"""Observe actual locals from one isolated public PlantCV watershed call.

This module imports no PlantCV, Qt or CUDA packages. It never reimplements a
public intermediate, replaces public helpers, or controls the public RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import sys
from typing import Any, Callable, Mapping

import numpy as np


class WatershedCaptureError(ValueError):
    """The returned public locals cannot satisfy the capture contract."""


class WatershedCaptureIdentityError(RuntimeError):
    """The callable changed while its public products were being observed."""


@dataclass(frozen=True)
class WatershedCapture:
    labels: np.ndarray
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]


SOURCE_LOCALS = {
    "distance": "dist_transform",
    "coordinates": "local_max_coordinates",
    "peaks": "local_max",
    "markers": "markers",
    "negative_distance": "dist_transform1",
    "labels": "labels",
    "diagnostic_bgr": "plt_img",
    "palette": "rand_color",
}

# Additional capture-only provenance; the canonical public manifest is unchanged.
CAPTURE_SOURCE_FILES = (
    ("watershed.py", 3222, "d5fd90bf4e064117ff47ee333d2b83e62134901fda36f6966ae1b0be4a82a712"),
    ("color_palette.py", 1143, "915191e9c21896155b7cbec9edeb1103f2f4a319dca86b0d02e0337a5bfefac0"),
    ("_helpers.py", 32744, "c05b0c944210657332ce307f2066caab55f2ebc8d8fea11988db3e476e7cdfba"),
    ("_debug.py", 864, "2f2363ce04653ac89b1ea6e5326dbba6848ff0284079f8a226d96af39b75b2f0"),
    ("classes.py", 18664, "8ee436f54c1bf10bdf6eaada1692205d61a209d65fb3687a636e87e97dc8b957"),
)
EMPTY_COORDINATES = {
    "schema": "seedfiddle.empty-watershed-coordinates.v1", "name": "coordinates",
    "shape": [0, 2], "dtype": "<i8", "nbytes": 0,
    "sha256": hashlib.sha256(b"").hexdigest(),
}


def capture_source_records() -> list[dict[str, Any]]:
    return [
        {"path": "plantcv/plantcv/" + name, "bytes": size, "sha256": digest}
        for name, size, digest in CAPTURE_SOURCE_FILES
    ]


def _limits(max_bytes: int, max_elements: int) -> None:
    if any(type(value) is not int or value < 1 for value in (max_bytes, max_elements)):
        raise WatershedCaptureError("Capture limits must be positive integers")


def _admit(shape: tuple[int, int], max_bytes: int, max_elements: int) -> None:
    _limits(max_bytes, max_elements)
    if len(shape) != 2 or any(type(axis) is not int or axis < 1 for axis in shape):
        raise WatershedCaptureError("Capture requires a nonempty HW frame")
    if np.dtype(np.intp).itemsize != 8:
        raise WatershedCaptureError("Capture requires the pinned intp64 environment")
    pixels = shape[0] * shape[1]
    # Includes the raw BGR diagnostic, a distinct RGB adapter, and palette.
    # This admits retained snapshots, not the public call's peak memory/RSS.
    if 42 * pixels + 3 > max_bytes or 16 * pixels + 3 > max_elements:
        raise WatershedCaptureError("Capture snapshot envelope exceeds its limit")


def _snapshot(
    local: Mapping[str, Any], returned: Any, mask: np.ndarray,
    *, max_bytes: int, max_elements: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    shape = mask.shape
    specs = {
        "distance": (np.dtype(np.float32), shape),
        "peaks": (np.dtype(np.bool_), shape),
        "markers": (np.dtype(np.int32), shape),
        "negative_distance": (np.dtype(np.float32), shape),
        "labels": (np.dtype(np.int32), shape),
        "diagnostic_bgr": (np.dtype(np.uint8), (*shape, 3)),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, (dtype, expected_shape) in specs.items():
        value = local.get(SOURCE_LOCALS[name])
        if type(value) is not np.ndarray or value.dtype != dtype or value.shape != expected_shape:
            raise WatershedCaptureError(f"Missing or malformed public local {SOURCE_LOCALS[name]}")
        arrays[name] = value
    coords = local.get("local_max_coordinates")
    if (
        type(coords) is not np.ndarray or coords.dtype != np.dtype(np.intp)
        or coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] > mask.size
    ):
        raise WatershedCaptureError("Missing or malformed public local local_max_coordinates")
    arrays["coordinates"] = coords
    palette = local.get("rand_color")
    if (
        type(palette) is not list or not 1 <= len(palette) <= mask.size
        or any(type(row) is not list or len(row) != 3 for row in palette)
        or any(type(value) is not int or not 0 <= value <= 255 for row in palette for value in row)
    ):
        raise WatershedCaptureError("Missing or malformed public local rand_color")
    # Charge every owned result, including palette conversion and RGB adapter,
    # before any contiguous snapshot allocation.
    total_bytes = sum(array.nbytes for array in arrays.values()) + len(palette) * 3 + arrays["diagnostic_bgr"].nbytes
    total_elements = sum(array.size for array in arrays.values()) + len(palette) * 3 + arrays["diagnostic_bgr"].size
    if total_bytes > max_bytes or total_elements > max_elements:
        raise WatershedCaptureError("Actual capture descriptors exceed their aggregate limit")
    distance = arrays["distance"]
    if not np.isfinite(distance).all() or not np.isfinite(arrays["negative_distance"]).all():
        raise WatershedCaptureError("Public distance locals contain nonfinite values")
    if np.any(distance < 0) or not np.array_equal(arrays["negative_distance"], -distance):
        raise WatershedCaptureError("Public negative-distance relation differs")
    if returned is not arrays["labels"]:
        raise WatershedCaptureError("Captured labels are not the actual public return")
    if coords.size:
        if np.any(coords < 0) or np.any(coords[:, 0] >= shape[0]) or np.any(coords[:, 1] >= shape[1]):
            raise WatershedCaptureError("Public peak coordinates leave their frame")
        if len(np.unique(coords, axis=0)) != len(coords):
            raise WatershedCaptureError("Public peak coordinates contain duplicates")
    peaks = arrays["peaks"]
    if np.count_nonzero(peaks) != len(coords) or (coords.size and not np.all(peaks[tuple(coords.T)])):
        raise WatershedCaptureError("Public peak support differs from actual coordinates")
    markers, labels = arrays["markers"], arrays["labels"]
    marker_ids, label_ids = np.unique(markers), np.unique(labels)
    if np.any(markers < 0) or np.any(labels < 0) or not np.array_equal(markers > 0, peaks):
        raise WatershedCaptureError("Public marker support differs from actual peaks")
    if np.any(labels[mask == 0] != 0) or not np.isin(label_ids[label_ids > 0], marker_ids).all():
        raise WatershedCaptureError("Public labels differ from marker/mask support")
    count = local.get("estimated_object_count")
    if type(count) is not int or count != len(marker_ids) - 1:
        raise WatershedCaptureError("Public estimated-object-count formula differs")
    if len(palette) != len(label_ids) or (label_ids.size and int(label_ids.max()) >= len(palette)):
        raise WatershedCaptureError("Public palette cannot index the actual label IDs")
    owned = {name: np.array(array, copy=True, order="C") for name, array in arrays.items()}
    owned["palette"] = np.array(palette, dtype=np.uint8, order="C")
    artifacts = {
        name: {
            "source_local": SOURCE_LOCALS[name], "shape": list(array.shape),
            "dtype": array.dtype.str, "nbytes": int(array.nbytes),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
        for name, array in owned.items()
    }
    return owned, {
        "schema": "seedfiddle.public-watershed-capture.v1",
        "status": "available", "estimated_object_count": count,
        "positive_marker_ids": marker_ids[marker_ids > 0].tolist(),
        "positive_label_ids": label_ids[label_ids > 0].tolist(),
        "label_max": int(labels.max()), "artifacts": artifacts,
        "palette_rng": "actual public shuffle; unseeded; state advancement preserved",
        "scientifically_validated": False,
    }


def capture_watershed(
    function: Callable[..., Any], *, mask: np.ndarray, kwargs: Mapping[str, Any],
    max_bytes: int, max_elements: int,
) -> WatershedCapture:
    """Call once, preserving public errors and existing profile/trace behavior.

    A capture contract failure retains a successful public label return. Callable
    identity changes abort instead. Cancellation belongs to the isolated runner,
    which can terminate the child and withhold all products.
    """
    if not inspect.isfunction(function):
        raise WatershedCaptureIdentityError("Capture requires the exact Python public callable")
    if type(mask) is not np.ndarray or mask.dtype != np.uint8 or mask.ndim != 2:
        raise WatershedCaptureError("Capture requires the public uint8 mask frame")
    _admit(mask.shape, max_bytes, max_elements)
    code, globals_ = function.__code__, function.__globals__
    previous = sys.getprofile()
    target_frame = None
    calls = 0
    products = None
    reason = None

    def observe(frame: Any, event: str, arg: Any) -> None:
        nonlocal target_frame, calls, products, reason
        if previous is not None:
            previous(frame, event, arg)
        if frame.f_code is not code or frame.f_globals is not globals_:
            return
        if function.__code__ is not code or function.__globals__ is not globals_:
            raise WatershedCaptureIdentityError("Public callable identity changed during capture")
        if event == "call":
            calls += 1
            if calls == 1:
                target_frame = frame
            else:
                reason = "Public target was called more than once"
                products = None
        elif event == "return" and frame is target_frame and arg is not None and calls == 1:
            try:
                products = _snapshot(frame.f_locals, arg, mask, max_bytes=max_bytes, max_elements=max_elements)
            except WatershedCaptureError as exc:
                reason = str(exc)

    try:
        sys.setprofile(observe)
        returned = function(**dict(kwargs))
    finally:
        sys.setprofile(previous)
    if function.__code__ is not code or function.__globals__ is not globals_:
        raise WatershedCaptureIdentityError("Public callable identity changed after capture")
    if type(returned) is not np.ndarray or returned.dtype != np.int32 or returned.shape != mask.shape:
        raise WatershedCaptureError("Public watershed did not return int32 labels in its input frame")
    if returned.nbytes > max_bytes or returned.size > max_elements:
        raise WatershedCaptureError("Public fallback labels exceed their storage limit")
    if np.any(returned < 0) or np.any(returned[mask == 0] != 0):
        raise WatershedCaptureError("Public fallback labels leave their mask support")
    if products is None or calls != 1:
        return WatershedCapture(np.array(returned, copy=True, order="C"), {}, {
            "schema": "seedfiddle.public-watershed-capture.v1",
            "status": "unavailable", "reason": reason or "Normal public return locals were not observed",
            "scientifically_validated": False,
        })
    arrays, metadata = products
    return WatershedCapture(returned, arrays, metadata)
