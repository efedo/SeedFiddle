"""Explicit CPU input snapshots and saved revisions for the desktop reference view."""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import os
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np

from . import native_image_inputs as inputs
from .checkpoints import (
    CheckpointCancelled, CheckpointIdentity, CheckpointLimits, CoordinateFrame,
    ExecutionRecord, ExecutionStatus, MeasurementContract, RasterRole, _canonical_json,
    _cancelled, publish_checkpoint,
)
from .recipes import RecipeSpec
from .reference_runner import MAX_ARRAY_BYTES, MAX_ARRAY_ELEMENTS, PlantCVReferenceRunner, _strict_json_loads
from .reference_seed_chain import ReferenceRasterInput, ReferenceSeedChainOutcome, run_reference_seed_chain
from .reference_seed_chain_reader import VerifiedReferenceSeedChain, read_reference_seed_chain
from .reference_watershed_capture import _admit


def reference_measurement(name: str) -> MeasurementContract:
    return MeasurementContract(name, "1", "public_reference_raster", "pixel", "not_applicable",
                               "not_evaluated", False, "Public execution does not certify biological measurement validity.")


def _poll(cancel: Any, **_kwargs) -> None:
    if _cancelled(cancel):
        raise CheckpointCancelled("Public reference session cancelled.")


@dataclass(frozen=True, slots=True)
class ReferenceCaptureSession:
    recipe: RecipeSpec
    outcome: ReferenceSeedChainOutcome | None
    verified: VerifiedReferenceSeedChain | None


def capture_reference_path(recipe: RecipeSpec, source_path: Path | str, store: Path | str,
                           runner: PlantCVReferenceRunner, *, expected_stamp=None,
                           limits: CheckpointLimits = CheckpointLimits(), cancel=None) -> ReferenceCaptureSession:
    """Prepare caller RGB/gray on CPU for isolated public calls, never the native cache.

The local decoder prepares inputs; it is not an invocation of PlantCV readimage.
Gray is derived from that same color decode. Owned array limits are separate
from codec scratch, metadata and public/process peak memory.
"""
    source_path, store = Path(source_path), Path(store)
    poll = lambda **kwargs: _poll(cancel, **kwargs)
    _poll(cancel)
    if expected_stamp is not None and inputs.stamp(source_path) != expected_stamp:
        raise ValueError("Selected image changed before reference input preparation.")
    source = inputs.source_identity(source_path, poll)
    raw = inputs.snapshot(source_path, source, poll)
    bgr = inputs.decode(raw)
    shape = bgr.shape[:2]
    _admit(shape, min(limits.max_total_array_bytes, MAX_ARRAY_BYTES), MAX_ARRAY_ELEMENTS)
    inputs.decoded_bound(int(bgr.shape[0] * bgr.shape[1]), color=True, gray=True)
    _poll(cancel)
    decoder = inputs.versions()
    transform = sha256(_canonical_json({"decoder": decoder, "source": source.sha256,
                                       "frame": list(shape)}, name="prepared reference frame")).hexdigest()
    frame = CoordinateFrame("prepared_public_reference_frame", shape, "top_left_pixel_centres",
                            "x_right_y_down", "pixel", transform)
    contract = {"decoder": decoder, "gray": "cv2.COLOR_BGR2GRAY from same decoded BGR",
                "rgb": "BGR to RGB channel reversal", "source": source.sha256, "frame": frame.to_dict()}
    digest = sha256(_canonical_json(contract, name="prepared reference contract")).hexdigest()
    common = CheckpointIdentity(
        source.sha256, frame, "prepared_reference_rgb", "seedfiddle_prepared_public_reference", "1",
        digest, digest, digest, "seedfiddle.reference_input", "1",
        {"opencv": str(cv2.__version__), "numpy": str(np.__version__)},
        {"color_order": "RGB", "dtype": "uint8", "value_range": [0, 255],
         "channel_conversion": "BGR to RGB from local OpenCV byte decode",
         "geometric_transform": "decoder orientation; no rectification", "interpolation": "decoder_defined"},
        {"foreground_polarity": "light", "representation": "uint8_0_255", "connectivity": "not_applicable",
         "validity_region": "full_frame", "border_policy": "not_applicable"},
        {"label_dtype": "int32", "background_label": 0, "label_id_mapping": "not_created",
         "split_boundary_ownership": "not_applicable", "visibility": "full_frame", "eligibility": "not_evaluated"},
        source.sha256,
    )
    rgb_array = np.ascontiguousarray(bgr[..., ::-1])
    gray_array = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bindings = {}
    for name, array in (("rgb", rgb_array), ("gray", gray_array)):
        _poll(cancel)
        identity = common if name == "rgb" else replace(common, stage_id="prepared_reference_gray", image_semantics={
            **dict(common.image_semantics), "color_order": "GRAY", "channel_conversion": contract["gray"]})
        role = RasterRole("image", "|u1", "HWC" if name == "rgb" else "HW", (0, 255))
        measurement = reference_measurement(f"prepared_reference_{name}")
        reference = publish_checkpoint(store / "inputs", identity, {name: array},
            execution=ExecutionRecord(ExecutionStatus.AVAILABLE), measurement=measurement,
            raster_roles={name: role}, metadata={"preparation": contract, "actual_public_reader": False},
            allow_cpu_transfer=True, limits=limits, cancel=cancel)
        bindings[name] = ReferenceRasterInput(store / "inputs", reference.checkpoint_sha256,
                                            identity, measurement, name, role)
    del bgr, rgb_array, gray_array, array, raw
    measurement = reference_measurement("public_reference_chain_output")
    outcome = run_reference_seed_chain(bindings["gray"], bindings["rgb"], recipe, runner, store,
        expected_source_sha256=source.sha256, expected_coordinate_frame=frame,
        expected_recipe_identity=recipe.identity, output_measurement=measurement,
        allow_cpu_transfer=True, limits=limits, cancel=cancel)
    verified = None
    if outcome.manifest_sha256 is not None:
        # The coordinator's unique call UUIDs make this manifest specific to
        # this attempt. Until the sidecar commits, a failed desktop session
        # retracts only its own hash-verified manifest; checkpoints survive.
        manifest_path = (store / "chains" / f"{outcome.manifest_sha256}.json").resolve()
        path = manifest_path.with_name(f"{outcome.manifest_sha256}.recipe.json")
        temporary = None
        try:
            _poll(cancel)
            verified = read_reference_seed_chain(store, outcome.manifest_sha256, gray=bindings["gray"], rgb=bindings["rgb"],
                recipe=recipe, expected_source_sha256=source.sha256, expected_coordinate_frame=frame,
                output_measurement=measurement, limits=limits, cancel=cancel)
            recipe_bytes = recipe.to_json().encode("utf-8")
            if len(recipe_bytes) > limits.max_metadata_bytes:
                raise ValueError("Selected recipe exceeds its saved revision metadata cap.")
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(recipe_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            _poll(cancel)
            os.replace(temporary, path)
            temporary = None
        except Exception:
            if (manifest_path.parent == (store / "chains").resolve()
                    and manifest_path == Path(outcome.manifest_path).resolve()
                    and manifest_path.is_file()):
                with manifest_path.open("rb") as stream:
                    own_bytes = stream.read(limits.max_metadata_bytes + 1)
                if len(own_bytes) <= limits.max_metadata_bytes and sha256(own_bytes).hexdigest() == outcome.manifest_sha256:
                    manifest_path.unlink()
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        # Sidecar replacement commits the desktop revision. Cancellation
        # arriving after this point cannot roll back a completed capture.
    else:
        _poll(cancel)
    return ReferenceCaptureSession(recipe, outcome, verified)


def open_reference_capture(path: Path | str, *, limits=CheckpointLimits(), cancel=None) -> ReferenceCaptureSession:
    """Open an explicitly selected stored revision; no current-image identity is inferred."""
    path = Path(path).resolve()
    digest = path.stem
    _poll(cancel)
    with path.open("rb") as stream:
        raw = stream.read(limits.max_metadata_bytes + 1)
    if len(raw) > limits.max_metadata_bytes or sha256(raw).hexdigest() != digest:
        raise ValueError("Selected capture bytes differ from their content-addressed revision.")
    record = _strict_json_loads(raw)
    with path.with_name(f"{digest}.recipe.json").open("rb") as stream:
        recipe_raw = stream.read(limits.max_metadata_bytes + 1)
    if len(recipe_raw) > limits.max_metadata_bytes:
        raise ValueError("Stored recipe exceeds its metadata cap.")
    recipe = RecipeSpec.from_json(recipe_raw.decode("utf-8"))
    store = path.parent.parent
    if path.parent.name != "chains":
        raise ValueError("Select a manifest from its captured chain store.")
    bindings = {}
    for name in ("gray", "rgb"):
        value = record["inputs"][name]
        bindings[name] = ReferenceRasterInput(store / "inputs", value["checkpoint_sha256"],
            CheckpointIdentity.from_dict(value["identity"]), MeasurementContract.from_dict(value["measurement"]),
            value["array_name"], RasterRole.from_dict(value["role"]))
    verified = read_reference_seed_chain(store, digest, gray=bindings["gray"], rgb=bindings["rgb"], recipe=recipe,
        expected_source_sha256=record["source_sha256"], expected_coordinate_frame=CoordinateFrame.from_dict(record["coordinate_frame"]),
        output_measurement=reference_measurement("public_reference_chain_output"), limits=limits, cancel=cancel)
    return ReferenceCaptureSession(recipe, None, verified)


__all__ = ["ReferenceCaptureSession", "capture_reference_path", "open_reference_capture", "reference_measurement"]
