"""Bounded resource accounting for cached analysis products.

The desktop deliberately caches node outputs, but cache ownership must remain
explicit because a single image can retain hundreds of CUDA tensors.  These
helpers walk only ordinary containers and dataclass fields; they never invoke
array conversion or other properties that could materialize a lazy raster.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Iterator

import numpy as np

from seedvision.cuda.ops import GpuRaster


@dataclass(frozen=True, slots=True)
class ResidentBytes:
    """Unique storage reachable from one cached object graph."""

    cuda: int = 0
    numpy: int = 0
    cpu_tensor: int = 0

    @property
    def total(self) -> int:
        return self.cuda + self.numpy + self.cpu_tensor


def _children(value: object) -> Iterator[object]:
    if isinstance(value, dict):
        yield from value.keys()
        yield from value.values()
    elif isinstance(value, (tuple, list, set, frozenset)):
        yield from value
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield getattr(value, field.name)


def resident_bytes(value: object) -> ResidentBytes:
    """Measure unique NumPy and PyTorch storage without downloading tensors."""

    try:
        import torch
    except ImportError:  # pragma: no cover - the desktop runtime requires torch
        torch = None

    seen_objects: set[int] = set()
    seen_storages: set[tuple[object, ...]] = set()
    cuda_bytes = 0
    numpy_bytes = 0
    cpu_tensor_bytes = 0

    def visit(item: object) -> None:
        nonlocal cuda_bytes, numpy_bytes, cpu_tensor_bytes
        identity = id(item)
        if identity in seen_objects:
            return
        seen_objects.add(identity)

        if isinstance(item, GpuRaster):
            visit(item.gpu_tensor())
            host = item.host_cache
            if host is not None:
                visit(host)
            return
        if isinstance(item, np.ndarray):
            owner = item
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            pointer = int(owner.__array_interface__["data"][0])
            key = ("numpy", pointer, int(owner.nbytes))
            if key not in seen_storages:
                seen_storages.add(key)
                numpy_bytes += int(owner.nbytes)
            return
        if torch is not None and torch.is_tensor(item):
            storage = item.untyped_storage()
            key = (
                "torch",
                str(item.device),
                int(storage.data_ptr()),
                int(storage.nbytes()),
            )
            if key not in seen_storages:
                seen_storages.add(key)
                if item.device.type == "cuda":
                    cuda_bytes += int(storage.nbytes())
                else:
                    cpu_tensor_bytes += int(storage.nbytes())
            return
        for child in _children(item):
            visit(child)

    visit(value)
    return ResidentBytes(cuda_bytes, numpy_bytes, cpu_tensor_bytes)


def release_host_caches(value: object) -> int:
    """Release every lazy CPU mirror reachable from ``value``.

    Returns the number of cache-owning objects cleared. CUDA tensors and source
    NumPy inputs are untouched.
    """

    seen: set[int] = set()
    released = 0

    def visit(item: object) -> None:
        nonlocal released
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(item, GpuRaster):
            if item.is_materialized:
                released += 1
            item.release_host_cache()
            return
        release = getattr(item, "release_host_cache", None)
        if callable(release):
            had_cache = getattr(item, "_host_cache", None) is not None
            release()
            released += int(had_cache)
        for child in _children(item):
            visit(child)

    visit(value)
    return released
