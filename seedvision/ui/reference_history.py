"""Bounded, compressed undo history for full-resolution reference rasters."""

from __future__ import annotations

from dataclasses import dataclass
import zlib

import numpy as np


@dataclass(frozen=True, slots=True)
class _CompressedRasterPatch:
    index: int
    x: int
    y: int
    shape: tuple[int, int]
    dtype: str
    payload: bytes

    @classmethod
    def encode(
        cls,
        index: int,
        x: int,
        y: int,
        values: np.ndarray,
    ) -> "_CompressedRasterPatch":
        contiguous = np.ascontiguousarray(values)
        return cls(
            index=int(index),
            x=int(x),
            y=int(y),
            shape=tuple(int(value) for value in contiguous.shape),
            dtype=contiguous.dtype.str,
            payload=zlib.compress(contiguous.tobytes(), level=1),
        )

    def decode(self) -> np.ndarray:
        raw = zlib.decompress(self.payload)
        return np.frombuffer(raw, dtype=np.dtype(self.dtype)).reshape(
            self.shape
        )


@dataclass(frozen=True, slots=True)
class _RasterUndoEntry:
    label: str
    context: str
    patches: tuple[_CompressedRasterPatch, ...]
    metadata: str | None = None


@dataclass(frozen=True, slots=True)
class RasterUndoResult:
    """State restored by one undo operation."""

    label: str
    context: str
    rasters: tuple[np.ndarray, ...]
    metadata: str | None


class RasterUndoHistory:
    """Store changed raster regions rather than full-resolution snapshots."""

    TILE_SIZE = 128

    def __init__(self, limit: int = 20) -> None:
        if int(limit) < 5:
            raise ValueError("Reference undo history must retain at least five edits.")
        self._limit = int(limit)
        self._entries: list[_RasterUndoEntry] = []

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def next_label(self) -> str | None:
        return None if not self._entries else self._entries[-1].label

    @property
    def next_context(self) -> str | None:
        return None if not self._entries else self._entries[-1].context

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def record(
        self,
        label: str,
        context: str,
        before: tuple[np.ndarray, ...],
        after: tuple[np.ndarray, ...],
        *,
        metadata: str | None = None,
    ) -> bool:
        """Record one atomic edit, returning false when it changed no pixels."""

        if len(before) != len(after) or not before:
            raise ValueError("Undo states must contain matching raster groups.")
        normalized: list[tuple[np.ndarray, np.ndarray]] = []
        for earlier, later in zip(before, after, strict=True):
            earlier_values = np.asarray(earlier)
            later_values = np.asarray(later)
            if earlier_values.ndim != 2 or later_values.ndim != 2:
                raise ValueError("Undo history accepts two-dimensional rasters only.")
            if earlier_values.shape != later_values.shape:
                raise ValueError("Before and after raster shapes must match.")
            normalized.append((earlier_values, later_values))

        height, width = normalized[0][0].shape
        tile = self.TILE_SIZE
        patches = []
        # Compare one tile at a time.  A full-image union mask would itself be
        # tens of megabytes for the source photographs and is unnecessary for
        # the sparse edits this history is designed to retain.
        for y0 in range(0, height, tile):
            y1 = min(y0 + tile, height)
            for x0 in range(0, width, tile):
                x1 = min(x0 + tile, width)
                for index, (earlier_values, later_values) in enumerate(normalized):
                    before_region = earlier_values[y0:y1, x0:x1]
                    after_region = later_values[y0:y1, x0:x1]
                    if np.array_equal(before_region, after_region):
                        continue
                    patches.append(
                        _CompressedRasterPatch.encode(
                            index, x0, y0, before_region
                        )
                    )
        if not patches:
            return False
        self._entries.append(
            _RasterUndoEntry(
                label=str(label),
                context=str(context),
                patches=tuple(patches),
                metadata=metadata,
            )
        )
        if len(self._entries) > self._limit:
            del self._entries[: len(self._entries) - self._limit]
        return True

    def undo(self, current: tuple[np.ndarray, ...]) -> RasterUndoResult | None:
        """Restore and remove the newest entry against the supplied current state."""

        if not self._entries:
            return None
        entry = self._entries.pop()
        restored = [np.asarray(value) for value in current]
        copied_indexes: set[int] = set()
        for patch in entry.patches:
            if patch.index >= len(restored):
                raise ValueError("Undo raster group no longer matches its history.")
            if patch.index not in copied_indexes:
                restored[patch.index] = restored[patch.index].copy()
                copied_indexes.add(patch.index)
            target = restored[patch.index]
            x0, y0 = patch.x, patch.y
            y1 = y0 + patch.shape[0]
            x1 = x0 + patch.shape[1]
            if target.shape[0] < y1 or target.shape[1] < x1:
                raise ValueError("Undo raster dimensions no longer match the image.")
            target[y0:y1, x0:x1] = patch.decode()
        return RasterUndoResult(
            label=entry.label,
            context=entry.context,
            rasters=tuple(restored),
            metadata=entry.metadata,
        )
