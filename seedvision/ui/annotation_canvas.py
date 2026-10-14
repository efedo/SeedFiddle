"""Compact provenance for the corrected CPU image actually displayed by Qt."""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from seedvision.annotation.coordinates import validated_transform
from seedvision.persistence.reference_regions import validated_source_sha256


@dataclass(frozen=True, slots=True)
class AnnotationCanvasBinding:
    epoch: int
    source_sha256: str
    corrected_image_sha256: str
    shape: tuple[int, int]
    source_to_corrected: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class AnnotationEditContext:
    """Reject delayed edits even when the source's generation history is unknown."""
    epoch: int
    binding: AnnotationCanvasBinding | None


def displayed_canvas_binding(result, bgr, epoch):
    """Capture only an analysis's own corrected display buffer, without CUDA I/O."""
    calibration = None if result is None else getattr(result, 'calibration', None)
    if calibration is None or bgr is not getattr(calibration, 'corrected_bgr', None):
        return None
    source = validated_source_sha256(getattr(result, 'source_sha256', None), 'analysis source SHA-256')
    if source is None:
        return None
    transform = validated_transform(calibration.affine_matrix)
    values = np.asarray(bgr)
    if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
        return None
    return AnnotationCanvasBinding(int(epoch), source,
        sha256(np.ascontiguousarray(values).tobytes()).hexdigest(), tuple(values.shape[:2]),
        tuple(tuple(float(value) for value in row) for row in transform))
