"""PlantCV 4.11.3 colour correction with CUDA-resident image arithmetic.

The pinned colour_correction.py source SHA-256 is
c0c3cf636db7a881aef820743501614fc3dccd0c1b732ac7df5d54a7607cce68.
Only compact chip matrices and their NumPy linear algebra cross the CPU boundary.
Source and corrected images, chip reductions and reusable pixel data stay on CUDA.
Floating reduction/BLAS differences require numerical conformance tests; these
functions do not claim byte-exact conversion or validated colour measurements.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from typing import Callable

import numpy as np
import torch
from . import tensor_callback


MAX_PIXELS = 64_000_000
TILE_PIXELS = 65_536
MAX_CHIPS = 255


class CorrectionCancelled(RuntimeError):
    pass


class CorrectionMutation(ValueError,tensor_callback.TensorCallbackMutation):
    """A callback changed a live correction input or private buffer."""


def _metadata_signature(value):
    if torch.is_tensor(value):return ('tensor',id(value),tensor_callback._signature(value))
    if isinstance(value,np.ndarray):
        base=value
        while isinstance(base,np.ndarray) and base.base is not None:base=base.base
        immutable=isinstance(base,bytes) and not value.flags.writeable
        return ('array',id(value),value.dtype.str,tuple(value.shape),value.strides,
            value.flags.writeable,value.__array_interface__['data'][0],id(base),
            None if immutable else sha256(value.tobytes()).hexdigest())
    if isinstance(value,Mapping):
        return ('mapping',id(value),tuple((key,_metadata_signature(item)) for key,item in value.items()))
    if type(value) in (tuple,list):return (type(value).__name__,tuple(_metadata_signature(item) for item in value))
    if is_dataclass(value) and not isinstance(value,type):
        return (type(value).__qualname__,tuple((field.name,_metadata_signature(getattr(value,field.name))) for field in fields(value)))
    return (type(value).__name__,value)


def _guarded(cancel,inputs=(),metadata=()):
    """Persistent input baselines plus scoped callback-time private buffers."""
    tensors=tuple(value for value in inputs if torch.is_tensor(value) and value.layout==torch.strided)
    baseline=tuple(tensor_callback._signature(value) for value in tensors)
    compact=_metadata_signature(metadata)
    def intact():
        if (tuple(tensor_callback._signature(value) for value in tensors)!=baseline
                or _metadata_signature(metadata)!=compact):
            raise CorrectionMutation('Correction input or compact metadata changed during calculation.')
    def guard(*private,metadata=()):
        intact()
        try:
            try:stopped=tensor_callback.checked_cancel(cancel,(*tensors,*private),
                identity=lambda:_metadata_signature(metadata))
            except tensor_callback.TensorCallbackMutation as error:
                raise CorrectionMutation(str(error)) from error
            if stopped:raise CorrectionCancelled('Colour correction was cancelled before a complete result.')
        finally:intact()
        return False
    return guard


@dataclass(frozen=True, slots=True)
class ColorCorrectionResult:
    corrected_bgr: torch.Tensor
    source_matrix: np.ndarray
    target_matrix: np.ndarray
    transformation_matrix: np.ndarray
    fit: str
    numpy_version: str


def _cancel(cancel,tensors=(),metadata=()):
    try:stopped=tensor_callback.checked_cancel(cancel,tensors,identity=lambda:_metadata_signature(metadata))
    except tensor_callback.TensorCallbackMutation as error:raise CorrectionMutation(str(error)) from error
    if stopped:
        raise CorrectionCancelled('Colour correction was cancelled before a complete result.')


def _image(image, cancel):
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be a zero-argument callable or None')
    _cancel(cancel,(image,))
    if (not torch.is_tensor(image) or image.layout != torch.strided
            or image.device.type != 'cuda' or image.dtype != torch.uint8
            or image.ndim != 3 or image.shape[-1] != 3 or not image.is_contiguous()):
        raise ValueError('Image must be contiguous CUDA HxWx3 uint8 BGR.')
    if not 1 <= image.shape[0] * image.shape[1] <= MAX_PIXELS:
        raise ValueError(f'Image must contain 1..{MAX_PIXELS} pixels.')
    return image


def get_color_matrix_cuda(bgr: torch.Tensor, mask: torch.Tensor, *, cancel=None):
    """Return PlantCV chip IDs and normalized RGB means as compact CPU metadata.

    The supported uint8 labelled mask includes background zero, as required by
    the pinned allocation behavior. A mask without background is rejected before
    calculating a misleading matrix. Empty masks return a 0x4 matrix.
    """
    guard=_guarded(cancel,(bgr,mask))
    _image(bgr, guard)
    if (not torch.is_tensor(mask) or mask.layout != torch.strided or mask.device != bgr.device
            or mask.dtype != torch.uint8 or tuple(mask.shape) != tuple(bgr.shape[:2])
            or not mask.is_contiguous()):
        raise ValueError('Chip mask must be contiguous CUDA HxW uint8 on the image device.')
    counts = torch.zeros(256, dtype=torch.int64, device=bgr.device)
    sums = torch.zeros((3, 256), dtype=torch.int64, device=bgr.device)
    pixels, labels = bgr.reshape(-1, 3), mask.reshape(-1)
    for start in range(0, labels.numel(), TILE_PIXELS):
        guard(counts,sums)
        ids = labels[start:start+TILE_PIXELS].to(torch.int64)
        values = pixels[start:start+TILE_PIXELS].to(torch.int64)
        counts += torch.bincount(ids, minlength=256)
        for channel in range(3):
            sums[channel].scatter_add_(0, ids, values[:, channel])
        guard(counts,sums,ids,values)
        del ids,values
    if int(counts[0].item()) == 0:
        raise ValueError('Pinned get_color_matrix requires background zero in the chip mask.')
    selected = torch.nonzero(counts[1:] > 0).reshape(-1) + 1
    result = torch.empty((selected.numel(), 4), dtype=torch.float64, device=bgr.device)
    result[:, 0] = selected
    result[:, 1:] = (sums[:, selected].flip(0).transpose(0, 1).to(torch.float64)
                     / counts[selected, None] / 255.0)
    guard(counts,sums,selected,result)
    matrix = result.cpu().numpy().copy()
    guard(counts,sums,selected,result,metadata=(matrix,))
    return ('chip_number', 'r_avg', 'g_avg', 'b_avg'), matrix


def _matrices(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    for value in (source, target):
        if (value.ndim != 2 or value.shape[1] != 4 or not 1 <= value.shape[0] <= MAX_CHIPS
                or not np.isfinite(value).all() or (value[:, 1:] < 0).any() or (value[:, 1:] > 1).any()):
            raise ValueError('Colour matrices must contain 1..255 rows [chip ID,R,G,B] with finite normalized RGB.')
    return source.copy(), target.copy()


def _apply(bgr, coefficients, *, affine, cancel):
    guard=_guarded(cancel,(bgr,),metadata=(coefficients,))
    _image(bgr, guard)
    coeff = torch.as_tensor(np.ascontiguousarray(coefficients), dtype=torch.float64, device=bgr.device)
    output = torch.empty_like(bgr)
    pixels, destination = bgr.reshape(-1, 3), output.reshape(-1, 3)
    for start in range(0, pixels.shape[0], TILE_PIXELS):
        guard(coeff,output)
        rgb = pixels[start:start+TILE_PIXELS].flip(1).to(torch.float64) / 255.0
        if affine:
            basis = torch.cat((rgb, torch.ones((rgb.shape[0], 1), dtype=torch.float64, device=bgr.device)), dim=1)
            corrected = torch.stack([basis @ coeff[:, channel] for channel in range(3)], dim=1)
        else:
            basis = torch.cat((rgb, rgb.square(), rgb.pow(3)), dim=1)
            # Preserve the source's left-to-right, nine-term summation order.
            corrected = basis[:, 0, None] * coeff[0, :3]
            for term in range(1, 9):
                corrected = corrected + basis[:, term, None] * coeff[term, :3]
        destination[start:start+TILE_PIXELS] = (255.0 * corrected.clamp(0, 1)).to(torch.uint8).flip(1)
        guard(coeff,output,rgb,basis,corrected)
        del rgb,basis,corrected
    guard(coeff,output)
    return output


def affine_color_correction_cuda(bgr, source_matrix, target_matrix, *, cancel=None):
    """Pinned affine RGB fit; chip IDs are ignored and rows correspond by position."""
    source,target=_matrices(source_matrix,target_matrix)
    guard=_guarded(cancel,(bgr,),metadata=(source_matrix,target_matrix,source,target))
    _image(bgr, guard)
    if source.shape != target.shape:
        raise ValueError('Mismatch between the colour matrices\' shapes.')
    augmented = np.concatenate((source[:, 1:], np.ones((source.shape[0], 1))), axis=1)
    inverse = np.linalg.pinv(augmented, rcond=1e-15)
    coefficients = np.column_stack([inverse @ target[:, channel] for channel in range(1, 4)])
    def fit_guard():return guard(metadata=(augmented,inverse,coefficients))
    fit_guard()
    corrected = _apply(bgr, coefficients, affine=True, cancel=fit_guard)
    guard(corrected,metadata=(augmented,inverse,coefficients))
    return ColorCorrectionResult(corrected, source, target, coefficients, 'affine', np.__version__)


def correct_color_cuda(target_bgr, target_mask, source_bgr, source_mask, *, cancel=None):
    """Pinned nine-term normal-equation correction from labelled colour chips.

    Correspondence follows the original source: equal-shaped tables use row order;
    unequal tables match chip IDs and retain unmatched zero rows. Singular fits
    propagate NumPy's LinAlgError, with no corrected image returned.
    All four input tensors and retained compact matrices stay guarded across
    extraction, fitting and application callbacks through the final check.
    """
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be a zero-argument callable or None')
    guard = _guarded(cancel, (target_bgr, target_mask, source_bgr, source_mask))
    _image(target_bgr, guard)
    _image(source_bgr, guard)
    _, target = get_color_matrix_cuda(target_bgr, target_mask, cancel=guard)
    target_guard = _guarded(guard, metadata=(target,))
    _, source = get_color_matrix_cuda(source_bgr, source_mask, cancel=target_guard)
    matrices_guard = _guarded(target_guard, metadata=(source,))
    source, target = _matrices(source, target)
    if source.shape == target.shape:
        source_rgb, target_rgb = source[:, 1:], target[:, 1:]
    else:
        pairs = np.zeros((source.shape[0], 6), dtype=np.float64)
        row = 0
        for target_row in target:
            for source_row in source:
                if target_row[0] == source_row[0]:
                    pairs[row] = np.concatenate((target_row[1:], source_row[1:]))
                    row += 1
        target_rgb, source_rgb = pairs[:, :3], pairs[:, 3:]
    a = np.concatenate((source_rgb, source_rgb**2, source_rgb**3), axis=1)
    b = np.concatenate((target_rgb, target_rgb**2, target_rgb**3), axis=1)
    inverse = np.linalg.solve(a.T @ a, a.T)
    coefficients = np.column_stack([inverse @ b[:, channel] for channel in range(9)])
    fit_guard = _guarded(matrices_guard, metadata=(source, target, a, b, inverse, coefficients))
    fit_guard()
    corrected = _apply(source_bgr, coefficients, affine=False, cancel=fit_guard)
    fit_guard(corrected)
    return ColorCorrectionResult(corrected, source, target, coefficients, 'polynomial_normal_equations', np.__version__)
