"""CUDA priority-queue watershed matching the pinned PlantCV internal call.

This implements scikit-image 0.25.2's two-dimensional, four-connected default
with compactness=0 and watershed_line=False. Priority, queue age, initial marker
order and heap ties are preserved. Dense rasters/queue storage stay on CUDA;
only scalar progress metadata crosses to the host between bounded kernel batches.
The application uses the compiler bundled with PyTorch, without a toolkit install.
"""
from hashlib import sha256
from pathlib import Path

import torch

from seedvision.cuda._nvrtc import fixed_kernel_program


SCIKIT_IMAGE_VERSION = '0.25.2'
SCIKIT_IMAGE_WRAPPER_SHA256 = 'efde33654577d7c1ce8dd8beb9d4cc8e78001d7d2594b1556fce900f7c488963'
MAX_RASTER_PIXELS = 64_000_000
MAX_WORKSPACE_BYTES = 512*1024*1024
KERNEL_BATCH_PIXELS = 4096
VALIDATION_TILE_PIXELS = 262_144
_CUDA_SOURCE = Path(__file__).with_name('_plantcv_watershed.cu').read_text(encoding='utf-8')
CUDA_SOURCE_SHA256 = sha256(_CUDA_SOURCE.encode('utf-8')).hexdigest()


class WatershedCancelled(RuntimeError):
    pass


class WatershedResourceError(ValueError):
    pass


def _cancel(cancel):
    if cancel is not None and cancel():
        raise WatershedCancelled('Watershed cancelled before a complete partition.')


def masked_watershed(elevation, markers, mask, *, cancel=None):
    """Return int32 labels on the input CUDA device, without changing any input.

    Inputs are contiguous same-shape HxW float32 elevation, int32 markers and
    uint8/bool mask. Any nonzero mask value is foreground. Nonzero marker IDs
    (including negative IDs) are retained; markers outside the mask are ignored.
    Mask components without a marker retain background zero, as in the reference.
    """
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be a zero-argument callable or None')
    _cancel(cancel)
    if not all(torch.is_tensor(tensor) for tensor in (elevation, markers, mask)):
        raise TypeError('Watershed inputs must be PyTorch tensors.')
    if (elevation.layout != torch.strided or elevation.device.type != 'cuda'
            or elevation.dtype != torch.float32 or elevation.ndim != 2):
        raise ValueError('Elevation must be a CUDA HxW float32 raster.')
    if (markers.dtype != torch.int32 or mask.dtype not in (torch.uint8, torch.bool)
            or any(tensor.layout != torch.strided or tensor.device != elevation.device
                or tuple(tensor.shape) != tuple(elevation.shape) or not tensor.is_contiguous()
                for tensor in (elevation, markers, mask))):
        raise ValueError('Require same-device contiguous HxW elevation, int32 markers and uint8/bool mask.')
    height, width = elevation.shape
    pixels = elevation.numel()
    if not 1 <= pixels <= MAX_RASTER_PIXELS:
        raise WatershedResourceError('Watershed exceeds its nonempty raster limit.')
    foreground = 0
    flat_elevation, flat_mask = elevation.reshape(-1), mask.reshape(-1)
    for start in range(0, pixels, VALIDATION_TILE_PIXELS):
        _cancel(cancel)
        if not bool(torch.isfinite(flat_elevation[start:start+VALIDATION_TILE_PIXELS]).all().item()):
            raise ValueError('Watershed elevation must be finite.')
        foreground += int(torch.count_nonzero(flat_mask[start:start+VALIDATION_TILE_PIXELS]).item())
    capacity = max(1, foreground)
    estimate = pixels*4 + capacity*12 + 16
    if estimate > MAX_WORKSPACE_BYTES:
        raise WatershedResourceError('Watershed output and fixed queue exceed the workspace byte limit.')
    _cancel(cancel)
    with torch.cuda.device(elevation.device):
        output = torch.empty_like(markers)
        values = torch.empty(capacity, dtype=torch.float32, device=elevation.device)
        ages = torch.empty(capacity, dtype=torch.int32, device=elevation.device)
        indices = torch.empty_like(ages)
        state = torch.tensor([0, 1, 0, 0], dtype=torch.int32, device=elevation.device)
        program = fixed_kernel_program(_CUDA_SOURCE, elevation.device)
        arguments = (elevation, markers, mask, output, values, ages, indices, state)
        integers = (height, width, capacity, KERNEL_BATCH_PIXELS)
        for _start in range(0, pixels, KERNEL_BATCH_PIXELS):
            _cancel(cancel)
            program.launch('seedvision_ws_init', arguments, integers)
            if int(state[3].item()):
                raise WatershedResourceError('Watershed queue exceeded its checked capacity.')
        while int(state[0].item()) > 0:
            _cancel(cancel)
            program.launch('seedvision_ws_step', arguments, integers)
            if int(state[3].item()):
                raise WatershedResourceError('Watershed queue exceeded its checked capacity.')
        _cancel(cancel)
        return output
