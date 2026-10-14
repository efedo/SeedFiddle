"""CUDA-resident PlantCV 4.11.3 ``threshold.custom_range`` primitive.

The pinned call receives BGR uint8 pixels from ``readimage``. Its ``channel='RGB'``
branch compares the three stored slots in B,G,R order; it does not reorder them.
All channel intervals use OpenCV ``inRange``'s inclusive lower and upper bounds.
HSV, Lab, and gray use byte-exact pinned OpenCV conversions. CMYK follows
PlantCV's float64 formula and its historical C/Y output-plane swap.

The LAB byte arithmetic follows OpenCV 4.11.0
``modules/imgproc/src/color_lab.cpp``, ``color.simd_helpers.hpp`` and
``modules/core/src/softfloat.cpp`` at
``https://github.com/opencv/opencv/tree/4.11.0`` (Apache-2.0; Copyright OpenCV
contributors). The softfloat source retains Berkeley SoftFloat and Sun fdlibm
notices; see ``docs/OPENCV_MATH_NOTICES.md``. The oracle is the pinned opencv-python 4.11.0.86 wheel
(SHA-256 ``085ad9b77c18853ea66283e98affefe2de8cc4c1f43eda4c100cf9b2721142ec``)
recorded in ``config/peer_reference_sources.json``. Only the compact 256-entry
sRGB and 3072-entry cube-root tables are generated on CPU; image pixels and all
per-pixel conversion remain on the input device.

The supported native input is one CUDA NCHW BGR raster, uint8 or finite float32
in the 0..255 image domain. Strided views are accepted when positive dimensions
are separable by the sorted-stride extent check; other interleaved layouts are
rejected even if their individual element addresses do not overlap. Float input
is rounded ties-to-even and saturated to uint8, matching this project's
``tensor_to_image`` boundary. The returned
``(mask, masked_image)`` is CUDA uint8: HxW 0/255 and 1x3xHxW respectively.
Only byte-range integer bounds are supported; OpenCV's implicit conversion of
floating/out-of-range scalar bounds is intentionally outside this API.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
import math
from numbers import Integral
import struct
from typing import Final

import torch


PLANTCV_VERSION: Final = "4.11.3"
PLANTCV_COMMIT: Final = "77e3322fd0cd5a9bef18bb99402f20c8f2db87a9"
PLANTCV_THRESHOLD_SOURCE_SHA256: Final = "04a671632e7e4decf5a662779cd88473b5aff750554afeee98eabe02eaa49907"
PLANTCV_HELPER_REFERENCE_ENV_SHA256: Final = "c05b0c944210657332ce307f2066caab55f2ebc8d8fea11988db3e476e7cdfba"
MAX_RASTER_PIXELS: Final = 64_000_000
MAX_OUTPUT_BYTES: Final = 256 * 1024 * 1024
MAX_TILE_PIXELS: Final = 262_144


def _float32(value: float) -> float:
    """Round a scalar through IEEE-754 binary32, as OpenCV's softfloat does."""
    return struct.unpack("=f", struct.pack("=f", value))[0]


_OPENCV_CBRT_COEFFICIENTS = tuple(
    struct.unpack("=d", struct.pack("=Q", int(raw, 16)))[0]
    for raw in (
        "4046a09e6653ba70", "406808f46c6116e0", "405dca97439cae14",
        "402add70d2827500", "3fc4f15f83f55d2d", "402d9e20660edb21",
        "4062ff15c0285815", "406510d06a8112ce", "4040fecbc9e2c375",
    )
)


def _opencv_softfloat_cbrt(value: float) -> float:
    """OpenCV 4.11 softfloat.cpp f32_cbrt rational approximation/bit pack.

    OpenCV attributes this approximation to Ken Turkowski. The scalar table
    calculation deliberately truncates the binary64 mantissa into binary32;
    the platform's rounded math.cbrt would change some byte Lab results.
    """
    bits = struct.unpack("=I", struct.pack("=f", value))[0]
    if bits & 0x7FFFFFFF == 0:
        return 0.0
    sign = bits >> 31
    exponent = ((bits >> 23) & 0xFF) - 127
    shift = exponent % 3 - 3
    output_exponent = (exponent - shift) // 3 - 1
    fraction = bits & 0x007FFFFF
    fr_bits = ((shift + 1023) << 52) | (fraction << 29)
    fr = struct.unpack("=d", struct.pack("=Q", fr_bits))[0]
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = _OPENCV_CBRT_COEFFICIENTS
    numerator = ((((a1 * fr + a2) * fr + a3) * fr + a4) * fr + a5)
    denominator = ((((a6 * fr + a7) * fr + a8) * fr + a9) * fr + 1.0)
    result_bits = struct.unpack("=Q", struct.pack("=d", numerator / denominator))[0]
    output_bits = (sign << 31) | ((output_exponent + 127) << 23) | ((result_bits & ((1 << 52) - 1)) >> 29)
    return struct.unpack("=f", struct.pack("=I", output_bits))[0]


def _lab_gamma_table() -> tuple[int, ...]:
    """Build OpenCV 4.11's compact uint8 sRGB linearization table."""
    gamma_threshold = 809.0 / 20_000.0
    gamma_low_scale = 323.0 / 25.0
    gamma_x_shift = 11.0 / 200.0
    gamma_power = 12.0 / 5.0
    int_scale = 255 * (1 << 3)
    table: list[int] = []
    for value in range(256):
        # OpenCV first divides in binary32, then promotes the sample to
        # softdouble for the sRGB power formula and casts back to softfloat.
        x = _float32(_float32(value) / _float32(255))
        if float(x) <= gamma_threshold:
            linear = _float32(float(x) / gamma_low_scale)
        else:
            powered = math.pow((float(x) + gamma_x_shift) / (1.0 + gamma_x_shift), gamma_power)
            linear = _float32(powered)
        table.append(int(round(_float32(_float32(int_scale) * linear))))
    return tuple(table)


def _lab_cbrt_table() -> tuple[int, ...]:
    """Build OpenCV's uint8 Lab transfer table from its pinned constants."""
    gamma_shift = 3
    table_size = 256 * 3 // 2 * (1 << gamma_shift)
    f255 = _float32(255)
    cbrt_scale = _float32(1.0 / _float32(f255 * (1 << gamma_shift)))
    threshold = _float32(_float32(216) / _float32(24389))
    scale = _float32(_float32(841) / _float32(108))
    bias = _float32(_float32(16) / _float32(116))
    lab_scale = 1 << (12 + gamma_shift)
    table: list[int] = []
    for index in range(table_size):
        x = _float32(cbrt_scale * _float32(index))
        if x < threshold:
            # The product of two binary32 values and this nearby addend fit
            # in binary64 before the single binary32 rounding.
            value = _float32(float(x) * float(scale) + float(bias))
        else:
            value = _opencv_softfloat_cbrt(x)
        table.append(int(round(_float32(_float32(lab_scale) * value))))
    return tuple(table)


_LAB_GAMMA_VALUES: Final = _lab_gamma_table()
_LAB_CBRT_VALUES: Final = _lab_cbrt_table()


@lru_cache(maxsize=8)
def _lab_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Upload only the small source-derived tables, once per active device."""
    gamma = torch.tensor(_LAB_GAMMA_VALUES, dtype=torch.int32, device=device)
    cbrt = torch.tensor(_LAB_CBRT_VALUES, dtype=torch.int32, device=device)
    return gamma, cbrt


class RangeResourceError(ValueError):
    """Raised when the requested output or raster exceeds its explicit budget."""


class RangeCancelledError(RuntimeError):
    """Raised when cancellation prevents a complete mask/image result."""


class UnsupportedRangeModeError(ValueError):
    """Raised when a requested mode cannot be evaluated with supported semantics."""


class UnsupportedRangeConversionError(ValueError):
    """Raised when a valid PlantCV mode lacks a byte-exact native conversion."""


def _check_cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise RangeCancelledError("Custom-range thresholding was cancelled.")


def _validate_image(image: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(image):
        raise TypeError("bgr must be a PyTorch tensor")
    if image.layout != torch.strided:
        raise TypeError("bgr must be a dense strided tensor")
    if image.device.type != "cuda":
        raise ValueError("bgr must already reside on a CUDA device")
    if image.dtype not in (torch.uint8, torch.float32):
        raise TypeError("bgr must have dtype torch.uint8 or torch.float32")
    if image.ndim != 4 or tuple(int(value) for value in image.shape[:2]) != (1, 3):
        raise ValueError("bgr must have shape 1x3xHxW in BGR channel order")
    height, width = (int(value) for value in image.shape[-2:])
    if height < 1 or width < 1:
        raise ValueError("bgr must have non-empty height and width")
    pixels = height * width
    if pixels > MAX_RASTER_PIXELS:
        raise RangeResourceError(
            f"bgr exceeds the reference runner limit of {MAX_RASTER_PIXELS:,} pixels"
        )
    output_bytes = pixels * 4
    if output_bytes > MAX_OUTPUT_BYTES:
        raise RangeResourceError(
            f"custom-range outputs need {output_bytes:,} bytes; limit is {MAX_OUTPUT_BYTES:,}"
        )
    dimensions = [
        (int(stride), int(size))
        for stride, size in zip(image.stride(), image.shape)
        if int(size) > 1
    ]
    if any(stride <= 0 for stride, _ in dimensions):
        raise ValueError("bgr must use positive strides for non-singleton dimensions")
    covered = 1
    for stride, size in sorted(dimensions):
        if stride < covered:
            raise ValueError("bgr layout is not supported by the sorted-stride extent check")
        covered = stride * size
    if image.dtype == torch.float32 and not bool(torch.isfinite(image).all().item()):
        raise ValueError("float32 bgr values must all be finite")
    return image


def _mode_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("channel must be a string")
    mode = value.upper()
    if mode not in {"RGB", "HSV", "LAB", "CMYK", "GRAY", "GREY"}:
        raise UnsupportedRangeModeError(
            "channel must be RGB, HSV, LAB, CMYK, GRAY, or GREY"
        )
    return "GRAY" if mode == "GREY" else mode


def _bounds(values: Sequence[int], name: str, count: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of {count} integer byte bounds")
    if len(values) != count:
        raise ValueError(f"{name} must contain exactly {count} values for this channel")
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name}[{index}] must be an integer")
        numeric = int(value)
        if not 0 <= numeric <= 255:
            raise ValueError(f"{name}[{index}] must be between 0 and 255 inclusive")
        result.append(numeric)
    return tuple(result)


def _range_controls(channel, lower, upper):
    """Validate controls without allocating image or device buffers."""
    mode = _mode_name(channel)
    count = 4 if mode == 'CMYK' else 1 if mode == 'GRAY' else 3
    return mode, _bounds(lower, 'lower', count), _bounds(upper, 'upper', count)


def _range_admission(pixels):
    """Conservative live tile scratch, including previous-tile rebind overlap.

    512 bytes per pixel covers both overlapping conversion lifetimes, byte
    copies, selection predicates and masked tile products for all five modes.
    The fixed allowance covers compact lookup tables and comparison chunks.
    This models allocations, not instructions, time or allocator reservation.
    """
    if type(pixels) is not int or not 1 <= pixels <= MAX_RASTER_PIXELS or 4*pixels > MAX_OUTPUT_BYTES:
        raise RangeResourceError('Custom-range raster/output exceeds its native byte domain.')
    return {'tile_pixels': min(pixels, MAX_TILE_PIXELS),
        'scratch_per_pixel': 512, 'tile_scratch_bytes': 512*min(pixels, MAX_TILE_PIXELS),
        'fixed_scratch_bytes': 4*1024*1024, 'pixel_limit': MAX_RASTER_PIXELS,
        'primitive_output_bytes': 4*pixels}


class RangeMutationError(RuntimeError):
    """An ordinary PyTorch callback write invalidated an in-flight product."""


class _RangeGuard:
    def __init__(self, image, callback):
        self.callback, self.watched = callback, {}
        self.watch(image)

    @staticmethod
    def signature(tensor):
        storage = tensor.untyped_storage()
        return (tensor._version, tuple(tensor.shape), tuple(tensor.stride()),
            tensor.storage_offset(), tensor.data_ptr(), storage.data_ptr(),
            storage.nbytes(), tensor.dtype, tensor.device, tensor.layout)

    def watch(self, *tensors):
        for tensor in tensors:
            tensor.record_stream(torch.cuda.current_stream(tensor.device))
            self.watched[id(tensor)] = (tensor, self.signature(tensor))

    def intact(self):
        if any(self.signature(tensor) != expected for tensor, expected in self.watched.values()):
            raise RangeMutationError('Custom-range input or owned output changed during a callback.')

    def poll(self):
        self.intact()
        try:
            if self.callback is not None:
                stopped = self.callback()
                if type(stopped) is not bool:
                    raise TypeError('Custom-range cancellation must return bool.')
                if stopped:
                    raise RangeCancelledError('Custom-range thresholding was cancelled.')
        finally:
            self.intact()


def _gray_from_bgr(pixels: torch.Tensor) -> torch.Tensor:
    """Exact 8-bit OpenCV BGR2GRAY fixed-point transform."""
    b, g, r = (pixels[:, index].to(torch.int32) for index in range(3))
    # Pinned OpenCV 4.11 uses B=3735, G=19235, R=9798, shift=15.
    return ((b * 3735 + g * 19235 + r * 9798 + 16384) >> 15).to(torch.uint8)


def _lab_from_bgr(pixels: torch.Tensor) -> torch.Tensor:
    """Byte-exact OpenCV 4.11.0 BGR2LAB for uint8 pixels on their current device."""
    gamma, cbrt = _lab_tables(pixels.device)
    bgr = pixels.to(torch.int32)
    # OpenCV's RGB2Lab_b coefficients are initialized from its softdouble
    # D65 matrix and rounded at xyz_shift=12, with blueIdx=0 for BGR input.
    blue_xyz = (778, 296, 3575)
    green_xyz = (1541, 2929, 448)
    red_xyz = (1777, 871, 73)
    blue = gamma[bgr[:, 0]]
    green = gamma[bgr[:, 1]]
    red = gamma[bgr[:, 2]]
    f_x = cbrt[(blue * blue_xyz[0] + green * green_xyz[0] + red * red_xyz[0] + 2048) >> 12]
    f_y = cbrt[(blue * blue_xyz[1] + green * green_xyz[1] + red * red_xyz[1] + 2048) >> 12]
    f_z = cbrt[(blue * blue_xyz[2] + green * green_xyz[2] + red * red_xyz[2] + 2048) >> 12]

    lab_shift2 = 15
    l_scale = (116 * 255 + 50) // 100
    l_shift = -((16 * 255 * (1 << lab_shift2) + 50) // 100)
    descale = 1 << (lab_shift2 - 1)
    lightness = (l_scale * f_y + l_shift + descale) >> lab_shift2
    a = (500 * (f_x - f_y) + 128 * (1 << lab_shift2) + descale) >> lab_shift2
    b = (200 * (f_y - f_z) + 128 * (1 << lab_shift2) + descale) >> lab_shift2
    return torch.stack((lightness, a, b), dim=1).clamp_(0, 255).to(torch.uint8)


def _hsv_from_bgr(pixels: torch.Tensor) -> torch.Tensor:
    """OpenCV's 8-bit HSV conversion using its 12-bit reciprocal tables."""
    b, g, r = (pixels[:, index].to(torch.int32) for index in range(3))
    maximum = torch.maximum(torch.maximum(r, g), b)
    minimum = torch.minimum(torch.minimum(r, g), b)
    difference = maximum - minimum
    # OpenCV builds these integer tables by cvRound(real-valued reciprocal).
    # Python's round is nearest-even, as are cvRound's pinned table entries.
    hdiv = [0] + [round((180 * 4096) / (6 * index)) for index in range(1, 256)]
    sdiv = [0] + [round((255 * 4096) / index) for index in range(1, 256)]
    device = pixels.device
    hdiv_table = torch.tensor(hdiv, dtype=torch.int32, device=device)
    sdiv_table = torch.tensor(sdiv, dtype=torch.int32, device=device)
    saturation = (difference * sdiv_table[maximum] + 2048) >> 12
    hue_numerator = torch.where(
        maximum == r,
        g - b,
        torch.where(maximum == g, b - r + 2 * difference, r - g + 4 * difference),
    )
    hue = (hue_numerator * hdiv_table[difference] + 2048) >> 12
    hue = torch.where(hue < 0, hue + 180, hue)
    hue = torch.where(difference == 0, 0, hue)
    return torch.stack((hue, saturation, maximum), dim=1).to(torch.uint8)


def _cmyk_from_bgr(pixels: torch.Tensor) -> torch.Tensor:
    """PlantCV float64 CMYK formula, including its returned C/Y swap."""
    # A CUDA tensor divisor retains true binary64 division. PyTorch's CPU
    # scalar fast path multiplies by a rounded reciprocal on CUDA, which can
    # move the later truncating uint8 cast across an integer boundary.
    normalizer = torch.tensor(255.0, dtype=torch.float64, device=pixels.device)
    b, g, r = (pixels[:, index].to(torch.float64) / normalizer for index in range(3))
    maximum = torch.maximum(torch.maximum(b, g), r)
    k = 1.0 - maximum
    denominator = 1.0 - k
    safe_denominator = torch.where(denominator == 0, 1.0, denominator)
    c_computed = (1.0 - r - k) / safe_denominator
    m_computed = (1.0 - g - k) / safe_denominator
    y_computed = (1.0 - b - k) / safe_denominator
    cmyk = torch.stack((c_computed, m_computed, y_computed, k), dim=1)
    cmyk = torch.where(denominator[:, None] == 0, torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64, device=pixels.device), cmyk)
    byte_values = (cmyk * 255.0).to(torch.uint8)  # NumPy astype truncates toward zero.
    # _rgb2cmyk splits dstack(c,m,y,k) as y,m,c,k and returns named channels.
    return byte_values[:, (2, 1, 0, 3)]


def _channels(pixels: torch.Tensor, mode: str) -> torch.Tensor:
    if mode in {"RGB"}:
        return pixels
    if mode == "GRAY":
        return _gray_from_bgr(pixels).unsqueeze(1)
    if mode == "HSV":
        return _hsv_from_bgr(pixels)
    if mode == "LAB":
        return _lab_from_bgr(pixels)
    if mode == "CMYK":
        return _cmyk_from_bgr(pixels)
    raise AssertionError(mode)


def threshold_custom_range(
    bgr: torch.Tensor,
    channel: str,
    lower: Sequence[int],
    upper: Sequence[int],
    *,
    cancel: Callable[[], bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply pinned PlantCV custom-range thresholding to CUDA BGR pixels.

    RGB deliberately means B,G,R storage-slot thresholding, as in the pinned
    ``readimage``/``custom_range`` path. Bounds are inclusive integer bytes.
    Cancellation raises :class:`RangeCancelledError`; no partial result escapes.
    """
    mode, lower_values, upper_values = _range_controls(channel, lower, upper)
    if cancel is not None and not callable(cancel):
        raise TypeError("cancel must be a zero-argument callable or None")
    bgr = _validate_image(bgr)
    height, width = (int(value) for value in bgr.shape[-2:])
    _range_admission(height*width)
    guard = _RangeGuard(bgr, cancel)
    guard.poll()
    mask = torch.empty((height, width), dtype=torch.uint8, device=bgr.device)
    masked = torch.empty((1, 3, height, width), dtype=torch.uint8, device=bgr.device)
    guard.watch(mask, masked)

    # Tile by pixels so color conversion scratch remains bounded even for a
    # single extremely wide row. Each returned slice is fully written before
    # advancing; cancellation never returns these partially populated tensors.
    rows_per_tile = max(1, MAX_TILE_PIXELS // width)
    for y0 in range(0, height, rows_per_tile):
        y1 = min(height, y0 + rows_per_tile)
        if width <= MAX_TILE_PIXELS:
            x_ranges = ((0, width),)
        else:
            x_ranges = tuple((x, min(width, x + MAX_TILE_PIXELS)) for x in range(0, width, MAX_TILE_PIXELS))
        for x0, x1 in x_ranges:
            guard.poll()
            source = bgr[0, :, y0:y1, x0:x1]
            if source.dtype == torch.float32:
                source_bytes = source.round().clamp(0.0, 255.0).to(torch.uint8)
            else:
                source_bytes = source
            pixels = source_bytes.permute(1, 2, 0).reshape(-1, 3)
            values = _channels(pixels, mode)
            selected = torch.ones((pixels.shape[0],), dtype=torch.bool, device=bgr.device)
            for index, (low, high) in enumerate(zip(lower_values, upper_values)):
                component = values[:, index]
                selected &= (component >= low) & (component <= high)
            tile_mask = selected.reshape(y1 - y0, x1 - x0).to(torch.uint8) * 255
            mask[y0:y1, x0:x1] = tile_mask
            tile_image = torch.where(tile_mask.unsqueeze(0) != 0, source_bytes, 0)
            masked[0, :, y0:y1, x0:x1] = tile_image
            # Only these private writes are authorized between callbacks.
            guard.watch(mask, masked)
    guard.poll()
    return mask, masked


__all__ = [
    "MAX_OUTPUT_BYTES",
    "MAX_RASTER_PIXELS",
    "MAX_TILE_PIXELS",
    "PLANTCV_COMMIT",
    "PLANTCV_HELPER_REFERENCE_ENV_SHA256",
    "PLANTCV_THRESHOLD_SOURCE_SHA256",
    "PLANTCV_VERSION",
    "RangeCancelledError",
    "RangeMutationError",
    "UnsupportedRangeConversionError",
    "RangeResourceError",
    "UnsupportedRangeModeError",
    "threshold_custom_range",
]
