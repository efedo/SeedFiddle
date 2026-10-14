"""Shared four-file correction inputs and execution for desktop and project jobs."""
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Mapping

import cv2
import numpy as np
import torch
from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QImageReader

from seedvision.benchmarking.checkpoints import CoordinateFrame
from seedvision.benchmarking.native_correction_recipe import (
    MAX_COMBINED_PIXELS, NativeCorrectionCancelled, NativeCorrectionError,
    RECIPE_ID, STAGE_ID, run_native_correction_recipe, _validate_recipe, _correction_output_signatures,
)
from seedvision.benchmarking.native_recipe import _tensor_signature, _module_record, _digest
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.model_io.asset_paths import validate_model_origin

_PORTS = ("target_rgb", "target_mask", "source_rgb", "source_mask")
_TITLES = {"target_rgb": "Target RGB image", "target_mask": "Target chip-label mask",
    "source_rgb": "Source RGB image", "source_mask": "Source chip-label mask"}
_CORRECTION_CACHE_KEY = f"reference.recipe:{RECIPE_ID}:{STAGE_ID}"
_HASH_CHUNK_BYTES = 1024*1024
# One explicitly capped prepared bundle; path changes release it.
MAX_PREPARED_CUDA_BYTES = 4*MAX_COMBINED_PIXELS
MAX_INPUT_FILE_BYTES = 256*1024*1024
MAX_DECODED_CPU_BYTES = 8*MAX_COMBINED_PIXELS


@dataclass(frozen=True, slots=True)
class NativeCorrectionInputs:
    file_paths: Mapping[str, Path]
    file_sha256s: Mapping[str, str]
    file_bytes: Mapping[str, int]
    tensors: Mapping[str, torch.Tensor]
    coordinate_frames: Mapping[str, CoordinateFrame]
    tensor_signatures: Mapping[str, tuple]
    preparation: Mapping[str, str]
    preparation_sha256: str
    decoder_versions: Mapping[str, str]
    device: torch.device
    cuda_bytes: int


def _cancel(cancel):
    if cancel is not None and cancel():
        raise NativeCorrectionCancelled("Correction input preparation was cancelled.")


def _hash_file(path, cancel=None):
    digest = sha256()
    count = 0
    with path.open("rb") as stream:
        while True:
            _cancel(cancel)
            block = stream.read(min(_HASH_CHUNK_BYTES, MAX_INPUT_FILE_BYTES+1-count))
            if not block:
                break
            count += len(block)
            if count > MAX_INPUT_FILE_BYTES:
                raise NativeCorrectionError('Correction input file exceeds its 256MiB byte limit.')
            digest.update(block)
    _cancel(cancel)
    return digest.hexdigest()


def _resolve_paths(paths):
    if not isinstance(paths, Mapping) or set(paths) != set(_PORTS):
        raise NativeCorrectionError("Select four explicit target/source image and chip-mask files.")
    resolved = {}
    for name in _PORTS:
        try:
            path = Path(paths[name]).expanduser().resolve(strict=True)
        except (OSError, TypeError, ValueError) as error:
            raise NativeCorrectionError(f"{_TITLES[name]} path is unavailable.") from error
        if not path.is_file():
            raise NativeCorrectionError(f"{_TITLES[name]} must be a regular file.")
        resolved[name] = path
    return resolved


def _file_hashes(paths, cancel=None):
    return {name: _hash_file(paths[name], cancel) for name in _PORTS}


def _read_decode_bytes(path, expected_size, expected_hash, cancel):
    """Decode the exact bounded bytes whose identity was selected for this run."""
    raw, digest = bytearray(), sha256()
    with path.open('rb') as stream:
        while True:
            _cancel(cancel)
            block = stream.read(min(_HASH_CHUNK_BYTES, expected_size+1-len(raw)))
            if not block:
                break
            raw.extend(block)
            if len(raw) > expected_size:
                raise NativeCorrectionError('An input file changed while OpenCV was decoding the correction inputs.')
            digest.update(block)
    if len(raw) != expected_size or digest.hexdigest() != expected_hash:
        raise NativeCorrectionError('An input file changed while OpenCV was decoding the correction inputs.')
    _cancel(cancel)
    return raw


def _signatures(tensors):
    return {name: _tensor_signature(tensors[name]) for name in _PORTS}


def _decode_header(raw, max_pixels):
    buffer = QBuffer()
    buffer.setData(QByteArray(raw))
    buffer.open(QIODevice.OpenModeFlag.ReadOnly)
    reader = QImageReader(buffer)
    format_name = bytes(reader.format()).decode('ascii', errors='replace').lower()
    size = reader.size()
    height, width = size.height(), size.width()
    if format_name not in {'png', 'jpeg', 'bmp', 'tiff'} or min(height, width) < 1:
        raise NativeCorrectionError('Correction inputs require a bounded PNG/JPEG/BMP/TIFF raster header.')
    if height*width > max_pixels:
        raise NativeCorrectionError('An input image exceeds the 64-million-pixel correction budget.')
    if reader.imageFormat().name not in {'Format_Mono', 'Format_MonoLSB', 'Format_Indexed8',
            'Format_RGB32', 'Format_ARGB32', 'Format_ARGB32_Premultiplied', 'Format_RGB888',
            'Format_BGR888', 'Format_RGBX8888', 'Format_RGBA8888',
            'Format_RGBA8888_Premultiplied', 'Format_Alpha8', 'Format_Grayscale8', 'Format_CMYK8888'}:
        raise NativeCorrectionError('Correction inputs require an eight-bit raster header before decoding.')
    return height, width


def _resolve_device(device):
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise NativeCorrectionError("Correction input preparation requires an available CUDA device.")
    index = torch.cuda.current_device() if selected.index is None else selected.index
    if index < 0 or index >= torch.cuda.device_count():
        raise NativeCorrectionError("Requested CUDA device is unavailable.")
    return torch.device("cuda", index)


def _same_prepared(previous, paths, hashes, sizes, device, preparation_sha256, decoder_versions):
    return (isinstance(previous, NativeCorrectionInputs)
        and previous.device == device
        and dict(previous.file_paths) == dict(paths)
        and dict(previous.file_sha256s) == dict(hashes)
        and dict(previous.file_bytes) == dict(sizes)
        and previous.preparation_sha256 == preparation_sha256
        and dict(previous.decoder_versions) == dict(decoder_versions)
        and dict(previous.tensor_signatures) == _signatures(previous.tensors))


def _preparation_identity():
    try:
        record = _digest(_module_record(sys.modules[__name__]))
    except ValueError as error:
        raise NativeCorrectionError('Correction input preparation differs from its verified source.') from error
    versions = {'opencv-python': version('opencv-python'), 'PySide6': version('PySide6'),
        'numpy': version('numpy'), 'torch': str(torch.__version__), 'torch_cuda': str(torch.version.cuda)}
    return record, versions


def prepare_native_correction_inputs(paths, *, previous=None, device="cuda", cancel=None,
                                     max_combined_pixels=MAX_COMBINED_PIXELS,
                                     max_prepared_cuda_bytes=MAX_PREPARED_CUDA_BYTES):
    """Validate four explicit decoded files before CUDA transfer.

    Qt reads bounded headers; OpenCV decodes the same byte snapshot as BGR images
    and unchanged uint8 masks on CPU. Image channel
    reversal runs on CUDA. Masks must match their corresponding image exactly.
    Before/after file hashes and tensor versions establish preparation reuse.
    """
    if cancel is not None and not callable(cancel):
        raise TypeError("cancel must be a zero-argument callable or None.")
    for value, maximum in ((max_combined_pixels, MAX_COMBINED_PIXELS),
                           (max_prepared_cuda_bytes, MAX_PREPARED_CUDA_BYTES)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise NativeCorrectionError('Correction input limits must be positive integers within the hard bounds.')
    _cancel(cancel)
    resolved, selected_device = _resolve_paths(paths), _resolve_device(device)
    sizes = {name: path.stat().st_size for name, path in resolved.items()}
    if any(not 1 <= size <= MAX_INPUT_FILE_BYTES for size in sizes.values()):
        raise NativeCorrectionError('Correction input files require 1..256MiB each.')
    before = _file_hashes(resolved, cancel)
    preparation_sha, decoder_versions = _preparation_identity()
    if _same_prepared(previous, resolved, before, sizes, selected_device, preparation_sha, decoder_versions):
        pixels = sum(previous.coordinate_frames[name].shape[0]*previous.coordinate_frames[name].shape[1]
                     for name in ('target_rgb', 'source_rgb'))
        if pixels > max_combined_pixels or previous.cuda_bytes > max_prepared_cuda_bytes:
            raise NativeCorrectionError('Cached correction inputs exceed the requested preparation budget.')
        _cancel(cancel)
        return previous
    decoded, shapes, header_bytes = {}, {}, 0
    for name in _PORTS:
        _cancel(cancel)
        raw = _read_decode_bytes(resolved[name], sizes[name], before[name], cancel)
        header = _decode_header(raw, max_combined_pixels)
        header_bytes += 4*header[0]*header[1]
        if header_bytes > min(MAX_DECODED_CPU_BYTES, 8*max_combined_pixels):
            raise NativeCorrectionError('Correction inputs exceed the aggregate decoded CPU-byte bound.')
        _cancel(cancel)
        flag = cv2.IMREAD_COLOR if name.endswith('_rgb') else cv2.IMREAD_UNCHANGED
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flag)
        del raw
        _cancel(cancel)
        if isinstance(image, np.ndarray) and tuple(image.shape[:2]) != header:
            raise NativeCorrectionError('Decoded correction pixels differ from their bounded raster header.')
        if name.endswith("_rgb"):
            if (not isinstance(image, np.ndarray) or image.ndim != 3
                    or image.shape[-1] != 3 or image.dtype != np.uint8
                    or min(image.shape[:2]) < 1):
                raise NativeCorrectionError(f"{_TITLES[name]} could not be decoded as an OpenCV BGR uint8 image.")
            shapes[name] = tuple(image.shape[:2])
            if image.shape[0]*image.shape[1] > max_combined_pixels:
                raise NativeCorrectionError("An input image exceeds the 64-million-pixel correction budget.")
            decoded[name] = np.ascontiguousarray(image)
        else:
            mask = image
            if (not isinstance(mask, np.ndarray) or mask.ndim != 2
                    or mask.dtype != np.uint8 or min(mask.shape) < 1):
                raise NativeCorrectionError(
                    f"{_TITLES[name]} must decode with IMREAD_UNCHANGED as a single-channel uint8 mask.")
            decoded[name] = np.ascontiguousarray(mask)
    for image_name, mask_name in (("target_rgb", "target_mask"), ("source_rgb", "source_mask")):
        if tuple(decoded[mask_name].shape) != shapes[image_name]:
            raise NativeCorrectionError(
                f"{_TITLES[mask_name]} shape must exactly match its corresponding RGB image.")
    if shapes["target_rgb"][0] != shapes["source_rgb"][0]:
        raise NativeCorrectionError(
            "Pinned correct_color requires equal target/source image heights for its diagnostic montage.")
    combined_pixels = sum(shapes[name][0]*shapes[name][1] for name in ("target_rgb", "source_rgb"))
    if combined_pixels > max_combined_pixels:
        raise NativeCorrectionError("Target and source images exceed the 64-million-pixel correction budget.")
    cuda_bytes = sum(value.nbytes for value in decoded.values())
    if cuda_bytes > max_prepared_cuda_bytes:
        raise NativeCorrectionError("Prepared correction inputs exceed their bounded CUDA residency budget.")
    if (_file_hashes(resolved, cancel) != before
            or {name: path.stat().st_size for name, path in resolved.items()} != sizes):
        raise NativeCorrectionError("An input file changed while OpenCV was decoding the correction inputs.")
    tensors = {}
    for name in _PORTS:
        _cancel(cancel)
        if name.endswith("_rgb"):
            bgr = torch.as_tensor(decoded[name], device=selected_device)
            tensors[name] = bgr.flip(-1).contiguous()
            del bgr
        else:
            tensors[name] = torch.as_tensor(decoded[name], device=selected_device).contiguous()
    target_frame = CoordinateFrame("target.decoded", shapes["target_rgb"],
        "top-left", "x-right,y-down", "pixel")
    source_frame = CoordinateFrame("source.decoded", shapes["source_rgb"],
        "top-left", "x-right,y-down", "pixel")
    frames = {"target_rgb": target_frame, "target_mask": target_frame,
        "source_rgb": source_frame, "source_mask": source_frame}
    preparation = {}
    for name in _PORTS:
        decoder = (f"OpenCV {cv2.__version__} IMREAD_COLOR BGR on CPU; "
            "CUDA uint8 transfer then BGR-to-RGB channel reversal") if name.endswith("_rgb") else (
            f"OpenCV {cv2.__version__} IMREAD_UNCHANGED; single-channel uint8 chip labels transferred to CUDA")
        preparation[name] = (f"{decoder}; path={resolved[name]}; sha256={before[name]}; "
            f"frame={frames[name].frame_id} {frames[name].shape}; device={selected_device}")
    signatures = _signatures(tensors)
    _cancel(cancel)
    return NativeCorrectionInputs(MappingProxyType(dict(resolved)),
        MappingProxyType(dict(before)), MappingProxyType(sizes), MappingProxyType(tensors),
        MappingProxyType(frames), MappingProxyType(signatures),
        MappingProxyType(preparation), preparation_sha, MappingProxyType(decoder_versions), selected_device, cuda_bytes)


def validate_correction_input_sources(value):
    """Validate four named byte/frame/decode records without probing their paths."""
    import re
    if not isinstance(value, Mapping) or set(value) != set(_PORTS):
        raise NativeCorrectionError('Correction output requires all four input-source records.')
    result, frames = {}, {}
    for name in _PORTS:
        row = value[name]
        if not isinstance(row, Mapping) or set(row) != {'path', 'sha256', 'bytes', 'frame', 'decode'}:
            raise NativeCorrectionError('Correction input-source record has missing or unknown fields.')
        if (type(row['path']) is not str or not row['path'].strip() or len(row['path']) > 4096
                or '\0' in row['path'] or type(row['sha256']) is not str
                or re.fullmatch('[0-9a-f]{64}', row['sha256']) is None
                or type(row['bytes']) is not int or not 1 <= row['bytes'] <= MAX_INPUT_FILE_BYTES
                or type(row['decode']) is not str or not row['decode'].strip() or len(row['decode']) > 8192):
            raise NativeCorrectionError('Correction input-source byte/path/decode identity is invalid.')
        try:
            validate_model_origin(row['path'])
        except ValueError as error:
            raise NativeCorrectionError('Correction input-source paths must be canonical absolute text.') from error
        if not isinstance(row['frame'], Mapping):
            raise NativeCorrectionError('Correction input-source frame must be a mapping.')
        frame = CoordinateFrame.from_dict(dict(row['frame']))
        if (frame.origin != 'top-left' or frame.axes != 'x-right,y-down' or frame.units != 'pixel'
                or frame.transform_sha256 is not None
                or frame.frame_id != ('target.decoded' if name.startswith('target_') else 'source.decoded')):
            raise NativeCorrectionError('Correction input-source frame must identify unchanged decoded pixels.')
        frames[name] = frame
        result[name] = dict(row) | {'frame': frame.to_dict()}
    if (frames['target_rgb'] != frames['target_mask'] or frames['source_rgb'] != frames['source_mask']
            or frames['target_rgb'].shape[0] != frames['source_rgb'].shape[0]
            or sum(frames[name].shape[0]*frames[name].shape[1] for name in ('target_rgb', 'source_rgb')) > MAX_COMBINED_PIXELS):
        raise NativeCorrectionError('Correction input-source coordinate frames are inconsistent.')
    return result


def run_native_correction_files(recipe, paths, *, previous=None, device='cuda', cache=None,
                                cancel=None, max_combined_pixels=MAX_COMBINED_PIXELS,
                                max_prepared_cuda_bytes=MAX_PREPARED_CUDA_BYTES):
    """Shared four-file execution with byte rechecks before cache reuse and return.

    Returns the actual result and reusable CUDA input bundle. Every role retains
    its own raw bytes and frame, including the two chip masks.
    """
    if cancel is not None and not callable(cancel):
        raise TypeError('cancel must be a zero-argument callable or None.')
    _cancel(cancel)
    _validate_recipe(recipe)
    if cache is None:
        cache = PipelineAnalysisCache()
    if not isinstance(cache, PipelineAnalysisCache):
        raise TypeError('cache must be a PipelineAnalysisCache or None.')
    try:
        prepared = prepare_native_correction_inputs(paths, previous=previous, device=device, cancel=cancel,
            max_combined_pixels=max_combined_pixels, max_prepared_cuda_bytes=max_prepared_cuda_bytes)
        if prepared is not previous:
            cache.values.pop(_CORRECTION_CACHE_KEY, None)
        result = run_native_correction_recipe(recipe,
            prepared.tensors['target_rgb'], prepared.tensors['target_mask'],
            prepared.tensors['source_rgb'], prepared.tensors['source_mask'],
            target_source_sha256=prepared.file_sha256s['target_rgb'],
            source_sha256=prepared.file_sha256s['source_rgb'],
            target_coordinate_frame=prepared.coordinate_frames['target_rgb'],
            coordinate_frame=prepared.coordinate_frames['source_rgb'], cache=cache, cancel=cancel)
        output_signatures = _correction_output_signatures(result.node(STAGE_ID).outputs)
        _cancel(cancel)
        sources = {name: {'path': str(prepared.file_paths[name]), 'sha256': prepared.file_sha256s[name],
            'bytes': prepared.file_bytes[name], 'frame': prepared.coordinate_frames[name].to_dict(),
            'decode': prepared.preparation[name]} for name in _PORTS}
        sources = validate_correction_input_sources(sources)
        sources = MappingProxyType({name: MappingProxyType(row | {'frame': MappingProxyType(
            row['frame'] | {'shape': tuple(row['frame']['shape'])})}) for name, row in sources.items()})
        # No caller callback may run after the final implementation, byte and
        # version guards. Preparation hashing remains cancellable per chunk;
        # the bounded final read is an indivisible publication audit.
        _cancel(cancel)
        preparation_sha, decoder_versions = _preparation_identity()
        if (preparation_sha != prepared.preparation_sha256
                or decoder_versions != dict(prepared.decoder_versions)):
            raise NativeCorrectionError('Correction input preparation changed while correction was running.')
        runtime = dict(result.runtime) | {'correction_tensor_executor': result.implementation_sha256,
            'correction_input_preparation': preparation_sha, 'PySide6': decoder_versions['PySide6']}
        implementation = correction_file_implementation_sha256(runtime)
        if (_file_hashes(prepared.file_paths) != dict(prepared.file_sha256s)
                or any(path.stat().st_size != prepared.file_bytes[name] for name, path in prepared.file_paths.items())):
            raise NativeCorrectionError('An input file changed while correction was running.')
        if _signatures(prepared.tensors) != dict(prepared.tensor_signatures):
            raise NativeCorrectionError('A prepared correction tensor changed while correction was running.')
        if _correction_output_signatures(result.node(STAGE_ID).outputs) != output_signatures:
            raise NativeCorrectionError('A correction output changed before the four-file result returned.')
        return replace(result, preparation=prepared.preparation, input_sources=sources,
            runtime=MappingProxyType(runtime), implementation_sha256=implementation), prepared
    except Exception:
        cache.values.pop(_CORRECTION_CACHE_KEY, None)
        raise


def correction_file_implementation_sha256(runtime):
    import re
    for name in ('correction_tensor_executor', 'correction_input_preparation'):
        if not isinstance(runtime.get(name), str) or re.fullmatch('[0-9a-f]{64}', runtime[name]) is None:
            raise NativeCorrectionError('Correction file execution requires verified producer identities.')
    if not isinstance(runtime.get('PySide6'), str) or not runtime['PySide6'].strip():
        raise NativeCorrectionError('Correction file execution requires its Qt header-reader version.')
    return _digest({'tensor_core_sha256': runtime['correction_tensor_executor'],
        'preparation_sha256': runtime['correction_input_preparation'], 'runtime': dict(runtime)})
