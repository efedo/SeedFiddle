"""Byte-identified OpenCV input; owned-array caps are not codec/RSS bounds."""
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import cv2
import numpy as np

MAX_ENCODED_BYTES = 256*1024*1024
READ_CHUNK_BYTES = 1024*1024
DECODE_FLAGS = 1
GRAY_DECODE_FLAGS = 0
MAX_DECODED_BYTES = 256*1024*1024
PLANTCV_VERSION = '4.11.3'
PLANTCV_READER_SHA256 = '09035b0caa1f91a503ecda82e39e082dfec2d84f78c8f8fdc8c3fe3854fbe770'
READER_MODE = 'rgb'
COLOR_CONTRACT = 'PlantCV readimage(mode=rgb): OpenCV IMREAD_COLOR BGR uint8; alpha dropped, depth quantized, decoder EXIF orientation; first decoded image.'
GRAY_CONTRACT = 'Pinned BGR2GRAY applied to the color-decoded byte raster; not codec-direct IMREAD_GRAYSCALE.'
CODEC_GRAY_CONTRACT = 'PlantCV readimage(mode=gray): OpenCV IMREAD_GRAYSCALE uint8; decoder EXIF orientation; first decoded image.'

class ImageInputError(ValueError):
    """Encoded identity or decoded color input is unavailable."""

@dataclass(frozen=True, slots=True)
class SourceIdentity:
    size: int
    sha256: str

def versions():
    return {'opencv':str(cv2.__version__),'opencv-python':version('opencv-python'),
        'numpy':str(np.__version__),'flags':DECODE_FLAGS,'reader_mode':READER_MODE,
        'gray_flags':GRAY_DECODE_FLAGS,
        'opencv_build_sha256':sha256(cv2.getBuildInformation().encode('utf-8')).hexdigest(),
        'plantcv_reader_sha256':PLANTCV_READER_SHA256}

def stamp(path):
    try:
        value=Path(path).stat()
        return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    except OSError:
        return None

def _poll(poll, metadata=()):
    if poll is not None:poll(metadata=metadata)

def _size(path):
    size=Path(path).stat().st_size
    if not 1<=size<=MAX_ENCODED_BYTES:
        raise ImageInputError('Native image files require 1..256MiB encoded bytes.')
    return size

def source_identity(path, poll=None):
    _poll(poll)
    _size(path)
    digest=sha256();size=0
    with Path(path).open('rb') as stream:
        while True:
            _poll(poll)
            block=stream.read(min(READ_CHUNK_BYTES,MAX_ENCODED_BYTES+1-size))
            if not block:break
            size+=len(block)
            if size>MAX_ENCODED_BYTES:
                raise ImageInputError('Native image encoded bytes exceed the 256MiB limit.')
            digest.update(block)
    if size<1:raise ImageInputError('The selected image is empty.')
    return SourceIdentity(size,digest.hexdigest())

def snapshot(path, expected, poll=None):
    if (not isinstance(expected,SourceIdentity) or type(expected.size) is not int
            or not 1<=expected.size<=MAX_ENCODED_BYTES or type(expected.sha256) is not str
            or len(expected.sha256)!=64 or any(c not in '0123456789abcdef' for c in expected.sha256)):
        raise ImageInputError('An image snapshot requires a bounded byte identity.')
    chunks=[];size=0
    with Path(path).open('rb') as stream:
        while True:
            _poll(poll,metadata=(expected,chunks))
            block=stream.read(min(READ_CHUNK_BYTES,expected.size+1-size))
            if not block:break
            chunks.append(block);size+=len(block)
            if size>expected.size:raise ImageInputError('Source image changed during its byte snapshot.')
    raw=b''.join(chunks)
    if len(raw)!=expected.size or sha256(raw).hexdigest()!=expected.sha256:
        raise ImageInputError('Source image changed during its byte snapshot.')
    return raw

def decode(raw):
    if type(raw) is not bytes or not 1<=len(raw)<=MAX_ENCODED_BYTES:
        raise ImageInputError('Decode requires one bounded immutable image snapshot.')
    try:
        decoded=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),DECODE_FLAGS)
    except cv2.error as error:
        raise ImageInputError('The image decoder could not read the selected bytes.') from error
    if (not isinstance(decoded,np.ndarray) or decoded.ndim!=3 or decoded.shape[-1]!=3
            or decoded.dtype!=np.uint8 or min(decoded.shape[:2])<1 or not decoded.flags.c_contiguous):
        raise ImageInputError('The image decoder could not produce a contiguous BGR uint8 raster.')
    return decoded


def decoded_bound(pixels, *, color=True, gray=False):
    """Bound owned arrays before the second decode or CUDA upload."""
    if type(pixels) is not int or pixels < 1 or pixels*(3*color+gray) > MAX_DECODED_BYTES:
        raise ImageInputError('Owned decoded image arrays exceed the 256MiB limit.')


def decode_gray(raw, shape):
    if type(raw) is not bytes or not 1<=len(raw)<=MAX_ENCODED_BYTES:
        raise ImageInputError('Decode requires one bounded immutable image snapshot.')
    try:
        decoded=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),GRAY_DECODE_FLAGS)
    except cv2.error as error:
        raise ImageInputError('The grayscale decoder could not read the selected bytes.') from error
    if (not isinstance(decoded,np.ndarray) or decoded.ndim!=2 or decoded.dtype!=np.uint8
            or tuple(decoded.shape)!=tuple(shape) or min(decoded.shape)<1 or not decoded.flags.c_contiguous):
        raise ImageInputError('Grayscale and color decoders must produce the same contiguous uint8 frame.')
    return decoded
