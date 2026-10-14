"""Standard-library numeric models for native PlantCV K-means prediction.

Only centre metadata is read on CPU. The format fixes the prediction patch
geometry and the public sigma-five preprocessing contract, independent of the
training estimator's implementation. Scientific model validation is separate.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import struct

SCHEMA = 'seedfiddle.plantcv-kmeans-centres.v1'
PREPROCESSING = 'plantcv-4.11.3-predict-sigma5-uint16-v1'
MAX_MODEL_BYTES = 8 * 1024 * 1024
MAX_CENTRE_VALUES = 1_048_576
MAX_CLUSTERS = 256
MAX_FEATURES = 4096


class KMeansModelError(ValueError):
    """Invalid numeric model or model identity."""


@dataclass(frozen=True, slots=True)
class KMeansModel:
    patch_size: int
    channels: int
    centres: tuple[tuple[float, ...], ...]
    source_sha256: str
    source_bytes: int
    centres_sha256: str


def _dimensions(patch_size, channels, centres):
    if type(patch_size) is not int or not 1 <= patch_size <= 31:
        raise KMeansModelError('patch_size must be an integer in [1,31]')
    if type(channels) is not int or channels not in (1, 3):
        raise KMeansModelError('channels must be one or three')
    features = patch_size * patch_size * channels
    if features > MAX_FEATURES or not 1 <= len(centres) <= MAX_CLUSTERS:
        raise KMeansModelError('model feature or cluster count exceeds the supported bound')
    if len(centres) * features > MAX_CENTRE_VALUES:
        raise KMeansModelError('centre table exceeds the compact metadata limit')
    if any(len(row) != features for row in centres):
        raise KMeansModelError('centre width must equal patch_size squared times channels')


def _centres_digest(patch_size, channels, centres):
    digest = sha256()
    digest.update((SCHEMA + '\0' + PREPROCESSING + '\0float64\0').encode('ascii'))
    digest.update(struct.pack('>III', patch_size, channels, len(centres)))
    for row in centres:
        for value in row:
            digest.update(struct.pack('>d', value))
    return digest.hexdigest()


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise KMeansModelError(f'duplicate model field: {key}')
        result[key] = value
    return result


def _nonfinite(value):
    raise KMeansModelError(f'nonfinite JSON number: {value}')


def parse_kmeans_model(source):
    """Parse an at-most-eight-MiB UTF-8 centres-only model with exact identity."""
    if isinstance(source, str):
        if len(source) > MAX_MODEL_BYTES:
            raise KMeansModelError('model exceeds the compact byte limit')
        try:
            raw = source.encode('utf-8')
        except UnicodeEncodeError as error:
            raise KMeansModelError('model must be UTF-8') from error
    elif isinstance(source, bytes):
        raw = source
    else:
        raise KMeansModelError('provide numeric model UTF-8 text or bytes')
    if not 0 < len(raw) <= MAX_MODEL_BYTES:
        raise KMeansModelError('model is empty or exceeds the compact byte limit')
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_object_pairs,
                           parse_constant=_nonfinite)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise KMeansModelError(f'invalid numeric model: {error}') from error
    fields = {'schema', 'preprocessing', 'dtype', 'patch_size', 'channels', 'centres'}
    if type(value) is not dict or set(value) != fields:
        raise KMeansModelError('model requires only schema, preprocessing, dtype, patch_size, channels and centres')
    if value['schema'] != SCHEMA or value['preprocessing'] != PREPROCESSING or value['dtype'] != 'float64':
        raise KMeansModelError('unsupported model schema, preprocessing or centre dtype')
    rows = value['centres']
    if type(rows) is not list or any(type(row) is not list for row in rows):
        raise KMeansModelError('centres must be a numeric row array')
    _dimensions(value['patch_size'], value['channels'], rows)
    centres = []
    for row in rows:
        normalized = []
        for item in row:
            if type(item) not in (int, float):
                raise KMeansModelError('centre values must be numeric scalars')
            try:
                number = float(item)
            except (ValueError, OverflowError) as error:
                raise KMeansModelError('unsupported centre value') from error
            if not math.isfinite(number) or not 0.0 <= number <= 255.0:
                raise KMeansModelError('centre values must be finite and in [0,255]')
            normalized.append(number)
        centres.append(tuple(normalized))
    centres = tuple(centres)
    return KMeansModel(value['patch_size'], value['channels'], centres,
                       sha256(raw).hexdigest(), len(raw),
                       _centres_digest(value['patch_size'], value['channels'], centres))


def load_kmeans_model(path):
    """Bound the read itself, including files that change after being selected."""
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(MAX_MODEL_BYTES + 1)
    except OSError as error:
        raise KMeansModelError(f'unable to read numeric K-means model: {error}') from error
    return parse_kmeans_model(raw)


def model_from_centres(centres, patch_size, channels):
    """Encode ordered numeric centre metadata into the same canonical model format."""
    if not isinstance(centres, (tuple, list)) or any(not isinstance(row, (tuple, list)) for row in centres):
        raise KMeansModelError('centres must be numeric rows')
    _dimensions(patch_size, channels, centres)
    normalized = []
    for row in centres:
        values = []
        for value in row:
            if type(value) not in (int, float):
                raise KMeansModelError('centre values must be numeric scalars')
            try:
                value = float(value)
            except (ValueError, OverflowError) as error:
                raise KMeansModelError('unsupported centre value') from error
            if not math.isfinite(value) or not 0.0 <= value <= 255.0:
                raise KMeansModelError('centre values must be finite and in [0,255]')
            values.append(value)
        normalized.append(tuple(values))
    try:
        raw = json.dumps(dict(schema=SCHEMA, preprocessing=PREPROCESSING, dtype='float64',
                              patch_size=patch_size, channels=channels, centres=normalized),
                         separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise KMeansModelError('unsupported numeric centre metadata') from error
    return parse_kmeans_model(raw)


def encode_kmeans_model(model):
    """Return bounded canonical UTF-8 metadata, preserving ordered numeric identity.

    Canonical bytes acquire their own raw hash when parsed. Re-encoding a file
    can change its raw hash/length while preserving its numeric table digest.
    """
    validate_kmeans_model(model)
    header = json.dumps(dict(schema=SCHEMA, preprocessing=PREPROCESSING, dtype='float64',
        patch_size=model.patch_size, channels=model.channels), separators=(',', ':'), allow_nan=False)
    encoder = json.JSONEncoder(separators=(',', ':'), allow_nan=False)
    output = bytearray()

    def append_text(text):
        encoded = text.encode('utf-8')
        if len(encoded) > MAX_MODEL_BYTES-len(output):
            raise KMeansModelError('canonical K-means model exceeds the byte limit')
        output.extend(encoded)

    append_text(header[:-1])
    append_text(',"centres":[')
    for index, row in enumerate(model.centres):
        if index:
            append_text(',')
        for fragment in encoder.iterencode(row):
            append_text(fragment)
    append_text(']}')
    return bytes(output)


def validate_kmeans_model(model):
    """Recheck immutable table identity at every direct native prediction call."""
    if type(model) is not KMeansModel or type(model.centres) is not tuple:
        raise KMeansModelError('expected an immutable parsed K-means model')
    if any(type(row) is not tuple for row in model.centres):
        raise KMeansModelError('model centre rows must be immutable tuples')
    _dimensions(model.patch_size, model.channels, model.centres)
    if (type(model.source_bytes) is not int or not 0 < model.source_bytes <= MAX_MODEL_BYTES
            or type(model.source_sha256) is not str or not re.fullmatch('[0-9a-f]{64}', model.source_sha256)
            or type(model.centres_sha256) is not str or not re.fullmatch('[0-9a-f]{64}', model.centres_sha256)):
        raise KMeansModelError('invalid model source/table identity')
    for row in model.centres:
        if any(type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 255.0 for value in row):
            raise KMeansModelError('unsupported frozen centre value')
    if _centres_digest(model.patch_size, model.channels, model.centres) != model.centres_sha256:
        raise KMeansModelError('model centres differ from their frozen numeric digest')
