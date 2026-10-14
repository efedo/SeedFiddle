"""Bounded sampled-RGB table training and interoperable PlantCV PDF output.

Input text is decoded on CPU; HSV samples and Gaussian training stay CUDA.
This API implements the multiclass sample-table entry point. Image/mask directory
sampling, plotting, desktop training and project-held training provenance are
separate integration work, not implied by this component.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import json
from pathlib import Path
import struct

import numpy as np
import torch

from seedvision.cuda import _nvrtc
from seedvision.cuda import plantcv_naive_bayes as _model
from seedvision.cuda import plantcv_naive_bayes_training as _fit
from seedvision.cuda import plantcv_range as _range

MAX_SAMPLE_BYTES = 16 * 1024 * 1024
MAX_LINE_BYTES = 64 * 1024
MAX_LABEL_BYTES = 255
CONVERSION_TILE = 16_384
PARSER_ID = 'plantcv-4.11.3-ordered-sampled-rgb-v1'


@dataclass(frozen=True, slots=True)
class BayesSampleTable:
    source_sha256: str
    source_bytes: int
    headers: tuple[str, ...]
    class_names: tuple[str, ...]
    rgb_bytes: tuple[bytes, ...]
    sample_counts: tuple[int, ...]
    parsed_sha256: str


@dataclass(frozen=True, slots=True)
class BayesTrainingCompletion:
    samples: tuple[torch.Tensor, ...]
    fit: _fit.BayesTrainingFit
    model: _model.NaiveBayesModel
    tsv_bytes: bytes
    # Canonical immutable JSON. Origin is provenance, not numeric model identity.
    provenance_json: bytes


def _poll(cancel):
    if cancel is not None and not callable(cancel):
        raise _fit.BayesTrainingError('cancel must be callable or None')
    answer = cancel() if cancel is not None else False
    if type(answer) is not bool:
        raise _fit.BayesTrainingError('cancel callback must return bool')
    if answer:
        raise _fit.BayesTrainingCancelled('Bayes sample training was cancelled')
    return False


def _sample_digest(names, data):
    digest = sha256()
    for name, pixels in zip(names, data):
        encoded = name.encode('utf-8')
        digest.update(struct.pack('>I', len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack('>I', len(pixels)))
        digest.update(pixels)
    return digest.hexdigest()


def parse_bayes_samples(source: bytes | str, *, cancel=None) -> BayesSampleTable:
    """Match supported public row/column semantics without cleaning labels.

    Duplicate headers combine columns in row/column order. Missing and empty
    cells are skipped; row quotes are removed, header quotes retained. Labels
    must round-trip through the existing PDF loader. Out-of-byte RGB values are
    rejected, matching the pinned NumPy 2.2.6 conversion rather than wrapping.
    """
    if cancel is not None and not callable(cancel):
        raise _fit.BayesTrainingError('cancel must be callable or None')
    _poll(cancel)
    if type(source) is str:
        if len(source) > MAX_SAMPLE_BYTES:
            raise _fit.BayesTrainingError('sample table exceeds the input byte cap')
        try:
            raw = source.encode('utf-8')
        except UnicodeError as error:
            raise _fit.BayesTrainingError('sample table must be UTF-8') from error
    elif type(source) is bytes:
        raw = source
    else:
        raise _fit.BayesTrainingError('sample table must be UTF-8 text or immutable bytes')
    if not 0 < len(raw) <= MAX_SAMPLE_BYTES:
        raise _fit.BayesTrainingError('sample table is empty or exceeds the input byte cap')
    try:
        text = raw.decode('utf-8')
    except UnicodeError as error:
        raise _fit.BayesTrainingError('sample table must be UTF-8') from error
    stream = io.StringIO(text, newline=None)
    header = stream.readline().rstrip('\n')
    headers = tuple(header.split('\t'))
    if (len(header.encode('utf-8')) > MAX_LINE_BYTES or len(headers) > _model.MAX_CLASSES
            or any(not name or len(name.encode('utf-8')) > MAX_LABEL_BYTES for name in headers)):
        raise _fit.BayesTrainingError('sample headers exceed supported nonempty label/column limits')
    classes = dict.fromkeys(headers)
    data = {name: bytearray() for name in classes}
    total = 0
    for row_number, row in enumerate(stream, start=2):
        if row_number % 128 == 0:
            _poll(cancel)
        if len(row.encode('utf-8')) > MAX_LINE_BYTES:
            raise _fit.BayesTrainingError(f'sample row {row_number} exceeds the line byte cap')
        row = row.rstrip('\n').replace('"', '')
        if not row:
            continue
        for column, point in enumerate(row.split('\t')):
            if not point:
                continue
            try:
                red, green, blue = map(int, point.split(','))
            except ValueError as error:
                raise _fit.BayesTrainingError(f'sample row {row_number} requires integer RGB triples') from error
            if column >= len(headers):
                raise _fit.BayesTrainingError(f'sample row {row_number} has an extra nonempty class column')
            if any(value < 0 or value > 255 for value in (red, green, blue)):
                raise _fit.BayesTrainingError(f'sample row {row_number} is outside the uint8 RGB domain')
            total += 1
            if total > _fit.MAX_SAMPLES:
                raise _fit.BayesTrainingError('sample table exceeds the global sample count cap')
            data[headers[column]].extend((red, green, blue))
    names = tuple(classes)
    pixels = tuple(bytes(data[name]) for name in names)
    _poll(cancel)
    return BayesSampleTable(sha256(raw).hexdigest(), len(raw), headers, names, pixels,
        tuple(len(p)//3 for p in pixels), _sample_digest(names, pixels))


def read_bayes_samples(path: str | Path, *, cancel=None) -> BayesSampleTable:
    _poll(cancel)
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(MAX_SAMPLE_BYTES+1)
    except OSError as error:
        raise _fit.BayesTrainingError(f'unable to read sample table: {error}') from error
    return parse_bayes_samples(raw, cancel=cancel)


def _validate_table(table):
    if (type(table) is not BayesSampleTable or type(table.class_names) is not tuple
            or type(table.rgb_bytes) is not tuple or type(table.headers) is not tuple
            or type(table.sample_counts) is not tuple
            or type(table.source_bytes) is not int
            or any(type(name) is not str for name in table.headers)
            or any(type(n) is not int or n < 0 for n in table.sample_counts)
            or not 1 <= len(table.class_names) <= _model.MAX_CLASSES
            or not 1 <= len(table.headers) <= _model.MAX_CLASSES
            or not 0 < table.source_bytes <= MAX_SAMPLE_BYTES
            or len(table.rgb_bytes) != len(table.class_names)
            or len(table.sample_counts) != len(table.class_names)
            or tuple(dict.fromkeys(table.headers)) != table.class_names
            or any(type(p) is not bytes or len(p) % 3 for p in table.rgb_bytes)
            or table.sample_counts != tuple(len(p)//3 for p in table.rgb_bytes)
            or sum(table.sample_counts) > _fit.MAX_SAMPLES):
        raise _fit.BayesTrainingError('parsed sample table has inconsistent immutable metadata')
    try:
        if (any(type(name) is not str or not name or '\t' in name or '\r' in name or '\n' in name
                or len(name.encode('utf-8')) > MAX_LABEL_BYTES for name in table.class_names)
                or _sample_digest(table.class_names, table.rgb_bytes) != table.parsed_sha256):
            raise _fit.BayesTrainingError('parsed sample identity or class labels are invalid')
        if (type(table.source_sha256) is not str or len(table.source_sha256) != 64
                or any(c not in '0123456789abcdef' for c in table.source_sha256)):
            raise _fit.BayesTrainingError('parsed sample source identity is invalid')
    except (UnicodeError, TypeError, ValueError) as error:
        raise _fit.BayesTrainingError('parsed sample encoding or identity is invalid') from error


def _calculation_records():
    # Verify the actual loaded Python/helper closure, as in native recipes.
    from seedvision.benchmarking.native_recipe import _module_record
    modules = (_fit, _nvrtc, _range, _model, __import__(__name__, fromlist=['train_bayes_samples']))
    return {module.__name__: _module_record(module) for module in modules}


def train_bayes_samples(table: BayesSampleTable, *, device='cuda', origin_path=None,
                        cancel=None) -> BayesTrainingCompletion:
    """Train parsed sample bytes, returning exact loadable TSV plus provenance.

    Only the compact PDF is downloaded for serialization. Original sample-file
    availability is unnecessary for subsequent inference with the produced TSV.
    Training remains an engineering computation, not scientific validation.
    """
    if cancel is not None and not callable(cancel):
        raise _fit.BayesTrainingError('cancel must be callable or None')
    _poll(cancel)
    _validate_table(table)
    if any(n < 2 for n in table.sample_counts):
        raise _fit.BayesTrainingError('each class needs at least two training samples')
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise _fit.BayesTrainingError('Bayes training requires an available CUDA device')
    # Combined conversion/fit bound before any upload. HSV is retained, while
    # decoder transport and converter scratch are limited to a single tile.
    total = sum(table.sample_counts)
    conversion_scratch = 1024*min(CONVERSION_TILE, max(table.sample_counts)) + _fit.HEADROOM_BYTES
    fit_scratch = 3*256*256*8 + len(table.class_names)*3*256*8 + 512 + _fit.HEADROOM_BYTES
    work = total*(3*3+3+3*256) + len(table.class_names)*(3*256*256+3*256)
    if (work > _fit.MAX_WORK or max(conversion_scratch, fit_scratch) + 3*total > _fit.MAX_CUDA_LIVE_BYTES
            or fit_scratch > _fit.MAX_SCRATCH_BYTES):
        raise _fit.BayesTrainingError('sample conversion and fit exceed their combined CUDA resource budget')
    records = _calculation_records()
    samples = []
    with torch.cuda.device(device):
        for raw, n in zip(table.rgb_bytes, table.sample_counts):
            _poll(cancel)
            hsv = torch.empty((n, 3), dtype=torch.uint8, device=device)
            for start in range(0, n, CONVERSION_TILE):
                _poll(cancel)
                end = min(start+CONVERSION_TILE, n)
                rgb = np.frombuffer(raw, dtype=np.uint8, count=(end-start)*3, offset=start*3).copy()
                bgr = torch.tensor(rgb, dtype=torch.uint8, device=device).reshape(-1, 3).flip(-1)
                hsv[start:end].copy_(_range._channels(bgr, 'HSV'))
                del rgb, bgr
            samples.append(hsv)
        samples = tuple(samples)
        fitted = _fit.fit_naive_bayes_hsv(samples, table.class_names, cancel=cancel)
        watched = (*samples, fitted.probabilities)
        signatures = tuple(_fit._tensor_signature(x) for x in watched)
        def publication_checkpoint():
            try:
                _poll(cancel)
            finally:
                if tuple(_fit._tensor_signature(x) for x in watched) != signatures:
                    raise _fit.BayesTrainingError('training samples or PDF changed before publication')
        publication_checkpoint()
        compact = fitted.probabilities.cpu().numpy()
        rows = ['class\tchannel\t'+'\t'.join(map(str, range(256)))]
        for c, channel in enumerate(_model.CHANNELS):
            for index, name in enumerate(table.class_names):
                rows.append('\t'.join((name, channel, *map(str, compact[index, c]))))
        payload = ('\n'.join(rows)+'\n').encode('utf-8')
        model = _model.parse_plantcv_pdf(payload)
        provenance = dict(kind='native_bayes_sample_training', parser_id=PARSER_ID,
            sample_origin_path=None if origin_path is None else str(Path(origin_path).resolve()),
            sample_sha256=table.source_sha256, sample_bytes=table.source_bytes,
            parsed_samples_sha256=table.parsed_sha256, headers=table.headers,
            class_names=table.class_names, sample_counts=table.sample_counts,
            model_sha256=model.source_sha256, tables_sha256=model.tables_sha256,
            fit_plan=asdict(fitted.plan), channel_fits=[[asdict(f) for f in cs] for cs in fitted.channels],
            calculation=records, scientific_validation=False,
            runtime=dict(torch=str(torch.__version__), numpy=np.__version__, torch_cuda=str(torch.version.cuda),
                device=str(samples[0].device), device_name=torch.cuda.get_device_name(samples[0].device),
                **_nvrtc.runtime_versions()),
            source_pins=dict(plantcv='4.11.3', scipy='1.15.3', numpy='2.2.6',
                plantcv_train_sha256=_fit.PLANTCV_TRAIN_SHA256, scipy_kde_sha256=_fit.SCIPY_KDE_SHA256,
                scipy_stats_sha256=_fit.SCIPY_STATS_SHA256, numpy_pairwise_sha256=_fit.NUMPY_PAIRWISE_SHA256))
        publication_checkpoint()
        if _calculation_records() != records:
            raise _fit.BayesTrainingError('training implementation changed before publication')
        return BayesTrainingCompletion(samples, fitted, model, payload,
            json.dumps(provenance, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8'))


def train_bayes_sample_file(path: str | Path, *, device='cuda', cancel=None):
    """Hash/read the selected bytes and reject a changed file before publishing."""
    path = Path(path).resolve()
    table = read_bayes_samples(path, cancel=cancel)
    result = train_bayes_samples(table, device=device, origin_path=path, cancel=cancel)
    watched = (*result.samples, result.fit.probabilities)
    signatures = tuple(_fit._tensor_signature(x) for x in watched)
    try:
        _poll(cancel)
    finally:
        if tuple(_fit._tensor_signature(x) for x in watched) != signatures:
            raise _fit.BayesTrainingError('training samples or PDF changed before publication')
    # Final selected-byte verification has no caller callback after the read.
    # A callback that edits the file at the last checkpoint is therefore caught.
    try:
        with path.open('rb') as stream:
            checked = stream.read(MAX_SAMPLE_BYTES+1)
    except OSError as error:
        raise _fit.BayesTrainingError(f'unable to verify selected sample table: {error}') from error
    if (sha256(checked).hexdigest(), len(checked)) != (table.source_sha256, table.source_bytes):
        raise _fit.BayesTrainingError('selected sample file changed during training')
    return result
