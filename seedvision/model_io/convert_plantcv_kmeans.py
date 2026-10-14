"""Convert a PlantCV MiniBatchKMeans model in its source training environment.

Invoke explicitly with python -m seedvision.model_io.convert_plantcv_kmeans.
The desktop reads only the resulting bounded numeric JSON artifact; sklearn and
joblib remain dependencies of this offline conversion operation alone.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import math
import os
from pathlib import Path
import stat
import tempfile

from seedvision.model_io import kmeans_model_artifact as _artifact
from seedvision.model_io import plantcv_kmeans_model as _model

MAX_INPUT_BYTES = 64 * 1024 * 1024


class KMeansConversionError(ValueError):
    """Unsupported source model/environment or unsuccessful atomic conversion."""


def _read_bounded(path):
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_INPUT_BYTES:
            raise KMeansConversionError('input must be a nonempty model file of at most64MiB')
        with path.open('rb') as stream:
            raw = stream.read(MAX_INPUT_BYTES+1)
        after = path.stat()
    except OSError as error:
        raise KMeansConversionError(f'unable to read source model: {path}') from error
    if (not 0 < len(raw) <= MAX_INPUT_BYTES or before.st_size != len(raw)
            or after.st_size != len(raw)
            or (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size)):
        raise KMeansConversionError('source model changed during its bounded read')
    return raw


def _geometry(width):
    if type(width) is not int or not 1 <= width <= _model.MAX_FEATURES:
        raise KMeansConversionError('unsupported source feature count')
    candidates = []
    for channels in (1, 3):
        p = math.isqrt(width//channels)
        if 1 <= p <= 31 and p*p*channels == width:
            candidates.append((p, channels))
    if len(candidates) != 1:
        raise KMeansConversionError('feature width does not identify one supported patch geometry')
    return candidates[0]


def _atomic_write(path, raw):
    if type(raw) is not bytes or not 0 < len(raw) <= _model.MAX_MODEL_BYTES:
        raise KMeansConversionError('complete output exceeds the native model byte bound')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent,
                prefix='.'+path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            if stream.write(raw) != len(raw):
                raise OSError('incomplete model write')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as error:
        raise KMeansConversionError(f'unable to atomically save converted model: {path}') from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def convert_plantcv_kmeans(input_path, output_path):
    """Read the source once; preserve centre order/binary64 values without refit."""
    try:
        source = Path(input_path).expanduser().resolve(strict=True)
        target = Path(output_path).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise KMeansConversionError('source or destination model path is unavailable') from error
    if source == target or (target.exists() and source.samefile(target)):
        raise KMeansConversionError('source and destination model files must differ')
    if not target.parent.is_dir():
        raise KMeansConversionError('destination folder does not exist')
    raw = _read_bounded(source)
    try:
        from importlib.metadata import version
        import joblib
        import numpy as np
        import sklearn
        from sklearn.cluster import MiniBatchKMeans
        plantcv_version = version('plantcv')
    except (ImportError, ModuleNotFoundError) as error:
        raise KMeansConversionError('run conversion in the PlantCV training environment') from error
    try:
        estimator = joblib.load(io.BytesIO(raw))
    except Exception as error:
        raise KMeansConversionError('unable to load the supplied PlantCV estimator') from error
    if type(estimator) is not MiniBatchKMeans:
        raise KMeansConversionError('expected an exact sklearn MiniBatchKMeans estimator')
    centres = getattr(estimator, 'cluster_centers_', None)
    clusters = getattr(estimator, 'n_clusters', None)
    width = getattr(estimator, 'n_features_in_', None)
    if (type(clusters) is not int or not 1 <= clusters <= _model.MAX_CLUSTERS
            or type(width) is not int or not 1 <= width <= _model.MAX_FEATURES
            or type(centres) is not np.ndarray or centres.dtype != np.dtype('float64')
            or centres.ndim != 2 or centres.shape != (clusters, width)
            or clusters*width > _model.MAX_CENTRE_VALUES):
        raise KMeansConversionError('estimator is unfitted or has unsupported centre dtype/geometry')
    patch_size, channels = _geometry(width)
    if not np.isfinite(centres).all() or bool(((centres < 0.) | (centres > 255.)).any()):
        raise KMeansConversionError('centres must be finite float64 values in[0,255]')
    # These are compact model metadata, not full-resolution training products.
    # tolist preserves binary64 values, row order, duplicates and signed zero.
    model = _model.model_from_centres(centres.tolist(), patch_size, channels)
    provenance = dict(kind='plantcv_conversion',
        numeric_identity=_artifact.numeric_model_identity(model),
        input=dict(source_sha256=sha256(raw).hexdigest(), source_bytes=len(raw),
            estimator_class=type(estimator).__module__+'.'+type(estimator).__qualname__),
        environment=dict(plantcv=plantcv_version, numpy=np.__version__,
            sklearn=sklearn.__version__, joblib=joblib.__version__),
        geometry=dict(patch_size=patch_size, channels=channels,
            method='unique-supported-feature-width'))
    payload = _artifact.encode_kmeans_artifact(model, provenance)
    _atomic_write(target, payload)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description='Convert a PlantCV K-means .fit model to native JSON.')
    parser.add_argument('input', help='model saved by PlantCV training')
    parser.add_argument('output', help='destination JSON model artifact')
    arguments = parser.parse_args(argv)
    try:
        target = convert_plantcv_kmeans(arguments.input, arguments.output)
    except (_model.KMeansModelError, KMeansConversionError, OSError) as error:
        parser.exit(2, f'Conversion failed: {error}\n')
    print(target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
