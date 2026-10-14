"""Write immutable, coordinate-bound per-seed image and mask artifacts."""

from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from seedvision.export.results import result_report


class CropArtifactError(ValueError):
    """The selected crop export is stale, incomplete, or cannot be encoded."""


class CropExportCancelled(CropArtifactError):
    """The operator stopped extraction or serialization before publication."""


def _cancelled(cancel):
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    event = getattr(cancel, 'is_set', None)
    if not callable(event):
        raise TypeError('cancel must be callable or an event with is_set().')
    return bool(event())


def _check_cancel(cancel):
    if _cancelled(cancel):
        raise CropExportCancelled('Seed crop export stopped before publication.')


def _metadata(record):
    """Only declared compact fields cross the metadata boundary."""
    return {field.name: getattr(record, field.name) for field in fields(record)
            if field.name not in {'image_crop', 'mask_crop'}}


def _image_bytes(tensor):
    """Download one explicitly exported crop, retaining the parent CUDA raster."""
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cuda':
        raise CropArtifactError('An available image crop must be a CUDA tensor.')
    image = tensor.detach()
    if image.ndim == 4 and tuple(image.shape[:2]) == (1, 3):
        image = image[0].permute(1, 2, 0)
    elif image.ndim == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    else:
        raise CropArtifactError('An image crop must have three BGR channels in NCHW or CHW order.')
    if image.numel() == 0:
        raise CropArtifactError('An available crop cannot have an empty image.')
    if image.dtype != torch.uint8:
        if not image.is_floating_point() or not bool(torch.isfinite(image).all().item()):
            raise CropArtifactError('Image crop values must be finite uint8 or float values.')
        if bool(((image < 0) | (image > 255)).any().item()):
            raise CropArtifactError('Image crop values lie outside the declared 0–255 range.')
        image = image.round().to(torch.uint8)
    return np.ascontiguousarray(image.cpu().numpy())


def _mask_bytes(tensor, shape):
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cuda':
        raise CropArtifactError('An available isolated mask must be a CUDA tensor.')
    mask = tensor.detach()
    if mask.ndim == 4 and tuple(mask.shape[:2]) == (1, 1):
        mask = mask[0, 0]
    if mask.ndim != 2 or tuple(mask.shape) != tuple(shape):
        raise CropArtifactError('Image and isolated mask crop dimensions differ.')
    if mask.dtype == torch.bool:
        mask = mask.to(torch.uint8) * 255
    elif mask.dtype != torch.uint8 or bool(((mask != 0) & (mask != 255)).any().item()):
        raise CropArtifactError('An isolated mask must contain bool or uint8 0/255 pixels.')
    return np.ascontiguousarray(mask.cpu().numpy())


def export_seed_crop_artifacts(directory, analysis, report, *, seed_ids, padding_px, cancel=None):
    """Publish crop PNGs and a complete manifest in one new revision directory.

    Cancellation/failure leaves a clearly incomplete private working directory;
    only a successfully written manifest is renamed to the published address.
    """
    from seedvision.export.crops import extract_seed_crops

    _check_cancel(cancel)
    identity = report.get('identity', {})
    method = report.get('method')
    if not isinstance(identity, dict) or not isinstance(method, str):
        raise CropArtifactError('Crop export requires an available exact result report.')
    fresh = result_report(analysis, enabled={method}, settings=identity.get('settings'),
                          provenance=identity.get('provenance'))
    if fresh['state'] != 'available_for_review' or fresh['result_revision'] != report.get('result_revision'):
        raise CropArtifactError('Result changed. Recompute or reopen its review before exporting crops.')
    identifiers = tuple(seed_ids)
    if not identifiers:
        raise CropArtifactError('Choose at least one visible seed for crop export.')
    batch = extract_seed_crops(analysis, method=method, seed_ids=identifiers,
                               padding_px=padding_px, result_revision=fresh['result_revision'], cancel=cancel)
    _check_cancel(cancel)
    if str(batch.state) == 'cancelled':
        raise CropExportCancelled('Seed crop extraction stopped before publication.')
    if str(batch.state) in {'failed', 'unavailable'}:
        reason = batch.reason or next((record.reason for record in batch.records if record.reason), '')
        raise CropArtifactError(reason or 'The selected seed crops are unavailable.')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    name = f"seed-crops-{fresh['result_revision'][:16]}-{uuid4().hex}"
    working = root / ('.' + name + '.incomplete')
    destination = root / name
    working.mkdir(exist_ok=False)
    records = []
    decisions = {row['seed_id']: row.get('review', 'unreviewed') for row in report.get('instances', ())}
    for record in batch.records:
        _check_cancel(cancel)
        item = _metadata(record)
        item['analysis_source_sha256'] = item['source_sha256']
        item['source_sha256'] = fresh['identity'].get('source_sha256')
        item['review'] = decisions.get(record.seed_id, 'unreviewed')
        item['artifacts'] = {}
        if record.image_crop is not None or record.mask_crop is not None:
            image = _image_bytes(record.image_crop)
            mask = _mask_bytes(record.mask_crop, image.shape[:2])
            x0, y0, x1, y1 = record.corrected_bbox
            reviewed_image = analysis.calibration.corrected_bgr[y0:y1, x0:x1]
            if not np.array_equal(image, reviewed_image):
                raise CropArtifactError('CUDA crop pixels differ from the reviewed corrected image. Recompute before exporting.')
            for kind, raster in (('image', image), ('mask', mask)):
                _check_cancel(cancel)
                filename = f'seed-{record.seed_id}-{kind}.png'
                if not cv2.imwrite(str(working / filename), raster):
                    raise OSError(f'Could not write seed {record.seed_id} {kind} crop.')
                item['artifacts'][kind] = {'path': filename,
                    'sha256': sha256((working / filename).read_bytes()).hexdigest(),
                    'shape': list(raster.shape), 'dtype': str(raster.dtype)}
        records.append(item)
    _check_cancel(cancel)
    package = Path(__file__).resolve().parents[1]
    manifest = {'format': 'seedfiddle.seed-crop-artifacts', 'version': 1,
        'result_revision': fresh['result_revision'], 'method': method,
        'source_identity': fresh['identity'], 'state': str(batch.state),
        'seed_ids': list(identifiers), 'padding_px': padding_px,
        'tensor_channel_order': 'BGR', 'image_file_channel_order': 'RGB', 'mask_values': [0, 255],
        'scientifically_validated': False, 'records': records,
        'artifact_identity_contract': 'Primitive crop identity binds result_revision and analysis_source_sha256; '
                                      'it is not a hash of the serialized record. source_sha256 is the verified report digest.',
        'implementation': {name: sha256((package / name).read_bytes()).hexdigest()
                           for name in ('export/crops.py', 'export/crop_artifacts.py')}}
    content = json.dumps(manifest, indent=2, allow_nan=False, sort_keys=True) + '\n'
    (working / 'seed-crops.json').write_text(content, encoding='utf-8')
    _check_cancel(cancel)
    final_report = result_report(analysis, enabled={method}, settings=identity.get('settings'),
                                 provenance=identity.get('provenance'))
    if final_report['state'] != 'available_for_review' or final_report['result_revision'] != fresh['result_revision']:
        raise CropArtifactError('Result changed during crop export. Recompute before publishing crops.')
    _check_cancel(cancel)
    working.rename(destination)
    return destination
