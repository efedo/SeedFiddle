"""Versioned per-instance colour/geometry features; raster work stays on CUDA."""
from __future__ import annotations

from dataclasses import dataclass
import json

import torch


TRAIT_FEATURE_SCHEMA = 'seed-classifier-pixel-v1'
TRAIT_FEATURE_NAMES = tuple(
    f'{channel}_{stat}' for channel in ('red', 'green', 'blue')
    for stat in ('mean', 'population_std', 'q10', 'q50', 'q90')
) + ('log_pixel_area', 'log_bbox_aspect', 'bbox_fill',
     'moment_anisotropy', 'grid_perimeter_over_sqrt_area')


def _identifiers(values, name):
    result = tuple(values)
    if any(type(value) is not int or value <= 0 for value in result):
        raise ValueError(f'{name} must contain positive integer instance IDs.')
    if len(result) != len(set(result)):
        raise ValueError(f'{name} must contain unique instance IDs.')
    return result


@dataclass(frozen=True, slots=True)
class TraitFeatureBatch:
    values: torch.Tensor
    identifiers: tuple[int, ...]
    rejected: tuple[tuple[int, str], ...]
    provenance_json: str
    schema: str = TRAIT_FEATURE_SCHEMA

    def __post_init__(self):
        object.__setattr__(self, 'identifiers', tuple(self.identifiers))
        object.__setattr__(self, 'rejected', tuple(tuple(value) for value in self.rejected))
        _identifiers(self.identifiers, 'Feature rows')
        _identifiers(tuple(value[0] for value in self.rejected), 'Rejected rows')
        if set(self.identifiers).intersection(value[0] for value in self.rejected):
            raise ValueError('Feature and rejected rows must be disjoint.')
        if not isinstance(self.values, torch.Tensor) or not self.values.is_cuda or self.values.ndim != 2 or not self.values.is_floating_point():
            raise ValueError('Compact trait features must be a CUDA floating matrix.')
        if self.values.shape != (len(self.identifiers), len(TRAIT_FEATURE_NAMES)):
            raise ValueError('Trait features must match the declared twenty-column schema.')
        if any(not isinstance(reason, str) or not reason for _, reason in self.rejected):
            raise ValueError('Rejected trait inputs require explicit reasons.')
        provenance = json.loads(self.provenance_json)
        if not isinstance(provenance, dict):
            raise ValueError('Trait feature provenance must be a JSON object.')
        json.dumps(provenance, allow_nan=False)


@torch.no_grad()
def _instance_features(rgb, labels, identifier):
    mask = labels == identifier
    coordinates = torch.nonzero(mask)
    count = coordinates.shape[0]
    if count == 0:
        return None, 'missing_instance'
    pixels = rgb[mask].to(torch.float64)
    if bool((~torch.isfinite(pixels) | (pixels < 0) | (pixels > 1)).any().item()):
        return None, 'invalid_pixels'
    mean = pixels.mean(dim=0)
    std = (pixels - mean).square().mean(dim=0).sqrt()
    quantiles = torch.quantile(pixels, pixels.new_tensor([.1, .5, .9]), dim=0)
    colour = torch.stack((mean, std, *quantiles), dim=1).reshape(-1)
    xy = coordinates.to(torch.float64)
    bbox = xy.amax(dim=0) - xy.amin(dim=0) + 1
    centred = xy - xy.mean(dim=0)
    moments = centred.T @ centred / count
    trace = moments.trace()
    separation = ((moments[0, 0] - moments[1, 1]).square()
                  + 4 * moments[0, 1].square()).sqrt()
    anisotropy = separation / trace if bool((trace > 0).item()) else trace.new_zeros(())
    perimeter = (
        (mask[1:] != mask[:-1]).sum(dtype=torch.int64)
        + (mask[:, 1:] != mask[:, :-1]).sum(dtype=torch.int64)
        + mask[0].sum(dtype=torch.int64) + mask[-1].sum(dtype=torch.int64)
        + mask[:, 0].sum(dtype=torch.int64) + mask[:, -1].sum(dtype=torch.int64)
    ).to(torch.float64)
    area = xy.new_tensor(count)
    geometry = torch.stack((area.log(), (bbox.max() / bbox.min()).log(),
                            area / bbox.prod(), anisotropy, perimeter / area.sqrt()))
    result = torch.cat((colour, geometry))
    if not bool(torch.isfinite(result).all().item()):
        return None, 'numeric_range'
    return result, None


@torch.no_grad()
def extract_trait_features(image, labels, requested_ids, eligible_ids, *,
                           encoding, image_frame, label_frame):
    """Extract eligible visible pixels in an explicitly shared corrected frame.

    Frames are supplied acquisition/analysis identities, not evidence of physical
    accuracy. Eligibility is supplied by the reviewed consumer. Log area is
    deliberately in pixels and remains resolution-dependent.
    """
    requested = _identifiers(requested_ids, 'Requested rows')
    eligible = set(_identifiers(eligible_ids, 'Eligible rows'))
    if (not isinstance(image, torch.Tensor) or not isinstance(labels, torch.Tensor)
            or not image.is_cuda or not labels.is_cuda or image.device != labels.device):
        raise ValueError('Trait image and labels must share a CUDA device.')
    if image.ndim != 3 or image.shape[-1] != 3 or labels.shape != image.shape[:2] or labels.numel() == 0:
        raise ValueError('Trait inputs require matching nonempty HWC/HW rasters.')
    if labels.dtype not in (torch.uint8, torch.int16, torch.int32, torch.int64):
        raise ValueError('Trait labels must use an integer dtype.')
    if encoding not in {'rgb_unit', 'bgr_unit', 'rgb_byte', 'bgr_byte'}:
        raise ValueError('Declare RGB/BGR and unit/byte encoding explicitly.')
    if (encoding.endswith('_byte') and image.dtype != torch.uint8) or (
            encoding.endswith('_unit') and not image.is_floating_point()):
        raise ValueError('Trait image dtype does not match the declared encoding.')
    reason = None
    if not isinstance(image_frame, str) or not image_frame or not isinstance(label_frame, str) or not label_frame:
        reason = 'missing_frame_provenance'
    elif image_frame != label_frame:
        reason = 'incompatible_frame'
    provenance = json.dumps({'feature_schema': TRAIT_FEATURE_SCHEMA,
        'image_frame': image_frame if isinstance(image_frame, str) else None,
        'label_frame': label_frame if isinstance(label_frame, str) else None,
        'encoding': encoding, 'image_shape': list(image.shape),
        'representation': 'visible corrected-frame pixels; no physical scale',
        'eligibility_source': 'explicit supplied eligible instance IDs'}, sort_keys=True)
    if reason:
        return TraitFeatureBatch(image.new_empty((0, 20), dtype=torch.float64), (),
                                 tuple((value, reason) for value in requested), provenance)
    rgb = image.to(torch.float64)
    if encoding.endswith('_byte'):
        rgb = rgb / 255
    if encoding.startswith('bgr'):
        rgb = rgb.flip(-1)
    rows, identifiers, rejected = [], [], []
    for identifier in requested:
        if identifier not in eligible:
            rejected.append((identifier, 'ineligible_instance'))
            continue
        row, reason = _instance_features(rgb, labels, identifier)
        if reason:
            rejected.append((identifier, reason))
        else:
            rows.append(row)
            identifiers.append(identifier)
    values = torch.stack(rows) if rows else image.new_empty((0, 20), dtype=torch.float64)
    return TraitFeatureBatch(values, tuple(identifiers), tuple(rejected), provenance)
