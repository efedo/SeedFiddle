"""Production-pipeline objectives with frozen, image-local reference domains.

Automatic runs are explicitly in-sample project adaptation, not validation.
No annotation is a watershed marker. All raster probability reductions stay
on the raster's device; CPU instance matching uses a bounded fixed image grid.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, fields
from hashlib import sha256
from pathlib import Path
from statistics import mean
import json
import os
import tempfile

import cv2
import numpy as np

from .registry import OBJECTIVES, affected_nodes
from .search import Ineligible, OptimizationCancelled
from seedvision.annotation.eligibility import eligible_instances


from seedvision.pipeline.settings import BASELINE_NODES, LAYER_NODES, production_settings
CUSTOM_OUTPUTS = {
    'illumination_decomposition': ('shadow_likelihood', 'highlight_likelihood'),
    'image_quality': ('sensor_noise', 'image_quality_risk'),
    'pattern_decomposition': ('pattern',), 'colour_probabilities': ('colour',),
    'touching_split': ('touching_split_likelihood',), 'ellipse_likelihood': ('ellipse_likelihood',),
    'assignment_confidence': ('instance_assignment_confidence',),
    'radial_profile': ('radial_profile_residual',), 'proposal_disagreement': ('proposal_disagreement',),
    'contact_graph': ('contact_pairs',), 'coat_damage': ('coat_damage_likelihood',),
    'calibration_residuals': ('calibration_residual_risk',),
}
DERIVED_DIAGNOSTICS = {'touching_split', 'ellipse_likelihood', 'assignment_confidence', 'proposal_disagreement', 'contact_graph'}


def eligibility_reason(graph, identifier, sample):
    kind = OBJECTIVES.get(identifier)
    has_instances = sample.labels is not None and bool(np.any(sample.labels))
    if kind in {'instances', 'unet', 'stardist', 'legacy instances', 'proposals', 'edges', 'ceiling edges', 'boundary'} and not has_instances:
        return 'Apply complete, shape-reviewed instance annotations.'
    if kind == 'instances' and ('procedural_instances' not in graph.nodes or not graph.node('procedural_instances').enabled):
        return 'Enable Procedural seed separation for this downstream objective.'
    if kind in {'unet', 'stardist'}:
        configured = Path(graph.node(identifier).parameters['checkpoint_path'])
        path = configured if configured.is_absolute() else Path(sample.kwargs.get('learning_root', '.')) / configured
        if not path.is_file():
            return 'Select an existing compatible learned checkpoint first.'
    if kind in {'material', 'colour', 'noise', 'texture'} and not has_instances and not any(
            value is not None and np.any(value) for key,value in sample.kwargs.items() if key.endswith('reference_mask') or key.endswith('exclusion_mask')):
        return 'Apply material class reference regions or complete seed instances.'
    if kind in {'traits', 'wrinkling'}:
        reviewed = [item for item in sample.kwargs.get('seed_instance_traits', ())
                    if has_instances and np.any(sample.labels == item.seed_id)
                    and (item.conditions_reviewed or (kind == 'traits' and item.coat_pattern))]
        if not reviewed:
            return 'Review compatible coat/condition labels on complete instances.'
    if kind == 'reviewed raster':
        derived = identifier in DERIVED_DIAGNOSTICS and has_instances
        supplied = any(key.startswith(identifier+'__') for key in (*sample.custom, *sample.target_keys))
        if not derived and not supplied:
            return 'Import reviewed targets for this output; seed outlines do not establish its meaning.'
    return ''


def array_digest(value):
    value = np.ascontiguousarray(value)
    return sha256(str((value.shape, str(value.dtype))).encode() + value.tobytes()).hexdigest()


def file_digest(path):
    from seedvision.persistence.reference_regions import file_sha256
    return file_sha256(Path(path))


@dataclass
class Sample:
    key: str
    path: Path
    kwargs: dict
    corrected_shape: tuple[int, int]
    transform: np.ndarray
    labels: np.ndarray | None = None
    complete: bool = False
    group: str = ''
    custom: dict = field(default_factory=dict)
    seed_diameter: float = 30.0
    fingerprint: str = ''
    target_path: Path | None = None
    target_fingerprint: str = ''
    target_keys: tuple[str, ...] = ()
    excluded_ids: tuple[int, ...] = ()

    def __post_init__(self):
        self.path = Path(self.path)
        self.fingerprint = self.fingerprint or file_digest(self.path)
        if self.target_path is not None:
            self.target_path = Path(self.target_path)
            self.target_fingerprint = file_digest(self.target_path)
            check_target_archive(self.target_path)
            with np.load(self.target_path, allow_pickle=False) as archive:
                self.target_keys = tuple(archive.files)
        self.transform = None if self.transform is None else np.asarray(self.transform).copy()
        original_ids = set() if self.labels is None else set(np.unique(self.labels)) - {0}
        self.labels = eligible_instances(self.labels, self.kwargs.get('seed_instance_traits', ()))
        retained_ids = set() if self.labels is None else set(np.unique(self.labels)) - {0}
        self.excluded_ids = tuple(sorted(int(value) for value in original_ids-retained_ids))
        self.kwargs = dict(self.kwargs)
        self.kwargs['seed_instance_annotations'] = self.labels
        for key, value in tuple(self.kwargs.items()):
            if isinstance(value, np.ndarray):
                value = value.copy()
                value.flags.writeable = False
                self.kwargs[key] = value
        if self.complete and (self.labels is None or not np.any(self.labels) or self.excluded_ids):
            raise ValueError('Whole-image coverage requires complete reviewed instances.')


def check_target_archive(path):
    from zipfile import ZipFile
    with ZipFile(path) as container:
        if len(container.infolist()) > 256 or sum(item.file_size for item in container.infolist()) > 1024**3:
            raise ValueError('Reviewed target archive exceeds the 1 GiB / 256-array resource limit.')


def target_shape(path):
    """Recover a persisted raster domain before an image has been analyzed."""
    check_target_archive(path)
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            parts = key.split('__')
            if len(parts) >= 2 and parts[1] in CUSTOM_OUTPUTS.get(parts[0], ()) and parts[1] != 'contact_pairs':
                value = archive[key]
                if value.ndim == 2:
                    return tuple(value.shape)
    raise ValueError('Targets without raster geometry require applied instance references or an analyzed image.')


def load_targets(path, sample):
    """Safe NPZ: immutable source + transform identity, NaN means unreviewed.

    Keys are node_id__output, with optional __class suffix for colour/pattern.
    Pair labels are N×3 (reference ID, reference ID, reviewed 0/1 relation).
    """
    check_target_archive(path)
    with np.load(path, allow_pickle=False) as archive:
        if str(archive['source_sha256'].item()) != sample.fingerprint:
            raise ValueError('Optimization target source fingerprint does not match.')
        if str(archive['transform_sha256'].item()) != array_digest(sample.transform):
            raise ValueError('Optimization targets use a different calibration transform.')
        result = {}
        for key in archive.files:
            if key in {'source_sha256', 'transform_sha256'}:
                continue
            parts = key.split('__')
            if len(parts) not in (2, 3) or parts[1] not in CUSTOM_OUTPUTS.get(parts[0], ()):
                raise ValueError(f'Unsupported optimization target {key}.')
            if (parts[1] in {'colour', 'pattern'}) != (len(parts) == 3):
                raise ValueError('Colour/pattern targets require exactly one named class suffix.')
            if parts[1] in {'colour', 'pattern'}:
                from seedvision.visualization.advanced import COLOUR_CLASS_NAMES, PATTERN_CLASS_NAMES
                allowed = COLOUR_CLASS_NAMES if parts[1] == 'colour' else PATTERN_CLASS_NAMES
                if parts[2] not in allowed:
                    raise ValueError(f'{key} must name one of: {allowed}')
            value = np.asarray(archive[key], dtype=np.float32).copy()
            if parts[1] == 'contact_pairs':
                columns = 3
                if value.ndim != 2 or value.shape[1] != columns or not np.isfinite(value).all():
                    raise ValueError(f'Invalid reviewed geometry for {key}.')
                if not len(value) or not np.isin(value[:,2], (0,1)).all() or np.any(value[:,:2] <= 0) or np.any(value[:,:2] != np.floor(value[:,:2])):
                    raise ValueError('Contact targets need positive integer reference IDs and explicit 0/1 labels.')
                if sample.labels is None or not np.isin(value[:,:2], np.unique(sample.labels)).all():
                    raise ValueError('Contact targets must identify complete reviewed seed instances in this image.')
                pairs = [tuple(sorted(pair)) for pair in value[:,:2]]
                if any(a == b for a,b in pairs) or len(pairs) != len(set(pairs)):
                    raise ValueError('Contact targets must contain distinct, nonduplicated pairs.')
            elif value.shape != sample.corrected_shape or np.isinf(value).any() or not np.isfinite(value).any():
                raise ValueError(f'{key} must be a corrected-image raster; NaN marks unreviewed pixels.')
            elif np.nanmin(value) < 0 or np.nanmax(value) > 1:
                raise ValueError(f'{key} targets must be probabilities in [0, 1].')
            value.flags.writeable = False
            result[key] = value
        return result



def tensor(value):
    import torch
    if value is None:
        raise Ineligible('Required production output is unavailable.')
    data = value.gpu_tensor() if hasattr(value, 'gpu_tensor') else torch.as_tensor(value)
    dtype = getattr(value, 'dtype', data.dtype)
    data = data.float().squeeze()
    if dtype in (np.dtype('uint8'), torch.uint8):
        data = data / 255.0
    return data


def raster_loss(value, target, result, reference_transform=None):
    """Balanced probability loss at fixed reviewed coordinates, on CUDA."""
    import torch
    import torch.nn.functional as functional
    data = tensor(value)
    if data.ndim != 2:
        raise ValueError('Scalar probability output required.')
    targets = torch.as_tensor(target, device=data.device, dtype=torch.float32)
    ys, xs = torch.where(torch.isfinite(targets))
    if not xs.numel():
        raise Ineligible('No reviewed pixels for this output.')
    crop_height, crop_width = result.layers.valid_mask.shape[-2:]
    sums = torch.zeros(2, device=data.device, dtype=torch.float64)
    counts = torch.zeros(2, device=data.device, dtype=torch.float64)
    mapping = None
    if reference_transform is not None and not np.allclose(reference_transform, result.calibration.affine_matrix):
        mapping = torch.as_tensor(result.calibration.affine_matrix @ np.linalg.inv(reference_transform), device=data.device, dtype=torch.float32)
    for start in range(0, xs.numel(), 262144):
        xx, yy = xs[start:start+262144], ys[start:start+262144]
        x, y = xx.float(), yy.float()
        if mapping is not None:
            coordinates = torch.stack((x,y,torch.ones_like(x)))
            projected = mapping @ coordinates
            x, y = projected[0]/projected[2], projected[1]/projected[2]
        x, y = x-result.crop_offset[0], y-result.crop_offset[1]
        grid = torch.stack(((x + .5) * 2 / crop_width - 1, (y + .5) * 2 / crop_height - 1), -1)
        prediction = functional.grid_sample(data[None,None], grid[None,None], align_corners=False, padding_mode='zeros').flatten().clamp(1e-5,1-1e-5)
        truth = targets[yy,xx]
        costs = -(truth*prediction.log() + (1-truth)*torch.log1p(-prediction))
        for index, mask in enumerate((truth >= .5, truth < .5)):
            sums[index] += costs[mask].double().sum()
            counts[index] += mask.sum()
    valid = counts > 0
    return float((sums[valid]/counts[valid]).mean().item())


def instance_loss(sample, labels, result, policy):
    from seedvision.segmentation.procedural_fit import score_procedural_instances
    if sample.labels is None or not np.any(sample.labels):
        raise Ineligible('Apply complete, shape-reviewed seed instances.')
    h, w = sample.corrected_shape
    scale = min(1.0, 1024 / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    truth = cv2.resize(sample.labels, size, interpolation=cv2.INTER_NEAREST)
    if set(np.unique(truth)) != set(np.unique(sample.labels)):
        raise Ineligible('Some reference instances disappear at the bounded evaluation resolution; review these tiny masks.')
    if hasattr(labels, 'gpu_tensor'):
        import torch.nn.functional as functional
        data = labels.gpu_tensor().float()
        factor = min(1., 1024 / max(data.shape[-2:]))
        if factor < 1:
            data = functional.interpolate(data, scale_factor=factor, mode='nearest')
        labels = data.squeeze().cpu().numpy()
    labels = np.asarray(labels, np.float32)
    ch, cw = result.layers.valid_mask.shape[-2:]
    placement = np.array([[cw / labels.shape[1], 0, result.crop_offset[0]],
                          [0, ch / labels.shape[0], result.crop_offset[1]], [0,0,1]], np.float64)
    mapping = sample.transform @ np.linalg.inv(result.calibration.affine_matrix)
    transform = np.diag((scale, scale, 1)) @ mapping @ placement
    prediction = cv2.warpPerspective(labels, transform, size, flags=cv2.INTER_NEAREST).astype(np.uint16)
    return score_procedural_instances(truth, prediction, seed_diameter_px=sample.seed_diameter*scale,
        annotations_are_complete=sample.complete,
        false_positive_weight=policy.reference_error_overreach_weight,
        overreach_distance_scale_fraction=policy.reference_error_distance_scale_fraction,
        minimum_match_iou=policy.reference_error_minimum_match_iou,
        missed_seed_weight=policy.reference_error_missed_seed_weight,
        incorrect_concavity_weight=policy.reference_error_concavity_weight).loss


def mask_target(sample, kind):
    target = np.full(sample.corrected_shape, np.nan, np.float32)
    foreground = sample.kwargs.get('foreground_reference_mask')
    background = sample.kwargs.get('background_reference_mask')
    other = sample.kwargs.get('foreground_exclusion_mask')
    if sample.labels is not None:
        foreground = (sample.labels > 0) if foreground is None else (foreground | (sample.labels > 0))
    for mask, value in ((background, 1 if kind == 'background' else 0),
                        (other, 1 if kind == 'other' else 0),
                        (foreground, 1 if kind == 'foreground' else 0)):
        if mask is not None:
            target[np.asarray(mask, bool)] = value
    return target


def boundary_target(sample):
    if sample.labels is None or not np.any(sample.labels):
        raise Ineligible('Complete reviewed instances are required for contour targets.')
    # The mask and erosion convention depend only on frozen references/scale.
    from seedvision.annotation.instance_references import instance_boundary_references
    refs = instance_boundary_references(sample.labels, sample.seed_diameter, interior_buffer_fraction=.08)
    target = np.full(sample.corrected_shape, np.nan, np.float32)
    target[refs.safe_interior] = 0
    target[refs.physical_edge] = 1
    return target


class ProductionEvaluator:
    mode = 'In-sample project adaptation; not independent validation'

    def __init__(self, graph, samples, *, cancelled=lambda: False, cache_limit=3):
        self.samples = tuple(samples)
        if not self.samples:
            raise Ineligible('No images with applied references are available.')
        self.cancelled = cancelled
        self.cache_limit = max(1, cache_limit)
        self.caches = OrderedDict()
        self.snapshots = {}
        self.policy = production_settings(graph)['procedural_settings']
        self.targets = {}
        self.loaded_target_files = set()
        self.previews = {}
        self.selected_samples = {}
        self.checkpoints = {}
        for identifier in ('unet_instances', 'stardist_instances'):
            if identifier in graph.nodes and graph.node(identifier).enabled:
                for sample in self.samples:
                    configured = Path(graph.node(identifier).parameters['checkpoint_path'])
                    path = configured if configured.is_absolute() else Path(sample.kwargs.get('learning_root', '.')) / configured
                    if not path.is_file():
                        raise Ineligible(f'{identifier}: load a compatible checkpoint before optimizing an enabled learned branch: {path}')
                    self.checkpoints[str(path.resolve())] = file_digest(path)
        self.provenance = {'objective_version': 1, 'image_weighting': 'equal per image',
            'checkpoints': self.checkpoints,
            'libraries': [str(getattr(getattr(s.kwargs.get('species_library'), 'manifest', None), 'content_sha256', 'none')) for s in self.samples],
            'sources': [{'key': s.key, 'path': str(s.path), 'sha256': s.fingerprint,
                'transform': None if s.transform is None else s.transform.tolist(), 'complete': s.complete, 'group': s.group,
                'annotations': None if s.labels is None else array_digest(s.labels), 'excluded_instance_ids': list(s.excluded_ids),
                'custom': {key: array_digest(value) for key, value in s.custom.items()},
                'target_path': None if s.target_path is None else str(s.target_path),
                'target_sha256': s.target_fingerprint,
                'reference_context': {key: repr(s.kwargs.get(key)) for key in (
                    'seed_instance_traits', 'seed_trait_species', 'seed_trait_coat_patterns',
                    'seed_trait_conditions', 'manual_seed_centres', 'species', 'biological_context')},
                'references': {key: array_digest(value) for key, value in s.kwargs.items() if isinstance(value, np.ndarray)}}
                for s in self.samples]}

    def check_snapshot(self):
        if self.cancelled():
            raise OptimizationCancelled()
        for path, digest in self.checkpoints.items():
            if file_digest(path) != digest:
                raise ValueError(f'Checkpoint changed during optimization: {path}')
        for sample in self.samples:
            if file_digest(sample.path) != sample.fingerprint:
                raise ValueError(f'Source changed during optimization: {sample.path.name}')
            if sample.target_path is not None and file_digest(sample.target_path) != sample.target_fingerprint:
                raise ValueError(f'Reviewed targets changed during optimization: {sample.target_path.name}')

    def run(self, graph, sample):
        from seedvision.segmentation.baseline import PipelineAnalysisCache, analyze_path
        if self.cancelled():
            raise OptimizationCancelled()
        signature = {key: dict(node.parameters) for key, node in graph.nodes.items()}
        previous = self.snapshots.get(sample.key)
        dirty = set(graph.nodes) if previous is None else set()
        if previous is not None:
            for key, values in signature.items():
                if previous.get(key) != values:
                    dirty.update(affected_nodes(graph, key))
        cache = self.caches.setdefault(sample.key, PipelineAnalysisCache())
        self.caches.move_to_end(sample.key)
        while len(self.caches) > self.cache_limit:
            evicted, _ = self.caches.popitem(last=False)
            self.snapshots.pop(evicted, None)
        kwargs = dict(sample.kwargs)
        kwargs.update(production_settings(graph))
        kwargs.update(enabled_nodes=frozenset(key for key, node in graph.nodes.items() if node.enabled),
                      node_cache=cache, dirty_nodes=frozenset(dirty), cancellation_requested=self.cancelled)
        kwargs['background_colour_enabled'] = graph.node('background_likelihood').parameters['background_colour_enabled']
        try:
            result = analyze_path(sample.path, **kwargs)
        except Exception as error:
            self.caches.pop(sample.key, None)
            self.snapshots.pop(sample.key, None)
            if self.cancelled():
                raise OptimizationCancelled() from error
            raise
        if sample.transform is None:
            sample.transform = np.asarray(result.calibration.affine_matrix).copy()
            sample.seed_diameter = result.estimated_seed_diameter_px
            for source in self.provenance['sources']:
                if source['key'] == sample.key:
                    source['transform'] = sample.transform.tolist()
        if not np.allclose(result.calibration.affine_matrix, sample.transform, rtol=0, atol=1e-7):
            self.caches.pop(sample.key, None)
            self.snapshots.pop(sample.key, None)
            raise ValueError('Candidate changes the frozen reference coordinate frame.')
        if sample.target_path is not None and sample.key not in self.loaded_target_files:
            sample.custom = load_targets(sample.target_path, sample)
            self.loaded_target_files.add(sample.key)
            for source in self.provenance['sources']:
                if source['key'] == sample.key:
                    source['custom'] = {key: array_digest(value) for key, value in sample.custom.items()}
                    source['target_path'] = str(sample.target_path)
                    source['target_sha256'] = sample.target_fingerprint
        self.snapshots[sample.key] = signature
        return result

    def parameters(self, graph, identifier, names):
        self.check_snapshot()
        exclusions = {}
        if identifier == 'illumination_decomposition':
            targets = set()
            for sample in self.samples:
                if not eligibility_reason(graph, identifier, sample):
                    self.run(graph, sample)
                    targets.update(sample.custom)
            shadow = identifier + '__shadow_likelihood' in targets
            highlight = identifier + '__highlight_likelihood' in targets
            for key, supported in (
                ('shadow_z_threshold', shadow), ('highlight_z_threshold', highlight),
                ('lighting_deviation_scale_fraction', shadow or highlight),
                ('lighting_extreme_softness', shadow or highlight)):
                if not supported:
                    exclusions[key] = 'Import reviewed shadow/highlight targets; these outputs do not drive instance separation.'
        if identifier == 'procedural_instances':
            requirements = {
                'boundary_semantic_floor': 'reference_edge_probability',
                'boundary_nonphysical_discount': 'reference_edge_probability',
                'boundary_physical_ridge_weight': 'reference_edge_probability',
                'boundary_trace_weight': 'edge_traces', 'trace_convexity_weight': 'edge_traces',
                'trace_minimum_length_fraction': 'edge_traces',
                'centre_validated_oval_weight': 'seed_edge_curves',
                'centre_flattened_grayscale_weight': 'illumination_decomposition',
                'boundary_surface_darkening_weight': 'surface_darkness_gradients'}
            for key, owner in requirements.items():
                if owner not in graph.nodes or not graph.node(owner).enabled:
                    exclusions[key] = f'Required {owner} input is disabled.'
            evidence = {
                'boundary_semantic_floor': ('edge_supported_physical_compatibility', 'edge_supported_nonphysical_compatibility', 'locally_normalized_net_physical_edge'),
                'boundary_nonphysical_discount': ('edge_supported_nonphysical_compatibility',),
                'boundary_physical_ridge_weight': ('normalized_net_reference_edge_ridges',),
                'boundary_trace_weight': ('edge_trace_continuity',),
                'trace_minimum_length_fraction': ('edge_trace_continuity',),
                'trace_convexity_weight': ('edge_trace_continuity',),
                'centre_validated_oval_weight': ('oval_centre_probability',),
            }
            present = {key: False for key in evidence}
            for sample in self.samples:
                if eligibility_reason(graph, identifier, sample):
                    continue
                result = self.run(graph, sample)
                for key, fields in evidence.items():
                    for name in fields:
                        value = getattr(result.layers, name, None)
                        if value is not None and bool(tensor(value).any().item()):
                            present[key] = True
            for key, available in present.items():
                if not available:
                    exclusions[key] = 'No nonzero supporting evidence on the selected reference images.'
        return tuple(key for key in names if key not in exclusions), exclusions

    def score(self, graph, identifier):
        selected = self.selected_samples.get(identifier)
        values = {}
        reasons = []
        for sample in self.samples:
            if selected is not None and sample.key not in selected:
                continue
            try:
                # Preflight target availability before expensive production work.
                reason = eligibility_reason(graph, identifier, sample)
                if reason:
                    raise Ineligible(reason)
                kind = OBJECTIVES[identifier]
                if kind in {'instances', 'unet', 'stardist', 'legacy instances', 'proposals', 'edges', 'boundary'}:
                    if sample.labels is None or not np.any(sample.labels):
                        raise Ineligible('Apply shape-reviewed complete instances.')
                derived = identifier in DERIVED_DIAGNOSTICS and sample.labels is not None and np.any(sample.labels)
                if kind == 'reviewed raster' and not derived and sample.target_path is None and not any(key.startswith(identifier+'__') for key in sample.custom):
                    raise Ineligible('Import reviewed targets for this diagnostic; seed outlines do not establish its classes.')
                result = self.run(graph, sample)
                values[sample.key] = float(self.score_result(identifier, sample, result))
            except Ineligible as error:
                if selected is not None:
                    raise ValueError(f'Candidate lost a previously eligible image: {sample.key}: {error}') from error
                reasons.append(f'{sample.path.name}: {error}')
        if not values:
            raise Ineligible('; '.join(reasons) or 'No compatible reference targets.')
        if selected is None:
            self.selected_samples[identifier] = frozenset(values)
            self.provenance.setdefault('excluded_images', {})[identifier] = reasons
        return values

    def score_result(self, identifier, sample, result):
        kind = OBJECTIVES[identifier]
        layers = result.layers
        if kind in {'instances', 'unet', 'stardist', 'legacy instances'}:
            output = {'instances': result.procedural_instances, 'unet': result.unet_instances,
                      'stardist': result.stardist_instances, 'legacy instances': layers.instance_labels}[kind]
            if output is None:
                raise Ineligible('Enable and connect the downstream instance node (and load its checkpoint if learned).')
            labels = output.labels if hasattr(output, 'labels') else output
            loss = instance_loss(sample, labels, result, self.policy)
            if identifier == 'illumination_decomposition':
                lighting = [raster_loss(result.advanced.rasters[name], sample.custom[identifier+'__'+name], result, sample.transform)
                            for name in CUSTOM_OUTPUTS[identifier] if identifier+'__'+name in sample.custom]
                if lighting:
                    loss = mean([loss, *lighting])
            return loss
        if kind in {'material', 'colour', 'noise', 'texture'}:
            outputs = {
                'material': (layers.seed_material_probability, layers.nonseed_material_probability),
                'colour': (result.foreground_colour_probability, layers.background_likelihood),
                'noise': (layers.foreground_noise_likelihood, layers.refined_background_likelihood),
                'texture': (layers.reference_seed_surface_probability, layers.reference_background_texture_probability)}[kind]
            for name in ('foreground', 'background'):
                if (sample.key, name) not in self.targets:
                    self.targets[sample.key, name] = mask_target(sample, name)
            targets = [self.targets[sample.key, name] for name in ('foreground', 'background')]
            if kind == 'material':
                targets[1] = 1 - targets[0]
            scores = [raster_loss(value, target, result, sample.transform) for value, target in zip(outputs, targets)]
            return mean(scores)
        if kind in {'edges', 'ceiling edges', 'boundary'}:
            if (sample.key, 'boundary') not in self.targets:
                self.targets[sample.key, 'boundary'] = boundary_target(sample)
            target = self.targets[sample.key, 'boundary']
            if kind == 'boundary':
                values = (result.advanced.rasters.get('boundary_magnitude'),)
            elif kind == 'ceiling edges':
                name = 'weak_lightening_surface_gradient' if identifier == 'lightening_gradient_ceiling' else 'weak_darkening_surface_gradient'
                values = (getattr(layers, name),)
            else:
                values = (layers.physical_edge_probability, layers.reference_edge_probability,
                          layers.locally_normalized_net_physical_edge, layers.normalized_net_reference_edge_ridges)
            return mean(raster_loss(value, target, result, sample.transform) for value in values)
        if kind in {'traits', 'wrinkling'}:
            scores = []
            for annotation in sample.kwargs.get('seed_instance_traits', ()):
                if sample.labels is None or not np.any(sample.labels == annotation.seed_id):
                    continue
                mask = sample.labels == annotation.seed_id
                values = []
                if kind == 'wrinkling' and annotation.conditions_reviewed:
                    values.append((result.advanced.rasters.get('wrinkling_likelihood'), float('wrinkled' in annotation.conditions)))
                if kind == 'traits':
                    if annotation.coat_pattern:
                        values.extend((value, float(name == annotation.coat_pattern)) for name, value in layers.reference_seed_coat_probabilities)
                    if annotation.conditions_reviewed:
                        values.extend((value, float(name in annotation.conditions)) for name, value in layers.reference_seed_condition_probabilities)
                for value, truth in values:
                    target = np.full(sample.corrected_shape, np.nan, np.float32)
                    target[mask] = truth
                    scores.append(raster_loss(value, target, result, sample.transform))
            if not scores:
                raise Ineligible('Review compatible coat/condition labels; unreviewed conditions are not negatives.')
            return mean(scores)
        if kind == 'proposals':
            return self.proposal_loss(sample, result)
        custom = {key: value for key, value in sample.custom.items() if key.startswith(identifier+'__')}
        if not custom and identifier in DERIVED_DIAGNOSTICS:
            cache_key = (sample.key, identifier)
            if cache_key not in self.targets:
                self.targets[cache_key] = self.derived_diagnostic_target(identifier, sample, result)
            custom = self.targets[cache_key]
        scores = []
        for key, target in custom.items():
            parts = key.split('__')
            if parts[0] != identifier:
                continue
            name = parts[1]
            if name == 'contact_pairs':
                mapping = self.reference_proposal_map(sample, result)
                pairs = {tuple(sorted((mapping[int(a)], mapping[int(b)]))) for a, b, _ in result.advanced.contact_pairs
                         if int(a) in mapping and int(b) in mapping}
                scores.append(mean((float(tuple(sorted((int(a), int(b)))) in pairs)-truth)**2 for a,b,truth in target))
                continue
            if name in {'colour', 'pattern'}:
                names = getattr(result.advanced, name+'_class_names')
                if len(parts) != 3 or parts[2] not in names:
                    raise Ineligible(f'Choose one of the actual {name} classes: {names}')
                value = getattr(result.advanced, name+'_probabilities')[names.index(parts[2])]
            elif name == 'ellipse_likelihood':
                value = result.advanced.hue_rasters[name][1]
            else:
                value = result.advanced.rasters.get(name)
            scores.append(raster_loss(value, target, result, sample.transform))
        if not scores:
            raise Ineligible('No compatible reviewed output targets.')
        return mean(scores)

    def reference_proposal_map(self, sample, result):
        from seedvision.segmentation.procedural_fit import _maximum_weight_assignment
        if sample.labels is None or not np.any(sample.labels):
            raise Ineligible('Complete instance references are needed to identify reviewed contacts.')
        ids = np.unique(sample.labels)
        ids = ids[ids > 0]
        centres = []
        for identifier in ids:
            ys, xs = np.where(sample.labels == identifier)
            centres.append((xs.mean(), ys.mean()))
        if not result.proposals:
            return {}
        points = np.array([(p.center_x, p.center_y) for p in result.proposals])
        distances = np.linalg.norm(np.asarray(centres)[:,None] - points[None], axis=-1)
        pairs = _maximum_weight_assignment(np.maximum(0., sample.seed_diameter*.5-distances))
        return {int(col)+1: int(ids[row]) for row,col in pairs}

    def derived_diagnostic_target(self, identifier, sample, result):
        if sample.labels is None or not np.any(sample.labels):
            raise Ineligible('Complete reviewed instances are required for this diagnostic objective.')
        labels = sample.labels
        boundary = boundary_target(sample)
        if identifier == 'ellipse_likelihood':
            return {identifier+'__ellipse_likelihood': boundary}
        if identifier in {'touching_split', 'contact_graph'}:
            ids = np.unique(labels)
            ids = ids[ids > 0]
            if len(ids) < 2:
                raise Ineligible('At least two complete references are needed to assess contact/split evidence.')
            # Positive labels describe visible adjacency only, not occlusion order.
            kernel = np.ones((3,3), np.uint8)
            contact = np.zeros(labels.shape, bool)
            pairs = []
            for index, a in enumerate(ids):
                expanded = cv2.dilate(np.uint8(labels == a), kernel) > 0
                neighbours = set(np.unique(labels[expanded])) - {0, a}
                for b in ids[index+1:]:
                    touching = b in neighbours
                    pairs.append((int(a), int(b), float(touching)))
                    if touching:
                        contact |= expanded & (cv2.dilate(np.uint8(labels == b), kernel)>0)
            if identifier == 'contact_graph':
                return {identifier+'__contact_pairs': np.asarray(pairs, np.float32)}
            target = np.where(np.isfinite(boundary), 0., np.nan).astype(np.float32)
            target[contact] = 1.
            if not contact.any():
                raise Ineligible('Add a reviewed touching pair for split optimization, or import explicit split targets.')
            return {identifier+'__touching_split_likelihood': target}
        # Assignment/risk targets compare a frozen baseline assignment against
        # reviewed identities. Only this compact categorical evaluation is CPU.
        import torch.nn.functional as functional
        prediction = tensor(result.layers.instance_labels)
        height, width = labels.shape
        scale = min(1., 1024/max(height,width))
        size = (max(1, round(width*scale)), max(1,round(height*scale)))
        ch,cw = result.layers.valid_mask.shape[-2:]
        local_size = (max(1,round(ch*scale)), max(1,round(cw*scale)))
        local = functional.interpolate(prediction[None,None], size=local_size, mode='nearest').squeeze().cpu().numpy()
        placement = np.array([[1,0,result.crop_offset[0]*scale],[0,1,result.crop_offset[1]*scale]], np.float32)
        predicted = cv2.warpAffine(local, placement, size, flags=cv2.INTER_NEAREST).astype(np.uint16)
        truth = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
        mapping = self.reference_proposal_map(sample, result)
        correct = np.zeros(truth.shape, np.float32)
        for predicted_id, reference_id in mapping.items():
            correct[(predicted == predicted_id) & (truth == reference_id)] = 1
        valid = (truth > 0) | (sample.complete & (predicted > 0))
        target = np.full(truth.shape, np.nan, np.float32)
        target[valid] = correct[valid] if identifier == 'assignment_confidence' else 1-correct[valid]
        full = cv2.resize(target, (width,height), interpolation=cv2.INTER_NEAREST)
        name = 'instance_assignment_confidence' if identifier == 'assignment_confidence' else 'proposal_disagreement'
        return {identifier+'__'+name: full}

    def proposal_loss(self, sample, result):
        from seedvision.segmentation.procedural_fit import _maximum_weight_assignment
        ids = np.unique(sample.labels)
        centres = []
        for identifier in ids[ids > 0]:
            ys, xs = np.where(sample.labels == identifier)
            centres.append((xs.mean(), ys.mean()))
        proposals = result.proposals
        if not proposals:
            return 1.0
        predicted = np.array([(p.center_x, p.center_y) for p in proposals])
        distances = np.linalg.norm(np.asarray(centres)[:, None] - predicted[None], axis=-1) / max(1, sample.seed_diameter)
        pairs = _maximum_weight_assignment(np.maximum(0., .5-distances))
        matched = {col for _, col in pairs}
        loss = 1 - len(pairs) / max(1, len(centres))
        loss += sum(distances[r,c]**2 for r,c in pairs) / max(1, len(centres))
        loss += sum((p.confidence - float(i in matched))**2 for i,p in enumerate(proposals)
                    if sample.complete or i in matched) / max(1, len(proposals))
        if sample.complete:
            loss += (len(proposals)-len(matched)) / max(1, len(centres))
        return float(loss)

    def accept(self, graph, identifier):
        # run() compares the full last-evaluated signature, including rejections;
        # the next evaluation therefore always repairs candidate cache state.
        pass

    def final_scores(self, graph):
        if 'procedural_instances' not in graph.nodes or not graph.node('procedural_instances').enabled:
            return {}
        values = {}
        for sample in self.samples:
            if sample.labels is not None and np.any(sample.labels):
                result = self.run(graph, sample)
                if result.procedural_instances is not None:
                    values[sample.key] = instance_loss(sample, result.procedural_instances.labels, result, self.policy)
                    import base64
                    labels = np.asarray(result.procedural_instances.labels)
                    preview = cv2.resize(labels.astype(np.float32), (384, max(1, round(384*labels.shape[0]/labels.shape[1]))), interpolation=cv2.INTER_NEAREST).astype(np.uint32)
                    colours = np.stack(((preview*67)%255, (preview*127)%255, (preview*199)%255), axis=-1).astype(np.uint8)
                    x, y = result.crop_offset
                    ch, cw = result.layers.valid_mask.shape[-2:]
                    crop = result.calibration.corrected_bgr[y:y+ch, x:x+cw]
                    rgb = cv2.resize(crop, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_AREA)
                    selected = preview > 0
                    rgb[selected] = (rgb[selected]*.55 + colours[selected]*.45).astype(np.uint8)
                    encoded = base64.b64encode(cv2.imencode('.png', rgb)[1]).decode('ascii')
                    record = self.previews.setdefault(sample.key, {'name': sample.path.name, 'before': encoded})
                    record['after'] = encoded
        return values


def write_record(path, payload):
    """Atomic, finite, inspectable project-adjacent optimization journal."""
    path = Path(path)
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def verify_record_sources(provenance):
    """Recheck external inputs after the user has reviewed a proposal."""
    for source in provenance.get('sources', ()):
        if file_digest(source['path']) != source['sha256']:
            raise ValueError('A source image changed while reviewing optimization results.')
        if source.get('target_path') and file_digest(source['target_path']) != source['target_sha256']:
            raise ValueError('Reviewed optimization targets changed while reviewing results.')
    for path, digest in provenance.get('checkpoints', {}).items():
        if file_digest(path) != digest:
            raise ValueError('A learned checkpoint changed while reviewing results.')
