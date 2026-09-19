"""One capability catalogue shared by the inspector, planner and search engine."""
from __future__ import annotations

from dataclasses import dataclass


EXEMPT = {
    'project': 'Project data and references are not search parameters.',
    'metadata': 'Species and specimen metadata are facts.',
    'species_reference_library': 'Library provenance is fixed; consuming nodes own blend controls.',
    'ruler_detection': 'Ruler length and tick spacing are physical facts.',
    'hue_only': 'Fixed colour conversion has no adjustable calculation.',
    'measurements': 'Deterministic measurements have no exposed estimator parameters.',
    'review': 'Human decisions are reference data; this card is not implemented.',
    'classification': 'This card is not implemented.',
    'aggregation': 'This card is not implemented.',
    'output': 'This card is not implemented.',
}

# Every implemented computational card has an objective. Additional reviewed
# raster targets can be supplied for the conditional diagnostic objectives.
OBJECTIVES = {
    'deskew_colour': 'material', 'layout_detection': 'material',
    'wavelet_decomposition': 'edges', 'seed_scale_estimation': 'instances',
    'background_likelihood': 'colour', 'refined_background_likelihood': 'noise',
    'edge_gradients': 'edges', 'surface_darkness_gradients': 'instances',
    'frequency_noise_masks': 'texture', 'reference_texture_prototypes': 'texture',
    'material_evidence_decision': 'material', 'reference_seed_traits': 'traits',
    'reference_edge_probability': 'edges', 'edge_traces': 'instances',
    'seed_edge_curves': 'instances', 'boundary_normals': 'boundary',
    'illumination_decomposition': 'instances', 'image_quality': 'reviewed raster',
    'procedural_instances': 'instances', 'unet_instances': 'unet',
    'stardist_instances': 'stardist', 'wrinkling': 'wrinkling',
    'pattern_decomposition': 'reviewed raster', 'colour_probabilities': 'reviewed raster',
    'circle_candidates': 'proposals', 'lightening_gradient_ceiling': 'ceiling edges',
    'darkening_gradient_ceiling': 'ceiling edges', 'distance_candidates': 'proposals',
    'identification': 'proposals', 'touching_split': 'reviewed raster',
    'instance_masks': 'legacy instances', 'ellipse_likelihood': 'reviewed raster',
    'assignment_confidence': 'reviewed raster', 'radial_profile': 'reviewed raster',
    'proposal_disagreement': 'reviewed raster', 'contact_graph': 'reviewed raster',
    'coat_damage': 'reviewed raster', 'calibration_residuals': 'reviewed raster',
}

FIXED = {
    'ruler_length_mm': 'Known physical ruler length.',
    'minor_tick_mm': 'Known physical tick spacing.',
    'expected_outer_diameter_mm': 'Known physical vessel diameter.',
    'shape_calibration_uncertainty_fraction': 'Measured calibration uncertainty.',
    'shape_boundary_perturbation_radius_px': 'Fixed reference sensitivity convention.',
    'checkpoint_path': 'Checkpoint identity; use Learning to train or select a model.',
    'reference_texture_weight': 'Legacy fallback only; resolved material is authoritative.',
    'tile_size': 'Frozen inference tiling policy.', 'tile_overlap': 'Frozen inference tiling policy.',
    'downsample_max_dimension': 'Frozen calibration working resolution.',
    'max_deskew_degrees': 'Frozen annotation coordinate frame; independent calibration geometry is not supplied by seed masks.',
    'apply_perspective_correction': 'Frozen annotation coordinate frame; independent calibration geometry is not supplied by seed masks.',
    'max_perspective_fraction': 'Frozen annotation coordinate frame; independent calibration geometry is not supplied by seed masks.',
}

OBJECTIVE_DESCRIPTIONS = {
    'material': 'Balanced binary log loss on foreground/nonseed material probabilities.',
    'colour': 'Balanced binary log loss on foreground/background colour probabilities.',
    'noise': 'Balanced binary log loss on foreground/background noise probabilities.',
    'texture': 'Balanced binary log loss on seed-surface/background-texture probabilities.',
    'edges': 'Mean balanced log loss on physical, supported, normalized and ridge outputs; fixed contours/interiors.',
    'ceiling edges': 'Balanced log loss on the filtered gradient magnitude; fixed contours/interiors.',
    'boundary': 'Balanced log loss on boundary magnitude; fixed contours/interiors.',
    'instances': 'One-to-one asymmetric procedural instance loss on a fixed reviewed domain.',
    'legacy instances': 'One-to-one asymmetric legacy instance loss on a fixed reviewed domain.',
    'unet': 'One-to-one asymmetric U-Net instance loss on a fixed reviewed domain.',
    'stardist': 'One-to-one asymmetric StarDist instance loss on a fixed reviewed domain.',
    'proposals': 'One-to-one centre recall, squared localization and confidence error; fixed seed scale.',
    'traits': 'Balanced log loss for reviewed coat/condition labels; unknown conditions ignored.',
    'wrinkling': 'Balanced log loss for reviewed wrinkle presence on complete instances.',
    'reviewed raster': 'Balanced log loss on reviewed output pixels; contact pairs use squared binary error.',
}


@dataclass(frozen=True)
class Capability:
    node_id: str
    objective: str
    searchable: tuple[str, ...]
    fixed: tuple[tuple[str, str], ...]
    reason: str = ''


def capability(node) -> Capability:
    if node.identifier not in OBJECTIVES and node.identifier not in EXEMPT:
        raise ValueError(f'Optimization disposition missing for {node.identifier}')
    fixed = []
    searchable = []
    for spec in node.parameter_specs:
        reason = EXEMPT.get(node.identifier) or FIXED.get(spec.key)
        if spec.key.startswith('reference_error_'):
            reason = 'Fixed scoring policy or coverage declaration; never minimize the penalty itself.'
        elif 'working_maximum_dimension' in spec.key or spec.key == 'reference_edge_minimum_working_seed_diameter_px':
            reason = 'Frozen analysis resolution and resource policy.'
        elif spec.key.endswith('_reference_source'):
            reason = 'Frozen training-source policy and provenance.'
        elif spec.display_only:
            reason = 'Presentation only.'
        if reason:
            fixed.append((spec.key, reason))
        else:
            if spec.kind not in {'int', 'float', 'bool', 'choice'}:
                raise ValueError(f'Unclassified optimization control: {node.identifier}.{spec.key}')
            searchable.append(spec.key)
    return Capability(node.identifier, OBJECTIVES.get(node.identifier, ''), tuple(searchable),
                      tuple(fixed), EXEMPT.get(node.identifier, ''))


def affected_nodes(graph, identifier):
    """Include actual shared producers, not just the owning inspector card."""
    roots = {identifier}
    if identifier == 'reference_edge_probability':
        roots.add('reference_texture_prototypes')
    affected = set(roots)
    for root in roots:
        if root in graph.nodes:
            affected.update(graph.downstream(root, recursive=True))
    return frozenset(affected)


def candidate_values(spec, current, pass_number=0):
    if spec.kind == 'bool':
        return (not current,)
    if spec.kind == 'choice':
        return tuple(value for value in spec.choices if value != current)
    # Relative steps avoid spending the entire search near huge range limits.
    step = max(float(spec.step or 1e-3), abs(float(current)) * .20) * (.5 ** pass_number)
    values = []
    for direction in (-1, 1):
        value = max(spec.minimum, min(spec.maximum, current + direction * step))
        if spec.kind == 'int':
            value = int(round(value))
            if value == current:
                value = int(max(spec.minimum, min(spec.maximum, current + direction)))
        if value != current and value not in values:
            values.append(value)
    return tuple(values)
