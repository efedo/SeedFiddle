"""Shared graph-to-calculation settings contract for desktop, batch and search."""
from dataclasses import fields

BASELINE_NODES = ('seed_scale_estimation', 'background_likelihood', 'distance_candidates', 'identification', 'circle_candidates')
LAYER_NODES = ('layout_detection', 'wavelet_decomposition', 'background_likelihood',
    'refined_background_likelihood', 'edge_gradients', 'surface_darkness_gradients',
    'lightening_gradient_ceiling', 'darkening_gradient_ceiling', 'frequency_noise_masks',
    'reference_texture_prototypes', 'material_evidence_decision', 'reference_seed_traits',
    'reference_edge_probability', 'edge_traces', 'instance_masks', 'seed_edge_curves')

def production_settings(graph):
    from seedvision.segmentation.baseline import BaselineSettings
    from seedvision.calibration import CalibrationSettings
    from seedvision.calibration.geometry import DishDetectionSettings
    from seedvision.visualization.layers import AnalysisLayerSettings
    from seedvision.visualization.advanced import AdvancedAnalysisSettings, ADVANCED_NODE_MODES
    from seedvision.segmentation.procedural import ProceduralInstanceSettings
    from seedvision.learning.pipeline import UNetPipelineSettings, StarDistPipelineSettings

    def build(cls, owners, *, active_only=True):
        values = {}
        for owner in owners:
            if owner in graph.nodes and (not active_only or graph.is_active(owner)):
                values.update(graph.node(owner).parameters)
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})
    return dict(settings=build(BaselineSettings, BASELINE_NODES),
        calibration_settings=build(CalibrationSettings, ('ruler_detection', 'deskew_colour'),active_only=False),
        dish_settings=build(DishDetectionSettings, ('layout_detection',),active_only=False),
        layer_settings=build(AnalysisLayerSettings, LAYER_NODES),
        advanced_settings=build(AdvancedAnalysisSettings, ADVANCED_NODE_MODES),
        procedural_settings=build(ProceduralInstanceSettings, ('procedural_instances',)),
        unet_settings=build(UNetPipelineSettings, ('unet_instances',),active_only=False),
        stardist_settings=build(StarDistPipelineSettings, ('stardist_instances',),active_only=False))

