"""Seed instance segmentation and review-proposal generation."""

from seedvision.segmentation.baseline import (
    BackgroundSamplingBand,
    BaselineAnalysis,
    BaselineSettings,
    PipelineAnalysisCache,
    SeedProposal,
    analyze_image,
    analyze_path,
)
from seedvision.calibration import CalibrationSettings, DishDetectionSettings
from seedvision.visualization import AdvancedAnalysisSettings, AnalysisLayerSettings
from seedvision.segmentation.procedural import (
    MANUAL_CENTRE_SUPPRESSION_DIAMETER_FRACTION,
    ManualSeedCentreMode,
    ManualSeedCentreSpace,
    ManualSeedCentres,
    PreparedProceduralInstanceInputs,
    ProceduralMarkerSource,
    ProceduralInstanceResult,
    ProceduralInstanceSettings,
    prepare_procedural_instance_inputs,
    procedural_seed_instances,
    procedural_seed_instances_from_prepared,
)
from seedvision.segmentation.procedural_fit import (
    DEFAULT_PROCEDURAL_FIT_PARAMETERS,
    ProceduralFitOptions,
    ProceduralFitParameter,
    ProceduralFitResult,
    ProceduralFitScore,
    ProceduralFitTrial,
    fit_procedural_settings,
    score_procedural_instances,
)
from seedvision.segmentation.reference_edge_fit import (
    ReferenceEdgeFitResult,
    ReferenceEdgeFitScore,
    evaluate_reference_edge_settings,
    fit_reference_edge_parameters,
)

__all__ = [
    "BaselineAnalysis",
    "BackgroundSamplingBand",
    "BaselineSettings",
    "CalibrationSettings",
    "DishDetectionSettings",
    "PipelineAnalysisCache",
    "MANUAL_CENTRE_SUPPRESSION_DIAMETER_FRACTION",
    "ManualSeedCentreMode",
    "ManualSeedCentreSpace",
    "ManualSeedCentres",
    "PreparedProceduralInstanceInputs",
    "ProceduralMarkerSource",
    "ProceduralInstanceResult",
    "ProceduralInstanceSettings",
    "DEFAULT_PROCEDURAL_FIT_PARAMETERS",
    "ProceduralFitOptions",
    "ProceduralFitParameter",
    "ProceduralFitResult",
    "ProceduralFitScore",
    "ProceduralFitTrial",
    "ReferenceEdgeFitResult",
    "ReferenceEdgeFitScore",
    "AnalysisLayerSettings",
    "AdvancedAnalysisSettings",
    "SeedProposal",
    "analyze_image",
    "analyze_path",
    "prepare_procedural_instance_inputs",
    "procedural_seed_instances",
    "procedural_seed_instances_from_prepared",
    "fit_procedural_settings",
    "evaluate_reference_edge_settings",
    "fit_reference_edge_parameters",
    "score_procedural_instances",
]
