"""Interactive seed-instance annotation algorithms."""

from seedvision.annotation.continuity import (
    InstanceContinuitySummary,
    summarize_instance_continuity,
)
from seedvision.annotation.instance_references import (
    InstanceBoundaryReferences,
    instance_boundary_references_from_edge_evidence,
    internal_edge_candidate_mask,
    instance_boundary_references,
)
from seedvision.annotation.shape_guided import (
    EllipseHypothesis,
    SHAPE_FILL_CONNECTIVITY,
    ShapeGuidedFillOptions,
    ShapeGuidedFillRegion,
    fit_rotated_edge_ellipse,
    shape_guided_fill_instance,
    shape_guided_fill_region,
    shape_outward_extension_pressure,
)
from seedvision.annotation.tools import (
    ANNOTATION_EDGE_SOURCES,
    EDGE_TRACE_EDGE_SOURCES,
    SMART_FILL_EDGE_SOURCES,
    EdgeTraceOptions,
    SmartFillOptions,
    SmartFillRegion,
    smart_fill_instance,
    smart_fill_region,
    snap_edge_point,
    trace_edge_path,
)
from seedvision.annotation.seed_traits import (
    SeedTraitCatalogue,
    SpeciesSeedTraitVocabulary,
    load_seed_trait_catalogue,
    trait_display_name,
)

__all__ = (
    "ANNOTATION_EDGE_SOURCES",
    "EDGE_TRACE_EDGE_SOURCES",
    "EdgeTraceOptions",
    "EllipseHypothesis",
    "InstanceBoundaryReferences",
    "InstanceContinuitySummary",
    "SHAPE_FILL_CONNECTIVITY",
    "SMART_FILL_EDGE_SOURCES",
    "ShapeGuidedFillOptions",
    "ShapeGuidedFillRegion",
    "SmartFillOptions",
    "SmartFillRegion",
    "SeedTraitCatalogue",
    "SpeciesSeedTraitVocabulary",
    "fit_rotated_edge_ellipse",
    "instance_boundary_references",
    "instance_boundary_references_from_edge_evidence",
    "internal_edge_candidate_mask",
    "load_seed_trait_catalogue",
    "shape_guided_fill_instance",
    "shape_guided_fill_region",
    "shape_outward_extension_pressure",
    "smart_fill_instance",
    "smart_fill_region",
    "snap_edge_point",
    "summarize_instance_continuity",
    "trace_edge_path",
    "trait_display_name",
)
