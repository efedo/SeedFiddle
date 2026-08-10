"""Toolkit-independent model for the visual seed-analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable


class NodeStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"
    WARNING = "warning"
    BLOCKED = "blocked"
    PLANNED = "planned"
    BYPASSED = "bypassed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    key: str
    label: str
    kind: str
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    description: str = ""
    choices: tuple[str, ...] = ()


@dataclass(slots=True)
class PipelineNode:
    identifier: str
    title: str
    category: str
    description: str
    x: float
    y: float
    details: str = ""
    enabled: bool = True
    implemented: bool = True
    bypassable: bool = False
    status: NodeStatus = NodeStatus.IDLE
    status_detail: str = "Not run"
    parameters: dict[str, Any] = field(default_factory=dict)
    parameter_specs: tuple[ParameterSpec, ...] = ()
    input_ports: tuple[tuple[str, str], ...] = ()
    output_ports: tuple[tuple[str, str], ...] = ()
    inline_parameters: tuple[tuple[str, str], ...] = ()
    calculation_seconds: float | None = None

    def set_parameter(self, key: str, value: Any) -> None:
        spec = next((item for item in self.parameter_specs if item.key == key), None)
        if spec is None:
            raise KeyError(f"Node {self.identifier!r} has no parameter {key!r}.")
        if spec.kind == "float":
            value = float(value)
        elif spec.kind == "int":
            value = int(value)
        elif spec.kind == "bool":
            value = bool(value)
        elif spec.kind == "choice":
            value = str(value)
            if value not in spec.choices:
                raise ValueError(f"{value!r} is not valid for {spec.label}.")
        if spec.minimum is not None and value < spec.minimum:
            raise ValueError(f"{spec.label} must be at least {spec.minimum}.")
        if spec.maximum is not None and value > spec.maximum:
            raise ValueError(f"{spec.label} must be at most {spec.maximum}.")
        self.parameters[key] = value


@dataclass(frozen=True, slots=True)
class PipelineConnection:
    source: str
    target: str
    data_type: str
    source_port: str = ""
    target_port: str = ""


class PipelineGraph:
    """Directed acyclic graph plus mutable node configuration and status."""

    def __init__(
        self,
        nodes: Iterable[PipelineNode],
        connections: Iterable[PipelineConnection],
    ) -> None:
        node_list = list(nodes)
        self.nodes = {node.identifier: node for node in node_list}
        if len(self.nodes) != len(node_list):
            raise ValueError("Pipeline node identifiers must be unique.")
        self.connections = tuple(connections)
        self.revision = 0
        self._validate_connections()
        self.topological_order()

    def node(self, identifier: str) -> PipelineNode:
        try:
            return self.nodes[identifier]
        except KeyError as error:
            raise KeyError(f"Unknown pipeline node {identifier!r}.") from error

    def upstream(self, identifier: str) -> tuple[str, ...]:
        return tuple(
            connection.source
            for connection in self.connections
            if connection.target == identifier
        )

    def downstream(self, identifier: str, *, recursive: bool = False) -> tuple[str, ...]:
        direct = [
            connection.target
            for connection in self.connections
            if connection.source == identifier
        ]
        if not recursive:
            return tuple(direct)
        visited: set[str] = set()
        pending = list(direct)
        while pending:
            current = pending.pop(0)
            if current in visited:
                continue
            visited.add(current)
            pending.extend(self.downstream(current))
        order = self.topological_order()
        return tuple(identifier for identifier in order if identifier in visited)

    def topological_order(self) -> tuple[str, ...]:
        indegree = {identifier: 0 for identifier in self.nodes}
        outgoing: dict[str, list[str]] = {identifier: [] for identifier in self.nodes}
        for connection in self.connections:
            indegree[connection.target] += 1
            outgoing[connection.source].append(connection.target)
        ready = [identifier for identifier in self.nodes if indegree[identifier] == 0]
        result: list[str] = []
        while ready:
            identifier = ready.pop(0)
            result.append(identifier)
            for target in outgoing[identifier]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if len(result) != len(self.nodes):
            raise ValueError("Pipeline connections must form a directed acyclic graph.")
        return tuple(result)

    def set_parameter(self, node_id: str, key: str, value: Any) -> tuple[str, ...]:
        node = self.node(node_id)
        previous = node.parameters.get(key)
        node.set_parameter(key, value)
        if node.parameters[key] == previous:
            return ()
        self.revision += 1
        affected = (node_id, *self.downstream(node_id, recursive=True))
        self.invalidate(affected)
        return affected

    def set_enabled(self, node_id: str, enabled: bool) -> tuple[str, ...]:
        node = self.node(node_id)
        enabled = bool(enabled)
        if node.enabled == enabled:
            return ()
        node.enabled = enabled
        node.status = NodeStatus.IDLE if enabled else NodeStatus.BYPASSED
        node.status_detail = "Not run" if enabled else "Bypassed"
        self.revision += 1
        affected = (node_id, *self.downstream(node_id, recursive=True))
        self.invalidate(affected, preserve_bypassed=True)
        return affected

    def set_status(
        self, node_id: str, status: NodeStatus, detail: str = ""
    ) -> None:
        node = self.node(node_id)
        node.status = status
        node.status_detail = detail or status.value.capitalize()

    def invalidate(
        self, node_ids: Iterable[str], *, preserve_bypassed: bool = False
    ) -> None:
        for node_id in node_ids:
            node = self.node(node_id)
            if preserve_bypassed and not node.enabled:
                node.status = NodeStatus.BYPASSED
                node.status_detail = "Bypassed"
            elif not node.implemented:
                node.status = NodeStatus.PLANNED
                node.status_detail = "Planned"
            else:
                node.status = NodeStatus.IDLE
                node.status_detail = "Not run"

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "nodes": [
                {
                    "id": node.identifier,
                    "title": node.title,
                    "category": node.category,
                    "description": node.description,
                    "details": node.details,
                    "position": [node.x, node.y],
                    "enabled": node.enabled,
                    "implemented": node.implemented,
                    "bypassable": node.bypassable,
                    "parameters": dict(node.parameters),
                    "input_ports": list(node.input_ports),
                    "output_ports": list(node.output_ports),
                    "inline_parameters": list(node.inline_parameters),
                    "calculation_seconds": node.calculation_seconds,
                }
                for node in self.nodes.values()
            ],
            "connections": [
                {
                    "source": connection.source,
                    "target": connection.target,
                    "data_type": connection.data_type,
                    "source_port": connection.source_port,
                    "target_port": connection.target_port,
                }
                for connection in self.connections
            ],
        }

    def _validate_connections(self) -> None:
        for connection in self.connections:
            if connection.source not in self.nodes:
                raise ValueError(f"Unknown connection source {connection.source!r}.")
            if connection.target not in self.nodes:
                raise ValueError(f"Unknown connection target {connection.target!r}.")
            if connection.source == connection.target:
                raise ValueError("A pipeline node cannot connect to itself.")


def build_default_pipeline() -> PipelineGraph:
    """Create the default raw-image-to-report workflow."""

    colour_reference_parameters = (
        ParameterSpec(
            "apply_colour_balance",
            "Apply neutral balance",
            "bool",
            description=(
                "Estimate BGR channel gains from the least-chromatic detected swatches."
            ),
        ),
    )
    ruler_parameters = (
        ParameterSpec(
            "ruler_length_mm",
            "Visible ruler span",
            "float",
            10.0,
            1000.0,
            1.0,
            "Nominal millimetres covered by the detected long ruler axis.",
        ),
    )
    deskew_parameters = (
        ParameterSpec(
            "max_deskew_degrees",
            "Maximum correction",
            "float",
            0.0,
            30.0,
            0.5,
            "Reject a combined reference rotation larger than this value.",
        ),
        ParameterSpec(
            "apply_perspective_correction",
            "Correct camera tilt",
            "bool",
            description=(
                "Rectify the detected card quadrilateral to remove mild projective "
                "keystone distortion as well as rotation."
            ),
        ),
        ParameterSpec(
            "max_perspective_fraction",
            "Maximum perspective strength",
            "float",
            0.0,
            0.50,
            0.01,
            "Reject projective correction when opposing card-side lengths differ "
            "by more than this fraction.",
        ),
    )
    scale_parameters = (
        ParameterSpec(
            "minor_tick_mm",
            "Minor tick interval",
            "float",
            0.1,
            10.0,
            0.1,
            "Physical spacing represented by adjacent minor ruler ticks.",
        ),
    )
    layout_parameters = (
        ParameterSpec("downsample_max_dimension", "Detection resolution", "int", 512, 8000, 128, "Maximum image dimension used for the CUDA dish-rim search."),
        ParameterSpec("hough_accumulator_threshold", "Rim support strictness", "int", 5, 200, 1, "Higher values require more radial gradient support around the rim."),
        ParameterSpec("min_radius_fraction", "Minimum radius / height", "float", 0.05, 0.47, 0.01, "Smallest dish radius as a fraction of corrected image height."),
        ParameterSpec("max_radius_fraction", "Maximum radius / height", "float", 0.06, 0.48, 0.01, "Largest dish radius as a fraction of corrected image height."),
        ParameterSpec("expected_center_x_fraction", "Expected centre X", "float", 0.05, 0.95, 0.01, "Weak selection prior as a fraction of image width."),
        ParameterSpec("expected_center_y_fraction", "Expected centre Y", "float", 0.05, 0.95, 0.01, "Weak selection prior as a fraction of image height."),
        ParameterSpec("expected_radius_fraction", "Expected radius / height", "float", 0.05, 0.48, 0.005, "Weak selection prior for choosing among detected circles."),
        ParameterSpec("rim_pair_search_fraction", "Dual-rim search / radius", "float", 0.03, 0.30, 0.005, "Radial range searched around the selected Petri-dish circle for the second concentric glass edge."),
        ParameterSpec("rim_pair_min_separation_fraction", "Minimum rim separation", "float", 0.005, 0.29, 0.005, "Smallest accepted separation between lower and upper glass edges, relative to dish radius."),
        ParameterSpec("rim_pair_max_separation_fraction", "Maximum rim separation", "float", 0.01, 0.30, 0.005, "Largest accepted separation between lower and upper glass edges, relative to dish radius."),
        ParameterSpec("rim_pair_expected_separation_fraction", "Expected rim separation", "float", 0.005, 0.30, 0.005, "Soft preference used when several concentric glass-edge pairs have comparable full-circumference support."),
        ParameterSpec("rim_pair_secondary_support_fraction", "Secondary-rim support", "float", 0.05, 1.0, 0.05, "Minimum strength of the second glass edge relative to the strongest dense radial-profile peak."),
    )
    seed_scale_parameters = (
        ParameterSpec(
            "reference_scale_factor",
            "Reference correction",
            "float",
            0.50,
            1.00,
            0.01,
            "Multiplies the median equivalent diameter of up to three isolated "
            "reference-seed components. Lower values compensate more strongly "
            "for low-contrast shadow included in those components.",
        ),
        ParameterSpec("reference_roi_x_min", "Reference ROI left", "float", 0.0, 0.99, 0.01, "Left edge as a fraction of corrected image width."),
        ParameterSpec("reference_roi_x_max", "Reference ROI right", "float", 0.01, 1.0, 0.01, "Right edge as a fraction of corrected image width."),
        ParameterSpec("reference_roi_y_min", "Reference ROI top", "float", 0.0, 0.99, 0.01, "Top edge as a fraction of corrected image height."),
        ParameterSpec("reference_roi_y_max", "Reference ROI bottom", "float", 0.01, 1.0, 0.01, "Bottom edge as a fraction of corrected image height."),
        ParameterSpec("reference_colour_distance_threshold", "Reference colour distance", "float", 2.0, 50.0, 0.5, "Minimum weighted Lab distance from the ROI median."),
        ParameterSpec("reference_max_aspect_ratio", "Maximum component aspect", "float", 1.0, 8.0, 0.1, "Reject more elongated reference components above this ratio."),
        ParameterSpec("reference_max_components", "Components used", "int", 1, 10, 1, "Maximum largest accepted components contributing to median diameter."),
        ParameterSpec("fallback_diameter_fraction", "Fallback diameter / dish radius", "float", 0.05, 0.40, 0.01, "Used only when no isolated reference component survives."),
    )
    foreground_parameters = (
        ParameterSpec(
            "inner_radius_fraction",
            "Outer-rim area used",
            "float",
            0.75,
            0.96,
            0.01,
            "Restricts segmentation and candidate centres to this fraction of "
            "the detected upper/outer dish radius, excluding the bright rim.",
        ),
        ParameterSpec(
            "foreground_otsu_fraction",
            "Foreground threshold",
            "float",
            0.40,
            1.40,
            0.01,
            "The actual threshold is max(8, Otsu × this value). Lower values "
            "retain more low-contrast material; higher values reject more dish background.",
        ),
        ParameterSpec("foreground_chroma_weight", "Lab chroma weight", "float", 0.5, 5.0, 0.1, "Relative contribution of each Lab chroma channel to foreground distance."),
        ParameterSpec("foreground_morphology_fraction", "Morphology kernel / diameter", "float", 0.01, 0.20, 0.005, "Opening/closing kernel size relative to estimated seed diameter."),
        ParameterSpec("foreground_background_prior_tolerance", "Perimeter colour tolerance", "float", 2.0, 100.0, 1.0, "Maximum weighted Lab distance from the outside-dish prior admitted when refining the automatic dish background."),
        ParameterSpec("foreground_probability_softness_fraction", "Probability softness", "float", 0.02, 1.0, 0.01, "Width of the soft probability transition around the foreground threshold, relative to that threshold."),
        ParameterSpec("foreground_reference_weight", "Foreground reference influence", "float", 0.0, 1.0, 0.05, "How strongly colours represented in the painted foreground distribution enhance seed probability across the dish."),
        ParameterSpec("foreground_local_contrast_scale_fraction", "Local contrast scale / diameter", "float", 0.03, 0.60, 0.01, "Gaussian neighbourhood, relative to seed diameter, used to distinguish locally bright seed surfaces from darker inter-seed gaps."),
        ParameterSpec("foreground_shadow_rejection_strength", "Shadow rejection", "float", 0.0, 3.0, 0.05, "Weight of seed-scale local lightness evidence in crowded dishes. It ramps down automatically when ample true tray is visible; increase when dark gaps are mistaken for seeds."),
        ParameterSpec("foreground_reference_components", "Reference colour modes", "int", 1, 8, 1, "Maximum robust Lab mixture components fitted to the painted foreground area."),
        ParameterSpec("foreground_distribution_fit_iterations", "Distribution fit rounds", "int", 1, 20, 1, "Robust clustering rounds used to fit each painted foreground colour distribution."),
        ParameterSpec("foreground_refinement_iterations", "Reference refinement rounds", "int", 0, 8, 1, "Number of cautious self-refinement rounds after fitting the painted foreground pixels."),
        ParameterSpec("foreground_refinement_min_probability", "Refinement acceptance", "float", 0.50, 0.99, 0.01, "Only pixels at or above this foreground colour-membership probability can enter a refinement round."),
        ParameterSpec("foreground_frequency_weight_power", "Mode frequency influence", "float", 0.0, 1.0, 0.05, "Weights each learned colour mode by how frequently it occurs in the painted foreground mask. Zero treats all represented colours equally; one uses their painted-area proportions directly."),
        ParameterSpec("foreground_distribution_scale_multiplier", "Colour tolerance multiplier", "float", 0.50, 3.00, 0.05, "Expands or contracts every learned foreground Lab colour mode. Increase when matching unpainted seed pixels receive too little probability."),
    )
    distance_parameters = (
        ParameterSpec("distance_blur_fraction", "Distance smoothing / diameter", "float", 0.0, 0.12, 0.005, "Gaussian smoothing of the distance map relative to seed diameter."),
        ParameterSpec("distance_neighborhood_fraction", "Peak neighbourhood / diameter", "float", 0.20, 1.20, 0.01, "Local-maximum suppression neighbourhood relative to seed diameter."),
        ParameterSpec("distance_min_depth_fraction", "Minimum peak depth / diameter", "float", 0.03, 0.40, 0.01, "Minimum distance from foreground boundary required for a peak."),
        ParameterSpec("distance_proposal_radius_fraction", "Proposal radius / diameter", "float", 0.10, 0.80, 0.01, "Provisional radius assigned to distance-only candidates."),
    )
    circle_parameters = (
        ParameterSpec(
            "circle_accumulator_threshold",
            "Circle strictness",
            "int",
            10,
            40,
            1,
            "Normalized CUDA ring-support threshold. Lower values accept weaker "
            "circular edge evidence and usually produce more candidates.",
        ),
        ParameterSpec(
            "dense_circle_relaxation",
            "Dense-image circle relaxation",
            "int",
            0,
            10,
            1,
            "Subtract this much ring strictness when the distance branch already "
            "indicates a densely packed seed image.",
        ),
        ParameterSpec(
            "dense_distance_candidate_threshold",
            "Dense-image peak count",
            "int",
            10,
            300,
            1,
            "Distance-peak count at which the circle detector becomes more sensitive.",
        ),
        ParameterSpec("circle_edge_threshold", "Circle edge threshold", "int", 10, 200, 1, "Minimum normalized CUDA gradient retained by the ring bank."),
        ParameterSpec("circle_working_maximum_dimension", "GPU working dimension", "int", 512, 4096, 128, "Maximum circle-bank image dimension. Resizing, convolution, and coordinate restoration all stay on GPU."),
        ParameterSpec("circle_min_distance_fraction", "Minimum centre spacing / diameter", "float", 0.20, 1.20, 0.01, "Local-maximum spacing between returned CUDA ring centres."),
        ParameterSpec("circle_min_radius_fraction", "Minimum radius / diameter", "float", 0.05, 0.80, 0.01, "Smallest tested CUDA ring radius relative to estimated seed diameter."),
        ParameterSpec("circle_max_radius_fraction", "Maximum radius / diameter", "float", 0.10, 1.20, 0.01, "Largest tested CUDA ring radius relative to estimated seed diameter."),
    )
    fusion_parameters = (
        ParameterSpec(
            "merge_distance_fraction",
            "Duplicate merge distance",
            "float",
            0.25,
            0.75,
            0.01,
            "A distance-transform candidate within this fraction of an existing "
            "CUDA ring centre is confidence-weighted into that candidate.",
        ),
        ParameterSpec("circle_confidence", "Circle candidate confidence", "float", 0.10, 1.00, 0.01, "Weight assigned to every CUDA ring proposal during fusion."),
        ParameterSpec("distance_confidence", "Distance candidate confidence", "float", 0.10, 1.00, 0.01, "Weight assigned to every distance-peak proposal during fusion."),
    )
    background_parameters = (
        ParameterSpec("background_sample_radius_fraction", "Reference brush radius / diameter", "float", 0.01, 0.20, 0.005, "Visible painted-brush radius relative to estimated seed diameter; every covered pixel enters the reference calculation."),
        ParameterSpec("background_chroma_percentile", "Automatic chroma percentile", "float", 1.0, 99.0, 1.0, "Automatic samples must be no more chromatic than this percentile."),
        ParameterSpec("background_lightness_percentile", "Automatic lightness percentile", "float", 1.0, 99.0, 1.0, "Automatic samples must be at least this lightness percentile."),
        ParameterSpec("background_minimum_sample_fraction", "Minimum automatic area", "float", 0.0001, 0.25, 0.001, "Minimum fraction of valid dish pixels retained as automatic background references; the lightest, least-chromatic candidates are added if percentile filters return less."),
        ParameterSpec("background_prior_tolerance", "Perimeter colour tolerance", "float", 2.0, 100.0, 1.0, "Maximum weighted Lab distance from the outside-dish median admitted to the automatic background training set."),
        ParameterSpec("background_lightness_scale_floor", "Lightness range floor", "float", 1.0, 40.0, 0.5, "Minimum robust Lab lightness spread used by the colour probability model."),
        ParameterSpec("background_chroma_scale_floor", "Chroma range floor", "float", 0.5, 30.0, 0.5, "Minimum robust Lab a/b spread used by the colour probability model."),
        ParameterSpec("background_colour_components", "Reference colour modes", "int", 1, 8, 1, "Maximum robust Lab mixture components fitted to the painted or automatic background samples."),
        ParameterSpec("background_distribution_fit_iterations", "Distribution fit rounds", "int", 1, 20, 1, "Robust clustering rounds used to fit the background colour distribution."),
        ParameterSpec("background_refinement_iterations", "Reference refinement rounds", "int", 0, 8, 1, "Number of cautious high-confidence expansion and refit rounds after the initial background distribution."),
        ParameterSpec("background_refinement_min_probability", "Refinement acceptance", "float", 0.50, 0.99, 0.01, "Only pixels at or above this background membership probability can enter a refinement round."),
        ParameterSpec("background_frequency_weight_power", "Mode frequency influence", "float", 0.0, 1.0, 0.05, "Weights each learned background colour mode by its occurrence frequency in the painted mask. Zero treats represented modes equally; one applies their area proportions directly."),
        ParameterSpec("background_distribution_scale_multiplier", "Colour tolerance multiplier", "float", 0.50, 3.00, 0.05, "Expands or contracts every learned background Lab colour mode before per-pixel probability is calculated."),
    )
    noise_parameters = (
        ParameterSpec("noise_medium_scale_fraction", "Medium band / diameter", "float", 0.005, 0.20, 0.005, "Medium Gaussian frequency boundary relative to seed diameter."),
        ParameterSpec("noise_coarse_scale_fraction", "Coarse band / diameter", "float", 0.01, 0.40, 0.005, "Coarse Gaussian frequency boundary relative to seed diameter."),
        ParameterSpec("noise_direction_step_degrees", "Direction interval (degrees)", "int", 5, 90, 5, "Angular interval between one-sided texture-continuation rays. Fifteen degrees produces 24 unique direction overlays."),
        ParameterSpec("noise_vector_length_fraction", "Ray length / diameter", "float", 0.05, 2.00, 0.05, "Length of every directed texture-continuation ray relative to the master seed diameter."),
        ParameterSpec("noise_vector_sample_count", "Samples per ray", "int", 2, 32, 1, "Number of positions geometrically averaged along each directed ray."),
        ParameterSpec("noise_vector_decay", "Ray sample decay", "float", 0.10, 1.00, 0.02, "Multiplicative weight retained by each successively more distant ray sample."),
        ParameterSpec("noise_direction_integration", "Direction integration", "choice", choices=("mean", "maximum", "minimum", "median"), description="How the directional overlays are merged. Mean is a normalized sum; maximum reaches object borders most readily; minimum requires background continuity in every direction."),
        ParameterSpec("noise_background_min_likelihood", "Confident background minimum", "int", 0, 255, 1, "Minimum colour-likelihood value used as a texture background pseudo-label."),
        ParameterSpec("noise_nonbackground_max_likelihood", "Confident non-background maximum", "int", 0, 255, 1, "Maximum colour-likelihood value used as a texture non-background pseudo-label."),
        ParameterSpec("noise_working_maximum_dimension", "GPU working dimension", "int", 512, 4096, 128, "Maximum directional-noise analysis dimension. Work and upsampling stay on GPU; selected overlays retain the full crop dimensions."),
    )
    edge_parameters = (
        ParameterSpec("edge_blur_sigma", "Pre-edge blur sigma", "float", 0.1, 5.0, 0.1, "Gaussian sigma applied in Lab before Scharr derivatives."),
        ParameterSpec("edge_chroma_weight", "Edge chroma weight", "float", 0.1, 5.0, 0.1, "Relative contribution of Lab a/b gradients."),
        ParameterSpec("edge_normalization_percentile", "Strength normalization percentile", "float", 80.0, 99.9, 0.1, "Response percentile mapped to maximum overlay brightness."),
        ParameterSpec("edge_strength_gamma", "Strength display gamma", "float", 0.10, 2.00, 0.05, "Below 1 brightens weak edges; above 1 suppresses them."),
    )
    shared_edge_values = {
        "edge_blur_sigma": 1.2,
        "edge_chroma_weight": 1.5,
        "edge_normalization_percentile": 99.0,
        "edge_strength_gamma": 0.65,
    }
    instance_parameters = (
        ParameterSpec("instance_min_extent_fraction", "Minimum mask extent / diameter", "float", 0.20, 1.20, 0.01, "Minimum circular extent retained around each proposal."),
        ParameterSpec("instance_max_extent_fraction", "Maximum mask extent / diameter", "float", 0.30, 1.50, 0.01, "Maximum circular extent retained around each proposal."),
        ParameterSpec("instance_radius_extent_multiplier", "Proposal-radius extent", "float", 0.50, 3.00, 0.05, "Multiplies each proposal radius when limiting its CUDA Voronoi region."),
    )
    curve_parameters = (
        ParameterSpec("curve_diameter_multiplier", "Master diameter multiplier", "float", 0.25, 3.00, 0.05, "Scales the image-specific global seed diameter before all curve radii and inward profile distances are calculated."),
        ParameterSpec("curve_radius_low_fraction", "Low curve radius / diameter", "float", 0.05, 1.00, 0.01, "Tightest of three curvature hypotheses. Every edge pixel is scored at all three radii and keeps the best result."),
        ParameterSpec("curve_radius_nominal_fraction", "Nominal curve radius / diameter", "float", 0.06, 1.10, 0.01, "Middle curvature hypothesis; 0.50 corresponds to the radius of a nominal seed diameter."),
        ParameterSpec("curve_radius_high_fraction", "High curve radius / diameter", "float", 0.07, 1.20, 0.01, "Broadest curvature hypothesis for larger seeds or flatter visible boundary sections."),
        ParameterSpec("curve_arc_angle_degrees", "Arc support angle", "float", 5.0, 80.0, 1.0, "How far around the tested circle to sample corroborating edges on both sides. It controls arc span, while the radius controls curvature."),
        ParameterSpec("curve_orientation_tolerance_degrees", "Tangent tolerance", "float", 1.0, 45.0, 1.0, "Gaussian sigma for directed-tangent error: support is exp(-0.5 × (error / tolerance)²)."),
        ParameterSpec("curve_near_distance_fraction", "Near profile distance / diameter", "float", 0.0, 0.60, 0.005, "First inward lightness sample distance."),
        ParameterSpec("curve_middle_distance_fraction", "Middle profile distance / diameter", "float", 0.005, 0.70, 0.005, "Second inward lightness sample distance."),
        ParameterSpec("curve_deep_distance_fraction", "Deep profile distance / diameter", "float", 0.01, 0.80, 0.005, "Deepest inward lightness sample distance."),
        ParameterSpec("curve_min_darkening", "Minimum inward lightness gain", "float", 0.0, 30.0, 0.5, "Lab lightness increase from rim to interior before support begins."),
        ParameterSpec("curve_full_darkening", "Full-support lightness gain", "float", 1.0, 100.0, 1.0, "Additional rim-to-interior Lab gain mapped to full darkening support."),
        ParameterSpec("curve_lightness_boost", "Lightness confidence boost", "float", 0.0, 3.0, 0.05, "Maximum multiplicative confidence enhancement from inward lightening. Zero ignores the profile; missing lightness evidence never rejects an otherwise supported arc."),
    )
    ridge_parameters = (
        ParameterSpec("ridge_nms_step_px", "Normal sampling step", "float", 0.25, 3.0, 0.05, "Distance sampled on both sides of each gradient maximum during subpixel non-maximum suppression."),
        ParameterSpec("ridge_low_threshold", "Hysteresis low", "float", 0.0, 0.95, 0.01, "Weak thinned edges retained only when connected to a strong ridge."),
        ParameterSpec("ridge_high_threshold", "Hysteresis high", "float", 0.01, 1.0, 0.01, "Strong thinned-edge threshold that seeds hysteresis reconstruction."),
        ParameterSpec("ridge_hysteresis_iterations", "Hysteresis reach", "int", 1, 32, 1, "Maximum GPU morphology steps through which connected weak ridges can be retained."),
    )
    trace_parameters = (
        ParameterSpec("trace_tangent_tolerance_degrees", "Link tangent tolerance", "float", 2.0, 60.0, 1.0, "Maximum axial tangent disagreement used to join ridge pixels into one trace."),
        ParameterSpec("trace_maximum_gap_px", "Maximum trace gap", "int", 1, 5, 1, "Largest tangent-aligned pixel gap bridged by the oriented GPU component linker."),
        ParameterSpec("trace_window_fraction", "Continuity window / diameter", "float", 0.03, 0.75, 0.01, "Distance followed in each tangent direction when measuring trace continuity."),
        ParameterSpec("trace_sample_count", "Continuity samples", "int", 2, 32, 1, "Number of tangent-following samples taken on each side of a ridge pixel."),
        ParameterSpec("trace_minimum_length_fraction", "Minimum trace length / diameter", "float", 0.02, 2.0, 0.01, "Oriented components with fewer pixels than this seed-relative length are rejected."),
        ParameterSpec("trace_junction_max_neighbors", "Junction neighbour limit", "int", 1, 8, 1, "Ridge pixels with more local neighbours are split before trace labelling to avoid merging crossing or touching boundaries."),
    )
    boundary_trace_parameters = (
        ParameterSpec("curve_diameter_multiplier", "Master diameter multiplier", "float", 0.25, 3.00, 0.05, "Scales the image-specific diameter used by tracing, radius priors and semantic-side sampling."),
        ParameterSpec("boundary_radius_min_fraction", "Minimum fitted radius / diameter", "float", 0.05, 1.40, 0.01, "Smallest circle-curvature hypothesis relative to the master diameter."),
        ParameterSpec("boundary_radius_max_fraction", "Maximum fitted radius / diameter", "float", 0.06, 1.50, 0.01, "Largest circle-curvature hypothesis; elongated lupin sides can require values above one half."),
        ParameterSpec("boundary_radius_sample_count", "Radius hypotheses", "int", 3, 25, 1, "Number of radii evaluated without quantizing the continuous source tangent field."),
        ParameterSpec("boundary_arc_span_degrees", "Arc span", "float", 10.0, 160.0, 2.0, "Total traced arc support sampled around a central ridge pixel."),
        ParameterSpec("boundary_arc_sample_count", "Arc samples", "int", 5, 41, 2, "Number of positions sampled across every circle hypothesis; missing samples are tolerated."),
        ParameterSpec("boundary_orientation_tolerance_degrees", "Arc tangent tolerance", "float", 2.0, 60.0, 1.0, "Angular tolerance between observed and predicted tangents along an arc."),
        ParameterSpec("boundary_missing_support_floor", "Missing-edge floor", "float", 0.0, 0.75, 0.01, "Support retained for an occluded or locally missing arc sample instead of making it a hard failure."),
        ParameterSpec("boundary_circle_residual_tolerance", "Circle residual tolerance", "float", 0.02, 0.80, 0.01, "Relative radial residual mapped to circle-fit confidence."),
        ParameterSpec("boundary_ellipse_residual_tolerance", "Ellipse residual tolerance", "float", 0.02, 0.80, 0.01, "Normalized conic residual mapped to ellipse-fit confidence."),
        ParameterSpec("boundary_max_axis_ratio", "Maximum ellipse axis ratio", "float", 1.0, 4.0, 0.05, "Largest seed ellipse axis ratio accepted as plausible."),
        ParameterSpec("boundary_radius_log_tolerance", "Radius-prior tolerance", "float", 0.05, 1.5, 0.05, "Log-space width of the soft global-diameter agreement term."),
        ParameterSpec("boundary_center_vote_weight", "Centre-vote influence", "float", 0.0, 1.0, 0.05, "Contribution from orientation-aware normal votes for compatible seed centres."),
        ParameterSpec("boundary_center_vote_blur_fraction", "Centre-vote blur / diameter", "float", 0.005, 0.30, 0.005, "Seed-relative smoothing of fragmented-arc centre votes."),
        ParameterSpec("boundary_semantic_weight", "Semantic-side influence", "float", 0.0, 1.0, 0.05, "Soft influence of foreground/background evidence on which side of a trace is seed interior."),
        ParameterSpec("boundary_polarity_boost", "Lightness-polarity boost", "float", 0.0, 1.0, 0.05, "Optional confidence boost when the fitted interior brightens away from the boundary; never a hard requirement."),
        ParameterSpec("boundary_minimum_confidence", "Accepted-boundary minimum", "float", 0.0, 1.0, 0.01, "Final confidence below which a ridge is categorized in the rejection-reason overlay."),
        ParameterSpec("boundary_geometry_max_candidates", "Displayed fit candidates", "int", 16, 1024, 16, "Maximum compact centre/ellipse hypotheses downloaded only when the vector fit overlay is requested."),
        ParameterSpec("boundary_working_maximum_dimension", "GPU working dimension", "int", 512, 4096, 128, "Maximum boundary-tracing dimension. Float tensors are resized only on GPU and full-resolution overlays remain GPU-resident until viewed."),
    )
    seed_interior_parameters = (
        ParameterSpec("compute_device", "Compute device", "choice", choices=("cuda", "auto", "cpu"), description="CUDA uses PyTorch tensors on the NVIDIA GPU. Auto prefers CUDA and CPU explicitly disables GPU analysis."),
        ParameterSpec("allow_cpu_fallback", "Allow CPU fallback", "bool", description="Permit analysis to continue on CPU if CUDA cannot be initialized. Disable this to make missing CUDA a hard error."),
        ParameterSpec("maximum_dimension", "GPU working dimension", "int", 256, 4096, 128, "Maximum dish-crop dimension processed by the diagnostic tensor branch. Outputs are resampled to full crop size."),
        ParameterSpec("interior_smoothing_fraction", "Interior smoothing / diameter", "float", 0.005, 0.30, 0.005, "Gaussian smoothing scale applied to foreground evidence relative to the global seed diameter."),
        ParameterSpec("interior_background_weight", "Background-evidence weight", "float", 0.0, 1.0, 0.05, "Blend between the foreground-strength model and inverse background evidence."),
    )
    boundary_normal_parameters = (
        ParameterSpec("boundary_width_fraction", "Boundary width / diameter", "float", 0.005, 0.20, 0.005, "Width of the probability morphology band combined with colour/lightness gradients."),
    )
    touching_split_parameters = (
        ParameterSpec("split_neck_fraction", "Neck depth / diameter", "float", 0.05, 1.20, 0.01, "Controls how strongly shallow foreground necks support a split between touching seeds."),
    )
    ellipse_parameters = (
        ParameterSpec("ellipse_radial_tolerance", "Radial tolerance", "float", 0.03, 0.80, 0.01, "Tolerance around a proposal's normalized radius when combining multiscale structure-tensor support."),
    )
    disagreement_parameters = (
        ParameterSpec("disagreement_scale_fraction", "Evidence spread / diameter", "float", 0.02, 0.80, 0.01, "Gaussian spread used to compare circle, distance, interior, and ellipse evidence."),
    )
    assignment_parameters = (
        ParameterSpec("assignment_boundary_penalty", "Boundary penalty", "float", 0.0, 1.0, 0.05, "Reduces assignment confidence at uncertain or shared instance boundaries."),
    )
    contact_parameters = (
        ParameterSpec("contact_distance_multiplier", "Contact distance multiplier", "float", 0.5, 3.0, 0.05, "Two proposals are connected when their centre spacing is within this multiple of their summed radii."),
    )
    illumination_parameters = (
        ParameterSpec("illumination_scale_fraction", "Illumination scale / diameter", "float", 0.10, 4.0, 0.05, "Large Gaussian scale used to separate illumination from seed reflectance."),
    )
    image_quality_parameters = (
        ParameterSpec("quality_noise_scale_fraction", "Noise scale / diameter", "float", 0.005, 0.30, 0.005, "Local high-frequency scale used for sensor/noise risk."),
    )
    radial_parameters = (
        ParameterSpec("radial_bin_count", "Radial profile bins", "int", 4, 128, 1, "Number of normalized centre-to-edge bins used to estimate expected seed lightness."),
    )
    wrinkling_parameters = (
        ParameterSpec("wrinkle_scale_fraction", "Wrinkle scale / diameter", "float", 0.005, 0.30, 0.005, "Seed-relative band scale at which ridge-like lightness curvature is measured."),
    )
    damage_parameters = (
        ParameterSpec("damage_anomaly_scale_fraction", "Damage context / diameter", "float", 0.02, 0.80, 0.01, "Neighbourhood used to detect local colour anomalies that coincide with edges or radial residuals."),
    )
    pattern_parameters = (
        ParameterSpec("pattern_scale_fraction", "Pattern scale / diameter", "float", 0.02, 1.0, 0.01, "Middle spatial scale used to separate spots, mottling, patches, stripes, and bicolour regions."),
    )
    colour_probability_parameters = (
        ParameterSpec("colour_temperature", "Colour probability sharpness", "float", 1.0, 100.0, 1.0, "Softmax sharpness for broad white, yellow, green, red, brown, and black colour prototypes."),
    )
    calibration_residual_parameters = (
        ParameterSpec("calibration_residual_gain", "Residual-risk gain", "float", 0.10, 5.0, 0.10, "Scales reference-confidence and spatial extrapolation risk away from the card and ruler."),
    )
    nodes = (
        PipelineNode(
            "raw_images", "Raw images", "Input", "Laboratory image batch", 0, 40,
            status_detail="Select an image",
        ),
        PipelineNode(
            "metadata", "Species & metadata", "Input", "Species, lot and notes", 240, 40,
            status_detail="Choose species",
        ),
        PipelineNode(
            "colour_reference",
            "Colour-card swatches",
            "Calibration",
            "Detect a regular swatch grid and estimate neutral colour balance",
            480,
            -120,
            parameters={"apply_colour_balance": True},
            parameter_specs=colour_reference_parameters,
        ),
        PipelineNode(
            "ruler_detection",
            "Ruler detection",
            "Calibration",
            "Locate scale endpoints after colour-card deskew",
            960,
            100,
            parameters={"ruler_length_mm": 150.0},
            parameter_specs=ruler_parameters,
        ),
        PipelineNode(
            "deskew_colour",
            "Deskew & colour balance",
            "Calibration",
            "Rectify from the colour card and apply gamut-safe neutral gains",
            720,
            40,
            parameters={
                "max_deskew_degrees": 10.0,
                "apply_perspective_correction": True,
                "max_perspective_fraction": 0.25,
            },
            parameter_specs=deskew_parameters,
        ),
        PipelineNode(
            "layout_detection",
            "Layout detection",
            "Calibration",
            "Detect the lower and upper concentric glass edges of a Petri dish",
            960,
            -160,
            details=(
                "The corrected image is reduced to the configured maximum dimension, "
                "converted to grayscale, and Gaussian-blurred on the tensor device. "
                "A CUDA circle bank scores radial-gradient support across the configured radius interval. "
                "The editable centre/radius prior selects a primary circular edge, then a dense "
                "concentric radial profile resolves the lower/inner and upper/outer glass edges. "
                "Both edges are retained for layout review; the outer edge defines the complete "
                "vessel extent and is the radius supplied to every downstream analysis node. "
                "The result is tagged as a Petri-dish vessel so "
                "future vessel detectors can replace this geometry without changing downstream nodes."
            ),
            parameters={
                "downsample_max_dimension": 1600,
                "hough_accumulator_threshold": 32,
                "min_radius_fraction": 0.16,
                "max_radius_fraction": 0.29,
                "expected_center_x_fraction": 0.58,
                "expected_center_y_fraction": 0.40,
                "expected_radius_fraction": 0.225,
                "rim_pair_search_fraction": 0.14,
                "rim_pair_min_separation_fraction": 0.015,
                "rim_pair_max_separation_fraction": 0.12,
                "rim_pair_expected_separation_fraction": 0.045,
                "rim_pair_secondary_support_fraction": 0.30,
            },
            parameter_specs=layout_parameters,
        ),
        PipelineNode(
            "scale_calibration",
            "Absolute ruler scale",
            "Calibration",
            "Estimate pixels per millimetre from deskewed ruler ticks",
            1200,
            100,
            parameters={"minor_tick_mm": 1.0},
            parameter_specs=scale_parameters,
        ),
        PipelineNode(
            "seed_scale_estimation",
            "Seed scale estimate",
            "Segmentation",
            "Estimate a global seed diameter from isolated reference seeds",
            1200,
            -160,
            details=(
                "The editable isolated-reference region is converted to CIE Lab. "
                "Pixels sufficiently unlike "
                "the region median form connected components after morphological "
                "opening and closing. Components touching the crop edge, with extreme "
                "area/aspect, or implausible relative to the dish are rejected. The "
                "median equivalent diameter of the configured number of components "
                "is corrected for shadow inflation. If none survive, the editable "
                "dish-radius fallback is used."
            ),
            parameters={
                "reference_scale_factor": 0.72,
                "reference_roi_x_min": 0.43,
                "reference_roi_x_max": 0.60,
                "reference_roi_y_min": 0.66,
                "reference_roi_y_max": 0.84,
                "reference_colour_distance_threshold": 10.0,
                "reference_max_aspect_ratio": 3.0,
                "reference_max_components": 3,
                "fallback_diameter_fraction": 0.16,
            },
            parameter_specs=seed_scale_parameters,
        ),
        PipelineNode(
            "foreground_segmentation",
            "Dish foreground mask",
            "Segmentation",
            "Separate seed-like colour from the estimated dish background",
            1440,
            -200,
            details=(
                "The median CIE Lab colour in the 0.5 cm band immediately outside "
                "the detected dish starts background selection, preventing pale seeds "
                "from being mistaken for the background class. Similar in-dish pixels "
                "refine that estimate; painted background areas override it. Foreground "
                "strength is a weighted Lab distance, Otsu centres a soft grayscale "
                "probability transition. Seed-scale local lightness then suppresses dark "
                "inter-seed gaps while retaining locally brighter seed surfaces, and a "
                "seed-scaled kernel cleans the derived binary proposal mask. Painted "
                "foreground pixels fit a multimodal Lab distribution; cautious refinement "
                "rounds enhance high-confidence matches across the dish while keeping the "
                "painted area anchored. It is also hard-included after morphology. "
                "The inspector plots the painted distribution—or a diagnostic fit to "
                "automatic high-confidence foreground pixels—as probability contours "
                "over a CIE Lab colour-gamut slice. "
                "Diagnostics cover the full dish, while proposals retain an editable "
                "inset measured from the upper/outer rim."
            ),
            parameters={
                "inner_radius_fraction": 0.90,
                "foreground_otsu_fraction": 0.90,
                "foreground_chroma_weight": 1.8,
                "foreground_morphology_fraction": 0.06,
                "foreground_background_prior_tolerance": 24.0,
                "foreground_probability_softness_fraction": 0.18,
                "foreground_reference_weight": 0.75,
                "foreground_local_contrast_scale_fraction": 0.18,
                "foreground_shadow_rejection_strength": 1.0,
                "foreground_reference_components": 4,
                "foreground_distribution_fit_iterations": 6,
                "foreground_refinement_iterations": 2,
                "foreground_refinement_min_probability": 0.82,
                "foreground_frequency_weight_power": 0.35,
                "foreground_distribution_scale_multiplier": 1.50,
            },
            parameter_specs=foreground_parameters,
        ),
        PipelineNode(
            "distance_candidates",
            "Distance-peak candidates",
            "Segmentation",
            "Find seed-centre peaks inside connected foreground",
            1680,
            -300,
            details=(
                "A Euclidean distance transform measures each foreground pixel's "
                "distance from the mask boundary. After light Gaussian smoothing, "
                "local maxima are found using the configured seed-relative neighbourhood "
                "and minimum boundary depth. "
                "This branch can propose centres in touching foreground masses."
            ),
            parameters={
                "distance_blur_fraction": 0.025,
                "distance_neighborhood_fraction": 0.58,
                "distance_min_depth_fraction": 0.14,
                "distance_proposal_radius_fraction": 0.43,
            },
            parameter_specs=distance_parameters,
        ),
        PipelineNode(
            "circle_candidates",
            "Circle candidates",
            "Segmentation",
            "Find seed-like circular edges in the dish crop",
            1680,
            -100,
            details=(
                "The grayscale dish crop is blurred on CUDA and convolved with a "
                "bank of normalized ring kernels. Radius and centre-spacing limits are "
                "editable fractions of the estimated seed diameter. "
                "Candidates outside the configured inset from the upper/outer dish rim are discarded. "
                "This branch is complementary to distance peaks and is currently "
                "biased toward roughly round visible seed boundaries."
            ),
            parameters={
                "circle_accumulator_threshold": 22,
                "dense_circle_relaxation": 4,
                "dense_distance_candidate_threshold": 30,
                "circle_edge_threshold": 80,
                "circle_working_maximum_dimension": 1280,
                "circle_min_distance_fraction": 0.58,
                "circle_min_radius_fraction": 0.22,
                "circle_max_radius_fraction": 0.62,
            },
            parameter_specs=circle_parameters,
        ),
        PipelineNode(
            "identification",
            "Seed identification",
            "Segmentation",
            "Fuse circle and distance-peak candidates into review proposals",
            1920,
            -200,
            details=(
                "Distance peaks establish seed interiors; CUDA ring candidates then enter using "
                "their editable confidence weights. A ring candidate near a distance peak "
                "is merged using confidence-weighted centre coordinates and the larger "
                "radius. Otherwise it is retained as another seed proposal. Results "
                "are sorted top-to-bottom then left-to-right. These are untrained "
                "review proposals—not validated instance counts—and overlaps still "
                "require later mask correction."
            ),
            parameters={
                "merge_distance_fraction": 0.48,
                "circle_confidence": 0.62,
                "distance_confidence": 0.48,
            },
            parameter_specs=fusion_parameters,
            bypassable=True,
        ),
        PipelineNode(
            "background_likelihood",
            "Background colour",
            "Diagnostic overlay",
            "Automatic or user-referenced per-pixel background likelihood",
            1320,
            180,
            parameters={
                "background_sample_radius_fraction": 0.10,
                "background_chroma_percentile": 50.0,
                "background_lightness_percentile": 55.0,
                "background_minimum_sample_fraction": 0.002,
                "background_prior_tolerance": 24.0,
                "background_lightness_scale_floor": 8.0,
                "background_chroma_scale_floor": 3.0,
                "background_colour_components": 4,
                "background_distribution_fit_iterations": 6,
                "background_refinement_iterations": 2,
                "background_refinement_min_probability": 0.82,
                "background_frequency_weight_power": 0.35,
                "background_distribution_scale_multiplier": 1.25,
            },
            parameter_specs=background_parameters,
            details=(
                "Painted areas, or pixels within a fixed weighted-colour tolerance of "
                "the median colour in the 0.5 cm outer dish-perimeter band, fit a multimodal "
                "CIE Lab probability distribution with a separate centre and spread for each mode. "
                "Optional iterative rounds admit only high-probability matches while retaining the "
                "painted pixels as anchors. Background and foreground reference areas are enforced "
                "as hard constraints in both colour and directional-noise maps. If too little "
                "matching tray is visible inside a crowded dish, the outer perimeter "
                "measurement remains authoritative instead of admitting seed colours. The node "
                "overlay outlines the exact annulus used for that initial estimate. The node "
                "inspector plots fitted membership contours over a local CIE Lab colour-gamut "
                "slice, including learned mode locations and reference frequencies. Its status "
                "reports the BGR range, selected area, and a warning when the final estimate "
                "deviates substantially from the perimeter prior."
            ),
            bypassable=True,
        ),
        PipelineNode(
            "instance_masks",
            "Instance colour masks",
            "Diagnostic overlay",
            "Unique, spatially contrasting provisional seed masks",
            2160,
            -200,
            parameters={
                "instance_min_extent_fraction": 0.72,
                "instance_max_extent_fraction": 0.95,
                "instance_radius_extent_multiplier": 1.55,
            },
            parameter_specs=instance_parameters,
        ),
        PipelineNode(
            "refined_background_likelihood",
            "Background noise profile",
            "Diagnostic overlay",
            "Directional frequency continuation learned from colour pseudo-labels",
            1440,
            180,
            parameters={
                "noise_medium_scale_fraction": 0.03,
                "noise_coarse_scale_fraction": 0.08,
                "noise_direction_step_degrees": 15,
                "noise_vector_length_fraction": 0.55,
                "noise_vector_sample_count": 9,
                "noise_vector_decay": 0.86,
                "noise_direction_integration": "maximum",
                "noise_background_min_likelihood": 190,
                "noise_nonbackground_max_likelihood": 65,
                "noise_working_maximum_dimension": 1280,
            },
            parameter_specs=noise_parameters,
            details=(
                "Fine, medium, and coarse texture distributions are learned from "
                "confident colour pseudo-labels. Texture likelihood is then "
                "separately evaluated across the exact dish-surrounding sampling "
                "annulus, without expanding unrelated downstream GPU crops. It is "
                "also "
                "geometrically accumulated along independent one-sided rays. At "
                "15° spacing there are 24 unique overlays (0° through 345°); "
                "the selected integration rule merges them into the refined map."
            ),
        ),
        PipelineNode(
            "edge_gradients",
            "Edge gradients",
            "GPU diagnostic",
            "Shared multichannel gradient field for both tangent encodings",
            1200,
            390,
            details=(
                "The corrected image is converted to CIE Lab once, blurred once, "
                "and differentiated with three CUDA Scharr filters. Weighted Lab "
                "gradients are fused before continuous atan2 orientation. The cached "
                "magnitude and tangent field feed the undirected and directed output "
                "ports without repeating image analysis."
            ),
            parameters=dict(shared_edge_values),
            parameter_specs=edge_parameters,
            input_ports=(("image", "Corrected image"),),
            output_ports=(
                ("undirected", "Undirected 0–180°"),
                ("directed", "Directed 0–360°"),
            ),
            inline_parameters=(
                ("edge_blur_sigma", "Blur σ"),
                ("edge_chroma_weight", "Chroma ×"),
                ("edge_normalization_percentile", "Norm %"),
                ("edge_strength_gamma", "Gamma"),
            ),
        ),
        PipelineNode(
            "undirected_edges",
            "Undirected edge tangents",
            "Diagnostic overlay",
            "Encode the shared tangent field axially; 0 degrees equals 180 degrees",
            1500,
            340,
        ),
        PipelineNode(
            "directed_edges",
            "Directed edge tangents",
            "Diagnostic overlay",
            "Encode shared polarity-aware tangents; 0 degrees equals 360 degrees",
            1500,
            500,
        ),
        PipelineNode(
            "edge_ridges",
            "Thinned edge ridges",
            "GPU diagnostic",
            "Non-maximum suppression and hysteresis on continuous gradients",
            1740,
            430,
            parameters={
                "ridge_nms_step_px": 1.0,
                "ridge_low_threshold": 0.10,
                "ridge_high_threshold": 0.24,
                "ridge_hysteresis_iterations": 8,
            },
            parameter_specs=ridge_parameters,
            inline_parameters=(
                ("ridge_nms_step_px", "NMS step"),
                ("ridge_low_threshold", "Low"),
                ("ridge_high_threshold", "High"),
                ("ridge_hysteresis_iterations", "Reach"),
            ),
            details=(
                "The float GPU gradient field is sampled on both sides of its normal. "
                "Only local maxima remain. Strong maxima seed hysteresis reconstruction, "
                "which retains connected weak ridges without downloading a Canny image."
            ),
        ),
        PipelineNode(
            "edge_traces",
            "Oriented edge traces",
            "GPU diagnostic",
            "Link tangent-compatible ridges, bridge short gaps and split junctions",
            1980,
            430,
            parameters={
                "trace_tangent_tolerance_degrees": 24.0,
                "trace_maximum_gap_px": 2,
                "trace_window_fraction": 0.20,
                "trace_sample_count": 7,
                "trace_minimum_length_fraction": 0.18,
                "trace_junction_max_neighbors": 4,
            },
            parameter_specs=trace_parameters,
            inline_parameters=(
                ("trace_tangent_tolerance_degrees", "Angle Â±Â°"),
                ("trace_maximum_gap_px", "Gap px"),
                ("trace_window_fraction", "Window Ã—D"),
                ("trace_minimum_length_fraction", "Min len Ã—D"),
            ),
            details=(
                "A GPU union/find graph joins ridge pixels only when endpoint tangents "
                "and their displacement agree. Configurable short gaps are bridged, "
                "over-connected junction pixels are split, and tangent-following samples "
                "measure continuity and missing-edge support on both sides."
            ),
        ),
        PipelineNode(
            "seed_edge_curves",
            "Seed-boundary confirmation",
            "Diagnostic overlay",
            "Trace, radius, circle, ellipse, centre-vote and semantic confirmation",
            2220,
            500,
            parameters={
                "curve_diameter_multiplier": 1.0,
                "boundary_radius_min_fraction": 0.25,
                "boundary_radius_max_fraction": 0.80,
                "boundary_radius_sample_count": 9,
                "boundary_arc_span_degrees": 70.0,
                "boundary_arc_sample_count": 15,
                "boundary_orientation_tolerance_degrees": 20.0,
                "boundary_missing_support_floor": 0.15,
                "boundary_circle_residual_tolerance": 0.16,
                "boundary_ellipse_residual_tolerance": 0.18,
                "boundary_max_axis_ratio": 2.20,
                "boundary_radius_log_tolerance": 0.45,
                "boundary_center_vote_weight": 0.35,
                "boundary_center_vote_blur_fraction": 0.08,
                "boundary_semantic_weight": 0.25,
                "boundary_polarity_boost": 0.20,
                "boundary_minimum_confidence": 0.12,
                "boundary_geometry_max_candidates": 256,
                "boundary_working_maximum_dimension": 1280,
            },
            parameter_specs=boundary_trace_parameters,
            inline_parameters=(
                ("curve_diameter_multiplier", "Diameter Ã—"),
                ("boundary_radius_min_fraction", "Radius min"),
                ("boundary_radius_max_fraction", "Radius max"),
                ("boundary_minimum_confidence", "Accept â‰¥"),
            ),
            details=(
                "Every trace is checked against a dense radius bank with many samples "
                "across each candidate arc; missing edge samples are tolerated. Normal "
                "centre votes reinforce fragmented arcs. Provisional instances provide "
                "batched circle and ellipse fits, with equivalent radius and axis ratio "
                "treated as soft evidence. Background/foreground side consistency and "
                "inward lightening can boost confidence but never define the boundary."
            ),
        ),
        PipelineNode(
            "seed_interior", "Seed-interior probability", "GPU diagnostic",
            "Fuse foreground and background evidence into a soft seed-interior map",
            1680, 180,
            details="CUDA PyTorch smooths the colour-derived foreground response, blends inverse background evidence, and retains a continuous probability instead of an early hard threshold.",
            parameters={"compute_device": "cuda", "allow_cpu_fallback": False, "maximum_dimension": 1024, "interior_smoothing_fraction": 0.055, "interior_background_weight": 0.55},
            parameter_specs=seed_interior_parameters,
        ),
        PipelineNode(
            "boundary_normals", "Boundary confidence & normals", "GPU diagnostic",
            "Estimate boundary strength and continuous outward normal direction",
            1920, 180,
            details="A seed-scaled probability morphology band is combined with Sobel lightness gradients. Hue shows a directed 0–360° normal; brightness shows confidence.",
            parameters={"boundary_width_fraction": 0.025},
            parameter_specs=boundary_normal_parameters,
        ),
        PipelineNode(
            "touching_split", "Touching-seed split likelihood", "GPU diagnostic",
            "Locate shared necks and distance-transform saddles inside seed masses",
            2160, 70,
            details="High likelihood requires boundary support near a shallow foreground neck or concave distance-transform saddle; it guides correction of provisional GPU assignments.",
            parameters={"split_neck_fraction": 0.36},
            parameter_specs=touching_split_parameters,
        ),
        PipelineNode(
            "ellipse_likelihood", "Multiscale ellipse likelihood", "GPU diagnostic",
            "Score seed-radius boundary support and local elliptical orientation",
            2160, 270,
            details="Structure-tensor anisotropy is combined with boundary evidence near each proposal radius. Hue shows axial ellipse orientation and brightness shows support.",
            parameters={"ellipse_radial_tolerance": 0.22},
            parameter_specs=ellipse_parameters,
        ),
        PipelineNode(
            "proposal_disagreement", "Proposal disagreement", "GPU diagnostic",
            "Show where circle, distance, interior, and ellipse evidence disagree",
            2400, 70,
            details="The normalized standard deviation among four independently derived evidence maps highlights fragile proposals and method-specific failures.",
            parameters={"disagreement_scale_fraction": 0.16},
            parameter_specs=disagreement_parameters,
        ),
        PipelineNode(
            "assignment_confidence", "Instance-assignment confidence", "GPU diagnostic",
            "Estimate confidence that each pixel belongs to its proposed seed",
            2400, 270,
            details="Interior confidence is discounted at strong boundaries and where neighbouring provisional labels meet. A separate contested-pixel intermediate is viewable.",
            parameters={"assignment_boundary_penalty": 0.70},
            parameter_specs=assignment_parameters,
        ),
        PipelineNode(
            "contact_graph", "Occlusion/contact graph", "GPU diagnostic",
            "Connect seed proposals whose extents touch or overlap",
            2640, 170,
            details="GPU pairwise distances create a proposal contact graph; the viewer overlays candidate contact lines on shared-boundary evidence.",
            parameters={"contact_distance_multiplier": 1.28},
            parameter_specs=contact_parameters,
        ),
        PipelineNode(
            "illumination_decomposition", "Illumination decomposition", "GPU diagnostic",
            "Separate broad illumination, reflectance, shadow, and glare",
            1200, 760,
            details="A seed-scaled low-frequency field estimates illumination. Dividing luminance by this field gives reflectance; additional intermediates expose shadow and specular risk.",
            parameters={"illumination_scale_fraction": 0.55},
            parameter_specs=illumination_parameters,
        ),
        PipelineNode(
            "image_quality", "Image-quality diagnostics", "GPU diagnostic",
            "Map focus, clipping, underexposure, glare, and local noise risk",
            1740, 760,
            details="The composite risk map is backed by separately viewable focus, clipped-highlight, underexposure, and sensor-noise maps.",
            parameters={"quality_noise_scale_fraction": 0.025},
            parameter_specs=image_quality_parameters,
        ),
        PipelineNode(
            "radial_profile", "Per-seed radial profiles", "GPU diagnostic",
            "Compare each assigned pixel with expected centre-to-edge lightness",
            2400, 540,
            details="Pixels are normalized by their assigned proposal radius and accumulated into a configurable global radial profile. The residual highlights unusual edge darkening or internal structure.",
            parameters={"radial_bin_count": 24},
            parameter_specs=radial_parameters,
        ),
        PipelineNode(
            "wrinkling", "Wrinkling likelihood", "GPU trait analysis",
            "Detect seed-relative ridge and valley texture away from boundaries",
            2640, 480,
            details="Band-limited lightness curvature is normalized within the dish and gated by soft seed-interior confidence, excluding the main seed boundary.",
            parameters={"wrinkle_scale_fraction": 0.035},
            parameter_specs=wrinkling_parameters,
        ),
        PipelineNode(
            "coat_damage", "Seed-coat damage likelihood", "GPU trait analysis",
            "Combine local colour anomaly, internal edges, and radial residuals",
            2880, 450,
            details="Damage confidence rises where an anomalous local colour coincides with internal edge or radial-profile evidence; normal outer boundaries are suppressed.",
            parameters={"damage_anomaly_scale_fraction": 0.12},
            parameter_specs=damage_parameters,
        ),
        PipelineNode(
            "pattern_decomposition", "Coat-pattern decomposition", "GPU trait analysis",
            "Provide broad plain, spotted, mottled, patch, stripe, and bicolour probabilities",
            2400, 720,
            details="Fine, medium, and coarse seed-relative frequency responses plus directional structure produce transparent heuristic class probabilities. Each class probability is available as an intermediate overlay.",
            parameters={"pattern_scale_fraction": 0.18},
            parameter_specs=pattern_parameters,
        ),
        PipelineNode(
            "colour_probabilities", "Broad colour probabilities", "GPU trait analysis",
            "Estimate white, yellow, green, red, brown, and black pixel probabilities",
            2400, 900,
            details="Colour-balanced RGB is compared with broad prototype colours using a softmax. The winning class, uncertainty, and every class probability are viewable independently.",
            parameters={"colour_temperature": 18.0},
            parameter_specs=colour_probability_parameters,
        ),
        PipelineNode(
            "calibration_residuals", "Calibration residual risk", "GPU diagnostic",
            "Map confidence loss and extrapolation away from calibration references",
            1200, 940,
            details="Detected card/ruler confidence seeds the risk surface; risk increases with distance from the closest transformed calibration anchor.",
            parameters={"calibration_residual_gain": 1.0},
            parameter_specs=calibration_residual_parameters,
        ),
        PipelineNode(
            "review", "Human review", "Review", "Add, delete, split and merge masks", 3120, -200,
            implemented=False, status=NodeStatus.PLANNED, status_detail="Planned",
        ),
        PipelineNode(
            "measurements", "Measurements", "Analysis", "Calibrated shape measurements", 3360, -310,
            bypassable=True, status=NodeStatus.BLOCKED, status_detail="Requires reviewed masks",
        ),
        PipelineNode(
            "classification", "Coat & condition", "Analysis", "Colour, pattern, wrinkling and damage", 3360, -90,
            implemented=False, bypassable=True, status=NodeStatus.PLANNED, status_detail="Planned",
        ),
        PipelineNode(
            "aggregation", "Lot aggregation", "Summary", "Counts, proportions and coverage", 3600, -200,
            implemented=False, status=NodeStatus.PLANNED, status_detail="Planned",
        ),
        PipelineNode(
            "output", "Final output", "Output", "CSV, JSON and annotated images", 3840, -200,
            implemented=False, status=NodeStatus.PLANNED, status_detail="Planned",
        ),
    )
    # Keep only the controls most useful during visual tuning on the graph itself.
    # The inspector remains the complete, tooltip-documented editor.
    inline_controls: dict[str, tuple[tuple[str, str], ...]] = {
        "colour_reference": (("apply_colour_balance", "Neutral balance"),),
        "ruler_detection": (("ruler_length_mm", "Span mm"),),
        "deskew_colour": (
            ("apply_perspective_correction", "Perspective"),
            ("max_deskew_degrees", "Max rotation"),
        ),
        "layout_detection": (
            ("downsample_max_dimension", "GPU dimension"),
            ("hough_accumulator_threshold", "Rim support"),
            ("expected_radius_fraction", "Expected radius"),
            ("rim_pair_expected_separation_fraction", "Expected rim gap"),
        ),
        "scale_calibration": (("minor_tick_mm", "Minor tick mm"),),
        "seed_scale_estimation": (
            ("reference_scale_factor", "Reference scale"),
            ("fallback_diameter_fraction", "Fallback diameter"),
        ),
        "foreground_segmentation": (
            ("foreground_otsu_fraction", "Otsu multiplier"),
            ("foreground_reference_weight", "Paint influence"),
            ("foreground_shadow_rejection_strength", "Reject shadows"),
        ),
        "distance_candidates": (
            ("distance_neighborhood_fraction", "Peak spacing"),
            ("distance_min_depth_fraction", "Peak depth"),
        ),
        "circle_candidates": (
            ("circle_accumulator_threshold", "Ring support"),
            ("circle_min_radius_fraction", "Radius min"),
            ("circle_max_radius_fraction", "Radius max"),
            ("circle_working_maximum_dimension", "GPU dimension"),
        ),
        "identification": (
            ("merge_distance_fraction", "Merge distance"),
            ("circle_confidence", "Circle weight"),
            ("distance_confidence", "Distance weight"),
        ),
        "background_likelihood": (
            ("background_colour_components", "Colour modes"),
            ("background_distribution_scale_multiplier", "Colour tolerance"),
            ("background_refinement_iterations", "Refine rounds"),
        ),
        "instance_masks": (
            ("instance_min_extent_fraction", "Extent min"),
            ("instance_max_extent_fraction", "Extent max"),
            ("instance_radius_extent_multiplier", "Radius extent"),
        ),
        "refined_background_likelihood": (
            ("noise_direction_step_degrees", "Direction step"),
            ("noise_vector_length_fraction", "Ray length"),
            ("noise_direction_integration", "Integration"),
            ("noise_working_maximum_dimension", "GPU dimension"),
        ),
        "edge_gradients": (
            ("edge_blur_sigma", "Blur sigma"),
            ("edge_chroma_weight", "Chroma weight"),
            ("edge_normalization_percentile", "Normalize %"),
            ("edge_strength_gamma", "Gamma"),
        ),
        "edge_ridges": (
            ("ridge_nms_step_px", "NMS step"),
            ("ridge_low_threshold", "Low"),
            ("ridge_high_threshold", "High"),
            ("ridge_hysteresis_iterations", "Reach"),
        ),
        "edge_traces": (
            ("trace_tangent_tolerance_degrees", "Angle +/-"),
            ("trace_maximum_gap_px", "Gap px"),
            ("trace_window_fraction", "Window xD"),
            ("trace_minimum_length_fraction", "Min length xD"),
        ),
        "seed_edge_curves": (
            ("curve_diameter_multiplier", "Diameter x"),
            ("boundary_radius_min_fraction", "Radius min"),
            ("boundary_radius_max_fraction", "Radius max"),
            ("boundary_minimum_confidence", "Accept min"),
        ),
        "seed_interior": (
            ("compute_device", "Device"),
            ("maximum_dimension", "GPU dimension"),
            ("interior_background_weight", "Background weight"),
        ),
        "boundary_normals": (("boundary_width_fraction", "Boundary width"),),
        "touching_split": (("split_neck_fraction", "Neck depth"),),
        "ellipse_likelihood": (("ellipse_radial_tolerance", "Radial tolerance"),),
        "proposal_disagreement": (("disagreement_scale_fraction", "Evidence spread"),),
        "assignment_confidence": (("assignment_boundary_penalty", "Boundary penalty"),),
        "contact_graph": (("contact_distance_multiplier", "Contact distance"),),
        "illumination_decomposition": (("illumination_scale_fraction", "Field scale"),),
        "image_quality": (("quality_noise_scale_fraction", "Noise scale"),),
        "radial_profile": (("radial_bin_count", "Radial bins"),),
        "wrinkling": (("wrinkle_scale_fraction", "Wrinkle scale"),),
        "coat_damage": (("damage_anomaly_scale_fraction", "Damage context"),),
        "pattern_decomposition": (("pattern_scale_fraction", "Pattern scale"),),
        "colour_probabilities": (("colour_temperature", "Sharpness"),),
        "calibration_residuals": (("calibration_residual_gain", "Risk gain"),),
    }
    for node in nodes:
        if node.identifier in inline_controls:
            node.inline_parameters = inline_controls[node.identifier]
    connections = (
        PipelineConnection("raw_images", "metadata", "ImageBatch"),
        PipelineConnection("metadata", "colour_reference", "TaggedImages"),
        PipelineConnection("colour_reference", "deskew_colour", "SwatchGrid"),
        PipelineConnection("deskew_colour", "ruler_detection", "CorrectedImage"),
        PipelineConnection("deskew_colour", "layout_detection", "CorrectedImage"),
        PipelineConnection("deskew_colour", "scale_calibration", "CorrectedImage"),
        PipelineConnection("ruler_detection", "scale_calibration", "RulerAxis"),
        PipelineConnection("deskew_colour", "seed_scale_estimation", "CorrectedImage"),
        PipelineConnection("layout_detection", "seed_scale_estimation", "VesselGeometry"),
        PipelineConnection("deskew_colour", "foreground_segmentation", "CorrectedImage"),
        PipelineConnection("layout_detection", "foreground_segmentation", "VesselGeometry"),
        PipelineConnection("seed_scale_estimation", "foreground_segmentation", "SeedDiameter"),
        PipelineConnection("foreground_segmentation", "distance_candidates", "ForegroundMask"),
        PipelineConnection("seed_scale_estimation", "distance_candidates", "SeedDiameter"),
        PipelineConnection("foreground_segmentation", "circle_candidates", "DishSearchRegion"),
        PipelineConnection("seed_scale_estimation", "circle_candidates", "SeedDiameter"),
        PipelineConnection("distance_candidates", "identification", "DistancePeaks"),
        PipelineConnection("circle_candidates", "identification", "CudaRings"),
        PipelineConnection(
            "deskew_colour", "background_likelihood", "CorrectedImage"
        ),
        PipelineConnection(
            "seed_scale_estimation", "background_likelihood", "SeedDiameter"
        ),
        PipelineConnection("identification", "instance_masks", "SeedProposals"),
        PipelineConnection(
            "background_likelihood", "instance_masks", "BackgroundLikelihood"
        ),
        PipelineConnection(
            "refined_background_likelihood", "instance_masks", "DirectedBackground"
        ),
        PipelineConnection(
            "background_likelihood",
            "refined_background_likelihood",
            "ColourPseudoLabels",
        ),
        PipelineConnection(
            "seed_scale_estimation",
            "refined_background_likelihood",
            "SeedDiameter",
        ),
        PipelineConnection(
            "deskew_colour", "edge_gradients", "CorrectedImage", target_port="image"
        ),
        PipelineConnection(
            "edge_gradients",
            "undirected_edges",
            "AxialTangents",
            source_port="undirected",
        ),
        PipelineConnection(
            "edge_gradients",
            "directed_edges",
            "DirectedTangents",
            source_port="directed",
        ),
        PipelineConnection("edge_gradients", "edge_ridges", "FloatGradientField"),
        PipelineConnection("edge_ridges", "edge_traces", "ThinnedRidges"),
        PipelineConnection("edge_gradients", "edge_traces", "ContinuousTangents"),
        PipelineConnection("edge_traces", "seed_edge_curves", "OrientedTraces"),
        PipelineConnection("edge_gradients", "seed_edge_curves", "FloatGradientField"),
        PipelineConnection("seed_scale_estimation", "seed_edge_curves", "SeedScale"),
        PipelineConnection("background_likelihood", "seed_edge_curves", "BackgroundProbability"),
        PipelineConnection("foreground_segmentation", "seed_edge_curves", "ForegroundProbability"),
        PipelineConnection("instance_masks", "seed_edge_curves", "ProvisionalInstances"),
        PipelineConnection("seed_scale_estimation", "instance_masks", "SeedScale"),
        PipelineConnection("foreground_segmentation", "seed_interior", "ForegroundEvidence"),
        PipelineConnection("background_likelihood", "seed_interior", "BackgroundProbability"),
        PipelineConnection("refined_background_likelihood", "seed_interior", "DirectionalBackground"),
        PipelineConnection("seed_interior", "boundary_normals", "InteriorProbability"),
        PipelineConnection("directed_edges", "boundary_normals", "ColourGradient"),
        PipelineConnection("boundary_normals", "touching_split", "BoundaryConfidence"),
        PipelineConnection("distance_candidates", "touching_split", "DistanceTransform"),
        PipelineConnection("boundary_normals", "ellipse_likelihood", "BoundaryNormals"),
        PipelineConnection("seed_scale_estimation", "ellipse_likelihood", "SeedDiameter"),
        PipelineConnection("circle_candidates", "proposal_disagreement", "CircleEvidence"),
        PipelineConnection("distance_candidates", "proposal_disagreement", "DistanceEvidence"),
        PipelineConnection("ellipse_likelihood", "proposal_disagreement", "EllipseEvidence"),
        PipelineConnection("instance_masks", "assignment_confidence", "ProvisionalInstances"),
        PipelineConnection("boundary_normals", "assignment_confidence", "BoundaryConfidence"),
        PipelineConnection("assignment_confidence", "contact_graph", "AssignmentConfidence"),
        PipelineConnection("identification", "contact_graph", "SeedProposals"),
        PipelineConnection("deskew_colour", "illumination_decomposition", "CorrectedImage"),
        PipelineConnection("illumination_decomposition", "image_quality", "IlluminationProducts"),
        PipelineConnection("directed_edges", "image_quality", "ImageGradients"),
        PipelineConnection("instance_masks", "radial_profile", "ProvisionalInstances"),
        PipelineConnection("seed_scale_estimation", "radial_profile", "SeedDiameter"),
        PipelineConnection("seed_interior", "radial_profile", "InteriorProbability"),
        PipelineConnection("radial_profile", "wrinkling", "RadialResidual"),
        PipelineConnection("seed_interior", "wrinkling", "InteriorProbability"),
        PipelineConnection("image_quality", "wrinkling", "ImageQuality"),
        PipelineConnection("radial_profile", "coat_damage", "RadialResidual"),
        PipelineConnection("boundary_normals", "coat_damage", "BoundaryConfidence"),
        PipelineConnection("image_quality", "coat_damage", "ImageQuality"),
        PipelineConnection("seed_interior", "pattern_decomposition", "InteriorProbability"),
        PipelineConnection("illumination_decomposition", "pattern_decomposition", "Reflectance"),
        PipelineConnection("seed_interior", "colour_probabilities", "InteriorProbability"),
        PipelineConnection("illumination_decomposition", "colour_probabilities", "Reflectance"),
        PipelineConnection("colour_reference", "calibration_residuals", "ReferenceConfidence"),
        PipelineConnection("ruler_detection", "calibration_residuals", "ReferenceConfidence"),
        PipelineConnection("deskew_colour", "calibration_residuals", "CalibrationTransform"),
        PipelineConnection("touching_split", "review", "SplitSuggestions"),
        PipelineConnection("proposal_disagreement", "review", "UncertainRegions"),
        PipelineConnection("assignment_confidence", "review", "AssignmentConfidence"),
        PipelineConnection("contact_graph", "review", "ContactGraph"),
        PipelineConnection("instance_masks", "review", "InstanceProposals"),
        PipelineConnection("review", "measurements", "ReviewedMasks"),
        PipelineConnection("scale_calibration", "measurements", "PixelsPerMillimetre"),
        PipelineConnection("review", "classification", "ReviewedMasks"),
        PipelineConnection("wrinkling", "classification", "WrinkleEvidence"),
        PipelineConnection("coat_damage", "classification", "DamageEvidence"),
        PipelineConnection("pattern_decomposition", "classification", "PatternProbabilities"),
        PipelineConnection("colour_probabilities", "classification", "ColourProbabilities"),
        PipelineConnection("calibration_residuals", "classification", "CalibrationRisk"),
        PipelineConnection("measurements", "aggregation", "MeasurementTable"),
        PipelineConnection("classification", "aggregation", "TraitTable"),
        PipelineConnection("aggregation", "output", "LotSummary"),
    )
    return PipelineGraph(nodes, connections)
