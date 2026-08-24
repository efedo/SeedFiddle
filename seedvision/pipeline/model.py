"""Toolkit-independent model for the visual seed-analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import Any, Iterable, Mapping


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
    display_only: bool = False


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
    input_port_types: dict[str, str] = field(default_factory=dict)
    output_port_types: dict[str, str] = field(default_factory=dict)
    inline_parameters: tuple[tuple[str, str], ...] = ()
    calculation_seconds: float | None = None
    default_parameters: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Keep an immutable-by-convention snapshot for the inspector's Reset
        # action.  Each node owns its dictionaries, so resetting one card can
        # never mutate a shared default object.
        self.parameters = dict(self.parameters)
        self.default_parameters = dict(self.parameters)
        self.input_port_types = dict(self.input_port_types)
        self.output_port_types = dict(self.output_port_types)

    def validated_parameter_value(self, key: str, value: Any) -> Any:
        """Normalize and validate one value without mutating this node."""

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
        elif spec.kind == "text":
            value = str(value).strip()
            if not value:
                raise ValueError(f"{spec.label} cannot be empty.")
        elif spec.kind != "bool":
            raise ValueError(f"Unsupported parameter kind {spec.kind!r}.")
        if spec.minimum is not None and value < spec.minimum:
            raise ValueError(f"{spec.label} must be at least {spec.minimum}.")
        if spec.maximum is not None and value > spec.maximum:
            raise ValueError(f"{spec.label} must be at most {spec.maximum}.")
        return value

    def set_parameter(self, key: str, value: Any) -> None:
        self.parameters[key] = self.validated_parameter_value(key, value)


@dataclass(frozen=True, slots=True)
class PipelineConnection:
    source: str
    target: str
    data_type: str
    source_port: str = ""
    target_port: str = ""


_PORT_LABELS = {
    "ImageBatch": "Images",
    "TaggedImages": "Tagged images",
    "CorrectedImage": "Corrected image",
    "SwatchGrid": "Colour swatches",
    "RulerAxis": "Ruler axis",
    "PixelsPerMillimetre": "Absolute scale",
    "VesselGeometry": "Dish geometry",
    "DishRegion": "Dish region",
    "DishSearchRegion": "Dish search",
    "SeedDiameter": "Seed diameter",
    "SeedScale": "Seed scale",
    "BackgroundReferences": "Background refs",
    "ForegroundReferences": "Foreground refs",
    "OtherReferences": "Other refs",
    "PaintedInstanceAnnotations": "Annotated seed IDs",
    "ManualSeedCentres": "Manual seed centres",
    "PhysicalEdgeProbability": "Physical-edge probability",
    "NonEdgeProbability": "Non-physical edge probability",
    "PhysicalPrototypeProbability": "Physical-edge prototype probability",
    "NonEdgePrototypeProbability": "Non-physical prototype probability",
    "PerimeterColourSamples": "Perimeter colours",
    "AutomaticBackgroundReferences": "Automatic BG refs",
    "AnnotatedForegroundReferences": "Annotated seed refs",
    "ForegroundColour": "FG colour",
    "ForegroundEvidence": "FG evidence",
    "ForegroundMask": "FG mask",
    "ForegroundProbability": "FG probability",
    "ForegroundNoise": "FG noise",
    "ForegroundNoiseProbability": "FG noise probability",
    "BackgroundColour": "BG colour",
    "BackgroundLikelihood": "BG likelihood",
    "BackgroundProbability": "BG probability",
    "BackgroundNoise": "BG noise",
    "BackgroundNoiseProbability": "BG noise probability",
    "DirectionalBackground": "Directional BG",
    "DirectedBackground": "Directed BG",
    "ColourPseudoLabels": "Colour pseudo-labels",
    "ReferenceSeedColours": "Reference seed colours",
    "EdgeMagnitude": "Edge magnitude",
    "FloatGradientField": "Gradient field",
    "AxialTangents": "Undirected tangents",
    "DirectedTangents": "Directed tangents",
    "ContinuousTangents": "Edge tangents",
    "ThinnedRidges": "Thinned ridges",
    "OrientedTraces": "Oriented traces",
    "OrientedTraceLabels": "Trace identities",
    "TraceContinuity": "Trace continuity",
    "TraceGapConfidence": "Trace gap confidence",
    "ThinnedReferenceRidges": "Reference ridges",
    "LocalShadow": "Local shadow",
    "LocalLighting": "Local lighting",
    "SensorNoise": "Sensor noise",
    "SensorNoiseLikelihood": "Sensor/noise likelihood",
    "FlattenedGrayscale": "Flattened grayscale",
    "ShadowLikelihood": "Shadow likelihood",
    "HighlightLikelihood": "Highlight likelihood",
    "GlareLikelihood": "Glare likelihood",
    "ImageQualityRisk": "Quality risk",
    "FocusQuality": "Focus quality",
    "ClippedHighlights": "Clipped highlights",
    "Underexposure": "Underexposure",
    "SpeciesCondition": "Species",
    "InteriorProbability": "Interior probability",
    "BoundaryConfidence": "Boundary confidence",
    "BoundaryNormals": "Boundary normals",
    "ContestedPixels": "Contested pixels",
    "ProvisionalInstances": "Provisional instances",
    "InstanceProposals": "Instance proposals",
    "SeedProposals": "Seed proposals",
    "ReviewedMasks": "Reviewed masks",
    "ReferenceConfidence": "Reference confidence",
    "MaterialLikelihood": "Material likelihood",
    "MaterialMask": "Material mask",
    "BoundaryCost": "Boundary cost",
    "CentreLikelihood": "Centre likelihood",
    "SeedInstances": "Seed instances",
    "InstanceConfidence": "Instance confidence",
    "InstanceConcavity": "Instance concavity",
    "InstanceCandidates": "Instance candidates",
    "CentreProbability": "Centre probability",
    "DistancePrediction": "Distance prediction",
    "ModelUncertainty": "Model uncertainty",
    "ObjectProbability": "Object probability",
    "RadialResidual": "Radial residual",
    "RadialCoordinate": "Radial coordinate",
    "PatternConfidence": "Pattern confidence",
    "ColourUncertainty": "Colour uncertainty",
}

_TERMINAL_OUTPUTS = {
    "procedural_instances": ("instances", "Seed instances", "SeedInstances"),
    "seed_edge_curves": (
        "boundaries",
        "Confirmed boundaries",
        "ConfirmedBoundaries",
    ),
    "output": ("reports", "Reports", "ExportedResults"),
}


def _port_identifier(value: str) -> str:
    """Return a stable, readable connector identifier."""

    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value).lower()
    words = re.sub(r"[^a-z0-9]+", "_", words).strip("_")
    return words or "data"


def _port_label(data_type: str) -> str:
    """Keep connector text short without hiding the datum it carries."""

    if data_type in _PORT_LABELS:
        return _PORT_LABELS[data_type]
    label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", data_type).strip()
    return label[:1].upper() + label[1:]


def _unique_port_id(candidate: str, used: dict[str, str], data_type: str) -> str:
    """Reuse a same-type output port, otherwise suffix a name collision."""

    if candidate not in used or used[candidate] == data_type:
        return candidate
    index = 2
    while f"{candidate}_{index}" in used:
        index += 1
    return f"{candidate}_{index}"


def _populate_connection_ports(
    nodes: dict[str, PipelineNode],
    connections: tuple[PipelineConnection, ...],
) -> tuple[PipelineConnection, ...]:
    """Give every authored dependency a concrete labelled input/output port."""

    explicit_target_counts: dict[tuple[str, str], int] = {}
    explicit_source_by_type: dict[tuple[str, str], str] = {}
    for connection in connections:
        if connection.target_port:
            key = (connection.target, connection.target_port)
            explicit_target_counts[key] = explicit_target_counts.get(key, 0) + 1
        if connection.source_port:
            explicit_source_by_type.setdefault(
                (connection.source, connection.data_type), connection.source_port
            )

    input_ports: dict[str, list[tuple[str, str]]] = {
        node_id: [] for node_id in nodes
    }
    output_ports: dict[str, list[tuple[str, str]]] = {
        node_id: list(node.output_ports) for node_id, node in nodes.items()
    }
    input_types: dict[str, dict[str, str]] = {
        node_id: dict(node.input_port_types) for node_id, node in nodes.items()
    }
    output_types: dict[str, dict[str, str]] = {
        node_id: dict(node.output_port_types) for node_id, node in nodes.items()
    }
    normalized: list[PipelineConnection] = []

    for connection in connections:
        source_used = output_types[connection.source]
        source_candidate = connection.source_port or explicit_source_by_type.get(
            (connection.source, connection.data_type),
            _port_identifier(connection.data_type),
        )
        source_port = _unique_port_id(
            source_candidate, source_used, connection.data_type
        )
        source_used[source_port] = connection.data_type
        if source_port not in {identifier for identifier, _ in output_ports[connection.source]}:
            output_ports[connection.source].append(
                (source_port, _port_label(connection.data_type))
            )

        # Inputs are single-cardinality dependency sockets. Some older cards
        # used one broad "image" name for eleven different tensors; expand
        # those aliases so the displayed graph states the real dependency.
        explicit_is_unique = bool(connection.target_port) and (
            explicit_target_counts[(connection.target, connection.target_port)] == 1
        )
        target_candidate = (
            connection.target_port
            if explicit_is_unique
            else _port_identifier(connection.data_type)
        )
        target_used = input_types[connection.target]
        if target_candidate in target_used:
            source_suffix = _port_identifier(connection.source)
            target_candidate = f"{target_candidate}_{source_suffix}"
        target_port = _unique_port_id(
            target_candidate, target_used, connection.data_type
        )
        # Inputs may not share even when their data type is the same.
        if target_port in target_used:
            index = 2
            base = target_port
            while f"{base}_{index}" in target_used:
                index += 1
            target_port = f"{base}_{index}"
        target_used[target_port] = connection.data_type
        if target_port not in {identifier for identifier, _ in input_ports[connection.target]}:
            input_ports[connection.target].append(
                (target_port, _port_label(connection.data_type))
            )
        normalized.append(
            PipelineConnection(
                connection.source,
                connection.target,
                connection.data_type,
                source_port,
                target_port,
            )
        )

    for node_id, node in nodes.items():
        # Preserve authored generated products even if nothing currently
        # consumes them. True terminal cards still expose their result.
        if not output_ports[node_id]:
            port_id, label, data_type = _TERMINAL_OUTPUTS.get(
                node_id,
                ("result", f"{node.title} result", "Result"),
            )
            output_ports[node_id].append((port_id, label))
            output_types[node_id][port_id] = data_type
        for port_id, label in input_ports[node_id]:
            input_types[node_id].setdefault(port_id, label)
        for port_id, label in output_ports[node_id]:
            output_types[node_id].setdefault(port_id, label)
        node.input_ports = tuple(input_ports[node_id])
        node.output_ports = tuple(output_ports[node_id])
        node.input_port_types = input_types[node_id]
        node.output_port_types = output_types[node_id]
    return tuple(normalized)


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
        self.unused_nodes: dict[str, PipelineNode] = {}
        self.connections = _populate_connection_ports(
            self.nodes, tuple(connections)
        )
        # This is the honest wiring catalogue. The calculation engine has
        # authored consumers, so the editor accepts only these typed endpoint
        # pairs rather than drawing substitutions the engine would ignore.
        self.connection_templates = self.connections
        self.unused_connections: tuple[PipelineConnection, ...] = ()
        self._connection_suspended: set[str] = set()
        self.revision = 0
        self._validate_connections()
        self.topological_order()

    def node(self, identifier: str) -> PipelineNode:
        if identifier in self.nodes:
            return self.nodes[identifier]
        if identifier in self.unused_nodes:
            return self.unused_nodes[identifier]
        raise KeyError(f"Unknown pipeline node {identifier!r}.")

    def is_active(self, identifier: str) -> bool:
        return identifier in self.nodes

    def shelve_node(self, identifier: str, *, record_revision: bool = True) -> None:
        """Move an active node and all incident edges to the unused toolbox."""

        if identifier not in self.nodes:
            raise ValueError(f"Pipeline node {identifier!r} is not active.")
        node = self.nodes.pop(identifier)
        retained = tuple(
            connection
            for connection in self.connections
            if identifier not in (connection.source, connection.target)
        )
        removed = tuple(
            connection
            for connection in self.connections
            if identifier in (connection.source, connection.target)
        )
        self.connections = retained
        self.unused_nodes[identifier] = node
        self.unused_connections = (*self.unused_connections, *removed)
        self._validate_connections()
        self.topological_order()
        if record_revision:
            self.revision += 1

    def restore_unused_node(self, identifier: str) -> tuple[PipelineConnection, ...]:
        """Restore a toolbox node and every now-resolvable preserved edge."""

        if identifier not in self.unused_nodes:
            raise ValueError(f"Unused pipeline node {identifier!r} was not found.")
        node = self.unused_nodes.pop(identifier)
        self.nodes[identifier] = node
        restored = tuple(
            connection
            for connection in self.unused_connections
            if connection.source in self.nodes and connection.target in self.nodes
        )
        self.unused_connections = tuple(
            connection
            for connection in self.unused_connections
            if connection not in restored
        )
        self.connections = (*self.connections, *restored)
        try:
            self._validate_connections()
            self.topological_order()
        except Exception:
            self.connections = tuple(
                connection
                for connection in self.connections
                if connection not in restored
            )
            self.unused_connections = (*self.unused_connections, *restored)
            self.unused_nodes[identifier] = self.nodes.pop(identifier)
            raise
        self.revision += 1
        return restored

    def _require_active(self, identifier: str) -> PipelineNode:
        if identifier in self.unused_nodes:
            raise ValueError(
                f"Pipeline node {identifier!r} is in the unused-node toolbox. "
                "Restore it before changing the active graph."
            )
        return self.node(identifier)

    def upstream(self, identifier: str) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                connection.source
                for connection in self.connections
                if connection.target == identifier
            )
        )

    def downstream(self, identifier: str, *, recursive: bool = False) -> tuple[str, ...]:
        direct = list(
            dict.fromkeys(
                connection.target
                for connection in self.connections
                if connection.source == identifier
            )
        )
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

    def downstream_from_port(
        self, identifier: str, port_id: str, *, recursive: bool = False
    ) -> tuple[str, ...]:
        """Return consumers of one authored output rather than the whole node."""

        direct = tuple(
            dict.fromkeys(
                connection.target
                for connection in self.connections
                if connection.source == identifier
                and connection.source_port == port_id
            )
        )
        if not recursive:
            return direct
        visited = set(direct)
        for target in direct:
            visited.update(self.downstream(target, recursive=True))
        order = self.topological_order()
        return tuple(node_id for node_id in order if node_id in visited)

    def connection_for_input(
        self, node_id: str, port_id: str
    ) -> PipelineConnection | None:
        """Return the active producer attached to a single input socket."""

        return next(
            (
                connection
                for connection in self.connections
                if connection.target == node_id
                and connection.target_port == port_id
            ),
            None,
        )

    def missing_input_ports(self, node_id: str) -> tuple[str, ...]:
        """List disconnected authored inputs whose endpoints are active."""

        self._require_active(node_id)
        required = {
            template.target_port
            for template in self.connection_templates
            if template.target == node_id and template.source in self.nodes
        }
        connected = {
            connection.target_port
            for connection in self.connections
            if connection.target == node_id
        }
        return tuple(
            port_id
            for port_id, _ in self.node(node_id).input_ports
            if port_id in required and port_id not in connected
        )

    def connection_template(
        self,
        source: str,
        source_port: str,
        target: str,
        target_port: str,
    ) -> PipelineConnection | None:
        """Resolve one calculation-supported connector pair."""

        return next(
            (
                template
                for template in self.connection_templates
                if template.source == source
                and template.source_port == source_port
                and template.target == target
                and template.target_port == target_port
            ),
            None,
        )

    def disconnect(self, connection: PipelineConnection) -> tuple[str, ...]:
        """Remove one active dependency and bypass its enabled consumer branch."""

        if connection not in self.connections:
            raise ValueError("That pipeline connection is not active.")
        affected = (
            connection.target,
            *self.downstream(connection.target, recursive=True),
        )
        self.connections = tuple(
            candidate
            for candidate in self.connections
            if candidate != connection
        )
        for node_id in affected:
            node = self.node(node_id)
            if node.enabled:
                self._connection_suspended.add(node_id)
            node.enabled = False
            node.status = NodeStatus.BYPASSED
            node.status_detail = "Disabled by disconnected input"
        self.revision += 1
        self.invalidate(affected, preserve_bypassed=True)
        for node_id in affected:
            node = self.node(node_id)
            node.status = NodeStatus.BYPASSED
            node.status_detail = "Disabled by disconnected input"
        self._validate_connections()
        self.topological_order()
        return affected

    def connect(
        self,
        source: str,
        source_port: str,
        target: str,
        target_port: str,
    ) -> tuple[str, ...]:
        """Restore an authored typed dependency and eligible suspended nodes."""

        self._require_active(source)
        self._require_active(target)
        template = self.connection_template(
            source, source_port, target, target_port
        )
        if template is None:
            source_type = self.node(source).output_port_types.get(source_port)
            target_type = self.node(target).input_port_types.get(target_port)
            if source_type != target_type:
                raise ValueError(
                    "These connectors carry different data types and cannot be joined."
                )
            raise ValueError(
                "This typed substitution is not an authored calculation input."
            )
        if template in self.connections:
            return ()
        occupied = self.connection_for_input(target, target_port)
        if occupied is not None:
            raise ValueError("Disconnect the existing input before reconnecting it.")
        self.connections = (*self.connections, template)
        try:
            self._validate_connections()
            self.topological_order()
        except Exception:
            self.connections = tuple(
                candidate
                for candidate in self.connections
                if candidate != template
            )
            raise

        affected = (target, *self.downstream(target, recursive=True))
        # Re-enable only cards that this editor auto-disabled. Deliberately
        # disabled learned/experimental branches remain untouched.
        for node_id in self.topological_order():
            if node_id not in self._connection_suspended:
                continue
            if self.missing_input_ports(node_id):
                continue
            if any(not self.node(parent).enabled for parent in self.upstream(node_id)):
                continue
            node = self.node(node_id)
            node.enabled = True
            node.status = NodeStatus.IDLE
            node.status_detail = "Not run"
            self._connection_suspended.remove(node_id)
        self.revision += 1
        self.invalidate(affected, preserve_bypassed=True)
        for node_id in affected:
            if node_id in self._connection_suspended:
                node = self.node(node_id)
                node.status = NodeStatus.BYPASSED
                node.status_detail = "Disabled by disconnected input"
        return affected

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
        return self.set_parameters(node_id, {key: value})

    def set_parameters(
        self, node_id: str, values: Mapping[str, Any]
    ) -> tuple[str, ...]:
        """Atomically update several parameters with one graph invalidation.

        Every supplied value is normalized and validated before any node state
        changes.  A failed member therefore cannot leave an earlier member
        applied, increment the revision, or invalidate cached node statuses.
        """

        node = self._require_active(node_id)
        normalized = {
            key: node.validated_parameter_value(key, value)
            for key, value in dict(values).items()
        }
        changed = {
            key: value
            for key, value in normalized.items()
            if node.parameters.get(key) != value
        }
        if not changed:
            return ()
        node.parameters.update(changed)
        changed_specs = {
            spec.key: spec
            for spec in node.parameter_specs
            if spec.key in changed
        }
        display_only = bool(changed) and all(
            changed_specs.get(key) is not None
            and changed_specs[key].display_only
            for key in changed
        )
        if display_only:
            # Presentation controls are persisted on their owning node, but
            # must not supersede an in-flight calculation or invalidate any
            # cached analytical product.  The owning node is returned so Qt
            # can repaint its card and current diagnostic.
            return (node_id,)
        self.revision += 1
        affected = (node_id, *self.downstream(node_id, recursive=True))
        self.invalidate(affected)
        return affected

    def reset_parameters(self, node_id: str) -> tuple[str, ...]:
        """Restore one node's authored defaults and invalidate its dependents."""

        node = self._require_active(node_id)
        if node.parameters == node.default_parameters:
            return ()
        changed_keys = {
            key
            for key, default in node.default_parameters.items()
            if node.parameters.get(key) != default
        }
        specs = {spec.key: spec for spec in node.parameter_specs}
        display_only = bool(changed_keys) and all(
            specs.get(key) is not None and specs[key].display_only
            for key in changed_keys
        )
        node.parameters = dict(node.default_parameters)
        if display_only:
            return (node_id,)
        self.revision += 1
        affected = (node_id, *self.downstream(node_id, recursive=True))
        self.invalidate(affected, preserve_bypassed=True)
        return affected

    def set_enabled(self, node_id: str, enabled: bool) -> tuple[str, ...]:
        node = self._require_active(node_id)
        enabled = bool(enabled)
        if node.enabled == enabled:
            return ()
        affected = (node_id, *self.downstream(node_id, recursive=True))
        if enabled:
            missing_ports = self.missing_input_ports(node_id)
            if missing_ports:
                labels = dict(node.input_ports)
                missing = ", ".join(labels[port_id] for port_id in missing_ports)
                raise ValueError(f"Reconnect required input(s) first: {missing}.")
            disabled_upstream = tuple(
                upstream_id
                for upstream_id in self.upstream(node_id)
                if not self.node(upstream_id).enabled
            )
            if disabled_upstream:
                titles = ", ".join(
                    self.node(upstream_id).title for upstream_id in disabled_upstream
                )
                raise ValueError(f"Enable upstream node(s) first: {titles}.")
            node.enabled = True
            node.status = NodeStatus.IDLE
            node.status_detail = "Not run"
        else:
            # A disabled calculation cannot leave consumers apparently active.
            # Re-enabling remains explicit so a user never unknowingly turns a
            # whole unfinished branch back on.
            for affected_id in affected:
                affected_node = self.node(affected_id)
                affected_node.enabled = False
                affected_node.status = NodeStatus.BYPASSED
                affected_node.status_detail = "Disabled by pipeline dependency"
                self._connection_suspended.discard(affected_id)
        self.revision += 1
        self.invalidate(affected, preserve_bypassed=True)
        return affected

    def set_status(
        self, node_id: str, status: NodeStatus, detail: str = ""
    ) -> None:
        if node_id in self.unused_nodes:
            return
        node = self._require_active(node_id)
        node.status = status
        node.status_detail = detail or status.value.capitalize()

    def invalidate(
        self, node_ids: Iterable[str], *, preserve_bypassed: bool = False
    ) -> None:
        for node_id in node_ids:
            if node_id in self.unused_nodes:
                continue
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
        def node_payload(node: PipelineNode) -> dict[str, Any]:
            return {
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
                "input_port_types": dict(node.input_port_types),
                "output_port_types": dict(node.output_port_types),
                "inline_parameters": list(node.inline_parameters),
                "calculation_seconds": node.calculation_seconds,
            }

        def connection_payload(connection: PipelineConnection) -> dict[str, str]:
            return {
                "source": connection.source,
                "target": connection.target,
                "data_type": connection.data_type,
                "source_port": connection.source_port,
                "target_port": connection.target_port,
            }

        return {
            "revision": self.revision,
            "nodes": [node_payload(node) for node in self.nodes.values()],
            "connections": [
                connection_payload(connection) for connection in self.connections
            ],
            "unused_nodes": [
                node_payload(node) for node in self.unused_nodes.values()
            ],
            "unused_connections": [
                connection_payload(connection)
                for connection in self.unused_connections
            ],
            "disconnected_connections": [
                connection_payload(connection)
                for connection in self.connection_templates
                if connection not in self.connections
                and connection not in self.unused_connections
            ],
        }

    def _validate_connections(self) -> None:
        occupied_inputs: set[tuple[str, str]] = set()
        for connection in self.connections:
            if connection.source not in self.nodes:
                raise ValueError(f"Unknown connection source {connection.source!r}.")
            if connection.target not in self.nodes:
                raise ValueError(f"Unknown connection target {connection.target!r}.")
            if connection.source == connection.target:
                raise ValueError("A pipeline node cannot connect to itself.")
            source = self.nodes[connection.source]
            target = self.nodes[connection.target]
            if connection.source_port not in dict(source.output_ports):
                raise ValueError(
                    f"Unknown output port {connection.source_port!r} on {source.title}."
                )
            if connection.target_port not in dict(target.input_ports):
                raise ValueError(
                    f"Unknown input port {connection.target_port!r} on {target.title}."
                )
            if (
                source.output_port_types.get(connection.source_port)
                != connection.data_type
                or target.input_port_types.get(connection.target_port)
                != connection.data_type
            ):
                raise ValueError("A pipeline connection has incompatible port types.")
            input_key = (connection.target, connection.target_port)
            if input_key in occupied_inputs:
                raise ValueError("A pipeline input can have only one producer.")
            occupied_inputs.add(input_key)


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
            "Expected visible span",
            "float",
            10.0,
            1000.0,
            1.0,
            "Approximate visible metric span used to reject implausible tick-family "
            "counts. Absolute scale is measured from adjacent metric ticks, not "
            "from the ruler endpoints.",
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
        ParameterSpec("expected_radius_fraction", "Fallback radius / height", "float", 0.05, 0.48, 0.005, "Fallback selection prior used only when a calibrated ruler scale or physical dish diameter is unavailable."),
        ParameterSpec("expected_outer_diameter_mm", "Expected outer diameter", "float", 0.0, 1000.0, 1.0, "Expected physical diameter of the exterior Petri-dish edge. Set to zero to use only the image-relative fallback radius."),
        ParameterSpec("calibrated_outer_radius_tolerance_fraction", "Outer diameter tolerance", "float", 0.0, 0.25, 0.005, "Allowed fractional deviation of the exterior glass edge from the ruler-calibrated expected radius."),
        ParameterSpec("rim_pair_search_fraction", "Dual-rim search / radius", "float", 0.03, 0.30, 0.005, "Radial range searched around the selected Petri-dish rim for the second concentric glass edge."),
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
            "Multiplies each locally fitted maximum width of up to three isolated "
            "reference seeds. Lower values retain an explicit empirical correction "
            "after the attached-shadow fit.",
        ),
        ParameterSpec("reference_roi_x_min", "Reference ROI left", "float", 0.0, 0.99, 0.01, "Left edge as a fraction of corrected image width."),
        ParameterSpec("reference_roi_x_max", "Reference ROI right", "float", 0.01, 1.0, 0.01, "Right edge as a fraction of corrected image width."),
        ParameterSpec("reference_roi_y_min", "Reference ROI top", "float", 0.0, 0.99, 0.01, "Top edge as a fraction of corrected image height."),
        ParameterSpec("reference_roi_y_max", "Reference ROI bottom", "float", 0.01, 1.0, 0.01, "Bottom edge as a fraction of corrected image height."),
        ParameterSpec("reference_colour_distance_threshold", "Reference colour distance", "float", 2.0, 50.0, 0.5, "Minimum weighted Lab distance from the ROI median."),
        ParameterSpec("reference_max_aspect_ratio", "Maximum component aspect", "float", 1.0, 8.0, 0.1, "Reject more elongated reference components above this ratio."),
        ParameterSpec("reference_max_components", "Components used", "int", 1, 10, 1, "Maximum largest accepted components contributing to median diameter."),
        ParameterSpec(
            "annotated_seed_top_fraction",
            "Largest annotated fraction",
            "float",
            0.05,
            1.00,
            0.05,
            "Average this largest fraction of complete annotated-instance maximum widths. The default top quartile discards smaller cutoff/partial seeds before overriding the isolated-reference estimate.",
        ),
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
        ParameterSpec("foreground_chroma_weight", "Lab chroma weight", "float", 0.5, 5.0, 0.1, "Relative contribution of each Lab chroma channel to foreground distance."),
        ParameterSpec("foreground_reference_weight", "Foreground colour influence", "float", 0.0, 1.0, 0.05, "Strength of the user-authored Foreground colour model. Without painted Foreground pixels or enabled annotated-seed interiors, Foreground probability remains zero."),
        ParameterSpec(
            "foreground_include_annotated_seed_instances",
            "Include annotated seeds as Foreground",
            "bool",
            description=(
                "Use safely inset interiors of applied seed-instance annotations as "
                "additional Foreground material examples for colour, directional-noise, "
                "and material-prototype fitting. Instance contours are excluded, and "
                "painted Background, Other, and exclusion evidence takes precedence. "
                "Unapplied annotation drafts are not analysis evidence."
            ),
        ),
        ParameterSpec("foreground_reference_components", "Maximum reference colour modes", "int", 1, 256, 4, "Maximum coverage-preserving quantized Lab colour modes retained from individual painted foreground pixels."),
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
        ParameterSpec(
            "circle_edge_magnitude_weight",
            "Edge-magnitude weight",
            "float",
            0.0,
            1.0,
            0.05,
            "Contribution of the shared Lab/Scharr edge-magnitude raster to ring support.",
        ),
        ParameterSpec(
            "circle_sensor_noise_weight",
            "Sensor/noise-boundary weight",
            "float",
            0.0,
            1.0,
            0.05,
            "Contribution of boundaries derived from the image-quality node's sensor/noise likelihood.",
        ),
        ParameterSpec(
            "circle_flattened_grayscale_weight",
            "Flattened-grayscale weight",
            "float",
            0.0,
            1.0,
            0.05,
            "Contribution of boundaries in the locally flattened grayscale raster.",
        ),
        ParameterSpec(
            "circle_shadow_weight",
            "Shadow-boundary weight",
            "float",
            0.0,
            1.0,
            0.05,
            "Contribution of transitions around nonlinear local-shadow areas.",
        ),
        ParameterSpec(
            "circle_highlight_weight",
            "Highlight-boundary weight",
            "float",
            0.0,
            1.0,
            0.05,
            "Contribution of transitions around nonlinear local-highlight areas.",
        ),
        ParameterSpec("circle_min_distance_fraction", "Minimum centre spacing / diameter", "float", 0.20, 1.20, 0.01, "Local-maximum spacing between returned CUDA ring centres."),
        ParameterSpec("circle_min_radius_fraction", "Minimum radius / diameter", "float", 0.05, 0.80, 0.01, "Smallest tested CUDA ring radius relative to estimated seed diameter."),
        ParameterSpec("circle_max_radius_fraction", "Maximum radius / diameter", "float", 0.10, 1.20, 0.01, "Largest tested CUDA ring radius relative to estimated seed diameter."),
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
    )
    fusion_parameters = (
        ParameterSpec("distance_confidence", "Distance candidate confidence", "float", 0.10, 1.00, 0.01, "Weight assigned to every distance-peak proposal during fusion."),
    )
    background_parameters = (
        ParameterSpec("background_sample_radius_fraction", "Reference brush radius / diameter", "float", 0.01, 0.20, 0.005, "Visible painted-brush radius relative to estimated seed diameter; every covered pixel enters the reference calculation."),
        ParameterSpec("background_chroma_percentile", "Automatic chroma percentile", "float", 1.0, 99.0, 1.0, "Automatic samples must be no more chromatic than this percentile."),
        ParameterSpec("background_lightness_percentile", "Automatic lightness percentile", "float", 1.0, 99.0, 1.0, "Automatic samples must be at least this lightness percentile."),
        ParameterSpec("background_minimum_sample_fraction", "Minimum automatic area", "float", 0.0001, 0.25, 0.001, "Minimum fraction of valid dish pixels retained as automatic background references; the lightest, least-chromatic candidates are added if percentile filters return less."),
        ParameterSpec("background_prior_tolerance", "Perimeter colour tolerance", "float", 2.0, 100.0, 1.0, "Maximum weighted Lab distance from the outside-dish median admitted to the automatic background training set."),
        ParameterSpec(
            "background_keep_perimeter_reference",
            "Keep perimeter reference",
            "bool",
            description=(
                "Keep only the colour-filtered pixels in the buffered perimeter "
                "ring as an automatic Background source alongside painted "
                "Background references. Similar-coloured pixels inside the dish "
                "are not recruited. Uncheck to train from paint alone, or from "
                "the non-perimeter light/low-chroma fallback when no Background "
                "area has been painted."
            ),
        ),
        ParameterSpec("background_lightness_scale_floor", "Lightness range floor", "float", 1.0, 40.0, 0.5, "Minimum robust Lab lightness spread used by the colour probability model."),
        ParameterSpec("background_chroma_scale_floor", "Chroma range floor", "float", 0.5, 30.0, 0.5, "Minimum robust Lab a/b spread used by the colour probability model."),
        ParameterSpec("background_colour_components", "Maximum reference colour modes", "int", 1, 256, 4, "Maximum robust Lab colour modes fitted to painted or automatic background samples. Higher values preserve more colour variation but increase calculation time."),
        ParameterSpec("background_distribution_fit_iterations", "Distribution fit rounds", "int", 1, 20, 1, "Robust clustering rounds used to fit the background colour distribution."),
        ParameterSpec("background_frequency_weight_power", "Mode frequency influence", "float", 0.0, 1.0, 0.05, "Weights each learned background colour mode by its occurrence frequency in the painted mask. Zero treats represented modes equally; one applies their area proportions directly."),
        ParameterSpec("background_distribution_scale_multiplier", "Colour tolerance multiplier", "float", 0.50, 3.00, 0.05, "Expands or contracts every learned background Lab colour mode before per-pixel probability is calculated."),
        ParameterSpec("background_automatic_evidence_floor", "Perimeter evidence floor", "float", 0.0, 1.0, 0.02, "Minimum fit authority retained by the automatic perimeter source after painted Background coverage is available."),
        ParameterSpec("background_reviewed_authority_half_life_seed_areas", "Perimeter authority half-life / seed areas", "float", 0.05, 8.0, 0.05, "Painted Background coverage, measured in nominal seed areas, required to halve perimeter-source authority above its floor."),
    )
    perimeter_background_parameters = (
        ParameterSpec(
            "perimeter_background_buffer_cm",
            "Outer-rim buffer (cm)",
            "float",
            0.0,
            2.0,
            0.05,
            "Clear distance between the detected exterior Petri-dish edge and "
            "the start of the automatic median-colour reference band.",
        ),
        ParameterSpec(
            "perimeter_background_band_thickness_cm",
            "Reference band thickness (cm)",
            "float",
            0.05,
            2.0,
            0.05,
            "Radial thickness of the automatic median-colour reference band.",
        ),
    )
    noise_parameters = (
        ParameterSpec("noise_medium_scale_fraction", "Medium band / diameter", "float", 0.005, 0.20, 0.005, "Medium Gaussian frequency boundary relative to seed diameter."),
        ParameterSpec("noise_coarse_scale_fraction", "Coarse band / diameter", "float", 0.01, 0.40, 0.005, "Coarse Gaussian frequency boundary relative to seed diameter."),
        ParameterSpec("noise_direction_step_degrees", "Direction interval (degrees)", "int", 5, 90, 5, "Angular interval between one-sided texture-continuation rays. Fifteen degrees produces 24 unique direction overlays."),
        ParameterSpec("noise_vector_length_fraction", "Ray length / diameter", "float", 0.05, 2.00, 0.05, "Length of every directed texture-continuation ray relative to the master seed diameter."),
        ParameterSpec("noise_vector_sample_count", "Samples per ray", "int", 2, 32, 1, "Number of positions geometrically averaged along each directed ray."),
        ParameterSpec("noise_vector_decay", "Ray sample decay", "float", 0.10, 1.00, 0.02, "Multiplicative weight retained by each successively more distant ray sample."),
        ParameterSpec("noise_direction_integration", "Direction integration", "choice", choices=("mean", "maximum", "minimum", "median"), description="How directional continuation is merged. Mean averages all rays; maximum accepts the strongest continuation; minimum requires every direction; median requires broad directional support."),
        ParameterSpec("noise_background_min_likelihood", "Confident background minimum", "int", 0, 255, 1, "Minimum colour-likelihood value used as a texture background pseudo-label."),
        ParameterSpec("noise_nonbackground_max_likelihood", "Confident non-background maximum", "int", 0, 255, 1, "Maximum colour-likelihood value used as a texture non-background pseudo-label."),
        ParameterSpec("noise_working_maximum_dimension", "GPU working dimension", "int", 512, 4096, 128, "Maximum directional-noise analysis dimension. Work and upsampling stay on GPU; selected overlays retain the full crop dimensions."),
    )
    foreground_noise_parameters = tuple(
        ParameterSpec(
            {
                "noise_background_min_likelihood": "foreground_noise_foreground_min_likelihood",
                "noise_nonbackground_max_likelihood": "foreground_noise_nonforeground_max_likelihood",
            }.get(spec.key, spec.key.replace("noise_", "foreground_noise_", 1)),
            spec.label.replace("background", "foreground").replace(
                "Background", "Foreground"
            ),
            spec.kind,
            spec.minimum,
            spec.maximum,
            spec.step,
            (
                "How foreground continuation is merged. The 1st tertile is "
                "the exact 33⅓ percentile: more conservative than median but "
                "less brittle than requiring every direction with minimum."
                if spec.key == "noise_direction_integration"
                else spec.description.replace("background", "foreground").replace(
                    "Background", "Foreground"
                )
            ),
            (
                (*spec.choices, "1st tertile")
                if spec.key == "noise_direction_integration"
                else spec.choices
            ),
        )
        for spec in noise_parameters
    )
    edge_parameters = (
        ParameterSpec("edge_gradient_method", "Gradient method", "choice", choices=("scharr", "sobel", "prewitt", "central_difference"), description="Derivative operator used for every selected original/wavelet source. Scharr has the best small-kernel rotational symmetry; Sobel/Prewitt smooth more, and central difference is the least smoothed."),
        ParameterSpec("edge_gradient_source_fusion", "Source fusion", "choice", choices=("maximum", "root_mean_square", "vector_sum"), description="How concurrently enabled original/wavelet gradient vectors are fused. Maximum keeps the locally strongest source, RMS combines strength while averaging direction, and vector sum permits cancellation."),
        ParameterSpec("edge_gradient_include_original", "Include original image", "bool", description="Include the corrected original image in the shared edge-gradient calculation."),
        ParameterSpec("edge_gradient_include_wavelet_detail_1", "Include wavelet detail 1", "bool", description="Include the finest signed wavelet detail band."),
        ParameterSpec("edge_gradient_include_wavelet_detail_2", "Include wavelet detail 2", "bool", description="Include the second signed wavelet detail band."),
        ParameterSpec("edge_gradient_include_wavelet_detail_3", "Include wavelet detail 3", "bool", description="Include the third signed wavelet detail band."),
        ParameterSpec("edge_gradient_include_wavelet_detail_4", "Include wavelet detail 4", "bool", description="Include the coarsest signed wavelet detail band."),
        ParameterSpec("edge_gradient_include_wavelet_residual", "Include wavelet residual", "bool", description="Include the low-pass wavelet residual as an additional gradient source."),
        ParameterSpec("edge_wavelet_detail_gain", "Wavelet detail gain", "float", 0.0, 8.0, 0.05, "Gain applied to signed detail-band derivatives before source fusion; it does not affect the original image or low-pass residual."),
        ParameterSpec("edge_blur_sigma", "Pre-edge blur sigma", "float", 0.0, 5.0, 0.1, "Gaussian sigma applied to each selected source before derivatives; zero disables the additional blur."),
        ParameterSpec("edge_chroma_weight", "Edge chroma weight", "float", 0.1, 5.0, 0.1, "Relative contribution of Lab a/b gradients."),
        ParameterSpec("edge_normalization_percentile", "Strength normalization percentile", "float", 80.0, 99.9, 0.1, "Response percentile mapped to maximum overlay brightness."),
        ParameterSpec("edge_strength_gamma", "Strength display gamma", "float", 0.10, 2.00, 0.05, "Below 1 brightens weak edges; above 1 suppresses them."),
    )
    shared_edge_values = {
        "edge_gradient_method": "scharr",
        "edge_gradient_source_fusion": "maximum",
        "edge_gradient_include_original": True,
        "edge_gradient_include_wavelet_detail_1": False,
        "edge_gradient_include_wavelet_detail_2": False,
        "edge_gradient_include_wavelet_detail_3": False,
        "edge_gradient_include_wavelet_detail_4": False,
        "edge_gradient_include_wavelet_residual": False,
        "edge_wavelet_detail_gain": 1.0,
        "edge_blur_sigma": 1.2,
        "edge_chroma_weight": 1.5,
        "edge_normalization_percentile": 99.0,
        "edge_strength_gamma": 0.65,
    }
    surface_gradient_parameters = (
        ParameterSpec("surface_gradient_blur_sigma", "Surface blur sigma", "float", 0.1, 8.0, 0.1, "Gaussian smoothing applied to CIE L* before one-sided surface slopes are measured."),
        ParameterSpec("surface_gradient_radius_fraction", "Ray length / diameter", "float", 0.05, 2.0, 0.05, "Farthest target pixel tested along each one-sided ray, relative to the image-specific seed diameter."),
        ParameterSpec("surface_gradient_direction_step_degrees", "Direction step", "int", 5, 90, 5, "Angular spacing between independently tested one-sided rays; smaller steps resolve direction more finely."),
        ParameterSpec("surface_gradient_sample_count", "Samples per ray", "int", 2, 32, 1, "Number of target distances tested along each direction before the maximum lightening and darkening slopes are selected."),
        ParameterSpec("surface_gradient_normalization_percentile", "Strength normalization percentile", "float", 80.0, 99.9, 0.1, "Raw L* slope percentile mapped to maximum overlay brightness without changing the physical cutoff values."),
        ParameterSpec("surface_gradient_strength_gamma", "Strength display gamma", "float", 0.10, 2.00, 0.05, "Display-only exponent applied after percentile normalization; below one reveals weak surface changes."),
        ParameterSpec("surface_gradient_working_maximum_dimension", "GPU working dimension", "int", 512, 4096, 128, "Maximum tensor dimension used for the ray bank; outputs are restored to the full dish crop on the GPU."),
    )
    surface_gradient_ceiling_parameters = (
        ParameterSpec("lightening_gradient_maximum_slope", "Maximum retained L* slope / px", "float", 0.05, 30.0, 0.05, "Zero lightening responses whose raw CIE L* change per original-image pixel exceeds this ceiling, suppressing hard edges."),
    )
    darkening_gradient_ceiling_parameters = (
        ParameterSpec("darkening_gradient_maximum_slope", "Maximum retained L* slope / px", "float", 0.05, 30.0, 0.05, "Zero darkening responses whose raw CIE L* change per original-image pixel exceeds this ceiling, suppressing hard edges."),
    )
    frequency_noise_mask_parameters = (
        ParameterSpec("frequency_noise_fine_scale_fraction", "Fine scale / diameter", "float", 0.002, 0.20, 0.002, "Smallest Gaussian scale used to isolate high-frequency local darkness and Lab-chroma variation."),
        ParameterSpec("frequency_noise_medium_scale_fraction", "Medium scale / diameter", "float", 0.005, 0.50, 0.005, "Middle Gaussian scale defining the fine-to-medium frequency band."),
        ParameterSpec("frequency_noise_coarse_scale_fraction", "Coarse scale / diameter", "float", 0.01, 1.50, 0.01, "Largest Gaussian scale defining the medium-to-coarse frequency band."),
        ParameterSpec("frequency_noise_context_fraction", "RMS context / diameter", "float", 0.002, 0.50, 0.002, "Surrounding neighbourhood over which each band-limited darkness or colour residual is converted to local RMS energy."),
        ParameterSpec("frequency_noise_normalization_percentile", "Mask normalization percentile", "float", 80.0, 99.9, 0.1, "Per-band response percentile mapped to full mask brightness."),
        ParameterSpec("frequency_noise_strength_gamma", "Mask display gamma", "float", 0.10, 2.00, 0.05, "Display exponent shared by all six frequency masks."),
        ParameterSpec("frequency_noise_working_maximum_dimension", "GPU working dimension", "int", 512, 4096, 128, "Maximum tensor dimension used for multiscale noise energy before full-resolution GPU restoration."),
    )
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
    reference_ridge_parameters = (
        ParameterSpec("reference_ridge_nms_step_px", "Normal sampling step", "float", 0.25, 3.0, 0.05, "Distance sampled on both sides of reference-trained physical-edge probability during subpixel non-maximum suppression."),
        ParameterSpec("reference_ridge_low_threshold", "Hysteresis low", "float", 0.0, 0.95, 0.01, "Weak reference maxima retained only when connected to a strong reference ridge."),
        ParameterSpec("reference_ridge_high_threshold", "Hysteresis high", "float", 0.01, 1.0, 0.01, "Strong reference-edge threshold that seeds hysteresis reconstruction."),
        ParameterSpec("reference_ridge_hysteresis_iterations", "Hysteresis reach", "int", 1, 32, 1, "Maximum GPU morphology steps through which connected weak reference ridges can be retained."),
        ParameterSpec("reference_ridge_working_maximum_dimension", "GPU working dimension", "int", 256, 4096, 64, "Maximum tensor dimension used for reference-probability thinning before the result is restored to source resolution."),
        ParameterSpec("reference_edge_normalization_radius_fraction", "Local normalization radius / diameter", "float", 0.05, 1.0, 0.01, "Seed-relative radius of the valid-weighted local support envelope. Larger values correct slower spatial variation but compare across more neighbouring edges."),
        ParameterSpec("reference_edge_normalization_target_support", "Target local support", "float", 0.05, 0.95, 0.01, "Target amplitude used to equalize the robust local total learned-edge envelope before semantic gating."),
        ParameterSpec("reference_edge_normalization_maximum_gain", "Maximum local gain", "float", 1.0, 8.0, 0.10, "Symmetric bound on local support amplification and attenuation. A value of 2.5 permits gains from 0.4x to 2.5x."),
        ParameterSpec("reference_edge_normalization_absolute_floor", "Absolute support floor", "float", 0.0, 0.50, 0.005, "Total learned physical-plus-internal evidence at or below this value remains zero after local normalization; the gate transitions smoothly to full support at twice the floor."),
    )
    trace_parameters = (
        ParameterSpec(
            "trace_edge_source",
            "Ridge source",
            "choice",
            choices=(
                "generic_ridges",
                "reference_ridges",
                "net_reference_ridges",
                "normalized_net_reference_ridges",
            ),
            description="Selects the thinned edge field linked into oriented traces. Tangents still come from the shared continuous image gradient.",
        ),
        ParameterSpec("trace_diameter_multiplier", "Trace diameter / master diameter", "float", 0.25, 3.0, 0.05, "Explicit multiplier linking trace minimum length and continuity-window radius to the image's master reference seed diameter."),
        ParameterSpec("trace_tangent_tolerance_degrees", "Link tangent tolerance", "float", 2.0, 60.0, 1.0, "Maximum axial tangent disagreement used to join ridge pixels into one trace."),
        ParameterSpec("trace_maximum_gap_px", "Maximum trace gap", "int", 1, 5, 1, "Largest Chebyshev separation between candidate endpoint coordinates in the working grid. A value of four does not mean four empty pixels between the endpoints."),
        ParameterSpec("trace_curvature_policy", "Convexity bias", "choice", choices=("off", "prefer", "require"), description="Curvature rule for initial non-adjacent links. Prefer rejects confident S-shaped bridges; require rejects every resolved S-shaped bridge; off preserves tangent-only linking."),
        ParameterSpec("trace_curvature_tolerance_degrees", "Curvature ambiguity", "float", 0.0, 45.0, 1.0, "Endpoint bend angles at or below this value are treated as too nearly straight to classify reliably. Prefer uses at least twice this deadband."),
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
        ParameterSpec("boundary_instance_edge_influence", "Instance-edge influence", "float", 0.0, 1.0, 0.05, "Soft influence of the instance-derived physical-edge probability on final confirmed curves."),
        ParameterSpec("boundary_nonphysical_edge_discount", "Non-physical discount", "float", 0.0, 1.0, 0.05, "Suppression applied where the instance-derived classifier identifies an apparent rather than physical edge."),
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
        ParameterSpec("flattening_contrast_gain", "Flattened contrast gain", "float", 0.10, 8.0, 0.10, "Gain applied to the log ratio between grayscale and its local illumination field."),
        ParameterSpec("lighting_deviation_scale_fraction", "Extreme context / diameter", "float", 0.01, 1.0, 0.01, "Local context used to standardize dark and bright deviations from flattened grayscale."),
        ParameterSpec("shadow_z_threshold", "Shadow deviation threshold", "float", 0.05, 5.0, 0.05, "Standardized negative local-lighting deviation at which shadow probability reaches its nonlinear transition."),
        ParameterSpec("highlight_z_threshold", "Highlight deviation threshold", "float", 0.05, 5.0, 0.05, "Standardized positive local-lighting deviation at which highlight probability reaches its nonlinear transition."),
        ParameterSpec("lighting_extreme_softness", "Extreme transition softness", "float", 0.05, 2.0, 0.05, "Width of the sigmoid transition used for both shadow and highlight likelihoods."),
    )
    reference_texture_parameters = (
        ParameterSpec("reference_texture_material_prototypes_per_class", "Maximum material prototypes / class", "int", 8, 256, 8, "Maximum coverage-preserving feature medoids retained independently for each of Background, Foreground, and Other."),
        ParameterSpec("reference_texture_minimum_samples_per_prototype", "Minimum support / prototype", "int", 4, 512, 4, "Minimum working-resolution painted samples per retained prototype; diverse classes can still use fewer prototypes when evidence is sparse."),
        ParameterSpec("reference_texture_fit_iterations", "Prototype fit iterations", "int", 1, 12, 1, "Robust farthest-first clustering iterations used to refine each image-local prototype bank."),
        ParameterSpec("reference_texture_similarity_scale", "Prototype tolerance", "float", 0.25, 4.0, 0.05, "Scales each prototype's robust diagonal feature spread when evaluating matching pixels throughout the dish."),
        ParameterSpec("reference_texture_context_fraction", "Feature context / diameter", "float", 0.005, 0.30, 0.005, "Seed-relative neighbourhood used for local Lab residuals, edge density, and tangent coherence."),
        ParameterSpec("reference_texture_patch_fraction", "Collage patch / diameter", "float", 0.10, 0.80, 0.02, "Side length of the source-image thumbnail retained for each prototype in the collage."),
        ParameterSpec("reference_texture_working_maximum_dimension", "GPU working dimension", "int", 256, 2048, 64, "Maximum dimension for prototype fitting and global feature matching; full-resolution probability outputs are restored on the GPU."),
    )
    material_evidence_parameters = (
        ParameterSpec("material_colour_weight", "Colour evidence weight", "float", 0.0, 3.0, 0.05, "Reliability weight shared by Foreground, Background, and Other colour evidence. Zero means no class support and one means maximum support; raw maps are not required to be normalized against each other."),
        ParameterSpec("material_noise_weight", "Texture evidence weight", "float", 0.0, 3.0, 0.05, "Reliability weight shared by the three directional texture evidence maps."),
        ParameterSpec("material_prototype_weight", "Prototype evidence weight", "float", 0.0, 3.0, 0.05, "Reliability weight for valid multiclass material-prototype probabilities. One-class similarities are refused upstream."),
        ParameterSpec("material_unknown_weight", "Unknown evidence weight", "float", 0.01, 3.0, 0.02, "Reserved mass when neither Seed nor Non-seed has convincing support."),
        ParameterSpec("material_decision_temperature", "Decision temperature", "float", 0.25, 4.0, 0.05, "Calibration temperature applied to Seed, Non-seed, Ambiguous, and Unknown masses before normalization. Lower values sharpen decisions."),
        ParameterSpec("material_seed_threshold", "Seed proposal threshold", "float", 0.05, 0.95, 0.01, "Minimum resolved Seed mass admitted to the authoritative binary seed-material proposal mask."),
        ParameterSpec("material_morphology_fraction", "Proposal morphology / diameter", "float", 0.0, 0.20, 0.005, "Opening/closing kernel size for the final binary seed-material proposal mask, relative to estimated seed diameter."),
    )
    reference_edge_probability_parameters = (
        ParameterSpec("reference_texture_edge_prototypes_per_class", "Maximum edge prototypes / class", "int", 8, 1024, 16, "Maximum coverage-preserving medoids retained independently for the instance-derived Physical-edge and Non-physical-edge classes."),
        ParameterSpec("reference_edge_minimum_samples_per_prototype", "Minimum edge support / prototype", "int", 4, 512, 4, "Minimum working-resolution edge samples per retained physical or non-physical prototype."),
        ParameterSpec("reference_edge_fit_iterations", "Edge prototype fit iterations", "int", 1, 12, 1, "Robust farthest-first clustering iterations for the two edge prototype banks."),
        ParameterSpec("reference_edge_similarity_scale", "Edge prototype tolerance", "float", 0.25, 4.0, 0.05, "Scales each edge prototype's robust feature spread during global matching."),
        ParameterSpec("reference_texture_edge_working_maximum_dimension", "Edge GPU working dimension", "int", 512, 4096, 128, "Hard maximum dimension for adaptive tangent-strip edge fitting and matching."),
        ParameterSpec("reference_edge_minimum_working_seed_diameter_px", "Minimum edge seed diameter / px", "float", 8.0, 96.0, 1.0, "Adaptive edge-strip resolution target in working pixels per seed diameter."),
        ParameterSpec("reference_edge_strip_normal_offset_fraction", "Strip side offset / diameter", "float", 0.01, 0.20, 0.005, "Distance from the edge centre to the separately pooled interior and exterior strips."),
        ParameterSpec("reference_edge_strip_tangent_half_length_fraction", "Strip tangent half-length / diameter", "float", 0.01, 0.25, 0.005, "Half-length of each narrow strip along the local edge tangent."),
        ParameterSpec("reference_edge_ridge_weight", "Ridge support", "float", 0.0, 1.0, 0.05, "Blend between continuous edge magnitude and thinned-ridge support before classification."),
        ParameterSpec("reference_texture_instance_interior_buffer_fraction", "Interior buffer / diameter", "float", 0.0, 0.50, 0.01, "Distance kept unlabeled between annotated contours and internal-edge examples."),
        ParameterSpec(
            "net_physical_edge_internal_scale",
            "Internal-edge subtraction",
            "float",
            0.0,
            2.0,
            0.05,
            "Multiplier applied to non-physical/internal-edge evidence: max(physical - multiplier × internal, 0). It does not alter either source probability raster, but it does recompute the thinned net ridge and downstream consumers.",
        ),
    )
    procedural_instance_parameters = (
        ParameterSpec("working_maximum_dimension", "Topology working dimension", "int", 256, 4096, 128, "Maximum raster dimension downloaded after GPU resizing for CPU marker-controlled watershed."),
        ParameterSpec("foreground_threshold_scale", "Material threshold scale", "float", 0.20, 2.00, 0.02, "Multiplier applied to the Otsu seed-material threshold."),
        ParameterSpec("occupancy_closing_fraction", "Material closing / diameter", "float", 0.01, 0.50, 0.01, "Seed-relative closing radius used to bridge interrupted material evidence."),
        ParameterSpec("occupancy_hole_area_fraction", "Filled-hole area / diameter²", "float", 0.0, 4.0, 0.05, "Largest enclosed low-probability coat region filled as seed material."),
        ParameterSpec("dish_margin_fraction", "Dish margin / diameter", "float", 0.0, 0.50, 0.01, "Inset from the valid dish boundary that prevents glass-rim instances."),
        ParameterSpec("boundary_edge_weight", "Candidate edge weight", "float", 0.0, 1.0, 0.02, "Contribution of continuous Lab edge magnitude to the initial boundary candidate."),
        ParameterSpec("boundary_ridge_weight", "Candidate ridge weight", "float", 0.0, 1.0, 0.02, "Contribution of generic thinned edge ridges to the initial boundary candidate."),
        ParameterSpec("boundary_semantic_floor", "Unclassified-edge floor", "float", 0.0, 1.0, 0.02, "Minimum candidate support retained where instance-derived physical/non-physical evidence is uncertain."),
        ParameterSpec("boundary_nonphysical_discount", "Non-physical discount", "float", 0.0, 1.0, 0.02, "How strongly instance-derived non-physical edge evidence suppresses the entire boundary candidate."),
        ParameterSpec("boundary_physical_ridge_weight", "Net-physical ridge weight", "float", 0.0, 1.0, 0.02, "Contribution of thinned physical-minus-scaled-internal edge probability to boundary support."),
        ParameterSpec("boundary_trace_weight", "Oriented-trace weight", "float", 0.0, 1.0, 0.02, "Contribution of coherent seed-scale oriented edge traces to boundary support."),
        ParameterSpec("boundary_surface_darkening_weight", "Surface-darkening boundary weight", "float", 0.0, 1.0, 0.02, "Contribution of one-sided seed-scale surface-darkening magnitude to the boundary candidate. This is independent of the local directional edge gradient."),
        ParameterSpec("trace_minimum_length_fraction", "Trace minimum / diameter", "float", 0.02, 2.0, 0.02, "Minimum trace length relative to seed diameter before it supports the watershed boundary."),
        ParameterSpec("trace_convexity_weight", "Trace convexity weight", "float", 0.0, 1.0, 0.02, "How strongly seed-scale elliptical convexity contributes to oriented-trace support."),
        ParameterSpec("centre_geometry_smoothing_fraction", "Centre smoothing / diameter", "float", 0.02, 0.50, 0.01, "Seed-relative smoothing applied to material occupancy before automatic centre detection."),
        ParameterSpec("centre_material_weight", "Centre material weight", "float", 0.0, 1.0, 0.02, "Contribution of seed-scale smoothed material evidence to marker likelihood."),
        ParameterSpec("centre_distance_weight", "Centre interior-depth weight", "float", 0.0, 1.0, 0.02, "Contribution of distance from the material and semantic-boundary barriers to marker likelihood."),
        ParameterSpec("centre_flattened_grayscale_weight", "Blurred flattened-gray weight", "float", 0.0, 1.0, 0.02, "Contribution of seed-scale blurred illumination-flattened grayscale contrast. Both dark and light coat centres are retained, while fine coat patterns are averaged before marker selection."),
        ParameterSpec("centre_minimum_separation_fraction", "Marker spacing / diameter", "float", 0.10, 1.00, 0.01, "Minimum automatic marker spacing in ordinary and crowded seed material."),
        ParameterSpec("sparse_centre_minimum_separation_fraction", "Sparse marker spacing / diameter", "float", 0.10, 1.00, 0.01, "Stricter marker spacing used when seed material covers less than 35% of the dish."),
        ParameterSpec("marker_count_multiplier", "Expected marker-count multiplier", "float", 0.50, 2.00, 0.01, "Scales the seed-area-derived expected marker count used to retain the strongest automatic centre candidates."),
        ParameterSpec("minimum_marker_score", "Minimum marker likelihood", "float", 0.0, 1.0, 0.01, "Reject automatic centre maxima below this combined likelihood."),
        ParameterSpec("minimum_instance_area_fraction", "Hard minimum area / diameter²", "float", 0.05, 1.40, 0.01, "Reject generated candidates below this calibrated seed-relative area. The default is approximately one third of a circular reference-seed area."),
        ParameterSpec("soft_minimum_instance_area_fraction", "Soft minimum area / diameter²", "float", 0.06, 1.45, 0.01, "Candidates below this area but above the hard minimum are retained with a progressively lower score."),
        ParameterSpec("maximum_instance_area_fraction", "Hard maximum area / diameter²", "float", 0.06, 3.00, 0.01, "Reject generated watershed regions larger than this calibrated seed-relative area; reviewed annotations remain authoritative."),
        ParameterSpec("soft_maximum_instance_width_fraction", "Soft maximum width / diameter", "float", 0.50, 3.50, 0.01, "Maximum side-to-side minimum-area-box width before a candidate receives a gradual score penalty."),
        ParameterSpec("hard_maximum_instance_width_fraction", "Hard maximum width / diameter", "float", 0.51, 4.00, 0.01, "Reject generated candidates whose side-to-side width exceeds this calibrated limit."),
        ParameterSpec("maximum_internal_concavity_fraction", "Hard maximum concavity", "float", 0.0, 1.0, 0.01, "Reject candidates when the area missing inside their convex-hull outline exceeds this fraction of identified seed area."),
        ParameterSpec("maximum_protrusion_area_fraction", "Hard maximum thin protrusion", "float", 0.0, 1.0, 0.01, "Reject candidates when a seed-scaled morphological opening identifies too much area in long narrow protrusions."),
        ParameterSpec("minimum_instance_solidity", "Hard minimum solidity", "float", 0.10, 1.00, 0.01, "Reject generated regions whose area divided by convex-hull area is below this value, preventing deeply concave multi-seed Frankenstein shapes."),
        ParameterSpec("maximum_instance_axis_ratio", "Hard maximum axis ratio", "float", 1.0, 8.0, 0.1, "Reject generated regions whose minimum-area bounding rectangle is more elongated than this ratio."),
        ParameterSpec("candidate_hypotheses_per_marker", "Candidates per centre", "int", 1, 9, 1, "Number of material-boundary thresholds evaluated for each retained centre before overlap-aware combination selection."),
        ParameterSpec("candidate_overlap_fraction", "Candidate overlap tolerance", "float", 0.0, 0.25, 0.005, "Largest overlap fraction tolerated while combining candidates; final assigned output pixels remain non-overlapping."),
    )
    unet_instance_parameters = (
        ParameterSpec("checkpoint_path", "Checkpoint path", "text", description="Project-relative or absolute path to a self-describing multi-head U-Net checkpoint."),
        ParameterSpec("tile_size", "Inference tile size", "int", 128, 2048, 64, "Square CUDA inference tile. Larger tiles reduce seams but consume more GPU memory."),
        ParameterSpec("tile_overlap", "Tile overlap", "int", 0, 512, 16, "Context blended between adjacent tiles; it must remain below half the tile size."),
        ParameterSpec("interior_threshold", "Interior threshold", "float", 0.05, 0.95, 0.01, "Minimum learned seed-interior probability admitted to watershed."),
        ParameterSpec("centre_threshold", "Centre threshold", "float", 0.05, 0.95, 0.01, "Minimum learned centre likelihood accepted as an automatic watershed marker."),
        ParameterSpec("centre_minimum_separation_fraction", "Centre spacing / diameter", "float", 0.05, 1.50, 0.01, "Minimum marker spacing relative to calibrated seed diameter."),
        ParameterSpec("minimum_instance_area_fraction", "Minimum area / diameter squared", "float", 0.01, 1.50, 0.01, "Decoded regions below this calibrated area are rejected."),
        ParameterSpec("physical_boundary_weight", "Physical-boundary weight", "float", 0.0, 1.0, 0.02, "Contribution of the learned physical boundary to watershed elevation."),
        ParameterSpec("distance_topography_weight", "Distance-topography weight", "float", 0.0, 1.0, 0.02, "Contribution of inverse learned interior depth to watershed elevation."),
        ParameterSpec("pattern_boundary_discount", "Pattern-boundary discount", "float", 0.0, 1.0, 0.02, "How strongly predicted non-physical coat-pattern boundaries suppress physical boundary cost."),
        ParameterSpec("uncertainty_penalty", "Uncertainty penalty", "float", 0.0, 1.0, 0.02, "Contribution of predicted dense-task uncertainty to watershed elevation and review confidence."),
        ParameterSpec("foreground_erosion_fraction", "Foreground erosion / diameter", "float", 0.0, 0.10, 0.005, "Optional calibrated inward correction of the learned interior support before watershed."),
    )
    stardist_instance_parameters = (
        ParameterSpec("checkpoint_path", "Checkpoint path", "text", description="Project-relative or absolute path to a self-describing native PyTorch StarDist checkpoint."),
        ParameterSpec("tile_size", "Inference tile size", "int", 128, 2048, 64, "Square CUDA inference tile. Larger tiles reduce seams but consume more GPU memory."),
        ParameterSpec("tile_overlap", "Tile overlap", "int", 0, 512, 16, "Context blended between adjacent tiles; it must remain below half the tile size."),
        ParameterSpec("object_threshold", "Object threshold", "float", 0.05, 0.95, 0.01, "Minimum learned StarDist object probability retained as a polygon centre."),
        ParameterSpec("nms_iou_threshold", "NMS overlap threshold", "float", 0.05, 0.95, 0.01, "Polygon overlap above which the lower-scored StarDist candidate is suppressed."),
        ParameterSpec("local_maximum_radius_fraction", "Peak radius / diameter", "float", 0.01, 0.50, 0.01, "Object-probability non-maximum radius relative to calibrated seed diameter."),
        ParameterSpec("minimum_instance_area_fraction", "Minimum area / diameter squared", "float", 0.01, 1.50, 0.01, "Decoded polygons below this calibrated area are rejected."),
        ParameterSpec("maximum_candidates", "Maximum polygon candidates", "int", 64, 16384, 64, "Safety cap applied after object-probability ranking and before polygon NMS."),
    )
    image_quality_parameters = (
        ParameterSpec("quality_noise_scale_fraction", "Noise scale / diameter", "float", 0.005, 0.30, 0.005, "Local high-frequency scale used for sensor/noise risk."),
    )
    radial_parameters = (
        ParameterSpec("radial_bin_count", "Radial profile bins", "int", 4, 128, 1, "Number of normalized centre-to-edge bins used to estimate expected seed lightness."),
    )
    wavelet_parameters = (
        ParameterSpec(
            "wavelet_level_count",
            "Wavelet detail levels",
            "int",
            1,
            4,
            1,
            "Number of undecimated B3-spline à trous detail bands. Unused fixed output bands are exactly zero.",
        ),
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
            "metadata", "Species and metadata", "Input", "Species, lot and notes", 240, 40,
            status_detail="Choose species",
        ),
        PipelineNode(
            "reference_layers",
            "Manual annotations",
            "Input",
            "Applied material classes, seed instances, and manual seed centres",
            1040,
            -680,
            details=(
                "This input owns every image-local manual annotation. Its four active "
                "full-resolution raster sublayers and one point layer are explicit: "
                "Background, foreground and other are mutually exclusive material classes; "
                "annotated seeds are integer instance identities. Optional manual centre "
                "points augment or replace automatically discovered procedural watershed "
                "markers without painting or expanding the material masks. Physical contours and "
                "internal non-physical edge examples are derived automatically from complete "
                "annotated instances rather than painted in a separate boundary layer. Each "
                "output has its own typed "
                "connector so every calculation that consumes human evidence is explicit "
                "in the graph. Painted coordinates are training evidence and are never "
                "forced directly into a probability output."
            ),
            output_ports=(
                ("background", "Background"),
                ("foreground", "Foreground"),
                ("other", "Other"),
                ("annotated_seeds", "Annotated seeds"),
                ("centres", "Manual seed centres"),
            ),
            output_port_types={"centres": "ManualSeedCentres"},
            status_detail="No applied references",
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
            "Identify the ruler outline and separate metric and imperial tick families",
            960,
            100,
            parameters={"ruler_length_mm": 150.0},
            parameter_specs=ruler_parameters,
        ),
        PipelineNode(
            "deskew_colour",
            "Deskew and colour balance",
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
            "Detect the lower and upper mildly elliptical glass edges of a Petri dish",
            1120,
            -400,
            details=(
                "The corrected image is reduced to the configured maximum dimension, "
                "converted to grayscale, and Gaussian-blurred on the tensor device. "
                "A CUDA circular bank scores radial-gradient support across the configured radius interval. "
                "When ruler scale is available, the editable physical outer-diameter prior excludes "
                "smaller circular seed-mass boundaries; otherwise the image-relative radius prior is used. "
                "The selected primary rim then seeds a dense "
                "concentric radial-profile search that resolves the lower/inner and "
                "upper/outer glass edges. "
                "Both edges are retained for layout review; the outer edge defines the complete "
                "vessel extent and is the radius supplied to every downstream analysis node. "
                "A bounded ellipse refinement retains mild camera-view ovality and uses the "
                "larger semi-axis for conservative circular consumers, so seeds are not clipped. "
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
                "expected_outer_diameter_mm": 96.0,
                "calibrated_outer_radius_tolerance_fraction": 0.025,
                "rim_pair_search_fraction": 0.14,
                "rim_pair_min_separation_fraction": 0.025,
                "rim_pair_max_separation_fraction": 0.12,
                "rim_pair_expected_separation_fraction": 0.045,
                "rim_pair_secondary_support_fraction": 0.30,
            },
            parameter_specs=layout_parameters,
        ),
        PipelineNode(
            "hue_only",
            "Hue only",
            "GPU diagnostic",
            "Display corrected-image hue at fixed neutral darkness",
            1240,
            -560,
            details=(
                "Converts the corrected dish crop to hue while holding displayed value "
                "at 62%. Saturated colours therefore differ only by hue; effectively "
                "achromatic pixels remain neutral gray instead of receiving an arbitrary hue."
            ),
            output_ports=(("hue", "Hue-only RGB"),),
        ),
        PipelineNode(
            "wavelet_decomposition",
            "Wavelet decomposition",
            "GPU diagnostic",
            "Undecimated multiscale B3-spline decomposition with exact reconstruction",
            1160,
            -560,
            parameters={"wavelet_level_count": 4},
            parameter_specs=wavelet_parameters,
            inline_parameters=(("wavelet_level_count", "Levels"),),
            output_ports=(
                ("detail_1", "Wavelet detail 1"),
                ("detail_2", "Wavelet detail 2"),
                ("detail_3", "Wavelet detail 3"),
                ("detail_4", "Wavelet detail 4"),
                ("residual", "Wavelet low-pass residual"),
                ("pyramid", "Complete wavelet pyramid"),
            ),
            details=(
                "Applies a stationary à trous transform with the separable B3-spline "
                "kernel [1, 4, 6, 4, 1]/16 at dyadic dilations. Each detail is the "
                "previous approximation minus the next; adding every detail to the "
                "final residual reconstructs the corrected crop exactly, apart from "
                "floating-point roundoff. Signed detail overlays use neutral gray for zero."
            ),
        ),
        PipelineNode(
            "scale_calibration",
            "Absolute ruler scale",
            "Calibration",
            "Estimate pixels per millimetre from deskewed ruler ticks",
            1040,
            240,
            parameters={"minor_tick_mm": 1.0},
            parameter_specs=scale_parameters,
        ),
        PipelineNode(
            "seed_scale_estimation",
            "Seed scale estimate",
            "Segmentation",
            "Estimate seed diameter from fitted references and reviewed instances",
            1200,
            -160,
            details=(
                "The editable isolated-reference region is converted to CIE Lab. "
                "Pixels sufficiently unlike "
                "the region median form connected components after morphological "
                "opening and closing. Components touching the crop edge, with extreme "
                "area/aspect, or implausible relative to the dish are rejected. A "
                "maximum-inscribed seed core then initializes a local foreground fit so "
                "attached cast shadows cannot define the final silhouette. The median "
                "corrected maximum width of the configured components provides the "
                "initial estimate. Those automatic fits estimate scale only and never "
                "become foreground-colour evidence. If no component survives, the editable "
                "dish-radius fallback is available. When at least "
                "two complete annotated instances exist, their largest configured fraction "
                "of exact maximum-Feret widths overrides the initial estimate; this drops "
                "small cutoff/partial masks while retaining the initial fit as a diagnostic."
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
                "annotated_seed_top_fraction": 0.25,
                "fallback_diameter_fraction": 0.16,
            },
            parameter_specs=seed_scale_parameters,
        ),
        PipelineNode(
            "perimeter_background_reference",
            "Perimeter background reference",
            "Segmentation",
            "Sample a buffered colour-reference band outside the Petri dish",
            1240,
            40,
            details=(
                "The ruler-calibrated controls define a clear buffer beyond the "
                "detected upper/outer glass edge followed by an independently "
                "adjustable annular sampling band. Every pixel in that band contributes "
                "to the initial median Lab colour and multimodal background reference. "
                "When ruler scale is unavailable, the detected 48 mm dish radius supplies "
                "the metric fallback. If too little of the requested outer annulus is "
                "visible, an orange inside-rim fallback preserves the configured gap and "
                "band thickness."
            ),
            parameters={
                "perimeter_background_buffer_cm": 0.35,
                "perimeter_background_band_thickness_cm": 0.50,
            },
            parameter_specs=perimeter_background_parameters,
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
            1800,
            -100,
            details=(
                "The shared Lab/Scharr edge magnitude plus boundaries in sensor/noise, "
                "flattened grayscale, local-shadow, and local-highlight rasters are "
                "weighted on CUDA and convolved with a bank of normalized "
                "ring kernels. Radius and centre-spacing limits are "
                "editable fractions of the estimated seed diameter. "
                "Candidates outside the configured inset from the upper/outer dish rim are discarded. "
                "This experimental toolbox branch remains biased toward roughly round "
                "visible seed boundaries and is not part of the default DAG."
            ),
            parameters={
                "circle_accumulator_threshold": 22,
                "dense_circle_relaxation": 4,
                "dense_distance_candidate_threshold": 30,
                "circle_edge_threshold": 80,
                "circle_working_maximum_dimension": 1280,
                "circle_edge_magnitude_weight": 0.50,
                "circle_sensor_noise_weight": 0.20,
                "circle_flattened_grayscale_weight": 0.10,
                "circle_shadow_weight": 0.10,
                "circle_highlight_weight": 0.10,
                "circle_min_distance_fraction": 0.58,
                "circle_min_radius_fraction": 0.22,
                "circle_max_radius_fraction": 0.62,
                "merge_distance_fraction": 0.48,
                "circle_confidence": 0.62,
            },
            parameter_specs=circle_parameters,
            input_ports=(
                ("region", "Dish search region"),
                ("scale", "Seed diameter"),
                ("edge", "Edge magnitude"),
                ("noise", "Sensor/noise likelihood"),
                ("flattened", "Flattened grayscale"),
                ("shadow", "Local shadow likelihood"),
                ("highlight", "Local highlight likelihood"),
            ),
            output_ports=(("candidates", "Circle candidates"),),
        ),
        PipelineNode(
            "identification",
            "Seed identification",
            "Segmentation",
            "Consolidate active seed-centre candidates into review proposals",
            1920,
            -200,
            details=(
                "The default graph turns distance peaks directly into spatially sorted "
                "review proposals. If Circle candidates is restored from the unused-node "
                "toolbox and enabled, nearby CUDA ring candidates are merged using the "
                "circle node's editable distance and confidence, while separated rings "
                "remain additional proposals. Results are sorted top-to-bottom then "
                "left-to-right. These are untrained review proposals—not validated "
                "instance counts—and overlaps still "
                "require later mask correction."
            ),
            parameters={
                "distance_confidence": 0.48,
            },
            parameter_specs=fusion_parameters,
            bypassable=True,
        ),
        PipelineNode(
            "background_likelihood",
            "Material colour probabilities",
            "Segmentation",
            "Independent foreground, background, and Other colour evidence",
            1320,
            180,
            parameters={
                "background_colour_enabled": True,
                "background_sample_radius_fraction": 0.10,
                "background_chroma_percentile": 50.0,
                "background_lightness_percentile": 55.0,
                "background_minimum_sample_fraction": 0.002,
                "background_prior_tolerance": 24.0,
                "background_keep_perimeter_reference": True,
                "background_lightness_scale_floor": 8.0,
                "background_chroma_scale_floor": 3.0,
                "background_colour_components": 32,
                "background_distribution_fit_iterations": 6,
                "background_frequency_weight_power": 0.35,
                "background_distribution_scale_multiplier": 1.25,
                "background_automatic_evidence_floor": 0.10,
                "background_reviewed_authority_half_life_seed_areas": 0.50,
                "inner_radius_fraction": 0.90,
                "foreground_chroma_weight": 1.8,
                "foreground_reference_weight": 0.75,
                "foreground_include_annotated_seed_instances": True,
                "foreground_reference_components": 64,
                "foreground_frequency_weight_power": 0.0,
                "foreground_distribution_scale_multiplier": 1.50,
            },
            parameter_specs=(
                ParameterSpec(
                    "background_colour_enabled",
                    "Use background colour analysis",
                    "bool",
                    description=(
                        "Enable the background and Other colour model without disabling "
                        "the independent, mandatory-reference Foreground model on this "
                        "combined node."
                    ),
                ),
                *background_parameters,
                *foreground_parameters,
            ),
            details=(
                "Foreground and Background remain independent evidence calculations; "
                "combining their controls on one card does not normalize them into "
                "complements or otherwise change either fitted equation. User-painted "
                "Foreground pixels and optional safely inset applied seed interiors fit "
                "the foreground Lab frequency model. With no authored Foreground source, "
                "that output is zero. "
                "Painted areas and, by default, pixels within a fixed weighted-colour "
                "tolerance of the median colour in the buffered outer dish-perimeter band "
                "jointly fit a multimodal "
                "CIE Lab probability distribution with a separate centre and spread for each mode. "
                "The perimeter source can be switched off explicitly; without painted "
                "Background, that opt-out uses the independent light/low-chroma fallback. "
                "No self-refinement pixels are admitted after the reference fit. Background and foreground reference areas remain "
                "semantic constraints in the colour map and become direct positive/negative "
                "texture samples in the directional-noise maps; noise probabilities are never "
                "overwritten merely because a coordinate was painted. "
                "Painted Other areas instead fit separate competing colour and texture "
                "evidence, attenuating background only where Other fits better and never "
                "overwriting painted coordinates. If too little "
                "matching tray is visible inside a crowded dish, the outer perimeter "
                "measurement remains authoritative instead of admitting seed colours. The node "
                "overlay outlines the exact annulus used for that initial estimate. The node "
                "node-owned full-pane overlay plots fitted membership contours over an "
                "exact HSV hue/saturation slice at the toolbar-selected Value, including "
                "learned mode locations and reference frequencies. Its status "
                "reports the BGR range, selected area, and a warning when the final estimate "
                "deviates substantially from the perimeter prior."
            ),
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
            details=(
                "Applied seed-instance annotations enter here as distinct interior "
                "identity constraints. Their centroids are prepended to automatic "
                "candidate centres, and a nearby automatic duplicate is suppressed. "
                "The provisional masks then grow within the editable radial extent "
                "and foreground/background gate while retaining each painted interior. "
                "This node never forces foreground probability from the integer IDs. "
                "The separate Foreground option can reuse safely inset applied interiors "
                "as fitting examples without forcing their output values. The node remains "
                "disabled by default until "
                "its separation quality has been validated."
            ),
        ),
        PipelineNode(
            "refined_background_likelihood",
            "Material noise probabilities",
            "Diagnostic overlay",
            "Independent foreground, background, and Other directional texture evidence",
            1440,
            180,
            parameters={
                "background_noise_enabled": True,
                "foreground_noise_enabled": True,
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
                "foreground_noise_medium_scale_fraction": 0.03,
                "foreground_noise_coarse_scale_fraction": 0.08,
                "foreground_noise_direction_step_degrees": 15,
                "foreground_noise_vector_length_fraction": 0.55,
                "foreground_noise_vector_sample_count": 9,
                "foreground_noise_vector_decay": 0.86,
                "foreground_noise_direction_integration": "1st tertile",
                "foreground_noise_foreground_min_likelihood": 190,
                "foreground_noise_nonforeground_max_likelihood": 65,
                "foreground_noise_working_maximum_dimension": 1280,
            },
            parameter_specs=(
                ParameterSpec(
                    "background_noise_enabled",
                    "Use background noise analysis",
                    "bool",
                    description=(
                        "Enable the independent Background/Other directional-texture "
                        "calculation on this combined node."
                    ),
                ),
                ParameterSpec(
                    "foreground_noise_enabled",
                    "Use foreground noise analysis",
                    "bool",
                    description=(
                        "Enable the independent mandatory-reference Foreground "
                        "directional-texture calculation on this combined node."
                    ),
                ),
                *noise_parameters,
                *foreground_noise_parameters,
            ),
            details=(
                "This card combines controls and connectors without combining the "
                "underlying evidence equations. Fine, medium, and coarse Background "
                "texture distributions are learned from the retained exterior annulus "
                "or painted Background and their non-Background examples. Foreground "
                "texture distributions are learned independently from painted "
                "Foreground and optional safely inset annotated-seed interiors; no "
                "automatic foreground source is synthesized. Painted locations remain "
                "training evidence and are never forced output values. Both calculations "
                "retain their own enable switch, scales, ray geometry, thresholds, "
                "integration rule, GPU working limit, cached raster, and timing. "
                "Background defaults to "
                "maximum across one-sided rays; Foreground defaults to the exact first "
                "tertile so matching texture needs broad directional support. Other "
                "noise remains the competing texture evidence fitted beside Background."
            ),
        ),
        PipelineNode(
            "edge_gradients",
            "Edge gradients",
            "GPU diagnostic",
            "Configurable original/wavelet gradients plus both tangent encodings",
            1340,
            390,
            details=(
                "Choose Scharr, Sobel, Prewitt, or central-difference derivatives and "
                "select the corrected original, any signed wavelet detail levels, and/or "
                "the low-pass residual. Selected source vectors are fused by local "
                "maximum, RMS, or vector sum. The same cached field produces magnitude, "
                "undirected axial tangent, and directed polarity-aware tangent overlays; "
                "the former standalone tangent nodes have been merged here."
            ),
            parameters=dict(shared_edge_values),
            parameter_specs=edge_parameters,
            input_ports=(
                ("image", "Corrected image"),
                ("wavelets", "Wavelet decomposition"),
            ),
            output_ports=(
                ("magnitude", "Edge magnitude"),
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
            "surface_darkness_gradients",
            "Directional surface darkness gradients",
            "GPU diagnostic",
            "Find maximum one-sided lightening and darkening L* slopes",
            1440,
            1080,
            details=(
                "CIE L* is smoothed once, then sampled at several distances along "
                "independent one-sided rays. Every query pixel keeps the largest positive "
                "target-minus-query slope as its lightening result and the largest positive "
                "query-minus-target slope as its darkening result. Magnitudes are reported "
                "as L* change per original-image pixel; hue encodes the query-to-target "
                "direction over 0–360°. This measures gradual surface variation rather "
                "than only the local Scharr derivative used by Edge gradients."
            ),
            parameters={
                "surface_gradient_blur_sigma": 1.6,
                "surface_gradient_radius_fraction": 0.45,
                "surface_gradient_direction_step_degrees": 15,
                "surface_gradient_sample_count": 8,
                "surface_gradient_normalization_percentile": 99.0,
                "surface_gradient_strength_gamma": 0.60,
                "surface_gradient_working_maximum_dimension": 1280,
            },
            parameter_specs=surface_gradient_parameters,
            bypassable=True,
            input_ports=(
                ("image", "Corrected image"),
                ("scale", "Seed diameter"),
            ),
            output_ports=(
                ("lightening_magnitude", "Lightening magnitude"),
                ("lightening_direction", "Lightening direction"),
                ("darkening_magnitude", "Darkening magnitude"),
                ("darkening_direction", "Darkening direction"),
            ),
        ),
        PipelineNode(
            "lightening_gradient_ceiling",
            "Lightening derivative upper cutoff",
            "GPU diagnostic",
            "Suppress lightening slopes above an editable upper magnitude",
            1740,
            1060,
            details=(
                "The raw lightening slope and its query-to-target direction are reused from "
                "Directional surface darkness gradients. Pixels above the editable CIE L* "
                "slope ceiling are set exactly to zero so strong physical edges do not "
                "participate in this gradual-surface diagnostic."
            ),
            parameters={"lightening_gradient_maximum_slope": 1.0},
            parameter_specs=surface_gradient_ceiling_parameters,
            bypassable=True,
            input_ports=(
                ("magnitude", "Lightening magnitude"),
                ("direction", "Lightening direction"),
            ),
            output_ports=(
                ("magnitude", "Filtered lightening magnitude"),
                ("direction", "Filtered lightening direction"),
            ),
        ),
        PipelineNode(
            "darkening_gradient_ceiling",
            "Darkening derivative upper cutoff",
            "GPU diagnostic",
            "Suppress darkening slopes above an editable upper magnitude",
            1740,
            1240,
            details=(
                "The raw darkening slope and its query-to-target direction are reused from "
                "Directional surface darkness gradients. Pixels above the editable CIE L* "
                "slope ceiling are set exactly to zero so strong physical edges do not "
                "participate in this gradual-surface diagnostic."
            ),
            parameters={"darkening_gradient_maximum_slope": 1.0},
            parameter_specs=darkening_gradient_ceiling_parameters,
            bypassable=True,
            input_ports=(
                ("magnitude", "Darkening magnitude"),
                ("direction", "Darkening direction"),
            ),
            output_ports=(
                ("magnitude", "Filtered darkening magnitude"),
                ("direction", "Filtered darkening direction"),
            ),
        ),
        PipelineNode(
            "frequency_noise_masks",
            "Multiscale darkness and colour noise",
            "GPU diagnostic",
            "Measure surrounding darkness and Lab-colour energy in three bands",
            1440,
            1420,
            details=(
                "A seed-relative Gaussian pyramid separates fine, medium, and coarse "
                "frequencies. Within each band, surrounding RMS energy is calculated "
                "independently for CIE L* darkness variation and Lab a*/b* colour "
                "variation. The six masks are image-derived energies rather than learned "
                "class probabilities and remain separate for downstream seed separation."
            ),
            parameters={
                "frequency_noise_fine_scale_fraction": 0.008,
                "frequency_noise_medium_scale_fraction": 0.030,
                "frequency_noise_coarse_scale_fraction": 0.100,
                "frequency_noise_context_fraction": 0.025,
                "frequency_noise_normalization_percentile": 99.0,
                "frequency_noise_strength_gamma": 0.65,
                "frequency_noise_working_maximum_dimension": 1280,
            },
            parameter_specs=frequency_noise_mask_parameters,
            bypassable=True,
            input_ports=(
                ("image", "Corrected image"),
                ("scale", "Seed diameter"),
            ),
            output_ports=(
                ("darkness_fine", "Fine darkness noise"),
                ("darkness_medium", "Medium darkness noise"),
                ("darkness_coarse", "Coarse darkness noise"),
                ("colour_fine", "Fine colour noise"),
                ("colour_medium", "Medium colour noise"),
                ("colour_coarse", "Coarse colour noise"),
            ),
        ),
        PipelineNode(
            "edge_ridges",
            "Thinned edge ridges",
            "GPU diagnostic",
            "Non-maximum suppression and hysteresis on continuous gradients",
            1440,
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
            "reference_texture_prototypes",
            "Reference texture prototypes",
            "GPU diagnostic",
            "Learn material prototypes and instance-derived edge classes",
            1560,
            690,
            parameters={
                "reference_texture_material_prototypes_per_class": 64,
                "reference_texture_minimum_samples_per_prototype": 16,
                "reference_texture_fit_iterations": 4,
                "reference_texture_similarity_scale": 1.0,
                "reference_texture_context_fraction": 0.04,
                "reference_texture_patch_fraction": 0.28,
                "reference_texture_working_maximum_dimension": 960,
            },
            parameter_specs=reference_texture_parameters,
            details=(
                "Painted Background, Foreground and Other samples, plus automatically "
                "derived instance-contour and internal-edge samples, are represented by "
                "separate coverage-preserving banks rather than one class average. "
                "The optional annotated-seed Foreground source adds safely inset instance "
                "interiors to the Foreground material bank without adding contour pixels. "
                "Material banks use corrected Lab, six multiscale noise bands, "
                "local residuals, and edge/ridge density. Edge banks instead use "
                "a narrow tangent-aligned descriptor with separately pooled interior, "
                "edge-centre, and exterior regions plus signed cross-edge Lab contrast. "
                "Annotated instance geometry fixes inside/outside while training; both "
                "normal polarities are tested at unlabelled pixels. Their independently "
                "bounded adaptive resolution retains detail on small seeds. Farthest-first "
                "robust clustering retains up to 64 medoids for each material class and "
                "256 for each physical/non-physical edge class by default. The two "
                "capacities are independent, and prototype evaluation is GPU-batched. "
                "Every bank is evaluated across the full dish with the same equation at "
                "painted and unpainted pixels. Edge thumbnails are tangent-aligned in the "
                "full-pane prototype collage. Every complete annotated seed contributes "
                "its contour as physical-edge evidence. Edge candidates safely inset from "
                "that contour become non-physical examples; flat interior pixels are not "
                "mislabelled merely because they lie inside a seed."
            ),
            output_ports=(
                ("seed_surface", "Seed surface"),
                ("background_texture", "Background texture"),
                ("other_texture", "Other texture"),
                ("physical_probability", "Physical edge"),
                ("non_edge_probability", "Non-physical edge"),
                ("prototype_profile", "Prototype collage"),
            ),
            inline_parameters=(
                ("reference_texture_material_prototypes_per_class", "Material max"),
                ("reference_texture_similarity_scale", "Tolerance"),
            ),
        ),
        PipelineNode(
            "material_evidence_decision",
            "Material evidence decision",
            "Segmentation",
            "Calibrate Seed versus Non-seed with explicit ambiguity and unknown mass",
            1660,
            260,
            parameters={
                "material_colour_weight": 1.0,
                "material_noise_weight": 0.55,
                "material_prototype_weight": 0.70,
                "material_unknown_weight": 0.35,
                "material_decision_temperature": 1.0,
                "material_seed_threshold": 0.68,
                "material_morphology_fraction": 0.06,
            },
            parameter_specs=material_evidence_parameters,
            inline_parameters=(
                ("material_seed_threshold", "Seed threshold"),
                ("material_unknown_weight", "Unknown mass"),
                ("material_decision_temperature", "Temperature"),
            ),
            details=(
                "This is the authoritative material decision. Raw colour, directional "
                "texture, and valid multiclass prototype maps remain independent evidence "
                "and need not sum to one. The node first combines Foreground as Seed "
                "support and the union of Background and Other as Non-seed support. "
                "Each source contribution is calibrated from its reviewed target median "
                "against the high tail of reviewed top-level non-target responses; "
                "Background and Other are never calibrated against one another. "
                "It then normalizes four mutually exclusive masses—resolved Seed, "
                "resolved Non-seed, conflicting/ambiguous, and insufficient/unknown—"
                "which sum to one at every valid pixel. A second diagnostic decision "
                "shows Background versus Other conditional on Non-seed; overlap is "
                "allowed, so pale glass can remain subtype-ambiguous. Class annotations "
                "calibrate source reliability but never directly discount or overwrite a "
                "raw evidence map. The node owns the only user-facing binary seed-material mask; "
                "procedural separation consumes its resolved Seed probability directly."
            ),
            input_ports=(
                ("foreground_colour", "Foreground colour evidence"),
                ("foreground_noise", "Foreground texture evidence"),
                ("background_colour", "Background colour evidence"),
                ("background_noise", "Background texture evidence"),
                ("other_colour", "Other colour evidence"),
                ("other_noise", "Other texture evidence"),
                ("reference_foreground", "Reference Foreground prototypes"),
                ("reference_background", "Reference Background prototypes"),
                ("reference_other", "Reference Other prototypes"),
                ("valid", "Valid dish region"),
                ("scale", "Seed diameter"),
            ),
            output_ports=(
                ("seed_probability", "Resolved Seed probability"),
                ("nonseed_probability", "Resolved Non-seed probability"),
                ("ambiguity", "Conflicting evidence"),
                ("unknown", "Unknown evidence"),
                ("background_subtype", "Conditional Background subtype"),
                ("other_subtype", "Conditional Other subtype"),
                ("subtype_ambiguity", "Conditional subtype ambiguity"),
                ("subtype_unknown", "Conditional subtype unknown"),
                ("seed_mask", "Seed material proposal mask"),
            ),
        ),
        PipelineNode(
            "reference_edge_probability",
            "Instance-derived edge probabilities",
            "GPU diagnostic",
            "Classify physical contours versus non-physical internal edges",
            1860,
            690,
            parameters={
                "reference_texture_edge_prototypes_per_class": 256,
                "reference_edge_minimum_samples_per_prototype": 16,
                "reference_edge_fit_iterations": 4,
                "reference_edge_similarity_scale": 1.0,
                "reference_texture_edge_working_maximum_dimension": 2048,
                "reference_edge_minimum_working_seed_diameter_px": 28.0,
                "reference_edge_strip_normal_offset_fraction": 0.05,
                "reference_edge_strip_tangent_half_length_fraction": 0.08,
                "reference_edge_ridge_weight": 0.35,
                "reference_texture_instance_interior_buffer_fraction": 0.08,
                "net_physical_edge_internal_scale": 0.50,
            },
            parameter_specs=reference_edge_probability_parameters,
            details=(
                "The upstream reference-texture node fits and globally evaluates many "
                "tangent-normalized physical and non-physical edge prototypes derived "
                "from complete annotated seed instances. This node "
                "publishes those two typed probability rasters for boundary confirmation "
                "and procedural watershed. With no annotated instances, both semantic "
                "rasters remain neutral rather than pretending generic texture is physical. "
                "The diagnostic-only internal-edge subtraction control adjusts the displayed "
                "positive physical margin without changing either authored output raster."
            ),
            output_ports=(
                ("physical_probability", "Physical edge"),
                ("non_edge_probability", "Non-physical edge"),
            ),
            inline_parameters=(
                ("reference_texture_edge_prototypes_per_class", "Edge max"),
                ("reference_edge_similarity_scale", "Tolerance"),
                ("reference_edge_strip_normal_offset_fraction", "Strip offset"),
                ("net_physical_edge_internal_scale", "Internal subtraction"),
            ),
        ),
        PipelineNode(
            "reference_edge_ridges",
            "Thinned reference edge ridge",
            "GPU diagnostic",
            "Thin reference-trained physical-edge probability into precise ridges",
            2100,
            1440,
            parameters={
                "reference_ridge_nms_step_px": 1.0,
                "reference_ridge_low_threshold": 0.10,
                "reference_ridge_high_threshold": 0.24,
                "reference_ridge_hysteresis_iterations": 8,
                "reference_ridge_working_maximum_dimension": 1280,
                "reference_edge_normalization_radius_fraction": 0.30,
                "reference_edge_normalization_target_support": 0.35,
                "reference_edge_normalization_maximum_gain": 2.50,
                "reference_edge_normalization_absolute_floor": 0.04,
            },
            parameter_specs=reference_ridge_parameters,
            input_ports=(
                ("physical_probability", "Physical-edge probability"),
                ("non_edge_probability", "Non-physical edge probability"),
                ("normals", "Continuous edge normals"),
                ("scale", "Seed diameter"),
            ),
            output_ports=(
                ("reference_ridges", "Reference ridge"),
                ("net_reference_ridges", "Net-physical ridge"),
                ("normalized_net_probability", "Locally normalized net edge"),
                ("normalized_net_ridges", "Normalized net ridge"),
            ),
            inline_parameters=(
                ("reference_ridge_low_threshold", "Low"),
                ("reference_ridge_high_threshold", "High"),
                ("reference_edge_normalization_radius_fraction", "Local radius"),
                ("reference_edge_normalization_maximum_gain", "Max gain"),
            ),
            details=(
                "The broad reference-trained physical-edge probability is sampled on "
                "both sides of the continuous image-gradient normal. A deterministic "
                "non-maximum suppression leaves local probability peaks, then CUDA "
                "high/low hysteresis retains connected weak peaks. Confidence remains "
                "continuous, and painted pixels are evidence rather than forced output. "
                "A second output applies the edge-probability node's scaled internal-edge "
                "subtraction before the same thinning operation. The locally normalized "
                "output separates that semantic margin from shared absolute support, "
                "equalizes support against a bounded seed-scale winsorized RMS envelope, "
                "and applies an absolute smooth floor before a third thinning pass. Raw "
                "probability and raw-net diagnostics remain unchanged."
            ),
        ),
        PipelineNode(
            "edge_traces",
            "Oriented edge traces",
            "GPU diagnostic",
            "Link tangent/curvature-compatible ridges, bridge gaps and split junctions",
            2280,
            430,
            parameters={
                "trace_edge_source": "generic_ridges",
                "trace_diameter_multiplier": 1.0,
                "trace_tangent_tolerance_degrees": 24.0,
                "trace_maximum_gap_px": 2,
                "trace_curvature_policy": "prefer",
                "trace_curvature_tolerance_degrees": 6.0,
                "trace_window_fraction": 0.20,
                "trace_sample_count": 7,
                "trace_minimum_length_fraction": 0.18,
                "trace_junction_max_neighbors": 4,
            },
            parameter_specs=trace_parameters,
            inline_parameters=(
                ("trace_edge_source", "Ridge source"),
                ("trace_diameter_multiplier", "Trace diameter"),
                ("trace_maximum_gap_px", "Gap px"),
                ("trace_curvature_policy", "Convexity"),
            ),
            output_ports=(
                ("trace_labels", "Trace identities"),
                ("continuity", "Trace continuity"),
                ("gap_confidence", "Gap confidence"),
            ),
            output_port_types={
                "trace_labels": "OrientedTraceLabels",
                "continuity": "TraceContinuity",
                "gap_confidence": "TraceGapConfidence",
            },
            details=(
                "The editable Ridge source selects generic, physical-reference, net, or "
                "locally normalized net ridges; the shared continuous image-gradient field "
                "still supplies subpixel tangents. A GPU union/find graph joins selected "
                "ridge pixels only when endpoint tangents "
                "and their displacement agree. Optional winding-independent convexity "
                "gating rejects S-shaped candidate gap links, with tolerant preference "
                "and stricter local-requirement modes. Configurable short gaps are bridged, "
                "over-connected junction pixels are split, and tangent-following samples "
                "measure continuity and missing-edge support on both sides. The curvature "
                "test constrains each initial bridge; global convexity is confirmed later "
                "by the circle/ellipse boundary model."
            ),
        ),
        PipelineNode(
            "seed_edge_curves",
            "Seed-boundary confirmation",
            "Diagnostic overlay",
            "Trace, radius, circle, ellipse, centre-vote and semantic confirmation",
            2520,
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
                "boundary_instance_edge_influence": 0.35,
                "boundary_nonphysical_edge_discount": 0.80,
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
            "Smooth the resolved Seed material probability at seed scale",
            1680, 180,
            details="CUDA PyTorch applies seed-relative Gaussian smoothing to the authoritative resolved Seed mass. Colour, texture, material prototypes, Background, Other, ambiguity, and unknown evidence are reconciled once in Material evidence decision and are not recombined or double-counted here.",
            parameters={"compute_device": "cuda", "allow_cpu_fallback": False, "maximum_dimension": 1024, "interior_smoothing_fraction": 0.055},
            parameter_specs=seed_interior_parameters,
        ),
        PipelineNode(
            "boundary_normals", "Boundary confidence and normals", "GPU diagnostic",
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
            2280, 270,
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
            output_ports=(
                ("confidence", "Assignment confidence"),
                ("contested_pixels", "Contested pixels"),
            ),
            output_port_types={
                "confidence": "AssignmentConfidence",
                "contested_pixels": "ContestedPixels",
            },
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
            "illumination_decomposition", "Grayscale and local lighting", "GPU diagnostic",
            "Flatten grayscale and identify locally unusual shadow/highlight areas",
            1440, 760,
            details=(
                "A valid-mask-aware, seed-scaled Gaussian field estimates local lighting. "
                "A clipped log grayscale/lighting ratio produces simple flattened grayscale. "
                "The residual is standardized by its local absolute deviation and passed "
                "through separate nonlinear sigmoid thresholds for shadow and highlight "
                "areas. Broad illumination, reflectance, and specular risk remain available."
            ),
            parameters={
                "illumination_scale_fraction": 0.55,
                "flattening_contrast_gain": 1.80,
                "lighting_deviation_scale_fraction": 0.10,
                "shadow_z_threshold": 0.75,
                "highlight_z_threshold": 0.75,
                "lighting_extreme_softness": 0.30,
            },
            parameter_specs=illumination_parameters,
            input_ports=(
                ("image", "Corrected image"),
                ("scale", "Seed diameter"),
            ),
            output_ports=(
                ("illumination", "Local illumination"),
                ("flattened", "Flattened grayscale"),
                ("shadow", "Local shadow likelihood"),
                ("highlight", "Local highlight likelihood"),
                ("reflectance", "Reflectance"),
                ("glare", "Glare likelihood"),
            ),
            output_port_types={
                "illumination": "IlluminationField",
                "flattened": "FlattenedGrayscale",
                "shadow": "ShadowLikelihood",
                "highlight": "HighlightLikelihood",
                "reflectance": "Reflectance",
                "glare": "GlareLikelihood",
            },
        ),
        PipelineNode(
            "image_quality", "Image-quality diagnostics", "GPU diagnostic",
            "Map focus, clipping, underexposure, glare, and local noise risk",
            1740, 760,
            details="The composite risk map is backed by separately viewable focus, clipped-highlight, underexposure, and sensor-noise maps.",
            parameters={"quality_noise_scale_fraction": 0.025},
            parameter_specs=image_quality_parameters,
            output_ports=(
                ("quality_risk", "Quality risk"),
                ("focus", "Focus quality"),
                ("clipped_highlights", "Clipped highlights"),
                ("underexposure", "Underexposure"),
                ("sensor_noise", "Sensor/noise likelihood"),
            ),
            output_port_types={
                "quality_risk": "ImageQualityRisk",
                "focus": "FocusQuality",
                "clipped_highlights": "ClippedHighlights",
                "underexposure": "Underexposure",
                "sensor_noise": "SensorNoiseLikelihood",
            },
        ),
        PipelineNode(
            "procedural_instances",
            "Procedural seed separation",
            "Segmentation",
            "Fuse material, physical-boundary and seed-scale evidence into reviewable instances",
            2340,
            820,
            details=(
                "The upstream hierarchical material decision supplies one resolved Seed "
                "probability that already accounts for Foreground, Background, Other, "
                "prototype, ambiguity, and unknown evidence. Seed-sized "
                "enclosed coat-pattern holes are filled, but "
                "the detected dish margin is excluded. Generic Lab edges and thinned ridges "
                "form boundary candidates; instance-derived physical-minus-non-physical "
                "evidence gates the entire candidate, while thinned physical ridges and "
                "long, continuous, seed-scale convex traces add geometric support. Internal "
                "coat texture is therefore suppressed instead of being counted repeatedly. "
                "Seed-scale blurred illumination-flattened grayscale, material geometry, "
                "and boundary-interior depth propose centres; "
                "distinct painted seed-instance IDs replace nearby automatic markers. A "
                "bounded CPU marker-controlled watershed evaluates several material-boundary "
                "hypotheses per centre. Hard area, width, concavity, protrusion, solidity, and "
                "axis-ratio limits reject impossible candidates; soft minimum-area and maximum-width "
                "limits lower scores without abrupt removal. An overlap-aware score ordering then "
                "selects a non-overlapping candidate combination. Each shape remains a marker-controlled "
                "watershed basin clipped by one material hypothesis—not a fitted ellipse. "
                "This remains a review aid: unannotated edge types use neutral semantic "
                "support, so low-confidence results must not be treated as validated counts."
            ),
            parameters={
                "working_maximum_dimension": 1600,
                "foreground_threshold_scale": 0.82,
                "occupancy_closing_fraction": 0.12,
                "occupancy_hole_area_fraction": 1.25,
                "dish_margin_fraction": 0.12,
                "boundary_edge_weight": 0.35,
                "boundary_ridge_weight": 0.65,
                "boundary_semantic_floor": 0.08,
                "boundary_nonphysical_discount": 0.95,
                "boundary_physical_ridge_weight": 0.55,
                "boundary_trace_weight": 0.45,
                "boundary_surface_darkening_weight": 0.30,
                "trace_minimum_length_fraction": 0.22,
                "trace_convexity_weight": 0.60,
                "centre_geometry_smoothing_fraction": 0.14,
                "centre_material_weight": 0.20,
                "centre_distance_weight": 0.20,
                "centre_flattened_grayscale_weight": 0.60,
                "centre_minimum_separation_fraction": 0.42,
                "sparse_centre_minimum_separation_fraction": 0.58,
                "marker_count_multiplier": 1.02,
                "minimum_marker_score": 0.12,
                "minimum_instance_area_fraction": 0.26,
                "soft_minimum_instance_area_fraction": 0.39,
                "maximum_instance_area_fraction": 1.45,
                "soft_maximum_instance_width_fraction": 1.15,
                "hard_maximum_instance_width_fraction": 1.35,
                "maximum_internal_concavity_fraction": 0.15,
                "maximum_protrusion_area_fraction": 0.10,
                "minimum_instance_solidity": 0.62,
                "maximum_instance_axis_ratio": 2.80,
                "candidate_hypotheses_per_marker": 5,
                "candidate_overlap_fraction": 0.02,
            },
            parameter_specs=procedural_instance_parameters,
            inline_parameters=(
                ("foreground_threshold_scale", "Material threshold"),
                ("centre_minimum_separation_fraction", "Marker spacing"),
                ("minimum_marker_score", "Marker minimum"),
            ),
            input_ports=(
                ("region", "Dish region"),
                ("scale", "Seed diameter"),
                ("material_probability", "Resolved Seed material probability"),
                ("edge_magnitude", "Edge magnitude"),
                ("ridges", "Thinned ridges"),
                ("physical_reference", "Physical-edge probability"),
                ("non_edge_reference", "Non-physical edge probability"),
                ("reference_ridges", "Thinned physical ridges"),
                ("trace_labels", "Oriented traces"),
                ("trace_continuity", "Trace continuity"),
                ("surface_darkening", "Surface darkening"),
                ("flattened_grayscale", "Flattened grayscale"),
                ("annotations", "Annotated seed instances"),
                ("manual_centres", "Manual centres"),
            ),
            output_ports=(
                ("material_likelihood", "Material likelihood"),
                ("material_mask", "Material mask"),
                ("boundary_cost", "Boundary cost"),
                ("centre_likelihood", "Centre likelihood"),
                ("instances", "Seed instances"),
                ("confidence", "Instance confidence"),
                ("concavity", "Internal concavity"),
                ("alternatives", "Alternative candidates"),
            ),
            output_port_types={
                "material_likelihood": "MaterialLikelihood",
                "material_mask": "MaterialMask",
                "boundary_cost": "BoundaryCost",
                "centre_likelihood": "CentreLikelihood",
                "instances": "SeedInstances",
                "confidence": "InstanceConfidence",
                "concavity": "InstanceConcavity",
                "alternatives": "InstanceCandidates",
            },
        ),
        PipelineNode(
            "unet_instances",
            "U-Net + watershed instances",
            "Learned segmentation",
            "Distinguish physical contacts from non-physical coat-pattern boundaries",
            2100,
            1060,
            details=(
                "A native PyTorch residual U-Net predicts seed interior, physical seed "
                "boundaries, apparent non-physical coat-pattern boundaries, centre/depth "
                "evidence, and calibrated auxiliary error probabilities. Marker-controlled watershed discounts "
                "predicted coat-pattern edges while following physical contacts. Applied "
                "instance annotations remain authoritative markers. Model logits are cached "
                "separately, so decoder-only setting changes do not repeat inference. This "
                "node is disabled until a compatible reviewed-data checkpoint exists."
            ),
            enabled=False,
            bypassable=True,
            status=NodeStatus.BYPASSED,
            status_detail="Disabled; compatible trained checkpoint required",
            parameters={
                "checkpoint_path": "models/unet_seed_instances.pt",
                "tile_size": 512,
                "tile_overlap": 96,
                "interior_threshold": 0.50,
                "centre_threshold": 0.30,
                "centre_minimum_separation_fraction": 0.32,
                "minimum_instance_area_fraction": 0.15,
                "physical_boundary_weight": 0.72,
                "distance_topography_weight": 0.28,
                "pattern_boundary_discount": 0.80,
                "uncertainty_penalty": 0.20,
                "foreground_erosion_fraction": 0.0,
            },
            parameter_specs=unet_instance_parameters,
            input_ports=(
                ("image", "Corrected image/evidence"),
                ("scale", "Seed diameter"),
                ("annotations", "Painted IDs"),
            ),
            output_ports=(
                ("interior", "Interior probability"),
                ("instances", "Instance identities"),
                ("physical", "Physical boundaries"),
                ("pattern", "Pattern boundaries"),
                ("centre", "Centre probability"),
                ("distance", "Distance prediction"),
                ("uncertainty", "Uncertainty"),
                ("confidence", "Instance confidence"),
            ),
            output_port_types={
                "interior": "InteriorProbability",
                "instances": "SeedInstances",
                "physical": "PhysicalEdgeProbability",
                "pattern": "PatternBoundaryProbability",
                "centre": "CentreProbability",
                "distance": "DistancePrediction",
                "uncertainty": "ModelUncertainty",
                "confidence": "InstanceConfidence",
            },
            inline_parameters=(
                ("interior_threshold", "Interior minimum"),
                ("centre_threshold", "Centre minimum"),
                ("pattern_boundary_discount", "Pattern discount"),
            ),
        ),
        PipelineNode(
            "stardist_instances",
            "StarDist seed instances",
            "Learned segmentation",
            "Predict one scored star-convex polygon for each visible seed",
            2400,
            1060,
            details=(
                "A native PyTorch StarDist model predicts object probability and radial "
                "distance along a checkpointed ray bank. Local maxima form star-convex "
                "seed polygons and overlap-aware non-maximum suppression removes duplicate "
                "proposals. The strong object-shape prior is intended to ignore internal "
                "coat transitions but may require U-Net boundary refinement for concave or "
                "occluded seeds. This node is disabled until a compatible checkpoint exists."
            ),
            enabled=False,
            bypassable=True,
            status=NodeStatus.BYPASSED,
            status_detail="Disabled; compatible trained checkpoint required",
            parameters={
                "checkpoint_path": "models/stardist_seed_instances.pt",
                "tile_size": 512,
                "tile_overlap": 96,
                "object_threshold": 0.45,
                "nms_iou_threshold": 0.35,
                "local_maximum_radius_fraction": 0.08,
                "minimum_instance_area_fraction": 0.15,
                "maximum_candidates": 4096,
            },
            parameter_specs=stardist_instance_parameters,
            input_ports=(
                ("image", "Corrected image/evidence"),
                ("scale", "Seed diameter"),
            ),
            output_ports=(
                ("instances", "Instance identities"),
                ("object", "Object probability"),
                ("uncertainty", "Radial uncertainty"),
                ("confidence", "Instance confidence"),
            ),
            output_port_types={
                "instances": "SeedInstances",
                "object": "ObjectProbability",
                "uncertainty": "ModelUncertainty",
                "confidence": "InstanceConfidence",
            },
            inline_parameters=(
                ("object_threshold", "Object minimum"),
                ("nms_iou_threshold", "NMS overlap"),
            ),
        ),
        PipelineNode(
            "radial_profile", "Per-seed radial profiles", "GPU diagnostic",
            "Compare each assigned pixel with expected centre-to-edge lightness",
            2400, 540,
            details="Pixels are normalized by their assigned proposal radius and accumulated into a configurable global radial profile. The residual highlights unusual edge darkening or internal structure.",
            parameters={"radial_bin_count": 24},
            parameter_specs=radial_parameters,
            output_ports=(
                ("residual", "Radial residual"),
                ("coordinate", "Radial coordinate"),
            ),
            output_port_types={
                "residual": "RadialResidual",
                "coordinate": "RadialCoordinate",
            },
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
            output_ports=(
                ("probabilities", "Pattern probabilities"),
                ("confidence", "Pattern confidence"),
            ),
            output_port_types={
                "probabilities": "PatternProbabilities",
                "confidence": "PatternConfidence",
            },
        ),
        PipelineNode(
            "colour_probabilities", "Broad colour probabilities", "GPU trait analysis",
            "Estimate white, yellow, green, red, brown, and black pixel probabilities",
            2400, 900,
            details="Colour-balanced RGB is compared with broad prototype colours using a softmax. The winning class, uncertainty, and every class probability are viewable independently.",
            parameters={"colour_temperature": 18.0},
            parameter_specs=colour_probability_parameters,
            output_ports=(
                ("probabilities", "Colour probabilities"),
                ("uncertainty", "Colour uncertainty"),
            ),
            output_port_types={
                "probabilities": "ColourProbabilities",
                "uncertainty": "ColourUncertainty",
            },
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
            "classification", "Coat and condition", "Analysis", "Colour, pattern, wrinkling and damage", 3360, -90,
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
            ("expected_outer_diameter_mm", "Outer diameter mm"),
            ("rim_pair_expected_separation_fraction", "Expected rim gap"),
        ),
        "scale_calibration": (("minor_tick_mm", "Minor tick mm"),),
        "seed_scale_estimation": (
            ("reference_scale_factor", "Reference scale"),
            ("annotated_seed_top_fraction", "Annotated top fraction"),
            ("fallback_diameter_fraction", "Fallback diameter"),
        ),
        "perimeter_background_reference": (
            ("perimeter_background_buffer_cm", "Buffer cm"),
            ("perimeter_background_band_thickness_cm", "Band cm"),
        ),
        "distance_candidates": (
            ("distance_neighborhood_fraction", "Peak spacing"),
            ("distance_min_depth_fraction", "Peak depth"),
        ),
        "circle_candidates": (
            ("circle_accumulator_threshold", "Ring support"),
            ("circle_min_radius_fraction", "Radius min"),
            ("circle_max_radius_fraction", "Radius max"),
            ("circle_confidence", "Circle weight"),
        ),
        "identification": (
            ("distance_confidence", "Distance weight"),
        ),
        "background_likelihood": (
            ("background_colour_enabled", "Use background colour"),
            ("background_colour_components", "Background modes"),
            ("foreground_reference_components", "Foreground modes"),
            ("foreground_reference_weight", "Foreground influence"),
        ),
        "instance_masks": (
            ("instance_min_extent_fraction", "Extent min"),
            ("instance_max_extent_fraction", "Extent max"),
            ("instance_radius_extent_multiplier", "Radius extent"),
        ),
        "refined_background_likelihood": (
            ("background_noise_enabled", "Use BG noise"),
            ("foreground_noise_enabled", "Use FG noise"),
            ("noise_direction_integration", "BG integration"),
            ("foreground_noise_direction_integration", "FG integration"),
        ),
        "edge_gradients": (
            ("edge_blur_sigma", "Blur sigma"),
            ("edge_chroma_weight", "Chroma weight"),
            ("edge_normalization_percentile", "Normalize %"),
            ("edge_strength_gamma", "Gamma"),
        ),
        "surface_darkness_gradients": (
            ("surface_gradient_radius_fraction", "Ray length"),
            ("surface_gradient_direction_step_degrees", "Direction step"),
            ("surface_gradient_sample_count", "Ray samples"),
            ("surface_gradient_working_maximum_dimension", "GPU dimension"),
        ),
        "lightening_gradient_ceiling": (
            ("lightening_gradient_maximum_slope", "Maximum slope"),
        ),
        "darkening_gradient_ceiling": (
            ("darkening_gradient_maximum_slope", "Maximum slope"),
        ),
        "frequency_noise_masks": (
            ("frequency_noise_fine_scale_fraction", "Fine scale"),
            ("frequency_noise_medium_scale_fraction", "Medium scale"),
            ("frequency_noise_coarse_scale_fraction", "Coarse scale"),
            ("frequency_noise_context_fraction", "RMS context"),
        ),
        "edge_ridges": (
            ("ridge_nms_step_px", "NMS step"),
            ("ridge_low_threshold", "Low"),
            ("ridge_high_threshold", "High"),
            ("ridge_hysteresis_iterations", "Reach"),
        ),
        "edge_traces": (
            ("trace_edge_source", "Ridge source"),
            ("trace_tangent_tolerance_degrees", "Angle +/-"),
            ("trace_maximum_gap_px", "Gap px"),
            ("trace_curvature_policy", "Convexity"),
        ),
        "seed_edge_curves": (
            ("curve_diameter_multiplier", "Diameter x"),
            ("boundary_radius_min_fraction", "Radius min"),
            ("boundary_radius_max_fraction", "Radius max"),
            ("boundary_minimum_confidence", "Accept min"),
        ),
        "seed_interior": (
            ("maximum_dimension", "GPU dimension"),
            ("interior_smoothing_fraction", "Smoothing"),
        ),
        "boundary_normals": (("boundary_width_fraction", "Boundary width"),),
        "touching_split": (("split_neck_fraction", "Neck depth"),),
        "ellipse_likelihood": (("ellipse_radial_tolerance", "Radial tolerance"),),
        "proposal_disagreement": (("disagreement_scale_fraction", "Evidence spread"),),
        "assignment_confidence": (("assignment_boundary_penalty", "Boundary penalty"),),
        "contact_graph": (("contact_distance_multiplier", "Contact distance"),),
        "illumination_decomposition": (
            ("illumination_scale_fraction", "Field scale"),
            ("flattening_contrast_gain", "Flatten gain"),
            ("shadow_z_threshold", "Shadow z"),
            ("highlight_z_threshold", "Highlight z"),
        ),
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
        PipelineConnection(
            "metadata", "deskew_colour", "TaggedImages", target_port="image"
        ),
        PipelineConnection("colour_reference", "deskew_colour", "SwatchGrid"),
        PipelineConnection("deskew_colour", "ruler_detection", "CorrectedImage"),
        PipelineConnection("deskew_colour", "layout_detection", "CorrectedImage"),
        PipelineConnection("deskew_colour", "hue_only", "CorrectedImage"),
        PipelineConnection("layout_detection", "hue_only", "DishRegion"),
        PipelineConnection("deskew_colour", "wavelet_decomposition", "CorrectedImage"),
        PipelineConnection("layout_detection", "wavelet_decomposition", "DishRegion"),
        PipelineConnection(
            "wavelet_decomposition",
            "edge_gradients",
            "WaveletPyramid",
            source_port="pyramid",
            target_port="wavelets",
        ),
        PipelineConnection("deskew_colour", "scale_calibration", "CorrectedImage"),
        PipelineConnection("ruler_detection", "scale_calibration", "RulerAxis"),
        PipelineConnection("scale_calibration", "layout_detection", "PixelsPerMillimetre"),
        PipelineConnection("deskew_colour", "seed_scale_estimation", "CorrectedImage"),
        PipelineConnection("layout_detection", "seed_scale_estimation", "VesselGeometry"),
        PipelineConnection(
            "reference_layers",
            "seed_scale_estimation",
            "PaintedInstanceAnnotations",
            source_port="annotated_seeds",
            target_port="annotations",
        ),
        PipelineConnection(
            "deskew_colour", "perimeter_background_reference", "CorrectedImage"
        ),
        PipelineConnection(
            "layout_detection", "perimeter_background_reference", "VesselGeometry"
        ),
        PipelineConnection(
            "scale_calibration",
            "perimeter_background_reference",
            "PixelsPerMillimetre",
        ),
        PipelineConnection("material_evidence_decision", "distance_candidates", "ForegroundMask", source_port="seed_mask"),
        PipelineConnection("seed_scale_estimation", "distance_candidates", "SeedDiameter"),
        PipelineConnection(
            "deskew_colour", "circle_candidates", "CorrectedImage",
            target_port="image",
        ),
        PipelineConnection(
            "layout_detection", "circle_candidates", "VesselGeometry",
            source_port="vessel_geometry", target_port="region",
        ),
        PipelineConnection(
            "distance_candidates", "circle_candidates", "DistancePeaks",
            target_port="distance_candidates",
        ),
        PipelineConnection(
            "seed_scale_estimation", "circle_candidates", "SeedDiameter",
            target_port="scale",
        ),
        PipelineConnection(
            "edge_gradients", "circle_candidates", "EdgeMagnitude",
            source_port="magnitude", target_port="edge",
        ),
        PipelineConnection(
            "image_quality", "circle_candidates", "SensorNoiseLikelihood",
            source_port="sensor_noise", target_port="noise",
        ),
        PipelineConnection(
            "illumination_decomposition", "circle_candidates", "FlattenedGrayscale",
            source_port="flattened", target_port="flattened",
        ),
        PipelineConnection(
            "illumination_decomposition", "circle_candidates", "ShadowLikelihood",
            source_port="shadow", target_port="shadow",
        ),
        PipelineConnection(
            "illumination_decomposition", "circle_candidates", "HighlightLikelihood",
            source_port="highlight", target_port="highlight",
        ),
        PipelineConnection("distance_candidates", "identification", "DistancePeaks"),
        PipelineConnection("circle_candidates", "identification", "CudaRings"),
        PipelineConnection(
            "seed_scale_estimation", "identification", "SeedDiameter",
            source_port="seed_diameter", target_port="scale",
        ),
        PipelineConnection(
            "deskew_colour", "background_likelihood", "CorrectedImage"
        ),
        PipelineConnection(
            "layout_detection", "background_likelihood", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection(
            "seed_scale_estimation", "background_likelihood", "SeedDiameter"
        ),
        PipelineConnection(
            "scale_calibration",
            "background_likelihood",
            "PixelsPerMillimetre",
            target_port="absolute_scale",
        ),
        PipelineConnection(
            "perimeter_background_reference",
            "background_likelihood",
            "PerimeterColourSamples",
        ),
        PipelineConnection("reference_layers", "background_likelihood", "BackgroundReferences", source_port="background", target_port="background_reference"),
        PipelineConnection("reference_layers", "background_likelihood", "ForegroundReferences", source_port="foreground", target_port="foreground_reference"),
        PipelineConnection("reference_layers", "background_likelihood", "OtherReferences", source_port="other", target_port="other_reference"),
        PipelineConnection(
            "reference_layers",
            "background_likelihood",
            "PaintedInstanceAnnotations",
            source_port="annotated_seeds",
            target_port="annotations",
        ),
        PipelineConnection("identification", "instance_masks", "SeedProposals"),
        PipelineConnection(
            "layout_detection", "instance_masks", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection(
            "material_evidence_decision", "instance_masks", "NonseedProbability",
            source_port="nonseed_probability", target_port="background_probability",
        ),
        PipelineConnection(
            "background_likelihood",
            "refined_background_likelihood",
            "BackgroundProbability",
            source_port="background_probability",
            target_port="background_probability",
        ),
        PipelineConnection(
            "deskew_colour", "refined_background_likelihood", "CorrectedImage",
            target_port="image",
        ),
        PipelineConnection(
            "layout_detection", "refined_background_likelihood", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection(
            "seed_scale_estimation",
            "refined_background_likelihood",
            "SeedDiameter",
        ),
        PipelineConnection(
            "perimeter_background_reference",
            "refined_background_likelihood",
            "PerimeterColourSamples",
            target_port="perimeter_band",
        ),
        PipelineConnection("reference_layers", "refined_background_likelihood", "BackgroundReferences", source_port="background", target_port="background_reference"),
        PipelineConnection("reference_layers", "refined_background_likelihood", "ForegroundReferences", source_port="foreground", target_port="foreground_reference"),
        PipelineConnection("reference_layers", "refined_background_likelihood", "OtherReferences", source_port="other", target_port="other_reference"),
        PipelineConnection(
            "background_likelihood",
            "refined_background_likelihood",
            "ForegroundProbability",
            source_port="foreground_probability",
            target_port="foreground_probability",
        ),
        PipelineConnection(
            "background_likelihood",
            "refined_background_likelihood",
            "AnnotatedForegroundReferences",
            source_port="annotated_foreground_reference_source",
            target_port="automatic_foreground_reference",
        ),
        PipelineConnection(
            "deskew_colour", "edge_gradients", "CorrectedImage", target_port="image"
        ),
        PipelineConnection(
            "layout_detection", "edge_gradients", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection(
            "deskew_colour",
            "surface_darkness_gradients",
            "CorrectedImage",
            target_port="image",
        ),
        PipelineConnection(
            "layout_detection",
            "surface_darkness_gradients",
            "DishRegion",
            source_port="dish_region",
            target_port="region",
        ),
        PipelineConnection(
            "seed_scale_estimation",
            "surface_darkness_gradients",
            "SeedDiameter",
            target_port="scale",
        ),
        PipelineConnection(
            "surface_darkness_gradients",
            "lightening_gradient_ceiling",
            "LighteningMagnitude",
            source_port="lightening_magnitude",
            target_port="magnitude",
        ),
        PipelineConnection(
            "surface_darkness_gradients",
            "lightening_gradient_ceiling",
            "LighteningDirection",
            source_port="lightening_direction",
            target_port="direction",
        ),
        PipelineConnection(
            "surface_darkness_gradients",
            "darkening_gradient_ceiling",
            "DarkeningMagnitude",
            source_port="darkening_magnitude",
            target_port="magnitude",
        ),
        PipelineConnection(
            "surface_darkness_gradients",
            "darkening_gradient_ceiling",
            "DarkeningDirection",
            source_port="darkening_direction",
            target_port="direction",
        ),
        PipelineConnection(
            "deskew_colour",
            "frequency_noise_masks",
            "CorrectedImage",
            target_port="image",
        ),
        PipelineConnection(
            "layout_detection",
            "frequency_noise_masks",
            "DishRegion",
            source_port="dish_region",
            target_port="region",
        ),
        PipelineConnection(
            "seed_scale_estimation",
            "frequency_noise_masks",
            "SeedDiameter",
            target_port="scale",
        ),
        PipelineConnection("edge_gradients", "edge_ridges", "FloatGradientField"),
        PipelineConnection("deskew_colour", "reference_texture_prototypes", "CorrectedImage", target_port="image"),
        PipelineConnection(
            "layout_detection", "reference_texture_prototypes", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection("seed_scale_estimation", "reference_texture_prototypes", "SeedDiameter", target_port="scale"),
        PipelineConnection("reference_layers", "reference_texture_prototypes", "BackgroundReferences", source_port="background", target_port="background_reference"),
        PipelineConnection("reference_layers", "reference_texture_prototypes", "ForegroundReferences", source_port="foreground", target_port="foreground_reference"),
        PipelineConnection("reference_layers", "reference_texture_prototypes", "OtherReferences", source_port="other", target_port="other_reference"),
        PipelineConnection(
            "background_likelihood",
            "reference_texture_prototypes",
            "AutomaticBackgroundReferences",
            source_port="background_reference_source",
            target_port="automatic_background_reference",
        ),
        PipelineConnection(
            "background_likelihood",
            "reference_texture_prototypes",
            "AnnotatedForegroundReferences",
            source_port="annotated_foreground_reference_source",
            target_port="automatic_foreground_reference",
        ),
        PipelineConnection("reference_layers", "reference_texture_prototypes", "PaintedInstanceAnnotations", source_port="annotated_seeds", target_port="annotations"),
        PipelineConnection("edge_gradients", "reference_texture_prototypes", "EdgeMagnitude", source_port="magnitude", target_port="edge"),
        PipelineConnection("edge_gradients", "reference_texture_prototypes", "AxialTangents", source_port="undirected", target_port="undirected"),
        PipelineConnection("edge_gradients", "reference_texture_prototypes", "DirectedTangents", source_port="directed", target_port="directed"),
        PipelineConnection("edge_ridges", "reference_texture_prototypes", "ThinnedRidges", target_port="ridges"),
        PipelineConnection("frequency_noise_masks", "reference_texture_prototypes", "FineDarknessNoise", source_port="darkness_fine", target_port="darkness_fine"),
        PipelineConnection("frequency_noise_masks", "reference_texture_prototypes", "MediumDarknessNoise", source_port="darkness_medium", target_port="darkness_medium"),
        PipelineConnection("frequency_noise_masks", "reference_texture_prototypes", "CoarseDarknessNoise", source_port="darkness_coarse", target_port="darkness_coarse"),
        PipelineConnection("frequency_noise_masks", "reference_texture_prototypes", "FineColourNoise", source_port="colour_fine", target_port="colour_fine"),
        PipelineConnection("frequency_noise_masks", "reference_texture_prototypes", "MediumColourNoise", source_port="colour_medium", target_port="colour_medium"),
        PipelineConnection("frequency_noise_masks", "reference_texture_prototypes", "CoarseColourNoise", source_port="colour_coarse", target_port="colour_coarse"),
        PipelineConnection("background_likelihood", "material_evidence_decision", "ForegroundColourProbability", source_port="foreground_probability", target_port="foreground_colour"),
        PipelineConnection("refined_background_likelihood", "material_evidence_decision", "ForegroundNoiseProbability", source_port="foreground_noise_probability", target_port="foreground_noise"),
        PipelineConnection("background_likelihood", "material_evidence_decision", "BackgroundProbability", source_port="background_probability", target_port="background_colour"),
        PipelineConnection("refined_background_likelihood", "material_evidence_decision", "BackgroundNoiseProbability", source_port="background_noise_probability", target_port="background_noise"),
        PipelineConnection("background_likelihood", "material_evidence_decision", "OtherColourProbability", source_port="other_colour_probability", target_port="other_colour"),
        PipelineConnection("refined_background_likelihood", "material_evidence_decision", "OtherNoiseProbability", source_port="other_noise_probability", target_port="other_noise"),
        PipelineConnection("reference_texture_prototypes", "material_evidence_decision", "ReferenceSeedSurface", source_port="seed_surface", target_port="reference_foreground"),
        PipelineConnection("reference_texture_prototypes", "material_evidence_decision", "ReferenceBackgroundMaterial", source_port="background_texture", target_port="reference_background"),
        PipelineConnection("reference_texture_prototypes", "material_evidence_decision", "ReferenceOtherMaterial", source_port="other_texture", target_port="reference_other"),
        PipelineConnection("background_likelihood", "material_evidence_decision", "ForegroundProposalRegion", source_port="proposal_region", target_port="valid"),
        PipelineConnection("seed_scale_estimation", "material_evidence_decision", "SeedDiameter", source_port="seed_diameter", target_port="scale"),
        PipelineConnection("reference_texture_prototypes", "reference_edge_probability", "PhysicalPrototypeProbability", source_port="physical_probability", target_port="physical_probability"),
        PipelineConnection("reference_texture_prototypes", "reference_edge_probability", "NonEdgePrototypeProbability", source_port="non_edge_probability", target_port="non_edge_probability"),
        PipelineConnection("reference_edge_probability", "reference_edge_ridges", "PhysicalEdgeProbability", source_port="physical_probability", target_port="physical_probability"),
        PipelineConnection("reference_edge_probability", "reference_edge_ridges", "NonEdgeProbability", source_port="non_edge_probability", target_port="non_edge_probability"),
        PipelineConnection("edge_gradients", "reference_edge_ridges", "ContinuousEdgeNormals", source_port="directed", target_port="normals"),
        PipelineConnection("seed_scale_estimation", "reference_edge_ridges", "SeedDiameter", source_port="seed_diameter", target_port="scale"),
        PipelineConnection("edge_ridges", "edge_traces", "ThinnedRidges"),
        PipelineConnection("reference_edge_ridges", "edge_traces", "ReferenceRidges", source_port="reference_ridges", target_port="reference_ridges"),
        PipelineConnection("reference_edge_ridges", "edge_traces", "NetReferenceRidges", source_port="net_reference_ridges", target_port="net_reference_ridges"),
        PipelineConnection("reference_edge_ridges", "edge_traces", "NormalizedNetReferenceRidges", source_port="normalized_net_ridges", target_port="normalized_net_reference_ridges"),
        PipelineConnection("edge_gradients", "edge_traces", "ContinuousTangents"),
        PipelineConnection(
            "seed_scale_estimation", "edge_traces", "SeedDiameter",
            source_port="seed_diameter", target_port="scale",
        ),
        PipelineConnection("edge_traces", "seed_edge_curves", "OrientedTraceLabels", source_port="trace_labels", target_port="traces"),
        PipelineConnection("edge_gradients", "seed_edge_curves", "FloatGradientField"),
        PipelineConnection(
            "seed_scale_estimation", "seed_edge_curves", "SeedDiameter",
            source_port="seed_diameter", target_port="scale",
        ),
        PipelineConnection(
            "material_evidence_decision", "seed_edge_curves", "NonseedProbability",
            source_port="nonseed_probability", target_port="background_probability",
        ),
        PipelineConnection(
            "material_evidence_decision", "seed_edge_curves", "ForegroundProbability",
            source_port="seed_probability",
        ),
        PipelineConnection("reference_edge_probability", "seed_edge_curves", "PhysicalEdgeProbability", source_port="physical_probability", target_port="physical_reference"),
        PipelineConnection("reference_edge_probability", "seed_edge_curves", "NonEdgeProbability", source_port="non_edge_probability", target_port="non_edge_reference"),
        PipelineConnection("instance_masks", "seed_edge_curves", "ProvisionalInstances"),
        PipelineConnection(
            "identification", "seed_edge_curves", "SeedProposals",
            target_port="proposals",
        ),
        PipelineConnection(
            "seed_scale_estimation", "instance_masks", "SeedDiameter",
            source_port="seed_diameter", target_port="scale",
        ),
        PipelineConnection(
            "material_evidence_decision", "seed_interior", "ForegroundProbability",
            source_port="seed_probability", target_port="foreground_probability",
        ),
        PipelineConnection(
            "layout_detection", "seed_interior", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection(
            "seed_scale_estimation", "seed_interior", "SeedDiameter",
            source_port="seed_diameter", target_port="scale",
        ),
        PipelineConnection("seed_interior", "boundary_normals", "InteriorProbability"),
        PipelineConnection(
            "deskew_colour", "boundary_normals", "CorrectedImage",
            target_port="image",
        ),
        PipelineConnection(
            "layout_detection", "boundary_normals", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection(
            "seed_scale_estimation", "boundary_normals", "SeedDiameter",
            source_port="seed_diameter", target_port="scale",
        ),
        PipelineConnection("boundary_normals", "touching_split", "BoundaryConfidence"),
        PipelineConnection("distance_candidates", "touching_split", "DistanceTransform"),
        PipelineConnection(
            "material_evidence_decision", "touching_split", "ForegroundMask",
            source_port="seed_mask", target_port="foreground_mask",
        ),
        PipelineConnection(
            "layout_detection", "touching_split", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection(
            "seed_scale_estimation", "touching_split", "SeedDiameter",
            source_port="seed_diameter", target_port="scale",
        ),
        PipelineConnection(
            "boundary_normals", "ellipse_likelihood", "BoundaryConfidence",
            target_port="boundary_confidence",
        ),
        PipelineConnection("seed_scale_estimation", "ellipse_likelihood", "SeedDiameter"),
        PipelineConnection(
            "deskew_colour", "ellipse_likelihood", "CorrectedImage",
            target_port="image",
        ),
        PipelineConnection(
            "layout_detection", "ellipse_likelihood", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection(
            "identification", "ellipse_likelihood", "SeedProposals",
            target_port="proposals",
        ),
        PipelineConnection(
            "instance_masks", "ellipse_likelihood", "ProvisionalInstances",
            target_port="instances",
        ),
        PipelineConnection("circle_candidates", "proposal_disagreement", "CircleEvidence"),
        PipelineConnection("distance_candidates", "proposal_disagreement", "DistanceEvidence"),
        PipelineConnection("ellipse_likelihood", "proposal_disagreement", "EllipseEvidence"),
        PipelineConnection(
            "seed_interior", "proposal_disagreement", "InteriorProbability",
            target_port="interior",
        ),
        PipelineConnection(
            "seed_scale_estimation", "proposal_disagreement", "SeedDiameter",
            source_port="seed_diameter", target_port="scale",
        ),
        PipelineConnection(
            "layout_detection", "proposal_disagreement", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection("instance_masks", "assignment_confidence", "ProvisionalInstances"),
        PipelineConnection("boundary_normals", "assignment_confidence", "BoundaryConfidence"),
        PipelineConnection(
            "seed_interior", "assignment_confidence", "InteriorProbability",
            target_port="interior",
        ),
        PipelineConnection(
            "material_evidence_decision", "assignment_confidence", "ForegroundMask",
            source_port="seed_mask", target_port="foreground_mask",
        ),
        PipelineConnection(
            "layout_detection", "assignment_confidence", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection(
            "identification", "assignment_confidence", "SeedProposals",
            target_port="proposals",
        ),
        PipelineConnection(
            "assignment_confidence", "contact_graph", "ContestedPixels",
            source_port="contested_pixels", target_port="contested_pixels",
        ),
        PipelineConnection("identification", "contact_graph", "SeedProposals"),
        PipelineConnection(
            "boundary_normals", "contact_graph", "BoundaryConfidence",
            target_port="boundary_confidence",
        ),
        PipelineConnection(
            "deskew_colour", "illumination_decomposition", "CorrectedImage",
            target_port="image",
        ),
        PipelineConnection(
            "seed_scale_estimation", "illumination_decomposition", "SeedDiameter",
            target_port="scale",
        ),
        PipelineConnection(
            "layout_detection", "illumination_decomposition", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection(
            "deskew_colour", "image_quality", "CorrectedImage",
            target_port="image",
        ),
        PipelineConnection(
            "layout_detection", "image_quality", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection(
            "seed_scale_estimation", "image_quality", "SeedDiameter",
            source_port="seed_diameter", target_port="scale",
        ),
        PipelineConnection("layout_detection", "procedural_instances", "DishRegion"),
        PipelineConnection("seed_scale_estimation", "procedural_instances", "SeedDiameter"),
        PipelineConnection("material_evidence_decision", "procedural_instances", "MaterialLikelihood", source_port="seed_probability", target_port="material_probability"),
        PipelineConnection("edge_gradients", "procedural_instances", "EdgeMagnitude"),
        PipelineConnection("edge_ridges", "procedural_instances", "ThinnedRidges"),
        PipelineConnection("reference_edge_probability", "procedural_instances", "PhysicalEdgeProbability", source_port="physical_probability", target_port="physical_reference"),
        PipelineConnection("reference_edge_probability", "procedural_instances", "NonEdgeProbability", source_port="non_edge_probability", target_port="non_edge_reference"),
        PipelineConnection("reference_edge_ridges", "procedural_instances", "NormalizedNetPhysicalProbability", source_port="normalized_net_probability", target_port="normalized_net_probability"),
        PipelineConnection("reference_edge_ridges", "procedural_instances", "ThinnedNormalizedNetReferenceRidges", source_port="normalized_net_ridges", target_port="reference_ridges"),
        PipelineConnection("edge_traces", "procedural_instances", "OrientedTraceLabels", source_port="trace_labels", target_port="trace_labels"),
        PipelineConnection("edge_traces", "procedural_instances", "TraceContinuity", source_port="continuity", target_port="trace_continuity"),
        PipelineConnection("surface_darkness_gradients", "procedural_instances", "DarkeningMagnitude", source_port="darkening_magnitude", target_port="surface_darkening"),
        PipelineConnection("illumination_decomposition", "procedural_instances", "FlattenedGrayscale", source_port="flattened", target_port="flattened_grayscale"),
        PipelineConnection("reference_layers", "procedural_instances", "PaintedInstanceAnnotations", source_port="annotated_seeds", target_port="annotations"),
        PipelineConnection("reference_layers", "procedural_instances", "ManualSeedCentres", source_port="centres", target_port="manual_centres"),
        PipelineConnection("metadata", "unet_instances", "SpeciesCondition", target_port="image"),
        PipelineConnection("deskew_colour", "unet_instances", "CorrectedImage", target_port="image"),
        PipelineConnection("layout_detection", "unet_instances", "DishRegion", target_port="image"),
        PipelineConnection("seed_scale_estimation", "unet_instances", "SeedDiameter", target_port="scale"),
        PipelineConnection("background_likelihood", "unet_instances", "ForegroundProbability", source_port="foreground_probability", target_port="foreground_probability"),
        PipelineConnection("refined_background_likelihood", "unet_instances", "ForegroundNoiseProbability", source_port="foreground_noise_probability", target_port="foreground_noise"),
        PipelineConnection("background_likelihood", "unet_instances", "BackgroundProbability", source_port="background_probability", target_port="background_probability"),
        PipelineConnection("refined_background_likelihood", "unet_instances", "BackgroundNoiseProbability", source_port="background_noise_probability", target_port="background_noise"),
        PipelineConnection("edge_gradients", "unet_instances", "EdgeMagnitude", target_port="image"),
        PipelineConnection("reference_edge_probability", "unet_instances", "PhysicalEdgeProbability", source_port="physical_probability", target_port="physical_reference"),
        PipelineConnection("reference_edge_probability", "unet_instances", "NonEdgeProbability", source_port="non_edge_probability", target_port="non_edge_reference"),
        PipelineConnection("illumination_decomposition", "unet_instances", "FlattenedGrayscale", source_port="flattened", target_port="flattened"),
        PipelineConnection("illumination_decomposition", "unet_instances", "ShadowLikelihood", source_port="shadow", target_port="shadow"),
        PipelineConnection("illumination_decomposition", "unet_instances", "HighlightLikelihood", source_port="highlight", target_port="highlight"),
        PipelineConnection("image_quality", "unet_instances", "SensorNoiseLikelihood", source_port="sensor_noise", target_port="sensor_noise"),
        PipelineConnection("reference_layers", "unet_instances", "PaintedInstanceAnnotations", source_port="annotated_seeds", target_port="annotations"),
        PipelineConnection("metadata", "stardist_instances", "SpeciesCondition", target_port="image"),
        PipelineConnection("deskew_colour", "stardist_instances", "CorrectedImage", target_port="image"),
        PipelineConnection("layout_detection", "stardist_instances", "DishRegion", target_port="image"),
        PipelineConnection("seed_scale_estimation", "stardist_instances", "SeedDiameter", target_port="scale"),
        PipelineConnection("background_likelihood", "stardist_instances", "ForegroundProbability", source_port="foreground_probability", target_port="foreground_probability"),
        PipelineConnection("refined_background_likelihood", "stardist_instances", "ForegroundNoiseProbability", source_port="foreground_noise_probability", target_port="foreground_noise"),
        PipelineConnection("background_likelihood", "stardist_instances", "BackgroundProbability", source_port="background_probability", target_port="background_probability"),
        PipelineConnection("refined_background_likelihood", "stardist_instances", "BackgroundNoiseProbability", source_port="background_noise_probability", target_port="background_noise"),
        PipelineConnection("edge_gradients", "stardist_instances", "EdgeMagnitude", target_port="image"),
        PipelineConnection("reference_edge_probability", "stardist_instances", "PhysicalEdgeProbability", source_port="physical_probability", target_port="physical_reference"),
        PipelineConnection("reference_edge_probability", "stardist_instances", "NonEdgeProbability", source_port="non_edge_probability", target_port="non_edge_reference"),
        PipelineConnection("illumination_decomposition", "stardist_instances", "FlattenedGrayscale", source_port="flattened", target_port="flattened"),
        PipelineConnection("illumination_decomposition", "stardist_instances", "ShadowLikelihood", source_port="shadow", target_port="shadow"),
        PipelineConnection("illumination_decomposition", "stardist_instances", "HighlightLikelihood", source_port="highlight", target_port="highlight"),
        PipelineConnection("image_quality", "stardist_instances", "SensorNoiseLikelihood", source_port="sensor_noise", target_port="sensor_noise"),
        PipelineConnection("reference_layers", "instance_masks", "PaintedInstanceAnnotations", source_port="annotated_seeds", target_port="annotations"),
        PipelineConnection("instance_masks", "radial_profile", "ProvisionalInstances"),
        PipelineConnection(
            "identification", "radial_profile", "SeedProposals",
            target_port="proposals",
        ),
        PipelineConnection(
            "deskew_colour", "radial_profile", "CorrectedImage",
            target_port="image",
        ),
        PipelineConnection(
            "layout_detection", "radial_profile", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection("seed_interior", "wrinkling", "InteriorProbability"),
        PipelineConnection(
            "boundary_normals", "wrinkling", "BoundaryConfidence",
            target_port="boundary_confidence",
        ),
        PipelineConnection(
            "deskew_colour", "wrinkling", "CorrectedImage",
            target_port="image",
        ),
        PipelineConnection(
            "layout_detection", "wrinkling", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection(
            "seed_scale_estimation", "wrinkling", "SeedDiameter",
            source_port="seed_diameter", target_port="scale",
        ),
        PipelineConnection(
            "radial_profile", "coat_damage", "RadialResidual",
            source_port="residual",
        ),
        PipelineConnection("boundary_normals", "coat_damage", "BoundaryConfidence"),
        PipelineConnection("seed_interior", "coat_damage", "InteriorProbability"),
        PipelineConnection(
            "deskew_colour", "coat_damage", "CorrectedImage",
            target_port="image",
        ),
        PipelineConnection(
            "layout_detection", "coat_damage", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection(
            "seed_scale_estimation", "coat_damage", "SeedDiameter",
            source_port="seed_diameter", target_port="scale",
        ),
        PipelineConnection("seed_interior", "pattern_decomposition", "InteriorProbability"),
        PipelineConnection(
            "deskew_colour", "pattern_decomposition", "CorrectedImage",
            target_port="image",
        ),
        PipelineConnection(
            "layout_detection", "pattern_decomposition", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection(
            "seed_scale_estimation", "pattern_decomposition", "SeedDiameter",
            source_port="seed_diameter", target_port="scale",
        ),
        PipelineConnection("seed_interior", "colour_probabilities", "InteriorProbability"),
        PipelineConnection(
            "deskew_colour", "colour_probabilities", "CorrectedImage",
            target_port="image",
        ),
        PipelineConnection(
            "layout_detection", "colour_probabilities", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection("colour_reference", "calibration_residuals", "ReferenceConfidence"),
        PipelineConnection("ruler_detection", "calibration_residuals", "ReferenceConfidence"),
        PipelineConnection("deskew_colour", "calibration_residuals", "CalibrationTransform"),
        PipelineConnection(
            "layout_detection", "calibration_residuals", "DishRegion",
            source_port="dish_region", target_port="region",
        ),
        PipelineConnection("touching_split", "review", "SplitSuggestions"),
        PipelineConnection("proposal_disagreement", "review", "UncertainRegions"),
        PipelineConnection(
            "assignment_confidence", "review", "AssignmentConfidence",
            source_port="confidence",
        ),
        PipelineConnection("contact_graph", "review", "ContactGraph"),
        PipelineConnection("instance_masks", "review", "InstanceProposals"),
        PipelineConnection("review", "measurements", "ReviewedMasks"),
        PipelineConnection("scale_calibration", "measurements", "PixelsPerMillimetre"),
        PipelineConnection("review", "classification", "ReviewedMasks"),
        PipelineConnection("wrinkling", "classification", "WrinkleEvidence"),
        PipelineConnection("coat_damage", "classification", "DamageEvidence"),
        PipelineConnection("pattern_decomposition", "classification", "PatternProbabilities", source_port="probabilities"),
        PipelineConnection("colour_probabilities", "classification", "ColourProbabilities", source_port="probabilities"),
        PipelineConnection("calibration_residuals", "classification", "CalibrationRisk"),
        PipelineConnection("measurements", "aggregation", "MeasurementTable"),
        PipelineConnection("classification", "aggregation", "TraitTable"),
        PipelineConnection("aggregation", "output", "LotSummary"),
    )
    graph = PipelineGraph(nodes, connections)
    disabled_roots = (
        "distance_candidates",
        "circle_candidates",
        "instance_masks",
        "seed_interior",
        "calibration_residuals",
    )
    disabled = set(disabled_roots)
    for root in disabled_roots:
        disabled.update(graph.downstream(root, recursive=True))
    for node_id in disabled:
        node = graph.node(node_id)
        node.enabled = False
        node.bypassable = True
        node.status = NodeStatus.BYPASSED
        node.status_detail = "Disabled by default while this branch is under review"
    for node_id in ("lightening_gradient_ceiling", "darkening_gradient_ceiling"):
        node = graph.node(node_id)
        node.enabled = False
        node.bypassable = True
        node.status = NodeStatus.BYPASSED
        node.status_detail = "Optional thresholded surface-gradient diagnostic"
    # Preserve experimental or unfinished branches and all of their authored
    # wiring without making them part of the default executable graph. The node
    # editor exposes every removed node through its unused-node toolbox.
    graph.shelve_node("circle_candidates", record_revision=False)
    graph.shelve_node("lightening_gradient_ceiling", record_revision=False)
    graph.shelve_node("darkening_gradient_ceiling", record_revision=False)
    distance_branch = (
        "distance_candidates",
        *graph.downstream("distance_candidates", recursive=True),
    )
    for node_id in distance_branch:
        graph.shelve_node(node_id, record_revision=False)
    graph.shelve_node("calibration_residuals", record_revision=False)
    return graph
