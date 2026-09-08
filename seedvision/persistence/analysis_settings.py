"""Portable, human-readable persistence for analytical pipeline settings.

The profile deliberately contains only values that can change an analysis:
node membership in the active graph, enabled state, authored parameters, and
whether each authored connection is connected.  Runtime progress, timings,
image-local annotations, cached products, and canvas presentation state belong
to other persistence layers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence, TypeAlias

from seedvision.pipeline import NodeStatus, ParameterSpec, PipelineConnection, PipelineGraph


ANALYSIS_SETTINGS_FORMAT = "seedfiddle-analysis-settings"
ANALYSIS_SETTINGS_VERSION = 19
_LEGACY_ANALYSIS_SETTINGS_VERSIONS = frozenset(
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
)
_PRE_COLOUR_MERGE_VERSIONS = frozenset({1, 2, 3, 4})
_PRE_NOISE_MERGE_VERSIONS = frozenset({1, 2, 3, 4, 5})
ANALYSIS_SETTINGS_FILE_SUFFIX = ".seedfiddle-settings.json"

_MAX_PROFILE_BYTES = 8 * 1024 * 1024
_MAX_NODES = 4096
_MAX_CONNECTIONS = 65536
_MAX_PARAMETERS_PER_NODE = 4096
_MAX_IDENTIFIER_CHARACTERS = 256
_MAX_TEXT_CHARACTERS = 16384

JsonScalar: TypeAlias = bool | int | float | str
ConnectionKey: TypeAlias = tuple[str, str, str, str]


class AnalysisSettingsError(ValueError):
    """Base error for a settings profile that cannot be used safely."""


class AnalysisSettingsIOError(AnalysisSettingsError):
    """A profile could not be read or atomically written."""


class InvalidAnalysisSettingsProfile(AnalysisSettingsError):
    """The file or embedded payload violates the versioned JSON schema."""


class IncompatibleAnalysisSettingsProfile(AnalysisSettingsError):
    """A valid profile does not match the current authored pipeline catalogue."""


@dataclass(frozen=True, slots=True)
class AnalysisNodeSettings:
    """Immutable analytical settings for one stable pipeline node identifier."""

    identifier: str
    active: bool
    enabled: bool
    connection_suspended: bool
    parameters: tuple[tuple[str, JsonScalar], ...]


@dataclass(frozen=True, slots=True)
class AnalysisConnectionSettings:
    """Connected/disconnected state for one authored endpoint pair."""

    source: str
    source_port: str
    target: str
    target_port: str
    connected: bool

    @property
    def key(self) -> ConnectionKey:
        return (self.source, self.source_port, self.target, self.target_port)


@dataclass(frozen=True, slots=True)
class AnalysisSettingsProfile:
    """A schema-validated but not yet graph-compatible settings profile."""

    format: str
    version: int
    nodes: tuple[AnalysisNodeSettings, ...]
    connections: tuple[AnalysisConnectionSettings, ...]


@dataclass(frozen=True, slots=True)
class AnalysisSettingsApplyResult:
    """Summary of one atomic profile application."""

    changed: bool
    affected_node_ids: tuple[str, ...]
    structure_changed: bool
    analytical_changed: bool


@dataclass(frozen=True, slots=True)
class _PreparedAnalysisSettings:
    profile: AnalysisSettingsProfile
    catalogue_order: tuple[str, ...]
    active_by_id: Mapping[str, bool]
    enabled_by_id: Mapping[str, bool]
    connection_suspended_by_id: Mapping[str, bool]
    parameters_by_id: Mapping[str, Mapping[str, JsonScalar]]
    connected_by_key: Mapping[ConnectionKey, bool]
    template_by_key: Mapping[ConnectionKey, PipelineConnection]


def analysis_settings_profile_from_graph(
    graph: PipelineGraph,
) -> AnalysisSettingsProfile:
    """Capture the complete analytical configuration of ``graph``.

    Every authored connection is recorded, including those preserved behind a
    toolbox node.  A false ``connected`` value is therefore distinguishable
    from a true connection whose endpoint is merely inactive.
    """

    catalogue = _graph_catalogue(graph)
    connected = set(graph.connections) | set(graph.unused_connections)
    captured_nodes: list[AnalysisNodeSettings] = []
    for node_id in sorted(catalogue):
        node = catalogue[node_id]
        captured_nodes.append(
            AnalysisNodeSettings(
                identifier=node_id,
                active=node_id in graph.nodes,
                enabled=bool(node.enabled),
                connection_suspended=node_id in graph._connection_suspended,
                parameters=tuple(
                    (spec.key, _json_scalar(node.parameters[spec.key], spec.key))
                    for spec in node.parameter_specs
                ),
            )
        )
    nodes = tuple(captured_nodes)
    connections = tuple(
        AnalysisConnectionSettings(
            source=template.source,
            source_port=template.source_port,
            target=template.target,
            target_port=template.target_port,
            connected=template in connected,
        )
        for template in graph.connection_templates
    )
    profile = AnalysisSettingsProfile(
        format=ANALYSIS_SETTINGS_FORMAT,
        version=ANALYSIS_SETTINGS_VERSION,
        nodes=nodes,
        connections=connections,
    )
    # Run the same structural validation as embedded or disk-sourced payloads.
    canonical = analysis_settings_profile_from_payload(
        _analysis_settings_profile_to_unchecked_payload(profile)
    )
    _prepare_analysis_settings(graph, canonical)
    return canonical


def analysis_settings_profile_to_payload(
    profile: AnalysisSettingsProfile,
) -> dict[str, object]:
    """Return a detached JSON-compatible payload for a profile or project file."""

    if not isinstance(profile, AnalysisSettingsProfile):
        raise TypeError("Expected an AnalysisSettingsProfile.")
    canonical = analysis_settings_profile_from_payload(
        _analysis_settings_profile_to_unchecked_payload(profile)
    )
    return _analysis_settings_profile_to_unchecked_payload(canonical)


def analysis_settings_profile_from_payload(
    payload: Mapping[str, object],
) -> AnalysisSettingsProfile:
    """Validate and detach one embedded versioned profile payload.

    This checks the file schema without needing a pipeline.  Compatibility
    with the current node/parameter/connection catalogue is intentionally a
    second validation pass performed before application.
    """

    root = _mapping(payload, "analysis settings profile")
    _require_exact_keys(root, {"format", "version", "nodes", "connections"}, "profile")
    format_name = _bounded_text(root["format"], "profile format")
    if format_name != ANALYSIS_SETTINGS_FORMAT:
        raise InvalidAnalysisSettingsProfile(
            f"Expected profile format {ANALYSIS_SETTINGS_FORMAT!r}, got "
            f"{format_name!r}."
        )
    version = root["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise InvalidAnalysisSettingsProfile("Profile version must be an integer.")
    if version not in {*_LEGACY_ANALYSIS_SETTINGS_VERSIONS, ANALYSIS_SETTINGS_VERSION}:
        raise IncompatibleAnalysisSettingsProfile(
            f"Unsupported analysis-settings version {version}; this application "
            f"supports versions 1 through {ANALYSIS_SETTINGS_VERSION}."
        )

    node_items = _sequence(root["nodes"], "profile nodes", _MAX_NODES)
    nodes: list[AnalysisNodeSettings] = []
    node_ids: set[str] = set()
    for index, item in enumerate(node_items):
        entry = _mapping(item, f"node record {index}")
        _require_exact_keys(
            entry,
            {
                "id",
                "active",
                "enabled",
                "connection_suspended",
                "parameters",
            },
            f"node record {index}",
        )
        node_id = _identifier(entry["id"], f"node record {index} id")
        if node_id in node_ids:
            raise InvalidAnalysisSettingsProfile(
                f"Node {node_id!r} appears more than once in the profile."
            )
        node_ids.add(node_id)
        active = _boolean(entry["active"], f"node {node_id!r} active")
        enabled = _boolean(entry["enabled"], f"node {node_id!r} enabled")
        connection_suspended = _boolean(
            entry["connection_suspended"],
            f"node {node_id!r} connection suspended",
        )
        parameter_payload = _mapping(
            entry["parameters"], f"parameters for node {node_id!r}"
        )
        if len(parameter_payload) > _MAX_PARAMETERS_PER_NODE:
            raise InvalidAnalysisSettingsProfile(
                f"Node {node_id!r} has too many parameters."
            )
        parameters: list[tuple[str, JsonScalar]] = []
        for raw_key, raw_value in parameter_payload.items():
            key = _identifier(raw_key, f"parameter name on node {node_id!r}")
            parameters.append((key, _json_scalar(raw_value, f"{node_id}.{key}")))
        nodes.append(
            AnalysisNodeSettings(
                identifier=node_id,
                active=active,
                enabled=enabled,
                connection_suspended=connection_suspended,
                parameters=tuple(parameters),
            )
        )

    connection_items = _sequence(
        root["connections"], "profile connections", _MAX_CONNECTIONS
    )
    connections: list[AnalysisConnectionSettings] = []
    connection_keys: set[ConnectionKey] = set()
    for index, item in enumerate(connection_items):
        entry = _mapping(item, f"connection record {index}")
        _require_exact_keys(
            entry,
            {"source", "source_port", "target", "target_port", "connected"},
            f"connection record {index}",
        )
        connection = AnalysisConnectionSettings(
            source=_identifier(entry["source"], f"connection {index} source"),
            source_port=_identifier(
                entry["source_port"], f"connection {index} source port"
            ),
            target=_identifier(entry["target"], f"connection {index} target"),
            target_port=_identifier(
                entry["target_port"], f"connection {index} target port"
            ),
            connected=_boolean(
                entry["connected"], f"connection record {index} connected"
            ),
        )
        if connection.key in connection_keys:
            raise InvalidAnalysisSettingsProfile(
                f"Connection {_format_connection_key(connection.key)} appears more "
                "than once in the profile."
            )
        connection_keys.add(connection.key)
        connections.append(connection)

    return AnalysisSettingsProfile(
        format=format_name,
        version=version,
        nodes=tuple(nodes),
        connections=tuple(connections),
    )


def save_analysis_settings_profile(
    path: Path | str,
    graph_or_profile: PipelineGraph | AnalysisSettingsProfile,
) -> Path:
    """Atomically write an indented UTF-8 JSON settings profile."""

    if isinstance(graph_or_profile, PipelineGraph):
        profile = analysis_settings_profile_from_graph(graph_or_profile)
    elif isinstance(graph_or_profile, AnalysisSettingsProfile):
        profile = graph_or_profile
    else:
        raise TypeError("Expected a PipelineGraph or AnalysisSettingsProfile.")
    payload = analysis_settings_profile_to_payload(profile)
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as error:
        raise InvalidAnalysisSettingsProfile(
            f"Analysis settings could not be encoded as JSON: {error}"
        ) from error
    destination = Path(path)
    descriptor: int | None = None
    temporary: Path | None = None
    operation_error: OSError | None = None
    cleanup_error: OSError | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        descriptor = None
        with stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except OSError as error:
        operation_error = error
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as error:
                cleanup_error = error
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError as error:
                cleanup_error = cleanup_error or error
    if operation_error is not None:
        detail = (
            f"; temporary-file cleanup also failed: {cleanup_error}"
            if cleanup_error is not None
            else ""
        )
        raise AnalysisSettingsIOError(
            f"Could not atomically save analysis settings to {destination}: "
            f"{operation_error}{detail}"
        ) from operation_error
    if cleanup_error is not None:
        raise AnalysisSettingsIOError(
            f"Analysis settings were saved to {destination}, but temporary-file "
            f"cleanup failed: {cleanup_error}"
        ) from cleanup_error
    return destination


def load_analysis_settings_profile(path: Path | str) -> AnalysisSettingsProfile:
    """Read and schema-validate an analysis settings JSON file."""

    source = Path(path)
    try:
        size = source.stat().st_size
        if size > _MAX_PROFILE_BYTES:
            raise InvalidAnalysisSettingsProfile(
                f"Analysis settings file is larger than {_MAX_PROFILE_BYTES} bytes."
            )
        encoded = source.read_text(encoding="utf-8")
    except InvalidAnalysisSettingsProfile:
        raise
    except (OSError, UnicodeError) as error:
        raise AnalysisSettingsIOError(
            f"Could not read analysis settings from {source}: {error}"
        ) from error
    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_non_finite_json_constant,
        )
    except InvalidAnalysisSettingsProfile:
        raise
    except json.JSONDecodeError as error:
        raise InvalidAnalysisSettingsProfile(
            f"Invalid JSON in analysis settings file {source}: {error.msg} "
            f"(line {error.lineno}, column {error.colno})."
        ) from error
    except RecursionError as error:
        raise InvalidAnalysisSettingsProfile(
            f"Analysis settings file {source} is nested too deeply."
        ) from error
    return analysis_settings_profile_from_payload(payload)


def apply_analysis_settings_profile(
    graph: PipelineGraph,
    profile: AnalysisSettingsProfile,
) -> AnalysisSettingsApplyResult:
    """Validate completely, then apply a profile as one graph transaction."""

    prepared = _prepare_analysis_settings(graph, profile)
    catalogue = _graph_catalogue(graph)
    old_active = set(graph.nodes)
    desired_active = {
        node_id
        for node_id, active in prepared.active_by_id.items()
        if active
    }
    old_connected = set(graph.connections) | set(graph.unused_connections)
    desired_connected = {
        prepared.template_by_key[key]
        for key, connected in prepared.connected_by_key.items()
        if connected
    }

    parameter_changes: dict[str, set[str]] = {}
    enabled_changes: set[str] = set()
    for node_id in prepared.catalogue_order:
        node = catalogue[node_id]
        changed_keys = {
            key
            for key, value in prepared.parameters_by_id[node_id].items()
            if node.parameters.get(key) != value
        }
        if changed_keys:
            parameter_changes[node_id] = changed_keys
        if node.enabled != prepared.enabled_by_id[node_id]:
            enabled_changes.add(node_id)

    old_suspended_nodes = set(graph._connection_suspended)
    desired_suspended_nodes = {
        node_id
        for node_id, suspended in prepared.connection_suspended_by_id.items()
        if suspended
    }
    suspended_changes = old_suspended_nodes ^ desired_suspended_nodes

    active_changes = old_active ^ desired_active
    connection_changes = old_connected ^ desired_connected
    structure_changed = bool(
        active_changes or connection_changes or suspended_changes
    )
    changed = bool(parameter_changes or enabled_changes or structure_changed)
    if not changed:
        return AnalysisSettingsApplyResult(False, (), False, False)

    non_display_parameter_nodes = {
        node_id
        for node_id, changed_keys in parameter_changes.items()
        if any(
            not _parameter_spec(catalogue[node_id], key).display_only
            for key in changed_keys
        )
    }
    analytical_changed = bool(
        structure_changed or enabled_changes or non_display_parameter_nodes
    )
    changed_roots = (
        set(parameter_changes)
        | enabled_changes
        | suspended_changes
        | active_changes
        | {connection.target for connection in connection_changes}
    )

    old_edges = tuple(graph.connections)
    desired_active_connections = tuple(
        template
        for template in graph.connection_templates
        if template in desired_connected
        and template.source in desired_active
        and template.target in desired_active
    )
    desired_unused_connections = tuple(
        template
        for template in graph.connection_templates
        if template in desired_connected
        and not (
            template.source in desired_active and template.target in desired_active
        )
    )
    if analytical_changed:
        affected = _downstream_closure(
            changed_roots,
            (*old_edges, *desired_active_connections),
        )
    else:
        affected = set(parameter_changes)
    affected_order = tuple(
        node_id for node_id in prepared.catalogue_order if node_id in affected
    )

    # Snapshot every field touched below.  Validation has already completed,
    # but rollback preserves the no-partial-mutation contract even if a future
    # graph invariant adds a failure after assignment.
    old_nodes = graph.nodes
    old_unused_nodes = graph.unused_nodes
    old_connections_tuple = graph.connections
    old_unused_connections_tuple = graph.unused_connections
    old_suspended = set(graph._connection_suspended)
    old_revision = graph.revision
    old_node_state = {
        node_id: (
            dict(node.parameters),
            node.enabled,
            node.status,
            node.status_detail,
        )
        for node_id, node in catalogue.items()
    }
    try:
        graph.nodes = {
            node_id: catalogue[node_id]
            for node_id in prepared.catalogue_order
            if node_id in desired_active
        }
        graph.unused_nodes = {
            node_id: catalogue[node_id]
            for node_id in prepared.catalogue_order
            if node_id not in desired_active
        }
        graph.connections = desired_active_connections
        graph.unused_connections = desired_unused_connections
        graph._connection_suspended.clear()
        graph._connection_suspended.update(desired_suspended_nodes)
        for node_id in prepared.catalogue_order:
            node = catalogue[node_id]
            node.parameters = dict(prepared.parameters_by_id[node_id])
            node.enabled = prepared.enabled_by_id[node_id]

        graph._validate_connections()
        graph.topological_order()
        if analytical_changed:
            graph.revision += 1
            graph.invalidate(affected_order, preserve_bypassed=True)
            for node_id in affected_order:
                if node_id in graph.nodes:
                    continue
                node = catalogue[node_id]
                if not node.enabled:
                    node.status = NodeStatus.BYPASSED
                    node.status_detail = "Bypassed"
                elif not node.implemented:
                    node.status = NodeStatus.PLANNED
                    node.status_detail = "Planned"
                else:
                    node.status = NodeStatus.IDLE
                    node.status_detail = "Not run"
    except Exception:
        graph.nodes = old_nodes
        graph.unused_nodes = old_unused_nodes
        graph.connections = old_connections_tuple
        graph.unused_connections = old_unused_connections_tuple
        graph._connection_suspended.clear()
        graph._connection_suspended.update(old_suspended)
        graph.revision = old_revision
        for node_id, (parameters, enabled, status, detail) in old_node_state.items():
            node = catalogue[node_id]
            node.parameters = parameters
            node.enabled = enabled
            node.status = status
            node.status_detail = detail
        raise

    return AnalysisSettingsApplyResult(
        changed=True,
        affected_node_ids=affected_order,
        structure_changed=structure_changed,
        analytical_changed=analytical_changed,
    )


def load_and_apply_analysis_settings_profile(
    path: Path | str,
    graph: PipelineGraph,
) -> AnalysisSettingsApplyResult:
    """Load, fully validate against ``graph``, and atomically apply a profile."""

    return apply_analysis_settings_profile(graph, load_analysis_settings_profile(path))


def _prepare_analysis_settings(
    graph: PipelineGraph,
    profile: AnalysisSettingsProfile,
) -> _PreparedAnalysisSettings:
    canonical = analysis_settings_profile_from_payload(
        analysis_settings_profile_to_payload(profile)
    )
    catalogue = _graph_catalogue(graph)
    expected_ids = set(catalogue)
    retired_node_ids = {
        "directed_edges",
        "undirected_edges",
        "foreground_segmentation",
        "foreground_noise_likelihood",
        "manual_seed_centres",
        "colour_reference",
        "scale_calibration",
        "edge_ridges",
        "reference_edge_ridges",
        "seed_interior",
        "raw_images",
        "reference_layers",
        "perimeter_background_reference",
    }
    canonical_by_id = {node.identifier: node for node in canonical.nodes}
    profile_by_id = {
        node.identifier: node
        for node in canonical.nodes
        if not (
            canonical.version in _LEGACY_ANALYSIS_SETTINGS_VERSIONS
            and node.identifier in retired_node_ids
        )
    }
    if (
        canonical.version in _LEGACY_ANALYSIS_SETTINGS_VERSIONS
        and "project" in expected_ids
        and "project" not in profile_by_id
    ):
        # Version nine replaces the Raw images and Manual annotations pseudo-
        # roots with one per-image Project root. It has no analytical settings,
        # so the authored current defaults are the complete migration.
        profile_by_id["project"] = AnalysisNodeSettings(
            identifier="project",
            active=True,
            enabled=True,
            connection_suspended=False,
            parameters=(),
        )
    if (
        canonical.version in _LEGACY_ANALYSIS_SETTINGS_VERSIONS
        and "species_reference_library" in expected_ids
        and "species_reference_library" not in profile_by_id
    ):
        # Version sixteen adds a compact resolver between Project/Metadata and
        # every reusable-reference consumer.  Exact pins live in the project
        # document rather than an analysis profile, so legacy profiles adopt
        # the authored node and connections without inventing a library.
        library_node = catalogue["species_reference_library"]
        profile_by_id["species_reference_library"] = AnalysisNodeSettings(
            identifier="species_reference_library",
            active=True,
            enabled=True,
            connection_suspended=False,
            parameters=tuple(
                (spec.key, library_node.parameters[spec.key])
                for spec in library_node.parameter_specs
            ),
        )
    if (
        canonical.version in _LEGACY_ANALYSIS_SETTINGS_VERSIONS
        and "reference_seed_traits" in expected_ids
        and "reference_seed_traits" not in profile_by_id
    ):
        # Version fourteen introduced a terminal diagnostic classifier whose
        # outputs do not feed any pre-existing calculation. Older profiles can
        # therefore adopt its authored defaults without changing their saved
        # analytical results or topology elsewhere in the graph.
        trait_node = catalogue["reference_seed_traits"]
        profile_by_id["reference_seed_traits"] = AnalysisNodeSettings(
            identifier="reference_seed_traits",
            active=True,
            enabled=True,
            connection_suspended=False,
            parameters=tuple(
                (spec.key, trait_node.parameters[spec.key])
                for spec in trait_node.parameter_specs
            ),
        )
    profile_ids = set(profile_by_id)
    missing_ids = expected_ids - profile_ids
    unknown_ids = profile_ids - expected_ids
    if missing_ids or unknown_ids:
        details: list[str] = []
        if missing_ids:
            details.append(f"missing nodes: {_formatted_names(missing_ids)}")
        if unknown_ids:
            details.append(f"unknown nodes: {_formatted_names(unknown_ids)}")
        raise IncompatibleAnalysisSettingsProfile(
            "Analysis-settings node catalogue mismatch (" + "; ".join(details) + ")."
        )

    active_by_id: dict[str, bool] = {}
    enabled_by_id: dict[str, bool] = {}
    connection_suspended_by_id: dict[str, bool] = {}
    parameters_by_id: dict[str, dict[str, JsonScalar]] = {}
    supplied_parameters_by_id = {
        node_id: dict(saved.parameters) for node_id, saved in profile_by_id.items()
    }
    if canonical.version < 18:
        supplied_parameters_by_id.get("seed_scale_estimation", {}).pop(
            "annotated_seed_top_fraction", None
        )
    if canonical.version in _LEGACY_ANALYSIS_SETTINGS_VERSIONS:
        # Version seven removes graph-only wrappers while preserving every
        # calculating control on the surviving owner.
        for retired_id, owner_id in (
            ("colour_reference", "deskew_colour"),
            ("scale_calibration", "ruler_detection"),
            ("edge_ridges", "edge_gradients"),
            ("reference_edge_ridges", "reference_edge_probability"),
            ("perimeter_background_reference", "layout_detection"),
        ):
            retired = canonical_by_id.get(retired_id)
            if retired is not None:
                supplied_parameters_by_id[owner_id].update(
                    dict(retired.parameters)
                )
    if canonical.version in _PRE_COLOUR_MERGE_VERSIONS:
        # Version five merged the two colour cards without changing either
        # equation, and folded the centre-point input into Manual annotations.
        # Carry the old foreground controls onto the combined background card;
        # the old Background-node enabled flag becomes its dedicated control.
        legacy_foreground = canonical_by_id.get("foreground_segmentation")
        legacy_background = canonical_by_id.get("background_likelihood")
        combined = supplied_parameters_by_id.get("background_likelihood", {})
        if legacy_foreground is not None:
            combined.update(dict(legacy_foreground.parameters))
        if legacy_background is not None:
            combined["background_colour_enabled"] = bool(
                legacy_background.enabled
            )
    if canonical.version in _PRE_NOISE_MERGE_VERSIONS:
        # Version six similarly combines the two noise cards while preserving
        # both calculations, parameter names, output ports, and cache keys.
        legacy_foreground_noise = canonical_by_id.get(
            "foreground_noise_likelihood"
        )
        combined_noise = supplied_parameters_by_id.get(
            "refined_background_likelihood", {}
        )
        legacy_background_noise = canonical_by_id.get(
            "refined_background_likelihood"
        )
        if legacy_background_noise is not None:
            combined_noise["background_noise_enabled"] = bool(
                legacy_background_noise.enabled
            )
        if legacy_foreground_noise is not None:
            combined_noise.update(dict(legacy_foreground_noise.parameters))
            combined_noise["foreground_noise_enabled"] = bool(
                legacy_foreground_noise.enabled
            )
    if canonical.version in _LEGACY_ANALYSIS_SETTINGS_VERSIONS:
        retired_parameters = {
            "foreground_segmentation": {
                "foreground_otsu_fraction",
                "foreground_background_prior_tolerance",
                "foreground_probability_softness_fraction",
                "foreground_local_contrast_scale_fraction",
                "foreground_shadow_rejection_strength",
                "foreground_distribution_fit_iterations",
                "foreground_refinement_iterations",
                "foreground_refinement_min_probability",
                "foreground_automatic_evidence_floor",
                "foreground_reviewed_authority_half_life_seed_areas",
            },
            "background_likelihood": {
                "background_refinement_iterations",
                "background_refinement_min_probability",
            },
            "material_evidence_decision": {
                "material_reviewed_foreground_weight",
            },
            "refined_background_likelihood": {
                "noise_background_min_likelihood",
                "noise_nonbackground_max_likelihood",
                "foreground_noise_foreground_min_likelihood",
                "foreground_noise_nonforeground_max_likelihood",
            },
        }
        for node_id, keys in retired_parameters.items():
            supplied = supplied_parameters_by_id.get(node_id, {})
            for key in keys:
                supplied.pop(key, None)
    # Older profiles stored the edge-classifier controls on the combined
    # prototype node. Move them to their new, visible owner before validation.
    moved_edge_keys = {
        "reference_texture_edge_prototypes_per_class",
        "reference_texture_edge_working_maximum_dimension",
        "reference_edge_minimum_working_seed_diameter_px",
        "reference_edge_strip_normal_offset_fraction",
        "reference_edge_strip_tangent_half_length_fraction",
        "reference_edge_ridge_weight",
        "reference_texture_instance_interior_buffer_fraction",
    }
    if canonical.version in _LEGACY_ANALYSIS_SETTINGS_VERSIONS:
        legacy_prototypes = supplied_parameters_by_id.get(
            "reference_texture_prototypes", {}
        )
        edge_parameters = supplied_parameters_by_id.get(
            "reference_edge_probability", {}
        )
        for key in moved_edge_keys:
            if key in legacy_prototypes and key not in edge_parameters:
                edge_parameters[key] = legacy_prototypes.pop(key)
        for new_key, legacy_key in (
            ("reference_edge_minimum_samples_per_prototype", "reference_texture_minimum_samples_per_prototype"),
            ("reference_edge_fit_iterations", "reference_texture_fit_iterations"),
            ("reference_edge_similarity_scale", "reference_texture_similarity_scale"),
        ):
            if new_key not in edge_parameters and legacy_key in legacy_prototypes:
                edge_parameters[new_key] = legacy_prototypes[legacy_key]
    for node_id, node in catalogue.items():
        saved = profile_by_id[node_id]
        supplied = supplied_parameters_by_id[node_id]
        expected_keys = {spec.key for spec in node.parameter_specs}
        supplied_keys = set(supplied)
        missing_keys = expected_keys - supplied_keys
        unknown_keys = supplied_keys - expected_keys
        if unknown_keys:
            details = []
            if unknown_keys:
                details.append(f"unknown parameters: {_formatted_names(unknown_keys)}")
            raise IncompatibleAnalysisSettingsProfile(
                f"Settings for node {node_id!r} do not match this application "
                f"({'; '.join(details)})."
            )
        if missing_keys:
            if canonical.version not in _LEGACY_ANALYSIS_SETTINGS_VERSIONS:
                raise IncompatibleAnalysisSettingsProfile(
                    f"Settings for node {node_id!r} do not match this application "
                    f"(missing parameters: {_formatted_names(missing_keys)})."
                )
            # Version-one profiles predate these controls. They adopt the
            # current authored defaults; current-version profiles remain strict.
            for key in missing_keys:
                supplied[key] = node.parameters[key]
        normalized: dict[str, JsonScalar] = {}
        for spec in node.parameter_specs:
            raw_value = supplied[spec.key]
            _require_parameter_json_type(spec, raw_value, node_id)
            try:
                value = node.validated_parameter_value(spec.key, raw_value)
            except (KeyError, TypeError, ValueError) as error:
                raise IncompatibleAnalysisSettingsProfile(
                    f"Invalid value for {node_id}.{spec.key}: {error}"
                ) from error
            normalized[spec.key] = _json_scalar(value, f"{node_id}.{spec.key}")
        active_by_id[node_id] = saved.active
        enabled_by_id[node_id] = (
            True
            if (
                canonical.version in _LEGACY_ANALYSIS_SETTINGS_VERSIONS
                and node_id == "background_likelihood"
            )
            or (
                canonical.version in _PRE_NOISE_MERGE_VERSIONS
                and node_id == "refined_background_likelihood"
            )
            else saved.enabled
        )
        connection_suspended_by_id[node_id] = saved.connection_suspended
        parameters_by_id[node_id] = normalized

    template_by_key: dict[ConnectionKey, PipelineConnection] = {}
    for template in graph.connection_templates:
        key = _connection_key(template)
        if key in template_by_key:
            raise IncompatibleAnalysisSettingsProfile(
                f"The application has duplicate authored connection "
                f"{_format_connection_key(key)}."
            )
        template_by_key[key] = template
    retired_foreground_targets = {
        "reviewed_foreground",
        "automatic_foreground_authority",
    }
    connected_by_key: dict[ConnectionKey, bool] = (
        {key: True for key in template_by_key}
        if canonical.version in _LEGACY_ANALYSIS_SETTINGS_VERSIONS
        else {}
    )
    for connection in canonical.connections:
        source = connection.source
        source_port = connection.source_port
        target = connection.target
        if canonical.version in _LEGACY_ANALYSIS_SETTINGS_VERSIONS:
            if target == "foreground_noise_likelihood":
                target = "refined_background_likelihood"
            if target in retired_node_ids:
                continue
            if source in {"directed_edges", "undirected_edges"}:
                continue
            if (
                target == "material_evidence_decision"
                and connection.target_port in retired_foreground_targets
            ):
                continue
            if (
                target == "refined_background_likelihood"
                and connection.target_port
                in {"background_probability", "foreground_probability"}
            ):
                continue
            if source == "foreground_segmentation":
                source = "background_likelihood"
            elif source == "foreground_noise_likelihood":
                source = "refined_background_likelihood"
            elif source == "manual_seed_centres":
                source = "project"
                source_port = "annotations"
            elif source == "raw_images":
                source = "project"
                source_port = "raw_image"
            elif source == "reference_layers":
                source = "project"
                source_port = "annotations"
            elif source == "perimeter_background_reference":
                source = "layout_detection"
                source_port = "perimeter_background"
            source = {
                "colour_reference": "deskew_colour",
                "scale_calibration": "ruler_detection",
                "edge_ridges": "edge_gradients",
                "reference_edge_ridges": "reference_edge_probability",
                "seed_interior": "material_evidence_decision",
            }.get(source, source)
            target = {
                "scale_calibration": "ruler_detection",
                "edge_ridges": "edge_gradients",
                "reference_edge_ridges": "reference_edge_probability",
            }.get(target, target)
            if target in {"colour_reference", "seed_interior"}:
                continue
            if source == target:
                continue
            if (
                target in {"procedural_instances", "unet_instances", "instance_masks"}
                and connection.target_port == "annotations"
            ):
                continue
        key = (
            source,
            source_port,
            target,
            connection.target_port,
        )
        if key in template_by_key:
            connected_by_key[key] = bool(connection.connected)
        else:
            # Keep an unrecognised endpoint so the catalogue comparison below
            # rejects it. Silently dropping arbitrary current or legacy wires
            # would turn a corrupt/profile-from-the-future connection into a
            # deceptively successful load. Every genuinely retired endpoint is
            # handled explicitly by the migration rules above.
            connected_by_key[key] = bool(connection.connected)
    legacy_reference_ridge_key = (
        "reference_edge_ridges",
        "reference_ridges",
        "procedural_instances",
        "reference_ridges",
    )
    net_reference_ridge_key = (
        "reference_edge_ridges",
        "net_reference_ridges",
        "procedural_instances",
        "reference_ridges",
    )
    if (
        canonical.version in _LEGACY_ANALYSIS_SETTINGS_VERSIONS
        and
        legacy_reference_ridge_key in connected_by_key
        and net_reference_ridge_key in template_by_key
        and net_reference_ridge_key not in connected_by_key
    ):
        connected_by_key[net_reference_ridge_key] = connected_by_key.pop(
            legacy_reference_ridge_key
        )
    expected_connections = set(template_by_key)
    supplied_connections = set(connected_by_key)
    missing_connections = expected_connections - supplied_connections
    unknown_connections = supplied_connections - expected_connections
    if unknown_connections:
        details = []
        if unknown_connections:
            details.append(
                "unknown connections: "
                + _formatted_connections(unknown_connections)
            )
        raise IncompatibleAnalysisSettingsProfile(
            "Analysis-settings connection catalogue mismatch ("
            + "; ".join(details)
            + ")."
        )
    if missing_connections:
        if canonical.version not in _LEGACY_ANALYSIS_SETTINGS_VERSIONS:
            raise IncompatibleAnalysisSettingsProfile(
                "Analysis-settings connection catalogue mismatch (missing connections: "
                + _formatted_connections(missing_connections)
                + ")."
            )
        # New authored connections use their template state when loading a
        # version-one otherwise compatible graph profile.
        authored_connected = set(graph.connections) | set(graph.unused_connections)
        for key in missing_connections:
            connected_by_key[key] = template_by_key[key] in authored_connected

    active_ids = {node_id for node_id, active in active_by_id.items() if active}
    connected_keys = {key for key, connected in connected_by_key.items() if connected}
    for node_id, active in active_by_id.items():
        if not active and enabled_by_id[node_id]:
            raise IncompatibleAnalysisSettingsProfile(
                f"Toolbox node {node_id!r} cannot be enabled."
            )
    disconnected_input_targets = {
        key[2]
        for key in expected_connections
        if key[0] in active_ids
        and key[2] in active_ids
        and key not in connected_keys
    }
    connected_active_edges = tuple(
        template_by_key[key]
        for key in connected_keys
        if key[0] in active_ids and key[2] in active_ids
    )
    suspended_branch = _downstream_closure(
        disconnected_input_targets, connected_active_edges
    )
    for node_id, suspended in connection_suspended_by_id.items():
        if not suspended:
            continue
        if node_id not in active_ids:
            raise IncompatibleAnalysisSettingsProfile(
                f"Toolbox node {node_id!r} cannot be connection-suspended."
            )
        if enabled_by_id[node_id]:
            raise IncompatibleAnalysisSettingsProfile(
                f"Enabled node {node_id!r} cannot be connection-suspended."
            )
        if node_id not in suspended_branch:
            raise IncompatibleAnalysisSettingsProfile(
                f"Node {node_id!r} is marked connection-suspended but is not in "
                "a branch with a disconnected required input."
            )
    for key in connected_keys:
        source, _source_port, target, _target_port = key
        if source not in expected_ids or target not in expected_ids:
            # The exact catalogue comparison above normally owns this error,
            # but this guard keeps the invariant explicit for future schemas.
            raise IncompatibleAnalysisSettingsProfile(
                f"Connection {_format_connection_key(key)} references an unknown node."
            )
    for node_id in active_ids:
        if not enabled_by_id[node_id]:
            continue
        missing_inputs = [
            key
            for key in expected_connections
            if key[2] == node_id
            and key[0] in active_ids
            and key not in connected_keys
        ]
        if missing_inputs:
            raise IncompatibleAnalysisSettingsProfile(
                f"Enabled node {node_id!r} is missing required input connection(s): "
                f"{_formatted_connections(missing_inputs)}."
            )
        disabled_upstream = sorted(
            {
                key[0]
                for key in connected_keys
                if key[2] == node_id
                and key[0] in active_ids
                and not enabled_by_id[key[0]]
            }
        )
        if disabled_upstream:
            raise IncompatibleAnalysisSettingsProfile(
                f"Enabled node {node_id!r} depends on disabled node(s): "
                f"{', '.join(disabled_upstream)}."
            )

    return _PreparedAnalysisSettings(
        profile=canonical,
        catalogue_order=tuple(catalogue),
        active_by_id=active_by_id,
        enabled_by_id=enabled_by_id,
        connection_suspended_by_id=connection_suspended_by_id,
        parameters_by_id=parameters_by_id,
        connected_by_key=connected_by_key,
        template_by_key=template_by_key,
    )


def _analysis_settings_profile_to_unchecked_payload(
    profile: AnalysisSettingsProfile,
) -> dict[str, object]:
    nodes: list[dict[str, object]] = []
    for node in profile.nodes:
        if not isinstance(node, AnalysisNodeSettings):
            raise InvalidAnalysisSettingsProfile(
                "Profile nodes must be AnalysisNodeSettings values."
            )
        parameters: dict[str, JsonScalar] = {}
        for item in node.parameters:
            if not isinstance(item, tuple) or len(item) != 2:
                raise InvalidAnalysisSettingsProfile(
                    f"Parameters for node {node.identifier!r} must be key/value pairs."
                )
            key, value = item
            key = _identifier(key, f"parameter name on node {node.identifier!r}")
            if key in parameters:
                raise InvalidAnalysisSettingsProfile(
                    f"Parameter {key!r} appears more than once on node "
                    f"{node.identifier!r}."
                )
            parameters[key] = _json_scalar(value, f"{node.identifier}.{key}")
        nodes.append(
            {
                "id": node.identifier,
                "active": node.active,
                "enabled": node.enabled,
                "connection_suspended": node.connection_suspended,
                "parameters": parameters,
            }
        )
    connections: list[dict[str, object]] = []
    for connection in profile.connections:
        if not isinstance(connection, AnalysisConnectionSettings):
            raise InvalidAnalysisSettingsProfile(
                "Profile connections must be AnalysisConnectionSettings values."
            )
        connections.append(
            {
                "source": connection.source,
                "source_port": connection.source_port,
                "target": connection.target,
                "target_port": connection.target_port,
                "connected": connection.connected,
            }
        )
    return {
        "format": profile.format,
        "version": profile.version,
        "nodes": nodes,
        "connections": connections,
    }


def _graph_catalogue(graph: PipelineGraph) -> dict[str, Any]:
    if not isinstance(graph, PipelineGraph):
        raise TypeError("Expected a PipelineGraph.")
    overlap = set(graph.nodes) & set(graph.unused_nodes)
    if overlap:
        raise IncompatibleAnalysisSettingsProfile(
            f"Pipeline nodes cannot be both active and unused: "
            f"{_formatted_names(overlap)}."
        )
    return {**graph.nodes, **graph.unused_nodes}


def _parameter_spec(node: Any, key: str) -> ParameterSpec:
    spec = next((candidate for candidate in node.parameter_specs if candidate.key == key), None)
    if spec is None:
        raise IncompatibleAnalysisSettingsProfile(
            f"Node {node.identifier!r} has no parameter {key!r}."
        )
    return spec


def _require_parameter_json_type(
    spec: ParameterSpec, value: JsonScalar, node_id: str
) -> None:
    valid = False
    if spec.kind == "bool":
        valid = isinstance(value, bool)
    elif spec.kind == "int":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif spec.kind == "float":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif spec.kind in {"choice", "text"}:
        valid = isinstance(value, str)
    if not valid:
        raise IncompatibleAnalysisSettingsProfile(
            f"Parameter {node_id}.{spec.key} must use JSON type {spec.kind!r}."
        )


def _connection_key(connection: PipelineConnection) -> ConnectionKey:
    return (
        connection.source,
        connection.source_port,
        connection.target,
        connection.target_port,
    )


def _downstream_closure(
    roots: set[str], connections: Sequence[PipelineConnection]
) -> set[str]:
    outgoing: dict[str, set[str]] = {}
    for connection in connections:
        outgoing.setdefault(connection.source, set()).add(connection.target)
    affected = set(roots)
    pending = list(roots)
    while pending:
        node_id = pending.pop()
        for target in outgoing.get(node_id, ()):
            if target in affected:
                continue
            affected.add(target)
            pending.append(target)
    return affected


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InvalidAnalysisSettingsProfile(f"{name.capitalize()} must be an object.")
    for key in value:
        if not isinstance(key, str):
            raise InvalidAnalysisSettingsProfile(
                f"Every key in {name} must be a string."
            )
    return value


def _sequence(value: object, name: str, maximum: int) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise InvalidAnalysisSettingsProfile(f"{name.capitalize()} must be an array.")
    if len(value) > maximum:
        raise InvalidAnalysisSettingsProfile(
            f"{name.capitalize()} contains more than {maximum} records."
        )
    return value


def _require_exact_keys(
    payload: Mapping[str, object], expected: set[str], name: str
) -> None:
    supplied = set(payload)
    missing = expected - supplied
    unknown = supplied - expected
    if not missing and not unknown:
        return
    details = []
    if missing:
        details.append(f"missing {_formatted_names(missing)}")
    if unknown:
        details.append(f"unknown {_formatted_names(unknown)}")
    raise InvalidAnalysisSettingsProfile(
        f"{name.capitalize()} has invalid fields ({'; '.join(details)})."
    )


def _identifier(value: object, name: str) -> str:
    text = _bounded_text(value, name, maximum=_MAX_IDENTIFIER_CHARACTERS)
    if not text or text.strip() != text:
        raise InvalidAnalysisSettingsProfile(
            f"{name.capitalize()} must be a non-empty trimmed string."
        )
    return text


def _bounded_text(
    value: object, name: str, *, maximum: int = _MAX_TEXT_CHARACTERS
) -> str:
    if not isinstance(value, str):
        raise InvalidAnalysisSettingsProfile(f"{name.capitalize()} must be text.")
    if len(value) > maximum:
        raise InvalidAnalysisSettingsProfile(
            f"{name.capitalize()} exceeds {maximum} characters."
        )
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise InvalidAnalysisSettingsProfile(f"{name.capitalize()} must be boolean.")
    return value


def _json_scalar(value: object, name: str) -> JsonScalar:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidAnalysisSettingsProfile(
                f"{name.capitalize()} must be a finite number."
            )
        return value
    if isinstance(value, str):
        return _bounded_text(value, name)
    raise InvalidAnalysisSettingsProfile(
        f"{name.capitalize()} must be a JSON string, number, or boolean."
    )


def _reject_duplicate_object_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidAnalysisSettingsProfile(
                f"Duplicate JSON object key {key!r} is not allowed."
            )
        result[key] = value
    return result


def _reject_non_finite_json_constant(value: str) -> object:
    raise InvalidAnalysisSettingsProfile(
        f"Non-finite JSON number {value!r} is not allowed."
    )


def _format_connection_key(key: ConnectionKey) -> str:
    return f"{key[0]}.{key[1]} -> {key[2]}.{key[3]}"


def _formatted_connections(keys: Sequence[ConnectionKey] | set[ConnectionKey]) -> str:
    values = sorted(_format_connection_key(key) for key in keys)
    if len(values) > 4:
        return ", ".join(values[:4]) + f", … (+{len(values) - 4} more)"
    return ", ".join(values)


def _formatted_names(values: set[str]) -> str:
    names = sorted(values)
    if len(names) > 8:
        return ", ".join(repr(name) for name in names[:8]) + (
            f", … (+{len(names) - 8} more)"
        )
    return ", ".join(repr(name) for name in names)
