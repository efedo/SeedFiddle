"""Bounded project persistence for immutable reference graph metadata.

Project files retain recipe, display and historical run identities only. A
historical checkpoint digest does not mean its arrays are present, restored or
verified; this project format always records checkpoint data as unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from seedvision.benchmarking.recipes import (
    MAX_RECIPE_BYTES,
    RecipeContractError,
    RecipeSpec,
    StageRunRecord,
    validate_stage_run,
)


REFERENCE_GRAPH_FORMAT = "seedfiddle.project-reference-graph"
REFERENCE_GRAPH_VERSION = 1
MAX_REFERENCE_GRAPHS = 16
MAX_REFERENCE_GRAPH_STAGES = 512
MAX_REFERENCE_GRAPH_ITEMS = 16_384
MAX_REFERENCE_GRAPH_BYTES = MAX_RECIPE_BYTES + 512 * 1024
MAX_REFERENCE_GRAPHS_BYTES = 8 * 1024 * 1024
_CHECKPOINT_DATA_STATUS = {
    "status": "unavailable",
    "reason": "Checkpoint arrays are not embedded, restored, or verified by project graph metadata.",
}


class ReferenceGraphPersistenceError(ValueError):
    """Reference graph metadata is invalid, stale, or exceeds its bounds."""


def _canonical_json(value: Any, *, maximum: int, label: str) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ReferenceGraphPersistenceError(f"{label} is not strict JSON metadata.") from error
    if len(encoded) > maximum:
        raise ReferenceGraphPersistenceError(f"{label} exceeds its {maximum}-byte limit.")
    return encoded


def _exact_object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ReferenceGraphPersistenceError(f"{label} has missing or unknown fields.")
    return value


def _recipe_payload_item_count(value: Any) -> int:
    if not isinstance(value, dict):
        raise ReferenceGraphPersistenceError("Reference recipe must be a JSON object.")
    raw_stages = value.get("stages")
    if not isinstance(raw_stages, list) or len(raw_stages) > MAX_REFERENCE_GRAPH_STAGES:
        raise ReferenceGraphPersistenceError(
            f"Reference recipe supports at most {MAX_REFERENCE_GRAPH_STAGES} stages."
        )
    item_count = 0
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, dict):
            continue  # The recipe decoder provides the precise schema error.
        for key in ("ports", "parameters", "source_mappings"):
            values = raw_stage.get(key)
            if isinstance(values, list):
                item_count += len(values)
                if item_count > MAX_REFERENCE_GRAPH_ITEMS:
                    raise ReferenceGraphPersistenceError(
                        f"Reference recipe exceeds the {MAX_REFERENCE_GRAPH_ITEMS}-item limit."
                    )
    for key in ("dependencies", "layout"):
        values = value.get(key)
        if key == "layout" and isinstance(values, list) and len(values) > MAX_REFERENCE_GRAPH_STAGES:
            raise ReferenceGraphPersistenceError(
                f"Reference recipe layout supports at most {MAX_REFERENCE_GRAPH_STAGES} stages."
            )
        if isinstance(values, list):
            item_count += len(values)
            if item_count > MAX_REFERENCE_GRAPH_ITEMS:
                raise ReferenceGraphPersistenceError(
                    f"Reference recipe exceeds the {MAX_REFERENCE_GRAPH_ITEMS}-item limit."
                )
    return item_count


def _validate_recipe_counts(recipe: RecipeSpec) -> None:
    if len(recipe.stages) > MAX_REFERENCE_GRAPH_STAGES:
        raise ReferenceGraphPersistenceError(
            f"Reference graphs support at most {MAX_REFERENCE_GRAPH_STAGES} stages."
        )
    if len(recipe.layout) > MAX_REFERENCE_GRAPH_STAGES:
        raise ReferenceGraphPersistenceError(
            f"Reference recipe layout supports at most {MAX_REFERENCE_GRAPH_STAGES} stages."
        )
    item_count = len(recipe.dependencies) + len(recipe.layout)
    for stage in recipe.stages:
        item_count += len(stage.ports) + len(stage.parameters) + len(stage.source_mappings)
        if item_count > MAX_REFERENCE_GRAPH_ITEMS:
            raise ReferenceGraphPersistenceError(
                f"Reference recipe exceeds the {MAX_REFERENCE_GRAPH_ITEMS}-item limit."
            )
    if item_count > MAX_REFERENCE_GRAPH_ITEMS:
        raise ReferenceGraphPersistenceError(
            f"Reference recipe exceeds the {MAX_REFERENCE_GRAPH_ITEMS}-item limit."
        )


def _recipe_from_payload(value: Any) -> RecipeSpec:
    _recipe_payload_item_count(value)
    encoded = _canonical_json(value, maximum=MAX_RECIPE_BYTES, label="Reference recipe")
    try:
        recipe = RecipeSpec.from_json(encoded)
        _validate_recipe_counts(recipe)
        return recipe
    except (RecipeContractError, TypeError, ValueError, KeyError, OverflowError, RecursionError) as error:
        if isinstance(error, ReferenceGraphPersistenceError):
            raise
        raise ReferenceGraphPersistenceError(f"Reference recipe is invalid: {error}") from error


@dataclass(frozen=True, slots=True)
class ReferenceGraphAttachment:
    """One exact recipe, per-stage historical status, and its view state."""

    recipe: RecipeSpec
    stage_runs: tuple[StageRunRecord, ...]
    selected_stage_id: str | None = None
    bundle_cables: bool = False
    route_around_nodes: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.recipe, RecipeSpec):
            raise ReferenceGraphPersistenceError("Reference graph requires a RecipeSpec.")
        _validate_recipe_counts(self.recipe)
        if not isinstance(self.stage_runs, tuple):
            raise ReferenceGraphPersistenceError("Reference graph stage_runs must be an immutable tuple.")
        expected_stage_ids = tuple(stage.stage_id for stage in self.recipe.stages)
        actual_stage_ids: list[str] = []
        for record in self.stage_runs:
            if not isinstance(record, StageRunRecord):
                raise ReferenceGraphPersistenceError("Reference graph run records must be StageRunRecord values.")
            actual_stage_ids.append(record.stage_id)
            try:
                validate_stage_run(self.recipe, record)
            except (RecipeContractError, KeyError) as error:
                raise ReferenceGraphPersistenceError(
                    f"Historical run for stage {record.stage_id!r} is stale or invalid: {error}"
                ) from error
        if len(set(actual_stage_ids)) != len(actual_stage_ids):
            raise ReferenceGraphPersistenceError("Reference graph stage run IDs must be unique.")
        if len(self.stage_runs) != len(self.recipe.stages):
            raise ReferenceGraphPersistenceError(
                "Reference graph must contain exactly one historical run record for every recipe stage."
            )
        if tuple(actual_stage_ids) != expected_stage_ids:
            raise ReferenceGraphPersistenceError(
                "Reference graph stage runs must exactly follow the recipe stage IDs in recipe order."
            )
        if self.selected_stage_id is not None and (
            not isinstance(self.selected_stage_id, str)
            or self.selected_stage_id not in set(expected_stage_ids)
        ):
            raise ReferenceGraphPersistenceError("Selected reference stage is missing from the recipe.")
        if type(self.bundle_cables) is not bool or type(self.route_around_nodes) is not bool:
            raise ReferenceGraphPersistenceError("Reference graph cable display state must be Boolean.")
        # Encode now so directly constructed attachments obey the same strict
        # per-graph cap as decoded project files.
        _canonical_json(self._payload(), maximum=MAX_REFERENCE_GRAPH_BYTES, label="Reference graph attachment")

    @property
    def layout(self):
        """The persisted recipe layout, exposed as graph view state."""

        return self.recipe.layout

    @property
    def checkpoint_data_status(self) -> str:
        """Historical digests never imply that checkpoint arrays are available."""

        return "unavailable"

    def _payload(self) -> dict[str, Any]:
        return {
            "format": REFERENCE_GRAPH_FORMAT,
            "version": REFERENCE_GRAPH_VERSION,
            "recipe": self.recipe.to_dict(),
            "stage_runs": [record.to_dict() for record in self.stage_runs],
            "selected_stage_id": self.selected_stage_id,
            "bundle_cables": self.bundle_cables,
            "route_around_nodes": self.route_around_nodes,
            "checkpoint_data": dict(_CHECKPOINT_DATA_STATUS),
        }

    def to_dict(self) -> dict[str, Any]:
        _validate_recipe_counts(self.recipe)
        payload = self._payload()
        _canonical_json(payload, maximum=MAX_REFERENCE_GRAPH_BYTES, label="Reference graph attachment")
        return payload

    @classmethod
    def from_dict(cls, value: Any) -> "ReferenceGraphAttachment":
        payload = _exact_object(
            value,
            {
                "format", "version", "recipe", "stage_runs", "selected_stage_id",
                "bundle_cables", "route_around_nodes", "checkpoint_data",
            },
            "Reference graph attachment",
        )
        _canonical_json(payload, maximum=MAX_REFERENCE_GRAPH_BYTES, label="Reference graph attachment")
        if payload["format"] != REFERENCE_GRAPH_FORMAT or type(payload["version"]) is not int or payload["version"] != REFERENCE_GRAPH_VERSION:
            raise ReferenceGraphPersistenceError("Unsupported reference graph format or version.")
        if payload["checkpoint_data"] != _CHECKPOINT_DATA_STATUS:
            raise ReferenceGraphPersistenceError(
                "Project graph metadata cannot claim checkpoint arrays are available or verified."
            )
        recipe = _recipe_from_payload(payload["recipe"])
        raw_runs = payload["stage_runs"]
        if not isinstance(raw_runs, list) or len(raw_runs) > MAX_REFERENCE_GRAPH_STAGES:
            raise ReferenceGraphPersistenceError("Reference graph stage_runs exceed the stage limit.")
        try:
            runs = tuple(StageRunRecord.from_dict(item) for item in raw_runs)
        except (RecipeContractError, TypeError, ValueError) as error:
            raise ReferenceGraphPersistenceError(f"Reference graph run record is invalid: {error}") from error
        try:
            return cls(
                recipe=recipe,
                stage_runs=runs,
                selected_stage_id=payload["selected_stage_id"],
                bundle_cables=payload["bundle_cables"],
                route_around_nodes=payload["route_around_nodes"],
            )
        except (RecipeContractError, TypeError, ValueError) as error:
            if isinstance(error, ReferenceGraphPersistenceError):
                raise
            raise ReferenceGraphPersistenceError(f"Reference graph attachment is invalid: {error}") from error


def validate_reference_graphs(value: object) -> tuple[ReferenceGraphAttachment, ...]:
    """Validate the immutable project collection and aggregate byte limits."""

    if not isinstance(value, tuple):
        raise ReferenceGraphPersistenceError("Project reference_graphs must be an immutable tuple.")
    if len(value) > MAX_REFERENCE_GRAPHS:
        raise ReferenceGraphPersistenceError(
            f"Projects support at most {MAX_REFERENCE_GRAPHS} reference graph attachments."
        )
    identities: set[tuple[str, str]] = set()
    total_bytes = 0
    for attachment in value:
        if not isinstance(attachment, ReferenceGraphAttachment):
            raise ReferenceGraphPersistenceError("Project reference_graphs entries must be graph attachments.")
        identity = (attachment.recipe.recipe_id, attachment.recipe.recipe_version)
        if identity in identities:
            raise ReferenceGraphPersistenceError("Project reference graph attachments must be unique.")
        identities.add(identity)
        encoded = _canonical_json(
            attachment.to_dict(),
            maximum=MAX_REFERENCE_GRAPH_BYTES,
            label="Reference graph attachment",
        )
        total_bytes += len(encoded)
        if total_bytes > MAX_REFERENCE_GRAPHS_BYTES:
            raise ReferenceGraphPersistenceError(
                f"Project reference graph metadata exceeds the {MAX_REFERENCE_GRAPHS_BYTES}-byte aggregate limit."
            )
    return value


def reference_graphs_to_payload(value: object) -> list[dict[str, Any]]:
    return [attachment.to_dict() for attachment in validate_reference_graphs(value)]


def reference_graphs_from_payload(value: Any) -> tuple[ReferenceGraphAttachment, ...]:
    if not isinstance(value, list) or len(value) > MAX_REFERENCE_GRAPHS:
        raise ReferenceGraphPersistenceError("Project reference_graphs must be a bounded JSON array.")
    try:
        attachments = tuple(ReferenceGraphAttachment.from_dict(item) for item in value)
        return validate_reference_graphs(attachments)
    except ReferenceGraphPersistenceError:
        raise
    except (TypeError, ValueError, KeyError, OverflowError, RecursionError) as error:
        raise ReferenceGraphPersistenceError(f"Project reference graph metadata is invalid: {error}") from error


__all__ = [
    "MAX_REFERENCE_GRAPHS",
    "MAX_REFERENCE_GRAPHS_BYTES",
    "MAX_REFERENCE_GRAPH_BYTES",
    "MAX_REFERENCE_GRAPH_ITEMS",
    "MAX_REFERENCE_GRAPH_STAGES",
    "REFERENCE_GRAPH_FORMAT",
    "REFERENCE_GRAPH_VERSION",
    "ReferenceGraphAttachment",
    "ReferenceGraphPersistenceError",
    "reference_graphs_from_payload",
    "reference_graphs_to_payload",
    "validate_reference_graphs",
]
