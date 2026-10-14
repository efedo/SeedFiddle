"""Bounded, read-only inspection of explicitly selected reference checkpoints.

This module verifies a one-stage checkpoint chain produced by the pinned
PlantCV threshold or image/mask bridge. It never executes analysis and never infers that a
checkpoint belongs to the image currently open in the desktop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from seedvision.benchmarking.checkpoints import (
    CheckpointError,
    CheckpointIdentity,
    CheckpointLimits,
    CoordinateFrame,
    ExecutionStatus,
    InvalidCheckpoint,
    RasterRole,
    load_checkpoint,
)
from seedvision.benchmarking.recipe_catalogue import (
    RecipeCatalogueError,
    _load_source_config,
)
from seedvision.benchmarking.recipes import (
    RecipeContractError,
    RecipeProvenanceMode,
    RecipeSpec,
    StageRunRecord,
    StageRunState,
    StaleRecipeIdentity,
    validate_stage_run,
)
from seedvision.benchmarking.reference_runner import (
    PINNED_PLANTCV_COMMIT,
    PINNED_PLANTCV_VERSION,
    PINNED_REFERENCE_PLATFORM,
    PINNED_REFERENCE_PYTHON,
    PINNED_REQUIREMENTS_SHA256,
    PINNED_SOURCE_AUDIT_SHA256,
)
from seedvision.benchmarking.reference_stage import (
    REFERENCE_STAGE_FORMAT,
    REFERENCE_STAGE_VERSION,
    ReferenceStageError,
    expected_reference_stage_identity,
)
from seedvision.benchmarking.reference_mask_stage import (
    REFERENCE_MASK_STAGE_FORMAT,
    REFERENCE_MASK_STAGE_VERSION,
    expected_reference_mask_stage_identity,
)


MAX_INSPECTED_ARRAY_BYTES = 24 * 1024 * 1024
MAX_INSPECTED_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_INSPECTED_METADATA_BYTES = 1024 * 1024
MAX_ENVIRONMENT_MANIFEST_BYTES = 1024 * 1024
MAX_INSPECTED_ARRAYS = 8
MAX_PREVIEW_PIXELS = 1_048_576
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NORMALIZE_PACKAGE = re.compile(r"[-_.]+")
_THRESHOLD_CALLABLES = {
    "threshold.binary": "plantcv.plantcv.threshold.binary",
    "threshold.mean": "plantcv.plantcv.threshold.mean",
    "threshold.otsu": "plantcv.plantcv.threshold.otsu",
}
_THRESHOLD_OPERATIONS = frozenset(_THRESHOLD_CALLABLES)
_MASK_CALLABLES = {
    "filter.median_blur": "plantcv.plantcv.median_blur",
    "morphology.fill": "plantcv.plantcv.fill",
    "morphology.erode": "plantcv.plantcv.erode",
    "morphology.dilate": "plantcv.plantcv.dilate",
}
_INPUT_GRAY_ROLE = RasterRole("image", np.dtype(np.uint8).str, "HW", (0, 255))
_OUTPUT_MASK_ROLE = RasterRole("mask", np.dtype(np.uint8).str, "HW")


class InspectionStatus(StrEnum):
    """An explicit result state; only VERIFIED carries preview arrays."""

    NOT_RUN = "not_run"
    RECORDED = "recorded_without_checkpoint"
    STORE_UNBOUND = "store_unbound"
    LOADING = "loading"
    VERIFIED = "verified"
    MISSING = "missing"
    MISMATCH = "mismatch"
    UNSUPPORTED = "unsupported"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class CheckpointInspection:
    """Checked checkpoint identities, compact facts, and display-only rasters."""

    status: InspectionStatus
    detail: str
    checkpoint_sha256: str | None = None
    source_sha256: str | None = None
    coordinate_frame: CoordinateFrame | None = None
    execution_status: ExecutionStatus | None = None
    array_inventory: tuple[Mapping[str, Any], ...] = ()
    compact_metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    preview_arrays: Mapping[str, np.ndarray] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.status, InspectionStatus):
            object.__setattr__(self, "status", InspectionStatus(self.status))
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ValueError("Inspection detail must be non-empty text.")
        if self.status is InspectionStatus.VERIFIED:
            if not isinstance(self.checkpoint_sha256, str) or not _SHA256.fullmatch(self.checkpoint_sha256):
                raise ValueError("A verified inspection must identify its output checkpoint digest.")
            if not isinstance(self.source_sha256, str) or not _SHA256.fullmatch(self.source_sha256):
                raise ValueError("A verified inspection must identify its checkpoint-bound source digest.")
            if not isinstance(self.coordinate_frame, CoordinateFrame):
                raise ValueError("A verified inspection must identify its exact coordinate frame.")
        else:
            if self.preview_arrays:
                raise ValueError("Only a verified inspection may contain display arrays.")
        frozen_inventory = tuple(MappingProxyType(dict(item)) for item in self.array_inventory)
        object.__setattr__(self, "array_inventory", frozen_inventory)
        object.__setattr__(self, "compact_metadata", MappingProxyType(dict(self.compact_metadata)))
        object.__setattr__(self, "preview_arrays", MappingProxyType(dict(self.preview_arrays)))


def _plain(value: Any) -> Any:
    """Copy frozen JSON-like checkpoint metadata into ordinary primitives."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _json_same(left: Any, right: Any) -> bool:
    try:
        return json.dumps(
            _plain(left), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) == json.dumps(
            _plain(right), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _effective_parameters_match(actual: Any, expected: Mapping[str, Any]) -> bool:
    """Allow the pinned runner's binary64 offset representation exactly."""
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
        return False
    wanted = dict(expected)
    observed = dict(actual)
    if "offset" not in wanted:
        return _json_same(observed, wanted)
    expected_offset = wanted.pop("offset")
    actual_offset = observed.pop("offset")
    if not _json_same(observed, wanted):
        return False
    if (
        isinstance(expected_offset, bool)
        or type(expected_offset) not in (int, float)
        or isinstance(actual_offset, bool)
        or type(actual_offset) not in (int, float)
    ):
        return False
    try:
        expected_float = float(expected_offset)
    except (OverflowError, ValueError):
        return False
    if not np.isfinite(expected_float) or not np.isfinite(actual_offset):
        return False
    if type(expected_offset) is int and int(expected_float) != expected_offset:
        return False
    return actual_offset == expected_float


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    is_set = getattr(cancel, "is_set", None)
    if not callable(is_set):
        raise TypeError("cancel must be a callable or an event with is_set().")
    return bool(is_set())


def _result(status: InspectionStatus, detail: str) -> CheckpointInspection:
    return CheckpointInspection(status=status, detail=detail)


def _normalized_environment(environment: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, version in environment.items():
        key = _NORMALIZE_PACKAGE.sub("-", name).lower()
        if key in result:
            raise ReferenceStageError("Recipe environment has duplicate normalized package names.")
        result[key] = version
    return result


def _verify_runner_identity(metadata: Mapping[str, Any], recipe: RecipeSpec) -> bool:
    runner = metadata.get("runner_identity")
    if not isinstance(runner, Mapping):
        raise CheckpointError("The output artifact omits its verified runner identity.")
    expected = {
        "reference": "plantcv",
        "version": PINNED_PLANTCV_VERSION,
        "commit": PINNED_PLANTCV_COMMIT,
        "python": PINNED_REFERENCE_PYTHON,
        "platform": PINNED_REFERENCE_PLATFORM,
        "source_audit_sha256": PINNED_SOURCE_AUDIT_SHA256,
        "requirements_lock_sha256": PINNED_REQUIREMENTS_SHA256,
    }
    for name, value in expected.items():
        if runner.get(name) != value:
            raise CheckpointMismatchError(f"Recorded runner {name} differs from the committed PlantCV pin.")
    digest = runner.get("identity_digest")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise CheckpointError("The recorded runner identity digest is malformed.")
    actual = runner.get("actual_distributions")
    expected_distributions = _normalized_environment(recipe.provenance.environment_versions)
    if not isinstance(actual, Mapping) or dict(actual) != expected_distributions:
        raise CheckpointMismatchError("Recorded runner distributions differ from the exact recipe environment.")
    try:
        source_config = _load_source_config()
        source_files = sorted(
            [
                {"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"]}
                for item in source_config["files"]
            ],
            key=lambda item: item["path"],
        )
        expected_source_content_digest = hashlib.sha256(json.dumps(
            {
                "version": PINNED_PLANTCV_VERSION,
                "commit": PINNED_PLANTCV_COMMIT,
                "files": source_files,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        package_inventory: dict[str, tuple[str, str]] = {}
        for package in source_config["environment"]["packages"]:
            name = _NORMALIZE_PACKAGE.sub("-", package["name"]).lower()
            if name in package_inventory:
                raise CheckpointMismatchError("Committed reference lock has duplicate normalized packages.")
            package_inventory[name] = (package["version"], package["archive_sha256"])
        expected_packages = [
            {"name": name, "version": version, "archive_sha256": archive_hash}
            for name, (version, archive_hash) in sorted(package_inventory.items())
        ]
    except (KeyError, TypeError, ValueError, RecipeCatalogueError) as error:
        raise CheckpointError(f"Committed PlantCV source or package inventory is unavailable: {error}") from error
    if runner.get("source_content_digest") != expected_source_content_digest:
        raise CheckpointMismatchError("Recorded PlantCV source-content digest differs from the committed source manifest.")
    environment_manifest_digest = runner.get("environment_manifest_sha256")
    if not isinstance(environment_manifest_digest, str) or not _SHA256.fullmatch(environment_manifest_digest):
        raise CheckpointError("The recorded reference environment manifest digest is malformed.")
    # Check the ignored raw inventory when present. A fresh checkout can still
    # verify the complete committed package/version/hash inventory without it.
    environment_path = Path(__file__).resolve().parents[2] / "artifacts" / "reference-sources" / "plantcv-environment.json"
    raw_environment_checked = False
    if environment_path.is_file():
        try:
            with environment_path.open("rb") as stream:
                raw_environment = stream.read(MAX_ENVIRONMENT_MANIFEST_BYTES + 1)
        except OSError as error:
            raise CheckpointError("Cannot read the captured PlantCV environment manifest.") from error
        if (
            len(raw_environment) > MAX_ENVIRONMENT_MANIFEST_BYTES
            or hashlib.sha256(raw_environment).hexdigest() != environment_manifest_digest
        ):
            raise CheckpointMismatchError("Recorded runner environment manifest hash differs from the captured inventory.")
        raw_environment_checked = True
    identity_payload = {
        "reference": runner["reference"],
        "version": runner["version"],
        "commit": runner["commit"],
        "python": runner["python"],
        "platform": runner["platform"],
        "source_audit_sha256": runner["source_audit_sha256"],
        "source_content_digest": expected_source_content_digest,
        "source_files": source_files,
        "environment_manifest_sha256": environment_manifest_digest,
        "requirements_lock_sha256": runner["requirements_lock_sha256"],
        "packages": expected_packages,
    }
    computed_identity_digest = hashlib.sha256(json.dumps(
        identity_payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    if computed_identity_digest != digest:
        raise CheckpointMismatchError("Recorded runner identity digest does not match its pinned source/environment fields.")
    return raw_environment_checked


class CheckpointMismatchError(CheckpointError):
    """A valid artifact disagrees with its stage record or current recipe."""


def _inventory(checkpoint, prefix: str) -> tuple[Mapping[str, Any], ...]:
    entries: list[Mapping[str, Any]] = []
    for name, array in checkpoint.arrays.items():
        role = checkpoint.raster_roles.get(name)
        entries.append({
            "artifact": prefix,
            "name": name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "nbytes": int(array.nbytes),
            "role": None if role is None else role.to_dict(),
        })
    return tuple(entries)


def _compact_metadata(
    metadata: Mapping[str, Any], *, raw_environment_checked: bool
) -> Mapping[str, Any]:
    keys = (
        "operation_id", "parameters", "effective_parameters", "observations",
        "intermediates", "visualizations", "channel_semantics", "computation_runtime_ns",
        "wall_runtime_ns", "stage_runtime_ns", "scientific_validation_status",
    )
    compact = {key: _plain(metadata[key]) for key in keys if key in metadata}
    runner = metadata.get("runner_identity")
    if isinstance(runner, Mapping):
        compact["verified_runner_identity"] = {
            key: _plain(runner[key])
            for key in (
                "reference", "version", "commit", "python", "platform",
                "source_audit_sha256", "source_content_digest",
                "environment_manifest_sha256", "requirements_lock_sha256",
                "identity_digest", "actual_distributions",
            )
            if key in runner
        }
        compact["verified_runner_identity"]["identity_digest_recomputed"] = True
        compact["verified_runner_identity"]["raw_environment_manifest_checked"] = raw_environment_checked
    encoded = json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_INSPECTED_METADATA_BYTES:
        raise CheckpointError("Compact checkpoint facts exceed the inspection metadata limit.")
    return compact


def inspect_reference_checkpoint(
    store: Path | str | None,
    recipe: RecipeSpec,
    record: StageRunRecord,
    *,
    cancel: Any = None,
) -> CheckpointInspection:
    """Verify one bridged preprocessing output and its exact upstream artifact.

    ``store`` is a directory selected explicitly by the caller. This function
    performs bounded content-addressed loads only; it never searches for a
    store or substitutes another backend. The returned arrays are actual,
    hash-verified checkpoint pixels for display and inspection.
    """
    if not isinstance(recipe, RecipeSpec) or not isinstance(record, StageRunRecord):
        raise TypeError("recipe and record must be immutable recipe-contract values.")
    if record.state is StageRunState.NOT_RUN:
        return _result(InspectionStatus.NOT_RUN, "This stage has not been recorded as executed.")
    if record.state not in {StageRunState.EXECUTED, StageRunState.EMPTY}:
        reason = f"Recorded state is {record.state.value.upper()}"
        if record.reason:
            reason += f": {record.reason}"
        return _result(InspectionStatus.RECORDED, reason + ". No output checkpoint is recorded.")
    try:
        validate_stage_run(recipe, record)
    except (RecipeContractError, StaleRecipeIdentity, KeyError, TypeError, ValueError) as error:
        return _result(InspectionStatus.MISMATCH, f"Stage run does not match this exact recipe: {error}")
    try:
        stage = recipe.stage(record.stage_id)
    except KeyError:
        return _result(InspectionStatus.MISMATCH, "The recorded stage ID is absent from the selected recipe.")
    if recipe.provenance.mode is not RecipeProvenanceMode.ORIGINAL_REFERENCE:
        return _result(InspectionStatus.UNSUPPORTED, "Only original-reference PlantCV preprocessing artifacts have a verifier here.")
    if recipe.provenance.reference_id != "plantcv" or stage.operation_id not in (_THRESHOLD_OPERATIONS | _MASK_CALLABLES.keys()):
        return _result(InspectionStatus.UNSUPPORTED, "No exact artifact verifier is defined for this backend operation.")
    is_threshold = stage.operation_id in _THRESHOLD_OPERATIONS
    is_median = stage.operation_id == "filter.median_blur"
    input_port = "mask" if stage.operation_id == "morphology.fill" else "gray"
    output_name = "image" if is_median else "mask"
    input_role = _INPUT_GRAY_ROLE if is_threshold or is_median else _OUTPUT_MASK_ROLE
    output_role = _INPUT_GRAY_ROLE if is_median else _OUTPUT_MASK_ROLE
    if not record.upstream_checkpoint_sha256:
        return _result(InspectionStatus.MISMATCH, "The stage record omits its exact upstream checkpoint digest.")
    if store is None:
        return _result(InspectionStatus.STORE_UNBOUND, "Select a checkpoint store directory to inspect this recorded digest.")
    root = Path(store).expanduser()
    if not root.is_dir():
        return _result(InspectionStatus.MISSING, "The explicitly selected checkpoint store directory is unavailable.")
    output_digest = record.checkpoint_sha256
    if not output_digest or not _SHA256.fullmatch(output_digest):
        return _result(InspectionStatus.MISMATCH, "The recorded output checkpoint digest is invalid.")
    output_path = root / f"{output_digest}.checkpoint"
    upstream_digest = record.upstream_checkpoint_sha256
    upstream_path = root / f"{upstream_digest}.checkpoint"
    if output_path.is_symlink() or not output_path.is_dir():
        return _result(InspectionStatus.MISSING, "The selected store has no output artifact at the recorded content address.")
    if upstream_path.is_symlink() or not upstream_path.is_dir():
        return _result(InspectionStatus.MISSING, "The selected store has no upstream artifact at the recorded content address.")
    limits = CheckpointLimits(
        max_total_array_bytes=MAX_INSPECTED_ARRAY_BYTES,
        max_array_bytes=MAX_INSPECTED_ARRAY_BYTES,
        max_metadata_bytes=MAX_INSPECTED_METADATA_BYTES,
        max_npz_bytes=MAX_INSPECTED_ARCHIVE_BYTES,
        max_arrays=MAX_INSPECTED_ARRAYS,
        max_dimensions=8,
    )
    try:
        if _cancelled(cancel):
            return _result(InspectionStatus.CANCELLED, "Inspection was cancelled before checkpoint loading.")
        output = load_checkpoint(root, output_digest, limits=limits)
        if _cancelled(cancel):
            return _result(InspectionStatus.CANCELLED, "Inspection was cancelled after loading the output artifact.")
        metadata = output.metadata
        input_identity_payload = metadata.get("input_checkpoint_identity")
        if not isinstance(input_identity_payload, Mapping):
            raise CheckpointError("The output artifact omits its exact upstream checkpoint identity.")
        input_identity = CheckpointIdentity.from_dict(_plain(input_identity_payload))
        upstream = load_checkpoint(
            root,
            upstream_digest,
            expected_identity=input_identity,
            limits=limits,
        )
        if _cancelled(cancel):
            return _result(InspectionStatus.CANCELLED, "Inspection was cancelled after loading the upstream artifact.")
        if output.checkpoint_sha256 != output_digest or upstream.checkpoint_sha256 != upstream_digest:
            raise CheckpointError("A loaded artifact does not match its requested content address.")
        if output.execution.status is not (
            ExecutionStatus.EMPTY if record.state is StageRunState.EMPTY else ExecutionStatus.AVAILABLE
        ):
            raise CheckpointMismatchError("Checkpoint execution state differs from the attached historical run record.")
        if upstream.execution.status not in {ExecutionStatus.AVAILABLE, ExecutionStatus.EMPTY}:
            raise CheckpointMismatchError("The recorded upstream artifact is not an available image input.")
        if len(stage.source_mappings) != 1:
            raise ReferenceStageError("The preprocessing stage does not have exactly one pinned source mapping.")
        mapping = stage.source_mappings[0]
        expected_parameters = recipe.parameters_for_runner(stage.stage_id)
        expected_effective = dict(expected_parameters)
        expected_effective["callable"] = (_THRESHOLD_CALLABLES if is_threshold else _MASK_CALLABLES)[stage.operation_id]
        expected_output_roles = {output_name: output_role.to_dict()}
        checks = {
            "format": REFERENCE_STAGE_FORMAT if is_threshold else REFERENCE_MASK_STAGE_FORMAT,
            "version": REFERENCE_STAGE_VERSION if is_threshold else REFERENCE_MASK_STAGE_VERSION,
            "stage_id": stage.stage_id,
            "operation_id": stage.operation_id,
            "recipe_identity": recipe.identity.to_dict(),
            "source_mapping": mapping.to_dict(),
            "source_sha256": input_identity.source_sha256,
            "coordinate_frame": input_identity.coordinate_frame.to_dict(),
            "upstream_checkpoint_sha256": upstream_digest,
            "input_checkpoint_identity": input_identity.to_dict(),
            "output_raster_roles": expected_output_roles,
            "parameters": expected_parameters,
            "scientific_validation_status": "not_certified",
        }
        for name, expected in checks.items():
            if not _json_same(metadata.get(name), expected):
                raise CheckpointMismatchError(f"Output artifact metadata field {name!r} differs from the exact stage contract.")
        if not _effective_parameters_match(metadata.get("effective_parameters"), expected_effective):
            raise CheckpointMismatchError("Output artifact effective parameters differ from the exact selected settings.")
        bindings = metadata.get("input_bindings")
        declared_input_roles = metadata.get("input_raster_roles")
        if (
            not isinstance(bindings, Mapping)
            or set(bindings) != {input_port}
            or not isinstance(bindings.get(input_port), str)
            or not bindings[input_port]
            or not isinstance(declared_input_roles, Mapping)
            or set(declared_input_roles) != {input_port}
        ):
            raise CheckpointMismatchError("Output artifact does not declare the exact preprocessing input binding and role.")
        try:
            bound_input_role = RasterRole.from_dict(_plain(declared_input_roles[input_port]))
        except (TypeError, ValueError) as error:
            raise CheckpointMismatchError("Output artifact declares a malformed gray input raster role.") from error
        bound_input_name = bindings[input_port]
        identity_factory = expected_reference_stage_identity if is_threshold else expected_reference_mask_stage_identity
        if output.identity != identity_factory(
            input_identity, recipe, stage.stage_id, upstream_digest
        ):
            raise CheckpointMismatchError("Output checkpoint identity differs from the exact recipe/source/frame/upstream contract.")
        if output.measurement.eligible:
            raise CheckpointMismatchError("This stage bridge must not mark scientific measurements eligible.")
        if not _json_same(metadata.get("measurement_contract"), output.measurement.to_dict()):
            raise CheckpointMismatchError("Recorded measurement facts differ from the checkpoint measurement contract.")
        if _cancelled(cancel):
            return _result(InspectionStatus.CANCELLED, "Inspection was cancelled before runner identity verification.")
        raw_environment_checked = _verify_runner_identity(metadata, recipe)
        if _cancelled(cancel):
            return _result(InspectionStatus.CANCELLED, "Inspection was cancelled during runner identity verification.")
        if (
            bound_input_name not in upstream.arrays
            or upstream.raster_roles.get(bound_input_name) != bound_input_role
            or bound_input_role != input_role
        ):
            raise CheckpointMismatchError("The explicitly bound upstream array is not the declared uint8 HxW input.")
        if set(output.arrays) != {output_name} or output.raster_roles.get(output_name) != output_role:
            raise CheckpointMismatchError("The output artifact is not the declared uint8 HxW preprocessing raster.")
        gray = upstream.arrays[bound_input_name]
        mask = output.arrays[output_name]
        frame_shape = input_identity.coordinate_frame.shape
        if (
            gray.dtype != np.dtype(np.uint8) or tuple(gray.shape) != frame_shape
            or mask.dtype != np.dtype(np.uint8) or tuple(mask.shape) != frame_shape
            or gray.ndim != 2 or mask.ndim != 2
        ):
            raise CheckpointMismatchError("Input and output pixels do not match the exact uint8 HxW coordinate frame.")
        if input_role == _OUTPUT_MASK_ROLE and not np.isin(gray, (0, 255)).all():
            raise CheckpointMismatchError("Input mask contains values outside the pinned uint8 0/255 representation.")
        if not is_median:
            if mask.size and not np.isin(mask, (0, 255)).all():
                raise CheckpointMismatchError("Output mask contains values outside the pinned uint8 0/255 representation.")
            if (not np.any(mask)) != (output.execution.status is ExecutionStatus.EMPTY):
                raise CheckpointMismatchError("Recorded execution empty/available state disagrees with the verified mask pixels.")
        elif output.execution.status is not ExecutionStatus.AVAILABLE:
            raise CheckpointMismatchError("Median grayscale output must be available, including an all-zero image.")
        expected_inventory = [{
            "name": output_name,
            "dtype": mask.dtype.str,
            "shape": list(mask.shape),
            "nbytes": int(mask.nbytes),
        }]
        if not _json_same(metadata.get("output_inventory"), expected_inventory):
            raise CheckpointMismatchError("Recorded output inventory differs from the verified mask array.")
        for name in ("observations", "visualizations", "channel_semantics"):
            if not isinstance(metadata.get(name), Mapping):
                raise CheckpointMismatchError(f"Recorded {name} facts are not an object.")
        if not isinstance(metadata.get("intermediates"), (tuple, list)):
            raise CheckpointMismatchError("Recorded intermediate declarations are not an array.")
        for name in ("computation_runtime_ns", "wall_runtime_ns", "stage_runtime_ns"):
            value = metadata.get(name)
            if type(value) is not int or value < 0:
                raise CheckpointMismatchError(f"Recorded {name} is not a non-negative measured integer duration.")
        inventory = _inventory(upstream, "upstream") + _inventory(output, "output")
        return CheckpointInspection(
            status=InspectionStatus.VERIFIED,
            detail=(
                "Both content-addressed checkpoint artifacts, the exact stage/run contract, "
                "source/frame identity, pinned PlantCV runner identity and dependency inventory, "
                "parameters and raster roles were verified. The captured raw environment manifest "
                "was crosschecked when available. "
                "This source binding is checkpoint-derived; the active desktop image was not independently matched."
            ),
            checkpoint_sha256=output_digest,
            source_sha256=output.identity.source_sha256,
            coordinate_frame=output.identity.coordinate_frame,
            execution_status=output.execution.status,
            array_inventory=inventory,
            compact_metadata=_compact_metadata(
                metadata, raw_environment_checked=raw_environment_checked
            ),
            preview_arrays={f"input.{input_port}": gray, f"output.{output_name}": mask},
        )
    except InvalidCheckpoint as error:
        return _result(InspectionStatus.MISMATCH, f"Checkpoint content or bounded array contract failed verification: {error}")
    except CheckpointMismatchError as error:
        return _result(InspectionStatus.MISMATCH, str(error))
    except CheckpointError as error:
        return _result(InspectionStatus.MISMATCH, f"Checkpoint contract mismatch: {error}")
    except (ReferenceStageError, RecipeCatalogueError, RecipeContractError, StaleRecipeIdentity, KeyError, TypeError, ValueError, OSError, OverflowError) as error:
        return _result(InspectionStatus.MISMATCH, f"Reference stage or source identity mismatch: {error}")


__all__ = [
    "CheckpointInspection",
    "InspectionStatus",
    "MAX_INSPECTED_ARRAY_BYTES",
    "MAX_INSPECTED_ARCHIVE_BYTES",
    "MAX_PREVIEW_PIXELS",
    "inspect_reference_checkpoint",
]
