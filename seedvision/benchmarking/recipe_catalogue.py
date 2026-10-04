"""Audited, declarative seed-workflow recipes from pinned references.

The catalogue contains operation contracts and graph topology only. It does
not execute PlantCV, recover the SMPTS capsule, or claim conformance/scientific
validity. PlantCV operation nodes are mapped to the audited 4.11.3 source. Its
watershed sub-stages are a source-inspection view of one public PlantCV call;
the isolated runner currently executes that call as ``segmentation.watershed``
and does not expose independent internal checkpoint outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from seedvision.benchmarking.recipes import (
    BranchCondition,
    DependencyContract,
    LayoutPosition,
    ParameterContract,
    PortContract,
    RecipeContractError,
    RecipeGraphMismatch,
    RecipeProvenance,
    RecipeProvenanceMode,
    RecipeSpec,
    RecipeStage,
    SourceMapping,
    SourceResolution,
    to_pipeline_graph,
    validate_pipeline_graph,
)


CATALOGUE_SCHEMA = "seedfiddle.reference-recipe-catalogue"
CATALOGUE_VERSION = "0.1.0"
CATALOGUE_PATH = "config/peer_reference_sources.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

HASH_POLICY = (
    "source_manifest_sha256 is SHA-256 over UTF-8 canonical JSON (sorted keys, "
    "compact separators, no NaN) of the reference-specific committed-config "
    "projection returned by source_manifest_projection(); PlantCV projection "
    "pins release, commit, retrieval date, license, and every listed source-file "
    "path/URL/hash; SMPTS projection pins manuscript identity/hash and the "
    "recorded original-capsule availability/ambiguity fields. The PlantCV "
    "dependency lock is bound separately by its committed SHA-256."
)

_EXECUTION_CLAIMS = {
    "runnable": False,
    "run_evidence": [],
    "statement": (
        "No recipe was executed; no recipe output hashes, conformance results, "
        "or scientific results are included. Source and dependency pins do not "
        "establish algorithm conformance or biological accuracy."
    ),
}

_SCOPE = {
    "accepted_feature_families": [f"F{i:02d}" for i in range(1, 23)],
    "catalogue_slice": [
        "F03.2.correct_color: explicit source-to-target correction operation contract only",
        "F04.1", "F04.2", "F04.3", "F07.1", "F07.2", "F07.3",
        "F09.1", "F10.1", "F11.1", "F15.1", "F17.1",
        "F22.1 recipe data and PipelineGraph projection contract only",
        "SMPTS paper-stage reconstruction contract, separate from unavailable original-code parity",
    ],
    "remaining_scope_is_still_accepted": True,
    "accepted_core_work_not_closed_by_this_slice": [
        "F01 decoding/channel/ROI/coordinate contracts and F02 physical calibration",
        "F03.1 neutral balance and remaining F03.2 auto card detection, matrix, and affine-correction paths",
        "F04.4 channel/range thresholding",
        "F06 original two-branch implementation parity; this catalogue only records unresolved paper-stage branches",
        "F08 exact component geometry and rectangle/contact-screening behavior",
        "F09.2 SMPTS targeted-component distance/morphology inputs",
        "F10.2 complete marker metadata and F11.2 returned diagnostic/result products",
        "F12.1 executable SMPTS targeted erosion and F14.1 individual crop/mask extraction",
        "F16 reviewed canonical geometry; F19 review/QC/export; F20 batch/resource controls; F21 conformance/science evaluation",
        "F22 native desktop views, checkpoint inspection, persistence, cache isolation, and Qt visual acceptance",
    ],
    "extended_seed_operations_remain_in_scope": [
        "PlantCV Gaussian, triangle, dual-channel and texture threshold variants",
        "PlantCV naive-Bayes and k-means classifiers with declared training access",
        "F18 reviewed per-seed semantic/hilum annotation and optional automatic trait classifiers",
        "remaining F01-F22 input, extraction, QC, export, execution, replay and science gates",
    ],
    "claim_boundary": (
        "This first catalogue slice does not close or exclude any accepted F01-F22 "
        "family or extended seed operation."
    ),
}


class RecipeCatalogueError(ValueError):
    """A catalogue, committed reference pin, or declared recipe is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise RecipeCatalogueError("Catalogue data is not strict canonical JSON.") from error


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_source_config() -> dict[str, Any]:
    path = _repo_root() / CATALOGUE_PATH
    try:
        raw = path.read_bytes()
        if len(raw) > 2 * 1024 * 1024:
            raise RecipeCatalogueError("Committed source configuration exceeds size limit.")
        config = _strict_json_loads(raw.decode("utf-8"))
    except RecipeCatalogueError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RecipeCatalogueError(f"Cannot read committed source configuration {path}.") from error
    if not isinstance(config, dict):
        raise RecipeCatalogueError("Committed source configuration must be an object.")
    return config


def _strict_json_loads(raw: str | bytes) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RecipeCatalogueError(f"Duplicate JSON field {key!r}.")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise RecipeCatalogueError(f"Non-finite JSON constant {value!r} is forbidden.")

    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def source_manifest_projection(config: Mapping[str, Any], reference: str) -> dict[str, Any]:
    """Return the stable, reference-specific projection hashed by this catalogue."""
    if reference == "plantcv":
        files = config.get("files")
        license_data = config.get("license")
        if not isinstance(files, list) or not isinstance(license_data, Mapping):
            raise RecipeCatalogueError("PlantCV source configuration is incomplete.")
        records = []
        for item in files:
            if not isinstance(item, Mapping):
                raise RecipeCatalogueError("PlantCV source file record is invalid.")
            path, url, digest = item.get("path"), item.get("url"), item.get("sha256")
            if not all(isinstance(part, str) and part for part in (path, url, digest)):
                raise RecipeCatalogueError("PlantCV source file record is missing path, URL, or hash.")
            if not _SHA256.fullmatch(digest):
                raise RecipeCatalogueError(f"PlantCV source hash is invalid for {path!r}.")
            records.append({"path": path, "url": url, "sha256": digest})
        records.sort(key=lambda item: item["path"])
        return {
            "reference": config.get("reference"),
            "version": config.get("version"),
            "commit": config.get("commit"),
            "retrieved_date": config.get("retrieved_date"),
            "license": license_data.get("plantcv"),
            "files": records,
        }
    if reference == "smpts_paper":
        paper = config.get("smpts")
        if not isinstance(paper, Mapping):
            raise RecipeCatalogueError("SMPTS paper source configuration is incomplete.")
        fields = (
            "reference", "version", "paper_url", "paper_sha256", "original_capsule",
            "original_status", "checked_date", "original_execution_verified", "ambiguous_operations",
        )
        result = {key: paper.get(key) for key in fields}
        if result["reference"] != "SMPTS" or not _SHA256.fullmatch(str(result["paper_sha256"])):
            raise RecipeCatalogueError("SMPTS paper identity or SHA-256 pin is invalid.")
        return result
    raise RecipeCatalogueError(f"Unknown source-manifest projection {reference!r}.")


def source_manifest_sha256(config: Mapping[str, Any], reference: str) -> str:
    return _sha256(_canonical_bytes(source_manifest_projection(config, reference)))


def _plantcv_file(config: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    for record in config.get("files", ()):
        if isinstance(record, Mapping) and record.get("path") == path:
            return record
    raise RecipeCatalogueError(f"No committed PlantCV source pin for {path!r}.")


def _plantcv_provenance(config: Mapping[str, Any]) -> RecipeProvenance:
    environment = config.get("environment")
    license_data = config.get("license")
    if not isinstance(environment, Mapping) or not isinstance(license_data, Mapping):
        raise RecipeCatalogueError("PlantCV environment or license pin is missing.")
    packages = environment.get("packages")
    if not isinstance(packages, list):
        raise RecipeCatalogueError("PlantCV environment packages must be listed.")
    versions = {
        str(item["name"]): str(item["version"])
        for item in packages
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        and isinstance(item.get("version"), str)
    }
    if len(versions) != len(packages):
        raise RecipeCatalogueError("PlantCV dependency version inventory is incomplete or duplicated.")
    return RecipeProvenance(
        mode=RecipeProvenanceMode.ORIGINAL_REFERENCE,
        backend_id="plantcv.reference",
        backend_version=str(config["version"]),
        reference_id="plantcv",
        reference_version=str(config["version"]),
        reference_commit=str(config["commit"]),
        source_manifest_sha256=source_manifest_sha256(config, "plantcv"),
        environment_lock_sha256=str(environment["lockfile_sha256"]),
        environment_versions=versions,
        license_id=str(license_data["plantcv"]),
        evidence_status=SourceResolution.PINNED,
        evidence_note=(
            "PlantCV release source files and dependency archives are pinned by "
            f"{CATALOGUE_PATH}; runtime is Python {environment['python']} on "
            f"{environment['platform']}; no recipe execution, native conformance, or scientific "
            "validation is asserted. SeedFiddle authored this operation order; it is "
            "not claimed as a workflow published by PlantCV."
        ),
    )


def _plantcv_mapping(
    config: Mapping[str, Any], path: str, symbol: str, *, note: str = "",
) -> SourceMapping:
    pin = _plantcv_file(config, path)
    return SourceMapping(
        reference_id="plantcv",
        reference_version=str(config["version"]),
        resolution=SourceResolution.PINNED,
        source_path=path,
        source_symbol=symbol,
        source_url=str(pin["url"]),
        source_sha256=str(pin["sha256"]),
        license_id=str(config["license"]["plantcv"]),
        note=note,
    )


def _unselected(parameter_id: str, description: str, *, editable: bool = False) -> ParameterContract:
    """Represent an unsupplied required recipe value as null, never a guess."""
    return ParameterContract(
        parameter_id=parameter_id,
        value_type="json",
        value=None,
        affects_calculation=True,
        editable=editable,
        description=f"UNSELECTED: {description}",
    )


def _image_semantics(color: str) -> dict[str, Any]:
    return {"color_order": color, "dtype": "uint8", "shape": "HxW" if color == "GRAY" else "HxWx3", "range": [0, 255]}


def _gray_output(stage_id: str, label: str = "Grayscale image") -> PortContract:
    return PortContract("image", "output", "GrayImage", label, semantics=_image_semantics("GRAY"))


def _gray_input(*, external: bool = False, label: str = "Grayscale image") -> PortContract:
    return PortContract("gray", "input", "GrayImage", label, external=external, semantics=_image_semantics("GRAY"))


def _mask_semantics() -> dict[str, Any]:
    return {"dtype": "uint8", "range": [0, 255], "foreground": 255, "background": 0, "shape": "HxW"}


def _label_semantics() -> dict[str, Any]:
    return {"dtype": "int32", "background": 0, "positive_values": "instance labels", "shape": "HxW"}


def _plantcv_input_stage(prefix: str) -> RecipeStage:
    return RecipeStage(
        stage_id=f"{prefix}_external_inputs",
        title="External prepared image inputs",
        operation_id="external.image_inputs",
        description=(
            "Contract boundary only. Caller supplies uint8 grayscale and RGB arrays. "
            "This recipe does not decode, convert channels, deskew, rectify, or apply "
            "color correction; the grayscale derivation and its relation to RGB must "
            "be recorded by the caller."
        ),
        ports=(
            PortContract("gray_image", "output", "GrayImage", "External grayscale input", semantics=_image_semantics("GRAY")),
            PortContract(
                "rgb_image", "output", "RGBImage", "External RGB input",
                semantics={**_image_semantics("RGB"), "channel_boundary": "RGB uint8 HWC"},
            ),
        ),
        resolution=SourceResolution.UNRESOLVED,
        resolution_note="External decoding, channel preparation, and input provenance are not executed in this recipe.",
    )


def _plantcv_seed_recipe(threshold_kind: str) -> RecipeSpec:
    config = _load_source_config()
    provenance = _plantcv_provenance(config)
    prefix = f"pcv_seed_{threshold_kind}"
    gray = _image_semantics("GRAY")
    rgb = {**_image_semantics("RGB"), "channel_boundary": "RGB uint8 HWC"}
    mask = _mask_semantics()
    labels = _label_semantics()

    median = RecipeStage(
        stage_id=f"{prefix}_median_blur",
        title="Median blur",
        operation_id="filter.median_blur",
        description=(
            "SeedFiddle-selected preprocessing order: median filter before threshold. "
            "This order is not a PlantCV-published seed workflow. The PlantCV API "
            "requires ksize and supplies no default; no fixture-grounded value is frozen."
        ),
        ports=(_gray_input(), _gray_output(prefix)),
        parameters=(_unselected(
            "ksize",
            "required kernel; the PlantCV API accepts integer or tuple, while the current isolated runner accepts an odd integer only; choose from the frozen acquisition scale and fixture",
        ),),
        source_mappings=(_plantcv_mapping(
            config, "plantcv/plantcv/median_blur.py", "plantcv.plantcv.median_blur.median_blur",
            note="Pinned API; ksize has no function default.",
        ),),
    )

    threshold_path = "plantcv/plantcv/threshold/threshold_methods.py"
    if threshold_kind == "binary":
        threshold_operation = "threshold.binary"
        threshold_title = "Fixed binary threshold"
        threshold_symbol = "plantcv.plantcv.threshold.binary"
        threshold_parameters = (
            _unselected("threshold", "required integer cutoff in [0,255]; no published seed-run cutoff selected"),
            ParameterContract(
                "object_type", "string", "light", True, False,
                description="PlantCV API default; polarity needs a documented image/channel boundary and fixture review.",
                choices=("light", "dark"),
            ),
        )
        threshold_note = "Fixed cutoff is a required function argument; the catalogue leaves it null rather than inventing a seed value."
    elif threshold_kind == "otsu":
        threshold_operation = "threshold.otsu"
        threshold_title = "Otsu global threshold"
        threshold_symbol = "plantcv.plantcv.threshold.otsu"
        threshold_parameters = (
            ParameterContract(
                "object_type", "string", "light", True, False,
                description="Pinned PlantCV API default; not validated for a selected seed image.",
                choices=("light", "dark"),
            ),
        )
        threshold_note = "Pinned call applies OpenCV Otsu with threshold argument zero and the selected light/dark polarity."
    elif threshold_kind == "mean":
        threshold_operation = "threshold.mean"
        threshold_title = "Adaptive mean threshold"
        threshold_symbol = "plantcv.plantcv.threshold.mean"
        threshold_parameters = (
            _unselected("ksize", "required block size >=3; PlantCV forces even sizes to odd, while the runner rejects even values"),
            _unselected("offset", "required adaptive-threshold offset; no published seed-run value selected"),
            ParameterContract(
                "object_type", "string", "light", True, False,
                description="Pinned PlantCV API default; not validated for a selected seed image.",
                choices=("light", "dark"),
            ),
        )
        threshold_note = "Pinned adaptive mean call; block size and offset have no function defaults and remain unselected."
    else:
        raise RecipeCatalogueError(f"Unsupported seed threshold family {threshold_kind!r}.")

    threshold = RecipeStage(
        stage_id=f"{prefix}_{threshold_kind}_threshold",
        title=threshold_title,
        operation_id=threshold_operation,
        description=(
            "SeedFiddle-authored threshold arm on the median-filtered external grayscale "
            "input. It is an individual method arm, not a combination of threshold masks. "
            f"{threshold_note}"
        ),
        ports=(
            PortContract("gray", "input", "GrayImage", "Median-filtered grayscale", semantics=gray),
            PortContract("mask", "output", "BinaryMask", "0/255 foreground mask", semantics=mask),
        ),
        parameters=threshold_parameters,
        source_mappings=(_plantcv_mapping(config, threshold_path, threshold_symbol, note=threshold_note),),
    )

    fill = RecipeStage(
        stage_id=f"{prefix}_small_object_filter",
        title="Small-component filter (PlantCV fill)",
        operation_id="morphology.fill",
        description=(
            "PlantCV fill removes small foreground objects; it is not hole filling. "
            "SeedFiddle selects it after threshold and before morphology. Required pixel "
            "area has no source default and remains unselected."
        ),
        ports=(
            PortContract("mask", "input", "BinaryMask", "Threshold mask", semantics=mask),
            PortContract("mask", "output", "BinaryMask", "Filtered mask", semantics=mask),
        ),
        parameters=(_unselected("size", "required minimum object area in pixels; no fixture-grounded cutoff selected"),),
        source_mappings=(_plantcv_mapping(
            config, "plantcv/plantcv/fill.py", "plantcv.plantcv.fill.fill",
            note="PlantCV fill removes small objects; it does not fill holes.",
        ),),
    )

    erode = RecipeStage(
        stage_id=f"{prefix}_erode",
        title="Binary erosion",
        operation_id="morphology.erode",
        description="SeedFiddle-authored square-kernel erosion stage; kernel and iteration values are not frozen.",
        ports=(
            PortContract("gray", "input", "BinaryMask", "Filtered mask", semantics=mask),
            PortContract("mask", "output", "BinaryMask", "Eroded mask", semantics=mask),
        ),
        parameters=(
            _unselected("ksize", "required square kernel size >1; no source default"),
            _unselected("iterations", "required number of passes; public reference runner uses this name and maps it to PlantCV i"),
        ),
        source_mappings=(_plantcv_mapping(
            config, "plantcv/plantcv/erode.py", "plantcv.plantcv.erode.erode",
            note="PlantCV callable parameter is i; isolated runner contract names it iterations.",
        ),),
    )

    dilate = RecipeStage(
        stage_id=f"{prefix}_dilate",
        title="Binary dilation",
        operation_id="morphology.dilate",
        description="SeedFiddle-authored square-kernel dilation stage after erosion; kernel and iteration values are not frozen.",
        ports=(
            PortContract("gray", "input", "BinaryMask", "Eroded mask", semantics=mask),
            PortContract("mask", "output", "BinaryMask", "Prepared foreground mask", semantics=mask),
        ),
        parameters=(
            _unselected("ksize", "required square kernel size >1; no source default"),
            _unselected("iterations", "required number of passes; public reference runner uses this name and maps it to PlantCV i"),
        ),
        source_mappings=(_plantcv_mapping(
            config, "plantcv/plantcv/dilate.py", "plantcv.plantcv.dilate.dilate",
            note="PlantCV callable parameter is i; isolated runner contract names it iterations.",
        ),),
    )

    watershed_group = (
        "Audited internal sub-stage of the single PlantCV 4.11.3 public call "
        "segmentation.watershed / watershed_segmentation. Current runner executes "
        "that original callable once and does not independently replay or checkpoint "
        "this internal result; future instrumentation must retain one-call semantics. "
        "The one call-level `distance` parameter is declared on local-maximum selection, "
        "the only sub-stage it changes."
    )
    watershed_file = "plantcv/plantcv/watershed.py"
    distance = RecipeStage(
        stage_id=f"{prefix}_ws_distance_map",
        title="Watershed distance map",
        operation_id="reference.internal.watershed.distance_map",
        description=watershed_group + " `distanceTransformWithLabels(mask, DIST_L2, maskSize=0)[0]`; returned labels are discarded.",
        ports=(
            PortContract("mask", "input", "BinaryMask", "Prepared foreground mask", semantics=mask),
            PortContract("distance", "output", "DistanceMap", "Euclidean distance map", semantics={"dtype": "float", "unit": "pixel", "source": "cv2.distanceTransformWithLabels, DIST_L2, maskSize=0"}),
        ),
        source_mappings=(_plantcv_mapping(
            config, watershed_file, "plantcv.plantcv.watershed.watershed_segmentation: cv2.distanceTransformWithLabels",
            note="Pinned source line 47; internal view only, not a second execution call.",
        ),),
    )
    peaks = RecipeStage(
        stage_id=f"{prefix}_ws_local_maxima",
        title="Local maxima",
        operation_id="reference.internal.watershed.local_maxima",
        description=watershed_group + " `peak_local_max(dist_transform, min_distance=distance, labels=mask)` then coordinates are painted into a boolean peak raster.",
        ports=(
            PortContract("distance", "input", "DistanceMap", "Distance map", semantics={"dtype": "float", "unit": "pixel", "source": "cv2.distanceTransformWithLabels, DIST_L2, maskSize=0"}),
            PortContract("mask", "input", "BinaryMask", "Prepared foreground mask", semantics=mask),
            PortContract("coordinates", "output", "PeakCoordinates", "Peak coordinates", semantics={"coordinate_order": "row,column", "selection": "pinned scikit-image peak_local_max defaults"}),
            PortContract("peaks", "output", "PeakMask", "Boolean local-maximum raster", semantics={"dtype": "bool", "background": False}),
        ),
        parameters=(ParameterContract(
            "distance", "integer", 10, True, False, units="pixels",
            description="PlantCV watershed_segmentation API default passed as peak_local_max min_distance.",
            minimum=1,
        ),),
        source_mappings=(_plantcv_mapping(
            config, watershed_file, "plantcv.plantcv.watershed.watershed_segmentation: peak_local_max and local_max raster assignment",
            note="Pinned source lines 49-51; library behavior is pinned by environment lock.",
        ),),
    )
    markers = RecipeStage(
        stage_id=f"{prefix}_ws_markers",
        title="Connected markers",
        operation_id="reference.internal.watershed.connected_markers",
        description=(
            watershed_group + " `ndi.label(local_max, structure=np.ones((3,3)))[0]` creates 8-connected markers. "
            "The marker-count observation is calculated after flooding and emitted later in the same public call; it is not the final label count."
        ),
        ports=(
            PortContract("peaks", "input", "PeakMask", "Boolean local maxima", semantics={"dtype": "bool", "background": False}),
            PortContract("markers", "output", "MarkerLabels", "8-connected marker labels", semantics={"dtype": "int32", "background": 0, "connectivity": 8}),
        ),
        source_mappings=(_plantcv_mapping(
            config, watershed_file,
            "plantcv.plantcv.watershed.watershed_segmentation: ndi.label(..., structure=np.ones((3,3)))",
            note="Pinned source line 53; internal view only.",
        ),),
    )
    inverted = RecipeStage(
        stage_id=f"{prefix}_ws_invert_distance",
        title="Negated distance map",
        operation_id="reference.internal.watershed.negate_distance",
        description=watershed_group + " `dist_transform1 = -dist_transform`; no new distance field is calculated.",
        ports=(
            PortContract("distance", "input", "DistanceMap", "Distance map", semantics={"dtype": "float", "unit": "pixel", "source": "cv2.distanceTransformWithLabels, DIST_L2, maskSize=0"}),
            PortContract("negative_distance", "output", "NegativeDistanceMap", "Negated distance map", semantics={"dtype": "float", "formula": "-distance"}),
        ),
        source_mappings=(_plantcv_mapping(config, watershed_file, "plantcv.plantcv.watershed.watershed_segmentation: dist_transform1 = -dist_transform", note="Pinned source line 54; internal view only."),),
    )
    flood = RecipeStage(
        stage_id=f"{prefix}_ws_masked_flood",
        title="Masked watershed flood",
        operation_id="reference.internal.watershed.masked_flood",
        description=(
            watershed_group + " `skimage.segmentation.watershed(-distance, markers, mask=mask)` returns "
            "integer instance labels. The public function also consumes RGB for its diagnostic overlay."
        ),
        ports=(
            PortContract("negative_distance", "input", "NegativeDistanceMap", "Negated distance map", semantics={"dtype": "float", "formula": "-distance"}),
            PortContract("markers", "input", "MarkerLabels", "Marker labels", semantics={"dtype": "int32", "background": 0, "connectivity": 8}),
            PortContract("mask", "input", "BinaryMask", "Prepared foreground mask", semantics=mask),
            PortContract("rgb", "input", "RGBImage", "External RGB for public-call diagnostic", semantics=rgb),
            PortContract("labels", "output", "InstanceLabels", "Watershed label raster", semantics=labels),
            PortContract("flood_complete", "output", "OperationCompletion", "Control token ordering post-flood observation", semantics={"role": "sequencing only", "numeric_payload": False}),
        ),
        source_mappings=(_plantcv_mapping(
            config, watershed_file, "plantcv.plantcv.watershed.watershed_segmentation: skimage.segmentation.watershed",
            note="Pinned source line 55; internal stage is not independently executed by the current runner.",
        ),),
    )
    marker_count = RecipeStage(
        stage_id=f"{prefix}_ws_marker_count_observation",
        title="Marker-count observation",
        operation_id="reference.internal.watershed.marker_count_observation",
        description=(
            watershed_group + " After the masked flood, PlantCV calculates "
            "`len(np.unique(markers)) - 1` and later adds `estimated_object_count` "
            "to the public call's observations. This count uses connected marker labels, "
            "not watershed labels; the flood-complete input encodes ordering only."
        ),
        ports=(
            PortContract("markers", "input", "MarkerLabels", "Connected marker labels", semantics={"dtype": "int32", "background": 0, "connectivity": 8}),
            PortContract("flood_complete", "input", "OperationCompletion", "Control token from masked flood", semantics={"role": "sequencing only", "numeric_payload": False}),
            PortContract("marker_count", "output", "Observation", "PlantCV estimated_object_count", semantics={"formula": "unique(marker_labels) - 1", "derived_from": "connected marker labels", "not_derived_from": "watershed labels", "timing": "calculated after flood; emitted later in the same public call", "method": "plantcv.plantcv.watershed"}),
        ),
        source_mappings=(_plantcv_mapping(
            config, watershed_file,
            "plantcv.plantcv.watershed.watershed_segmentation: estimated_object_count and outputs.add_observation",
            note="Pinned source line 56 calculates the count after the line-55 flood; lines 76-78 emit it later. Count derives from markers, not final watershed labels.",
        ),),
        parameters=(ParameterContract(
            "label", "string", "SeedFiddle", True, False,
            description="Isolated runner's explicit observation sample label for the public watershed call.",
        ),),
    )

    size = RecipeStage(
        stage_id=f"{prefix}_analyze_size",
        title="PlantCV size and shape observations",
        operation_id="analysis.size",
        description=(
            "Pinned analyze.size emits per-label observations and a diagnostic image. "
            "Runner input must be the caller's RGB uint8 image (explicitly converted "
            "to PlantCV BGR) or grayscale; analysis remains pixel-scaled. Dynamic "
            "n_labels must be bound to the actual label raster before execution."
        ),
        ports=(
            PortContract("image", "input", "RGBImage", "External RGB source image", semantics=rgb),
            PortContract("labels", "input", "InstanceLabels", "Watershed labels", semantics=labels),
            PortContract(
                "observations", "output", "SizeObservations", "Per-label size/shape observations",
                semantics={
                    "variables": [
                        "area", "convex_hull_area", "solidity", "perimeter", "total_edge_length",
                        "width", "height", "longest_path", "center_of_mass", "convex_hull_vertices",
                        "object_in_frame", "ellipse_center", "ellipse_major_axis", "ellipse_minor_axis",
                        "ellipse_angle", "ellipse_eccentricity",
                    ],
                    "units": "pixels unless trait is unitless/degrees",
                    "physical_calibration": "not supplied",
                    "definitions": "PlantCV 4.11.3 analyze.size; not substituted with SeedFiddle canonical geometry",
                },
            ),
            PortContract("diagnostic", "output", "RGBImage", "Size-analysis overlay", semantics={"dtype": "uint8", "runner_channel_order": "RGB", "source_image": "caller RGB boundary"}),
        ),
        parameters=(
            _unselected("n_labels", "positive per-image object count required by PlantCV/runner; bind from the actual labels"),
            ParameterContract("label", "string", "SeedFiddle", True, False, description="Explicit isolated-runner observation label; overrides PlantCV's global sample label."),
        ),
        source_mappings=(_plantcv_mapping(
            config, "plantcv/plantcv/analyze/size.py", "plantcv.plantcv.analyze.size.size and _analyze_size",
            note="Pinned API lines 12-39 and observations at lines 148-196; runner reports pixels and converts RGB boundary to/from PlantCV BGR.",
        ),),
    )
    color = RecipeStage(
        stage_id=f"{prefix}_analyze_color",
        title="PlantCV per-label colour observations",
        operation_id="analysis.color",
        description=(
            "Pinned analyze.color emits selected RGB/Lab/HSV histograms, HSV summaries, "
            "circular hue statistics and a histogram visualization. `all` is explicitly "
            "selected to include all documented spaces. Source documentation says RGB; "
            "source code uses OpenCV BGR conversions, so the catalogue records the "
            "runner's RGB-to-BGR adapter and does not claim biological colour validity."
        ),
        ports=(
            PortContract("rgb", "input", "RGBImage", "External RGB source image", semantics=rgb),
            PortContract("labels", "input", "InstanceLabels", "Watershed labels", semantics=labels),
            PortContract(
                "observations", "output", "ColorObservations", "Per-label histograms and summaries",
                semantics={
                    "variables": [
                        "blue_frequencies", "green_frequencies", "red_frequencies",
                        "lightness_frequencies", "green-magenta_frequencies", "blue-yellow_frequencies",
                        "saturation_mean", "saturation_median", "value_mean", "value_median",
                        "hue_frequencies", "saturation_frequencies", "value_frequencies",
                        "hue_circular_mean", "hue_circular_std", "hue_median",
                    ],
                    "spaces": ["rgb", "lab", "hsv"],
                    "histogram_bins": 256,
                    "hue_frequency_bins": 180,
                    "hue_encoding": "PlantCV OpenCV hue 0-179; emitted frequency bins are 0-179 and hue labels use degrees x2",
                    "science": "not validated",
                },
            ),
            PortContract("histogram", "output", "Visualization", "Hue histogram diagnostic", semantics={"source_return": "PlantCV analyze.color hue_chart"}),
        ),
        parameters=(
            _unselected("n_labels", "positive per-image object count required by PlantCV/runner; bind from the actual labels"),
            ParameterContract("colorspaces", "string", "all", True, False, description="Explicitly selected to request all supported RGB, Lab and HSV output groups.", choices=("all", "rgb", "lab", "hsv")),
            ParameterContract("label", "string", "SeedFiddle", True, False, description="Explicit isolated-runner observation label; overrides PlantCV's global sample label."),
        ),
        source_mappings=(_plantcv_mapping(
            config, "plantcv/plantcv/analyze/color.py", "plantcv.plantcv.analyze.color.color and _analyze_color",
            note="Pinned API lines 14-42; BGR conversion/output observations at lines 104-211; runner explicitly adapts RGB boundaries to BGR.",
        ),),
    )

    stages = (
        _plantcv_input_stage(prefix), median, threshold, fill, erode, dilate,
        distance, peaks, markers, inverted, flood, marker_count, size, color,
    )
    dependencies = (
        DependencyContract(f"{prefix}_external_inputs", "gray_image", median.stage_id, "gray", "GrayImage"),
        DependencyContract(median.stage_id, "image", threshold.stage_id, "gray", "GrayImage"),
        DependencyContract(threshold.stage_id, "mask", fill.stage_id, "mask", "BinaryMask"),
        DependencyContract(fill.stage_id, "mask", erode.stage_id, "gray", "BinaryMask"),
        DependencyContract(erode.stage_id, "mask", dilate.stage_id, "gray", "BinaryMask"),
        DependencyContract(dilate.stage_id, "mask", distance.stage_id, "mask", "BinaryMask"),
        DependencyContract(distance.stage_id, "distance", peaks.stage_id, "distance", "DistanceMap"),
        DependencyContract(dilate.stage_id, "mask", peaks.stage_id, "mask", "BinaryMask"),
        DependencyContract(peaks.stage_id, "peaks", markers.stage_id, "peaks", "PeakMask"),
        DependencyContract(distance.stage_id, "distance", inverted.stage_id, "distance", "DistanceMap"),
        DependencyContract(inverted.stage_id, "negative_distance", flood.stage_id, "negative_distance", "NegativeDistanceMap"),
        DependencyContract(markers.stage_id, "markers", flood.stage_id, "markers", "MarkerLabels"),
        DependencyContract(dilate.stage_id, "mask", flood.stage_id, "mask", "BinaryMask"),
        DependencyContract(flood.stage_id, "labels", size.stage_id, "labels", "InstanceLabels"),
        DependencyContract(flood.stage_id, "labels", color.stage_id, "labels", "InstanceLabels"),
        DependencyContract(markers.stage_id, "markers", marker_count.stage_id, "markers", "MarkerLabels"),
        DependencyContract(flood.stage_id, "flood_complete", marker_count.stage_id, "flood_complete", "OperationCompletion"),
        DependencyContract(f"{prefix}_external_inputs", "rgb_image", size.stage_id, "image", "RGBImage"),
        DependencyContract(f"{prefix}_external_inputs", "rgb_image", flood.stage_id, "rgb", "RGBImage"),
        DependencyContract(f"{prefix}_external_inputs", "rgb_image", color.stage_id, "rgb", "RGBImage"),
    )
    layout = tuple(LayoutPosition(stage.stage_id, float(index * 240), 0) for index, stage in enumerate(stages))
    return RecipeSpec(
        recipe_id=f"plantcv_seed_{threshold_kind}",
        recipe_version="1.0.0",
        title=f"PlantCV seed workflow — {threshold_title}",
        description=(
            "SeedFiddle-authored stage order using pinned PlantCV 4.11.3 functions; "
            "PlantCV did not publish this exact end-to-end seed workflow. Required "
            "values with no documented source default remain explicit null/unselected. "
            "This recipe has not been executed and is not a conformance or science claim. "
            "External decoded grayscale/RGB boundaries are caller supplied."
        ),
        provenance=provenance,
        stages=stages,
        dependencies=dependencies,
        layout=layout,
    )


def _color_correction_recipe() -> RecipeSpec:
    config = _load_source_config()
    provenance = _plantcv_provenance(config)
    rgb = {**_image_semantics("RGB"), "channel_boundary": "runner accepts RGB uint8 and explicitly converts to PlantCV BGR"}
    mask = {"dtype": "integer", "background": 0, "foreground": "distinct label per color chip", "shape": "HxW"}
    matrix = {"dtype": "float", "shape": "Nx4", "columns": ["chip_number", "mean_R", "mean_G", "mean_B"], "RGB_scale": "normalized [0,1]"}
    stage = RecipeStage(
        stage_id="pcv_color_correct_color",
        title="PlantCV source-to-target colour-card correction",
        operation_id="transform.correct_color",
        description=(
            "Exact pinned correct_color API using externally supplied target/source RGB "
            "images and masks with uniquely labeled chip regions. No image decoder, card "
            "detector, matrix source, orientation choice, or reference run is included. "
            "The isolated runner converts RGB boundaries to PlantCV BGR and adapts the "
            "returned corrected raster and matrix data."
        ),
        ports=(
            PortContract("target_rgb", "input", "RGBImage", "Target reference image", external=True, semantics=rgb),
            PortContract("target_mask", "input", "ChipLabelMask", "Target chip label mask", external=True, semantics=mask),
            PortContract("source_rgb", "input", "RGBImage", "Image to correct", external=True, semantics=rgb),
            PortContract("source_mask", "input", "ChipLabelMask", "Source chip label mask", external=True, semantics=mask),
            PortContract("corrected_image", "output", "RGBImage", "Corrected RGB boundary image", semantics={**rgb, "runner_output_adapter": "PlantCV BGR back to RGB"}),
            PortContract("target_matrix", "output", "ColorMatrix", "Measured target chips", semantics=matrix),
            PortContract("source_matrix", "output", "ColorMatrix", "Measured source chips", semantics=matrix),
            PortContract("transformation_matrix", "output", "ColorTransform", "Fitted polynomial transform", semantics={"shape": "9x9", "meaning": "PlantCV correct_color result"}),
        ),
        source_mappings=(_plantcv_mapping(
            config, "plantcv/plantcv/transform/color_correction.py", "plantcv.plantcv.transform.color_correction.correct_color",
            note="Pinned source lines 446-496; isolated runner maps to transform.correct_color and provides a temporary output directory.",
        ),),
    )
    return RecipeSpec(
        recipe_id="plantcv_color_card_correct_color",
        recipe_version="1.0.0",
        title="PlantCV colour-card correction operation",
        description=(
            "Single-operation reference contract for PlantCV correct_color. This is a "
            "color-calibration stage, not a complete seed-analysis workflow. Matrices, "
            "chip masks, RGB channel handling, and target definition require external "
            "review; no run or held-out patch validation is claimed."
        ),
        provenance=provenance,
        stages=(stage,),
        dependencies=(),
        layout=(LayoutPosition(stage.stage_id, 0, 0),),
    )


def _paper_mapping(config: Mapping[str, Any], *, note: str) -> tuple[SourceMapping, SourceMapping]:
    paper = config["smpts"]
    return (
        SourceMapping(
            reference_id="smpts_paper",
            reference_version=str(paper["version"]),
            resolution=SourceResolution.UNRESOLVED,
            source_url=str(paper["paper_url"]),
            note=note,
        ),
        SourceMapping(
            reference_id="smpts_original",
            reference_version="v1",
            resolution=SourceResolution.UNAVAILABLE,
            source_url=str(paper["original_capsule"]),
            note=(
                "Original CodeOcean page/API returned HTTP 403 on " + str(paper["checked_date"])
                + "; source bytes, capsule license, execution and parity remain unavailable."
            ),
        ),
    )


def _smpts_paper_recipe() -> RecipeSpec:
    config = _load_source_config()
    paper = config["smpts"]
    provenance = RecipeProvenance(
        mode=RecipeProvenanceMode.PAPER_RECONSTRUCTION,
        backend_id="smpts.paper_port.contract",
        backend_version="0.1.0-provisional",
        reference_id="smpts_paper",
        reference_version=str(paper["version"]),
        source_manifest_sha256=source_manifest_sha256(config, "smpts_paper"),
        evidence_status=SourceResolution.PINNED,
        evidence_note=(
            f"Manuscript v2 SHA-256 is pinned; original capsule {paper['original_status']} "
            f"on {paper['checked_date']}. This paper reconstruction is a stage contract only, "
            "not recovered source, not an executable implementation, and not an original-code parity claim."
        ),
    )
    image_raster = {
        "raster_kind": "threshold input image",
        "shape": "unresolved",
        "dtype": "unresolved",
        "channel_conversion": "unresolved",
        "intensity_range": "unresolved",
    }
    mask = {
        "raster_kind": "candidate foreground mask",
        "shape": "unresolved",
        "dtype": "unresolved",
        "foreground_value": "unresolved",
        "background_value": "unresolved",
        "polarity": "unresolved",
        "channel_conversion": "unresolved",
    }
    cores = {
        "raster_kind": "conditional eroded regions",
        "shape": "unresolved",
        "dtype": "unresolved",
        "mask_polarity": "unresolved",
        "not_measurement_mask": True,
    }
    rectangles = {"coordinate_frame": "unresolved", "rectangle_format": "unresolved", "decision_rule": "unresolved"}
    stage_mappings = lambda detail: _paper_mapping(config, note=detail)
    branch_a = RecipeStage(
        stage_id="smpts_paper_branch_a_threshold",
        title="Threshold branch A — algorithm unresolved",
        operation_id="paper_reconstruction.threshold_branch_a",
        description="The manuscript describes a first threshold-derived foreground branch; exact fixed/Otsu/local method, polarity and parameters remain ambiguous.",
        ports=(
            PortContract("image", "input", "ImageRaster", "External threshold input image", external=True, semantics=image_raster),
            PortContract("mask", "output", "BinaryMask", "Branch A mask", semantics=mask),
        ),
        parameters=(
            _unselected("method", "branch algorithm not resolved from manuscript (Otsu versus fixed-threshold ambiguity)"),
            _unselected("polarity", "foreground polarity/channel conversion not specified conclusively"),
            _unselected("threshold_settings", "cutoff/local method/block/offset not resolved"),
        ),
        source_mappings=stage_mappings("Paper stage is described at high level; operation details are ambiguous and original code is unavailable."),
        resolution=SourceResolution.UNRESOLVED,
        resolution_note="Paper supports a threshold-derived branch but not an exact executable operation or parameters.",
    )
    branch_b = RecipeStage(
        stage_id="smpts_paper_branch_b_threshold",
        title="Threshold branch B — algorithm unresolved",
        operation_id="paper_reconstruction.threshold_branch_b",
        description="The manuscript describes a second threshold-derived branch; branch-specific method, polarity, channel conversion and settings are unresolved.",
        ports=(
            PortContract("image", "input", "ImageRaster", "External threshold input image", external=True, semantics=image_raster),
            PortContract("mask", "output", "BinaryMask", "Branch B mask", semantics=mask),
        ),
        parameters=(
            _unselected("method", "branch algorithm and relation to Otsu/fixed/local thresholds are ambiguous"),
            _unselected("polarity", "foreground polarity/channel conversion not specified conclusively"),
            _unselected("threshold_settings", "cutoff/local method/block/offset not resolved"),
        ),
        source_mappings=stage_mappings("Paper stage is described at high level; operation details are ambiguous and original code is unavailable."),
        resolution=SourceResolution.UNRESOLVED,
        resolution_note="Paper supports a second threshold-derived branch but not an exact executable operation or parameters.",
    )
    filter_a = RecipeStage(
        stage_id="smpts_paper_branch_a_filter",
        title="Filter branch A",
        operation_id="paper_reconstruction.filter_branch_a",
        description="Filtering is a manuscript stage; operation, branch order, kernel, border handling and parameters are unresolved.",
        ports=(PortContract("mask", "input", "BinaryMask", "Branch A mask", semantics=mask), PortContract("filtered", "output", "BinaryMask", "Filtered branch A", semantics=mask)),
        parameters=(_unselected("filter_contract", "filter method, kernel, border behavior and exact parameters are unresolved"),),
        source_mappings=stage_mappings("Paper describes branch filtering; exact filter call and source code are unavailable."),
        resolution=SourceResolution.UNRESOLVED,
        resolution_note="The manuscript does not freeze a reproducible filter contract here.",
    )
    filter_b = RecipeStage(
        stage_id="smpts_paper_branch_b_filter",
        title="Filter branch B",
        operation_id="paper_reconstruction.filter_branch_b",
        description="Filtering is a manuscript stage; operation, branch order, kernel, border handling and parameters are unresolved.",
        ports=(PortContract("mask", "input", "BinaryMask", "Branch B mask", semantics=mask), PortContract("filtered", "output", "BinaryMask", "Filtered branch B", semantics=mask)),
        parameters=(_unselected("filter_contract", "filter method, kernel, border behavior and exact parameters are unresolved"),),
        source_mappings=stage_mappings("Paper describes branch filtering; exact filter call and source code are unavailable."),
        resolution=SourceResolution.UNRESOLVED,
        resolution_note="The manuscript does not freeze a reproducible filter contract here.",
    )
    combine = RecipeStage(
        stage_id="smpts_paper_mask_combine",
        title="Combine filtered foreground branches",
        operation_id="paper_reconstruction.mask_combine",
        description="The manuscript combines filtered branch masks; the boolean/weighted rule, polarity and handling of disagreement are unresolved. No AND/OR rule is assumed.",
        ports=(
            PortContract("branch_a", "input", "BinaryMask", "Filtered branch A", semantics=mask),
            PortContract("branch_b", "input", "BinaryMask", "Filtered branch B", semantics=mask),
            PortContract("combined", "output", "BinaryMask", "Combined foreground", semantics=mask),
        ),
        parameters=(_unselected("combination_rule", "mask-combination operation is ambiguous in the available paper evidence"),),
        source_mappings=stage_mappings("Paper-level combination step only; exact operation and original-code behavior unavailable."),
        resolution=SourceResolution.UNRESOLVED,
        resolution_note="Combination semantics must not be guessed from two input branches.",
    )
    components = RecipeStage(
        stage_id="smpts_paper_contact_rectangles",
        title="Component and rectangle localization",
        operation_id="paper_reconstruction.component_rectangles",
        description="Paper describes bounding-rectangle localization for contact screening; exact connected-component definition and rectangle semantics are unresolved.",
        ports=(
            PortContract("mask", "input", "BinaryMask", "Combined foreground", semantics=mask),
            PortContract("rectangles", "output", "RectangleCandidates", "Component rectangles", semantics=rectangles),
        ),
        parameters=(_unselected("component_contract", "connectivity, rectangle coordinates and candidate geometry are not pinned"),),
        source_mappings=stage_mappings("Manuscript-level bounding-rectangle stage; no recovered code symbol or exact thresholds."),
        resolution=SourceResolution.UNRESOLVED,
        resolution_note="Rectangle-based contact localization is described but not executable from the available details.",
    )
    contact = RecipeStage(
        stage_id="smpts_paper_contact_screen",
        title="Conditional contact screening",
        operation_id="paper_reconstruction.contact_screen",
        description="A rectangle-based decision determines targeted processing; contact thresholds and decision reasons remain unresolved.",
        ports=(
            PortContract("rectangles", "input", "RectangleCandidates", "Component rectangles", semantics=rectangles),
            PortContract("contacts", "output", "ContactCandidates", "Selected contact rectangles", semantics={"decision_rule": "unresolved", "thresholds": "unresolved"}),
        ),
        parameters=(_unselected("contact_rule", "rectangle screening thresholds/decision rule are not resolved"),),
        source_mappings=stage_mappings("Paper describes contact screening; original source unavailable and decision thresholds ambiguous."),
        resolution=SourceResolution.UNRESOLVED,
        resolution_note="Contact/no-contact criterion is not recoverable from the manuscript description alone.",
    )
    erode = RecipeStage(
        stage_id="smpts_paper_targeted_erosion",
        title="Conditional targeted erosion",
        operation_id="paper_reconstruction.targeted_erosion",
        description="Erosion applies only when contact screening selects a target; kernel shape/size, iterations, border policy and recovery semantics are unresolved.",
        ports=(
            PortContract("mask", "input", "BinaryMask", "Combined foreground", semantics=mask),
            PortContract("contacts", "input", "ContactCandidates", "Selected contact rectangles", semantics={"decision_rule": "unresolved", "thresholds": "unresolved"}),
            PortContract("cores", "output", "ErodedCores", "Eroded target regions", semantics=cores),
        ),
        parameters=(
            _unselected("kernel_shape", "erosion kernel shape is ambiguous"),
            _unselected("kernel_size", "erosion kernel size is not supplied"),
            _unselected("iterations", "erosion iteration count is not supplied"),
            _unselected("border_policy", "erosion border behavior is not supplied"),
        ),
        source_mappings=stage_mappings("Paper-stage erosion only; no PlantCV erosion is substituted and no exact code source is claimed."),
        resolution=SourceResolution.UNRESOLVED,
        resolution_note="The manuscript supports targeted erosion, but its exact operator parameters and boundary semantics remain unresolved.",
        condition=BranchCondition("contact_screen selects one or more candidates under its unresolved rule", "Contact candidate selected"),
    )
    extract = RecipeStage(
        stage_id="smpts_paper_extract_seeds",
        title="Individual seed extraction and crops",
        operation_id="paper_reconstruction.extract_seed_crops",
        description="The paper describes individual-seed extraction; crop padding, retained masks and dense-instance output semantics are not inferred.",
        ports=(
            PortContract("image", "input", "ImageRaster", "External image source", external=True, semantics={**image_raster, "channel_order": "unresolved"}),
            PortContract("mask", "input", "BinaryMask", "Combined foreground", semantics=mask),
            PortContract("contacts", "input", "ContactCandidates", "Selected contact rectangles", semantics={"decision_rule": "unresolved", "thresholds": "unresolved"}),
            PortContract("cores", "input", "ErodedCores", "Conditional eroded regions", semantics=cores),
            PortContract("crops", "output", "SeedCrops", "Extracted individual image crops", semantics={"crop_padding": "unresolved", "mask_output": "not inferred"}),
            PortContract("rectangles", "output", "CropRectangles", "Crop-to-source rectangles", semantics={"coordinate_frame": "unresolved", "padding": "unresolved"}),
            PortContract("status", "output", "ExtractionStatus", "Extraction success/failure status", semantics={"failure_accounting": "unresolved"}),
        ),
        parameters=(_unselected("extraction_contract", "crop selection, padding, mask retention, and dense-instance semantics are unresolved"),),
        source_mappings=stage_mappings("Paper describes individual-seed extraction; exact output products and original implementation remain unavailable."),
        resolution=SourceResolution.UNRESOLVED,
        resolution_note="Crop and dense-mask semantics cannot be reconstructed or claimed from the available paper evidence.",
    )
    stages = (branch_a, branch_b, filter_a, filter_b, combine, components, contact, erode, extract)
    dependencies = (
        DependencyContract(branch_a.stage_id, "mask", filter_a.stage_id, "mask", "BinaryMask"),
        DependencyContract(branch_b.stage_id, "mask", filter_b.stage_id, "mask", "BinaryMask"),
        DependencyContract(filter_a.stage_id, "filtered", combine.stage_id, "branch_a", "BinaryMask"),
        DependencyContract(filter_b.stage_id, "filtered", combine.stage_id, "branch_b", "BinaryMask"),
        DependencyContract(combine.stage_id, "combined", components.stage_id, "mask", "BinaryMask"),
        DependencyContract(components.stage_id, "rectangles", contact.stage_id, "rectangles", "RectangleCandidates"),
        DependencyContract(combine.stage_id, "combined", erode.stage_id, "mask", "BinaryMask"),
        DependencyContract(contact.stage_id, "contacts", erode.stage_id, "contacts", "ContactCandidates"),
        DependencyContract(combine.stage_id, "combined", extract.stage_id, "mask", "BinaryMask"),
        DependencyContract(contact.stage_id, "contacts", extract.stage_id, "contacts", "ContactCandidates"),
        DependencyContract(erode.stage_id, "cores", extract.stage_id, "cores", "ErodedCores"),
    )
    layout_positions = {
        branch_a.stage_id: LayoutPosition(branch_a.stage_id, 0, 0),
        branch_b.stage_id: LayoutPosition(branch_b.stage_id, 0, 220),
        filter_a.stage_id: LayoutPosition(filter_a.stage_id, 260, 0),
        filter_b.stage_id: LayoutPosition(filter_b.stage_id, 260, 220),
        combine.stage_id: LayoutPosition(combine.stage_id, 520, 110),
        components.stage_id: LayoutPosition(components.stage_id, 780, 110),
        contact.stage_id: LayoutPosition(contact.stage_id, 1040, 110),
        erode.stage_id: LayoutPosition(erode.stage_id, 1300, 110),
        extract.stage_id: LayoutPosition(extract.stage_id, 1560, 110),
    }
    return RecipeSpec(
        recipe_id="smpts_paper_reconstruction",
        recipe_version="0.1.0-provisional",
        title="SMPTS manuscript-stage reconstruction contract",
        description=(
            "A provisional graph of recoverable manuscript-level stages only. "
            "This is not the unavailable original CodeOcean implementation. Threshold "
            "choices, filtering/combine rules, rectangle criteria, erosion settings, "
            "channel handling, crop semantics and executable behavior remain unresolved."
        ),
        provenance=provenance,
        stages=stages,
        dependencies=dependencies,
        layout=tuple(layout_positions[item.stage_id] for item in stages),
    )


@dataclass(frozen=True, slots=True)
class RecipeCatalogue:
    version: str
    recipes: tuple[RecipeSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version or self.version != self.version.strip():
            raise RecipeCatalogueError("Catalogue version must be non-empty trimmed text.")
        recipes = tuple(self.recipes)
        if not recipes or any(not isinstance(item, RecipeSpec) for item in recipes):
            raise RecipeCatalogueError("Catalogue requires RecipeSpec entries.")
        ids = [item.recipe_id for item in recipes]
        if len(ids) != len(set(ids)):
            raise RecipeCatalogueError("Catalogue recipe IDs must be unique.")
        object.__setattr__(self, "recipes", tuple(sorted(recipes, key=lambda item: item.recipe_id)))

    def recipe(self, recipe_id: str) -> RecipeSpec:
        for recipe in self.recipes:
            if recipe.recipe_id == recipe_id:
                return recipe
        raise KeyError(recipe_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CATALOGUE_SCHEMA,
            "version": self.version,
            "source_manifest_hash_policy": HASH_POLICY,
            "feature_scope": json.loads(_canonical_bytes(_SCOPE)),
            "execution_claims": json.loads(_canonical_bytes(_EXECUTION_CLAIMS)),
            "recipes": [recipe.to_dict() for recipe in self.recipes],
        }

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @property
    def sha256(self) -> str:
        return _sha256(self.to_json().encode("utf-8"))

    @classmethod
    def from_json(cls, value: str | bytes) -> "RecipeCatalogue":
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise RecipeCatalogueError("Catalogue JSON must be UTF-8.") from error
        if not isinstance(value, str):
            raise RecipeCatalogueError("Catalogue JSON must be bounded UTF-8 text.")
        try:
            encoded_size = len(value.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise RecipeCatalogueError("Catalogue JSON must be valid UTF-8 text.") from error
        if encoded_size > 4 * 1024 * 1024:
            raise RecipeCatalogueError("Catalogue JSON must be bounded UTF-8 text.")
        try:
            data = _strict_json_loads(value)
        except (json.JSONDecodeError, RecipeCatalogueError) as error:
            raise RecipeCatalogueError(f"Invalid strict catalogue JSON: {error}") from error
        expected_fields = {
            "schema", "version", "source_manifest_hash_policy", "feature_scope",
            "execution_claims", "recipes",
        }
        if not isinstance(data, dict) or set(data) != expected_fields or data.get("schema") != CATALOGUE_SCHEMA:
            raise RecipeCatalogueError("Catalogue JSON has unknown fields or an unsupported schema.")
        if data["source_manifest_hash_policy"] != HASH_POLICY or data["feature_scope"] != _SCOPE:
            raise RecipeCatalogueError("Catalogue scope or source-manifest hash policy differs from this version.")
        if not isinstance(data["recipes"], list):
            raise RecipeCatalogueError("Catalogue recipes must be an array.")
        if data["execution_claims"] != _EXECUTION_CLAIMS:
            raise RecipeCatalogueError("Catalogue run evidence or execution claims are not the frozen empty state.")
        return cls(data["version"], tuple(RecipeSpec.from_json(_canonical_bytes(item)) for item in data["recipes"]))


def validate_recipe_catalogue(catalogue: RecipeCatalogue) -> None:
    """Audit source pins, declared graphs, round-trips, and honest run states."""
    if not isinstance(catalogue, RecipeCatalogue):
        raise TypeError("catalogue must be a RecipeCatalogue.")
    config = _load_source_config()
    if catalogue.version != CATALOGUE_VERSION:
        raise RecipeCatalogueError("Catalogue version differs from the frozen catalogue implementation.")
    required_ids = {
        "plantcv_seed_binary", "plantcv_seed_otsu", "plantcv_seed_mean",
        "plantcv_color_card_correct_color", "smpts_paper_reconstruction",
    }
    if {recipe.recipe_id for recipe in catalogue.recipes} != required_ids:
        raise RecipeCatalogueError("Catalogue recipe set is missing, extra, or renamed.")
    plantcv_manifest = source_manifest_sha256(config, "plantcv")
    paper_manifest = source_manifest_sha256(config, "smpts_paper")
    environment = config["environment"]
    expected_versions = {str(item["name"]): str(item["version"]) for item in environment["packages"]}
    lock_path = _repo_root() / str(environment["lockfile"])
    try:
        lock_sha = _sha256(lock_path.read_bytes())
    except OSError as error:
        raise RecipeCatalogueError(f"Cannot read the committed dependency lock {lock_path}.") from error
    if lock_sha != str(environment["lockfile_sha256"]):
        raise RecipeCatalogueError("Committed PlantCV dependency lock differs from its source-config pin.")
    file_pins = {str(item["path"]): item for item in config["files"]}
    for recipe in catalogue.recipes:
        restored = RecipeSpec.from_json(recipe.to_json())
        if restored != recipe:
            raise RecipeCatalogueError(f"Recipe {recipe.recipe_id!r} failed its strict JSON round-trip.")
        graph = to_pipeline_graph(recipe)
        if any(node.status.value == "complete" for node in graph.nodes.values()):
            raise RecipeCatalogueError(f"Recipe {recipe.recipe_id!r} has unsupported completed run state.")
        if recipe.recipe_id == "smpts_paper_reconstruction":
            if recipe.provenance.mode is not RecipeProvenanceMode.PAPER_RECONSTRUCTION:
                raise RecipeCatalogueError("SMPTS paper graph must remain paper-reconstruction mode.")
            if recipe.provenance.reference_id != "smpts_paper" or recipe.provenance.source_manifest_sha256 != paper_manifest:
                raise RecipeCatalogueError("SMPTS paper graph differs from the pinned manuscript source.")
            if recipe.provenance.evidence_status is not SourceResolution.PINNED:
                raise RecipeCatalogueError("SMPTS manuscript identity must remain explicitly pinned.")
            if config["smpts"]["original_execution_verified"] is not False:
                raise RecipeCatalogueError("The current source record no longer says original SMPTS execution is unavailable.")
            paper = config["smpts"]
            if not isinstance(paper, Mapping):
                raise RecipeCatalogueError("SMPTS source record is invalid.")
            for stage in recipe.stages:
                mappings = {mapping.reference_id: mapping for mapping in stage.source_mappings}
                manuscript = mappings.get("smpts_paper")
                original = mappings.get("smpts_original")
                if (
                    len(stage.source_mappings) != 2
                    or
                    manuscript is None
                    or manuscript.resolution is not SourceResolution.UNRESOLVED
                    or manuscript.reference_version != str(paper["version"])
                    or manuscript.source_url != str(paper["paper_url"])
                    or original is None
                    or original.resolution is not SourceResolution.UNAVAILABLE
                    or original.reference_version != "v1"
                    or original.source_url != str(paper["original_capsule"])
                    or str(paper["checked_date"]) not in original.note
                    or ("http_403" in str(paper["original_status"]) and "HTTP 403" not in original.note)
                ):
                    raise RecipeCatalogueError(
                        f"SMPTS stage {stage.stage_id!r} must keep pinned-paper reconstruction separate from unavailable original code."
                    )
                if stage.resolution is not SourceResolution.UNRESOLVED:
                    raise RecipeCatalogueError("Every SMPTS paper-reconstruction stage must remain unresolved.")
        else:
            provenance = recipe.provenance
            if (
                provenance.mode is not RecipeProvenanceMode.ORIGINAL_REFERENCE
                or provenance.backend_id != "plantcv.reference"
                or provenance.backend_version != str(config["version"])
                or provenance.reference_id != "plantcv"
                or provenance.reference_version != str(config["version"])
                or provenance.reference_commit != str(config["commit"])
                or provenance.source_manifest_sha256 != plantcv_manifest
                or provenance.environment_lock_sha256 != str(environment["lockfile_sha256"])
                or provenance.license_id != str(config["license"]["plantcv"])
                or provenance.evidence_status is not SourceResolution.PINNED
                or dict(provenance.environment_versions) != expected_versions
                or f"Python {environment['python']} on {environment['platform']}" not in provenance.evidence_note
            ):
                raise RecipeCatalogueError(f"Recipe {recipe.recipe_id!r} differs from PlantCV source/environment pins.")
            for stage in recipe.stages:
                if stage.operation_id == "external.image_inputs":
                    if stage.source_mappings or stage.resolution is not SourceResolution.UNRESOLVED:
                        raise RecipeCatalogueError("External PlantCV image input must remain unresolved without source decoding.")
                    continue
                if not stage.source_mappings or any(
                    mapping.reference_id != "plantcv" or mapping.resolution is not SourceResolution.PINNED
                    for mapping in stage.source_mappings
                ):
                    raise RecipeCatalogueError(f"PlantCV stage {stage.stage_id!r} needs exact pinned source mappings.")
                for mapping in stage.source_mappings:
                    if mapping.reference_id == "plantcv" and mapping.resolution is SourceResolution.PINNED:
                        pin = file_pins.get(mapping.source_path)
                        if (
                            not isinstance(pin, Mapping)
                            or pin.get("sha256") != mapping.source_sha256
                            or pin.get("url") != mapping.source_url
                            or mapping.license_id != config["license"]["plantcv"]
                            or mapping.reference_version != str(config["version"])
                        ):
                            raise RecipeCatalogueError(
                                f"Source mapping for {stage.stage_id!r} differs from committed source config."
                            )
        try:
            validate_pipeline_graph(recipe, graph)
        except (RecipeContractError, RecipeGraphMismatch) as error:
            raise RecipeCatalogueError(f"Recipe graph {recipe.recipe_id!r} failed exact validation.") from error


def build_recipe_catalogue() -> RecipeCatalogue:
    """Build the fixed recipe slice from committed reference-source pins."""
    catalogue = RecipeCatalogue(
        version=CATALOGUE_VERSION,
        recipes=(
            _plantcv_seed_recipe("binary"),
            _plantcv_seed_recipe("otsu"),
            _plantcv_seed_recipe("mean"),
            _color_correction_recipe(),
            _smpts_paper_recipe(),
        ),
    )
    validate_recipe_catalogue(catalogue)
    return catalogue
