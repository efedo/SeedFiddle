"""Strict, atomic persistence for immutable species-reference libraries."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Mapping
import zipfile
import zlib

import numpy as np

from seedvision.reference_library.contracts import (
    BiologicalContext,
    LibraryArraySpec,
    LibraryProduct,
    LibrarySourceRecord,
    LibraryStatus,
    ProductValidation,
    ShapeObservation,
    ShapePopulationComponent,
    SpeciesDimensionsShapeBank,
    SpeciesEdgePrototypeBank,
    SpeciesForegroundColourBank,
    SpeciesForegroundNoiseBank,
    SpeciesLibraryArtifact,
    SpeciesLibraryManifest,
    SpeciesLibraryPin,
    SpeciesMaterialPrototypeBank,
    SpeciesSeedTraitBank,
    SpeciesShapeSummary,
    SourceProfileBank,
    SourcePrototypeBank,
    ValidationTier,
    SPECIES_LIBRARY_EXTENSION,
    SPECIES_LIBRARY_FORMAT,
    SPECIES_LIBRARY_MAX_ARRAY_BYTES,
    SPECIES_LIBRARY_SCHEMA_VERSION,
)

MANIFEST_NAME = "manifest.json"
BANKS_NAME = "banks.npz"
PROVENANCE_NAME = "provenance.npz"
INDEX_NAME = "index.json"
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
MAX_ZIP_RATIO = 200.0


class SpeciesLibraryError(ValueError):
    pass


class InvalidSpeciesLibrary(SpeciesLibraryError):
    pass


class SpeciesLibraryIOError(SpeciesLibraryError):
    pass


class SpeciesLibraryNotFound(SpeciesLibraryError):
    pass


def default_species_library_root() -> Path:
    """Resolve the per-user root through Qt without making Qt a package import."""

    try:
        from PySide6.QtCore import QStandardPaths
        base = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        )
    except Exception:
        base = ""
    if not base:
        base = str(Path.home() / ".seed-fiddle")
    return Path(base) / "species-libraries"


class SpeciesLibraryStore:
    """Publish, resolve, import, export, and retire immutable library versions."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = (
            default_species_library_root() if root is None else Path(root)
        ).resolve()

    def version_path(self, species_id: str, library_id: str, version: str) -> Path:
        _safe_token(species_id, "species ID")
        _safe_token(library_id, "library ID", numeric_first=True)
        _safe_token(version, "version", numeric_first=True)
        return self.root / species_id / library_id / version

    def publish(self, artifact: SpeciesLibraryArtifact) -> SpeciesLibraryArtifact:
        arrays, bank_metadata = _artifact_arrays(artifact)
        specs = tuple(
            _array_spec(name, values) for name, values in sorted(arrays.items())
        )
        manifest = replace(
            artifact.manifest,
            status=LibraryStatus.PUBLISHED,
            arrays=specs,
            content_sha256="",
        )
        digest = _content_digest(_manifest_payload(manifest, bank_metadata), arrays)
        manifest = replace(manifest, content_sha256=digest)
        finalized = replace(artifact, manifest=manifest)
        destination = self.version_path(
            manifest.species_id, manifest.library_id, manifest.version
        )
        if destination.exists():
            try:
                return self.load(
                    SpeciesLibraryPin(
                        manifest.library_id,
                        manifest.version,
                        manifest.species_id,
                        digest,
                    )
                )
            except SpeciesLibraryError as error:
                raise SpeciesLibraryIOError(
                    "Published library versions are immutable and this version already exists."
                ) from error
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{manifest.version}.", dir=destination.parent)
        )
        try:
            _write_json(temporary / MANIFEST_NAME, _manifest_payload(manifest, bank_metadata))
            with (temporary / BANKS_NAME).open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            with (temporary / PROVENANCE_NAME).open("wb") as stream:
                np.savez_compressed(
                    stream,
                    source_sha256=np.asarray(
                        [item.source_sha256 for item in manifest.sources], dtype="U64"
                    ),
                    annotation_sha256=np.asarray(
                        [item.annotation_sha256 for item in manifest.sources], dtype="U64"
                    ),
                )
            _load_directory(temporary, None)
            os.replace(temporary, destination)
            self._update_index(manifest, retired=False)
            return finalized
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise

    def load(self, pin: SpeciesLibraryPin) -> SpeciesLibraryArtifact:
        source = self.version_path(pin.species_id, pin.library_id, pin.version)
        if not source.is_dir():
            raise SpeciesLibraryNotFound(
                f"Pinned species library {pin.library_id} version {pin.version} is not installed."
            )
        return _load_directory(source, pin)

    def list_manifests(
        self, species_id: str | None = None
    ) -> tuple[SpeciesLibraryManifest, ...]:
        roots = (
            (self.root / species_id,)
            if species_id is not None
            else tuple(path for path in self.root.glob("*") if path.is_dir())
        )
        manifests = []
        for species_root in roots:
            if not species_root.is_dir():
                continue
            retired_keys: set[tuple[str, str]] = set()
            index_path = species_root / INDEX_NAME
            if index_path.is_file():
                try:
                    index_payload = _read_json(index_path)
                    retired_keys = {
                        (str(item.get("library_id")), str(item.get("version")))
                        for item in index_payload.get("versions", ())
                        if isinstance(item, dict) and bool(item.get("retired"))
                    }
                except SpeciesLibraryError:
                    retired_keys = set()
            for path in species_root.glob(f"*/*/{MANIFEST_NAME}"):
                try:
                    manifest = _load_directory(path.parent, None).manifest
                    if (manifest.library_id, manifest.version) in retired_keys:
                        manifest = replace(manifest, status=LibraryStatus.RETIRED)
                    manifests.append(manifest)
                except SpeciesLibraryError:
                    continue
        return tuple(
            sorted(
                manifests,
                key=lambda item: (item.species_id, item.library_id, item.version),
            )
        )

    def retire(self, pin: SpeciesLibraryPin, *, retired: bool = True) -> None:
        self._update_index(self.load(pin).manifest, retired=retired)

    def export_library(self, pin: SpeciesLibraryPin, destination: Path | str) -> Path:
        self.load(pin)
        source = self.version_path(pin.species_id, pin.library_id, pin.version)
        target = Path(destination)
        if target.suffix.lower() != ".zip":
            target = target.with_name(target.name + SPECIES_LIBRARY_EXTENSION)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for name in (MANIFEST_NAME, BANKS_NAME, PROVENANCE_NAME):
                    archive.write(source / name, arcname=name)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target

    def import_bundle(self, source: Path | str) -> SpeciesLibraryArtifact:
        bundle = Path(source)
        if not bundle.is_file() or bundle.stat().st_size > MAX_BUNDLE_BYTES:
            raise InvalidSpeciesLibrary("Species-library bundle is missing or too large.")
        import_root = self.root / ".imports"
        import_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix="library.", dir=import_root))
        try:
            with zipfile.ZipFile(bundle, "r") as archive:
                infos = archive.infolist()
                if {item.filename for item in infos} != {
                    MANIFEST_NAME, BANKS_NAME, PROVENANCE_NAME
                }:
                    raise InvalidSpeciesLibrary(
                        "Portable library has an unexpected file set."
                    )
                for info in infos:
                    _safe_archive_member(info)
                    with archive.open(info, "r") as input_stream, (
                        temporary / info.filename
                    ).open("wb") as output_stream:
                        shutil.copyfileobj(
                            input_stream, output_stream, length=1024 * 1024
                        )
            return self.publish(_load_directory(temporary, None))
        except (zipfile.BadZipFile, OSError) as error:
            raise InvalidSpeciesLibrary(
                f"Could not import species library: {error}"
            ) from error
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def _update_index(
        self, manifest: SpeciesLibraryManifest, *, retired: bool
    ) -> None:
        species_root = self.root / manifest.species_id
        species_root.mkdir(parents=True, exist_ok=True)
        path = species_root / INDEX_NAME
        entries = []
        if path.is_file():
            try:
                payload = _read_json(path)
                entries = list(payload.get("versions", ()))
            except SpeciesLibraryError:
                entries = []
        key = (manifest.library_id, manifest.version)
        entries = [
            item
            for item in entries
            if isinstance(item, dict)
            and (item.get("library_id"), item.get("version")) != key
        ]
        entries.append(
            {
                "library_id": manifest.library_id,
                "version": manifest.version,
                "sha256": manifest.content_sha256,
                "retired": bool(retired),
                "created_utc": manifest.created_utc,
            }
        )
        _write_json(
            path,
            {
                "format": "seedfiddle-species-library-index",
                "version": 1,
                "versions": entries,
            },
        )


def create_manifest(
    *,
    library_id: str,
    version: str,
    species_id: str,
    species_display_name: str,
    trait_vocabulary_sha256: str,
    seed_fiddle_version: str,
    sources: tuple[LibrarySourceRecord, ...],
    products: tuple[ProductValidation, ...],
    descriptor_schemas: Mapping[str, str],
    extraction_settings_sha256: str,
    aggregation_settings_sha256: str,
    corrected_colour_space_version: str,
    parent_content_sha256: str | None = None,
) -> SpeciesLibraryManifest:
    return SpeciesLibraryManifest(
        library_id, version, "", species_id, species_display_name,
        trait_vocabulary_sha256, seed_fiddle_version,
        datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        LibraryStatus.DRAFT, sources, products, descriptor_schemas,
        extraction_settings_sha256, aggregation_settings_sha256,
        corrected_colour_space_version, (), parent_content_sha256,
    )


def _artifact_arrays(
    artifact: SpeciesLibraryArtifact,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays: dict[str, np.ndarray] = {}
    banks: dict[str, object] = {}

    def profile(name: str, bank: SourceProfileBank | None) -> None:
        if bank is None:
            return
        prefix = f"{name}."
        arrays.update(
            {
                prefix + "centres": bank.centres,
                prefix + "scales": bank.scales,
                prefix + "weights": bank.weights,
                prefix + "source_indices": bank.source_indices,
                prefix + "sample_counts": bank.sample_counts,
                prefix + "half_distances": bank.half_distances,
            }
        )
        banks[name] = {
            "kind": "profile",
            "schema_id": bank.schema_id,
            "feature_names": list(bank.feature_names),
        }

    def prototype(name: str, bank: SourcePrototypeBank | None) -> None:
        if bank is None:
            return
        prefix = f"{name}."
        arrays.update(
            {
                prefix + "centres": bank.centres,
                prefix + "scales": bank.scales,
                prefix + "weights": bank.weights,
                prefix + "source_indices": bank.source_indices,
                prefix + "seed_ids": bank.seed_ids,
                prefix + "class_ids": bank.class_ids,
                prefix + "sample_counts": bank.sample_counts,
            }
        )
        banks[name] = {
            "kind": "prototype",
            "schema_id": bank.schema_id,
            "feature_names": list(bank.feature_names),
            "class_names": list(bank.class_names),
        }

    profile("foreground_colour", artifact.foreground_colour)
    profile("foreground_noise", artifact.foreground_noise)
    prototype("material_prototypes", artifact.material_prototypes)
    prototype("edge_prototypes", artifact.edge_prototypes)
    prototype("seed_traits", artifact.seed_traits)
    for name, value in (
        ("shape_summary", artifact.shape_summary),
        ("dimensions_shape", artifact.dimensions_shape),
    ):
        if value is None:
            continue
        prefix = f"{name}."
        observations = value.observations
        dimension = len(value.measurement_names)
        arrays[prefix + "measurements"] = np.asarray(
            [item.measurement for item in observations], np.float64
        ).reshape(len(observations), dimension)
        arrays[prefix + "uncertainties"] = np.asarray(
            [item.measurement_uncertainty for item in observations], np.float64
        ).reshape(len(observations), dimension)
        arrays[prefix + "source_indices"] = np.asarray(
            [item.source_index for item in observations], np.int32
        )
        arrays[prefix + "seed_ids"] = np.asarray(
            [item.seed_id for item in observations], np.int32
        )
        contour_length = max(
            (len(item.contour_signature) for item in observations), default=0
        )
        if any(
            len(item.contour_signature) not in {0, contour_length}
            for item in observations
        ):
            raise InvalidSpeciesLibrary(
                "Shape contour signatures must use one shared sample count."
            )
        arrays[prefix + "contours"] = np.asarray(
            [
                item.contour_signature
                if item.contour_signature
                else (0.0,) * contour_length
                for item in observations
            ],
            np.float32,
        ).reshape(len(observations), contour_length)
        banks[name] = {
            "kind": name,
            "schema_id": value.schema_id,
            "measurement_names": list(value.measurement_names),
            "observations": [
                _shape_observation_metadata(item) for item in observations
            ],
        }
        if isinstance(value, SpeciesDimensionsShapeBank):
            banks[name]["components"] = [
                _shape_component_payload(item) for item in value.components
            ]
            banks[name]["fit_settings"] = {
                "minimum_component_seeds": value.minimum_component_seeds,
                "shrinkage_seed_count": value.shrinkage_seed_count,
                "maximum_contour_modes": value.maximum_contour_modes,
            }
    return (
        {name: np.ascontiguousarray(value) for name, value in arrays.items()},
        {"banks": banks},
    )


def _load_directory(
    source: Path, expected_pin: SpeciesLibraryPin | None
) -> SpeciesLibraryArtifact:
    manifest, bank_metadata = _manifest_from_payload(
        _read_json(source / MANIFEST_NAME)
    )
    if expected_pin is not None and (
        manifest.library_id,
        manifest.version,
        manifest.species_id,
        manifest.content_sha256,
    ) != (
        expected_pin.library_id,
        expected_pin.version,
        expected_pin.species_id,
        expected_pin.sha256,
    ):
        raise InvalidSpeciesLibrary(
            "Installed species library does not match the exact project pin."
        )
    try:
        with np.load(source / BANKS_NAME, allow_pickle=False) as archive:
            if set(archive.files) != {item.name for item in manifest.arrays}:
                raise InvalidSpeciesLibrary(
                    "Library numeric array set does not match its manifest."
                )
            arrays = {
                name: np.asarray(archive[name]).copy() for name in archive.files
            }
    except InvalidSpeciesLibrary:
        raise
    except (
        OSError,
        ValueError,
        EOFError,
        zipfile.BadZipFile,
        zlib.error,
    ) as error:
        raise InvalidSpeciesLibrary(
            f"Could not read library numeric banks: {error}"
        ) from error
    for spec in manifest.arrays:
        if _array_spec(spec.name, arrays[spec.name]) != spec:
            raise InvalidSpeciesLibrary(
                f"Library array {spec.name!r} fails its dtype/shape/hash contract."
            )
    calculated = _content_digest(
        _manifest_payload(replace(manifest, content_sha256=""), bank_metadata),
        arrays,
    )
    if calculated != manifest.content_sha256:
        raise InvalidSpeciesLibrary("Species library content hash is invalid.")
    return _artifact_from_arrays(manifest, bank_metadata, arrays)


def _artifact_from_arrays(manifest, metadata, arrays):
    banks = metadata.get("banks")
    if not isinstance(banks, dict):
        raise InvalidSpeciesLibrary("Library bank metadata is invalid.")

    def profile(name, cls):
        item = banks.get(name)
        if item is None:
            return None
        _exact_keys(
            item, {"kind", "schema_id", "feature_names"}, f"{name} metadata"
        )
        if item["kind"] != "profile":
            raise InvalidSpeciesLibrary(f"{name} is not a profile bank.")
        prefix = f"{name}."
        return cls(
            arrays[prefix + "centres"], arrays[prefix + "scales"],
            arrays[prefix + "weights"], arrays[prefix + "source_indices"],
            arrays[prefix + "sample_counts"], item["schema_id"],
            tuple(item["feature_names"]), arrays[prefix + "half_distances"],
        )

    def prototype(name, cls):
        item = banks.get(name)
        if item is None:
            return None
        _exact_keys(
            item,
            {"kind", "schema_id", "feature_names", "class_names"},
            f"{name} metadata",
        )
        if item["kind"] != "prototype":
            raise InvalidSpeciesLibrary(f"{name} is not a prototype bank.")
        prefix = f"{name}."
        return cls(
            arrays[prefix + "centres"], arrays[prefix + "scales"],
            arrays[prefix + "weights"], arrays[prefix + "source_indices"],
            arrays[prefix + "seed_ids"], arrays[prefix + "class_ids"],
            arrays[prefix + "sample_counts"], item["schema_id"],
            tuple(item["feature_names"]), tuple(item["class_names"]),
        )

    def shape(name, cls):
        item = banks.get(name)
        if item is None:
            return None
        expected = {"kind", "schema_id", "measurement_names", "observations"}
        if cls is SpeciesDimensionsShapeBank:
            expected.add("components")
            expected.add("fit_settings")
        _exact_keys(item, expected, f"{name} metadata")
        prefix = f"{name}."
        measurements = arrays[prefix + "measurements"]
        uncertainties = arrays[prefix + "uncertainties"]
        sources = arrays[prefix + "source_indices"]
        seed_ids = arrays[prefix + "seed_ids"]
        contours = arrays[prefix + "contours"]
        records = item["observations"]
        if not isinstance(records, list) or not (
            len(records)
            == len(measurements)
            == len(uncertainties)
            == len(sources)
            == len(seed_ids)
            == len(contours)
        ):
            raise InvalidSpeciesLibrary(
                f"{name} observation arrays are inconsistent."
            )
        observations = tuple(
            _shape_observation_from_payload(
                record,
                int(sources[index]),
                int(seed_ids[index]),
                tuple(float(value) for value in measurements[index]),
                tuple(float(value) for value in uncertainties[index]),
                tuple(float(value) for value in contours[index]),
            )
            for index, record in enumerate(records)
        )
        names = tuple(item["measurement_names"])
        if cls is SpeciesShapeSummary:
            return cls(observations, names, item["schema_id"])
        fit_settings = item["fit_settings"]
        _exact_keys(
            fit_settings,
            {
                "minimum_component_seeds",
                "shrinkage_seed_count",
                "maximum_contour_modes",
            },
            "dimensions/shape fit settings",
        )
        return cls(
            observations,
            tuple(
                _shape_component_from_payload(value)
                for value in item["components"]
            ),
            names,
            item["schema_id"],
            int(fit_settings["minimum_component_seeds"]),
            float(fit_settings["shrinkage_seed_count"]),
            int(fit_settings["maximum_contour_modes"]),
        )

    return SpeciesLibraryArtifact(
        manifest,
        profile("foreground_colour", SpeciesForegroundColourBank),
        profile("foreground_noise", SpeciesForegroundNoiseBank),
        prototype("material_prototypes", SpeciesMaterialPrototypeBank),
        prototype("edge_prototypes", SpeciesEdgePrototypeBank),
        prototype("seed_traits", SpeciesSeedTraitBank),
        shape("shape_summary", SpeciesShapeSummary),
        shape("dimensions_shape", SpeciesDimensionsShapeBank),
    )


def _manifest_payload(
    manifest: SpeciesLibraryManifest, bank_metadata: Mapping[str, object]
) -> dict[str, object]:
    return {
        "format": SPECIES_LIBRARY_FORMAT,
        "schema_version": SPECIES_LIBRARY_SCHEMA_VERSION,
        "library_id": manifest.library_id,
        "version": manifest.version,
        "content_sha256": manifest.content_sha256,
        "species_id": manifest.species_id,
        "species_display_name": manifest.species_display_name,
        "trait_vocabulary_sha256": manifest.trait_vocabulary_sha256,
        "seed_fiddle_version": manifest.seed_fiddle_version,
        "created_utc": manifest.created_utc,
        "status": manifest.status.value,
        "sources": [_source_payload(item) for item in manifest.sources],
        "products": [_validation_payload(item) for item in manifest.products],
        "descriptor_schemas": dict(manifest.descriptor_schemas),
        "extraction_settings_sha256": manifest.extraction_settings_sha256,
        "aggregation_settings_sha256": manifest.aggregation_settings_sha256,
        "corrected_colour_space_version": manifest.corrected_colour_space_version,
        "arrays": [asdict(item) for item in manifest.arrays],
        "parent_content_sha256": manifest.parent_content_sha256,
        "bank_metadata": dict(bank_metadata),
    }


def _manifest_from_payload(payload):
    required = {
        "format", "schema_version", "library_id", "version", "content_sha256",
        "species_id", "species_display_name", "trait_vocabulary_sha256",
        "seed_fiddle_version", "created_utc", "status", "sources", "products",
        "descriptor_schemas", "extraction_settings_sha256",
        "aggregation_settings_sha256", "corrected_colour_space_version",
        "arrays", "parent_content_sha256", "bank_metadata",
    }
    _exact_keys(payload, required, "library manifest")
    if (
        payload["format"] != SPECIES_LIBRARY_FORMAT
        or payload["schema_version"] != SPECIES_LIBRARY_SCHEMA_VERSION
    ):
        raise InvalidSpeciesLibrary(
            "Species-library format or schema version is unsupported."
        )
    for name in ("sources", "products", "arrays"):
        if not isinstance(payload[name], list):
            raise InvalidSpeciesLibrary(f"Library {name} must be a JSON list.")
    manifest = SpeciesLibraryManifest(
        payload["library_id"], payload["version"], payload["content_sha256"],
        payload["species_id"], payload["species_display_name"],
        payload["trait_vocabulary_sha256"], payload["seed_fiddle_version"],
        payload["created_utc"], LibraryStatus(payload["status"]),
        tuple(_source_from_payload(item) for item in payload["sources"]),
        tuple(_validation_from_payload(item) for item in payload["products"]),
        payload["descriptor_schemas"], payload["extraction_settings_sha256"],
        payload["aggregation_settings_sha256"],
        payload["corrected_colour_space_version"],
        tuple(_array_spec_from_payload(item) for item in payload["arrays"]),
        payload["parent_content_sha256"],
    )
    return manifest, payload["bank_metadata"]


def _source_payload(value: LibrarySourceRecord) -> dict[str, object]:
    return {
        "source_sha256": value.source_sha256,
        "annotation_sha256": value.annotation_sha256,
        "display_label": value.display_label,
        "source_shape": list(value.source_shape),
        "capture_group_id": value.capture_group_id,
        "biological_context": (
            None if value.biological_context is None
            else asdict(value.biological_context)
        ),
        "review_status": value.review_status,
    }


def _source_from_payload(value) -> LibrarySourceRecord:
    _exact_keys(
        value,
        {"source_sha256", "annotation_sha256", "display_label", "source_shape",
         "capture_group_id", "biological_context", "review_status"},
        "source record",
    )
    context = value["biological_context"]
    if context is not None:
        _exact_keys(
            context,
            {"species_id", "lineage_group_id", "accession_id", "seed_lot_id"},
            "biological context",
        )
        context = BiologicalContext(**context)
    return LibrarySourceRecord(
        value["source_sha256"], value["annotation_sha256"],
        value["display_label"], tuple(value["source_shape"]),
        value["capture_group_id"], context, value["review_status"],
    )


def _validation_payload(value: ProductValidation) -> dict[str, object]:
    return {
        "product": value.product.value,
        "tier": value.tier.value,
        "source_count": value.source_count,
        "seed_count": value.seed_count,
        "sample_count": value.sample_count,
        "prototype_count": value.prototype_count,
        "effective_weight": value.effective_weight,
        "metrics": dict(value.metrics),
        "warnings": list(value.warnings),
    }


def _validation_from_payload(value) -> ProductValidation:
    _exact_keys(
        value,
        {"product", "tier", "source_count", "seed_count", "sample_count",
         "prototype_count", "effective_weight", "metrics", "warnings"},
        "product validation",
    )
    return ProductValidation(
        LibraryProduct(value["product"]), ValidationTier(value["tier"]),
        value["source_count"], value["seed_count"], value["sample_count"],
        value["prototype_count"], value["effective_weight"], value["metrics"],
        tuple(value["warnings"]),
    )


def _array_spec_from_payload(value) -> LibraryArraySpec:
    _exact_keys(
        value, {"name", "dtype", "shape", "byte_count", "sha256"},
        "array record",
    )
    return LibraryArraySpec(
        value["name"], value["dtype"], tuple(value["shape"]),
        value["byte_count"], value["sha256"],
    )


def _shape_observation_metadata(item: ShapeObservation) -> dict[str, object]:
    return {
        "physical_seed_id": item.physical_seed_id,
        "hierarchy_path": list(item.hierarchy_path),
        "pose": item.pose,
        "visibility": item.visibility,
        "calibrated": item.calibrated,
    }


def _shape_observation_from_payload(
    value, source_index, seed_id, measurement, uncertainty, contour_signature
) -> ShapeObservation:
    _exact_keys(
        value,
        {"physical_seed_id", "hierarchy_path", "pose", "visibility", "calibrated"},
        "shape observation",
    )
    return ShapeObservation(
        source_index, seed_id, value["physical_seed_id"],
        tuple(value["hierarchy_path"]), value["pose"], value["visibility"],
        bool(value["calibrated"]), measurement, uncertainty, contour_signature,
    )


def _shape_component_payload(item: ShapePopulationComponent) -> dict[str, object]:
    return {
        "hierarchy_path": list(item.hierarchy_path), "pose": item.pose,
        "mean": list(item.mean),
        "covariance": [list(row) for row in item.covariance],
        "mean_uncertainty": list(item.mean_uncertainty),
        "predictive_interval_95": [list(row) for row in item.predictive_interval_95],
        "effective_physical_seed_count": item.effective_physical_seed_count,
        "source_count": item.source_count,
        "contour_mean": list(item.contour_mean),
        "contour_modes": [list(row) for row in item.contour_modes],
        "contour_variances": list(item.contour_variances),
        "physical_dimensions_available": item.physical_dimensions_available,
        "calibrated_physical_seed_count": item.calibrated_physical_seed_count,
        "dimensionless_physical_seed_count": item.dimensionless_physical_seed_count,
    }


def _shape_component_from_payload(value) -> ShapePopulationComponent:
    required = {
        "hierarchy_path", "pose", "mean", "covariance", "mean_uncertainty",
        "predictive_interval_95", "effective_physical_seed_count", "source_count",
        "contour_mean", "contour_modes", "contour_variances",
        "physical_dimensions_available", "calibrated_physical_seed_count",
        "dimensionless_physical_seed_count",
    }
    _exact_keys(value, required, "shape component")
    return ShapePopulationComponent(
        tuple(value["hierarchy_path"]), value["pose"], tuple(value["mean"]),
        tuple(tuple(row) for row in value["covariance"]),
        tuple(value["mean_uncertainty"]),
        tuple(tuple(row) for row in value["predictive_interval_95"]),
        value["effective_physical_seed_count"], value["source_count"],
        tuple(value["contour_mean"]),
        tuple(tuple(row) for row in value["contour_modes"]),
        tuple(value["contour_variances"]),
        bool(value["physical_dimensions_available"]),
        value["calibrated_physical_seed_count"],
        value["dimensionless_physical_seed_count"],
    )


def _array_spec(name: str, values: np.ndarray) -> LibraryArraySpec:
    array = np.ascontiguousarray(values)
    if array.dtype.hasobject or array.nbytes > SPECIES_LIBRARY_MAX_ARRAY_BYTES:
        raise InvalidSpeciesLibrary(
            f"Library array {name!r} has an unsafe dtype or size."
        )
    digest = sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return LibraryArraySpec(
        name, array.dtype.str, tuple(array.shape), array.nbytes, digest.hexdigest()
    )


def _content_digest(payload, arrays) -> str:
    root = dict(payload)
    root["content_sha256"] = ""
    digest = sha256(_canonical_json(root))
    for name, values in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        digest.update(_array_spec(name, values).sha256.encode("ascii"))
    return digest.hexdigest()


def _canonical_json(payload) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_json(path: Path, payload) -> None:
    data = _canonical_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path):
    try:
        if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
            raise InvalidSpeciesLibrary(
                f"Library JSON is missing or too large: {path}"
            )
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                InvalidSpeciesLibrary(f"Non-finite JSON number {value!r}.")
            ),
        )
    except InvalidSpeciesLibrary:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidSpeciesLibrary(
            f"Could not read library JSON: {error}"
        ) from error


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise InvalidSpeciesLibrary(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise InvalidSpeciesLibrary(f"{name.capitalize()} has an invalid schema.")


def _safe_token(value: str, name: str, *, numeric_first: bool = False) -> None:
    text = str(value)
    first = bool(text) and (text[0].isalnum() if numeric_first else text[0].islower())
    if (
        not first
        or len(text) > 128
        or text in {".", ".."}
        or not all(
            character.islower()
            or character.isdigit()
            or character in "_.-"
            for character in text
        )
    ):
        raise InvalidSpeciesLibrary(f"Unsafe {name}: {value!r}.")


def _safe_archive_member(info: zipfile.ZipInfo) -> None:
    path = PurePosixPath(info.filename)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise InvalidSpeciesLibrary("Portable library contains an unsafe path.")
    if info.file_size > MAX_BUNDLE_BYTES:
        raise InvalidSpeciesLibrary("Portable library member is too large.")
    if info.compress_size and info.file_size / info.compress_size > MAX_ZIP_RATIO:
        raise InvalidSpeciesLibrary(
            "Portable library compression ratio is unsafe."
        )
