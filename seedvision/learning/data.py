"""Versioned learning manifests, annotation I/O, audits, and tile datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

from seedvision.learning.contracts import FeatureStackSpec, ModelFamily
from seedvision.learning.targets import build_dense_targets, build_stardist_targets


MANIFEST_VERSION = 1


def rotation_ray_mapping(ray_count: int, quarter_turns: int) -> np.ndarray:
    """Map new screen-coordinate ray indices to pre-rotation indices."""

    ray_count = int(ray_count)
    if ray_count < 4 or ray_count % 4:
        raise ValueError("Quarter-turn ray mapping requires a multiple of four rays.")
    return (np.arange(ray_count) + int(quarter_turns) * (ray_count // 4)) % ray_count


@dataclass(frozen=True, slots=True)
class LearningSample:
    identifier: str
    features: str
    instances: str
    species: str
    group: str
    split: str
    reviewed: bool
    pattern_boundary: str | None = None
    pattern_valid: str | None = None
    image: str | None = None
    annotation_author: str | None = None
    annotation_revision: str | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class LearningManifest:
    dataset_id: str
    feature_spec: FeatureStackSpec
    samples: tuple[LearningSample, ...]
    provenance: str = "observational"
    scientific_validation_eligible: bool = False
    version: int = MANIFEST_VERSION

    @classmethod
    def load(cls, path: Path | str) -> "LearningManifest":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if int(payload.get("version", -1)) != MANIFEST_VERSION:
            raise ValueError(f"Unsupported learning-manifest version in {source}.")
        return cls(
            dataset_id=str(payload["dataset_id"]),
            feature_spec=FeatureStackSpec(**payload["feature_spec"]),
            samples=tuple(LearningSample(**item) for item in payload["samples"]),
            provenance=str(payload.get("provenance", "observational")),
            scientific_validation_eligible=bool(
                payload.get("scientific_validation_eligible", False)
            ),
            version=int(payload["version"]),
        )

    def save(self, path: Path | str) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "dataset_id": self.dataset_id,
            "feature_spec": asdict(self.feature_spec),
            "samples": [asdict(item) for item in self.samples],
            "provenance": self.provenance,
            "scientific_validation_eligible": self.scientific_validation_eligible,
        }
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
        return destination


def resolve_sample_path(manifest_path: Path | str, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(manifest_path).resolve().parent / path


def read_label_image(path: Path | str) -> np.ndarray:
    values = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if values is None:
        raise ValueError(f"Could not read label raster {path}.")
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"Label raster must be a single-channel integer image: {path}")
    if np.any(values < 0):
        raise ValueError(f"Label raster contains negative identifiers: {path}")
    return values.astype(np.int32, copy=False)


def write_label_image(path: Path | str, labels: np.ndarray) -> Path:
    destination = Path(path)
    values = np.asarray(labels)
    if values.ndim != 2 or np.any(values < 0):
        raise ValueError("Instance labels must be a non-negative two-dimensional raster.")
    maximum = int(values.max(initial=0))
    if maximum > np.iinfo(np.uint16).max:
        raise ValueError("PNG instance labels cannot exceed identifier 65,535.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    if not cv2.imwrite(str(temporary), values.astype(np.uint16)):
        raise OSError(f"Could not write label raster {destination}.")
    temporary.replace(destination)
    return destination


def _safe_identifier(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._")
    if not identifier:
        raise ValueError("A non-empty sample identifier is required.")
    return identifier


def export_learning_sample(
    manifest_path: Path | str,
    *,
    dataset_id: str,
    feature_spec: FeatureStackSpec,
    identifier: str,
    features: np.ndarray,
    labels: np.ndarray,
    image_bgr: np.ndarray | None,
    species: str,
    group: str,
    split: str = "train",
    reviewed: bool = False,
    pattern_boundary: np.ndarray | None = None,
    pattern_valid: np.ndarray | None = None,
    annotation_author: str | None = None,
    annotation_revision: str | None = None,
    notes: str | None = None,
    replace_existing: bool = False,
) -> LearningSample:
    """Atomically persist one corrected-coordinate supervised sample.

    This deliberately exports an explicit model feature stack together with
    the source view and labels.  A later pipeline change therefore cannot
    silently reinterpret old training data.
    """

    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    root.mkdir(parents=True, exist_ok=True)
    identifier = _safe_identifier(identifier)
    split = str(split).strip().lower()
    if split not in {"train", "validation", "test"}:
        raise ValueError("Learning split must be train, validation, or test.")
    values = np.asarray(features)
    label_values = np.asarray(labels)
    if values.ndim != 3 or values.shape[0] != feature_spec.input_channels:
        raise ValueError("Feature stack must be CHW with the specified channel count.")
    if label_values.ndim != 2 or tuple(values.shape[1:]) != label_values.shape:
        raise ValueError("Features and instance labels must share raster dimensions.")
    if not np.isfinite(values).all():
        raise ValueError("Feature stack contains non-finite values.")
    if image_bgr is not None:
        display = np.asarray(image_bgr)
        if display.shape != (*label_values.shape, 3) or display.dtype != np.uint8:
            raise ValueError("Display image must be uint8 BGR with the label dimensions.")
    else:
        display = None
    if (pattern_boundary is None) != (pattern_valid is None):
        raise ValueError("Pattern boundary and validity rasters must be supplied together.")
    pattern = None
    pattern_known = None
    if pattern_boundary is not None:
        pattern = np.asarray(pattern_boundary) > 0
        pattern_known = np.asarray(pattern_valid) > 0
        if pattern.shape != label_values.shape or pattern_known.shape != label_values.shape:
            raise ValueError("Pattern rasters must share the instance-label dimensions.")
        if np.any(pattern & ~pattern_known):
            raise ValueError("Every positive pattern-boundary pixel must be marked valid.")

    if manifest_path.is_file():
        manifest = LearningManifest.load(manifest_path)
        if manifest.dataset_id != str(dataset_id):
            raise ValueError("The existing manifest uses a different dataset identifier.")
        if manifest.feature_spec != feature_spec:
            raise ValueError("The existing manifest uses a different feature specification.")
        existing = {item.identifier: item for item in manifest.samples}
        if identifier in existing and not replace_existing:
            raise FileExistsError(f"Sample {identifier!r} already exists in the manifest.")
    else:
        manifest = LearningManifest(
            dataset_id=str(dataset_id),
            feature_spec=feature_spec,
            samples=(),
            scientific_validation_eligible=False,
        )

    feature_name = f"{identifier}.features.npz"
    instance_name = f"{identifier}.instances.png"
    image_name = f"{identifier}.png" if display is not None else None
    pattern_name = f"{identifier}.pattern.png" if pattern is not None else None
    pattern_valid_name = (
        f"{identifier}.pattern_valid.png" if pattern_known is not None else None
    )
    temporary_feature = root / f"{identifier}.features.tmp.npz"
    np.savez_compressed(temporary_feature, features=values.astype(np.float16))
    temporary_feature.replace(root / feature_name)
    write_label_image(root / instance_name, label_values)
    if display is not None and not cv2.imwrite(str(root / image_name), display):
        raise OSError(f"Could not write display image for {identifier}.")
    if pattern is not None:
        write_label_image(root / pattern_name, np.uint8(pattern) * 255)
        write_label_image(root / pattern_valid_name, np.uint8(pattern_known) * 255)
    sample = LearningSample(
        identifier=identifier,
        features=feature_name,
        instances=instance_name,
        species=str(species),
        group=str(group),
        split=split,
        reviewed=bool(reviewed),
        pattern_boundary=pattern_name,
        pattern_valid=pattern_valid_name,
        image=image_name,
        annotation_author=annotation_author,
        annotation_revision=annotation_revision,
        notes=notes,
    )
    samples = [item for item in manifest.samples if item.identifier != identifier]
    samples.append(sample)
    updated = LearningManifest(
        dataset_id=manifest.dataset_id,
        feature_spec=manifest.feature_spec,
        samples=tuple(sorted(samples, key=lambda item: item.identifier)),
        provenance=manifest.provenance,
        scientific_validation_eligible=manifest.scientific_validation_eligible,
        version=manifest.version,
    )
    updated.save(manifest_path)
    return sample


def file_sha256(path: Path | str) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_manifest(path: Path | str) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = LearningManifest.load(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []
    split_groups: dict[str, set[str]] = {}
    instance_count = 0
    reviewed_count = 0
    identifiers: set[str] = set()
    for sample in manifest.samples:
        if sample.identifier in identifiers:
            errors.append(f"Duplicate sample identifier: {sample.identifier}")
        identifiers.add(sample.identifier)
        if sample.split not in {"train", "validation", "test"}:
            errors.append(f"Invalid split for {sample.identifier}: {sample.split!r}")
        if not sample.group.strip():
            errors.append(f"Missing lot/capture group for {sample.identifier}.")
        split_groups.setdefault(sample.group, set()).add(sample.split)
        feature_path = resolve_sample_path(manifest_path, sample.features)
        label_path = resolve_sample_path(manifest_path, sample.instances)
        if not feature_path.is_file():
            errors.append(f"Missing feature archive for {sample.identifier}: {feature_path}")
            continue
        if not label_path.is_file():
            errors.append(f"Missing labels for {sample.identifier}: {label_path}")
            continue
        with np.load(feature_path, allow_pickle=False) as archive:
            if "features" not in archive:
                errors.append(f"Feature archive has no 'features': {feature_path}")
                continue
            feature_shape = archive["features"].shape
            if not np.isfinite(archive["features"]).all():
                errors.append(f"Non-finite feature values for {sample.identifier}.")
        labels = read_label_image(label_path)
        if len(feature_shape) != 3 or feature_shape[0] != manifest.feature_spec.input_channels:
            errors.append(
                f"Feature count mismatch for {sample.identifier}: {feature_shape}"
            )
        elif tuple(feature_shape[1:]) != labels.shape:
            errors.append(f"Feature/label dimensions differ for {sample.identifier}.")
        instance_count += int(labels.max(initial=0))
        present = np.unique(labels)
        present = present[present > 0]
        if len(present) and not np.array_equal(
            present, np.arange(1, int(present[-1]) + 1)
        ):
            errors.append(f"Instance IDs are not consecutive for {sample.identifier}.")
        for instance_id in present:
            components, _component_labels = cv2.connectedComponents(
                np.uint8(labels == instance_id), connectivity=8
            )
            if components > 2:
                errors.append(
                    f"Instance {int(instance_id)} is disconnected in {sample.identifier}."
                )
        reviewed_count += int(sample.reviewed)
        if sample.reviewed and not sample.annotation_author:
            warnings.append(
                f"Reviewed sample {sample.identifier} has no annotation author."
            )
        if sample.pattern_boundary and not sample.pattern_valid:
            errors.append(
                f"Pattern boundary lacks an explicit validity mask: {sample.identifier}"
            )
        if sample.pattern_boundary and sample.pattern_valid:
            pattern = read_label_image(
                resolve_sample_path(manifest_path, sample.pattern_boundary)
            )
            pattern_valid = read_label_image(
                resolve_sample_path(manifest_path, sample.pattern_valid)
            )
            if pattern.shape != labels.shape or pattern_valid.shape != labels.shape:
                errors.append(f"Pattern/label dimensions differ for {sample.identifier}.")
            elif np.any((pattern > 0) & (pattern_valid == 0)):
                errors.append(
                    f"Pattern positives fall outside the validity mask for {sample.identifier}."
                )
    for group, splits in split_groups.items():
        if len(splits) > 1:
            errors.append(f"Group {group!r} crosses dataset splits: {sorted(splits)}")
    if not manifest.samples:
        errors.append("The manifest contains no samples.")
    if reviewed_count < len(manifest.samples):
        warnings.append(
            f"Only {reviewed_count}/{len(manifest.samples)} samples are marked reviewed."
        )
    return {
        "dataset_id": manifest.dataset_id,
        "provenance": manifest.provenance,
        "scientific_validation_eligible": manifest.scientific_validation_eligible,
        "sample_count": len(manifest.samples),
        "reviewed_sample_count": reviewed_count,
        "instance_count": instance_count,
        "splits": {
            split: sum(item.split == split for item in manifest.samples)
            for split in sorted({item.split for item in manifest.samples})
        },
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


class SeedTileDataset:
    """Torch-compatible deterministic tile dataset backed by feature archives."""

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        split: str,
        family: ModelFamily | str,
        tile_size: int = 256,
        tiles_per_sample: int = 16,
        ray_count: int = 32,
        augment: bool = False,
        random_seed: int = 0,
        derived_cache_directory: Path | str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = LearningManifest.load(self.manifest_path)
        self.samples = tuple(item for item in self.manifest.samples if item.split == split)
        self.family = ModelFamily(family)
        self.tile_size = int(tile_size)
        self.tiles_per_sample = int(tiles_per_sample)
        self.ray_count = int(ray_count)
        self.augment = bool(augment)
        self.random_seed = int(random_seed)
        self.derived_cache_directory = Path(
            derived_cache_directory
            if derived_cache_directory is not None
            else self.manifest_path.parent / ".seedfiddle-cache"
        )
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = {}
        if self.tile_size < 32 or self.tiles_per_sample < 1:
            raise ValueError("tile_size must be >= 32 and tiles_per_sample must be positive.")
        if not self.samples:
            raise ValueError(f"No samples use split {split!r}.")

    def __len__(self) -> int:
        return len(self.samples) * self.tiles_per_sample

    def _load(self, sample: LearningSample):
        cached = self._cache.get(sample.identifier)
        if cached is not None:
            return cached
        feature_path = resolve_sample_path(self.manifest_path, sample.features)
        label_path = resolve_sample_path(self.manifest_path, sample.instances)
        with np.load(feature_path, allow_pickle=False) as archive:
            features = archive["features"].astype(np.float32, copy=False)
        labels = read_label_image(label_path)
        pattern = None
        pattern_valid = None
        if sample.pattern_boundary:
            pattern = read_label_image(
                resolve_sample_path(self.manifest_path, sample.pattern_boundary)
            ) > 0
            pattern_valid = read_label_image(
                resolve_sample_path(self.manifest_path, sample.pattern_valid or "")
            ) > 0
        label_digest = file_sha256(label_path)[:16]
        family_suffix = (
            "unet" if self.family is ModelFamily.UNET_WATERSHED else f"stardist{self.ray_count}"
        )
        target_cache = self.derived_cache_directory / (
            f"{_safe_identifier(sample.identifier)}.{family_suffix}.{label_digest}.npz"
        )
        target_cache.parent.mkdir(parents=True, exist_ok=True)
        cached_targets = None
        if target_cache.is_file():
            with np.load(target_cache, allow_pickle=False) as archive:
                cached_targets = {name: archive[name] for name in archive.files}
        if cached_targets is not None:
            targets = cached_targets
        elif self.family is ModelFamily.UNET_WATERSHED:
            built = build_dense_targets(
                labels,
                pattern_boundary=pattern,
                pattern_valid=pattern_valid,
            )
            targets = {
                "interior": built.interior[None],
                "physical_boundary": built.physical_boundary[None],
                "pattern_boundary": built.pattern_boundary[None],
                "pattern_valid": built.pattern_valid[None],
                "centre": built.centre[None],
                "distance": built.distance[None],
                "valid": built.valid[None],
            }
        else:
            built = build_stardist_targets(labels, ray_count=self.ray_count)
            targets = {
                "object_probability": built.object_probability[None],
                "radial_distances": built.radial_distances,
                "radial_valid": built.radial_valid,
                "valid": built.valid[None],
            }
        if cached_targets is None:
            temporary = target_cache.with_name(target_cache.stem + ".tmp.npz")
            np.savez_compressed(temporary, **targets)
            temporary.replace(target_cache)
        cached = (features, labels, targets)
        self._cache[sample.identifier] = cached
        return cached

    def __getitem__(self, index: int):
        import torch

        sample = self.samples[index // self.tiles_per_sample]
        features, labels, targets = self._load(sample)
        height, width = labels.shape
        tile = min(self.tile_size, height, width)
        rng = np.random.default_rng(self.random_seed + index * 104729)
        if np.any(labels > 0) and rng.random() < 0.8:
            rows, columns = np.nonzero(labels > 0)
            selected = int(rng.integers(0, len(rows)))
            centre_y, centre_x = int(rows[selected]), int(columns[selected])
            y0 = int(np.clip(centre_y - rng.integers(tile // 4, 3 * tile // 4 + 1), 0, height - tile))
            x0 = int(np.clip(centre_x - rng.integers(tile // 4, 3 * tile // 4 + 1), 0, width - tile))
        else:
            y0 = int(rng.integers(0, max(1, height - tile + 1)))
            x0 = int(rng.integers(0, max(1, width - tile + 1)))
        slices = (slice(y0, y0 + tile), slice(x0, x0 + tile))
        feature_tile = features[:, slices[0], slices[1]].copy()
        target_tiles = {
            name: values[:, slices[0], slices[1]].copy()
            for name, values in targets.items()
        }
        label_tile = labels[slices].copy()
        if self.augment:
            feature_tile, label_tile, target_tiles = self._augment(
                feature_tile, label_tile, target_tiles, rng
            )
        return {
            "features": torch.from_numpy(np.ascontiguousarray(feature_tile)),
            "labels": torch.from_numpy(np.ascontiguousarray(label_tile.astype(np.int64))),
            "targets": {
                name: torch.from_numpy(np.ascontiguousarray(values.astype(np.float32)))
                for name, values in target_tiles.items()
            },
            "sample_id": sample.identifier,
        }

    def _augment(self, features, labels, targets, rng):
        ray_count = self.ray_count
        if rng.random() < 0.5:
            features = features[..., ::-1]
            labels = labels[:, ::-1]
            targets = {name: values[..., ::-1] for name, values in targets.items()}
            if self.family is ModelFamily.STARDIST:
                mapping = (ray_count // 2 - np.arange(ray_count)) % ray_count
                targets["radial_distances"] = targets["radial_distances"][mapping]
                targets["radial_valid"] = targets["radial_valid"][mapping]
        if rng.random() < 0.5:
            features = features[..., ::-1, :]
            labels = labels[::-1, :]
            targets = {name: values[..., ::-1, :] for name, values in targets.items()}
            if self.family is ModelFamily.STARDIST:
                mapping = (-np.arange(ray_count)) % ray_count
                targets["radial_distances"] = targets["radial_distances"][mapping]
                targets["radial_valid"] = targets["radial_valid"][mapping]
        if ray_count % 4 == 0:
            turns = int(rng.integers(0, 4))
        else:
            turns = 0
        if turns:
            features = np.rot90(features, turns, axes=(-2, -1))
            labels = np.rot90(labels, turns)
            targets = {
                name: np.rot90(values, turns, axes=(-2, -1))
                for name, values in targets.items()
            }
            if self.family is ModelFamily.STARDIST:
                # np.rot90 turns screen coordinates counter-clockwise, so a
                # new ray at angle theta samples the old ray at theta + 90Â°.
                mapping = rotation_ray_mapping(ray_count, turns)
                targets["radial_distances"] = targets["radial_distances"][mapping]
                targets["radial_valid"] = targets["radial_valid"][mapping]
        return features, labels, targets
