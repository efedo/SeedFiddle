"""Deterministic source-balanced assembly of compact library contributions."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from hashlib import sha256
from typing import Iterable

import numpy as np

from seedvision.measurement import (
    build_species_shape_summary,
    fit_species_dimensions_shape_bank,
)
from seedvision.reference_library.contracts import (
    SHAPE_SUMMARY_SCHEMA,
    SpeciesEdgePrototypeBank,
    SpeciesForegroundColourBank,
    SpeciesForegroundNoiseBank,
    SpeciesLibraryArtifact,
    SpeciesMaterialPrototypeBank,
    SpeciesSeedTraitBank,
    SourceProfileBank,
    SourcePrototypeBank,
)
from seedvision.reference_library.extraction import ExtractedSourceContribution


@dataclass(frozen=True, slots=True)
class LibraryAggregationSettings:
    maximum_colour_profiles: int = 2048
    maximum_noise_profiles: int = 512
    maximum_material_prototypes_per_class: int = 1024
    maximum_edge_prototypes_per_class: int = 4096
    maximum_trait_prototypes_per_class: int = 1024
    shape_minimum_component_seeds: int = 2
    shape_shrinkage_seed_count: float = 5.0
    maximum_contour_modes: int = 4

    def __post_init__(self) -> None:
        if min(
            self.maximum_colour_profiles,
            self.maximum_noise_profiles,
            self.maximum_material_prototypes_per_class,
            self.maximum_edge_prototypes_per_class,
            self.maximum_trait_prototypes_per_class,
        ) < 1:
            raise ValueError("All library bank capacities must be positive.")
        if self.shape_minimum_component_seeds < 1:
            raise ValueError("Shape components need at least one physical seed.")
        if self.shape_shrinkage_seed_count < 0:
            raise ValueError("Shape shrinkage support cannot be negative.")


def aggregate_contributions(
    contributions: Iterable[ExtractedSourceContribution],
    *,
    settings: LibraryAggregationSettings | None = None,
) -> dict[str, object]:
    """Return source-balanced banks independent of input and pixel duplication."""

    settings = settings or LibraryAggregationSettings()
    values = tuple(contributions)
    hashes = tuple(item.source.source_sha256 for item in values)
    if len(hashes) != len(set(hashes)):
        raise ValueError("A source image may contribute only once to a library version.")
    source_indices = tuple(item.source_index for item in values)
    if len(source_indices) != len(set(source_indices)):
        raise ValueError("Source contribution indices must be unique before aggregation.")
    # Canonical source order makes rebuilt banks independent of UI selection order.
    values = tuple(sorted(values, key=lambda item: item.source.source_sha256))
    remap = {
        contribution.source_index: new_index
        for new_index, contribution in enumerate(values)
    }
    observations = tuple(
        _remapped_observation(item, remap)
        for contribution in values
        for item in contribution.shape_observations
    )
    return {
        "sources": tuple(item.source for item in values),
        "foreground_colour": _aggregate_profiles(
            values, remap, "foreground_colour", SpeciesForegroundColourBank,
            settings.maximum_colour_profiles,
        ),
        "foreground_noise": _aggregate_profiles(
            values, remap, "foreground_noise", SpeciesForegroundNoiseBank,
            settings.maximum_noise_profiles,
        ),
        "material_prototypes": _aggregate_prototypes(
            values, remap, "material_prototypes", SpeciesMaterialPrototypeBank,
            settings.maximum_material_prototypes_per_class,
        ),
        "edge_prototypes": _aggregate_prototypes(
            values, remap, "edge_prototypes", SpeciesEdgePrototypeBank,
            settings.maximum_edge_prototypes_per_class,
        ),
        "seed_traits": _aggregate_prototypes(
            values, remap, "seed_traits", SpeciesSeedTraitBank,
            settings.maximum_trait_prototypes_per_class,
        ),
        "shape_summary": build_species_shape_summary(observations),
        "dimensions_shape": fit_species_dimensions_shape_bank(
            observations,
            minimum_component_seeds=settings.shape_minimum_component_seeds,
            shrinkage_seed_count=settings.shape_shrinkage_seed_count,
            maximum_contour_modes=settings.maximum_contour_modes,
        ),
    }


def contributions_from_artifact(
    artifact: SpeciesLibraryArtifact,
) -> tuple[ExtractedSourceContribution, ...]:
    """Recover the retained compact contribution for each immutable source.

    This is the fork/rebuild seam promised by the on-disk contract.  It does
    not reconstruct source images or annotation masks; only inference-ready
    profiles, prototypes, reviewed measurements, and provenance are retained.
    """

    def profile_for(bank, source_index):
        if bank is None:
            return None
        selected = np.asarray(bank.source_indices) == source_index
        if not np.any(selected):
            return None
        weights = np.asarray(bank.weights)[selected]
        weights = weights / max(float(weights.sum()), 1e-12)
        return type(bank)(
            bank.centres[selected], bank.scales[selected], weights,
            np.full(int(np.count_nonzero(selected)), source_index, np.int32),
            bank.sample_counts[selected], bank.schema_id, bank.feature_names,
            bank.half_distances[selected],
        )

    def prototype_for(bank, source_index):
        if bank is None:
            return None
        selected = np.asarray(bank.source_indices) == source_index
        if not np.any(selected):
            return None
        weights = np.asarray(bank.weights)[selected].copy()
        classes = np.asarray(bank.class_ids)[selected]
        for class_id in np.unique(classes):
            class_selected = classes == class_id
            weights[class_selected] /= max(
                float(weights[class_selected].sum()), 1e-12
            )
        return type(bank)(
            bank.centres[selected], bank.scales[selected], weights,
            np.full(int(np.count_nonzero(selected)), source_index, np.int32),
            bank.seed_ids[selected], classes, bank.sample_counts[selected],
            bank.schema_id, bank.feature_names, bank.class_names,
        )

    return tuple(
        ExtractedSourceContribution(
            source_index=index,
            source=source,
            foreground_colour=profile_for(artifact.foreground_colour, index),
            foreground_noise=profile_for(artifact.foreground_noise, index),
            material_prototypes=prototype_for(
                artifact.material_prototypes, index
            ),
            edge_prototypes=prototype_for(artifact.edge_prototypes, index),
            seed_traits=prototype_for(artifact.seed_traits, index),
            shape_observations=tuple(
                observation
                for observation in (
                    ()
                    if artifact.shape_summary is None or artifact.shape_summary.schema_id != SHAPE_SUMMARY_SCHEMA
                    else artifact.shape_summary.observations
                )
                if observation.source_index == index
            ),
        )
        for index, source in enumerate(artifact.manifest.sources)
    )


def reindex_contribution(
    contribution: ExtractedSourceContribution, source_index: int
) -> ExtractedSourceContribution:
    """Assign one collision-free source index before deterministic assembly."""

    def reindex_bank(bank):
        if bank is None:
            return None
        return replace(
            bank,
            source_indices=np.full(len(bank.source_indices), source_index, np.int32),
        )

    return replace(
        contribution,
        source_index=source_index,
        foreground_colour=reindex_bank(contribution.foreground_colour),
        foreground_noise=reindex_bank(contribution.foreground_noise),
        material_prototypes=reindex_bank(contribution.material_prototypes),
        edge_prototypes=reindex_bank(contribution.edge_prototypes),
        seed_traits=reindex_bank(contribution.seed_traits),
        shape_observations=tuple(
            replace(item, source_index=source_index)
            for item in contribution.shape_observations
        ),
    )


def _aggregate_profiles(values, remap, attribute, cls, maximum):
    banks = [getattr(item, attribute) for item in values]
    banks = [item for item in banks if item is not None]
    if not banks:
        return None
    first = banks[0]
    if any(
        item.schema_id != first.schema_id
        or item.feature_names != first.feature_names
        for item in banks
    ):
        raise ValueError("Profile contribution schemas do not match.")
    centres = np.concatenate([item.centres for item in banks])
    scales = np.concatenate([item.scales for item in banks])
    weights = np.concatenate([item.weights for item in banks])
    sources = np.concatenate([item.source_indices for item in banks])
    counts = np.concatenate([item.sample_counts for item in banks])
    halves = np.concatenate([item.half_distances for item in banks])
    sources = np.asarray([remap[int(value)] for value in sources], np.int32)
    weights = _source_balanced_weights(weights, sources)
    keep = _bounded_diverse_selection(centres / scales, weights, maximum, sources)
    retained_weights = _source_balanced_weights(weights[keep], sources[keep])
    return cls(
        centres[keep], scales[keep], retained_weights, sources[keep], counts[keep],
        first.schema_id, first.feature_names, halves[keep],
    )


def _aggregate_prototypes(values, remap, attribute, cls, maximum_per_class):
    banks = [getattr(item, attribute) for item in values]
    banks = [item for item in banks if item is not None]
    if not banks:
        return None
    first = banks[0]
    if any(
        item.schema_id != first.schema_id
        or item.feature_names != first.feature_names
        or item.class_names != first.class_names
        for item in banks
    ):
        raise ValueError("Prototype contribution schemas do not match.")
    centres = np.concatenate([item.centres for item in banks])
    scales = np.concatenate([item.scales for item in banks])
    weights = np.concatenate([item.weights for item in banks])
    sources = np.concatenate([item.source_indices for item in banks])
    seeds = np.concatenate([item.seed_ids for item in banks])
    classes = np.concatenate([item.class_ids for item in banks])
    counts = np.concatenate([item.sample_counts for item in banks])
    sources = np.asarray([remap[int(value)] for value in sources], np.int32)
    keep_parts = []
    for class_id in range(len(first.class_names)):
        candidates = np.flatnonzero(classes == class_id)
        if not len(candidates):
            continue
        balanced = _source_balanced_weights(weights[candidates], sources[candidates])
        weights[candidates] = balanced
        selected = _bounded_diverse_selection(
            centres[candidates] / scales[candidates],
            balanced,
            maximum_per_class,
            sources[candidates],
        )
        keep_parts.append(candidates[selected])
    keep = np.sort(np.concatenate(keep_parts)) if keep_parts else np.empty(0, int)
    retained_weights = weights[keep].copy()
    for class_id in range(len(first.class_names)):
        selected = np.flatnonzero(classes[keep] == class_id)
        if len(selected):
            retained_weights[selected] = _source_balanced_weights(
                retained_weights[selected], sources[keep][selected]
            )
    return cls(
        centres[keep], scales[keep], retained_weights, sources[keep], seeds[keep],
        classes[keep], counts[keep], first.schema_id, first.feature_names,
        first.class_names,
    )


def aggregation_settings_sha256(settings: LibraryAggregationSettings) -> str:
    payload = "\n".join(
        f"{name}={getattr(settings, name)!r}"
        for name in settings.__dataclass_fields__
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _remapped_observation(item, remap):
    from dataclasses import replace
    return replace(item, source_index=remap[item.source_index])


def _source_balanced_weights(weights, source_indices):
    result = np.asarray(weights, np.float64).copy()
    sources = np.unique(source_indices)
    for source in sources:
        selected = source_indices == source
        result[selected] /= max(float(result[selected].sum()), 1e-12)
        result[selected] /= max(len(sources), 1)
    return result.astype(np.float32)


def _bounded_diverse_selection(features, weights, maximum, sources):
    if len(features) <= maximum:
        return np.arange(len(features))
    # Reserve one representative for every source before global farthest-first.
    selected = []
    for source in sorted(np.unique(sources)):
        candidates = np.flatnonzero(sources == source)
        selected.append(int(candidates[np.argmax(weights[candidates])]))
        if len(selected) == maximum:
            return np.asarray(selected, np.int64)
    centre = np.median(features, axis=0)
    scale = np.maximum(1e-6, 1.4826 * np.median(np.abs(features - centre), axis=0))
    normalized = (features - centre) / scale
    nearest = np.full(len(features), np.inf)
    for index in selected:
        nearest = np.minimum(nearest, np.sum((normalized - normalized[index]) ** 2, axis=1))
    while len(selected) < maximum:
        score = nearest * (0.25 + 0.75 * weights / max(float(weights.max()), 1e-12))
        score[selected] = -1
        index = int(np.argmax(score))
        selected.append(index)
        nearest = np.minimum(nearest, np.sum((normalized - normalized[index]) ** 2, axis=1))
    return np.asarray(sorted(selected), np.int64)
