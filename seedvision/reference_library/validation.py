"""Grouped validation and leakage/duplication audits for species libraries.

The validator deliberately operates on the compact, source-tagged contribution
banks which are also used at runtime.  It never makes random-pixel splits: an
entire image, capture group, physical seed, or accession is withheld at once.
The resulting metrics are descriptive validation diagnostics rather than a
claim that descriptor compatibility is a calibrated posterior probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from seedvision.reference_library.contracts import (
    LibraryProduct,
    ProductValidation,
    ValidationTier,
)


@dataclass(frozen=True, slots=True)
class LibraryValidationReport:
    products: tuple[ProductValidation, ...]
    publish_eligible: bool
    warnings: tuple[str, ...]

    def for_product(self, product: LibraryProduct) -> ProductValidation:
        return next(item for item in self.products if item.product is product)


def validate_aggregated_library(aggregated: Mapping[str, object]) -> LibraryValidationReport:
    sources = tuple(aggregated.get("sources", ()))
    source_count = len(sources)
    capture_groups = {
        item.capture_group_id for item in sources if item.capture_group_id is not None
    }
    validations = []
    warnings = []

    def tier(count):
        if count == 0:
            return ValidationTier.UNAVAILABLE
        if count < 3:
            return ValidationTier.PROVISIONAL
        if len(capture_groups) >= 2:
            return ValidationTier.MULTI_CONTEXT_VALIDATED
        return ValidationTier.VALIDATED

    for product, key in (
        (LibraryProduct.FOREGROUND_COLOUR, "foreground_colour"),
        (LibraryProduct.FOREGROUND_NOISE, "foreground_noise"),
        (LibraryProduct.MATERIAL_PROTOTYPES, "material_prototypes"),
        (LibraryProduct.EDGE_PROTOTYPES, "edge_prototypes"),
        (LibraryProduct.SEED_TRAITS, "seed_traits"),
    ):
        bank = aggregated.get(key)
        contributing = 0 if bank is None else len(np.unique(bank.source_indices))
        sample_count = 0 if bank is None else int(np.sum(bank.sample_counts))
        prototype_count = 0 if bank is None else int(len(bank.weights))
        metrics = {}
        product_warnings = []
        if bank is not None:
            metrics["maximum_source_weight"] = _maximum_source_weight(bank)
            metrics["source_balance_error"] = _source_balance_error(bank)
            metrics["leave_one_source_out_distance"] = _leave_one_source_out_distance(bank)
            metrics["leave_one_capture_group_out_distance"] = (
                _leave_one_capture_group_out_distance(bank, sources)
            )
            metrics["capture_group_count"] = float(
                len(
                    {
                        sources[int(index)].capture_group_id
                        for index in np.unique(bank.source_indices)
                        if sources[int(index)].capture_group_id is not None
                    }
                )
            )
            if metrics["source_balance_error"] > 1e-5:
                product_warnings.append("Source-balance invariant failed.")
        if contributing and contributing < 3:
            product_warnings.append("Fewer than three independent source images.")
        validations.append(
            ProductValidation(
                product, tier(contributing), contributing,
                sample_count=sample_count, prototype_count=prototype_count,
                effective_weight=float(contributing), metrics=metrics,
                warnings=tuple(product_warnings),
            )
        )

    summary = aggregated.get("shape_summary")
    observations = () if summary is None else summary.observations
    shape_sources = len({item.source_index for item in observations})
    physical_seeds = {
        item.physical_seed_id or f"{item.source_index}:{item.seed_id}"
        for item in observations
    }
    shape_warnings = []
    if observations and not any(item.calibrated for item in observations):
        shape_warnings.append(
            "No calibrated observations; only dimensionless diagnostics are portable."
        )
    shape_metrics = _shape_grouped_metrics(observations, sources)
    validations.append(
        ProductValidation(
            LibraryProduct.SHAPE_SUMMARY,
            tier(shape_sources),
            shape_sources,
            seed_count=len(physical_seeds),
            sample_count=len(observations),
            effective_weight=float(len(physical_seeds)),
            metrics={
                "physical_seed_duplication_ratio": (
                    len(observations) / max(len(physical_seeds), 1)
                ),
                **shape_metrics,
            },
            warnings=tuple(shape_warnings),
        )
    )
    dimensions = aggregated.get("dimensions_shape")
    component_count = 0 if dimensions is None else len(dimensions.components)
    validations.append(
        ProductValidation(
            LibraryProduct.DIMENSIONS_SHAPE,
            ValidationTier.UNAVAILABLE
            if component_count == 0
            else tier(shape_sources),
            shape_sources,
            seed_count=len(physical_seeds),
            prototype_count=component_count,
            effective_weight=float(len(physical_seeds)),
            metrics=shape_metrics,
            warnings=(
                ("No supported calibrated hierarchy/pose component.",)
                if observations and component_count == 0
                else ()
            ),
        )
    )
    available = [
        item for item in validations if item.tier is not ValidationTier.UNAVAILABLE
    ]
    if not available:
        warnings.append("No publishable library product has reviewed support.")
    if len({item.source_sha256 for item in sources}) != len(sources):
        warnings.append("Duplicate source-image fingerprint detected.")
    return LibraryValidationReport(
        tuple(validations), bool(available) and not warnings, tuple(warnings)
    )


def assert_no_held_out_source(bank, source_index: int) -> None:
    if bank is not None and np.any(bank.source_indices == source_index):
        raise AssertionError("Held-out source remains in the runtime library bank.")


def source_weight_signature(bank) -> tuple[tuple[int, float], ...]:
    if bank is None:
        return ()
    return tuple(
        (int(source), float(np.sum(bank.weights[bank.source_indices == source])))
        for source in sorted(np.unique(bank.source_indices))
    )


def _maximum_source_weight(bank) -> float:
    signature = source_weight_signature(bank)
    return max((value for _source, value in signature), default=0.0)


def _source_balance_error(bank) -> float:
    """Maximum departure from equal source authority within each class.

    Prototype classes are independently normalized, whereas a profile bank has
    only one implicit class.  Counting raw pixels or prototypes here would
    recreate the duplication failure this audit is intended to catch.
    """

    class_ids = getattr(bank, "class_ids", np.zeros(len(bank.weights), np.int16))
    maximum = 0.0
    for class_id in np.unique(class_ids):
        selected = class_ids == class_id
        sources = np.unique(bank.source_indices[selected])
        if not len(sources):
            continue
        expected = 1.0 / len(sources)
        totals = [
            float(np.sum(bank.weights[selected & (bank.source_indices == source)]))
            for source in sources
        ]
        normalization = max(sum(totals), 1e-12)
        maximum = max(
            maximum,
            max(abs(value / normalization - expected) for value in totals),
        )
    return float(maximum)


def _leave_one_source_out_distance(bank) -> float:
    sources = np.unique(bank.source_indices)
    if len(sources) < 2 or not len(bank.centres):
        return 0.0
    distances = []
    for source in sources:
        held = bank.centres[bank.source_indices == source]
        training = bank.centres[bank.source_indices != source]
        training_scales = bank.scales[bank.source_indices != source]
        if not len(held) or not len(training):
            continue
        chunk = ((held[:, None] - training[None]) / training_scales[None]) ** 2
        distances.extend(np.min(np.mean(chunk, axis=2), axis=1))
    return float(np.median(distances)) if distances else 0.0


def _leave_one_capture_group_out_distance(bank, sources) -> float:
    """Descriptor-domain shift with each complete capture group held out."""

    if bank is None or not len(bank.centres):
        return 0.0
    groups = np.asarray(
        [
            "" if sources[int(index)].capture_group_id is None
            else sources[int(index)].capture_group_id
            for index in bank.source_indices
        ],
        dtype="U128",
    )
    named = tuple(value for value in np.unique(groups) if value)
    if len(named) < 2:
        return 0.0
    distances = []
    class_ids = getattr(bank, "class_ids", np.zeros(len(bank.centres), np.int16))
    for group in named:
        for class_id in np.unique(class_ids):
            held_mask = (groups == group) & (class_ids == class_id)
            train_mask = (groups != group) & (class_ids == class_id)
            held = bank.centres[held_mask]
            training = bank.centres[train_mask]
            scales = bank.scales[train_mask]
            if not len(held) or not len(training):
                continue
            distance = ((held[:, None] - training[None]) / scales[None]) ** 2
            distances.extend(np.min(np.mean(distance, axis=2), axis=1))
    return float(np.median(distances)) if distances else 0.0


def _shape_grouped_metrics(observations, sources) -> dict[str, float]:
    """Return physical-seed/image/accession/capture grouped shape diagnostics."""

    if not observations:
        return {
            "physical_seed_count": 0.0,
            "accession_count": 0.0,
            "pose_family_count": 0.0,
            "leave_one_physical_seed_predictive_coverage": 0.0,
            "leave_one_accession_standardized_distance": 0.0,
            "leave_one_capture_group_standardized_distance": 0.0,
        }
    groups: dict[str, list[object]] = {}
    for item in observations:
        key = item.physical_seed_id or f"{item.source_index}:{item.seed_id}"
        groups.setdefault(key, []).append(item)
    grouped = []
    for key, records in sorted(groups.items()):
        grouped.append(
            (
                key,
                np.mean([item.measurement for item in records], axis=0),
                records[0].pose,
                records[0].hierarchy_path,
                tuple(sorted({item.source_index for item in records})),
            )
        )
    accession_ids = {
        sources[int(source_id)].biological_context.accession_id
        for item in grouped
        for source_id in item[4]
        if sources[int(source_id)].biological_context is not None
        and sources[int(source_id)].biological_context.accession_id is not None
    }
    poses = {item[2] for item in grouped}
    coverage_hits = 0
    coverage_total = 0
    # Leave out a complete physical seed.  The robust interval is fitted from
    # other physical seeds in the same pose; repeated views can never leak.
    for index, (_key, vector, pose, _path, _source_ids) in enumerate(grouped):
        training = np.asarray(
            [item[1] for offset, item in enumerate(grouped) if offset != index and item[2] == pose],
            np.float64,
        )
        if len(training) < 2:
            continue
        centre = np.median(training, axis=0)
        scale = np.maximum(1e-8, 1.4826 * np.median(np.abs(training - centre), axis=0))
        # The most interpretable first-pass coverage uses maximum span,
        # ovality, area, non-ellipticity, and solidity rather than claiming 15
        # correlated dimensions are independent trials.
        selected = np.asarray((0, 3, 4, 6, 8), np.int64)
        coverage_hits += int(
            np.all(np.abs(vector[selected] - centre[selected]) <= 1.96 * scale[selected])
        )
        coverage_total += 1

    def grouped_distance(group_for_source):
        labels = []
        vectors = []
        for _key, vector, _pose, _path, source_ids in grouped:
            values = {
                group_for_source(source_id)
                for source_id in source_ids
                if group_for_source(source_id) is not None
            }
            if len(values) == 1:
                labels.append(next(iter(values)))
                vectors.append(vector)
        unique = tuple(sorted(set(labels)))
        if len(unique) < 2:
            return 0.0
        vectors = np.asarray(vectors, np.float64)
        distances = []
        for label in unique:
            held = vectors[np.asarray(labels) == label]
            training = vectors[np.asarray(labels) != label]
            if not len(held) or len(training) < 2:
                continue
            centre = np.median(training, axis=0)
            scale = np.maximum(1e-8, 1.4826 * np.median(np.abs(training - centre), axis=0))
            distances.extend(np.sqrt(np.mean(((held - centre) / scale) ** 2, axis=1)))
        return float(np.median(distances)) if distances else 0.0

    return {
        "physical_seed_count": float(len(grouped)),
        "accession_count": float(len(accession_ids)),
        "pose_family_count": float(len(poses)),
        "leave_one_physical_seed_predictive_coverage": (
            float(coverage_hits / coverage_total) if coverage_total else 0.0
        ),
        "leave_one_accession_standardized_distance": grouped_distance(
            lambda source_index: (
                sources[int(source_index)].biological_context.accession_id
                if sources[int(source_index)].biological_context is not None
                else None
            )
        ),
        "leave_one_capture_group_standardized_distance": grouped_distance(
            lambda source_index: sources[int(source_index)].capture_group_id
        ),
    }
