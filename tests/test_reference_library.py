from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from seedvision.reference_library.aggregation import (
    LibraryAggregationSettings,
    aggregate_contributions,
    contributions_from_artifact,
    reindex_contribution,
)
from seedvision.reference_library.contracts import (
    BiologicalContext,
    FOREGROUND_COLOUR_SCHEMA,
    LibrarySourceRecord,
    ShapeObservation,
    SpeciesForegroundColourBank,
    SpeciesLibraryArtifact,
    SpeciesLibraryPin,
)
from seedvision.reference_library.extraction import ExtractedSourceContribution
from seedvision.reference_library.extraction import (
    EDGE_FEATURE_NAMES,
    MATERIAL_FEATURE_NAMES,
    NOISE_FEATURE_NAMES,
    LibraryExtractionSettings,
    LibrarySourceInput,
    extract_source_contribution,
)
from seedvision.persistence.reference_regions import (
    ReferenceRegionBundle,
    SeedInstanceAnnotation,
    file_sha256,
)
from seedvision.reference_library.persistence import (
    InvalidSpeciesLibrary,
    SpeciesLibraryStore,
    create_manifest,
)
from seedvision.reference_library.validation import (
    assert_no_held_out_source,
    source_weight_signature,
    validate_aggregated_library,
)


def _source(index: int) -> LibrarySourceRecord:
    token = f"{index + 1:064x}"
    return LibrarySourceRecord(
        token,
        f"{index + 101:064x}",
        f"image-{index}.png",
        (64, 96),
        capture_group_id=f"capture_{index % 2}",
        biological_context=BiologicalContext(
            "lupinus_mutabilis", accession_id=f"accession_{index % 2}"
        ),
    )


def _profile(index: int, *, duplicate_rows: int = 1) -> SpeciesForegroundColourBank:
    centres = np.repeat(
        np.asarray([[35.0 + index, 128.0, 128.0]], np.float32),
        duplicate_rows,
        axis=0,
    )
    return SpeciesForegroundColourBank(
        centres,
        np.ones_like(centres) * np.asarray((8.0, 4.0, 4.0), np.float32),
        np.ones(duplicate_rows, np.float32) / duplicate_rows,
        np.full(duplicate_rows, index, np.int32),
        np.full(duplicate_rows, 100, np.int64),
        FOREGROUND_COLOUR_SCHEMA,
        ("L*", "a*", "b*"),
    )


def _shape(index: int, seed_id: int, physical_seed_id: str) -> ShapeObservation:
    measurement = (
        8.0 + index,
        7.8 + index,
        6.2 + index * 0.2,
        1.25,
        38.0 + index,
        0.94,
        0.04,
        0.03,
        0.97,
        0.02,
        0.01,
        0.90,
        0.01,
        0.04,
        0.08,
        1.0,
        0.0,
        0.8,
        1.0,
    )
    return ShapeObservation(
        index,
        seed_id,
        physical_seed_id,
        BiologicalContext(
            "lupinus_mutabilis", accession_id=f"accession_{index % 2}"
        ).hierarchy_path(),
        "flat",
        "complete",
        True,
        measurement,
        tuple(0.02 for _ in measurement),
        tuple(1.0 + 0.02 * np.cos(value) for value in np.linspace(0, 2 * np.pi, 32, endpoint=False)),
    )


def _contribution(index: int, *, duplicate_rows: int = 1) -> ExtractedSourceContribution:
    return ExtractedSourceContribution(
        index,
        _source(index),
        _profile(index, duplicate_rows=duplicate_rows),
        None,
        None,
        None,
        None,
        (_shape(index, index + 1, f"physical:{index}"),),
    )


class SpeciesReferenceLibraryTests(unittest.TestCase):
    def test_extraction_uses_production_descriptor_schemas(self) -> None:
        height, width = 96, 112
        yy, xx = np.mgrid[:height, :width]
        labels = np.zeros((height, width), np.uint16)
        seed = ((xx - 56.0) / 28.0) ** 2 + ((yy - 48.0) / 20.0) ** 2 <= 1.0
        labels[seed] = 1
        image = np.full((height, width, 3), 215, np.uint8)
        image[seed] = (150, 165, 175)
        # A prominent coat-pattern transition supplies true non-physical edge
        # examples inside the safely inset reviewed seed.
        stripe = seed & (np.abs(xx - 56) <= 3) & (np.abs(yy - 48) <= 12)
        image[stripe] = (55, 65, 75)
        foreground = cv2.erode(
            seed.astype(np.uint8), np.ones((5, 5), np.uint8)
        ).astype(bool)
        references = ReferenceRegionBundle(
            (height, width),
            foreground=foreground,
            annotated_seeds=labels,
            seed_annotations=(
                SeedInstanceAnnotation(
                    1,
                    coat_pattern="white",
                    conditions_reviewed=True,
                    shape_reviewed=True,
                    outline_visibility="complete",
                    pose="flat",
                    physical_seed_id="seed-1",
                ),
            ),
            annotation_species="lupinus_mutabilis",
        )
        with tempfile.TemporaryDirectory() as temporary:
            source_path = Path(temporary) / "source.png"
            self.assertTrue(cv2.imwrite(str(source_path), image))
            extracted = extract_source_contribution(
                LibrarySourceInput(
                    source_path,
                    image,
                    references,
                    "a" * 64,
                    "Lupinus mutabilis",
                    BiologicalContext("lupinus_mutabilis"),
                    pixels_per_mm=10.0,
                    coat_pattern_vocabulary=("white", "banded_light"),
                    condition_vocabulary=("immature",),
                ),
                source_index=0,
                settings=LibraryExtractionSettings(
                    minimum_samples_per_prototype=2,
                    noise_working_maximum_dimension=512,
                    frequency_noise_working_maximum_dimension=512,
                    material_working_maximum_dimension=256,
                    trait_working_maximum_dimension=256,
                    edge_working_maximum_dimension=512,
                ),
            )
        self.assertEqual(extracted.foreground_noise.feature_names, NOISE_FEATURE_NAMES)
        self.assertEqual(extracted.material_prototypes.feature_names, MATERIAL_FEATURE_NAMES)
        self.assertEqual(extracted.edge_prototypes.feature_names, EDGE_FEATURE_NAMES)
        self.assertEqual(
            extracted.edge_prototypes.class_names,
            ("physical_edge", "non_edge"),
        )
        self.assertEqual(extracted.seed_traits.feature_names, MATERIAL_FEATURE_NAMES)
        self.assertTrue(extracted.shape_observations)

    def test_aggregation_is_order_independent_and_source_balanced(self) -> None:
        first = aggregate_contributions((_contribution(0), _contribution(1)))
        second = aggregate_contributions((_contribution(1), _contribution(0)))

        for key in ("foreground_colour",):
            left = first[key]
            right = second[key]
            self.assertTrue(np.array_equal(left.centres, right.centres))
            self.assertTrue(np.array_equal(left.weights, right.weights))
            self.assertEqual(
                source_weight_signature(left), ((0, 0.5), (1, 0.5))
            )
        self.assertEqual(
            tuple(item.source_sha256 for item in first["sources"]),
            tuple(item.source_sha256 for item in second["sources"]),
        )

    def test_duplicate_rows_cannot_increase_source_authority(self) -> None:
        ordinary = aggregate_contributions((_contribution(0), _contribution(1)))
        duplicated = aggregate_contributions(
            (_contribution(0, duplicate_rows=12), _contribution(1))
        )
        self.assertTrue(
            np.allclose(
                [value for _source, value in source_weight_signature(ordinary["foreground_colour"])],
                [value for _source, value in source_weight_signature(duplicated["foreground_colour"])],
            )
        )

    def test_source_exclusion_removes_profiles_and_refits_shape(self) -> None:
        aggregated = aggregate_contributions(
            (_contribution(0), _contribution(1), _contribution(2)),
            settings=LibraryAggregationSettings(
                shape_minimum_component_seeds=2,
                shape_shrinkage_seed_count=7.0,
                maximum_contour_modes=2,
            ),
        )
        artifact = SpeciesLibraryArtifact(
            _manifest(aggregated),
            foreground_colour=aggregated["foreground_colour"],
            shape_summary=aggregated["shape_summary"],
            dimensions_shape=aggregated["dimensions_shape"],
        )

        filtered, excluded = artifact.excluding_source_sha256(
            artifact.manifest.sources[0].source_sha256
        )

        self.assertEqual(excluded, (0,))
        assert_no_held_out_source(filtered.foreground_colour, 0)
        self.assertFalse(
            any(item.source_index == 0 for item in filtered.shape_summary.observations)
        )
        self.assertEqual(filtered.dimensions_shape.minimum_component_seeds, 2)
        self.assertEqual(filtered.dimensions_shape.shrinkage_seed_count, 7.0)
        self.assertEqual(filtered.dimensions_shape.maximum_contour_modes, 2)

    def test_fork_recovery_preserves_per_source_contributions(self) -> None:
        aggregated = aggregate_contributions((_contribution(0), _contribution(1)))
        artifact = SpeciesLibraryArtifact(
            _manifest(aggregated),
            foreground_colour=aggregated["foreground_colour"],
            shape_summary=aggregated["shape_summary"],
            dimensions_shape=aggregated["dimensions_shape"],
        )
        recovered = contributions_from_artifact(artifact)
        rebuilt = aggregate_contributions(
            tuple(reindex_contribution(value, index) for index, value in enumerate(recovered))
        )
        self.assertTrue(
            np.array_equal(
                aggregated["foreground_colour"].centres,
                rebuilt["foreground_colour"].centres,
            )
        )
        self.assertEqual(
            tuple(aggregated["shape_summary"].observations),
            tuple(rebuilt["shape_summary"].observations),
        )

    def test_publish_load_export_import_and_tamper_detection(self) -> None:
        aggregated = aggregate_contributions((_contribution(0), _contribution(1)))
        artifact = SpeciesLibraryArtifact(
            _manifest(aggregated),
            foreground_colour=aggregated["foreground_colour"],
            shape_summary=aggregated["shape_summary"],
            dimensions_shape=aggregated["dimensions_shape"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = SpeciesLibraryStore(root / "catalogue")
            published = store.publish(artifact)
            pin = SpeciesLibraryPin(
                published.manifest.library_id,
                published.manifest.version,
                published.manifest.species_id,
                published.manifest.content_sha256,
            )
            loaded = store.load(pin)
            self.assertEqual(loaded.manifest.content_sha256, pin.sha256)
            changed_manifest = replace(
                published.manifest,
                aggregation_settings_sha256="d" * 64,
            )
            with self.assertRaisesRegex(Exception, "immutable"):
                store.publish(replace(artifact, manifest=changed_manifest))

            bundle = store.export_library(pin, root / "portable")
            imported_store = SpeciesLibraryStore(root / "imported")
            imported = imported_store.import_bundle(bundle)
            self.assertEqual(imported.manifest.content_sha256, pin.sha256)

            banks_path = imported_store.version_path(
                pin.species_id, pin.library_id, pin.version
            ) / "banks.npz"
            payload = bytearray(banks_path.read_bytes())
            payload[len(payload) // 2] ^= 0x01
            banks_path.write_bytes(payload)
            with self.assertRaises(InvalidSpeciesLibrary):
                imported_store.load(pin)

    def test_validation_reports_grouped_shape_and_source_metrics(self) -> None:
        aggregated = aggregate_contributions(
            (_contribution(0), _contribution(1), _contribution(2))
        )
        report = validate_aggregated_library(aggregated)
        colour = report.for_product(next(item.product for item in report.products if item.product.value == "foreground_colour"))
        shape = report.for_product(next(item.product for item in report.products if item.product.value == "shape_summary"))
        self.assertAlmostEqual(colour.metrics["source_balance_error"], 0.0)
        self.assertIn("leave_one_source_out_distance", colour.metrics)
        self.assertEqual(shape.metrics["physical_seed_count"], 3.0)
        self.assertEqual(shape.metrics["accession_count"], 2.0)


def _manifest(aggregated):
    report = validate_aggregated_library(aggregated)
    return create_manifest(
        library_id="library-1",
        version="1",
        species_id="lupinus_mutabilis",
        species_display_name="Lupinus mutabilis",
        trait_vocabulary_sha256="a" * 64,
        seed_fiddle_version="test",
        sources=aggregated["sources"],
        products=report.products,
        descriptor_schemas={"foreground_colour": FOREGROUND_COLOUR_SCHEMA},
        extraction_settings_sha256="b" * 64,
        aggregation_settings_sha256="c" * 64,
        corrected_colour_space_version="opencv-lab-u8-v1",
    )


if __name__ == "__main__":
    unittest.main()
