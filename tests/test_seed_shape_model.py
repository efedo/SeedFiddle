from __future__ import annotations

import unittest

import cv2
import numpy as np

from seedvision.measurement import (
    assemble_seed_dimensions_shape_model,
    assess_candidate_shape,
    fit_species_dimensions_shape_bank,
    measure_reviewed_seed_instances,
    measure_shape_mask,
    shape_observations_from_summary,
)
from seedvision.persistence.reference_regions import SeedInstanceAnnotation
from seedvision.reference_library.contracts import BiologicalContext


def _ellipse_mask(
    *, angle: float = 0.0, axes: tuple[int, int] = (38, 25), nub: bool = False
) -> np.ndarray:
    mask = np.zeros((160, 180), np.uint8)
    cv2.ellipse(mask, (90, 80), axes, angle, 0, 360, 1, -1)
    if nub:
        cv2.circle(mask, (90 + axes[0], 80), 7, 1, -1)
    return mask > 0


def _reviewed(seed_id: int, **changes) -> SeedInstanceAnnotation:
    values = dict(
        seed_id=seed_id,
        shape_reviewed=True,
        outline_visibility="complete",
        pose="flat",
    )
    values.update(changes)
    return SeedInstanceAnnotation(**values)


class SeedDimensionsShapeTests(unittest.TestCase):
    def test_synthetic_ellipse_is_rotation_invariant(self) -> None:
        first = measure_shape_mask(_ellipse_mask(angle=0), contour_samples=128)
        rotated = measure_shape_mask(_ellipse_mask(angle=43), contour_samples=128)
        self.assertAlmostEqual(first.maximum_span, rotated.maximum_span, delta=2.5)
        self.assertAlmostEqual(first.ovality, rotated.ovality, delta=0.08)
        self.assertAlmostEqual(first.projected_area, rotated.projected_area, delta=80)
        self.assertGreater(first.ovality, 1.35)

    def test_circle_has_uncertain_orientation_and_nub_is_recorded(self) -> None:
        circle = measure_shape_mask(_ellipse_mask(axes=(30, 30)))
        nub = measure_shape_mask(
            _ellipse_mask(axes=(34, 25), nub=True),
            hilum_point=(124.0, 80.0),
        )
        self.assertGreater(circle.ellipse.orientation_uncertainty_degrees, 50.0)
        self.assertIsNotNone(nub.local_feature)
        self.assertGreater(nub.non_ellipticity, 0.01)

    def test_review_and_visibility_are_explicit_eligibility_gates(self) -> None:
        labels = np.zeros((180, 220), np.uint16)
        labels[20:180, :180][_ellipse_mask()] = 1
        second = np.zeros_like(labels, np.uint8)
        cv2.ellipse(second, (175, 80), (30, 22), 0, 0, 360, 1, -1)
        labels[second > 0] = 2
        summary = measure_reviewed_seed_instances(
            labels,
            (
                _reviewed(1),
                _reviewed(2, outline_visibility="partly_occluded"),
            ),
        )
        self.assertEqual(summary.eligible_count, 1)
        rejected = next(item for item in summary.observations if item.seed_id == 2)
        self.assertIn("partly_occluded", rejected.exclusion_reason)

    def test_repeated_views_count_as_one_physical_seed(self) -> None:
        observations = []
        context = BiologicalContext("lupinus_mutabilis", accession_id="line_a")
        for source_index, physical_id, angle in (
            (0, "seed:a", 0),
            (1, "seed:a", 35),
            (2, "seed:b", 15),
        ):
            labels = np.zeros((160, 180), np.uint16)
            labels[_ellipse_mask(angle=angle)] = 1
            summary = measure_reviewed_seed_instances(
                labels,
                (_reviewed(1, physical_seed_id=physical_id),),
                pixels_per_mm=10.0,
            )
            observations.extend(
                shape_observations_from_summary(
                    summary,
                    source_index=source_index,
                    biological_context=context,
                )
            )
        bank = fit_species_dimensions_shape_bank(observations)
        component, path = bank.select_component(context, "flat")
        self.assertEqual(path, context.hierarchy_path())
        self.assertIsNotNone(component)
        self.assertEqual(component.effective_physical_seed_count, 2.0)
        self.assertEqual(component.calibrated_physical_seed_count, 2.0)
        self.assertEqual(component.source_count, 3)

    def test_uncalibrated_shapes_form_dimensionless_family_only(self) -> None:
        observations = []
        context = BiologicalContext("lupinus_mutabilis")
        for source_index, axes in enumerate(((38, 25), (42, 27), (34, 22))):
            labels = np.zeros((160, 180), np.uint16)
            labels[_ellipse_mask(axes=axes)] = 1
            summary = measure_reviewed_seed_instances(labels, (_reviewed(1),))
            observations.extend(
                shape_observations_from_summary(
                    summary,
                    source_index=source_index,
                    biological_context=context,
                )
            )
        bank = fit_species_dimensions_shape_bank(observations)
        component, _path = bank.select_component(context, "flat")
        self.assertIsNotNone(component)
        self.assertFalse(component.physical_dimensions_available)
        self.assertEqual(component.calibrated_physical_seed_count, 0.0)
        self.assertEqual(component.dimensionless_physical_seed_count, 3.0)
        self.assertAlmostEqual(component.mean[0], 1.0, places=6)

    def test_hierarchy_falls_back_without_confusing_optional_levels(self) -> None:
        context_a = BiologicalContext(
            "lupinus_mutabilis", accession_id="accession_a"
        )
        context_b = BiologicalContext(
            "lupinus_mutabilis", lineage_group_id="lineage_b"
        )
        self.assertNotEqual(context_a.hierarchy_path(), context_b.hierarchy_path())
        observations = []
        for index, context in enumerate((context_a, context_a, context_b, context_b)):
            labels = np.zeros((160, 180), np.uint16)
            labels[_ellipse_mask(axes=(36 + index, 24))] = 1
            summary = measure_reviewed_seed_instances(
                labels, (_reviewed(1),), pixels_per_mm=10.0
            )
            observations.extend(
                shape_observations_from_summary(
                    summary,
                    source_index=index,
                    biological_context=context,
                )
            )
        bank = fit_species_dimensions_shape_bank(observations)
        component, selected = bank.select_component(context_a, "flat")
        self.assertIsNotNone(component)
        self.assertEqual(selected, context_a.hierarchy_path())
        unseen = BiologicalContext(
            "lupinus_mutabilis", accession_id="unseen_accession"
        )
        _fallback, fallback_path = bank.select_component(unseen, "flat")
        self.assertEqual(fallback_path, ("lupinus_mutabilis",))

    def test_uncalibrated_local_shapes_do_not_update_physical_library_prior(self) -> None:
        context = BiologicalContext("lupinus_mutabilis")
        calibrated = []
        for index in range(2):
            labels = np.zeros((160, 180), np.uint16)
            labels[_ellipse_mask(axes=(38 + index, 25))] = 1
            summary = measure_reviewed_seed_instances(
                labels, (_reviewed(1),), pixels_per_mm=10.0
            )
            calibrated.extend(
                shape_observations_from_summary(
                    summary,
                    source_index=index,
                    biological_context=context,
                )
            )
        bank = fit_species_dimensions_shape_bank(calibrated)
        local_labels = np.zeros((160, 180), np.uint16)
        local_labels[_ellipse_mask(axes=(55, 15))] = 1
        local = measure_reviewed_seed_instances(local_labels, (_reviewed(1),))
        model = assemble_seed_dimensions_shape_model(
            local_summary=local,
            legacy_processing_diameter_px=90.0,
            source_mode="species_library_prior_and_current",
            biological_context=context,
            library_bank=bank,
        )
        self.assertFalse(model.in_sample_local_update)
        self.assertTrue(any("uncalibrated" in value for value in model.warnings))
        self.assertEqual(model.compatibility_seed_diameter_px, 90.0)

    def test_out_of_family_assessment_abstains(self) -> None:
        context = BiologicalContext("lupinus_mutabilis")
        observations = []
        for index, axes in enumerate(((38, 25), (39, 25), (37, 24))):
            labels = np.zeros((160, 180), np.uint16)
            labels[_ellipse_mask(axes=axes)] = 1
            summary = measure_reviewed_seed_instances(
                labels, (_reviewed(1),), pixels_per_mm=10.0
            )
            observations.extend(
                shape_observations_from_summary(
                    summary,
                    source_index=index,
                    biological_context=context,
                )
            )
        bank = fit_species_dimensions_shape_bank(observations)
        component, _path = bank.select_component(context, "flat")
        extreme = np.asarray(component.mean) * 1.0
        extreme[3] = 5.0
        assessment = assess_candidate_shape(extreme, component)
        self.assertTrue(assessment.out_of_family)
        self.assertLess(assessment.compatibility, 0.2)


if __name__ == "__main__":
    unittest.main()
