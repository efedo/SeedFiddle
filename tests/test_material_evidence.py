from __future__ import annotations

import unittest

import numpy as np

from seedvision.cuda import CudaContext, hierarchical_material_evidence


class MaterialEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = CudaContext.resolve(requested="cpu")
        self.valid = np.full((9, 9), 255, np.uint8)

    def _constant(self, probability: float) -> np.ndarray:
        return np.full(
            self.valid.shape, round(float(probability) * 255.0), np.uint8
        )

    def _result(self, **overrides):
        arguments = dict(
            valid_mask=self.valid,
            seed_diameter=12.0,
            foreground_colour=self._constant(0.80),
            foreground_noise=self._constant(0.80),
            background_colour=self._constant(0.20),
            background_noise=self._constant(0.20),
            morphology_fraction=0.0,
            cuda_context=self.context,
        )
        arguments.update(overrides)
        return hierarchical_material_evidence(**arguments)

    def test_four_decision_masses_sum_to_one(self) -> None:
        result = self._result()
        total = sum(
            np.asarray(raster, dtype=np.int16)
            for raster in (
                result.seed_probability,
                result.nonseed_probability,
                result.ambiguity_probability,
                result.unknown_probability,
            )
        )
        np.testing.assert_allclose(total, 255, atol=2)
        subtype_total = sum(
            np.asarray(raster, dtype=np.int16)
            for raster in (
                result.conditional_background_probability,
                result.conditional_other_probability,
                result.conditional_ambiguity_probability,
                result.conditional_unknown_probability,
            )
        )
        np.testing.assert_allclose(subtype_total, 255, atol=2)
        self.assertGreater(
            int(np.asarray(result.seed_probability)[4, 4]),
            int(np.asarray(result.nonseed_probability)[4, 4]),
        )

    def test_invalid_pixels_have_no_material_mass(self) -> None:
        valid = self.valid.copy()
        valid[:3] = 0
        result = self._result(valid_mask=valid)
        for raster in (
            result.seed_probability,
            result.nonseed_probability,
            result.ambiguity_probability,
            result.unknown_probability,
        ):
            self.assertFalse(np.any(np.asarray(raster)[:3]))

    def test_seed_annotation_does_not_overwrite_raw_material_evidence(self) -> None:
        without_annotation = self._result(
            foreground_colour=self._constant(0.25),
            foreground_noise=self._constant(0.25),
            background_colour=self._constant(0.95),
            background_noise=self._constant(0.95),
        )
        seed_annotation = np.zeros_like(self.valid)
        seed_annotation[4, 4] = 255
        with_annotation = self._result(
            foreground_colour=self._constant(0.25),
            foreground_noise=self._constant(0.25),
            background_colour=self._constant(0.95),
            background_noise=self._constant(0.95),
            seed_reference_mask=seed_annotation,
        )
        np.testing.assert_array_equal(
            np.asarray(with_annotation.seed_probability),
            np.asarray(without_annotation.seed_probability),
        )
        np.testing.assert_array_equal(
            np.asarray(with_annotation.background_support),
            np.asarray(without_annotation.background_support),
        )

    def test_other_is_positive_nonseed_without_suppressing_background(self) -> None:
        result = self._result(
            foreground_colour=self._constant(0.10),
            foreground_noise=self._constant(0.10),
            background_colour=self._constant(0.90),
            background_noise=self._constant(0.90),
            other_colour=self._constant(0.90),
            other_noise=self._constant(0.90),
        )
        self.assertGreater(int(np.asarray(result.nonseed_support)[4, 4]), 250)
        background_subtype = int(
            np.asarray(result.conditional_background_probability)[4, 4]
        )
        other_subtype = int(
            np.asarray(result.conditional_other_probability)[4, 4]
        )
        subtype_ambiguity = int(
            np.asarray(result.conditional_ambiguity_probability)[4, 4]
        )
        self.assertLess(background_subtype, 20)
        self.assertLess(other_subtype, 20)
        self.assertLess(abs(background_subtype - other_subtype), 3)
        self.assertGreater(subtype_ambiguity, 220)

    def test_background_annotation_does_not_zero_foreground_evidence(self) -> None:
        background_annotation = np.zeros_like(self.valid)
        background_annotation[4, 4] = 255
        result = self._result(
            foreground_colour=self._constant(0.95),
            foreground_noise=self._constant(0.95),
            background_colour=self._constant(0.95),
            background_noise=self._constant(0.95),
            background_reference_mask=background_annotation,
        )
        self.assertGreater(int(np.asarray(result.seed_support)[4, 4]), 220)
        self.assertGreater(int(np.asarray(result.background_support)[4, 4]), 220)
        self.assertGreater(int(np.asarray(result.ambiguity_probability)[4, 4]), 220)

    def test_conflicting_seed_and_nonseed_is_ambiguous(self) -> None:
        result = self._result(
            foreground_colour=self._constant(0.95),
            foreground_noise=self._constant(0.95),
            background_colour=self._constant(0.95),
            background_noise=self._constant(0.95),
        )
        ambiguity = int(np.asarray(result.ambiguity_probability)[4, 4])
        seed = int(np.asarray(result.seed_probability)[4, 4])
        nonseed = int(np.asarray(result.nonseed_probability)[4, 4])
        self.assertGreater(ambiguity, seed * 10)
        self.assertGreater(ambiguity, nonseed * 10)

    def test_cross_matching_source_loses_authority_on_reviewed_masks(self) -> None:
        seed_reference = np.zeros_like(self.valid)
        seed_reference[:4] = 255
        other_reference = np.zeros_like(self.valid)
        other_reference[5:] = 255
        result = self._result(
            foreground_colour=self._constant(0.80),
            foreground_noise=None,
            background_colour=self._constant(0.0),
            background_noise=None,
            other_colour=self._constant(0.90),
            other_noise=None,
            seed_reference_mask=seed_reference,
            other_reference_mask=other_reference,
        )
        reliabilities = dict(result.source_reliabilities)
        self.assertEqual(reliabilities["other_colour"], 0.0)
        self.assertEqual(int(np.asarray(result.other_support)[4, 4]), 0)

    def test_background_other_overlap_is_not_a_subtype_negative(self) -> None:
        seed_reference = np.zeros_like(self.valid)
        seed_reference[:3] = 255
        background_reference = np.zeros_like(self.valid)
        background_reference[3:6] = 255
        other_reference = np.zeros_like(self.valid)
        other_reference[6:] = 255
        shared_nonseed = np.full_like(self.valid, 230)
        shared_nonseed[:3] = 10
        result = self._result(
            foreground_colour=self._constant(0.1),
            foreground_noise=None,
            background_colour=shared_nonseed,
            background_noise=None,
            other_colour=shared_nonseed,
            other_noise=None,
            seed_reference_mask=seed_reference,
            background_reference_mask=background_reference,
            other_reference_mask=other_reference,
        )
        reliabilities = dict(result.source_reliabilities)
        self.assertGreater(reliabilities["background_colour"], 0.95)
        self.assertGreater(reliabilities["other_colour"], 0.95)


if __name__ == "__main__":
    unittest.main()
