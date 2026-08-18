from __future__ import annotations

import unittest

import cv2
import numpy as np

from seedvision.annotation.instance_references import (
    instance_boundary_references,
    instance_boundary_references_from_edge_evidence,
)


class InstanceReferenceEvidenceTests(unittest.TestCase):
    def test_distinct_instance_contact_is_physical_and_interiors_are_inset(self) -> None:
        labels = np.zeros((64, 72), np.uint16)
        labels[12:52, 8:36] = 1
        labels[12:52, 36:64] = 2
        candidates = np.zeros(labels.shape, bool)
        candidates[20:44, 20] = True
        candidates[20:44, 52] = True
        candidates[20:44, 34] = True  # Too close to the shared contact.

        references = instance_boundary_references(
            labels,
            20.0,
            interior_buffer_fraction=0.15,
            internal_edge_candidates=candidates,
        )

        # Both labelled sides of the shared contact remain physical evidence.
        self.assertTrue(np.all(references.physical_edge[13:51, 35:37]))
        self.assertTrue(references.non_edge[32, 20])
        self.assertTrue(references.non_edge[32, 52])
        self.assertFalse(references.non_edge[32, 34])
        self.assertTrue(references.safe_interior[32, 20])
        self.assertFalse(np.any(references.physical_edge & references.non_edge))
        self.assertAlmostEqual(references.interior_buffer_px, 3.0)

    def test_small_or_partial_regions_do_not_invent_non_edge_interiors(self) -> None:
        labels = np.zeros((28, 28), np.uint16)
        labels[10:15, 10:15] = 7

        references = instance_boundary_references(
            labels,
            30.0,
            interior_buffer_fraction=0.20,
        )

        self.assertGreater(np.count_nonzero(references.physical_edge), 0)
        self.assertEqual(np.count_nonzero(references.non_edge), 0)

    def test_export_style_edge_evidence_selects_only_sparse_internal_ridges(self) -> None:
        labels = np.zeros((80, 80), np.uint16)
        cv2.circle(labels, (40, 40), 28, 1, -1)
        edge = np.zeros(labels.shape, np.uint8)
        ridges = np.zeros(labels.shape, np.uint8)
        cv2.line(edge, (22, 40), (58, 40), 220, 3)
        cv2.line(ridges, (22, 40), (58, 40), 255, 1)

        references = instance_boundary_references_from_edge_evidence(
            labels,
            56.0,
            edge,
            ridges,
        )

        self.assertTrue(references.non_edge[40, 40])
        self.assertFalse(references.non_edge[40, 18])
        self.assertTrue(references.physical_edge[40, 12])
        self.assertLess(
            np.count_nonzero(references.non_edge),
            np.count_nonzero(references.safe_interior) // 8,
        )

    def test_rejects_non_integer_and_negative_instance_ids(self) -> None:
        with self.assertRaises(ValueError):
            instance_boundary_references(np.zeros((4, 4), np.float32), 10.0)
        with self.assertRaises(ValueError):
            instance_boundary_references(-np.ones((4, 4), np.int16), 10.0)

    def test_retired_manual_boundary_arguments_are_not_accepted(self) -> None:
        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        image = np.full((96, 96, 3), 220, np.uint8)
        cv2.circle(image, (48, 48), 26, (40, 85, 165), -1)
        cv2.line(image, (31, 48), (65, 48), (205, 205, 205), 3)
        labels = np.zeros((96, 96), np.uint16)
        cv2.circle(labels, (48, 48), 26, 1, -1)
        obsolete_physical = np.zeros((96, 96), bool)
        obsolete_non_edge = np.zeros((96, 96), bool)
        obsolete_physical[5:20, 5:20] = True
        obsolete_non_edge[35:60, 35:60] = True
        common = dict(
            crop=image,
            valid_mask=np.full((96, 96), 255, np.uint8),
            centers=np.empty((0, 2), np.float32),
            radii=np.empty((0,), np.float32),
            seed_diameter=52.0,
            offset_x=0,
            offset_y=0,
            seed_instance_annotations=labels,
        )

        with self.assertRaises(TypeError):
            build_analysis_layers(
                **common,
                physical_edge_reference_mask=obsolete_physical,
                non_edge_reference_mask=obsolete_non_edge,
                settings=AnalysisLayerSettings(),
            )

    def test_generic_edges_do_not_publish_semantic_probabilities(self) -> None:
        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        image = np.zeros((96, 96, 3), np.uint8)
        image[:, 48:] = 255
        material_foreground = np.zeros((96, 96), bool)
        material_foreground[20:36, 60:76] = True
        layers = build_analysis_layers(
            crop=image,
            valid_mask=np.full((96, 96), 255, np.uint8),
            centers=np.empty((0, 2), np.float32),
            radii=np.empty((0,), np.float32),
            seed_diameter=40.0,
            offset_x=0,
            offset_y=0,
            foreground_reference_mask=material_foreground,
            settings=AnalysisLayerSettings(),
            instance_masks_enabled=False,
        )

        counts = dict(layers.reference_texture_profile.class_sample_counts)
        self.assertEqual(counts["physical_edge"], 0)
        self.assertEqual(counts["non_edge"], 0)
        self.assertEqual(int(np.asarray(layers.physical_edge_probability).max()), 0)
        self.assertEqual(int(np.asarray(layers.non_edge_probability).max()), 0)

    def test_flat_annotated_interiors_do_not_flood_non_edge_training(self) -> None:
        from seedvision.visualization import AnalysisLayerSettings, build_analysis_layers

        image = np.full((96, 96, 3), 220, np.uint8)
        cv2.circle(image, (48, 48), 28, (60, 100, 150), -1)
        labels = np.zeros((96, 96), np.uint16)
        cv2.circle(labels, (48, 48), 28, 1, -1)
        layers = build_analysis_layers(
            crop=image,
            valid_mask=np.full((96, 96), 255, np.uint8),
            centers=np.empty((0, 2), np.float32),
            radii=np.empty((0,), np.float32),
            seed_diameter=56.0,
            offset_x=0,
            offset_y=0,
            seed_instance_annotations=labels,
            settings=AnalysisLayerSettings(),
            instance_masks_enabled=False,
        )

        counts = dict(layers.reference_texture_profile.class_sample_counts)
        self.assertGreater(counts["physical_edge"], 0)
        self.assertEqual(counts["non_edge"], 0)
        self.assertEqual(int(np.asarray(layers.non_edge_probability).max()), 0)


if __name__ == "__main__":
    unittest.main()
