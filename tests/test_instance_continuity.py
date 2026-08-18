from __future__ import annotations

import unittest

import numpy as np

from seedvision.annotation import summarize_instance_continuity


class InstanceContinuityTests(unittest.TestCase):
    def test_empty_raster_has_an_empty_summary(self) -> None:
        summary = summarize_instance_continuity(
            np.zeros((0, 12), dtype=np.uint16)
        )

        self.assertEqual(summary.identifiers, ())
        self.assertEqual(summary.pixel_count, 0)
        self.assertEqual(summary.component_counts, ())
        self.assertEqual(summary.disconnected, ())
        self.assertEqual(summary.disconnected_identifiers, ())
        self.assertEqual(summary.component_count(9), 0)

    def test_reports_exact_component_counts_for_multiple_ids(self) -> None:
        labels = np.zeros((14, 18), dtype=np.uint16)
        labels[1:4, 1:5] = 1
        labels[8:11, 11:15] = 1
        labels[2:7, 8:12] = 7
        labels[12, 1] = 12
        labels[12, 4] = 12
        labels[12, 7] = 12

        summary = summarize_instance_continuity(labels)

        self.assertEqual(summary.identifiers, (1, 7, 12))
        self.assertEqual(summary.pixel_count, int(np.count_nonzero(labels)))
        self.assertEqual(summary.component_counts, (2, 1, 3))
        self.assertEqual(summary.component_count(7), 1)
        self.assertEqual(summary.disconnected, ((1, 2), (12, 3)))
        self.assertEqual(summary.disconnected_identifiers, (1, 12))

    def test_diagonal_pixels_connect_only_when_their_ids_match(self) -> None:
        # The two categorical components cross diagonally.  A single binary
        # foreground pass would merge all four pixels, while independent
        # 8-connectivity correctly gives one component for each ID.
        labels = np.array([[1, 2], [2, 1]], dtype=np.uint16)

        summary = summarize_instance_continuity(labels)

        self.assertEqual(summary.identifiers, (1, 2))
        self.assertEqual(summary.component_counts, (1, 1))
        self.assertEqual(summary.disconnected, ())

    def test_later_row_can_join_two_active_runs(self) -> None:
        labels = np.array(
            [
                [4, 4, 0, 4, 4],
                [4, 4, 4, 4, 4],
            ],
            dtype=np.uint16,
        )

        summary = summarize_instance_continuity(labels)

        self.assertEqual(summary.component_count(4), 1)

    def test_empty_row_finalizes_separate_areas(self) -> None:
        labels = np.array(
            [
                [0, 9, 9, 0],
                [0, 0, 0, 0],
                [0, 9, 9, 0],
            ],
            dtype=np.uint16,
        )

        summary = summarize_instance_continuity(labels)

        self.assertEqual(summary.component_count(9), 2)
        self.assertEqual(summary.disconnected, ((9, 2),))

    def test_non_contiguous_uint16_view_and_largest_id_are_supported(self) -> None:
        storage = np.zeros((6, 12), dtype=np.uint16)
        labels = storage[:, ::2]
        self.assertFalse(labels.flags.c_contiguous)
        labels[1, 1] = np.iinfo(np.uint16).max
        labels[2, 2] = np.iinfo(np.uint16).max

        summary = summarize_instance_continuity(labels)

        self.assertEqual(summary.identifiers, (np.iinfo(np.uint16).max,))
        self.assertEqual(summary.pixel_count, 2)
        self.assertEqual(summary.component_counts, (1,))

    def test_matches_independent_opencv_components_on_random_small_maps(self) -> None:
        import cv2

        generator = np.random.default_rng(20260817)
        for _case in range(40):
            labels = generator.integers(0, 6, size=(11, 15), dtype=np.uint16)
            # Keep a realistic amount of empty space while retaining many
            # adversarial same-ID diagonal contacts.
            labels[generator.random(labels.shape) < 0.45] = 0
            summary = summarize_instance_continuity(labels)
            expected = {}
            for identifier in np.unique(labels):
                if identifier == 0:
                    continue
                count, _components = cv2.connectedComponents(
                    np.uint8(labels == identifier), connectivity=8
                )
                expected[int(identifier)] = count - 1
            self.assertEqual(
                dict(zip(summary.identifiers, summary.component_counts, strict=True)),
                expected,
            )

    def test_rejects_non_integer_or_out_of_range_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            summarize_instance_continuity(np.zeros((2, 3, 1), dtype=np.uint16))
        with self.assertRaisesRegex(TypeError, "integer"):
            summarize_instance_continuity(np.zeros((2, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "between 0 and 65,535"):
            summarize_instance_continuity(np.array([[0, -1]], dtype=np.int32))
        with self.assertRaisesRegex(ValueError, "between 0 and 65,535"):
            summarize_instance_continuity(np.array([[65_536]], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
