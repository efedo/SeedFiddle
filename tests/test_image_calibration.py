from __future__ import annotations

import unittest


class ImageCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(f"Calibration dependencies unavailable: {error}")

    @staticmethod
    def _synthetic_calibration_image(
        rotation_degrees: float = 0.0,
        *,
        swatch_offset: tuple[int, int] = (0, 0),
        swatch_half_size: tuple[int, int] = (38, 43),
    ):
        import cv2
        import numpy as np

        image = np.full((800, 1200, 3), 238, dtype=np.uint8)
        cv2.rectangle(image, (45, 35), (465, 755), (25, 25, 25), -1)
        colours = (
            (235, 235, 235), (180, 110, 40), (35, 150, 225), (165, 205, 90),
            (205, 205, 205), (180, 45, 155), (210, 70, 60), (180, 145, 215),
            (165, 165, 165), (30, 220, 225), (80, 80, 210), (60, 125, 65),
            (125, 125, 125), (45, 45, 210), (155, 70, 90), (210, 155, 105),
            (85, 85, 85), (70, 180, 65), (65, 195, 160), (155, 120, 75),
            (45, 45, 45), (210, 65, 40), (40, 175, 220), (95, 65, 135),
        )
        for row in range(6):
            for column in range(4):
                center_x = 105 + column * 100
                center_y = 100 + row * 118
                center_x += swatch_offset[0]
                center_y += swatch_offset[1]
                colour = colours[row * 4 + column]
                cv2.rectangle(
                    image,
                    (
                        center_x - swatch_half_size[0],
                        center_y - swatch_half_size[1],
                    ),
                    (
                        center_x + swatch_half_size[0],
                        center_y + swatch_half_size[1],
                    ),
                    colour,
                    -1,
                )

        cv2.rectangle(image, (510, 635), (1135, 760), (224, 224, 224), -1)
        cv2.rectangle(image, (510, 635), (1135, 760), (55, 55, 55), 2)
        for tick in range(151):
            x = 525 + tick * 4
            tick_height = 48 if tick % 10 == 0 else 32 if tick % 5 == 0 else 20
            cv2.line(image, (x, 645), (x, 645 + tick_height), (25, 25, 25), 1)

        cast = image.astype(np.float32)
        cast[:, :, 0] *= 1.12
        cast[:, :, 2] *= 0.88
        image = np.uint8(np.clip(cast, 0, 255))
        if rotation_degrees:
            matrix = cv2.getRotationMatrix2D((600, 400), rotation_degrees, 1.0)
            image = cv2.warpAffine(
                image,
                matrix,
                (1200, 800),
                borderMode=cv2.BORDER_REPLICATE,
            )
        return image

    def test_detects_swatch_grid_and_ruler_independently(self) -> None:
        from seedvision.calibration import detect_colour_card, detect_ruler

        image = self._synthetic_calibration_image(rotation_degrees=2.0)
        card = detect_colour_card(image)
        ruler = detect_ruler(image)
        self.assertGreaterEqual(card.detected_swatch_count, 18)
        self.assertEqual(len(card.swatches), 24)
        self.assertGreater(card.confidence, 0.45)
        self.assertGreater(ruler.length_px, 500)
        self.assertGreater(ruler.confidence, 0.6)
        self.assertAlmostEqual(card.angle_degrees, -2.0, delta=1.0)
        self.assertAlmostEqual(ruler.angle_degrees, -2.0, delta=1.0)

    def test_reference_card_and_printed_ruler_scale_are_aligned(self) -> None:
        from seedvision.calibration import detect_colour_card, detect_ruler

        image = self._synthetic_calibration_image()
        card = detect_colour_card(image)
        ruler = detect_ruler(image)
        self.assertEqual(card.detected_swatch_count, 24)
        bounds = sorted(card.bounds, key=lambda point: point[0] + point[1])
        self.assertAlmostEqual(bounds[0][0], 45.0, delta=6.0)
        self.assertAlmostEqual(bounds[0][1], 35.0, delta=6.0)
        ruler_x = sorted((ruler.endpoint_a[0], ruler.endpoint_b[0]))
        self.assertAlmostEqual(ruler_x[0], 525.0, delta=4.0)
        self.assertAlmostEqual(ruler_x[1], 1125.0, delta=4.0)
        self.assertAlmostEqual(ruler.length_px, 600.0, delta=5.0)
        self.assertAlmostEqual(ruler.tick_pitch_px, 4.0, delta=0.15)
        self.assertGreaterEqual(ruler.matched_tick_count, 145)
        self.assertEqual(
            len(ruler.supported_tick_points), ruler.matched_tick_count
        )
        self.assertEqual(len(ruler.outline_points), 4)
        self.assertEqual(
            len(ruler.metric_tick_points), ruler.matched_tick_count
        )
        self.assertEqual(len(ruler.metric_tick_segments), 151)
        self.assertEqual(set(ruler.metric_tick_classes), {0, 1, 2})
        self.assertEqual(
            ruler.metric_unit_divider_indices,
            tuple(range(0, 151, 10)),
        )
        self.assertAlmostEqual(ruler.metric_pixels_per_mm, 4.0, delta=0.15)
        self.assertGreaterEqual(len(ruler.metric_labels), 16)
        self.assertTrue(ruler.scale_reliable)

    def test_metric_tick_family_beats_competing_reverse_scale_and_barcode(self) -> None:
        import cv2

        from seedvision.calibration import detect_ruler

        image = self._synthetic_calibration_image()
        for tick in range(97):
            x = round(520 + tick * 6.25)
            tick_height = (
                52
                if tick % 16 == 0
                else 43
                if tick % 8 == 0
                else 35
                if tick % 4 == 0
                else 28
                if tick % 2 == 0
                else 20
            )
            cv2.line(image, (x, 748), (x, 748 - tick_height), (20, 20, 20), 1)
        for index, x in enumerate(range(930, 1106, 4)):
            if index % 3:
                cv2.line(image, (x, 708), (x, 742), (8, 8, 8), 2)

        ruler = detect_ruler(image)

        ruler_x = sorted((ruler.endpoint_a[0], ruler.endpoint_b[0]))
        self.assertAlmostEqual(ruler_x[0], 525.0, delta=4.0)
        self.assertAlmostEqual(ruler_x[1], 1125.0, delta=4.0)
        self.assertAlmostEqual(ruler.tick_pitch_px, 4.0, delta=0.15)
        self.assertGreaterEqual(ruler.matched_tick_count, 140)
        self.assertEqual(len(ruler.metric_tick_segments), 151)
        self.assertEqual(len(ruler.imperial_tick_segments), 97)
        self.assertEqual(set(ruler.metric_tick_classes), {0, 1, 2})
        self.assertGreaterEqual(len(set(ruler.imperial_tick_classes)), 4)
        self.assertEqual(
            ruler.imperial_unit_divider_indices,
            tuple(range(0, 97, 16)),
        )
        self.assertTrue(ruler.imperial_scale_reliable)
        self.assertTrue(ruler.imperial_hierarchy_reliable)
        self.assertGreater(ruler.imperial_hierarchy_consistency, 0.80)
        self.assertGreater(ruler.imperial_pixels_per_mm, 0.0)
        self.assertIsNotNone(ruler.scale_disagreement_percent)
        # This synthetic competing scale uses only approximate hand-drawn
        # imperial spacing.  It should remain a plausible cross-check while
        # exposing the residual disagreement instead of silently merging the
        # two estimates.
        self.assertLess(ruler.scale_disagreement_percent, 8.0)

    def test_real_dual_scale_ruler_uses_visible_metric_tick_span(self) -> None:
        import cv2
        import numpy as np
        from pathlib import Path

        from seedvision.calibration import calibrate_image

        path = Path(__file__).resolve().parents[1] / "images" / "IMG_9546.JPG"
        if not path.exists():
            self.skipTest("New-batch ruler regression fixture is not present")
        result = calibrate_image(cv2.imread(str(path), cv2.IMREAD_COLOR))
        ruler = result.ruler
        self.assertIsNotNone(ruler)
        ruler_x = sorted((ruler.endpoint_a[0], ruler.endpoint_b[0]))
        # Full-resolution row tracing rejects the plastic left edge that the
        # old detector called tick zero.  The visible metric print runs from
        # approximately x=3077 through x=5805 in the rectified fixture.
        self.assertAlmostEqual(ruler_x[0], 3076.0, delta=8.0)
        self.assertAlmostEqual(ruler_x[1], 5805.0, delta=8.0)
        self.assertAlmostEqual(ruler.tick_pitch_px, 18.0, delta=0.15)
        self.assertEqual(ruler.matched_tick_count, 151)
        self.assertEqual(len(ruler.metric_tick_segments), 151)
        self.assertEqual(len(ruler.imperial_tick_segments), 97)
        self.assertAlmostEqual(ruler.imperial_tick_pitch_px, 29.0, delta=0.15)
        self.assertAlmostEqual(ruler.metric_pixels_per_mm, 18.0, delta=0.15)
        self.assertAlmostEqual(
            ruler.imperial_pixels_per_mm,
            29.0 * 16.0 / 25.4,
            delta=0.15,
        )
        self.assertIsNotNone(ruler.scale_disagreement_percent)
        self.assertLess(ruler.scale_disagreement_percent, 3.0)
        self.assertTrue(ruler.metric_hierarchy_reliable)
        self.assertTrue(ruler.imperial_hierarchy_reliable)
        self.assertGreater(ruler.metric_hierarchy_consistency, 0.90)
        self.assertGreater(ruler.imperial_hierarchy_consistency, 0.90)
        metric_lengths = np.asarray(
            [
                np.linalg.norm(np.subtract(inner, outer))
                for outer, inner in ruler.metric_tick_segments
            ],
            dtype=np.float64,
        )
        metric_classes = np.asarray(ruler.metric_tick_classes, dtype=np.int32)
        metric_medians = np.asarray(
            [np.median(metric_lengths[metric_classes == level]) for level in range(3)]
        )
        self.assertTrue(np.all(np.diff(metric_medians) > 3.0), metric_medians)
        self.assertTrue(
            all(
                metric_classes[index] == 2
                for index in ruler.metric_unit_divider_indices
            )
        )
        self.assertGreaterEqual(
            sum(label.observed for label in ruler.metric_labels), 8
        )
        self.assertGreaterEqual(
            sum(label.observed for label in ruler.imperial_labels), 4
        )

        angle = np.deg2rad(ruler.angle_degrees)
        along = np.asarray((np.cos(angle), np.sin(angle)), dtype=np.float64)
        across = np.asarray((-np.sin(angle), np.cos(angle)), dtype=np.float64)
        outline = np.asarray(ruler.outline_points, dtype=np.float64)
        metric = np.asarray(ruler.metric_tick_segments, dtype=np.float64)
        imperial = np.asarray(ruler.imperial_tick_segments, dtype=np.float64)
        outline_u = outline @ along
        terminal_u = np.concatenate((metric[:, 0] @ along, imperial[:, 0] @ along))
        self.assertLess(float(np.min(outline_u)), float(np.min(terminal_u)))
        self.assertGreater(float(np.max(outline_u)), float(np.max(terminal_u)))
        self.assertLess(
            float(np.min(terminal_u) - np.min(outline_u)),
            ruler.imperial_tick_pitch_px * 2.0,
        )
        self.assertLess(
            float(np.max(outline_u) - np.max(terminal_u)),
            ruler.imperial_tick_pitch_px * 2.0,
        )
        outline_v = outline @ across
        top_outline_v = outline_v[:2]
        bottom_outline_v = outline_v[2:]
        self.assertLess(
            float(np.max(top_outline_v)), float(np.min(metric[:, 0] @ across))
        )
        self.assertGreater(
            float(np.min(bottom_outline_v)),
            float(np.max(imperial[:, 0] @ across)),
        )
        self.assertLess(
            float(np.min(metric[:, 0] @ across) - np.min(top_outline_v)),
            ruler.tick_pitch_px * 1.6,
        )
        self.assertLess(
            float(np.max(bottom_outline_v) - np.max(imperial[:, 0] @ across)),
            ruler.tick_pitch_px * 1.6,
        )
        self.assertLess(float(np.ptp(top_outline_v)), ruler.tick_pitch_px * 1.5)
        self.assertLess(float(np.ptp(bottom_outline_v)), ruler.tick_pitch_px * 1.5)
        # The fitted plastic perimeter is a mild quadrilateral, not a rectangle
        # reconstructed from the coarse shared ruler angle.
        self.assertGreater(float(np.ptp(bottom_outline_v)), 1.0)
        self.assertGreater(
            abs(float(outline_u[0] - outline_u[3])), 1.0
        )
        self.assertGreater(
            abs(float(outline_u[1] - outline_u[2])), 1.0
        )

    def test_reliable_metric_lattice_maps_ticks_hidden_by_local_glare(self) -> None:
        import cv2
        import numpy as np

        from seedvision.calibration import detect_ruler

        image = self._synthetic_calibration_image(rotation_degrees=1.5)
        # Erase several complete minor dashes and weaken others.  The remaining
        # periodic family is still authoritative and must map every one of the
        # 151 designed positions rather than emitting a sparse threshold mask.
        inverse = cv2.getRotationMatrix2D((600, 400), 1.5, 1.0)
        for tick in (7, 18, 29, 46, 63, 82, 101, 117, 139):
            x = 525 + tick * 4
            point = inverse @ np.asarray((x, 655, 1.0), dtype=np.float64)
            cv2.circle(
                image,
                (round(float(point[0])), round(float(point[1]))),
                5,
                (224, 224, 224),
                -1,
            )

        ruler = detect_ruler(image)

        self.assertTrue(ruler.scale_reliable)
        self.assertEqual(len(ruler.metric_tick_segments), 151)
        self.assertGreaterEqual(ruler.matched_tick_count, 135)
        self.assertAlmostEqual(ruler.tick_pitch_px, 4.0, delta=0.18)

    def test_scale_is_rejected_when_longest_ticks_are_not_major_increments(self) -> None:
        import cv2

        from seedvision.calibration import detect_ruler

        image = self._synthetic_calibration_image()
        # Shorten every designed centimetre mark so the longest surviving
        # family occurs at 5 mm. Periodic positions alone must not be allowed to
        # call this a valid 0--15 cm alignment.
        for tick in range(0, 151, 10):
            x = 525 + tick * 4
            cv2.line(image, (x, 666), (x, 696), (224, 224, 224), 3)

        ruler = detect_ruler(image)

        self.assertEqual(len(ruler.metric_tick_segments), 151)
        self.assertFalse(ruler.metric_hierarchy_reliable)
        self.assertFalse(ruler.scale_reliable)
        self.assertNotEqual(ruler.metric_unit_divider_indices[0], 0)

    def test_real_img9533_imperial_lengths_align_major_increment_hierarchy(self) -> None:
        import cv2
        import numpy as np
        from pathlib import Path

        from seedvision.calibration import calibrate_image

        path = Path(__file__).resolve().parents[1] / "images" / "IMG_9533.JPG"
        if not path.exists():
            self.skipTest("Ruler hierarchy regression fixture is not present")
        ruler = calibrate_image(
            cv2.imread(str(path), cv2.IMREAD_COLOR)
        ).ruler
        self.assertIsNotNone(ruler)
        self.assertTrue(ruler.metric_hierarchy_reliable)
        self.assertTrue(ruler.imperial_hierarchy_reliable)
        self.assertGreater(ruler.metric_hierarchy_consistency, 0.90)
        self.assertGreater(ruler.imperial_hierarchy_consistency, 0.90)
        self.assertEqual(
            ruler.imperial_unit_divider_indices, tuple(range(0, 97, 16))
        )

        expected_classes = []
        for index in range(97):
            expected_classes.append(
                4
                if index % 16 == 0
                else 3
                if index % 8 == 0
                else 2
                if index % 4 == 0
                else 1
                if index % 2 == 0
                else 0
            )
        self.assertEqual(
            ruler.imperial_tick_classes, tuple(expected_classes)
        )
        lengths = np.asarray(
            [
                np.linalg.norm(np.subtract(inner, outer))
                for outer, inner in ruler.imperial_tick_segments
            ],
            dtype=np.float64,
        )
        classes = np.asarray(ruler.imperial_tick_classes, dtype=np.int32)
        medians = np.asarray(
            [np.median(lengths[classes == level]) for level in range(5)]
        )
        self.assertTrue(np.all(np.diff(medians) > 3.0), medians)
        self.assertTrue(
            all(classes[index] == 4 for index in ruler.imperial_unit_divider_indices)
        )
        number_labels = [
            item.text for item in ruler.imperial_labels if item.kind == "number"
        ]
        self.assertEqual(number_labels[0], "6")
        self.assertEqual(number_labels[-1], "0")

    def test_swatch_grid_snaps_to_shared_printed_edges(self) -> None:
        import numpy as np

        from seedvision.calibration import detect_colour_card

        image = self._synthetic_calibration_image(
            swatch_offset=(5, 4), swatch_half_size=(34, 39)
        )
        card = detect_colour_card(image)
        for swatch in card.swatches:
            center_x = 105 + swatch.column * 100 + 5
            center_y = 100 + swatch.row * 118 + 4
            expected = np.asarray(
                (
                    (center_x - 34, center_y - 39),
                    (center_x + 34, center_y - 39),
                    (center_x + 34, center_y + 39),
                    (center_x - 34, center_y + 39),
                ),
                dtype=np.float64,
            )
            detected = np.asarray(swatch.corners, dtype=np.float64)
            self.assertLess(float(np.max(np.abs(detected - expected))), 2.5)

    def test_projective_deskew_rectifies_mild_camera_tilt(self) -> None:
        import cv2
        import numpy as np

        from seedvision.calibration import calibrate_image

        image = self._synthetic_calibration_image()
        source = np.float32(((0, 0), (1199, 0), (1199, 799), (0, 799)))
        target = np.float32(((18, 12), (1175, 34), (1190, 775), (5, 748)))
        tilted = cv2.warpPerspective(
            image,
            cv2.getPerspectiveTransform(source, target),
            (1200, 800),
            borderMode=cv2.BORDER_REPLICATE,
        )
        result = calibrate_image(tilted)
        self.assertTrue(result.perspective_corrected)
        self.assertGreater(result.perspective_strength, 0.01)
        bounds = result.transform_points(result.colour_card.bounds)
        sides = [bounds[(index + 1) % 4] - bounds[index] for index in range(4)]
        lengths = [float(np.linalg.norm(side)) for side in sides]
        self.assertAlmostEqual(lengths[0], lengths[2], delta=1.0)
        self.assertAlmostEqual(lengths[1], lengths[3], delta=1.0)
        corner_cosine = abs(float(np.dot(sides[0], sides[1]))) / (
            lengths[0] * lengths[1]
        )
        self.assertLess(corner_cosine, 0.01)

    def test_calibration_corrects_cast_deskews_and_assigns_scale(self) -> None:
        from seedvision.calibration import calibrate_image

        result = calibrate_image(
            self._synthetic_calibration_image(rotation_degrees=2.0)
        )
        self.assertIsNotNone(result.colour_card)
        self.assertIsNotNone(result.ruler)
        self.assertAlmostEqual(result.deskew_degrees, -2.0, delta=1.0)
        blue_gain, _, red_gain = result.channel_gains_bgr
        self.assertLess(blue_gain, 1.0)
        self.assertGreater(red_gain, 1.0)
        self.assertIsNotNone(result.pixels_per_mm)
        self.assertAlmostEqual(result.pixels_per_mm, 4.0, delta=1.0)
        self.assertAlmostEqual(result.ruler.angle_degrees, 0.0, delta=1.0)
        self.assertEqual(result.scale_bar_mm, 50.0)
        self.assertGreater(result.scale_confidence, 0.1)
        self.assertGreaterEqual(result.corrected_bgr.shape[0], 800)
        self.assertGreaterEqual(result.corrected_bgr.shape[1], 1200)
        self.assertEqual(result.corrected_bgr.shape[2], 3)

    def test_missing_references_produce_a_safe_uncalibrated_result(self) -> None:
        import numpy as np

        from seedvision.calibration import calibrate_image

        image = np.full((300, 400, 3), 180, dtype=np.uint8)
        result = calibrate_image(image)
        self.assertIsNone(result.colour_card)
        self.assertIsNone(result.ruler)
        self.assertIsNone(result.pixels_per_mm)
        self.assertEqual(result.deskew_degrees, 0.0)
        self.assertTrue(result.warnings)

    def test_detects_both_concentric_petri_dish_edges(self) -> None:
        import cv2
        import numpy as np

        from seedvision.calibration.geometry import (
            DishDetectionSettings,
            detect_dish,
        )

        image = np.full((800, 1200, 3), 232, dtype=np.uint8)
        center = (700, 350)
        cv2.circle(image, center, 180, (65, 65, 65), 4)
        cv2.circle(image, center, 196, (92, 92, 92), 4)
        dish = detect_dish(
            image,
            DishDetectionSettings(
                min_radius_fraction=0.18,
                max_radius_fraction=0.28,
                expected_center_x_fraction=center[0] / image.shape[1],
                expected_center_y_fraction=center[1] / image.shape[0],
                expected_radius_fraction=180 / image.shape[0],
            ),
        )

        self.assertEqual(dish.vessel_type, "petri_dish")
        self.assertTrue(dish.rim_pair_detected)
        self.assertAlmostEqual(dish.lower_edge_radius, 180, delta=5)
        self.assertAlmostEqual(dish.upper_edge_radius, 196, delta=5)
        self.assertLess(dish.inner_radius, dish.outer_radius)
        self.assertEqual(dish.radius, dish.outer_radius)

    def test_clear_rim_overrides_an_inconsistent_ruler_scale(self) -> None:
        import cv2
        import numpy as np

        from seedvision.calibration.geometry import detect_dish

        image = np.full((800, 1200, 3), 232, dtype=np.uint8)
        center = (700, 350)
        cv2.circle(image, center, 180, (60, 60, 60), 4)
        cv2.circle(image, center, 196, (88, 88, 88), 4)

        # 5.2 px/mm incorrectly predicts a 250 px outer radius for a 96 mm
        # dish. The coherent visible pair must win over that bad scale.
        dish = detect_dish(image, pixels_per_mm=5.2)

        self.assertAlmostEqual(dish.center_x, center[0], delta=8)
        self.assertAlmostEqual(dish.center_y, center[1], delta=8)
        self.assertAlmostEqual(dish.inner_radius, 180, delta=6)
        self.assertAlmostEqual(dish.outer_radius, 196, delta=6)
        self.assertTrue(dish.rim_pair_detected)

    def test_clear_new_batch_rims_are_not_forced_to_bad_ruler_scale(self) -> None:
        import cv2
        from pathlib import Path

        from seedvision.calibration.geometry import detect_dish
        from seedvision.calibration.image import calibrate_image
        from seedvision.cuda import CudaContext

        root = Path(__file__).resolve().parents[1]
        expected = {
            "IMG_9533.JPG": (3854, 1993, 810, 853),
            "IMG_9546.JPG": (3817, 1940, 792, 862),
        }
        if not all((root / "images" / name).exists() for name in expected):
            self.skipTest("New-batch Petri-rim regression fixtures are not present")
        context = CudaContext.resolve()
        for name, target in expected.items():
            image = cv2.imread(str(root / "images" / name), cv2.IMREAD_COLOR)
            calibration = calibrate_image(image, cuda_context=context)
            dish = detect_dish(
                calibration.corrected_bgr,
                cuda_context=context,
                image_tensor=calibration.gpu_corrected_bgr,
                pixels_per_mm=calibration.pixels_per_mm,
            )
            with self.subTest(name=name):
                self.assertAlmostEqual(dish.center_x, target[0], delta=18)
                self.assertAlmostEqual(dish.center_y, target[1], delta=18)
                self.assertAlmostEqual(dish.inner_radius, target[2], delta=16)
                self.assertAlmostEqual(dish.outer_radius, target[3], delta=16)
                self.assertTrue(dish.rim_pair_detected)

    def test_mild_elliptical_rim_is_retained_with_conservative_bounds(self) -> None:
        import cv2
        import numpy as np
        import torch

        from seedvision.calibration.geometry import (
            DishCircle,
            _fit_mild_dish_ellipse,
        )

        edge = np.zeros((256, 272), np.float32)
        cv2.ellipse(edge, (136, 126), (92, 84), 13, 0, 360, 1.0, 2)
        fitted = _fit_mild_dish_ellipse(
            torch.from_numpy(edge)[None, None], 136.0, 126.0, 88.0
        )
        self.assertIsNotNone(fitted)
        assert fitted is not None
        centre_x, centre_y, axis_x, axis_y, angle, confidence = fitted
        self.assertGreater(max(axis_x, axis_y) / min(axis_x, axis_y), 1.05)
        self.assertGreater(confidence, 0.5)
        dish = DishCircle(
            round(centre_x),
            round(centre_y),
            int(np.ceil(max(axis_x, axis_y))),
            confidence,
            outer_radius_x=axis_x,
            outer_radius_y=axis_y,
            ellipse_angle_degrees=angle,
        )
        left, top, right, bottom = dish.bounds
        self.assertLess(left, centre_x - min(axis_x, axis_y))
        self.assertGreater(right, centre_x + min(axis_x, axis_y))
        self.assertLessEqual(dish.outer_radius_x, dish.radius)
        self.assertLessEqual(dish.outer_radius_y, dish.radius)
        self.assertLess(top, bottom)

    def test_pilot_batch_rims_match_the_calibrated_96_mm_outer_edges(self) -> None:
        import cv2
        from pathlib import Path

        from seedvision.calibration.geometry import detect_dish
        from seedvision.calibration.image import calibrate_image
        from seedvision.cuda import CudaContext

        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "images").glob("*.JPG"))
        if not all(path.exists() for path in paths):
            self.skipTest("Petri-rim regression fixtures are not present")
        if len(paths) != 11:
            self.skipTest("The complete eleven-image pilot batch is not present")
        context = CudaContext.resolve()
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            calibration = calibrate_image(image, cuda_context=context)
            dish = detect_dish(
                calibration.corrected_bgr,
                cuda_context=context,
                image_tensor=calibration.gpu_corrected_bgr,
                pixels_per_mm=calibration.pixels_per_mm,
            )
            with self.subTest(path=path.name):
                self.assertIsNotNone(calibration.pixels_per_mm)
                inner_radius_mm = dish.inner_radius / calibration.pixels_per_mm
                outer_radius_mm = dish.outer_radius / calibration.pixels_per_mm
                self.assertGreaterEqual(inner_radius_mm, 43.5)
                self.assertLessEqual(inner_radius_mm, 47.0)
                self.assertGreaterEqual(outer_radius_mm, 47.0)
                self.assertLessEqual(outer_radius_mm, 49.2)
                self.assertTrue(dish.rim_pair_detected)


if __name__ == "__main__":
    unittest.main()
