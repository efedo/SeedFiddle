from __future__ import annotations

from tests.functional_units import functional_units

import unittest

# Swatch centres of the synthetic card drawn at (45, 35)-(465, 755), laid out
# like the Datacolor card in the image fixtures.
SWATCH_CENTRES_X = (113, 209, 303, 397)
SWATCH_CENTRES_Y = (112, 227, 341, 454, 566, 678)


@functional_units
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
        swatch_half_size: tuple[int, int] = (39, 45),
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
                center_x = SWATCH_CENTRES_X[column]
                center_y = SWATCH_CENTRES_Y[row]
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

    @staticmethod
    def _rotate_expanded(image, degrees: float, border=None):
        """Rotate about the centre onto a canvas holding the whole image."""

        import cv2

        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
        cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_width = int(height * sine + width * cosine)
        new_height = int(height * cosine + width * sine)
        matrix[0, 2] += new_width / 2 - width / 2
        matrix[1, 2] += new_height / 2 - height / 2
        if border is None:
            return cv2.warpAffine(
                image, matrix, (new_width, new_height), borderMode=cv2.BORDER_REPLICATE
            )
        return cv2.warpAffine(
            image,
            matrix,
            (new_width, new_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )

    @staticmethod
    def _camera_response(bgr):
        """The tone curve and white-balance cast applied to synthetic charts."""

        import numpy as np

        response = 255.0 * (np.asarray(bgr, np.float64) / 255.0) ** 0.8
        return np.clip(response * np.float64((1.10, 1.0, 0.88)), 0.0, 255.0)

    @classmethod
    def _synthetic_chart_scene(cls, rgb, rows: int, columns: int, degrees: float):
        """A swatch chart on a black card, through a camera response, tilted."""

        import numpy as np

        patch, gap, margin = 56, 12, 40
        height = margin * 2 + rows * patch + (rows - 1) * gap
        width = margin * 2 + columns * patch + (columns - 1) * gap
        scene = np.full((height + 300, width + 300, 3), 150, np.uint8)
        scene[150:150 + height, 150:150 + width] = 20
        for index, colour in enumerate(rgb):
            row, column = divmod(index, columns)
            y = 150 + margin + row * (patch + gap)
            x = 150 + margin + column * (patch + gap)
            scene[y:y + patch, x:x + patch] = colour[::-1]
        scene = np.uint8(np.round(cls._camera_response(scene)))
        return cls._rotate_expanded(scene, degrees, border=(150, 150, 150))

    @staticmethod
    def _synthetic_ruler_scene(
        degrees: float,
        pitch: float,
        *,
        millimetres: int = 150,
        margin: float = 16.0,
        centre: tuple[float, float] = (500.0, 450.0),
        five_mm_marks: bool = True,
    ):
        """A ruler at any angle with dashes at a sub-pixel pitch."""

        import cv2
        import numpy as np

        image = np.full((900, 1000, 3), 236, np.uint8)
        ruler_width = 120.0
        length = millimetres * pitch + 2 * margin
        radians = np.deg2rad(degrees)
        along = np.float64((np.cos(radians), np.sin(radians)))
        across = np.float64((-np.sin(radians), np.cos(radians)))

        def point(u: float, v: float) -> tuple[int, int]:
            xy = centre + (u - length / 2) * along + (v - ruler_width / 2) * across
            # Sixteenths of a pixel (shift=4) place every dash at its exact position.
            return tuple(int(round(value * 16)) for value in xy)

        body = np.int32(
            [point(0, 0), point(length, 0), point(length, ruler_width), point(0, ruler_width)]
        )
        cv2.fillConvexPoly(image, body, (222, 222, 222), cv2.LINE_AA, 4)
        cv2.polylines(image, [body], True, (55, 55, 55), 2, cv2.LINE_AA, 4)
        for tick in range(millimetres + 1):
            u = margin + tick * pitch
            tick_length = (
                46 if tick % 10 == 0 else 30 if tick % 5 == 0 and five_mm_marks else 18
            )
            cv2.line(
                image, point(u, 9), point(u, 9 + tick_length), (25, 25, 25), 1, cv2.LINE_AA, 4
            )
        return image

    def check_test_colorchecker_is_identified_and_numbered_at_any_angle(self) -> None:
        import numpy as np

        from seedvision.calibration import detect_colour_card
        from seedvision.calibration.colour_chart import COLORCHECKER_CLASSIC_24

        # A tilt, the same tilt upside down, and a portrait-turned chart.
        for degrees, rotation, angle in ((37.0, 0, -37.0), (217.0, 180, -37.0), (-61.0, 90, -29.0)):
            with self.subTest(degrees=degrees):
                card = detect_colour_card(
                    self._synthetic_chart_scene(COLORCHECKER_CLASSIC_24.rgb, 4, 6, degrees)
                )
                self.assertEqual(card.chart, COLORCHECKER_CLASSIC_24.name)
                self.assertEqual(card.rotation_degrees, rotation)
                self.assertEqual(card.patch_count, 24)
                self.assertLess(abs(card.angle_degrees - angle), 0.5)
                # X-Rite's numbers follow the chart, not the image: 19-24 are
                # the greys from white to black, whichever way the card lies.
                numbers = sorted(swatch.number for swatch in card.swatches)
                self.assertEqual(numbers, list(range(1, 25)))
                for swatch in card.swatches:
                    expected = self._camera_response(
                        COLORCHECKER_CLASSIC_24.rgb[swatch.number - 1][::-1]
                    )
                    self.assertLess(float(np.max(np.abs(swatch.mean_bgr - expected))), 4.0)

    def check_test_unrecognised_grid_is_numbered_in_reading_order(self) -> None:
        import numpy as np

        from seedvision.calibration import detect_colour_card

        generator = np.random.default_rng(7)
        colours = [tuple(int(value) for value in generator.integers(40, 230, 3)) for _ in range(20)]
        card = detect_colour_card(self._synthetic_chart_scene(colours, 4, 5, 15.0))
        self.assertIsNone(card.chart)
        self.assertEqual((card.rows, card.columns, card.patch_count), (4, 5, 20))
        for swatch in card.swatches:
            self.assertEqual(swatch.number, swatch.row * 5 + swatch.column + 1)
            expected = self._camera_response(colours[swatch.number - 1][::-1])
            self.assertLess(float(np.max(np.abs(swatch.mean_bgr - expected))), 4.0)

    def check_test_ruler_is_measured_at_any_angle_to_a_fraction_of_a_pixel(self) -> None:
        import numpy as np

        from seedvision.calibration import detect_ruler

        for degrees, pitch in ((0.0, 4.3), (90.0, 4.3), (30.0, 5.15)):
            with self.subTest(degrees=degrees, pitch=pitch):
                ruler = detect_ruler(self._synthetic_ruler_scene(degrees, pitch))
                self.assertTrue(ruler.scale_reliable)
                self.assertEqual(len(ruler.metric_tick_segments), 151)
                self.assertEqual(ruler.metric_unit_divider_indices[0], 0)
                # A whole-pixel pitch would be 7% out at 4.3 px per millimetre.
                self.assertAlmostEqual(ruler.metric_pixels_per_mm / pitch, 1.0, delta=0.001)
                direction = np.subtract(ruler.endpoint_b, ruler.endpoint_a)
                direction /= np.linalg.norm(direction)
                radians = np.deg2rad(degrees)
                cross = direction[0] * np.sin(radians) - direction[1] * np.cos(radians)
                self.assertLess(abs(cross), np.sin(np.deg2rad(0.5)))

    def check_test_ruler_without_its_configured_span_is_measured_over_its_visible_run(
        self,
    ) -> None:
        """A 15 cm ruler cut by the frame, and a 6 cm mm-and-cm scale."""

        from seedvision.calibration import detect_ruler

        cases = (
            # The frame cuts the ruler near 11.7 cm; blank plastic precedes 0.
            ("cut by the frame", 0.0, 6.0, 150, (750.0, 450.0), True),
            ("6 cm without 5 mm marks", 12.0, 7.3, 60, (500.0, 450.0), False),
        )
        for label, degrees, pitch, millimetres, centre, five_mm in cases:
            with self.subTest(case=label):
                ruler = detect_ruler(
                    self._synthetic_ruler_scene(
                        degrees,
                        pitch,
                        millimetres=millimetres,
                        margin=30.0,
                        centre=centre,
                        five_mm_marks=five_mm,
                    )
                )
                self.assertTrue(ruler.scale_reliable)
                self.assertAlmostEqual(ruler.metric_pixels_per_mm / pitch, 1.0, delta=0.001)
                self.assertEqual(ruler.metric_unit_divider_indices[0], 0)
                # Blank plastic before the first dash shows that it is zero.
                self.assertEqual(ruler.metric_labels[0].text, "0")
                if millimetres == 60:
                    self.assertEqual(len(ruler.metric_tick_segments), 61)
                    self.assertEqual(ruler.metric_unit_divider_indices, tuple(range(0, 61, 10)))
                else:
                    self.assertGreaterEqual(len(ruler.metric_tick_segments), 110)

    def check_test_low_quality_photographs_take_scale_from_partial_rulers(self) -> None:
        """GRIN seed photographs: rulers cut by the frame, the chart or a seed.

        Reviewed scales were read from printed centimetre marks in zoomed crops
        of the corrected photographs, about 0.1% accurate: 0 to 13 cm on
        pi_504415_s, 1 to 6 cm on the mm-and-cm scale of 19U_7126. The last
        three photographs have no ruler; one has barcodes, which are periodic,
        and one a printed sample label beneath the seeds.
        """

        import cv2
        from pathlib import Path

        from seedvision.calibration.image import calibrate_image
        from seedvision.cuda import CudaContext

        expected = {
            "pi_504415_s.jpg": 5.314,
            "pi_457995_s.jpg": None,
            "pi_478523_seed.jpg": None,
            "pi_478525_s.jpg": None,
            "pi_510573_s.jpg": None,
            "pi_615395_s.jpg": None,
            "19U_7126__SD_PI_546487__03-05-2023_01_Seed_SD.JPG": 7.368,
            "19U_7255__SD_PI_538398__03-04-2023_01_Seed_SD.JPG": None,
            "19U_7513__SD_PI_539867__03-06-2023_01_Seed_SD.JPG": None,
        }
        without_ruler = (
            "PI054859_03U-1711_SD_3-21-2013_01.JPG",
            "18S_5012__SD_PI_553049__09-15-2020_01_Seed_SD.jpg",
            "PI542972_11S-2361_SD_6-2-2014_1.JPG",
        )
        root = Path(__file__).resolve().parents[1] / "images"
        if not all((root / name).exists() for name in (*expected, *without_ruler)):
            self.skipTest("GRIN photographs are not present")
        context = CudaContext.resolve()
        for name in (*expected, *without_ruler):
            with self.subTest(path=name):
                result = calibrate_image(
                    cv2.imread(str(root / name), cv2.IMREAD_COLOR), cuda_context=context
                )
                if name in without_ruler:
                    self.assertIsNone(result.ruler)
                    self.assertIsNone(result.pixels_per_mm)
                    continue
                self.assertIsNotNone(result.colour_card)
                self.assertTrue(result.ruler.scale_reliable)
                if expected[name] is not None:
                    self.assertAlmostEqual(
                        result.pixels_per_mm / expected[name], 1.0, delta=0.005
                    )
                if name.startswith("pi_"):
                    # Blank plastic precedes these rulers' zero.
                    self.assertEqual(result.ruler.metric_labels[0].text, "0")

    def check_test_rotated_and_tilted_fixture_keeps_chart_numbers_and_scale(self) -> None:
        """A quarter turn and a tilt beyond deskew of a committed photograph."""

        import cv2
        import numpy as np
        from pathlib import Path

        from seedvision.calibration import detect_colour_card, detect_ruler
        from seedvision.calibration.image import _chart_exclusion_region
        from seedvision.cuda import CudaContext

        path = Path(__file__).resolve().parents[1] / "images" / "IMG_9632c.JPG"
        if not path.exists():
            self.skipTest("IMG_9632c.JPG fixture is not present")
        original = cv2.imread(str(path), cv2.IMREAD_COLOR)
        context = CudaContext.resolve()
        variants = (
            ("upright", original, 0, 0.0),
            ("quarter turn", cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE), 90, 90.0),
            ("25 degree tilt", self._rotate_expanded(original, 25.0), 0, -25.0),
        )
        reference = None
        for label, image, rotation, turn in variants:
            with self.subTest(variant=label):
                card = detect_colour_card(np.ascontiguousarray(image), cuda_context=context)
                ruler = detect_ruler(
                    np.ascontiguousarray(image),
                    cuda_context=context,
                    excluded_regions=(_chart_exclusion_region(card, np.eye(3)),),
                )
                self.assertEqual(card.chart, "Datacolor SpyderCHECKR 24")
                self.assertEqual(card.rotation_degrees, rotation)
                self.assertTrue(ruler.scale_reliable)
                colours = {swatch.number: np.asarray(swatch.mean_bgr) for swatch in card.swatches}
                self.assertEqual(sorted(colours), list(range(1, 25)))
                if reference is None:
                    reference = (colours, ruler.metric_pixels_per_mm, ruler.angle_degrees)
                    continue
                for number, colour in colours.items():
                    self.assertLess(float(np.max(np.abs(colour - reference[0][number]))), 6.0)
                # Scale in the source frame is unchanged by rotation.
                self.assertAlmostEqual(ruler.metric_pixels_per_mm / reference[1], 1.0, delta=0.002)
                difference = (ruler.angle_degrees - reference[2] - turn + 90.0) % 180.0 - 90.0
                self.assertLess(abs(difference), 1.0)

    def check_test_scale_is_rejected_when_longest_ticks_are_not_major_increments(self) -> None:
        import cv2
        import numpy as np

        from seedvision.calibration import detect_colour_card, detect_ruler
        from seedvision.calibration.image import _chart_exclusion_region

        image = self._synthetic_calibration_image()
        # Shorten every designed centimetre mark so the longest surviving
        # family occurs at 5 mm. Periodic positions alone must not be allowed to
        # call this a valid 0--15 cm alignment.
        for tick in range(0, 151, 10):
            x = 525 + tick * 4
            cv2.line(image, (x, 666), (x, 696), (224, 224, 224), 3)

        # The ruler's long edges continue the chart's last swatch row and
        # card edge, so, as in calibrate_image, the chart is excluded.
        card = detect_colour_card(image)
        ruler = detect_ruler(
            image, excluded_regions=(_chart_exclusion_region(card, np.eye(3)),)
        )

        self.assertEqual(len(ruler.metric_tick_segments), 151)
        self.assertFalse(ruler.metric_hierarchy_reliable)
        self.assertFalse(ruler.scale_reliable)
        self.assertNotEqual(ruler.metric_unit_divider_indices[0], 0)

    def check_test_printed_label_is_not_a_ruler(self) -> None:
        """A sample label's two long edges and letter strokes form no tick train."""

        import cv2
        import numpy as np

        from seedvision.calibration import detect_ruler
        from seedvision.calibration.image import CalibrationDetectionError

        image = np.full((900, 1400, 3), (60, 110, 150), np.uint8)
        cv2.rectangle(image, (150, 560), (1250, 700), (235, 235, 235), -1)
        for text, x in (("PI 542972", 190), ("11S-2361", 720)):
            cv2.putText(
                image, text, (x, 665), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (25, 25, 25), 9, cv2.LINE_AA
            )
        # The strokes count as some forty loose ticks along the band.
        with self.assertRaisesRegex(CalibrationDetectionError, "Ruler was not detected"):
            detect_ruler(image)


    def check_test_swatch_grid_snaps_to_shared_printed_edges(self) -> None:
        import numpy as np

        from seedvision.calibration import detect_colour_card

        image = self._synthetic_calibration_image(
            swatch_offset=(5, 4), swatch_half_size=(36, 42)
        )
        card = detect_colour_card(image)
        for swatch in card.swatches:
            center_x = SWATCH_CENTRES_X[swatch.column] + 5
            center_y = SWATCH_CENTRES_Y[swatch.row] + 4
            expected = np.asarray(
                (
                    (center_x - 36, center_y - 42),
                    (center_x + 36, center_y - 42),
                    (center_x + 36, center_y + 42),
                    (center_x - 36, center_y + 42),
                ),
                dtype=np.float64,
            )
            detected = np.asarray(swatch.corners, dtype=np.float64)
            self.assertLess(float(np.max(np.abs(detected - expected))), 2.5)



    def check_test_missing_references_produce_a_safe_uncalibrated_result(self) -> None:
        import numpy as np

        from seedvision.calibration import calibrate_image

        image = np.full((300, 400, 3), 180, dtype=np.uint8)
        result = calibrate_image(image)
        self.assertIsNone(result.colour_card)
        self.assertIsNone(result.ruler)
        self.assertIsNone(result.pixels_per_mm)
        self.assertEqual(result.deskew_degrees, 0.0)
        self.assertTrue(result.warnings)




    def check_test_mild_elliptical_rim_is_retained_with_conservative_bounds(self) -> None:
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

    @staticmethod
    def _faint_rim_scene(*, seed_mass: bool = True):
        """A thin, low-contrast glass rim (r 200) around a dark seed mass (r 150)."""

        import cv2
        import numpy as np

        image = np.full((800, 1200, 3), 235, dtype=np.uint8)
        cv2.circle(image, (700, 330), 200, (165, 165, 165), 2, cv2.LINE_AA)
        cv2.circle(image, (700, 330), 192, (205, 205, 205), 2, cv2.LINE_AA)
        if seed_mass:
            cv2.circle(image, (712, 322), 150, (60, 60, 60), -1, cv2.LINE_AA)
        return image

    def check_test_faint_rim_replaces_an_enclosed_seed_mass_with_a_ruler_scale(self) -> None:
        from unittest import mock

        from seedvision.calibration import geometry

        with mock.patch.object(
            geometry, "_search_calibrated_rim", wraps=geometry._search_calibrated_rim
        ) as search:
            dish = geometry.detect_dish(self._faint_rim_scene(), pixels_per_mm=200 / 48.0)
        # The generic bank prefers the much stronger seed-mass edge; only the
        # calibrated search recovers the glass.
        search.assert_called_once()
        self.assertLessEqual(abs(dish.center_x - 700), 2)
        self.assertLessEqual(abs(dish.center_y - 330), 2)
        self.assertGreaterEqual(dish.outer_radius, 198)
        self.assertLessEqual(dish.outer_radius, 204)
        self.assertLess(dish.inner_radius, dish.outer_radius)
        self.assertTrue(dish.rim_pair_detected)

    def check_test_rim_agreeing_with_the_ruler_or_contradicting_it_clearly_is_kept(self) -> None:
        from unittest import mock

        from seedvision.calibration import geometry

        image = self._faint_rim_scene(seed_mass=False)
        uncalibrated = geometry.detect_dish(image)
        for pixels_per_mm, searches in ((200 / 48.0, 0), (230 / 48.0, 1)):
            with self.subTest(pixels_per_mm=pixels_per_mm), mock.patch.object(
                geometry, "_search_calibrated_rim", wraps=geometry._search_calibrated_rim
            ) as search:
                dish = geometry.detect_dish(image, pixels_per_mm=pixels_per_mm)
                # A 230 px ruler radius disagrees with the clear 200 px glass;
                # circles of that size only graze it, so the glass is kept.
                self.assertEqual(search.call_count, searches)
                self.assertEqual(
                    (dish.center_x, dish.center_y, dish.outer_radius),
                    (uncalibrated.center_x, uncalibrated.center_y, uncalibrated.outer_radius),
                )

    def check_test_dish_size_that_cannot_fit_the_image_is_searched_without_it(self) -> None:
        """A ruler scale that puts a 96 mm dish outside the radius range is ignored.

        At 400/48 px/mm the dish would span the whole 800 px image height; at
        100/48 it would be smaller than the smallest radius searched. Neither
        can be in view, so the search runs exactly as without a ruler rather
        than failing, and the rim-support check decides whether a vessel is
        present.
        """

        from unittest import mock

        from seedvision.calibration import geometry

        image = self._faint_rim_scene(seed_mass=False)
        uncalibrated = geometry.detect_dish(image)
        for pixels_per_mm in (400 / 48.0, 100 / 48.0):
            with self.subTest(pixels_per_mm=pixels_per_mm), mock.patch.object(
                geometry, "_search_calibrated_rim", wraps=geometry._search_calibrated_rim
            ) as search:
                dish = geometry.detect_dish(image, pixels_per_mm=pixels_per_mm)
                search.assert_not_called()
                self.assertEqual(dish, uncalibrated)

    def check_test_fixture_rims_match_the_calibrated_96_mm_dish(self) -> None:
        """Fixtures where a seed mass or colour-card arcs outscored faint glass.

        Centres were read on a millimetre grid (about 1 mm accuracy) and are
        kept in source-image pixels, so they follow whatever correction the
        calibration applies. IMG_9974c is a control whose rim the generic
        search already resolves.
        """

        import cv2
        import numpy as np
        from pathlib import Path
        from unittest import mock

        from seedvision.calibration import geometry
        from seedvision.calibration.image import calibrate_image
        from seedvision.cuda import CudaContext

        reviewed_centres = {
            "IMG_9403.JPG": (3899.4, 1998.8),
            "IMG_9636c.JPG": (1841.1, 745.7),
            "IMG_9641c.JPG": (1859.1, 773.6),
            "IMG_9668c.JPG": (1829.5, 784.3),
            "IMG_9719.JPG": (3604.2, 1483.6),
            "IMG_9974c.JPG": None,
        }
        root = Path(__file__).resolve().parents[1] / "images"
        if not all((root / name).exists() for name in reviewed_centres):
            self.skipTest("Petri-rim regression fixtures are not present")
        context = CudaContext.resolve()
        for name, centre in reviewed_centres.items():
            calibration = calibrate_image(
                cv2.imread(str(root / name), cv2.IMREAD_COLOR), cuda_context=context
            )
            with self.subTest(path=name), mock.patch.object(
                geometry, "_search_calibrated_rim", wraps=geometry._search_calibrated_rim
            ) as search:
                dish = geometry.detect_dish(
                    calibration.corrected_bgr,
                    cuda_context=context,
                    image_tensor=calibration.gpu_corrected_bgr,
                    pixels_per_mm=calibration.pixels_per_mm,
                )
                pixels_per_mm = calibration.pixels_per_mm
                self.assertIsNotNone(pixels_per_mm)
                self.assertEqual(search.call_count, 0 if centre is None else 1)
                exterior_diameter_mm = sum(dish.outer_axes) / pixels_per_mm
                self.assertGreaterEqual(exterior_diameter_mm, 92.5)
                self.assertLessEqual(exterior_diameter_mm, 99.5)
                self.assertLess(dish.inner_radius, dish.outer_radius)
                self.assertTrue(dish.rim_pair_detected)
                if centre is not None:
                    reviewed_x, reviewed_y = calibration.transform_points((centre,))[0]
                    offset = np.hypot(dish.center_x - reviewed_x, dish.center_y - reviewed_y)
                    self.assertLessEqual(offset / pixels_per_mm, 2.5)

    def test_fixture_rims_match_the_calibrated_96_mm_dish(self):
        self.scenario(self.check_test_fixture_rims_match_the_calibrated_96_mm_dish, labelled=False)

    def test_low_quality_reference_photographs(self):
        self.scenario(
            self.check_test_low_quality_photographs_take_scale_from_partial_rulers,
            labelled=False,
        )

    def test_validation_and_recovery(self):
        """3 related ImageCalibrationTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_scale_is_rejected_when_longest_ticks_are_not_major_increments)
        self.scenario(self.check_test_printed_label_is_not_a_ruler)
        self.scenario(self.check_test_missing_references_produce_a_safe_uncalibrated_result)

    def test_reference_orientation(self):
        """5 ImageCalibrationTests scenarios: charts and rulers at any angle or in part."""
        self.scenario(self.check_test_colorchecker_is_identified_and_numbered_at_any_angle)
        self.scenario(self.check_test_unrecognised_grid_is_numbered_in_reading_order)
        self.scenario(self.check_test_ruler_is_measured_at_any_angle_to_a_fraction_of_a_pixel)
        self.scenario(
            self.check_test_ruler_without_its_configured_span_is_measured_over_its_visible_run
        )
        self.scenario(self.check_test_rotated_and_tilted_fixture_keeps_chart_numbers_and_scale)

    def test_workflow_lifecycle(self):
        """5 related ImageCalibrationTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_swatch_grid_snaps_to_shared_printed_edges)
        self.scenario(self.check_test_mild_elliptical_rim_is_retained_with_conservative_bounds)
        self.scenario(self.check_test_faint_rim_replaces_an_enclosed_seed_mass_with_a_ruler_scale)
        self.scenario(self.check_test_rim_agreeing_with_the_ruler_or_contradicting_it_clearly_is_kept)
        self.scenario(self.check_test_dish_size_that_cannot_fit_the_image_is_searched_without_it)


if __name__ == "__main__":
    unittest.main()
