"""Colour correction against a chart's published reference colours."""

from tests.functional_units import functional_units
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.calibration import chart_colour as cc
from seedvision.calibration import image as c
from seedvision.cuda import CudaContext, image_to_tensor

# A plausible camera: channel crosstalk, unequal channel sensitivity and flare.
CAMERA = np.array(((0.78, 0.16, 0.03), (0.06, 0.86, 0.06), (0.02, 0.12, 0.66)))
FLARE = np.array((0.012, 0.010, 0.018))


def encode(linear):
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055) * 255.0


def photographed(chart, camera=CAMERA, flare=FLARE):
    """8-bit sRGB patch colours as a camera would record the chart."""

    linear = np.clip(cc.lab_d50_to_linear_rgb(np.asarray(chart.lab_d50)), 0.0, None)
    return encode(linear @ camera.T + flare)


def reference_srgb(chart):
    return encode(np.clip(cc.lab_d50_to_linear_rgb(np.asarray(chart.lab_d50)), 0.0, None))


def card_image(chart=cc.SPYDERCHECKR_24, *, name=None):
    """A grey image holding the photographed chart, and its detection."""

    colours = photographed(chart)
    pitch_x, pitch_y, size = 34, 30, 22
    image = np.full((30 + chart.rows * pitch_y, 30 + chart.columns * pitch_x, 3), 128, np.uint8)
    swatches = []
    for index, rgb in enumerate(colours):
        row, column = divmod(index, chart.columns)
        x0, y0 = 15 + column * pitch_x, 15 + row * pitch_y
        bgr = np.rint(rgb[::-1]).astype(np.uint8)
        image[y0:y0 + size, x0:x0 + size] = bgr
        corners = ((x0 + 3.0, y0 + 3.0), (x0 + size - 4.0, y0 + 3.0),
                   (x0 + size - 4.0, y0 + size - 4.0), (x0 + 3.0, y0 + size - 4.0))
        swatches.append(c.ColourSwatch(index + 1, row, column, x0 + size / 2, y0 + size / 2,
                                       corners, tuple(float(v) for v in bgr)))
    height, width = image.shape[:2]
    card = c.ColourCardDetection(
        tuple(swatches), ((10.0, 10.0), (width - 10.0, 10.0), (width - 10.0, height - 10.0), (10.0, height - 10.0)),
        0.0, 0.9, len(swatches), chart.name if name is None else name, 0, chart.rows, chart.columns,
        chart.rows * chart.columns)
    return image, card


def ruler(*args, **kwargs):
    return c.RulerDetection((45., 50.), (90., 50.), 67.5, 50., 45., 6., 0., .8,
        tick_pitch_px=4., metric_pixels_per_mm=4.)


@functional_units
class ChartColourCorrectionTests(unittest.TestCase):
    def calibrate(self, image, card, settings=None):
        settings = settings or c.CalibrationSettings(apply_perspective_correction=False)
        with patch.object(c, 'detect_colour_card', return_value=card), \
                patch.object(c, 'detect_ruler', side_effect=ruler):
            return c.calibrate_image(image, settings, cuda_context=CudaContext.resolve())

    def check_test_published_lab_values_reproduce_the_published_srgb_table(self):
        # sRGB column of the same published table, except G2 "Blueprint", whose
        # printed 99/86/96 is a typo for the 58/88/159 its Lab values give.
        published = {
            "Card White": (249, 242, 238), "Primary Cyan": (0, 127, 159),
            "Primary Orange": (222, 118, 32), "Aqua": (98, 187, 166),
            "Primary Yellow": (245, 205, 0), "Blueprint": (58, 88, 159),
            "Classic Light Skin": (197, 145, 125), "Card Black": (43, 41, 43),
            "Classic Dark Skin": (112, 76, 60), "Sunflower": (236, 158, 25),
        }
        chart = cc.SPYDERCHECKR_24
        srgb = reference_srgb(chart)
        for name, expected in published.items():
            with self.subTest(name=name):
                index = chart.patch_names.index(name)
                np.testing.assert_allclose(srgb[index], expected, atol=3.0)
        # Upright layout: white top left, greys down the left column, the two
        # skin tones at the foot of the right column.
        self.assertEqual([chart.patch_names[i] for i in (0, 20, 19, 23)],
                         ["Card White", "Card Black", "Classic Light Skin", "Classic Dark Skin"])

    def check_test_ciede2000_matches_published_test_pairs(self):
        # Sharma, Wu and Dalal (2005) test data.
        pairs = (((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
                 ((50, 2.49, -0.001), (50, -2.49, 0.0009), 7.1792),
                 ((2.0776, 0.0795, -1.135), (0.9033, -0.0636, -0.5514), 0.9082),
                 ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630))
        for first, second, expected in pairs:
            self.assertAlmostEqual(float(cc.delta_e_2000(np.array(first), np.array(second))), expected, places=4)

    def check_test_identified_chart_fit_recovers_the_camera(self):
        chart = cc.SPYDERCHECKR_24
        colours = photographed(chart)
        patches = [(i + 1, i // 4, i % 4, tuple(colours[i, ::-1])) for i in range(24)]
        facts, matrix, offset = cc.fit_chart_correction(patches, chart_name=chart.name)
        self.assertEqual(facts.model, "chart_reference_fit")
        self.assertLess(facts.mean_delta_e, 0.6)
        self.assertLess(facts.held_out_mean_delta_e, 1.0)
        # Automatic anchors are the chart's own greys.
        self.assertEqual(facts.anchor_patch_ids, (1, 5, 9, 13, 17, 21))
        np.testing.assert_allclose(matrix @ CAMERA, np.eye(3), atol=0.03)

    def check_test_unidentified_grids_find_the_chart_and_its_rotation(self):
        spyder = cc.SPYDERCHECKR_24
        colours = photographed(spyder)
        upside_down = np.rot90(np.arange(24).reshape(6, 4), 2)
        patches = [(0, r, k, tuple(colours[upside_down[r, k], ::-1])) for r in range(6) for k in range(4)]
        facts, *_ = cc.fit_chart_correction(patches)
        self.assertEqual((facts.chart, facts.rotation_degrees), (spyder.name, 180))
        checker = cc.COLORCHECKER_CLASSIC_24
        colours = photographed(checker)
        turned = cc._reference_grid(checker, 1)
        patches = [(0, r, k, tuple(colours[turned[r, k], ::-1]))
                   for r in range(turned.shape[0]) for k in range(turned.shape[1])]
        facts, *_ = cc.fit_chart_correction(patches)
        self.assertEqual((facts.chart, facts.rotation_degrees), (checker.name, 90))

    def check_test_unknown_or_clipped_charts_give_a_reason(self):
        rng = np.random.default_rng(5)
        patches = [(0, i // 4, i % 4, tuple(rng.uniform(20, 230, 3))) for i in range(24)]
        self.assertIn("closely enough", cc.fit_chart_correction(patches))
        colours = photographed(cc.SPYDERCHECKR_24)
        clipped = [(i + 1, i // 4, i % 4, (255.0, 255.0, 255.0) if i >= 10 else tuple(colours[i, ::-1]))
                   for i in range(24)]
        reason = cc.fit_chart_correction(clipped, chart_name=cc.SPYDERCHECKR_24.name)
        self.assertIn("needs 12", reason)
        self.assertIn("No reference colours", cc.fit_chart_correction(clipped, chart_name="Mystery card"))

    def check_test_neutral_patch_selection_changes_the_fit(self):
        chart = cc.SPYDERCHECKR_24
        rng = np.random.default_rng(11)
        colours = np.clip(photographed(chart) + rng.normal(0, 4, (24, 3)), 0, 249)
        patches = [(i + 1, i // 4, i % 4, tuple(colours[i, ::-1])) for i in range(24)]
        automatic = cc.fit_chart_correction(patches, chart_name=chart.name)
        manual = cc.fit_chart_correction(patches, chart_name=chart.name, anchor_patch_numbers=(1, 2))
        self.assertEqual(manual[0].anchor_patch_ids, (1, 2))
        self.assertGreater(np.abs(automatic[1] - manual[1]).max(), 1e-4)

    def check_test_gpu_application_matches_the_numpy_transform(self):
        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, (9, 13, 3), dtype=np.uint8)
        matrix = np.array(((1.1, -0.05, 0.0), (0.02, 0.95, 0.03), (-0.01, 0.04, 1.2)))
        offset = np.array((0.004, -0.002, 0.01))
        tensor = image_to_tensor(image, CudaContext.resolve()).float()
        result = cc.apply_linear_rgb_correction(tensor, matrix, offset)
        self.assertEqual(result.device.type, tensor.device.type)
        linear = cc.srgb_to_linear(image[..., ::-1] / 255.0) @ matrix.T + offset
        peak = np.maximum(linear.max(axis=-1, keepdims=True), 1.0)
        expected = encode(np.clip(linear / peak, 0.0, 1.0))[..., ::-1]
        np.testing.assert_allclose(result[0].permute(1, 2, 0).cpu().numpy(), expected, atol=0.05)

    def check_test_calibration_corrects_patches_to_their_reference_colours(self):
        chart = cc.SPYDERCHECKR_24
        image, card = card_image(chart)
        result = self.calibrate(image, card)
        facts = result.colour_correction
        self.assertEqual((facts.model, facts.chart), ("chart_reference_fit", chart.name))
        self.assertEqual(result.channel_gains_bgr, (1.0, 1.0, 1.0))
        self.assertEqual(result.gpu_corrected_bgr.device.type, 'cuda')
        expected = reference_srgb(chart)
        for swatch in card.swatches:
            x, y = result.transform_points(((swatch.center_x, swatch.center_y),))[0]
            corrected = result.corrected_bgr[int(round(y)), int(round(x))][::-1].astype(float)
            with self.subTest(patch=swatch.number):
                np.testing.assert_allclose(corrected, expected[swatch.number - 1], atol=4.0)
        self.assertFalse(any("grey balance only" in warning for warning in result.warnings))
        from seedvision.export.calibration_facts import calibration_facts
        exported = calibration_facts(result)
        self.assertEqual(exported['colour_correction']['chart'], chart.name)

    def check_test_grey_balance_and_unknown_charts_use_channel_gains(self):
        image, card = card_image()
        grey = self.calibrate(image, card, c.CalibrationSettings(
            apply_perspective_correction=False, colour_correction_model='Grey balance only'))
        self.assertEqual(grey.colour_correction.model, 'neutral_balance')
        self.assertNotEqual(grey.channel_gains_bgr, (1.0, 1.0, 1.0))
        self.assertFalse(any("grey balance only" in warning for warning in grey.warnings))
        unknown = self.calibrate(image, replace(card, chart="Mystery card"))
        self.assertEqual(unknown.colour_correction.model, 'neutral_balance')
        self.assertTrue(any("grey balance only" in warning for warning in unknown.warnings))
        off = self.calibrate(image, card, c.CalibrationSettings(
            apply_perspective_correction=False, apply_colour_balance=False))
        self.assertEqual(off.colour_correction.model, 'none')
        np.testing.assert_array_equal(off.corrected_bgr[15:37, 15:37], image[15:37, 15:37])

    def check_test_model_control_reaches_the_calculation(self):
        from seedvision.pipeline import build_default_pipeline
        from seedvision.pipeline.settings import production_settings
        graph = build_default_pipeline()
        self.assertEqual(production_settings(graph)['calibration_settings'].colour_correction_model,
                         'Chart reference fit')
        graph.set_parameter('deskew_colour', 'colour_correction_model', 'Grey balance only')
        self.assertEqual(production_settings(graph)['calibration_settings'].colour_correction_model,
                         'Grey balance only')
        with self.assertRaises(ValueError):
            c.CalibrationSettings(colour_correction_model='Curves')

    def test_reference_colours(self):
        """3 related ChartColourCorrectionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_published_lab_values_reproduce_the_published_srgb_table)
        self.scenario(self.check_test_ciede2000_matches_published_test_pairs)
        self.scenario(self.check_test_model_control_reaches_the_calculation)

    def test_fitting(self):
        """4 related ChartColourCorrectionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_identified_chart_fit_recovers_the_camera)
        self.scenario(self.check_test_unidentified_grids_find_the_chart_and_its_rotation)
        self.scenario(self.check_test_unknown_or_clipped_charts_give_a_reason)
        self.scenario(self.check_test_neutral_patch_selection_changes_the_fit)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for calibration')
    def test_calibration(self):
        """3 related ChartColourCorrectionTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_gpu_application_matches_the_numpy_transform)
        self.scenario(self.check_test_calibration_corrects_patches_to_their_reference_colours)
        self.scenario(self.check_test_grey_balance_and_unknown_charts_use_channel_gains)


if __name__ == '__main__':
    unittest.main()
