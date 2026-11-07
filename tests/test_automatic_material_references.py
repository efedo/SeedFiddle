"""Initial Foreground, Background and Other references without painting."""

from tests.functional_units import functional_units
from dataclasses import replace
import unittest

import cv2
import numpy as np
import torch

from seedvision.calibration.geometry import DishCircle
from seedvision.cuda import CudaContext, bgr_to_lab, image_to_tensor
from seedvision.segmentation import automatic_references as ar
from seedvision.segmentation.baseline import (
    BaselineSettings,
    _confident_vessel_seeds,
    _foreground_feature,
)

FLOOR = (205, 210, 214)
SEED = (60, 92, 128)


def dish_image(*, rim=True, size=(900, 900), centre=(450, 450), radius=380):
    """A pale vessel floor, optionally with a visible dark glass rim."""

    image = np.full((*size, 3), FLOOR, np.uint8)
    if rim:
        cv2.circle(image, centre, radius, (120, 124, 128), 14)
    dish = DishCircle(
        centre[0], centre[1], radius + 7, 1.0,
        lower_edge_radius=radius - 7, upper_edge_radius=radius + 7,
        rim_pair_detected=True,
    )
    return image, dish


def place_seed(image, centre, axes=(30, 20), angle=20.0, colour=SEED):
    cv2.ellipse(image, centre, axes, angle, 0.0, 360.0, colour, -1)


def lab_of(image):
    return bgr_to_lab(image_to_tensor(image, CudaContext.resolve()))


def settings(**changes):
    return BaselineSettings(**changes)


@functional_units
class AutomaticMaterialReferenceTests(unittest.TestCase):
    def check_test_rim_support_separates_a_visible_vessel_from_a_fitted_circle(self):
        image, dish = dish_image(rim=True)
        self.assertGreater(ar.vessel_rim_support(lab_of(image), dish), 0.95)
        plain, _ = dish_image(rim=False)
        self.assertLess(ar.vessel_rim_support(lab_of(plain), dish), 0.05)
        half = plain.copy()
        cv2.ellipse(half, (450, 450), (380, 380), 0.0, 0.0, 180.0, (120, 124, 128), 14)
        self.assertAlmostEqual(ar.vessel_rim_support(lab_of(half), dish), 0.5, delta=0.08)

    def check_test_vessel_search_keeps_isolated_compact_seeds_of_one_size(self):
        image, dish = dish_image()
        isolated = ((300, 300), (520, 280), (330, 560), (600, 600), (450, 430))
        for centre in isolated:
            place_seed(image, centre)
        # A touching pair, a stick and a speck must not count.
        place_seed(image, (250, 430))
        place_seed(image, (290, 450))
        cv2.rectangle(image, (560, 420), (660, 432), SEED, -1)
        place_seed(image, (640, 300), axes=(6, 5))
        size, count, seeds, ovals, diameters = _confident_vessel_seeds(
            image, dish, settings(), CudaContext.resolve(), vessel_present=True
        )
        self.assertEqual(count, len(isolated))
        self.assertEqual(len(seeds), len(isolated))
        self.assertTrue(all(seed.source == "vessel" for seed in seeds))
        # The processing width keeps its meaning: the corrected long side.
        self.assertAlmostEqual(size, 0.72 * 60.0, delta=2.5)
        centres = sorted((round(oval.center_x), round(oval.center_y)) for oval in ovals)
        for found, expected in zip(centres, sorted(isolated)):
            self.assertLessEqual(np.hypot(found[0] - expected[0], found[1] - expected[1]), 2.0)
        for seed in seeds:
            self.assertEqual(seed.mask.dtype, np.uint8)
            self.assertLess(seed.mask.shape[0], 120)

    def check_test_too_few_confident_seeds_leave_the_size_unset(self):
        image, dish = dish_image()
        place_seed(image, (300, 300))
        place_seed(image, (520, 280))
        found = _confident_vessel_seeds(
            image, dish, settings(), CudaContext.resolve(), vessel_present=True
        )
        self.assertIsNone(found[0])
        found = _confident_vessel_seeds(
            image, dish, settings(confident_seed_minimum_count=2),
            CudaContext.resolve(), vessel_present=True,
        )
        self.assertEqual(found[1], 2)

    def check_test_scene_search_skips_the_chart_ruler_and_border(self):
        image = np.full((700, 1000, 3), FLOOR, np.uint8)
        # A brighter patch of floor behind the row is lighting, not seed.
        cv2.rectangle(image, (150, 170), (430, 232), tuple(v + 15 for v in FLOOR), -1)
        # The row is sown 16 px apart (a third of a seed width): inside their
        # bounding boxes plus any margin, but clear of each outline.
        scene = ((200, 200), (274, 200), (348, 200), (260, 420), (450, 380))
        for centre in scene:
            place_seed(image, centre)
        # Dust beside a seed does not make it crowded.
        cv2.circle(image, (450, 345), 3, SEED, -1)
        # A pair only 10 px apart is not confidently isolated.
        place_seed(image, (150, 560))
        place_seed(image, (218, 560))
        # Seed-like patches on a chart, and a seed cut by the image border.
        chart = np.array(((620.0, 120.0), (900.0, 120.0), (900.0, 560.0), (620.0, 560.0)))
        cv2.rectangle(image, (620, 120), (900, 560), (40, 40, 40), -1)
        for y in (180, 300, 420):
            cv2.rectangle(image, (660, y), (720, y + 60), (180, 120, 200), -1)
        place_seed(image, (12, 600))
        dish = DishCircle(500, 350, 300, 1.0)
        size, count, seeds, ovals, _diameters = _confident_vessel_seeds(
            image, dish, settings(), CudaContext.resolve(),
            vessel_present=False, excluded_polygons=(chart,),
        )
        self.assertEqual(count, len(scene))
        self.assertTrue(all(seed.source == "scene" for seed in seeds))
        self.assertAlmostEqual(size, 0.72 * 60.0, delta=2.5)
        for oval in ovals:
            self.assertLess(oval.center_x, 600)
            self.assertGreater(oval.center_x, 60)
        # Specks agree with each other but cannot outvote fewer, larger seeds.
        dusty = np.full((700, 1000, 3), FLOOR, np.uint8)
        for centre in ((200, 200), (400, 200), (300, 450)):
            place_seed(dusty, centre)
        for x in range(560, 960, 50):
            for y in (150, 300, 450):
                place_seed(dusty, (x, y), axes=(6, 5))
        size, count, seeds, _ovals, _diameters = _confident_vessel_seeds(
            dusty, dish, settings(), CudaContext.resolve(), vessel_present=False
        )
        self.assertEqual(count, 3)
        self.assertAlmostEqual(size, 0.72 * 60.0, delta=2.5)

    def check_test_printed_characters_and_marks_off_the_backdrop_are_not_seeds(self):
        from unittest.mock import patch

        from seedvision.segmentation import baseline

        context = CudaContext.resolve()
        dish = DishCircle(500, 350, 300, 1.0)

        def search(image):
            return _confident_vessel_seeds(
                image, dish, settings(), context, vessel_present=False
            )

        # Isolated, compact and of one size, but drawn in thin strokes.
        label = np.full((700, 1000, 3), FLOOR, np.uint8)
        for text, x, y in (("0", 150, 250), ("8", 400, 250), ("6", 650, 250),
                           ("9", 300, 520), ("0", 600, 520)):
            cv2.putText(label, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.4,
                        (25, 25, 25), 7, cv2.LINE_AA)
        self.assertIsNone(search(label)[0])
        with patch.object(baseline, "_CONFIDENT_SEED_MINIMUM_STROKE", 0.0):
            self.assertEqual(search(label)[1], 5)

        # Dark marks on a wooden wall: clear of each other, but the floor
        # around them is the wall, not the backdrop the seeds lie on.
        wall = np.full((700, 1000, 3), FLOOR, np.uint8)
        cv2.rectangle(wall, (0, 0), (1000, 260), (70, 120, 170), -1)
        for x in (150, 350, 550, 750):
            cv2.ellipse(wall, (x, 130), (14, 10), 0.0, 0.0, 360.0, (30, 40, 50), -1)
        self.assertIsNone(search(wall)[0])
        with patch.object(baseline, "_CONFIDENT_SEED_BACKDROP_MINIMUM_DISTANCES", 1000.0):
            self.assertEqual(search(wall)[1], 4)
        seeds = ((250, 450), (500, 470), (750, 440))
        for centre in seeds:
            place_seed(wall, centre)
        _size, count, _seeds, ovals, _diameters = search(wall)
        self.assertEqual(count, len(seeds))
        self.assertEqual(
            sorted((round(oval.center_x), round(oval.center_y)) for oval in ovals),
            sorted(seeds),
        )

    def check_test_blobs_near_the_mask_cleaning_scale_are_not_seeds(self):
        from unittest.mock import patch

        from seedvision.segmentation import baseline

        # Compact, isolated and of one size, but only two cleaning-kernel
        # widths across, like barcode bars chopped into blobs.
        blobs = np.full((700, 1000, 3), FLOOR, np.uint8)
        for centre in ((200, 200), (450, 220), (700, 180), (300, 480), (620, 500)):
            cv2.circle(blobs, centre, 3, (25, 25, 25), -1)
        dish = DishCircle(500, 350, 300, 1.0)

        def search():
            return _confident_vessel_seeds(
                blobs, dish, settings(), CudaContext.resolve(), vessel_present=False
            )

        self.assertIsNone(search()[0])
        with patch.object(baseline, "_CONFIDENT_SEED_MINIMUM_KERNEL_WIDTHS", 0.0):
            self.assertEqual(search()[1], 5)

    def check_test_each_class_comes_from_its_source(self):
        context = CudaContext.resolve()
        image, dish = dish_image()
        centres = ((300, 300), (520, 280), (330, 560), (600, 600))
        for centre in centres:
            place_seed(image, centre)
        # A seed heaped against the wall: not confident, and not Other either.
        place_seed(image, (78, 450), angle=90.0)
        _size, _count, seeds, _ovals, _diameters = _confident_vessel_seeds(
            image, dish, settings(), context, vessel_present=True
        )
        self.assertEqual(len(seeds), len(centres))
        lab = lab_of(image)
        chart = np.array(((20.0, 20.0), (120.0, 20.0), (120.0, 90.0), (20.0, 90.0)))
        floor_lab = lab[0, :, 450, 450].float()
        agreeing = floor_lab[None, :].repeat(64, 1)
        references = ar.automatic_material_references(
            lab, dish, seeds, 43.0, 4.0, vessel_present=True,
            perimeter_samples_lab=agreeing, chart_polygon=chart, settings=settings(),
        )
        self.assertEqual(references.foreground_source, "confident seeds in the vessel")
        self.assertEqual(references.background_source, "perimeter ring")
        self.assertEqual(references.other_sources, ("vessel wall", "colour chart"))
        self.assertFalse(references.replaces_perimeter)
        foreground = references.foreground.cpu().numpy()
        seed_pixels = np.all(image == SEED, axis=2)
        self.assertTrue(foreground.any())
        self.assertFalse(np.any(foreground & ~seed_pixels))
        # The inset keeps edges out: every Foreground pixel lies well inside.
        inside_distance = cv2.distanceTransform(seed_pixels.astype(np.uint8), cv2.DIST_L2, 5)
        self.assertGreaterEqual(float(inside_distance[foreground].min()), 0.15 * 43.0 - 1.0)
        other = references.other.cpu().numpy()
        self.assertTrue(other[450, 450 + 380])
        self.assertTrue(other[450 - 380, 450])
        self.assertFalse(other[450, 450])
        heaped = np.zeros(other.shape, np.uint8)
        cv2.ellipse(heaped, (78, 450), (30, 20), 90.0, 0.0, 360.0, 1, -1)
        self.assertFalse(np.any(other & (heaped > 0)))
        self.assertTrue(other[50, 60])

        # A vessel floor unlike the surface outside replaces the perimeter ring.
        disagreeing = (floor_lab + torch.tensor((0.0, 40.0, 0.0), device=floor_lab.device))[None, :].repeat(64, 1)
        rings = ar.automatic_material_references(
            lab, dish, seeds, 43.0, 4.0, vessel_present=True,
            perimeter_samples_lab=disagreeing, settings=settings(),
        )
        self.assertEqual(rings.background_source, "around confident seeds")
        self.assertTrue(rings.replaces_perimeter)
        self.assertGreater(rings.perimeter_disagreement, 15.0)
        background = rings.background.cpu().numpy()
        self.assertFalse(np.any(background & seed_pixels))
        self.assertIn("perimeter ring differs", ar.describe(rings))

        # Without seeds or a perimeter ring, the image border is Background,
        # and without a visible vessel its wall is not Other.
        border = ar.automatic_material_references(
            lab, dish, (), 43.0, 4.0, vessel_present=False, chart_polygon=chart,
            settings=settings(),
        )
        self.assertEqual(border.foreground_source, "none")
        self.assertIsNone(border.foreground)
        self.assertEqual(border.background_source, "image border")
        self.assertEqual(border.other_sources, ("colour chart",))
        self.assertTrue(border.background.cpu().numpy()[5, 450])
        self.assertFalse(border.background.cpu().numpy()[50, 60])

    def check_test_controls_switch_classes_and_shape_the_areas(self):
        context = CudaContext.resolve()
        image, dish = dish_image()
        for centre in ((300, 300), (520, 280), (330, 560), (600, 600)):
            place_seed(image, centre)
        _size, _count, seeds, _ovals, _diameters = _confident_vessel_seeds(
            image, dish, settings(), context, vessel_present=True
        )
        lab = lab_of(image)
        off = ar.automatic_material_references(
            lab, dish, seeds, 43.0, 4.0, vessel_present=True,
            settings=settings(automatic_foreground_reference=False,
                              automatic_background_reference=False,
                              automatic_other_reference=False),
        )
        self.assertIsNone(off.foreground)
        self.assertIsNone(off.background)
        self.assertIsNone(off.other)
        narrow = ar.automatic_material_references(
            lab, dish, seeds, 43.0, 4.0, vessel_present=True,
            settings=settings(automatic_seed_inset_fraction=0.30),
        )
        wide = ar.automatic_material_references(
            lab, dish, seeds, 43.0, 4.0, vessel_present=True, settings=settings(),
        )
        self.assertLess(int(narrow.foreground.sum()), int(wide.foreground.sum()))
        thick_wall = ar.automatic_material_references(
            lab, replace(dish, rim_pair_detected=False), seeds, 43.0, 4.0, vessel_present=True,
            settings=settings(automatic_wall_width_mm=8.0),
        )
        self.assertGreater(int(thick_wall.other.sum()), int(wide.other.sum()))

    def check_test_painted_references_thin_the_automatic_area(self):
        mask = torch.ones((200, 200), dtype=torch.bool)
        self.assertIs(ar.thinned(mask, 0, 0.5), mask)
        self.assertEqual(ar.authority(0, 100.0, 0.1, 0.5), 1.0)
        weight = ar.authority(400, 100.0, 0.1, 0.5)
        self.assertAlmostEqual(weight, 0.1 + 0.9 * 2.0 ** (-8.0), places=6)
        kept = int(ar.thinned(mask, 400, weight).sum())
        self.assertAlmostEqual(kept, weight * 400, delta=0.15 * weight * 400 + 5)
        # The kept pixels spread across the whole area.
        thinned = ar.thinned(mask, 4000, 0.5).numpy()
        self.assertTrue(thinned[:100].any() and thinned[100:].any())
        self.assertTrue(thinned[:, :100].any() and thinned[:, 100:].any())

    def check_test_automatic_seed_references_become_foreground_evidence(self):
        context = CudaContext.resolve()
        crop = np.full((128, 128, 3), (180, 190, 200), np.uint8)
        crop[36:92, 18:52] = (55, 65, 75)
        crop[36:92, 80:110] = (55, 65, 75)
        seed_interior = np.zeros((128, 128), dtype=bool)
        seed_interior[48:80, 26:44] = True
        base = BaselineSettings(foreground_reference_weight=0.90)
        unreferenced = _foreground_feature(crop, base, context, seed_diameter=28.0)
        self.assertFalse(np.any(unreferenced[1]))
        self.assertIsNone(unreferenced[-2])
        referenced = _foreground_feature(
            crop, base, context, seed_diameter=28.0,
            automatic_foreground_reference_mask=torch.from_numpy(seed_interior),
        )
        probability = np.asarray(referenced[1])
        profile = referenced[-2]
        # The model reaches the second, unreferenced seed but not the floor.
        self.assertGreater(int(probability[64, 95]), 200)
        self.assertLess(int(probability[10, 10]), 20)
        self.assertEqual(profile.source, "automatic_seeds")
        source = np.asarray(referenced[-1]) > 0
        self.assertTrue(np.array_equal(source, seed_interior))
        # Painted Background takes precedence over the automatic seed area.
        painted = np.zeros((128, 128), dtype=bool)
        painted[48:80, 26:44] = True
        overruled = _foreground_feature(
            crop, base, context, seed_diameter=28.0,
            background_reference_mask=painted,
            automatic_foreground_reference_mask=torch.from_numpy(seed_interior),
        )
        self.assertIsNone(overruled[-2])

    def check_test_node_controls_reach_the_settings(self):
        from seedvision.pipeline.model import build_default_pipeline
        from seedvision.pipeline.settings import production_settings

        graph = build_default_pipeline()
        node = graph.node("automatic_material_references")
        for key in node.parameters:
            self.assertTrue(hasattr(BaselineSettings(), key), key)
        node.parameters["automatic_perimeter_disagreement"] = 7.5
        graph.node("seed_scale_estimation").parameters["vessel_minimum_rim_support"] = 0.6
        built = production_settings(graph)["settings"]
        self.assertEqual(built.automatic_perimeter_disagreement, 7.5)
        self.assertEqual(built.vessel_minimum_rim_support, 0.6)
        self.assertEqual(
            set(graph.downstream("automatic_material_references")),
            {"background_likelihood", "refined_background_likelihood",
             "reference_texture_prototypes", "material_evidence_decision"},
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the reference search")
    def test_sources(self):
        """7 related AutomaticMaterialReferenceTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_rim_support_separates_a_visible_vessel_from_a_fitted_circle)
        self.scenario(self.check_test_vessel_search_keeps_isolated_compact_seeds_of_one_size)
        self.scenario(self.check_test_too_few_confident_seeds_leave_the_size_unset)
        self.scenario(self.check_test_scene_search_skips_the_chart_ruler_and_border)
        self.scenario(self.check_test_printed_characters_and_marks_off_the_backdrop_are_not_seeds)
        self.scenario(self.check_test_blobs_near_the_mask_cleaning_scale_are_not_seeds)
        self.scenario(self.check_test_each_class_comes_from_its_source)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the reference search")
    def test_weighting(self):
        """4 related AutomaticMaterialReferenceTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_controls_switch_classes_and_shape_the_areas)
        self.scenario(self.check_test_painted_references_thin_the_automatic_area)
        self.scenario(self.check_test_automatic_seed_references_become_foreground_evidence)
        self.scenario(self.check_test_node_controls_reach_the_settings)


if __name__ == "__main__":
    unittest.main()
