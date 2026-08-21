from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.cuda import GpuRaster
from seedvision.segmentation.procedural import (
    MANUAL_CENTRE_SUPPRESSION_DIAMETER_FRACTION,
    ManualSeedCentreMode,
    ManualSeedCentreSpace,
    ManualSeedCentres,
    ProceduralInstanceSettings,
    ProceduralMarkerSource,
    prepare_procedural_instance_inputs,
    procedural_seed_instances,
    procedural_seed_instances_from_prepared,
)


class ProceduralInstanceTests(unittest.TestCase):
    @staticmethod
    def _touching_seed_scene():
        shape = (240, 240)
        valid = np.zeros(shape, np.uint8)
        cv2.circle(valid, (120, 120), 112, 255, -1)
        identities = np.zeros(shape, np.int32)
        for identifier, centre, angle in (
            (1, (78, 120), -18),
            (2, (120, 120), 12),
            (3, (162, 120), -8),
        ):
            cv2.ellipse(identities, centre, (29, 22), angle, 0, 360, identifier, -1)
        material = np.uint8(identities > 0) * 235
        material = cv2.GaussianBlur(material, (0, 0), 2.0)
        boundary = np.zeros(shape, np.uint8)
        for identifier in (1, 2, 3):
            region = np.uint8(identities == identifier) * 255
            boundary = np.maximum(
                boundary,
                cv2.morphologyEx(region, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)),
            )
        background = np.uint8(255 - material)
        return valid, identities, material, background, boundary

    @staticmethod
    def _packed_material_scene():
        shape = (320, 320)
        valid = np.zeros(shape, np.uint8)
        cv2.circle(valid, (160, 160), 150, 255, -1)
        material = np.zeros(shape, np.uint8)
        cv2.circle(material, (160, 160), 135, 235, -1)
        return valid, material, 255 - material

    @classmethod
    def _manual_centre_scene_arguments(cls):
        valid, _identities, material, background, boundary = (
            cls._touching_seed_scene()
        )
        settings = ProceduralInstanceSettings(
            centre_minimum_separation_fraction=0.58,
            minimum_marker_score=0.10,
            marker_count_multiplier=0.50,
        )
        arguments = dict(
            foreground_probability=material,
            foreground_noise_probability=material,
            background_probability=background,
            refined_background_probability=background,
            edge_magnitude=boundary,
            edge_ridges=boundary,
            settings=settings,
        )
        return valid, arguments

    def test_no_manual_centre_edits_are_bit_exact_with_legacy_result(self) -> None:
        valid, arguments = self._manual_centre_scene_arguments()
        legacy = procedural_seed_instances(valid, 54.0, **arguments)
        explicit_none = procedural_seed_instances(
            valid, 54.0, **arguments, manual_seed_centres=None
        )
        empty_augment = procedural_seed_instances(
            valid,
            54.0,
            **arguments,
            manual_seed_centres=ManualSeedCentres(),
        )

        for candidate in (explicit_none, empty_augment):
            for field_name in (
                "labels",
                "centres_xy",
                "marker_scores",
                "instance_confidences",
                "occupancy_likelihood",
                "occupancy_mask",
                "boundary_cost",
                "centre_likelihood",
                "marker_centres_xy",
                "marker_sources",
                "rejected_manual_centres_xy",
            ):
                np.testing.assert_array_equal(
                    getattr(candidate, field_name), getattr(legacy, field_name)
                )
            self.assertEqual(
                candidate.rejected_manual_centre_reasons,
                legacy.rejected_manual_centre_reasons,
            )

    def test_augment_adds_and_overrides_without_duplicate_markers(self) -> None:
        valid, arguments = self._manual_centre_scene_arguments()
        baseline = procedural_seed_instances(valid, 54.0, **arguments)
        baseline_automatic = baseline.marker_centres_for_source(
            ProceduralMarkerSource.AUTOMATIC
        )
        self.assertEqual(len(baseline_automatic), 2)

        # The low marker-count prior omits the left seed. Augmenting that seed
        # must preserve both untouched automatic markers.
        added_point = np.asarray((78.0, 120.0), np.float32)
        added = procedural_seed_instances(
            valid,
            54.0,
            **arguments,
            manual_seed_centres=ManualSeedCentres(
                (tuple(added_point),), ManualSeedCentreMode.AUGMENT
            ),
        )
        np.testing.assert_allclose(
            added.marker_centres_for_source(ProceduralMarkerSource.MANUAL),
            added_point[None],
            atol=0.01,
        )
        np.testing.assert_array_equal(
            added.marker_centres_for_source(ProceduralMarkerSource.AUTOMATIC),
            baseline_automatic,
        )

        # Moving a manual point onto an automatic marker replaces that marker,
        # but cannot leave a duplicate within the documented 0.45-diameter band.
        override_point = baseline_automatic[0] + np.asarray((-2.0, -2.0))
        overridden = procedural_seed_instances(
            valid,
            54.0,
            **arguments,
            manual_seed_centres=ManualSeedCentres(
                (tuple(override_point),), ManualSeedCentreMode.AUGMENT
            ),
        )
        automatic = overridden.marker_centres_for_source(
            ProceduralMarkerSource.AUTOMATIC
        )
        manual = overridden.marker_centres_for_source(
            ProceduralMarkerSource.MANUAL
        )
        self.assertEqual(len(manual), 1)
        self.assertEqual(len(automatic), len(baseline_automatic) - 1)
        self.assertTrue(
            np.all(
                np.linalg.norm(automatic - manual[0], axis=1)
                >= 54.0 * MANUAL_CENTRE_SUPPRESSION_DIAMETER_FRACTION
            )
        )

    def test_replace_mode_and_empty_replace_delete_automatic_markers(self) -> None:
        valid, arguments = self._manual_centre_scene_arguments()
        replacement = procedural_seed_instances(
            valid,
            54.0,
            **arguments,
            manual_seed_centres=ManualSeedCentres(
                ((78.0, 120.0),),
                ManualSeedCentreMode.REPLACE_AUTOMATIC,
            ),
        )
        np.testing.assert_allclose(
            replacement.marker_centres_xy,
            np.asarray(((78.0, 120.0),), np.float32),
        )
        np.testing.assert_array_equal(
            replacement.marker_sources,
            np.asarray((int(ProceduralMarkerSource.MANUAL),), np.uint8),
        )

        deleted = procedural_seed_instances(
            valid,
            54.0,
            **arguments,
            manual_seed_centres=ManualSeedCentres(
                (), ManualSeedCentreMode.REPLACE_AUTOMATIC
            ),
        )
        self.assertEqual(deleted.count, 0)
        self.assertEqual(deleted.marker_centres_xy.shape, (0, 2))
        self.assertEqual(deleted.marker_sources.shape, (0,))
        self.assertFalse(np.any(deleted.labels))

    def test_annotations_remain_whole_mask_authority_over_manual_centres(self) -> None:
        valid, arguments = self._manual_centre_scene_arguments()
        annotations = np.zeros(valid.shape, np.uint16)
        cv2.circle(annotations, (78, 120), 10, 7, -1)
        result = procedural_seed_instances(
            valid,
            54.0,
            **arguments,
            seed_instance_annotations=annotations,
            manual_seed_centres=ManualSeedCentres(
                ((80.0, 120.0), (162.0, 120.0)),
                ManualSeedCentreMode.REPLACE_AUTOMATIC,
            ),
        )

        annotation_labels = np.unique(result.labels[annotations > 0])
        self.assertEqual(len(annotation_labels), 1)
        self.assertGreater(int(annotation_labels[0]), 0)
        np.testing.assert_array_equal(
            result.marker_sources,
            np.asarray(
                (
                    int(ProceduralMarkerSource.ANNOTATED),
                    int(ProceduralMarkerSource.MANUAL),
                ),
                np.uint8,
            ),
        )
        np.testing.assert_allclose(
            result.editable_marker_centres_xy,
            np.asarray(((162.0, 120.0),), np.float32),
        )
        self.assertEqual(
            result.rejected_manual_centre_reasons,
            ("near_annotated_instance",),
        )
        np.testing.assert_allclose(
            result.rejected_manual_centres_xy,
            np.asarray(((80.0, 120.0),), np.float32),
        )

    def test_marker_metadata_is_not_replaced_by_instance_centroids(self) -> None:
        valid, arguments = self._manual_centre_scene_arguments()
        marker = np.asarray((60.0, 120.0), np.float32)
        result = procedural_seed_instances(
            valid,
            54.0,
            **arguments,
            manual_seed_centres=ManualSeedCentres(
                (tuple(marker),), ManualSeedCentreMode.REPLACE_AUTOMATIC
            ),
        )

        self.assertEqual(result.count, 1)
        np.testing.assert_allclose(result.marker_centres_xy[0], marker)
        self.assertEqual(
            int(result.marker_sources[0]), int(ProceduralMarkerSource.MANUAL)
        )
        self.assertGreater(
            float(np.linalg.norm(result.centres_xy[0] - marker)), 8.0
        )

    def test_invalid_outside_duplicate_and_culled_manual_centres_are_reported(self) -> None:
        with self.assertRaises(ValueError):
            ManualSeedCentres(((float("nan"), 4.0),))
        with self.assertRaises(ValueError):
            ManualSeedCentres((), mode="unknown")
        with self.assertRaises(ValueError):
            ManualSeedCentres((), coordinate_space="screen")

        valid, arguments = self._manual_centre_scene_arguments()
        rejected = procedural_seed_instances(
            valid,
            54.0,
            **arguments,
            manual_seed_centres=ManualSeedCentres(
                (
                    (-1.0, 10.0),
                    (0.0, 0.0),
                    (120.0, 20.0),
                    (78.0, 120.0),
                    (78.2, 120.2),
                ),
                ManualSeedCentreMode.REPLACE_AUTOMATIC,
            ),
        )
        self.assertEqual(
            rejected.rejected_manual_centre_reasons,
            (
                "outside_source",
                "outside_valid_region",
                "outside_material",
                "duplicate_working_pixel",
            ),
        )
        np.testing.assert_allclose(
            rejected.rejected_manual_centres_xy,
            np.asarray(
                ((-1.0, 10.0), (0.0, 0.0), (120.0, 20.0), (78.2, 120.2)),
                np.float32,
            ),
        )

        culled_arguments = dict(arguments)
        culled_arguments["settings"] = ProceduralInstanceSettings(
            centre_minimum_separation_fraction=0.58,
            minimum_marker_score=0.10,
            marker_count_multiplier=0.50,
            minimum_instance_area_fraction=1.44,
            maximum_instance_area_fraction=1.45,
        )
        culled = procedural_seed_instances(
            valid,
            54.0,
            **culled_arguments,
            manual_seed_centres=ManualSeedCentres(
                ((78.0, 120.0),), ManualSeedCentreMode.REPLACE_AUTOMATIC
            ),
        )
        self.assertEqual(
            culled.rejected_manual_centre_reasons,
            ("instance_below_minimum_area",),
        )
        np.testing.assert_allclose(
            culled.rejected_manual_centres_xy,
            np.asarray(((78.0, 120.0),), np.float32),
        )

    def test_source_centres_transform_once_to_corrected_coordinates(self) -> None:
        from types import SimpleNamespace

        from seedvision.segmentation.baseline import (
            _manual_centres_in_corrected_coordinates,
        )

        calls: list[tuple[tuple[float, float], ...]] = []

        def transform(points):
            calls.append(tuple(tuple(value) for value in points))
            values = np.asarray(points, np.float64)
            return values * np.asarray((1.5, 0.75)) + np.asarray((11.0, -4.0))

        source = ManualSeedCentres(
            ((10.0, 20.0), (30.0, 40.0)),
            ManualSeedCentreMode.AUGMENT,
            ManualSeedCentreSpace.SOURCE_IMAGE,
        )
        corrected = _manual_centres_in_corrected_coordinates(
            source, SimpleNamespace(transform_points=transform)
        )
        assert corrected is not None
        self.assertEqual(
            corrected.coordinate_space, ManualSeedCentreSpace.CORRECTED_IMAGE
        )
        np.testing.assert_allclose(
            corrected.centres_xy, ((26.0, 11.0), (56.0, 26.0))
        )
        self.assertEqual(calls, [source.centres_xy])
        self.assertIs(
            _manual_centres_in_corrected_coordinates(
                corrected, SimpleNamespace(transform_points=lambda _points: None)
            ),
            corrected,
        )
        np.testing.assert_allclose(
            corrected.translated(-6.0, -8.0).centres_xy,
            ((20.0, 3.0), (50.0, 18.0)),
        )

    def test_redesigned_boundary_settings_are_validated(self) -> None:
        invalid_settings = (
            {"boundary_edge_weight": 0.0, "boundary_ridge_weight": 0.0},
            {"boundary_semantic_floor": 1.01},
            {"boundary_physical_ridge_weight": -0.01},
            {"boundary_trace_weight": 1.01},
            {"trace_minimum_length_fraction": 0.01},
            {"trace_convexity_weight": -0.01},
            {"centre_geometry_smoothing_fraction": 0.01},
        )
        for parameters in invalid_settings:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                ProceduralInstanceSettings(**parameters)

    def test_retired_sensor_and_shadow_boundary_inputs_are_not_accepted(self) -> None:
        valid, _identities, material, background, boundary = self._touching_seed_scene()
        arguments = dict(
            foreground_probability=material,
            foreground_noise_probability=material,
            background_probability=background,
            refined_background_probability=background,
            edge_magnitude=boundary,
            edge_ridges=boundary,
        )
        with self.assertRaises(TypeError):
            procedural_seed_instances(
                valid,
                54.0,
                **arguments,
                sensor_noise=boundary,
            )
        with self.assertRaises(TypeError):
            prepare_procedural_instance_inputs(
                valid,
                54.0,
                **arguments,
                shadow_likelihood=boundary,
            )

    def test_separates_touching_seed_shapes_and_exposes_diagnostics(self) -> None:
        valid, _identities, material, background, boundary = self._touching_seed_scene()
        result = procedural_seed_instances(
            valid,
            54.0,
            foreground_probability=material,
            foreground_noise_probability=material,
            background_probability=background,
            refined_background_probability=background,
            edge_magnitude=boundary,
            edge_ridges=boundary,
            settings=ProceduralInstanceSettings(
                centre_minimum_separation_fraction=0.58,
                minimum_marker_score=0.10,
            ),
        )
        self.assertEqual(result.labels.shape, valid.shape)
        self.assertGreaterEqual(result.count, 3)
        self.assertEqual(int(result.labels.max()), result.count)
        self.assertEqual(result.instance_rgba().shape, (*valid.shape, 4))
        self.assertEqual(result.confidence_raster().shape, valid.shape)
        self.assertGreater(np.count_nonzero(result.boundary_cost), 0)
        self.assertGreater(np.count_nonzero(result.centre_likelihood), 0)

    def test_dense_component_supplements_suppressed_regional_maxima(self) -> None:
        valid, material, background = self._packed_material_scene()
        empty = np.zeros(valid.shape, np.uint8)
        settings = ProceduralInstanceSettings(minimum_marker_score=0.08)
        result = procedural_seed_instances(
            valid,
            40.0,
            foreground_probability=material,
            foreground_noise_probability=material,
            background_probability=background,
            refined_background_probability=background,
            edge_magnitude=empty,
            edge_ridges=empty,
            settings=settings,
        )
        occupied_area = int(np.count_nonzero(result.occupancy_mask))
        expected = round(
            occupied_area
            / (settings.packed_seed_cell_fraction * 40.0 * 40.0)
        )
        target = round(expected * settings.marker_count_multiplier)

        # A featureless packed component has only one regional maximum.  The
        # calibrated packing prior should recover its seed-scale population,
        # while the explicit target remains a hard upper bound.
        self.assertGreaterEqual(result.count, round(expected * 0.90))
        self.assertLessEqual(result.count, target)

    def test_supplemental_markers_do_not_split_a_complete_annotation(self) -> None:
        valid, material, background = self._packed_material_scene()
        empty = np.zeros(valid.shape, np.uint8)
        annotations = np.zeros(valid.shape, np.uint16)
        cv2.ellipse(annotations, (160, 160), (18, 14), 20, 0, 360, 7, -1)
        result = procedural_seed_instances(
            valid,
            40.0,
            foreground_probability=material,
            foreground_noise_probability=material,
            background_probability=background,
            refined_background_probability=background,
            edge_magnitude=empty,
            edge_ridges=empty,
            seed_instance_annotations=annotations,
            settings=ProceduralInstanceSettings(minimum_marker_score=0.08),
        )

        annotation_labels = np.unique(result.labels[annotations > 0])
        self.assertEqual(len(annotation_labels), 1)
        self.assertGreater(int(annotation_labels[0]), 0)
        self.assertGreater(result.count, 20)

    def test_internal_edge_texture_cannot_exceed_calibrated_marker_target(self) -> None:
        valid, material, background = self._packed_material_scene()
        texture = np.zeros(valid.shape, np.uint8)
        for coordinate in range(50, 281, 13):
            cv2.line(texture, (35, coordinate), (285, coordinate), 255, 2)
            cv2.line(texture, (coordinate, 35), (coordinate, 285), 255, 2)
        settings = ProceduralInstanceSettings(minimum_marker_score=0.08)
        result = procedural_seed_instances(
            valid,
            40.0,
            foreground_probability=material,
            foreground_noise_probability=material,
            background_probability=background,
            refined_background_probability=background,
            edge_magnitude=texture,
            edge_ridges=texture,
            non_edge_probability=texture,
            settings=settings,
        )
        occupied_area = int(np.count_nonzero(result.occupancy_mask))
        expected = round(
            occupied_area
            / (settings.packed_seed_cell_fraction * 40.0 * 40.0)
        )
        target = round(expected * settings.marker_count_multiplier)

        # Even adversarial high-frequency internal ridges cannot nominate an
        # unbounded number of centres; count comes from calibrated geometry.
        self.assertGreater(result.count, 1)
        self.assertLessEqual(result.count, target)

    def test_distinct_painted_instances_become_authoritative_markers(self) -> None:
        valid, identities, material, background, boundary = self._touching_seed_scene()
        annotations = np.zeros_like(identities, dtype=np.uint16)
        for identifier, centre in ((1, (78, 120)), (2, (120, 120)), (3, (162, 120))):
            cv2.circle(annotations, centre, 4, identifier, -1)
        valid = cv2.resize(valid, (480, 480), interpolation=cv2.INTER_NEAREST)
        material = cv2.resize(material, (480, 480), interpolation=cv2.INTER_LINEAR)
        background = cv2.resize(background, (480, 480), interpolation=cv2.INTER_LINEAR)
        boundary = cv2.resize(boundary, (480, 480), interpolation=cv2.INTER_LINEAR)
        annotations = cv2.resize(annotations, (480, 480), interpolation=cv2.INTER_NEAREST)
        result = procedural_seed_instances(
            valid,
            108.0,
            foreground_probability=material,
            foreground_noise_probability=material,
            background_probability=background,
            refined_background_probability=background,
            edge_magnitude=boundary,
            edge_ridges=boundary,
            seed_instance_annotations=annotations,
            settings=ProceduralInstanceSettings(
                minimum_marker_score=0.95,
                working_maximum_dimension=256,
            ),
        )
        self.assertGreaterEqual(result.count, 3)
        for expected in ((156, 240), (240, 240), (324, 240)):
            distances = np.hypot(
                result.centres_xy[:, 0] - expected[0],
                result.centres_xy[:, 1] - expected[1],
            )
            self.assertLess(float(distances.min()), 30.0)

    def test_low_level_prepared_runner_rejects_untransformed_source_coordinates(self) -> None:
        valid, arguments = self._manual_centre_scene_arguments()
        settings = arguments.pop("settings")
        prepared = prepare_procedural_instance_inputs(
            valid,
            60.0,
            **arguments,
            working_maximum_dimension=settings.working_maximum_dimension,
        )
        with self.assertRaisesRegex(ValueError, "corrected-image"):
            procedural_seed_instances_from_prepared(
                prepared,
                manual_seed_centres=ManualSeedCentres(
                    ((78.0, 120.0),),
                    coordinate_space=ManualSeedCentreSpace.SOURCE_IMAGE,
                ),
                settings=settings,
            )

    def test_gpu_rasters_resize_without_materializing_full_host_caches(self) -> None:
        import torch

        valid, _identities, material, background, boundary = self._touching_seed_scene()
        valid = cv2.resize(valid, (480, 480), interpolation=cv2.INTER_NEAREST)
        material = cv2.resize(material, (480, 480), interpolation=cv2.INTER_LINEAR)
        background = cv2.resize(background, (480, 480), interpolation=cv2.INTER_LINEAR)
        boundary = cv2.resize(boundary, (480, 480), interpolation=cv2.INTER_LINEAR)

        def raster(values: np.ndarray, name: str) -> GpuRaster:
            return GpuRaster(
                torch.from_numpy(values)[None, None],
                numpy_dtype=np.uint8,
                name=name,
            )

        inputs = [
            raster(valid, "valid"),
            raster(material, "material"),
            raster(material, "material noise"),
            raster(background, "background"),
            raster(background, "background noise"),
            raster(boundary, "edge"),
            raster(boundary, "ridges"),
        ]
        result = procedural_seed_instances(
            inputs[0],
            108.0,
            foreground_probability=inputs[1],
            foreground_noise_probability=inputs[2],
            background_probability=inputs[3],
            refined_background_probability=inputs[4],
            edge_magnitude=inputs[5],
            edge_ridges=inputs[6],
            settings=ProceduralInstanceSettings(working_maximum_dimension=256),
        )
        self.assertGreater(result.count, 0)
        self.assertEqual(result.source_shape, (480, 480))
        self.assertLessEqual(max(result.labels.shape), 256)
        self.assertEqual(result.occupancy_likelihood.shape, result.labels.shape)
        self.assertEqual(result.boundary_cost.shape, result.labels.shape)
        self.assertTrue(all(not item.is_materialized for item in inputs))

    def test_prepared_inputs_match_direct_result_and_are_reused(self) -> None:
        import seedvision.segmentation.procedural as procedural_module

        valid, _identities, material, background, boundary = self._touching_seed_scene()
        settings = ProceduralInstanceSettings(
            centre_minimum_separation_fraction=0.58,
            minimum_marker_score=0.10,
        )
        arguments = {
            "foreground_probability": material,
            "foreground_noise_probability": material,
            "background_probability": background,
            "refined_background_probability": background,
            "edge_magnitude": boundary,
            "edge_ridges": boundary,
            "physical_edge_probability": boundary,
            "non_edge_probability": np.zeros_like(boundary),
            "thinned_reference_edge_ridges": boundary,
            "oriented_edge_trace_labels": np.zeros_like(boundary, np.int32),
            "oriented_edge_trace_continuity": boundary,
            "reference_surface_probability": material,
        }
        direct = procedural_seed_instances(
            valid,
            54.0,
            **arguments,
            settings=settings,
        )

        with patch.object(
            procedural_module,
            "_working_u8",
            wraps=procedural_module._working_u8,
        ) as working_u8, patch.object(
            procedural_module,
            "_working_labels",
            wraps=procedural_module._working_labels,
        ) as working_labels:
            prepared = prepare_procedural_instance_inputs(
                valid,
                54.0,
                **arguments,
                working_maximum_dimension=settings.working_maximum_dimension,
            )
            preparation_calls = working_u8.call_count
            label_preparation_calls = working_labels.call_count
            first = procedural_seed_instances_from_prepared(
                prepared,
                settings=settings,
            )
            procedural_seed_instances_from_prepared(
                prepared,
                settings=ProceduralInstanceSettings(
                    centre_minimum_separation_fraction=0.48,
                    minimum_marker_score=0.10,
                ),
            )
            self.assertEqual(working_u8.call_count, preparation_calls)
            self.assertEqual(working_labels.call_count, label_preparation_calls)

        self.assertEqual(preparation_calls, 12)
        self.assertEqual(label_preparation_calls, 1)
        for field in (
            "labels",
            "centres_xy",
            "marker_scores",
            "instance_confidences",
            "occupancy_likelihood",
            "occupancy_mask",
            "boundary_cost",
            "centre_likelihood",
        ):
            np.testing.assert_array_equal(getattr(first, field), getattr(direct, field))
        self.assertEqual(first.source_shape, direct.source_shape)
        self.assertEqual(first.working_scale, direct.working_scale)

    def test_prepared_annotation_targets_exclude_invalid_pixels(self) -> None:
        valid, _identities, material, background, boundary = self._touching_seed_scene()
        prepared = prepare_procedural_instance_inputs(
            valid,
            54.0,
            foreground_probability=material,
            foreground_noise_probability=material,
            background_probability=background,
            refined_background_probability=background,
            edge_magnitude=boundary,
            edge_ridges=boundary,
        )
        annotations = np.ones(valid.shape, np.uint16)
        targets = prepared.annotation_targets(annotations)
        self.assertFalse(np.any(targets[~prepared.valid_mask]))
        self.assertTrue(np.all(targets[prepared.valid_mask] == 1))

    def test_annotation_semantics_suppress_internal_coat_stripe_without_split(self) -> None:
        shape = (220, 220)
        valid = np.zeros(shape, np.uint8)
        cv2.circle(valid, (110, 110), 104, 255, -1)
        seed = np.zeros(shape, np.uint8)
        cv2.ellipse(seed, (110, 110), (34, 24), 18, 0, 360, 255, -1)
        material = cv2.GaussianBlur(seed, (0, 0), 1.2)
        background = 255 - material
        outer_edge = cv2.morphologyEx(
            seed, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        )
        stripe = np.zeros(shape, np.uint8)
        cv2.line(stripe, (86, 103), (134, 117), 255, 3)
        candidate_edges = np.maximum(outer_edge, stripe)

        result = procedural_seed_instances(
            valid,
            68.0,
            foreground_probability=material,
            foreground_noise_probability=material,
            background_probability=background,
            refined_background_probability=background,
            edge_magnitude=candidate_edges,
            edge_ridges=candidate_edges,
            physical_edge_probability=outer_edge,
            non_edge_probability=stripe,
            thinned_reference_edge_ridges=outer_edge,
            settings=ProceduralInstanceSettings(minimum_marker_score=0.08),
        )

        self.assertEqual(result.count, 1)
        boundary_cost = result.boundary_cost.astype(np.float32)
        self.assertGreater(
            float(np.mean(boundary_cost[outer_edge > 0])),
            4.0 * float(np.mean(boundary_cost[stripe > 0])),
        )

    def test_true_touching_boundary_remains_a_watershed_cut(self) -> None:
        shape = (240, 240)
        valid = np.zeros(shape, np.uint8)
        cv2.circle(valid, (120, 120), 112, 255, -1)
        left = np.zeros(shape, np.uint8)
        right = np.zeros(shape, np.uint8)
        cv2.ellipse(left, (96, 120), (32, 25), -8, 0, 360, 255, -1)
        cv2.ellipse(right, (144, 120), (32, 25), 8, 0, 360, 255, -1)
        seed_union = np.maximum(left, right)
        material = cv2.GaussianBlur(seed_union, (0, 0), 1.2)
        background = 255 - material
        physical = np.maximum(
            cv2.morphologyEx(left, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)),
            cv2.morphologyEx(right, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)),
        )
        result = procedural_seed_instances(
            valid,
            64.0,
            foreground_probability=material,
            foreground_noise_probability=material,
            background_probability=background,
            refined_background_probability=background,
            edge_magnitude=physical,
            edge_ridges=physical,
            physical_edge_probability=physical,
            non_edge_probability=np.zeros(shape, np.uint8),
            thinned_reference_edge_ridges=physical,
            settings=ProceduralInstanceSettings(
                centre_minimum_separation_fraction=0.52,
                minimum_marker_score=0.08,
            ),
        )
        self.assertEqual(result.count, 2)
        left_label = int(result.labels[120, 96])
        right_label = int(result.labels[120, 144])
        self.assertGreater(left_label, 0)
        self.assertGreater(right_label, 0)
        self.assertNotEqual(left_label, right_label)

    def test_sparse_semantic_training_keeps_unknown_edges_neutral(self) -> None:
        shape = (180, 180)
        valid = np.full(shape, 255, np.uint8)
        material = np.zeros(shape, np.uint8)
        cv2.circle(material, (90, 90), 62, 255, -1)
        candidate = np.zeros(shape, np.uint8)
        cv2.line(candidate, (42, 72), (138, 72), 220, 2)
        cv2.line(candidate, (42, 108), (138, 108), 220, 2)
        known_physical = np.zeros(shape, np.uint8)
        cv2.line(known_physical, (42, 72), (138, 72), 255, 2)
        unknown = np.zeros(shape, np.uint8)
        cv2.line(unknown, (42, 108), (138, 108), 255, 2)
        common = dict(
            foreground_probability=material,
            foreground_noise_probability=material,
            background_probability=255 - material,
            refined_background_probability=255 - material,
            edge_magnitude=candidate,
            edge_ridges=candidate,
            settings=ProceduralInstanceSettings(
                boundary_physical_ridge_weight=0.0,
                boundary_trace_weight=0.0,
                minimum_marker_score=0.08,
            ),
        )
        neutral = procedural_seed_instances(valid, 70.0, **common)
        sparse_semantics = procedural_seed_instances(
            valid,
            70.0,
            **common,
            physical_edge_probability=known_physical,
            non_edge_probability=np.zeros(shape, np.uint8),
        )
        neutral_unknown = float(np.mean(neutral.boundary_cost[unknown > 0]))
        classified_unknown = float(
            np.mean(sparse_semantics.boundary_cost[unknown > 0])
        )
        self.assertGreater(classified_unknown, 0.85 * neutral_unknown)

    def test_long_continuous_convex_trace_increases_boundary_support(self) -> None:
        shape = (220, 220)
        valid = np.zeros(shape, np.uint8)
        cv2.circle(valid, (110, 110), 104, 255, -1)
        seed = np.zeros(shape, np.uint8)
        cv2.ellipse(seed, (110, 110), (34, 25), 0, 0, 360, 255, -1)
        material = cv2.GaussianBlur(seed, (0, 0), 1.2)
        background = 255 - material
        outer = cv2.morphologyEx(
            seed, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        )
        trace_labels = np.zeros(shape, np.int32)
        cv2.ellipse(trace_labels, (110, 110), (28, 19), 0, 15, 175, 7, 1)
        trace_pixels = trace_labels > 0
        weak_arc = np.uint8(trace_pixels) * 60
        candidate = np.maximum(outer, weak_arc)
        common = dict(
            foreground_probability=material,
            foreground_noise_probability=material,
            background_probability=background,
            refined_background_probability=background,
            edge_magnitude=candidate,
            edge_ridges=candidate,
            physical_edge_probability=np.zeros(shape, np.uint8),
            non_edge_probability=np.zeros(shape, np.uint8),
            settings=ProceduralInstanceSettings(minimum_marker_score=0.08),
        )
        without_trace = procedural_seed_instances(valid, 68.0, **common)
        with_trace = procedural_seed_instances(
            valid,
            68.0,
            **common,
            oriented_edge_trace_labels=trace_labels,
            oriented_edge_trace_continuity=np.uint8(trace_pixels) * 255,
        )
        self.assertGreater(
            float(np.mean(with_trace.boundary_cost[trace_pixels])),
            1.35 * float(np.mean(without_trace.boundary_cost[trace_pixels])),
        )


if __name__ == "__main__":
    unittest.main()
