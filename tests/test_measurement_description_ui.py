"""Reviewed overlay facts use the same measurement records as their values."""

from tests.functional_units import functional_units
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QGraphicsLineItem, QGraphicsTextItem

from seedvision.measurement import geometry as g, shape_model as model
from seedvision.persistence.reference_regions import SeedInstanceAnnotation
from seedvision.ui.image_view import ImageView


@functional_units
class MeasurementDescriptionUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        path = Path(self.directory.name) / 'image.png'
        image = QImage(1000, 800, QImage.Format.Format_RGB888)
        image.fill(0)
        self.assertTrue(image.save(str(path)))
        self.view = ImageView()
        self.view.load_image(path)
        labels = np.zeros((800, 1000), np.uint16)
        cv2.ellipse(labels, (300, 450), (65, 40), 35, 0, 360, 1, -1)
        annotation = SeedInstanceAnnotation(1, shape_reviewed=True, outline_visibility='complete', pose='flat')
        self.summary = model.measure_reviewed_seed_instances(labels, (annotation,), pixels_per_mm=10)
        self.result = SimpleNamespace(calibration=SimpleNamespace(pixels_per_mm=10))

    def tearDown(self):
        self.view.close()
        self.directory.cleanup()

    def check_test_body_span_and_point_tooltips_bind_definitions_units_and_frame_without_refitting(self):
        with patch.object(g, 'measure_shape_mask', side_effect=AssertionError('unexpected refit')):
            self.view._render_reviewed_shape_geometry(self.result, self.summary, uncertainty=True)
            self.view._render_size_ovality_distribution(self.result, self.summary, None)
        tooltips = '\n'.join(item.toolTip() for item in self.view._overlay_items)
        self.assertIn('body length:', tooltips)
        self.assertIn('maximum span:', tooltips)
        self.assertIn('Greatest pairwise distance', tooltips)
        self.assertIn('mm', tooltips)
        self.assertIn('coordinates: input_mask', tooltips)
        self.assertIn('Visibility: complete; source image frame: unknown', tooltips)
        self.assertIn('not a confidence interval', tooltips)
        text = '\n'.join(item.toPlainText() for item in self.view._overlay_items if isinstance(item, QGraphicsTextItem))
        self.assertNotIn('1σ', text)
        self.assertIn('sensitivity estimates', text)

    def chart_line_count(self, measurement):
        reviewed = replace(self.summary.observations[0], measurement=measurement)
        summary = replace(self.summary, observations=(reviewed,))
        self.view._clear_overlay_items()
        self.view._render_size_ovality_distribution(self.result, summary, None)
        return sum(isinstance(item, QGraphicsLineItem) for item in self.view._overlay_items)

    def check_test_unavailable_legacy_and_boundary_errors_omit_bars_but_valid_zero_is_available(self):
        measurement = self.summary.observations[0].measurement
        legacy = replace(measurement, provenance=None)
        baseline = self.chart_line_count(legacy)
        tooltips = '\n'.join(item.toolTip() for item in self.view._overlay_items)
        self.assertIn('Definition: unknown', tooltips)
        self.assertIn('Sensitivity estimate unavailable', tooltips)
        missing = replace(measurement, provenance=replace(measurement.provenance,
            perturbation_radii_px=(), boundary_alternative_count=0))
        self.assertEqual(self.chart_line_count(missing), baseline)
        successful_zero = replace(measurement, measurement_uncertainty=(0.,) * 19)
        self.assertEqual(self.chart_line_count(successful_zero), baseline + 2)
        mask = np.zeros((800, 1000), np.uint8)
        cv2.ellipse(mask, (300, 450), (65, 40), 35, 0, 360, 1, -1)
        scale_only = g.measure_shape_mask(mask, perturbation_radii=(), pixels_per_mm=10,
                                         calibration_relative_uncertainty=.02)
        self.assertEqual(self.chart_line_count(scale_only), baseline + 1)

    def check_test_visibility_refresh_changes_metadata_without_changing_measurement_geometry(self):
        reviewed = self.summary.observations[0]
        records = {r.key: r for r in reviewed.measurement.described_values}
        initial = self.view._reviewed_shape_tooltip(reviewed, records, ('maximum_span',))
        changed_measurement = replace(reviewed.measurement, provenance=replace(
            reviewed.measurement.provenance, visibility='partly_occluded'))
        changed = replace(reviewed, annotation=replace(reviewed.annotation,
            outline_visibility='partly_occluded'), measurement=changed_measurement)
        updated = self.view._reviewed_shape_tooltip(changed,
            {r.key: r for r in changed_measurement.described_values}, ('maximum_span',))
        self.assertIn('Visibility: complete', initial)
        self.assertIn('Visibility: partly_occluded', updated)
        self.assertEqual(changed_measurement.vector(), reviewed.measurement.vector())
        self.assertEqual(changed_measurement.maximum_span_endpoints,
                         reviewed.measurement.maximum_span_endpoints)

    def check_test_proven_corrected_frame_reaches_native_overlay_from_stored_bundle(self):
        from seedvision.annotation.coordinates import reproject_bundle
        from seedvision.measurement.lineage import reviewed_measurement_context
        from seedvision.persistence.reference_regions import ReferenceRegionBundle
        labels = np.zeros((96, 128), np.uint16)
        cv2.ellipse(labels, (64, 48), (24, 14), 17, 0, 360, 1, -1)
        context = reviewed_measurement_context(None, np.zeros((96, 128, 3), np.uint8),
            np.eye(3), np.eye(3), transport_applied=True)
        transported = reproject_bundle(ReferenceRegionBundle(labels.shape,
            annotated_seeds=labels, seed_annotations=(self.summary.observations[0].annotation,),
            source_to_corrected=np.eye(3)), np.eye(3), labels.shape)
        summary = model.measure_reviewed_seed_instances(transported.annotated_seeds,
            transported.seed_annotations, pixels_per_mm=10, input_context=context)
        with patch.object(g, 'measure_shape_mask', side_effect=AssertionError('unexpected refit')):
            self.view._render_reviewed_shape_geometry(self.result, summary, uncertainty=True)
        tooltips = '\n'.join(item.toolTip() for item in self.view._overlay_items)
        self.assertIn('source image frame: corrected_image', tooltips)
        self.assertIn('coordinates: input_mask', tooltips)

    def check_test_model_overlay_explains_numeric_unavailability_and_retained_calibrated_evidence(self):
        mask = np.zeros((96, 128), np.uint8)
        cv2.ellipse(mask, (64, 48), (24, 14), 17, 0, 360, 1, -1)
        measurement = g.measure_shape_mask(mask, pixels_per_mm=1e-79)
        component = model._local_component((measurement,), 'flat')
        self.view._render_pose_shape_distributions(self.result,
            SimpleNamespace(families=(SimpleNamespace(pose='flat', component=component),)))
        tooltips = '\n'.join(item.toolTip() for item in self.view._overlay_items)
        self.assertIn('physical fit unavailable: numeric range', tooltips)
        self.assertIn('1.0 retained calibrated seed(s)', tooltips)

    def check_test_actual_model_pooling_failure_reaches_native_tooltip_without_refitting(self):
        from seedvision.reference_library.contracts import BiologicalContext, SpeciesDimensionsShapeBank
        component = model._local_component(
            (self.summary.observations[0].measurement,), 'flat')
        prior = replace(component, hierarchy_path=('lupinus_mutabilis',),
            covariance=tuple((0.,)*19 for _ in range(19)))
        assembled = model.assemble_seed_dimensions_shape_model(local_summary=self.summary,
            legacy_processing_diameter_px=87, source_mode='species_library_prior_and_current',
            biological_context=BiologicalContext('lupinus_mutabilis'),
            library_bank=SpeciesDimensionsShapeBank((), (prior,)), minimum_component_seeds=1)
        self.assertIs(assembled.family('flat').component, prior)
        with patch.object(model, '_local_component', side_effect=AssertionError('unexpected refit')):
            self.view._render_pose_shape_distributions(self.result, assembled)
        tooltips = '\n'.join(item.toolTip() for item in self.view._overlay_items)
        self.assertIn('local pooling unavailable: singular_precision', tooltips)
        self.assertIn('retained library component', tooltips)
        self.assertNotIn('local pooling applied', tooltips)

    def check_test_actual_conditioned_model_update_reaches_native_view_without_refitting(self):
        from seedvision.reference_library.contracts import BiologicalContext, SpeciesDimensionsShapeBank
        from tests.test_local_shape_units import measurements
        prior = replace(model._local_component(measurements(scale=4.), 'flat'),
                        hierarchy_path=('lupinus_mutabilis',))
        assembled = model.assemble_seed_dimensions_shape_model(local_summary=self.summary,
            legacy_processing_diameter_px=87, source_mode='species_library_prior_and_current',
            biological_context=BiologicalContext('lupinus_mutabilis'),
            library_bank=SpeciesDimensionsShapeBank((), (prior,)), minimum_component_seeds=1)
        family = assembled.family('flat')
        self.assertEqual(family.update_state, 'applied')
        self.assertEqual(family.pooling_method, 'coordinate_conditioned_precision')
        self.assertEqual(len(family.conditioning_scales), 19)
        self.assertEqual(family.source, 'library+local')
        self.assertTrue(assembled.in_sample_local_update)
        self.assertEqual(assembled.compatibility_seed_diameter_px, 87)
        with patch.object(model, '_local_component', side_effect=AssertionError('unexpected refit')):
            self.view._render_pose_shape_distributions(self.result, assembled)
        tooltips = '\n'.join(item.toolTip() for item in self.view._overlay_items)
        self.assertIn('local pooling applied', tooltips)
        self.assertNotIn('pooling unavailable', tooltips)

    def test_workflow_lifecycle(self):
        """5 related MeasurementDescriptionUITests scenarios, each with private fixture state."""
        self.scenario(self.check_test_body_span_and_point_tooltips_bind_definitions_units_and_frame_without_refitting)
        self.scenario(self.check_test_visibility_refresh_changes_metadata_without_changing_measurement_geometry)
        self.scenario(self.check_test_proven_corrected_frame_reaches_native_overlay_from_stored_bundle)
        self.scenario(self.check_test_model_overlay_explains_numeric_unavailability_and_retained_calibrated_evidence)
        self.scenario(self.check_test_actual_conditioned_model_update_reaches_native_view_without_refitting)

    def test_validation_and_recovery(self):
        """2 related MeasurementDescriptionUITests scenarios, each with private fixture state."""
        self.scenario(self.check_test_unavailable_legacy_and_boundary_errors_omit_bars_but_valid_zero_is_available)
        self.scenario(self.check_test_actual_model_pooling_failure_reaches_native_tooltip_without_refitting)
