"""Numeric scale availability is separate from biological/physical validity."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.measurement import geometry as g
from seedvision.measurement.shape_model import (
    measure_reviewed_seed_instances, shape_observations_from_summary,
)
from seedvision.persistence.reference_regions import SeedInstanceAnnotation
from seedvision.reference_library.contracts import BiologicalContext


def mask():
    value=np.zeros((96,128),np.uint8)
    cv2.ellipse(value,(64,48),(24,14),0,0,360,1,-1)
    return value


def reviewed(seed_id):
    return SeedInstanceAnnotation(seed_id,shape_reviewed=True,outline_visibility='complete',pose='flat')


class MeasurementScaleSafetyTests(unittest.TestCase):
    def test_invalid_scales_reject_before_raster_measurement(self):
        invalid=(True,False,np.bool_(True),0.,-1.,np.nan,np.inf,-np.inf,'2',[],object(),10**1000)
        for value in invalid:
            with self.subTest(value=type(value).__name__), \
                    patch.object(g,'image_to_tensor',side_effect=AssertionError('unexpected CUDA upload')), \
                    patch.object(g,'_largest_contour',side_effect=AssertionError('unexpected contour fit')):
                for measure in (g.measure_mask,g.measure_shape_mask):
                    with self.assertRaisesRegex(ValueError,'finite positive real'):
                        measure(mask(),pixels_per_mm=value)

    def test_ordinary_shape_conversion_preserves_exact_dimensional_arithmetic(self):
        px=g.measure_shape_mask(mask(),calibration_relative_uncertainty=.02)
        mm=g.measure_shape_mask(mask(),pixels_per_mm=2.5,calibration_relative_uncertainty=.02)
        self.assertEqual(mm.maximum_span,px.maximum_span/2.5)
        self.assertEqual(mm.ellipse.body_length,px.ellipse.body_length/2.5)
        self.assertEqual(mm.ellipse.body_width,px.ellipse.body_width/2.5)
        self.assertEqual(mm.projected_area,px.projected_area/(2.5*2.5))
        self.assertEqual(mm.measurement_uncertainty,tuple(v/2.5 if i in (0,1,2)
            else v/(2.5*2.5) if i==4 else v for i,v in enumerate(px.measurement_uncertainty)))
        self.assertEqual(mm.maximum_span_endpoints,px.maximum_span_endpoints)
        self.assertEqual(mm.ellipse.center_xy,px.ellipse.center_xy)
        self.assertEqual(mm.contour_signature,px.contour_signature)
        self.assertEqual(mm.ovality,px.ovality)

    def test_area_intermediate_overflow_or_underflow_uses_representable_sequential_result(self):
        self.assertEqual(g._scaled_dimension(1e308,1e160,area=True),(1e308/1e160)/1e160)
        self.assertEqual(g._scaled_dimension(1e-320,1e-200,area=True),(1e-320/1e-200)/1e-200)
        self.assertEqual(g._scaled_dimension(0.,1e-200,area=True),0.)

    def test_nonfinite_or_underflowed_final_dimensions_and_uncertainties_are_withheld(self):
        for value,scale,area in ((1.,1e-308,True),(1.,1e308,True),(1e-300,1e100,False)):
            with self.subTest(scale=scale,area=area),self.assertRaisesRegex(ValueError,'not representable'):
                g._scaled_dimension(value,scale,area=area)
        px=g.measure_shape_mask(mask(),perturbation_radii=())
        errors=list(px.measurement_uncertainty); errors[0]=1e-300
        with self.assertRaisesRegex(ValueError,'Measurement uncertainty'):
            g._scale_shape_measurement(replace(px,measurement_uncertainty=tuple(errors)),1e50)
        for measure in (g.measure_shape_mask,g.measure_mask):
            with self.assertRaisesRegex(ValueError,'not representable'):
                measure(mask(),pixels_per_mm=1e308)

    def test_subnormal_area_and_tiny_width_keep_pixel_ovality_and_dimensionless_features(self):
        px=g.measure_shape_mask(mask(),perturbation_radii=())
        for scale in (1e12,1e160):
            mm=g._scale_shape_measurement(px,scale)
            self.assertGreater(mm.ellipse.body_width,0)
            self.assertLess(mm.ellipse.body_width,1e-9)
            self.assertEqual(mm.ovality,px.ovality)
            for index in (3,*range(5,19)):
                self.assertEqual(mm.vector()[index],px.vector()[index])
            self.assertTrue(all(math.isfinite(v) for v in mm.vector()))
        self.assertGreater(mm.projected_area,0)
        self.assertLess(mm.projected_area,np.finfo(float).tiny)

    def test_invalid_or_missing_scale_preserves_pixel_summaries_even_when_empty(self):
        labels=mask().astype(np.uint16)
        expected=measure_reviewed_seed_instances(labels,(reviewed(1),))
        for scale in (None,True,0,np.nan,np.inf,'bad'):
            with self.subTest(scale=scale):
                for empty in (None,np.zeros((8,8),np.uint16)):
                    summary=measure_reviewed_seed_instances(empty,(),pixels_per_mm=scale)
                    self.assertEqual((summary.length_unit,summary.calibrated_count,summary.calibration_state),('px',0,'unavailable'))
                    self.assertTrue(summary.calibration_reason)
                summary=measure_reviewed_seed_instances(labels,(reviewed(1),),pixels_per_mm=scale)
                self.assertEqual((summary.eligible_count,summary.calibrated_count,summary.length_unit),(1,0,'px'))
                np.testing.assert_array_equal(summary.vectors,expected.vectors)
                self.assertEqual(summary.size_statistics,expected.size_statistics)
                self.assertIs(summary.observations[0].measurement,summary.observations[0].pixel_measurement)
                self.assertTrue(summary.observations[0].calibration_reason)

    def test_all_valid_summary_retains_mm_working_units_and_actual_calibrated_count(self):
        summary=measure_reviewed_seed_instances(mask().astype(np.uint16),(reviewed(1),),pixels_per_mm=2.5)
        self.assertEqual((summary.length_unit,summary.calibrated_count,summary.calibration_state),('mm',1,'available'))
        item=summary.observations[0]
        self.assertIs(item.measurement,item.calibrated_measurement)
        self.assertEqual(item.pixel_measurement.length_unit,'px')
        self.assertEqual(item.measurement,g._scale_shape_measurement(item.pixel_measurement,2.5))

    def test_mixed_per_instance_availability_never_mixes_summary_or_model_units(self):
        labels=np.zeros((96,128),np.uint16)
        cv2.circle(labels,(24,48),5,1,-1)
        cv2.circle(labels,(88,48),25,2,-1)
        summary=measure_reviewed_seed_instances(labels,(reviewed(1),reviewed(2)),pixels_per_mm=3e-153)
        self.assertEqual((summary.eligible_count,summary.calibrated_count,summary.length_unit,summary.calibration_state),(2,1,'px','partial'))
        a,b=summary.observations
        self.assertIsNotNone(a.calibrated_measurement)
        self.assertIsNone(b.calibrated_measurement)
        self.assertTrue(b.calibration_reason)
        self.assertTrue(all(x.measurement.length_unit=='px' for x in summary.observations))
        np.testing.assert_array_equal(summary.vectors,[x.pixel_measurement.vector() for x in summary.observations])
        self.assertEqual(summary.size_statistics[0],np.mean([x.pixel_measurement.maximum_span for x in summary.observations]))
        observations=shape_observations_from_summary(summary,source_index=0,biological_context=BiologicalContext('lupinus_mutabilis'))
        self.assertEqual([x.calibrated for x in observations],[True,False])
        self.assertEqual(observations[0].measurement,a.calibrated_measurement.vector())
        self.assertEqual(observations[1].measurement,b.pixel_measurement.vector())

    def test_aggregate_overflow_or_underflow_is_explicitly_unavailable(self):
        from seedvision.measurement.shape_model import SeedMeasurementSummary,ReviewedSeedShape
        px=g.measure_shape_mask(mask(),perturbation_radii=())
        for large in (1e200,1e-200):
            errors=list(px.measurement_uncertainty); errors[0]=large
            measurement=replace(px,maximum_span=large,measurement_uncertainty=tuple(errors))
            summary=SeedMeasurementSummary((ReviewedSeedShape(1,reviewed(1),measurement,True),),1,0,'px')
            self.assertIsNone(summary.size_statistics)
            self.assertTrue(summary.size_statistics_unavailable_reason)

    def test_conversion_rejects_double_scaling_or_incomplete_uncertainty_bundles(self):
        px=g.measure_shape_mask(mask(),perturbation_radii=())
        mm=g._scale_shape_measurement(px,2.5)
        with self.assertRaisesRegex(ValueError,'pixel measurement bundle'):
            g._scale_shape_measurement(mm,2.5)
        for errors in ((),(0.,)*18,(*px.measurement_uncertainty[:-1],np.nan),
                (*px.measurement_uncertainty[:-1],-1.),(*px.measurement_uncertainty[:-1],True)):
            with self.subTest(errors=errors),self.assertRaisesRegex(ValueError,'complete finite'):
                g._scale_shape_measurement(replace(px,measurement_uncertainty=errors),2.5)

    def test_legacy_default_aspect_and_malformed_overrides_have_explicit_behavior(self):
        from dataclasses import asdict
        import json
        px=g.measure_shape_mask(mask(),perturbation_radii=())
        values=asdict(px); values.pop('body_aspect_ratio')
        values.pop('provenance')
        values['ellipse']=px.ellipse
        values['local_feature']=px.local_feature
        restored=g.SeedShapeMeasurement(**values)
        self.assertEqual(restored.vector(),px.vector())
        self.assertIsNone(restored.body_aspect_ratio)
        for ratio in (True,0.,-1.,np.nan,np.inf,'2',10**1000):
            with self.subTest(ratio=ratio),self.assertRaisesRegex(ValueError,'Body aspect ratio'):
                replace(px,body_aspect_ratio=ratio)
        object.__setattr__(restored,'body_aspect_ratio',np.inf)
        with self.assertRaisesRegex(ValueError,'Body aspect ratio'):
            restored.vector()
        mm=g._scale_shape_measurement(px,2.5)
        self.assertEqual(json.loads(json.dumps(asdict(mm),allow_nan=False))['body_aspect_ratio'],px.ovality)

    def test_mixed_summary_population_fit_uses_actual_calibrated_and_dimensionless_groups(self):
        from dataclasses import asdict
        import json
        from seedvision.measurement.shape_model import fit_species_dimensions_shape_bank
        labels=np.zeros((96,128),np.uint16)
        cv2.circle(labels,(24,48),5,1,-1)
        cv2.circle(labels,(88,48),25,2,-1)
        convert=g._scale_shape_measurement
        def selected_conversion(measurement,scale):
            if measurement.projected_area>1000:
                raise ValueError('Declared second-instance conversion failure.')
            return convert(measurement,scale)
        with patch('seedvision.measurement.shape_model._scale_shape_measurement',side_effect=selected_conversion):
            summary=measure_reviewed_seed_instances(labels,(reviewed(1),reviewed(2)),pixels_per_mm=2.5)
        self.assertEqual(summary.length_unit,'px')
        context=BiologicalContext('lupinus_mutabilis')
        observations=shape_observations_from_summary(summary,source_index=0,biological_context=context)
        bank=fit_species_dimensions_shape_bank(observations,minimum_component_seeds=1)
        component,_=bank.select_component(context,'flat')
        self.assertEqual(component.calibrated_physical_seed_count,1)
        self.assertEqual(component.dimensionless_physical_seed_count,2)
        self.assertTrue(np.isfinite(component.mean).all())
        saved=json.loads(json.dumps(asdict(summary),allow_nan=False))
        self.assertEqual(saved['length_unit'],'px')
        self.assertEqual(saved['calibrated_count'],1)
        self.assertEqual(saved['observations'][0]['calibrated_measurement']['length_unit'],'mm')
        self.assertIsNone(saved['observations'][1]['calibrated_measurement'])
        self.assertTrue(saved['observations'][1]['calibration_reason'])
