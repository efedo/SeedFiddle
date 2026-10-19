
from tests.functional_units import functional_units
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch

from seedvision.cuda import plantcv_watershed_diagnostic as drawing

REFERENCE = Path(__file__).resolve().parents[1] / 'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
import importlib, json, sys
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import numpy as np
from plantcv import plantcv as pcv
assert version('plantcv') == '4.11.3'
assert version('numpy') == '2.2.6'
assert version('opencv-python') == '4.11.0.86'
assert version('matplotlib') == '3.11.2'
root = Path(pcv.__file__).parent
for name, digest in {
 'watershed.py':'d5fd90bf4e064117ff47ee333d2b83e62134901fda36f6966ae1b0be4a82a712',
 'color_palette.py':'915191e9c21896155b7cbec9edeb1103f2f4a319dca86b0d02e0337a5bfefac0',
 '_helpers.py':'c05b0c944210657332ce307f2066caab55f2ebc8d8fea11988db3e476e7cdfba',
 '_debug.py':'2f2363ce04653ac89b1ea6e5326dbba6848ff0284079f8a226d96af39b75b2f0',
 'classes.py':'8ee436f54c1bf10bdf6eaada1692205d61a209d65fb3687a636e87e97dc8b957',
}.items():
 assert sha256((root/name).read_bytes()).hexdigest() == digest
payload = json.load(sys.stdin)
output = []
module = importlib.import_module('plantcv.plantcv.watershed')
debug = importlib.import_module('plantcv.plantcv._debug')
original_debug = module._debug
for case in payload['cases']:
 np.random.seed(case['random_seed'])
 pcv.params.color_scale = case['color_scale']
 pcv.params.color_sequence = 'sequential'
 pcv.params.saved_color_scale = None
 pcv.params.device = 11
 if payload['mode'] == 'palette':
  pcv.params.color_sequence = 'random'
  output.append(pcv.color_palette(case['count']))
  continue
 labels = np.asarray(case['labels'],np.int32)
 bgr = np.asarray(case['rgb'],np.uint8)[...,::-1].copy()
 pcv.params.line_thickness = case['line_thickness']
 pcv.params.debug = case.get('debug',None)
 captured, side_effects = [], []
 def capture(visual,filename=None,**kwargs):
  captured.append(visual.copy())
  original_debug(visual,filename,**kwargs)
 module._debug = capture
 debug.print_image = lambda **kwargs: side_effects.append('print')
 debug.plot_image = lambda *args,**kwargs: side_effects.append('plot')
 # Keep the public drawing/return/observation/debug path intact, but inject the
 # label product so arbitrary forests test the exact source drawing block.
 module.watershed = lambda *args,**kwargs: labels
 returned = module.watershed_segmentation(bgr,(labels>0).astype(np.uint8)*255,
     distance=1,label='oracle')
 output.append({'image':captured[0][...,::-1].tolist(),
  'palette':pcv.params.saved_color_scale, 'device':pcv.params.device,
  'sequence':pcv.params.color_sequence, 'side_effects':side_effects,
  'returns_labels_only':returned is labels})
print(json.dumps(output))
'''


def oracle(mode, cases):
    result = subprocess.run([str(REFERENCE), '-c', SCRIPT],
        input=json.dumps({'mode': mode, 'cases': cases}), text=True,
        capture_output=True, timeout=120)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


def fixtures():
    labels = np.zeros((43,57),np.int32)
    labels[0:21,0:23] = 1
    labels[4:17,5:18] = 0
    labels[8:12,9:14] = 1  # Nested island inside a hole.
    labels[21:43,23:57] = 2
    labels[27:38,30:49] = 0
    labels[1:3,49:51] = 1  # Disconnected region with the same ID.
    tiny = np.zeros((9,13),np.int32)
    tiny[0,0],tiny[-1,-1],tiny[4,5] = 1,2,3
    diagonal = np.zeros((17,19),np.int32)
    for i in range(15):
        diagonal[i,i] = 1
        diagonal[i,i+1] = 2
    return (labels,tiny,diagonal,np.zeros((3,5),np.int32),
            np.asarray([[0,1,1,0,2,2,0]],np.int32))


class WatershedPaletteTests(unittest.TestCase):
    def test_exact_public_palette_sampling_shuffle_and_global_rng_preservation(self):
        cases = [dict(count=count,color_scale=scale,random_seed=seed)
                 for count,scale,seed in ((1,'gist_rainbow',0),(3,'gist_rainbow',9),
                    (129,'viridis',0xffffffff),(4097,'tab20',123),(7,'jet',411003))]
        expected = oracle('palette',cases)
        before = np.random.get_state()
        for case, colors in zip(cases,expected):
            palette, metadata = drawing.watershed_palette(**case)
            self.assertEqual(palette,tuple(tuple(c) for c in colors))
            self.assertEqual(metadata['palette_count'],case['count'])
            self.assertEqual(metadata['rng_profile'],drawing.RNG_PROFILE)
            self.assertEqual(metadata['palette_source_sha256'],drawing.PALETTE_SOURCE_SHA256)
            self.assertEqual(metadata['reference_numpy_version'],'2.2.6')
        after = np.random.get_state()
        self.assertEqual(before[0],after[0])
        np.testing.assert_array_equal(before[1],after[1])
        self.assertEqual(before[2:],after[2:])

    def test_invalid_controls_and_runtime_lut_binding(self):
        for values in ({'count':0},{'count':4098},{'count':True},{'count':2,'random_seed':True},
                       {'count':2,'random_seed':-1},{'count':2,'random_seed':2**32},
                       {'count':2,'color_scale':'no_such_palette'}):
            with self.assertRaises(drawing.WatershedDiagnosticError):
                drawing.watershed_palette(**values)
        identity = drawing.palette_runtime_identity()
        self.assertGreater(len(identity['registry']),100)
        self.assertEqual(identity['rng_profile'],drawing.RNG_PROFILE)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
@functional_units
class WatershedDiagnosticTests(unittest.TestCase):
    def check_test_long_contour_streams_exact_public_pixels_with_bounded_cpu_geometry(self):
        labels = np.zeros((10000,19),np.int32)
        labels[2:-2,5:14] = 1
        rgb = np.zeros((*labels.shape,3),np.uint8)
        case = dict(labels=labels.tolist(),rgb=rgb.tolist(),line_thickness=5,
                    color_scale='gist_rainbow',random_seed=0)
        expected = oracle('overlay',[case])[0]
        actual = drawing.render_watershed_diagnostic_rgb(torch.tensor(rgb,device='cuda'),
            torch.tensor(labels,device='cuda'))
        np.testing.assert_array_equal(actual.image.cpu().numpy(),expected['image'])
        self.assertGreater(actual.metadata['contour_points'],19000)
        self.assertLess(actual.metadata['max_cpu_geometry_estimate_bytes'],16*1024*1024)

    def check_test_exact_public_debug_pixels_nested_disconnected_diagonal_edges_and_thickness(self):
        cases = []
        for index, labels in enumerate(fixtures()):
            rgb = np.random.default_rng(index).integers(0,256,(*labels.shape,3),dtype=np.uint8)
            for thickness in (1,2,5,13):
                cases.append(dict(labels=labels.tolist(),rgb=rgb.tolist(),line_thickness=thickness,
                    color_scale='gist_rainbow' if index%2 else 'viridis',random_seed=411003+index,
                    debug=(None,'print','plot')[index%3]))
        expected = oracle('overlay',cases)
        for case, source in zip(cases,expected):
            rgb = torch.tensor(case['rgb'],device='cuda',dtype=torch.uint8)
            labels = torch.tensor(case['labels'],device='cuda',dtype=torch.int32)
            signatures = (rgb._version,labels._version)
            actual = drawing.render_watershed_diagnostic_rgb(rgb,labels,
                **{k:case[k] for k in ('line_thickness','color_scale','random_seed')})
            np.testing.assert_array_equal(actual.image.cpu().numpy(),source['image'])
            self.assertEqual(actual.metadata['palette_source_triples'],tuple(tuple(c) for c in source['palette']))
            self.assertEqual((rgb._version,labels._version),signatures)
            self.assertNotEqual(actual.image.data_ptr(),rgb.data_ptr())
            self.assertEqual(actual.image.device,rgb.device)
            self.assertTrue(source['returns_labels_only'])
            self.assertEqual(source['sequence'],'sequential')
            self.assertEqual(source['device'],12)
            self.assertEqual(source['side_effects'],[] if case['debug'] is None else [case['debug']])
            with self.assertRaises(TypeError):
                actual.metadata['line_thickness'] = 99

    def check_test_bounding_boxes_preserve_global_frame_and_avoid_per_label_full_scans(self):
        labels = torch.zeros((300,400),dtype=torch.int32,device='cuda')
        labels[77:86,95:105] = 1
        labels[199:207,301:314] = 2
        image = torch.zeros((300,400,3),dtype=torch.uint8,device='cuda')
        original = drawing.contours.trace_binary_contours_cuda
        shapes = []
        def trace(mask,**kwargs):
            shapes.append(tuple(mask.shape))
            return original(mask,**kwargs)
        with patch.object(drawing.contours,'trace_binary_contours_cuda',side_effect=trace):
            result = drawing.render_watershed_diagnostic_rgb(image,labels)
        self.assertEqual(shapes,[(9,10),(8,13)])
        self.assertEqual(result.metadata['roi_pixels'],194)
        self.assertEqual(result.metadata['scan_work'],16*120000+32*194)
        self.assertEqual(result.metadata['label_ids'],(1,2))

    def check_test_invalid_inputs_sparse_source_palette_and_preallocation_resource_caps(self):
        image = torch.zeros((9,13,3),dtype=torch.uint8,device='cuda')
        labels = torch.zeros((9,13),dtype=torch.int32,device='cuda');labels[4,5]=1
        for a,b,options in ((image.cpu(),labels,{}),(image,labels.long(),{}),
                (image,labels[:3],{}),(image,labels,{'line_thickness':True}),
                (image.transpose(0,1),labels.T,{})):
            with self.assertRaises(drawing.WatershedDiagnosticError):
                drawing.render_watershed_diagnostic_rgb(a,b,**options)
        for value in (-1,65537):
            bad = labels.clone();bad[4,5]=value
            with self.assertRaises(drawing.WatershedDiagnosticError):
                drawing.render_watershed_diagnostic_rgb(image,bad)
        for bad in (torch.full_like(labels,1),labels*2):
            with self.assertRaises(drawing.WatershedDiagnosticUnsupported):
                drawing.render_watershed_diagnostic_rgb(image,bad)
        with patch.object(drawing,'MAX_SCAN_WORK',1), patch.object(torch.Tensor,'clone',
                side_effect=AssertionError('canvas allocated before scan admission')):
            with self.assertRaises(drawing.WatershedDiagnosticResourceError):
                drawing.render_watershed_diagnostic_rgb(image,labels)
        with patch.object(drawing,'MAX_POINTS',0):
            with self.assertRaises(drawing.WatershedDiagnosticResourceError):
                drawing.render_watershed_diagnostic_rgb(image,labels)

    def check_test_cancellation_and_input_mutation_withhold_private_canvas(self):
        labels = torch.tensor(fixtures()[0],device='cuda',dtype=torch.int32)
        image = torch.zeros((*labels.shape,3),device='cuda',dtype=torch.uint8)
        for stop in (1,4,12,40):
            visits = [0]
            def cancel():
                visits[0] += 1
                return visits[0] >= stop
            with self.assertRaises(drawing.WatershedDiagnosticCancelled):
                drawing.render_watershed_diagnostic_rgb(image,labels,cancel=cancel)
            self.assertEqual(image._version,0)
            self.assertEqual(labels._version,0)
        for value in (image,labels):
            visits = [0]
            def mutate():
                visits[0] += 1
                if visits[0] == 4:
                    value.add_(1)
                return False
            with self.assertRaisesRegex(drawing.WatershedDiagnosticError,'changed'):
                drawing.render_watershed_diagnostic_rgb(image,labels,cancel=mutate)
            if value is image:
                image.zero_()
            else:
                labels.sub_(1)

    def check_test_shared_topology_work_and_cpu_geometry_admission_withhold_the_canvas(self):
        labels = torch.tensor(fixtures()[0],device='cuda',dtype=torch.int32)
        image = torch.zeros((*labels.shape,3),device='cuda',dtype=torch.uint8)
        original = drawing.contours.trace_binary_contours_cuda
        visits = []
        def consume(mask,**kwargs):
            budget = kwargs['work_budget']
            visits.append(budget[0])
            result = original(mask,**kwargs)
            budget[0] = drawing.size.MAX_CONTOUR_WORK_UNITS-1
            return result
        with patch.object(drawing.contours,'trace_binary_contours_cuda',side_effect=consume):
            with self.assertRaises(drawing.WatershedDiagnosticResourceError):
                drawing.render_watershed_diagnostic_rgb(image,labels)
        self.assertEqual(visits,[0,drawing.size.MAX_CONTOUR_WORK_UNITS-1])
        with patch.object(drawing,'MAX_CPU_GEOMETRY_BYTES',1), patch.object(drawing.raster,
                'iter_adjacent_closed_polyline_spans') as expand:
            with self.assertRaises(drawing.WatershedDiagnosticResourceError):
                drawing.render_watershed_diagnostic_rgb(image,labels)
            expand.assert_not_called()
        with patch.object(drawing,'MAX_GEOMETRY_WORK',1):
            with self.assertRaises(drawing.WatershedDiagnosticResourceError):
                drawing.render_watershed_diagnostic_rgb(image,labels)
        self.assertEqual((image._version,labels._version),(0,0))

    def check_test_existing_contour_api_shared_budget_accumulates_and_rejects_invalid_before_scan(self):
        mask = (torch.tensor(fixtures()[0],device='cuda')==1).to(torch.uint8)*255
        budget = [0]
        first = drawing.contours.trace_binary_contours_cuda(mask,work_budget=budget)
        used = budget[0]
        self.assertGreater(used,0)
        second = drawing.contours.trace_binary_contours_cuda(mask,work_budget=budget)
        self.assertEqual(budget[0],2*used)
        self.assertEqual([c.tolist() for c in first.contours],[c.tolist() for c in second.contours])
        for invalid in ([],[-1],[True],[drawing.size.MAX_CONTOUR_WORK_UNITS+1],(0,)):
            with patch.object(drawing.size,'_validate_mask') as scan:
                with self.assertRaises(drawing.contours.BinaryContourResourceError):
                    drawing.contours.trace_binary_contours_cuda(mask,work_budget=invalid)
                scan.assert_not_called()

    def check_test_strict_callback_finally_checks_and_owned_canvas_data_and_storage(self):
        labels = torch.tensor(fixtures()[0],device='cuda',dtype=torch.int32)
        image = torch.zeros((*labels.shape,3),device='cuda',dtype=torch.uint8)
        for value in (None,0,1,[],np.bool_(False)):
            with self.assertRaisesRegex(drawing.WatershedDiagnosticError,'exactly bool'):
                drawing.render_watershed_diagnostic_rgb(image,labels,cancel=lambda:value)
        def bad_callback():
            image.add_(1)
            raise RuntimeError('callback failure')
        with self.assertRaisesRegex(drawing.WatershedDiagnosticError,'input changed'):
            drawing.render_watershed_diagnostic_rgb(image,labels,cancel=bad_callback)
        image.zero_()
        visits = [0]
        def count():
            visits[0] += 1
            return False
        drawing.render_watershed_diagnostic_rgb(image,labels,cancel=count)
        total = visits[0]
        original_clone = torch.Tensor.clone
        for storage_change in (False,True):
            canvases,visits = [],[0]
            def capture(value,*args,**kwargs):
                result = original_clone(value,*args,**kwargs)
                if value is image:
                    canvases.append(result)
                return result
            def mutate_final():
                visits[0] += 1
                if visits[0] == total:
                    canvas = canvases[0]
                    if storage_change:
                        canvas.untyped_storage().resize_(canvas.untyped_storage().nbytes()+512)
                    else:
                        canvas.add_(1)
                return False
            with patch.object(torch.Tensor,'clone',capture):
                with self.assertRaisesRegex(drawing.WatershedDiagnosticError,'canvas changed'):
                    drawing.render_watershed_diagnostic_rgb(image,labels,cancel=mutate_final)

    def test_workflow_lifecycle(self):
        """4 related WatershedDiagnosticTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_long_contour_streams_exact_public_pixels_with_bounded_cpu_geometry)
        self.scenario(self.check_test_exact_public_debug_pixels_nested_disconnected_diagonal_edges_and_thickness)
        self.scenario(self.check_test_bounding_boxes_preserve_global_frame_and_avoid_per_label_full_scans)
        self.scenario(self.check_test_strict_callback_finally_checks_and_owned_canvas_data_and_storage)

    def test_validation_and_recovery(self):
        """4 related WatershedDiagnosticTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_invalid_inputs_sparse_source_palette_and_preallocation_resource_caps)
        self.scenario(self.check_test_cancellation_and_input_mutation_withhold_private_canvas)
        self.scenario(self.check_test_shared_topology_work_and_cpu_geometry_admission_withhold_the_canvas)
        self.scenario(self.check_test_existing_contour_api_shared_budget_accumulates_and_rejects_invalid_before_scan)
