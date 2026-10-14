import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.cuda import plantcv_card_detection as card
from seedvision.cuda.plantcv_card_preprocess import ColorCardPreprocessPrecisionError
from seedvision.cuda.plantcv_contours import BinaryContourResourceError

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'artifacts/plantcv-reference-env/Scripts/python.exe'
SCRIPT = r'''
import hashlib, importlib, json, sys
from pathlib import Path
from importlib.metadata import version
import cv2,numpy as np
from plantcv import plantcv as pcv
assert version('plantcv')=='4.11.3' and version('numpy')=='2.2.6'
assert version('opencv-python')=='4.11.0.86' and cv2.__version__=='4.11.0'
module=importlib.import_module('plantcv.plantcv.transform.detect_color_card')
assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()=='e3ad4028e8a831e523815e33ee057f08ba73f5a3f60b3933f11504dd71b7bb49'
auto=importlib.import_module('plantcv.plantcv.transform.auto_correct_color')
assert hashlib.sha256(Path(auto.__file__).read_bytes()).hexdigest()=='bed2eb9e50e42e39f7ad4bbef157150190585268178f06083092edb19ffaf57a'
payload=json.load(sys.stdin)
image=np.load(sys.argv[1]);pcv.params.debug=None;pcv.outputs.clear()
kwargs=payload['kwargs'];bounds=kwargs.pop('roi_bounds_xyxy',None)
roi=None
if bounds is not None:
    x0,y0,x1,y1=bounds
    if payload.get('rectangle'):
        roi=pcv.roi.rectangle(img=image,x=x0,y=y0,w=x1-x0+1,h=y1-y0+1)
    else:
        roi=pcv.Objects(contours=[[np.array([[[x0,y0]],[[x0,y1]],[[x1,y1]],[[x1,y0]]],dtype=np.int32)]],
            hierarchy=[np.array([[[-1,-1,-1,-1]]],dtype=np.int32)])
else: x0,y0,x1,y1=0,0,image.shape[1],image.shape[0]
centers=[];transforms=[];corners=[]
original_transform=cv2.transform
original_perspective=cv2.getPerspectiveTransform
def transform(*args,**kw):
    result=original_transform(*args,**kw);centers.append(result[0][:,:2].copy());return result
def perspective(*args,**kw):
    result=original_perspective(*args,**kw)
    corners.append(args[1].copy());transforms.append(result.copy());return result
cv2.transform=transform;cv2.getPerspectiveTransform=perspective
if payload['operation']=='auto':
    corrected=pcv.transform.auto_correct_color(image,roi=roi,**kwargs)
mask=pcv.transform.detect_color_card(image,roi=roi,**kwargs)
metadata={k:float(v['value'][0]) for k,v in pcv.outputs.metadata.items()}
scale=[pcv.params.px_width,pcv.params.px_height,pcv.params.unit]
options={k:v for k,v in kwargs.items() if k!='color_chip_size'}
local=image[y0:y1,x0:x1]
bbox=np.zeros(image.shape[:2],dtype=np.uint8)
bbox[y0:y1,x0:x1]=pcv.transform.mask_color_card(local,**options)
raw=module._color_card_detection(local,**options)
result={'mask':mask,'bbox':bbox,'centers':centers[0], 'corners':corners[0],
    'transformation':transforms[0],'areas':np.asarray(raw[2]),'heights':np.asarray(raw[3]),'widths':np.asarray(raw[4])}
if payload['operation']=='auto':
    result['corrected']=corrected
    result['source']=pcv.transform.get_color_matrix(image,mask)[1]
    result['target']=pcv.transform.std_color_matrix(pos=3)
np.savez(sys.argv[2],**result)
print(json.dumps({'metadata':metadata,'scale':scale}))
'''


def fixture(*, varied=False, orientation=0):
    image=np.full((620,460,3),220,dtype=np.uint8)
    palette=((20,20,20),(40,40,40),(20,50,30),(60,20,40),(30,30,70),(60,30,20))
    for j in range(6):
        for i in range(4):
            x,y=100+i*80,100+j*80
            image[y-25:y+26,x-25:x+26]=palette[(j*4+i)%len(palette)] if varied else 40
    cv2.circle(image,(100,100),5,(255,255,255),-1)
    return np.ascontiguousarray(np.rot90(image,orientation))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class AutomaticCardTests(unittest.TestCase):
    def oracle(self,image,operation='detect',rectangle=False,**kwargs):
        with tempfile.TemporaryDirectory(prefix='card-oracle-',dir=ROOT/'artifacts') as temporary:
            source=Path(temporary)/'image.npy';output=Path(temporary)/'result.npz'
            np.save(source,image)
            result=subprocess.run([str(REFERENCE),'-c',SCRIPT,str(source),str(output)],
                input=json.dumps({'operation':operation,'kwargs':kwargs,'rectangle':rectangle}),
                text=True,capture_output=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stderr)
            metadata=json.loads(result.stdout)
            with np.load(output) as values:
                arrays={k:values[k].copy() for k in values.files}
        return arrays,metadata

    def compare(self,image,**kwargs):
        original,metadata=self.oracle(image,**kwargs)
        source=torch.as_tensor(image,device='cuda');version=source._version
        result=card.detect_color_card_cuda(source,**kwargs)
        self.assertEqual(result.labeled_mask.dtype,torch.float64)
        self.assertEqual(result.bounding_mask.dtype,torch.uint8)
        self.assertEqual(result.labeled_mask.device,source.device)
        self.assertEqual(result.reference_sources,card.REFERENCE_SOURCES)
        self.assertEqual(result.grid_box_sha256,card.GRID_BOX_SHA256)
        np.testing.assert_array_equal(result.labeled_mask.cpu().numpy(),original['mask'])
        np.testing.assert_array_equal(result.bounding_mask.cpu().numpy(),original['bbox'])
        geometry=result.geometry
        for name,expected in [('centers_xy','centers'),('corners_xy','corners'),
                ('transformation','transformation'),('areas','areas'),('widths','widths'),('heights','heights')]:
            np.testing.assert_array_equal(getattr(geometry,name),original[expected])
        for name,value in metadata['metadata'].items():
            self.assertEqual(getattr(geometry,name),value)
        if kwargs.get('color_chip_size') is not None:
            self.assertEqual([geometry.px_width,geometry.px_height,geometry.unit],metadata['scale'])
        for array in (geometry.centers_xy,geometry.corners_xy,geometry.transformation):
            with self.assertRaises(ValueError): array.setflags(write=True)
        self.assertEqual(source._version,version)
        np.testing.assert_array_equal(source.cpu().numpy(),image)
        return result

    def test_actual_complete_public_detector_and_bbox_both_methods_default_domain(self):
        image=fixture()
        for method in (0,1):
            self.compare(image,adaptive_method=method)
        source=torch.as_tensor(image,device='cuda')
        expected,_=self.oracle(image)
        np.testing.assert_array_equal(card.mask_color_card_cuda(source).cpu().numpy(),expected['bbox'])

    def test_card_rotations_white_corner_ties_and_scaling_named_and_custom(self):
        for orientation,name in enumerate(('nano','passport','cameratrax','classic')):
            self.compare(fixture(orientation=orientation),adaptive_method=0,color_chip_size=name)
        self.compare(fixture(),adaptive_method=0,color_chip_size=(7,11))
        # With identical corner pixels argmin uses the first source box corner;
        # its order matters even when the rectangle describes the same shape.
        tied=fixture();tied[95:106,95:106]=40
        self.compare(tied,adaptive_method=0)
        transform=cv2.getRotationMatrix2D((230,310),13,1)
        transform[:,2]+=np.array([170,90])
        rotated=cv2.warpAffine(fixture(),transform,(800,800),flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,borderValue=(220,220,220))
        self.compare(rotated,adaptive_method=0)

    def test_roi_half_open_rectangle_mapping_and_full_frame_edges(self):
        image=fixture()
        for bounds in ((40,40,420,580),(40,40,460,580),(40,40,420,620),(40,40,460,620),(0,0,460,620)):
            result=self.compare(image,adaptive_method=0,roi_bounds_xyxy=bounds)
            self.assertEqual(result.roi_bounds_xyxy,bounds)
        expected,_=self.oracle(image,adaptive_method=0,roi_bounds_xyxy=(40,40,420,580),rectangle=True)
        actual=card.detect_color_card_cuda(torch.as_tensor(image,device='cuda'),adaptive_method=0,
            roi_bounds_xyxy=(40,40,420,580))
        np.testing.assert_array_equal(actual.labeled_mask.cpu().numpy(),expected['mask'])
        self.compare(image,adaptive_method=0,radius=100,roi_bounds_xyxy=(40,40,420,580))

    def test_effective_radius_filters_and_block_size_preserve_strict_source_boundaries(self):
        image=fixture();source=torch.as_tensor(image,device='cuda')
        small=self.compare(image,adaptive_method=0,radius=0,block_size=35)
        overlap=self.compare(image,adaptive_method=0,radius=70,block_size=35)
        self.assertEqual(int((small.labeled_mask!=0).sum()),24)
        self.assertGreater(int((overlap.labeled_mask!=0).sum()),24)
        ordinary=card.detect_color_card_cuda(source,adaptive_method=0)
        self.assertNotEqual(small.geometry.median_color_chip_size,ordinary.geometry.median_color_chip_size)
        for changed in ({'min_size':10000},{'aspect_ratio':1},{'solidity':1}):
            with self.assertRaises(card.AutomaticCardNotFound):
                card.detect_color_card_cuda(source,adaptive_method=0,**changed)
        self.compare(image,adaptive_method=0,aspect_ratio=1/1.27)

    def test_actual_public_auto_affine_correction_and_compact_matrices(self):
        image=fixture(varied=True)
        original,metadata=self.oracle(image,operation='auto',adaptive_method=0,color_chip_size='mini')
        source=torch.as_tensor(image,device='cuda');version=source._version
        actual=card.auto_correct_color_cuda(source,adaptive_method=0,color_chip_size='mini')
        np.testing.assert_array_equal(actual.detection.labeled_mask.cpu().numpy(),original['mask'])
        np.testing.assert_allclose(actual.correction.source_matrix,original['source'],atol=2e-15,rtol=0)
        np.testing.assert_array_equal(actual.correction.target_matrix,original['target'])
        difference=np.abs(actual.correction.corrected_bgr.cpu().numpy().astype(int)-original['corrected'].astype(int))
        self.assertLessEqual(int(difference.max()),1)
        for array in (actual.correction.source_matrix,actual.correction.target_matrix,
                      actual.correction.transformation_matrix):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError): array.setflags(write=True)
        self.assertEqual(actual.correction.fit,'affine')
        self.assertEqual(source._version,version)

    def test_no_card_numeric_uncertainty_and_topology_resources_remain_distinct(self):
        source=torch.full((31,37,3),40,dtype=torch.uint8,device='cuda')
        with self.assertRaises(card.AutomaticCardNotFound):card.detect_color_card_cuda(source)
        uncertain=fixture();uncertain[uncertain==40]=60
        with self.assertRaises(ColorCardPreprocessPrecisionError):
            card.detect_color_card_cuda(torch.as_tensor(uncertain,device='cuda'))
        source=torch.as_tensor(fixture(),device='cuda')
        with patch.object(card,'trace_binary_contours_cuda',side_effect=BinaryContourResourceError('bound')):
            with self.assertRaises(BinaryContourResourceError):card.detect_color_card_cuda(source,adaptive_method=0)

    def test_missing_interior_chip_keeps_source_24_grid_labels_and_23_contours(self):
        image=fixture();image[235:286,155:206]=220
        result=self.compare(image,adaptive_method=0)
        self.assertEqual(len(result.geometry.selected_contour_indices),23)
        self.assertEqual(len(result.geometry.centers_xy),24)
        self.assertEqual(set(result.labeled_mask.unique().cpu().tolist()),set(range(0,241,10)))

    def test_reusable_filter_and_final_mask_mutation_and_grid_metadata_tamper(self):
        source=torch.as_tensor(fixture(),device='cuda')
        prepare=card.preprocessing.preprocess_color_card_cuda
        captured=[]
        def remember(*args,**kwargs):
            result=prepare(*args,**kwargs);captured.append(result);return result
        def change_filter():
            if captured: captured[0].gray[0,0]=219
            return False
        with patch.object(card.preprocessing,'preprocess_color_card_cuda',side_effect=remember):
            with self.assertRaisesRegex(card.AutomaticCardError,'changed'):
                card.detect_color_card_cuda(source,adaptive_method=0,cancel=change_filter)
        calls=0
        def count():
            nonlocal calls
            calls+=1
            return False
        card.detect_color_card_cuda(source,adaptive_method=0,cancel=count)
        final=calls;calls=0;captured=[]
        zeros=torch.zeros
        def remember_mask(*args,**kwargs):
            result=zeros(*args,**kwargs)
            if result.shape==source.shape[:2] and result.dtype==torch.float64:captured.append(result)
            return result
        def change_mask():
            nonlocal calls
            calls+=1
            if calls==final: captured[0][0,0]=10
            return False
        with patch.object(card.torch,'zeros',side_effect=remember_mask):
            with self.assertRaisesRegex(card.AutomaticCardError,'changed'):
                card.detect_color_card_cuda(source,adaptive_method=0,cancel=change_mask)
        with patch.object(card,'GRID_BOX_411',card.GRID_BOX_411.copy()):
            with self.assertRaisesRegex(card.AutomaticCardError,'Fixed grid'):
                card.detect_color_card_cuda(source,adaptive_method=0)

    def test_invalid_roi_storage_work_and_circle_budget_reject_before_filter_allocation(self):
        source=torch.as_tensor(fixture(),device='cuda')
        invalid=[{'roi_bounds_xyxy':bounds} for bounds in
            ((0,0,0,620),(0,0,461,620),(-1,0,400,600),(0,0,400,621),(0,0,False,600),(0,0,2.0,3))]
        invalid += [{'color_chip_size':'unknown'},{'color_chip_size':(0,1)}, {'solidity':float('nan')},
            {'aspect_ratio':0},{'radius':True},{'radius':-1}]
        for options in invalid:
            with patch.object(card.preprocessing,'preprocess_color_card_cuda') as prepare:
                with self.assertRaises(ValueError): card.detect_color_card_cuda(source,**options)
                prepare.assert_not_called()
        for operation in (card.detect_color_card_cuda,card.auto_correct_color_cuda):
            with patch.object(card,'MAX_RETAINED_BYTES',1),patch.object(card.preprocessing,'preprocess_color_card_cuda') as prepare:
                with self.assertRaises(card.AutomaticCardResourceError):operation(source)
                prepare.assert_not_called()
        with patch.object(card,'MAX_CIRCLE_WORK',1):
            with self.assertRaises(card.AutomaticCardResourceError):card.detect_color_card_cuda(source)
        with patch.object(card,'MAX_GEOMETRY_WORK',1):
            with self.assertRaises(card.AutomaticCardResourceError):card.detect_color_card_cuda(source,adaptive_method=0)

    def test_cancel_and_mutation_first_middle_final_callbacks_withhold_whole_result(self):
        source=torch.as_tensor(fixture(),device='cuda')
        with self.assertRaises(card.AutomaticCardCancelled):
            card.detect_color_card_cuda(source,cancel=lambda:True)
        for operation in (card.detect_color_card_cuda,card.auto_correct_color_cuda):
            calls=0
            def count():
                nonlocal calls
                calls+=1
                return False
            operation(source,adaptive_method=0,cancel=count)
            final=calls
            for target in (1,final//2,final):
                calls=0
                def mutate():
                    nonlocal calls
                    calls+=1
                    if calls==target: source[0,0,0]=219
                    return False
                with self.assertRaisesRegex(card.AutomaticCardError,'changed'):
                    operation(source,adaptive_method=0,cancel=mutate)
                source[0,0,0]=220

    def test_actual_multi_tile_custom_stream_retained_and_scratch_memory(self):
        source=torch.as_tensor(fixture(),device='cuda')
        stream=torch.cuda.Stream(device=source.device);torch.cuda.synchronize()
        baseline=torch.cuda.memory_allocated(source.device);torch.cuda.reset_peak_memory_stats(source.device)
        with torch.cuda.stream(stream):actual=card.auto_correct_color_cuda(source,adaptive_method=0)
        stream.synchronize()
        self.assertEqual(actual.correction.corrected_bgr.device,source.device)
        self.assertLessEqual(torch.cuda.max_memory_allocated(source.device)-baseline,
                            card.MAX_RETAINED_BYTES+256*1024*1024)


if __name__=='__main__':unittest.main()
