"""Pure recovery contracts, maximum inventory and allocation-free admission."""
from collections.abc import Mapping
from unittest.mock import patch
import unittest

from seedvision.benchmarking import native_recovery_replay as contracts
from seedvision.benchmarking.checkpoints import CoordinateFrame, _freeze_json

FRAME=CoordinateFrame('source.decoded',(17,19),'top-left','x-right,y-down','pixel')
BINDING=_freeze_json({'coordinate_frame':FRAME.to_dict(),'source_sha256':'1'*64},name='test source binding')


def encode(value):
    if isinstance(value,Mapping):return {'kind':'mapping','items':[[key,encode(item)] for key,item in value.items()]}
    if type(value) in (tuple,list):return {'kind':'tuple' if type(value) is tuple else 'list','items':[encode(item) for item in value]}
    return value


class RecoveryReplayContractTests(unittest.TestCase):
    def test_primitive_decoder_preserves_types_and_exact_byte_depth_item_caps(self):
        value={'tuple':(1,'é\n'), 'list':[True,None]}
        self.assertEqual(contracts.decode(encode(value)),contracts.canonical(value))
        self.assertIsInstance(contracts.decode(encode(value))['tuple'],tuple)
        self.assertIsInstance(contracts.decode(encode(value))['list'],list)
        with patch.object(contracts,'MAX_METADATA_BYTES',6):
            self.assertEqual(contracts.canonical('é\n'),'é\n')
        for attr,limit,value in (('MAX_METADATA_BYTES',5,'é\n'),('MAX_METADATA_DEPTH',0,(1,)),
                ('MAX_METADATA_ITEMS',1,(1,))):
            with patch.object(contracts,attr,limit),self.assertRaises(contracts.NativeRecoveryReplayError):
                contracts.canonical(value)
        cycle=[];cycle.append(cycle)
        for value in (cycle,float('nan'),1<<256,object(),'\ud800'):
            with self.assertRaises(contracts.NativeRecoveryReplayError):contracts.canonical(value)
        for value in ({'kind':'dataclass','items':[]},{'kind':'mapping','items':[['x',1],['x',2]]}):
            with self.assertRaises(contracts.NativeRecoveryReplayError):contracts.decode(value)

    def test_inventory_maximum_sparse_ids_and_downstream_limit_are_distinct(self):
        frame=CoordinateFrame('source.decoded',(1,65535),'top-left','x-right,y-down','pixel')
        binding=_freeze_json({'coordinate_frame':frame.to_dict(),'source_sha256':'1'*64},name='maximum test binding')
        value={'connectivity':4,'core_pixels':32768,
            'objects':tuple({'label_id':i,'original_core_pixels':1} for i in range(1,65536,2))}
        decoded=contracts.decode(encode(value))
        contracts.validate(decoded,'inventory',frame,binding,{})
        self.assertEqual(len(decoded['objects']),32768)
        self.assertEqual(contracts.compare(decoded,value)['mismatched_elements'],0)
        with self.assertRaises(contracts.native.core.CoreRecoveryResourceError):
            contracts.native.graph_metadata_bound(8065)
        changed={**value,'objects':value['objects']+({'label_id':65536,'original_core_pixels':1},)}
        with self.assertRaises(contracts.NativeRecoveryReplayError):
            contracts.validate(changed,'inventory',frame,binding,{})

    def test_malformed_inventory_types_order_and_totals(self):
        value={'connectivity':4,'core_pixels':3,'objects':(
            {'label_id':2,'original_core_pixels':1},{'label_id':17,'original_core_pixels':2})}
        contracts.validate(value,'inventory',FRAME,BINDING,{})
        for changed in ({**value,'connectivity':True},{**value,'core_pixels':True},
                {**value,'objects':list(value['objects'])},{**value,'core_pixels':4},
                {**value,'objects':tuple(reversed(value['objects']))},
                {**value,'objects':({'label_id':True,'original_core_pixels':3},)}):
            with self.assertRaises(contracts.NativeRecoveryReplayError):
                contracts.validate(changed,'inventory',FRAME,BINDING,{})
        changed={**value,'objects':({'label_id':2,'original_core_pixels':2},{'label_id':17,'original_core_pixels':1})}
        comparison=contracts.compare(value,changed)
        self.assertEqual(comparison['mismatched_elements'],2)
        self.assertEqual(comparison['compared_elements'],6)

    def test_source_roi_controls_and_metadata_bind_exact_full_frame(self):
        contracts.validate(BINDING,'source_binding',FRAME,BINDING,{})
        for bounds in (None,[0,0,19,17],[0,0,1,1],[18,16,19,17]):
            parameters={'validity_roi':None if bounds is None else {'shape':[17,19],'bounds':bounds}}
            selected=contracts.bounds(FRAME.shape,parameters)
            contracts.validate({'binding':BINDING,'bounds_xyxy':selected},'validity_metadata',FRAME,BINDING,parameters)
        for roi in ({'shape':[17,19],'bounds':[0,0,0,1]}, {'shape':[17,19],'bounds':[0,0,20,17]},
                {'shape':[16,19],'bounds':[0,0,1,1]}, {'shape':[17,19],'bounds':[False,0,1,1]}):
            with self.assertRaises(contracts.NativeRecoveryReplayError):contracts.bounds(FRAME.shape,{'validity_roi':roi})
        with self.assertRaises(contracts.NativeRecoveryReplayError):
            contracts.validate({**BINDING,'source_sha256':'2'*64},'source_binding',FRAME,BINDING,{})

    def test_source_and_replay_admission_before_any_cuda_allocation(self):
        for operation,parameters,bpp in (('recovery.core_labels',{},5),
                ('recovery.validity',{'validity_roi':None},2),('recovery.clip_cores',{},9)):
            plan=contracts.plan(operation,FRAME.shape,parameters,selected_bytes=323*bpp)
            self.assertGreater(plan['cuda_bytes'],323*(bpp+1))
            for kwargs in ({'selected_limit':323*bpp-1},{'cuda_limit':plan['cuda_bytes']-1}):
                with self.assertRaises(contracts.NativeRecoveryReplayUnavailable):
                    contracts.plan(operation,FRAME.shape,parameters,selected_bytes=323*bpp,**kwargs)
        pixels=1
        # Find the actual source envelope boundary, including its work cap.
        lo,hi=1,contracts.native.MAX_RASTER_PIXELS
        while lo<hi:
            middle=(lo+hi+1)//2
            try:contracts.native.retained_bound(middle)
            except contracts.native.core.CoreRecoveryResourceError:hi=middle-1
            else:lo=middle
        pixels=lo
        contracts.plan('recovery.core_labels',(1,pixels),{},selected_bytes=5*pixels)
        with self.assertRaises(contracts.NativeRecoveryReplayUnavailable):
            contracts.plan('recovery.core_labels',(1,pixels+1),{},selected_bytes=5*(pixels+1))

    def test_clipping_controls_and_exact_small_allocation_boundary(self):
        shape=(1,1)
        plan=contracts.plan('recovery.clip_cores',shape,{},selected_bytes=9)
        compare=contracts._rounded(1)+contracts._rounded(8)+contracts.ALLOCATION_BYTES
        self.assertEqual(plan['cuda_bytes'],4*512+compare+contracts.AUDIT_CUDA_BYTES)
        self.assertEqual(plan['primitive_scratch_bytes'],0)
        self.assertEqual(plan['primitive_cpu_metadata_bytes'],0)
        self.assertEqual(plan['queue_admission'],'no queue required')
        contracts.plan('recovery.clip_cores',shape,{},selected_bytes=9,
            selected_limit=9,cuda_limit=plan['cuda_bytes'])
        for parameters in ({'validity_roi':None},{'relabel':True},None):
            with self.assertRaises(contracts.NativeRecoveryReplayError):
                contracts.plan('recovery.clip_cores',shape,parameters,selected_bytes=9)
        with self.assertRaises(contracts.NativeRecoveryReplayUnavailable):
            contracts.plan('recovery.clip_cores',(1,contracts.native.MAX_RASTER_PIXELS+1),{},selected_bytes=0)

    def test_remaining_plans_deduplicate_rounding_and_ignore_archived_domain_claims(self):
        pixels=323;partition=(pixels,4*pixels,pixels,4*pixels,4*pixels,pixels,pixels,pixels,8)
        for operation,parameters,selected,inputs in (
                ('recovery.partition',{},partition,partition[:4]),
                ('recovery.annotation',{'uncertainty_radius':8,'measurement_policy':'flag_uncertain'},(*partition,pixels),partition),
                ('recovery.measurements',{},(),())):
            kwargs={'selected_bytes':sum(selected),'selected_allocations':selected,'input_allocations':inputs,
                'label_count':2,'inventory_count':2}
            plan=contracts.plan(operation,FRAME.shape,parameters,**kwargs)
            self.assertGreaterEqual(plan['cuda_bytes'],sum(contracts._rounded(size) for size in selected)+contracts.AUDIT_CUDA_BYTES)
            contracts.plan(operation,FRAME.shape,parameters,**kwargs,cuda_limit=plan['cuda_bytes'])
            with self.assertRaises(contracts.NativeRecoveryReplayUnavailable):
                contracts.plan(operation,FRAME.shape,parameters,**kwargs,cuda_limit=plan['cuda_bytes']-1)
            with self.assertRaises(contracts.NativeRecoveryReplayUnavailable):
                contracts.plan(operation,FRAME.shape,parameters,**{**kwargs,'inventory_count':8065})
        kwargs={'selected_bytes':sum(partition),'selected_allocations':partition,'input_allocations':partition[:4]}
        plans=[contracts.plan('recovery.partition',FRAME.shape,{},**kwargs,label_count=count,inventory_count=2) for count in (0,2,32768)]
        self.assertEqual(plans[0]['cuda_bytes'],plans[-1]['cuda_bytes'])
        self.assertIn('worst-case',plans[0]['queue_admission'])
        with patch.object(contracts.native.core._watershed,'MAX_WORKSPACE_BYTES',16*pixels+15),self.assertRaises(contracts.NativeRecoveryReplayUnavailable):
            contracts.plan('recovery.partition',FRAME.shape,{},**kwargs,inventory_count=2)

    def test_maximum_downstream_measurement_metadata_fits_bounded_decoder(self):
        frame=CoordinateFrame('source.decoded',(1,8064),'top-left','x-right,y-down','pixel')
        value={'core_'+str(label):{name:{'value':1,'available':True,'trait':name.replace('_',' '),
            'method':'seedfiddle.experimental.core_recovery','scale':'pixel_squared' if name=='assigned_area' else 'pixel',
            'datatype':"<class 'int'>",'reasons':(),'scientific_validation':False} for name in contracts.TRAITS}
            for label in range(1,8065)}
        decoded=contracts.decode(encode(value));contracts.validate(decoded,'observations',frame,None,{})
        self.assertEqual(len(decoded),8064)
        self.assertEqual(contracts.compare(value,decoded)['mismatched_elements'],0)
        with patch.object(contracts,'MAX_METADATA_BYTES',1024*1024),self.assertRaises(contracts.NativeRecoveryReplayError):
            contracts.decode(encode(value))


if __name__=='__main__':unittest.main()
