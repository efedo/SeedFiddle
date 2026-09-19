from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.learning.contracts import FeatureStackSpec
from seedvision.learning.data import export_learning_sample, LearningManifest, file_sha256, audit_manifest, SeedTileDataset


class CriticalIntegrityTests(unittest.TestCase):
    def test_checkpoint_cache_uses_content_and_result_rejects_replaced_weights(self):
        import os
        from seedvision.learning.pipeline import cached_checkpoint
        from seedvision.export.results import result_report
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/'weights.pt';checkpoint.write_bytes(b'old weights')
            stamp = checkpoint.stat().st_mtime_ns
            digest = file_sha256(checkpoint)
            with patch('seedvision.learning.pipeline.load_checkpoint',side_effect=lambda *a,**k:(object(),object(),{})) as loader:
                cached_checkpoint(root,'weights.pt',device='cpu')
                checkpoint.write_bytes(b'new weights');os.utime(checkpoint,ns=(stamp,stamp))
                cached_checkpoint(root,'weights.pt',device='cpu')
                self.assertEqual(loader.call_count,2)
            analysis = SimpleNamespace(image_path=None,crop_offset=(0,0),warnings=(),
                calibration=SimpleNamespace(affine_matrix=np.eye(3),corrected_bgr=np.zeros((8,8,3),np.uint8)),
                unet_instances=SimpleNamespace(checkpoint_id=str(checkpoint)+'#sha256='+digest))
            with self.assertRaisesRegex(ValueError,'Checkpoint changed'):
                result_report(analysis,enabled={'unet_instances'})

    def test_streamed_evaluation_both_families_and_hybrid(self):
        import json
        from seedvision.learning.models import MultiHeadSeedUNet, SeedStarDist2D
        from seedvision.learning.checkpoint import save_checkpoint
        from seedvision.learning.evaluation import evaluate_checkpoint
        from seedvision.learning.hybrid_evaluation import evaluate_hybrid_checkpoints
        spec = FeatureStackSpec(include_species_planes=False)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root/'manifest.json'
            self.sample(manifest,split='test')
            checkpoints = []
            for family,cls in (('unet_watershed',MultiHeadSeedUNet),('stardist',SeedStarDist2D)):
                path = root/(family+'.pt')
                save_checkpoint(path,cls(spec.input_channels,base_channels=8,depth=2),family=family,feature_spec=spec)
                report = evaluate_checkpoint(path,manifest,root/family,device='cpu',tile_size=64,overlap=8)
                self.assertFalse(report['scientifically_validated'])
                self.assertEqual(report['sample_count'],1)
                json.dumps(report,allow_nan=False)
                checkpoints.append(path)
            report = evaluate_hybrid_checkpoints(*checkpoints,manifest,root/'hybrid',split='test',device='cpu',tile_size=64,overlap=8)
            self.assertFalse(report['scientifically_validated'])
            json.dumps(report,allow_nan=False)

    def test_inference_cancellation_and_spool_budget(self):
        import torch
        from seedvision.learning.inference import tiled_predict
        from seedvision.learning.resources import PredictionSpool
        from seedvision.timing import AnalysisCancelled
        with self.assertRaises(AnalysisCancelled):
            tiled_predict(torch.nn.Identity(),torch.zeros(1,1,8,8),tile_size=64,overlap=8,cancellation_requested=lambda:True)
        spool = PredictionSpool(1)
        root = spool.root
        try:
            with self.assertRaises(MemoryError):
                spool.append({'tensor':torch.zeros(1)})
        finally:
            spool.close()
        self.assertFalse(root.exists())

    def test_clipped_reprojection_withholds_complete_shape_and_landmark(self):
        from seedvision.annotation.coordinates import reproject_bundle
        from seedvision.persistence.reference_regions import ReferenceRegionBundle, SeedInstanceAnnotation
        labels = np.zeros((8,8),np.uint16)
        labels[2:6,2:6] = 1
        annotation = SeedInstanceAnnotation(seed_id=1,shape_reviewed=True,outline_visibility='complete',hilum_point=(5.,4.))
        bundle = ReferenceRegionBundle(shape=labels.shape,annotated_seeds=labels,
            seed_annotations=(annotation,),source_to_corrected=np.eye(3))
        shifted = reproject_bundle(bundle,np.array([[1,0,4],[0,1,0],[0,0,1]],float),labels.shape)
        self.assertFalse(shifted.seed_annotations[0].shape_reviewed)
        self.assertEqual(shifted.seed_annotations[0].outline_visibility,'image_cutoff')
        self.assertIsNone(shifted.seed_annotations[0].hilum_point)

    def test_custom_pickle_checkpoint_is_rejected_without_execution(self):
        import torch
        import pickle
        from seedvision.learning.checkpoint import load_checkpoint
        class Untrusted:
            def __reduce__(self):
                return (eval,('1 + 2',))
        with TemporaryDirectory() as directory:
            path = Path(directory)/'untrusted.pt'
            torch.save({'unexpected':Untrusted()},path)
            with self.assertRaisesRegex(pickle.UnpicklingError,'Weights only load failed'):
                load_checkpoint(path)

    def test_decoder_selection_rejects_held_out_group_overlap(self):
        import json
        from dataclasses import asdict
        from seedvision.learning.evaluation_protocol import audit_decoder_source
        from seedvision.learning.decode import UNetWatershedSettings
        settings = UNetWatershedSettings()
        identity = {'samples':[{'group':'held-out','components':{'features':'same'}}]}
        with TemporaryDirectory() as directory:
            path = Path(directory)/'decoder.json'
            path.write_text(json.dumps({'decoder_settings':asdict(settings),'split':'validation','dataset_identity':identity}))
            with self.assertRaisesRegex(ValueError,'overlaps'):
                audit_decoder_source(path,settings,identity,independent=True)

    def test_reference_transport_same_canvas_translation_and_rotation(self):
        from seedvision.annotation.coordinates import reproject_raster
        labels = np.zeros((8,8),np.uint16)
        labels[2,3] = 37
        shift = np.array([[1,0,2],[0,1,1],[0,0,1]],float)
        shifted = reproject_raster(labels,np.eye(3),shift,labels.shape)
        self.assertEqual(shifted[3,5],37)
        rotation = np.array([[0,-1,7],[1,0,0],[0,0,1]],float)
        rotated = reproject_raster(labels,np.eye(3),rotation,(10,12))
        self.assertEqual(rotated[3,5],37)
        np.testing.assert_array_equal(reproject_raster(shifted,shift,np.eye(3),labels.shape),labels)

    def test_legacy_reference_bundle_requires_alignment_review(self):
        from seedvision.annotation.coordinates import reproject_bundle
        from seedvision.persistence.reference_regions import ReferenceRegionBundle
        with self.assertRaisesRegex(ValueError,'Legacy'):
            reproject_bundle(ReferenceRegionBundle(shape=(8,8)),np.eye(3),(8,8))

    def test_saved_reference_transform_round_trip(self):
        from seedvision.persistence.reference_regions import ReferenceRegionStore, ReferenceRegionBundle
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'image.png'
            cv2.imwrite(str(source),np.zeros((8,8,3),np.uint8))
            transform = np.array([[1,0,2],[0,1,1],[0,0,1]],float)
            store = ReferenceRegionStore(root)
            store.save(source,ReferenceRegionBundle(shape=(8,8),source_to_corrected=transform))
            np.testing.assert_array_equal(store.load_if_present(source,None).source_to_corrected,transform)

    def test_results_distinguish_missing_evidence_from_empty_region(self):
        from seedvision.export.results import result_report
        instances = SimpleNamespace(labels=np.zeros((8,8),np.int32),instance_confidences=np.empty(0))
        analysis = SimpleNamespace(image_path=None,calibration=SimpleNamespace(affine_matrix=np.eye(3),corrected_bgr=np.zeros((8,8,3),np.uint8)),
            crop_offset=(0,0),layers=SimpleNamespace(foreground_colour_profile=None),warnings=(),procedural_instances=instances)
        report = result_report(analysis)
        self.assertEqual(report['state'],'unavailable')
        self.assertIsNone(report['proposed_count'])
        analysis.layers.foreground_colour_profile = object()
        self.assertEqual(result_report(analysis)['proposed_count'],0)

    def test_legacy_ambiguous_foreground_recipe_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Re-export'):
            FeatureStackSpec(version=1)
        self.assertEqual(FeatureStackSpec(version=1,channels=('lab_l',)).version,1)

    def sample(self, path, **overrides):
        spec = FeatureStackSpec(include_species_planes=False)
        labels = np.zeros((32,32),np.uint16)
        labels[8:20,8:20] = 1
        arguments = dict(dataset_id='regression',feature_spec=spec,identifier='a',
            features=np.zeros((spec.input_channels,32,32),np.float32),labels=labels,
            image_bgr=None,species='soybean',group='capture-a',reviewed=True,
            annotation_author='Test reviewer')
        arguments.update(overrides)
        return export_learning_sample(path,**arguments)

    def test_failed_component_replacement_preserves_complete_old_revision(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            original = self.sample(path)
            before = path.read_bytes()
            digest = file_sha256(Path(directory)/original.features)
            with patch('seedvision.learning.data.write_label_image',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    self.sample(path,replace_existing=True,features=np.ones((13,32,32),np.float32))
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(file_sha256(Path(directory)/original.features),digest)
            self.assertTrue(audit_manifest(path)['valid'])

    def test_failed_manifest_switch_preserves_old_revision(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            self.sample(path)
            before = path.read_bytes()
            original_replace = Path.replace
            def fail_manifest(source, destination):
                if Path(destination) == path:
                    raise OSError('rename failed')
                return original_replace(source,destination)
            with patch.object(Path,'replace',fail_manifest):
                with self.assertRaises(OSError):
                    self.sample(path,replace_existing=True)
            self.assertEqual(path.read_bytes(),before)
            self.assertTrue(audit_manifest(path)['valid'])

    def test_duplicate_content_rejected_before_model_load(self):
        from seedvision.learning.evaluation import evaluate_checkpoint
        with TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            sample = self.sample(path)
            manifest = LearningManifest.load(path)
            replace(manifest,samples=(sample,replace(sample,identifier='alias',group='other',split='test'))).save(path)
            with patch('seedvision.learning.evaluation.load_checkpoint') as load:
                with self.assertRaisesRegex(ValueError,'audit failed'):
                    evaluate_checkpoint(Path(directory)/'model.pt',path,Path(directory)/'evaluation',device='cpu')
            load.assert_not_called()

    def test_component_edit_is_detected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            sample = self.sample(path)
            np.savez_compressed(Path(directory)/sample.features,features=np.ones((13,32,32),np.float16))
            audit = audit_manifest(path)
            self.assertFalse(audit['valid'])
            self.assertTrue(any('fingerprinted' in value for value in audit['errors']))

    def test_dataset_cache_does_not_retain_an_oversized_sample(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            self.sample(path)
            dataset = SeedTileDataset(path,split='train',family='unet_watershed',tile_size=32,sample_cache_bytes=1)
            self.assertEqual(dataset[0]['features'].shape[-2:],(32,32))
            self.assertEqual(len(dataset._cache),0)

    def test_raw_cache_detects_equal_size_source_replacement(self):
        from seedvision.segmentation.baseline import analyze_path, PipelineAnalysisCache
        with TemporaryDirectory() as directory:
            path = Path(directory)/'source.bmp'
            cache = PipelineAnalysisCache()
            with patch('seedvision.segmentation.baseline.analyze_image',side_effect=lambda image,**kwargs: float(image.mean())):
                cv2.imwrite(str(path),np.full((16,16,3),10,np.uint8))
                self.assertEqual(analyze_path(path,node_cache=cache),10.)
                cv2.imwrite(str(path),np.full((16,16,3),240,np.uint8))
                self.assertEqual(analyze_path(path,node_cache=cache),240.)

    def test_export_and_inference_share_raw_foreground_semantics(self):
        from seedvision.learning.features import pipeline_evidence
        from seedvision.learning.export import analysis_evidence
        raw,resolved = object(),object()
        layers = SimpleNamespace(foreground_noise_likelihood=None,background_likelihood=None,
            refined_background_likelihood=None,edge_likelihood=None)
        advanced = SimpleNamespace(rasters={name:None for name in ('sensor_noise','flattened_grayscale','shadow_likelihood','highlight_likelihood')})
        result = SimpleNamespace(foreground_colour_probability=raw,foreground_probability=resolved,layers=layers,advanced=advanced)
        self.assertEqual(analysis_evidence(result),pipeline_evidence(raw,layers,advanced))
        self.assertIs(analysis_evidence(result)['foreground_colour'],raw)

    def test_global_matching_maximizes_cardinality_at_low_threshold(self):
        from seedvision.learning.metrics import evaluate_instances
        # Contingency [[intersection 40, 30], [30, 0]]: greedy selects .4
        # and misses the second match; global matching selects both .3 pairs.
        truth = np.array([1]*70+[2]*30,np.uint16).reshape(10,10)
        predicted = np.array([1]*40+[2]*30+[1]*30,np.uint16).reshape(10,10)
        metrics = evaluate_instances(truth,predicted,iou_threshold=.29)
        self.assertEqual(metrics.true_positives,2)

    def test_restricted_checkpoints_round_trip_both_families(self):
        from seedvision.learning.models import MultiHeadSeedUNet, SeedStarDist2D
        from seedvision.learning.checkpoint import save_checkpoint,load_checkpoint
        spec = FeatureStackSpec(include_species_planes=False)
        with TemporaryDirectory() as directory:
            for family,cls in (('unet_watershed',MultiHeadSeedUNet),('stardist',SeedStarDist2D)):
                model = cls(spec.input_channels,base_channels=8,depth=2)
                path = Path(directory)/(family+'.pt')
                save_checkpoint(path,model,family=family,feature_spec=spec,training_metadata={'seed':4})
                restored,returned,payload = load_checkpoint(path)
                self.assertEqual(returned,spec)
                self.assertEqual(payload['training_metadata'],{'seed':4})
                self.assertEqual(set(restored.state_dict()),set(model.state_dict()))


if __name__ == '__main__':
    unittest.main()

