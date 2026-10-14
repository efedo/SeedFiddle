"""Actual relocated native results, artifact facts and private review history."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from seedvision.export import artifact_identity as artifacts
from seedvision.export.review import lookup_result_review
from seedvision.export.results import result_report
from seedvision.persistence import ProjectAnalysisStore, analysis_settings_profile_from_graph
from seedvision.persistence.snapshot import create_snapshot, restore_snapshot
import tests.test_authoritative_parity as parity_fixture
import tests.test_result_prediction_revision as report_fixture


class ArtifactIdentityTests(unittest.TestCase):
    def capture(self, root, selection, *, enabled=(), checkpoints=None):
        return artifacts.capture_artifact_selections(root,
            selections={'unet_instances': selection}, enabled=enabled, checkpoints=checkpoints)

    def test_equivalent_verified_locations_keep_paths_but_share_content_identity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = (root / 'first.pt', root / 'second.pt')
            for path in paths:
                path.write_bytes(b'opaque disabled checkpoint bytes')
            reports = []
            for path in paths:
                analysis = report_fixture._analysis()
                analysis.artifact_selections_json = self.capture(root, str(path))
                settings = {'nodes': [{'id': 'unet_instances', 'parameters': {'checkpoint_path': str(path)}}],
                            'unrelated_path': 'caller-owned/path'}
                reports.append(result_report(analysis, settings=settings, provenance={'opaque': 'a/path'}))
            self.assertEqual(reports[0]['identity'], reports[1]['identity'])
            self.assertEqual(reports[0]['result_revision'], reports[1]['result_revision'])
            self.assertNotEqual(reports[0]['native_artifact_execution'], reports[1]['native_artifact_execution'])
            self.assertEqual(reports[0]['identity']['settings']['unrelated_path'], 'caller-owned/path')
            self.assertEqual(reports[0]['identity']['provenance'], {'opaque': 'a/path'})
            state = reports[0]['identity']['native_artifact_selections']['unet_instances']
            self.assertEqual(state['state'], 'bytes_recorded_format_unproven')
            self.assertFalse(state['used'])
            rebuilt = result_report(analysis, settings=reports[1]['identity']['settings'], provenance={'opaque': 'a/path'})
            self.assertEqual(rebuilt['result_revision'], reports[1]['result_revision'])
            self.assertEqual(rebuilt['native_artifact_execution'], reports[1]['native_artifact_execution'])

    def test_changed_disappeared_and_newly_available_selections_reject_stale_capture(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'weights.pt'
            path.write_bytes(b'original')
            captured = self.capture(root, str(path))
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed after analysis'):
                artifacts.verified_artifact_facts(captured)
            path.unlink()
            with self.assertRaises(ValueError):
                artifacts.verified_artifact_facts(captured)
            absent = self.capture(root, str(path))
            path.write_bytes(b'appeared')
            with self.assertRaises(ValueError):
                artifacts.verified_artifact_facts(absent)

    def test_actual_checkpoint_correspondence_and_legacy_opaque_ids_remain_distinct(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'weights.pt'
            path.write_bytes(b'weights')
            digest = sha256(path.read_bytes()).hexdigest()
            producer = str(path.resolve()) + '#sha256=' + digest
            captured = self.capture(root, str(path), enabled={'unet_instances'},
                                    checkpoints={'unet_instances': producer})
            facts = artifacts.verified_artifact_facts(captured)
            token = artifacts.semantic_checkpoint(producer, 'unet_instances', facts)
            self.assertEqual(token['source_sha256'], digest)
            self.assertEqual(token['state'], 'used_checkpoint_bytes_verified')
            with self.assertRaisesRegex(ValueError, 'changed during inference'):
                self.capture(root, str(path), enabled={'unet_instances'},
                             checkpoints={'unet_instances': str(path) + '#sha256=' + 'a' * 64})
            with self.assertRaises(ValueError):
                artifacts.semantic_checkpoint('other-producer', 'unet_instances', facts)
            legacy = artifacts.verified_artifact_facts(self.capture(root, str(path),
                enabled={'unet_instances'}, checkpoints={'unet_instances': 'legacy opaque ID'}))
            self.assertEqual(artifacts.semantic_checkpoint('legacy opaque ID', 'unet_instances', legacy),
                             'legacy opaque ID')
            self.assertEqual(artifacts.semantic_artifact(legacy['unet_instances'], 'unet_instances')
                             ['opaque_producer_id'], 'legacy opaque ID')

    def test_forged_canonical_settings_and_malformed_captures_do_not_become_verified(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'weights.pt'
            path.write_bytes(b'weights')
            captured = self.capture(root, str(path))
            facts = artifacts.verified_artifact_facts(captured)
            token = artifacts.semantic_artifact(facts['unet_instances'], 'unet_instances')
            token['source_sha256'] = 'a' * 64
            with self.assertRaisesRegex(ValueError, 'captured selection'):
                artifacts.semantic_settings({'nodes': [{'id': 'unet_instances',
                    'parameters': {'checkpoint_path': token}}]}, facts)
            for payload in ('[]', '{}', None, json.dumps({'contract': artifacts.CONTRACT, 'records':
                    {'unet_instances': {'enabled': True}}})):
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    artifacts.verified_artifact_facts(payload)
            broken = json.loads(captured)
            broken['records']['unet_instances']['used'] = True
            with self.assertRaises(ValueError):
                artifacts.verified_artifact_facts(json.dumps(broken))

    def test_missing_invalid_and_none_trait_states_are_explicit_and_stale_changes_reject(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'trait.json'
            def capture(selected):
                return artifacts.capture_artifact_selections(root,
                    selections={'trait_classifier': selected}, enabled=())
            missing = capture(str(path))
            absent = artifacts.verified_artifact_facts(capture('None'))['trait_classifier']
            self.assertEqual(absent['state'], 'no_frozen_model')
            self.assertNotIn('opaque_selection', artifacts.semantic_artifact(absent, 'trait_classifier'))
            path.write_bytes(b'{}')
            invalid = artifacts.verified_artifact_facts(capture(str(path)))['trait_classifier']
            self.assertEqual(invalid['state'], 'invalid_model')
            self.assertEqual(invalid['source_sha256'], sha256(b'{}').hexdigest())
            with self.assertRaises(ValueError):
                artifacts.verified_artifact_facts(missing)


class PrivateReviewLookupTests(unittest.TestCase):
    def test_only_absent_primary_uses_project_private_review_without_writing_shared_history(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / 'restored-projects' / 'private' / 'working.seedfiddle-project.json'
            revision = 'a' * 64
            private = project.parent / 'result-reviews' / (revision + '.json')
            private.parent.mkdir(parents=True)
            raw = json.dumps({'result_revision': revision, 'decisions': {'1': 'accepted'}}).encode()
            private.write_bytes(raw)
            self.assertEqual(lookup_result_review(root, revision, project_path=project)[1], 'matched')
            self.assertFalse((root / 'result-reviews').exists())
            primary = root / 'result-reviews' / private.name
            primary.parent.mkdir()
            primary.write_bytes(b'{')
            self.assertEqual(lookup_result_review(root, revision, project_path=project), (None, 'malformed'))
            primary.write_text(json.dumps({'result_revision': 'b' * 64, 'decisions': {}}))
            self.assertEqual(lookup_result_review(root, revision, project_path=project), (None, 'incompatible'))
            primary.write_text(json.dumps({'result_revision': revision, 'decisions': {'1': 'excluded'}}))
            self.assertEqual(lookup_result_review(root, revision, project_path=project)[0]['decisions'],
                             {'1': 'excluded'})
            self.assertEqual(private.read_bytes(), raw)
            with patch('pathlib.Path.open', side_effect=PermissionError('denied')):
                with self.assertRaises(PermissionError):
                    lookup_result_review(root, revision, project_path=project)


@unittest.skipUnless(torch.cuda.is_available(), 'Actual portable native parity requires CUDA')
class PortableAuthoritativeTests(unittest.TestCase):
    # Share setup and route helpers, without inheriting or collecting prior tests.
    setUpClass = classmethod(parity_fixture.AuthoritativeParityTests.setUpClass.__func__)
    setUp = parity_fixture.AuthoritativeParityTests.setUp
    save_project = parity_fixture.AuthoritativeParityTests.save_project
    install_profile = parity_fixture.AuthoritativeParityTests.install_profile
    decode = staticmethod(parity_fixture.AuthoritativeParityTests.decode)
    paired = parity_fixture.AuthoritativeParityTests.paired
    save_desktop_review = parity_fixture.AuthoritativeParityTests.save_desktop_review

    def transport(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def relocate(self, base, snapshot):
        self.window.close()
        original = self.root.resolve()
        self.assertTrue(original.is_relative_to(base.parent.resolve()))
        self.assertNotEqual(original, base.resolve())
        shutil.rmtree(original)
        self.root = base / 'restored'
        (self.root / 'config').mkdir(parents=True)
        shutil.copyfile(Path(__file__).parents[1] / 'config' / 'traits.json', self.root / 'config' / 'traits.json')
        self.master = restore_snapshot(snapshot, self.root)
        loaded = ProjectAnalysisStore(self.root).load(self.master)
        self.assertFalse(loaded.issues)
        self.source = loaded.images[0].path
        self.key = str(self.source.resolve()).casefold()
        from seedvision.ui.main_window import MainWindow
        self.window = MainWindow(self.root)
        self.addCleanup(self.window.close)
        self.window._current_image_key = lambda: self.key
        self.install_profile()

    def test_actual_relocation_preserves_complete_identity_predictions_crops_and_normal_reviews(self):
        before, _ = self.paired()
        self.save_desktop_review()
        reviewed = self.window._current_report(self.result)
        old_review = self.window._review_path(before).read_bytes()
        from seedvision.export.crop_artifacts import export_seed_crop_artifacts
        crops = export_seed_crop_artifacts(self.root / 'crops', self.result, reviewed,
                                           seed_ids=(1, 2), padding_px=2)
        old_crops = json.loads((crops / 'seed-crops.json').read_text())
        base = self.transport()
        snapshot = create_snapshot(self.root, self.master, base / 'snapshots')
        self.relocate(base, snapshot)
        after, batch = self.paired()
        self.assertEqual(after['identity'], before['identity'])
        self.assertEqual(after['result_revision'], before['result_revision'])
        self.assertEqual(after['instances'], reviewed['instances'])
        self.assertEqual(after['reviewed_measurements'], before['reviewed_measurements'])
        self.assertEqual(after['identity']['crop_offset'], [22, 6])
        self.assertEqual(batch['review_lookup'], 'matched')
        self.assertEqual(after['trait_classifier']['predictions'], before['trait_classifier']['predictions'])
        self.assertNotEqual(after['trait_classifier']['model_path'], before['trait_classifier']['model_path'])
        self.assertNotEqual(after['native_artifact_execution'], before['native_artifact_execution'])
        self.assertNotIn('model_path', after['identity']['trait_classifier'])
        self.assertTrue(self.task.node_cache.values['trait_classifier.features'].values.is_cuda)
        private = self.master.parent / 'result-reviews' / (before['result_revision'] + '.json')
        self.assertEqual(private.read_bytes(), old_review)
        self.assertFalse((self.root / 'result-reviews').exists())
        new_crops = export_seed_crop_artifacts(self.root / 'crops', self.result, after,
                                              seed_ids=(1, 2), padding_px=2)
        new_payload = json.loads((new_crops / 'seed-crops.json').read_text())
        self.assertEqual(new_payload, old_crops)
        # Two restores keep their own bytes; no shared approvals are created.
        second = restore_snapshot(snapshot, self.root)
        self.assertNotEqual(second.parent, self.master.parent)
        self.assertEqual((second.parent / 'result-reviews' / private.name).read_bytes(), old_review)
        with self.assertRaisesRegex(ValueError, 'captured selection|changed|binding'):
            # Model-path selection change must not be silently applied to a cached result.
            self.window.pipeline.set_parameter('trait_classifier', 'trait_model_path', 'None')
            self.window._current_report(self.result)
        restored_model = after['native_artifact_execution']['selections']['trait_classifier']['selection']
        self.window.pipeline.set_parameter('trait_classifier', 'trait_model_path', restored_model)
        self.window.pipeline.set_parameter('trait_classifier', 'trait_abstention_threshold', .99)
        loaded = ProjectAnalysisStore(self.root).load(self.master)
        ProjectAnalysisStore(self.root).save(replace(loaded.document,
            analysis_settings=analysis_settings_profile_from_graph(self.window.pipeline)), self.master)
        self.window._cache_dirty_nodes.clear()
        changed, _ = self.paired()
        self.assertNotEqual(changed['result_revision'], after['result_revision'])
        self.assertEqual(changed['review_lookup'], 'not_recorded')
        model_path = Path(changed['trait_classifier']['model_path'])
        model_path.write_bytes(model_path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'changed after analysis'):
            self.window._current_report(self.result)
        changed_model, _ = self.paired()
        self.assertNotEqual(changed_model['result_revision'], changed['result_revision'])
        self.assertEqual(changed_model['review_lookup'], 'not_recorded')
        self.assertEqual(private.read_bytes(), old_review)

    def learned_relocation(self, node, family, model_class):
        from seedvision.learning.checkpoint import save_checkpoint
        from seedvision.learning.contracts import FeatureStackSpec
        spec = FeatureStackSpec(include_species_planes=False)
        model = model_class(spec.input_channels, base_channels=8, depth=2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        path = self.root / (node + '.pt')
        save_checkpoint(path, model, family=family, feature_spec=spec)
        self.graph.set_parameter(node, 'checkpoint_path', str(path))
        self.graph.set_parameter(node, 'tile_size', 128)
        self.graph.set_parameter(node, 'tile_overlap', 16)
        self.graph.set_enabled(node, True)
        self.save_project()
        self.install_profile()
        before, _ = self.paired()
        self.assertEqual(before['method'], node)
        self.assertEqual(before['identity']['checkpoint_id']['source_sha256'], sha256(path.read_bytes()).hexdigest())
        self.assertTrue(before['identity']['checkpoint_id']['used'])
        review_path = self.window._review_path(before)
        review_path.parent.mkdir(exist_ok=True)
        raw = json.dumps({'result_revision': before['result_revision'], 'decisions': {}}).encode()
        review_path.write_bytes(raw)
        base = self.transport()
        snapshot = create_snapshot(self.root, self.master, base / 'snapshots')
        self.relocate(base, snapshot)
        after, _ = self.paired()
        self.assertEqual(after['identity'], before['identity'])
        self.assertEqual(after['result_revision'], before['result_revision'])
        self.assertEqual(after['instances'], before['instances'])
        self.assertEqual(after['review_lookup'], 'matched')
        current = after['native_artifact_execution']['selections'][node]
        self.assertTrue(Path(current['resolved_path']).is_relative_to(self.root))
        self.assertEqual(current['state'], 'used_checkpoint_bytes_verified')
        self.assertNotEqual(current['resolved_path'], str(path))
        self.assertEqual((self.master.parent / 'result-reviews' / review_path.name).read_bytes(), raw)

    def test_actual_unet_checkpoint_and_cuda_inference_preserve_revision_after_relocation(self):
        from seedvision.learning.models import MultiHeadSeedUNet
        self.learned_relocation('unet_instances', 'unet_watershed', MultiHeadSeedUNet)

    def test_actual_stardist_checkpoint_and_cuda_inference_preserve_revision_after_relocation(self):
        from seedvision.learning.models import SeedStarDist2D
        self.learned_relocation('stardist_instances', 'stardist', SeedStarDist2D)

    def test_disabled_selected_trait_and_learned_bytes_relocate_without_invented_model_validation(self):
        self.graph.set_enabled('trait_classifier', False)
        checkpoint = self.root / 'unused.pt'
        checkpoint.write_bytes(b'not an executable model; selected while disabled')
        self.graph.set_parameter('unet_instances', 'checkpoint_path', str(checkpoint))
        self.save_project()
        self.install_profile()
        before, _ = self.paired()
        base = self.transport()
        snapshot = create_snapshot(self.root, self.master, base / 'snapshots')
        self.relocate(base, snapshot)
        after, _ = self.paired()
        self.assertEqual(before['identity'], after['identity'])
        self.assertIsNone(after.get('trait_classifier'))
        record = after['identity']['native_artifact_selections']['unet_instances']
        self.assertEqual(record['state'], 'bytes_recorded_format_unproven')
        self.assertFalse(record['enabled'])
        self.assertFalse(record['used'])
        resolved = Path(after['native_artifact_execution']['selections']['unet_instances']['resolved_path'])
        self.assertTrue(resolved.is_relative_to(self.root))
        self.assertEqual(resolved.read_bytes(), b'not an executable model; selected while disabled')

    def test_corrupt_and_unsafe_supplementary_review_inventory_cannot_publish_working_project(self):
        before, _ = self.paired()
        self.save_desktop_review()
        base = self.transport()
        snapshot = create_snapshot(self.root, self.master, base / 'snapshots')
        index_path = snapshot / 'snapshot-index.json'
        original = json.loads(index_path.read_text())
        review_name = 'result-reviews/' + before['result_revision'] + '.json'
        object_path = original['supplementary_evidence'][review_name]
        for name, object_name in (('result-reviews/../elsewhere.json', object_path),
                ('result-reviews\\bad.json', object_path), ('Result-Reviews/bad.json', object_path),
                (review_name, 'objects/not-in-inventory.json')):
            bad = deepcopy(original)
            bad['supplementary_evidence'] = {name: object_name}
            index_path.write_text(json.dumps(bad))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'review evidence'):
                restore_snapshot(snapshot, base / 'rejected')
        index_path.write_text(json.dumps(original))
        (snapshot / object_path).write_bytes(b'changed review')
        with self.assertRaisesRegex(ValueError, 'verification failed|changed bytes'):
            restore_snapshot(snapshot, base / 'rejected')
        self.assertFalse(list((base / 'rejected').rglob('working.seedfiddle-project.json')))


if __name__ == '__main__':
    unittest.main()
