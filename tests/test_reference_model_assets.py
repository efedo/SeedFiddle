"""Real classifier bytes survive project moves and portable snapshot restore."""

from tests.catalogue_fixtures import CatalogueTestCase

from tests.functional_units import functional_units
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch

import torch

from seedvision.benchmarking.native_recipe import run_native_recipe_path, NativeRecipeError
from seedvision.export import native_results as outputs
from seedvision.persistence.project_analysis import (
    ProjectAnalysisStore, ProjectImageSpec, InvalidProjectAnalysis,
)
from seedvision.persistence import reference_models as assets
from seedvision.persistence.snapshot import create_snapshot, restore_snapshot
from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
from seedvision.pipeline import build_default_pipeline
from seedvision.segmentation.baseline import PipelineAnalysisCache
from tests import test_native_result_export as fixtures
from tests.test_native_result_export import _attachment
from tests.test_native_naive_bayes_recipe import _recipe_with_model
from tests.test_plantcv_naive_bayes import _constant_pdf


@functional_units
class ReferenceModelAssetTests(CatalogueTestCase):
    setUp = fixtures.NativeResultExportTests.setUp
    tearDown = fixtures.NativeResultExportTests.tearDown
    make_recipe = fixtures.NativeResultExportTests.make_recipe

    def bayes_recipe(self):
        path = self.root/'pdf.tsv'
        path.write_bytes(_constant_pdf({'first': 1., 'second': 2.}))
        return _recipe_with_model(path)[0]

    def project(self, recipes=None):
        workspace = self.root/'workspace'
        workspace.mkdir()
        shutil.copyfile(self.source, workspace/'seeds.png')
        (workspace/'config').mkdir()
        shutil.copyfile(Path(__file__).parents[1]/'config'/'traits.json', workspace/'config'/'traits.json')
        graphs = tuple(_attachment(recipe) for recipe in (recipes or (self.recipe,)))
        master = ProjectAnalysisStore(workspace).capture_and_save(
            analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
            images=(ProjectImageSpec(workspace/'seeds.png', (13, 17)),), reference_graphs=graphs,
            destination=workspace/'test.seedfiddle-project.json')
        return workspace, master, graphs

    def assert_products_equal(self, before, after):
        self.assertEqual(before.recipe_identity, after.recipe_identity)
        for previous, current in zip(before.nodes, after.nodes, strict=True):
            for name, value in previous.outputs.items():
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(value, current.outputs[name]))
                    self.assertEqual(current.outputs[name].device.type, 'cuda')
                elif name == 'class_masks':
                    self.assertEqual(tuple(value), tuple(current.outputs[name]))
                    for key, tensor in value.items():
                        self.assertTrue(torch.equal(tensor, current.outputs[name][key]))

    def check_test_capture_preserves_history_and_deduplicates_exact_objects(self):
        second = replace(self.recipe, recipe_id=self.recipe.recipe_id+'_history')
        workspace, master, graphs = self.project((self.recipe, second))
        loaded = ProjectAnalysisStore(workspace).load(master)
        self.assertFalse(loaded.issues)
        self.assertEqual(loaded.document.reference_graphs, graphs)
        self.assertEqual(len(loaded.document.reference_model_assets), 2)
        self.assertEqual(len(list((workspace/'native-models').iterdir())), 1)
        self.assertEqual(len({asset.path for asset in loaded.document.reference_model_assets}), 1)
        self.assertEqual(json.loads(master.read_bytes())['version'], 6)

    def check_test_legacy_v3_migrates_metadata_without_inventing_model_closure(self):
        workspace, master, graphs = self.project()
        payload = json.loads(master.read_bytes())
        payload['version'] = 3
        payload.pop('correction_bindings')
        payload.pop('reference_model_assets')
        payload.pop('native_recipes')
        master.write_text(json.dumps(payload), encoding='utf-8')
        store = ProjectAnalysisStore(workspace)
        loaded = store.load(master)
        self.assertEqual(loaded.document.reference_model_assets, ())
        self.assertEqual(loaded.document.reference_graphs, graphs)
        store.save(loaded.document, master)
        self.assertEqual(json.loads(master.read_bytes())['reference_model_assets'], [])
        with self.assertRaises(ValueError):
            assets.verify_reference_model_assets(workspace, (), graphs, require_complete=True)
        snapshot = create_snapshot(workspace, master, self.root/'snapshots')
        portable = ProjectAnalysisStore(snapshot).load(snapshot/'snapshot.seedfiddle-project.json')
        self.assertEqual(len(portable.document.reference_model_assets), 1)

    def check_test_invalid_or_stale_binding_never_replaces_master(self):
        workspace, master, graphs = self.project()
        store = ProjectAnalysisStore(workspace)
        loaded = store.load(master)
        original = master.read_bytes()
        asset = loaded.document.reference_model_assets[0]
        for change in ({'selection_sha256': '0'*64}, {'source_sha256': '0'*64}, {'source_bytes': 1}):
            with self.assertRaises(InvalidProjectAnalysis):
                store.save(replace(loaded.document, reference_model_assets=(replace(asset, **change),)), master)
            self.assertEqual(master.read_bytes(), original)
        with self.assertRaises(ValueError):
            replace(asset, path='../outside.model')
        with self.assertRaises(InvalidProjectAnalysis):
            store.save(replace(loaded.document, reference_model_assets=(asset, asset)), master)

    def check_test_aggregate_bounds_precede_any_model_read_or_copy(self):
        graphs = (_attachment(self.recipe), _attachment(self.bayes_recipe()))
        for name, bound in (('MAX_MODEL_ASSETS', 1), ('MAX_TOTAL_MODEL_BYTES', 1)):
            with patch.object(assets, name, bound), patch.object(assets, 'verified_asset_bytes') as read:
                with self.assertRaises(ValueError):
                    assets.capture_reference_model_assets(self.root, graphs)
                read.assert_not_called()

    def check_test_missing_legacy_model_preserves_save_and_reports_unavailability(self):
        self.model_path.unlink()
        workspace, master, graphs = self.project()
        store = ProjectAnalysisStore(workspace)
        loaded = store.load(master)
        self.assertEqual(loaded.document.reference_graphs, graphs)
        self.assertEqual(loaded.document.reference_model_assets, ())
        self.assertEqual(len(loaded.issues), 1)
        self.assertTrue(loaded.issues[0].role.startswith('classifier_model:'))
        store.save(loaded.document, master)
        self.assertEqual(store.load(master).document.reference_graphs, graphs)
        with self.assertRaises(ValueError):
            create_snapshot(workspace, master, self.root/'snapshots')
        self.assertFalse((self.root/'snapshots').exists())

    def check_test_unavailable_bound_asset_recapture_preserves_descriptor_without_fallback(self):
        workspace, master, graphs = self.project()
        loaded = ProjectAnalysisStore(workspace).load(master)
        bound = loaded.document.reference_model_assets
        asset_path = workspace/bound[0].path
        asset_path.write_bytes(b'broken pinned model')
        recaptured = assets.capture_reference_model_assets(workspace, graphs, bound, allow_unavailable=True)
        self.assertEqual(recaptured, bound)
        self.assertEqual(asset_path.read_bytes(), b'broken pinned model')
        self.assertTrue(self.model_path.is_file())
        with self.assertRaises(ValueError):
            assets.capture_reference_model_assets(workspace, graphs, bound)

    def check_test_unexpected_model_validation_failure_is_not_hidden_by_metadata_save(self):
        with patch.object(assets, 'verified_asset_bytes', side_effect=RuntimeError('unexpected defect')):
            with self.assertRaisesRegex(RuntimeError, 'unexpected defect'):
                assets.capture_reference_model_assets(self.root, (_attachment(self.recipe),),
                                                      allow_unavailable=True)

    def check_test_foreign_origin_requires_verified_local_transport(self):
        from seedvision.benchmarking import native_kmeans_recipe as factory
        from seedvision.model_io.asset_paths import resolve_model_path
        selection = dict(self.recipe.parameters_for_runner(factory.MODEL_STAGE_ID)['model_source'])
        foreign = '/foreign/model.json' if os.name == 'nt' else r'Z:\foreign\model.json'
        selection['path'] = foreign
        with patch('seedvision.model_io.asset_paths.Path.resolve') as resolve:
            with self.assertRaises(ValueError):
                resolve_model_path(foreign)
            resolve.assert_not_called()
        model = factory.load_selected_model(selection, resolved_path=str(self.model_path.resolve()))
        identity = factory.selected_model_identity(selection, model, resolved_path=str(self.model_path.resolve()))
        self.assertEqual(identity['path'], foreign)

    def check_test_model_folder_redirection_is_rejected_before_copy(self):
        workspace = self.root/'workspace'
        outside = self.root/'outside'
        workspace.mkdir()
        outside.mkdir()
        link = workspace/'native-models'
        if os.name == 'nt':
            import subprocess
            completed = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(outside)],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
        else:
            link.symlink_to(outside, target_is_directory=True)
        try:
            with self.assertRaisesRegex(ValueError, 'escapes'):
                assets.capture_reference_model_assets(workspace, (_attachment(self.recipe),))
            self.assertEqual(list(outside.iterdir()), [])
        finally:
            if os.name == 'nt':
                os.rmdir(link)  # Remove only the verified junction, never its target.
            else:
                link.unlink()

    def check_test_snapshot_copy_mutation_and_interruption_never_publish_completion(self):
        workspace, master, _ = self.project()
        bound = ProjectAnalysisStore(workspace).load(master).document.reference_model_assets[0]
        model_path = workspace/bound.path
        raw = model_path.read_bytes()
        write = Path.write_bytes
        def changed_model(destination, data):
            value = write(destination, data)
            if destination.parent.name == 'objects' and destination.suffix == '.model':
                write(model_path, raw+b' ')
            return value
        with patch.object(Path, 'write_bytes', changed_model):
            with self.assertRaisesRegex(ValueError, 'Copy verification failed'):
                create_snapshot(workspace, master, self.root/'snapshots')
        model_path.write_bytes(raw)
        def interrupted_model(destination, data):
            value = write(destination, data)
            if destination.parent.name == 'objects' and destination.suffix == '.model':
                raise OSError('interrupted model copy')
            return value
        with patch.object(Path, 'write_bytes', interrupted_model):
            with self.assertRaisesRegex(OSError, 'interrupted model copy'):
                create_snapshot(workspace, master, self.root/'snapshots')
        for partial in (self.root/'snapshots').iterdir():
            self.assertFalse((partial/'snapshot-index.json').exists())

    def check_test_model_growth_during_snapshot_read_stays_bounded(self):
        from contextlib import contextmanager
        from seedvision.persistence import snapshot as snapshots
        workspace, master, _ = self.project()
        bound = ProjectAnalysisStore(workspace).load(master).document.reference_model_assets[0]
        model_path = workspace/bound.path
        original = model_path.read_bytes()
        opening = Path.open
        read_sizes = []
        class GrowingReader:
            def __init__(self, stream):
                self.stream = stream
            def read(self, size):
                read_sizes.append(size)
                with opening(model_path, 'ab') as appended:
                    appended.write(b' '*100_000)
                return self.stream.read(size)
        @contextmanager
        def open_growing(path, *args, **kwargs):
            with opening(path, *args, **kwargs) as stream:
                yield GrowingReader(stream) if path == model_path and args == ('rb',) else stream
        with patch.object(Path, 'open', open_growing):
            with self.assertRaisesRegex(ValueError, 'Copy verification failed'):
                snapshots._model_bytes(model_path, bound)
        self.assertEqual(read_sizes, [bound.source_bytes+1])
        model_path.write_bytes(original)
        portable = create_snapshot(workspace, master, self.root/'snapshots')
        portable_document = ProjectAnalysisStore(portable).load(portable/'snapshot.seedfiddle-project.json').document
        portable_model = portable/portable_document.reference_model_assets[0].path
        portable_model.write_bytes(original+b' '*100_000)
        with patch.object(snapshots, 'file_sha256', wraps=snapshots.file_sha256) as hash_file:
            with self.assertRaises(ValueError):
                restore_snapshot(portable, self.root/'restore')
        self.assertNotIn(portable_model.resolve(), [Path(call.args[0]) for call in hash_file.call_args_list])
        self.assertFalse((self.root/'restore').exists())








    def _assert_legacy_archive(self, version):
        from seedvision.benchmarking.checkpoints import load_checkpoint, publish_checkpoint
        result = run_native_recipe_path(self.recipe, self.source)
        current = outputs.export_native_recipe_result(self.root/'v2', self.recipe, result)
        report = json.loads(current.read_bytes())
        report.pop('result_revision')
        groups = report.pop('groups')
        if version == 1:
            report.pop('model_asset_paths')
        if version < 3:
            report.pop('correction_inputs')
            report.pop('correction_implementation_sha256')
        if version < 4:
            report.pop('correction_input_origins')
        for node in report['nodes']:
            node.pop('calculation')  # Introduced in v5, absent in v1-v4 records.
        report['version'] = version
        binding = sha256(outputs._json_bytes(report)).hexdigest()
        legacy = self.root/f'v{version}-legacy'
        legacy.mkdir()
        saved = []
        for index, group in enumerate(groups):
            checkpoint = load_checkpoint(current.parent/'checkpoints', group['checkpoint_sha256'])
            reference = publish_checkpoint(legacy/'checkpoints', checkpoint.identity, checkpoint.arrays,
                execution=checkpoint.execution, measurement=checkpoint.measurement,
                raster_roles=checkpoint.raster_roles,
                metadata={'format': outputs.FORMAT, 'version': version, 'run_binding_sha256': binding,
                          'group_index': index, 'arrays': group['arrays']}, allow_cpu_transfer=True)
            saved.append({'checkpoint_sha256': reference.checkpoint_sha256, 'arrays': group['arrays']})
        report['groups'] = saved
        report['result_revision'] = sha256(outputs._json_bytes(report)).hexdigest()
        (legacy/'native-results.json').write_bytes(outputs._json_bytes(report))
        restored, arrays = outputs.load_native_recipe_result(legacy)
        self.assertEqual(restored['version'], version)
        self.assertEqual('model_asset_paths' in restored, version >= 2)
        self.assertEqual('correction_inputs' in restored, version >= 3)
        self.assertEqual('correction_input_origins' in restored, version >= 4)
        self.assertEqual(restored['recipe_identity'], result.recipe_identity.to_dict())
        self.assertTrue(arrays)






    def test_workflow_lifecycle(self):
        """5 related ReferenceModelAssetTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_capture_preserves_history_and_deduplicates_exact_objects)
        self.scenario(self.check_test_legacy_v3_migrates_metadata_without_inventing_model_closure)
        self.scenario(self.check_test_aggregate_bounds_precede_any_model_read_or_copy)
        self.scenario(self.check_test_foreign_origin_requires_verified_local_transport)
        self.scenario(self.check_test_model_growth_during_snapshot_read_stays_bounded)

    def test_validation_and_recovery(self):
        """6 related ReferenceModelAssetTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_invalid_or_stale_binding_never_replaces_master)
        self.scenario(self.check_test_missing_legacy_model_preserves_save_and_reports_unavailability)
        self.scenario(self.check_test_unavailable_bound_asset_recapture_preserves_descriptor_without_fallback)
        self.scenario(self.check_test_unexpected_model_validation_failure_is_not_hidden_by_metadata_save)
        self.scenario(self.check_test_model_folder_redirection_is_rejected_before_copy)
        self.scenario(self.check_test_snapshot_copy_mutation_and_interruption_never_publish_completion)
