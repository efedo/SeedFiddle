from __future__ import annotations

from tests.functional_units import functional_units

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from seedvision.pipeline import build_default_pipeline
from seedvision.optimization.search import optimize
from tests.test_node_optimization import FakeEvaluator


@functional_units
class NodeOptimizationUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def window(self, directory):
        from seedvision.ui.main_window import MainWindow
        from tests.review_fixtures import isolated_window
        with patch.object(MainWindow, '_load_workspace_images'):
            window = isolated_window(Path(__file__).resolve().parents[1],
                                     settings_path=Path(directory) / 'ui.ini')
        def dispose():
            window._optimization_task = None
            window._project_tracking_enabled = False
            window._project_dirty = False
            window._instance_annotations_dirty.clear()
            window._reference_masks_dirty.clear()
            window._unsaved_reference_sidecars.clear()
            window.close()
        self.addCleanup(dispose)
        return window

    def check_test_all_computational_cards_offer_shared_action(self):
        from seedvision.ui.pipeline_inspector import PipelineInspector
        from seedvision.optimization.registry import capability
        inspector = PipelineInspector()
        self.addCleanup(inspector.close)
        inspector.enable_unified_optimization()
        inspector.set_optimization_context(ready=True, busy=False)
        graph = build_default_pipeline()
        emitted = []
        inspector.node_action_requested.connect(lambda *args: emitted.append(args))
        for node in (*graph.nodes.values(), *graph.unused_nodes.values()):
            inspector.set_node(node)
            self.assertEqual(not inspector.optimization_button.isHidden(), bool(capability(node).searchable))
            self.assertTrue(inspector.procedural_fit_container.isHidden())
            self.assertTrue(inspector.reference_edge_fit_container.isHidden())
            if inspector.optimization_button.isEnabled():
                inspector.optimization_button.click()
                self.assertEqual(emitted[-1][:2], (node.identifier, inspector.OPTIMIZE_NODE_ACTION))
        inspector.set_node(graph.node('procedural_instances'))
        inspector.set_optimization_context(ready=True, busy=True)
        self.assertFalse(inspector.optimization_button.isEnabled())
        inspector.set_optimization_context(ready=True, busy=False)
        inspector.set_editing_locked(True)
        self.assertFalse(inspector.optimization_button.isEnabled())
        self.assertIn('Pipeline is frozen', inspector.optimization_status.text())

    def check_test_project_command_and_dispatch_are_connected(self):
        from seedvision.ui.pipeline_inspector import PipelineInspector
        with TemporaryDirectory() as directory:
            window = self.window(directory)
            with patch.object(window, '_start_node_optimization') as start:
                window.optimize_project_action.trigger()
                start.assert_called_once_with()
                window._pipeline_node_action_requested('background_likelihood', PipelineInspector.OPTIMIZE_NODE_ACTION, None)
                self.assertEqual(start.call_args.args, ('background_likelihood',))

    def check_test_classifier_manual_controls_remain_editable_without_optimization(self):
        from PySide6.QtWidgets import QDoubleSpinBox, QLineEdit
        from seedvision.ui.pipeline_inspector import PipelineInspector
        from seedvision.optimization.registry import EXEMPT
        inspector = PipelineInspector()
        self.addCleanup(inspector.close)
        graph = build_default_pipeline()
        classifier = graph.node('trait_classifier')
        classifier.enabled = True
        inspector.parameter_changed.connect(graph.set_parameter)
        changes, actions = [], []
        inspector.parameter_changed.connect(lambda *args: changes.append(args))
        inspector.node_action_requested.connect(lambda *args: actions.append(args))
        inspector.enable_unified_optimization()
        inspector.set_optimization_context(ready=True, busy=False)
        inspector.set_node(classifier)
        self.assertTrue(inspector.optimization_button.isHidden())
        self.assertFalse(inspector.optimization_button.isEnabled())
        self.assertEqual(inspector.optimization_status.text(), EXEMPT['trait_classifier'])
        path = inspector.findChild(QLineEdit, 'parameter_trait_model_path')
        threshold = inspector.findChild(QDoubleSpinBox, 'parameter_trait_abstention_threshold')
        self.assertIsNotNone(path)
        self.assertIsNotNone(threshold)
        self.assertTrue(path.isEnabled())
        self.assertFalse(path.isReadOnly())
        self.assertTrue(threshold.isEnabled())
        self.assertFalse(threshold.isReadOnly())
        path.setText('selected-frozen-traits.json')
        path.editingFinished.emit()
        threshold.setValue(.63)
        threshold.editingFinished.emit()
        self.assertEqual(classifier.parameters['trait_model_path'], 'selected-frozen-traits.json')
        self.assertEqual(classifier.parameters['trait_abstention_threshold'], .63)
        self.assertEqual(changes, [('trait_classifier', 'trait_model_path', 'selected-frozen-traits.json'),
                                  ('trait_classifier', 'trait_abstention_threshold', .63)])
        self.assertFalse(actions)
        for enabled in (False, True):
            classifier.enabled = enabled
            inspector.set_node(classifier)
            self.assertEqual(inspector.findChild(QLineEdit, 'parameter_trait_model_path').isEnabled(), enabled)
            self.assertEqual(inspector.findChild(QDoubleSpinBox, 'parameter_trait_abstention_threshold').isEnabled(), enabled)
            self.assertTrue(inspector.optimization_button.isHidden())
            self.assertFalse(inspector.optimization_button.isEnabled())

    def check_test_completed_progress_does_not_cancel_success_and_apply_is_atomic(self):
        from PySide6.QtWidgets import QDialog, QProgressDialog
        with TemporaryDirectory() as directory:
            window = self.window(directory)
            window.pipeline.node('wavelet_decomposition').set_parameter('wavelet_level_count', 4)
            report = optimize(window.pipeline, FakeEvaluator(), selected={'wavelet_decomposition'})
            path = Path(directory)/'review.seedfiddle-project.json'
            window._current_project_path = path
            task = SimpleNamespace(revision=window.pipeline.revision, stop=Event(), project_path=path,
                annotation_snapshot=window._optimization_annotation_snapshot())
            window._optimization_task = task
            progress = QProgressDialog('Optimizing', 'Cancel', 0, 0, window)
            progress.canceled.connect(task.stop.set)
            progress.show()
            self.app.processEvents()
            window._optimization_progress = progress
            with patch.object(QDialog, 'exec', return_value=QDialog.DialogCode.Accepted), patch.object(window, '_optimization_refresh') as refresh:
                window._optimization_completed(report)
            self.assertFalse(task.stop.is_set())
            self.assertEqual(window.pipeline.node('wavelet_decomposition').parameters['wavelet_level_count'], 3)
            self.assertIsNone(window._optimization_task)
            refresh.assert_called_once()
            import json
            record = json.loads(path.with_suffix('.optimization.json').read_text())
            self.assertEqual(record['disposition'], 'applied')
            self.assertEqual(len(list(Path(directory).glob('*.optimization-*.json'))), 1)

    def check_test_stale_proposal_never_opens_apply_dialog(self):
        from PySide6.QtWidgets import QDialog
        with TemporaryDirectory() as directory:
            window = self.window(directory)
            report = optimize(window.pipeline, FakeEvaluator(), selected={'wavelet_decomposition'})
            path = Path(directory)/'review.seedfiddle-project.json'
            window._current_project_path = path
            window._optimization_task = SimpleNamespace(revision=-1, stop=Event(), project_path=path,
                annotation_snapshot=window._optimization_annotation_snapshot())
            with patch.object(QDialog, 'exec') as dialog:
                window._optimization_completed(report)
            dialog.assert_not_called()
            self.assertIsNone(window._optimization_task)

    def check_test_rollback_refuses_later_edits_to_another_control(self):
        import json
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        with TemporaryDirectory() as directory:
            window = self.window(directory)
            node = window.pipeline.node('edge_gradients')
            original = dict(node.parameters)
            changed = {'edge_blur_sigma': original['edge_blur_sigma'] + .1}
            window.pipeline.set_parameters(node.identifier, changed)
            other = next(spec for spec in node.parameter_specs if spec.key != 'edge_blur_sigma' and spec.kind == 'float')
            node.set_parameter(other.key, other.minimum if node.parameters[other.key] != other.minimum else other.maximum)
            expected = dict(node.parameters)
            project = Path(directory)/'project.json'
            path = project.with_suffix('.optimization.json')
            window._current_project_path = project
            path.write_text(json.dumps({'started':'test', 'mode':'test', 'disposition':'applied',
                'original':{node.identifier:original}, 'nodes':[{'node_id':node.identifier,'changes':changed}]}))
            with patch.object(QFileDialog, 'getOpenFileName', return_value=(str(path),'')), \
                 patch.object(QMessageBox, 'exec', return_value=QMessageBox.StandardButton.Yes), \
                 patch.object(QMessageBox, 'information') as information:
                window._show_optimization_history()
            information.assert_called_once()
            self.assertEqual(node.parameters, expected)

    def check_test_save_as_preserves_companions_and_target_location(self):
        import json
        with TemporaryDirectory() as directory:
            window = self.window(directory)
            source = Path(directory)/'original.json'
            destination = Path(directory)/'subdir'/'copy.json'
            destination.parent.mkdir()
            window._current_project_path = source
            source.with_suffix('.optimization-targets.json').write_text(json.dumps({'sha':'targets.npz'}))
            source.with_suffix('.optimization-policy.json').write_text(json.dumps({'sha':{'complete':True}}))
            window._copy_optimization_companions(destination)
            targets = json.loads(destination.with_suffix('.optimization-targets.json').read_text())
            self.assertEqual(Path(targets['sha']), Path(directory)/'targets.npz')
            self.assertTrue(json.loads(destination.with_suffix('.optimization-policy.json').read_text())['sha']['complete'])

    def check_test_sample_preflight_allows_local_refs_and_tracks_image_context(self):
        import numpy as np
        from seedvision.reference_library.contracts import BiologicalContext
        from seedvision.ui.main_window import _path_identity
        from seedvision.persistence.reference_regions import SeedInstanceAnnotation
        from seedvision.ui.optimization import OptimizationDialog
        with TemporaryDirectory() as directory:
            window = self.window(directory)
            paths = [Path(directory) / (name + '.png') for name in ('first', 'second', 'draft')]
            for path in paths:
                path.write_bytes(b'fingerprinted source')
            keys = [_path_identity(path) for path in paths]
            window._image_paths = dict(zip(keys, paths))
            window._project_species_library_pin = None
            contexts = [BiologicalContext('unknown', seed_lot_id='lot_a'),
                        BiologicalContext('unknown', seed_lot_id='lot_b')]
            window._project_image_biological_contexts = dict(zip(keys, contexts))
            labels = np.zeros((20, 24), np.uint16)
            labels[3:10, 3:10] = 1
            labels[12:17, 12:17] = 2
            for key in keys:
                window._applied_instance_annotations[key] = labels
                window._applied_seed_annotations[key] = {1: SeedInstanceAnnotation(
                    1, shape_reviewed=True, outline_visibility='complete')}
            window._instance_annotations_dirty.add(keys[2])
            resolved_contexts = []
            def resolve(pin, *, context, **kwargs):
                resolved_contexts.append(context)
                return SimpleNamespace(error='No species library is pinned.', artifact=None)
            with patch.object(window._species_library_service, 'resolve', side_effect=resolve):
                samples = window._optimization_samples()
            self.assertEqual(len(samples), 2)
            self.assertEqual(resolved_contexts, contexts)
            self.assertEqual([sample.kwargs['biological_context'] for sample in samples], contexts)
            self.assertTrue(all(sample.kwargs['species_library'] is None for sample in samples))
            self.assertTrue(all(sample.excluded_ids == (2,) for sample in samples))
            self.assertIn('Unapplied draft', window._optimization_exclusions[0][1])
            dialog = OptimizationDialog(window.pipeline, samples, None,
                excluded_images=window._optimization_exclusions)
            self.addCleanup(dialog.close)
            self.assertIn('1 excluded', dialog.readiness.text())
            self.assertIn('1 eligible; 1 excluded', dialog.images.item(0, 1).text())
            snapshot = window._optimization_annotation_snapshot()
            window._project_image_biological_contexts[keys[0]] = contexts[1]
            self.assertNotEqual(snapshot, window._optimization_annotation_snapshot())
            window.pipeline.node('reference_texture_prototypes').parameters['material_prototype_reference_source'] = 'Species library only'
            with patch.object(window._species_library_service, 'resolve', side_effect=resolve):
                with self.assertRaisesRegex(ValueError, 'pin a compatible species library'):
                    window._optimization_samples()
            window._project_species_library_pin = object()
            with patch.object(window._species_library_service, 'resolve', side_effect=resolve):
                with self.assertRaisesRegex(ValueError, 'No species library is pinned'):
                    window._optimization_samples()

    def test_workflow_lifecycle(self):
        """6 related NodeOptimizationUITests scenarios, each with private fixture state."""
        self.scenario(self.check_test_all_computational_cards_offer_shared_action)
        self.scenario(self.check_test_project_command_and_dispatch_are_connected)
        self.scenario(self.check_test_classifier_manual_controls_remain_editable_without_optimization)
        self.scenario(self.check_test_rollback_refuses_later_edits_to_another_control)
        self.scenario(self.check_test_save_as_preserves_companions_and_target_location)
        self.scenario(self.check_test_sample_preflight_allows_local_refs_and_tracks_image_context)

    def test_validation_and_recovery(self):
        """2 related NodeOptimizationUITests scenarios, each with private fixture state."""
        self.scenario(self.check_test_completed_progress_does_not_cancel_success_and_apply_is_atomic)
        self.scenario(self.check_test_stale_proposal_never_opens_apply_dialog)


if __name__ == '__main__':
    unittest.main()
