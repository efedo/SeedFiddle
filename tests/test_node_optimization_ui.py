from __future__ import annotations

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


class NodeOptimizationUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def window(self, directory):
        from PySide6.QtCore import QSettings
        from seedvision.ui.main_window import MainWindow
        with patch.object(MainWindow, '_load_workspace_images'), patch('seedvision.ui.main_window.QSettings',
                return_value=QSettings(str(Path(directory)/'ui.ini'), QSettings.Format.IniFormat)):
            window = MainWindow(Path(__file__).resolve().parents[1])
        def dispose():
            window._optimization_task = None
            window._project_tracking_enabled = False
            window._project_dirty = False
            window.close()
        self.addCleanup(dispose)
        return window

    def test_all_computational_cards_offer_shared_action(self):
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

    def test_project_command_and_dispatch_are_connected(self):
        from seedvision.ui.pipeline_inspector import PipelineInspector
        with TemporaryDirectory() as directory:
            window = self.window(directory)
            with patch.object(window, '_start_node_optimization') as start:
                window.optimize_project_action.trigger()
                start.assert_called_once_with()
                window._pipeline_node_action_requested('background_likelihood', PipelineInspector.OPTIMIZE_NODE_ACTION, None)
                self.assertEqual(start.call_args.args, ('background_likelihood',))

    def test_completed_progress_does_not_cancel_success_and_apply_is_atomic(self):
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

    def test_stale_proposal_never_opens_apply_dialog(self):
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

    def test_rollback_refuses_later_edits_to_another_control(self):
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

    def test_save_as_preserves_companions_and_target_location(self):
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


if __name__ == '__main__':
    unittest.main()
