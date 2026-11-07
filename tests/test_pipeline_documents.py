"""Standalone graph/layout transport, factory protection and frozen file loads."""


from tests.functional_units import functional_units
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QComboBox, QMainWindow, QMessageBox

from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
from seedvision.persistence.pipeline_documents import (
    PIPELINE_EXTENSION, PipelineDocument, PipelineDocumentError,
    PipelineDocumentStore, PipelineLayoutDocument,
)
from seedvision.pipeline import build_default_pipeline
from seedvision.pipeline.catalogue_nodes import create_instance, display_id
from seedvision.ui.pipeline_canvas import PipelineCanvas
from seedvision.ui.pipeline_documents import PipelineDocumentsController


@functional_units
class PipelineDocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def check_test_full_roundtrip_preserves_settings_instances_and_layout(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            default = build_default_pipeline()
            default.set_parameter('ruler_detection', 'ruler_length_mm', 183.0)
            profile = analysis_settings_profile_from_graph(default)
            instances = (create_instance('default:seed_scale_estimation'),
                         create_instance('default:seed_scale_estimation'))
            selection = display_id(instances[1], instances[1].target_node)
            layout = PipelineLayoutDocument(((selection, 454.5, -31.0),), selected_node=selection,
                                            selected_output='diameter', bundle_cables=True)
            store = PipelineDocumentStore(root)
            document = PipelineDocument('Custom measured seeds', profile, instances, layout)
            path = root / ('measured' + PIPELINE_EXTENSION)
            store.save_pipeline(document, path)
            reopened = store.load_pipeline(path)
            self.assertEqual(reopened.to_dict(), document.to_dict())
            self.assertEqual(reopened.analysis_settings, profile)
            self.assertEqual(reopened.node_instances, instances)
            self.assertEqual(reopened.layout.selected_node, selection)

            # An earlier format, such as one holding a removed recipe workflow, is refused plainly.
            payload = document.to_dict()
            for changed, message in ((payload | {'version': 1, 'pipeline_workflow': None}, 'reads only version 2'),
                                     (payload | {'backend_default': 'native'}, 'missing or unknown'),
                                     (payload | {'format': 'other'}, 'not a Seed Fiddle pipeline')):
                stale = root / ('stale' + PIPELINE_EXTENSION)
                stale.write_text(json.dumps(changed), encoding='utf-8')
                with self.subTest(message=message), self.assertRaisesRegex(PipelineDocumentError, message):
                    store.load_pipeline(stale)

    def check_test_invalid_files_atomic_failure_and_factory_names_cannot_replace_existing_bytes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = PipelineDocument('Custom profile', analysis_settings_profile_from_graph(build_default_pipeline()))
            protected = root / ('bundled' + PIPELINE_EXTENSION)
            store = PipelineDocumentStore(root, factory_names=('Official seed pipeline',), factory_paths=(protected,))
            path = store.save_pipeline(document, root / ('custom' + PIPELINE_EXTENSION))
            original = path.read_bytes()
            for name in ('../escape', 'folder\\name', 'name:stream', 'line\nname', 'dot.'):
                with self.subTest(name=name), self.assertRaises(PipelineDocumentError):
                    replace(document, name=name)
            with patch('seedvision.persistence.pipeline_documents.os.replace', side_effect=OSError('simulated replacement failure')):
                with self.assertRaises(PipelineDocumentError):
                    store.save_pipeline(replace(document, name='Changed custom'), path)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(tuple(root.glob('*.tmp')))
            for target, name in ((protected, 'Custom profile'),
                                 (root / ('official seed pipeline' + PIPELINE_EXTENSION), 'Custom profile'),
                                 (root / ('custom' + PIPELINE_EXTENSION), '  Official seed pipeline  '),
                                 (root / 'config' / ('custom' + PIPELINE_EXTENSION), 'Custom profile')):
                with self.subTest(target=target, name=name), self.assertRaises(ValueError):
                    store.save_pipeline(replace(document, name=name), target)
            self.assertEqual(path.read_bytes(), original)
            malformed = root / ('invalid' + PIPELINE_EXTENSION)
            malformed.write_text('{"format":"seedfiddle-pipeline","format":"duplicate"}', encoding='utf-8')
            with self.assertRaises(PipelineDocumentError):
                store.load_pipeline(malformed)
            with patch('seedvision.persistence.pipeline_documents.MAX_DOCUMENT_BYTES', 32):
                with self.assertRaises(PipelineDocumentError):
                    store.load_pipeline(path)
            with self.assertRaises(PipelineDocumentError):
                PipelineLayoutDocument((('project', float('nan'), 0),))

    def check_test_controller_locks_defaults_and_layout_does_not_invalidate_calculations(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            window = QMainWindow()
            window._root = root
            window._background_work_is_active = Mock(return_value=False)
            window._thread_pool = SimpleNamespace(activeThreadCount=lambda: 0)
            window._set_project_dirty = Mock()
            graph = build_default_pipeline()
            canvas = PipelineCanvas(graph, window)
            window.pipeline_canvas = canvas
            window.pipeline_workflow = SimpleNamespace()
            window.catalogue_nodes = None
            window.image_workspace_action = QAction(window)
            window.pipeline_workspace_action = QAction(window)
            for action in (window.image_workspace_action, window.pipeline_workspace_action):
                action.setCheckable(True)
                action.setChecked(True)
            window.overlay_combo = QComboBox(window)
            window.overlay_combo.addItem('Raw', 'raw_image')
            profile = analysis_settings_profile_from_graph(graph)
            document = PipelineDocument('Custom inspection', profile)
            store = PipelineDocumentStore(root, factory_names=('Official source',))
            path = store.save_pipeline(document, root / ('inspection' + PIPELINE_EXTENSION))
            install = Mock()
            controller = PipelineDocumentsController(window, capture=lambda name: replace(document, name=name),
                                                      install=install, store=store)
            locks = Mock()
            controller.lock_changed.connect(locks)
            live_name = {'value': 'Original live name'}
            controller.named.connect(lambda name: live_name.update(value=name))
            self.assertTrue(controller.locked)
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No):
                self.assertFalse(controller.can_edit())
            controller.create_copy('Editable inspection')
            self.assertTrue(controller.can_edit())
            controller.loaded('Official source')
            self.assertTrue(controller.locked)
            self.assertTrue(locks.call_args.args[0])
            with patch.object(QMessageBox, 'warning'):
                self.assertFalse(controller.save_pipeline_as(root / ('copy' + PIPELINE_EXTENSION), 'Official source'))
                self.assertFalse(controller.load_pipeline(root / 'does-not-exist.json'))
                with patch.object(store, 'save_pipeline', side_effect=PipelineDocumentError('Simulated save failure')):
                    self.assertFalse(controller.save_pipeline_as(root / ('copy' + PIPELINE_EXTENSION), 'Custom failed name'))
            self.assertEqual(live_name['value'], 'Editable inspection')
            self.assertEqual(controller.current_name, 'Official source')
            install.assert_not_called()
            self.assertTrue(controller.save_pipeline_as(root / ('copy' + PIPELINE_EXTENSION), 'Custom saved name'))
            self.assertEqual(live_name['value'], 'Custom saved name')
            self.assertTrue(controller.load_pipeline(path))
            install.assert_called_once()
            self.assertFalse(controller.locked)
            revision, edges = graph.revision, graph.connections
            cached = object()
            window._analysis_caches = {'test-image': cached}
            graph.node('project').calculation_seconds = 12.1
            layout = PipelineLayoutDocument((('project', 231.5, -98.0), ('absent-stable-id', 1, 2)),
                                            selected_node='project', selected_output='raw_image', bundle_cables=True)
            with patch.object(graph, 'invalidate', wraps=graph.invalidate) as invalidate:
                self.assertIn('absent-stable-id', controller.apply_layout(layout))
            invalidate.assert_not_called()
            self.assertEqual((graph.node('project').x, graph.node('project').y), (231.5, -98.0))
            self.assertIs(window._analysis_caches['test-image'], cached)
            self.assertEqual(graph.node('project').calculation_seconds, 12.1)
            self.assertIs(graph.connections, edges)
            self.assertEqual(graph.revision, revision)
            self.assertEqual(analysis_settings_profile_from_graph(graph), profile)
            self.assertFalse(controller.locked)
            window.close()
            self.application.processEvents()

    def test_workflow_lifecycle(self):
        """1 related PipelineDocumentTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_full_roundtrip_preserves_settings_instances_and_layout)

    def test_validation_and_recovery(self):
        """2 related PipelineDocumentTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_invalid_files_atomic_failure_and_factory_names_cannot_replace_existing_bytes)
        self.scenario(self.check_test_controller_locks_defaults_and_layout_does_not_invalidate_calculations)
