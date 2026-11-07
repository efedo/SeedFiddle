"""Default pipelines stay locked; their edit controls offer a named, editable copy."""


from tests.functional_units import functional_units
from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QInputDialog, QMessageBox, QStyle,
    QStyledItemDelegate, QStyleOptionViewItem,
)

from seedvision.persistence.analysis_settings import save_analysis_settings_profile
from seedvision.pipeline import build_default_pipeline
from seedvision.ui.canvas_controls import COMPACT_ICON_SIZE, load_icon, rename_icon, save_icon
from seedvision.ui.pipeline_workflow import ENTRY_KIND, SELECTOR_WIDTH
from tests.review_fixtures import dispose_window, isolated_window

YES, NO = QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No
ADDED = 'default:seed_scale_estimation'
OPTIONAL_NODE = 'circle_candidates'
DEFAULT_NODE, DEFAULT_KEY, DEFAULT_FACTORY = 'procedural_instances', 'working_maximum_dimension', 1600


@contextmanager
def answers(answer=YES, names=()):
    """Answer the offer and each name prompt; one prompt too many fails the test."""
    replies = [name if isinstance(name, tuple) else (name, True) for name in names]
    with (patch.object(QMessageBox, 'question', return_value=answer) as question,
          patch.object(QInputDialog, 'getText', side_effect=replies) as prompt):
        yield question, prompt


@functional_units
class DefaultPipelineCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.window = isolated_window(self.root, editable=False)
        self.presenter = self.window.pipeline_workflow
        self.documents = self.window.pipeline_documents
        self.canvas = self.window.pipeline_canvas
        self.selector = self.presenter.selector

    def tearDown(self):
        dispose_window(self.window)
        self.application.processEvents()
        self.temporary.cleanup()

    def default_value(self):
        return self.window.pipeline.node(DEFAULT_NODE).parameters[DEFAULT_KEY]

    def default_positions(self):
        return {node.identifier: (node.x, node.y) for node in self.window.pipeline.nodes.values()}

    def copy_names(self):
        return [self.selector.itemText(index) for index in range(self.selector.count())
                if self.selector.itemData(index, ENTRY_KIND) == 'custom']

    def show_copy(self, name):
        self.selector.setCurrentIndex(self.selector.findData(name))
        self.assertEqual(self.documents.current_name, name)

    def make_copy(self, name):
        with answers(names=[name]):
            self.assertTrue(self.documents.rename_current())
        self.assertFalse(self.documents.locked)

    def assert_locked_default(self, name='Procedural seed analysis'):
        self.assertTrue(self.documents.locked)
        self.assertEqual(self.documents.current_name, name)
        self.assertEqual(self.selector.currentText(), name)
        self.assertTrue(self.canvas.editing_locked)

    def assert_on_copy(self, name):
        self.assertFalse(self.documents.locked)
        self.assertEqual(self.documents.current_name, name)
        self.assertEqual(self.selector.currentText(), name)
        self.assertFalse(self.canvas.editing_locked)
        self.assertFalse(self.window.pipeline_inspector.editing_locked)

    def check_test_add_node_names_line_up_whether_used_or_not(self):
        self.canvas._populate_node_catalogue()
        view = self.canvas.node_catalogue_list
        delegate = QStyledItemDelegate()

        def text_left(item):
            option = QStyleOptionViewItem()
            option.initFrom(view)
            option.decorationSize = view.iconSize()
            delegate.initStyleOption(option, view.indexFromItem(item))
            option.rect = view.visualItemRect(item)
            return QApplication.style().subElementRect(QStyle.SubElement.SE_ItemViewItemText, option, view).left()

        entries = self.canvas.catalogue_entry_items()
        used = [item for item in entries if item.foreground().color().name() == '#1768ad']
        unused = [item for item in entries if item.foreground().color().name() == '#218044']
        self.assertTrue(used)
        self.assertTrue(unused)
        self.assertEqual({text_left(item) for item in entries}, {text_left(used[0])})
        self.assertFalse(any(item.text().startswith(('✓', ' ')) for item in entries))
        self.assertFalse(any(item.icon().isNull() for item in entries))
        # Section headings have no icon column, so they stand out to the left.
        headings = [view.item(index) for index in range(view.count()) if view.item(index) not in entries]
        self.assertTrue(headings)
        self.assertTrue(all(item.icon().isNull() for item in headings))
        self.assertLess(text_left(headings[0]), text_left(used[0]))

    def check_test_selector_is_wider_with_rename_beside_it_and_no_lock_control(self):
        self.window.resize(1440, 900)
        self.window.show()
        self.application.processEvents()
        layout = self.presenter.strip.layout()
        self.assertEqual(self.selector.width(), SELECTOR_WIDTH)
        self.assertEqual(layout.indexOf(self.presenter.rename_button), layout.indexOf(self.selector) + 1)
        self.assertLess(self.selector.geometry().right(), self.presenter.rename_button.geometry().left())
        self.assertEqual(self.presenter.rename_button.accessibleName(), 'Rename pipeline')
        self.assertFalse(any('lock' in box.text().casefold() for box in self.window.findChildren(QCheckBox)))
        self.assertFalse(hasattr(self.documents, 'set_frozen'))

    def check_test_load_and_save_icons_sit_beside_rename_and_offer_no_copy(self):
        self.window.resize(1440, 900)
        self.window.show()
        self.application.processEvents()
        presenter = self.presenter
        layout = presenter.strip.layout()
        rename = layout.indexOf(presenter.rename_button)
        self.assertEqual(layout.indexOf(presenter.load_button), rename + 1)
        self.assertEqual(layout.indexOf(presenter.save_button), rename + 2)

        # Each is a compact, icon-only, named command in the same flat colour.
        images = []
        for button, icon, name in ((presenter.rename_button, rename_icon(), 'Rename pipeline'),
                                   (presenter.load_button, load_icon(), 'Load pipeline'),
                                   (presenter.save_button, save_icon(), 'Save pipeline')):
            self.assertEqual(button.text(), '')
            self.assertFalse(button.icon().isNull())
            self.assertEqual(button.iconSize(), COMPACT_ICON_SIZE)
            self.assertLessEqual(button.width(), 30)
            self.assertEqual(button.accessibleName(), name)
            self.assertTrue(button.toolTip().startswith(name))
            image = icon.pixmap(COMPACT_ICON_SIZE).toImage()
            opaque = {image.pixelColor(x, y).name() for x in range(image.width()) for y in range(image.height())
                      if image.pixelColor(x, y).alpha() == 255}
            self.assertEqual(opaque, {'#f2f6fa'})
            images.append(image)
        self.assertEqual(len({bytes(image.constBits()) for image in images}), 3)

        # On a default pipeline neither asks about a copy, and the factory name rule holds.
        destination = self.root / 'saved.seedfiddle-pipeline.json'
        with (answers() as (question, _),
              patch.object(QFileDialog, 'getOpenFileName', return_value=('', '')) as choose):
            presenter.load_button.click()
        choose.assert_called_once()
        question.assert_not_called()
        self.assert_locked_default()
        with (answers(names=['Procedural seed analysis']) as (question, _),
              patch.object(QFileDialog, 'getSaveFileName', return_value=(str(destination), '')),
              patch.object(QMessageBox, 'warning') as warning):
            presenter.save_button.click()
        warning.assert_called_once()
        question.assert_not_called()
        self.assertFalse(destination.exists())
        self.assert_locked_default()
        with (answers(names=['Saved seeds']) as (question, _),
              patch.object(QFileDialog, 'getSaveFileName', return_value=(str(destination), ''))):
            presenter.save_button.click()
        question.assert_not_called()
        self.assertTrue(destination.is_file())
        self.assert_on_copy('Saved seeds')

        presenter.select_template(None)
        with (answers() as (question, _),
              patch.object(QFileDialog, 'getOpenFileName', return_value=(str(destination), ''))):
            presenter.load_button.click()
        question.assert_not_called()
        self.assert_on_copy('Saved seeds 2')

    def check_test_more_menu_holds_only_node_removal_and_export_when_narrow(self):
        menu = self.presenter.more_button.menu()
        self.assertEqual([action.text() for action in menu.actions()],
                         ['Export run…', 'Remove selected added node'])
        self.window.resize(1440, 900)
        self.window.show()
        self.application.processEvents()
        self.assertFalse(self.presenter.export_button.isHidden())
        self.assertEqual([action.text() for action in menu.actions() if action.isVisible()],
                         ['Remove selected added node'])
        self.window.resize(1050, 650)
        self.application.processEvents()
        self.assertEqual(self.presenter.export_menu_action.isVisible(), self.presenter.export_button.isHidden())
        for removed in ('open_saved', 'show_source', 'show_qc', 'replay', 'replay_completed', 'replay_action', 'save'):
            self.assertFalse(hasattr(self.presenter, removed), removed)
        self.assertFalse(hasattr(self.window.catalogue_nodes, 'show_source_selected'))

    def offered(self):
        """What the open list shows, and what a person can pick from it."""
        view, model = self.selector.view(), self.selector.model()
        shown = [row for row in range(self.selector.count()) if not view.isRowHidden(row)]
        return ([self.selector.itemText(row) for row in shown],
                [self.selector.itemText(row) for row in shown if model.item(row).isEnabled()])

    def check_test_list_offers_the_default_pipelines_then_the_project_copies(self):
        defaults = ['Procedural seed analysis', 'Cellpose seed analysis']
        menus = {entry.text().replace('&', ''): entry for entry in self.window.menuBar().actions()}
        analysis = {entry.text() for entry in menus['Analysis'].menu().actions()}
        self.assertNotIn('Settings', analysis)
        shown, selectable = self.offered()
        self.assertEqual(selectable, defaults)
        self.assertEqual(shown, ['DEFAULT PIPELINES', *defaults])
        # The wheel reaches each default and nothing else.
        self.selector.setFocus()
        self.assertEqual(self.selector.currentText(), defaults[0])
        self.selector.wheelEvent(QWheelEvent(QPointF(), QPointF(), QPoint(), QPoint(0, -120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
        self.assertEqual(self.selector.currentText(), defaults[1])
        self.presenter.select_template('procedural_seed_analysis')
        with answers(names=['My copy']):
            self.presenter.rename_button.click()
        shown, selectable = self.offered()
        self.assertEqual(selectable, [*defaults, 'My copy'])
        self.assertEqual(shown, ['DEFAULT PIPELINES', *defaults, 'SAVED IN THIS PROJECT', 'My copy'])

    def check_test_declining_keeps_the_default_and_never_asks_for_a_name(self):
        self.assert_locked_default()
        for control in (self.canvas.add_node_button, self.canvas.move_to_unused_button,
                        self.presenter.rename_button, self.presenter.remove_added_action):
            self.assertTrue(control.isEnabled(), control)
        instances = self.window.catalogue_nodes.node_instance_ids
        shelved = dict(self.window.pipeline.unused_nodes)
        self.assertIn(OPTIONAL_NODE, shelved)
        with answers(NO) as (question, prompt):
            self.assertFalse(self.window.catalogue_nodes.add_entry(ADDED))
            self.canvas._restore_unused_node(OPTIONAL_NODE)
            self.canvas.move_to_unused_button.click()
            self.assertFalse(self.window.catalogue_nodes.remove_selected())
            self.assertFalse(self.documents.rename_current())
        self.assertEqual(question.call_count, 5)
        prompt.assert_not_called()
        for call in question.call_args_list:
            self.assertIn('Default pipelines cannot be edited', call.args[2])
            self.assertIn('editable copy', call.args[2])
        self.assertEqual(self.window.catalogue_nodes.node_instance_ids, instances)
        self.assertEqual(dict(self.window.pipeline.unused_nodes), shelved)
        self.assertEqual(self.documents.custom, {})
        self.assertEqual(self.copy_names(), [])
        self.assertIn('was not changed', self.window.statusBar().currentMessage())
        self.assert_locked_default()

    def check_test_cancelling_the_name_leaves_the_default_and_drops_the_edit(self):
        instances = self.window.catalogue_nodes.node_instance_ids
        with answers(names=[('Unused name', False)]) as (question, prompt):
            self.assertFalse(self.window.catalogue_nodes.add_entry(ADDED))
        question.assert_called_once()
        prompt.assert_called_once()
        self.assertEqual(self.window.catalogue_nodes.node_instance_ids, instances)
        self.assertEqual(self.copy_names(), [])
        self.assertIn('was not changed', self.window.statusBar().currentMessage())
        self.assert_locked_default()

    def check_test_accepting_loads_the_copy_and_applies_the_add_that_asked(self):
        self.canvas.add_node_menu.popup(QPoint(0, 0))
        self.application.processEvents()
        with answers(names=['My seeds']) as (question, prompt):
            self.canvas._add_catalogue_item(self.canvas.catalogue_entry_items()[0])
        question.assert_called_once()
        self.assertEqual(prompt.call_args.kwargs['text'], 'Procedural seed analysis copy')
        self.assert_on_copy('My seeds')
        self.assertEqual(len(self.window.catalogue_nodes.instances), 1)
        self.assertEqual(self.copy_names(), ['My seeds'])
        # Later edits ask nothing, and the default pipeline remains listed.
        with answers() as (question, _):
            self.window.catalogue_nodes.add_entry(ADDED)
        question.assert_not_called()
        self.assertEqual(len(self.window.catalogue_nodes.instances), 2)
        self.assertGreaterEqual(self.presenter.index_of(None), 0)

    def check_test_accepting_applies_move_to_unused_and_remove(self):
        with answers(names=['Restored']):
            self.canvas._restore_unused_node(OPTIONAL_NODE)
        self.assert_on_copy('Restored')
        self.assertNotIn(OPTIONAL_NODE, self.window.pipeline.unused_nodes)

        # A project without a named copy can open locked on a graph holding these nodes.
        self.canvas.select_node(OPTIONAL_NODE)
        self.documents.loaded('Procedural seed analysis')
        with answers(NO):
            self.canvas.move_to_unused_button.click()
        self.assertNotIn(OPTIONAL_NODE, self.window.pipeline.unused_nodes)
        with answers(names=['Shelved']):
            self.canvas.move_to_unused_button.click()
        self.assert_on_copy('Shelved')
        self.assertIn(OPTIONAL_NODE, self.window.pipeline.unused_nodes)

        added = self.window.catalogue_nodes.add_entry(ADDED)
        self.canvas.select_node(added)
        self.documents.loaded('Procedural seed analysis')
        self.assertIsNotNone(self.window.catalogue_nodes.selected_instance)
        with answers(NO):
            self.presenter.remove_added_action.trigger()
        self.assertEqual(len(self.window.catalogue_nodes.instances), 1)
        with answers(names=['Removed']):
            self.presenter.remove_added_action.trigger()
        self.assert_on_copy('Removed')
        self.assertEqual(len(self.window.catalogue_nodes.instances), 0)

    def check_test_refused_names_ask_again_with_the_reason_and_typed_text(self):
        self.make_copy('Taken')
        self.presenter.select_template(None)
        self.assertTrue(self.documents.locked)
        with answers(names=['   ', 'procedural seed ANALYSIS', 'taken', 'Bad\nname', '  Fresh  ']) as (_, prompt):
            self.assertTrue(self.documents.rename_current())
        labels = [call.args[2] for call in prompt.call_args_list]
        texts = [call.kwargs['text'] for call in prompt.call_args_list]
        self.assertEqual(labels[0], 'Pipeline name:')
        self.assertIn('Enter a name', labels[1])
        self.assertIn('is a default pipeline', labels[2])
        self.assertIn('already has a pipeline called taken', labels[3])
        self.assertNotEqual(labels[4], 'Pipeline name:')
        self.assertEqual(texts[3:], ['taken', 'Bad\nname'])
        self.assert_on_copy('Fresh')
        self.assertEqual(self.copy_names(), ['Taken', 'Fresh'])

    def check_test_rename_offers_a_copy_on_a_default_and_renames_a_copy(self):
        with answers(NO) as (question, prompt):
            self.presenter.rename_button.click()
        question.assert_called_once()
        prompt.assert_not_called()
        self.assert_locked_default()
        with answers(names=['First']) as (question, _):
            self.presenter.rename_button.click()
        question.assert_called_once()
        self.assert_on_copy('First')

        # A copy renames in place, without asking about copies.
        self.window._set_project_dirty(False)
        with answers(names=['Second']) as (question, prompt):
            self.presenter.rename_button.click()
        question.assert_not_called()
        self.assertEqual(prompt.call_args.kwargs['text'], 'First')
        self.assert_on_copy('Second')
        self.assertEqual(self.copy_names(), ['Second'])
        self.assertTrue(self.window._project_dirty)
        # Cancelling, or keeping the name, changes nothing.
        with answers(names=[('Third', False)]):
            self.assertFalse(self.documents.rename_current())
        with answers(names=['Second']):
            self.assertFalse(self.documents.rename_current())
        self.assert_on_copy('Second')

    def check_test_switching_keeps_copies_and_defaults_stay_factory(self):
        arranged = self.default_positions()
        self.make_copy('Plain')
        self.window._pipeline_parameter_changed(DEFAULT_NODE, DEFAULT_KEY, 1728)
        self.window.pipeline.node(DEFAULT_NODE).x += 500
        moved = self.default_positions()
        self.window.catalogue_nodes.add_entry(ADDED)

        self.presenter.select_template(None)
        self.assertEqual(self.documents.custom['Plain'].name, 'Plain')
        self.assert_locked_default()
        self.assertEqual(self.default_value(), DEFAULT_FACTORY)
        self.assertEqual(self.default_positions(), arranged)
        self.assertEqual(self.window.catalogue_nodes.instances, ())
        self.make_copy('Second')
        self.window._pipeline_parameter_changed(DEFAULT_NODE, DEFAULT_KEY, 1280)

        self.show_copy('Plain')
        self.assert_on_copy('Plain')
        self.assertEqual(self.default_value(), 1728)
        self.assertEqual(self.default_positions(), moved)
        self.assertEqual(len(self.window.catalogue_nodes.instances), 1)
        self.show_copy('Second')
        self.assert_on_copy('Second')
        self.assertEqual(self.default_value(), 1280)
        self.assertEqual(self.window.catalogue_nodes.instances, ())
        self.presenter.select_template(None)
        self.assertEqual(self.default_value(), DEFAULT_FACTORY)
        self.assertEqual(self.default_positions(), arranged)

    def check_test_project_keeps_every_copy_and_reopens_on_the_current_one(self):
        destination = self.root / 'copies.seedfiddle-project.json'
        self.make_copy('Plain')
        self.window._pipeline_parameter_changed(DEFAULT_NODE, DEFAULT_KEY, 1728)
        self.presenter.select_template(None)
        self.make_copy('Second')
        self.window._pipeline_parameter_changed(DEFAULT_NODE, DEFAULT_KEY, 1280)
        self.save_project(destination)
        payload = json.loads(destination.read_text(encoding='utf-8'))
        self.assertEqual(payload['active_pipeline'], 'Second')
        self.assertEqual([copy['name'] for copy in payload['custom_pipelines']], ['Plain'])

        self.presenter.select_template(None)
        self.open_project(destination)
        self.assert_on_copy('Second')
        self.assertEqual(self.default_value(), 1280)
        self.assertEqual(self.copy_names(), ['Plain', 'Second'])
        self.show_copy('Plain')
        self.assertEqual(self.default_value(), 1728)

    def check_test_project_without_copies_saves_no_copy_keys_and_opens_locked(self):
        destination = self.root / 'default.seedfiddle-project.json'
        self.save_project(destination)
        payload = json.loads(destination.read_text(encoding='utf-8'))
        self.assertNotIn('active_pipeline', payload)
        self.assertNotIn('custom_pipelines', payload)
        self.make_copy('Discarded')
        self.open_project(destination)
        self.assert_locked_default()
        self.assertEqual(self.copy_names(), [])

    def check_test_lock_notice_settings_profile_and_finished_work_respect_the_lock(self):
        self.window.pipeline_canvas.select_node(DEFAULT_NODE)
        notice = self.window.pipeline_inspector.lock_notice
        self.assertIn('Default pipelines cannot be edited', notice.text())
        profile = save_analysis_settings_profile(self.root / 'edited.seedfiddle-settings.json',
                                                 self.edited_default_graph())
        # Declining the offer, after agreeing to replace the settings, leaves them alone.
        with patch.object(QMessageBox, 'question', side_effect=[YES, NO]):
            self.assertFalse(self.window._load_analysis_settings_profile(profile))
        self.assertEqual(self.default_value(), DEFAULT_FACTORY)
        # Work finishing later must not raise a prompt on its own.
        with answers(NO) as (question, _):
            self.assertFalse(self.window._pipeline_editable())
        question.assert_not_called()

        with (patch.object(QMessageBox, 'question', side_effect=[YES, YES]),
              patch.object(QInputDialog, 'getText', return_value=('Profiled', True))):
            self.assertTrue(self.window._load_analysis_settings_profile(profile))
        self.assert_on_copy('Profiled')
        self.assertEqual(self.default_value(), 1728)
        self.assertTrue(self.window._pipeline_editable())

        self.presenter.select_template(None)
        with answers(names=['From the notice']) as (question, _):
            notice.linkActivated.emit('copy')
        question.assert_called_once()
        self.assert_on_copy('From the notice')

    def edited_default_graph(self):
        graph = build_default_pipeline()
        graph.node(DEFAULT_NODE).parameters[DEFAULT_KEY] = 1728
        return graph

    def save_project(self, destination):
        with (patch.object(QFileDialog, 'getSaveFileName', return_value=(str(destination), '')),
              patch.object(QMessageBox, 'warning'), patch.object(QMessageBox, 'critical') as critical):
            self.window.save_project_action.trigger()
            critical.assert_not_called()
        self.assertTrue(destination.is_file())

    def open_project(self, destination):
        with (patch.object(self.window, '_confirm_project_replacement', return_value=True),
              patch.object(QMessageBox, 'warning'), patch.object(QMessageBox, 'critical') as critical):
            self.assertTrue(self.window._open_project(destination))
            critical.assert_not_called()

    def test_workflow_lifecycle(self):
        """9 related DefaultPipelineCopyTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_list_offers_the_default_pipelines_then_the_project_copies)
        self.scenario(self.check_test_load_and_save_icons_sit_beside_rename_and_offer_no_copy)
        self.scenario(self.check_test_more_menu_holds_only_node_removal_and_export_when_narrow)
        self.scenario(self.check_test_add_node_names_line_up_whether_used_or_not)
        self.scenario(self.check_test_selector_is_wider_with_rename_beside_it_and_no_lock_control)
        self.scenario(self.check_test_accepting_loads_the_copy_and_applies_the_add_that_asked)
        self.scenario(self.check_test_accepting_applies_move_to_unused_and_remove)
        self.scenario(self.check_test_rename_offers_a_copy_on_a_default_and_renames_a_copy)
        self.scenario(self.check_test_switching_keeps_copies_and_defaults_stay_factory)

    def test_refusal_and_recovery(self):
        """6 related DefaultPipelineCopyTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_declining_keeps_the_default_and_never_asks_for_a_name)
        self.scenario(self.check_test_cancelling_the_name_leaves_the_default_and_drops_the_edit)
        self.scenario(self.check_test_refused_names_ask_again_with_the_reason_and_typed_text)
        self.scenario(self.check_test_project_keeps_every_copy_and_reopens_on_the_current_one)
        self.scenario(self.check_test_project_without_copies_saves_no_copy_keys_and_opens_locked)
        self.scenario(self.check_test_lock_notice_settings_profile_and_finished_work_respect_the_lock)


if __name__ == '__main__':
    unittest.main()
