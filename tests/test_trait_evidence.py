"""Trait training from a project's own labels, and models kept with the project."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvision.learning.trait_evidence import (
    CALIBRATION, COAT_PATTERN, TRAIN, TraitImage, TraitImageSource, TraitSeed, build_trait_manifest,
    default_trait_splits, evidence_problem, new_trait_model_path, read_trait_image, seed_label,
    target_seeds, target_title, trait_evidence_path, trait_targets,
)
from seedvision.learning.trait_training import (
    TraitTrainingConfiguration, TraitTrainingRequest, load_frozen_trait_model, read_trait_manifest,
    train_trait_manifest,
)
from seedvision.persistence.project_annotations import copy_trait_model
from seedvision.persistence.reference_regions import (
    ReferenceRegionBundle, ReferenceRegionStore, SeedInstanceAnnotation, file_sha256,
)
from tests.functional_units import functional_units
from tests.review_fixtures import dispose_window, workspace_window

SPLIT = 'condition:split'


def annotated(seed_id, **changes):
    values = dict(shape_reviewed=True, outline_visibility='complete', pose='flat')
    return SeedInstanceAnnotation(seed_id, **{**values, **changes})


def labelled(key, seeds, *, group=None):
    """A read image for rules that need no archive: (label, reason) per seed."""
    return TraitImage(
        TraitImageSource(key, key, Path(f'{key}.png'), Path(f'{key}.npz'), group=group or key),
        tuple(TraitSeed(index, label, (), False, None, reason) for index, (label, reason) in enumerate(seeds, 1)),
        'a' * 64, 'b' * 64)


class _Workspace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / 'config').mkdir()
        (self.root / 'images').mkdir()
        shutil.copyfile(Path(__file__).parents[1] / 'config' / 'traits.json', self.root / 'config' / 'traits.json')
        self.definitions = self.root / 'config' / 'traits.json'
        self.store = ReferenceRegionStore(self.root / 'store')

    def write_image(self, name, annotations, *, transform=np.eye(3), brightness=None, edge_seed=None):
        """Save an image whose seeds are 4x4 blocks, one per annotation, and their archive."""
        path = self.root / 'images' / f'{name}.png'
        image = np.full((40, 40, 3), 40, np.uint8)
        image[0, 0, 0] = sum(map(ord, name)) % 255
        labels = np.zeros((40, 40), np.int32)
        for index, annotation in enumerate(annotations):
            top, left = 3 + 6 * (index // 5), 3 + 6 * (index % 5)
            labels[top:top + 4, left:left + 4] = annotation.seed_id
            image[top:top + 4, left:left + 4, 2] = 30 if brightness is None else brightness(annotation)
        if edge_seed is not None:
            labels[0, 0] = edge_seed
        self.assertTrue(cv2.imwrite(str(path), image))
        archive = self.store.save(path, ReferenceRegionBundle(
            labels.shape, annotated_seeds=labels, source_to_corrected=transform,
            seed_annotations=tuple(annotations), annotation_snapshot_source_sha256=file_sha256(path)))
        return path, archive

    def read(self, name, annotations, *, group=None, **changes):
        path, archive = self.write_image(name, annotations, **changes)
        return read_trait_image(TraitImageSource(name, name, path, archive, group=group or name), self.store)


@functional_units
class TraitEvidenceTests(_Workspace):
    def check_seeds_are_counted_by_class_and_skip_reason(self):
        image = self.read('lot', (
            annotated(1, coat_pattern='white'), annotated(2, coat_pattern='white'),
            annotated(3, coat_pattern='banded_dark'),
            annotated(4, coat_pattern='white', shape_reviewed=False, outline_visibility='unknown'),
            annotated(5, coat_pattern='white', outline_visibility='partly_occluded'),
            annotated(6, coat_pattern='white', shape_exclusion_reason='chipped'),
            annotated(7, coat_pattern='white'), annotated(8)), edge_seed=7)
        self.assertIsNone(image.problem)
        self.assertEqual({seed.seed_id for seed in image.seeds}, set(range(1, 8)))  # 8 has no trait label.
        usable, skipped = target_seeds(COAT_PATTERN, image)
        self.assertEqual(usable, {'white': 2, 'banded_dark': 1})
        self.assertEqual(skipped, {'outline or pose not set': 1, 'outline not complete': 1,
                                   'excluded from modelling': 1, 'touches the image edge': 1})

    def check_seed_beyond_the_decoded_image_is_skipped(self):
        image = self.read('shifted', (annotated(1, coat_pattern='white'),),
                          transform=np.array(((1., 0., 20.), (0., 1., 0.), (0., 0., 1.))))
        self.assertEqual(image.seeds[0].exclusion, 'outside the decoded image')

    def check_conditions_say_nothing_about_unreviewed_seeds(self):
        reviewed_clean = TraitSeed(1, None, (), True, None, None)
        split = TraitSeed(2, None, ('split',), True, None, None)
        stained = TraitSeed(3, None, ('stained',), True, None, None)
        unreviewed = TraitSeed(4, 'white', (), False, None, None)
        self.assertEqual([seed_label(SPLIT, seed) for seed in (reviewed_clean, split, stained, unreviewed)],
                         ['not_split', 'split', 'not_split', None])
        image = TraitImage(TraitImageSource('a', 'a', Path('a.png'), None), (reviewed_clean, split, stained, unreviewed))
        self.assertEqual(trait_targets((image,)), (COAT_PATTERN, 'condition:split', 'condition:stained'))
        self.assertEqual(target_title(SPLIT), 'Condition: split')
        self.assertEqual(trait_targets((replace(image, seeds=(reviewed_clean,)),)), ())

    def check_unreadable_archive_is_that_images_problem(self):
        path, archive = self.write_image('broken', (annotated(1, coat_pattern='white'),))
        archive.write_bytes(b'not an archive')
        image = read_trait_image(TraitImageSource('broken', 'broken', path, archive), self.store)
        self.assertIn('could not be read', image.problem)
        self.assertEqual(image.seeds, ())
        draft = TraitImageSource('draft', 'draft', path, archive, problem='Apply the draft.')
        self.assertEqual(read_trait_image(draft, self.store).problem, 'Apply the draft.')

    def check_readiness_says_what_is_missing(self):
        both = labelled('both', (('white', None), ('dark', None)))
        white = labelled('white', (('white', None),))
        dark = labelled('dark', (('dark', None),))
        for images, splits, expected in (
            ((both,), {}, 'No image'),
            ((labelled('skipped', (('white', 'outline not complete'),)),), {'skipped': TRAIN}, 'No image'),
            ((white,), {'white': TRAIN}, 'two classes'),
            ((white, dark), {'white': TRAIN, 'dark': CALIBRATION}, 'Dark has no seeds in a Training'),
            ((both,), {'both': TRAIN}, 'Calibration'),
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, evidence_problem(COAT_PATTERN, images, splits))
        self.assertIsNone(evidence_problem(COAT_PATTERN, (both, white), {'both': TRAIN, 'white': CALIBRATION}))

    def check_default_splits_hold_out_a_lot_and_keep_every_class_in_training(self):
        large = labelled('large', [('white', None)] * 9 + [('dark', None)] * 9)
        small = labelled('small', (('white', None), ('dark', None)))
        only_rare = labelled('rare', (('rare', None),))
        self.assertEqual(default_trait_splits(COAT_PATTERN, (large, small)), {'large': TRAIN, 'small': CALIBRATION})
        # The rare class exists on one image only, so that image has to train.
        self.assertEqual(default_trait_splits(COAT_PATTERN, (large, small, only_rare))['rare'], TRAIN)
        # Two images of one lot stay together.
        first = labelled('first', (('white', None), ('dark', None)), group='lot')
        second = labelled('second', (('white', None),), group='lot')
        splits = default_trait_splits(COAT_PATTERN, (large, first, second))
        self.assertEqual(splits['first'], splits['second'])
        # One lot has nothing to hold out, which readiness then explains.
        self.assertEqual(default_trait_splits(COAT_PATTERN, (large,)), {'large': TRAIN})
        unusable = labelled('unusable', (('white', 'outline not complete'),))
        self.assertNotIn('unusable', default_trait_splits(COAT_PATTERN, (large, unusable)))

    def check_manifest_passes_the_training_checks_without_unusable_seeds(self):
        training = self.read('train', (annotated(1, coat_pattern='white', physical_seed_id='seed-a'),
                                       annotated(2, coat_pattern='banded_dark'),
                                       annotated(3, coat_pattern='white', outline_visibility='image_cutoff')))
        calibration = self.read('calibrate', (annotated(1, coat_pattern='white'),))
        payload = build_trait_manifest(COAT_PATTERN, (training, calibration), {'train': TRAIN, 'calibrate': CALIBRATION},
                                       definition_source=self.definitions)
        path = self.root / 'evidence.json'
        path.write_text(json.dumps(payload), encoding='utf-8')
        manifest = read_trait_manifest(path).payload
        self.assertEqual([item['example_id'] for item in manifest['examples']],
                         ['train-seed-1', 'train-seed-2', 'calibrate-seed-1'])
        self.assertEqual(manifest['examples'][0]['physical_id'], 'seed-a')
        self.assertEqual([item['identifier'] for item in manifest['vocabulary']['traits']], ['banded_dark', 'white'])
        self.assertEqual(manifest['vocabulary']['definition_source_sha256'], file_sha256(self.definitions))
        excluded = build_trait_manifest(COAT_PATTERN, (training, calibration), {'train': TRAIN, 'calibrate': ''},
                                        definition_source=self.definitions)
        self.assertEqual({item['split'] for item in excluded['examples']}, {TRAIN})

    def check_new_model_files_never_reuse_a_name(self):
        first = new_trait_model_path(self.root, 'coat-pattern')
        first.write_text('{}')
        second = new_trait_model_path(self.root, 'coat-pattern')
        self.assertNotEqual(first, second)
        self.assertEqual(trait_evidence_path(first).name, first.stem + '.evidence.json')

    def check_copying_a_model_keeps_its_evidence_and_other_models(self):
        source = self.root / 'trained' / 'model.json'
        source.parent.mkdir()
        source.write_text('first')
        trait_evidence_path(source).write_text('evidence')
        folder = self.root / 'project' / 'trait-models'
        copied = copy_trait_model(source, folder)
        self.assertEqual(copied.read_text(), 'first')
        self.assertEqual(trait_evidence_path(copied).read_text(), 'evidence')
        self.assertEqual(copy_trait_model(source, folder), copied)
        source.write_text('second')
        other = copy_trait_model(source, folder)
        self.assertNotEqual(other, copied)
        self.assertEqual((copied.read_text(), other.read_text()), ('first', 'second'))

    def test_project_evidence(self):
        """9 related TraitEvidenceTests scenarios, each with private fixture state."""
        self.scenario(self.check_seeds_are_counted_by_class_and_skip_reason)
        self.scenario(self.check_seed_beyond_the_decoded_image_is_skipped)
        self.scenario(self.check_conditions_say_nothing_about_unreviewed_seeds)
        self.scenario(self.check_unreadable_archive_is_that_images_problem)
        self.scenario(self.check_readiness_says_what_is_missing)
        self.scenario(self.check_default_splits_hold_out_a_lot_and_keep_every_class_in_training)
        self.scenario(self.check_manifest_passes_the_training_checks_without_unusable_seeds)
        self.scenario(self.check_new_model_files_never_reuse_a_name)
        self.scenario(self.check_copying_a_model_keeps_its_evidence_and_other_models)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for native trait training')
@functional_units
class TraitEvidenceTrainingTests(_Workspace):
    def model(self):
        """Train a real model on project-style annotations, as the dialog's request does."""
        tone = lambda item: 30 if item.coat_pattern == 'white' else 220
        training = self.read('train', [annotated(index, coat_pattern='white' if index % 2 else 'banded_dark')
                                       for index in range(1, 7)], brightness=tone)
        calibration = self.read('calibrate', [annotated(index, coat_pattern='white' if index % 2 else 'banded_dark')
                                              for index in range(1, 5)], brightness=tone)
        payload = build_trait_manifest(COAT_PATTERN, (training, calibration), {'train': TRAIN, 'calibrate': CALIBRATION},
                                       definition_source=self.definitions)
        output = self.root / 'trained' / 'coat-pattern.json'
        output.parent.mkdir(exist_ok=True)
        manifest = trait_evidence_path(output)
        manifest.write_text(json.dumps(payload), encoding='utf-8')
        report = train_trait_manifest(TraitTrainingRequest(
            manifest, output, TraitTrainingConfiguration(iterations=40, calibration_iterations=7),
            manifest_sha256=file_sha256(manifest)))
        return output, report

    def check_project_labels_train_a_model_that_loads(self):
        output, report = self.model()
        model = load_frozen_trait_model(output)
        self.assertEqual(model.model_sha256, report['model_sha256'])
        self.assertEqual([item['identifier'] for item in model.payload['vocabulary']['traits']], ['banded_dark', 'white'])

    def check_import_validates_copies_and_uses_the_model(self):
        output, _ = self.model()
        window = workspace_window(self.root)
        self.addCleanup(lambda: dispose_window(window))
        bad = self.root / 'bad.json'
        bad.write_text('{}')
        with patch('seedvision.ui.trait_training_workflow.QFileDialog.getOpenFileName', return_value=(str(bad), '')), \
             patch('seedvision.ui.trait_training_workflow.QMessageBox.warning') as warning:
            window._import_trait_model()
        warning.assert_called_once()
        self.assertFalse(window._trait_model_directory().exists())
        self.assertEqual(window.pipeline.node('trait_classifier').parameters['trait_model_path'], 'None')
        with patch('seedvision.ui.trait_training_workflow.QFileDialog.getOpenFileName', return_value=(str(output), '')), \
             patch.object(window, '_pipeline_editing_allowed', return_value=True):
            window._import_trait_model()
        node = window.pipeline.node('trait_classifier')
        self.assertTrue(node.enabled)
        self.assertEqual(node.parameters['trait_model_path'], 'projects/trait-models/coat-pattern.json')
        self.assertEqual((self.root / node.parameters['trait_model_path']).read_bytes(), output.read_bytes())
        self.assertTrue((self.root / 'projects/trait-models/coat-pattern.evidence.json').is_file())

    def test_training_on_project_labels(self):
        """2 related TraitEvidenceTrainingTests scenarios, each with private fixture state."""
        self.scenario(self.check_project_labels_train_a_model_that_loads)
        self.scenario(self.check_import_validates_copies_and_uses_the_model)


@functional_units
class TraitModelWindowTests(_Workspace):
    def window(self):
        window = workspace_window(self.root)
        self.addCleanup(lambda: dispose_window(window))
        return window

    def dialog(self, images=()):
        from seedvision.ui.trait_training_workflow import TraitTrainingDialog
        dialog = TraitTrainingDialog(project_root=self.root, model_directory=self.root / 'models', images=images)
        self.addCleanup(dialog.close)
        return dialog

    def project_images(self):
        return (self.read('large', [annotated(index, coat_pattern='white' if index % 2 else 'banded_dark')
                                    for index in range(1, 9)]),
                self.read('small', (annotated(1, coat_pattern='white'), annotated(2, coat_pattern='banded_dark'),
                                    annotated(3, coat_pattern='white', outline_visibility='partly_occluded'))))

    def check_first_attempt_without_labels_explains_instead_of_failing(self):
        dialog = self.dialog()
        self.assertFalse(dialog.train_button.isEnabled())
        self.assertIn('No seed in this project has a trait label', dialog.problem_label.text())
        self.assertFalse(dialog.problem_label.isHidden())
        self.assertEqual(dialog.manifest_row.edit.text(), '')
        self.assertFalse(dialog.manifest_row.isEnabled())
        dialog.evidence_file_checkbox.setChecked(True)
        self.assertEqual(dialog.problem_label.text(), 'Choose an evidence file.')
        self.assertFalse(dialog.train_button.isEnabled())
        missing = str(self.root / 'absent.json')
        dialog.manifest_row.edit.setText(missing)
        self.assertTrue(dialog.train_button.isEnabled())
        with patch('seedvision.ui.trait_training_workflow.QMessageBox.warning') as warning:
            dialog.accept()
        warning.assert_called_once()
        self.assertIsNone(dialog.request())

    def check_first_attempt_with_labels_needs_no_files_chosen(self):
        dialog = self.dialog(self.project_images())
        self.assertEqual(dialog.target_combo.currentData(), COAT_PATTERN)
        self.assertEqual(dialog.image_table.rowCount(), 2)
        table = dialog.image_table
        self.assertEqual(table.item(0, dialog.SEEDS_COLUMN).text(), 'Banded dark 4, White 4')
        self.assertEqual(table.cellWidget(0, dialog.SPLIT_COLUMN).currentData(), TRAIN)
        self.assertEqual(table.cellWidget(1, dialog.SPLIT_COLUMN).currentData(), CALIBRATION)
        self.assertIn('Outline not complete 1', table.item(1, dialog.NOTE_COLUMN).text())
        self.assertEqual(dialog.class_summary.text(),
                         'Training: Banded dark 4, White 4. Calibration: Banded dark 1, White 1.')
        self.assertTrue(dialog.train_button.isEnabled())
        dialog.accept()
        request = dialog.request()
        self.assertTrue(request.activate_after_training)
        self.assertEqual(request.output_path.parent, (self.root / 'models').resolve())
        self.assertTrue(request.output_path.name.startswith('coat-pattern-'))
        self.assertEqual(request.manifest_path, trait_evidence_path(request.output_path))
        self.assertEqual(request.manifest_sha256, file_sha256(request.manifest_path))
        self.assertEqual(len(read_trait_manifest(request.manifest_path).payload['examples']), 10)

    def check_splits_keep_a_lot_together_and_update_readiness(self):
        first = labelled('first', (('white', None), ('dark', None)), group='lot')
        second = labelled('second', (('white', None),), group='lot')
        other = labelled('other', (('white', None), ('dark', None)), group='other')
        dialog = self.dialog((first, second, other))
        combos = {image.source.key: dialog.image_table.cellWidget(row, dialog.SPLIT_COLUMN)
                  for row, image in enumerate(dialog._rows)}
        self.assertEqual(combos['first'].currentData(), combos['second'].currentData())
        combos['first'].setCurrentIndex(combos['first'].findData(CALIBRATION))
        self.assertEqual(combos['second'].currentData(), CALIBRATION)
        combos['other'].setCurrentIndex(combos['other'].findData(''))
        self.assertFalse(dialog.train_button.isEnabled())
        self.assertIn('Training image', dialog.problem_label.text())
        combos['other'].setCurrentIndex(combos['other'].findData(TRAIN))
        self.assertTrue(dialog.train_button.isEnabled())
        self.assertTrue(dialog.problem_label.isHidden())

    def check_changing_the_trait_lists_only_images_that_speak_to_it(self):
        clean = TraitImage(TraitImageSource('clean', 'clean', Path('clean.png'), Path('clean.npz'), group='clean'),
                           (TraitSeed(1, None, (), True, None, None),), 'a' * 64, 'b' * 64)
        split = TraitImage(TraitImageSource('split', 'split', Path('split.png'), Path('split.npz'), group='split'),
                           (TraitSeed(1, None, ('split',), True, None, None), TraitSeed(2, None, (), True, None, None)),
                           'a' * 64, 'b' * 64)
        coat = labelled('coat', (('white', None),))
        dialog = self.dialog((clean, split, coat))
        self.assertEqual([dialog.target_combo.itemData(index) for index in range(dialog.target_combo.count())],
                         [COAT_PATTERN, SPLIT])
        self.assertEqual([image.source.key for image in dialog._rows], ['coat'])
        self.assertIn('two classes', dialog.problem_label.text())
        dialog.target_combo.setCurrentIndex(1)
        self.assertEqual([image.source.key for image in dialog._rows], ['clean', 'split'])
        self.assertEqual(dialog.image_table.cellWidget(0, dialog.SPLIT_COLUMN).currentData(), CALIBRATION)
        self.assertEqual(dialog.image_table.cellWidget(1, dialog.SPLIT_COLUMN).currentData(), TRAIN)
        self.assertTrue(dialog.train_button.isEnabled())

    def check_window_reads_labels_and_blocks_drafts(self):
        from PySide6.QtWidgets import QDialog
        window = self.window()
        self.assertEqual(window._trait_model_directory(), self.root / 'projects' / 'trait-models')
        with patch('seedvision.ui.trait_training_workflow.TraitTrainingDialog') as dialog:
            dialog.return_value.exec.return_value = QDialog.DialogCode.Rejected
            window._start_trait_training()
        self.assertEqual(dialog.call_args.kwargs['images'], ())
        self.assertEqual(dialog.call_args.kwargs['model_directory'], self.root / 'projects' / 'trait-models')
        self.store = window._reference_region_store
        path, _ = self.write_image('labelled', (annotated(1, coat_pattern='white'),))
        window._add_image(path, open_now=False, mark_project_dirty=False)
        key = next(item for item, value in window._image_paths.items() if Path(value).resolve() == path.resolve())
        with patch('seedvision.ui.trait_training_workflow.TraitTrainingDialog') as dialog:
            dialog.return_value.exec.return_value = QDialog.DialogCode.Rejected
            window._start_trait_training()
        image = next(item for item in dialog.call_args.kwargs['images'] if item.source.key == key)
        self.assertEqual([seed.coat_pattern for seed in image.seeds], ['white'])
        window._instance_annotations_dirty.add(key)
        source = next(item for item in window._trait_image_sources() if item.key == key)
        self.assertIn('draft', source.problem)

    def check_trained_model_is_saved_with_the_project_and_follows_save_as(self):
        window = self.window()
        trained = window._trait_model_directory() / 'coat-pattern.json'
        trained.parent.mkdir(parents=True)
        trained.write_text('model bytes')
        trait_evidence_path(trained).write_text('evidence bytes')
        node = window.pipeline.node('trait_classifier')
        window.pipeline.set_parameter('trait_classifier', 'trait_model_path', 'projects/trait-models/coat-pattern.json')
        first = self.root / 'first.seedfiddle-project.json'
        self.assertTrue(window._write_project(first))
        kept = self.root / 'first.seedfiddle-annotations' / 'trait-models' / 'coat-pattern.json'
        self.assertEqual(kept.read_text(), 'model bytes')
        self.assertEqual(trait_evidence_path(kept).read_text(), 'evidence bytes')
        self.assertEqual(node.parameters['trait_model_path'], 'first.seedfiddle-annotations/trait-models/coat-pattern.json')
        revision = window.pipeline.revision
        self.assertTrue(window._write_project(first))
        self.assertEqual(window.pipeline.revision, revision)
        second = self.root / 'second.seedfiddle-project.json'
        self.assertTrue(window._write_project(second))
        self.assertEqual(node.parameters['trait_model_path'], 'second.seedfiddle-annotations/trait-models/coat-pattern.json')
        self.assertEqual((self.root / node.parameters['trait_model_path']).read_text(), 'model bytes')
        self.assertTrue(kept.is_file())
        reopened = self.window()
        with patch.object(reopened, '_confirm_project_replacement', return_value=True):
            self.assertTrue(reopened._open_project(second))
        self.assertEqual(reopened.pipeline.node('trait_classifier').parameters['trait_model_path'],
                         'second.seedfiddle-annotations/trait-models/coat-pattern.json')

    def check_save_leaves_a_missing_or_absent_model_alone(self):
        window = self.window()
        node = window.pipeline.node('trait_classifier')
        project = self.root / 'plain.seedfiddle-project.json'
        self.assertTrue(window._write_project(project))
        self.assertEqual(node.parameters['trait_model_path'], 'None')
        self.assertFalse((self.root / 'plain.seedfiddle-annotations' / 'trait-models').exists())
        window.pipeline.set_parameter('trait_classifier', 'trait_model_path', 'gone.json')
        self.assertTrue(window._write_project(project))
        self.assertEqual(node.parameters['trait_model_path'], 'gone.json')

    def check_export_copies_the_selected_model_or_says_there_is_none(self):
        window = self.window()
        with patch('seedvision.ui.trait_training_workflow.QFileDialog.getSaveFileName') as chooser:
            window._export_trait_model()
        chooser.assert_not_called()
        self.assertIn('no model', window.statusBar().currentMessage())
        model = self.root / 'selected.json'
        model.write_text('model bytes')
        window.pipeline.set_parameter('trait_classifier', 'trait_model_path', 'selected.json')
        destination = self.root / 'exported' / 'copy.json'
        with patch('seedvision.ui.trait_training_workflow.QFileDialog.getSaveFileName', return_value=(str(destination), '')):
            window._export_trait_model()
        self.assertEqual(destination.read_text(), 'model bytes')
        self.assertIn(str(destination), window.statusBar().currentMessage())

    def check_learning_menu_offers_import_and_export(self):
        from PySide6.QtWidgets import QMenu
        window = self.window()
        actions = {action.text() for menu in window.menuBar().findChildren(QMenu) for action in menu.actions()}
        self.assertTrue({'Train object-level seed traits…', 'Import seed-trait model…', 'Export seed-trait model…'} <= actions)
        self.assertFalse(any('Activate frozen' in text for text in actions))

    def test_dialog(self):
        """4 related TraitModelWindowTests scenarios, each with private fixture state."""
        self.scenario(self.check_first_attempt_without_labels_explains_instead_of_failing)
        self.scenario(self.check_first_attempt_with_labels_needs_no_files_chosen)
        self.scenario(self.check_splits_keep_a_lot_together_and_update_readiness)
        self.scenario(self.check_changing_the_trait_lists_only_images_that_speak_to_it)

    def test_models_with_the_project(self):
        """5 related TraitModelWindowTests scenarios, each with private fixture state."""
        self.scenario(self.check_window_reads_labels_and_blocks_drafts)
        self.scenario(self.check_trained_model_is_saved_with_the_project_and_follows_save_as)
        self.scenario(self.check_save_leaves_a_missing_or_absent_model_alone)
        self.scenario(self.check_export_copies_the_selected_model_or_says_there_is_none)
        self.scenario(self.check_learning_menu_offers_import_and_export)


if __name__ == '__main__':
    unittest.main()
