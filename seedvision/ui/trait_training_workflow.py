"""Train seed-trait models from a project's labels, and keep them with the project."""
from functools import partial
import json
from threading import Event
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QHeaderView, QLabel, QMessageBox, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)

from seedvision.annotation.seed_traits import trait_display_name
from seedvision.learning.trait_classifier import TraitTrainingCancelled
from seedvision.learning.trait_evidence import (
    CALIBRATION, TRAIN, TraitImageSource, build_trait_manifest, default_trait_splits,
    evidence_problem, new_trait_model_path, read_trait_images, seed_label, split_counts,
    target_file_stem, target_seeds, target_title, trait_evidence_path, trait_targets,
)
from seedvision.learning.trait_training import (
    TraitTrainingConfiguration, TraitTrainingRequest, load_frozen_trait_model,
    read_trait_manifest, train_trait_manifest, workspace_path_text,
)
from seedvision.persistence.project_analysis import REFERENCE_REGIONS_SIDECAR
from seedvision.persistence.project_annotations import (
    copy_annotation_file, copy_trait_model, project_annotation_directory, trait_model_folder,
)
from seedvision.ui.learning_workflow import _PathRow, _unique_names
from seedvision.ui.persistence_jobs import run_persistence

_NO_LABELS = ('No seed in this project has a trait label yet. On images from at least two seed lots, '
              'select seeds, set their coat pattern or condition, set Outline to Complete and choose '
              'a pose, then apply.')


def _counted(counts):
    return ', '.join(f'{trait_display_name(label)} {count:,}' for label, count in sorted(counts.items()))


class TraitTrainingDialog(QDialog):
    IMAGE_COLUMN, SPECIES_COLUMN, GROUP_COLUMN, SEEDS_COLUMN, SPLIT_COLUMN, NOTE_COLUMN = range(6)

    def __init__(self, *, project_root, model_directory, images=(), parent=None):
        super().__init__(parent)
        self.setWindowTitle('Train object-level seed traits')
        self.setMinimumWidth(760)
        self._definition_source = Path(project_root) / 'config' / 'traits.json'
        self._model_directory = Path(model_directory)
        self._images = tuple(images)
        self._rows = ()
        self._accepted_request = None
        layout = QVBoxLayout(self)
        target_form = QFormLayout()
        self.target_combo = QComboBox(self)
        for target in trait_targets(self._images):
            self.target_combo.addItem(target_title(target), target)
        self.target_combo.setEnabled(self.target_combo.count() > 0)
        target_form.addRow('Trait', self.target_combo)
        layout.addLayout(target_form)
        self.image_table = QTableWidget(0, 6, self)
        self.image_table.setHorizontalHeaderLabels(('Image', 'Species', 'Seed lot', 'Usable seeds', 'Split', 'Note'))
        self.image_table.verticalHeader().hide()
        self.image_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.image_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.image_table.horizontalHeader().setSectionResizeMode(self.NOTE_COLUMN, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.image_table)
        self.class_summary = QLabel(self)
        self.class_summary.setWordWrap(True)
        layout.addWidget(self.class_summary)
        self.problem_label = QLabel(self)
        self.problem_label.setObjectName('trait_training_problem')
        self.problem_label.setWordWrap(True)
        self.problem_label.setStyleSheet('font-weight: bold;')
        layout.addWidget(self.problem_label)
        form = QFormLayout()
        self.iterations_spin = QSpinBox(self)
        self.iterations_spin.setRange(1, 10000)
        self.iterations_spin.setValue(200)
        self.calibration_spin = QSpinBox(self)
        self.calibration_spin.setRange(2, 1024)
        self.calibration_spin.setValue(65)
        self.learning_rate_spin = self._number(.000001, 1., .03, 6)
        self.l2_spin = self._number(0., 100., .01, 6)
        self.lower_temperature_spin = self._number(.000001, 1000000., .05, 6)
        self.upper_temperature_spin = self._number(.000001, 1000000., 20., 6)
        self.threshold_spin = self._number(0., 1., .8, 4)
        form.addRow('Training iterations', self.iterations_spin)
        form.addRow('Learning rate', self.learning_rate_spin)
        form.addRow('L2 weight penalty', self.l2_spin)
        form.addRow('Temperature lower bound', self.lower_temperature_spin)
        form.addRow('Temperature upper bound', self.upper_temperature_spin)
        form.addRow('Calibration log-grid candidates', self.calibration_spin)
        form.addRow('Abstention threshold', self.threshold_spin)
        self.activate_checkbox = QCheckBox('Use the model in Object-level seed traits', self)
        self.activate_checkbox.setChecked(True)
        form.addRow('When training completes', self.activate_checkbox)
        self.evidence_file_checkbox = QCheckBox('Train from a prepared evidence file instead', self)
        form.addRow(self.evidence_file_checkbox)
        self.manifest_row = _PathRow(self)
        self.manifest_row.browse.clicked.connect(self._choose_manifest)
        form.addRow('Evidence file', self.manifest_row)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        self.train_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.train_button.setText('Train')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.target_combo.currentIndexChanged.connect(self._show_target)
        self.evidence_file_checkbox.toggled.connect(self._update_readiness)
        self.manifest_row.edit.textChanged.connect(self._update_readiness)
        self._show_target()

    def _number(self, lower, upper, value, decimals):
        spin = QDoubleSpinBox(self)
        spin.setDecimals(decimals)
        spin.setRange(lower, upper)
        spin.setValue(value)
        return spin

    def _choose_manifest(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Choose trait evidence', self.manifest_row.edit.text(), 'JSON (*.json)')
        if path:
            self.manifest_row.edit.setText(path)

    def _show_target(self):
        """List the images that say something about the chosen trait, with default splits."""
        target = self.target_combo.currentData()
        self._rows = tuple(image for image in self._images if image.problem is not None or (
            target is not None and any(seed_label(target, seed) is not None for seed in image.seeds)))
        splits = {} if target is None else default_trait_splits(target, self._rows)
        self.image_table.setRowCount(len(self._rows))
        for row, image in enumerate(self._rows):
            usable, skipped = ((), ()) if target is None else target_seeds(target, image)
            for column, text in ((self.IMAGE_COLUMN, image.source.path.name), (self.SPECIES_COLUMN, image.source.species),
                                 (self.GROUP_COLUMN, image.source.group), (self.SEEDS_COLUMN, _counted(usable) if usable else '0')):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.image_table.setItem(row, column, item)
            split = QComboBox(self.image_table)
            split.addItem('Training', TRAIN)
            split.addItem('Calibration', CALIBRATION)
            split.addItem('Not used', '')
            split.setCurrentIndex(max(0, split.findData(splits.get(image.source.key, ''))))
            split.setEnabled(image.source.key in splits)
            split.currentIndexChanged.connect(lambda _index, current=row: self._split_changed(current))
            self.image_table.setCellWidget(row, self.SPLIT_COLUMN, split)
            note = image.problem or (f'Skipped: {_counted(skipped)}' if skipped else '')
            item = QTableWidgetItem(note)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            item.setToolTip(note)
            self.image_table.setItem(row, self.NOTE_COLUMN, item)
        self.image_table.resizeRowsToContents()
        self._update_readiness()

    def _split_changed(self, changed):
        """Keep each seed lot on one side, because a lot may not train and calibrate."""
        group = self._rows[changed].source.group
        for row, image in enumerate(self._rows):
            combo = self.image_table.cellWidget(row, self.SPLIT_COLUMN)
            if row != changed and image.source.group == group and combo.isEnabled():
                with QSignalBlocker(combo):
                    combo.setCurrentIndex(self.image_table.cellWidget(changed, self.SPLIT_COLUMN).currentIndex())
        self._update_readiness()

    def _splits(self):
        return {image.source.key: self.image_table.cellWidget(row, self.SPLIT_COLUMN).currentData()
                for row, image in enumerate(self._rows)}

    def _update_readiness(self):
        from_file = self.evidence_file_checkbox.isChecked()
        target = self.target_combo.currentData()
        self.manifest_row.setEnabled(from_file)
        self.target_combo.setEnabled(not from_file and self.target_combo.count() > 0)
        self.image_table.setEnabled(not from_file)
        summary = ''
        if from_file:
            problem = None if self.manifest_row.edit.text().strip() else 'Choose an evidence file.'
        elif target is None:
            problem = _NO_LABELS
        else:
            problem = evidence_problem(target, self._rows, self._splits())
            counts = split_counts(target, self._rows, self._splits())
            summary = (f'Training: {_counted(counts[TRAIN]) or "no seeds"}. '
                       f'Calibration: {_counted(counts[CALIBRATION]) or "no seeds"}.')
        self.class_summary.setText(summary)
        self.class_summary.setVisible(bool(summary))
        self.problem_label.setText(problem or '')
        self.problem_label.setVisible(problem is not None)
        self.train_button.setEnabled(problem is None)

    def request(self):
        return self._accepted_request

    def _prepared_request(self):
        """Write the evidence beside the new model, and let the training rules read it first."""
        configuration = TraitTrainingConfiguration(
            self.iterations_spin.value(), self.learning_rate_spin.value(), self.l2_spin.value(),
            (self.lower_temperature_spin.value(), self.upper_temperature_spin.value()),
            self.calibration_spin.value(), self.threshold_spin.value())
        self._model_directory.mkdir(parents=True, exist_ok=True)
        if self.evidence_file_checkbox.isChecked():
            manifest_path = Path(self.manifest_row.edit.text().strip())
            output_path = new_trait_model_path(self._model_directory, 'seed-traits')
            manifest = read_trait_manifest(manifest_path)
        else:
            target = self.target_combo.currentData()
            output_path = new_trait_model_path(self._model_directory, target_file_stem(target))
            manifest_path = trait_evidence_path(output_path)
            payload = build_trait_manifest(target, self._rows, self._splits(), definition_source=self._definition_source)
            manifest_path.write_text(json.dumps(payload, indent=1), encoding='utf-8')
            try:
                manifest = read_trait_manifest(manifest_path)
            except ValueError:
                manifest_path.unlink(missing_ok=True)
                raise
        return TraitTrainingRequest(manifest_path, output_path, configuration,
                                    self.activate_checkbox.isChecked(), manifest.content_sha256)

    def accept(self):
        try:
            self._accepted_request = self._prepared_request()
        except (OSError, ValueError, TypeError, KeyError) as error:
            QMessageBox.warning(self, 'Trait evidence is not ready', str(error))
            return
        super().accept()


class _TraitTrainingSignals(QObject):
    progress = Signal(object, str, int, int)
    completed = Signal(object, object)
    failed = Signal(object, str)
    cancelled = Signal(object)


class TraitTrainingTask(QRunnable):
    def __init__(self, request):
        super().__init__()
        self.request = request
        self.signals = _TraitTrainingSignals()
        self._cancel = Event()
        self.pipeline_identity = None
        self.pipeline_revision = None

    def cancel(self):
        self._cancel.set()

    def cancellation_requested(self):
        return self._cancel.is_set()

    def run(self):
        try:
            report = train_trait_manifest(self.request, cancellation_requested=self.cancellation_requested,
                progress_callback=lambda stage, current, maximum: self.signals.progress.emit(self, stage, current, maximum))
        except TraitTrainingCancelled:
            self.signals.cancelled.emit(self)
        except Exception as error:
            self.signals.failed.emit(self, str(error))
        else:
            self.signals.completed.emit(self, report)


class TraitModelController:
    """Start trait training from the project's own labels, and keep models with the project."""

    def _trait_model_directory(self):
        project = self._current_project_path
        return trait_model_folder(project_annotation_directory(project) if project is not None else self._root / 'projects')

    def _trait_image_sources(self):
        """Describe each project image's current annotation archive for reading off the GUI thread."""
        archives = self._project_annotation_sources()
        images = tuple(Path(path).resolve() for path in self._image_paths.values())
        keys = tuple(self._image_paths)
        names = _unique_names(tuple(TraitImageSource(key, '', path, None) for key, path in zip(keys, images)))
        sources = []
        for key, path in zip(keys, images):
            context = self._project_image_biological_contexts.get(key)
            archive = archives.get((key, REFERENCE_REGIONS_SIDECAR))
            problem = None
            if key in self._instance_annotations_dirty or key in self._reference_masks_dirty:
                problem = 'Apply or discard the draft on this image first.'
            elif key in self._unsaved_reference_sidecars:
                problem = 'The last edit on this image was not written to disk. Save the project first.'
            sources.append(TraitImageSource(
                key, names[key], path, None if archive is None else archive[1],
                species=self._training_species_name(key, ''),
                group=(context.seed_lot_id if context is not None else None) or names[key], problem=problem))
        return tuple(sources)

    def _start_trait_training(self):
        if self._background_work_is_active():
            self.statusBar().showMessage('Wait for the current background work to finish.')
            return
        try:
            images = run_persistence(self, partial(read_trait_images, self._trait_image_sources(), self._reference_region_store),
                                     'Reading seed labels…')
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, 'Could not read seed labels', str(error))
            return
        dialog = TraitTrainingDialog(project_root=self._root, model_directory=self._trait_model_directory(),
                                     images=images, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self._begin_trait_training(dialog.request())
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, 'Trait training cannot start', str(error))

    def _selected_trait_model_path(self):
        selected = str(self.pipeline.node('trait_classifier').parameters['trait_model_path']).strip()
        if selected.casefold() == 'none':
            return None
        path = Path(selected).expanduser()
        return (path if path.is_absolute() else self._root / path).resolve()

    def _import_trait_model(self):
        """Copy a model from elsewhere into the project, then use it."""
        if self._background_work_is_active():
            self.statusBar().showMessage('Wait for the current background work to finish.')
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Import seed-trait model', str(self._root), 'JSON (*.json)')
        if not path:
            return
        try:
            load_frozen_trait_model(path)
            if not self._pipeline_editing_allowed():
                return
            self._activate_trait_model(copy_trait_model(path, self._trait_model_directory()))
        except (OSError, ValueError, KeyError, TypeError) as error:
            QMessageBox.warning(self, 'Could not import the seed-trait model', str(error))

    def _export_trait_model(self):
        source = self._selected_trait_model_path()
        if source is None or not source.is_file():
            self.statusBar().showMessage('Object-level seed traits has no model to export.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export seed-trait model', source.name, 'JSON (*.json)')
        if not path:
            return
        try:
            copy_annotation_file(source, path)
        except OSError as error:
            QMessageBox.warning(self, 'Could not export the seed-trait model', str(error))
            return
        self.statusBar().showMessage(f'Exported seed-trait model to {path}.')

    def _adopt_trait_model(self, destination):
        """Keep the selected model with the project being saved, and point the node at that copy."""
        if 'trait_classifier' not in self.pipeline.nodes:
            return
        source = self._selected_trait_model_path()
        if source is None or not source.is_file():
            return
        folder = trait_model_folder(project_annotation_directory(destination)).resolve()
        if source.parent == folder:
            return
        affected = self.pipeline.set_parameter('trait_classifier', 'trait_model_path',
                                               workspace_path_text(copy_trait_model(source, folder), self._root))
        for image_key in set(self._analysis_caches) | set(self._analyses):
            self._cache_dirty_nodes.setdefault(image_key, set()).update(affected)
        self.pipeline_canvas.refresh(affected)
