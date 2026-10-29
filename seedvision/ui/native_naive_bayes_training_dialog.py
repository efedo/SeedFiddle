"""Asynchronous Bayes sample/directory training and atomic PlantCV TSV saving."""
from pathlib import Path
from threading import Event, Lock

from PySide6.QtCore import QObject, QIODevice, QRunnable, QSaveFile, QThreadPool, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from seedvision.benchmarking import naive_bayes_training_directory as _directory
from seedvision.benchmarking import native_naive_bayes_training as _adapter
from seedvision.benchmarking.native_naive_bayes_recipe import model_source_for_trained_model
from seedvision.cuda.plantcv_naive_bayes_sampling import BayesSamplingCancelled
from seedvision.cuda.plantcv_naive_bayes import parse_plantcv_pdf
from seedvision.model_io import naive_bayes_provenance as _provenance
from seedvision.export import naive_bayes_plots as _plots

_DIRECTORY_TRAINING_LOCK = Lock()


class _TrainingSignals(QObject):
    completed = Signal(object)


class _TrainingWorker(QRunnable):
    def __init__(self, operation, path, cancel, *, directory=False):
        super().__init__()
        self.operation, self.path, self.cancel = operation, path, cancel
        self.directory = directory
        self.signals = _TrainingSignals()

    def run(self):
        try:
            if self.directory:
                # Serialize whole GUI jobs sharing the public global NumPy
                # stream. A queued job remains cancellable without RNG draws.
                while not _DIRECTORY_TRAINING_LOCK.acquire(timeout=.05):
                    if self.cancel.is_set():
                        raise BayesSamplingCancelled('Queued Bayes directory training stopped')
                try:
                    if self.cancel.is_set():
                        raise BayesSamplingCancelled('Queued Bayes directory training stopped')
                    result = self.operation(*self.path, cancel=self.cancel.is_set)
                finally:
                    _DIRECTORY_TRAINING_LOCK.release()
            else:
                result = self.operation(self.path, cancel=self.cancel.is_set)
        except Exception as error:
            result = error
        self.signals.completed.emit(result)


class _PlotWorker(QRunnable):
    def __init__(self, operation, model, cancel):
        super().__init__()
        self.operation, self.model, self.cancel = operation, model, cancel
        self.signals = _TrainingSignals()

    def run(self):
        try:
            result = self.operation(self.model, cancel=self.cancel.is_set)
        except Exception as error:
            result = error
        self.signals.completed.emit(result)


class NativeNaiveBayesTrainingDialog(QDialog):
    def __init__(self, parent=None, *, operation=None, directory_operation=None, plot_operation=None):
        super().__init__(parent)
        self.setWindowTitle('Train Naive Bayes')
        self.resize(760, 370)
        self._operation = operation or _adapter.train_bayes_sample_file
        self._directory_operation = directory_operation or _directory.train_bayes_directories
        self._plot_operation = plot_operation or _plots.render_naive_bayes_plots
        self._pending_plot_save = None
        self._publishing = False
        self._worker = self._cancel = self._completion = None
        self._revision = self._run_revision = 0
        self._run_path = None
        self._run_mode = 'samples'
        self._close_requested = False
        self.saved_model_source = None
        self.saved_model_path = None
        self.saved_plot_paths = ()
        self.plot_save_errors = ()
        self.rendered_plot_metadata = None
        self.input_mode = QComboBox()
        self.input_mode.setObjectName('nativeNaiveBayesTrainingMode')
        self.input_mode.setAccessibleName('Training input')
        self.input_mode.addItem('Sampled RGB table', 'samples')
        self.input_mode.addItem('Images and masks', 'directories')
        self.samples_edit = QLineEdit()
        self.samples_edit.setObjectName('nativeNaiveBayesTrainingSamples')
        self.samples_edit.setReadOnly(True)
        self.choose_samples_button = QPushButton('Choose samples…')
        self.choose_samples_button.setObjectName('nativeNaiveBayesChooseSamples')
        self.image_directory_edit = QLineEdit()
        self.image_directory_edit.setObjectName('nativeNaiveBayesTrainingImages')
        self.image_directory_edit.setReadOnly(True)
        self.mask_directory_edit = QLineEdit()
        self.mask_directory_edit.setObjectName('nativeNaiveBayesTrainingMasks')
        self.mask_directory_edit.setReadOnly(True)
        self.choose_images_button = QPushButton('Choose images…')
        self.choose_masks_button = QPushButton('Choose masks…')
        self.choose_images_button.setObjectName('nativeNaiveBayesChooseTrainingImages')
        self.choose_masks_button.setObjectName('nativeNaiveBayesChooseTrainingMasks')
        self.description = QLabel()
        self.description.setWordWrap(True)
        self.description.setTextFormat(Qt.TextFormat.PlainText)
        self.status_label = QLabel('Choose a sampled RGB table to begin.')
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.train_button = QPushButton('Train')
        self.stop_button = QPushButton('Stop')
        self.save_button = QPushButton('Save model…')
        self.save_plots_checkbox = QCheckBox('Save probability density plots (SVG) beside the model')
        self.save_plots_checkbox.setObjectName('nativeNaiveBayesSavePDFPlots')
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.train_button.setDefault(True)
        self.samples_row, self.images_row, self.masks_row = QWidget(), QWidget(), QWidget()
        source_row, image_row, mask_row = (QHBoxLayout(widget)
            for widget in (self.samples_row, self.images_row, self.masks_row))
        actions = QHBoxLayout()
        for row in (source_row, image_row, mask_row):
            row.setContentsMargins(0, 0, 0, 0)
        source_row.addWidget(self.samples_edit, 1)
        source_row.addWidget(self.choose_samples_button)
        image_row.addWidget(QLabel('Images'))
        image_row.addWidget(self.image_directory_edit, 1)
        image_row.addWidget(self.choose_images_button)
        mask_row.addWidget(QLabel('Masks'))
        mask_row.addWidget(self.mask_directory_edit, 1)
        mask_row.addWidget(self.choose_masks_button)
        for widget in (self.train_button, self.stop_button, self.save_button, self.button_box):
            actions.addWidget(widget)
        layout = QVBoxLayout(self)
        layout.addWidget(self.input_mode)
        layout.addWidget(self.description)
        for widget in (self.samples_row, self.images_row, self.masks_row):
            layout.addWidget(widget)
        layout.addWidget(self.status_label, 1)
        layout.addWidget(self.save_plots_checkbox)
        layout.addLayout(actions)
        self.samples_edit.textChanged.connect(self._selection_changed)
        self.image_directory_edit.textChanged.connect(self._selection_changed)
        self.mask_directory_edit.textChanged.connect(self._selection_changed)
        self.input_mode.currentIndexChanged.connect(self._mode_changed)
        self.choose_samples_button.clicked.connect(self._choose_samples)
        self.choose_images_button.clicked.connect(lambda:self._choose_directory(self.image_directory_edit, 'Choose training images'))
        self.choose_masks_button.clicked.connect(lambda:self._choose_directory(self.mask_directory_edit, 'Choose training masks'))
        self.train_button.clicked.connect(self._start_training)
        self.stop_button.clicked.connect(self._stop_training)
        self.save_button.clicked.connect(self._save_model)
        self.button_box.rejected.connect(self.reject)
        self._mode_changed()
        self._set_running(False)

    def _mode_changed(self, *_):
        directory = self.input_mode.currentData() == 'directories'
        self.samples_row.setVisible(not directory)
        self.images_row.setVisible(directory)
        self.masks_row.setVisible(directory)
        self.description.setText(
            'Images pair with masks of the same filename in the mask directory. '
            'Black (0) is background; white (255) is foreground. PNG/JPG selection and '
            'sampling follow PlantCV order using the current random stream. '
            'Decoded dimensions must match, with varying samples in all three HSV channels.'
            if directory else
            'Train from tab-delimited class columns containing RGB triples. '
            'Each class needs at least two samples and variation in all three HSV channels. '
            'The source Gaussian estimator uses its default bandwidth.')
        self._selection_changed()

    def _choose_directory(self, edit, title):
        if self._worker is not None:
            return
        path = QFileDialog.getExistingDirectory(self, title, edit.text())
        if path:
            if edit.text() == path:
                self._selection_changed()
            else:
                edit.setText(path)

    def _choose_samples(self):
        if self._worker is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Choose sampled RGB table', self.samples_edit.text(),
            'Sampled RGB tables (*.tsv *.txt);;All files (*)')
        if path:
            if self.samples_edit.text() == path:
                self._selection_changed()
            else:
                self.samples_edit.setText(path)

    def _selection_changed(self, *_):
        self._revision += 1
        self._completion = None
        self.saved_model_source = self.saved_model_path = None
        self.saved_plot_paths, self.plot_save_errors = (), ()
        self.rendered_plot_metadata = None
        self.save_button.setEnabled(False)
        if self._cancel is not None:
            self._cancel.set()
        if self._worker is None:
            self.status_label.setText('Choose training inputs, then Train to create a new model.')

    def _start_training(self):
        if self._worker is not None or self._publishing:
            return
        self._completion = None
        self.saved_model_source = self.saved_model_path = None
        self.saved_plot_paths, self.plot_save_errors = (), ()
        self.rendered_plot_metadata = None
        self.save_button.setEnabled(False)
        self._run_mode = self.input_mode.currentData()
        directory = self._run_mode == 'directories'
        edits = (self.image_directory_edit, self.mask_directory_edit) if directory else (self.samples_edit,)
        if any(not edit.text() for edit in edits):
            self.status_label.setText('Choose both image and mask directories first.' if directory
                else 'Choose a sampled RGB table first.')
            return
        try:
            paths = tuple(str(Path(edit.text()).resolve()) for edit in edits)
        except (OSError, ValueError, RuntimeError) as error:
            self.status_label.setText(f'Training input is unavailable: {error}')
            return
        self._run_path = paths if directory else paths[0]
        self._run_revision = self._revision
        self._cancel = Event()
        operation = self._directory_operation if directory else self._operation
        worker = _TrainingWorker(operation, self._run_path, self._cancel, directory=directory)
        worker.signals.completed.connect(self._completed)
        self._worker = worker
        self._set_running(True)
        self.status_label.setText('Training Gaussian probability tables…')
        QThreadPool.globalInstance().start(worker)

    def _validated_completion(self, result):
        directory = self._run_mode == 'directories'
        expected_type = _directory.BayesDirectoryTrainingCompletion if directory else _adapter.BayesTrainingCompletion
        if type(result) is not expected_type:
            raise ValueError('Training returned incomplete results.')
        model = parse_plantcv_pdf(result.tsv_bytes)
        if model != result.model:
            raise ValueError('Training model differs from its actual TSV bytes.')
        record = _provenance.parse_training_record(result.provenance_json, model)
        metadata = record.metadata
        if directory:
            selection = metadata.get('selection', {})
            if (metadata['kind'] != 'native_bayes_directory_training'
                    or (selection.get('image_root_origin'), selection.get('mask_root_origin')) != self._run_path):
                raise ValueError('Training result differs from the selected directory origins.')
        elif (metadata['kind'] != 'native_bayes_sample_training'
                or metadata['sample_origin_path'] != self._run_path):
            raise ValueError('Training result differs from the selected sample origin.')
        return metadata

    def _completed(self, result):
        cancelled = self._cancel.is_set()
        self._worker = self._cancel = None
        self._completion = None
        self._set_running(False)
        if self._close_requested:
            super().reject()
            return
        if cancelled:
            self.status_label.setText('Training stopped.')
            return
        if self._run_revision != self._revision:
            self.status_label.setText('Training inputs changed. Train again to update the model.')
            return
        if isinstance(result, Exception):
            self.status_label.setText(f'Training failed: {result}')
            return
        try:
            metadata = self._validated_completion(result)
        except (ValueError, TypeError) as error:
            self.status_label.setText(f'Training failed: {error}')
            return
        self._completion = result
        self.save_button.setEnabled(True)
        counts = ', '.join(f'{name}: {n}' for name, n in zip(metadata['class_names'], metadata['sample_counts']))
        facts = (f"{len(metadata['occurrences'])} image/mask pairs. Ordered channel samples — {counts}.\n"
            f"Sampling state: {metadata['rng_after']['sha256'][:16]}.\n"
            if self._run_mode == 'directories' else f'Ordered samples — {counts}.\n')
        self.status_label.setText('Ready to save. '+facts+
            'Numerical conformance is empirical; training accuracy is not validated.\n'
            'Save creates a PlantCV TSV. Keep the selected recipe in your project to retain its training provenance.')

    def _stop_training(self):
        if self._cancel is not None:
            self._cancel.set()
            self.status_label.setText('Stopping…')

    def _save_model(self):
        if self._worker is not None or self._publishing or self._completion is None:
            return
        try:
            self._validated_completion(self._completion)
        except (ValueError, TypeError) as error:
            self._completion = None
            self.save_button.setEnabled(False)
            self.status_label.setText(f'Training result is unavailable: {error}')
            return
        frozen_completion, frozen_revision = self._completion, self._revision
        path, _ = QFileDialog.getSaveFileName(self, 'Save Naive Bayes model', 'naive-bayes-model.tsv',
            'PlantCV probability tables (*.tsv *.txt)')
        if not path:
            return
        if (self._completion is not frozen_completion or self._revision != frozen_revision
                or self._close_requested):
            self.status_label.setText('Training inputs changed. No model or plots were saved.')
            return
        try:
            self._validated_completion(self._completion)
        except (ValueError, TypeError) as error:
            self.status_label.setText(f'Training result is unavailable: {error}')
            return
        if self.save_plots_checkbox.isChecked():
            root = Path(path).parent
            try:
                destination = Path(path).resolve()
                if any(str(destination).casefold() == str((root/name).resolve()).casefold()
                       for name in _plots.FILENAMES):
                    raise ValueError('The model filename must differ from the three PDF SVG filenames.')
            except (OSError, ValueError, RuntimeError) as error:
                QMessageBox.warning(self, 'Save model and plots', str(error))
                return
            self._pending_plot_save = (path, self._completion, self._revision)
            self._cancel = Event()
            worker = _PlotWorker(self._plot_operation, self._completion.model, self._cancel)
            worker.signals.completed.connect(self._plots_completed)
            self._worker = worker
            self._set_running(True)
            self.status_label.setText('Rendering probability density SVG plots…')
            QThreadPool.globalInstance().start(worker)
            return
        self._publish_model(path)

    def _plots_completed(self, result):
        cancelled = self._cancel.is_set()
        pending = self._pending_plot_save
        self._pending_plot_save = None
        self._worker = self._cancel = None
        self._set_running(False)
        if self._close_requested:
            super().reject()
            return
        if cancelled:
            self.status_label.setText('Plotting stopped. '+('The trained model remains ready to save.'
                if self._completion is not None else 'Train again after changing inputs.'))
            return
        if pending is None or pending[1] is not self._completion or pending[2] != self._revision:
            self.status_label.setText('Training inputs changed. No model or plots were saved.')
            return
        try:
            if isinstance(result, Exception):
                raise result
            self._validated_completion(self._completion)
            _plots.validate_naive_bayes_plots(result, self._completion.model)
        except Exception as error:
            self.status_label.setText(f'Plots failed; no files saved: {error}')
            return
        self._publish_model(pending[0], plots=result)

    def _publish_model(self, path, *, plots=None):
        self._publishing = True
        self._set_running(True)
        self.stop_button.setEnabled(False)
        try:
            saved = self._publish_model_files(path, plots=plots)
        finally:
            self._publishing = False
            self._set_running(False)
        if self._close_requested:
            super().reject()
        elif saved:
            self.accept()

    def _publish_model_files(self, path, *, plots=None):
        target = QSaveFile(path)
        if not target.open(QIODevice.OpenModeFlag.WriteOnly):
            QMessageBox.warning(self, 'Save model', target.errorString())
            return
        payload = self._completion.tsv_bytes
        if target.write(payload) != len(payload) or not target.commit():
            error = target.errorString()
            target.cancelWriting()
            QMessageBox.warning(self, 'Save model', error)
            return
        try:
            selected = model_source_for_trained_model(path, payload, self._completion.provenance_json)
        except (ValueError, OSError, TypeError) as error:
            QMessageBox.warning(self, 'Choose trained model', str(error))
            return
        self.saved_model_path = str(Path(path).resolve())
        self.saved_model_source = selected
        self.saved_plot_paths, self.plot_save_errors = (), ()
        self.rendered_plot_metadata = None
        if plots is not None:
            written, failed = [], []
            for plot in plots.plots:
                plot_path = Path(path).parent/plot.filename
                try:
                    self._write_plot(plot_path, plot.svg_bytes)
                    written.append(str(plot_path.resolve()))
                except (OSError, ValueError, RuntimeError) as error:
                    failed.append((plot.filename, str(error)))
            self.saved_plot_paths, self.plot_save_errors = tuple(written), tuple(failed)
            self.rendered_plot_metadata = plots.metadata_json
            if failed:
                QMessageBox.warning(self, 'Model saved; some PDF plots failed',
                    'The verified TSV model is selected.\nSaved plots: '+
                    (', '.join(Path(p).name for p in written) or 'none')+
                    '\nFailed plots: '+', '.join(f'{name}: {error}' for name, error in failed))
        return True

    @staticmethod
    def _write_plot(path, payload):
        target = QSaveFile(str(path))
        if not target.open(QIODevice.OpenModeFlag.WriteOnly):
            raise OSError(target.errorString())
        if target.write(payload) != len(payload) or not target.commit():
            error = target.errorString()
            target.cancelWriting()
            raise OSError(error)
        # Each sibling is atomic individually; the TSV plus three SVGs are not
        # a filesystem transaction. Report any postcommit mismatch honestly.
        with Path(path).open('rb') as stream:
            actual = stream.read(_plots.MAX_SVG_BYTES+1)
        if actual != payload:
            raise OSError('Saved PDF plot differs from its rendered SVG bytes.')

    def _set_running(self, running):
        for widget in (self.input_mode, self.samples_edit, self.choose_samples_button,
                self.image_directory_edit, self.mask_directory_edit,
                self.choose_images_button, self.choose_masks_button):
            widget.setEnabled(not running)
        self.train_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.save_button.setEnabled(not running and self._completion is not None)
        self.save_plots_checkbox.setEnabled(not running)

    def _request_close(self):
        if self._publishing:
            self._close_requested = True
            return True
        if self._worker is None:
            return False
        self._close_requested = True
        self._stop_training()
        return True

    def reject(self):
        if not self._request_close():
            super().reject()

    def accept(self):
        if not self._request_close():
            super().accept()

    def closeEvent(self, event):
        if self._request_close():
            event.ignore()
        else:
            super().closeEvent(event)
