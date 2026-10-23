"""Asynchronous CUDA directory training and atomic centre-model saving."""
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QIODevice, QRunnable, QSaveFile, QThreadPool, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QVBoxLayout)

from seedvision.benchmarking import native_kmeans_training as _adapter


class _TrainingSignals(QObject):
    completed = Signal(object)


class _TrainingWorker(QRunnable):
    def __init__(self, operation, request, cache, cancel):
        super().__init__()
        self.operation, self.request, self.cache, self.cancel = operation, request, cache, cancel
        self.signals = _TrainingSignals()

    def run(self):
        try:
            result = self.operation(self.request, cached_rows=self.cache, cancel=self.cancel.is_set)
        except Exception as error:
            result = error
        self.signals.completed.emit(result)


def _spin(low, high, value):
    widget = QSpinBox()
    widget.setRange(low, high)
    widget.setValue(value)
    return widget


class NativeKMeansTrainingDialog(QDialog):
    def __init__(self, parent=None, *, operation=None):
        super().__init__(parent)
        self.setWindowTitle('Train K-means centres')
        self._operation = operation or _adapter.train_kmeans_directory
        self._worker = self._cancel = self._cached_rows = self._completion = None
        self._close_requested = False
        self._revision = self._run_revision = 0
        self.saved_model_path = None
        self.directory_edit = QLineEdit()
        self.directory_edit.setReadOnly(True)
        self.choose_directory_button = QPushButton('Choose folder…')
        self.prefix_edit = QLineEdit()
        self.image_count = _spin(0, 256, 0)
        self.patch_size = _spin(1, 31, 10)
        self.sampling_mode = QComboBox()
        self.sampling_mode.addItems(['All patches', 'Patch count', 'Patch fraction'])
        self.sampling_count = _spin(1, 1_000_000, 1000)
        self.sampling_fraction = QDoubleSpinBox()
        self.sampling_fraction.setDecimals(6)
        self.sampling_fraction.setRange(.000001, .999999)
        self.sampling_fraction.setValue(.1)
        self.clusters = _spin(1, 256, 2)
        self.seed_edit = QLineEdit('1')
        self.seed_edit.setMaxLength(10)
        self.n_init = _spin(1, 32, 10)
        form = QFormLayout()
        folder = QHBoxLayout()
        folder.addWidget(self.directory_edit, 1)
        folder.addWidget(self.choose_directory_button)
        sampling = QHBoxLayout()
        for widget in (self.sampling_mode, self.sampling_count, self.sampling_fraction):
            sampling.addWidget(widget)
        form.addRow('Image folder', folder)
        form.addRow('Filename contains', self.prefix_edit)
        form.addRow('Image count (0 = all)', self.image_count)
        form.addRow('Patch size', self.patch_size)
        form.addRow('Sampling', sampling)
        form.addRow('Clusters', self.clusters)
        form.addRow('Random seed', self.seed_edit)
        form.addRow('Initializations', self.n_init)
        self.status_label = QLabel()
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.status_label.setWordWrap(True)
        self.train_button = QPushButton('Train')
        self.stop_button = QPushButton('Stop')
        self.save_button = QPushButton('Save model…')
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.train_button.setDefault(True)
        actions = QHBoxLayout()
        for widget in (self.train_button, self.stop_button, self.save_button, self.button_box):
            actions.addWidget(widget)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.status_label)
        layout.addLayout(actions)
        self._selection_controls = (self.directory_edit, self.prefix_edit, self.image_count)
        self._row_controls = (self.patch_size, self.sampling_mode, self.sampling_count, self.sampling_fraction)
        self._fit_controls = (self.clusters, self.seed_edit, self.n_init)
        self.directory_edit.textChanged.connect(self._selection_changed)
        self.prefix_edit.textChanged.connect(self._selection_changed)
        self.image_count.valueChanged.connect(self._selection_changed)
        self.patch_size.valueChanged.connect(self._rows_changed)
        self.sampling_mode.currentIndexChanged.connect(self._rows_changed)
        self.sampling_count.valueChanged.connect(self._rows_changed)
        self.sampling_fraction.valueChanged.connect(self._rows_changed)
        self.clusters.valueChanged.connect(self._fit_changed)
        self.seed_edit.textChanged.connect(self._fit_changed)
        self.n_init.valueChanged.connect(self._fit_changed)
        self.choose_directory_button.clicked.connect(self._choose_directory)
        self.train_button.clicked.connect(self._start_training)
        self.stop_button.clicked.connect(self._stop_training)
        self.save_button.clicked.connect(self._save_model)
        self.button_box.rejected.connect(self.reject)
        self._set_running(False)

    def _choose_directory(self):
        if self._worker is not None:
            return
        path = QFileDialog.getExistingDirectory(self, 'Choose training image folder', self.directory_edit.text())
        if path:
            self.directory_edit.setText(path)
            self._selection_changed()  # Choosing the same folder requests a fresh selection too.

    def _make_request(self):
        directory = self.directory_edit.text().strip()
        if not directory:
            raise ValueError('Choose an image folder first.')
        text = self.seed_edit.text().strip()
        if not text or not text.isascii() or not text.isdecimal() or not 0 <= int(text) <= 0xFFFFFFFF:
            raise ValueError('Random seed must be an integer from 0 through 4294967295.')
        mode = self.sampling_mode.currentIndex()
        sampling = None if mode == 0 else (self.sampling_count.value() if mode == 1 else float(self.sampling_fraction.value()))
        return _adapter.KMeansTrainingRequest(directory, self.prefix_edit.text(), self.image_count.value(),
            self.patch_size.value(), sampling, self.clusters.value(), int(text), self.n_init.value())

    def _start_training(self):
        if self._worker is not None:
            return
        self._completion = None
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText('Cancel')
        self.save_button.setEnabled(False)
        try:
            request = self._make_request()
        except ValueError as error:
            self.status_label.setText(str(error))
            return
        self._cancel = Event()
        self._run_revision = self._revision
        worker = _TrainingWorker(self._operation, request, self._cached_rows, self._cancel)
        worker.signals.completed.connect(self._completed)
        self._worker = worker
        self._set_running(True)
        self.status_label.setText('Training…')
        QThreadPool.globalInstance().start(worker)

    def _completed(self, result):
        cancelled = self._cancel.is_set()
        self._worker = self._cancel = None
        self._set_running(False)
        if self._close_requested:
            super().reject()
            return
        if cancelled:
            self.status_label.setText('Training stopped.')
            return
        if self._run_revision != self._revision:
            self.status_label.setText('Training settings changed. Train again to update the model.')
            return
        if isinstance(result, Exception):
            self.status_label.setText(f'Training failed: {result}')
            return
        if type(result) is not _adapter.KMeansTrainingCompletion:
            self.status_label.setText('Training returned incomplete results.')
            return
        self._completion, self._cached_rows = result, result.cache
        self.save_button.setEnabled(True)
        # Training finished: leaving no longer abandons work, so say Close.
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setText('Close')
        self.status_label.setText(f'Ready to save: {result.summary.rows} patches, {result.summary.iterations} iterations.')

    def _stop_training(self):
        if self._cancel is not None:
            self._cancel.set()
            self.status_label.setText('Stopping…')

    def _save_model(self):
        if self._worker is not None or self._completion is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save K-means model', 'kmeans-model.json', 'JSON model (*.json)')
        if not path:
            return
        target = QSaveFile(path)
        if not target.open(QIODevice.OpenModeFlag.WriteOnly):
            QMessageBox.warning(self, 'Save model', target.errorString())
            return
        payload = self._completion.artifact_payload
        if target.write(payload) != len(payload) or not target.commit():
            error = target.errorString()
            target.cancelWriting()
            QMessageBox.warning(self, 'Save model', error)
            return
        self.saved_model_path = str(Path(path).resolve())
        self.accept()

    def _selection_changed(self, *_):
        self._cached_rows = None
        self._fit_changed()

    def _rows_changed(self, *_):
        if self._cached_rows is not None:
            self._cached_rows = self._cached_rows.without_rows()
        self._fit_changed()
        self._set_sampling_enabled()

    def _fit_changed(self, *_):
        self._revision += 1
        self._completion = None
        self.save_button.setEnabled(False)

    def _set_sampling_enabled(self):
        self.sampling_count.setEnabled(self._worker is None and self.sampling_mode.currentIndex() == 1)
        self.sampling_fraction.setEnabled(self._worker is None and self.sampling_mode.currentIndex() == 2)

    def _set_running(self, running):
        for widget in self._selection_controls+self._row_controls+self._fit_controls+(self.choose_directory_button,):
            widget.setEnabled(not running)
        self.train_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.save_button.setEnabled(not running and self._completion is not None)
        self._set_sampling_enabled()

    def _request_close(self):
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
