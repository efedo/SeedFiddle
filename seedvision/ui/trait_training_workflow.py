"""Native trait-training controls and a cancellable serial-pool worker."""
from threading import Event
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QLabel, QMessageBox, QSpinBox, QVBoxLayout,
)

from seedvision.learning.trait_classifier import TraitTrainingCancelled
from seedvision.learning.trait_training import (
    TraitTrainingConfiguration, TraitTrainingRequest, read_trait_manifest, train_trait_manifest,
)
from seedvision.ui.learning_workflow import _PathRow


class TraitTrainingDialog(QDialog):
    def __init__(self, *, project_root, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Train object-level seed traits')
        self.setMinimumWidth(620)
        self._accepted_request = None
        layout = QVBoxLayout(self)
        explanation = QLabel('Load explicit trait definitions, permitted truth and isolated train/calibration groups. '
            'Inputs may be supplied twenty-column features or reviewed reference-seed archives. '
            'Training diagnostics do not establish biological validity.', self)
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        form = QFormLayout()
        self.manifest_row = _PathRow(self)
        self.manifest_row.edit.setText(str(Path(project_root) / 'trait-training.json'))
        self.output_row = _PathRow(self)
        self.output_row.edit.setText(str(Path(project_root) / 'trait-model.json'))
        self.manifest_row.browse.clicked.connect(self._choose_manifest)
        self.output_row.browse.clicked.connect(self._choose_output)
        form.addRow('Evidence manifest', self.manifest_row)
        form.addRow('Frozen model output', self.output_row)
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
        self.activate_checkbox = QCheckBox('Activate the saved model when training completes', self)
        form.addRow('After saving', self.activate_checkbox)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('Train')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

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

    def _choose_output(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save frozen trait model', self.output_row.edit.text(), 'JSON (*.json)')
        if path:
            self.output_row.edit.setText(path)

    def request(self):
        if self._accepted_request is not None:
            return self._accepted_request
        if not self.manifest_row.edit.text().strip() or not self.output_row.edit.text().strip():
            raise ValueError('Choose both evidence and output files.')
        return TraitTrainingRequest(Path(self.manifest_row.edit.text()), Path(self.output_row.edit.text()),
            TraitTrainingConfiguration(self.iterations_spin.value(), self.learning_rate_spin.value(), self.l2_spin.value(),
                (self.lower_temperature_spin.value(), self.upper_temperature_spin.value()),
                self.calibration_spin.value(), self.threshold_spin.value()), self.activate_checkbox.isChecked())

    def accept(self):
        from dataclasses import replace
        try:
            request = self.request()
            manifest = read_trait_manifest(request.manifest_path)
            if not request.output_path.parent.is_dir():
                raise ValueError('Choose an existing output directory.')
            if request.output_path.exists() and QMessageBox.question(self, 'Replace frozen model?',
                    f'Replace {request.output_path.name} only after successful training?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
            self._accepted_request = replace(request, manifest_sha256=manifest.content_sha256)
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
