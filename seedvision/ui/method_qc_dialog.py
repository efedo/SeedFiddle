"""Review portable recipe QC without installing predictions or approving seeds."""
import getpass
import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout, QInputDialog)

from seedvision.export.method_qc import (MethodQC, acknowledge_method_qc, export_method_qc,
    load_method_qc, load_method_qc_acknowledgment)


class MethodQCDialog(QDialog):
    def __init__(self, quality=None, parent=None, *, project_root=None, project_path=None):
        super().__init__(parent)
        self._project_root = project_root
        self._project_path = project_path
        self.quality = None
        self.acknowledgment = None
        self.setWindowTitle('Recorded recipe QC')
        self.resize(860, 650)
        layout = QVBoxLayout(self)
        self.summary = QLabel(self)
        self.summary.setObjectName('methodQCSummary')
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.summary)
        self.facts = QPlainTextEdit(self)
        self.facts.setObjectName('methodQCRecordedFacts')
        self.facts.setReadOnly(True)
        layout.addWidget(self.facts, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel('Recorded operator account:', self))
        self.operator = QLineEdit(getpass.getuser(), self)
        self.operator.setObjectName('methodQCOperator')
        row.addWidget(self.operator, 1)
        layout.addLayout(row)
        self.status = QLabel(self)
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self.load_button = QPushButton('Load recorded QC…', self)
        self.batch_button = QPushButton('Load batch row QC…', self)
        self.export_button = QPushButton('Export QC…', self)
        self.ack_button = QPushButton('Save acknowledgment…', self)
        self.load_ack_button = QPushButton('Load acknowledgment…', self)
        self.retain_button = QPushButton('Retain in project history' if project_path is not None
                                        else 'Retain in workspace history', self)
        self.retain_button.setObjectName('retainMethodQC')
        layout.addWidget(self.retain_button)
        for button in (self.load_button, self.batch_button, self.export_button, self.ack_button, self.load_ack_button):
            buttons.addButton(button, QDialogButtonBox.ButtonRole.ActionRole)
        self.load_button.clicked.connect(self._load)
        self.batch_button.clicked.connect(self._load_batch_row)
        self.export_button.clicked.connect(self._export)
        self.ack_button.clicked.connect(self._acknowledge)
        self.load_ack_button.clicked.connect(self._load_acknowledgment)
        self.retain_button.clicked.connect(self._retain)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.set_quality(quality)

    def set_quality(self, quality):
        if quality is not None and not isinstance(quality, MethodQC):
            raise TypeError('QC review requires a validated recorded QC record.')
        self.quality = quality
        self.acknowledgment = None
        for button in (self.export_button, self.ack_button, self.load_ack_button):
            button.setEnabled(quality is not None)
        self.retain_button.setEnabled(quality is not None and self._project_root is not None)
        if quality is None:
            self.summary.setText('No recorded recipe QC selected.')
            self.facts.clear()
            self.status.setText('Select a recorded QC file to review its captured facts.')
            return
        payload = quality.payload
        identity = payload['identity']
        recipe = identity['recipe_identity']
        route = identity['native_artifact']
        inspection = identity['recorded_inspection']
        heading = (f"Recipe: {recipe['recipe_id']} · {identity['recipe_mode']}\n"
                   f"Declared recipe backend: {recipe['backend_id']} {recipe['backend_version']}\n"
                   if recipe else 'Selected recipe identity was not captured.\n')
        attempt = identity.get('attempt_record')
        batch = identity.get('batch_record')
        reference = identity.get('reference_capture')
        if batch:
            heading = f"Batch destination: {batch['destination']}\n"
        recorded = (f"Recorded public capture: verified complete manifest {reference['manifest_sha256']}\n"
                    f"Public backend: {reference['execution_backend']}\n" if reference and identity['scope'] == 'whole_reference_capture' else
                    f"Recorded reference attempt: {reference['state']} · {reference['phase']}\n"
                    'A complete public capture was not verified for this attempt.\n' if reference else
                    f"Recorded batch attempt: {batch['state']} · {batch['phase']}\n"
                    f"Recorded rows: {len(batch['completed_rows'])}/{len(batch['requested_recipes'])}; publication did not complete.\n"
                    f"Staged archive state: {batch['archives_state']}. Current archive availability is not verified.\n" if batch else
                    f"Recorded recipe attempt: {attempt['state']} · {attempt['phase']}\n"
                    'Output archive and node outcomes were not verified for this attempt.\n' if attempt else
                    f"Recorded native execution: {route['execution_mode']} · {route['status']}\n" if route else
               f"Selected stage: {identity['stage_record']['stage_id']} · recorded state {identity['stage_record']['state']}\n"
               f"Recorded checkpoint inspection: {inspection['status']}\n")
        self.summary.setText(heading+recorded
            + f"QC revision: {quality.revision}\n"
              'Loading this record does not reverify original artifacts. Acknowledgment applies to this exact record only.')
        self.facts.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
        self.status.setText('No acknowledgment loaded for this QC revision. Measurement and scientific eligibility remain unknown.')

    def _attempt(self, action):
        try:
            action()
        except (ValueError, OSError, TypeError) as error:
            self.status.setText(f'QC action failed: {type(error).__name__}: {error}')

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load recorded recipe QC', filter='JSON records (*.json)')
        if path:
            self._attempt(lambda: self.set_quality(load_method_qc(path)))

    def _load_batch_row(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load recorded native graph ledger', filter='JSON records (*.json)')
        if path:
            def load():
                from seedvision.export.recipe_batch_qc import read_project_graph_qc
                qualities = read_project_graph_qc(path)
                if not qualities:
                    raise ValueError('This ledger has no recorded graph rows.')
                index, accepted = QInputDialog.getInt(self, 'Recorded batch row QC', 'Graph row index (starting at zero):',
                                                    0, 0, len(qualities)-1)
                if accepted:
                    self.set_quality(qualities[index])
                    if qualities[index] is None:
                        self.status.setText('This historical row has no recorded QC; its facts remain unknown.')
            self._attempt(load)

    def _export(self):
        if self.quality is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export recorded recipe QC', 'method-qc.json', 'JSON records (*.json)')
        if path:
            def save():
                export_method_qc(self.quality, path)
                self.status.setText(f'Recorded QC exported: {path}')
            self._attempt(save)

    def _acknowledge(self):
        if self.quality is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save QC acknowledgment',
            self.quality.revision+'-acknowledgment.json', 'JSON records (*.json)')
        if path:
            def save():
                acknowledge_method_qc(self.quality, path, self.operator.text())
                self._install_acknowledgment(path)
            self._attempt(save)

    def _install_acknowledgment(self, path):
        value = load_method_qc_acknowledgment(path, self.quality)
        self.acknowledgment = value
        self.status.setText(f"Recorded acknowledgment: {value['operator_account']} · {value['reviewed_at_utc']}\n"
                            'Seeds and scientific validity are not approved by this acknowledgment.')

    def _load_acknowledgment(self):
        if self.quality is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Load acknowledgment for this exact QC', filter='JSON records (*.json)')
        if path:
            self._attempt(lambda: self._install_acknowledgment(path))

    def _retain(self):
        if self.quality is None or self._project_root is None:
            return
        from seedvision.persistence.method_qc_history import retain_method_qc
        def save():
            paths = retain_method_qc(self._project_root, self.quality, project_path=self._project_path,
                                     acknowledgment=self.acknowledgment)
            self.status.setText(f'Retained {len(paths)} historical record(s) for this exact analytical QC revision.\n'
                                'Retaining records does not approve seeds or reverify original artifacts.')
        self._attempt(save)
