"""Typed editors for small native vector, ROI and category parameters."""
import json
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QLabel, QLineEdit


class WorkflowParameterEditor(QWidget):
    changed = Signal(object)

    def __init__(self, key, value, parent=None):
        super().__init__(parent)
        parsed = json.loads(value) if isinstance(value, str) else value
        self.key = key
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.fields = []
        self.optional = None
        self.card = None
        if key == 'cat_list':
            self.optional = QCheckBox('Select cluster IDs (otherwise all)', self)
            self.optional.setChecked(parsed is not None)
            layout.addWidget(self.optional, 0, 0, 1, 2)
            self.categories = QLineEdit(', '.join(map(str, parsed or [])), self)
            self.categories.setPlaceholderText('0, 2, 5')
            layout.addWidget(self.categories, 1, 0, 1, 2)
            self.categories.editingFinished.connect(self.emit_value)
            self.optional.toggled.connect(self.emit_value)
            self.categories.setEnabled(self.optional.isChecked())
            return
        roi = key in ('roi_bounds_xyxy', 'seed_roi_xyxy', 'validity_roi')
        chip = key == 'color_chip_size'
        labels = ['X start', 'Y start', 'X end', 'Y end'] if roi else ['Width mm', 'Height mm'] if chip else ['X1', 'Y1', 'X2', 'Y2'] if key == 'points' else [f'Channel {i+1}' for i in range(len(parsed or [0, 0, 0]))]
        if roi or chip:
            self.optional = QCheckBox('Use region' if roi else 'Use physical chip dimensions', self)
            self.optional.setChecked(parsed is not None)
            layout.addWidget(self.optional, 0, 0, 1, 2)
        if chip:
            self.card = QComboBox(self)
            self.card.addItems(['Custom', 'CLASSIC', 'PASSPORT', 'CAMERATRAX', 'NANO', 'MINI'])
            if isinstance(parsed, str):
                self.card.setCurrentText(parsed.upper())
            layout.addWidget(self.card, 1, 0, 1, 2)
            self.card.currentTextChanged.connect(self.emit_value)
        values = ([0, 0, 100, 100] if roi else [12, 12] if chip else [0]*len(labels)) if parsed is None or isinstance(parsed, str) else [item for row in parsed for item in row] if key == 'points' else list(parsed)
        for index, (label, selected) in enumerate(zip(labels, values)):
            field = QDoubleSpinBox(self)
            field.setDecimals(12 if chip or key == 'points' else 0)
            field.setRange(0 if key in ('lower', 'upper') or roi else 0.001 if chip else -1000000,
                255 if key in ('lower', 'upper') else 1000000)
            field.setValue(selected)
            field.setObjectName(f'{key}_{index}')
            self.fields.append(field)
            layout.addWidget(QLabel(label, self), index+2, 0)
            layout.addWidget(field, index+2, 1)
            field.editingFinished.connect(self.emit_value)
        if self.optional is not None:
            self.optional.toggled.connect(self.emit_value)
        self.sync_fields()

    def sync_fields(self):
        enabled = self.optional is None or self.optional.isChecked()
        for field in self.fields:
            field.setEnabled(enabled and (self.card is None or self.card.currentText() == 'Custom'))
        if self.card is not None:
            self.card.setEnabled(enabled)
        if hasattr(self, 'categories'):
            self.categories.setEnabled(enabled)

    def emit_value(self, *args):
        self.sync_fields()
        if self.optional is not None and not self.optional.isChecked():
            self.changed.emit('null')
            return
        if self.key == 'cat_list':
            try:
                value = [int(part.strip()) for part in self.categories.text().split(',')]
            except ValueError:
                self.categories.setToolTip('Enter comma-separated integer cluster IDs.')
                return
        elif self.card is not None and self.card.currentText() != 'Custom':
            value = self.card.currentText()
        else:
            value = [float(field.value()) if self.key in ('color_chip_size', 'points') else int(field.value()) for field in self.fields]
            if self.key == 'points':
                value = [value[:2], value[2:]]
        self.changed.emit(json.dumps(value))
