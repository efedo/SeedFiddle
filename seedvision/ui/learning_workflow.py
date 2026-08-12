"""Native dialogs and value objects for the supervised-learning workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from seedvision.learning.data import LearningManifest
from seedvision.learning.training import TrainingConfiguration


@dataclass(frozen=True, slots=True)
class LearningExportOptions:
    manifest_path: Path
    dataset_id: str
    identifier: str
    group: str
    split: str
    reviewed: bool
    annotation_author: str | None
    annotation_revision: str | None
    notes: str | None


@dataclass(frozen=True, slots=True)
class LearningTrainingRequest:
    manifest_path: Path
    output_path: Path
    configuration: TrainingConfiguration
    activate_after_training: bool


class _PathRow(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(self)
        self.browse = QPushButton("Browse…", self)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.browse)


class LearningExportDialog(QDialog):
    """Collect provenance and split metadata at the point of export."""

    def __init__(
        self,
        *,
        default_directory: Path,
        default_identifier: str,
        default_group: str,
        default_revision: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export reviewed learning sample")
        self.setMinimumWidth(610)
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Export a complete instance mask and the exact model input features. "
            "All captures from one biological lot or imaging session must keep the "
            "same group and must never cross dataset splits.",
            self,
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form = QFormLayout()
        self.directory_row = _PathRow(self)
        self.directory_row.edit.setText(str(default_directory))
        self.directory_row.browse.clicked.connect(self._choose_directory)
        self.dataset_id_edit = QLineEdit("seed-fiddle-human-annotations", self)
        self.identifier_edit = QLineEdit(default_identifier, self)
        self.group_edit = QLineEdit(default_group, self)
        self.group_edit.setPlaceholderText("e.g. lot-2026-08/capture-session-a")
        self.split_combo = QComboBox(self)
        self.split_combo.addItem("Training", "train")
        self.split_combo.addItem("Validation", "validation")
        self.split_combo.addItem("Locked test (never used for refinement)", "test")
        self.revision_edit = QLineEdit(default_revision, self)
        self.author_edit = QLineEdit(self)
        self.author_edit.setPlaceholderText("Annotator or reviewer name")
        self.reviewed_checkbox = QCheckBox(
            "Complete mask has been checked seed-by-seed by a human", self
        )
        self.notes_edit = QLineEdit(self)
        form.addRow("Dataset folder", self.directory_row)
        form.addRow("Dataset ID", self.dataset_id_edit)
        form.addRow("Sample ID", self.identifier_edit)
        form.addRow("Lot / capture group", self.group_edit)
        form.addRow("Dataset split", self.split_combo)
        form.addRow("Annotation revision", self.revision_edit)
        form.addRow("Annotator / reviewer", self.author_edit)
        form.addRow("Review state", self.reviewed_checkbox)
        form.addRow("Notes", self.notes_edit)
        layout.addLayout(form)

        caution = QLabel(
            "Unchecked samples may be saved for later review, but Seed Fiddle will "
            "refuse to train from them. Test labels must remain locked until final evaluation.",
            self,
        )
        caution.setWordWrap(True)
        layout.addWidget(caution)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Save,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.directory_row.edit.editingFinished.connect(self._load_existing_dataset)
        self._load_existing_dataset()

    def _choose_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Choose or create a learning dataset folder", self.directory_row.edit.text()
        )
        if selected:
            self.directory_row.edit.setText(selected)
            self._load_existing_dataset()

    def _load_existing_dataset(self) -> None:
        manifest_path = Path(self.directory_row.edit.text().strip()) / "manifest.json"
        if not manifest_path.is_file():
            self.dataset_id_edit.setReadOnly(False)
            return
        try:
            dataset_id = LearningManifest.load(manifest_path).dataset_id
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            self.dataset_id_edit.setReadOnly(False)
            return
        self.dataset_id_edit.setText(dataset_id)
        self.dataset_id_edit.setReadOnly(True)

    def options(self) -> LearningExportOptions:
        author = self.author_edit.text().strip() or None
        revision = self.revision_edit.text().strip() or None
        notes = self.notes_edit.text().strip() or None
        return LearningExportOptions(
            manifest_path=(
                Path(self.directory_row.edit.text().strip()).expanduser() / "manifest.json"
            ).resolve(),
            dataset_id=self.dataset_id_edit.text().strip(),
            identifier=self.identifier_edit.text().strip(),
            group=self.group_edit.text().strip(),
            split=str(self.split_combo.currentData()),
            reviewed=self.reviewed_checkbox.isChecked(),
            annotation_author=author,
            annotation_revision=revision,
            notes=notes,
        )

    def accept(self) -> None:
        options = self.options()
        if not options.dataset_id or not options.identifier or not options.group:
            QMessageBox.warning(
                self,
                "Incomplete learning metadata",
                "Dataset ID, sample ID, and lot/capture group are required.",
            )
            return
        if options.reviewed and not options.annotation_author:
            QMessageBox.warning(
                self,
                "Reviewer required",
                "Enter the person who checked the complete mask before marking it reviewed.",
            )
            return
        if options.manifest_path.is_file():
            try:
                manifest = LearningManifest.load(options.manifest_path)
            except Exception as error:  # noqa: BLE001 - user-facing validation boundary
                QMessageBox.warning(self, "Invalid learning manifest", str(error))
                return
            incompatible = {
                sample.split
                for sample in manifest.samples
                if sample.group == options.group and sample.split != options.split
            }
            if incompatible:
                QMessageBox.warning(
                    self,
                    "Dataset leakage prevented",
                    f"Group {options.group!r} already belongs to {sorted(incompatible)}. "
                    "All related captures and revisions must remain in one split.",
                )
                return
        super().accept()


class LearningTrainingDialog(QDialog):
    """Configure a new or checkpoint-refinement training run."""

    def __init__(self, *, project_root: Path, parent=None) -> None:
        super().__init__(parent)
        self._project_root = Path(project_root)
        self.setWindowTitle("Train or refine learned seed model")
        self.setMinimumWidth(650)
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Training uses only human-reviewed training and validation samples. "
            "Choose an initial checkpoint to refine it; leave that field empty to train "
            "a new model. The locked test split is never used here.",
            self,
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        paths = QGroupBox("Dataset and checkpoint", self)
        path_form = QFormLayout(paths)
        self.manifest_row = _PathRow(paths)
        self.manifest_row.edit.setText(str(self._project_root / "learning-data" / "manifest.json"))
        self.manifest_row.browse.clicked.connect(self._choose_manifest)
        self.family_combo = QComboBox(paths)
        self.family_combo.addItem("U-Net + watershed", "unet_watershed")
        self.family_combo.addItem("StarDist", "stardist")
        self.output_row = _PathRow(paths)
        self.output_row.edit.setText(str(self._project_root / "models" / "unet_seed_instances.pt"))
        self.output_row.browse.clicked.connect(self._choose_output)
        self.initial_row = _PathRow(paths)
        self.initial_row.edit.setPlaceholderText("Optional checkpoint to refine")
        self.initial_row.browse.clicked.connect(self._choose_initial)
        path_form.addRow("Learning manifest", self.manifest_row)
        path_form.addRow("Model", self.family_combo)
        path_form.addRow("Output checkpoint", self.output_row)
        path_form.addRow("Initial checkpoint", self.initial_row)
        layout.addWidget(paths)

        settings = QGroupBox("Training settings", self)
        settings_form = QFormLayout(settings)
        self.epochs_spin = QSpinBox(settings)
        self.epochs_spin.setRange(1, 10_000)
        self.epochs_spin.setValue(40)
        self.batch_spin = QSpinBox(settings)
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(4)
        self.tile_spin = QSpinBox(settings)
        self.tile_spin.setRange(32, 4096)
        self.tile_spin.setSingleStep(32)
        self.tile_spin.setValue(256)
        self.tiles_per_sample_spin = QSpinBox(settings)
        self.tiles_per_sample_spin.setRange(1, 1024)
        self.tiles_per_sample_spin.setValue(16)
        self.learning_rate_spin = QDoubleSpinBox(settings)
        self.learning_rate_spin.setDecimals(7)
        self.learning_rate_spin.setRange(0.0000001, 1.0)
        self.learning_rate_spin.setValue(0.0002)
        self.rays_spin = QSpinBox(settings)
        self.rays_spin.setRange(8, 256)
        self.rays_spin.setSingleStep(4)
        self.rays_spin.setValue(32)
        self.base_channels_spin = QSpinBox(settings)
        self.base_channels_spin.setRange(8, 256)
        self.base_channels_spin.setValue(24)
        self.depth_spin = QSpinBox(settings)
        self.depth_spin.setRange(1, 5)
        self.depth_spin.setValue(4)
        self.activate_checkbox = QCheckBox(
            "Activate the trained checkpoint in its pipeline node", settings
        )
        self.activate_checkbox.setChecked(True)
        settings_form.addRow("Maximum epochs", self.epochs_spin)
        settings_form.addRow("Batch size", self.batch_spin)
        settings_form.addRow("Tile size", self.tile_spin)
        settings_form.addRow("Tiles per image / epoch", self.tiles_per_sample_spin)
        settings_form.addRow("Learning rate", self.learning_rate_spin)
        settings_form.addRow("StarDist rays", self.rays_spin)
        settings_form.addRow("Base channels", self.base_channels_spin)
        settings_form.addRow("Network depth", self.depth_spin)
        settings_form.addRow("After training", self.activate_checkbox)
        layout.addWidget(settings)

        validation_note = QLabel(
            "The best validation-loss checkpoint and a JSON training report are saved. "
            "Training success is not scientific validation; final metrics require a frozen, "
            "independently reviewed test set.",
            self,
        )
        validation_note.setWordWrap(True)
        layout.addWidget(validation_note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Start training")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.family_combo.currentIndexChanged.connect(self._family_changed)
        self._family_changed()

    def _choose_manifest(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Choose learning manifest", self.manifest_row.edit.text(), "JSON (*.json)"
        )
        if selected:
            self.manifest_row.edit.setText(selected)

    def _choose_output(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save learned checkpoint",
            self.output_row.edit.text(),
            "PyTorch checkpoint (*.pt)",
        )
        if selected:
            self.output_row.edit.setText(selected)

    def _choose_initial(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Choose checkpoint to refine",
            self.initial_row.edit.text() or str(self._project_root / "models"),
            "PyTorch checkpoint (*.pt)",
        )
        if selected:
            self.initial_row.edit.setText(selected)
            try:
                from seedvision.learning.checkpoint import load_checkpoint

                _model, _feature_spec, payload = load_checkpoint(selected, device="cpu")
                family = str(payload["family"])
                index = self.family_combo.findData(family)
                if index >= 0:
                    self.family_combo.setCurrentIndex(index)
                configuration = payload["model_configuration"]
                self.base_channels_spin.setValue(int(configuration["base_channels"]))
                self.depth_spin.setValue(int(configuration["depth"]))
                if family == "stardist":
                    self.rays_spin.setValue(int(configuration["ray_count"]))
            except Exception as error:  # noqa: BLE001 - user-facing checkpoint boundary
                self.initial_row.edit.clear()
                QMessageBox.warning(
                    self, "Incompatible initial checkpoint", str(error)
                )

    def _family_changed(self) -> None:
        family = str(self.family_combo.currentData())
        filename = "unet_seed_instances.pt" if family == "unet_watershed" else "stardist_seed_instances.pt"
        self.output_row.edit.setText(str(self._project_root / "models" / filename))
        self.rays_spin.setEnabled(family == "stardist")

    def request(self) -> LearningTrainingRequest:
        manifest = Path(self.manifest_row.edit.text().strip()).expanduser().resolve()
        output = Path(self.output_row.edit.text().strip()).expanduser().resolve()
        initial_text = self.initial_row.edit.text().strip()
        initial = str(Path(initial_text).expanduser().resolve()) if initial_text else None
        configuration = TrainingConfiguration(
            family=str(self.family_combo.currentData()),
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            tile_size=self.tile_spin.value(),
            tiles_per_sample=self.tiles_per_sample_spin.value(),
            ray_count=self.rays_spin.value(),
            base_channels=self.base_channels_spin.value(),
            depth=self.depth_spin.value(),
            learning_rate=self.learning_rate_spin.value(),
            initial_checkpoint=initial,
        )
        return LearningTrainingRequest(
            manifest_path=manifest,
            output_path=output,
            configuration=configuration,
            activate_after_training=self.activate_checkbox.isChecked(),
        )

    def accept(self) -> None:
        try:
            request = self.request()
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "Invalid training settings", str(error))
            return
        if not request.manifest_path.is_file():
            QMessageBox.warning(self, "Manifest not found", str(request.manifest_path))
            return
        initial = request.configuration.initial_checkpoint
        if initial and not Path(initial).is_file():
            QMessageBox.warning(self, "Initial checkpoint not found", initial)
            return
        if initial and Path(initial).resolve() == request.output_path:
            QMessageBox.warning(
                self,
                "Choose a new output checkpoint",
                "Refinement must write a new checkpoint so the source checkpoint remains recoverable.",
            )
            return
        if request.output_path.exists():
            answer = QMessageBox.question(
                self,
                "Replace checkpoint?",
                f"{request.output_path}\n\nReplace this checkpoint after validation improves?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        super().accept()


def format_learning_audit(audit: dict) -> str:
    """Return a concise, user-facing audit summary."""

    splits = audit.get("splits", {})
    lines = [
        f"Dataset: {audit.get('dataset_id', 'unknown')}",
        f"Samples: {audit.get('sample_count', 0):,}",
        f"Reviewed: {audit.get('reviewed_sample_count', 0):,}",
        f"Instances: {audit.get('instance_count', 0):,}",
        "Splits: " + ", ".join(f"{name}={count}" for name, count in sorted(splits.items())),
    ]
    errors = list(audit.get("errors", ()))
    warnings = list(audit.get("warnings", ()))
    if errors:
        lines.extend(("", "Errors:", *(f"• {item}" for item in errors)))
    if warnings:
        lines.extend(("", "Warnings:", *(f"• {item}" for item in warnings)))
    if not errors and not warnings:
        lines.extend(("", "No audit errors or warnings."))
    return "\n".join(lines)
