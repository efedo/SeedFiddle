"""Native dialogs and value objects for the supervised-learning workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from seedvision.learning.training import TrainingConfiguration


@dataclass(frozen=True, slots=True)
class TrainingImage:
    """A project image with seed outlines, as Train first lists it."""

    key: str
    path: Path
    seed_count: int
    species: str = ""
    reviewed: bool = False
    problem: str | None = None
    group: str = ""


@dataclass(frozen=True, slots=True)
class TrainingDatasetImage:
    """One image chosen for the dataset Train builds."""

    key: str
    path: Path
    identifier: str
    species: str
    group: str
    split: str
    reviewed: bool


@dataclass(frozen=True, slots=True)
class LearningTrainingRequest:
    manifest_path: Path
    output_path: Path
    configuration: TrainingConfiguration
    activate_after_training: bool
    images: tuple[TrainingDatasetImage, ...] = ()
    reviewer: str | None = None


def default_training_splits(images) -> dict[str, str]:
    """Hold out about one group in five per species, never a species' largest."""

    groups: dict[str, list[TrainingImage]] = {}
    for image in images:
        groups.setdefault(image.group or image.key, []).append(image)
    ranked = sorted(
        groups.values(),
        key=lambda members: (-sum(item.seed_count for item in members), members[0].path.name),
    )
    by_species: dict[str, list[list[TrainingImage]]] = {}
    for members in ranked:
        by_species.setdefault(members[0].species, []).append(members)
    held_out = [members for species in by_species.values() for members in species[1::5]]
    if len(ranked) > 1 and not held_out:
        # Every species has a single group, so hold out the second largest overall.
        held_out = [ranked[1]]
    splits = {image.key: "train" for image in images}
    for members in held_out:
        for image in members:
            splits[image.key] = "validation"
    return splits


def _unique_names(images) -> dict[str, str]:
    """Name each image after its file, keeping same-named files apart."""

    names: dict[str, str] = {}
    for image in images:
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", image.path.stem).strip("._-") or "image"
        name, suffix = stem, 2
        while name in names.values():
            name, suffix = f"{stem}_{suffix}", suffix + 1
        names[image.key] = name
    return names


class _PathRow(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(self)
        self.browse = QPushButton("Browse…", self)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.browse)


class LearningTrainingDialog(QDialog):
    """Configure a new or checkpoint-refinement training run."""

    IMAGE_COLUMN, SEEDS_COLUMN, SPECIES_COLUMN, GROUP_COLUMN, SPLIT_COLUMN, REVIEWED_COLUMN, NOTE_COLUMN = range(7)

    def __init__(
        self,
        *,
        project_root: Path,
        images: tuple[TrainingImage, ...] = (),
        species_names: tuple[str, ...] = (),
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._project_root = Path(project_root)
        self._images = tuple(images)
        self._species_names = tuple(species_names)
        self._dataset_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.setWindowTitle("Train or refine learned seed model")
        self.setMinimumWidth(860)
        layout = QVBoxLayout(self)

        dataset = QGroupBox("Images", self)
        dataset_layout = QVBoxLayout(dataset)
        explanation = QLabel(
            "Images are analysed if needed, saved as a dataset, then trained on. "
            "Give related captures the same group: a group stays in one split.",
            dataset,
        )
        explanation.setWordWrap(True)
        dataset_layout.addWidget(explanation)
        self.image_table = self._build_image_table(dataset)
        dataset_layout.addWidget(self.image_table)
        reviewer_form = QFormLayout()
        self.reviewer_edit = QLineEdit(dataset)
        self.reviewer_edit.setPlaceholderText("Who checked the images marked Reviewed")
        reviewer_form.addRow("Reviewer", self.reviewer_edit)
        dataset_layout.addLayout(reviewer_form)
        layout.addWidget(dataset)

        paths = QGroupBox("Checkpoint", self)
        path_form = QFormLayout(paths)
        self.family_combo = QComboBox(paths)
        self.family_combo.addItem("U-Net + watershed", "unet_watershed")
        self.family_combo.addItem("StarDist", "stardist")
        self.output_row = _PathRow(paths)
        self.output_row.edit.setText(str(self._project_root / "models" / "unet_seed_instances.pt"))
        self.output_row.browse.clicked.connect(self._choose_output)
        self.initial_row = _PathRow(paths)
        self.initial_row.edit.setPlaceholderText("Optional checkpoint to refine")
        self.initial_row.browse.clicked.connect(self._choose_initial)
        path_form.addRow("Model", self.family_combo)
        path_form.addRow("Output checkpoint", self.output_row)
        path_form.addRow("Initial checkpoint", self.initial_row)
        layout.addWidget(paths)

        settings = QGroupBox("Training settings", self)
        settings_layout = QVBoxLayout(settings)
        settings_columns = QHBoxLayout()
        first_settings = QFormLayout()
        second_settings = QFormLayout()
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
        first_settings.addRow("Maximum epochs", self.epochs_spin)
        first_settings.addRow("Batch size", self.batch_spin)
        first_settings.addRow("Tile size", self.tile_spin)
        first_settings.addRow("Tiles per image / epoch", self.tiles_per_sample_spin)
        second_settings.addRow("Learning rate", self.learning_rate_spin)
        second_settings.addRow("StarDist rays", self.rays_spin)
        second_settings.addRow("Base channels", self.base_channels_spin)
        second_settings.addRow("Network depth", self.depth_spin)
        settings_columns.addLayout(first_settings)
        settings_columns.addLayout(second_settings)
        settings_layout.addLayout(settings_columns)
        settings_layout.addWidget(self.activate_checkbox)
        layout.addWidget(settings)

        validation_note = QLabel(
            "The best validation-loss checkpoint and a JSON training report are saved.",
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

    def _build_image_table(self, parent) -> QTableWidget:
        table = QTableWidget(len(self._images), 7, parent)
        table.setHorizontalHeaderLabels(
            ("Image", "Seeds", "Species", "Group", "Split", "Reviewed", "Note")
        )
        table.verticalHeader().hide()
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.NOTE_COLUMN, QHeaderView.ResizeMode.Stretch)
        usable = [image for image in self._images if image.problem is None]
        splits = default_training_splits(usable)
        names = _unique_names(self._images)
        for row, image in enumerate(self._images):
            enabled = image.problem is None
            base = Qt.ItemFlag.ItemIsEnabled if enabled else Qt.ItemFlag.NoItemFlags
            name = QTableWidgetItem(image.path.name)
            name.setFlags(base | Qt.ItemFlag.ItemIsUserCheckable)
            name.setCheckState(Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked)
            table.setItem(row, self.IMAGE_COLUMN, name)
            seeds = QTableWidgetItem(f"{image.seed_count:,}")
            seeds.setFlags(base)
            seeds.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, self.SEEDS_COLUMN, seeds)
            species = QComboBox(table)
            species.addItem("Choose…", "")
            for option in self._species_names:
                species.addItem(option, option)
            species.setCurrentIndex(max(0, species.findData(image.species)))
            species.setEnabled(enabled)
            table.setCellWidget(row, self.SPECIES_COLUMN, species)
            group = QTableWidgetItem(image.group or names[image.key])
            group.setFlags(base | Qt.ItemFlag.ItemIsEditable if enabled else base)
            table.setItem(row, self.GROUP_COLUMN, group)
            split = QComboBox(table)
            split.addItem("Training", "train")
            split.addItem("Validation", "validation")
            split.setCurrentIndex(max(0, split.findData(splits.get(image.key, "train"))))
            split.setEnabled(enabled)
            split.currentIndexChanged.connect(lambda _index, current=row: self._split_changed(current))
            table.setCellWidget(row, self.SPLIT_COLUMN, split)
            reviewed = QTableWidgetItem()
            reviewed.setFlags(base | Qt.ItemFlag.ItemIsUserCheckable)
            reviewed.setCheckState(
                Qt.CheckState.Checked if image.reviewed and enabled else Qt.CheckState.Unchecked
            )
            table.setItem(row, self.REVIEWED_COLUMN, reviewed)
            note = QTableWidgetItem(image.problem or "")
            note.setFlags(Qt.ItemFlag.ItemIsEnabled)
            note.setToolTip(image.problem or "")
            table.setItem(row, self.NOTE_COLUMN, note)
        table.itemChanged.connect(self._image_item_changed)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.setMinimumHeight(
            table.horizontalHeader().sizeHint().height()
            + sum(table.rowHeight(row) for row in range(min(8, len(self._images))))
            + 2 * table.frameWidth()
        )
        return table

    def _group_of(self, row: int) -> str:
        return self.image_table.item(row, self.GROUP_COLUMN).text().strip()

    def _split_combo(self, row: int) -> QComboBox:
        return self.image_table.cellWidget(row, self.SPLIT_COLUMN)

    def _image_item_changed(self, item: QTableWidgetItem) -> None:
        """Move an image into the split of any group it joins."""

        if item.column() != self.GROUP_COLUMN:
            return
        group = self._group_of(item.row())
        for row in range(self.image_table.rowCount()):
            if row != item.row() and group and self._group_of(row) == group:
                with QSignalBlocker(self._split_combo(item.row())):
                    self._split_combo(item.row()).setCurrentIndex(self._split_combo(row).currentIndex())
                return

    def _split_changed(self, changed: int) -> None:
        """Keep every image of a group in the split just chosen for it."""

        group = self._group_of(changed)
        for row in range(self.image_table.rowCount()):
            if row != changed and group and self._group_of(row) == group:
                with QSignalBlocker(self._split_combo(row)):
                    self._split_combo(row).setCurrentIndex(self._split_combo(changed).currentIndex())

    def selected_images(self) -> tuple[TrainingDatasetImage, ...]:
        names = _unique_names(self._images)
        selected = []
        for row, image in enumerate(self._images):
            if self.image_table.item(row, self.IMAGE_COLUMN).checkState() != Qt.CheckState.Checked:
                continue
            selected.append(TrainingDatasetImage(
                key=image.key,
                path=image.path,
                identifier=names[image.key],
                species=str(self.image_table.cellWidget(row, self.SPECIES_COLUMN).currentData()),
                group=self._group_of(row),
                split=str(self._split_combo(row).currentData()),
                reviewed=self.image_table.item(row, self.REVIEWED_COLUMN).checkState() == Qt.CheckState.Checked,
            ))
        return tuple(selected)

    @staticmethod
    def _dataset_problem(request: LearningTrainingRequest) -> str | None:
        images = request.images
        splits = {image.split for image in images}
        if "train" not in splits or "validation" not in splits:
            return "Use at least one Training image and one Validation image."
        unnamed = [image.path.name for image in images if not image.species]
        if unnamed:
            return "Choose a species for: " + ", ".join(unnamed) + "."
        ungrouped = [image.path.name for image in images if not image.group]
        if ungrouped:
            return "Give a group to: " + ", ".join(ungrouped) + "."
        if any(image.reviewed for image in images) and not request.reviewer:
            return "Enter who checked the images marked Reviewed."
        return None

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
        dataset = self._project_root / "learning-data" / f"{output.stem}-{self._dataset_stamp}"
        return LearningTrainingRequest(
            manifest_path=dataset / "manifest.json",
            output_path=output,
            configuration=configuration,
            activate_after_training=self.activate_checkbox.isChecked(),
            images=self.selected_images(),
            reviewer=self.reviewer_edit.text().strip() or None,
        )

    def accept(self) -> None:
        try:
            request = self.request()
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "Invalid training settings", str(error))
            return
        problem = self._dataset_problem(request)
        if problem:
            QMessageBox.warning(self, "Dataset is not ready for training", problem)
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
