"""Project-facing manager for immutable species-reference library versions."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from seedvision.reference_library import (
    SPECIES_LIBRARY_EXTENSION,
    BiologicalContext,
    SpeciesLibraryPin,
    SpeciesLibraryService,
)


class SpeciesLibraryManagerDialog(QDialog):
    """Build, audit, publish, fork, import, and pin immutable libraries."""

    pin_selected = Signal(object)
    build_requested = Signal(object, str, object, object, object, object)

    def __init__(
        self,
        *,
        service: SpeciesLibraryService,
        context: BiologicalContext,
        image_paths: tuple[Path, ...],
        current_pin: SpeciesLibraryPin | None,
        capture_group_id: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.service = service
        self.context = context
        self.current_pin = current_pin
        self._manifests = ()
        self.setWindowTitle("Species reference libraries")
        self.resize(1100, 760)
        root = QVBoxLayout(self)
        self.summary_label = QLabel(self)
        self.summary_label.setWordWrap(True)
        root.addWidget(self.summary_label)
        self.tabs = QTabWidget(self)
        root.addWidget(self.tabs, 1)

        # Tab 1: choose exactly which immutable version this project uses.
        libraries = QWidget(self.tabs)
        libraries_layout = QVBoxLayout(libraries)
        library_help = QLabel(
            "Choose an installed immutable version for future analyses in this "
            "project. Pinning never edits annotations or library contents, and a "
            "project always keeps the exact selected content hash—it does not "
            "silently follow a newer version.",
            libraries,
        )
        library_help.setWordWrap(True)
        libraries_layout.addWidget(library_help)
        self.version_table = QTableWidget(0, 7, libraries)
        self.version_table.setHorizontalHeaderLabels(
            (
                "Used by project",
                "Version",
                "Availability",
                "Training images",
                "Included evidence",
                "Published",
                "Content ID (SHA-256)",
            )
        )
        self.version_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.version_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.version_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        libraries_layout.addWidget(self.version_table, 1)
        self.selected_library_label = QLabel("No installed version is selected.", libraries)
        self.selected_library_label.setWordWrap(True)
        libraries_layout.addWidget(self.selected_library_label)
        buttons = QHBoxLayout()
        self.pin_button = QPushButton("Use selected version for this project", libraries)
        self.pin_button.setToolTip(
            "Pin this exact library ID, version, species, and content hash to the project."
        )
        self.export_button = QPushButton("Export selected…", libraries)
        self.export_button.setToolTip("Save the selected immutable library as a portable bundle.")
        self.import_button = QPushButton("Import library file…", libraries)
        self.import_button.setToolTip("Install and verify a portable species-library bundle.")
        self.retire_button = QPushButton("Retire selected version", libraries)
        self.retire_button.setToolTip(
            "Retiring hides a version from normal selection without deleting its immutable data."
        )
        for button in (self.pin_button, self.export_button, self.import_button, self.retire_button):
            buttons.addWidget(button)
        libraries_layout.addLayout(buttons)
        self.tabs.addTab(libraries, "1. Use a library")

        # Tab 2: one linear create/fork workflow rather than separate Sources
        # and Publish tabs whose relationship was not visible.
        sources = QWidget(self.tabs)
        sources_layout = QVBoxLayout(sources)
        source_group = QGroupBox("1. Choose reviewed project images", sources)
        source_group_layout = QVBoxLayout(source_group)
        source_help = QLabel(
            "Checked images contribute only their saved, reviewed annotations. "
            "Before publication, each source is revalidated against its file "
            "fingerprint, dimensions, species, and trait vocabulary; invalid sources "
            "are reported and never silently skipped.",
            source_group,
        )
        source_help.setWordWrap(True)
        source_group_layout.addWidget(source_help)
        self.source_selection_label = QLabel(source_group)
        source_group_layout.addWidget(self.source_selection_label)
        source_buttons = QHBoxLayout()
        self.select_all_sources_button = QPushButton("Select all", source_group)
        self.clear_sources_button = QPushButton("Clear selection", source_group)
        source_buttons.addWidget(self.select_all_sources_button)
        source_buttons.addWidget(self.clear_sources_button)
        source_buttons.addStretch(1)
        source_group_layout.addLayout(source_buttons)
        self.source_list = QListWidget(source_group)
        for path in image_paths:
            item = QListWidgetItem(path.name, self.source_list)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
        source_group_layout.addWidget(self.source_list, 1)
        sources_layout.addWidget(source_group, 2)

        context_group = QGroupBox("2. Describe these sources (optional)", sources)
        context_group_layout = QVBoxLayout(context_group)
        context_help = QLabel(
            "These identifiers let future models partially pool related accessions, "
            "lots, and repeated photographs without treating them as the same seed.",
            context_group,
        )
        context_help.setWordWrap(True)
        context_group_layout.addWidget(context_help)
        context_form = QFormLayout()
        self.lineage_edit = QLineEdit(
            context.lineage_group_id or "", context_group
        )
        self.accession_edit = QLineEdit(context.accession_id or "", context_group)
        self.seed_lot_edit = QLineEdit(context.seed_lot_id or "", context_group)
        self.capture_group_edit = QLineEdit(capture_group_id or "", context_group)
        for widget in (
            self.lineage_edit,
            self.accession_edit,
            self.seed_lot_edit,
            self.capture_group_edit,
        ):
            widget.setMaxLength(128)
        self.lineage_edit.setToolTip(
            "Broad genetic or breeding lineage shared by related accessions."
        )
        self.accession_edit.setToolTip(
            "Accession or line represented by the selected images."
        )
        self.seed_lot_edit.setToolTip(
            "Specific harvested or supplied seed lot within an accession."
        )
        self.capture_group_edit.setToolTip(
            "Use the same value for repeated photographs from one capture session; "
            "this prevents near-duplicate views being counted as independent evidence."
        )
        context_form.addRow("Lineage / group", self.lineage_edit)
        context_form.addRow("Accession", self.accession_edit)
        context_form.addRow("Seed lot", self.seed_lot_edit)
        context_form.addRow("Repeated-photo group", self.capture_group_edit)
        context_group_layout.addLayout(context_form)
        sources_layout.addWidget(context_group)

        publish = QGroupBox("3. Validate and publish an immutable version", sources)
        publish_layout = QVBoxLayout(publish)
        notice = QLabel(
            "Publishing creates a new read-only version. Applying or saving image "
            "annotations never changes an installed library.",
            publish,
        )
        notice.setWordWrap(True)
        publish_layout.addWidget(notice)
        form = QFormLayout()
        self.version_edit = QLineEdit("1", publish)
        self.version_edit.setMaxLength(128)
        self.version_edit.setToolTip("A stable name or number for this new immutable version.")
        form.addRow("New version name / number", self.version_edit)
        self.fork_checkbox = QCheckBox(
            "Start from the selected installed version", publish
        )
        self.fork_checkbox.setToolTip(
            "Carry the selected version's sources forward, then optionally remove "
            "individual inherited sources below."
        )
        form.addRow("Existing sources", self.fork_checkbox)
        publish_layout.addLayout(form)
        self.retained_sources_label = QLabel(
            "Inherited sources to keep (uncheck any source to remove it from the new version):",
            publish,
        )
        self.retained_sources_label.setWordWrap(True)
        publish_layout.addWidget(self.retained_sources_label)
        self.retained_source_list = QListWidget(publish)
        self.retained_source_list.setMaximumHeight(120)
        publish_layout.addWidget(self.retained_source_list)
        self.build_button = QPushButton("Validate sources and publish new version", publish)
        publish_layout.addWidget(self.build_button)
        self.build_status_label = QLabel(
            "No build is running. Validation problems will be shown here before "
            "anything is published.",
            publish,
        )
        self.build_status_label.setWordWrap(True)
        publish_layout.addWidget(self.build_status_label)
        sources_layout.addWidget(publish)
        self.tabs.addTab(sources, "2. Create a version")

        # Tab 3: put product coverage and detailed validation together.
        inspect = QWidget(self.tabs)
        inspect_layout = QVBoxLayout(inspect)
        inspect_help = QLabel(
            "Select a version on the first tab to inspect exactly which evidence "
            "products it contains, their support, and any validation warnings.",
            inspect,
        )
        inspect_help.setWordWrap(True)
        inspect_layout.addWidget(inspect_help)
        self.coverage_table = QTableWidget(0, 8, inspect)
        self.coverage_table.setHorizontalHeaderLabels(
            (
                "Evidence product",
                "Quality tier",
                "Images",
                "Seeds",
                "Samples",
                "Prototypes",
                "Effective weight",
                "Warnings",
            )
        )
        self.coverage_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        inspect_layout.addWidget(self.coverage_table, 2)
        self.validation_label = QLabel(inspect)
        self.validation_label.setWordWrap(True)
        self.validation_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        inspect_layout.addWidget(self.validation_label, 1)
        self.tabs.addTab(inspect, "3. Inspect selected version")

        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.accept)
        root.addWidget(close_button)
        self.pin_button.clicked.connect(self._pin_selected)
        self.export_button.clicked.connect(self._export_selected)
        self.import_button.clicked.connect(self._import_bundle)
        self.retire_button.clicked.connect(self._toggle_retired)
        self.build_button.clicked.connect(self._request_build)
        self.version_table.itemSelectionChanged.connect(self._show_selected_validation)
        self.fork_checkbox.toggled.connect(self._fork_toggled)
        self.source_list.itemChanged.connect(self._source_selection_changed)
        self.select_all_sources_button.clicked.connect(self._select_all_sources)
        self.clear_sources_button.clicked.connect(self._clear_source_selection)
        self._source_selection_changed()
        self._fork_toggled(False)
        self.refresh()

    def refresh(self) -> None:
        manifests = tuple(
            value for value in self.service.store.list_manifests()
            if value.species_id == self.context.species_id
        )
        self._manifests = manifests
        self.version_table.setRowCount(len(manifests))
        for row, manifest in enumerate(manifests):
            pin = SpeciesLibraryPin(
                manifest.library_id, manifest.version, manifest.species_id,
                manifest.content_sha256,
            )
            values = (
                "●" if self.current_pin == pin else "",
                manifest.version,
                manifest.status.value,
                len(manifest.sources),
                ", ".join(f"{item.product.value}: {item.tier.value}" for item in manifest.products),
                manifest.created_utc,
                manifest.content_sha256,
            )
            for column, value in enumerate(values):
                self.version_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.version_table.resizeColumnsToContents()
        self.summary_label.setText(
            f"Current project species: {self.context.species_id}. "
            f"{len(manifests)} immutable version(s) are installed. "
            + (
                "The project is not using a species library."
                if self.current_pin is None
                else f"The project uses version {self.current_pin.version} "
                f"({self.current_pin.sha256[:12]}…)."
            )
        )
        if manifests and self.version_table.currentRow() < 0:
            self.version_table.selectRow(0)
        self._show_selected_validation()

    def _source_selection_changed(self, *_unused) -> None:
        selected = sum(
            self.source_list.item(index).checkState() == Qt.CheckState.Checked
            for index in range(self.source_list.count())
        )
        self.source_selection_label.setText(
            f"{selected} of {self.source_list.count()} project image(s) selected."
        )

    def _select_all_sources(self) -> None:
        for index in range(self.source_list.count()):
            self.source_list.item(index).setCheckState(Qt.CheckState.Checked)

    def _clear_source_selection(self) -> None:
        for index in range(self.source_list.count()):
            self.source_list.item(index).setCheckState(Qt.CheckState.Unchecked)

    def _fork_toggled(self, checked: bool) -> None:
        available = checked and self._selected_manifest() is not None
        self.retained_sources_label.setVisible(checked)
        self.retained_source_list.setVisible(checked)
        self.retained_source_list.setEnabled(available)
        if checked and not available:
            self.retained_sources_label.setText(
                "Select an installed version on the first tab before carrying "
                "its sources forward."
            )
        else:
            self.retained_sources_label.setText(
                "Inherited sources to keep (uncheck any source to remove it from "
                "the new version):"
            )

    def _refresh_version_actions(self) -> None:
        manifest = self._selected_manifest()
        selected = manifest is not None
        self.pin_button.setEnabled(selected)
        self.export_button.setEnabled(selected)
        self.retire_button.setEnabled(selected)
        self.retire_button.setText(
            "Restore selected version"
            if selected and manifest.status.value == "retired"
            else "Retire selected version"
        )
        self._fork_toggled(self.fork_checkbox.isChecked())

    def set_build_busy(self, busy: bool, message: str = "") -> None:
        for widget in (
            self.build_button, self.source_list, self.version_edit,
            self.fork_checkbox, self.retained_source_list,
            self.lineage_edit, self.accession_edit, self.seed_lot_edit,
            self.capture_group_edit,
        ):
            widget.setEnabled(not busy)
        self.select_all_sources_button.setEnabled(not busy)
        self.clear_sources_button.setEnabled(not busy)
        if not busy:
            self._refresh_version_actions()
        if message:
            self.validation_label.setText(message)
            self.build_status_label.setText(message)

    def published(self, pin: SpeciesLibraryPin) -> None:
        self.current_pin = pin
        self.set_build_busy(False, "The validated immutable version was published and pinned.")
        self.refresh()
        self.build_status_label.setText(
            "Published successfully and selected for this project."
        )
        self.pin_selected.emit(pin)

    def build_failed(self, message: str) -> None:
        self.set_build_busy(False, message)
        QMessageBox.critical(self, "Species-library build failed", message)

    def _selected_manifest(self):
        row = self.version_table.currentRow()
        return self._manifests[row] if 0 <= row < len(self._manifests) else None

    def _selected_pin(self):
        manifest = self._selected_manifest()
        return None if manifest is None else SpeciesLibraryPin(
            manifest.library_id, manifest.version, manifest.species_id,
            manifest.content_sha256,
        )

    def _pin_selected(self) -> None:
        pin = self._selected_pin()
        if pin is None:
            return
        self.current_pin = pin
        self.refresh()
        self.pin_selected.emit(pin)

    def _export_selected(self) -> None:
        manifest = self._selected_manifest()
        pin = self._selected_pin()
        if manifest is None or pin is None:
            return
        target, _selected = QFileDialog.getSaveFileName(
            self, "Export species library",
            f"{manifest.species_id}-{manifest.version}{SPECIES_LIBRARY_EXTENSION}",
            f"Seed Fiddle species library (*{SPECIES_LIBRARY_EXTENSION})",
        )
        if not target:
            return
        try:
            self.service.store.export_library(pin, Path(target))
        except Exception as error:
            QMessageBox.critical(self, "Export failed", str(error))

    def _import_bundle(self) -> None:
        source, _selected = QFileDialog.getOpenFileName(
            self, "Import species library", "",
            f"Seed Fiddle species library (*{SPECIES_LIBRARY_EXTENSION})",
        )
        if not source:
            return
        try:
            artifact = self.service.store.import_bundle(Path(source))
        except Exception as error:
            QMessageBox.critical(self, "Import failed", str(error))
            return
        if artifact.manifest.species_id != self.context.species_id:
            QMessageBox.information(
                self,
                "Library installed for another species",
                f"Installed the valid library for {artifact.manifest.species_display_name}; "
                "it is not eligible for the current project's species.",
            )
        self.refresh()

    def _toggle_retired(self) -> None:
        manifest = self._selected_manifest()
        pin = self._selected_pin()
        if manifest is None or pin is None:
            return
        try:
            self.service.store.retire(pin, retired=manifest.status.value != "retired")
        except Exception as error:
            QMessageBox.critical(self, "Could not change library status", str(error))
            return
        self.refresh()

    def _request_build(self) -> None:
        version = self.version_edit.text().strip()
        paths = tuple(
            Path(item.data(Qt.ItemDataRole.UserRole))
            for index in range(self.source_list.count())
            if (item := self.source_list.item(index)).checkState() == Qt.CheckState.Checked
        )
        parent_pin = self._selected_pin() if self.fork_checkbox.isChecked() else None
        if not version or (not paths and parent_pin is None):
            QMessageBox.warning(
                self, "Incomplete build",
                "Enter a version and select project sources or a parent version.",
            )
            return
        removed = tuple(
            str(item.data(Qt.ItemDataRole.UserRole))
            for index in range(self.retained_source_list.count())
            if (item := self.retained_source_list.item(index)).checkState() != Qt.CheckState.Checked
        ) if parent_pin is not None else ()
        optional = lambda value: value.strip() or None
        try:
            context = BiologicalContext(
                self.context.species_id,
                lineage_group_id=optional(self.lineage_edit.text()),
                accession_id=optional(self.accession_edit.text()),
                seed_lot_id=optional(self.seed_lot_edit.text()),
            )
            capture_group = optional(self.capture_group_edit.text())
            # Apply the same stable-identifier validation used by persisted
            # source records before dispatching expensive extraction.
            if capture_group is not None:
                from seedvision.reference_library.contracts import LibrarySourceRecord
                LibrarySourceRecord(
                    "0" * 64, "1" * 64, "validation", (1, 1),
                    capture_group_id=capture_group,
                    biological_context=context,
                )
        except ValueError as error:
            QMessageBox.warning(self, "Invalid biological context", str(error))
            return
        self.set_build_busy(True, "Preparing reviewed source contributions…")
        self.build_requested.emit(
            paths, version, parent_pin, removed, context, capture_group
        )

    def _show_selected_validation(self) -> None:
        manifest = self._selected_manifest()
        self.coverage_table.setRowCount(0 if manifest is None else len(manifest.products))
        self.retained_source_list.clear()
        if manifest is None:
            self.validation_label.setText("No installed version is selected.")
            self.selected_library_label.setText(
                "No installed version is selected. Import a library file or create "
                "a version on the next tab."
            )
            self._refresh_version_actions()
            return
        pin = self._selected_pin()
        self.selected_library_label.setText(
            f"Selected: {manifest.library_id} version {manifest.version} "
            f"({manifest.status.value}); {len(manifest.sources)} training image(s). "
            + (
                "This exact version is currently used by the project."
                if pin == self.current_pin
                else "Use the button below to make this the project's exact version."
            )
        )
        lines = [
            f"Library {manifest.library_id} version {manifest.version}",
            f"Content: {manifest.content_sha256}",
            f"Parent: {manifest.parent_content_sha256 or 'none'}",
            "",
        ]
        for row, product in enumerate(manifest.products):
            values = (
                product.product.value, product.tier.value, product.source_count,
                product.seed_count, product.sample_count, product.prototype_count,
                f"{product.effective_weight:.2f}", "; ".join(product.warnings),
            )
            for column, value in enumerate(values):
                self.coverage_table.setItem(row, column, QTableWidgetItem(str(value)))
            lines.append(
                f"{product.product.value}: {product.tier.value}; "
                f"{product.source_count} source(s), {product.seed_count} seed(s), "
                f"{product.prototype_count} prototype(s)"
            )
            lines.extend(
                f"  {name.replace('_', ' ')}: {value:.4g}"
                for name, value in product.metrics.items()
            )
            lines.extend(f"  Warning: {warning}" for warning in product.warnings)
        self.coverage_table.resizeColumnsToContents()
        self.validation_label.setText("\n".join(lines))
        for source in manifest.sources:
            item = QListWidgetItem(
                f"{source.display_label} — {source.review_status}",
                self.retained_source_list,
            )
            item.setData(Qt.ItemDataRole.UserRole, source.source_sha256)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            context = source.biological_context
            context_text = "unspecified" if context is None else " / ".join(
                value or "—" for value in (
                    context.lineage_group_id, context.accession_id, context.seed_lot_id
                )
            )
            item.setToolTip(
                f"SHA-256: {source.source_sha256}\n"
                f"Capture group: {source.capture_group_id or 'unspecified'}\n"
                f"Lineage / accession / lot: {context_text}"
            )
        self._refresh_version_actions()
