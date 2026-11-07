"""Project-facing manager for immutable species-reference library versions."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from seedvision.reference_library import (
    SPECIES_LIBRARY_EXTENSION,
    BiologicalContext,
    SpeciesLibraryPin,
    SpeciesLibraryService,
)

_UNKNOWN_SPECIES_ID = "unknown"


def _coverage_label(value):
    return "legacy coverage tier" if value == "validated" else value


def _next_version_name(manifests) -> str:
    """One more than the largest whole-number version, or the count plus one."""

    numbers = [int(item.version) for item in manifests if str(item.version).isdigit()]
    return str(max(numbers, default=len(manifests)) + 1)


class SpeciesLibraryManagerDialog(QDialog):
    """Create, use, inspect, import and export immutable species libraries.

    A library describes one species. The species chosen at the top decides
    which versions are listed, which project images can contribute to a new
    version, and which species that version is built for. Only a library of
    the open image's species can be used by the project.
    """

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
        species: tuple[tuple[str, str], ...] = (),
        image_species: dict[str, str] | None = None,
        annotated_paths: frozenset[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.service = service
        self.context = context
        self.current_pin = current_pin
        # The species the open image uses: only its libraries can be pinned.
        self.project_species_id = context.species_id
        self._image_paths = tuple(image_paths)
        self._image_species = None if image_species is None else dict(image_species)
        self._annotated_paths = annotated_paths
        self._manifests = ()
        self._version_edited = False
        self._species_names = {species_id: name for species_id, name in species}
        self.setWindowTitle("Species reference libraries")
        self.resize(980, 720)
        root = QVBoxLayout(self)

        # Header: the species every tab works on, and what the project uses.
        header = QHBoxLayout()
        species_label = QLabel("Species", self)
        species_font = QFont(species_label.font())
        species_font.setBold(True)
        species_label.setFont(species_font)
        header.addWidget(species_label)
        self.species_combo = QComboBox(self)
        self.species_combo.setToolTip(
            "A reference library describes one species. Choose the species to "
            "list its libraries or to create one from your images of it."
        )
        known = dict(species)
        if context.species_id not in known:
            known = {context.species_id: self._display_name(context.species_id), **known}
        for species_id, name in known.items():
            if species_id in {"", _UNKNOWN_SPECIES_ID}:
                self.species_combo.addItem("Choose a species…", _UNKNOWN_SPECIES_ID)
            else:
                self.species_combo.addItem(name, species_id)
        self.species_combo.setCurrentIndex(
            max(0, self.species_combo.findData(context.species_id))
        )
        self.species_combo.setMinimumContentsLength(18)
        header.addWidget(self.species_combo)
        header.addSpacing(16)
        self.summary_label = QLabel(self)
        self.summary_label.setWordWrap(True)
        header.addWidget(self.summary_label, 1)
        root.addLayout(header)

        self.tabs = QTabWidget(self)
        root.addWidget(self.tabs, 1)

        # Libraries: the versions of this species and what to do with them.
        libraries = QWidget(self.tabs)
        libraries_layout = QVBoxLayout(libraries)
        self.empty_state = QFrame(libraries)
        self.empty_state.setFrameShape(QFrame.Shape.StyledPanel)
        empty_layout = QVBoxLayout(self.empty_state)
        self.empty_label = QLabel(self.empty_state)
        self.empty_label.setWordWrap(True)
        empty_layout.addWidget(self.empty_label)
        empty_buttons = QHBoxLayout()
        self.create_first_button = QPushButton("Create a library…", self.empty_state)
        self.create_first_button.setToolTip("Build one from your annotated images.")
        empty_buttons.addWidget(self.create_first_button)
        empty_buttons.addStretch(1)
        empty_layout.addLayout(empty_buttons)
        self.empty_state.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        libraries_layout.addWidget(self.empty_state)
        self.version_table = QTableWidget(0, 6, libraries)
        self.version_table.setHorizontalHeaderLabels(
            ("Used by project", "Version", "Status", "Images", "Evidence", "Published")
        )
        self.version_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.version_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.version_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.version_table.verticalHeader().setVisible(False)
        self.version_table.horizontalHeader().setStretchLastSection(True)
        libraries_layout.addWidget(self.version_table, 1)
        # Takes the room the table leaves when there is no version yet.
        self.empty_spacer = QWidget(libraries)
        libraries_layout.addWidget(self.empty_spacer, 1)
        self.selected_library_label = QLabel(libraries)
        self.selected_library_label.setWordWrap(True)
        libraries_layout.addWidget(self.selected_library_label)
        buttons = QHBoxLayout()
        self.pin_button = QPushButton("Use for this project", libraries)
        self.pin_button.setToolTip(
            "Analyses of this species then use this exact version. The project keeps "
            "this version until you choose another; it never follows newer ones."
        )
        self.export_button = QPushButton("Export…", libraries)
        self.export_button.setToolTip("Save the selected version as a portable file.")
        self.retire_button = QPushButton("Retire", libraries)
        self.retire_button.setToolTip(
            "Hide the selected version from normal use. Nothing is deleted."
        )
        self.import_button = QPushButton("Import library file…", libraries)
        self.import_button.setToolTip("Install a library file exported from Seed Fiddle.")
        for button in (self.pin_button, self.export_button, self.retire_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(self.import_button)
        libraries_layout.addLayout(buttons)
        self.tabs.addTab(libraries, "Libraries")

        # Create: images of this species, then a version, then build.
        create = QWidget(self.tabs)
        create_layout = QVBoxLayout(create)
        self.create_heading = QLabel(create)
        heading_font = QFont(self.create_heading.font())
        heading_font.setBold(True)
        heading_font.setPointSizeF(heading_font.pointSizeF() * 1.15)
        self.create_heading.setFont(heading_font)
        self.create_heading.setWordWrap(True)
        create_layout.addWidget(self.create_heading)
        self.create_help = QLabel(create)
        self.create_help.setWordWrap(True)
        create_layout.addWidget(self.create_help)

        source_group = QGroupBox("1. Images", create)
        source_group_layout = QVBoxLayout(source_group)
        source_controls = QHBoxLayout()
        self.source_selection_label = QLabel(source_group)
        source_controls.addWidget(self.source_selection_label, 1)
        self.select_all_sources_button = QPushButton("Select all", source_group)
        self.clear_sources_button = QPushButton("Clear", source_group)
        source_controls.addWidget(self.select_all_sources_button)
        source_controls.addWidget(self.clear_sources_button)
        source_group_layout.addLayout(source_controls)
        self.source_list = QListWidget(source_group)
        source_group_layout.addWidget(self.source_list, 1)
        self.unset_species_label = QLabel(source_group)
        self.unset_species_label.setWordWrap(True)
        source_group_layout.addWidget(self.unset_species_label)
        self.other_species_label = QLabel(source_group)
        self.other_species_label.setWordWrap(True)
        source_group_layout.addWidget(self.other_species_label)
        create_layout.addWidget(source_group, 1)

        version_group = QGroupBox("2. Version", create)
        version_layout = QVBoxLayout(version_group)
        form = QFormLayout()
        self.version_edit = QLineEdit("1", version_group)
        self.version_edit.setMaxLength(128)
        self.version_edit.setToolTip("A name or number for the new version.")
        form.addRow("Name", self.version_edit)
        self.base_combo = QComboBox(version_group)
        self.base_combo.setToolTip(
            "Start empty, or keep the images of an existing version and add the "
            "ones checked above."
        )
        form.addRow("Start from", self.base_combo)
        version_layout.addLayout(form)
        self.retained_sources_label = QLabel(
            "Images kept from that version (uncheck any to leave it out):",
            version_group,
        )
        self.retained_sources_label.setWordWrap(True)
        version_layout.addWidget(self.retained_sources_label)
        self.retained_source_list = QListWidget(version_group)
        self.retained_source_list.setMaximumHeight(110)
        version_layout.addWidget(self.retained_source_list)
        create_layout.addWidget(version_group)

        # Provenance identifiers are optional and rarely changed: keep them
        # behind a disclosure button.
        self.details_toggle = QToolButton(create)
        self.details_toggle.setText("Source details (optional)")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.details_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.details_toggle.setAutoRaise(True)
        create_layout.addWidget(self.details_toggle)
        self.details_panel = QWidget(create)
        details_layout = QVBoxLayout(self.details_panel)
        details_layout.setContentsMargins(18, 0, 0, 0)
        details_help = QLabel(
            "These identifiers let models pool related accessions, lots and repeated "
            "photographs without counting them as independent seeds.",
            self.details_panel,
        )
        details_help.setWordWrap(True)
        details_layout.addWidget(details_help)
        context_form = QFormLayout()
        self.lineage_edit = QLineEdit(context.lineage_group_id or "", self.details_panel)
        self.accession_edit = QLineEdit(context.accession_id or "", self.details_panel)
        self.seed_lot_edit = QLineEdit(context.seed_lot_id or "", self.details_panel)
        self.capture_group_edit = QLineEdit(capture_group_id or "", self.details_panel)
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
        self.accession_edit.setToolTip("Accession or line represented by the images.")
        self.seed_lot_edit.setToolTip(
            "Specific harvested or supplied seed lot within an accession."
        )
        self.capture_group_edit.setToolTip(
            "Use the same value for repeated photographs from one capture session, so "
            "near-duplicate views are not counted as independent evidence."
        )
        context_form.addRow("Lineage / group", self.lineage_edit)
        context_form.addRow("Accession", self.accession_edit)
        context_form.addRow("Seed lot", self.seed_lot_edit)
        context_form.addRow("Repeated-photo group", self.capture_group_edit)
        details_layout.addLayout(context_form)
        self.details_panel.setVisible(False)
        create_layout.addWidget(self.details_panel)

        build_row = QHBoxLayout()
        self.build_status_label = QLabel(create)
        self.build_status_label.setWordWrap(True)
        build_row.addWidget(self.build_status_label, 1)
        self.build_button = QPushButton(create)
        self.build_button.setDefault(True)
        build_row.addWidget(self.build_button)
        create_layout.addLayout(build_row)
        self.tabs.addTab(create, "Create a library")

        # Details: product coverage and validation of the selected version.
        inspect = QWidget(self.tabs)
        inspect_layout = QVBoxLayout(inspect)
        inspect_help = QLabel(
            "The evidence the version selected on Libraries contains, its support "
            "and any validation warnings. Coverage counts evidence; it is not accuracy.",
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
        self.coverage_table.verticalHeader().setVisible(False)
        inspect_layout.addWidget(self.coverage_table, 2)
        self.validation_label = QLabel(inspect)
        self.validation_label.setWordWrap(True)
        self.validation_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        inspect_layout.addWidget(self.validation_label, 1)
        self.tabs.addTab(inspect, "Details")

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.accept)
        close_row.addWidget(close_button)
        root.addLayout(close_row)

        self.species_combo.currentIndexChanged.connect(self._species_changed)
        self.create_first_button.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        self.pin_button.clicked.connect(self._pin_selected)
        self.export_button.clicked.connect(self._export_selected)
        self.import_button.clicked.connect(self._import_bundle)
        self.retire_button.clicked.connect(self._toggle_retired)
        self.build_button.clicked.connect(self._request_build)
        self.version_table.itemSelectionChanged.connect(self._show_selected_validation)
        self.base_combo.currentIndexChanged.connect(self._base_changed)
        self.source_list.itemChanged.connect(self._source_selection_changed)
        self.select_all_sources_button.clicked.connect(self._select_all_sources)
        self.clear_sources_button.clicked.connect(self._clear_source_selection)
        self.version_edit.textEdited.connect(self._version_name_edited)
        self.details_toggle.toggled.connect(self._details_toggled)
        self._species_changed()

    # Species -----------------------------------------------------------------

    @property
    def species_id(self) -> str:
        """The species the dialog lists libraries for and builds them for."""

        return str(self.species_combo.currentData() or _UNKNOWN_SPECIES_ID)

    def _display_name(self, species_id: str) -> str:
        if species_id in self._species_names:
            return self._species_names[species_id]
        for manifest in self._manifests:
            if manifest.species_id == species_id:
                return manifest.species_display_name
        return species_id.replace("_", " ").capitalize()

    def _species_chosen(self) -> bool:
        return self.species_id not in {"", _UNKNOWN_SPECIES_ID}

    def _species_changed(self, *_unused) -> None:
        self._populate_sources()
        self._version_edited = False
        self.refresh()

    def _image_species_id(self, path: Path) -> str:
        if self._image_species is None:
            return self.species_id
        return self._image_species.get(str(path), _UNKNOWN_SPECIES_ID) or _UNKNOWN_SPECIES_ID

    def _populate_sources(self) -> None:
        """List the chosen species' images, then those with no species set.

        Annotated images are checked. An image with no species set takes the
        library's species when it is included, so a new project does not have
        to set Species image by image before creating its first library.
        """

        species_id = self.species_id
        name = self._display_name(species_id)
        with_species = [
            path for path in self._image_paths if self._image_species_id(path) == species_id
        ]
        unset = [
            path
            for path in self._image_paths
            if self._image_species_id(path) == _UNKNOWN_SPECIES_ID
        ]
        other = len(self._image_paths) - len(with_species) - len(unset)
        self.source_list.blockSignals(True)
        self.source_list.clear()
        if self._species_chosen():
            for path in (*with_species, *unset):
                annotated = (
                    self._annotated_paths is None or str(path) in self._annotated_paths
                )
                notes = (
                    ("species not set",) if path in unset else ()
                ) + (() if annotated else ("no saved annotations",))
                item = QListWidgetItem(
                    " — ".join((path.name, ", ".join(notes))) if notes else path.name,
                    self.source_list,
                )
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                item.setToolTip(str(path))
                if annotated:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked)
                else:
                    item.setFlags(
                        (item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        & ~Qt.ItemFlag.ItemIsEnabled
                    )
                    item.setCheckState(Qt.CheckState.Unchecked)
        self.source_list.blockSignals(False)
        self.unset_species_label.setVisible(bool(unset) and self._species_chosen())
        self.unset_species_label.setText(
            f"{len(unset)} image(s) have no species set and are listed too. Creating "
            f"the library records the checked ones as {name}."
        )
        self.other_species_label.setVisible(other > 0 and self._species_chosen())
        self.other_species_label.setText(
            f"{other} project image(s) set to another species are not listed. To "
            "include one, open it and change Species on the Image tab of the right "
            "panel."
        )
        if self._species_chosen():
            self.create_heading.setText(f"New {name} library version")
            self.create_help.setText(
                f"A library collects the saved annotations of your {name} images: seed "
                "outlines and seed traits, and painted Foreground, Background and Other "
                "areas. Analyses of other images of this species can then use them. "
                "Images are checked before anything is published; publishing creates a "
                "read-only version, and later annotation edits never change it."
            )
            self.build_button.setText(f"Create {name} library")
        else:
            self.create_heading.setText("Choose a species first")
            self.create_help.setText(
                "A library describes one species. Choose it in Species at the top; its "
                "images in this project are then listed here."
            )
            self.build_button.setText("Create library")
        self._source_selection_changed()

    # Libraries ---------------------------------------------------------------

    def refresh(self) -> None:
        species_id = self.species_id
        manifests = tuple(
            value for value in self.service.store.list_manifests()
            if value.species_id == species_id
        )
        self._manifests = manifests
        name = self._display_name(species_id)
        self.version_table.setRowCount(len(manifests))
        for row, manifest in enumerate(manifests):
            pin = SpeciesLibraryPin(
                manifest.library_id, manifest.version, manifest.species_id,
                manifest.content_sha256,
            )
            values = (
                "●" if self.current_pin == pin else "",
                manifest.version,
                manifest.status.value.capitalize(),
                len(manifest.sources),
                ", ".join(
                    f"{item.product.value}: {_coverage_label(item.tier.value)}"
                    for item in manifest.products
                ),
                manifest.created_utc,
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                cell.setToolTip(
                    f"{manifest.library_id} version {manifest.version}\n"
                    f"Content ID (SHA-256): {manifest.content_sha256}"
                )
                self.version_table.setItem(row, column, cell)
        self.version_table.resizeColumnsToContents()
        has_versions = bool(manifests)
        self.version_table.setVisible(has_versions)
        self.empty_state.setVisible(not has_versions)
        self.empty_spacer.setVisible(not has_versions)
        self.empty_label.setText(
            f"There is no {name} reference library yet. A library is built from your "
            f"annotated {name} images, so their reviewed seeds and painted areas can "
            "guide the analysis of other images of this species."
            if self._species_chosen()
            else "Choose a species above to see its libraries or to create one."
        )
        self.create_first_button.setVisible(self._species_chosen())
        self.import_button.setEnabled(True)
        used = self.current_pin
        self.summary_label.setText(
            "This project does not use a species library."
            if used is None
            else f"This project uses the {self._display_name(used.species_id)} library, "
            f"version {used.version}."
        )
        self._populate_base_versions()
        if not self._version_edited:
            self.version_edit.setText(_next_version_name(manifests))
        if manifests and self.version_table.currentRow() < 0:
            used_row = next(
                (
                    row for row, manifest in enumerate(manifests)
                    if used is not None and manifest.content_sha256 == used.sha256
                ),
                0,
            )
            self.version_table.selectRow(used_row)
        self._show_selected_validation()

    def _populate_base_versions(self) -> None:
        previous = self.base_combo.currentData()
        self.base_combo.blockSignals(True)
        self.base_combo.clear()
        self.base_combo.addItem("Nothing: a new library", None)
        for index, manifest in enumerate(self._manifests):
            if manifest.status.value == "retired":
                continue
            self.base_combo.addItem(
                f"Version {manifest.version} ({len(manifest.sources)} image(s)), "
                "adding the images checked above",
                index,
            )
        restored = self.base_combo.findData(previous)
        self.base_combo.setCurrentIndex(max(0, restored))
        self.base_combo.setEnabled(self.base_combo.count() > 1)
        self.base_combo.blockSignals(False)
        self._base_changed()

    def _base_manifest(self):
        index = self.base_combo.currentData()
        return None if index is None or index >= len(self._manifests) else self._manifests[index]

    def _base_changed(self, *_unused) -> None:
        manifest = self._base_manifest()
        self.retained_sources_label.setVisible(manifest is not None)
        self.retained_source_list.setVisible(manifest is not None)
        self.retained_source_list.clear()
        if manifest is None:
            self._source_selection_changed()
            return
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
        self._source_selection_changed()

    def _checked_sources(self) -> tuple[Path, ...]:
        return tuple(
            Path(item.data(Qt.ItemDataRole.UserRole))
            for index in range(self.source_list.count())
            if (item := self.source_list.item(index)).checkState() == Qt.CheckState.Checked
        )

    def _source_selection_changed(self, *_unused) -> None:
        listed = self.source_list.count()
        usable = sum(
            bool(self.source_list.item(index).flags() & Qt.ItemFlag.ItemIsEnabled)
            for index in range(listed)
        )
        selected = len(self._checked_sources())
        name = self._display_name(self.species_id)
        if not self._species_chosen():
            text = ""
        elif not listed:
            text = f"This project has no {name} images."
        else:
            text = f"{selected} of {listed} image(s) selected"
            if usable < listed:
                text += f"; {listed - usable} have no saved annotations yet"
            text += "."
        self.source_selection_label.setText(text)
        ready = self._species_chosen() and (selected > 0 or self._base_manifest() is not None)
        self.build_button.setEnabled(ready and self.source_list.isEnabled())
        if not self._species_chosen():
            status = ""
        elif ready:
            status = "Ready. Problems found while checking the images are shown here."
        elif listed and not usable:
            status = (
                f"Annotate and save at least one {name} image (Seeds or Materials on "
                "the toolbar, then Apply + save) to create a library."
            )
        else:
            status = "Check at least one image."
        self.build_status_label.setText(status)

    def _select_all_sources(self) -> None:
        for index in range(self.source_list.count()):
            item = self.source_list.item(index)
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(Qt.CheckState.Checked)

    def _clear_source_selection(self) -> None:
        for index in range(self.source_list.count()):
            self.source_list.item(index).setCheckState(Qt.CheckState.Unchecked)

    def _version_name_edited(self, _text: str) -> None:
        self._version_edited = True

    def _details_toggled(self, shown: bool) -> None:
        self.details_panel.setVisible(shown)
        self.details_toggle.setArrowType(
            Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow
        )

    def _refresh_version_actions(self) -> None:
        manifest = self._selected_manifest()
        selected = manifest is not None
        usable_here = selected and manifest.species_id == self.project_species_id
        self.pin_button.setEnabled(usable_here and manifest.status.value != "retired")
        self.export_button.setEnabled(selected)
        self.retire_button.setEnabled(selected)
        self.retire_button.setText(
            "Restore" if selected and manifest.status.value == "retired" else "Retire"
        )

    def set_build_busy(self, busy: bool, message: str = "") -> None:
        for widget in (
            self.species_combo, self.source_list, self.version_edit,
            self.base_combo, self.retained_source_list,
            self.lineage_edit, self.accession_edit, self.seed_lot_edit,
            self.capture_group_edit, self.select_all_sources_button,
            self.clear_sources_button,
        ):
            widget.setEnabled(not busy)
        self.build_button.setEnabled(not busy)
        if not busy:
            self.base_combo.setEnabled(self.base_combo.count() > 1)
            self._source_selection_changed()
            self._refresh_version_actions()
        if message:
            self.validation_label.setText(message)
            self.build_status_label.setText(message)

    def update_image_species(
        self, image_species: dict[str, str], open_image_species_id: str
    ) -> None:
        """Take the species the project recorded for its images during a build.

        The source list keeps its checks until the build ends, so a failed
        build can be retried as it was.
        """

        self._image_species = dict(image_species)
        self.project_species_id = open_image_species_id
        self._refresh_version_actions()

    def published(self, pin: SpeciesLibraryPin) -> None:
        name = self._display_name(pin.species_id)
        self.set_build_busy(False)
        self._populate_sources()
        self._version_edited = False
        usable_here = pin.species_id == self.project_species_id
        if usable_here:
            self.current_pin = pin
        self.refresh()
        message = (
            f"Published {name} version {pin.version}; this project now uses it."
            if usable_here
            else f"Published {name} version {pin.version}. Open a {name} image and "
            "choose Use for this project on Libraries to use it."
        )
        self.build_status_label.setText(message)
        self.validation_label.setText(message)
        self.tabs.setCurrentIndex(0)
        if usable_here:
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
        species_id = artifact.manifest.species_id
        if self.species_combo.findData(species_id) < 0:
            self._species_names[species_id] = artifact.manifest.species_display_name
            self.species_combo.addItem(artifact.manifest.species_display_name, species_id)
        # Show the library just installed, whatever its species.
        if species_id != self.species_id:
            self.species_combo.setCurrentIndex(self.species_combo.findData(species_id))
        else:
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
        paths = self._checked_sources()
        parent_pin = None
        base = self._base_manifest()
        if base is not None:
            parent_pin = SpeciesLibraryPin(
                base.library_id, base.version, base.species_id, base.content_sha256
            )
        if not self._species_chosen():
            QMessageBox.warning(self, "No species", "Choose the library's species first.")
            return
        if not version or (not paths and parent_pin is None):
            QMessageBox.warning(
                self, "Incomplete version",
                "Enter a version name and check at least one image.",
            )
            return
        if any(manifest.version == version for manifest in self._manifests):
            QMessageBox.warning(
                self, "Version exists",
                f"Version {version} already exists for this species. Enter another name.",
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
                self.species_id,
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
            QMessageBox.warning(self, "Invalid source details", str(error))
            return
        self.set_build_busy(True, "Checking the images and building the library…")
        self.build_requested.emit(
            paths, version, parent_pin, removed, context, capture_group
        )

    def _show_selected_validation(self) -> None:
        manifest = self._selected_manifest()
        self.coverage_table.setRowCount(0 if manifest is None else len(manifest.products))
        name = self._display_name(self.species_id)
        if manifest is None:
            self.validation_label.setText("No version is selected on Libraries.")
            self.selected_library_label.setText("")
            self._refresh_version_actions()
            return
        pin = self._selected_pin()
        if pin == self.current_pin:
            usage = "This project uses this version."
        elif manifest.species_id != self.project_species_id:
            usage = (
                f"The open image is not {name}: open a {name} image to use this "
                "library for the project."
            )
        elif manifest.status.value == "retired":
            usage = "Retired versions cannot be chosen; restore it first."
        else:
            usage = "Choose Use for this project to analyse with this version."
        self.selected_library_label.setText(
            f"Version {manifest.version}: {len(manifest.sources)} image(s), "
            f"published {manifest.created_utc}. {usage}"
        )
        lines = [
            f"Library {manifest.library_id} version {manifest.version}",
            f"Content: {manifest.content_sha256}",
            f"Parent: {manifest.parent_content_sha256 or 'none'}",
            "",
        ]
        for row, product in enumerate(manifest.products):
            values = (
                product.product.value, _coverage_label(product.tier.value), product.source_count,
                product.seed_count, product.sample_count, product.prototype_count,
                f"{product.effective_weight:.2f}", "; ".join(product.warnings),
            )
            for column, value in enumerate(values):
                self.coverage_table.setItem(row, column, QTableWidgetItem(str(value)))
            lines.append(
                f"{product.product.value}: {_coverage_label(product.tier.value)}; "
                f"{product.source_count} source(s), {product.seed_count} seed(s), "
                f"{product.prototype_count} prototype(s)"
            )
            lines.extend(
                f"  {metric.replace('_', ' ')}: " + ('unavailable' if value is None else f'{value:.4g}')
                for metric, value in product.metrics.items()
            )
            lines.extend(f"  Warning: {warning}" for warning in product.warnings)
        self.coverage_table.resizeColumnsToContents()
        self.validation_label.setText("\n".join(lines))
        self._refresh_version_actions()
