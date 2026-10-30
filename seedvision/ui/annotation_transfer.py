"""Export annotations from one project and import them into another."""

from __future__ import annotations

import tempfile
from functools import partial
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from seedvision.persistence import (
    ANNOTATION_BUNDLE_EXTENSION,
    MANUAL_SEED_CENTRES_SIDECAR,
    REFERENCE_REGIONS_SIDECAR,
    AnnotationExportItem,
    AnnotationTransferError,
    InstanceMaskImportError,
    ManualSeedCentreStoreError,
    ProjectAnalysisError,
    ReferenceRegionError,
    file_sha256,
    read_annotation_bundle,
    read_source_raster_shape,
    write_annotation_bundle,
)
from seedvision.persistence.annotation_transfer import (
    ANNOTATION_KIND_LABELS,
    extract_annotation,
)
from seedvision.ui.persistence_jobs import run_persistence


_KIND_ORDER = (MANUAL_SEED_CENTRES_SIDECAR, REFERENCE_REGIONS_SIDECAR)
_TRANSFER_ERRORS = (
    AnnotationTransferError,
    InstanceMaskImportError,
    ManualSeedCentreStoreError,
    ReferenceRegionError,
    OSError,
)
# Per-image reference state forgotten when an import replaces the layers.
_REFERENCE_STATE = (
    "_draft_background_reference_masks", "_draft_foreground_reference_masks",
    "_applied_background_reference_masks", "_applied_foreground_reference_masks",
    "_draft_background_exclusion_masks", "_draft_foreground_exclusion_masks",
    "_applied_background_exclusion_masks", "_applied_foreground_exclusion_masks",
    "_draft_physical_edge_reference_masks", "_draft_non_edge_reference_masks",
    "_applied_physical_edge_reference_masks", "_applied_non_edge_reference_masks",
    "_draft_instance_annotations", "_applied_instance_annotations",
    "_draft_instance_annotation_origins", "_applied_instance_annotation_origins",
    "_draft_seed_annotations", "_applied_seed_annotations",
    "_draft_seed_annotation_species", "_applied_seed_annotation_species",
    "_reference_dirty_classes", "_instance_continuity_cache",
    "_reference_undo_histories", "_instance_undo_histories",
    "_reference_layer_shapes", "_reference_transforms",
    "_reference_source_identities", "_draft_instance_generation_sources",
    "_pending_unbound_reference_bundles", "_reference_region_load_status",
)


def _kinds_text(kinds) -> str:
    return ", ".join(ANNOTATION_KIND_LABELS[kind] for kind in _KIND_ORDER if kind in kinds)


class AnnotationSelectionDialog(QDialog):
    """Choose images and annotation kinds for an export or import."""

    def __init__(self, parent, *, title, intro, rows, accept_text, note=""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(520, 420)
        layout = QVBoxLayout(self)
        label = QLabel(intro)
        label.setWordWrap(True)
        layout.addWidget(label)
        present = {kind for _label, kinds, _detail in rows for kind in kinds}
        kind_row = QHBoxLayout()
        self.kind_boxes = {}
        for kind in _KIND_ORDER:
            if kind in present:
                box = QCheckBox(ANNOTATION_KIND_LABELS[kind].capitalize())
                box.setChecked(True)
                kind_row.addWidget(box)
                self.kind_boxes[kind] = box
        kind_row.addStretch(1)
        layout.addLayout(kind_row)
        self.image_list = QListWidget()
        self._row_kinds = []
        for text, kinds, detail in rows:
            item = QListWidgetItem(f"{text} — {_kinds_text(kinds)}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setToolTip(detail)
            self.image_list.addItem(item)
            self._row_kinds.append(tuple(kinds))
        layout.addWidget(self.image_list, 1)
        selection_row = QHBoxLayout()
        for text, state in (
            ("Select all", Qt.CheckState.Checked),
            ("Select none", Qt.CheckState.Unchecked),
        ):
            button = QPushButton(text)
            button.clicked.connect(partial(self._check_all, state))
            selection_row.addWidget(button)
        selection_row.addStretch(1)
        layout.addLayout(selection_row)
        if note:
            note_label = QLabel(note)
            note_label.setWordWrap(True)
            layout.addWidget(note_label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(accept_text)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _check_all(self, state) -> None:
        for index in range(self.image_list.count()):
            self.image_list.item(index).setCheckState(state)

    def selection(self) -> list[tuple[int, tuple[str, ...]]]:
        """Return each checked row with the annotation kinds still chosen."""

        chosen = {kind for kind, box in self.kind_boxes.items() if box.isChecked()}
        selected = []
        for index, kinds in enumerate(self._row_kinds):
            if self.image_list.item(index).checkState() != Qt.CheckState.Checked:
                continue
            kept = tuple(kind for kind in kinds if kind in chosen)
            if kept:
                selected.append((index, kept))
        return selected


def _bundle_matches(source: Path, images: tuple[Path, ...]):
    """Read a bundle and fingerprint the project's images (runs off the GUI thread)."""

    entries = read_annotation_bundle(source)
    digests = tuple((image, file_sha256(image)) for image in images)
    return entries, digests


class AnnotationTransferController:
    """File ▸ Import / export ▸ Export and Import annotations."""

    def _init_annotation_transfer(self) -> None:
        self.export_annotations_action = QAction("Export annotations…", self)
        self.export_annotations_action.setToolTip(
            "Write manual seed centres and reference layers for chosen images to "
            "one file that another project can import."
        )
        self.export_annotations_action.triggered.connect(self._export_annotations)
        self.import_annotations_action = QAction("Import annotations…", self)
        self.import_annotations_action.setToolTip(
            "Add annotations exported from another project to the matching images "
            "here. Images match by exact file contents; imports are unsaved edits."
        )
        self.import_annotations_action.triggered.connect(self._import_annotations)

    def _listed_images(self) -> tuple[Path, ...]:
        return tuple(
            Path(self.image_list.item(index).data(Qt.ItemDataRole.UserRole)).resolve()
            for index in range(self.image_list.count())
        )

    def _annotation_bundle_folder(self) -> Path:
        return (
            self._current_project_path.parent
            if self._current_project_path is not None
            else self._root
        )

    @Slot()
    def _export_annotations(self) -> None:
        from seedvision.ui.main_window import _path_identity

        if self._background_work_is_active():
            QMessageBox.information(
                self,
                "Analysis is busy",
                "Wait for the current work to finish before exporting annotations.",
            )
            return
        if not self._resolve_unapplied_project_drafts():
            return
        if not self._retry_unsaved_project_sidecars():
            return
        try:
            sources = self._project_annotation_sources()
        except ProjectAnalysisError as error:
            QMessageBox.critical(self, "Could not export annotations", str(error))
            return
        by_image: dict[str, tuple[Path, dict[str, Path]]] = {}
        for (key, kind), (image, archive) in sources.items():
            by_image.setdefault(key, (image, {}))[1][kind] = archive
        ordered = [
            by_image[key]
            for key in (_path_identity(image) for image in self._listed_images())
            if key in by_image
        ]
        if not ordered:
            QMessageBox.information(
                self,
                "Export annotations",
                "No image in this project has manual seed centres or applied "
                "reference layers to export.",
            )
            return
        dialog = AnnotationSelectionDialog(
            self,
            title="Export annotations",
            intro=(
                "Choose the images and annotations to export. Another project can "
                "import them for the same image files."
            ),
            rows=[(image.name, tuple(archives), str(image)) for image, archives in ordered],
            accept_text="Export…",
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        items = [
            AnnotationExportItem(ordered[index][0], kind, ordered[index][1][kind])
            for index, kinds in dialog.selection()
            for kind in kinds
        ]
        if not items:
            return
        name = (
            self._current_project_path.name.split(".")[0]
            if self._current_project_path is not None
            else "annotations"
        )
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export annotations",
            str(self._annotation_bundle_folder() / f"{name}{ANNOTATION_BUNDLE_EXTENSION}"),
            f"Seed Fiddle annotations (*{ANNOTATION_BUNDLE_EXTENSION});;All files (*)",
        )
        if not selected:
            return
        destination = self._path_with_required_suffix(
            Path(selected), ANNOTATION_BUNDLE_EXTENSION
        )
        try:
            entries = run_persistence(
                self,
                partial(
                    write_annotation_bundle,
                    destination,
                    items,
                    workspace_root=self._root,
                ),
                "Exporting annotations…",
            )
        except _TRANSFER_ERRORS as error:
            QMessageBox.critical(self, "Could not export annotations", str(error))
            return
        images = len({entry.image_sha256 for entry in entries})
        self.statusBar().showMessage(
            f"Exported {len(entries)} annotation file(s) for {images} image(s) to "
            f"{destination}."
        )

    @Slot()
    def _import_annotations(self) -> None:
        from seedvision.ui.main_window import _path_identity

        if self._background_work_is_active():
            QMessageBox.information(
                self,
                "Analysis is busy",
                "Wait for the current work to finish before importing annotations.",
            )
            return
        if not self._resolve_unapplied_project_drafts():
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Import annotations",
            str(self._annotation_bundle_folder()),
            f"Seed Fiddle annotations (*{ANNOTATION_BUNDLE_EXTENSION});;All files (*)",
        )
        if not selected:
            return
        source = Path(selected)
        try:
            entries, digests = run_persistence(
                self,
                partial(_bundle_matches, source, self._listed_images()),
                "Reading annotations…",
            )
        except _TRANSFER_ERRORS as error:
            QMessageBox.critical(self, "Could not import annotations", str(error))
            return
        images_by_digest: dict[str, list[Path]] = {}
        for image, digest in digests:
            images_by_digest.setdefault(digest, []).append(image)
        matched: dict[Path, dict[str, object]] = {}
        unmatched = 0
        for entry in entries:
            images = images_by_digest.get(entry.image_sha256)
            if not images:
                unmatched += 1
                continue
            for image in images:
                matched.setdefault(image, {})[entry.kind] = entry
        if not matched:
            QMessageBox.information(
                self,
                "Import annotations",
                f"None of the {len(entries)} annotation file(s) in {source.name} "
                "belong to an image in this project. Images are matched by their "
                "exact file contents, so add the same image files first.",
            )
            return
        ordered = [image for image in self._listed_images() if image in matched]
        rows = []
        for image in ordered:
            key = _path_identity(image)
            replaced = [
                kind
                for kind, recorded in (
                    (REFERENCE_REGIONS_SIDECAR, self._project_reference_sidecar_paths),
                    (MANUAL_SEED_CENTRES_SIDECAR, self._project_manual_centre_sidecar_paths),
                )
                if kind in matched[image] and key in recorded
            ]
            detail = str(image) + (
                f"\nReplaces existing {_kinds_text(replaced)}." if replaced else ""
            )
            rows.append((image.name, tuple(matched[image]), detail))
        dialog = AnnotationSelectionDialog(
            self,
            title="Import annotations",
            intro=(
                f"Choose which annotations from {source.name} to add. They replace "
                "the same kind of annotation on that image and stay unsaved until "
                "you save the project."
            ),
            rows=rows,
            accept_text="Import",
            note=(
                f"{unmatched} file(s) in the bundle match no image in this project."
                if unmatched
                else ""
            ),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = [
            (ordered[index], kind, matched[ordered[index]][kind])
            for index, kinds in dialog.selection()
            for kind in kinds
        ]
        imported: set[str] = set()
        count = 0
        with tempfile.TemporaryDirectory() as temporary:
            for image, kind, entry in chosen:
                try:
                    archive = extract_annotation(source, entry, temporary)
                    self._install_imported_annotation(image, kind, archive)
                except _TRANSFER_ERRORS as error:
                    QMessageBox.critical(
                        self,
                        "Could not import annotations",
                        f"{image.name}: {error}\n\n{count} earlier annotation "
                        "file(s) were imported.",
                    )
                    break
                imported.add(_path_identity(image))
                count += 1
        if not imported:
            return
        self._set_project_dirty()
        current = self.image_view.image_path
        current_key = None if current is None else _path_identity(current)
        if current_key in imported:
            self._open_path(current, mark_project_dirty=False)
            if current_key in self._analysis_caches:
                self._analyze_current_image(
                    dirty_nodes=set(self._cache_dirty_nodes.get(current_key, ()))
                )
        self.statusBar().showMessage(
            f"Imported {count} annotation file(s) for {len(imported)} image(s) "
            f"from {source.name}."
            + (
                " Save the project to keep them."
                if self._project_annotation_directory is not None
                else ""
            )
        )

    def _install_imported_annotation(self, image: Path, kind: str, archive: Path) -> Path:
        """Validate one extracted archive and make it this image's annotation."""

        from seedvision.ui.main_window import _path_identity

        key = _path_identity(image)
        if kind == MANUAL_SEED_CENTRES_SIDECAR:
            source_shape = read_source_raster_shape(image)
            stored = self._manual_seed_centre_store.load_project_archive(
                image, archive, source_shape
            )
            if stored is None:
                raise AnnotationTransferError(f"{archive.name} could not be read.")
            destination = self._manual_seed_centre_store.save(
                image,
                stored.centres_xy,
                mode=stored.mode,
                source_shape=source_shape,
                suppressed_xy=stored.suppressed_xy,
            )
            self._manual_seed_centre_states.pop(key, None)
            self._manual_seed_centre_histories.pop(key, None)
            self._manual_seed_centre_autoload_attempted.discard(key)
            self._unsaved_manual_centre_sidecars.discard(key)
            self._withheld_manual_centre_sidecars.discard(key)
            self._project_unresolved_manual_centre_sidecars.discard(key)
            self._project_manual_centre_sidecar_paths[key] = destination
            self._project_manual_centre_loads.add(key)
            affected = {
                "procedural_instances",
                *self.pipeline.downstream("procedural_instances", recursive=True),
            }
        else:
            bundle = self._reference_region_store.load_project_archive(
                image, archive, None
            )
            if bundle is None:
                raise AnnotationTransferError(f"{archive.name} could not be read.")
            destination = self._reference_region_store.save(image, bundle)
            for name in _REFERENCE_STATE:
                getattr(self, name).pop(key, None)
            for values in (
                self._reference_masks_dirty,
                self._instance_annotations_dirty,
                self._reference_region_autoload_attempted,
                self._unsaved_reference_sidecars,
                self._withheld_reference_sidecars,
                self._project_unresolved_reference_sidecars,
            ):
                values.discard(key)
            self._project_reference_sidecar_paths[key] = destination
            self._project_unbound_reference_loads.add(key)
            affected = {
                "project",
                *self.pipeline.downstream_from_port(
                    "project", "annotations", recursive=True
                ),
            }
        self.pipeline.invalidate(affected)
        self._cache_dirty_nodes.setdefault(key, set()).update(affected - {"project"})
        self._analyses.pop(key, None)
        return destination
