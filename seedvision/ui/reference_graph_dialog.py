"""Read-only Qt view of the frozen PlantCV and SMPTS reference graph slice.

This dialog projects existing recipe contracts into the existing pipeline canvas.
Historical graphs remain read-only and do not infer arrays from project metadata.
The explicit native calculation action opens a separate working session, without
rewriting original run history or presenting native pixels as original checkpoints.
"""

from __future__ import annotations

from dataclasses import replace
import html
import json
import math
from pathlib import Path
import threading
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Qt
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QSizePolicy,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
import numpy as np

from seedvision.benchmarking.recipe_catalogue import (
    RecipeCatalogue,
    build_recipe_catalogue,
    validate_recipe_catalogue,
)
from seedvision.benchmarking.native_extended_threshold_recipes import build_native_extended_threshold_recipes
from seedvision.benchmarking.native_range_recipe import build_native_range_recipe
from seedvision.benchmarking.checkpoint_inspection import (
    CheckpointInspection,
    InspectionStatus,
    MAX_PREVIEW_PIXELS,
    inspect_reference_checkpoint,
)
from seedvision.benchmarking.native_checkpoint_inspection import inspect_native_checkpoint
from seedvision.benchmarking.recipes import (
    LayoutPosition,
    RecipeContractError,
    RecipeProvenanceMode,
    RecipeSpec,
    SourceResolution,
    StageRunRecord,
    StageRunState,
    initial_stage_run,
    to_pipeline_graph,
)
from seedvision.persistence.reference_graphs import (
    MAX_REFERENCE_GRAPHS,
    ReferenceGraphAttachment,
    ReferenceGraphPersistenceError,
    validate_reference_graphs,
)
from seedvision.persistence.native_recipes import validate_native_recipes, select_native_recipe, recipe_key
from seedvision.pipeline import NodeStatus, PipelineNode
from seedvision.ui.pipeline_canvas import PipelineCanvas
from seedvision.ui.pipeline_inspector import PipelineInspector


def _matches_catalogue_with_selected_values(recipe: RecipeSpec, catalogue: RecipeSpec | None) -> bool:
    """Permit explicit parameter choices without relaxing source or topology identity."""
    if catalogue is None:
        return False
    try:
        stages = tuple(
            replace(actual, parameters=tuple(
                replace(parameter, value=pinned.value)
                for parameter, pinned in zip(actual.parameters, canonical.parameters, strict=True)
            ))
            for actual, canonical in zip(recipe.stages, catalogue.stages, strict=True)
        )
        return replace(recipe, stages=stages, layout=catalogue.layout) == catalogue
    except (TypeError, ValueError):
        return False


def _matches_native_catalogue(recipe: RecipeSpec, catalogue: RecipeSpec | None) -> bool:
    """Check topology/source/controls while the native verifier checks runtime pins."""
    if catalogue is None or recipe.provenance.mode is not RecipeProvenanceMode.NATIVE_COMPATIBLE:
        return False
    if recipe.provenance.reference_id != "plantcv":
        return False
    return _matches_catalogue_with_selected_values(
        replace(recipe, provenance=catalogue.provenance), catalogue,
    )


def _inspect_graph_checkpoint(store, recipe, record, *, cancel=None):
    if recipe.provenance.mode is RecipeProvenanceMode.NATIVE_COMPATIBLE:
        return inspect_native_checkpoint(store, recipe, record, cancel=cancel)
    return inspect_reference_checkpoint(store, recipe, record, cancel=cancel)


class _CheckpointPreviewLabel(QLabel):
    """Fit a bounded display image after layout assigns the actual label size."""
    def __init__(self, text, parent=None):
        super().__init__(text,parent)
        self._display_image = None
        self.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)

    def clear(self):
        self._display_image = None
        super().clear()

    def set_display_image(self, image):
        self._display_image = image
        self._fit_image()

    def _fit_image(self):
        if self._display_image is None:
            return
        target=self.contentsRect().size()
        super().setPixmap(QPixmap.fromImage(self._display_image).scaled(
            max(1,target.width()-8),max(1,target.height()-8),
            Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.FastTransformation,
        ))

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self._fit_image()


class _InspectionSignals(QObject):
    completed = Signal(int, object)


class _InspectionWorker(QRunnable):
    """One bounded checkpoint inspection on Qt's shared worker pool."""

    def __init__(self, token, store, recipe, record, cancel):
        super().__init__()
        self.token = token
        self.store = store
        self.recipe = recipe
        self.record = record
        self.cancel = cancel
        self.signals = _InspectionSignals()
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            result = _inspect_graph_checkpoint(
                self.store, self.recipe, self.record, cancel=self.cancel
            )
        except Exception as error:  # keep unexpected inspection errors visible in the UI
            result = CheckpointInspection(
                InspectionStatus.MISMATCH,
                f"Checkpoint inspection stopped safely: {type(error).__name__}: {error}",
            )
        self.signals.completed.emit(self.token, result)


def _escaped(value: object) -> str:
    return html.escape(str(value), quote=True)


def _format_value(value: object) -> str:
    if value is None:
        return "UNSELECTED"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return repr(value)
    return str(value)


class ReferenceGraphDialog(QDialog):
    """Display frozen reference recipes with a read-only graph and facts panel.

    ``reference_graphs`` is an immutable project-metadata tuple. The dialog
    emits the replacement tuple whenever a selection or display-only graph
    setting changes, so its caller can use the existing project metadata API.
    A checkpoint store is read only after the user explicitly binds its folder.
    """

    reference_graphs_changed = Signal(object)
    native_recipes_changed = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        reference_graphs: tuple[ReferenceGraphAttachment, ...] = (),
        catalogue: RecipeCatalogue | None = None,
        project_root=None,
        project_path=None,
        reference_model_assets=(),
        native_recipes=(),
        correction_bindings=(),
        correction_context_factory=None,
        keep_correction=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("PlantCV and SMPTS reference graphs")
        self.setMinimumSize(1050, 650)
        self.resize(1440, 900)

        self.catalogue = catalogue or build_recipe_catalogue()
        if catalogue is not None:
            validate_recipe_catalogue(self.catalogue)
        self._catalogue_by_id = {
            recipe.recipe_id: recipe for recipe in self.catalogue.recipes
        }
        self._reference_graphs = validate_reference_graphs(reference_graphs)
        self._project_root = project_root
        self._project_path = project_path
        self._reference_model_assets = tuple(reference_model_assets)
        self._native_recipes = validate_native_recipes(native_recipes)
        self._correction_bindings = tuple(correction_bindings)
        self._correction_context_factory = correction_context_factory
        self._keep_correction_callback = keep_correction
        self._active_attachment: ReferenceGraphAttachment | None = None
        self._active_recipe: RecipeSpec | None = None
        self._active_entry_kind: str | None = None
        self._active_is_transient = False
        self._transient_reason: str | None = None
        self._canvas: PipelineCanvas | None = None
        self._inspector: PipelineInspector | None = None
        self._details_panel: QWidget | None = None
        self._updating_controls = False
        self._checkpoint_store: Path | None = None
        self._inspection_token = 0
        self._inspection_cancel: threading.Event | None = None
        self._inspection_workers: dict[int, _InspectionWorker] = {}
        self._inspection_result: CheckpointInspection | None = None
        self._inspection_closed = False
        self._inspection_stage_id: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Reference graph:"))
        self.recipe_selector = QComboBox(self)
        self.recipe_selector.setObjectName("referenceRecipeSelector")
        self._populate_recipe_selector()
        selector_row.addWidget(self.recipe_selector, 1)
        self.backend_label = QLabel(self)
        self.backend_label.setObjectName("referenceBackendIdentity")
        self.backend_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        selector_row.addWidget(self.backend_label)
        layout.addLayout(selector_row)

        self.claim_label = QLabel(self)
        self.claim_label.setObjectName("referenceProvenanceClaim")
        self.claim_label.setWordWrap(True)
        self.claim_label.setStyleSheet(
            "font-weight: 600; padding: 6px; border: 1px solid #7b8792;"
        )
        layout.addWidget(self.claim_label)

        self.attachment_label = QLabel(self)
        self.attachment_label.setObjectName("referenceAttachmentStatus")
        self.attachment_label.setWordWrap(True)
        layout.addWidget(self.attachment_label)

        store_row = QHBoxLayout()
        self._store_bind_button = QPushButton("Select checkpoint store…", self)
        self._store_bind_button.setObjectName("bindReferenceCheckpointStore")
        self._store_clear_button = QPushButton("Clear store", self)
        self._store_clear_button.setObjectName("clearReferenceCheckpointStore")
        self._store_clear_button.setEnabled(False)
        self._native_run_button = QPushButton('Run native calculation…', self)
        self._native_run_button.setObjectName('runNativeReferenceRecipe')
        self._native_run_button.clicked.connect(self._open_native_recipe)
        self._extended_threshold_button = QPushButton('Run extended threshold…', self)
        self._extended_threshold_button.setObjectName('runNativeExtendedThreshold')
        self._extended_threshold_button.clicked.connect(self._open_native_extended_threshold)
        self._naive_bayes_button = QPushButton('Run frozen Naive Bayes…', self)
        self._naive_bayes_button.setObjectName('runNativeNaiveBayes')
        self._naive_bayes_button.setToolTip('Select a PlantCV probability table and calculate per-class masks.')
        self._naive_bayes_button.clicked.connect(self._open_native_naive_bayes)
        self._kmeans_button = QPushButton('Run frozen K-means…', self)
        self._kmeans_button.setObjectName('runNativeKMeans')
        self._kmeans_button.setToolTip('Select numeric centres and calculate cluster labels and category masks.')
        self._kmeans_button.clicked.connect(self._open_native_kmeans)
        self._automatic_card_button = QPushButton('Automatic colour card…', self)
        self._automatic_card_button.setObjectName('runNativeAutomaticCard')
        self._automatic_card_button.setToolTip('Detect card chips and apply affine colour correction.')
        self._automatic_card_button.clicked.connect(self._open_native_automatic_card)
        self._calibrated_seed_button = QPushButton('Calibrated seeds…', self)
        self._calibrated_seed_button.setObjectName('runNativeCalibratedSeed')
        self._calibrated_seed_button.setToolTip('Correct card colours, crop seeds, segment and measure traits.')
        self._calibrated_seed_button.clicked.connect(self._open_native_calibrated_seed)
        self._core_recovery_button = QPushButton('Experimental core recovery…', self)
        self._core_recovery_button.setObjectName('runNativeCoreRecovery')
        self._core_recovery_button.setToolTip('Recover original foreground from eroded cores and inspect boundary uncertainty.')
        self._core_recovery_button.clicked.connect(self._open_native_core_recovery)
        self._paper_port_button = QPushButton('SMPTS paper port…', self)
        self._paper_port_button.setObjectName('runNativeSmptsPaper')
        self._paper_port_button.setToolTip('Run the provisional fixed/mean reconstruction with explicit assumptions and original RGB crops.')
        self._paper_port_button.clicked.connect(self._open_native_smpts_paper)
        self._store_label = QLabel("Checkpoint store: not selected", self)
        self._store_label.setObjectName("referenceCheckpointStoreBinding")
        self._store_label.setTextFormat(Qt.TextFormat.PlainText)
        self._store_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        store_row.addWidget(self._store_bind_button)
        store_row.addWidget(self._store_clear_button)
        store_row.addWidget(self._native_run_button)
        store_row.addWidget(self._extended_threshold_button)
        store_row.addWidget(self._store_label, 1)
        layout.addLayout(store_row)
        classifier_row = QHBoxLayout()
        classifier_row.addWidget(self._naive_bayes_button)
        classifier_row.addWidget(self._kmeans_button)
        classifier_row.addWidget(self._automatic_card_button)
        classifier_row.addWidget(self._calibrated_seed_button)
        classifier_row.addWidget(self._core_recovery_button)
        self._saved_native_button = QPushButton('Project native recipes', self)
        self._saved_native_button.setObjectName('projectNativeRecipes')
        self._native_menu = QMenu(self._saved_native_button)
        self._saved_native_button.setMenu(self._native_menu)
        self._refresh_native_menu()
        classifier_row.addWidget(self._saved_native_button)
        classifier_row.addStretch(1)
        layout.addLayout(classifier_row)
        paper_row = QHBoxLayout()
        paper_row.addWidget(self._paper_port_button)
        self._public_capture_button = QPushButton('Capture public seed calls…', self)
        self._public_capture_button.setObjectName('capturePublicSeedCalls')
        self._public_capture_button.clicked.connect(self._open_public_capture)
        paper_row.addWidget(self._public_capture_button)
        self._calibrated_public_capture_button = QPushButton('Capture calibrated public seeds…', self)
        self._calibrated_public_capture_button.setObjectName('captureCalibratedPublicSeedCalls')
        self._calibrated_public_capture_button.clicked.connect(self._open_calibrated_public_capture)
        paper_row.addWidget(self._calibrated_public_capture_button)
        self._qc_history_button = QPushButton('Retained recipe QC…', self)
        self._qc_history_button.setObjectName('projectMethodQCHistory')
        self._qc_history_button.setEnabled(project_root is not None)
        self._qc_history_button.clicked.connect(self._open_qc_history)
        paper_row.addWidget(self._qc_history_button)
        paper_row.addStretch(1)
        layout.addLayout(paper_row)

        self.graph_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.graph_splitter.setObjectName("referenceGraphSplitter")
        self.graph_splitter.setChildrenCollapsible(False)
        layout.addWidget(self.graph_splitter, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self.recipe_selector.currentIndexChanged.connect(self._recipe_selected)
        self._store_bind_button.clicked.connect(self._choose_checkpoint_store)
        self._store_clear_button.clicked.connect(self.clear_checkpoint_store)
        initial_index = self._initial_recipe_index()
        self.recipe_selector.setCurrentIndex(initial_index)
        self._recipe_selected(initial_index)

    @property
    def reference_graphs(self) -> tuple[ReferenceGraphAttachment, ...]:
        """Return the validated immutable metadata tuple for project saving."""

        return self._reference_graphs

    def _open_native_extended_threshold(self):
        recipes = (*build_native_extended_threshold_recipes().values(), build_native_range_recipe())
        selected, accepted = QInputDialog.getItem(
            self, 'Select native threshold', 'Threshold operation:',
            [recipe.title for recipe in recipes], 0, False)
        if not accepted:
            return
        recipe = next((item for item in recipes if item.title == selected), None)
        if recipe is None:
            return
        self._open_working_native_recipe(recipe)

    def _open_public_capture(self):
        if self._active_recipe is None or not self._public_capture_button.isEnabled():
            return
        from seedvision.ui.public_reference_capture_dialog import PublicReferenceCaptureDialog
        PublicReferenceCaptureDialog(self._active_recipe, self).exec()

    def _open_calibrated_public_capture(self):
        if self._active_recipe is None or not self._calibrated_public_capture_button.isEnabled():
            return
        from seedvision.ui.calibrated_public_capture_dialog import CalibratedPublicReferenceCaptureDialog
        CalibratedPublicReferenceCaptureDialog(self._active_recipe, self).exec()

    def _open_native_naive_bayes(self):
        from seedvision.benchmarking.native_naive_bayes_recipe import build_native_naive_bayes_recipe
        self._open_working_native_recipe(build_native_naive_bayes_recipe())

    def _open_native_kmeans(self):
        from seedvision.benchmarking.native_kmeans_recipe import build_native_kmeans_recipe
        self._open_working_native_recipe(build_native_kmeans_recipe())

    def _open_native_automatic_card(self):
        from seedvision.benchmarking.native_card_recipe import build_native_card_recipe
        self._open_working_native_recipe(build_native_card_recipe())

    def _open_native_calibrated_seed(self):
        from seedvision.benchmarking.native_calibrated_seed_recipe import build_native_calibrated_seed_recipe
        selected, accepted = QInputDialog.getItem(self, 'Calibrated seed analysis',
            'Threshold operation:', ['binary','mean','otsu','range'], 0, False)
        if accepted:
            self._open_working_native_recipe(build_native_calibrated_seed_recipe(selected))

    def _open_native_core_recovery(self):
        from seedvision.benchmarking.native_core_recovery_recipe import build_native_core_recovery_recipe
        self._open_working_native_recipe(build_native_core_recovery_recipe())

    def _open_native_smpts_paper(self):
        from seedvision.benchmarking.native_smpts_recipe import build_native_smpts_recipe
        self._open_working_native_recipe(build_native_smpts_recipe())

    def _open_native_recipe(self):
        recipe = self.active_attachment.recipe
        if recipe.provenance.mode is RecipeProvenanceMode.NATIVE_COMPATIBLE:
            self._open_saved_native_recipe(recipe)
            return
        canonical = self._catalogue_by_id.get(recipe.recipe_id)
        if recipe.recipe_id not in ('plantcv_seed_binary', 'plantcv_seed_mean', 'plantcv_seed_otsu', 'plantcv_color_card_correct_color') or not _matches_catalogue_with_selected_values(recipe, canonical):
            return
        if recipe.recipe_id == 'plantcv_color_card_correct_color':
            self._open_correction_recipe(recipe)
            return
        from seedvision.benchmarking.native_recipe import build_native_seed_recipe
        self._open_working_native_recipe(build_native_seed_recipe(recipe))

    def _open_saved_native_recipe(self, recipe):
        if recipe.recipe_id == 'plantcv_color_card_correct_color':
            self._open_correction_recipe(recipe)
            return
        from seedvision.benchmarking.native_recipe import _validate_recipe
        from seedvision.persistence.reference_models import model_asset_paths
        try:
            _validate_recipe(recipe)
            paths = {} if self._project_root is None else model_asset_paths(
                self._project_root, recipe, self._reference_model_assets)
        except ValueError as error:
            QMessageBox.warning(self, 'Native recipe unavailable', str(error))
            return
        if recipe.recipe_id == 'plantcv_kmeans_predict_masks':
            from seedvision.ui.native_kmeans_dialog import NativeKMeansDialog as Dialog
        elif recipe.recipe_id == 'plantcv_naive_bayes_classifier':
            from seedvision.ui.native_naive_bayes_dialog import NativeNaiveBayesDialog as Dialog
        else:
            from seedvision.ui.native_recipe_dialog import NativeRecipeDialog as Dialog
        options = {'model_asset_paths': paths}
        if self._project_root is not None:
            options['keep_in_project'] = self._keep_native_recipe
        dialog = Dialog(recipe, self, **options)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def _open_correction_recipe(self, recipe):
        from seedvision.benchmarking.native_correction_recipe import build_native_correction_recipe
        from seedvision.persistence.correction_inputs import validate_correction_binding_records
        from seedvision.ui.native_correction_dialog import NativeCorrectionDialog
        try:
            recipe = build_native_correction_recipe(recipe)
            context = None if self._correction_context_factory is None else self._correction_context_factory()
            bindings = validate_correction_binding_records(self._correction_bindings)
            key = None if context is None else (context.image.identifier, recipe.recipe_id,
                                                recipe.recipe_version, recipe.identity.recipe_sha256)
            binding = next((item for item in bindings if item.key == key), None)
            dialog = NativeCorrectionDialog(recipe, self, project_context=context, project_binding=binding,
                keep_in_project=self._keep_correction if context is not None and self._keep_correction_callback else None)
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, 'Correction unavailable', str(error))
            return
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def _keep_correction(self, request, cancel, current):
        from seedvision.persistence.correction_inputs import select_correction_binding
        from seedvision.persistence.correction_session import validate_correction_keep_result
        captured = self._keep_correction_callback(request, cancel, current)
        validate_correction_keep_result(request, captured)
        self._native_recipes = select_native_recipe(self._native_recipes, request.recipe)
        self._correction_bindings = select_correction_binding(self._correction_bindings, captured)
        self._refresh_native_menu()
        self.native_recipes_changed.emit(self._native_recipes)
        return captured

    @property
    def native_recipes(self):
        return self._native_recipes

    def _open_working_native_recipe(self, recipe):
        selected = next((item for item in self._native_recipes if recipe_key(item) == recipe_key(recipe)), recipe)
        self._open_saved_native_recipe(selected)

    def _keep_native_recipe(self, recipe):
        from seedvision.benchmarking.native_recipe import _validate_recipe
        _validate_recipe(recipe)
        updated = select_native_recipe(self._native_recipes, recipe)
        if updated != self._native_recipes:
            self._native_recipes = updated
            self._refresh_native_menu()
            self.native_recipes_changed.emit(updated)

    def _remove_native_recipe(self, key):
        updated = validate_native_recipes(tuple(recipe for recipe in self._native_recipes if recipe_key(recipe) != key))
        if updated != self._native_recipes:
            self._native_recipes = updated
            from seedvision.persistence.correction_inputs import retain_selected_correction_bindings
            self._correction_bindings = retain_selected_correction_bindings(
                self._correction_bindings, self._reference_graphs, updated)
            self._refresh_native_menu()
            self.native_recipes_changed.emit(updated)

    def _refresh_native_menu(self):
        self._native_menu.clear()
        for previous in getattr(self, '_native_submenus', ()):
            previous.deleteLater()
        self._native_submenus = []
        self._saved_native_button.setEnabled(bool(self._native_recipes))
        for recipe in self._native_recipes:
            menu = QMenu(recipe.title, self._native_menu)
            self._native_submenus.append(menu)
            self._native_menu.addMenu(menu)
            menu.addAction('Open working recipe').triggered.connect(
                lambda checked=False, recipe=recipe: self._open_saved_native_recipe(recipe))
            menu.addAction('Remove from project').triggered.connect(
                lambda checked=False, key=recipe_key(recipe): self._remove_native_recipe(key))

    @property
    def canvas(self) -> PipelineCanvas:
        if self._canvas is None:
            raise RuntimeError("No reference recipe is selected.")
        return self._canvas

    @property
    def active_attachment(self) -> ReferenceGraphAttachment:
        if self._active_attachment is None:
            raise RuntimeError("No reference recipe is selected.")
        return self._active_attachment

    def bind_checkpoint_store(self, path: Path | str) -> None:
        """Bind a user-selected content-addressed store for this dialog session."""
        selected = Path(path).expanduser().resolve(strict=False)
        self._checkpoint_store = selected
        self._store_label.setText(f"Checkpoint store (this session): {selected}")
        self._store_clear_button.setEnabled(True)
        if self._active_attachment is not None:
            self._show_stage(self._inspection_stage_id or self.active_attachment.selected_stage_id or self.active_attachment.recipe.stages[0].stage_id)

    def clear_checkpoint_store(self) -> None:
        """Remove the ephemeral store binding without changing project metadata."""
        self._checkpoint_store = None
        self._store_label.setText("Checkpoint store: not selected")
        self._store_clear_button.setEnabled(False)
        if self._active_attachment is not None:
            self._show_stage(self._inspection_stage_id or self.active_attachment.selected_stage_id or self.active_attachment.recipe.stages[0].stage_id)

    def _choose_checkpoint_store(self) -> None:
        initial = "" if self._checkpoint_store is None else str(self._checkpoint_store)
        selected = QFileDialog.getExistingDirectory(
            self, "Select explicit checkpoint store", initial
        )
        if selected:
            self.bind_checkpoint_store(selected)

    def _cancel_checkpoint_inspection(self) -> None:
        self._inspection_token += 1
        if self._inspection_cancel is not None:
            self._inspection_cancel.set()
        self._inspection_cancel = None
        # Workers auto-delete in Qt after run() returns. Keep cancellation
        # cooperative and discard their queued result by token; tryTake() can
        # dereference an already-deleted QRunnable before its queued slot runs.

    def _inspection_completed(self, token: int, result: object) -> None:
        self._inspection_workers.pop(token, None)
        if (
            self._inspection_closed
            or token != self._inspection_token
            or not isinstance(result, CheckpointInspection)
            or self._inspection_cancel is not None and self._inspection_cancel.is_set()
        ):
            return
        stage_id = self._inspection_stage_id
        if stage_id is None or self._active_attachment is None:
            return
        if self._inspection_stage_id != stage_id:
            return
        self._inspection_cancel = None
        self._set_inspection_result(result)

    def _set_inspection_result(self, result: CheckpointInspection) -> None:
        self._inspection_result = result
        self._inspection_status_label.setText(
            f"{result.status.value.replace('_', ' ').upper()} — {result.detail}"
        )
        self._inspection_identity_label.setText(
            "Source and coordinate-frame identity: unavailable."
        )
        for label in self._inspection_preview_labels.values():
            label.clear()
            label.setText("No verified pixels")
        for label in self._inspection_preview_descriptions.values():
            label.clear()
        if result.status is InspectionStatus.VERIFIED:
            frame = result.coordinate_frame
            self._inspection_identity_label.setText(
                "Checkpoint-bound source SHA-256: "
                f"{result.source_sha256}\nCoordinate frame: "
                f"{frame.frame_id}; shape {frame.shape}; origin {frame.origin}; "
                f"axes {frame.axes}; units {frame.units}; transform SHA-256 "
                f"{frame.transform_sha256}.\nActive desktop image identity was not independently verified; "
                "these pixels are not overlaid on native outputs."
            )
            for slot, alternatives in (
                ("input.gray", ("input.gray", "input.mask", "input.peaks")),
                ("output.mask", ("output.mask", "output.image", "output.distance", "output.markers")),
            ):
                for key in alternatives:
                    array = result.preview_arrays.get(key)
                    if array is not None:
                        self._set_preview_array(slot, array)
                        break
            inventory_lines = [
                f"{item['artifact']}.{item['name']}: {item['dtype']} {item['shape']} "
                f"({item['nbytes']} bytes), role={item['role']}"
                for item in result.array_inventory
            ]
            metadata = dict(result.compact_metadata)
            metadata["verified_array_inventory"] = inventory_lines
            metadata["execution_status"] = (
                None if result.execution_status is None else result.execution_status.value
            )
            self._inspection_metadata_browser.setPlainText(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2)
            )
        else:
            self._inspection_metadata_browser.clear()
        self._refresh_stage_facts()

    def _set_preview_array(self, key: str, array: np.ndarray) -> None:
        if array.ndim != 2 or array.dtype not in (np.dtype(np.uint8), np.dtype(np.bool_), np.dtype(np.int32), np.dtype(np.float32)):
            self._inspection_preview_labels[key].setText(
                "Preview unsupported: expected a uint8/bool/int32/float32 H×W raster."
            )
            return
        height, width = array.shape
        if not height or not width:
            self._inspection_preview_labels[key].setText("Empty raster; no preview pixels.")
            return
        axis_budget = max(1, math.isqrt(MAX_PREVIEW_PIXELS))
        stride_y = max(1, (height + axis_budget - 1) // axis_budget)
        stride_x = max(1, (width + axis_budget - 1) // axis_budget)
        preview_height = (height + stride_y - 1) // stride_y
        preview_width = (width + stride_x - 1) // stride_x
        if preview_height * preview_width > MAX_PREVIEW_PIXELS:
            raise RuntimeError("Per-axis checkpoint preview bound failed before allocation.")
        sampled = array[::stride_y, ::stride_x]
        display_note = "Original uint8 intensities."
        image_format = QImage.Format.Format_Grayscale8
        if array.dtype == np.dtype(np.bool_):
            preview = np.ascontiguousarray(sampled.astype(np.uint8)*255)
            display_note = "Boolean peaks: black=false, white=true."
        elif array.dtype == np.dtype(np.int32):
            if np.any(sampled < 0):
                self._inspection_preview_labels[key].setText("Preview unsupported: negative instance labels.")
                return
            values = sampled.astype(np.int64)
            colours = (values*0x9e3779b1) & 0xffffff
            preview = np.stack(tuple(((colours >> shift) & 255).astype(np.uint8) for shift in (16,8,0)),axis=-1)
            preview[values == 0] = 0
            preview = np.ascontiguousarray(preview)
            image_format = QImage.Format.Format_RGB888
            display_note = "Categorical label colours; label 0 is black. Exact IDs remain in the checkpoint."
        elif array.dtype == np.dtype(np.float32):
            lower, upper = float(np.min(array)), float(np.max(array))
            if not math.isfinite(lower) or not math.isfinite(upper):
                self._inspection_preview_labels[key].setText("Preview unsupported: nonfinite distance values.")
                return
            if upper > lower:
                preview = np.clip(np.rint((sampled.astype(np.float64)-lower)*(255.0/(upper-lower))),0,255).astype(np.uint8)
            else:
                preview = np.full(sampled.shape,255 if upper > 0 else 0,dtype=np.uint8)
            preview = np.ascontiguousarray(preview)
            display_note = f"Distance display scaled from {lower:g} to {upper:g} pixels; archived float32 values are unchanged."
        else:
            preview = np.ascontiguousarray(sampled)
        image = QImage(
            preview.data,
            int(preview.shape[1]),
            int(preview.shape[0]),
            int(preview.strides[0]),
            image_format,
        ).copy()
        image_label = self._inspection_preview_labels[key]
        image_label.set_display_image(image)
        self._inspection_preview_descriptions[key].setText(
            f"Original {width}×{height}; display {preview.shape[1]}×{preview.shape[0]} "
            f"(nearest-neighbour strides y={stride_y}, x={stride_x}; "
            f"{preview_height*preview_width:,} preview pixels). {display_note}"
        )

    def _detail_tab_changed(self, index: int) -> None:
        if index == 1:
            page_layout = self._inspection_layout_refs[0]
            page_layout.invalidate()
            page_layout.activate()

    def _shutdown_inspection(self) -> None:
        if self._inspection_closed:
            return
        self._inspection_closed = True
        self._cancel_checkpoint_inspection()
        for worker in tuple(self._inspection_workers.values()):
            try:
                worker.signals.completed.disconnect(self._inspection_completed)
            except (RuntimeError, TypeError):
                pass
        self._inspection_workers.clear()

    def accept(self) -> None:
        self._shutdown_inspection()
        super().accept()

    def reject(self) -> None:
        self._shutdown_inspection()
        super().reject()

    def done(self, result: int) -> None:
        self._shutdown_inspection()
        super().done(result)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._shutdown_inspection()
        super().closeEvent(event)

    @staticmethod
    def _attachment_key(recipe: RecipeSpec) -> tuple[str, str]:
        return recipe.recipe_id, recipe.recipe_version

    def _populate_recipe_selector(self) -> None:
        """Show current catalogue entries and every exact stored recipe version."""

        stored_by_key = {
            self._attachment_key(item.recipe): item
            for item in self._reference_graphs
        }
        catalogue_keys = {
            self._attachment_key(recipe): recipe for recipe in self.catalogue.recipes
        }
        for recipe in self.catalogue.recipes:
            key = self._attachment_key(recipe)
            stored = stored_by_key.get(key)
            if stored is not None:
                label = f"{stored.recipe.title} — stored v{stored.recipe.recipe_version}"
                entry_kind = "stored"
            else:
                label = f"{recipe.title} — catalogue v{recipe.recipe_version}"
                entry_kind = "catalogue"
            self.recipe_selector.addItem(label, recipe.recipe_id)
            self.recipe_selector.setItemData(
                self.recipe_selector.count() - 1,
                (entry_kind, *key),
                Qt.ItemDataRole.UserRole + 1,
            )

        for attachment in self._reference_graphs:
            recipe = attachment.recipe
            key = self._attachment_key(recipe)
            if key in catalogue_keys:
                continue
            if recipe.recipe_id in self._catalogue_by_id:
                label = f"{recipe.title} — stored historical v{recipe.recipe_version}"
            else:
                label = (
                    f"{recipe.title} — stored {recipe.recipe_id} v{recipe.recipe_version} "
                    "(no current catalogue entry)"
                )
            self.recipe_selector.addItem(label, recipe.recipe_id)
            self.recipe_selector.setItemData(
                self.recipe_selector.count() - 1,
                ("stored", *key),
                Qt.ItemDataRole.UserRole + 1,
            )

    def _initial_recipe_index(self) -> int:
        if self._reference_graphs:
            first = self._reference_graphs[0].recipe
            for index in range(self.recipe_selector.count()):
                if self.recipe_selector.itemData(
                    index, Qt.ItemDataRole.UserRole + 1
                ) == ("stored", first.recipe_id, first.recipe_version):
                    return index
        return 0

    def _attachment_for(self, recipe: RecipeSpec) -> ReferenceGraphAttachment:
        existing = next(
            (
                item
                for item in self._reference_graphs
                if (item.recipe.recipe_id, item.recipe.recipe_version)
                == (recipe.recipe_id, recipe.recipe_version)
            ),
            None,
        )
        if existing is not None:
            return existing
        stage_ids = tuple(stage.stage_id for stage in recipe.stages)
        selected = next(
            (
                stage.stage_id
                for stage in recipe.stages
                if stage.resolution is SourceResolution.PINNED
            ),
            stage_ids[0],
        )
        return ReferenceGraphAttachment(
            recipe=recipe,
            stage_runs=tuple(
                initial_stage_run(recipe, stage_id) for stage_id in stage_ids
            ),
            selected_stage_id=selected,
        )

    def _upsert_attachment(self, attachment: ReferenceGraphAttachment) -> None:
        key = self._attachment_key(attachment.recipe)
        if (
            self._active_is_transient
            and self._active_attachment is not None
            and self._attachment_key(self._active_attachment.recipe) == key
        ):
            return
        exists = any(
            self._attachment_key(item.recipe) == key
            for item in self._reference_graphs
        )
        if not exists and len(self._reference_graphs) >= MAX_REFERENCE_GRAPHS:
            # Preserve all stored historical graphs. The unrepresented current
            # catalogue recipe remains usable as an explicitly transient view.
            if self._active_attachment is not None and (
                self._attachment_key(self._active_attachment.recipe) == key
            ):
                self._mark_active_transient(
                    f"the {MAX_REFERENCE_GRAPHS}-graph attachment limit is reached"
                )
            return
        retained = tuple(
            attachment if self._attachment_key(item.recipe) == key else item
            for item in self._reference_graphs
        )
        updated = (
            retained
            if exists
            else (*self._reference_graphs, attachment)
        )
        try:
            updated = validate_reference_graphs(updated)
        except ReferenceGraphPersistenceError as error:
            if not self._is_capacity_error(error):
                raise
            self._mark_active_transient(
                f"project graph metadata reached its persistence capacity ({error})"
            )
            return
        if updated == self._reference_graphs:
            return
        self._reference_graphs = updated
        if self._active_attachment is not None and (
            self._attachment_key(self._active_attachment.recipe) == key
        ):
            self._active_is_transient = False
        self.reference_graphs_changed.emit(updated)

    @staticmethod
    def _is_capacity_error(error: Exception) -> bool:
        """Identify persistence bounds without masking malformed metadata."""

        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "-byte limit",
                "-byte aggregate limit",
                "-item limit",
                "support at most ",
                "supports at most ",
            )
        )

    def _mark_active_transient(self, reason: str) -> None:
        self._active_is_transient = True
        if self._transient_reason is None:
            self._transient_reason = reason
        self._refresh_attachment_status()

    def _recipe_selected(self, index: int) -> None:
        if index < 0:
            return
        entry = self.recipe_selector.itemData(
            index, Qt.ItemDataRole.UserRole + 1
        )
        if not isinstance(entry, tuple) or len(entry) != 3:
            return
        entry_kind, recipe_id, recipe_version = entry
        self._active_entry_kind = entry_kind
        self._active_is_transient = False
        self._transient_reason = None
        if entry_kind == "stored":
            attachment = next(
                (
                    item for item in self._reference_graphs
                    if self._attachment_key(item.recipe) == (recipe_id, recipe_version)
                ),
                None,
            )
            if attachment is None:
                return
        elif entry_kind == "catalogue":
            catalogue_recipe = self._catalogue_by_id.get(recipe_id)
            if catalogue_recipe is None or catalogue_recipe.recipe_version != recipe_version:
                return
            attachment = self._attachment_for(catalogue_recipe)
        else:
            return
        self._active_attachment = attachment
        self._active_recipe = attachment.recipe
        self._upsert_attachment(attachment)
        self._render_active_graph()

    def _render_active_graph(self) -> None:
        attachment = self.active_attachment
        recipe = attachment.recipe
        catalogue_recipe = self._catalogue_by_id.get(recipe.recipe_id)
        native_supported = False
        if recipe.provenance.mode is RecipeProvenanceMode.NATIVE_COMPATIBLE:
            try:
                from seedvision.benchmarking.native_recipe import _validate_recipe
                _validate_recipe(recipe)
                native_supported = True
            except ValueError:
                pass
        self._native_run_button.setEnabled(
            native_supported or recipe.recipe_id in ('plantcv_seed_binary', 'plantcv_seed_mean', 'plantcv_seed_otsu', 'plantcv_color_card_correct_color')
            and _matches_catalogue_with_selected_values(recipe, catalogue_recipe))
        self._public_capture_button.setEnabled(
            recipe.provenance.mode is RecipeProvenanceMode.ORIGINAL_REFERENCE
            and recipe.recipe_id in ('plantcv_seed_binary', 'plantcv_seed_mean', 'plantcv_seed_otsu')
            and _matches_catalogue_with_selected_values(recipe, catalogue_recipe))
        self._calibrated_public_capture_button.setEnabled(self._public_capture_button.isEnabled())
        same_identity = catalogue_recipe is not None and (
            recipe.identity.analytical_dict()
            == catalogue_recipe.identity.analytical_dict()
        )

        if self._canvas is not None:
            self._canvas.setParent(None)
            self._canvas.deleteLater()
        if self._details_panel is not None:
            self._details_panel.setParent(None)
            self._details_panel.deleteLater()
        self._canvas = None
        self._inspector = None
        self._details_panel = None

        graph = to_pipeline_graph(recipe, attachment.stage_runs)
        selected = attachment.selected_stage_id or recipe.stages[0].stage_id
        canvas = PipelineCanvas(
            graph,
            read_only=True,
            reference_view=True,
            initial_focus_node_id=selected,
        )
        canvas.setObjectName("referencePipelineCanvas")
        inspector = PipelineInspector(self, read_only=True)
        inspector.setObjectName("referencePipelineInspector")
        inspector.setMinimumWidth(320)
        inspector.setMinimumHeight(145)
        facts = QTextBrowser(self)
        facts.setObjectName("referenceStageFacts")
        facts.setOpenExternalLinks(False)
        facts.setMinimumHeight(240)

        detail_tabs = QTabWidget(self)
        detail_tabs.setObjectName("referenceStageDetailTabs")
        detail_tabs.addTab(facts, "Stage facts")
        inspection_page = QWidget(detail_tabs)
        inspection_layout = QVBoxLayout(inspection_page)
        inspection_layout.setContentsMargins(6, 6, 6, 6)
        inspection_status = QLabel("No checkpoint inspection requested.", inspection_page)
        inspection_status.setObjectName("referenceCheckpointInspectionStatus")
        inspection_status.setWordWrap(True)
        inspection_status.setTextFormat(Qt.TextFormat.PlainText)
        inspection_identity = QLabel("Source and coordinate-frame identity: unavailable.", inspection_page)
        inspection_identity.setObjectName("referenceCheckpointIdentity")
        inspection_identity.setWordWrap(True)
        inspection_identity.setTextFormat(Qt.TextFormat.PlainText)
        inspection_identity.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        inspection_layout.addWidget(inspection_status)
        inspection_layout.addWidget(inspection_identity)
        method_qc_button = QPushButton('Recorded stage QC…', inspection_page)
        method_qc_button.setObjectName('referenceStageMethodQC')
        method_qc_button.clicked.connect(self._review_stage_qc)
        inspection_layout.addWidget(method_qc_button)
        preview_row = QHBoxLayout()
        preview_images: dict[str, QLabel] = {}
        preview_descriptions: dict[str, QLabel] = {}
        preview_group_layouts = []
        for key, title in (
            ("input.gray", "Input pixels · verified upstream checkpoint"),
            ("output.mask", "Output pixels · verified stage checkpoint"),
        ):
            group = QWidget(inspection_page)
            group.setMinimumSize(220, 238)
            group_layout = QVBoxLayout(group)
            preview_group_layouts.append(group_layout)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.setSpacing(4)
            title_label = QLabel(title, group)
            title_label.setWordWrap(True)
            title_label.setStyleSheet("font-weight: 600;")
            image_label = _CheckpointPreviewLabel("No verified pixels", group)
            image_label.setObjectName(
                "referenceCheckpointInputPreview"
                if key == "input.gray" else "referenceCheckpointOutputPreview"
            )
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # Captions can wrap to different heights. Let the preview shrink
            # within its shared row rather than force it across a caption.
            image_label.setMinimumSize(220, 64)
            image_label.setStyleSheet("background: #20252a; color: #e8edf1; border: 1px solid #7b8792;")
            description_label = QLabel("", group)
            description_label.setWordWrap(True)
            description_label.setMinimumHeight(36)
            group_layout.addWidget(title_label)
            group_layout.addWidget(image_label, 1)
            group_layout.addWidget(description_label)
            preview_row.addWidget(group, 1)
            preview_images[key] = image_label
            preview_descriptions[key] = description_label
        inspection_layout.addLayout(preview_row, 2)
        inspection_metadata = QTextBrowser(inspection_page)
        inspection_metadata.setObjectName("referenceCheckpointMetadata")
        inspection_metadata.setOpenExternalLinks(False)
        inspection_metadata.setPlaceholderText("Compact artifact observations and runtime appear after verification.")
        inspection_layout.addWidget(inspection_metadata, 1)
        detail_tabs.addTab(inspection_page, "Checkpoint inspection")

        details_panel = QWidget(self.graph_splitter)
        details_layout = QVBoxLayout(details_panel)
        details_layout.setContentsMargins(8, 0, 0, 0)
        details_layout.addWidget(inspector, 1)
        details_layout.addWidget(detail_tabs, 4)
        self.graph_splitter.addWidget(canvas)
        self.graph_splitter.addWidget(details_panel)
        self.graph_splitter.setStretchFactor(0, 3)
        self.graph_splitter.setStretchFactor(1, 2)

        self._canvas = canvas
        self._inspector = inspector
        self._facts_browser = facts
        self._detail_tabs = detail_tabs
        self._inspection_status_label = inspection_status
        self._inspection_identity_label = inspection_identity
        self._inspection_metadata_browser = inspection_metadata
        self._inspection_preview_labels = preview_images
        self._inspection_preview_descriptions = preview_descriptions
        self._inspection_layout_refs = (
            inspection_layout, preview_row, *preview_group_layouts
        )
        detail_tabs.currentChanged.connect(self._detail_tab_changed)
        self._details_panel = details_panel
        self._render_identity_header(recipe, same_identity)

        self._updating_controls = True
        try:
            canvas.set_cable_bundling_enabled(attachment.bundle_cables)
            canvas.set_obstacle_routing_enabled(attachment.route_around_nodes)
            canvas.select_node(selected)
            self._show_stage(selected)
        finally:
            self._updating_controls = False

        canvas.node_selected.connect(self._stage_selected)
        canvas.layout_changed.connect(self._layout_changed)
        canvas.presentation_changed.connect(self._presentation_changed)

    def _render_identity_header(self, recipe: RecipeSpec, same_identity: bool) -> None:
        provenance = recipe.provenance
        self.backend_label.setText(
            f"{provenance.backend_id} · {provenance.backend_version} · "
            f"{provenance.mode.value}"
        )
        catalogue_recipe = self._catalogue_by_id.get(recipe.recipe_id)
        if provenance.mode is RecipeProvenanceMode.PAPER_RECONSTRUCTION:
            if provenance.reference_id == "smpts_paper":
                claim = (
                    "SMPTS paper reconstruction only. The original implementation is "
                    "unavailable in the stored source record; this reconstruction does "
                    "not claim recovered code or original-code parity."
                )
            else:
                claim = (
                    "Paper-reconstruction provenance. This graph does not claim to be "
                    "recovered original code or to have original-code parity."
                )
        elif provenance.mode is RecipeProvenanceMode.NATIVE_COMPATIBLE:
            claim = (
                "Native-compatible SeedFiddle contract. It is not an original-reference "
                "implementation or a reference-parity claim."
            )
        elif provenance.mode is RecipeProvenanceMode.ENHANCED:
            claim = (
                "Enhanced SeedFiddle contract. It is not an original-reference "
                "implementation or a reference-parity claim."
            )
        elif (
            provenance.mode is RecipeProvenanceMode.ORIGINAL_REFERENCE
            and provenance.reference_id == "plantcv"
            and provenance.backend_id == "plantcv.reference"
            and (same_identity or _matches_catalogue_with_selected_values(recipe, catalogue_recipe))
        ):
            if recipe.recipe_id == "plantcv_color_card_correct_color":
                claim = (
                    "Audited original PlantCV function: one correct_color operation. "
                    "This is a single-operation contract, not a complete color-calibration workflow."
                )
            else:
                claim = (
                    "Audited original PlantCV functions; this stage order was selected by "
                    "SeedFiddle and is not a published end-to-end PlantCV seed workflow."
                )
        elif provenance.mode is RecipeProvenanceMode.ORIGINAL_REFERENCE:
            reference_name = provenance.reference_id or "unspecified reference"
            claim = (
                f"Stored metadata declares original-reference provenance for {reference_name}. "
                "Only the saved source mappings below are shown; current-source audit "
                "status is not independently asserted."
            )
        else:
            claim = "Stored reference recipe metadata. This view does not execute operations."
        claim += (
            " This view does not execute operations; checkpoint pixels appear only "
            "after an explicit store binding and artifact verification."
        )
        self.claim_label.setText(claim)
        if self._active_is_transient:
            reason = self._transient_reason or "a project metadata persistence limit"
            state = (
                f"TRANSIENT — NOT SAVED: {reason}. This graph remains inspectable; "
                "stored project attachments are unchanged."
            )
        elif catalogue_recipe is None:
            state = (
                f"HISTORICAL STORED RECIPE: {recipe.recipe_id} v{recipe.recipe_version} "
                "has no current catalogue entry. Showing this exact saved recipe."
            )
        elif same_identity:
            state = "Stored graph matches the current frozen catalogue identity."
        elif _matches_catalogue_with_selected_values(recipe, catalogue_recipe):
            state = (
                "SELECTED PARAMETERS: source, topology and parameter declarations match "
                "the current catalogue. Showing the exact saved parameterized recipe identity."
            )
        elif _matches_native_catalogue(recipe, catalogue_recipe):
            state = (
                "NATIVE RECIPE: source, topology and parameter declarations match the "
                "current catalogue. Saved native code and runtime provenance are checked "
                "when an archived stage is inspected."
            )
        else:
            state = (
                f"STALE CATALOGUE IDENTITY: showing exact stored recipe "
                f"version {recipe.recipe_version}; it differs from current catalogue "
                f"version {catalogue_recipe.recipe_version}."
            )
        self.attachment_label.setText(
            f"{state} Historical run records are metadata only. "
            "Checkpoint arrays are unavailable in project graph metadata."
        )

    def _refresh_attachment_status(self) -> None:
        if self._active_attachment is None:
            return
        catalogue_recipe = self._catalogue_by_id.get(
            self._active_attachment.recipe.recipe_id
        )
        same_identity = catalogue_recipe is not None and (
            self._active_attachment.recipe.identity.analytical_dict()
            == catalogue_recipe.identity.analytical_dict()
        )
        self._render_identity_header(self._active_attachment.recipe, same_identity)

    def _generic_inspector_node(self, stage_id: str) -> PipelineNode:
        """Create a collision-proof generic node for the native read-only inspector."""

        recipe = self.active_attachment.recipe
        stage = recipe.stage(stage_id)
        run = next(item for item in self.active_attachment.stage_runs if item.stage_id == stage_id)
        status, detail = {
            StageRunState.UNRESOLVED: (NodeStatus.PLANNED, "Source operation is unresolved"),
            StageRunState.NOT_RUN: (NodeStatus.IDLE, "Not run"),
            StageRunState.UNAVAILABLE: (NodeStatus.BLOCKED, "Reference source or execution is unavailable"),
            StageRunState.EXECUTED: (NodeStatus.COMPLETE, "Historical execution metadata"),
            StageRunState.EMPTY: (NodeStatus.COMPLETE, "Historical empty result metadata"),
            StageRunState.BYPASSED: (NodeStatus.BYPASSED, "Historical bypass metadata"),
            StageRunState.FAILED: (NodeStatus.FAILED, "Historical failure metadata"),
            StageRunState.UNSUPPORTED: (NodeStatus.BLOCKED, "Operation is unsupported by this view"),
        }[run.state]
        if run.reason:
            detail = f"{detail}: {run.reason}"
        return PipelineNode(
            identifier=f"reference_inspector_{stage_id}",
            title=stage.title,
            category="reference contract",
            description=stage.description,
            details="",
            x=0.0,
            y=0.0,
            enabled=True,
            implemented=False,
            status=status,
            status_detail=detail,
        )

    def _stage_selected(self, stage_id: str) -> None:
        if self._updating_controls:
            return
        attachment = self.active_attachment
        if stage_id not in {stage.stage_id for stage in attachment.recipe.stages}:
            return
        try:
            updated = replace(attachment, selected_stage_id=stage_id)
        except (RecipeContractError, ReferenceGraphPersistenceError) as error:
            if not self._is_capacity_error(error):
                raise
            self._mark_active_transient(
                f"project graph metadata reached its persistence capacity ({error})"
            )
            self._show_stage(stage_id)
            return
        self._active_attachment = updated
        self._upsert_attachment(updated)
        self._show_stage(stage_id)

    def _layout_changed(self) -> None:
        if self._updating_controls:
            return
        attachment = self.active_attachment
        positions = tuple(
            LayoutPosition(stage_id, float(item.pos().x()), float(item.pos().y()))
            for stage_id, item in self.canvas.node_items.items()
        )
        try:
            recipe = replace(attachment.recipe, layout=positions)
            updated = replace(attachment, recipe=recipe)
        except (RecipeContractError, ReferenceGraphPersistenceError) as error:
            if not self._is_capacity_error(error):
                raise
            self._mark_active_transient(
                f"project graph metadata reached its persistence capacity ({error})"
            )
            return
        if recipe.identity.analytical_dict() != attachment.recipe.identity.analytical_dict():
            raise AssertionError("A layout-only edit changed analytical recipe identity.")
        self._active_attachment = updated
        self._active_recipe = recipe
        self._upsert_attachment(updated)

    def _presentation_changed(self) -> None:
        if self._updating_controls:
            return
        try:
            updated = replace(
                self.active_attachment,
                bundle_cables=self.canvas.cable_bundling_enabled,
                route_around_nodes=self.canvas.obstacle_routing_enabled,
            )
        except (RecipeContractError, ReferenceGraphPersistenceError) as error:
            if not self._is_capacity_error(error):
                raise
            self._mark_active_transient(
                f"project graph metadata reached its persistence capacity ({error})"
            )
            return
        self._active_attachment = updated
        self._upsert_attachment(updated)

    def _show_stage(self, stage_id: str) -> None:
        if self._inspector is None or self._active_attachment is None:
            return
        self._cancel_checkpoint_inspection()
        token = self._inspection_token
        self._inspection_stage_id = stage_id
        self._inspection_result = None
        recipe = self.active_attachment.recipe
        stage = recipe.stage(stage_id)
        record = next(
            item for item in self.active_attachment.stage_runs
            if item.stage_id == stage_id
        )
        self._inspector.set_node(self._generic_inspector_node(stage_id))
        # This native inspector fact is specific to the editable analysis graph;
        # reference recipe contracts do not have an enabled/disabled switch.
        self._inspector.enabled_fact_label.hide()
        catalogue_recipe = self._catalogue_by_id.get(recipe.recipe_id)
        current_identity = catalogue_recipe is not None and (
            recipe.identity.analytical_dict()
            == catalogue_recipe.identity.analytical_dict()
        )
        if record.state not in {StageRunState.EXECUTED, StageRunState.EMPTY}:
            self._set_inspection_result(_inspect_graph_checkpoint(None, recipe, record))
            return
        if self._checkpoint_store is None:
            self._set_inspection_result(_inspect_graph_checkpoint(None, recipe, record))
            return
        if not current_identity and not (
            _matches_catalogue_with_selected_values(recipe, catalogue_recipe)
            or _matches_native_catalogue(recipe, catalogue_recipe)
        ):
            self._set_inspection_result(CheckpointInspection(
                InspectionStatus.UNSUPPORTED,
                "This exact historical or stale recipe is preserved for inspection; "
                "its checkpoint is not loaded because its stage identity is not the current frozen catalogue entry.",
            ))
            return
        if recipe.provenance.mode not in {RecipeProvenanceMode.ORIGINAL_REFERENCE, RecipeProvenanceMode.NATIVE_COMPATIBLE}:
            self._set_inspection_result(CheckpointInspection(
                InspectionStatus.UNSUPPORTED,
                "No exact original/native-compatible PlantCV artifact verifier is available for this recipe mode.",
            ))
            return
        loading = CheckpointInspection(
            InspectionStatus.LOADING,
            "Loading bounded content-addressed artifacts from the explicitly selected store.",
        )
        self._set_inspection_result(loading)
        cancel = threading.Event()
        self._inspection_cancel = cancel
        worker = _InspectionWorker(
            token, self._checkpoint_store, recipe, record, cancel
        )
        worker.signals.completed.connect(self._inspection_completed)
        self._inspection_workers[token] = worker
        QThreadPool.globalInstance().start(worker)

    def _refresh_stage_facts(self) -> None:
        if self._active_attachment is None or self._inspection_stage_id is None:
            return
        recipe = self.active_attachment.recipe
        stage = recipe.stage(self._inspection_stage_id)
        record = next(
            item for item in self.active_attachment.stage_runs
            if item.stage_id == self._inspection_stage_id
        )
        self._facts_browser.setHtml(
            self._stage_facts_html(recipe, stage, record, self._inspection_result)
        )

    def _review_stage_qc(self) -> None:
        if self._active_attachment is None or self._inspection_stage_id is None:
            return
        result = self._inspection_result
        if result is None or result.status is InspectionStatus.LOADING:
            return
        from seedvision.export.method_qc import qc_from_stage_inspection
        from seedvision.ui.method_qc_dialog import MethodQCDialog
        recipe = self.active_attachment.recipe
        record = next(item for item in self.active_attachment.stage_runs
                      if item.stage_id == self._inspection_stage_id)
        try:
            quality = qc_from_stage_inspection(recipe, record, result)
        except (ValueError, TypeError) as error:
            self._inspection_status_label.setText(f'QC record unavailable: {error}')
            return
        MethodQCDialog(quality, self, project_root=self._project_root, project_path=self._project_path).exec()

    def _open_qc_history(self):
        if self._project_root is None:
            return
        from seedvision.persistence.method_qc_history import load_project_method_qc, _quality, _ack, _identities
        from seedvision.ui.method_qc_dialog import MethodQCDialog
        try:
            history = load_project_method_qc(self._project_root, self._project_path)
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, 'Retained QC unavailable', str(error))
            return
        if history.diagnostics:
            QMessageBox.warning(self, 'Retained QC diagnostics', '\n'.join(history.diagnostics[:8]))
        if not history.records:
            return
        qualities = [_quality(raw) for raw in history.records]
        labels = [f"{index+1}. {quality.payload['identity']['recipe_identity']['recipe_id']} · "
                  f"{quality.payload['identity']['scope']} · {quality.revision[:16]}"
                  for index, quality in enumerate(qualities)]
        selected, accepted = QInputDialog.getItem(self, 'Retained recipe QC', 'Select a historical QC view:', labels, 0, False)
        if not accepted or selected not in labels:
            return
        quality = qualities[labels.index(selected)]
        dialog = MethodQCDialog(quality, self, project_root=self._project_root, project_path=self._project_path)
        identities = _identities(history.records)
        events = [_ack(raw, identities) for raw in history.acknowledgments]
        events = [event for event in events if event['qc_revision'] == quality.revision]
        if events:
            dialog.status.setText(f'Recorded acknowledgments for this analytical QC revision ({len(events)} total; up to eight shown):\n'+ '\n'.join(
                f"{event['operator_account']} · {event['reviewed_at_utc']}" for event in events[:8])
                + '\nHistorical records do not approve seeds or reverify original artifacts.')
        dialog.exec()

    @staticmethod
    def _stage_facts_html(
        recipe: RecipeSpec,
        stage,
        record: StageRunRecord,
        inspection: CheckpointInspection | None = None,
    ) -> str:
        esc = _escaped
        rows: list[str] = []
        rows.append("<h2>Reference contract</h2>")
        rows.append(
            f"<p><b>{esc(stage.title)}</b><br>"
            f"Operation ID: <code>{esc(stage.operation_id)}</code><br>"
            f"Stage ID: <code>{esc(stage.stage_id)}</code></p>"
        )
        rows.append(f"<p><i>Recipe scope: {esc(recipe.description)}</i></p>")
        rows.append(f"<p>{esc(stage.description)}</p>")

        rows.append("<h3>Typed ports</h3><ul>")
        for port in stage.ports:
            flags = []
            if port.required:
                flags.append("required")
            else:
                flags.append("optional")
            if port.external:
                flags.append("external input")
            semantics = (
                f"; semantics {esc(dict(port.semantics))}"
                if port.semantics else ""
            )
            rows.append(
                f"<li><b>{esc(port.direction.title())}</b> "
                f"<code>{esc(port.name)}</code> — {esc(port.label)}; "
                f"type <code>{esc(port.data_type)}</code> ({', '.join(flags)})"
                f"{semantics}</li>"
            )
        rows.append("</ul>")

        rows.append("<h3>Declared parameters</h3>")
        if stage.parameters:
            if inspection is not None and inspection.status is InspectionStatus.VERIFIED:
                parameter_note = (
                    "These are recipe-contract values. Effective values were recorded "
                    "in the hash-verified stage artifact and appear in the inspection tab."
                )
            else:
                parameter_note = (
                    "These are recipe-contract values. No operation result has been "
                    "verified here, so effective runtime parameter values are not "
                    "recorded as checkpoint facts."
                )
            rows.append(f"<p>{esc(parameter_note)}</p><ul>")
            for parameter in stage.parameters:
                if parameter.value is None:
                    value = "UNSELECTED — required choice/value is absent."
                else:
                    value = (
                        f"Declared/fixed value: <code>{esc(_format_value(parameter.value))}</code>."
                    )
                status = "calculation-affecting" if parameter.affects_calculation else "display-only"
                editable = (
                    "catalogue marks this field editable; the reference graph view is read-only"
                    if parameter.editable
                    else "fixed contract field"
                )
                extra = []
                if parameter.units:
                    extra.append(f"units: {parameter.units}")
                if parameter.minimum is not None or parameter.maximum is not None:
                    extra.append(f"range: {parameter.minimum}…{parameter.maximum}")
                if parameter.choices:
                    extra.append(f"choices: {parameter.choices!r}")
                rows.append(
                    f"<li><code>{esc(parameter.parameter_id)}</code> ({esc(parameter.value_type)}; "
                    f"{status}; {editable}) — {value} {esc(parameter.description)}"
                    f"{' [' + esc('; '.join(extra)) + ']' if extra else ''}</li>"
                )
            rows.append("</ul>")
        else:
            rows.append("<p>No parameter values are declared for this stage.</p>")

        rows.append("<h3>Source mapping</h3>")
        if stage.source_mappings:
            rows.append("<ul>")
            for source in stage.source_mappings:
                source_line = (
                    f"<b>{esc(source.reference_id)}</b> {esc(source.reference_version)} — "
                    f"{esc(source.resolution.value)}"
                )
                if source.source_symbol:
                    source_line += f"<br>Function/symbol: <code>{esc(source.source_symbol)}</code>"
                if source.source_path:
                    source_line += f"<br>Source path: <code>{esc(source.source_path)}</code>"
                if source.source_sha256:
                    source_line += f"<br>Source SHA-256: <code>{esc(source.source_sha256)}</code>"
                if source.source_url:
                    source_line += f"<br>Source URL: {esc(source.source_url)}"
                if source.license_id:
                    source_line += f"<br>License: {esc(source.license_id)}"
                if source.note:
                    source_line += f"<br>{esc(source.note)}"
                rows.append(f"<li>{source_line}</li>")
            rows.append("</ul>")
        else:
            rows.append(
                f"<p>Source mapping unavailable for this stage. {esc(stage.resolution_note)}</p>"
            )
        rows.append(
            f"<p>Stage source resolution: <b>{esc(stage.resolution.value)}</b>. "
            f"Backend: {esc(recipe.provenance.backend_id)} "
            f"{esc(recipe.provenance.backend_version)}; mode: "
            f"{esc(recipe.provenance.mode.value)}.</p>"
        )

        if stage.condition is not None:
            rows.append(
                "<h3>Conditional stage</h3><p>Condition label: "
                f"<b>{esc(stage.condition.label)}</b><br>Expression: "
                f"<code>{esc(stage.condition.expression)}</code><br>"
                f"If true: {esc(stage.condition.outcome_when_true)}. "
                "The condition has not been evaluated by this view.</p>"
            )

        rows.append("<h3>Historical run state</h3>")
        rows.append(f"<p>State: <b>{esc(record.state.value.upper())}</b>.")
        if record.reason:
            rows.append(f"<br>Recorded reason: {esc(record.reason)}")
        if record.upstream_checkpoint_sha256:
            rows.append(
                "<br>Recorded upstream checkpoint digest: <code>"
                f"{esc(record.upstream_checkpoint_sha256)}</code>"
            )
        if record.state in {StageRunState.EXECUTED, StageRunState.EMPTY}:
            if inspection is not None and inspection.status is InspectionStatus.VERIFIED:
                rows.append(
                    "<br>Recorded checkpoint digest: <code>"
                    f"{esc(record.checkpoint_sha256)}</code>; matching checkpoint content "
                    "and its upstream chain were verified in the explicitly selected store."
                )
            else:
                rows.append(
                    "<br>Historical checkpoint digest: <code>"
                    f"{esc(record.checkpoint_sha256)}</code>. This digest alone is not "
                    "evidence that artifact pixels are present or verified."
                )
        else:
            rows.append("<br>No executed checkpoint is recorded for this stage.")
        if inspection is None or inspection.status is InspectionStatus.STORE_UNBOUND:
            inspection_text = (
                "Checkpoint arrays: UNAVAILABLE until a checkpoint store is explicitly selected. "
                "Project graph metadata does not embed, restore, or verify checkpoint pixels. "
                "No output preview is available in this dialog."
            )
        elif inspection.status is InspectionStatus.VERIFIED:
            inspection_text = (
                "Checkpoint arrays: VERIFIED from the explicitly selected store. Source and frame "
                "identity are bound to the artifact chain; the active desktop image was not "
                "independently matched, and no overlay with native outputs is shown."
            )
        else:
            inspection_text = (
                f"Checkpoint arrays: {esc(inspection.status.value.upper())}. "
                f"{esc(inspection.detail)} No output preview is available in this dialog."
            )
        rows.append(f"</p><p><b>{inspection_text}</b></p>")
        if "watershed" in stage.operation_id or "watershed" in stage.stage_id:
            if recipe.provenance.mode is RecipeProvenanceMode.NATIVE_COMPATIBLE:
                scope = (
                    "Native stage artifacts are inspected only through their exact native verifier. "
                    "The original public PlantCV call does not expose these internal checkpoints. "
                    "This dialog does not calculate or reconstruct intermediate pixels."
                )
            else:
                scope = (
                    "These nodes are source-inspection facts about internals of one public "
                    "PlantCV watershed call. They are not independent original replay calls, "
                    "and no intermediate watershed images are reconstructed here."
                )
            rows.append(
                "<h3>Watershed scope</h3><p>"+scope+"</p>"
            )
        return "".join(rows)


__all__ = ["ReferenceGraphDialog"]
