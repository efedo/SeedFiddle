"""Read-only Qt view of the frozen PlantCV and SMPTS reference graph slice.

This dialog projects the existing recipe contracts into the existing pipeline
canvas. It does not execute operations, synthesize intermediate images, or
claim checkpoint arrays are present in project metadata.
"""

from __future__ import annotations

from dataclasses import replace
import html
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from seedvision.benchmarking.recipe_catalogue import (
    RecipeCatalogue,
    build_recipe_catalogue,
)
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
from seedvision.pipeline import NodeStatus, PipelineNode
from seedvision.ui.pipeline_canvas import PipelineCanvas
from seedvision.ui.pipeline_inspector import PipelineInspector


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
    No image or checkpoint store is read by this view.
    """

    reference_graphs_changed = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        reference_graphs: tuple[ReferenceGraphAttachment, ...] = (),
        catalogue: RecipeCatalogue | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("PlantCV and SMPTS reference graphs")
        self.setMinimumSize(1050, 650)
        self.resize(1440, 900)

        self.catalogue = catalogue or build_recipe_catalogue()
        self._catalogue_by_id = {
            recipe.recipe_id: recipe for recipe in self.catalogue.recipes
        }
        self._reference_graphs = validate_reference_graphs(reference_graphs)
        self._active_attachment: ReferenceGraphAttachment | None = None
        self._active_recipe: RecipeSpec | None = None
        self._active_entry_kind: str | None = None
        self._active_is_transient = False
        self._transient_reason: str | None = None
        self._canvas: PipelineCanvas | None = None
        self._inspector: PipelineInspector | None = None
        self._details_panel: QWidget | None = None
        self._updating_controls = False

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

        self.graph_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.graph_splitter.setObjectName("referenceGraphSplitter")
        self.graph_splitter.setChildrenCollapsible(False)
        layout.addWidget(self.graph_splitter, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self.recipe_selector.currentIndexChanged.connect(self._recipe_selected)
        initial_index = self._initial_recipe_index()
        self.recipe_selector.setCurrentIndex(initial_index)
        self._recipe_selected(initial_index)

    @property
    def reference_graphs(self) -> tuple[ReferenceGraphAttachment, ...]:
        """Return the validated immutable metadata tuple for project saving."""

        return self._reference_graphs

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

        details_panel = QWidget(self.graph_splitter)
        details_layout = QVBoxLayout(details_panel)
        details_layout.setContentsMargins(8, 0, 0, 0)
        details_layout.addWidget(inspector, 1)
        details_layout.addWidget(facts, 4)
        self.graph_splitter.addWidget(canvas)
        self.graph_splitter.addWidget(details_panel)
        self.graph_splitter.setStretchFactor(0, 3)
        self.graph_splitter.setStretchFactor(1, 2)

        self._canvas = canvas
        self._inspector = inspector
        self._facts_browser = facts
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
            and same_identity
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
        claim += " This view does not execute operations or provide reference output pixels."
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
        self._facts_browser.setHtml(self._stage_facts_html(recipe, stage, record))

    @staticmethod
    def _stage_facts_html(
        recipe: RecipeSpec,
        stage,
        record: StageRunRecord,
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
            rows.append(
                "<p>These are recipe-contract values. No operation has run here, "
                "so effective runtime parameter values are not recorded.</p><ul>"
            )
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
            rows.append(
                "<br>Historical checkpoint digest: <code>"
                f"{esc(record.checkpoint_sha256)}</code>. The digest is metadata only; "
                "its raster arrays are unavailable here."
            )
        else:
            rows.append("<br>No executed checkpoint is recorded for this stage.")
        rows.append(
            "</p><p><b>Checkpoint arrays: UNAVAILABLE.</b> Project graph metadata "
            "does not embed, restore, or verify checkpoint pixels. No output preview "
            "is available in this dialog.</p>"
        )
        if "watershed" in stage.operation_id or "watershed" in stage.stage_id:
            rows.append(
                "<h3>Watershed scope</h3><p>These nodes are source-inspection facts "
                "about internals of one public PlantCV watershed call. They are not "
                "independent replay calls, and no intermediate watershed images are "
                "reconstructed here.</p>"
            )
        return "".join(rows)


__all__ = ["ReferenceGraphDialog"]
