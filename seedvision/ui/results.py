"""Results, review decisions and immutable export, independent of node inspection."""
from dataclasses import asdict
from html import escape
import json
from PySide6.QtCore import Qt, QObject, QRunnable, Signal, QTimer, QByteArray, QPoint, QSignalBlocker
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QComboBox, QLineEdit, QPushButton, QFileDialog, QMessageBox, QProgressDialog, QInputDialog, QPlainTextEdit)
from seedvision.export.results import result_report, export_report
from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph, analysis_settings_profile_to_payload


def _trait_presentation(row, report):
    """Describe the stored prediction without consulting or changing a model."""
    assignment = row.get('trait_assignment')
    if assignment is None:
        return 'Unavailable', 'No stored trait assignment.'
    text = 'Abstained: ' + assignment[len('abstained: '):].replace('_', ' ') \
        if assignment.startswith('abstained: ') else assignment
    details = [assignment]
    prediction = row.get('trait_prediction')
    if prediction is not None:
        if prediction['confidence'] is not None:
            details.append(f"Maximum predictive probability: {prediction['confidence']:.3f}. "
                           'This is not a probability of correctness.')
        classifier = report.get('trait_classifier') or {}
        model = classifier.get('frozen_model') or {}
        if not prediction['abstained']:
            definitions = model.get('vocabulary', {}).get('traits', ())
            definition = next((item for item in definitions
                               if item['identifier'] == prediction['predicted_trait_id']), None)
            details.append('Supplied definition: ' + definition['definition'] if definition
                           else 'Definition unavailable.')
            if definition:
                details.append('Definition evidence: ' + definition['evidence_reference'])
        if prediction.get('model_sha256'):
            details.append('Model SHA256: ' + prediction['model_sha256'])
        details.append('Biological accuracy has not been independently validated.')
    return text, '\n'.join(details)


class _ExportSignals(QObject):
    done = Signal(object, str)


class _ExportTask(QRunnable):
    def __init__(self, directory, result, report):
        super().__init__()
        self.signals = _ExportSignals()
        self.directory, self.result, self.report = directory, result, report

    def run(self):
        try:
            self.signals.done.emit(export_report(self.directory,self.result,self.report), '')
        except Exception as error:
            self.signals.done.emit(None,str(error))


class _CropExportTask(QRunnable):
    def __init__(self, directory, result, report, seed_ids, padding_px):
        super().__init__()
        from threading import Event
        self.signals = _ExportSignals()
        self.directory, self.result, self.report = directory, result, report
        self.seed_ids, self.padding_px = tuple(seed_ids), padding_px
        self.cancel_event = Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        from seedvision.export.crop_artifacts import export_seed_crop_artifacts
        try:
            path = export_seed_crop_artifacts(self.directory, self.result, self.report,
                seed_ids=self.seed_ids, padding_px=self.padding_px, cancel=self.cancel_event)
            payload = json.loads((path / 'seed-crops.json').read_text(encoding='utf-8'))
            summary = {'path': path, 'available': sum(record['status'] == 'available' for record in payload['records']),
                       'incomplete': sum(record['status'] != 'available' for record in payload['records'])}
            self.signals.done.emit(summary, '')
        except Exception as error:
            self.signals.done.emit(None, str(error))


class ResultsController:
    def _init_results(self):
        self._result_export_task = None
        self._image_review_states = {}
        self._image_run_outcomes = {}
        self._run_outcome_history = {}
        self._selected_history_outcome = None
        self.results_action = QAction('Review / export results…',self)
        self.results_action.setShortcut('Ctrl+R')
        self.results_action.triggered.connect(self._review_results)
        self.addAction(self.results_action)
        self.terminal_history_action = QAction('Review terminal run history…', self)
        self.terminal_history_action.triggered.connect(self._review_terminal_history)
        self.addAction(self.terminal_history_action)
        for action in self._optimization_menu_actions:
            if action.text() == '&Analysis':
                action.menu().addAction(self.results_action)
                action.menu().addAction(self.terminal_history_action)
        button = QPushButton('Review / export results…',self)
        button.clicked.connect(self._review_results)
        self.baseline_section.layout().addWidget(button)
        self.reference_readiness_button = QPushButton('Supply Foreground references',self)
        self.reference_readiness_button.clicked.connect(lambda: self._pipeline_node_selected('background_likelihood'))
        self.baseline_section.layout().addWidget(self.reference_readiness_button)
        self.workflow_toolbar.addAction(self.results_action)
        self.workflow_toolbar.addAction(self.stop_action)
        self.search_node_action = QAction('Find pipeline node…',self)
        self.search_node_action.setShortcut('Ctrl+K')
        self.search_node_action.triggered.connect(self._find_pipeline_node)
        self.addAction(self.search_node_action)
        self.workflow_toolbar.addAction(self.search_node_action)
        self.redo_reference_action = QAction('Redo annotation edit',self)
        self.redo_reference_action.setShortcut('Ctrl+Shift+Z')
        self.redo_reference_action.triggered.connect(lambda: self._undo_instance_reference_edit(redo=True)
            if self.annotate_instances_action.isChecked() else self._undo_reference_mask_edit(redo=True)
            if self.paint_background_action.isChecked() else None)
        self.addAction(self.redo_reference_action)
        self.workflow_toolbar.addAction(self.redo_reference_action)
        self.image_workspace_action.toggled.connect(self._ensure_workspace_visible)
        self.pipeline_workspace_action.toggled.connect(self._ensure_workspace_visible)
        state = self._application_settings.value('review-ui/main-splitter')
        if isinstance(state,QByteArray):
            self.main_splitter.restoreState(state)
        position = self._application_settings.value('review-ui/annotation-window-position')
        if isinstance(position,QPoint):
            self.image_view._context_panel_saved_window_position = position
        viewport_action = QAction('Save current image viewport…',self)
        viewport_action.setShortcut('Ctrl+Shift+E')
        viewport_action.triggered.connect(self._save_current_viewport)
        self.addAction(viewport_action)
        self._optimization_menu_actions[0].menu().addAction(viewport_action)
        for direction,shortcut in ((1,'Alt+Right'),(-1,'Alt+Left')):
            action = QAction('Next annotated seed' if direction>0 else 'Previous annotated seed',self)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda _checked=False,direction=direction:self._step_annotated_seed(direction))
            self.addAction(action)
        for tool,shortcut in (('brush','Ctrl+Alt+B'),('edge_trace','Ctrl+Alt+T'),('smart_fill','Ctrl+Alt+F')):
            action = QAction(f'Annotation tool: {tool}',self)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda _checked=False,tool=tool:self._instance_tool_selected(tool,True)
                if self.annotate_instances_action.isChecked() else None)
            self.addAction(action)
        for title,method in (('Create immutable portable snapshot…',self._create_snapshot),
                             ('Create snapshot with reference checkpoints…',self._create_snapshot_with_reference_checkpoints),
                             ('Restore portable snapshot…',self._restore_snapshot)):
            action = QAction(title,self)
            action.triggered.connect(method)
            self.addAction(action)
            self._optimization_menu_actions[0].menu().addAction(action)

    def _create_snapshot(self):
        self._create_snapshot_impl()

    def _create_snapshot_with_reference_checkpoints(self):
        self._create_snapshot_impl(include_reference_checkpoints=True)

    def _create_snapshot_impl(self, *, include_reference_checkpoints=False):
        if self._background_work_is_active():
            return
        if not self._resolve_unapplied_project_drafts(recompute=False):
            return
        saved = self._write_project(self._current_project_path) if self._current_project_path else self._save_project_as(drafts_resolved=True)
        if not saved:
            return
        directory = QFileDialog.getExistingDirectory(self,'Choose snapshot destination')
        if not directory:
            return
        checkpoint_store = None
        if include_reference_checkpoints:
            checkpoint_store = QFileDialog.getExistingDirectory(self,'Choose reference checkpoint store to include')
            if not checkpoint_store:
                return
        from seedvision.persistence.snapshot import create_snapshot
        from seedvision.ui.persistence_jobs import run_persistence
        try:
            path = run_persistence(self,lambda:create_snapshot(self._root,self._current_project_path,directory,
                library_store=self._species_library_service.store,
                reference_checkpoint_store=checkpoint_store),'Copying and verifying immutable project snapshot…')
            self.statusBar().showMessage(f'Immutable snapshot created: {path}')
        except (ValueError,OSError) as error:
            QMessageBox.warning(self,'Snapshot failed',str(error))

    def _save_review_ui_state(self):
        self._application_settings.setValue('review-ui/main-splitter',self.main_splitter.saveState())
        if self.image_view._context_panel_user_position is not None:
            position = self.annotation_workspace.mapTo(
                self, self.image_view._context_panel_user_position
            )
            self._application_settings.setValue('review-ui/annotation-window-position',position)

    def _step_annotated_seed(self, direction):
        if not self.annotate_instances_action.isChecked() or self._background_work_is_active():
            return
        ids = sorted(self.image_view.instance_centres())
        if not ids:
            return
        current = self.instance_id_spin.value()
        index = ids.index(current) if current in ids else -1
        self.instance_id_spin.setValue(ids[(index+direction)%len(ids)])

    def _save_current_viewport(self):
        path,_ = QFileDialog.getSaveFileName(self,'Save current image viewport','','PNG image (*.png)')
        if not path:
            return
        image = self.image_view.viewport().grab().toImage()
        from PySide6.QtGui import QImage, QPainter, QColor, QFont
        ratio = image.devicePixelRatio()
        annotated = QImage(image.width(),image.height()+round(88*ratio),QImage.Format.Format_ARGB32)
        annotated.setDevicePixelRatio(ratio)
        annotated.fill(QColor('white'))
        painter = QPainter(annotated)
        painter.drawImage(0,0,image)
        painter.setPen(QColor('black'))
        painter.setFont(QFont('Segoe UI',9))
        y = round(image.height()/ratio)
        painter.drawText(12,y+20,'Overlay: '+self.overlay_combo.currentText())
        painter.drawText(12,y+40,'Visible image coordinates. Physical units unvalidated; scores are uncalibrated.')
        zoom = abs(self.image_view.transform().m11())
        pixels = 100
        while pixels*zoom > image.width()/ratio*.4 and pixels > 1:
            pixels //= 10
        if zoom > 0:
            length = max(1,round(pixels*zoom))
            painter.drawLine(12,y+62,12+length,y+62)
            painter.drawLine(12,y+58,12,y+66)
            painter.drawLine(12+length,y+58,12+length,y+66)
            painter.drawText(22+length,y+66,f'{pixels} image px')
        painter.end()
        image = annotated
        from seedvision.ui.persistence_jobs import run_persistence
        if not run_persistence(self,lambda:image.save(path,'PNG'),'Saving current display…'):
            QMessageBox.warning(self,'Save failed','Could not write the viewport image.')

    def _restore_snapshot(self):
        if self._background_work_is_active():
            return
        directory = QFileDialog.getExistingDirectory(self,'Choose snapshot folder')
        if not directory:
            return
        from seedvision.persistence.snapshot import restore_snapshot
        from seedvision.ui.persistence_jobs import run_persistence
        try:
            path = run_persistence(self,lambda:restore_snapshot(directory,self._root,
                library_store=self._species_library_service.store),'Restoring verified working copy…')
            self.statusBar().showMessage(f'Restored working project: {path}. Use File > Open project to open it.')
        except (ValueError,OSError) as error:
            QMessageBox.warning(self,'Restore failed',str(error))

    def _ensure_workspace_visible(self):
        QTimer.singleShot(0,self._restore_empty_workspace)

    def _restore_empty_workspace(self):
        if not self.image_workspace_action.isChecked() and not self.pipeline_workspace_action.isChecked():
            self.image_workspace_action.setChecked(True)

    def _find_pipeline_node(self):
        names = {f'{node.title} ({identifier})':identifier for identifier,node in self.pipeline.nodes.items()
            if identifier in self.pipeline_canvas.node_items}
        text, accepted = QInputDialog.getItem(self,'Find pipeline node','Type a node name',sorted(names),0,True)
        if not accepted:
            return
        matches = [name for name in names if text.lower() in name.lower()]
        if text in names:
            matches = [text]
        if not matches:
            return
        identifier = names[matches[0]]
        self._show_pipeline_workspace()
        self.pipeline_canvas.select_node(identifier)
        neighbours = {identifier,*self.pipeline.upstream(identifier),*self.pipeline.downstream(identifier)}
        bounds = self.pipeline_canvas.node_items[identifier].sceneBoundingRect()
        for neighbour in neighbours:
            if neighbour in self.pipeline_canvas.node_items:
                bounds = bounds.united(self.pipeline_canvas.node_items[neighbour].sceneBoundingRect())
        self.pipeline_canvas.fitInView(bounds.adjusted(-30,-30,30,30),Qt.AspectRatioMode.KeepAspectRatio)

    def _current_report(self, result):
        key = self._current_image_key()
        if self._cache_dirty_nodes.get(key):
            raise ValueError('Pipeline inputs changed. Recompute the image before reviewing or exporting this result.')
        provenance = {}
        if getattr(result, 'analysis_inputs_json', None) is None:
            # Legacy/synthetic results have no captured inputs. Do not invent history.
            from seedvision.optimization.runtime import array_digest
            annotation_arrays = {name: None if getattr(self, attr).get(key) is None else array_digest(getattr(self, attr).get(key)) for name,attr in (
                ('instances','_applied_instance_annotations'), ('foreground','_applied_foreground_reference_masks'),
                ('background','_applied_background_reference_masks'))}
            provenance = {'annotations': annotation_arrays, 'species': self.species_combo.currentText(),
                'annotation_metadata': [asdict(item) for item in self._applied_seed_annotations.get(key,{}).values()],
                'analysis_mode': 'image_local_adaptation', 'biological_context': asdict(self._effective_biological_context()),
                'species_library': None if self._project_species_library_pin is None else asdict(self._project_species_library_pin)}
            from pathlib import Path
            from seedvision.learning.data import file_sha256
            provenance['checkpoints'] = {}
            for identifier in ('unet_instances','stardist_instances'):
                if self.pipeline.node(identifier).enabled:
                    path = Path(self.pipeline.node(identifier).parameters['checkpoint_path'])
                    path = path if path.is_absolute() else self._root/path
                    provenance['checkpoints'][identifier] = file_sha256(path) if path.is_file() else 'unavailable'
        from seedvision.learning.data import file_sha256
        project = self._current_project_path
        reference = self._project_reference_sidecar_paths.get(key)
        execution = {'route': 'desktop',
            'project_sha256': file_sha256(project) if project is not None and project.is_file() else None,
            'reference_sha256': file_sha256(reference) if reference is not None and reference.is_file() else None,
            'library': None if self._project_species_library_pin is None else asdict(self._project_species_library_pin)}
        settings = analysis_settings_profile_to_payload(analysis_settings_profile_from_graph(self.pipeline))
        report = result_report(result, enabled={identifier for identifier,node in self.pipeline.nodes.items() if node.enabled},
            settings=settings, provenance=provenance, execution_provenance=execution)
        from seedvision.export.review import lookup_result_review
        review, review_state = lookup_result_review(self._root, report['result_revision'], project_path=project)
        if review is not None:
            report = result_report(result, enabled={identifier for identifier,node in self.pipeline.nodes.items() if node.enabled},
                settings=settings, provenance=provenance,review=review, execution_provenance=execution)
        report['review_lookup'] = review_state
        return report

    def _review_path(self, report):
        return self._root / 'result-reviews' / (report['result_revision']+'.json')

    def _review_run_outcome(self, outcome):
        from seedvision.export.run_outcomes import export_run_outcome, acknowledge_run_outcome
        from seedvision.persistence.terminal_history import load_project_acknowledgments, load_acknowledgment
        from seedvision.ui.persistence_jobs import run_persistence
        payload = outcome.payload
        dialog = QDialog(self)
        dialog.setWindowTitle('Review terminal analysis record')
        dialog.resize(720, 360)
        layout = QVBoxLayout(dialog)
        summary = QLabel(f"Run {payload['identity']['state']}: "
                         + payload['identity']['reason_code'].replace('_', ' ') + '\n'
                         + ('Preparation phase: ' + payload['identity']['phase'].replace('_', ' ') + '\n'
                            if 'phase' in payload['identity'] else '')
                         + (payload['diagnostic'] or '') + '\n\n'
                         'No authoritative prediction or calibration was captured. '
                         'Acknowledgment records that you reviewed this outcome; it cannot approve seeds.', dialog)
        summary.setWordWrap(True)
        layout.addWidget(summary)
        saved = run_persistence(self, lambda: load_project_acknowledgments(
            self._root, self._current_project_path), 'Reading terminal acknowledgments…')
        matching = [load_acknowledgment(raw) for raw in saved.records
                    if load_acknowledgment(raw)['outcome_revision'] == payload['outcome_revision']]
        if matching or saved.diagnostics:
            text = '\n'.join('Recorded acknowledgment: ' + record['operator_account'] + ' at '
                + record['reviewed_at_utc'] for record in matching[:8])
            if len(matching) > 8:
                text += f'\n{len(matching) - 8} additional recorded acknowledgments.'
            if saved.diagnostics:
                text += '\nAcknowledgment history has diagnostics: ' + '; '.join(saved.diagnostics[:3])
            note = QPlainTextEdit(dialog)
            note.setReadOnly(True)
            note.setMaximumHeight(100)
            note.setPlainText(text)
            layout.addWidget(note)
        acknowledge = QPushButton('Acknowledge this terminal outcome', dialog)
        layout.addWidget(acknowledge)

        def save_acknowledgment():
            import getpass
            try:
                run_persistence(self, lambda: acknowledge_run_outcome(
                    self._root / 'artifacts' / 'run-outcome-acknowledgments', outcome, getpass.getuser()),
                    'Saving exact outcome acknowledgment…')
            except (ValueError, OSError) as error:
                QMessageBox.warning(dialog, 'Acknowledgment failed', str(error))
                return
            acknowledge.setText('Acknowledged for this exact outcome revision')

        acknowledge.clicked.connect(save_acknowledgment)
        export = QPushButton('Export terminal JSON record…', dialog)
        layout.addWidget(export)

        def export_record():
            directory = QFileDialog.getExistingDirectory(dialog, 'Export terminal analysis record')
            if not directory:
                return
            try:
                path = run_persistence(self, lambda: export_run_outcome(directory, outcome),
                                       'Saving terminal run evidence…')
                self.statusBar().showMessage(f'Terminal analysis record saved: {path}')
            except (ValueError, OSError) as error:
                QMessageBox.warning(dialog, 'Export failed', str(error))

        export.clicked.connect(export_record)
        dialog.exec()

    def _open_terminal_record(self):
        from seedvision.export.run_outcome_history import load_terminal_record
        from seedvision.ui.persistence_jobs import run_persistence
        selected, _ = QFileDialog.getOpenFileName(self, 'Open terminal run record',
            str(self._root / 'artifacts' / 'run-outcomes'), 'Terminal JSON (*.json)')
        if not selected:
            return
        try:
            outcome = run_persistence(self, lambda: load_terminal_record(selected), 'Reading terminal run record…')
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, 'Terminal record unavailable', str(error))
            return
        self._selected_history_outcome = outcome
        self._review_run_outcome(outcome)

    def _review_terminal_history(self):
        from seedvision.persistence.terminal_history import load_project_terminal_history
        from seedvision.ui.persistence_jobs import run_persistence
        history = run_persistence(self, lambda: load_project_terminal_history(
            self._root, self._current_project_path, live=tuple(self._run_outcome_history.values())),
            'Reading terminal run history…')
        outcomes = history.outcomes
        dialog = QDialog(self)
        dialog.setWindowTitle('Terminal run history')
        dialog.resize(850, 450)
        layout = QVBoxLayout(dialog)
        label = QLabel('These records contain observations made during earlier attempts. '
            'Reviewing them does not change the current analysis or approve seeds.', dialog)
        label.setWordWrap(True)
        layout.addWidget(label)
        table = QTableWidget(len(outcomes), 5, dialog)
        table.setHorizontalHeaderLabels(('Subject', 'Preparation / last node', 'Outcome', 'File', 'Revision'))
        for row, outcome in enumerate(outcomes):
            payload = outcome.payload
            identity = payload['identity']
            path = outcome.image_path or outcome.subject_path
            values = (identity.get('subject_kind', 'image').replace('_', ' '),
                (identity.get('phase') or identity.get('last_started_node') or 'analysis').replace('_', ' '),
                identity['state'], path.name, payload['outcome_revision'][:16])
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(str(path) if column == 3 else payload['outcome_revision'] if column == 4 else value)
                table.setItem(row, column, item)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.resizeColumnsToContents()
        layout.addWidget(table)
        diagnostics = history.diagnostics
        if diagnostics:
            note = QPlainTextEdit(dialog)
            note.setReadOnly(True)
            note.setMaximumHeight(100)
            note.setPlainText('\n'.join(diagnostics[:12]) + ('\nFurther diagnostics omitted.' if len(diagnostics) > 12 else ''))
            layout.addWidget(note)
        review = QPushButton('Review selected terminal record…', dialog)
        review.setEnabled(False)
        table.itemSelectionChanged.connect(lambda: review.setEnabled(table.currentRow() >= 0))
        def selected_record():
            row = table.currentRow()
            if row < 0:
                return
            self._selected_history_outcome = outcomes[row]
            self._review_run_outcome(outcomes[row])
        review.clicked.connect(selected_record)
        layout.addWidget(review)
        open_file = QPushButton('Open one terminal JSON record…', dialog)
        open_file.clicked.connect(self._open_terminal_record)
        layout.addWidget(open_file)
        dialog.exec()

    def _review_results(self):
        if self._background_work_is_active():
            QMessageBox.information(self,'Results','Wait for the active calculation to finish or use Stop.')
            return
        result = self._analyses.get(self._current_image_key())
        if result is None:
            terminal = self._image_run_outcomes.get(self._current_image_key())
            if terminal is not None and terminal[0] == self.pipeline.revision:
                self._review_run_outcome(terminal[1])
                return
            QMessageBox.information(self,'Results','Analyze the current image first.')
            return
        try:
            report = self._current_report(result)
        except (ValueError,OSError) as error:
            QMessageBox.warning(self,'Results unavailable',str(error))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Review visible seed instances')
        dialog.resize(960,640)
        layout = QVBoxLayout(dialog)
        summary = QLabel(dialog)
        summary.setWordWrap(True)
        layout.addWidget(summary)
        def refresh_summary():
            rows = report['instances']
            assigned = sum(row.get('trait_assignment') is not None
                           and not row['trait_assignment'].startswith('abstained: ') for row in rows)
            abstained = sum((row.get('trait_assignment') or '').startswith('abstained: ') for row in rows)
            unavailable_traits = len(rows) - assigned - abstained
            text = report['unavailable_reason'] or (
                f"{report['method']}: {len(rows)} proposed; {sum(r['review']=='accepted' for r in rows)} accepted; "
                f"{sum(r['review']=='excluded' for r in rows)} excluded. Measurements: visible 2-D pixels. "
                f'Trait predictions: {assigned} assigned, {abstained} abstained, {unavailable_traits} unavailable. '
                'Physical units remain withheld pending validation. Instance scores are uncalibrated; '
                'trait predictions have no independent biological validation.')
            from seedvision.export.backend_outcomes import outcome_descriptions
            outcomes = report['backend_outcomes']
            if outcomes['outcomes'] is not None and any(record['state'] in {'failed', 'unavailable'}
                                                       for record in outcomes['outcomes'].values()):
                text += '\n' + '\n'.join(outcome_descriptions(outcomes))
            summary.setText(text)
        refresh_summary()
        search = QLineEdit(dialog)
        search.setPlaceholderText('Find seed ID…')
        layout.addWidget(search)
        review_filter = QComboBox(dialog)
        review_filter.addItems(('All decisions','unreviewed','accepted','excluded','reviewed'))
        review_filter.setAccessibleName('Filter seeds by review decision')
        layout.addWidget(review_filter)
        next_seed = QPushButton('Next matching seed (Alt+N)',dialog)
        next_seed.setShortcut('Alt+N')
        layout.addWidget(next_seed)
        compare = QPushButton('Compare completed instance methods / outcomes…',dialog)
        def compare_methods():
            from seedvision.export.backend_outcomes import outcome_descriptions
            lines = outcome_descriptions(report['backend_outcomes'])
            if report['backend_outcomes']['outcomes'] is None and report['method'] is not None:
                lines.append(f"{report['method']}: {report['proposed_count']} proposed instances in the stored report; "
                             'producer execution status was not recorded.')
            QMessageBox.information(dialog,'Method comparison','\n\n'.join(lines)+
                '\n\nCounts are unreviewed proposals. Compare boundaries using each method overlay; agreement alone does not establish accuracy. Enable and run a missing method before comparing it.')
        compare.clicked.connect(compare_methods)
        layout.addWidget(compare)
        alternatives = QPushButton('Inspect unselected procedural candidates in the image',dialog)
        alternatives.clicked.connect(lambda: (dialog.accept(), self._show_image_workspace(),
            self.overlay_combo.setCurrentIndex(self.overlay_combo.findData('procedural_alternative_candidates'))))
        alternatives.setEnabled(report['method']=='procedural_instances')
        layout.addWidget(alternatives)
        table = QTableWidget(len(report['instances']),7,dialog)
        table.setHorizontalHeaderLabels(('Seed ID','Decision','Area px²','Major px','Minor px','Score','Predicted trait'))
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(table)
        editors = []
        for index,row in enumerate(report['instances']):
            trait_text, trait_details = _trait_presentation(row, report)
            values = (row['seed_id'],None,row['visible_area_px2'],row['visible_major_extent_px'],
                row['visible_minor_extent_px'],row['uncalibrated_instance_score'],trait_text)
            for column,value in enumerate(values):
                if value is not None:
                    table.setItem(index,column,QTableWidgetItem(f'{value:.3f}' if isinstance(value,float) else str(value)))
            table.item(index,6).setToolTip('<p>' + escape(trait_details).replace('\n', '<br>') + '</p>')
            choice = QComboBox(table)
            choice.addItems(('unreviewed','accepted','excluded'))
            choice.setCurrentText(row['review'])
            def changed(value, row=row, choice=choice):
                previous = row['review']
                row['review'] = value
                if not save_review():
                    row['review'] = previous
                    with QSignalBlocker(choice):
                        choice.setCurrentText(previous)
                refresh_summary()
                filter_rows()
            choice.currentTextChanged.connect(changed)
            table.setCellWidget(index,1,choice)
            editors.append(choice)
        def filter_rows():
            decision = review_filter.currentText()
            for i,row in enumerate(report['instances']):
                matches = decision == 'All decisions' or row['review'] == decision or (decision == 'reviewed' and row['review'] != 'unreviewed')
                table.setRowHidden(i,not matches or search.text().strip() not in str(row['seed_id']))
        search.textChanged.connect(filter_rows)
        review_filter.currentTextChanged.connect(filter_rows)
        def advance_seed():
            start = table.currentRow()
            for offset in range(1,table.rowCount()+1):
                row = (start+offset)%table.rowCount()
                if not table.isRowHidden(row):
                    table.selectRow(row)
                    table.scrollToItem(table.item(row,0))
                    editors[row].setFocus()
                    break
        next_seed.clicked.connect(advance_seed)
        table.resizeColumnsToContents()
        def inspect_seed(index, _column):
            row = report['instances'][index]
            dialog.accept()
            self._show_image_workspace()
            overlay = self.overlay_combo.findData(report['method'])
            if overlay >= 0:
                self.overlay_combo.setCurrentIndex(overlay)
            if report['method']=='procedural_instances':
                self.image_view._selected_procedural_label = row['seed_id']
                self.image_view._render_analysis()
            self.image_view.centerOn(row['centre_x_px'],row['centre_y_px'])
        table.cellDoubleClicked.connect(inspect_seed)
        table.setToolTip('Double-click a seed to inspect it in the image. Each review decision is saved atomically for this result revision.')
        save = QPushButton('Save review decisions',dialog)
        layout.addWidget(save)
        def save_review():
            from datetime import datetime, timezone
            from uuid import uuid4
            import getpass
            path = self._review_path(report)
            path.parent.mkdir(parents=True,exist_ok=True)
            payload = {'result_revision': report['result_revision'],
                'reviewed_at_utc': datetime.now(timezone.utc).isoformat(),
                'operator_account': getpass.getuser(),
                'decisions': {str(row['seed_id']):row['review'] for row in report['instances']}}
            temporary = path.with_suffix('.tmp')
            try:
                history = path.parent/'history'/report['result_revision']
                history.mkdir(parents=True,exist_ok=True)
                (history/(uuid4().hex+'.json')).write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
                temporary.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
                temporary.replace(path)
            except OSError as error:
                QMessageBox.warning(dialog,'Review save failed',str(error))
                return False
            save.setText('Review saved for this exact result revision')
            self._image_review_states[self._current_image_key()] = (
                f"{sum(row['review'] != 'unreviewed' for row in report['instances'])}/{len(report['instances'])} reviewed")
            self._update_analysis_availability()
            return True
        save.clicked.connect(save_review)
        export = QPushButton('Export JSON, CSV and annotated image…',dialog)
        layout.addWidget(export)
        def start_export():
            directory = QFileDialog.getExistingDirectory(dialog,'Export result revision')
            if not directory:
                return
            for state in ('accepted','excluded','unreviewed'):
                report[state+'_count'] = sum(row['review']==state for row in report['instances'])
            task = _ExportTask(directory,result,json.loads(json.dumps(report)))
            progress = QProgressDialog('Writing result revision…','',0,0,self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            self._result_export_task = task
            def finished(path,error):
                self._result_export_task = None
                progress.close()
                if error:
                    QMessageBox.warning(self,'Export failed',error)
                else:
                    self.statusBar().showMessage(f'Result revision exported to {path}')
            task.signals.done.connect(finished)
            dialog.accept()
            self._thread_pool.start(task)
        export.clicked.connect(start_export)
        crops = QPushButton('Export crops for visible rows…', dialog)
        crops.setAccessibleName('Export seed image and mask crops')
        crops.setToolTip('Export the seed rows currently visible in this table, with image, isolated mask and coordinate metadata.')
        crops.setEnabled(bool(report['instances']) and report['state'] == 'available_for_review')
        layout.addWidget(crops)
        def start_crop_export():
            identifiers = tuple(row['seed_id'] for index, row in enumerate(report['instances'])
                                if not table.isRowHidden(index))
            if not identifiers:
                QMessageBox.information(dialog, 'Seed crops', 'No seed rows match the current filters.')
                return
            padding, accepted = QInputDialog.getInt(dialog, 'Seed crop padding',
                'Padding around each visible seed (image pixels):', 0, 0, 4096)
            if not accepted:
                return
            directory = QFileDialog.getExistingDirectory(dialog, 'Export seed crops')
            if not directory:
                return
            try:
                current = self._current_report(result)
                if current['result_revision'] != report['result_revision']:
                    raise ValueError('Result changed. Reopen its review before exporting crops.')
            except (ValueError, OSError) as error:
                QMessageBox.warning(dialog, 'Seed crops unavailable', str(error))
                return
            task = _CropExportTask(directory, result, json.loads(json.dumps(current)), identifiers, padding)
            progress = QProgressDialog('Extracting and writing seed crops…', 'Stop', 0, 0, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.canceled.connect(task.cancel)
            self._result_export_task = task
            def finished_crops(summary, error):
                self._result_export_task = None
                progress.close()
                if self._closing_after_work:
                    return
                if error:
                    QMessageBox.warning(self, 'Seed crop export incomplete', error)
                else:
                    path = summary['path']
                    self.statusBar().showMessage(f"{summary['available']} seed crops exported to {path}")
                    if summary['incomplete']:
                        QMessageBox.warning(self, 'Some seed crops are unavailable',
                            f"{summary['available']} crops exported; {summary['incomplete']} requested seeds have no crop. "
                            f'Individual reasons are recorded in {path / "seed-crops.json"}.')
            task.signals.done.connect(finished_crops)
            progress.show()
            dialog.accept()
            self._thread_pool.start(task)
        crops.clicked.connect(start_crop_export)
        dialog.exec()
