"""Results, review decisions and immutable export, independent of node inspection."""
from dataclasses import asdict
import json
from PySide6.QtCore import Qt, QObject, QRunnable, Signal, QTimer, QByteArray, QPoint, QSignalBlocker
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QComboBox, QLineEdit, QPushButton, QFileDialog, QMessageBox, QProgressDialog, QInputDialog)
from seedvision.export.results import result_report, export_report
from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph, analysis_settings_profile_to_payload


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


class ResultsController:
    def _init_results(self):
        self._result_export_task = None
        self._image_review_states = {}
        self.results_action = QAction('Review / export results…',self)
        self.results_action.setShortcut('Ctrl+R')
        self.results_action.triggered.connect(self._review_results)
        self.addAction(self.results_action)
        for action in self._optimization_menu_actions:
            if action.text() == '&Analysis':
                action.menu().addAction(self.results_action)
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
                             ('Restore portable snapshot…',self._restore_snapshot)):
            action = QAction(title,self)
            action.triggered.connect(method)
            self.addAction(action)
            self._optimization_menu_actions[0].menu().addAction(action)

    def _create_snapshot(self):
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
        from seedvision.persistence.snapshot import create_snapshot
        from seedvision.ui.persistence_jobs import run_persistence
        try:
            path = run_persistence(self,lambda:create_snapshot(self._root,self._current_project_path,directory,
                library_store=self._species_library_service.store),'Copying and verifying immutable project snapshot…')
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
        settings = analysis_settings_profile_to_payload(analysis_settings_profile_from_graph(self.pipeline))
        report = result_report(result, enabled={identifier for identifier,node in self.pipeline.nodes.items() if node.enabled},
            settings=settings, provenance=provenance)
        path = self._review_path(report)
        if path.exists():
            review = json.loads(path.read_text(encoding='utf-8'))
            report = result_report(result, enabled={identifier for identifier,node in self.pipeline.nodes.items() if node.enabled},
                settings=settings, provenance=provenance,review=review)
        return report

    def _review_path(self, report):
        return self._root / 'result-reviews' / (report['result_revision']+'.json')

    def _review_results(self):
        if self._background_work_is_active():
            QMessageBox.information(self,'Results','Wait for the active calculation to finish or use Stop.')
            return
        result = self._analyses.get(self._current_image_key())
        if result is None:
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
            summary.setText(report['unavailable_reason'] or
                f"{report['method']}: {len(rows)} proposed; {sum(r['review']=='accepted' for r in rows)} accepted; "
                f"{sum(r['review']=='excluded' for r in rows)} excluded. Measurements: visible 2-D pixels. "
                'Physical units and trait assignments withheld pending validation. Scores are uncalibrated.')
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
        compare = QPushButton('Compare completed instance methods…',dialog)
        def compare_methods():
            from seedvision.export.results import select_result
            import numpy as np
            lines = []
            for name in ('procedural_instances','unet_instances','stardist_instances'):
                _, instances, reason = select_result(result,{name})
                count = None if instances is None else len(np.unique(np.asarray(instances.labels)[np.asarray(instances.labels)>0]))
                lines.append(f'{name}: {reason if count is None else str(count)+" proposed instances"}')
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
        table.setHorizontalHeaderLabels(('Seed ID','Decision','Area px²','Major px','Minor px','Score','Traits'))
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(table)
        editors = []
        for index,row in enumerate(report['instances']):
            values = (row['seed_id'],None,row['visible_area_px2'],row['visible_major_extent_px'],
                row['visible_minor_extent_px'],row['uncalibrated_instance_score'],'Abstained')
            for column,value in enumerate(values):
                if value is not None:
                    table.setItem(index,column,QTableWidgetItem(f'{value:.3f}' if isinstance(value,float) else str(value)))
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
        dialog.exec()
