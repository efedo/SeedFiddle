"""Native project/node optimization UI and cancellable production worker."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from threading import Event
from datetime import datetime, timezone
from html import escape

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QProgressDialog,
    QSpinBox, QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout)

from seedvision.optimization.registry import capability, OBJECTIVE_DESCRIPTIONS
from seedvision.optimization.runtime import Sample, ProductionEvaluator, load_targets, write_record, eligibility_reason
from seedvision.optimization.search import optimize, apply_report


class FitSignals(QObject):
    progress = Signal(str, int, float)
    completed = Signal(object)
    failed = Signal(str)


class FitTask(QRunnable):
    def __init__(self, graph, samples, selected, passes, budget):
        super().__init__()
        self.graph = deepcopy(graph)
        self.samples = samples
        self.selected = selected
        self.passes = passes
        self.budget = budget
        self.revision = graph.revision
        self.stop = Event()
        self.signals = FitSignals()

    def cancel(self):
        self.stop.set()

    def run(self):
        try:
            evaluator = ProductionEvaluator(self.graph, self.samples, cancelled=self.stop.is_set)
            result = optimize(self.graph, evaluator, selected=self.selected,
                passes=self.passes, maximum_evaluations=self.budget,
                cancelled=self.stop.is_set, progress=self.signals.progress.emit)
        except Exception as error:
            from seedvision.diagnostics import LOGGER
            LOGGER.exception('Project optimization failed')
            self.signals.failed.emit(str(error))
            return
        self.signals.completed.emit(result)


class OptimizationDialog(QDialog):
    def __init__(self, graph, samples, selected, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Optimize from project references')
        self.resize(840, 650)
        layout = QVBoxLayout(self)
        intro = QLabel('In-sample project adaptation — not independent validation.\n'
            'Each node is fitted across the selected images, then downstream nodes use its proposal.\n'
            'Only complete, shape-reviewed instances are eligible. Physical facts, scoring costs,\n'
            'reference-source policies and resolution limits stay fixed. You review the result before applying.')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.images = QTableWidget(len(samples), 3, self)
        self.images.setHorizontalHeaderLabels(['Use image', 'Eligible instances', 'Every seed reviewed'])
        for row, sample in enumerate(samples):
            item = QTableWidgetItem(sample.path.name)
            item.setToolTip(str(sample.path))
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.images.setItem(row, 0, item)
            count = 0 if sample.labels is None else np.count_nonzero(np.unique(sample.labels))
            self.images.setItem(row, 1, QTableWidgetItem(f'{count} eligible; {len(sample.excluded_ids)} excluded'))
            checkbox = QCheckBox()
            checkbox.setToolTip('Declare exhaustive coverage for this image only. Unchecked leaves disjoint predictions unscored.')
            checkbox.setEnabled(count > 0 and not sample.excluded_ids)
            checkbox.setChecked(sample.complete and checkbox.isEnabled())
            self.images.setCellWidget(row, 2, checkbox)
        self.images.horizontalHeader().setStretchLastSection(True)
        self.images.resizeColumnsToContents()
        self.images.setMaximumHeight(150)
        layout.addWidget(self.images)
        self.nodes = QListWidget(self)
        self.nodes.setWordWrap(True)
        for key in graph.topological_order():
            node = graph.node(key)
            contract = capability(node)
            reasons = [eligibility_reason(graph, key, sample) for sample in samples] if contract.searchable else []
            count = sum(not reason for reason in reasons)
            readiness = ('Node disabled.' if not node.enabled else
                         f'{count}/{len(samples)} image(s) eligible' if count else next(iter(reasons), ''))
            item = QListWidgetItem(f'{node.title}: {len(contract.searchable)} parameters; {contract.objective or contract.reason}; {readiness}')
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setToolTip(OBJECTIVE_DESCRIPTIONS.get(contract.objective, contract.reason)
                + '\nEqual weight per image; lower is better. In-sample adaptation.\n\nSearchable controls:\n' + '\n'.join(contract.searchable)
                + '\n\nFixed controls:\n' + '\n'.join(f'{name}: {reason}' for name, reason in contract.fixed))
            available = bool(node.enabled and node.implemented and contract.searchable and count)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | (Qt.ItemFlag.ItemIsUserCheckable if available else Qt.ItemFlag.NoItemFlags))
            item.setCheckState(Qt.CheckState.Checked if available and (selected is None or key in selected) else Qt.CheckState.Unchecked)
            self.nodes.addItem(item)
        layout.addWidget(self.nodes, 1)
        form = QFormLayout()
        self.passes = QSpinBox()
        self.passes.setRange(1, 10)
        self.budget = QSpinBox()
        self.budget.setRange(0, 10000)
        self.budget.setSpecialValueText('Complete sweep of every eligible control')
        form.addRow('Search passes', self.passes)
        form.addRow('Maximum evaluations per node', self.budget)
        layout.addLayout(form)
        note = QLabel('Hover a node for fixed-control reasons. Missing target types are recorded as skipped.\n'
                      'Complete sweeps may be lengthy; cancellation leaves project settings unchanged.')
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('Start optimization')
        blockers = []
        for identifier in ('unet_instances', 'stardist_instances'):
            if identifier in graph.nodes and graph.node(identifier).enabled:
                for sample in samples:
                    configured = Path(graph.node(identifier).parameters['checkpoint_path'])
                    path = configured if configured.is_absolute() else Path(sample.kwargs.get('learning_root', '.')) / configured
                    if not path.is_file():
                        blockers.append(f'{graph.node(identifier).title}: select a compatible checkpoint or disable the branch first.')
                        break
        if blockers:
            note.setText('\n'.join(blockers))
            buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._graph, self._samples = graph, samples
        self.images.itemChanged.connect(self._refresh_node_eligibility)

    def _refresh_node_eligibility(self, *_args):
        samples = [sample for row,sample in enumerate(self._samples)
                   if self.images.item(row,0).checkState() == Qt.CheckState.Checked]
        for row in range(self.nodes.count()):
            item = self.nodes.item(row)
            identifier = item.data(Qt.ItemDataRole.UserRole)
            node = self._graph.node(identifier)
            contract = capability(node)
            count = sum(not eligibility_reason(self._graph,identifier,sample) for sample in samples) if contract.searchable else 0
            available = bool(node.enabled and node.implemented and contract.searchable and count)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | (Qt.ItemFlag.ItemIsUserCheckable if available else Qt.ItemFlag.NoItemFlags))
            if not available:
                item.setCheckState(Qt.CheckState.Unchecked)
            item.setText(f'{node.title}: {len(contract.searchable)} parameters; {contract.objective or contract.reason}; {count}/{len(samples)} image(s) eligible')


class OptimizationController:
    """Mixin keeps optimization orchestration out of the main window's workers."""
    def _copy_optimization_companions(self, destination):
        source = self._current_project_path
        destination = Path(destination)
        if source is None or source.resolve() == destination.resolve():
            return
        for path in source.parent.glob(source.stem + '.optimization*.json'):
            record = json.loads(path.read_text(encoding='utf-8'))
            suffix = path.name[len(source.stem):]
            if suffix == '.optimization-targets.json':
                record = {key: str((source.parent / value).resolve()) for key, value in record.items()}
            write_record(destination.parent / (destination.stem + suffix), record)

    def _init_optimization(self):
        self._optimization_task = None
        self._optimization_progress = None
        self._optimization_targets = {}
        self._optimization_menu_actions = self.menuBar().actions()
        action = next(item for item in self._optimization_menu_actions if item.text() == '&Analysis')
        menu = action.menu()
        self.optimize_project_action = QAction('Optimize eligible nodes from project references…', self)
        self.optimize_project_action.triggered.connect(lambda: self._start_node_optimization())
        menu.addAction(self.optimize_project_action)
        self.import_optimization_targets_action = QAction('Import reviewed optimization targets…', self)
        self.import_optimization_targets_action.triggered.connect(self._import_optimization_targets)
        menu.addAction(self.import_optimization_targets_action)
        self.optimization_history_action = QAction('Optimization history / restore previous settings…', self)
        self.optimization_history_action.triggered.connect(self._show_optimization_history)
        menu.addAction(self.optimization_history_action)

    def _optimization_samples(self):
        samples = []
        target_paths = {}
        policies = {}
        if self._current_project_path is not None:
            index = self._current_project_path.with_suffix('.optimization-targets.json')
            if index.exists():
                target_paths = json.loads(index.read_text(encoding='utf-8'))
            policy_path = self._current_project_path.with_suffix('.optimization-policy.json')
            if policy_path.exists():
                policies = json.loads(policy_path.read_text(encoding='utf-8'))
        for key, path in self._image_paths.items():
            if key in self._instance_annotations_dirty or key in self._reference_masks_dirty:
                continue
            masks = {name: getattr(self, attr).get(key) for name, attr in (
                ('background_reference_mask', '_applied_background_reference_masks'),
                ('foreground_reference_mask', '_applied_foreground_reference_masks'),
                ('background_exclusion_mask', '_applied_background_exclusion_masks'),
                ('foreground_exclusion_mask', '_applied_foreground_exclusion_masks'),
                ('physical_edge_reference_mask', '_applied_physical_edge_reference_masks'),
                ('non_edge_reference_mask', '_applied_non_edge_reference_masks'))}
            labels = self._applied_instance_annotations.get(key)
            arrays = [value for value in (labels, *masks.values()) if value is not None]
            result = self._analyses.get(key)
            from seedvision.optimization.runtime import file_digest
            fingerprint = file_digest(path)
            target_path = target_paths.get(fingerprint)
            if target_path is not None:
                target_path = self._current_project_path.parent / target_path
            if not arrays and result is None and target_path is None:
                continue
            library = self._resolved_species_library_for_current_image(path)
            if library.error:
                raise ValueError(f'{path.name}: {library.error}')
            kwargs = dict(masks, seed_instance_traits=tuple(sorted(self._applied_seed_annotations.get(key, {}).values(), key=lambda item: item.seed_id)),
                reference_transform=self._reference_transforms.get(key),
                seed_trait_species=self._applied_seed_annotation_species.get(key, ''),
                seed_trait_coat_patterns=self._current_seed_trait_vocabulary().coat_patterns,
                seed_trait_conditions=self._seed_trait_catalogue.conditions,
                manual_seed_centres=self._manual_seed_centres_for_analysis(key),
                learning_root=self._root, species=self.species_combo.currentText(),
                biological_context=self._effective_biological_context(), species_library=library.artifact,
                library_shape_bank=None if library.artifact is None else library.artifact.dimensions_shape)
            if arrays:
                shape = tuple(arrays[0].shape)
            elif result is not None:
                shape = tuple(result.calibration.corrected_bgr.shape[:2])
            else:
                from seedvision.optimization.runtime import target_shape
                shape = target_shape(target_path)
            sample = Sample(key, path, kwargs, shape,
                None if result is None else result.calibration.affine_matrix,
                labels=labels, seed_diameter=30 if result is None else result.estimated_seed_diameter_px,
                fingerprint=fingerprint, target_path=target_path)
            sample.custom = self._optimization_targets.get(key, {})
            from seedvision.optimization.runtime import array_digest
            policy = policies.get(fingerprint, {})
            if sample.labels is not None and policy.get('annotations') == array_digest(sample.labels) and not sample.excluded_ids:
                sample.complete = bool(policy.get('complete', False))
            samples.append(sample)
        return samples

    def _start_node_optimization(self, node_id=None):
        if self._background_work_is_active():
            return
        if not self._project_tracking_enabled or self._current_project_path is None:
            QMessageBox.information(self, 'Save a project first', 'Optimization settings and their history belong to a saved project.')
            return
        try:
            samples = self._optimization_samples()
            if not samples:
                raise ValueError('Apply and save reference regions or complete reviewed instances first. Images with unapplied drafts are excluded.')
            selected = None if node_id is None else {node_id}
            dialog = OptimizationDialog(self.pipeline, samples, selected, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            chosen = []
            for row, sample in enumerate(samples):
                if dialog.images.item(row, 0).checkState() == Qt.CheckState.Checked:
                    sample.complete = dialog.images.cellWidget(row, 2).isChecked()
                    chosen.append(sample)
            selected = {dialog.nodes.item(row).data(Qt.ItemDataRole.UserRole)
                for row in range(dialog.nodes.count()) if dialog.nodes.item(row).checkState() == Qt.CheckState.Checked}
            if not chosen or not selected:
                raise ValueError('Select at least one image and one eligible node.')
            from seedvision.optimization.runtime import array_digest
            policy_path = self._current_project_path.with_suffix('.optimization-policy.json')
            policies = json.loads(policy_path.read_text(encoding='utf-8')) if policy_path.exists() else {}
            for sample in chosen:
                policies[sample.fingerprint] = {'annotations': None if sample.labels is None else array_digest(sample.labels), 'complete': sample.complete}
            write_record(policy_path, policies)
            task = FitTask(self.pipeline, chosen, selected, dialog.passes.value(), dialog.budget.value())
            task.annotation_snapshot = self._optimization_annotation_snapshot()
            task.project_path = self._current_project_path
            task.signals.progress.connect(self._optimization_progressed)
            task.signals.completed.connect(self._optimization_completed)
            task.signals.failed.connect(self._optimization_failed)
            progress = QProgressDialog('Preparing production baselines…', 'Cancel', 0, 0, self)
            progress.setWindowTitle('Optimize project nodes')
            progress.setMinimumDuration(0)
            progress.setAutoClose(False)
            progress.canceled.connect(task.cancel)
            self._optimization_task, self._optimization_progress = task, progress
            self._update_analysis_availability()
            self._sync_background_controls()
            progress.show()
            self._thread_pool.start(task)
        except Exception as error:
            QMessageBox.warning(self, 'Optimization unavailable', str(error))

    def _optimization_annotation_snapshot(self):
        from seedvision.optimization.runtime import array_digest
        return [(key, array_digest(value)) for attr in (
            '_applied_instance_annotations', '_applied_background_reference_masks',
            '_applied_foreground_reference_masks', '_applied_background_exclusion_masks',
            '_applied_foreground_exclusion_masks', '_applied_physical_edge_reference_masks',
            '_applied_non_edge_reference_masks') for key, value in sorted(getattr(self, attr).items())] + [
            ('traits', repr(self._applied_seed_annotations)), ('species', self.species_combo.currentText()),
            ('centres', repr(self._manual_seed_centre_states)),
            ('targets', repr({key: {name: array_digest(value) for name, value in targets.items()}
                              for key, targets in self._optimization_targets.items()})),
            ('library', repr(self._project_species_library_pin))]

    def _optimization_progressed(self, node_id, evaluation, loss):
        if self._optimization_progress is not None:
            self._optimization_progress.setLabelText(f'{self.pipeline.node(node_id).title}\nEvaluation {evaluation}; best mean loss {loss:.5f}')

    def _finish_optimization(self):
        task = self._optimization_task
        if self._optimization_progress is not None:
            self._optimization_progress.blockSignals(True)
            self._optimization_progress.close()
        self._optimization_progress = None
        self._optimization_task = None
        self._update_analysis_availability()
        self._sync_background_controls()
        return task

    def _optimization_completed(self, report):
        task = self._optimization_task
        if task is None:
            return
        if self._optimization_progress is not None:
            self._optimization_progress.blockSignals(True)
            self._optimization_progress.close()
            self._optimization_progress = None
        refresh = None
        stale = (task.revision != self.pipeline.revision or task.project_path != self._current_project_path
                 or task.annotation_snapshot != self._optimization_annotation_snapshot())
        record = asdict(report)
        record['disposition'] = 'cancelled' if report.cancelled or task.stop.is_set() else 'stale' if stale else 'proposal'
        journal = task.project_path.with_suffix('.optimization.json')
        archive = task.project_path.with_suffix('.optimization-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
        try:
            write_record(journal, record)
            write_record(archive, record)
            if stale or report.cancelled or task.stop.is_set():
                self.statusBar().showMessage('Optimization discarded: cancelled or project inputs changed. History was saved.')
                return
            rows = sorted(report.nodes, key=lambda row: row.status == 'skipped')
            detail = '\n\n'.join(f'{row.node_id}: {row.status}\n{row.reason}\nBefore: {row.before}\nAfter: {row.after}\nChanges: {row.changes}' for row in rows)
            message = QDialog(self)
            message.resize(900, 700)
            message.setWindowTitle('Review project optimization')
            layout = QVBoxLayout(message)
            from statistics import mean
            before = mean(report.final_before.values()) if report.final_before else None
            after = mean(report.final_after.values()) if report.final_after else None
            regressions = sum(value > report.final_before[key] + 1e-9 for key,value in report.final_after.items() if key in report.final_before)
            summary = QLabel(f'{len(report.changes)} node(s) improved. This is in-sample adaptation.\n'
                f'Final mean instance loss: {before} → {after}. Increased loss on {regressions} image(s).\n\n'
                'Inspect previews and details below. Applying changes shared settings for every project image.')
            summary.setWordWrap(True)
            layout.addWidget(summary)
            browser = QTextBrowser(message)
            html = []
            for preview in report.previews.values():
                html.append(f'<h3>{escape(preview["name"])}</h3><p>Instance overlays (crop view)</p>'
                    '<table><tr><th>Before</th><th>After</th></tr><tr>'
                    f'<td><img width="384" src="data:image/png;base64,{preview["before"]}"/></td>'
                    f'<td><img width="384" src="data:image/png;base64,{preview["after"]}"/></td></tr></table>')
            html.append('<pre style="white-space: pre-wrap">' + escape(detail) + '</pre>')
            browser.setHtml(''.join(html))
            layout.addWidget(browser)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
            buttons.button(QDialogButtonBox.StandardButton.Apply).setEnabled(bool(report.changes))
            buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(message.accept)
            buttons.rejected.connect(message.reject)
            layout.addWidget(buttons)
            if message.exec() != QDialog.DialogCode.Accepted:
                record['disposition'] = 'kept original'
                write_record(journal, record)
                write_record(archive, record)
                return
            # Recheck after review: a modal dialog can still process queued events.
            if task.annotation_snapshot != self._optimization_annotation_snapshot():
                raise ValueError('References changed while reviewing the proposal.')
            from seedvision.optimization.runtime import verify_record_sources
            verify_record_sources(report.provenance)
            affected = apply_report(self.pipeline, report)
            record['disposition'] = 'applied'
            try:
                write_record(journal, record)
                write_record(archive, record)
            except Exception:
                for identifier in report.changes:
                    self.pipeline.set_parameters(identifier, report.original[identifier])
                self.pipeline.invalidate(affected)
                raise
            refresh = affected
        except Exception as error:
            QMessageBox.warning(self, 'Optimization result', str(error))
        finally:
            self._finish_optimization()
            if refresh:
                self._optimization_refresh(refresh)
            self._start_pending_analysis()

    def _optimization_refresh(self, affected):
        self._set_project_dirty()
        for key in set(self._analysis_caches) | set(self._analyses):
            self._cache_dirty_nodes.setdefault(key, set()).update(affected)
        self._analyses.clear()
        self.pipeline_canvas.refresh(affected)
        self.pipeline_inspector.set_node(self.pipeline.node(self._selected_pipeline_node))
        self._analyze_current_image(dirty_nodes=affected)

    def _optimization_failed(self, error):
        task = self._finish_optimization()
        if task is not None and task.project_path is not None:
            path = task.project_path.with_suffix('.optimization-failed-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
            try:
                write_record(path, {'disposition': 'failed', 'error': str(error), 'settings_changed': False})
            except OSError:
                pass
        QMessageBox.warning(self, 'Optimization stopped; settings unchanged', error)
        self._start_pending_analysis()

    def _import_optimization_targets(self):
        if self._background_work_is_active():
            return
        try:
            key = self._current_image_key()
            sample = next((item for item in self._optimization_samples() if item.key == key), None)
            if sample is None or sample.transform is None:
                raise ValueError('Analyze the current image first. Target files must match its source and calibration fingerprints.')
            path, _ = QFileDialog.getOpenFileName(self, 'Reviewed optimization targets', '', 'Reviewed targets (*.npz)')
            if path:
                self._optimization_targets[key] = load_targets(path, sample)
                if self._current_project_path is not None:
                    index = self._current_project_path.with_suffix('.optimization-targets.json')
                    entries = json.loads(index.read_text(encoding='utf-8')) if index.exists() else {}
                    target_path = Path(path).resolve()
                    try:
                        stored = str(target_path.relative_to(self._current_project_path.parent))
                    except ValueError:
                        stored = str(target_path)
                    entries[sample.fingerprint] = stored
                    write_record(index, entries)
                self.statusBar().showMessage(f'Loaded {len(self._optimization_targets[key])} reviewed optimization targets.')
        except Exception as error:
            QMessageBox.warning(self, 'Target import unavailable', str(error))

    def _show_optimization_history(self):
        if self._current_project_path is None or self._background_work_is_active():
            return
        latest = self._current_project_path.with_suffix('.optimization.json')
        selected, _ = QFileDialog.getOpenFileName(self, 'Select optimization record', str(latest),
                                                 'Optimization records (*.optimization*.json)')
        if not selected:
            return
        path = Path(selected)
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            dialog = QMessageBox(self)
            dialog.setWindowTitle('Optimization record')
            dialog.setText(f"Run: {record['started']}\n{record['mode']}\nStatus: {record['disposition']}\nRestore the settings from before this applied run?")
            dialog.setDetailedText(json.dumps(record, indent=2))
            dialog.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            dialog.setDefaultButton(QMessageBox.StandardButton.No)
            if dialog.exec() != QMessageBox.StandardButton.Yes or record['disposition'] != 'applied':
                return
            from seedvision.optimization.registry import affected_nodes
            proposed = deepcopy(self.pipeline)
            affected = set()
            for row in record['nodes']:
                if not row['changes']:
                    continue
                node_id = row['node_id']
                expected = dict(record['original'][node_id], **row['changes'])
                if proposed.node(node_id).parameters != expected:
                    raise ValueError('Settings changed after this fit; automatic rollback would overwrite later edits.')
                proposed.set_parameters(node_id, record['original'][node_id])
                affected.update(affected_nodes(proposed, node_id))
            previous = {row['node_id']: dict(self.pipeline.node(row['node_id']).parameters)
                        for row in record['nodes'] if row['changes']}
            for row in record['nodes']:
                if row['changes']:
                    self.pipeline.set_parameters(row['node_id'], record['original'][row['node_id']])
            self.pipeline.invalidate(affected)
            record['disposition'] = 'restored'
            try:
                write_record(path, record)
            except Exception:
                for node_id, parameters in previous.items():
                    self.pipeline.set_parameters(node_id, parameters)
                self.pipeline.invalidate(affected)
                raise
            self._optimization_refresh(affected)
        except Exception as error:
            QMessageBox.information(self, 'Optimization history', str(error))
