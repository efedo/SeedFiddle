"""Responsive cancellation and worker-draining shutdown for the native window."""
from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction

# Window attributes holding one cancellable background task each.
CANCELLABLE_TASKS = ('_learning_training_task', '_procedural_fit_task', '_reference_edge_fit_task',
                     '_optimization_task', '_species_library_build_task', '_result_export_task')


class WorkController:
    def _init_work_control(self):
        self._closing_after_work = False
        self._stop_requested = False
        self.stop_action = QAction('Stop', self)
        self.stop_action.setToolTip('Stop the current pipeline or background calculation (Ctrl+.)')
        self.stop_action.setShortcut('Ctrl+.')
        self.stop_action.triggered.connect(self._stop_current_work)
        self.addAction(self.stop_action)
        for action in self._optimization_menu_actions:
            if action.text() == '&Analysis':
                action.menu().addAction(self.stop_action)
        self._shutdown_timer = QTimer(self)
        self._shutdown_timer.setInterval(100)
        self._shutdown_timer.timeout.connect(self._poll_shutdown)
        self._sync_run_controls()

    def _stoppable_work_is_active(self):
        """Return whether Stop has anything to cancel; the one definition of running work."""
        return bool(
            getattr(getattr(self, 'catalogue_nodes', None), 'busy', False)
            or getattr(getattr(self, 'pipeline_workflow', None), '_job', None) is not None
            or self._active_tasks
            or any(getattr(self, name, None) is not None for name in CANCELLABLE_TASKS))

    def _sync_run_controls(self):
        """Offer Run when idle and Stop while work runs; return whether work is running."""
        running = self._stoppable_work_is_active()
        available = not running and self.image_view.image_path is not None
        self.analyze_action.setEnabled(available)
        if hasattr(self, 'analyze_button'):
            self.analyze_button.setEnabled(available)
        if hasattr(self, 'stop_action'):
            self.stop_action.setEnabled(running)
        return running

    def _stop_current_work(self):
        self.catalogue_nodes.stop()
        self._stop_requested = True
        self._pending_analysis_key = None
        self._pending_analysis_scope = None
        tasks = list(self._active_tasks.values()) + [getattr(self, name, None) for name in CANCELLABLE_TASKS]
        for task in tasks:
            if task is not None and callable(getattr(task, 'cancel', None)):
                task.cancel()
        self.statusBar().showMessage('Stop requested — waiting for the current calculation to reach a cancellation point.')

    def _poll_shutdown(self):
        if self._thread_pool.activeThreadCount() == 0:
            self._shutdown_timer.stop()
            self._closing_after_work = False
            self.close()
