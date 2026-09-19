"""Responsive cancellation and worker-draining shutdown for the native window."""
from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction


class WorkController:
    def _init_work_control(self):
        self._closing_after_work = False
        self._stop_requested = False
        self.stop_action = QAction('Stop current work', self)
        self.stop_action.setShortcut('Ctrl+.')
        self.stop_action.triggered.connect(self._stop_current_work)
        self.addAction(self.stop_action)
        for action in self._optimization_menu_actions:
            if action.text() == '&Analysis':
                action.menu().addAction(self.stop_action)
        self._shutdown_timer = QTimer(self)
        self._shutdown_timer.setInterval(100)
        self._shutdown_timer.timeout.connect(self._poll_shutdown)

    def _stop_current_work(self):
        self._stop_requested = True
        self._pending_analysis_key = None
        self._pending_analysis_scope = None
        tasks = list(self._active_tasks.values()) + [getattr(self, name, None) for name in
            ('_learning_training_task', '_procedural_fit_task', '_reference_edge_fit_task', '_optimization_task', '_species_library_build_task')]
        for task in tasks:
            if task is not None and callable(getattr(task, 'cancel', None)):
                task.cancel()
        self.statusBar().showMessage('Stop requested — waiting for the current calculation to reach a cancellation point.')

    def _poll_shutdown(self):
        if self._thread_pool.activeThreadCount() == 0:
            self._shutdown_timer.stop()
            self._closing_after_work = False
            self.close()
