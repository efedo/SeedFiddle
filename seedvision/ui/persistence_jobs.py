"""Run blocking persistence in a worker while Qt continues to process events."""
from PySide6.QtCore import QObject, QRunnable, Signal, QEventLoop, Qt
from PySide6.QtWidgets import QProgressDialog
from seedvision.persistence.reference_regions import ReferenceRegionStore


class _Signals(QObject):
    completed = Signal(object, object)


class PersistenceJob(QRunnable):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.signals = _Signals()

    def run(self):
        try:
            self.signals.completed.emit(self.callback(),None)
        except Exception as error:
            self.signals.completed.emit(None,error)


def run_persistence(window, callback, title='Saving verified data…'):
    """Preserve transactional call semantics without blocking the GUI event loop."""
    loop = QEventLoop(window)
    progress = QProgressDialog(title,'',0,0,window)
    progress.setWindowModality(Qt.WindowModality.ApplicationModal)
    progress.setMinimumDuration(150)
    progress.setCancelButton(None)
    task = PersistenceJob(callback)
    outcome = []
    def done(value,error):
        outcome.extend((value,error))
        loop.quit()
    task.signals.completed.connect(done)
    window._persistence_job = task
    progress.show()
    window._thread_pool.start(task)
    loop.exec()
    progress.close()
    window._persistence_job = None
    if outcome[1] is not None:
        raise outcome[1]
    return outcome[0]


class ResponsiveReferenceRegionStore(ReferenceRegionStore):
    def __init__(self, root, window):
        super().__init__(root)
        self.window = window

    def save(self, *args, **kwargs):
        from seedvision.ui.main_window import _path_identity
        from seedvision.persistence.reference_regions import file_sha256, ReferenceRegionError
        source = args[0] if args else kwargs['image_path']
        expected = self.window._source_fingerprints.get(_path_identity(source))
        if expected is not None and file_sha256(source) != expected:
            raise ReferenceRegionError('Source changed since these annotations were opened. Restore the original source before saving its references.')
        save = super().save
        return run_persistence(self.window,lambda:save(*args,**kwargs),'Saving reference annotation revision…')
