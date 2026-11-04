"""Deferred callbacks die with their owner and report what they raise.

QTimer.singleShot(0, callback) does neither. Its error is stored during
processEvents() and static dialogs, and then crashes graphics items that are
painting for the first time, so a failure here can end the process.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QCoreApplication, QEvent, QRectF, QTimer
from PySide6.QtWidgets import (QApplication, QGraphicsObject, QGraphicsScene, QGraphicsView,
                               QMessageBox, QWidget)

from seedvision.ui.deferred import run_soon

ROOT = Path(__file__).resolve().parents[1]


class _Card(QGraphicsObject):
    def __init__(self):
        super().__init__()
        self.paints = 0

    def boundingRect(self):
        return QRectF(0, 0, 40, 40)

    def paint(self, painter, option, widget=None):
        self.paints += 1
        painter.drawRect(0, 0, 40, 40)


@contextmanager
def reported_errors():
    errors = []
    with patch.object(sys, 'excepthook', lambda *details: errors.append(str(details[1]))):
        yield errors


def fail():
    raise RuntimeError('raised by a deferred callback')


class DeferredCallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _fresh_cards(self):
        """A shown view whose cards have not painted yet, as in a new canvas."""
        scene = QGraphicsScene()
        cards = [_Card() for _ in range(5)]
        for index, card in enumerate(cards):
            card.setPos(index * 10, index * 10)
            scene.addItem(card)
        view = QGraphicsView(scene)
        # The view does not own its scene.
        self.addCleanup(lambda kept=scene: view.close())
        view.resize(300, 300)
        view.show()
        return view, cards

    def _pump_until(self, finished):
        for _ in range(100):
            self.app.processEvents()
            if finished():
                return
            time.sleep(0.01)
        self.fail('The expected event never arrived.')

    def test_callback_runs_in_one_event_pass_in_order_and_leaves_no_timer(self):
        owner, calls = QWidget(), []
        run_soon(owner, lambda: calls.append('first'))
        run_soon(owner, lambda: calls.append('second'))
        self.assertEqual(calls, [])
        self.app.processEvents()
        self.assertEqual(calls, ['first', 'second'])
        self._pump_until(lambda: not owner.findChildren(QTimer))

    def test_deleting_the_owner_cancels_the_callback(self):
        owner, calls = QWidget(), []
        run_soon(owner, lambda: calls.append('late'))
        owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.assertEqual(calls, [])

    def test_raising_callback_is_reported_and_new_cards_paint_during_process_events(self):
        view, cards = self._fresh_cards()
        with reported_errors() as errors:
            run_soon(view, fail)
            self._pump_until(lambda: all(card.paints for card in cards))
        self.assertEqual(errors, ['raised by a deferred callback'])

    def test_raising_callback_is_reported_and_new_cards_paint_during_a_static_dialog(self):
        view, cards = self._fresh_cards()
        closing = QTimer(view)
        closing.setSingleShot(True)
        closing.timeout.connect(lambda: QApplication.activeModalWidget().close())
        closing.start(100)
        with reported_errors() as errors:
            run_soon(view, fail)
            QMessageBox.information(None, 'Deferred', 'Cards paint behind this dialog.')
        self.assertEqual(errors, ['raised by a deferred callback'])
        self.assertTrue(all(card.paints for card in cards))

    def test_deleted_window_with_a_pending_workspace_restore_cannot_harm_a_new_canvas(self):
        from seedvision.pipeline import build_default_pipeline
        from seedvision.ui.pipeline_canvas import PipelineCanvas, PipelineNodeItem
        from tests.review_fixtures import isolated_window
        painted = []
        real_paint = PipelineNodeItem.paint
        def paint(item, *arguments):
            painted.append(item)
            return real_paint(item, *arguments)
        with tempfile.TemporaryDirectory() as directory, reported_errors() as errors, \
                patch.object(PipelineNodeItem, 'paint', paint):
            window = isolated_window(Path(directory), settings_path=Path(directory) / 'settings.ini')
            # Hiding the image defers a check that a workspace is still showing.
            window.image_workspace_action.setChecked(False)
            window.deleteLater()
            del window
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            canvas = PipelineCanvas(build_default_pipeline())
            self.addCleanup(canvas.close)
            canvas.resize(1600, 1000)
            canvas.show()
            self._pump_until(lambda: len(painted) >= 10)
        self.assertEqual(errors, [])

    def test_application_code_defers_only_through_owned_timers(self):
        direct = re.compile(r'\bsingleShot\s*\(')
        users = [path.relative_to(ROOT).as_posix() for path in (ROOT / 'seedvision').rglob('*.py')
                 if path.name != 'deferred.py' and direct.search(path.read_text(encoding='utf-8'))]
        self.assertEqual(users, [], 'Use seedvision.ui.deferred.run_soon, not QTimer.singleShot.')
