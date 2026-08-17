from __future__ import annotations

import unittest

import numpy as np

from seedvision.ui.reference_history import RasterUndoHistory


class RasterUndoHistoryTests(unittest.TestCase):
    def test_grouped_reference_edit_is_restored_atomically(self) -> None:
        shape = (300, 310)
        background = np.zeros(shape, dtype=bool)
        foreground = np.zeros(shape, dtype=bool)
        other = np.zeros(shape, dtype=bool)
        foreground[42:58, 47:63] = True
        before = (background, foreground, other)
        after = tuple(value.copy() for value in before)
        after[0][45:62, 50:67] = True
        after[1][45:62, 50:67] = False

        history = RasterUndoHistory(limit=8)
        self.assertTrue(
            history.record(
                "background stroke", "material", before, after
            )
        )
        restored = history.undo(after)

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.context, "material")
        self.assertEqual(restored.label, "background stroke")
        for actual, expected in zip(restored.rasters, before, strict=True):
            self.assertTrue(np.array_equal(actual, expected))
        self.assertEqual(len(history), 0)

    def test_multiple_tiles_and_commands_undo_in_reverse_order(self) -> None:
        original = np.zeros((390, 390), dtype=np.uint16)
        first = original.copy()
        first[2:9, 3:11] = 4
        first[300:315, 280:300] = 4
        second = first.copy()
        second[135:150, 129:145] = 8
        history = RasterUndoHistory(limit=6)

        self.assertTrue(
            history.record("first stroke", "instances", (original,), (first,))
        )
        self.assertTrue(
            history.record("second stroke", "instances", (first,), (second,))
        )
        undone_second = history.undo((second,))
        assert undone_second is not None
        self.assertTrue(np.array_equal(undone_second.rasters[0], first))
        undone_first = history.undo(undone_second.rasters)
        assert undone_first is not None
        self.assertTrue(np.array_equal(undone_first.rasters[0], original))

    def test_no_op_does_not_consume_history(self) -> None:
        values = np.zeros((16, 18), dtype=bool)
        history = RasterUndoHistory(limit=5)

        self.assertFalse(
            history.record("no-op", "boundary", (values,), (values.copy(),))
        )
        self.assertEqual(len(history), 0)
        self.assertIsNone(history.undo((values,)))

    def test_queue_is_bounded_and_preserves_instance_origin(self) -> None:
        history = RasterUndoHistory(limit=5)
        states = [np.zeros((12, 20), dtype=np.uint16)]
        for index in range(8):
            next_state = states[-1].copy()
            next_state[1, index] = index + 1
            self.assertTrue(
                history.record(
                    f"edit {index}",
                    "instances",
                    (states[-1],),
                    (next_state,),
                    metadata=f"origin-{index}",
                )
            )
            states.append(next_state)

        self.assertEqual(len(history), 5)
        current = (states[-1],)
        labels = []
        origins = []
        while len(history):
            result = history.undo(current)
            assert result is not None
            current = result.rasters
            labels.append(result.label)
            origins.append(result.metadata)
        self.assertEqual(labels, ["edit 7", "edit 6", "edit 5", "edit 4", "edit 3"])
        self.assertEqual(
            origins,
            ["origin-7", "origin-6", "origin-5", "origin-4", "origin-3"],
        )
        # The three commands evicted by the bounded queue remain applied.
        self.assertTrue(np.array_equal(current[0], states[3]))

    def test_limit_must_retain_at_least_five_commands(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least five"):
            RasterUndoHistory(limit=4)


if __name__ == "__main__":
    unittest.main()
