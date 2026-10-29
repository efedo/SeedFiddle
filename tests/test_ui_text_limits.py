"""Keep command labels short and tooltips readable across the desktop UI.

Commands (buttons, actions, menus) name a thing; tooltips say what happens.
Longer explanation belongs in a node's How it works text or the operator
guide. Checkbox captions are exempt because they are sentences by design.
"""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
UI_SOURCES = ROOT / "seedvision" / "ui"
COMMAND_CALLS = frozenset({"QPushButton", "QAction", "QToolButton", "addAction", "addMenu", "QMenu"})
COMMAND_LIMIT = 45
TOOLTIP_LIMIT = 450


def constant_strings(names):
    """Yield (location, text) for literal first arguments of the named calls."""

    for path in sorted(UI_SOURCES.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            name = (
                function.id if isinstance(function, ast.Name)
                else function.attr if isinstance(function, ast.Attribute)
                else None
            )
            argument = node.args[0]
            if name in names and isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                yield f"{path.name}:{node.lineno}", argument.value


class UiTextLimitTests(unittest.TestCase):
    def test_command_labels_stay_short(self) -> None:
        long_labels = {
            location: text
            for location, text in constant_strings(COMMAND_CALLS)
            if len(text) > COMMAND_LIMIT
        }
        self.assertEqual(long_labels, {}, f"command labels must be at most {COMMAND_LIMIT} characters")

    def test_tooltips_stay_readable(self) -> None:
        long_tooltips = {
            location: len(text)
            for location, text in constant_strings({"setToolTip"})
            if len(text) > TOOLTIP_LIMIT
        }
        self.assertEqual(long_tooltips, {}, f"tooltips must be at most {TOOLTIP_LIMIT} characters")

    def test_scan_sees_the_interface(self) -> None:
        # Guards against a scan that silently matches nothing.
        self.assertGreater(sum(1 for _ in constant_strings(COMMAND_CALLS)), 150)
        self.assertGreater(sum(1 for _ in constant_strings({"setToolTip"})), 100)


if __name__ == "__main__":
    unittest.main()
