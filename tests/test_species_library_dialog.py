from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _EmptyLibraryStore:
    def list_manifests(self):
        return ()


class SpeciesLibraryDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    def test_dialog_presents_three_explicit_workflow_steps(self) -> None:
        from seedvision.reference_library import BiologicalContext
        from seedvision.ui.species_library_dialog import (
            SpeciesLibraryManagerDialog,
        )

        dialog = SpeciesLibraryManagerDialog(
            service=SimpleNamespace(store=_EmptyLibraryStore()),
            context=BiologicalContext("lupinus_mutabilis"),
            image_paths=(Path("first.jpg"), Path("second.jpg")),
            current_pin=None,
        )
        self.assertEqual(dialog.tabs.count(), 3)
        self.assertEqual(dialog.tabs.tabText(0), "1. Use a library")
        self.assertEqual(dialog.tabs.tabText(1), "2. Create a version")
        self.assertEqual(dialog.tabs.tabText(2), "3. Inspect selected version")
        self.assertEqual(
            dialog.pin_button.text(), "Use selected version for this project"
        )
        self.assertEqual(
            dialog.build_button.text(),
            "Validate sources and publish new version",
        )
        self.assertIn("2 of 2", dialog.source_selection_label.text())
        self.assertFalse(dialog.pin_button.isEnabled())
        self.assertTrue(dialog.retained_source_list.isHidden())
        dialog.close()

    def test_source_selection_summary_and_fork_guidance_are_immediate(self) -> None:
        from PySide6.QtCore import Qt
        from seedvision.reference_library import BiologicalContext
        from seedvision.ui.species_library_dialog import (
            SpeciesLibraryManagerDialog,
        )

        dialog = SpeciesLibraryManagerDialog(
            service=SimpleNamespace(store=_EmptyLibraryStore()),
            context=BiologicalContext("soybean"),
            image_paths=(Path("only.jpg"),),
            current_pin=None,
        )
        dialog.source_list.item(0).setCheckState(Qt.CheckState.Unchecked)
        self.assertIn("0 of 1", dialog.source_selection_label.text())
        dialog.fork_checkbox.setChecked(True)
        self.assertIn("Select an installed version", dialog.retained_sources_label.text())
        self.assertFalse(dialog.retained_source_list.isEnabled())
        dialog.close()


if __name__ == "__main__":
    unittest.main()
