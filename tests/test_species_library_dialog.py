from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SPECIES = (("soybean", "Soybean"), ("lupinus_mutabilis", "Lupinus mutabilis"))
IMAGES = (Path("soy1.jpg"), Path("soy2.jpg"), Path("soy3.jpg"), Path("lupin.jpg"))
IMAGE_SPECIES = {
    "soy1.jpg": "soybean",
    "soy2.jpg": "soybean",
    "soy3.jpg": "soybean",
    "lupin.jpg": "lupinus_mutabilis",
}


class _LibraryStore:
    def __init__(self, manifests=()):
        self.manifests = tuple(manifests)

    def list_manifests(self):
        return self.manifests


def _manifest(species_id="soybean", version="1", status="published"):
    source = SimpleNamespace(
        display_label="soy1.jpg",
        review_status="reviewed",
        source_sha256="b" * 64,
        biological_context=None,
        capture_group_id=None,
    )
    return SimpleNamespace(
        library_id=f"{species_id}-library",
        version=version,
        species_id=species_id,
        species_display_name=dict(SPECIES)[species_id],
        content_sha256="a" * 64,
        status=SimpleNamespace(value=status),
        sources=(source,),
        products=(),
        created_utc="2026-11-06T12:00:00Z",
        parent_content_sha256=None,
    )


class SpeciesLibraryDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    def dialog(self, species_id="soybean", manifests=(), annotated=("soy1.jpg", "soy2.jpg")):
        from seedvision.reference_library import BiologicalContext
        from seedvision.ui.species_library_dialog import SpeciesLibraryManagerDialog

        dialog = SpeciesLibraryManagerDialog(
            service=SimpleNamespace(store=_LibraryStore(manifests)),
            context=BiologicalContext(species_id),
            image_paths=IMAGES,
            current_pin=None,
            species=SPECIES,
            image_species=IMAGE_SPECIES,
            annotated_paths=frozenset(annotated),
        )
        self.addCleanup(dialog.close)
        return dialog

    def test_species_without_a_library_leads_to_creating_one(self) -> None:
        from PySide6.QtCore import Qt

        dialog = self.dialog()
        self.assertEqual(
            [dialog.tabs.tabText(index) for index in range(dialog.tabs.count())],
            ["Libraries", "Create a library", "Details"],
        )
        self.assertEqual(dialog.species_combo.currentText(), "Soybean")
        self.assertFalse(dialog.empty_state.isHidden())
        self.assertTrue(dialog.version_table.isHidden())
        self.assertIn("no Soybean reference library yet", dialog.empty_label.text())
        self.assertFalse(dialog.pin_button.isEnabled())
        dialog.create_first_button.click()
        self.assertEqual(dialog.tabs.currentIndex(), 1)

        # The Create tab names the species and lists only its images; one
        # without saved annotations is shown but cannot be checked.
        self.assertEqual(dialog.create_heading.text(), "New Soybean library version")
        self.assertEqual(dialog.build_button.text(), "Create Soybean library")
        listed = [dialog.source_list.item(index) for index in range(dialog.source_list.count())]
        self.assertEqual(
            [item.text() for item in listed],
            ["soy1.jpg", "soy2.jpg", "soy3.jpg — no saved annotations"],
        )
        self.assertEqual(
            [item.checkState() == Qt.CheckState.Checked for item in listed], [True, True, False]
        )
        self.assertFalse(listed[2].flags() & Qt.ItemFlag.ItemIsEnabled)
        self.assertEqual(
            dialog.source_selection_label.text(),
            "2 of 3 image(s) selected; 1 have no saved annotations yet.",
        )
        self.assertIn("1 project image(s) set to another species", dialog.other_species_label.text())
        self.assertTrue(dialog.unset_species_label.isHidden())
        self.assertEqual(dialog.version_edit.text(), "1")
        self.assertTrue(dialog.retained_source_list.isHidden())
        self.assertTrue(dialog.details_panel.isHidden())
        dialog.details_toggle.click()
        self.assertFalse(dialog.details_panel.isHidden())

        requests = []
        dialog.build_requested.connect(lambda *values: requests.append(values))
        listed[0].setCheckState(Qt.CheckState.Unchecked)
        self.assertTrue(dialog.build_button.isEnabled())
        dialog.build_button.click()
        paths, version, parent, removed, context, _capture = requests[0]
        self.assertEqual(paths, (Path("soy2.jpg"),))
        self.assertEqual((version, parent, removed), ("1", None, ()))
        self.assertEqual(context.species_id, "soybean")

    def test_choosing_another_species_lists_its_images_and_builds_for_it(self) -> None:
        dialog = self.dialog()
        dialog.species_combo.setCurrentIndex(dialog.species_combo.findData("lupinus_mutabilis"))
        self.assertEqual(dialog.create_heading.text(), "New Lupinus mutabilis library version")
        self.assertEqual(dialog.source_list.count(), 1)
        self.assertEqual(dialog.source_list.item(0).text(), "lupin.jpg — no saved annotations")
        self.assertFalse(dialog.build_button.isEnabled())
        self.assertIn("Annotate and save at least one", dialog.build_status_label.text())

    def test_unknown_species_must_be_chosen_before_creating(self) -> None:
        dialog = self.dialog(species_id="unknown")
        self.assertEqual(dialog.species_combo.currentText(), "Choose a species…")
        self.assertEqual(dialog.create_heading.text(), "Choose a species first")
        self.assertEqual(dialog.source_list.count(), 0)
        self.assertFalse(dialog.build_button.isEnabled())
        self.assertFalse(dialog.create_first_button.isVisibleTo(dialog))

    def test_existing_version_is_used_extended_and_only_pinned_for_its_species(self) -> None:
        from PySide6.QtCore import Qt
        from seedvision.reference_library import SpeciesLibraryPin

        dialog = self.dialog(manifests=(_manifest(),))
        self.assertTrue(dialog.empty_state.isHidden())
        self.assertEqual(dialog.version_table.rowCount(), 1)
        self.assertTrue(dialog.pin_button.isEnabled())
        self.assertEqual(dialog.version_edit.text(), "2")
        pinned = []
        dialog.pin_selected.connect(pinned.append)
        dialog.pin_button.click()
        self.assertEqual(pinned[0].version, "1")
        self.assertEqual(dialog.version_table.item(0, 0).text(), "●")
        self.assertIn("uses the Soybean library, version 1", dialog.summary_label.text())

        # Starting from version 1 keeps its images, each removable.
        dialog.base_combo.setCurrentIndex(1)
        self.assertFalse(dialog.retained_source_list.isHidden())
        self.assertEqual(dialog.retained_source_list.count(), 1)
        requests = []
        dialog.build_requested.connect(lambda *values: requests.append(values))
        dialog.retained_source_list.item(0).setCheckState(Qt.CheckState.Unchecked)
        dialog.build_button.click()
        _paths, version, parent, removed, _context, _capture = requests[0]
        self.assertEqual((version, parent.version, removed), ("2", "1", ("b" * 64,)))

        # A version that exists already is refused before any build.
        dialog.set_build_busy(False)
        dialog.version_edit.setText("1")
        from unittest.mock import patch

        with patch("seedvision.ui.species_library_dialog.QMessageBox.warning") as warning:
            dialog.build_button.click()
        self.assertEqual(len(requests), 1)
        self.assertIn("already exists", warning.call_args.args[2])

        # A library published for another species is not pinned here.
        dialog.published(SpeciesLibraryPin("lupin-library", "1", "lupinus_mutabilis", "c" * 64))
        self.assertEqual(len(pinned), 1)
        self.assertIn("Open a Lupinus mutabilis image", dialog.build_status_label.text())

    def test_other_species_library_cannot_be_pinned_from_this_image(self) -> None:
        dialog = self.dialog(manifests=(_manifest("lupinus_mutabilis"),))
        dialog.species_combo.setCurrentIndex(dialog.species_combo.findData("lupinus_mutabilis"))
        self.assertEqual(dialog.version_table.rowCount(), 1)
        self.assertFalse(dialog.pin_button.isEnabled())
        self.assertIn("open a Lupinus mutabilis image", dialog.selected_library_label.text())


if __name__ == "__main__":
    unittest.main()
