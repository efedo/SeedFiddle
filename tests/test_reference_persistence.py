from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from seedvision.persistence import (
    ImageFingerprintMismatch,
    InvalidReferenceArchive,
    ReferenceRegionBundle,
    ReferenceRegionStore,
)


class ReferenceRegionStoreTests(unittest.TestCase):
    def test_round_trip_preserves_material_and_instance_outputs_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "sample.jpg"
            image_path.parent.mkdir()
            image_path.write_bytes(b"unchanged source image bytes")
            shape = (18, 24)
            background = np.zeros(shape, dtype=bool)
            foreground = np.zeros(shape, dtype=bool)
            other = np.zeros(shape, dtype=bool)
            instances = np.zeros(shape, dtype=np.uint16)
            background[1:4, 2:7] = True
            foreground[6:11, 3:9] = True
            other[12:16, 15:20] = True
            instances[6:9, 3:7] = 17
            instances[12:16, 15:20] = 42
            store = ReferenceRegionStore(root)

            destination = store.save(
                image_path,
                ReferenceRegionBundle(
                    shape=shape,
                    background=background,
                    foreground=foreground,
                    other=other,
                    annotated_seeds=instances,
                    annotation_origin="pipeline:procedural_instances",
                ),
            )
            loaded = store.load_if_present(image_path, shape)

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertTrue(destination.is_relative_to(root / "projects"))
            self.assertTrue(np.array_equal(loaded.background, background))
            self.assertTrue(np.array_equal(loaded.foreground, foreground))
            self.assertTrue(np.array_equal(loaded.other, other))
            self.assertIsNone(loaded.physical_edge)
            self.assertIsNone(loaded.non_edge)
            self.assertTrue(np.array_equal(loaded.annotated_seeds, instances))
            self.assertEqual(
                loaded.annotation_origin, "pipeline:procedural_instances"
            )
            with np.load(destination, allow_pickle=False) as archive:
                self.assertEqual(int(archive["version"]), 2)
                self.assertEqual(archive["material"].dtype, np.uint8)
                self.assertNotIn("boundary", archive.files)
                self.assertEqual(archive["annotated_seeds"].dtype, np.uint16)
            self.assertEqual(tuple(destination.parent.glob("*.tmp")), ())

    def test_version_one_boundary_layer_is_validated_but_retired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "capture.png"
            image_path.write_bytes(b"legacy source")
            shape = (7, 9)
            store = ReferenceRegionStore(root)
            destination = store.save(
                image_path,
                ReferenceRegionBundle(shape=shape),
            )
            with np.load(destination, allow_pickle=False) as archive:
                payload = {name: np.array(archive[name]) for name in archive.files}
            boundary = np.zeros(shape, dtype=np.uint8)
            boundary[2, 3] = 1
            boundary[4, 5] = 2
            payload["version"] = np.asarray(1, dtype=np.uint16)
            payload["boundary"] = boundary
            with destination.open("wb") as stream:
                np.savez_compressed(stream, **payload)

            loaded = store.load_if_present(image_path, shape)

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertIsNone(loaded.physical_edge)
            self.assertIsNone(loaded.non_edge)

    def test_changed_image_is_rejected_before_reference_rasters_are_used(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "capture.png"
            image_path.write_bytes(b"first image")
            store = ReferenceRegionStore(root)
            store.save(
                image_path,
                ReferenceRegionBundle(shape=(4, 5)),
            )
            image_path.write_bytes(b"modified image")

            with self.assertRaises(ImageFingerprintMismatch) as raised:
                store.load_if_present(image_path, (4, 5))

            self.assertNotEqual(
                raised.exception.expected_sha256,
                raised.exception.actual_sha256,
            )

    def test_invalid_categorical_archive_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "capture.png"
            image_path.write_bytes(b"source")
            store = ReferenceRegionStore(root)
            destination = store.save(
                image_path,
                ReferenceRegionBundle(shape=(4, 5)),
            )
            with np.load(destination, allow_pickle=False) as archive:
                payload = {name: np.array(archive[name]) for name in archive.files}
            payload["material"][2, 3] = 9
            with destination.open("wb") as stream:
                np.savez_compressed(stream, **payload)

            with self.assertRaises(InvalidReferenceArchive):
                store.load_if_present(image_path, (4, 5))

    def test_corrupt_archive_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "capture.png"
            image_path.write_bytes(b"source")
            store = ReferenceRegionStore(root)
            archive = store.path_for(image_path)
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"not a zip archive")

            with self.assertRaises(InvalidReferenceArchive):
                store.load_if_present(image_path, (4, 5))

    def test_invalid_archive_can_be_backed_up_and_replaced_with_empty_current_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "capture.png"
            image_path.write_bytes(b"source")
            store = ReferenceRegionStore(root)
            destination = store.save(
                image_path,
                ReferenceRegionBundle(
                    shape=(7, 9),
                    foreground=np.ones((7, 9), dtype=bool),
                ),
            )
            original_bytes = destination.read_bytes()

            backup, replacement = store.backup_and_replace_invalid(
                image_path, (4, 5)
            )

            self.assertEqual(replacement, destination)
            self.assertTrue(backup.is_file())
            self.assertEqual(backup.read_bytes(), original_bytes)
            self.assertIn(".invalid-", backup.name)
            self.assertTrue(backup.name.endswith(".bak"))
            loaded = store.load_if_present(image_path, (4, 5))
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.shape, (4, 5))
            self.assertIsNone(loaded.background)
            self.assertIsNone(loaded.foreground)
            self.assertIsNone(loaded.other)
            self.assertIsNone(loaded.annotated_seeds)

    def test_same_named_external_images_receive_distinct_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            first = Path(temporary) / "one" / "capture.png"
            second = Path(temporary) / "two" / "capture.png"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            store = ReferenceRegionStore(root)
            self.assertNotEqual(store.path_for(first), store.path_for(second))


class ReferenceRegionMainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest(f"PySide6 unavailable: {error}")
        cls.application = QApplication.instance() or QApplication([])

    @staticmethod
    def _write_image(path: Path, colour: str) -> None:
        from PySide6.QtGui import QColor, QImage

        path.parent.mkdir(parents=True, exist_ok=True)
        image = QImage(64, 48, QImage.Format.Format_RGB888)
        image.fill(QColor(colour))
        if not image.save(str(path)):
            raise RuntimeError(f"Could not write test image {path}.")

    def _save_complete_snapshot(self, root: Path, image_path: Path) -> None:
        from seedvision.ui.main_window import MainWindow

        window = MainWindow(root)
        key = window._current_image_key()
        self.assertIsNotNone(key)
        assert key is not None
        shape = (48, 64)
        background = np.zeros(shape, dtype=bool)
        foreground = np.zeros(shape, dtype=bool)
        other = np.zeros(shape, dtype=bool)
        instances = np.zeros(shape, dtype=np.uint16)
        background[2:7, 3:9] = True
        foreground[10:18, 12:22] = True
        other[30:35, 50:55] = True
        instances[10:18, 12:22] = 7
        window._applied_background_reference_masks[key] = background
        window._applied_foreground_reference_masks[key] = foreground
        window._applied_background_exclusion_masks[key] = other
        window._applied_foreground_exclusion_masks[key] = other
        window._applied_instance_annotations[key] = instances
        window._applied_instance_annotation_origins[key] = "manual:test"
        window._save_reference_regions()
        self.assertTrue(window._reference_region_store.path_for(image_path).is_file())
        window.close()

    def test_saved_snapshot_automatically_loads_as_immutable_applied_state(self) -> None:
        from seedvision.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "capture.png"
            self._write_image(image_path, "#38684a")
            self._save_complete_snapshot(root, image_path)

            restored = MainWindow(root)
            key = restored._current_image_key()
            self.assertIsNotNone(key)
            assert key is not None
            self.assertTrue(restored._applied_background_reference_masks[key][3, 4])
            self.assertTrue(restored._applied_foreground_reference_masks[key][12, 15])
            self.assertTrue(restored._applied_background_exclusion_masks[key][32, 52])
            self.assertIs(
                restored._applied_background_exclusion_masks[key],
                restored._applied_foreground_exclusion_masks[key],
            )
            self.assertNotIn(key, restored._applied_physical_edge_reference_masks)
            self.assertNotIn(key, restored._applied_non_edge_reference_masks)
            self.assertEqual(restored._applied_instance_annotations[key][12, 15], 7)
            self.assertEqual(
                restored._applied_instance_annotation_origins[key], "manual:test"
            )
            self.assertFalse(
                restored._applied_foreground_reference_masks[key].flags.writeable
            )
            self.assertFalse(
                restored._applied_instance_annotations[key].flags.writeable
            )
            self.assertIs(
                restored.image_view.reference_mask("foreground", copy=False),
                restored._applied_foreground_reference_masks[key],
            )
            self.assertIs(
                restored.image_view._instance_annotations,
                restored._applied_instance_annotations[key],
            )
            self.assertNotIn(key, restored._reference_masks_dirty)
            self.assertNotIn(key, restored._instance_annotations_dirty)
            restored.close()

    def test_loose_corrected_shape_is_deferred_instead_of_rejected_against_raw_image(self) -> None:
        from types import SimpleNamespace

        from seedvision.ui.main_window import MainWindow, _path_identity

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "capture.png"
            self._write_image(image_path, "#38684a")
            corrected_shape = (54, 70)
            foreground = np.zeros(corrected_shape, dtype=bool)
            foreground[8:24, 13:31] = True
            ReferenceRegionStore(root).save(
                image_path,
                ReferenceRegionBundle(
                    shape=corrected_shape, foreground=foreground
                ),
            )

            with patch("seedvision.ui.main_window.QMessageBox.warning") as warning:
                window = MainWindow(root)
            key = _path_identity(image_path)
            self.assertIn(key, window._pending_unbound_reference_bundles)
            self.assertNotIn(key, window._applied_foreground_reference_masks)
            warning.assert_not_called()

            result = SimpleNamespace(
                image_path=image_path,
                calibration=SimpleNamespace(
                    corrected_bgr=np.zeros((*corrected_shape, 3), np.uint8)
                ),
            )
            affected = window._resolve_pending_project_reference_bundle(key, result)

            self.assertIn("reference_layers", affected)
            np.testing.assert_array_equal(
                window._applied_foreground_reference_masks[key], foreground
            )
            self.assertNotIn(key, window._withheld_reference_sidecars)
            window.close()

    def test_genuine_loose_shape_mismatch_offers_backup_and_empty_replacement(self) -> None:
        from types import SimpleNamespace

        from seedvision.ui.main_window import MainWindow, _path_identity

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "capture.png"
            self._write_image(image_path, "#38684a")
            saved_shape = (54, 70)
            corrected_shape = (56, 72)
            foreground = np.zeros(saved_shape, dtype=bool)
            foreground[8:24, 13:31] = True
            store = ReferenceRegionStore(root)
            archive = store.save(
                image_path,
                ReferenceRegionBundle(shape=saved_shape, foreground=foreground),
            )
            original_bytes = archive.read_bytes()
            window = MainWindow(root)
            key = _path_identity(image_path)
            self.assertIn(key, window._pending_unbound_reference_bundles)
            result = SimpleNamespace(
                image_path=image_path,
                calibration=SimpleNamespace(
                    corrected_bgr=np.zeros((*corrected_shape, 3), np.uint8)
                ),
            )
            dialog = MagicMock()
            replace_button = object()
            keep_button = object()
            dialog.addButton.side_effect = (replace_button, keep_button)
            dialog.clickedButton.return_value = replace_button

            with patch(
                "seedvision.ui.main_window.QMessageBox", return_value=dialog
            ) as message_box:
                affected = window._resolve_pending_project_reference_bundle(
                    key, result
                )

            self.assertEqual(affected, set())
            self.assertEqual(
                [call.args[0] for call in dialog.addButton.call_args_list],
                ["Back up + replace", "Keep unchanged"],
            )
            message_box.information.assert_called_once()
            backups = tuple(
                archive.parent.glob(f"{archive.name}.invalid-*.bak")
            )
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original_bytes)
            replacement = store.load_if_present(image_path, corrected_shape)
            self.assertIsNotNone(replacement)
            assert replacement is not None
            self.assertIsNone(replacement.foreground)
            self.assertIsNone(replacement.annotated_seeds)
            self.assertNotIn(key, window._withheld_reference_sidecars)
            window.close()

    def test_switching_images_keeps_newer_unsaved_in_memory_applied_state(self) -> None:
        from seedvision.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path = root / "images" / "a.png"
            second_path = root / "images" / "b.png"
            self._write_image(first_path, "#38684a")
            self._write_image(second_path, "#4a3868")
            old = np.zeros((48, 64), dtype=bool)
            old[2:5, 2:5] = True
            ReferenceRegionStore(root).save(
                first_path,
                ReferenceRegionBundle(shape=(48, 64), background=old),
            )
            window = MainWindow(root)
            key = window._current_image_key()
            self.assertIsNotNone(key)
            assert key is not None
            newer = np.zeros((48, 64), dtype=bool)
            newer[30:36, 40:47] = True
            newer.flags.writeable = False
            window._applied_background_reference_masks[key] = newer

            window._open_path(second_path)
            window._open_path(first_path)

            self.assertIs(window._applied_background_reference_masks[key], newer)
            self.assertFalse(window.image_view.reference_mask("background")[3, 3])
            self.assertTrue(window.image_view.reference_mask("background")[32, 42])
            self.assertGreater(
                len(window.image_view._overlay_items),
                0,
                "Restored references must be repainted even without a cached analysis.",
            )
            window.close()

    def test_association_panel_explains_pending_corrected_seed_annotations(self) -> None:
        from seedvision.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "capture.png"
            self._write_image(image_path, "#38684a")
            saved_shape = (52, 68)
            background = np.zeros(saved_shape, dtype=bool)
            foreground = np.zeros(saved_shape, dtype=bool)
            other = np.zeros(saved_shape, dtype=bool)
            labels = np.zeros(saved_shape, dtype=np.uint16)
            background[1:5, 2:8] = True
            foreground[10:18, 12:21] = True
            other[30:34, 40:47] = True
            labels[10:18, 12:21] = 3
            labels[33:41, 48:59] = 9
            ReferenceRegionStore(root).save(
                image_path,
                ReferenceRegionBundle(
                    shape=saved_shape,
                    background=background,
                    foreground=foreground,
                    other=other,
                    annotated_seeds=labels,
                ),
            )

            window = MainWindow(root)
            key = window._current_image_key()
            self.assertIsNotNone(key)
            assert key is not None
            panel = window.reference_association_label.text()
            self.assertIn(
                "Validated; waiting for corrected-image calibration", panel
            )
            self.assertIn("Saved corrected coordinates: 68 × 52 px", panel)
            self.assertIn("Annotated seeds 2 IDs / 160 px", panel)
            self.assertIn(str(ReferenceRegionStore(root).path_for(image_path)), panel)
            self.assertNotIn(key, window._applied_instance_annotations)

            result = SimpleNamespace(
                calibration=SimpleNamespace(
                    corrected_bgr=np.zeros((*saved_shape, 3), dtype=np.uint8)
                ),
                image_path=image_path,
            )
            affected = window._resolve_pending_project_reference_bundle(key, result)
            self.assertTrue(affected)
            self.assertTrue(
                np.array_equal(window._applied_instance_annotations[key], labels)
            )
            # The completion handler must adopt corrected dimensions before it
            # binds the newly installed corrected-coordinate sidecar.
            window.image_view.show_analysis(result, render=False)
            window.image_view.prepare_analysis_coordinates()
            self.assertEqual(window.image_view.image_size, (68, 52))
            window._sync_reference_masks_to_view(key, render=False)
            window.image_view.set_instance_annotations(
                window._applied_instance_annotations[key],
                copy=False,
                render=False,
            )
            self.assertIn(
                "Loaded and available to analysis",
                window.reference_association_label.text(),
            )
            window.close()

    def test_empty_applied_snapshot_overwrites_an_older_nonempty_archive(self) -> None:
        from seedvision.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "capture.png"
            self._write_image(image_path, "#38684a")
            self._save_complete_snapshot(root, image_path)
            window = MainWindow(root)
            key = window._current_image_key()
            self.assertIsNotNone(key)
            assert key is not None
            for store in (
                window._applied_background_reference_masks,
                window._applied_foreground_reference_masks,
                window._applied_background_exclusion_masks,
                window._applied_foreground_exclusion_masks,
                window._applied_physical_edge_reference_masks,
                window._applied_non_edge_reference_masks,
                window._applied_instance_annotations,
            ):
                store.pop(key, None)
            window._applied_instance_annotation_origins.pop(key, None)

            window._save_reference_regions()
            loaded = window._reference_region_store.load_if_present(
                image_path, (48, 64)
            )

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertIsNone(loaded.background)
            self.assertIsNone(loaded.foreground)
            self.assertIsNone(loaded.other)
            self.assertIsNone(loaded.physical_edge)
            self.assertIsNone(loaded.non_edge)
            self.assertIsNone(loaded.annotated_seeds)
            window.close()

    def test_modified_image_warns_and_loads_no_saved_regions(self) -> None:
        from seedvision.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "capture.png"
            self._write_image(image_path, "#38684a")
            self._save_complete_snapshot(root, image_path)
            self._write_image(image_path, "#8b4038")

            with patch("seedvision.ui.main_window.QMessageBox.warning") as warning:
                rejected = MainWindow(root)
            key = rejected._current_image_key()
            self.assertIsNotNone(key)
            assert key is not None
            warning.assert_called_once()
            self.assertIn("changed", warning.call_args.args[2])
            self.assertNotIn(key, rejected._applied_background_reference_masks)
            self.assertNotIn(key, rejected._applied_foreground_reference_masks)
            self.assertNotIn(key, rejected._applied_instance_annotations)
            rejected.close()

    def test_dirty_draft_blocks_save_without_replacing_archive(self) -> None:
        from seedvision.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "capture.png"
            self._write_image(image_path, "#38684a")
            window = MainWindow(root)
            key = window._current_image_key()
            self.assertIsNotNone(key)
            assert key is not None
            window._reference_masks_dirty.add(key)
            with patch("seedvision.ui.main_window.QMessageBox.warning") as warning:
                window._save_reference_regions()
            warning.assert_called_once()
            self.assertFalse(window._reference_region_store.path_for(image_path).exists())
            window.close()

    def test_applying_seed_instances_automatically_saves_and_reloads_them(self) -> None:
        from seedvision.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "capture.png"
            self._write_image(image_path, "#38684a")
            window = MainWindow(root)
            key = window._current_image_key()
            self.assertIsNotNone(key)
            assert key is not None
            labels = np.zeros((48, 64), dtype=np.uint16)
            labels[7:15, 9:18] = 3
            labels[28:39, 42:54] = 11
            window._analyze_current_image = lambda **_kwargs: None

            window._instance_annotations_edited(labels)
            self.assertFalse(
                window._reference_region_store.path_for(image_path).exists()
            )
            window._apply_instance_annotations()

            archive = window._reference_region_store.path_for(image_path)
            self.assertTrue(archive.is_file())
            self.assertEqual(
                window.apply_instance_annotations_button.text(), "Apply + save"
            )
            loaded = window._reference_region_store.load_if_present(
                image_path, (48, 64)
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertTrue(np.array_equal(loaded.annotated_seeds, labels))
            self.assertEqual(loaded.annotation_origin, "manual")
            window.close()

            restored = MainWindow(root)
            restored_key = restored._current_image_key()
            self.assertIsNotNone(restored_key)
            assert restored_key is not None
            self.assertTrue(
                np.array_equal(
                    restored._applied_instance_annotations[restored_key], labels
                )
            )
            restored.close()

    def test_seed_instance_autosave_failure_warns_but_keeps_applied_state(self) -> None:
        from seedvision.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "capture.png"
            self._write_image(image_path, "#38684a")
            window = MainWindow(root)
            key = window._current_image_key()
            self.assertIsNotNone(key)
            assert key is not None
            labels = np.zeros((48, 64), dtype=np.uint16)
            labels[12:22, 15:27] = 5
            window._analyze_current_image = lambda **_kwargs: None
            window._instance_annotations_edited(labels)

            with (
                patch.object(
                    window._reference_region_store,
                    "save",
                    side_effect=OSError("disk full"),
                ),
                patch("seedvision.ui.main_window.QMessageBox.critical") as critical,
            ):
                window._apply_instance_annotations()

            critical.assert_called_once()
            self.assertEqual(
                critical.call_args.args[1], "Automatic reference save failed"
            )
            self.assertIn("remain available", critical.call_args.args[2])
            self.assertTrue(
                np.array_equal(window._applied_instance_annotations[key], labels)
            )
            self.assertNotIn(key, window._instance_annotations_dirty)
            self.assertNotIn(key, window._draft_instance_annotations)
            self.assertIn(
                "Automatic disk save failed", window.statusBar().currentMessage()
            )
            self.assertFalse(
                window._reference_region_store.path_for(image_path).exists()
            )
            window.close()


if __name__ == "__main__":
    unittest.main()
