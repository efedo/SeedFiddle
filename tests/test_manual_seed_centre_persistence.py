from __future__ import annotations

from tests.functional_units import functional_units

import tempfile
import unittest
from pathlib import Path

import numpy as np

from seedvision.persistence import (
    InvalidManualSeedCentreArchive,
    ManualSeedCentreFingerprintMismatch,
    ManualSeedCentreStore,
    ManualSeedCentreStoreError,
)


@functional_units
class ManualSeedCentreStoreTests(unittest.TestCase):
    @staticmethod
    def _image(root: Path, name: str, payload: bytes) -> Path:
        path = root / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def check_test_round_trip_preserves_exact_source_floats_mode_and_empty_replace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = self._image(root, "capture.png", b"source photograph")
            store = ManualSeedCentreStore(root)
            centres = np.asarray(
                ((3.125, 4.875), (29.75, 18.0625)), dtype=np.float64
            )

            destination = store.save(
                image,
                centres,
                mode="augment",
                source_shape=(24, 36),
            )
            loaded = store.load_if_present(image, (24, 36))

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertTrue(destination.is_relative_to(root / "projects"))
            self.assertEqual(loaded.source_shape, (24, 36))
            self.assertEqual(loaded.mode, "augment")
            np.testing.assert_array_equal(loaded.centres_xy, centres)
            self.assertFalse(loaded.centres_xy.flags.writeable)

            # An empty Replace state is an intentional deletion of every
            # automatic centre, not an absent sidecar.
            empty = np.empty((0, 2), dtype=np.float64)
            store.save(
                image,
                empty,
                mode="replace_automatic",
                source_shape=(24, 36),
            )
            deleted = store.load_if_present(image, (24, 36))
            self.assertIsNotNone(deleted)
            assert deleted is not None
            self.assertEqual(deleted.mode, "replace_automatic")
            self.assertEqual(deleted.centres_xy.shape, (0, 2))
            self.assertEqual(deleted.suppressed_xy.shape, (0, 2))

            # Augment also keeps the automatic centres the user removed.
            removed = np.asarray(((7.5, 9.25),), dtype=np.float64)
            store.save(
                image,
                centres,
                mode="augment",
                source_shape=(24, 36),
                suppressed_xy=removed,
            )
            restored = store.load_if_present(image, (24, 36))
            assert restored is not None
            np.testing.assert_array_equal(restored.centres_xy, centres)
            np.testing.assert_array_equal(restored.suppressed_xy, removed)
            self.assertFalse(restored.suppressed_xy.flags.writeable)

    def check_test_version_one_sidecar_loads_without_suppressed_centres(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = self._image(root, "capture.png", b"source photograph")
            store = ManualSeedCentreStore(root)
            centres = np.asarray(((3.0, 4.0),), dtype=np.float64)
            sidecar = store.save(
                image,
                centres,
                mode="replace_automatic",
                source_shape=(24, 36),
                suppressed_xy=np.asarray(((5.0, 6.0),)),
            )
            with np.load(sidecar, allow_pickle=False) as archive:
                fields = {
                    name: archive[name]
                    for name in archive.files
                    if name != "suppressed_xy"
                }
            fields["version"] = np.asarray(1, dtype=np.uint16)
            with sidecar.open("wb") as stream:
                np.savez_compressed(stream, **fields)

            loaded = store.load_if_present(image, (24, 36))
            assert loaded is not None
            self.assertEqual(loaded.mode, "replace_automatic")
            np.testing.assert_array_equal(loaded.centres_xy, centres)
            self.assertEqual(loaded.suppressed_xy.shape, (0, 2))

            fields["version"] = np.asarray(2, dtype=np.uint16)
            with sidecar.open("wb") as stream:
                np.savez_compressed(stream, **fields)
            with self.assertRaisesRegex(InvalidManualSeedCentreArchive, "suppressed_xy"):
                store.load_if_present(image, (24, 36))

    def check_test_each_image_has_independent_source_bound_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._image(root, "first.png", b"first source")
            second = self._image(root, "second.png", b"second source")
            store = ManualSeedCentreStore(root)
            first_centres = np.asarray(((4.0, 5.0),), dtype=np.float32)
            second_centres = np.asarray(
                ((17.25, 8.5), (20.0, 11.0)), dtype=np.float64
            )

            store.save(
                first,
                first_centres,
                mode="augment",
                source_shape=(20, 30),
            )
            store.save(
                second,
                second_centres,
                mode="replace_automatic",
                source_shape=(20, 30),
            )

            self.assertNotEqual(store.path_for(first), store.path_for(second))
            loaded_first = store.load_if_present(first, (20, 30))
            loaded_second = store.load_if_present(second, (20, 30))
            self.assertIsNotNone(loaded_first)
            self.assertIsNotNone(loaded_second)
            assert loaded_first is not None and loaded_second is not None
            np.testing.assert_array_equal(loaded_first.centres_xy, first_centres)
            np.testing.assert_array_equal(loaded_second.centres_xy, second_centres)
            self.assertEqual(loaded_first.mode, "augment")
            self.assertEqual(loaded_second.mode, "replace_automatic")

    def check_test_changed_source_bytes_are_rejected_without_mutating_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = self._image(root, "capture.png", b"original source")
            store = ManualSeedCentreStore(root)
            sidecar = store.save(
                image,
                np.asarray(((4.0, 5.0),), dtype=np.float64),
                mode="augment",
                source_shape=(20, 30),
            )
            before = sidecar.read_bytes()
            image.write_bytes(b"changed source bytes")

            with self.assertRaises(ManualSeedCentreFingerprintMismatch):
                store.load_if_present(image, (20, 30))

            self.assertEqual(sidecar.read_bytes(), before)

    def check_test_same_named_external_images_do_not_share_a_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            project = temporary_root / "project"
            first = temporary_root / "one" / "capture.png"
            second = temporary_root / "two" / "capture.png"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            store = ManualSeedCentreStore(project)

            self.assertNotEqual(store.path_for(first), store.path_for(second))

    def check_test_invalid_saved_or_loaded_coordinates_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = self._image(root, "capture.png", b"source")
            store = ManualSeedCentreStore(root)
            with self.assertRaises(ManualSeedCentreStoreError):
                store.save(
                    image,
                    np.asarray(((3, 4),), dtype=np.int32),
                    mode="augment",
                    source_shape=(20, 30),
                )
            with self.assertRaises(ManualSeedCentreStoreError):
                store.save(
                    image,
                    np.asarray(((30.0, 4.0),), dtype=np.float64),
                    mode="augment",
                    source_shape=(20, 30),
                )

            store.save(
                image,
                np.asarray(((3.0, 4.0),), dtype=np.float64),
                mode="augment",
                source_shape=(20, 30),
            )
            with self.assertRaises(InvalidManualSeedCentreArchive):
                store.load_if_present(image, (21, 30))

    def test_workflow_lifecycle(self):
        """4 related ManualSeedCentreStoreTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_round_trip_preserves_exact_source_floats_mode_and_empty_replace)
        self.scenario(self.check_test_version_one_sidecar_loads_without_suppressed_centres)
        self.scenario(self.check_test_each_image_has_independent_source_bound_state)
        self.scenario(self.check_test_same_named_external_images_do_not_share_a_sidecar)

    def test_validation_and_recovery(self):
        """2 related ManualSeedCentreStoreTests scenarios, each with private fixture state."""
        self.scenario(self.check_test_changed_source_bytes_are_rejected_without_mutating_sidecar)
        self.scenario(self.check_test_invalid_saved_or_loaded_coordinates_are_rejected)


if __name__ == "__main__":
    unittest.main()
