from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from seedvision.persistence import (
    InstanceMaskFingerprintMismatch,
    InstanceMaskImportError,
    file_sha256,
    load_bundled_instance_mask,
    load_corrected_instance_mask,
)


class InstanceMaskImportTests(unittest.TestCase):
    def test_corrected_png_tiff_and_npz_preserve_sparse_ids_exactly(self) -> None:
        labels = np.zeros((9, 13), dtype=np.uint16)
        labels[1:4, 2:6] = 7
        labels[5:8, 8:12] = 41_003
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = (
                root / "labels.png",
                root / "labels.tiff",
                root / "labels.npz",
            )
            self.assertTrue(cv2.imwrite(str(paths[0]), labels))
            self.assertTrue(cv2.imwrite(str(paths[1]), labels))
            np.savez_compressed(
                paths[2],
                labels=labels,
                coordinate_space=np.asarray("corrected"),
            )

            for path in paths:
                with self.subTest(path=path.suffix):
                    imported = load_corrected_instance_mask(path, labels.shape)
                    self.assertEqual(imported.coordinate_space, "corrected")
                    self.assertEqual(imported.instance_count, 2)
                    self.assertEqual(imported.labels.dtype, np.uint16)
                    self.assertTrue(np.array_equal(imported.labels, labels))
                    self.assertEqual(set(np.unique(imported.labels)), {0, 7, 41_003})

    def test_corrected_import_rejects_shape_type_range_and_corruption(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong_shape = root / "wrong.npz"
            floating = root / "floating.npz"
            negative = root / "negative.npz"
            excessive = root / "excessive.npz"
            colour = root / "colour.png"
            corrupt = root / "corrupt.npz"
            source_space = root / "source.npz"
            np.savez_compressed(wrong_shape, labels=np.zeros((3, 4), np.uint16))
            np.savez_compressed(floating, labels=np.zeros((5, 6), np.float32))
            np.savez_compressed(negative, labels=np.full((5, 6), -1, np.int16))
            np.savez_compressed(excessive, labels=np.full((5, 6), 65_536, np.int32))
            self.assertTrue(cv2.imwrite(str(colour), np.zeros((5, 6, 3), np.uint8)))
            corrupt.write_bytes(b"not an npz archive")
            np.savez_compressed(
                source_space,
                labels=np.zeros((5, 6), np.uint16),
                coordinate_space=np.asarray("source"),
            )

            cases = (
                (wrong_shape, "Nothing was resized or loaded"),
                (floating, "must be integers"),
                (negative, "nonnegative"),
                (excessive, "65,535"),
                (colour, "single-channel"),
                (corrupt, "Could not read"),
                (source_space, "fingerprinted bundled-reference manifest"),
            )
            for path, message in cases:
                with self.subTest(path=path.name):
                    with self.assertRaisesRegex(InstanceMaskImportError, message):
                        load_corrected_instance_mask(path, (5, 6))

    def _write_bundle(
        self,
        root: Path,
    ) -> tuple[Path, np.ndarray, np.ndarray, dict[str, object]]:
        image_path = root / "images" / "capture.png"
        image_path.parent.mkdir(parents=True)
        source_image = np.zeros((8, 10, 3), dtype=np.uint8)
        source_image[..., 1] = 127
        self.assertTrue(cv2.imwrite(str(image_path), source_image))

        labels = np.zeros((8, 10), dtype=np.uint16)
        labels[1:4, 1:4] = 3
        labels[4:7, 6:9] = 50_001
        reference_root = root / "seed-instance-references"
        mask_path = reference_root / "masks" / "capture.instances.png"
        mask_path.parent.mkdir(parents=True)
        self.assertTrue(cv2.imwrite(str(mask_path), labels))
        entry: dict[str, object] = {
            "image_path": "images/capture.png",
            "image_sha256": file_sha256(image_path),
            "width": 10,
            "height": 8,
            "mask_path": "masks/capture.instances.png",
            "mask_sha256": file_sha256(mask_path),
            "coordinate_space": "source",
            "instance_count": 2,
            "dish_instance_count": 1,
            "external_reference_count": 1,
            "reviewed": False,
            "notes": "Machine-prepared starting reference; requires review.",
        }
        payload = {"version": 1, "entries": [entry]}
        (reference_root / "manifest.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return image_path, labels, source_image, entry

    def test_bundled_source_mask_is_fingerprint_bound_and_warped_nearest(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path, source_labels, source_image, _entry = self._write_bundle(root)
            transform = np.asarray(
                ((1.0, 0.0, 2.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
                dtype=np.float64,
            )
            imported = load_bundled_instance_mask(
                root,
                image_path,
                source_shape=source_image.shape[:2],
                corrected_shape=(10, 13),
                source_to_corrected=transform,
            )
            self.assertIsNotNone(imported)
            assert imported is not None
            expected = cv2.warpPerspective(
                source_labels,
                transform,
                (13, 10),
                flags=cv2.INTER_NEAREST,
            )
            self.assertTrue(np.array_equal(imported.labels, expected))
            self.assertEqual(set(np.unique(imported.labels)), {0, 3, 50_001})
            self.assertEqual(imported.coordinate_space, "source->corrected")
            self.assertFalse(imported.reviewed)
            self.assertEqual(imported.external_reference_count, 1)
            self.assertIn("source-sha256=", imported.origin)
            self.assertIn("reviewed=false", imported.origin)

    def test_bundled_mask_rejects_stale_image_mask_and_declared_count(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path, _labels, source_image, entry = self._write_bundle(root)
            manifest = root / "seed-instance-references" / "manifest.json"
            identity = np.eye(3, dtype=np.float64)

            stale_entry = dict(entry, image_sha256="0" * 64)
            manifest.write_text(
                json.dumps({"version": 1, "entries": [stale_entry]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                InstanceMaskFingerprintMismatch, "Nothing was loaded"
            ):
                load_bundled_instance_mask(
                    root,
                    image_path,
                    source_shape=source_image.shape[:2],
                    corrected_shape=source_image.shape[:2],
                    source_to_corrected=identity,
                )

            bad_mask_entry = dict(entry, mask_sha256="f" * 64)
            manifest.write_text(
                json.dumps({"version": 1, "entries": [bad_mask_entry]}),
                encoding="utf-8",
            )
            with self.assertRaises(InstanceMaskFingerprintMismatch):
                load_bundled_instance_mask(
                    root,
                    image_path,
                    source_shape=source_image.shape[:2],
                    corrected_shape=source_image.shape[:2],
                    source_to_corrected=identity,
                )

            bad_count_entry = dict(entry, instance_count=3)
            manifest.write_text(
                json.dumps({"version": 1, "entries": [bad_count_entry]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(InstanceMaskImportError, "not the manifest"):
                load_bundled_instance_mask(
                    root,
                    image_path,
                    source_shape=source_image.shape[:2],
                    corrected_shape=source_image.shape[:2],
                    source_to_corrected=identity,
                )

            missing_entry = dict(entry, mask_path="masks/missing.png")
            manifest.write_text(
                json.dumps({"version": 1, "entries": [missing_entry]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                InstanceMaskImportError, "Could not fingerprint bundled mask"
            ):
                load_bundled_instance_mask(
                    root,
                    image_path,
                    source_shape=source_image.shape[:2],
                    corrected_shape=source_image.shape[:2],
                    source_to_corrected=identity,
                )

    def test_bundled_manifest_requires_every_integrity_field(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path, _labels, source_image, entry = self._write_bundle(root)
            manifest = root / "seed-instance-references" / "manifest.json"
            identity = np.eye(3, dtype=np.float64)

            required_fields = (
                "coordinate_space",
                "mask_sha256",
                "instance_count",
                "dish_instance_count",
                "external_reference_count",
                "reviewed",
                "notes",
            )
            for field in required_fields:
                with self.subTest(field=field):
                    incomplete = dict(entry)
                    incomplete.pop(field)
                    manifest.write_text(
                        json.dumps({"version": 1, "entries": [incomplete]}),
                        encoding="utf-8",
                    )
                    with self.assertRaises(InstanceMaskImportError):
                        load_bundled_instance_mask(
                            root,
                            image_path,
                            source_shape=source_image.shape[:2],
                            corrected_shape=source_image.shape[:2],
                            source_to_corrected=identity,
                        )

    def test_bundled_manifest_requires_partition_counts_to_sum_to_total(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path, _labels, source_image, entry = self._write_bundle(root)
            manifest = root / "seed-instance-references" / "manifest.json"
            inconsistent = dict(entry, dish_instance_count=2)
            manifest.write_text(
                json.dumps({"version": 1, "entries": [inconsistent]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                InstanceMaskImportError,
                "dish_instance_count plus external_reference_count",
            ):
                load_bundled_instance_mask(
                    root,
                    image_path,
                    source_shape=source_image.shape[:2],
                    corrected_shape=source_image.shape[:2],
                    source_to_corrected=np.eye(3, dtype=np.float64),
                )

    def test_bundled_warp_must_preserve_every_nonzero_identifier(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path, _labels, source_image, _entry = self._write_bundle(root)
            transform = np.asarray(
                ((1.0, 0.0, 100.0), (0.0, 1.0, 100.0), (0.0, 0.0, 1.0)),
                dtype=np.float64,
            )
            with self.assertRaisesRegex(
                InstanceMaskImportError, "complete set of nonzero seed IDs"
            ):
                load_bundled_instance_mask(
                    root,
                    image_path,
                    source_shape=source_image.shape[:2],
                    corrected_shape=source_image.shape[:2],
                    source_to_corrected=transform,
                )

    def test_no_current_image_manifest_entry_returns_none(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _image_path, _labels, source_image, _entry = self._write_bundle(root)
            other_path = root / "images" / "other.png"
            self.assertTrue(cv2.imwrite(str(other_path), source_image))
            self.assertIsNone(
                load_bundled_instance_mask(
                    root,
                    other_path,
                    source_shape=source_image.shape[:2],
                    corrected_shape=source_image.shape[:2],
                    source_to_corrected=np.eye(3),
                )
            )


if __name__ == "__main__":
    unittest.main()
