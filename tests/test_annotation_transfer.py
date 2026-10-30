from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from seedvision.persistence import (
    MANUAL_SEED_CENTRES_SIDECAR,
    REFERENCE_REGIONS_SIDECAR,
    AnnotationExportItem,
    AnnotationTransferError,
    ManualSeedCentreFingerprintMismatch,
    ManualSeedCentreStore,
    ReferenceRegionBundle,
    ReferenceRegionStore,
    annotation_folder,
    file_sha256,
    project_annotation_directory,
    read_annotation_bundle,
    write_annotation_bundle,
)
from seedvision.persistence.annotation_transfer import extract_annotation
from seedvision.persistence.project_annotations import (
    copy_annotation_file,
    discard_unsaved_annotations,
    unsaved_annotation_files,
)


def _image(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((36, 48, 3), value, dtype=np.uint8))
    return path.resolve()


class ProjectAnnotationFolderTests(unittest.TestCase):
    def test_folder_sits_beside_the_project_file(self) -> None:
        project = Path("D:/work/analysis2.seedfiddle-project.json")
        directory = project_annotation_directory(project)
        self.assertEqual(directory, Path("D:/work/analysis2.seedfiddle-annotations"))
        self.assertEqual(
            annotation_folder(directory, MANUAL_SEED_CENTRES_SIDECAR, unsaved=True),
            directory / "unsaved" / "manual-seed-centres",
        )
        self.assertEqual(
            annotation_folder(directory, REFERENCE_REGIONS_SIDECAR),
            directory / "reference-regions",
        )

    def test_unsaved_files_are_listed_and_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = _image(root / "images" / "a.png", 90)
            directory = project_annotation_directory(root / "p.seedfiddle-project.json")
            store = ManualSeedCentreStore(
                root, annotation_folder(directory, MANUAL_SEED_CENTRES_SIDECAR, unsaved=True)
            )
            draft = store.save(
                image, np.asarray(((3.0, 4.0),)), mode="augment", source_shape=(36, 48)
            )
            saved = copy_annotation_file(
                draft, ManualSeedCentreStore(
                    root, annotation_folder(directory, MANUAL_SEED_CENTRES_SIDECAR)
                ).path_for(image)
            )
            self.assertEqual(unsaved_annotation_files(directory), (draft,))
            self.assertEqual(saved.read_bytes(), draft.read_bytes())
            discard_unsaved_annotations(directory)
            self.assertEqual(unsaved_annotation_files(directory), ())
            self.assertTrue(saved.is_file())


class AnnotationBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.first = _image(self.root / "images" / "a.png", 90)
        self.second = _image(self.root / "images" / "b.png", 140)
        self.centres = ManualSeedCentreStore(self.root).save(
            self.first,
            np.asarray(((11.5, 13.25), (20.0, 8.0))),
            mode="replace_automatic",
            source_shape=(36, 48),
        )
        self.foreground = np.zeros((36, 48), dtype=bool)
        self.foreground[4:16, 6:20] = True
        self.references = ReferenceRegionStore(self.root).save(
            self.second,
            ReferenceRegionBundle(
                source_to_corrected=np.eye(3), shape=(36, 48), foreground=self.foreground
            ),
        )
        self.bundle = self.root / "out" / "x.seedfiddle-annotations.zip"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _export(self) -> None:
        write_annotation_bundle(
            self.bundle,
            (
                AnnotationExportItem(self.first, MANUAL_SEED_CENTRES_SIDECAR, self.centres),
                AnnotationExportItem(self.second, REFERENCE_REGIONS_SIDECAR, self.references),
            ),
            workspace_root=self.root,
        )

    def _rewrite_manifest(self, change) -> None:
        with zipfile.ZipFile(self.bundle) as bundle:
            members = {name: bundle.read(name) for name in bundle.namelist()}
        manifest = json.loads(members["manifest.json"])
        change(manifest)
        members["manifest.json"] = json.dumps(manifest).encode("utf-8")
        with zipfile.ZipFile(self.bundle, "w") as bundle:
            for name, content in members.items():
                bundle.writestr(name, content)

    def test_round_trip_restores_both_kinds_for_the_same_images(self) -> None:
        self._export()
        entries = read_annotation_bundle(self.bundle)
        self.assertEqual(
            [(entry.image_name, entry.kind) for entry in entries],
            [("a.png", MANUAL_SEED_CENTRES_SIDECAR), ("b.png", REFERENCE_REGIONS_SIDECAR)],
        )
        self.assertEqual(entries[0].image_sha256, file_sha256(self.first))
        self.assertEqual(entries[0].source_shape, (36, 48))
        with tempfile.TemporaryDirectory() as extracted:
            centres = ManualSeedCentreStore(self.root).load_project_archive(
                self.first, extract_annotation(self.bundle, entries[0], extracted), (36, 48)
            )
            references = ReferenceRegionStore(self.root).load_project_archive(
                self.second, extract_annotation(self.bundle, entries[1], extracted), None
            )
        np.testing.assert_allclose(centres.centres_xy, ((11.5, 13.25), (20.0, 8.0)))
        self.assertEqual(centres.mode, "replace_automatic")
        np.testing.assert_array_equal(references.foreground, self.foreground)

    def test_export_rejects_an_archive_drawn_on_another_image(self) -> None:
        with self.assertRaises(ManualSeedCentreFingerprintMismatch):
            write_annotation_bundle(
                self.bundle,
                (AnnotationExportItem(self.second, MANUAL_SEED_CENTRES_SIDECAR, self.centres),),
                workspace_root=self.root,
            )
        self.assertFalse(self.bundle.exists())

    def test_reading_rejects_unsafe_or_ambiguous_manifests(self) -> None:
        self._export()
        cases = {
            "path escape": lambda m: m["entries"][0].update(member="../escape.npz"),
            "missing member": lambda m: m["entries"][0].update(
                member="manual-seed-centres/9999-gone.npz"
            ),
            "kind folder": lambda m: m["entries"][0].update(kind=REFERENCE_REGIONS_SIDECAR),
            "duplicate image": lambda m: m["entries"].append(dict(m["entries"][0])),
            "bad digest": lambda m: m["entries"][0].update(image_sha256="0" * 63),
            "unknown field": lambda m: m.update(extra=True),
            "version": lambda m: m.update(version=99),
        }
        original = self.bundle.read_bytes()
        for name, change in cases.items():
            with self.subTest(name):
                self.bundle.write_bytes(original)
                self._rewrite_manifest(change)
                with self.assertRaises(AnnotationTransferError):
                    read_annotation_bundle(self.bundle)
        self.bundle.write_bytes(b"not a zip file")
        with self.assertRaises(AnnotationTransferError):
            read_annotation_bundle(self.bundle)


if __name__ == "__main__":
    unittest.main()
