from __future__ import annotations

import copy
from hashlib import sha256
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.persistence import (
    EXTERNAL_ABSOLUTE,
    MANUAL_SEED_CENTRES_SIDECAR,
    PROJECT_ANALYSIS_EXTENSION,
    PROJECT_ANALYSIS_MAX_IMAGES,
    PROJECT_RELATIVE,
    REFERENCE_REGIONS_SIDECAR,
    InvalidProjectAnalysis,
    InvalidManualSeedCentreArchive,
    InvalidReferenceArchive,
    ManualSeedCentreStore,
    ProjectAnalysisIOError,
    ProjectAnalysisDocument,
    ProjectAnalysisStore,
    ProjectFileStatus,
    ProjectImageSpec,
    ProjectUiState,
    ReferenceRegionBundle,
    ReferenceRegionStore,
    analysis_settings_profile_from_graph,
    analysis_settings_profile_to_payload,
    file_sha256,
)
from seedvision.pipeline import build_default_pipeline


class ProjectAnalysisPersistenceTests(unittest.TestCase):
    @staticmethod
    def _source(
        path: Path, shape: tuple[int, int], value: int
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full(shape, value, dtype=np.uint8)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write test image {path}.")
        return path

    @staticmethod
    def _profile():
        return analysis_settings_profile_from_graph(build_default_pipeline())

    @staticmethod
    def _sidecars(root: Path, image: Path, shape: tuple[int, int]) -> tuple[Path, Path]:
        foreground = np.zeros(shape, dtype=bool)
        foreground[1:3, 2:5] = True
        references = ReferenceRegionStore(root).save(
            image,
            ReferenceRegionBundle(shape=shape, foreground=foreground),
        )
        centres = ManualSeedCentreStore(root).save(
            image,
            np.asarray(((3.25, 2.75),), dtype=np.float64),
            mode="augment",
            source_shape=shape,
        )
        return references, centres

    @staticmethod
    def _rewrite_archive_identity(
        source: Path, destination: Path, identity: str
    ) -> Path:
        with np.load(source, allow_pickle=False) as archive:
            payload = {
                name: np.asarray(archive[name]).copy() for name in archive.files
            }
        payload["image_identity"] = np.asarray(identity)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        return destination

    def test_round_trip_is_compact_ordered_and_resolves_relative_and_external_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "workspace"
            internal = self._source(root / "images" / "inside.jpg", (8, 11), 55)
            external = self._source(base / "external" / "outside.jpg", (9, 13), 95)
            references, centres = self._sidecars(root, internal, (8, 11))
            profile = self._profile()
            selected_node = profile.nodes[0].identifier
            ui_state = ProjectUiState(
                node_positions={selected_node: (12.5, -8.25)},
                bundle_cables=True,
                route_around_nodes=True,
                selected_node=selected_node,
                selected_overlay="foreground_mask",
            )
            store = ProjectAnalysisStore(root)

            document = store.capture(
                analysis_settings=profile,
                images=(
                    ProjectImageSpec(internal, (8, 11)),
                    ProjectImageSpec(external, (9, 13)),
                ),
                species="Lupinus mexicanus",
                selected_image=internal,
                ui_state=ui_state,
            )
            destination = store.save(document)
            loaded = store.load()

            self.assertTrue(destination.name.endswith(PROJECT_ANALYSIS_EXTENSION))
            self.assertEqual(len(loaded.images), 2)
            self.assertEqual(loaded.images[0].record.source.location, PROJECT_RELATIVE)
            self.assertEqual(loaded.images[1].record.source.location, EXTERNAL_ABSOLUTE)
            self.assertEqual(loaded.images[0].path, internal.resolve())
            self.assertEqual(loaded.images[1].path, external.resolve())
            self.assertTrue(
                all(item.status is ProjectFileStatus.AVAILABLE for item in loaded.images)
            )
            self.assertEqual(loaded.issues, ())
            self.assertEqual(loaded.document.species, "Lupinus mexicanus")
            self.assertIsNotNone(loaded.selected_image)
            assert loaded.selected_image is not None
            self.assertEqual(loaded.selected_image.path, internal.resolve())
            self.assertEqual(
                analysis_settings_profile_to_payload(loaded.document.analysis_settings),
                analysis_settings_profile_to_payload(profile),
            )
            self.assertEqual(
                loaded.document.ui_state.node_positions[selected_node],
                (12.5, -8.25),
            )
            self.assertEqual(
                {
                    item.reference.kind
                    for item in loaded.images[0].sidecars
                },
                {REFERENCE_REGIONS_SIDECAR, MANUAL_SEED_CENTRES_SIDECAR},
            )
            self.assertEqual(
                loaded.images[0].sidecar(REFERENCE_REGIONS_SIDECAR).path,
                references.resolve(),
            )
            self.assertEqual(
                loaded.images[0].sidecar(MANUAL_SEED_CENTRES_SIDECAR).path,
                centres.resolve(),
            )

            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload),
                {"format", "version", "analysis_settings", "project", "images", "ui"},
            )
            self.assertEqual(
                set(payload["images"][0]),
                {
                    "id",
                    "source",
                    "image_sha256",
                    "source_shape",
                    "sidecars",
                    "biological_context",
                    "capture_group_id",
                },
            )
            encoded = destination.read_text(encoding="utf-8")
            self.assertNotIn("corrected_bgr", encoded)
            self.assertNotIn("occupancy_likelihood", encoded)
            self.assertLess(destination.stat().st_size, 512 * 1024)

    def test_moving_the_complete_workspace_preserves_relative_images_and_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            original = base / "original"
            image = self._source(original / "images" / "capture.jpg", (7, 10), 85)
            self._sidecars(original, image, (7, 10))
            store = ProjectAnalysisStore(original)
            store.capture_and_save(
                analysis_settings=self._profile(),
                images=(ProjectImageSpec(image, (7, 10)),),
                selected_image=image,
            )
            moved = base / "moved"
            shutil.copytree(original, moved)

            loaded = ProjectAnalysisStore(moved).load()

            self.assertEqual(loaded.issues, ())
            self.assertEqual(loaded.images[0].path, moved / "images" / "capture.jpg")
            self.assertTrue(
                all(
                    item.status is ProjectFileStatus.AVAILABLE
                    for item in loaded.images[0].sidecars
                )
            )
            restored = ReferenceRegionStore(moved).load_if_present(
                moved / "images" / "capture.jpg", (7, 10)
            )
            self.assertIsNotNone(restored)

    def test_explicit_project_sidecars_survive_foreign_host_identity_and_path(self) -> None:
        """A master path, source digest, and contents outrank host path casing."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            image = self._source(root / "Images" / "CAPTURE.PNG", (8, 11), 85)
            references, centres = self._sidecars(root, image, (8, 11))
            reference_store = ReferenceRegionStore(root)
            centre_store = ManualSeedCentreStore(root)
            native_identity = reference_store.image_identity(image)
            foreign_identity = (
                native_identity.casefold()
                if native_identity != native_identity.casefold()
                else "project:Images/CAPTURE.PNG"
            )
            self.assertNotEqual(native_identity, foreign_identity)

            # Simulate a sidecar written on a host with the opposite path-case
            # normalization and retained under the exact filename in its master.
            self._rewrite_archive_identity(references, references, foreign_identity)
            self._rewrite_archive_identity(centres, centres, foreign_identity)
            explicit_references = self._rewrite_archive_identity(
                references,
                root / "projects" / "portable" / "legacy.references.npz",
                foreign_identity,
            )
            explicit_centres = self._rewrite_archive_identity(
                centres,
                root / "projects" / "portable" / "legacy.centres.npz",
                foreign_identity,
            )

            # Loose-workspace discovery retains its strict native identity.
            with self.assertRaises(InvalidReferenceArchive):
                reference_store.load_if_present(image, None)
            with self.assertRaises(InvalidManualSeedCentreArchive):
                centre_store.load_if_present(image, (8, 11))

            store = ProjectAnalysisStore(root)
            with (
                patch.object(
                    ReferenceRegionStore,
                    "path_for",
                    side_effect=AssertionError(
                        "explicit project reference path must not be recomputed"
                    ),
                ),
                patch.object(
                    ManualSeedCentreStore,
                    "path_for",
                    side_effect=AssertionError(
                        "explicit project centre path must not be recomputed"
                    ),
                ),
            ):
                destination = store.capture_and_save(
                    analysis_settings=self._profile(),
                    images=(
                        ProjectImageSpec(
                            image,
                            (8, 11),
                            reference_regions_path=explicit_references,
                            manual_seed_centres_path=explicit_centres,
                        ),
                    ),
                )
            loaded = store.load(destination)

            self.assertEqual(loaded.issues, ())
            self.assertEqual(
                loaded.images[0].sidecar(REFERENCE_REGIONS_SIDECAR).path,
                explicit_references.resolve(),
            )
            self.assertEqual(
                loaded.images[0].sidecar(MANUAL_SEED_CENTRES_SIDECAR).path,
                explicit_centres.resolve(),
            )
            restored_references = reference_store.load_project_archive(
                image, explicit_references, None
            )
            restored_centres = centre_store.load_project_archive(
                image, explicit_centres, (8, 11)
            )
            self.assertIsNotNone(restored_references)
            self.assertIsNotNone(restored_centres)

    @unittest.skipIf(os.name == "nt", "requires a case-sensitive source filesystem")
    def test_posix_project_can_store_source_locators_that_differ_only_by_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            upper = self._source(root / "images" / "Seed.png", (6, 7), 45)
            lower = self._source(root / "images" / "seed.png", (6, 7), 95)
            store = ProjectAnalysisStore(root)

            destination = store.capture_and_save(
                analysis_settings=self._profile(),
                images=(
                    ProjectImageSpec(upper, (6, 7)),
                    ProjectImageSpec(lower, (6, 7)),
                ),
            )
            loaded = store.load(destination)

            self.assertEqual(loaded.issues, ())
            self.assertEqual(
                [image.record.source.path for image in loaded.images],
                ["images/Seed.png", "images/seed.png"],
            )

    def test_explicit_sidecar_override_must_be_included_existing_and_root_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "workspace"
            image = self._source(root / "images" / "capture.jpg", (8, 11), 75)
            references, _centres = self._sidecars(root, image, (8, 11))
            store = ProjectAnalysisStore(root)

            with self.assertRaisesRegex(InvalidProjectAnalysis, "while excluding"):
                store.capture(
                    analysis_settings=self._profile(),
                    images=(
                        ProjectImageSpec(
                            image,
                            (8, 11),
                            include_reference_regions=False,
                            reference_regions_path=references,
                        ),
                    ),
                )
            with self.assertRaisesRegex(InvalidProjectAnalysis, "does not exist"):
                store.capture(
                    analysis_settings=self._profile(),
                    images=(
                        ProjectImageSpec(
                            image,
                            (8, 11),
                            reference_regions_path=root / "projects" / "missing.npz",
                            include_manual_seed_centres=False,
                        ),
                    ),
                )
            external = base / "outside.references.npz"
            shutil.copyfile(references, external)
            with self.assertRaisesRegex(InvalidProjectAnalysis, "escapes"):
                store.capture(
                    analysis_settings=self._profile(),
                    images=(
                        ProjectImageSpec(
                            image,
                            (8, 11),
                            reference_regions_path=external,
                            include_manual_seed_centres=False,
                        ),
                    ),
                )

    def test_missing_and_changed_files_are_structured_issues_not_schema_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "workspace"
            internal = self._source(root / "images" / "inside.jpg", (8, 11), 50)
            external = self._source(base / "external" / "outside.jpg", (9, 13), 90)
            _references, centres = self._sidecars(root, internal, (8, 11))
            store = ProjectAnalysisStore(root)
            store.capture_and_save(
                analysis_settings=self._profile(),
                images=(
                    ProjectImageSpec(internal, (8, 11)),
                    ProjectImageSpec(external, (9, 13)),
                ),
                selected_image=external,
            )
            internal.unlink()
            self._source(external, (9, 13), 125)
            centres.write_bytes(b"changed compact sidecar")

            loaded = store.load()

            issue_pairs = {(item.role, item.status) for item in loaded.issues}
            self.assertIn(("source_image", ProjectFileStatus.MISSING), issue_pairs)
            self.assertIn(
                ("source_image", ProjectFileStatus.FINGERPRINT_MISMATCH),
                issue_pairs,
            )
            self.assertIn(
                (
                    MANUAL_SEED_CENTRES_SIDECAR,
                    ProjectFileStatus.FINGERPRINT_MISMATCH,
                ),
                issue_pairs,
            )
            self.assertEqual(sum(item.role == "source_image" for item in loaded.issues), 2)
            self.assertIsNotNone(loaded.selected_image)
            assert loaded.selected_image is not None
            self.assertEqual(
                loaded.selected_image.status,
                ProjectFileStatus.FINGERPRINT_MISMATCH,
            )

            unchecked = store.load(verify_files=False)
            self.assertEqual(unchecked.issues, ())
            self.assertTrue(
                all(item.status is ProjectFileStatus.UNCHECKED for item in unchecked.images)
            )

    def test_strict_schema_rejects_unknown_fields_traversal_duplicates_and_nan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            image = self._source(root / "images" / "capture.jpg", (8, 11), 75)
            self._sidecars(root, image, (8, 11))
            profile = self._profile()
            node_id = profile.nodes[0].identifier
            store = ProjectAnalysisStore(root)
            destination = store.capture_and_save(
                analysis_settings=profile,
                images=(ProjectImageSpec(image, (8, 11)),),
                ui_state=ProjectUiState(node_positions={node_id: (1.0, 2.0)}),
            )
            valid = json.loads(destination.read_text(encoding="utf-8"))

            variants: list[dict[str, object]] = []
            unknown = copy.deepcopy(valid)
            unknown["large_tensor"] = []
            variants.append(unknown)
            traversal = copy.deepcopy(valid)
            traversal["images"][0]["source"]["path"] = "../escape.jpg"
            variants.append(traversal)
            sidecar_traversal = copy.deepcopy(valid)
            sidecar_traversal["images"][0]["sidecars"][
                REFERENCE_REGIONS_SIDECAR
            ]["path"] = "../escape.npz"
            variants.append(sidecar_traversal)
            invalid_profile = copy.deepcopy(valid)
            del invalid_profile["analysis_settings"]["format"]
            variants.append(invalid_profile)
            unknown_node = copy.deepcopy(valid)
            unknown_node["ui"]["node_positions"] = {"not_a_node": [1.0, 2.0]}
            variants.append(unknown_node)
            invalid_version = copy.deepcopy(valid)
            invalid_version["version"] = True
            variants.append(invalid_version)

            for index, variant in enumerate(variants):
                with self.subTest(index=index):
                    candidate = root / "projects" / f"invalid-{index}{PROJECT_ANALYSIS_EXTENSION}"
                    candidate.write_text(json.dumps(variant), encoding="utf-8")
                    with self.assertRaises(InvalidProjectAnalysis):
                        store.load(candidate)

            duplicate = root / "projects" / f"duplicate{PROJECT_ANALYSIS_EXTENSION}"
            duplicate.write_text(
                '{"format":"seedfiddle-project","format":"seedfiddle-project"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(InvalidProjectAnalysis, "Duplicate JSON"):
                store.load(duplicate)

            nonfinite = root / "projects" / f"nan{PROJECT_ANALYSIS_EXTENSION}"
            nan_payload = copy.deepcopy(valid)
            nan_payload["ui"]["node_positions"][node_id] = [float("nan"), 2.0]
            nonfinite.write_text(json.dumps(nan_payload), encoding="utf-8")
            with self.assertRaises(InvalidProjectAnalysis):
                store.load(nonfinite)

    def test_failed_atomic_replace_preserves_the_previous_master_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            image = self._source(root / "images" / "capture.jpg", (8, 11), 75)
            store = ProjectAnalysisStore(root)
            original = store.capture(
                analysis_settings=self._profile(),
                images=(ProjectImageSpec(image, (8, 11)),),
                species="original",
            )
            destination = store.save(original)
            before = destination.read_bytes()
            replacement = store.capture(
                analysis_settings=self._profile(),
                images=(ProjectImageSpec(image, (8, 11)),),
                species="replacement",
            )

            with patch(
                "seedvision.persistence.project_analysis.Path.mkdir",
                side_effect=OSError("directory unavailable"),
            ):
                with self.assertRaises(ProjectAnalysisIOError):
                    store.save(replacement)
            self.assertEqual(destination.read_bytes(), before)

            with patch(
                "seedvision.persistence.project_analysis.os.replace",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaises(ProjectAnalysisIOError):
                    store.save(replacement)

            self.assertEqual(destination.read_bytes(), before)
            self.assertEqual(tuple(destination.parent.glob(f".{destination.name}.*.tmp")), ())
            with self.assertRaises(InvalidProjectAnalysis):
                store.save(replacement, root / "projects" / "wrong.json")

    def test_capture_rejects_a_declared_shape_that_does_not_match_the_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            image = self._source(root / "images" / "capture.jpg", (8, 11), 75)

            with self.assertRaisesRegex(InvalidProjectAnalysis, "decodes as"):
                ProjectAnalysisStore(root).capture(
                    analysis_settings=self._profile(),
                    images=(ProjectImageSpec(image, (9, 11)),),
                )

    def test_resolve_rejects_a_tampered_shape_even_when_source_hash_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            image = self._source(root / "images" / "capture.jpg", (8, 11), 75)
            store = ProjectAnalysisStore(root)
            destination = store.capture_and_save(
                analysis_settings=self._profile(),
                images=(ProjectImageSpec(image, (8, 11)),),
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))
            payload["images"][0]["source_shape"] = [9, 11]
            destination.write_text(json.dumps(payload), encoding="utf-8")

            loaded = store.load(destination)

            self.assertIs(
                loaded.images[0].status,
                ProjectFileStatus.DIMENSION_MISMATCH,
            )
            self.assertEqual(loaded.issues[0].role, "source_image")

    def test_capture_refuses_corrupt_and_stale_image_bound_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            image = self._source(root / "images" / "capture.jpg", (8, 11), 75)
            reference_store = ReferenceRegionStore(root)
            corrupt = reference_store.path_for(image)
            corrupt.parent.mkdir(parents=True, exist_ok=True)
            corrupt.write_bytes(b"not a reference archive")
            store = ProjectAnalysisStore(root)
            with self.assertRaisesRegex(InvalidProjectAnalysis, "invalid reference"):
                store.capture(
                    analysis_settings=self._profile(),
                    images=(
                        ProjectImageSpec(
                            image,
                            (8, 11),
                            include_manual_seed_centres=False,
                        ),
                    ),
                )

            corrupt.unlink()
            _references, _centres = self._sidecars(root, image, (8, 11))
            self._source(image, (8, 11), 105)
            with self.assertRaisesRegex(
                InvalidProjectAnalysis, "manual_seed_centres"
            ):
                store.capture(
                    analysis_settings=self._profile(),
                    images=(
                        ProjectImageSpec(
                            image,
                            (8, 11),
                            include_reference_regions=False,
                        ),
                    ),
                )

    def test_resolve_validates_sidecar_contents_against_the_current_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            image = self._source(root / "images" / "capture.jpg", (8, 11), 75)
            _references, _centres = self._sidecars(root, image, (8, 11))
            store = ProjectAnalysisStore(root)
            destination = store.capture_and_save(
                analysis_settings=self._profile(),
                images=(ProjectImageSpec(image, (8, 11)),),
            )
            # Keep both recorded file hashes internally consistent while making
            # the saved sidecars stale relative to replacement source bytes.
            self._source(image, (8, 11), 115)
            payload = json.loads(destination.read_text(encoding="utf-8"))
            payload["images"][0]["image_sha256"] = file_sha256(image)
            destination.write_text(json.dumps(payload), encoding="utf-8")

            loaded = store.load(destination)

            self.assertIs(loaded.images[0].status, ProjectFileStatus.AVAILABLE)
            self.assertTrue(loaded.images[0].sidecars)
            self.assertTrue(
                all(
                    sidecar.status is ProjectFileStatus.INVALID_SIDECAR
                    for sidecar in loaded.images[0].sidecars
                )
            )

    def test_reference_sidecar_uses_its_corrected_shape_not_raw_source_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            source_shape = (8, 11)
            corrected_shape = (10, 13)
            image = self._source(
                root / "images" / "capture.jpg", source_shape, 75
            )
            foreground = np.zeros(corrected_shape, dtype=bool)
            foreground[2:5, 3:7] = True
            references = ReferenceRegionStore(root).save(
                image,
                ReferenceRegionBundle(
                    shape=corrected_shape,
                    foreground=foreground,
                ),
            )
            ManualSeedCentreStore(root).save(
                image,
                np.asarray(((3.25, 2.75),), dtype=np.float64),
                mode="augment",
                source_shape=source_shape,
            )
            store = ProjectAnalysisStore(root)

            destination = store.capture_and_save(
                analysis_settings=self._profile(),
                images=(ProjectImageSpec(image, source_shape),),
            )
            loaded = store.load(destination)

            self.assertEqual(loaded.issues, ())
            self.assertIs(
                loaded.images[0].sidecar(REFERENCE_REGIONS_SIDECAR).status,
                ProjectFileStatus.AVAILABLE,
            )
            restored = ReferenceRegionStore(root).load_if_present(image, None)
            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.shape, corrected_shape)
            self.assertTrue(references.is_file())

    def test_foreign_absolute_path_is_never_probed_as_a_native_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            image = self._source(root / "images" / "capture.jpg", (8, 11), 75)
            store = ProjectAnalysisStore(root)
            destination = store.capture_and_save(
                analysis_settings=self._profile(),
                images=(ProjectImageSpec(image, (8, 11)),),
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))
            foreign_path = (
                "/var/lib/seed-fiddle/capture.jpg"
                if os.name == "nt"
                else "C:/SeedFiddle/capture.jpg"
            )
            image_payload = payload["images"][0]
            image_payload["source"] = {
                "location": EXTERNAL_ABSOLUTE,
                "path": foreign_path,
            }
            image_payload["id"] = sha256(
                f"{EXTERNAL_ABSOLUTE}\0{foreign_path}".encode("utf-8")
            ).hexdigest()
            destination.write_text(json.dumps(payload), encoding="utf-8")

            with patch(
                "seedvision.persistence.project_analysis.file_sha256"
            ) as fingerprint:
                loaded = store.load(destination)

            fingerprint.assert_not_called()
            self.assertIs(loaded.images[0].status, ProjectFileStatus.UNREADABLE)

    def test_cleanup_failures_are_wrapped_without_masking_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            image = self._source(root / "images" / "capture.jpg", (8, 11), 75)
            store = ProjectAnalysisStore(root)
            document = store.capture(
                analysis_settings=self._profile(),
                images=(ProjectImageSpec(image, (8, 11)),),
            )
            destination = store.save(document)
            before = destination.read_bytes()

            with patch(
                "seedvision.persistence.project_analysis.os.replace",
                side_effect=OSError("replace failed"),
            ), patch.object(Path, "unlink", side_effect=OSError("unlink failed")):
                with self.assertRaisesRegex(
                    ProjectAnalysisIOError, "replace failed.*cleanup also failed"
                ):
                    store.save(document)

            self.assertEqual(destination.read_bytes(), before)
            for leftover in destination.parent.glob(f".{destination.name}.*.tmp"):
                leftover.unlink()

    def test_deep_json_and_oversized_direct_document_fail_as_schema_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            store = ProjectAnalysisStore(root)
            nested = root / "projects" / f"nested{PROJECT_ANALYSIS_EXTENSION}"
            nested.parent.mkdir(parents=True, exist_ok=True)
            nested.write_text("[" * 10_000 + "0" + "]" * 10_000, encoding="utf-8")
            with self.assertRaisesRegex(InvalidProjectAnalysis, "nested too deeply"):
                store.load(nested)

            invalid = ProjectAnalysisDocument(
                analysis_settings=self._profile(),
                images=(None,) * (PROJECT_ANALYSIS_MAX_IMAGES + 1),  # type: ignore[arg-type]
            )
            with self.assertRaisesRegex(InvalidProjectAnalysis, "at most"):
                store.save(invalid)


if __name__ == "__main__":
    unittest.main()
