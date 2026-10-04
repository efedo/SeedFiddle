from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from seedvision.benchmarking.recipes import (
    LayoutPosition,
    ParameterContract,
    PortContract,
    RecipeProvenance,
    RecipeProvenanceMode,
    RecipeSpec,
    RecipeStage,
    SourceMapping,
    SourceResolution,
    StageRunRecord,
    StageRunState,
)
from seedvision.persistence.analysis_settings import analysis_settings_profile_from_graph
from seedvision.persistence.project_analysis import (
    InvalidProjectAnalysis,
    ProjectAnalysisDocument,
    ProjectAnalysisStore,
    ProjectImageSpec,
)
from seedvision.persistence import reference_graphs as reference_graphs_module
from seedvision.persistence.reference_graphs import (
    MAX_REFERENCE_GRAPHS,
    MAX_REFERENCE_GRAPH_STAGES,
    REFERENCE_GRAPH_FORMAT,
    ReferenceGraphAttachment,
    ReferenceGraphPersistenceError,
    reference_graphs_from_payload,
    validate_reference_graphs,
)
from seedvision.persistence.snapshot import create_snapshot, restore_snapshot
from seedvision.pipeline import build_default_pipeline


_SHA = "a" * 64


def _recipe(*, recipe_id: str = "graph_demo", description: str = "Test graph metadata only.") -> RecipeSpec:
    source = SourceMapping(
        reference_id="graph_demo_source",
        reference_version="test-only",
        resolution=SourceResolution.PINNED,
        source_path="reference_ops.py",
        source_symbol="threshold_image",
        source_sha256=_SHA,
        license_id="test-only",
    )
    provenance = RecipeProvenance(
        mode=RecipeProvenanceMode.NATIVE_COMPATIBLE,
        backend_id="native_graph_demo",
        backend_version="test-only",
        reference_id="graph_demo_source",
        reference_version="test-only",
        source_manifest_sha256="b" * 64,
        environment_lock_sha256="c" * 64,
        environment_versions={"python": "3.12-test"},
        evidence_status=SourceResolution.PINNED,
    )
    stage = RecipeStage(
        stage_id="threshold",
        title="Threshold",
        operation_id="demo.threshold",
        description="Test-only operation metadata; no execution or science claim.",
        ports=(
            PortContract(
                "image", "input", "Image", "Input image", external=True,
                semantics={"dtype": "uint8", "color_order": "RGB"},
            ),
            PortContract(
                "mask", "output", "Mask", "Output mask",
                semantics={"dtype": "uint8", "foreground": "white"},
            ),
        ),
        parameters=(
            ParameterContract(
                "cutoff", "integer", 127, True, editable=True,
                minimum=0, maximum=255,
            ),
        ),
        source_mappings=(source,),
    )
    return RecipeSpec(
        recipe_id=recipe_id,
        recipe_version="1",
        title="Synthetic graph fixture",
        description=description,
        provenance=provenance,
        stages=(stage,),
        layout=(LayoutPosition("threshold", 34.5, -20.25),),
    )


def _attachment(*, recipe: RecipeSpec | None = None) -> ReferenceGraphAttachment:
    recipe = recipe or _recipe()
    record = StageRunRecord(
        stage_id="threshold",
        recipe_identity=recipe.identity,
        state=StageRunState.EXECUTED,
        checkpoint_sha256="d" * 64,
    )
    return ReferenceGraphAttachment(
        recipe=recipe,
        stage_runs=(record,),
        selected_stage_id="threshold",
        bundle_cables=True,
        route_around_nodes=False,
    )


class ReferenceGraphPersistenceTests(unittest.TestCase):
    def test_attachment_round_trip_retains_recipe_run_and_display_state(self) -> None:
        attachment = _attachment()
        restored = ReferenceGraphAttachment.from_dict(attachment.to_dict())

        self.assertEqual(restored, attachment)
        self.assertEqual(restored.recipe.identity, attachment.recipe.identity)
        self.assertEqual(restored.layout, (LayoutPosition("threshold", 34.5, -20.25),))
        self.assertEqual(restored.selected_stage_id, "threshold")
        self.assertTrue(restored.bundle_cables)
        self.assertFalse(restored.route_around_nodes)
        self.assertEqual(restored.stage_runs[0].checkpoint_sha256, "d" * 64)
        self.assertEqual(restored.checkpoint_data_status, "unavailable")
        self.assertEqual(
            restored.to_dict()["checkpoint_data"]["status"],
            "unavailable",
        )
        with self.assertRaises(TypeError):
            restored.recipe.provenance.environment_versions["python"] = "altered"

    def test_attachment_rejects_missing_extra_duplicate_and_stale_stage_runs(self) -> None:
        attachment = _attachment()
        run = attachment.stage_runs[0]
        with self.assertRaisesRegex(ReferenceGraphPersistenceError, "exactly one"):
            replace(attachment, stage_runs=())
        with self.assertRaisesRegex(ReferenceGraphPersistenceError, "unique"):
            replace(attachment, stage_runs=(run, run))

        stale_identity = replace(run.recipe_identity, settings_sha256="e" * 64)
        stale = replace(run, recipe_identity=stale_identity)
        with self.assertRaisesRegex(ReferenceGraphPersistenceError, "stale or invalid"):
            replace(attachment, stage_runs=(stale,))

        with self.assertRaisesRegex(ReferenceGraphPersistenceError, "Selected reference stage"):
            replace(attachment, selected_stage_id="missing")
        with self.assertRaisesRegex(ReferenceGraphPersistenceError, "must be Boolean"):
            replace(attachment, bundle_cables=1)

    def test_payload_rejects_unknown_fields_false_availability_and_duplicate_graphs(self) -> None:
        payload = _attachment().to_dict()
        original_payload = json.loads(json.dumps(payload))
        self.assertEqual(ReferenceGraphAttachment.from_dict(payload), _attachment())
        self.assertEqual(ReferenceGraphAttachment.from_dict(payload), _attachment())
        self.assertEqual(payload, original_payload)

        with self.assertRaisesRegex(ReferenceGraphPersistenceError, "unknown fields"):
            ReferenceGraphAttachment.from_dict(payload | {"tensor": []})

        claims_arrays = json.loads(json.dumps(payload))
        claims_arrays["checkpoint_data"]["status"] = "available"
        with self.assertRaisesRegex(ReferenceGraphPersistenceError, "cannot claim"):
            ReferenceGraphAttachment.from_dict(claims_arrays)

        with self.assertRaisesRegex(ReferenceGraphPersistenceError, "must be unique"):
            validate_reference_graphs((_attachment(), _attachment()))
        with self.assertRaisesRegex(ReferenceGraphPersistenceError, "at most"):
            validate_reference_graphs((_attachment(),) * (MAX_REFERENCE_GRAPHS + 1))
        with self.assertRaisesRegex(ReferenceGraphPersistenceError, "unknown fields"):
            reference_graphs_from_payload([payload | {"old_arrays": []}])

        excessive_stages = json.loads(json.dumps(payload))
        excessive_stages["recipe"]["stages"] *= MAX_REFERENCE_GRAPH_STAGES + 1
        with self.assertRaisesRegex(ReferenceGraphPersistenceError, "supports at most"):
            ReferenceGraphAttachment.from_dict(excessive_stages)

    def test_constructor_save_and_load_share_the_same_count_and_byte_limits(self) -> None:
        attachment = _attachment()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            store = ProjectAnalysisStore(root)
            document = ProjectAnalysisDocument(
                analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
                images=(),
                reference_graphs=(attachment,),
            )
            existing_project = store.save(document, root / "existing.seedfiddle-project.json")
            never_written = root / "rejected.seedfiddle-project.json"

            run = attachment.stage_runs[0]
            with patch.object(reference_graphs_module, "MAX_REFERENCE_GRAPH_ITEMS", 4), patch.object(
                reference_graphs_module,
                "_canonical_json",
                side_effect=AssertionError("item limit must run before graph encoding"),
            ):
                with self.assertRaisesRegex(ReferenceGraphPersistenceError, "item limit"):
                    ReferenceGraphAttachment(
                        recipe=attachment.recipe,
                        stage_runs=(run,),
                        selected_stage_id="threshold",
                        bundle_cables=True,
                    )

            with patch.object(reference_graphs_module, "MAX_REFERENCE_GRAPH_ITEMS", 4):
                with self.assertRaisesRegex(InvalidProjectAnalysis, "item limit"):
                    store.save(document, never_written)
                with self.assertRaisesRegex(InvalidProjectAnalysis, "item limit"):
                    store.load(existing_project)
            self.assertFalse(never_written.exists())

            with patch.object(reference_graphs_module, "MAX_REFERENCE_GRAPH_BYTES", 128):
                with self.assertRaisesRegex(ReferenceGraphPersistenceError, "byte limit"):
                    ReferenceGraphAttachment(
                        recipe=attachment.recipe,
                        stage_runs=(run,),
                        selected_stage_id="threshold",
                        bundle_cables=True,
                    )
                with self.assertRaisesRegex(InvalidProjectAnalysis, "byte limit"):
                    store.save(document, never_written)
                with self.assertRaisesRegex(InvalidProjectAnalysis, "byte limit"):
                    store.load(existing_project)
            self.assertFalse(never_written.exists())

    def test_malformed_recipe_enum_is_normalized_at_project_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            store = ProjectAnalysisStore(root)
            document = ProjectAnalysisDocument(
                analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
                images=(),
                reference_graphs=(_attachment(),),
            )
            project = store.save(document, root / "invalid-enum.seedfiddle-project.json")
            payload = json.loads(project.read_text(encoding="utf-8"))
            payload["reference_graphs"][0]["recipe"]["provenance"]["mode"] = "bogus"
            project.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(InvalidProjectAnalysis, "reference graph metadata"):
                store.load(project)

    def test_aggregate_graph_metadata_size_is_bounded(self) -> None:
        description = "x" * 1_700_000
        attachments = tuple(
            _attachment(recipe=_recipe(recipe_id=f"graph_{index}", description=description))
            for index in range(5)
        )
        with self.assertRaisesRegex(ReferenceGraphPersistenceError, "aggregate limit"):
            validate_reference_graphs(attachments)

    def test_project_and_snapshot_round_trip_graph_metadata_without_checkpoint_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            image = root / "images" / "capture.png"
            image.parent.mkdir(parents=True)
            self.assertTrue(cv2.imwrite(str(image), np.zeros((12, 16, 3), dtype=np.uint8)))
            attachment = _attachment()
            store = ProjectAnalysisStore(root)
            document = store.capture(
                analysis_settings=analysis_settings_profile_from_graph(build_default_pipeline()),
                images=(ProjectImageSpec(image, (12, 16)),),
                reference_graphs=(attachment,),
            )
            project = store.save(document, root / "project.seedfiddle-project.json")
            loaded = store.load(project)

            self.assertEqual(loaded.document.reference_graphs, (attachment,))
            self.assertEqual(loaded.document.reference_graphs[0].checkpoint_data_status, "unavailable")
            payload = json.loads(project.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], 3)
            self.assertEqual(payload["reference_graphs"][0]["format"], REFERENCE_GRAPH_FORMAT)

            snapshot = create_snapshot(root, project, root / "snapshots")
            restored_project = restore_snapshot(snapshot, root)
            restored = store.load(restored_project)
            self.assertEqual(restored.document.reference_graphs, (attachment,))
            self.assertEqual(restored.document.reference_graphs[0].stage_runs[0].checkpoint_sha256, "d" * 64)
            self.assertEqual(restored.document.reference_graphs[0].checkpoint_data_status, "unavailable")
            self.assertFalse(any(path.name == "arrays.npz" for path in snapshot.rglob("*")))

            image.write_bytes(b"tampered source image")
            with self.assertRaisesRegex(ValueError, "Resolve missing, changed"):
                create_snapshot(root, project, root / "tampered-source-snapshots")

            master = snapshot / "snapshot.seedfiddle-project.json"
            master.write_bytes(master.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "Snapshot verification failed"):
                restore_snapshot(snapshot, root)


if __name__ == "__main__":
    unittest.main()
