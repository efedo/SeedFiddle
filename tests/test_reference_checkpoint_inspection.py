from __future__ import annotations

from dataclasses import replace
import shutil
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from seedvision.benchmarking.checkpoint_inspection import (
    InspectionStatus,
    inspect_reference_checkpoint,
)
from seedvision.benchmarking.checkpoints import (
    ExecutionStatus,
    RasterRole,
    load_checkpoint,
    publish_checkpoint,
)
from seedvision.benchmarking.recipes import (
    StageRunRecord,
    StageRunState,
    initial_stage_run,
)
from seedvision.benchmarking.reference_stage import (
    expected_reference_stage_identity,
    run_reference_stage,
)
from seedvision.persistence.reference_graphs import ReferenceGraphAttachment

from tests.test_reference_stage import HAS_REFERENCE_ENV, _fixture, _recipe


@unittest.skipUnless(HAS_REFERENCE_ENV, "the ignored pinned PlantCV reference environment is unavailable")
class ReferenceCheckpointInspectionBridgeTests(unittest.TestCase):
    """Exercise inspection against a real output published by the pinned bridge."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        args, cls.recipe, cls.stage, cls.input_reference, _measurement = _fixture(
            str(cls.root / "bridge"), operation_id="threshold.otsu"
        )
        cls.outcome = run_reference_stage(**args)
        if cls.outcome.checkpoint_reference is None:
            raise AssertionError(f"Pinned PlantCV bridge did not publish: {cls.outcome.run_record}")
        cls.output_reference = cls.outcome.checkpoint_reference
        cls.store = cls.root / "store"
        cls.store.mkdir()
        for source_root, digest in (
            (args["input_root"], cls.input_reference.checkpoint_sha256),
            (args["output_root"], cls.output_reference.checkpoint_sha256),
        ):
            shutil.copytree(source_root / f"{digest}.checkpoint", cls.store / f"{digest}.checkpoint")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()







    def _republished_with_metadata(self, name: str, mutate):
        source = load_checkpoint(self.store, self.output_reference.checkpoint_sha256)
        metadata = _plain(source.metadata)
        mutate(metadata)
        store = self.root / name
        store.mkdir()
        shutil.copytree(
            self.store / f"{self.input_reference.checkpoint_sha256}.checkpoint",
            store / f"{self.input_reference.checkpoint_sha256}.checkpoint",
        )
        reference = publish_checkpoint(
            store,
            source.identity,
            source.arrays,
            execution=source.execution,
            measurement=source.measurement,
            raster_roles=source.raster_roles,
            metadata=metadata,
            allow_cpu_transfer=True,
        )
        record = replace(self.outcome.run_record, checkpoint_sha256=reference.checkpoint_sha256)
        return store, record












def _plain(value):
    if hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


if __name__ == "__main__":
    unittest.main()
