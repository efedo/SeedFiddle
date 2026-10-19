"""Actual public execution stays distinct through archive, QC and replay."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from seedvision.benchmarking.native_recipe import run_native_recipe
from seedvision.export.native_results import export_native_recipe_result, load_native_recipe_result, _validate_payload, NativeResultExportError
from seedvision.export.method_qc import load_method_qc
from seedvision.benchmarking.native_stage_replay import replay_native_stage
from seedvision.pipeline.backend_execution import validate_public_otsu_calculation
from tests.test_pipeline_backend_execution import workflow, RecordingRunner, REFERENCE
from tests.test_native_recipe import fixture, FRAME, PREPARATION, SOURCE


@unittest.skipUnless(torch.cuda.is_available() and REFERENCE.is_file(), 'Requires CUDA and the pinned public runtime')
class PipelineBackendArchiveTests(unittest.TestCase):
    pass
