"""Saved workspace workflows execute through the ordinary project export route."""

from tests.catalogue_fixtures import CatalogueTestCase
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from seedvision.benchmarking.native_recipe import build_native_seed_recipe
from seedvision.export.native_results import read_native_recipe_inventory
from seedvision.export.project_native_graphs import export_project_native_graphs
from seedvision.export.recipe_batch_qc import read_project_graph_qc
from seedvision.pipeline.workflows import create_workflow, workflow_graph, workflow_from_graph
from tests.test_native_recipe import fixture, selected_recipe


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for workflow execution.')
class PipelineWorkflowBatchTests(CatalogueTestCase):
    pass

