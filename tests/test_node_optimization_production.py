"""Production integration on two generated images; no accuracy claims."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import torch

from seedvision.pipeline import build_default_pipeline
from seedvision.optimization.runtime import Sample, ProductionEvaluator, production_settings
from seedvision.optimization.search import optimize, apply_report
from seedvision.persistence.reference_regions import SeedInstanceAnnotation
from seedvision.segmentation.baseline import analyze_path


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA production pipeline required')
class ProductionOptimizationTests(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
