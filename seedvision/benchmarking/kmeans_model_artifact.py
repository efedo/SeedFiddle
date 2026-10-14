"""Compatibility import for the shared standard-library model artifact codec."""
import sys

from seedvision.model_io import kmeans_model_artifact as _implementation

sys.modules[__name__] = _implementation
