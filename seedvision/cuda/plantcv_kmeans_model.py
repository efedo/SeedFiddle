"""Compatibility import for the shared standard-library numeric model codec."""
import sys

from seedvision.model_io import plantcv_kmeans_model as _implementation

# Preserve a single module/global namespace, including legacy patched controls.
sys.modules[__name__] = _implementation
