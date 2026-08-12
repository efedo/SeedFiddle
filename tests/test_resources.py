from __future__ import annotations

import unittest

import numpy as np

from seedvision.cuda import GpuRaster
from seedvision.resources import release_host_caches, resident_bytes


class ResourceOwnershipTests(unittest.TestCase):
    def test_counts_unique_lazy_storage_and_releases_host_mirrors(self) -> None:
        import torch

        tensor = torch.arange(64, dtype=torch.uint8).reshape(1, 1, 8, 8)
        raster = GpuRaster(tensor, numpy_dtype=np.uint8, name="resource test")
        values = {"first": raster, "duplicate": raster}

        before = resident_bytes(values)
        self.assertEqual(before.cpu_tensor, 64)
        self.assertEqual(before.numpy, 0)
        raster.numpy()
        materialized = resident_bytes(values)
        self.assertEqual(materialized.cpu_tensor, 64)
        self.assertEqual(materialized.numpy, 64)
        self.assertEqual(release_host_caches(values), 1)
        self.assertFalse(raster.is_materialized)
        self.assertEqual(resident_bytes(values).numpy, 0)


if __name__ == "__main__":
    unittest.main()
