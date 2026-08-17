from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np


class IconAssetTests(unittest.TestCase):
    def test_taskbar_icon_uses_the_square_canvas_at_small_sizes(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "seedvision"
            / "assets"
            / "seed_vision_icon.png"
        )
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        self.assertIsNotNone(image)
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 4)
        self.assertEqual(image.shape[0], image.shape[1])

        alpha = image[:, :, 3]
        opaque_y, opaque_x = np.where(alpha >= 128)
        self.assertGreater(opaque_x.size, 0)
        occupied_width = int(opaque_x.max() - opaque_x.min() + 1)
        occupied_height = int(opaque_y.max() - opaque_y.min() + 1)
        self.assertGreaterEqual(occupied_width / alpha.shape[1], 0.90)
        self.assertGreaterEqual(occupied_height / alpha.shape[0], 0.90)

        small_alpha = cv2.resize(alpha, (32, 32), interpolation=cv2.INTER_AREA)
        self.assertGreaterEqual(int(np.count_nonzero(small_alpha >= 128)), 400)


if __name__ == "__main__":
    unittest.main()
