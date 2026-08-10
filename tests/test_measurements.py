from __future__ import annotations

import math
import unittest


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(f"Measurement dependencies unavailable: {error}")

    def test_circle_measurements_are_calibrated(self) -> None:
        import cv2
        import numpy as np

        from seedvision.measurement.geometry import measure_mask

        mask = np.zeros((240, 240), dtype=np.uint8)
        cv2.circle(mask, (120, 120), 50, 255, -1)
        result = measure_mask(mask, pixels_per_mm=10.0)
        self.assertAlmostEqual(result.maximum_feret, 10.0, delta=0.15)
        self.assertAlmostEqual(result.minimum_feret, 10.0, delta=0.15)
        self.assertAlmostEqual(result.area, math.pi * 25.0, delta=1.5)
        self.assertGreater(result.circularity, 0.85)
        self.assertGreater(result.solidity, 0.98)
        self.assertEqual(result.length_unit, "mm")
        self.assertEqual(result.area_unit, "mm²")

    def test_empty_mask_is_rejected(self) -> None:
        import numpy as np

        from seedvision.measurement.geometry import measure_mask

        with self.assertRaises(ValueError):
            measure_mask(np.zeros((20, 20), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
