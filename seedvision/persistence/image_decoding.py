"""Path-safe raster decoding shared by the analysis, persistence and learning code.

``cv2.imread`` takes a byte string path and, on Windows, cannot open files whose
path contains characters outside the active code page; it silently returns
``None``.  Reading the bytes with NumPy and decoding them with ``cv2.imdecode``
behaves identically for ordinary paths and also works for non-ASCII ones.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path | str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Decode an image file like ``cv2.imread`` but without its path limitations.

    Returns ``None`` when the file is missing, unreadable or not a decodable
    image, matching ``cv2.imread`` so call sites keep their existing checks.
    """

    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    try:
        return cv2.imdecode(encoded, flags)
    except cv2.error:
        return None
