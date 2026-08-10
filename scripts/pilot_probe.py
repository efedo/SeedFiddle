"""Probe dish geometry and simple seed-center baselines on pilot images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def detect_dish(image: np.ndarray) -> tuple[int, int, int]:
    height, width = image.shape[:2]
    scale = min(1.0, 1600.0 / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2.0)
    small_height, small_width = gray.shape
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=small_height * 0.18,
        param1=90,
        param2=36,
        minRadius=int(small_height * 0.16),
        maxRadius=int(small_height * 0.29),
    )
    if circles is None:
        raise RuntimeError("dish circle not found")

    candidates = circles[0]

    def score(circle: np.ndarray) -> float:
        x, y, radius = circle
        expected_x = small_width * 0.58
        expected_y = small_height * 0.40
        expected_radius = small_height * 0.225
        return (
            abs(x - expected_x) / small_width
            + abs(y - expected_y) / small_height
            + 1.5 * abs(radius - expected_radius) / small_height
        )

    x, y, radius = min(candidates, key=score)
    inverse = 1.0 / scale
    return round(x * inverse), round(y * inverse), round(radius * inverse)


def probe_centers(image: np.ndarray, dish: tuple[int, int, int]) -> np.ndarray:
    center_x, center_y, radius = dish
    x0 = max(0, center_x - radius)
    y0 = max(0, center_y - radius)
    x1 = min(image.shape[1], center_x + radius)
    y1 = min(image.shape[0], center_y + radius)
    crop = image[y0:y1, x0:x1]
    scale = min(1.0, 900.0 / max(crop.shape[:2]))
    small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    seed_radius = max(4, round(radius * scale * 0.035))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(7, seed_radius * 1.15),
        param1=80,
        param2=16,
        minRadius=max(4, round(seed_radius * 0.5)),
        maxRadius=max(8, round(seed_radius * 1.7)),
    )
    return np.empty((0, 3), dtype=np.float32) if circles is None else circles[0]


def probe_reference_components(image: np.ndarray) -> list[tuple[int, int, int]]:
    height, width = image.shape[:2]
    x0, x1 = round(width * 0.42), round(width * 0.58)
    y0, y1 = round(height * 0.68), round(height * 0.82)
    crop = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = np.uint8(gray < 222) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    components: list[tuple[int, int, int]] = []
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        if area < 40:
            continue
        if component_width > crop.shape[1] * 0.25 or component_height > crop.shape[0] * 0.5:
            continue
        components.append((int(area), int(component_width), int(component_height)))
    return sorted(components, reverse=True)[:8]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path, nargs="?", default=Path("images"))
    args = parser.parse_args()
    for path in sorted(args.image_dir.glob("*")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        try:
            dish = detect_dish(image)
            centers = probe_centers(image, dish)
            components = probe_reference_components(image)
            print(
                f"{path.name}\tdish={dish}\tcenters={len(centers)}"
                f"\treference_components={components}"
            )
        except RuntimeError as error:
            print(f"{path.name}\tERROR={error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
