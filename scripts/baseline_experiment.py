"""Explore foreground masks and watershed markers on the pilot images.

This script is intentionally separate from the application.  It prints compact
diagnostics used to tune the transparent OpenCV baseline before its constants
are promoted into ``seedvision.segmentation``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pilot_probe import detect_dish


def dish_crop(image: np.ndarray, dish: tuple[int, int, int]) -> tuple[np.ndarray, int, int]:
    center_x, center_y, radius = dish
    x0 = max(0, center_x - radius)
    y0 = max(0, center_y - radius)
    x1 = min(image.shape[1], center_x + radius)
    y1 = min(image.shape[0], center_y + radius)
    return image[y0:y1, x0:x1], x0, y0


def foreground_feature(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = crop.shape[:2]
    center = (width // 2, height // 2)
    radius = round(min(height, width) * 0.40)
    valid = np.zeros((height, width), np.uint8)
    cv2.circle(valid, center, radius, 255, -1)

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Use the coolest, least saturated half of the inner dish as a robust
    # estimate of the neutral paper/dish background.
    chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
    lightness = lab[:, :, 0]
    candidates = (valid > 0) & (chroma < np.percentile(chroma[valid > 0], 55))
    candidates &= lightness > np.percentile(lightness[valid > 0], 45)
    background = np.median(lab[candidates], axis=0)
    delta = np.sqrt(
        (lab[:, :, 0] - background[0]) ** 2
        + 1.8 * (lab[:, :, 1] - background[1]) ** 2
        + 1.8 * (lab[:, :, 2] - background[2]) ** 2
    )
    return delta, valid


def component_summary(mask: np.ndarray) -> tuple[int, list[int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = sorted(
        (
            int(stats[index, cv2.CC_STAT_AREA])
            for index in range(1, count)
            if 80 <= stats[index, cv2.CC_STAT_AREA] <= 20_000
        ),
        reverse=True,
    )
    return len(areas), areas[:20]


def inspect_reference(image: np.ndarray) -> float | None:
    height, width = image.shape[:2]
    x0, x1 = round(width * 0.43), round(width * 0.60)
    y0, y1 = round(height * 0.66), round(height * 0.84)
    crop = image[y0:y1, x0:x1]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    background = np.median(lab.reshape(-1, 3), axis=0)
    delta = np.sqrt(
        (lab[:, :, 0] - background[0]) ** 2
        + 1.8 * (lab[:, :, 1] - background[1]) ** 2
        + 1.8 * (lab[:, :, 2] - background[2]) ** 2
    )
    summaries: list[str] = []
    for threshold in (3, 4, 5, 6, 8, 10):
        mask = np.uint8(delta >= threshold) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, areas = component_summary(mask)
        summaries.append(f"{threshold}:{count}/{areas[:4]}")
    print("  references=" + " ".join(summaries))

    mask = np.uint8(delta >= 10) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    details = []
    diameters: list[float] = []
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        if area >= 80:
            details.append(
                (int(area), int(x), int(y), int(component_width), int(component_height),
                 tuple(round(float(v), 1) for v in centroids[index]))
            )
        touches_edge = (
            x <= 1
            or y <= 1
            or x + component_width >= crop.shape[1] - 1
            or y + component_height >= crop.shape[0] - 1
        )
        aspect = max(component_width, component_height) / max(1, min(component_width, component_height))
        if 500 <= area <= crop.size * 0.15 and not touches_edge and aspect <= 2.5:
            diameters.append(float(np.sqrt(4.0 * area / np.pi)))
    print(f"  reference_details={sorted(details, reverse=True)[:8]}")
    diameter = float(np.median(diameters)) if diameters else None
    print(f"  reference_diameter={diameter}")
    return diameter


def local_maxima(
    mask: np.ndarray, seed_diameter: float
) -> tuple[list[tuple[float, float]], np.ndarray]:
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    distance = cv2.GaussianBlur(distance, (0, 0), max(1.0, seed_diameter * 0.025))
    neighborhood = max(5, round(seed_diameter * 0.58))
    if neighborhood % 2 == 0:
        neighborhood += 1
    dilated = cv2.dilate(
        distance,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (neighborhood, neighborhood)),
    )
    peaks = np.uint8(
        (distance >= dilated - 1e-4)
        & (distance >= max(2.0, seed_diameter * 0.14))
    ) * 255
    count, _, stats, centroids = cv2.connectedComponentsWithStats(peaks)
    centers: list[tuple[float, float]] = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] > neighborhood * neighborhood:
            continue
        centers.append((float(centroids[index, 0]), float(centroids[index, 1])))
    return centers, peaks


def inspect(path: Path, output_dir: Path) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {path}")
    dish = detect_dish(image)
    crop, _, _ = dish_crop(image, dish)
    delta, valid = foreground_feature(crop)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{path.name}: dish={dish} crop={crop.shape[1]}x{crop.shape[0]}")
    seed_diameter = inspect_reference(image) or dish[2] * 0.16
    values = np.clip(delta[valid > 0], 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(
        "  feature="
        f"p25={np.percentile(values, 25):.1f} p50={np.percentile(values, 50):.1f} "
        f"p75={np.percentile(values, 75):.1f} otsu={otsu:.1f}"
    )
    adaptive_threshold = max(8.0, otsu * 0.25)
    adaptive_mask = np.uint8((delta >= adaptive_threshold) & (valid > 0)) * 255
    kernel_size = max(3, round(seed_diameter * 0.06))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    adaptive_mask = cv2.morphologyEx(adaptive_mask, cv2.MORPH_OPEN, kernel)
    adaptive_mask = cv2.morphologyEx(adaptive_mask, cv2.MORPH_CLOSE, kernel)
    centers, _ = local_maxima(adaptive_mask, seed_diameter)
    print(
        f"  adaptive_threshold={adaptive_threshold:.1f} seed_diameter={seed_diameter:.1f} "
        f"markers={len(centers)}"
    )
    overlay = crop.copy()
    for center_x, center_y in centers:
        cv2.circle(overlay, (round(center_x), round(center_y)), 5, (0, 0, 255), -1)
        cv2.circle(
            overlay,
            (round(center_x), round(center_y)),
            round(seed_diameter * 0.43),
            (0, 255, 255),
            2,
        )
    cv2.imwrite(str(output_dir / f"{path.stem}_adaptive.png"), adaptive_mask)
    cv2.imwrite(str(output_dir / f"{path.stem}_markers.jpg"), overlay)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    hough_counts: list[str] = []
    hough_overlay = crop.copy()
    for accumulator in (16, 20, 22, 24, 28):
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.1,
            minDist=max(8.0, seed_diameter * 0.58),
            param1=80,
            param2=accumulator,
            minRadius=max(4, round(seed_diameter * 0.22)),
            maxRadius=max(8, round(seed_diameter * 0.62)),
        )
        accepted: list[np.ndarray] = []
        if circles is not None:
            accepted = [
                circle
                for circle in circles[0]
                if np.hypot(circle[0] - crop.shape[1] / 2, circle[1] - crop.shape[0] / 2)
                <= min(crop.shape[:2]) * 0.46
            ]
        hough_counts.append(f"{accumulator}:{len(accepted)}")
        if accumulator == 22:
            for center_x, center_y, circle_radius in accepted:
                cv2.circle(
                    hough_overlay,
                    (round(float(center_x)), round(float(center_y))),
                    round(float(circle_radius)),
                    (255, 0, 255),
                    2,
                )
    print("  hough=" + " ".join(hough_counts))
    cv2.imwrite(str(output_dir / f"{path.stem}_hough.jpg"), hough_overlay)
    for threshold in (4, 6, 8, 10, 12, 15, 20):
        mask = np.uint8((delta >= threshold) & (valid > 0)) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        component_count, areas = component_summary(mask)
        print(f"  threshold={threshold:>2}: components={component_count:>3} areas={areas}")
        if threshold in (6, 10, 15):
            cv2.imwrite(str(output_dir / f"{path.stem}_mask_{threshold}.png"), mask)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/pilot"))
    args = parser.parse_args()
    paths: list[Path] = []
    for path in args.images:
        if path.is_dir():
            paths.extend(
                candidate
                for candidate in sorted(path.iterdir())
                if candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            )
        else:
            paths.append(path)
    for path in paths:
        inspect(path, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
