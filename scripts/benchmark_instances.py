"""Score predicted seed-instance masks against the bundled reference masks.

Each ``--predictions NAME=DIR`` names one method whose directory holds
``<stem>.instances.png`` ``uint16`` label rasters in *source* (original
photograph) coordinates, matching ``seed-instance-references/manifest.json``.
Every image is scored with ``seedvision.learning.metrics.evaluate_instances``
(precision/recall/F1 at IoU 0.5, mean matched IoU, panoptic quality, count
error, split and merge counts, boundary F1). When ``--regions DIR`` names
``<stem>.region.json`` dish polygons, a second score set restricted to
instances whose centroids fall inside the dish is reported as well, because the
references also label the isolated calibration seeds outside the dish that the
pipelines never see.

The references are machine pre-annotations marked ``reviewed: false``. Scores
describe agreement with those pre-annotations, not accuracy against ground
truth; the JSON report repeats this caveat.

``--review STEM`` renders full-resolution side-by-side overlays (reference on
the left, each method on the right) for a human to inspect.

Example::

    .venv\\Scripts\\python.exe scripts\\benchmark_instances.py \\
        --predictions procedural=artifacts/instance-benchmark/predictions/procedural \\
        --predictions cellpose-sam-48=artifacts/instance-benchmark/predictions/cellpose-sam-48 \\
        --regions artifacts/instance-benchmark/predictions/procedural \\
        --review IMG_9632c --review IMG_0002c
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

CAVEAT = (
    "The reference masks in seed-instance-references/ are machine-generated "
    "pre-annotations marked reviewed: false in manifest.json. Every number here "
    "measures agreement with those pre-annotations, not accuracy against human "
    "ground truth; the numbers are relative between methods and can change once "
    "the references are reviewed."
)
SUMMARY_FIELDS = ("f1", "panoptic_quality", "mean_matched_iou", "count_error",
                  "split_instances", "merged_instances", "boundary_f1")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labels(path: Path, shape: tuple[int, int]) -> np.ndarray:
    labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise ValueError(f"Could not read label raster {path}")
    if labels.ndim != 2 or labels.dtype != np.uint16:
        raise ValueError(f"{path} must be a single-channel uint16 raster, got {labels.dtype} {labels.shape}")
    if labels.shape != tuple(shape):
        raise ValueError(f"{path} has shape {labels.shape}; the reference frame is {tuple(shape)}")
    return labels


def restrict_to_region(labels: np.ndarray, polygon: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep instances whose centroid lies inside ``polygon``; return (labels, dropped)."""
    labels = np.asarray(labels).astype(np.int32)
    count = int(labels.max(initial=0))
    if count == 0:
        return labels.astype(np.uint16), 0
    ys, xs = np.nonzero(labels)
    ids = labels[ys, xs]
    area = np.maximum(np.bincount(ids, minlength=count + 1), 1).astype(np.float64)
    cx = np.bincount(ids, weights=xs, minlength=count + 1) / area
    cy = np.bincount(ids, weights=ys, minlength=count + 1) / area
    contour = np.asarray(polygon, np.float32).reshape(-1, 1, 2)
    present = np.flatnonzero(np.bincount(ids, minlength=count + 1))
    keep = [int(i) for i in present
            if cv2.pointPolygonTest(contour, (float(cx[i]), float(cy[i])), False) >= 0]
    mapping = np.zeros(count + 1, np.uint16)
    mapping[keep] = np.arange(1, len(keep) + 1, dtype=np.uint16)
    return mapping[labels], len(present) - len(keep)


def score(reference: np.ndarray, prediction: np.ndarray) -> dict:
    from seedvision.learning.metrics import evaluate_instances
    metrics = evaluate_instances(reference, prediction).to_dict()
    return {key: (round(value, 4) if isinstance(value, float) else value) for key, value in metrics.items()}


def aggregate(rows: list[dict]) -> dict:
    """Pool TP/FP/FN across images and average the per-image scores."""
    if not rows:
        return {}
    tp = sum(r["true_positives"] for r in rows)
    fp = sum(r["false_positives"] for r in rows)
    fn = sum(r["false_negatives"] for r in rows)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    matched = [r["mean_matched_iou"] for r in rows if r["mean_matched_iou"] is not None]
    pooled_iou_sum = sum(r["mean_matched_iou"] * r["true_positives"] for r in rows if r["mean_matched_iou"] is not None)
    return {
        "images": len(rows),
        "true_instances": sum(r["true_instances"] for r in rows),
        "predicted_instances": sum(r["predicted_instances"] for r in rows),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "pooled_precision": round(precision, 4), "pooled_recall": round(recall, 4),
        "pooled_f1": round(2 * precision * recall / max(1e-12, precision + recall), 4),
        "pooled_panoptic_quality": round(pooled_iou_sum / max(1e-12, tp + 0.5 * fp + 0.5 * fn), 4),
        "mean_f1": round(float(np.mean([r["f1"] for r in rows])), 4),
        "mean_panoptic_quality": round(float(np.mean([r["panoptic_quality"] for r in rows])), 4),
        "mean_matched_iou": None if not matched else round(float(np.mean(matched)), 4),
        "total_count_error": sum(r["count_error"] for r in rows),
        "total_absolute_count_error": sum(r["absolute_count_error"] for r in rows),
        "total_split_instances": sum(r["split_instances"] for r in rows),
        "total_merged_instances": sum(r["merged_instances"] for r in rows),
        "mean_boundary_f1": round(float(np.mean([r["boundary_f1"] for r in rows])), 4),
    }


def render_overlay(image: np.ndarray, labels: np.ndarray, title: str, colour: tuple[int, int, int]) -> np.ndarray:
    from seedvision.learning.targets import physical_boundary_mask
    canvas = image.copy()
    boundary = physical_boundary_mask(np.asarray(labels).astype(np.int32), width=3) > 0
    canvas[boundary] = colour
    count = len(set(np.unique(labels).tolist()) - {0})
    scale = max(1.0, image.shape[1] / 1600)
    cv2.putText(canvas, f"{title}: {count} instances", (int(20 * scale), int(60 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6 * scale, (0, 0, 0), int(8 * scale), cv2.LINE_AA)
    cv2.putText(canvas, f"{title}: {count} instances", (int(20 * scale), int(60 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6 * scale, (255, 255, 255), int(3 * scale), cv2.LINE_AA)
    return canvas


def write_review(image_path: Path, stem: str, reference: np.ndarray, predictions: dict[str, np.ndarray],
                 review_dir: Path) -> list[str]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {image_path}")
    review_dir.mkdir(parents=True, exist_ok=True)
    left = render_overlay(image, reference, "reference (unreviewed pre-annotation)", (0, 255, 0))
    written = []
    for name, labels in predictions.items():
        right = render_overlay(image, labels, name, (0, 0, 255))
        side_by_side = np.concatenate([left, right], axis=1)
        target = review_dir / f"{stem}.reference-vs-{name}.jpg"
        cv2.imwrite(str(target), side_by_side, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written.append(str(target))
    return written


def parse_predictions(values: list[str]) -> dict[str, Path]:
    methods = {}
    for value in values:
        name, separator, directory = value.partition("=")
        if not separator or not name or not directory:
            raise argparse.ArgumentTypeError(f"--predictions expects NAME=DIR, got {value!r}")
        methods[name] = Path(directory)
    return methods


def run_benchmark(references: Path, methods: dict[str, Path], *, regions: Path | None = None,
                  images: list[str] | None = None, review: list[str] | None = None,
                  review_dir: Path | None = None, image_root: Path = ROOT) -> dict:
    manifest = json.loads((references / "manifest.json").read_text(encoding="utf-8"))
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caveat": CAVEAT, "iou_threshold": 0.5,
        "reference_directory": str(references), "methods": {name: str(path) for name, path in methods.items()},
        "images": [], "aggregate": {}, "review_overlays": [],
    }
    per_method_rows: dict[str, dict[str, list[dict]]] = {name: {"full_frame": [], "dish_region": []} for name in methods}
    unreviewed = 0
    for entry in manifest["entries"]:
        stem = Path(entry["image_path"]).stem
        if images and stem not in images:
            continue
        shape = (entry["height"], entry["width"])
        mask_path = references / entry["mask_path"]
        if file_sha256(mask_path) != entry["mask_sha256"]:
            raise ValueError(f"Reference mask {mask_path.name} does not match its manifest SHA-256.")
        reference = load_labels(mask_path, shape)
        unreviewed += not entry.get("reviewed", False)
        polygon = None
        if regions is not None and (regions / f"{stem}.region.json").is_file():
            polygon = np.asarray(json.loads((regions / f"{stem}.region.json").read_text(encoding="utf-8"))["polygon_xy"])
        record = {"stem": stem, "image_path": entry["image_path"], "reviewed": bool(entry.get("reviewed", False)),
                  "reference_instances": int(entry["instance_count"]),
                  "reference_dish_instances": int(entry.get("dish_instance_count", entry["instance_count"])),
                  "dish_region_available": polygon is not None, "methods": {}}
        region_reference, reference_dropped = (restrict_to_region(reference, polygon) if polygon is not None else (None, 0))
        if polygon is not None:
            record["reference_instances_outside_dish"] = int(reference_dropped)
        predictions_for_review = {}
        for name, directory in methods.items():
            prediction_path = directory / f"{stem}.instances.png"
            if not prediction_path.is_file():
                record["methods"][name] = {"status": "missing", "path": str(prediction_path)}
                continue
            prediction = load_labels(prediction_path, shape)
            result = {"status": "scored", "path": str(prediction_path), "full_frame": score(reference, prediction)}
            per_method_rows[name]["full_frame"].append(result["full_frame"])
            if polygon is not None:
                region_prediction, prediction_dropped = restrict_to_region(prediction, polygon)
                result["dish_region"] = score(region_reference, region_prediction)
                result["dish_region"]["prediction_instances_outside_dish"] = int(prediction_dropped)
                per_method_rows[name]["dish_region"].append(result["dish_region"])
            run_path = directory / f"{stem}.run.json"
            if run_path.is_file():
                run = json.loads(run_path.read_text(encoding="utf-8"))
                result["run"] = {key: run[key] for key in ("elapsed_seconds", "foreground", "diameter_hint_px", "warnings")
                                 if key in run}
            record["methods"][name] = result
            if review and stem in review:
                predictions_for_review[name] = prediction
        if review and stem in review:
            written = write_review(image_root / entry["image_path"], stem, reference, predictions_for_review,
                                   review_dir or references.parent / "artifacts" / "instance-benchmark" / "review")
            report["review_overlays"].extend(written)
        report["images"].append(record)
    report["unreviewed_reference_images"] = unreviewed
    for name, rows in per_method_rows.items():
        report["aggregate"][name] = {scope: aggregate(items) for scope, items in rows.items() if items}
    return report


def print_summary(report: dict) -> None:
    print(report["caveat"])
    print()
    scopes = ("dish_region", "full_frame")
    for scope in scopes:
        methods = [name for name, values in report["aggregate"].items() if scope in values]
        if not methods:
            continue
        print(f"== {scope} ==")
        header = f"{'image':<12}" + "".join(f"{name:>34}" for name in methods)
        print(header)
        print(f"{'':<12}" + "".join(f"{'F1    PQ   dCount  split merge':>34}" for _ in methods))
        for record in report["images"]:
            cells = []
            for name in methods:
                values = record["methods"].get(name, {}).get(scope)
                if values is None:
                    cells.append(f"{'missing':>34}")
                    continue
                cells.append(f"{values['f1']:>8.3f}{values['panoptic_quality']:>6.3f}{values['count_error']:>8d}"
                             f"{values['split_instances']:>7d}{values['merged_instances']:>6d}   ")
            print(f"{record['stem']:<12}" + "".join(cells))
        cells = []
        for name in methods:
            agg = report["aggregate"][name][scope]
            cells.append(f"{agg['pooled_f1']:>8.3f}{agg['pooled_panoptic_quality']:>6.3f}{agg['total_count_error']:>8d}"
                         f"{agg['total_split_instances']:>7d}{agg['total_merged_instances']:>6d}   ")
        print(f"{'pooled':<12}" + "".join(cells))
        print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--references", type=Path, default=ROOT / "seed-instance-references")
    parser.add_argument("--regions", type=Path, default=None, help="Directory holding <stem>.region.json dish polygons")
    parser.add_argument("--images", nargs="*", default=None, help="Restrict to these image stems")
    parser.add_argument("--review", action="append", default=[], help="Image stem to render side-by-side overlays for")
    parser.add_argument("--review-dir", type=Path, default=ROOT / "artifacts" / "instance-benchmark" / "review")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "instance-benchmark" / "benchmark.json")
    args = parser.parse_args(argv)
    try:
        methods = parse_predictions(args.predictions)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    report = run_benchmark(args.references, methods, regions=args.regions, images=args.images,
                           review=args.review, review_dir=args.review_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report)
    print(f"Report: {args.output}")
    for path in report["review_overlays"]:
        print(f"Overlay: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
