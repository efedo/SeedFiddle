"""Zero-shot Cellpose-SAM spike on the calibrated dish crops exported by the procedural run.

This is an experiment, not an application feature: it runs in a scratch
environment that has ``cellpose>=4`` installed (the project ``.venv`` must not
receive ``opencv-python-headless``). It needs the ``<stem>.dish.png`` and
``<stem>.frame.json`` files written by ``predict_procedural_instances.py`` so both
methods see exactly the same calibrated RGB dish crop, runs the pretrained
``cpsam`` model with each requested diameter hint, drops labels whose centroid
lies outside the detected dish, warps the labels back to the photograph's
source frame and writes ``<variant>/<stem>.instances.png`` (``uint16``).

Example (scratch venv layered on the project venv)::

    <scratch>\\Scripts\\python.exe scripts\\predict_cellpose_instances.py \\
        --frames artifacts/instance-benchmark/predictions/procedural \\
        --output artifacts/instance-benchmark/predictions \\
        --diameters default 48 estimated
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import cv2
import numpy as np

from predict_procedural_instances import labels_to_source


def drop_outside_dish(labels, dish, offset):
    """Zero labels whose centroid lies outside the dish outer circle (crop frame)."""
    labels = np.asarray(labels).astype(np.int32)
    count = int(labels.max(initial=0))
    if count == 0:
        return labels.astype(np.uint16), 0
    ys, xs = np.nonzero(labels)
    ids = labels[ys, xs]
    area = np.bincount(ids, minlength=count + 1).astype(np.float64)
    cx = np.bincount(ids, weights=xs, minlength=count + 1) / np.maximum(area, 1)
    cy = np.bincount(ids, weights=ys, minlength=count + 1) / np.maximum(area, 1)
    centre_x = dish["center_x"] - offset[0]
    centre_y = dish["center_y"] - offset[1]
    inside = np.hypot(cx - centre_x, cy - centre_y) <= dish["outer_radius"]
    inside[0] = False
    keep = np.flatnonzero(inside)
    mapping = np.zeros(count + 1, np.uint16)
    mapping[keep] = np.arange(1, len(keep) + 1, dtype=np.uint16)
    return mapping[labels], int(count - len(keep))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, required=True, help="Directory with <stem>.dish.png and <stem>.frame.json")
    parser.add_argument("--output", type=Path, required=True, help="Parent directory; one subdirectory per diameter variant")
    parser.add_argument("--images", nargs="*", default=None)
    parser.add_argument("--diameters", nargs="+", default=["default", "48"],
                        help="'default' (no rescaling), 'estimated' (pipeline seed diameter) or a pixel value")
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from cellpose import models, version as cellpose_version

    model = models.CellposeModel(gpu=not args.cpu)
    frames = sorted(args.frames.glob("*.frame.json"))
    if not frames:
        raise SystemExit(f"No frame files in {args.frames}; run predict_procedural_instances.py first.")
    for frame_path in frames:
        frame = json.loads(frame_path.read_text(encoding="utf-8"))
        stem = frame["stem"]
        if args.images and stem not in args.images:
            continue
        crop_bgr = cv2.imread(str(args.frames / f"{stem}.dish.png"), cv2.IMREAD_COLOR)
        if crop_bgr is None:
            raise SystemExit(f"Missing dish crop for {stem}.")
        crop_rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
        for spec in args.diameters:
            if spec == "default":
                diameter = None
            elif spec == "estimated":
                diameter = float(frame["estimated_seed_diameter_px"])
            else:
                diameter = float(spec)
            variant = f"cellpose-sam-{spec}"
            out_dir = args.output / variant
            out_dir.mkdir(parents=True, exist_ok=True)
            if not args.cpu:
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            masks, flows, _styles = model.eval(
                crop_rgb, diameter=diameter, channel_axis=2, batch_size=args.batch_size,
                flow_threshold=args.flow_threshold, cellprob_threshold=args.cellprob_threshold)
            elapsed = time.perf_counter() - started
            raw_count = int(np.asarray(masks).max(initial=0))
            kept, dropped = drop_outside_dish(masks, frame["dish_corrected"], frame["crop_offset_xy"])
            source_labels = labels_to_source(kept, frame["crop_shape_hw"], frame["crop_offset_xy"],
                                             frame["source_to_corrected"], frame["source_shape_hw"])
            cv2.imwrite(str(out_dir / f"{stem}.instances.png"), source_labels)
            run = {
                "stem": stem, "method": variant, "model": "cpsam", "cellpose_version": cellpose_version,
                "diameter_hint_px": diameter, "flow_threshold": args.flow_threshold,
                "cellprob_threshold": args.cellprob_threshold, "crop_shape_hw": frame["crop_shape_hw"],
                "elapsed_seconds": round(elapsed, 3), "raw_instance_count": raw_count,
                "dropped_outside_dish": dropped,
                "instance_count": int(len(set(np.unique(source_labels).tolist()) - {0})),
                "peak_cuda_bytes": None if args.cpu else int(torch.cuda.max_memory_allocated()),
                "device": "cpu" if args.cpu else torch.cuda.get_device_name(0),
            }
            (out_dir / f"{stem}.run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
            print(f"{stem} [{variant}]: {run['instance_count']} instances "
                  f"({dropped} outside dish) in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
