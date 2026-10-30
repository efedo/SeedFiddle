# Seed-instance benchmark

`scripts/benchmark_instances.py` scores predicted seed-instance masks against the
bundled reference masks in `seed-instance-references/` using
`seedvision.learning.metrics.evaluate_instances`: precision, recall and F1 at
IoU 0.5, mean matched IoU, panoptic quality (PQ), signed count error, split and
merge counts, and boundary F1 within 2 px. Reports are written as JSON under
`artifacts/instance-benchmark/` (ignored by git).

## What the references can and cannot support

Every entry in `seed-instance-references/manifest.json` is marked
`reviewed: false`. The masks are machine pre-annotations: rotated-ellipse
support around the legacy procedural proposal centres, edge-snapped and clipped
at contact bisectors. They have not been accepted by a person at full
resolution. Consequently:

- Scores measure **agreement with the pre-annotations**, not accuracy. A method
  can score badly because it is wrong or because the pre-annotation is wrong.
- Numbers are useful **relatively** (method A versus method B on the same
  references) and as a regression guard for the harness itself.
- The legacy proposal stage generated the reference centres, so scoring it
  would be circular. The benchmark therefore scores the `procedural_instances`
  node, which is what the desktop presents for review.
- Each reference also labels the two isolated calibration seeds outside the
  dish (`external_reference_count`). The pipelines only see the dish crop, so
  the harness reports a second `dish_region` score set restricted to instances
  whose centroids fall inside the detected dish polygon.

The harness repeats this caveat in every JSON report and in its console output.
Promote a reference to ground truth only after a human has corrected it in the
desktop and the manifest entry says `reviewed: true`.

## Running the benchmark

All commands run from the repository root with the project `.venv`.

1. Produce procedural masks in source coordinates. This runs the default
   pipeline graph up to `procedural_instances` (the desktop's active-node set)
   and also exports the calibrated RGB dish crop and frame geometry per image:

   ```powershell
   .venv\Scripts\python.exe scripts\predict_procedural_instances.py --output artifacts\instance-benchmark\predictions\procedural
   ```

   The procedural pipeline produces no foreground without seed-colour
   references. By default the script paints eight eroded reference instances as
   the foreground sample (`--foreground-source reference-sample`, recorded in
   `<stem>.run.json`). `--foreground-source painted --project-root <workspace>
   --project <project.json>` uses saved reference-region sidecars where a
   project has them.

2. Score one or more prediction directories. Each `--predictions NAME=DIR`
   directory holds `<stem>.instances.png` `uint16` rasters in source
   coordinates; `--regions` points at the directory holding
   `<stem>.region.json` dish polygons (written in step 1); `--review STEM`
   renders full-resolution side-by-side overlays under `--review-dir`:

   ```powershell
   .venv\Scripts\python.exe scripts\benchmark_instances.py --predictions procedural=artifacts\instance-benchmark\predictions\procedural --regions artifacts\instance-benchmark\predictions\procedural --review IMG_9632c --review IMG_0002c --output artifacts\instance-benchmark\benchmark.json
   ```

   The JSON report lists per-image metrics for `full_frame` and `dish_region`,
   per-method aggregates (pooled TP/FP/FN precision, recall, F1 and PQ; mean
   per-image F1 and PQ; total count error, splits and merges), the run facts
   copied from each `<stem>.run.json`, and the overlay paths.

## Cellpose-SAM spike (experimental, not an application feature)

`scripts/predict_cellpose_instances.py` runs the pretrained Cellpose-SAM
`cpsam` model zero-shot on the dish crops exported in step 1 and writes
source-frame masks for the harness. It is **not** part of the desktop
application and must not be installed into the project `.venv`: `cellpose`
depends on `opencv-python-headless`, which overwrites the `cv2` package the
application pins. Use a scratch virtual environment layered on the project
site-packages instead, for example:

```powershell
.venv\Scripts\python.exe -m venv %TEMP%\cpvenv
echo D:\Programming\SeedFiddle\.venv\Lib\site-packages > %TEMP%\cpvenv\Lib\site-packages\_project_venv.pth
%TEMP%\cpvenv\Scripts\python.exe -m pip install --no-deps cellpose scipy natsort tifffile tqdm fastremap imagecodecs roifile fill-voids segment_anything
%TEMP%\cpvenv\Scripts\python.exe -m pip install --no-deps torchvision --index-url https://download.pytorch.org/whl/cu126
%TEMP%\cpvenv\Scripts\python.exe scripts\predict_cellpose_instances.py --frames artifacts\instance-benchmark\predictions\procedural --output artifacts\instance-benchmark\predictions --diameters default 48 estimated
```

Each diameter hint writes its own `cellpose-sam-<hint>` directory, which is
scored with `--predictions cellpose-sam-48=...` exactly like the procedural
output. The first run downloads the `cpsam` weights (about 1.3 GB) into
`~/.cellpose/models`. Confirm the application is unaffected afterwards with
`python seed_vision.py --diagnostics`.

## Tests

`tests/test_benchmark_instances.py` exercises the harness on synthetic masks:
scoring, dish-region restriction, aggregate pooling, overlay rendering, the
reviewed-reference caveat, and rejection of wrongly framed or non-`uint16`
rasters. Run it with:

```powershell
.venv\Scripts\python.exe -m tests tests.test_benchmark_instances
```
