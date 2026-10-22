# Evaluation of the seed-detection pipelines

Date: 21 October 2026. Scope: the instance-segmentation methods in this repository compared with current literature, and a recommended direction. This is an engineering assessment, not a scientific validation.

## Summary

No method in the repository has a measured accuracy on a real image, because there are no human-reviewed instance masks. The procedural watershed pipeline is the only method wired into the default workflow; the learned U-Net and StarDist models were trained only on synthetic data from engineered feature channels and have never been scored against real labels. The literature says marker watershed is the known failure mode for touching seeds and that pretrained flow-field models (Cellpose family, now with a SAM transformer backbone) are the strongest fit for densely packed convex objects with internal texture.

Recommendation, in order: build a small reviewed benchmark first; run Cellpose-SAM zero-shot on calibrated RGB crops and score it with the same harness; fine-tune it human-in-the-loop using the existing annotation tools; keep the classical pipeline for calibration, dish, scale and colour, and demote watershed to an explainable fallback. A pretrained transformer-backbone model is in order, but architecture is the second problem; the reviewed test set is the first.

## What the images demand

`images/IMG_0002c.JPG` (6240x4160) shows a dish of about 630 lupin seeds, densely packed and touching, each with a mottled coat and a dark hilum line, photographed through a glass dish with glare, at roughly 48 px per seed after calibration. The strongest edges in such an image are inside seeds (coat pattern, hilum); the seed-to-seed contacts are the weakest edges. Any method that begins from colour likelihood and gradient magnitude is working against the data.

## What the repository has today

| Method | How it works | State |
| --- | --- | --- |
| Procedural (`seedvision/segmentation/procedural.py`) | Hand-built evidence rasters (colour likelihood, edge magnitude, ridges, oriented traces, surface darkening, distance) scored into markers, then OpenCV marker-controlled watershed at a 1600 px working scale | About 45 tuned parameters in `ProceduralInstanceSettings`; the only instance method in the default workflow |
| U-Net + watershed and StarDist (`seedvision/learning/`) | Small native nets (24 base channels, depth 3) trained from scratch on synthetic data; inputs are 13 engineered feature channels from the classical pipeline plus species planes, not RGB | `artifacts/learned/synthetic_hard_v2/unet_b24d3_pattern15.training.json` records best epoch 1 (did not converge); every fixture review in `artifacts/learned/fixture_review/` states that no human instance masks exist and accuracy cannot be estimated |

Labelled data on hand:

- `seed-instance-references/`: 11 `uint16` masks in source coordinates, every entry marked `reviewed: false` in `manifest.json` (machine pre-annotations with documented contact, count and boundary errors).
- `learning-data/`: one sample (`IMG_0002c`), exported from applied annotations.
- `seedvision/learning/metrics.py` already implements `evaluate_instances` (precision, recall, F1 at IoU 0.5, mean matched IoU, panoptic quality, count error, split and merge counts, boundary F1). Nothing real has been scored with it.

The observation that no pipeline matches human seed differentiation is therefore an impression rather than a measurement. A measurement is the cheapest thing to add.

## Against the literature

- Watershed is the documented failure mode. Toda et al. (2020) report that watershed requires per-cultivar parameter search and fails on touching seeds; their Mask R-CNN trained only on synthetic images reached 96% recall and 95% AP on real touching barley seeds. The repository's synthetic-to-real idea is sound, but it uses a far weaker learner (tiny from-scratch nets on engineered features) than the paper, which compounds the domain gap instead of bridging it.
- Flow-field models dominate packed convex objects. Cellpose-style dense flow prediction is the standard for touching blobs with internal texture. Cellpose-SAM (2025) replaces the Cellpose backbone with SAM's pretrained ViT and reports generalisation exceeding inter-human agreement on its benchmarks, with fine-tuning and human-in-the-loop training built in. The transformer matters for pretraining and robustness across species and lighting, not for attention as such.
- Box-based detectors work but cost more labels. Mask R-CNN beat YOLOv8-seg on soybean pods (count MAE 3.57 vs 4.66); both need a box and a mask for every seed, i.e. thousands of annotations. A few-shot LoRA adaptation of SAM for small dense agricultural objects (chickpea pods, 2025) trained about 4% of SAM's parameters, which shows that parameter-efficient fine-tuning transfers to this regime.
- Zero-shot SAM 2 is the wrong tool. Its automatic mask generator is unreliable on dense small objects (class-agnostic benchmarks show large variance across scenarios). Do not expect it to split 600 seeds unprompted.

## Is a transformer-based method in order?

Yes, as a pretrained flow-field model, with the benchmark built first. Ranked:

1. Build the benchmark (days). Human-correct two or three of the existing pre-annotation masks at full resolution (about 800 seeds) and lock them as a held-out set. Score the procedural pipeline with `evaluate_instances`. Until this exists no method, including the current one, can be ranked.
2. Run Cellpose-SAM zero-shot on calibrated RGB dish crops (hours). It fits the project constraints: native PyTorch on CUDA, no server, one pip dependency. Given the geometry it should beat marker watershed on contact splitting out of the box; the benchmark will say by how much.
3. Fine-tune human-in-the-loop. Use the GUI's existing annotation tools to correct predictions and mix in the synthetic generator (the Toda recipe). Fifty corrected images is a reasonable target for a species-general model.
4. Keep the classical pipeline for calibration, dish and ruler detection, colour correction and scale. Demote watershed to an explainable fallback. Retire the from-scratch U-Net/StarDist-on-engineered-features track: feeding a learner the output of the heuristic it is meant to replace caps it at the heuristic's quality.
5. Only if steps 2 and 3 plateau on contacts: a parameter-efficient SAM or DINOv2 fine-tune with a Cellpose-style flow head, trained natively. Same data, more work.

## Follow-up in progress

A separate session, "Seed-instance benchmark and Cellpose-SAM zero-shot spike", is building `scripts/benchmark_instances.py`, scoring the procedural pipeline and zero-shot Cellpose-SAM against the 11 reference masks, and rendering review overlays under `artifacts/instance-benchmark/`. Its numbers are relative to unreviewed pre-annotations and must not be presented as ground truth.

## Sources

- Toda et al. 2020, Training instance segmentation neural network with synthetic datasets for crop seed phenotyping, Communications Biology: https://www.nature.com/articles/s42003-020-0905-5
- Seed Phenotyping on Neural Networks using Domain Randomization and Transfer Learning: https://arxiv.org/pdf/2012.13259
- Cellpose-SAM: superhuman generalization for cellular segmentation (bioRxiv 2025): https://www.biorxiv.org/content/10.1101/2025.04.28.651001.full.pdf
- Revisiting foundation models for cell instance segmentation (MIDL 2026): https://arxiv.org/abs/2603.17845
- Soybean pod and seed counting using unions of deep neural networks (2025): https://arxiv.org/pdf/2502.15286
- Few-shot parameter-efficient SAM adaptation for small dense agricultural objects (2025): https://arxiv.org/html/2509.25805v1
- High-throughput instance segmentation and shape restoration of overlapping vegetable seeds based on sim2real: https://www.sciencedirect.com/science/article/abs/pii/S0263224122016116
