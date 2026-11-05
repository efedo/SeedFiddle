# Seed-instance method comparison

Current as of 5 November 2026. This page holds the current comparison of the seed-instance methods and what to do next, and is updated in place as evidence arrives. It starts with the two methods trained here; Cellpose-SAM appears only where it affects them. Nothing here is scientific validation, because no method has been scored against human-reviewed masks of real photographs.

Two short names are used throughout, since the current ones blur what is being compared (see [Naming](#naming)):

- **Boundary U-Net:** the `unet_instances` node, currently titled "U-Net + watershed instances".
- **Polygon U-Net:** the `stardist_instances` node, currently titled "StarDist seed instances".

## Current recommendations

1. **Keep both until they are scored on reviewed masks.** Neither theory nor the literature shows that either is the weaker method for touching seeds, and the evidence in this repository cannot settle it.
2. **Expect different strengths and test for them.** The polygon U-Net should separate and count seeds better, especially mottled ones. The boundary U-Net should trace outlines better. Both are predictions, not findings.
3. **Train both properly before comparing them.** Each current checkpoint had only about 160 optimiser steps on 16 repeated tiles, so neither says much about its method. The [training experiment](#a-shared-network) now sets a real step budget, and new training weights the polygon U-Net's radial loss as the paper does and draws new tiles each epoch.
4. **Rename both** so that names state what differs. See [Naming](#naming).
5. **Agree the decision rule before opening a test set.** See [Head-to-head protocol](#head-to-head-protocol).
6. **Retire both if Cellpose-SAM beats them on count and outline.** That follows the [seed-detection evaluation](SEED_DETECTION_EVALUATION.md). The removal also takes the training flow with it, and Cellpose fine-tuning, which would replace it, is not built.
7. **Test a shared network before restructuring the graph.** If both methods stay, they should share one network, provided joint training does not hurt either. The experiment is built and needs the reviewed masks. See [A shared network](#a-shared-network).

## What the two methods are

"U-Net" and "StarDist" are not the same kind of thing. U-Net is a family of network architectures. StarDist is a way of representing objects, normally run on a U-Net. In this repository both models are built on the same [`UNetBackbone`](../seedvision/learning/models.py) class and differ only in what comes after it.

| Layer | Boundary U-Net | Polygon U-Net |
|---|---|---|
| Input | 18 channels: 13 engineered evidence channels and 5 species planes | Same |
| Backbone | Residual U-Net, 24 base channels, 4 downsampling levels, 6,330,480 parameters | Same design and size, trained separately |
| Output heads | Interior, physical boundary, coat-pattern boundary, centre and distance, uncertainty (13,150 parameters) | Object probability, 32 radial distances, uncertainty (8,254 parameters) |
| Training target | Per-pixel maps | Per-pixel whole-object geometry |
| Decoder | Marker-controlled watershed | Polygon non-maximum suppression |

The backbone holds 99.8% of one model's parameters and 99.9% of the other's. The comparison is therefore between output representations and decoders, not between architectures.

### What "U-Net" can mean

U-Net configurations vary in depth, width, block type, normalisation, downsampling, upsampling, skip connections, encoder, input and output heads. These choices matter enough that nnU-Net exists to configure them automatically. Three examples relevant here:

| | Original U-Net (2015) | U-Net in the StarDist paper | This repository |
|---|---|---|---|
| Base channels | 64 | 32 | 24 |
| Downsampling levels | 4 | 3 | 4 now; 3 for the archived synthetic results |
| Block | Two plain convolutions | Two convolutions | Residual, group normalisation, SiLU |
| Input | Image | Image | 13 engineered channels and 5 species planes |

## How they work

**Boundary U-Net.** Each pixel predicts local properties. Centre markers are peaks of the centre map, at least 0.32 seed diameters apart by default. Watershed then floods a landscape of predicted physical boundaries and distance from those markers, discounting predicted coat-pattern edges, and gives one region per marker. The markers set the seed count and the flooding sets the outlines. A missed centre merges two seeds and a false centre splits one, but a gap in a predicted boundary only moves a dividing line.

**Polygon U-Net.** Each pixel predicts whether it is inside a seed and the distance to that seed's edge along 32 fixed directions, which together describe a whole polygon. High-scoring polygons are kept and overlapping lower-scoring ones removed. Overlap removal sets the seed count and the 32 distances set the outlines. It cannot leak two seeds together through a boundary gap, but it can only represent shapes where every edge point is visible from the centre.

## Theoretical basis

Neither method has a guarantee of instance accuracy. The polygon method has the better basis for separating seeds, which is a judgement rather than a measurement.

- **U-Net:** an empirically successful architecture that says nothing about objects. Pixel losses barely register a few wrong pixels at a contact, even when those pixels change the seed count.
- **Watershed:** the strongest mathematics of any part here, since watershed cuts are proven optimal as minimum spanning forests. That optimality is relative to the landscape it floods. Here the landscape's weights are hand-set and not trained, so the rigorous part is not where the errors arise.
- **Star-convex polygons:** an explicit, testable assumption about what a seed is, which compact seeds meet. Every prediction is one whole object, and each pixel must reason about the seed's far edge. The weak points are the greedy overlap removal, which sits outside training, and shapes that break the assumption, such as crescents.

## Published evidence

U-Net is far more successful overall. For touching round objects the record is split, and no independent test of a well-built multi-channel U-Net with watershed against StarDist was found. No comparison on touching seeds was found either.

Average precision on nuclei, from the StarDist authors' own comparison ([Schmidt et al. 2018](https://ar5iv.labs.arxiv.org/html/1806.03535)):

| IoU threshold | StarDist | Mask R-CNN | U-Net, 3-class |
|---|---:|---:|---:|
| 0.50 | **0.864** | 0.832 | 0.806 |
| 0.70 | **0.685** | 0.684 | 0.654 |
| 0.80 | 0.450 | 0.489 | **0.491** |
| 0.90 | 0.119 | 0.189 | **0.226** |

StarDist finds objects better and traces them less well. That U-Net baseline obtains instances from connected components, which is much simpler than the boundary U-Net here.

| Source | Task | Result | Limits |
|---|---|---|---|
| [Caicedo et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6919559/) | 2018 Data Science Bowl, 3,891 teams | First place: 32 U-Net and FPN networks with a boundary class and watershed. The authors credit pre- and post-processing more than architecture for separating the winners | An ensemble, and StarDist was not among the top three |
| [Weigert and Schmidt 2022](https://arxiv.org/abs/2203.02284v2) | CoNIC, histopathology nuclei | Authors report first place with StarDist plus a semantic head; [the organisers](https://grand-challenge.org/blogs/conic-challenge/) list them among the winners | Full leaderboard not checked |
| [Stringer et al. 2020](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v1.full-text) | Cell images, AP at IoU 0.5 | Cellpose 0.88, Mask R-CNN 0.80, StarDist 0.76; on the generalised set 0.79, 0.66, 0.67 | Its authors' own benchmark, preprint figures |
| [Isensee et al. 2021](https://pubmed.ncbi.nlm.nih.gov/33288961/) | nnU-Net, 23 public competition datasets | Matched or beat most specialised methods | Semantic segmentation, not separating touching objects |
| [Corcoran et al. 2023](https://www.frontiersin.org/articles/10.3389/fpls.2023.1120182/full) | Oilseed rape seeds in pods, micro-CT | StarDist-3D, test F1 99.3% on 209 seeds | 3D CT, not photographs of touching seeds |

## Evidence in this repository

**Synthetic images.** These results come from an earlier 3-level backbone on a 40 px canonical seed diameter ([archived results](archive/LEARNED_SEGMENTATION_RESULTS.md)). They show that the software works, not which method is better on real seeds.

| Method and split | Instance F1 | Matched IoU | Relative count error | Boundary F1 |
|---|---:|---:|---:|---:|
| Boundary U-Net, locked test | 1.000 | 0.820 | 0.000 | 0.987 |
| Polygon U-Net, validation | 0.926 | 0.816 | 0.162 | 0.743 |
| Polygon U-Net, locked test | 0.996 | 0.829 | 0.009 | 0.776 |
| Hybrid, validation | 0.998 | 0.810 | 0.005 | 0.966 |

The polygon U-Net's region overlap matches the boundary U-Net's even though its boundary F1 is lower. Polygon resolution does not explain that gap. On a 40 px seed, 32 rays miss the true outline by about 0.1 px if it is round and about 0.4 px at the tip if it is 2:1, against a 2 px scoring tolerance. The cause has not been diagnosed; the [departures below](#departures-from-published-stardist) are candidates.

**Current checkpoints.** Both were trained with the same settings on the same two photographs, one for training and one for validation, with 192 seeds and none reviewed. Neither has been scored. Both predate the radial weighting and the per-epoch tile draw, so retraining would change them.

| | Boundary U-Net | Polygon U-Net |
|---|---|---|
| Epochs run, of 40 | 39, stopped by patience | 40, schedule ended |
| Optimiser steps | 156 | 160 |
| Best epoch | 31 | 39 |
| Training time | 402 s | 78 s |

Neither model has trained for long. One training image at 16 tiles per sample and batch size 4 gives only 4 optimiser steps per epoch, and the cosine schedule takes the learning rate to zero by epoch 40 whatever the dataset size. Each epoch also repeated the same 16 tiles with the same augmentation, so each model saw only 16 crops. The boundary U-Net's patience stop is therefore not evidence of convergence, since its learning rate had already nearly reached zero. Both runs used the same random seed, so they started from the same backbone weights, and after training those weights still correlate at 0.996.

The polygon U-Net's learning rate reached zero while its validation loss was still falling. Its radial loss ended at 1.88, which by estimate corresponds to radii wrong by a factor of 5 to 9 on average, so its polygons have probably not yet learned seed size. Loss values cannot be compared between the two methods. The training times are not a fair comparison either, because machine load was not controlled.

## Departures from published StarDist

The polygon U-Net is not the published method, so its results should not be attributed to StarDist as such.

| | Published StarDist | Here |
|---|---|---|
| Input | The image | 18 engineered and species channels |
| Radial loss | Absolute error weighted by object probability | [Log-domain and relative error](../seedvision/learning/losses.py) with the same weighting, plus an angular curvature term |
| Object loss | Binary cross-entropy | Balanced binary cross-entropy plus Dice |
| Candidates | Pixels above a probability threshold | [Local peaks only](../seedvision/learning/decode.py), radius 0.08 seed diameters by default |
| Candidate score | Object probability | Object probability times (1 - predicted uncertainty) |

Only peak pixels near a seed's centre are used at inference, so edge pixels' polygons are never used. Radial error was originally unweighted, which spent equal effort on those edge pixels. It now follows the paper's weighting, and whether that helps here is untested.

## A shared network

**Status: the training experiment is built and has no result on reviewed masks; the graph is unchanged.** In the application the two methods are separate networks with the same design and the same 15 inputs. They share a design, not weights, so each image takes two forward passes, and training takes two checkpoints and two runs.

**Why share.** One network gives one forward pass, one checkpoint and one training run, and the existing hybrid decoder no longer needs a second pass. Both sets of targets come from the same masks, so no extra labelling is needed. The polygon method's object-probability target is [a copy of the boundary method's distance target](../seedvision/learning/targets.py), so one head could serve both. Adding a head to a shared StarDist backbone has also worked in practice, in the [CoNIC entry](https://arxiv.org/abs/2203.02284v2).

**Graph shape.** Use one network node and two decoder nodes, not a backbone node and head nodes.

| Node | Controls | Output |
|---|---|---|
| Seed U-Net | Checkpoint, tile size, tile overlap | All predicted maps |
| Boundary watershed decoder | Interior and centre thresholds, landscape weights | Seed instances |
| Star-convex polygon decoder | Object threshold, overlap threshold, candidate limits | Seed instances |

Head nodes would add nothing. Heads are 0.1–0.2% of the parameters, have no controls, are only valid with the weights they were trained with, and run with the backbone during tiled prediction. Inference and decoding are already cached separately, so decoder changes would still skip the network.

**Risks of joint training.**

- **Loss balance:** the loss terms have different scales and the radial term learns far more slowly, so one task could dominate the shared weights.
- **Model selection:** a single best epoch may suit one head and not the other.
- **Task conflict:** sharing does not always help ([Standley et al. 2020](https://arxiv.org/abs/1905.07553)). The heads here are only two layers, so almost everything is shared. If the tasks conflict, separate final decoder stages are a middle option; sharing only the encoder would still share about 70% of the parameters.
- **Coupling:** changing either head's loss means retraining both.
- **Comparison:** a joint model shows which output is better from a shared network, not which method is better alone.

**Experiment.** [`--run-training-experiment`](LEARNED_INSTANCE_SEGMENTATION.md#training-experiment) trains three arms with the same seeds and step budget, with no graph changes: boundary alone, polygon alone, and both together. The shared arm keeps both sets of heads and adds the two losses at equal weight. Each output is scored against the test masks with its decoder chosen on validation.

- **Adopt the shared network** if each output does at least as well as its separately trained counterpart, within run-to-run spread.
- **Keep them separate** if joint training clearly hurts either output and separate final decoder stages do not fix it.

The report applies the first rule to F1 at IoU 0.5 and 0.75, but only when the test masks are reviewed and there are at least three seeds. On anything less it opens with NOT VALIDATION and decides nothing.

**Cost if adopted.** The node catalogue changes, so the settings format changes and existing projects need upgrading. Both models must also be retrained, which costs little since neither has trained for long. The proposed names in [Naming](#naming) would become Seed U-Net, Boundary watershed decoder and Star-convex polygon decoder.

## What to cite

Only Cellpose-SAM runs a published method directly, using the [published package and weights](https://www.biorxiv.org/content/10.1101/2025.04.28.651001.full.pdf). The polygon U-Net can be cited as based on one paper. The boundary U-Net has no single source and must be described as this project's own method. Published accuracy does not transfer to either, so both need their own validation.

**Polygon U-Net:** based on [Schmidt et al. 2018](https://ar5iv.labs.arxiv.org/html/1806.03535), which supplies the star-convex polygons, the object probability, the radial distances and the greedy polygon suppression. It is a native PyTorch reimplementation that does not use the `stardist` package, so the [departures above](#departures-from-published-stardist) must be stated alongside the citation.

**Boundary U-Net:** cite each component for what it contributes.

| Component | Source | Relationship |
|---|---|---|
| Encoder, decoder and skip connections | [Ronneberger et al. 2015](https://arxiv.org/abs/1505.04597) | Same family, different configuration |
| Residual blocks in a U-Net | [Zhang et al. 2018](https://arxiv.org/abs/1711.10684) | Same idea, different blocks |
| Predicting contours with objects to split touching ones | [Chen et al. 2016](https://arxiv.org/abs/1604.02677) | Same idea, different network and decoding |
| Regressing a distance map for watershed | [Naylor et al. 2019](https://doi.org/10.1109/TMI.2018.2865709) | Same idea; here one of several inputs to the landscape |
| Learned watershed landscape | [Bai and Urtasun 2017](https://arxiv.org/abs/1611.08303) | Related; it cuts a learned energy map rather than flooding from markers |
| Marker-based watershed | Meyer 1992, "Color image segmentation", through [OpenCV](https://kyamagu.github.io/mexopencv/matlab/watershed.html) | Used directly |
| Coat-pattern boundary head, separation loss and discount | None found | Original here and not validated on real seeds |
| Marker score and landscape weights | None | Original here and hand-set |
| Uncertainty head | None found | Original here; predicts the network's own absolute error |

"None found" reflects a short search, not a systematic review. A publication using the boundary U-Net would need to present the coat-pattern approach as a contribution and validate it as one.

## Naming

**Status: proposal, not yet applied.** The names should say what differs, make the shared backbone visible, and avoid attributing results to a published method that was changed.

| Where | Now | Proposed |
|---|---|---|
| Node title | U-Net + watershed instances | U-Net boundary watershed |
| Node title | StarDist seed instances | U-Net star-convex polygons |
| Training dialog | U-Net + watershed; StarDist | The same two names |
| Learning menu | Train or refine U-Net / StarDist… | Train or refine seed model… |
| Exported method text | StarDist with 32 rays and polygon NMS | U-Net 24/4 star-convex polygons, 32 rays, read from the checkpoint |
| Exported method text | multi-head U-Net with pattern-aware marker-controlled watershed | U-Net 24/4 boundary watershed, read from the checkpoint |
| U-Net uncertainty text | "Aleatoric uncertainty"; "calibrated auxiliary error probabilities" | Predicted error |

"U-Net 24/4" means 24 base channels and 4 downsampling levels. Reading it from the checkpoint means every exported result records the configuration that produced it. The node's details text keeps the word StarDist, so people searching for it still find it.

The uncertainty wording overclaims. The head is trained to predict the network's own absolute error, which is not aleatoric uncertainty, and the model, loss, training, decoder and pipeline code contain no calibration step.

Full wording for methods sections:

- *A method developed for this project, building on Ronneberger et al. 2015, Chen et al. 2016 and Naylor et al. 2019: a residual U-Net (24 base channels, 4 downsampling levels) predicting interior, physical-boundary, coat-pattern-boundary, centre and distance maps, decoded by marker-controlled watershed (Meyer 1992).*
- *The same residual U-Net with a 32-ray star-convex polygon head after StarDist (Schmidt et al. 2018), with a modified loss and peak-only candidates, decoded by polygon non-maximum suppression.*

Keep the internal identifiers (`unet_instances`, `stardist_instances`, the `family` values and the checkpoint filenames). Settings profiles require an exact node catalogue, so changing them forces a format change and a project upgrade, with no benefit to anyone reading the interface.

## Head-to-head protocol

1. **Reference masks.** Correct 2–3 masks by hand, starting from **Learning → Seed outlines → Copy Cellpose-SAM seeds to outlines**, and mark them reviewed. Starting from Cellpose-SAM could favour it where outlines are ambiguous, so judge each outline from the image. Keep at least one complete image group out of all training.
2. **Training.** Run the [training experiment](LEARNED_INSTANCE_SEGMENTATION.md#training-experiment), which trains on the same images and split with a step budget far beyond the current 160 and three random seeds to show run-to-run spread, and includes the shared-network arm from [A shared network](#a-shared-network).
3. **Decoder settings.** Choose them on validation only and freeze them before opening the test image.
4. **Scoring.** Use `evaluate_instances` on the test image for both methods and Cellpose-SAM: F1 at IoU 0.5 and 0.75, matched IoU, count error, splits, merges and boundary F1 at 2 px. The training experiment reports these for its arms; Cellpose-SAM must be scored separately. Check mottled seeds separately, since that is where the methods should differ most.
5. **Decision rule.**
   - One method better on both count and outline by more than the run-to-run spread: remove the other.
   - Results split between count and outline: try the existing hybrid, which uses polygon centres as watershed markers. Keep both only if it beats each alone; it did not on synthetic validation.
   - Cellpose-SAM better than both on count and outline: retire both.

## Open questions

- **Inputs:** both models learn from procedural evidence rather than the image alone. Would a colour-only arm do better?
- **Priority:** which matters more for the intended use, count or outline? Trait measurements depend on outlines, so this could decide a split result.
- **Coat patterns:** does the boundary U-Net's separate coat-pattern head earn its place, or does the polygon shape assumption handle mottled coats without it?

## Sources

- Meyer 1992, "Color image segmentation", International Conference on Image Processing and its Applications, pp. 303–306, [as referenced for OpenCV's `watershed`](https://kyamagu.github.io/mexopencv/matlab/watershed.html)
- [Cousty et al. 2009, Watershed Cuts](https://hal-upec-upem.archives-ouvertes.fr/hal-00622036)
- [Ronneberger et al. 2015, U-Net](https://arxiv.org/abs/1505.04597)
- [Chen et al. 2016, DCAN](https://arxiv.org/abs/1604.02677)
- [Bai and Urtasun 2017, Deep Watershed Transform](https://arxiv.org/abs/1611.08303)
- [Schmidt et al. 2018, Cell Detection with Star-convex Polygons](https://ar5iv.labs.arxiv.org/html/1806.03535)
- [Zhang et al. 2018, Deep Residual U-Net](https://arxiv.org/abs/1711.10684)
- [Caicedo et al. 2019, 2018 Data Science Bowl](https://pmc.ncbi.nlm.nih.gov/articles/PMC6919559/)
- [Naylor et al. 2019, distance map regression](https://doi.org/10.1109/TMI.2018.2865709)
- [Standley et al. 2020, Which Tasks Should Be Learned Together](https://arxiv.org/abs/1905.07553)
- [Stringer et al. 2020, Cellpose preprint](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v1.full-text)
- [Isensee et al. 2021, nnU-Net](https://pubmed.ncbi.nlm.nih.gov/33288961/)
- [Weigert and Schmidt 2022, StarDist at CoNIC](https://arxiv.org/abs/2203.02284v2)
- [Corcoran et al. 2023, seeds in micro-CT](https://www.frontiersin.org/articles/10.3389/fpls.2023.1120182/full)
- [Cellpose-SAM preprint, 2025](https://www.biorxiv.org/content/10.1101/2025.04.28.651001.full.pdf)
