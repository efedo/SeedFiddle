# Node control audit

Date: 2026-08-26

## Scope and method

This audit covers every control rendered in the right-hand node inspector for
all active and toolbox nodes. The graph currently contains 46 nodes, of which
38 are configurable, with 271 visible controls arranged into 99 operation-
ordered sections.

The audit used four independent checks:

1. Every visible control was matched to its owning computational settings
   field and to a non-constructor read in the analysis implementation.
2. Every field in the calibration, layout, baseline, analysis-layer, advanced,
   procedural, U-Net, and StarDist settings dataclasses was compared with the
   visible control catalogue.
3. Every intentionally private or compatibility-only field was named and given
   a reason in `INTENTIONALLY_UNEXPOSED_SETTINGS`; an unaccounted field now
   fails the test suite.
4. Every configurable node was assigned an exhaustive section catalogue. The
   application now rejects missing, duplicated, unknown, or unsectioned
   controls while building the graph.

## Findings and corrections

- No previously visible control was unused. Each one affects computation or an
  explicitly diagnostic display product, so none was removed.
- Three important procedural settings were being calculated from fixed values
  but were absent from the inspector. The audit exposed reference-surface
  occupancy weight, expected sparse seed area, and expected packed seed-cell
  area. Their existing defaults remain unchanged.
- Misleading display-oriented names were corrected for edge-strength gamma,
  surface-strength gamma, noise-energy gamma, maximum oval candidates, and the
  internal-candidate ridge weight. Their descriptions now name downstream
  analytical effects.
- Calibration controls are now validated together at edit time. Wavelet
  decomposition also participates in analysis-layer cross-validation rather
  than failing later in the worker.
- Settings profile version 8 records the expanded procedural control schema;
  version 7 profiles migrate using the prior fixed defaults.

## Inspector organization

The order within each node follows the calculation: input/source selection,
preparation and scale, fitting or response, post-processing, normalization,
then acceptance or output selection. The section catalogue is summarized
below. Nodes with no settings intentionally have no section heading.

| Node | Controls | Inspector sections |
|---|---:|---|
| Project | 0 | — |
| Species and metadata | 0 | — |
| Ruler detection and scale | 2 | Scale interpretation |
| Deskew and colour balance | 4 | Colour correction; Geometric correction |
| Layout detection | 16 | Rim search; Image-relative bounds and priors; Physical-size prior; Dual-rim pairing; Exterior background reference annulus |
| Hue only | 0 | — |
| Wavelet decomposition | 1 | Decomposition |
| Seed scale estimate | 10 | Isolated-reference search region; Isolated-reference candidates; Annotated-instance override; Fallback |
| Material colour probabilities | 22 | Analysis region; Reference sources; Automatic background selection; Background colour model; Automatic-source authority; Foreground colour model |
| Material noise probabilities | 18 | Class availability; Background frequency bands; Background directional continuation; Background performance; Foreground frequency bands; Foreground directional continuation; Foreground performance |
| Edge gradients | 17 | Gradient sources; Gradient response; Thinned ridges |
| Directional surface darkness gradients | 7 | Surface preparation; Directional sampling; Surface response; Performance |
| Multiscale darkness and colour noise | 7 | Frequency bands; Local energy response; Performance |
| Reference texture prototypes | 8 | Material prototype fitting; Material descriptor and matching; Prototype collage |
| Material evidence decision | 7 | Evidence contributions; Decision calibration; Binary material proposal |
| Reference edges | 21 | Optional conservative net evidence; Training-example selection; Strip descriptor geometry and resolution; Edge prototype fitting and matching; Supported ridge extraction; Reference-edge support normalization |
| Oriented edge traces | 10 | Ridge input and seed scale; Pixel linking; Continuity filtering |
| Seed-boundary confirmation | 21 | Scale and radius hypotheses; Arc evidence; Oval geometry fit; Centre voting; Semantic and polarity evidence; Acceptance and performance |
| Boundary confidence and normals | 1 | Boundary band |
| Grayscale and local lighting | 6 | Illumination flattening; Local shadows and highlights |
| Image-quality diagnostics | 1 | Sensor-noise estimate |
| Procedural seed separation | 36 | Working raster and dish region; Seed-material occupancy; Boundary evidence; Centre likelihood; Automatic marker selection; Candidate geometry limits; Candidate search and combination |
| U-Net + watershed instances | 12 | Model; Tiled inference; Interior and markers; Watershed topography |
| StarDist seed instances | 8 | Model; Tiled inference; Candidate detection; Polygon acceptance |
| Wrinkling likelihood | 1 | Wrinkle response |
| Coat-pattern decomposition | 1 | Pattern decomposition |
| Broad colour probabilities | 1 | Probability calibration |

Toolbox controls use the same contract: Circle candidates is divided into
detection sensitivity, boundary evidence, ring geometry, proposal fusion, and
performance; Distance-peak candidates is divided into distance field and peak
geometry; Instance colour masks is divided into mask extents and proposal
scaling. Each remaining configurable toolbox node has one concise section that
matches its single operation.

## Enforcement

The regression suite now checks section completeness, unique membership,
non-empty control labels and explanations, visible-setting computational reads,
and complete exposure or explicit justification of computational fields. This
makes the audit a maintained graph invariant rather than a static document.
