Seed Fiddle
This is a scratch file in which Seed Fiddle prompts are drafted and archived.

---

- When an editor node is selected, please also select adjacent nodes (but with a different outline colour; maybe purple?)
- Instance colour masks are not currently working, add an "Enabled" check-box to this node that defaults to disabled for now. Obviously this should also disable all dependent nodes. Come to think of it, also disable by default distance-peak candidates, circle candidates, seed-interior probabilities, and all downstream nodes, since none of these are working either.
- The Layout detection is having a lot of difficulties detecting the upper/outside petri dish border (should be on the exterior edge of this border), but it should be very obvious visually. Try to fix this and verify the fix by looking at the test output yourself.
- The user should be able to overlay nodes if they want. Add a button to a one-line node editor control panel to automatically reposition nodes (which should eliminate overlaps while minimizing edge criss-crossing).
- Add zoom indicators and controls to both the image viewer and node editor.
- Make the application tray icon larger. Actually, start fresh and make it look like the bow of a fiddle being played over a seed. Also rename the application "Seed Fiddle" and get rid of the "-- visual analysis pipeline" from the title.
- For the background and foreground colour b*/a* plots, don't dim the plots according to contour probabilities; this makes it impossible to tell which areas are selected.
- The foreground mask still thinks seed patterns are background. Painting patterns as references does not help except directly under the painted region. Also, you should not ever be "cheating" with the painting by directly forcing the foreground probability under a manually-painted reference region to 1. The foreground calculation should instead be corrected until this occurs naturally (or close to it, anyway).
- For the foreground mask, it seems like painted foreground reference pixels are just being averaged. That's not how it's supposed to work, and that method doesn't work with complex seed patterns (which have both light and dark patches). Reference pixel colours should be accumulated and colour probabilities calculated from individual colour frequencies, not from regional averages.
- Add a reset button to the "Settings" section of the right (detailed) node settings panel, which resets all parameters to default.

---

1. For automatic background colour reference selection from the dish-surrounding ring (enabled via "Keep perimeter-matched source"), only include dish-surrounding regions with colour close to the median ring colour (so that e.g. parts of the ring that intersect the colour calibration card are excluded).
2. Use a colour other than light gray for painting "Other" regions: currently very hard to see where has been painted as "Other"
3. For the Reference texture prototypes, the "yellow centre line" and the "two polarity-neutral cyan side strips" do not seem to be aligned correctly with edges in prototype images.
4. Expose parameters for the instance-derived edge probabilities node, and then add instance-based parameter optimization for physical edge probability and non-physical edge probability.
5. Add a thinned net-physical edge ridge overlay.
6. Procedural centre likelihoods are one of the worst part of the prediction model at the moment (although physical boundary costs are also pretty bad compared to plain physical edge probabilities or net physical edge probabilities), and are throwing everything else off. It would not be possible to do much worse by just throwing darts at an image. In particular, using blurred flattened grayscale images could work much better for centre assignment with many seed coat patterns.
7. Procedural seed instances often generate improbable Frankentein seeds whose shape and size should both be prohibited by any reasonable model. It is not very clear from the settings exactly how these final instance shapes are being determined.
8. For seed instance annotation, it's not clear whether the "Apply + save" button saves the annotation painting locally (within the open project) or globally, or whether File > Save Project is still required to permanent save the painted refence instances.
9. For seed instance annotation, smart fill should probably be improved to not just blindly flood through narrow gaps in edges. This is particularly egregeous with oriented edge traces that are almost fully complete minus a couple of hole pixels. Maybe a pressure/waterfall model?
10. "Fit procedural settings" should only apply within a given project for now (not globally in the broad sense; unclear what is meant by the current warning box)
11. Add Directional surface darkness gradients back to the default graph, and enable them by default. Visually, they clearly extract useful information as many seeds can be readily distinguished by eye in the surface darkening direction overlay (and to a lesser extent in the surface darkening magnitude and surface lightening direction plots). They should theoretically be extracting additional information beyond that extracted by directional edge gradients. The problem is that I have no idea what to do with this information or how to incorporate it into seed edge finding or centre positioning.
12. The outward soft half-life for the shape fill tool needs to start working inside the oval to account for seeds that have divets/are not perfectly oval. Also something is wrong with the current half-life implementation: Even setting it to 0.005 x preferred diameter makes no difference, it floods way outside of the oval.
13. What exactly does the trace continuity overlay show? It seems like it could be quite useful.
14. Changing the "Maximum edge prototypes / class" seems to disable the buttons material reference and seed annotation painting, as well as for saving the project via the file menu.

---

- The species and metadata node should be in the same tier as and not depend on the raw images node
- The colour-card swatches node can be merged into the Deskew and colour balance node
- The ruler detection and absolute ruler scale nodes can be merged into a "Ruler detection and scale" node
- Combine the thinned edge ridges node with the edge gradients node
- Combine the thinned reference edge ridge node with the instance-derived edge probabilities node (which you can rename the reference edge node)
- Various nodes have outputs that are unused and no longer appear to correspond to extant overlays, with some apparently having been renamed but the previous entry not deleted (e.g.  "Thinned edge ridges"). Clean all of these up, but be sure not to remove any extant overlays.
- Oriented edge traces is working very poorly with normalized net reference ridges; even though the original ridges are clean, oriented edge traces seems to produce double edges everywhere (see image); why? Can this be fixed?
- Why do material noise probabilities depend on material colour probabilities?
- Instead of separating physical vs. non-physical prototype probability from edge strength, you seem to have instead simply moved the previous edge-strength depended amalgamations to the upstream node? Or am I misinterpreting?
- The shared edge magnitude overlay doesn't seem to change when the gradient method is changed
- Changing the gradient method for the edge gradient mode results in the whole pipeline (including upstream nodes) being recalculated. This should never occur, you are supposed to be caching nodes and only recomputing the portion fo the graph that is required; remember?
- Is the "Seed-Interior probability" node just a worse version of the "Resolved Seed probability" overlay? If so, remove.
- Also, is the "Seed-material likelihood" in the Procedural seed separation node just a clone of the "Resolved Seed probability" overlay? If so, remove it.
- It looks like Procedural seed separation is somehow cheating hard during fitting of settings, since all manually annotated seeds are identified and have near-identical outlines with automatic detection compared with their manually-painted areas. This _must_ be fixed and you must add tests to ensure that none of the fitting or learning models cheat like this.

---

- Add an overlay to the appropriate node that shows the most likely edge ovals across the petri dish
- Create an inferred seed centre probability overlay based on the calculated likelihood that a given pixel is at the centre of an oval fitting the curved edges detected in the image. You may have to calculate in the reverse direction (starting with detected curved edges) for this to be computationally-feasible


---

1. Does the Boundary confidence and normals node still do anything? It's overlays appear to be solid black and solid blue.
2. For the "Material evidence decision node": i. The foreground binary proposal mask (first overlay) is very bad and I'm not clear where it is coming from; can this be removed? ii. "Background/Other subtype ambiguity" and "Unknown Non-seed subtype" overlays appear to currently be empty.
3. Remove the separate perimeter background referencde node and move its overlay to the layout detection node
4. For the Reference edge node, remove the "Locally" from "Locally normalized net physical edge". Remove the "Thinned reference edge ridge" overlay.
5. The idea was that normalization was supposed to enhance weak connected edges in the Reference edge node. Instead, edges appear weaker in the "Thinned normalized net-physical ridge" overlay compared with the "Thinned net-physical ridge" overlay.

---

1. Remove the unused "Boundary confidence and normals" node from the Overlay selection list. There was supposed to be a test so that this is always kept up to date and only contains active nodes.
2. 

- Remove small spots: identify small groups of dark pixels in flattened grayscale surrounded by light pixels; fill in with light pixels
- Don't just trace any edge, try to find edges with continuin colour (on one side at least), curve
- Seed-boundary confirmation needs to identify the edge types it is using
- Should also incorporate grayscale evidence where useful
- Overreach penalty for procedural seed separation fitting needs to lay out all of its parameters
- Directly matching seeds?

---

Fix the other-noise classifier with the suggested appropriate redesign:
- Keep Background and Foreground comparison distributions separate and class-balanced.
- Let only Foreground act as the direct contrast class for Other’s top-level evidence, preserving legitimate Background/Other overlap.
- Add oriented/multiscale texture descriptors or multiple prototypes per class.
- Add invariance tests ensuring that duplicating Background samples cannot overwhelm Foreground negatives.
- Correct the stale inspector explanation.

After you have finshed, verify that this same error mode does not affect foreground or background noise classifiers.

---

- Although despeckled flattened grayscale to be run multiple iterations via a control (but default to a single iteration)


- For reference edges, are the calculations currently correctly using true edges and then weighting them by the physical vs. non-physical edge prototype probability? The net physical edge probability overlay has very wide edge strips if this is the case. 

The new regression test exposed the exact mechanism behind the visual complaint: a strong ridge inside the normalization window raised the local RMS enough that the immediately adjacent weak continuation stayed below the hysteresis threshold. I’m correcting that boundary case explicitly—the gain will still use the robust local envelope, but a strong neighbour can no longer suppress the gain assigned to the current weak edge pixel.

Reference edges: Renamed the overlay to “Normalized net physical edge” and removed the physical-only “Thinned reference edge ridge” viewer overlay. Its internal tensor remains available to annotation tools and trace sources.


--

1. Procedural centre likelihoods are very bad with the Procedural seed separation node compared with the centre votes from the seed-boundary confirmation node (although I'm not clear on the difference between the centre vote layouts in that mode). Can you fix this?
2. Can you show an overlay underreach/overreach pixels for each procedural seed candidate that has a matching reference control, with intensity of shading equal to the cost of that pixel (and maybe different colours for underreach and overreach)?
3. I have no idea what the attached error message affecting project saving even means.

Did this flaw only affect the "Other" class or were other classes also affected?

---

- For the detected ruler, can you also show the imperial measurement line?
- For the Reviewed seed measurements overlay, it is not clear which border of the reference seed (inner grey, yellow, or outer grey) correspondes to the measured width. Also there is supposed to be an error associated with this measurement.
- The Material references/Annotate seed instances control should be made into a properly control widget; resizable with a drag-able top bar (no "Move painting controls" button)
- The Annotate seed instances control should have an option to select an existing seed annotation. Also, the "Show selected seed only" control should be immediately under the seed selector spinner. And there are huge vertical expanses of white space below the "Brush radius" control and below the "Freehand interior painting..." label. Also, under the seed selector spinner, there should be a note in red that says "empty" if a numbered seed instance is currently empty.
- For reviewed seed measurements, even in the absence of fully annotated nodes, you should be able to use the seed width perpendicular to the maximum width axis to obtain an initial minimum ovality estimate