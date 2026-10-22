"""Names the calculation engine uses that are not pipeline graph node ids.

The engine (``seedvision.segmentation.baseline``) and the desktop window both
translate a few stage names to the graph node that owns them.  Keeping the
table here, in one place, lets ``tests/test_node_vocabulary.py`` prove that
every name the engine compares against the dirty set is either a graph node
or declared below, so a renamed node cannot silently stop invalidating its
cache.
"""

from __future__ import annotations

# Engine stage or progress name -> graph node id that owns it.
ENGINE_NODE_ALIASES: dict[str, str] = {
    "foreground_segmentation": "background_likelihood",
    "foreground_noise_likelihood": "refined_background_likelihood",
    "colour_reference": "deskew_colour",
    "scale_calibration": "ruler_detection",
    "edge_ridges": "edge_gradients",
    "reference_edge_ridges": "reference_edge_probability",
    "perimeter_background_reference": "layout_detection",
}

# Dirty markers accepted by the engine that name shared inputs rather than
# nodes.  ``raw_images`` and ``reference_layers`` are also connection-source
# names in saved analysis settings (``persistence.analysis_settings``).
INPUT_DIRTY_MARKERS: frozenset[str] = frozenset(
    {"project_image", "raw_images", "reference_layers"}
)


def owning_node(name: str) -> str:
    """Return the graph node id for an engine stage or progress name."""

    return ENGINE_NODE_ALIASES.get(name, name)
