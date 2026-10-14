"""Detached facts from the inputs actually supplied to native analysis."""
from dataclasses import asdict
from hashlib import sha256
import json

import numpy as np


def analysis_input_facts(captured):
    if not isinstance(captured, str):
        raise ValueError('Captured analysis inputs must be immutable JSON text.')
    facts = json.loads(captured)
    if not isinstance(facts, dict) or facts.get('contract') != 'seedfiddle.analysis-inputs-v1':
        raise ValueError('Captured analysis inputs have an unsupported contract.')
    return facts


def _array_fact(value):
    if value is None:
        return None
    # These are the existing CPU annotation inputs, never CUDA products.
    array = np.ascontiguousarray(value)
    return {'shape': list(array.shape), 'dtype': array.dtype.str,
            'sha256': sha256(array.tobytes()).hexdigest()}


def capture_analysis_inputs(*, masks, annotations, species, biological_context,
                            trait_species, coat_patterns, conditions,
                            background_points, foreground_points, manual_centres,
                            enabled_nodes, background_colour_enabled,
                            reviewed_input_key, species_library, library_shape_bank):
    """Capture after alignment/validation, before mutable UI state can change."""
    shape_bank = None if library_shape_bank is None else asdict(library_shape_bank)
    manifest = None if species_library is None else species_library.manifest
    facts = {
        'contract': 'seedfiddle.analysis-inputs-v1',
        'analysis_mode': 'image_local_adaptation',
        'annotations': {name: _array_fact(value) for name, value in masks.items()},
        'annotation_metadata': [asdict(item) for item in annotations],
        'species': species,
        'biological_context': None if biological_context is None else asdict(biological_context),
        'trait_vocabulary': {'species': trait_species, 'coat_patterns': coat_patterns,
                             'conditions': conditions},
        'background_reference_points': background_points,
        'foreground_reference_points': foreground_points,
        'manual_seed_centres': None if manual_centres is None else asdict(manual_centres),
        'enabled_nodes': None if enabled_nodes is None else sorted(enabled_nodes),
        'background_colour_enabled': background_colour_enabled,
        'reviewed_input_key': reviewed_input_key,
        'species_library': None if manifest is None else {
            'library_id': manifest.library_id, 'version': manifest.version,
            'species_id': manifest.species_id, 'sha256': manifest.content_sha256},
        'library_shape_bank_sha256': None if shape_bank is None else sha256(
            json.dumps(shape_bank, sort_keys=True, allow_nan=False).encode()).hexdigest(),
    }
    # An immutable string prevents callers from mutating the stored snapshot.
    return json.dumps(facts, sort_keys=True, allow_nan=False)
