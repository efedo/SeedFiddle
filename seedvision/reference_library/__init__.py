"""Reusable, immutable species-reference library support.

The package deliberately keeps its top level light.  Shape measurement imports
annotation persistence, and project persistence imports the library contracts;
eagerly importing builders here would close that loop during ordinary startup.
Public implementation objects remain available through lazy attributes.
"""

from __future__ import annotations

from importlib import import_module

from seedvision.reference_library.contracts import *  # noqa: F403


_LAZY_MODULES = {
    "InvalidSpeciesLibrary": "persistence",
    "SpeciesLibraryError": "persistence",
    "SpeciesLibraryIOError": "persistence",
    "SpeciesLibraryNotFound": "persistence",
    "SpeciesLibraryStore": "persistence",
    "create_manifest": "persistence",
    "default_species_library_root": "persistence",
    "LibraryAggregationSettings": "aggregation",
    "aggregate_contributions": "aggregation",
    "aggregation_settings_sha256": "aggregation",
    "ExtractedSourceContribution": "extraction",
    "LibraryExtractionSettings": "extraction",
    "LibrarySourceInput": "extraction",
    "extract_source_contribution": "extraction",
    "extraction_settings_sha256": "extraction",
    "SpeciesLibraryService": "service",
    "LibraryValidationReport": "validation",
    "assert_no_held_out_source": "validation",
    "source_weight_signature": "validation",
    "validate_aggregated_library": "validation",
}


def __getattr__(name: str):
    module_name = _LAZY_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(
        import_module(f"seedvision.reference_library.{module_name}"), name
    )
    globals()[name] = value
    return value
