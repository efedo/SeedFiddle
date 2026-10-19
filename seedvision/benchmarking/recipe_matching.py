"""Recipe comparison helpers shared by saved evidence and execution adapters."""
from dataclasses import replace
from seedvision.benchmarking.recipes import RecipeProvenanceMode, RecipeSpec


def _matches_catalogue_with_selected_values(recipe: RecipeSpec, catalogue: RecipeSpec | None) -> bool:
    """Permit explicit parameter choices without relaxing source or topology identity."""
    if catalogue is None:
        return False
    try:
        stages = tuple(
            replace(actual, parameters=tuple(
                replace(parameter, value=pinned.value)
                for parameter, pinned in zip(actual.parameters, canonical.parameters, strict=True)
            ))
            for actual, canonical in zip(recipe.stages, catalogue.stages, strict=True)
        )
        return replace(recipe, stages=stages, layout=catalogue.layout) == catalogue
    except (TypeError, ValueError):
        return False


def _matches_native_catalogue(recipe: RecipeSpec, catalogue: RecipeSpec | None) -> bool:
    """Check topology/source/controls while the native verifier checks runtime pins."""
    if catalogue is None or recipe.provenance.mode is not RecipeProvenanceMode.NATIVE_COMPATIBLE:
        return False
    if recipe.provenance.reference_id != "plantcv":
        return False
    return _matches_catalogue_with_selected_values(
        replace(recipe, provenance=catalogue.provenance), catalogue,
    )
