"""Bounded editable project recipes, separate from historical run records."""
from seedvision.benchmarking.recipes import MAX_RECIPE_BYTES, RecipeSpec, RecipeProvenanceMode
from seedvision.persistence.reference_graphs import (
    _canonical_json, _recipe_from_payload, _validate_recipe_counts,
)

MAX_NATIVE_RECIPES = 16
MAX_NATIVE_RECIPES_BYTES = 8 * 1024 * 1024


def recipe_key(recipe):
    return recipe.recipe_id, recipe.recipe_version


def validate_native_recipes(value):
    if type(value) is not tuple or len(value) > MAX_NATIVE_RECIPES:
        raise ValueError('Project native_recipes must be a bounded immutable tuple.')
    seen, total = set(), 0
    for recipe in value:
        from seedvision.benchmarking.native_smpts_recipe import is_executable_paper_recipe
        paper_recipe = is_executable_paper_recipe(recipe)
        if not isinstance(recipe, RecipeSpec) or (recipe.provenance.mode is not RecipeProvenanceMode.NATIVE_COMPATIBLE and not paper_recipe):
            raise ValueError('Working project recipes must be native-compatible RecipeSpec values.')
        if paper_recipe:
            from seedvision.benchmarking.native_recipe import _validate_recipe
            _validate_recipe(recipe)  # Admit this canonical executable graph, never arbitrary paper contracts.
        _validate_recipe_counts(recipe)
        encoded = _canonical_json(recipe.to_dict(), maximum=MAX_RECIPE_BYTES, label='Working native recipe')
        _recipe_from_payload(recipe.to_dict())  # Strictly revalidate directly constructed metadata.
        key = recipe_key(recipe)
        if key in seen:
            raise ValueError('Only one working selection per native recipe id/version is allowed.')
        seen.add(key)
        total += len(encoded)
        if total > MAX_NATIVE_RECIPES_BYTES:
            raise ValueError('Working native recipe metadata exceeds its aggregate byte limit.')
    return value


def native_recipes_to_payload(value):
    return [recipe.to_dict() for recipe in validate_native_recipes(value)]


def native_recipes_from_payload(value):
    if type(value) is not list or len(value) > MAX_NATIVE_RECIPES:
        raise ValueError('Project native_recipes must be a bounded JSON array.')
    return validate_native_recipes(tuple(_recipe_from_payload(item) for item in value))


def select_native_recipe(current, selected):
    current = validate_native_recipes(current)
    validate_native_recipes((selected,))
    key = recipe_key(selected)
    found = any(recipe_key(recipe) == key for recipe in current)
    updated = tuple(selected if recipe_key(recipe) == key else recipe for recipe in current)
    return validate_native_recipes(updated if found else updated+(selected,))
