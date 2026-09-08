"""Species-specific semantic labels for reviewed reference seed instances."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}")


def trait_display_name(identifier: str) -> str:
    """Return a compact UI label for a stable trait identifier."""

    return str(identifier).replace("_", " ").capitalize()


@dataclass(frozen=True, slots=True)
class SpeciesSeedTraitVocabulary:
    species_id: str
    display_name: str
    coat_patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SeedTraitCatalogue:
    species: tuple[SpeciesSeedTraitVocabulary, ...]
    conditions: tuple[str, ...]

    def for_species(self, value: str) -> SpeciesSeedTraitVocabulary:
        """Resolve either the stable species ID or its UI display name."""

        query = str(value).strip().casefold()
        for vocabulary in self.species:
            if query in {
                vocabulary.species_id.casefold(),
                vocabulary.display_name.casefold(),
            }:
                return vocabulary
        return SpeciesSeedTraitVocabulary("", str(value), ())


def load_seed_trait_catalogue(path: Path | str) -> SeedTraitCatalogue:
    """Load and strictly validate the semantic annotation vocabulary."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The traits manifest root must be a JSON object.")
    conditions = _identifier_tuple(
        payload.get("reference_seed_conditions", ()),
        "reference_seed_conditions",
    )
    species_items = payload.get("species", ())
    if not isinstance(species_items, list):
        raise ValueError("species must be a JSON list.")
    species: list[SpeciesSeedTraitVocabulary] = []
    seen_species: set[str] = set()
    seen_display_names: set[str] = set()
    for item in species_items:
        if not isinstance(item, dict):
            raise ValueError("Every traits species entry must be an object.")
        species_id = str(item.get("id", ""))
        display_name = str(item.get("display_name", "")).strip()
        if _IDENTIFIER.fullmatch(species_id) is None or not display_name:
            raise ValueError("Traits species IDs and display names must be valid.")
        if species_id in seen_species:
            raise ValueError(f"Duplicate traits species ID {species_id!r}.")
        display_key = display_name.casefold()
        if display_key in seen_display_names:
            raise ValueError(f"Duplicate traits species name {display_name!r}.")
        seen_species.add(species_id)
        seen_display_names.add(display_key)
        species.append(
            SpeciesSeedTraitVocabulary(
                species_id=species_id,
                display_name=display_name,
                coat_patterns=_identifier_tuple(
                    item.get("reference_seed_coat_patterns", ()),
                    f"{species_id}.reference_seed_coat_patterns",
                ),
            )
        )
    return SeedTraitCatalogue(tuple(species), conditions)


def _identifier_tuple(values: object, name: str) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a JSON list.")
    normalized = tuple(str(value) for value in values)
    if any(_IDENTIFIER.fullmatch(value) is None for value in normalized):
        raise ValueError(f"{name} contains an invalid stable identifier.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicate identifiers.")
    return normalized
