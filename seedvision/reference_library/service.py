"""UI-facing build, publish, resolve, and compatibility service."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from seedvision.annotation.seed_traits import load_seed_trait_catalogue
from seedvision.persistence.reference_regions import file_sha256
from seedvision.reference_library.aggregation import (
    LibraryAggregationSettings,
    aggregate_contributions,
    aggregation_settings_sha256,
    contributions_from_artifact,
    reindex_contribution,
)
from seedvision.reference_library.contracts import (
    EDGE_PROTOTYPE_SCHEMA,
    FOREGROUND_COLOUR_SCHEMA,
    FOREGROUND_NOISE_SCHEMA,
    MATERIAL_PROTOTYPE_SCHEMA,
    SEED_TRAIT_SCHEMA,
    SHAPE_SUMMARY_SCHEMA,
    DIMENSIONS_SHAPE_SCHEMA,
    BiologicalContext,
    ResolvedSpeciesLibrary,
    SpeciesLibraryArtifact,
    SpeciesLibraryPin,
    SpeciesLibraryProvenance,
)
from seedvision.reference_library.extraction import (
    LibraryExtractionSettings,
    LibrarySourceInput,
    extract_source_contribution,
    extraction_settings_sha256,
)
from seedvision.reference_library.persistence import (
    SpeciesLibraryError,
    SpeciesLibraryStore,
    create_manifest,
)
from seedvision.reference_library.validation import validate_aggregated_library


class SpeciesLibraryService:
    def __init__(self, store: SpeciesLibraryStore | None = None) -> None:
        self.store = store or SpeciesLibraryStore()

    def build_draft(
        self,
        sources: tuple[LibrarySourceInput, ...],
        *,
        version: str,
        trait_vocabulary_path: Path | str,
        seed_fiddle_version: str,
        library_id: str | None = None,
        extraction_settings: LibraryExtractionSettings | None = None,
        aggregation_settings: LibraryAggregationSettings | None = None,
        base_artifact: SpeciesLibraryArtifact | None = None,
        removed_source_sha256: tuple[str, ...] = (),
        cancellation_requested=None,
    ) -> SpeciesLibraryArtifact:
        if not sources and base_artifact is None:
            raise ValueError("At least one reviewed or retained source is required.")
        species = (
            sources[0].biological_context.species_id
            if sources
            else base_artifact.manifest.species_id
        )
        if any(item.biological_context.species_id != species for item in sources):
            raise ValueError("One library version cannot mix species.")
        if base_artifact is not None and base_artifact.manifest.species_id != species:
            raise ValueError("A fork cannot change the species of its parent library.")
        extraction_settings = extraction_settings or LibraryExtractionSettings()
        aggregation_settings = aggregation_settings or LibraryAggregationSettings()
        if (
            base_artifact is not None
            and base_artifact.manifest.extraction_settings_sha256
            != extraction_settings_sha256(extraction_settings)
        ):
            raise ValueError(
                "The parent uses different extraction settings; re-extract its "
                "source images instead of mixing incompatible compact descriptors."
            )
        catalogue = load_seed_trait_catalogue(trait_vocabulary_path)
        vocabulary = catalogue.for_species(species)
        ordered = tuple(
            sorted(sources, key=lambda item: file_sha256(item.source_path))
        )
        ordered = tuple(
            replace(
                item,
                coat_pattern_vocabulary=vocabulary.coat_patterns,
                condition_vocabulary=catalogue.conditions,
            )
            for item in ordered
        )
        new_contributions = tuple(
            extract_source_contribution(
                item,
                source_index=index,
                settings=extraction_settings,
                cancellation_requested=cancellation_requested,
            )
            for index, item in enumerate(ordered)
        )
        retained = (
            ()
            if base_artifact is None
            else tuple(
                item
                for item in contributions_from_artifact(base_artifact)
                if item.source.source_sha256 not in set(removed_source_sha256)
            )
        )
        contribution_values = (*retained, *new_contributions)
        if not contribution_values:
            raise ValueError("The draft would contain no retained or new source.")
        hashes = [item.source.source_sha256 for item in contribution_values]
        if len(hashes) != len(set(hashes)):
            raise ValueError(
                "A source image already retained by the parent cannot be added twice."
            )
        contributions = tuple(
            reindex_contribution(item, index)
            for index, item in enumerate(contribution_values)
        )
        aggregated = aggregate_contributions(
            contributions, settings=aggregation_settings
        )
        validation = validate_aggregated_library(aggregated)
        if not validation.publish_eligible:
            raise ValueError(
                "Library draft has no publishable reviewed product: "
                + "; ".join(validation.warnings)
            )
        descriptor_schemas = {
            "foreground_colour": FOREGROUND_COLOUR_SCHEMA,
            "foreground_noise": FOREGROUND_NOISE_SCHEMA,
            "material_prototypes": MATERIAL_PROTOTYPE_SCHEMA,
            "edge_prototypes": EDGE_PROTOTYPE_SCHEMA,
            "seed_traits": SEED_TRAIT_SCHEMA,
            "shape_summary": SHAPE_SUMMARY_SCHEMA,
            "dimensions_shape": DIMENSIONS_SHAPE_SCHEMA,
        }
        manifest = create_manifest(
            library_id=(
                library_id
                or (
                    base_artifact.manifest.library_id
                    if base_artifact is not None
                    else str(uuid4())
                )
            ),
            version=version,
            species_id=species,
            species_display_name=(
                ordered[0].species_display_name
                if ordered
                else base_artifact.manifest.species_display_name
            ),
            trait_vocabulary_sha256=file_sha256(trait_vocabulary_path),
            seed_fiddle_version=seed_fiddle_version,
            sources=aggregated["sources"],
            products=validation.products,
            descriptor_schemas=descriptor_schemas,
            extraction_settings_sha256=extraction_settings_sha256(
                extraction_settings
            ),
            aggregation_settings_sha256=aggregation_settings_sha256(
                aggregation_settings
            ),
            corrected_colour_space_version="opencv-lab-u8-v1",
            parent_content_sha256=(
                None
                if base_artifact is None
                else base_artifact.manifest.content_sha256
            ),
        )
        return SpeciesLibraryArtifact(
            manifest,
            aggregated["foreground_colour"],
            aggregated["foreground_noise"],
            aggregated["material_prototypes"],
            aggregated["edge_prototypes"],
            aggregated["seed_traits"],
            aggregated["shape_summary"],
            aggregated["dimensions_shape"],
        )

    def publish(self, draft: SpeciesLibraryArtifact) -> SpeciesLibraryArtifact:
        return self.store.publish(draft)

    def resolve(
        self,
        pin: SpeciesLibraryPin | None,
        *,
        context: BiologicalContext | None,
        current_source_sha256: str | None,
        trait_vocabulary_sha256: str,
    ) -> ResolvedSpeciesLibrary:
        if pin is None:
            return ResolvedSpeciesLibrary(
                None, None, {}, "No species library is pinned to this project."
            )
        if context is None or context.species_id != pin.species_id:
            return ResolvedSpeciesLibrary(
                None, None, {}, "Project biological context does not match the pin."
            )
        try:
            artifact = self.store.load(pin)
        except SpeciesLibraryError as error:
            return ResolvedSpeciesLibrary(None, None, {}, str(error))
        compatibility = {}
        vocabulary_matches = (
            artifact.manifest.trait_vocabulary_sha256 == trait_vocabulary_sha256
        )
        for product in artifact.manifest.products:
            status = product.tier.value
            if product.product.value == "seed_traits" and not vocabulary_matches:
                status = "incompatible_trait_vocabulary"
            compatibility[product.product.value] = status
        filtered = artifact
        shape_warnings = []
        for name, schema in (("shape_summary", SHAPE_SUMMARY_SCHEMA),
                             ("dimensions_shape", DIMENSIONS_SHAPE_SCHEMA)):
            bank = getattr(filtered, name)
            if bank is not None and bank.schema_id != schema:
                filtered = replace(filtered, **{name: None})
                compatibility[name] = "incompatible_shape_measurements"
                shape_warnings.append(
                    f"{name}: rebuild this library from saved annotations; its older "
                    "ellipse orientation/uncertainty measurements are incompatible.")
        excluded = ()
        if current_source_sha256 is not None:
            filtered, excluded = filtered.excluding_source_sha256(
                current_source_sha256
            )
        if not vocabulary_matches and filtered.seed_traits is not None:
            # A trait class index is meaningful only against the exact
            # vocabulary used to publish it.  Reporting incompatibility while
            # still handing the bank to runtime consumers would make a renamed
            # or reordered class silently acquire another class's evidence.
            filtered = replace(filtered, seed_traits=None)
        selected_path = ()
        if filtered.dimensions_shape is not None:
            _component, selected_path = filtered.dimensions_shape.select_component(
                context, "flat"
            )
        provenance = SpeciesLibraryProvenance(
            pin,
            len(artifact.manifest.sources) - len(excluded),
            excluded,
            selected_path,
            warnings=(
                ("Trait vocabulary differs; trait product disabled.",)
                if not vocabulary_matches
                else ()
            ) + tuple(shape_warnings),
        )
        return ResolvedSpeciesLibrary(filtered, provenance, compatibility)
