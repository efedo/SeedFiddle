"""Run the saved project recipe and fingerprinted references without a GUI."""
from pathlib import Path
from dataclasses import asdict
from seedvision.pipeline import build_default_pipeline
from seedvision.pipeline.settings import production_settings
from seedvision.persistence.analysis_settings import apply_analysis_settings_profile, analysis_settings_profile_to_payload
from seedvision.persistence.project_analysis import ProjectAnalysisStore
from seedvision.persistence.reference_regions import ReferenceRegionStore, file_sha256
from seedvision.persistence.manual_seed_centres import ManualSeedCentreStore
from seedvision.segmentation.procedural import ManualSeedCentres, ManualSeedCentreMode, ManualSeedCentreSpace
from seedvision.segmentation.baseline import analyze_path
from seedvision.export.results import result_report, export_report


def run_project(root, project_path, output_directory):
    root = Path(root)
    loaded = ProjectAnalysisStore(root).load(project_path)
    if loaded.issues:
        raise ValueError('Project contains unresolved or changed sources/sidecars; repair them before batch execution.')
    graph = build_default_pipeline()
    apply_analysis_settings_profile(graph, loaded.document.analysis_settings)
    from seedvision.reference_library.service import SpeciesLibraryService
    from seedvision.annotation.seed_traits import load_seed_trait_catalogue
    catalogue = load_seed_trait_catalogue(root/'config'/'traits.json')
    vocabulary = catalogue.for_species(loaded.document.species or 'unknown')
    service = SpeciesLibraryService()
    reports = []
    for item in loaded.images:
        kwargs = production_settings(graph)
        bundle_ref = item.sidecar('reference_regions')
        bundle = None
        if bundle_ref is not None:
            bundle = ReferenceRegionStore(root).load_project_archive(item.path,bundle_ref.path,None)
            if bundle.source_to_corrected is None:
                raise ValueError(f'{item.path.name}: reference coordinates need explicit alignment review in the desktop.')
            kwargs.update(background_reference_mask=bundle.background,foreground_reference_mask=bundle.foreground,
                background_exclusion_mask=bundle.other,foreground_exclusion_mask=bundle.other,
                seed_instance_annotations=bundle.annotated_seeds,seed_instance_traits=bundle.seed_annotations,
                reference_transform=bundle.source_to_corrected,seed_trait_species=bundle.annotation_species,
                seed_trait_coat_patterns=vocabulary.coat_patterns,seed_trait_conditions=catalogue.conditions)
        centres_ref = item.sidecar('manual_seed_centres')
        if centres_ref is not None:
            centres = ManualSeedCentreStore(root).load_project_archive(item.path,centres_ref.path,item.record.source_shape)
            kwargs['manual_seed_centres'] = ManualSeedCentres(tuple(map(tuple,centres.centres_xy)),ManualSeedCentreMode(centres.mode),ManualSeedCentreSpace.SOURCE_IMAGE)
        context = item.record.biological_context or loaded.document.biological_context
        library = service.resolve(loaded.document.species_library,context=context,
            current_source_sha256=item.record.source.sha256,trait_vocabulary_sha256=file_sha256(root/'config'/'traits.json'))
        if loaded.document.species_library is not None and library.error:
            raise ValueError(library.error)
        enabled = frozenset(identifier for identifier,node in graph.nodes.items() if node.enabled)
        kwargs.update(enabled_nodes=enabled,learning_root=root,species=loaded.document.species,
            biological_context=context,species_library=library.artifact,
            library_shape_bank=None if library.artifact is None else library.artifact.dimensions_shape,
            background_colour_enabled=graph.node('background_likelihood').parameters['background_colour_enabled'])
        result = analyze_path(item.path,**kwargs)
        report = result_report(result,enabled=enabled,settings=analysis_settings_profile_to_payload(loaded.document.analysis_settings),
            provenance={'project_sha256':file_sha256(project_path),'analysis_mode':'image_local_adaptation',
                'reference_sha256':None if bundle_ref is None else bundle_ref.reference.sha256,
                'library':None if loaded.document.species_library is None else asdict(loaded.document.species_library)})
        reports.append(export_report(output_directory,result,report))
        del result
    return reports
