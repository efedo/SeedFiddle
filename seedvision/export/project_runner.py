"""Run the saved project recipe and fingerprinted references without a GUI."""
from pathlib import Path
from dataclasses import asdict
from seedvision.pipeline import build_default_pipeline
from seedvision.pipeline.settings import production_settings
from seedvision.pipeline.catalogue_nodes import instance_default_context
from seedvision.persistence.analysis_settings import apply_analysis_settings_profile, analysis_settings_profile_to_payload
from seedvision.persistence.project_analysis import ProjectAnalysisStore
from seedvision.persistence.reference_regions import ReferenceRegionStore, file_sha256
from seedvision.persistence.manual_seed_centres import ManualSeedCentreStore
from seedvision.segmentation.procedural import ManualSeedCentres, ManualSeedCentreMode, ManualSeedCentreSpace
from seedvision.segmentation.baseline import analyze_path
from seedvision.export.results import result_report, export_report
from seedvision.export.project_native_graphs import export_project_native_graphs, export_project_node_instances
from seedvision.export.project_reference_captures import export_project_reference_captures
from seedvision.segmentation.baseline import PipelineAnalysisCache
from seedvision.export.run_outcomes import (run_native_analysis, NativeRunOutcome,
    export_run_outcome, AnalysisInputIntegrityError)
from seedvision.export.preflight_outcomes import PreflightRunOutcome, record_preflight


def run_project(root, project_path, output_directory, *, crop_padding_px=None,
                reference_capture_runner_factory=None):
    root = Path(root)
    terminal_directory = Path(output_directory) / 'run-outcomes'
    with record_preflight(project_path, 'project', 'project_arguments', terminal_directory):
        if crop_padding_px is not None and (type(crop_padding_px) is not int or crop_padding_px < 0):
            raise ValueError('crop_padding_px must be a nonnegative integer or None.')
    with record_preflight(project_path, 'project', 'project_load', terminal_directory):
        loaded = ProjectAnalysisStore(root).load(project_path)
    with record_preflight(project_path, 'project', 'project_sources', terminal_directory):
        from seedvision.persistence.correction_inputs import PORTS
        correction_issue_roles = {'correction_input:'+role for role in PORTS}
        if any(issue.role not in correction_issue_roles for issue in loaded.issues):
            raise ValueError('Project contains unresolved or changed sources/sidecars; repair them before batch execution.')
    with record_preflight(project_path, 'project', 'project_settings', terminal_directory):
        graph = build_default_pipeline()
        apply_analysis_settings_profile(graph, loaded.document.analysis_settings)
    with record_preflight(project_path, 'project', 'project_traits', terminal_directory):
        from seedvision.reference_library.service import SpeciesLibraryService
        from seedvision.annotation.seed_traits import load_seed_trait_catalogue
        catalogue = load_seed_trait_catalogue(root/'config'/'traits.json')
        vocabulary = catalogue.for_species(loaded.document.species or 'unknown')
        service = SpeciesLibraryService()
    reports = []
    native_cache = PipelineAnalysisCache()
    for item in loaded.images:
        with record_preflight(item.path, 'image', 'image_settings', terminal_directory):
            kwargs = production_settings(graph)
            kwargs.update(seed_trait_coat_patterns=vocabulary.coat_patterns,
                          seed_trait_conditions=catalogue.conditions)
        with record_preflight(item.path, 'image', 'image_references', terminal_directory):
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
                    reference_archive_source_sha256=bundle.archive_source_sha256,
                    reference_annotation_snapshot_source_sha256=bundle.annotation_snapshot_source_sha256,
                    seed_trait_coat_patterns=vocabulary.coat_patterns,seed_trait_conditions=catalogue.conditions)
        with record_preflight(item.path, 'image', 'image_manual_centres', terminal_directory):
            centres_ref = item.sidecar('manual_seed_centres')
            if centres_ref is not None:
                centres = ManualSeedCentreStore(root).load_project_archive(item.path,centres_ref.path,item.record.source_shape)
                kwargs['manual_seed_centres'] = ManualSeedCentres(tuple(map(tuple,centres.centres_xy)),ManualSeedCentreMode(centres.mode),ManualSeedCentreSpace.SOURCE_IMAGE)
        with record_preflight(item.path, 'image', 'image_library', terminal_directory):
            from seedvision.reference_library.contracts import BiologicalContext
            context = (item.record.biological_context or loaded.document.biological_context
                       or BiologicalContext(vocabulary.species_id or 'unknown'))
            library = service.resolve(loaded.document.species_library,context=context,
                current_source_sha256=item.record.source.sha256,trait_vocabulary_sha256=file_sha256(root/'config'/'traits.json'))
            if loaded.document.species_library is not None and library.error:
                raise ValueError(library.error)
        with record_preflight(item.path, 'image', 'image_request', terminal_directory):
            enabled = frozenset(identifier for identifier,node in graph.nodes.items() if node.enabled)
            kwargs.update(enabled_nodes=enabled,learning_root=root,species=loaded.document.species or 'unknown',
                biological_context=context,species_library=library.artifact,
                library_shape_bank=None if library.artifact is None else library.artifact.dimensions_shape,
                background_colour_enabled=graph.node('background_likelihood').parameters['background_colour_enabled'])
        result = run_native_analysis(item.path, analysis_runner=analyze_path, **kwargs)
        if isinstance(result, (NativeRunOutcome, PreflightRunOutcome)):
            destination = export_run_outcome(Path(output_directory) / 'run-outcomes', result)
            payload = result.payload
            state = payload['identity']['state']
            diagnostic = payload['diagnostic'] or payload['identity']['reason_code']
            del result
            if state == 'cancelled':
                from seedvision.timing import AnalysisCancelled
                raise AnalysisCancelled(f'{diagnostic} Terminal record: {destination}')
            if state == 'rejected':
                raise AnalysisInputIntegrityError(f'{diagnostic} Terminal record: {destination}')
            if (payload['format'] != 'seedfiddle-native-run-outcome' or state != 'failed'
                    or payload['identity']['reason_code'] == 'device_failure'):
                raise RuntimeError(f'{diagnostic} Terminal record: {destination}')
            if file_sha256(item.path) != item.record.source.sha256:
                raise AnalysisInputIntegrityError('Source changed after failed analysis; companion execution is blocked.')
            try:
                export_project_native_graphs(loaded.document.reference_graphs, item.path,
                    destination / 'native_graphs', expected_source_sha256=item.record.source.sha256,
                    cache=native_cache, project_root=root,
                    reference_model_assets=loaded.document.reference_model_assets,
                    native_recipes=loaded.document.native_recipes,
                    **({'pipeline_workflow': loaded.document.pipeline_workflow}
                       if getattr(loaded.document, 'pipeline_workflow', None) is not None else {}),
                    correction_bindings=loaded.document.correction_bindings, project_image_record=item.record,
                    project_path=project_path)
            finally:
                native_cache.values.clear()
            export_project_reference_captures(loaded.document.reference_graphs, item.path,
                destination / 'reference_captures', expected_source_sha256=item.record.source.sha256,
                runner_factory=reference_capture_runner_factory, project_root=root, project_path=project_path)
            if loaded.document.node_instances:
                export_project_node_instances(loaded.document.node_instances, item.path,
                    destination / 'catalogue_nodes', expected_source_sha256=item.record.source.sha256,
                    project_root=root, reference_model_assets=loaded.document.reference_model_assets,
                    correction_bindings=loaded.document.correction_bindings, project_image_record=item.record,
                    default_context=instance_default_context(kwargs))
            reports.append(destination)
            continue
        report_kwargs = dict(enabled=enabled,settings=analysis_settings_profile_to_payload(loaded.document.analysis_settings),
            execution_provenance={'route':'batch','project_sha256':file_sha256(project_path),'analysis_mode':'image_local_adaptation',
                'reference_sha256':None if bundle_ref is None else bundle_ref.reference.sha256,
                'library':None if loaded.document.species_library is None else asdict(loaded.document.species_library)})
        report = result_report(result, **report_kwargs)
        from seedvision.export.review import lookup_result_review
        review, review_state = lookup_result_review(root, report.get('result_revision'), project_path=project_path)
        if review is not None:
            report = result_report(result, **report_kwargs, review=review)
        report['review_lookup'] = review_state
        if crop_padding_px is None:
            destination = export_report(output_directory,result,report)
        else:
            from uuid import uuid4
            from seedvision.export.crop_artifacts import export_seed_crop_artifacts
            staging = Path(output_directory) / ('.seed-crop-job-' + uuid4().hex + '.incomplete')
            staging.mkdir(parents=True, exist_ok=False)
            destination = export_report(staging, result, report)
            if report['state'] == 'available_for_review' and report['instances']:
                export_seed_crop_artifacts(destination, result, report,
                    seed_ids=tuple(row['seed_id'] for row in report['instances']), padding_px=crop_padding_px)
            published = Path(output_directory) / destination.name
            destination.rename(published)
            staging.rmdir()
            destination = published
        # Baseline/crop serialization is complete; release its CUDA image before
        # starting the saved native graphs for the same source.
        del result
        try:
            export_project_native_graphs(loaded.document.reference_graphs, item.path,
                destination/'native_graphs', expected_source_sha256=item.record.source.sha256,
                cache=native_cache, project_root=root,
                reference_model_assets=loaded.document.reference_model_assets,
                native_recipes=loaded.document.native_recipes,
                **({'pipeline_workflow': loaded.document.pipeline_workflow}
                   if getattr(loaded.document, 'pipeline_workflow', None) is not None else {}),
                correction_bindings=loaded.document.correction_bindings, project_image_record=item.record,
                project_path=project_path)
        finally:
            # Reuse node products within this image's saved graphs, then release
            # dense products before processing another full-resolution image.
            native_cache.values.clear()
        export_project_reference_captures(loaded.document.reference_graphs, item.path,
            destination / 'reference_captures', expected_source_sha256=item.record.source.sha256,
            runner_factory=reference_capture_runner_factory, project_root=root, project_path=project_path)
        if loaded.document.node_instances:
            export_project_node_instances(loaded.document.node_instances, item.path,
                destination / 'catalogue_nodes', expected_source_sha256=item.record.source.sha256,
                project_root=root, reference_model_assets=loaded.document.reference_model_assets,
                correction_bindings=loaded.document.correction_bindings, project_image_record=item.record,
                default_context=instance_default_context(kwargs))
        reports.append(destination)
    return reports
