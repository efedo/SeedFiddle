"""Synchronize the shared graph inputs with the desktop's selected image."""
from pathlib import Path

from PySide6.QtCore import Qt
from seedvision.pipeline.model import NodeStatus


def refresh_project_context(window):
    from seedvision.ui.main_window import _path_identity
    path = window.image_view.image_path
    presenter = getattr(window, 'pipeline_workflow', None)
    graphs = [window.pipeline]
    if presenter is not None and presenter.active:
        graphs.append(presenter.graph)
    rows = [(str(window.image_list.item(i).text()),
             str(window.image_list.item(i).data(Qt.ItemDataRole.UserRole)))
            for i in range(window.image_list.count())]
    current = next((i for i, (_, filename) in enumerate(rows)
                    if path is not None and _path_identity(Path(filename)) == _path_identity(path)), -1)
    window.pipeline_inspector.set_image_choices(rows, current)
    species = window.species_combo.currentText()
    pin = window._project_species_library_pin
    library_status, library_text = NodeStatus.COMPLETE, 'No species library pinned.'
    if pin is not None:
        library_text = f'{pin.library_id} · version {pin.version}'
        library_status = NodeStatus.IDLE
        if path is not None:
            # Cache only compact immutable-library resolution for this exact source
            # and context. Explicitly selecting the library node refreshes it.
            key = (pin, window._source_fingerprints.get(_path_identity(path)),
                   window._effective_biological_context(path))
            if key != getattr(window, '_context_library_key', None):
                try:
                    resolved = window._resolved_species_library_for_current_image(path)
                    if resolved.error:
                        library_status, library_text = NodeStatus.BLOCKED, resolved.error
                    elif resolved.provenance is not None:
                        library_status = NodeStatus.COMPLETE
                        library_text += f'\n{resolved.provenance.eligible_source_count} eligible source(s); current-image contributions excluded.'
                except (ValueError, OSError) as error:
                    library_status, library_text = NodeStatus.BLOCKED, str(error)
                window._context_library_key = key
                window._context_library_status = (library_status, library_text)
            else:
                library_status, library_text = window._context_library_status
    details = {
        'image_selector': 'No image selected.' if path is None else f'Raw source: {Path(path).name}',
        'metadata': f'Species: {species}\nChange species in Image metadata above.',
        'species_reference_library': library_text,
    }
    if presenter is not None and presenter.active:
        details['species_reference_library'] += '\nThis recipe has no species-library inputs; its calculations do not use these banks.'
    window.pipeline_inspector.set_context_details(details)
    for graph in graphs:
        for identifier, status, text in (
                ('project', NodeStatus.COMPLETE, f'{len(rows)} project image(s)'),
                ('image_selector', NodeStatus.COMPLETE if path else NodeStatus.IDLE,
                 Path(path).name if path else 'No image selected'),
                ('metadata', NodeStatus.COMPLETE if path else NodeStatus.IDLE, species),
                ('species_reference_library', library_status, library_text)):
            if identifier in graph.nodes:
                graph.set_status(identifier, status, text)
    window.pipeline_inspector.set_project_summary(window._project_summary_text())
    window.pipeline_canvas.refresh(('project', 'image_selector', 'metadata', 'species_reference_library'))
    window.pipeline_inspector.refresh_status()
