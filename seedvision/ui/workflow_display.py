"""Bind the shared image controls to the active workflow and its coordinate frame."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap


class WorkflowDisplay:
    def __init__(self, controller):
        self.controller = controller
        self.window = controller.window
        self.mode = 'source'
        self.title = 'Source image'
        self.blend_available = False

    def sync_controls(self):
        controller, window = self.controller, self.window
        if not controller.active:
            return
        ready = window.image_view.image_path is not None
        busy = controller._job is not None
        window.overlay_button.setEnabled(ready)
        window.overlay_combo.setEnabled(ready)
        window.overlay_opacity_slider.setEnabled(self.blend_available and not busy)
        window.overlay_opacity_slider.setToolTip(
            'Blend the selected output over its input image.' if self.blend_available
            else 'Opacity requires an image in the same coordinate frame as this output.')
        # These are workspace controls, independent of the selected recipe.
        for action in (window.opacity_toolbar_label_action,
                       window.overlay_opacity_slider_action, window.overlay_opacity_label_action):
            action.setVisible(True)
        for widget in (window.opacity_toolbar_label, window.overlay_opacity_slider,
                       window.overlay_opacity_label):
            widget.show()
        for action in (window.hsv_value_toolbar_label_action, window.hsv_value_slider_action,
                       window.hsv_value_label_action, window.hsv_peak_button_action):
            action.setVisible(False)
        key = window._current_image_key()
        editable = window._analyses.get(key or '') is not None
        explanation = ('Edit labels in the calibrated image; return to node outputs through Overlay.'
            if editable else 'Run Default seed analysis for this image to establish the calibrated annotation frame.')
        for action in (window.paint_background_action, window.annotate_instances_action):
            action.setEnabled(editable and not busy)
            action.setToolTip(explanation)
        # Annotation visibility is meaningful only in the verified editing frame.
        for widget in (window.show_material_references_checkbox,
                       window.show_instance_annotations_checkbox, window.annotation_opacity_slider):
            widget.setEnabled(editable and self.mode == 'annotations' and not busy)
            widget.setToolTip('Shown in the calibrated annotation image. Choose it from Overlay or click Materials / Seeds.')
        self.rebuild_menu()

    def rebuild_menu(self):
        window, controller = self.window, self.controller
        window.overlay_menu.clear()
        window.overlay_menu.addAction('Source image', self.show_source)
        annotation = window.overlay_menu.addAction('Calibrated image and annotations', self.show_annotations)
        annotation.setEnabled(window._analyses.get(window._current_image_key() or '') is not None)
        if controller.graph is not None:
            for stage in controller.workflow.recipe.stages:
                submenu = window.overlay_menu.addMenu(stage.title)
                node = None
                if controller.result is not None:
                    try:
                        node = controller.result.node(stage.stage_id)
                    except StopIteration:
                        pass
                for port in stage.ports:
                    if port.direction != 'output':
                        continue
                    action = submenu.addAction(port.label or port.name)
                    action.setEnabled(node is not None and port.name in node.outputs)
                    action.triggered.connect(lambda _checked=False, sid=stage.stage_id, name=port.name:
                        self.show_output(sid, name))
        window.overlay_button.setText(window.overlay_button.fontMetrics().elidedText(
            self.title, Qt.TextElideMode.ElideRight, window.overlay_button.maximumWidth() - 28))
        window.overlay_button.setAccessibleName(self.title)
        window.overlay_button.setToolTip(self.title + '\nChoose the source image, annotation image, or a calculated node output.')

    def show_source(self):
        controller, window = self.controller, self.window
        window._stop_reference_point_editing()
        source = window.image_view._source_pixmap
        self.mode, self.title, self.blend_available = 'source', 'Source image', False
        if source is None:
            controller.preview.setText('Open an image to run this workflow.')
        else:
            controller.preview.setPixmap(source.scaled(1024, 1024,
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            controller.preview.message.setText(f'Source image · {source.width()}×{source.height()} pixels · annotations use the calibrated image')
        window.image_display_stack.setCurrentWidget(controller.preview)
        self.sync_controls()

    def show_output(self, node_id, output):
        controller = self.controller
        self.window._stop_reference_point_editing()
        self.window.pipeline_canvas.select_node(node_id)
        controller._preferred_output = output
        controller.select_node(node_id)

    def show_annotations(self):
        window = self.window
        result = window._analyses.get(window._current_image_key() or '')
        if result is None:
            window.statusBar().showMessage('Run Default seed analysis for this image before editing calibrated annotations.')
            return False
        self.mode, self.title, self.blend_available = 'annotations', 'Calibrated image and annotations', False
        window.image_view.show_analysis(result)
        window.image_view.set_overlay_mode('calibrated_image')
        window.image_display_stack.setCurrentWidget(window.image_view)
        window.image_workspace_action.setChecked(True)
        self.sync_controls()
        return True

    def rendered(self, raster, semantics):
        self.mode = 'output'
        stage = self.controller.workflow.recipe.stage(self.controller._selected)
        self.title = stage.title
        self.blend_available = False
        base = None
        frame = semantics.get('frame', self.controller.result.coordinate_frame.frame_id)
        if self.controller.result is not None and frame:
            # Select an actual RGB output with identical frame and dimensions.
            # Never resize corrected annotations onto source or crop coordinates.
            for item in self.controller.result.nodes:
                spec = self.controller.workflow.recipe.stage(item.stage_id)
                for port in spec.ports:
                    candidate = item.outputs.get(port.name)
                    candidate_frame = port.semantics.get('frame')
                    if candidate_frame is None and spec.operation_id == 'external.image_inputs':
                        candidate_frame = self.controller.result.coordinate_frame.frame_id
                    if (port.direction == 'output' and candidate_frame == frame
                            and port.semantics.get('color_order') == 'RGB'
                            and getattr(candidate, 'ndim', 0) == 3
                            and tuple(candidate.shape[:2]) == tuple(raster.shape[:2])):
                        from seedvision.ui.native_workflow_helpers import native_preview_image
                        base = QPixmap.fromImage(native_preview_image(candidate, port.semantics))
                        break
                if base is not None:
                    break
        if base is not None and raster.ndim == 2:
            self.controller.preview.set_base(base)
            self.blend_available = True
            self.set_opacity(self.window.overlay_opacity_slider.value())
        self.sync_controls()

    def set_opacity(self, value):
        if self.blend_available:
            self.controller.preview.output_item.setOpacity(value / 100.0)
