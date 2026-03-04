import sys
import json
import os
print("Starting application...")
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                            QMenuBar, QMenu, QAction, QFileDialog, QMessageBox, QSplitter,
                            QFormLayout, QComboBox, QDoubleSpinBox, QLabel)
print("PyQt5 imported successfully")
from PyQt5.QtCore import Qt
from image_viewer import ImageViewer
print("ImageViewer imported")
from plugin_manager import PluginManager
print("PluginManager imported")
from processing_pipeline import ProcessingPipeline
print("ProcessingPipeline imported")
from parameter_panel import ParameterPanel
print("ParameterPanel imported")
from edge_measurement_panel import EdgeMeasurementPanel
from roi_manager import ROIManager
print("ROIManager imported")
from histogram_widget import HistogramWidget

SESSION_FILE = os.path.join(os.path.dirname(__file__), '.last_session.json')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionTool")
        self.setGeometry(100, 100, 1200, 800)
        self.last_image_path = None

        # Central: one or two image viewers + (parameters + Edge Measurement results) side-by-side
        self.image_viewer = ImageViewer()
        self.image_viewer_compare = ImageViewer()
        self.image_viewer_compare.hide()
        self.compare_original_image = None
        self.compare_mode_enabled = False
        self._syncing_view = False
        self.edge_measure_panel = EdgeMeasurementPanel()
        self.parameter_panel = ParameterPanel()

        # Center-right analysis widget (parameters + Edge Measurement summary)
        self.right_center_widget = QWidget()
        right_center_layout = QVBoxLayout(self.right_center_widget)
        right_center_layout.setContentsMargins(0, 0, 0, 0)
        right_center_layout.setSpacing(4)
        # Parameters for the selected pipeline step on top, Edge Measurement results below
        right_center_layout.addWidget(self.parameter_panel)
        right_center_layout.addWidget(self.edge_measure_panel)

        # Left side: splitter that can host one or two synchronized viewers
        self.viewer_splitter = QSplitter(Qt.Horizontal)
        self.viewer_splitter.addWidget(self.image_viewer)
        # Do not add compare viewer yet; it is added when compare mode is enabled

        center_splitter = QSplitter(Qt.Horizontal)
        center_splitter.addWidget(self.viewer_splitter)
        center_splitter.addWidget(self.right_center_widget)
        center_splitter.setStretchFactor(0, 3)
        center_splitter.setStretchFactor(1, 2)
        self.setCentralWidget(center_splitter)

        # Right: ROI controls, Histogram, Plugins, Pipeline
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        self.roi_manager = ROIManager(self.image_viewer)
        self.histogram_widget = HistogramWidget()
        self.plugin_manager = PluginManager()
        self.processing_pipeline = ProcessingPipeline(self.image_viewer, self.edge_measure_panel)
        self.roi_manager.setMinimumHeight(120)
        self.histogram_widget.setMinimumHeight(140)
        # Make Filters/Detectors list more compact so parameters get more space
        self.plugin_manager.setMinimumHeight(80)

        # Color / view controls (below histogram)
        color_ctrl = QWidget()
        color_layout = QFormLayout(color_ctrl)
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.setSpacing(2)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["RGB", "Grayscale", "HSV - H", "HSV - S", "HSV - V"])
        self.view_mode_combo.setCurrentText("RGB")
        self.view_mode_combo.currentTextChanged.connect(self.on_view_mode_changed)
        color_layout.addRow(QLabel("View mode:"), self.view_mode_combo)

        self.s_scale_spin = QDoubleSpinBox()
        self.s_scale_spin.setRange(0.0, 3.0)
        self.s_scale_spin.setSingleStep(0.1)
        self.s_scale_spin.setDecimals(2)
        self.s_scale_spin.setValue(1.0)
        self.s_scale_spin.valueChanged.connect(self.on_hsv_scale_changed)
        color_layout.addRow(QLabel("S scale:"), self.s_scale_spin)

        self.v_scale_spin = QDoubleSpinBox()
        self.v_scale_spin.setRange(0.0, 3.0)
        self.v_scale_spin.setSingleStep(0.1)
        self.v_scale_spin.setDecimals(2)
        self.v_scale_spin.setValue(1.0)
        self.v_scale_spin.valueChanged.connect(self.on_hsv_scale_changed)
        color_layout.addRow(QLabel("V scale:"), self.v_scale_spin)

        # Container for the pipeline list
        pipeline_container = QWidget()
        pipeline_layout = QVBoxLayout(pipeline_container)
        pipeline_layout.setContentsMargins(0, 0, 0, 0)
        pipeline_layout.setSpacing(4)
        self.processing_pipeline.setMinimumHeight(160)
        pipeline_layout.addWidget(self.processing_pipeline)

        right_layout.addWidget(self.roi_manager)
        right_layout.addWidget(self.histogram_widget)
        right_layout.addWidget(color_ctrl)
        right_layout.addWidget(self.plugin_manager)
        right_layout.addWidget(pipeline_container)
        # Give more vertical space to the pipeline + parameter panel
        right_layout.setStretch(0, 1)  # ROI manager
        right_layout.setStretch(1, 1)  # histogram
        right_layout.setStretch(2, 0)  # color controls
        right_layout.setStretch(3, 1)  # plugin list (Filters/Detectors)
        right_layout.setStretch(4, 3)  # pipeline + parameters
        self.right_dock = QDockWidget("", self)
        self.right_dock.setWidget(right_widget)
        self.right_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.right_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)

        # Create menu bar
        self.create_menu_bar()

        # Connect signals
        self.image_viewer.load_image_requested.connect(self.load_image)
        self.image_viewer.roi_changed.connect(self.roi_manager.set_roi)
        # Allow ROI edits started from the compare viewer to drive the shared ROI
        self.image_viewer_compare.roi_changed.connect(self.roi_manager.set_roi)
        self.image_viewer.view_changed.connect(self.on_view_changed_from_main)
        self.image_viewer_compare.view_changed.connect(self.on_view_changed_from_compare)
        self.plugin_manager.plugin_selected.connect(self.on_plugin_selected)
        self.processing_pipeline.pipeline_updated.connect(self.on_pipeline_updated)
        self.parameter_panel.parameter_changed.connect(self.on_parameter_changed)
        self.processing_pipeline.plugin_selected.connect(self.parameter_panel.set_plugin)
        self.roi_manager.roi_changed.connect(self.on_roi_changed)
        self.histogram_widget.lut_changed.connect(self.on_lut_changed)

        self.restore_session()

    def create_menu_bar(self):
        """Create the menu bar with File menu"""
        menubar = self.menuBar()
        
        # Create File menu
        file_menu = menubar.addMenu('File')
        
        # Add Load Image action
        load_action = QAction('Load Image', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)

        # Add Load Compare Image action
        load_compare_action = QAction('Load Compare Image...', self)
        load_compare_action.setShortcut('Ctrl+Shift+O')
        load_compare_action.triggered.connect(self.load_compare_image)
        file_menu.addAction(load_compare_action)

        # Add Clear Compare Image action
        clear_compare_action = QAction('Clear Compare Image', self)
        clear_compare_action.triggered.connect(self.clear_compare_image)
        file_menu.addAction(clear_compare_action)
        
        # Add separator
        file_menu.addSeparator()
        
        # Add Save Pipeline action
        save_pipeline_action = QAction('Save Pipeline', self)
        save_pipeline_action.setShortcut('Ctrl+S')
        save_pipeline_action.triggered.connect(self.processing_pipeline.save_pipeline)
        file_menu.addAction(save_pipeline_action)
        
        # Add Load Pipeline action
        load_pipeline_action = QAction('Load Pipeline', self)
        load_pipeline_action.setShortcut('Ctrl+L')
        load_pipeline_action.triggered.connect(self.load_pipeline)
        file_menu.addAction(load_pipeline_action)

        # Add separator
        file_menu.addSeparator()

        # Add Export View action
        export_view_action = QAction('Export View', self)
        export_view_action.setShortcut('Ctrl+E')
        export_view_action.triggered.connect(self.export_view_with_annotations)
        file_menu.addAction(export_view_action)
        
        # Add separator
        file_menu.addSeparator()
        
        # Add Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Create View menu
        view_menu = menubar.addMenu('View')

        # Toggle analysis panel (ROI / histogram / plugins / pipeline)
        self.toggle_analysis_action = QAction('Show Analysis Panel', self)
        self.toggle_analysis_action.setCheckable(True)
        self.toggle_analysis_action.setChecked(True)
        self.toggle_analysis_action.setShortcut('F9')
        self.toggle_analysis_action.toggled.connect(self.toggle_analysis_panel)
        view_menu.addAction(self.toggle_analysis_action)

    def export_view_with_annotations(self):
        """Export the current viewer pixmap (including ROI and measurements) as an image."""
        pixmap = self.image_viewer.image_label.pixmap()
        if pixmap is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export View",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)"
        )

        if file_name:
            pixmap.save(file_name)

    def set_compare_mode(self, enabled: bool):
        """Enable or disable dual-image compare mode."""
        enabled = bool(enabled)
        if enabled == self.compare_mode_enabled:
            return

        self.compare_mode_enabled = enabled

        if enabled:
            if self.image_viewer_compare not in self._iter_splitter_widgets(self.viewer_splitter):
                self.viewer_splitter.addWidget(self.image_viewer_compare)
            self.image_viewer_compare.show()
            # Try to start with the same view as the main viewer
            self.image_viewer_compare.apply_view_state(self.image_viewer.get_view_state())
        else:
            # Remove compare viewer from splitter and clear its state
            index = self.viewer_splitter.indexOf(self.image_viewer_compare)
            if index != -1:
                self.viewer_splitter.widget(index).hide()
                self.viewer_splitter.removeWidget(self.image_viewer_compare)
            self.image_viewer_compare.clear_edge_overlay()
            self.image_viewer_compare.set_roi(None)
            self.image_viewer_compare.update_image(None) if hasattr(self.image_viewer_compare, "update_image") else None
            self.compare_original_image = None

    def _iter_splitter_widgets(self, splitter):
        """Yield all direct child widgets of a QSplitter."""
        return [splitter.widget(i) for i in range(splitter.count())]

    def load_compare_image(self):
        """Load an image into the compare viewer and enable compare mode."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Compare Image File",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )

        if not file_name:
            return

        if self.image_viewer_compare.load_image(file_name):
            # Remember original compare image for pipeline processing
            img = self.image_viewer_compare.get_current_image()
            self.compare_original_image = img.copy() if img is not None else None
            self.image_viewer_compare.set_image_path(file_name)
            if not self.compare_mode_enabled:
                self.set_compare_mode(True)
            # Apply current ROI to compare viewer if present
            roi = self.roi_manager.get_roi()
            if roi:
                self.image_viewer_compare.set_roi(roi)
            # If a pipeline result exists, run it on the compare image as well
            if self.processing_pipeline.original_image is not None and self.compare_original_image is not None:
                roi_for_processing = self.roi_manager.get_roi()
                compare_result = self.processing_pipeline.run_on_image(
                    self.compare_original_image,
                    roi=roi_for_processing,
                )
                if compare_result is not None:
                    self.image_viewer_compare.update_image(compare_result)
        else:
            QMessageBox.warning(self, "Error", f"Failed to load compare image: {file_name}")

    def clear_compare_image(self):
        """Clear and hide the compare image, disabling compare mode."""
        self.compare_original_image = None
        self.set_compare_mode(False)

    def load_image(self):
        """Open file dialog to load an image and auto-set ROI"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        
        if file_name:
            if self.image_viewer.load_image(file_name):
                self.setWindowTitle(f"VisionTool - {file_name}")
                self.last_image_path = file_name
                img = self.image_viewer.get_current_image()
                if img is not None:
                    self.processing_pipeline.set_original_image(img)  # Ensure pipeline has the original image
                    self.histogram_widget.set_image(img)
                    h, w = img.shape[:2]
                    roi_w = w // 4
                    roi_h = h // 4
                    roi_x = (w - roi_w) // 2
                    roi_y = (h - roi_h) // 2
                    roi = (roi_x, roi_y, roi_w, roi_h)
                    # Always update stored ROI values, but only show/apply it
                    # to the viewers when ROI is currently enabled.
                    self.roi_manager.set_roi(roi)
                    if self.roi_manager.roi_enabled:
                        self.image_viewer.set_roi(roi)
                self.save_session()
            else:
                QMessageBox.warning(self, "Error", f"Failed to load image: {file_name}")

    def toggle_analysis_panel(self, visible: bool):
        """Show or hide analysis panels (right dock and center-right widgets)."""
        is_visible = bool(visible)
        self.right_dock.setVisible(is_visible)
        self.right_center_widget.setVisible(is_visible)

    def on_plugin_selected(self, plugin):
        """Handle plugin selection"""
        self.processing_pipeline.add_plugin(plugin)
        self.parameter_panel.set_plugin(plugin)
        self.save_session()

    def load_pipeline(self):
        """Load a pipeline from a file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Load Pipeline",
            "",
            "Pipeline Files (*.json)"
        )
        
        if file_name:
            with open(file_name, 'r') as f:
                pipeline_data = json.load(f)
            
            # Clear current pipeline
            self.processing_pipeline.clear_pipeline()
            
            # Add plugins to pipeline
            for plugin_data in pipeline_data:
                plugin = self.plugin_manager.create_plugin(plugin_data['name'])
                if plugin:
                    # Set parameters if provided
                    if 'parameters' in plugin_data:
                        for name, value in plugin_data['parameters'].items():
                            plugin.set_parameter(name, value)
                    self.processing_pipeline.add_plugin(plugin)
                else:
                    QMessageBox.warning(
                        self,
                        "Plugin Not Found",
                        f"Plugin '{plugin_data['name']}' not found. Pipeline may be incomplete."
                    )

    def on_parameter_changed(self):
        """Handle parameter changes by running the pipeline"""
        if self.image_viewer.get_current_image() is not None:
            self.processing_pipeline.run_pipeline(self.image_viewer.get_current_image())
        self.save_session()

    def on_lut_changed(self, low, high):
        """Update viewer LUT from histogram widget."""
        self.image_viewer.set_lut(low, high)
        if self.compare_mode_enabled and self.compare_original_image is not None:
            self.image_viewer_compare.set_lut(low, high)

    def on_view_mode_changed(self, text):
        """Change viewer color / grayscale / HSV view mode."""
        mapping = {
            "RGB": "rgb",
            "Grayscale": "gray",
            "HSV - H": "h",
            "HSV - S": "s",
            "HSV - V": "v",
        }
        mode = mapping.get(text, "rgb")
        self.image_viewer.set_view_mode(mode)
        if self.compare_mode_enabled and self.compare_original_image is not None:
            self.image_viewer_compare.set_view_mode(mode)

    def on_hsv_scale_changed(self, *args):
        """Update HSV S/V scaling in the viewer."""
        s_scale = float(self.s_scale_spin.value())
        v_scale = float(self.v_scale_spin.value())
        self.image_viewer.set_hsv_scales(s_scale, v_scale)
        if self.compare_mode_enabled and self.compare_original_image is not None:
            self.image_viewer_compare.set_hsv_scales(s_scale, v_scale)

    def on_roi_changed(self, roi):
        """Handle ROI changes by updating the image viewer.

        The processing pipeline is normally run explicitly via the Run
        button. However, when the ROI is disabled/cleared, we also
        reset the pipeline output so the image no longer shows a
        previously processed ROI region.
        """
        # Update the image viewer ROI
        self.image_viewer.set_roi(roi)
        if self.compare_mode_enabled and self.compare_original_image is not None:
            self.image_viewer_compare.set_roi(roi)
        # Force a repaint of the image viewer
        self.image_viewer.repaint()

        # If ROI is disabled/cleared, reset pipeline output to original image
        if roi is None:
            self.processing_pipeline.reset_pipeline()

        self.save_session()

    def on_pipeline_updated(self, result):
        """Update viewers and histogram when the pipeline output changes."""
        if result is not None:
            self.image_viewer.update_image(result)
            self.histogram_widget.set_image(result)

        # Also run the same pipeline on the compare image, if available
        if self.compare_mode_enabled and self.compare_original_image is not None:
            roi = self.roi_manager.get_roi()
            compare_result = self.processing_pipeline.run_on_image(
                self.compare_original_image,
                roi=roi,
            )
            if compare_result is not None:
                self.image_viewer_compare.update_image(compare_result)

    def on_view_changed_from_main(self, state):
        """Mirror main viewer's view changes to the compare viewer."""
        if not self.compare_mode_enabled or self.compare_original_image is None:
            return
        if self._syncing_view:
            return
        self._syncing_view = True
        try:
            self.image_viewer_compare.apply_view_state(state)
        finally:
            self._syncing_view = False

    def on_view_changed_from_compare(self, state):
        """Optionally mirror compare viewer changes back to the main viewer."""
        if not self.compare_mode_enabled or self.compare_original_image is None:
            return
        if self._syncing_view:
            return
        self._syncing_view = True
        try:
            self.image_viewer.apply_view_state(state)
        finally:
            self._syncing_view = False

    def save_session(self):
        session = {
            'image_path': self.last_image_path,
            'roi': self.roi_manager.get_roi(),
            'pipeline': [
                {
                    'name': plugin.get_name(),
                    'parameters': plugin.get_parameters()
                } for plugin in self.processing_pipeline.get_pipeline()
            ]
        }
        try:
            with open(SESSION_FILE, 'w') as f:
                json.dump(session, f, indent=2)
        except Exception as e:
            print(f"Failed to save session: {e}")

    def restore_session(self):
        if not os.path.exists(SESSION_FILE):
            return
        try:
            with open(SESSION_FILE, 'r') as f:
                session = json.load(f)
            # Restore image
            image_path = session.get('image_path')
            if image_path and os.path.exists(image_path):
                self.image_viewer.load_image(image_path)
                self.last_image_path = image_path
                img = self.image_viewer.get_current_image()
                if img is not None:
                    self.processing_pipeline.set_original_image(img)  # Ensure pipeline has the original image
            # Restore ROI
            roi = session.get('roi')
            if roi:
                self.roi_manager.set_roi(tuple(roi))
                if self.roi_manager.roi_enabled:
                    self.image_viewer.set_roi(tuple(roi))
            # Restore pipeline
            pipeline = session.get('pipeline', [])
            self.processing_pipeline.clear_pipeline()
            for plugin_data in pipeline:
                plugin = self.plugin_manager.create_plugin(plugin_data['name'])
                if plugin:
                    for name, value in plugin_data['parameters'].items():
                        plugin.set_parameter(name, value)
                    self.processing_pipeline.add_plugin(plugin)
        except Exception as e:
            print(f"Failed to restore session: {e}")

def main():
    print("Entering main function")
    app = QApplication(sys.argv)
    print("QApplication created")
    window = MainWindow()
    print("MainWindow created")
    window.show()
    print("Window shown")
    sys.exit(app.exec_())
    print("Application exited")

if __name__ == "__main__":
    main() 