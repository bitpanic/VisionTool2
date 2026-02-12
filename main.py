import sys
import json
import os
print("Starting application...")
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                            QMenuBar, QMenu, QAction, QFileDialog, QMessageBox, QSplitter)
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

        # Central image viewer
        self.image_viewer = ImageViewer()
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.addWidget(self.image_viewer)
        self.setCentralWidget(central_widget)

        # Left: Only Parameters panel (fixed width)
        self.parameter_panel = ParameterPanel()
        left_dock = QDockWidget("Parameters", self)
        left_dock.setWidget(self.parameter_panel)
        left_dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        left_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        fixed_width = 260
        left_dock.setMinimumWidth(fixed_width)
        left_dock.setMaximumWidth(fixed_width)
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)

        # Right: ROI controls, Histogram, Plugins, Pipeline
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        self.roi_manager = ROIManager(self.image_viewer)
        self.histogram_widget = HistogramWidget()
        self.plugin_manager = PluginManager()
        self.processing_pipeline = ProcessingPipeline(self.image_viewer)
        self.roi_manager.setMinimumHeight(120)
        self.histogram_widget.setMinimumHeight(140)
        self.plugin_manager.setMinimumHeight(180)
        self.processing_pipeline.setMinimumHeight(180)
        right_layout.addWidget(self.roi_manager)
        right_layout.addWidget(self.histogram_widget)
        right_layout.addWidget(self.plugin_manager)
        right_layout.addWidget(self.processing_pipeline)
        right_layout.setStretch(0, 1)
        right_layout.setStretch(1, 1)
        right_layout.setStretch(2, 2)
        right_layout.setStretch(3, 2)
        right_dock = QDockWidget("", self)
        right_dock.setWidget(right_widget)
        right_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        right_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)

        # Create menu bar
        self.create_menu_bar()

        # Connect signals
        self.image_viewer.load_image_requested.connect(self.load_image)
        self.image_viewer.roi_changed.connect(self.roi_manager.set_roi)
        self.plugin_manager.plugin_selected.connect(self.on_plugin_selected)
        self.processing_pipeline.pipeline_updated.connect(self.image_viewer.update_image)
        self.processing_pipeline.pipeline_updated.connect(self.histogram_widget.set_image)
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
                    self.roi_manager.set_roi(roi)
                    self.image_viewer.set_roi(roi)
                self.save_session()
            else:
                QMessageBox.warning(self, "Error", f"Failed to load image: {file_name}")

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

    def on_roi_changed(self, roi):
        """Handle ROI changes by updating the image viewer and running the pipeline"""
        # First update the image viewer
        self.image_viewer.set_roi(roi)
        # Force a repaint of the image viewer
        self.image_viewer.repaint()
        # Then run the pipeline with the original image
        if self.image_viewer.get_current_image() is not None:
            # Get the original image from the pipeline
            original_image = self.image_viewer.get_current_image()
            # Run pipeline with original image
            self.processing_pipeline.run_pipeline(original_image)
        self.save_session()

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