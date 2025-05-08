import sys
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                            QMenuBar, QMenu, QAction, QFileDialog, QMessageBox, QSplitter)
from PyQt5.QtCore import Qt
from image_viewer import ImageViewer
from plugin_manager import PluginManager
from processing_pipeline import ProcessingPipeline
from parameter_panel import ParameterPanel
from roi_manager import ROIManager

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionTool")
        self.setGeometry(100, 100, 1200, 800)

        # Central image viewer
        self.image_viewer = ImageViewer()
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.addWidget(self.image_viewer)
        self.setCentralWidget(central_widget)

        # Create ROI manager dock
        self.roi_manager = ROIManager(self.image_viewer)
        roi_dock = QDockWidget("ROI Controls", self)
        roi_dock.setWidget(self.roi_manager)
        roi_dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        roi_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.LeftDockWidgetArea, roi_dock)

        # Remove right-side tabbing, use a composite widget for right dock
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        self.plugin_manager = PluginManager()
        self.processing_pipeline = ProcessingPipeline(self.image_viewer)
        right_layout.addWidget(self.plugin_manager, stretch=2)
        right_layout.addWidget(self.processing_pipeline, stretch=1)
        right_dock = QDockWidget("", self)
        right_dock.setWidget(right_widget)
        right_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        right_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)

        # Create parameter panel dock
        self.parameter_panel = ParameterPanel()
        param_dock = QDockWidget("Parameters", self)
        param_dock.setWidget(self.parameter_panel)
        param_dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        param_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.LeftDockWidgetArea, param_dock)
        self.tabifyDockWidget(roi_dock, param_dock)
        roi_dock.raise_()

        # Create menu bar
        self.create_menu_bar()

        # Connect signals
        self.plugin_manager.plugin_selected.connect(self.on_plugin_selected)
        self.processing_pipeline.pipeline_updated.connect(self.image_viewer.update_image)
        self.parameter_panel.parameter_changed.connect(self.on_parameter_changed)
        self.processing_pipeline.plugin_selected.connect(self.parameter_panel.set_plugin)

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
        
        # Add Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

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
                # Auto-set ROI in center
                img = self.image_viewer.get_current_image()
                if img is not None:
                    h, w = img.shape[:2]
                    roi_w = w // 4
                    roi_h = h // 4
                    roi_x = (w - roi_w) // 2
                    roi_y = (h - roi_h) // 2
                    roi = (roi_x, roi_y, roi_w, roi_h)
                    self.roi_manager.set_roi(roi)
                    self.image_viewer.set_roi(roi)
            else:
                QMessageBox.warning(self, "Error", f"Failed to load image: {file_name}")

    def on_plugin_selected(self, plugin):
        """Handle plugin selection"""
        self.processing_pipeline.add_plugin(plugin)
        self.parameter_panel.set_plugin(plugin)

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

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 