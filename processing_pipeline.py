import json
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QListWidget,
                             QPushButton, QHBoxLayout, QListWidgetItem,
                             QFileDialog)
from PyQt5.QtCore import pyqtSignal, Qt

class ProcessingPipeline(QWidget):
    pipeline_updated = pyqtSignal(object)
    plugin_added = pyqtSignal(object)
    plugin_selected = pyqtSignal(object)

    def __init__(self, image_viewer, edge_measure_panel=None):
        super().__init__()
        self.image_viewer = image_viewer
        self.edge_measure_panel = edge_measure_panel
        self.init_ui()
        self.pipeline = []
        self.original_image = None  # Store the original image

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create pipeline list
        self.pipeline_list = QListWidget()
        self.pipeline_list.currentItemChanged.connect(self.on_plugin_selected)
        self.pipeline_list.itemChanged.connect(self.on_item_changed)
        layout.addWidget(self.pipeline_list)
        
        # Create control buttons
        button_layout = QHBoxLayout()
        
        self.move_up_btn = QPushButton("Move Up")
        self.move_up_btn.clicked.connect(self.move_up)
        button_layout.addWidget(self.move_up_btn)
        
        self.move_down_btn = QPushButton("Move Down")
        self.move_down_btn.clicked.connect(self.move_down)
        button_layout.addWidget(self.move_down_btn)
        
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_plugin)
        button_layout.addWidget(self.remove_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_pipeline)
        button_layout.addWidget(self.reset_btn)

        # Explicit run button so pipeline only runs when requested
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run_now)
        button_layout.addWidget(self.run_btn)
        
        layout.addLayout(button_layout)

        # Create save/load buttons
        save_load_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save Pipeline")
        self.save_btn.clicked.connect(self.save_pipeline)
        save_load_layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("Load Pipeline")
        self.load_btn.clicked.connect(self.load_pipeline)
        save_load_layout.addWidget(self.load_btn)
        
        layout.addLayout(save_load_layout)

    def add_plugin(self, plugin):
        """Add a plugin to the pipeline"""
        self.pipeline.append(plugin)
        # Optionally provide context to plugins that support it
        if hasattr(plugin, "set_context"):
            try:
                plugin.set_context(self.image_viewer, self.edge_measure_panel)
            except TypeError:
                # Backwards compatibility if set_context has a different signature
                try:
                    plugin.set_context(self.image_viewer)
                except Exception:
                    pass
        item = QListWidgetItem(plugin.get_name())
        # Make the item checkable so steps can be enabled/disabled
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setCheckState(Qt.Checked)
        self.pipeline_list.addItem(item)
        self.plugin_added.emit(plugin)
        # Do not auto-run here; user can press Run

    def remove_plugin(self):
        """Remove the selected plugin from the pipeline"""
        current_row = self.pipeline_list.currentRow()
        if current_row >= 0:
            self.pipeline_list.takeItem(current_row)
            self.pipeline.pop(current_row)
            # Do not auto-run here; user can press Run

    def move_up(self):
        """Move the selected plugin up in the pipeline"""
        current_row = self.pipeline_list.currentRow()
        if current_row > 0:
            # Move in list widget
            item = self.pipeline_list.takeItem(current_row)
            self.pipeline_list.insertItem(current_row - 1, item)
            self.pipeline_list.setCurrentItem(item)
            
            # Move in pipeline list
            self.pipeline[current_row], self.pipeline[current_row - 1] = \
                self.pipeline[current_row - 1], self.pipeline[current_row]
            
            # Do not auto-run here; user can press Run

    def move_down(self):
        """Move the selected plugin down in the pipeline"""
        current_row = self.pipeline_list.currentRow()
        if current_row < self.pipeline_list.count() - 1:
            # Move in list widget
            item = self.pipeline_list.takeItem(current_row)
            self.pipeline_list.insertItem(current_row + 1, item)
            self.pipeline_list.setCurrentItem(item)
            
            # Move in pipeline list
            self.pipeline[current_row], self.pipeline[current_row + 1] = \
                self.pipeline[current_row + 1], self.pipeline[current_row]
            
            # Run pipeline to update the image
            if self.image_viewer.get_current_image() is not None:
                self.run_pipeline(self.image_viewer.get_current_image())

    def set_original_image(self, image):
        """Explicitly set the original image (should be called when loading a new image)"""
        if image is not None:
            self.original_image = image.copy()

    def run_pipeline(self, image=None):
        """Run the pipeline on the input image"""
        # Do not set self.original_image here!
        if self.original_image is None:
            return None

        # If no pipeline, return original image
        if not self.pipeline:
            self.pipeline_updated.emit(self.original_image.copy())
            return self.original_image.copy()

        roi = self.image_viewer.get_roi()
        # Ensure ROI is a tuple of 4 ints
        if isinstance(roi, tuple) and len(roi) == 4 and all(isinstance(v, int) for v in roi):
            x, y, w, h = roi
            roi_image = self.original_image[y:y+h, x:x+w].copy()
            result = roi_image.copy()
            for idx, plugin in enumerate(self.pipeline):
                # Skip disabled steps (unchecked items)
                item = self.pipeline_list.item(idx)
                if item is not None and item.checkState() == Qt.Unchecked:
                    continue
                if result is not None:
                    result = plugin.process(result)
            output = self.original_image.copy()
            output[y:y+h, x:x+w] = result
            result = output
        else:
            # No valid ROI, show original image (do not process)
            result = self.original_image.copy()

        if result is not None:
            self.pipeline_updated.emit(result)
        return result

    def get_pipeline(self):
        """Get the current pipeline"""
        return self.pipeline.copy()

    def reset_pipeline(self):
        """Reset the pipeline by showing the original image"""
        if self.original_image is not None:
            self.pipeline_updated.emit(self.original_image.copy())

    def clear_pipeline(self):
        """Clear the pipeline"""
        self.pipeline.clear()
        self.pipeline_list.clear()
        # Do not auto-run here; user can press Run

    def save_pipeline(self):
        """Save the current pipeline to a JSON file"""
        if not self.pipeline:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Pipeline",
            "",
            "Pipeline Files (*.json)"
        )
        
        if file_name:
            pipeline_data = []
            for plugin in self.pipeline:
                plugin_data = {
                    'name': plugin.get_name(),
                    'parameters': plugin.get_parameters()
                }
                pipeline_data.append(plugin_data)
            
            with open(file_name, 'w') as f:
                json.dump(pipeline_data, f, indent=4)

    def load_pipeline(self):
        """Load a pipeline from a JSON file"""
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
            self.clear_pipeline()
            
            # Add plugins to pipeline
            for plugin_data in pipeline_data:
                self.plugin_added.emit(plugin_data)
            
            # Do not auto-run here; user can press Run

    def on_plugin_selected(self, current, previous):
        """Handle plugin selection in the pipeline list"""
        if current is not None:
            row = self.pipeline_list.row(current)
            if 0 <= row < len(self.pipeline):
                self.plugin_selected.emit(self.pipeline[row])

    def on_item_changed(self, item):
        """Handle step enable/disable; pipeline runs only when user presses Run."""
        return

    def run_now(self):
        """Run the pipeline explicitly when the user presses the Run button."""
        if self.image_viewer.get_current_image() is not None:
            self.run_pipeline(self.image_viewer.get_current_image())