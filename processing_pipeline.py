import json
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QListWidget, 
                             QPushButton, QHBoxLayout, QListWidgetItem,
                             QFileDialog)
from PyQt5.QtCore import pyqtSignal

class ProcessingPipeline(QWidget):
    pipeline_updated = pyqtSignal(object)
    plugin_added = pyqtSignal(object)
    plugin_selected = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.pipeline = []

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create pipeline list
        self.pipeline_list = QListWidget()
        self.pipeline_list.currentItemChanged.connect(self.on_plugin_selected)
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
        
        self.run_btn = QPushButton("Run Pipeline")
        self.run_btn.clicked.connect(self.run_pipeline)
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
        item = QListWidgetItem(plugin.get_name())
        self.pipeline_list.addItem(item)
        self.plugin_added.emit(plugin)

    def remove_plugin(self):
        """Remove the selected plugin from the pipeline"""
        current_row = self.pipeline_list.currentRow()
        if current_row >= 0:
            self.pipeline_list.takeItem(current_row)
            self.pipeline.pop(current_row)

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

    def run_pipeline(self, image=None):
        """Run the pipeline on the input image"""
        if not self.pipeline or image is None:
            return image

        # Get ROI from image viewer
        roi = self.parent().parent().image_viewer.get_roi()
        
        if roi is None:
            # If no ROI, process entire image
            result = image.copy()
            for plugin in self.pipeline:
                if result is not None:
                    result = plugin.process(result)
        else:
            # Process only ROI region
            x, y, w, h = roi
            roi_image = image[y:y+h, x:x+w].copy()
            
            # Process ROI
            result = roi_image.copy()
            for plugin in self.pipeline:
                if result is not None:
                    result = plugin.process(result)
            
            # Create output image with processed ROI
            output = image.copy()
            output[y:y+h, x:x+w] = result
            result = output
        
        if result is not None:
            self.pipeline_updated.emit(result)
        return result

    def get_pipeline(self):
        """Get the current pipeline"""
        return self.pipeline.copy()

    def clear_pipeline(self):
        """Clear the pipeline"""
        self.pipeline.clear()
        self.pipeline_list.clear()

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

    def on_plugin_selected(self, current, previous):
        """Handle plugin selection in the pipeline list"""
        if current is not None:
            row = self.pipeline_list.row(current)
            if 0 <= row < len(self.pipeline):
                self.plugin_selected.emit(self.pipeline[row]) 