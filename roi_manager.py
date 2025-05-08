from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                             QLabel, QSpinBox, QHBoxLayout)
from PyQt5.QtCore import Qt, pyqtSignal

class ROIManager(QWidget):
    roi_changed = pyqtSignal(object)

    def __init__(self, image_viewer):
        super().__init__()
        self.image_viewer = image_viewer
        self.init_ui()
        self.roi = None

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # ROI coordinates
        coords_layout = QHBoxLayout()
        
        # X coordinate
        x_layout = QVBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.x_spin = QSpinBox()
        self.x_spin.setRange(0, 9999)
        self.x_spin.valueChanged.connect(self.on_coords_changed)
        x_layout.addWidget(self.x_spin)
        coords_layout.addLayout(x_layout)
        
        # Y coordinate
        y_layout = QVBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.y_spin = QSpinBox()
        self.y_spin.setRange(0, 9999)
        self.y_spin.valueChanged.connect(self.on_coords_changed)
        y_layout.addWidget(self.y_spin)
        coords_layout.addLayout(y_layout)
        
        layout.addLayout(coords_layout)
        
        # ROI dimensions
        dims_layout = QHBoxLayout()
        
        # Width
        width_layout = QVBoxLayout()
        width_layout.addWidget(QLabel("Width:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 9999)
        self.width_spin.valueChanged.connect(self.on_coords_changed)
        width_layout.addWidget(self.width_spin)
        dims_layout.addLayout(width_layout)
        
        # Height
        height_layout = QVBoxLayout()
        height_layout.addWidget(QLabel("Height:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 9999)
        self.height_spin.valueChanged.connect(self.on_coords_changed)
        height_layout.addWidget(self.height_spin)
        dims_layout.addLayout(height_layout)
        
        layout.addLayout(dims_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear ROI")
        self.clear_btn.clicked.connect(self.clear_roi)
        button_layout.addWidget(self.clear_btn)
        
        self.apply_btn = QPushButton("Apply ROI")
        self.apply_btn.clicked.connect(self.apply_roi)
        button_layout.addWidget(self.apply_btn)
        
        layout.addLayout(button_layout)
        
        # Add stretch to push everything to the top
        layout.addStretch()

    def on_coords_changed(self):
        """Handle coordinate changes"""
        x = self.x_spin.value()
        y = self.y_spin.value()
        width = self.width_spin.value()
        height = self.height_spin.value()
        
        self.roi = (x, y, width, height)
        self.roi_changed.emit(self.roi)

    def clear_roi(self):
        """Clear the current ROI"""
        self.roi = None
        self.x_spin.setValue(0)
        self.y_spin.setValue(0)
        self.width_spin.setValue(1)
        self.height_spin.setValue(1)
        self.roi_changed.emit(None)

    def apply_roi(self):
        """Apply the current ROI to the image viewer"""
        if self.roi:
            self.image_viewer.set_roi(self.roi)

    def get_roi(self):
        """Get the current ROI"""
        return self.roi

    def set_roi(self, roi):
        """Set the ROI values"""
        if roi:
            x, y, width, height = roi
            self.x_spin.setValue(x)
            self.y_spin.setValue(y)
            self.width_spin.setValue(width)
            self.height_spin.setValue(height)
            self.roi = roi
        else:
            self.clear_roi() 