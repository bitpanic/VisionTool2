from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton,
                             QLabel, QSpinBox, QHBoxLayout, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal

class ROIManager(QWidget):
    roi_changed = pyqtSignal(object)

    def __init__(self, image_viewer):
        super().__init__()
        self.image_viewer = image_viewer
        self.init_ui()
        self.roi = None
        self.roi_enabled = True

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

        # Enable/disable ROI toggle
        self.enable_checkbox = QCheckBox("Enable ROI")
        self.enable_checkbox.setChecked(True)
        self.enable_checkbox.stateChanged.connect(self.on_enable_changed)
        button_layout.addWidget(self.enable_checkbox)
        
        layout.addLayout(button_layout)

        # Usage hints
        hints = QLabel(
            "Hints:\n"
            "• Hold Ctrl and drag in the image to create or move/resize the ROI.\n"
            "• Uncheck 'Enable ROI' to hide the ROI without losing its values.\n"
            "• Use 'Apply ROI' after editing values here to update the image."
        )
        hints.setWordWrap(True)
        hints.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(hints)

        # Add stretch to push everything to the top
        layout.addStretch()

    def on_coords_changed(self):
        """Handle coordinate changes"""
        x = self.x_spin.value()
        y = self.y_spin.value()
        width = self.width_spin.value()
        height = self.height_spin.value()
        
        self.roi = (x, y, width, height)
        if self.roi_enabled:
            self.image_viewer.set_roi(self.roi)
            self.roi_changed.emit(self.roi)

    def clear_roi(self):
        """Clear the current ROI"""
        self.roi = None
        # Block spinbox signals so we don't emit multiple roi_changed events
        for spin, value in (
            (self.x_spin, 0),
            (self.y_spin, 0),
            (self.width_spin, 1),
            (self.height_spin, 1),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        # Single update to viewer and listeners
        if self.roi_enabled:
            self.image_viewer.set_roi(None)
            self.roi_changed.emit(None)

    def apply_roi(self):
        """Apply the current ROI to the image viewer and trigger pipeline update"""
        if self.roi is not None and self.roi_enabled:
            self.image_viewer.set_roi(self.roi)
            self.roi_changed.emit(self.roi)

    def get_roi(self):
        """Get the current ROI"""
        if self.roi and isinstance(self.roi, tuple) and len(self.roi) == 4:
            return self.roi
        return None

    def set_roi(self, roi):
        """Set the ROI values"""
        if roi is not None:
            x, y, width, height = roi
            # Block signals while updating UI from code to avoid
            # repeated roi_changed emissions and slow updates
            for spin, value in (
                (self.x_spin, x),
                (self.y_spin, y),
                (self.width_spin, width),
                (self.height_spin, height),
            ):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
            self.roi = roi
            if self.roi_enabled:
                self.image_viewer.set_roi(self.roi)
                self.roi_changed.emit(self.roi)
        else:
            self.clear_roi() 

    def on_enable_changed(self, state):
        """Toggle ROI effect on/off without losing coordinates."""
        self.roi_enabled = state == Qt.Checked
        if self.roi_enabled:
            # Reapply stored ROI (if any)
            if self.roi is not None:
                self.image_viewer.set_roi(self.roi)
                self.roi_changed.emit(self.roi)
        else:
            # Temporarily disable ROI but keep coordinates in the spin boxes
            self.image_viewer.set_roi(None)
            self.roi_changed.emit(None)