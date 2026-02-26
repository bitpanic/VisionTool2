from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, 
                             QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox)
from PyQt5.QtCore import pyqtSignal

class ParameterPanel(QWidget):
    parameter_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_plugin = None
        self.parameter_widgets = {}

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create form layout for parameters
        self.form_layout = QFormLayout()
        layout.addLayout(self.form_layout)
        
        # Add stretch to push everything to the top
        layout.addStretch()

    def set_plugin(self, plugin):
        """Set the current plugin and update the parameter panel"""
        self.current_plugin = plugin
        self.clear_parameters()
        
        if plugin:
            self.add_parameters(plugin.get_parameters())

    def clear_parameters(self):
        """Clear all parameter widgets"""
        # Remove all widgets from the form layout
        while self.form_layout.rowCount() > 0:
            self.form_layout.removeRow(0)
        
        self.parameter_widgets.clear()

    def add_parameters(self, parameters):
        """Add parameter widgets based on parameter types"""
        for name, value in parameters.items():
            if isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
                widget.stateChanged.connect(lambda state, n=name: self.on_parameter_changed(n, state))
            
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(-999999, 999999)
                widget.setValue(value)
                widget.valueChanged.connect(lambda v, n=name: self.on_parameter_changed(n, v))
            
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setRange(-999999.0, 999999.0)
                widget.setDecimals(3)
                widget.setValue(value)
                widget.valueChanged.connect(lambda v, n=name: self.on_parameter_changed(n, v))
            
            elif isinstance(value, (list, tuple)):
                widget = QComboBox()
                for item in value:
                    widget.addItem(str(item))
                widget.currentTextChanged.connect(lambda v, n=name: self.on_parameter_changed(n, v))
            
            # Strings are used by some plugins (e.g. Edge Measurement) for
            # enum-like parameters such as mode or gradient method. For these,
            # show a combo box with known choices.
            elif isinstance(value, str):
                choices_map = {
                    "Measurement mode": [
                        "Along line (single profile)",
                        "Perpendicular cross-sections (multi profile)",
                    ],
                    "Sampling mode": [
                        "Step (px)",
                        "Num samples",
                    ],
                    "Smoothing type": [
                        "None",
                        "Gaussian",
                        "Savitzky-Golay",
                    ],
                    "Gradient method": [
                        "Sobel",
                        "Scharr",
                    ],
                    "Overlay metric": [
                        "peak_to_peak",
                        "edge_width",
                    ],
                }
                if name in choices_map:
                    widget = QComboBox()
                    for item in choices_map[name]:
                        widget.addItem(str(item))
                    # Try to restore previous selection if possible
                    idx = widget.findText(value)
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                    widget.currentTextChanged.connect(lambda v, n=name: self.on_parameter_changed(n, v))
                else:
                    continue  # Skip unsupported plain string parameters

            else:
                continue  # Skip unsupported parameter types
            
            self.parameter_widgets[name] = widget
            self.form_layout.addRow(name, widget)

    def on_parameter_changed(self, name, value):
        """Handle parameter value changes"""
        if self.current_plugin:
            self.current_plugin.set_parameter(name, value)
            self.parameter_changed.emit()

    def get_parameter_values(self):
        """Get current values of all parameters"""
        values = {}
        for name, widget in self.parameter_widgets.items():
            if isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                values[name] = widget.value()
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentText()
        return values 