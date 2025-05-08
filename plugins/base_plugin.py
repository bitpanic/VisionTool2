from abc import ABC, abstractmethod
import cv2
import numpy as np

class BasePlugin(ABC):
    def __init__(self):
        self.name = "Base Plugin"
        self.description = "Base plugin class"
        self.parameters = {}

    @abstractmethod
    def process(self, image):
        """Process the input image and return the result"""
        pass

    def set_parameter(self, name, value):
        """Set a parameter value"""
        if name in self.parameters:
            self.parameters[name] = value
            return True
        return False

    def get_parameter(self, name):
        """Get a parameter value"""
        return self.parameters.get(name)

    def get_parameters(self):
        """Get all parameters"""
        return self.parameters.copy()

    def get_name(self):
        """Get plugin name"""
        return self.name

    def get_description(self):
        """Get plugin description"""
        return self.description

class FilterPlugin(BasePlugin):
    """Base class for filter plugins"""
    def __init__(self):
        super().__init__()
        self.category = "Filter"

class DetectorPlugin(BasePlugin):
    """Base class for detector plugins"""
    def __init__(self):
        super().__init__()
        self.category = "Detector" 