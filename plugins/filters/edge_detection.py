import cv2
import numpy as np
from plugins.base_plugin import FilterPlugin

class EdgeDetection(FilterPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Edge Detection"
        self.description = "Detects edges using the Canny algorithm."
        self.parameters = {
            "threshold1": 100,
            "threshold2": 200
        }

    def process(self, image):
        if image is None:
            return None
            
        # Create a copy of the input image to avoid modifying the original
        working_image = image.copy()
            
        # Convert to grayscale if needed
        if len(working_image.shape) == 3:
            gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_image.copy()
            
        # Apply edge detection
        edges = cv2.Canny(gray, self.parameters["threshold1"], self.parameters["threshold2"])
        
        # Create output image with same format as input
        if len(working_image.shape) == 3:
            # For color images, create a 3-channel image with edges
            result = np.zeros_like(working_image)
            result[..., 0] = edges  # Blue channel
            result[..., 1] = edges  # Green channel
            result[..., 2] = edges  # Red channel
        else:
            # For grayscale images, return edges directly
            result = edges
            
        return result 