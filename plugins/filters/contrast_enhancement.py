import cv2
import numpy as np
from plugins.base_plugin import FilterPlugin

class ContrastEnhancement(FilterPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Contrast Enhancement"
        self.description = "Enhances image contrast using CLAHE"
        self.parameters = {
            "clip_limit": 2.0,
            "tile_size": 8,
            "apply_clahe": True
        }

    def process(self, image):
        """Apply contrast enhancement to the image"""
        if image is None:
            return None

        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        if self.parameters["apply_clahe"]:
            # Sanitize parameters to avoid OpenCV crashes when values go negative/zero
            clip_limit = float(self.parameters.get("clip_limit", 2.0))
            tile_size = int(self.parameters.get("tile_size", 8))

            # OpenCV requires clipLimit > 0 and tileGridSize >= 1
            if clip_limit <= 0:
                clip_limit = 0.01
            if tile_size < 1:
                tile_size = 1

            # Create CLAHE object with safe parameters
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=(tile_size, tile_size)
            )
            
            # Apply CLAHE to L channel
            l = clahe.apply(l)

        # Merge channels
        enhanced = cv2.merge([l, a, b])
        
        # Convert back to BGR
        result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return result 