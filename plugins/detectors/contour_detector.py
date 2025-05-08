import cv2
import numpy as np
from plugins.base_plugin import DetectorPlugin

class ContourDetector(DetectorPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Contour Detector"
        self.description = "Detects contours in the image."
        self.parameters = {
            "threshold": 127
        }

    def process(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, thresh = cv2.threshold(gray, self.parameters["threshold"], 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = image.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        return result 