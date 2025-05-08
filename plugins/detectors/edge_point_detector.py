import cv2
import numpy as np
from plugins.base_plugin import DetectorPlugin

class EdgePointDetector(DetectorPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Edge Point Detector"
        self.description = "Detects strong edge points using Canny and marks them."
        self.parameters = {
            "threshold1": 100,
            "threshold2": 200
        }

    def process(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, self.parameters["threshold1"], self.parameters["threshold2"])
        points = np.column_stack(np.where(edges > 0))
        result = image.copy()
        for pt in points:
            cv2.circle(result, (pt[1], pt[0]), 1, (255, 0, 0), -1)
        return result 