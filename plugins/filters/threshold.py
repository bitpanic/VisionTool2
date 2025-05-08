import cv2
from plugins.base_plugin import FilterPlugin

class Threshold(FilterPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Threshold"
        self.description = "Applies a binary threshold to the image."
        self.parameters = {
            "thresh": 127,
            "maxval": 255
        }

    def process(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, result = cv2.threshold(gray, self.parameters["thresh"], self.parameters["maxval"], cv2.THRESH_BINARY)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR) if len(image.shape) == 3 else result 