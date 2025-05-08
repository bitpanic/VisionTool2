import cv2
from plugins.base_plugin import FilterPlugin

class GaussianBlur(FilterPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Gaussian Blur"
        self.description = "Applies Gaussian blur to the image."
        self.parameters = {
            "kernel_size": 5
        }

    def process(self, image):
        k = int(self.parameters["kernel_size"])
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1  # kernel size must be odd
        return cv2.GaussianBlur(image, (k, k), 0) 