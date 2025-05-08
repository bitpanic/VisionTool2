import cv2
import numpy as np
from plugins.base_plugin import DetectorPlugin

class BlobDetector(DetectorPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Blob Detector"
        self.description = "Detects blobs in the image using SimpleBlobDetector"
        self.parameters = {
            "min_threshold": 10,
            "max_threshold": 200,
            "threshold_step": 10,
            "min_area": 100,
            "max_area": 5000,
            "min_circularity": 0.1,
            "min_convexity": 0.87,
            "min_inertia_ratio": 0.01
        }

    def process(self, image):
        """Detect blobs in the image"""
        if image is None:
            return None

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Change thresholds
        params.minThreshold = self.parameters["min_threshold"]
        params.maxThreshold = self.parameters["max_threshold"]
        params.thresholdStep = self.parameters["threshold_step"]
        
        # Filter by Area
        params.filterByArea = True
        params.minArea = self.parameters["min_area"]
        params.maxArea = self.parameters["max_area"]
        
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = self.parameters["min_circularity"]
        
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = self.parameters["min_convexity"]
        
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = self.parameters["min_inertia_ratio"]
        
        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(gray)
        
        # Draw detected blobs
        result = image.copy()
        result = cv2.drawKeypoints(
            result, 
            keypoints, 
            np.array([]), 
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        return result 