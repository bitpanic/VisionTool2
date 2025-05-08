import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize, QPoint, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QCursor, QPainter, QPen, QColor

class ImageViewer(QWidget):
    load_image_requested = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_image = None
        self.roi = None
        self.scale_factor = 1.0
        self.pan_start = None
        self.last_pos = None
        self.display_image_data = None  # Store the original image data

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create toolbar with load and zoom buttons
        toolbar = QHBoxLayout()
        
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_image_requested.emit)
        toolbar.addWidget(self.load_image_btn)
        
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.clicked.connect(lambda: self.zoom(1.25))
        toolbar.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(lambda: self.zoom(0.8))
        toolbar.addWidget(self.zoom_out_btn)
        
        self.zoom_fit_btn = QPushButton("Zoom to Fit")
        self.zoom_fit_btn.clicked.connect(self.zoom_to_fit)
        toolbar.addWidget(self.zoom_fit_btn)
        
        self.zoom_roi_btn = QPushButton("Zoom to ROI")
        self.zoom_roi_btn.clicked.connect(self.zoom_to_roi)
        toolbar.addWidget(self.zoom_roi_btn)
        
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        # Create scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)
        
        # Create image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        
        # Enable mouse tracking for pan
        self.image_label.setMouseTracking(True)
        
        # Set focus policy to accept wheel events
        self.setFocusPolicy(Qt.StrongFocus)

    def load_image(self, image_path):
        """Load an image from file"""
        self.current_image = cv2.imread(image_path)
        if self.current_image is not None:
            self.display_image(self.current_image)
            self.zoom_to_fit()
            return True
        return False

    def display_image(self, image):
        """Display the given image"""
        if image is None:
            return

        # Convert BGR to RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        
        # Store the image data
        self.display_image_data = image.copy()
        
        # Create QImage from numpy array
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Create pixmap and scale it
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            int(width * self.scale_factor),
            int(height * self.scale_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # Draw ROI if exists
        if self.roi is not None:
            painter = QPainter(scaled_pixmap)
            
            # Draw ROI border
            pen = QPen(QColor(255, 0, 0))  # Red color
            pen.setWidth(2)
            painter.setPen(pen)
            
            x, y, w, h = self.roi
            scaled_x = int(x * self.scale_factor)
            scaled_y = int(y * self.scale_factor)
            scaled_w = int(w * self.scale_factor)
            scaled_h = int(h * self.scale_factor)
            
            # Draw semi-transparent overlay
            painter.setBrush(QColor(255, 0, 0, 30))  # Semi-transparent red
            painter.drawRect(scaled_x, scaled_y, scaled_w, scaled_h)
            
            # Draw border
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(scaled_x, scaled_y, scaled_w, scaled_h)
            
            painter.end()
        
        self.image_label.setPixmap(scaled_pixmap)

    def update_image(self, image):
        """Update the displayed image"""
        self.current_image = image.copy()
        self.display_image(image)

    def get_current_image(self):
        """Return the current image"""
        return self.current_image.copy() if self.current_image is not None else None

    def set_roi(self, roi):
        """Set the Region of Interest"""
        self.roi = roi
        if self.current_image is not None:
            self.display_image(self.current_image)
            self.repaint()  # Force immediate repaint

    def get_roi(self):
        """Get the current Region of Interest"""
        return self.roi

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if self.current_image is not None:
            # Get the amount of scroll
            delta = event.angleDelta().y()
            
            # Calculate zoom factor based on scroll direction
            factor = 1.25 if delta > 0 else 0.8
            
            # Get cursor position relative to the image
            pos = self.image_label.mapFrom(self, event.pos())
            
            # Zoom around cursor position
            self.zoom(factor, pos)

    def zoom(self, factor, pos=None):
        """Zoom the image by the given factor"""
        if self.current_image is None:
            return
            
        old_factor = self.scale_factor
        self.scale_factor *= factor
        
        # Limit zoom range
        self.scale_factor = max(0.1, min(10.0, self.scale_factor))
        
        # Update display
        self.display_image(self.current_image)
        
        # Adjust scroll position to keep the point under cursor fixed
        if pos is not None and self.scale_factor != old_factor:
            scroll_x = self.scroll_area.horizontalScrollBar()
            scroll_y = self.scroll_area.verticalScrollBar()
            
            rel_x = pos.x() / self.image_label.width()
            rel_y = pos.y() / self.image_label.height()
            
            new_x = int(rel_x * self.image_label.width() - pos.x() + scroll_x.value())
            new_y = int(rel_y * self.image_label.height() - pos.y() + scroll_y.value())
            
            scroll_x.setValue(new_x)
            scroll_y.setValue(new_y)

    def zoom_to_fit(self):
        """Zoom to fit the window"""
        if self.current_image is None:
            return
            
        # Calculate scale factor to fit the window
        view_size = self.scroll_area.size()
        image_size = self.current_image.shape[:2][::-1]  # width, height
        
        scale_x = view_size.width() / image_size[0]
        scale_y = view_size.height() / image_size[1]
        
        self.scale_factor = min(scale_x, scale_y)
        self.display_image(self.current_image)

    def zoom_to_roi(self):
        """Zoom to the current ROI"""
        if self.current_image is None or self.roi is None:
            return
            
        x, y, width, height = self.roi
        view_size = self.scroll_area.size()
        
        scale_x = view_size.width() / width
        scale_y = view_size.height() / height
        
        self.scale_factor = min(scale_x, scale_y) * 0.9  # 90% to show some context
        self.display_image(self.current_image)
        
        # Center ROI in view
        scroll_x = self.scroll_area.horizontalScrollBar()
        scroll_y = self.scroll_area.verticalScrollBar()
        
        scroll_x.setValue(int(x * self.scale_factor - (view_size.width() - width * self.scale_factor) / 2))
        scroll_y.setValue(int(y * self.scale_factor - (view_size.height() - height * self.scale_factor) / 2))

    def mousePressEvent(self, event):
        """Handle mouse press for panning"""
        if event.button() == Qt.LeftButton:
            self.pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            self.last_pos = event.pos()

    def mouseReleaseEvent(self, event):
        """Handle mouse release for panning"""
        if event.button() == Qt.LeftButton:
            self.pan_start = None
            self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, event):
        """Handle mouse move for panning"""
        if self.pan_start is not None:
            # Calculate movement
            delta = event.pos() - self.last_pos
            self.last_pos = event.pos()
            
            # Update scroll bars
            scroll_x = self.scroll_area.horizontalScrollBar()
            scroll_y = self.scroll_area.verticalScrollBar()
            
            scroll_x.setValue(scroll_x.value() - delta.x())
            scroll_y.setValue(scroll_y.value() - delta.y()) 