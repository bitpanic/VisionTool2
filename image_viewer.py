import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize, QPoint, pyqtSignal, QRect
from PyQt5.QtGui import QImage, QPixmap, QCursor, QPainter, QPen, QColor

class ImageViewer(QWidget):
    load_image_requested = pyqtSignal()
    roi_changed = pyqtSignal(tuple)  # Signal to notify ROI changes

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_image = None
        self.roi = None
        self.scale_factor = 1.0
        self.pan_start = None
        self.last_pos = None
        self.display_image_data = None
        self.resize_handle = None  # Current resize handle being dragged
        self.resize_start = None   # Starting position for resize
        self.resize_start_roi = None  # ROI at start of resize
        self.handle_size = 8  # Size of resize handles in pixels
        self.new_roi_start = None  # Starting point for new ROI creation
        self.is_ctrl_pressed = False  # Track Ctrl key state

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
        self.scroll_area.wheelEvent = self.scroll_area_wheel_event  # Override wheel event
        layout.addWidget(self.scroll_area)
        
        # Create image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        
        # Enable mouse tracking for pan and ROI
        self.image_label.setMouseTracking(True)
        self.setMouseTracking(True)
        
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

    def get_handle_rect(self, x, y, w, h, handle):
        """Get the rectangle for a resize handle"""
        scaled_x = int(x * self.scale_factor)
        scaled_y = int(y * self.scale_factor)
        scaled_w = int(w * self.scale_factor)
        scaled_h = int(h * self.scale_factor)
        
        if handle == 'top_left':
            return QRect(scaled_x - self.handle_size//2, scaled_y - self.handle_size//2, 
                        self.handle_size, self.handle_size)
        elif handle == 'top_right':
            return QRect(scaled_x + scaled_w - self.handle_size//2, scaled_y - self.handle_size//2, 
                        self.handle_size, self.handle_size)
        elif handle == 'bottom_left':
            return QRect(scaled_x - self.handle_size//2, scaled_y + scaled_h - self.handle_size//2, 
                        self.handle_size, self.handle_size)
        elif handle == 'bottom_right':
            return QRect(scaled_x + scaled_w - self.handle_size//2, scaled_y + scaled_h - self.handle_size//2, 
                        self.handle_size, self.handle_size)
        elif handle == 'top':
            return QRect(scaled_x + scaled_w//2 - self.handle_size//2, scaled_y - self.handle_size//2, 
                        self.handle_size, self.handle_size)
        elif handle == 'bottom':
            return QRect(scaled_x + scaled_w//2 - self.handle_size//2, scaled_y + scaled_h - self.handle_size//2, 
                        self.handle_size, self.handle_size)
        elif handle == 'left':
            return QRect(scaled_x - self.handle_size//2, scaled_y + scaled_h//2 - self.handle_size//2, 
                        self.handle_size, self.handle_size)
        elif handle == 'right':
            return QRect(scaled_x + scaled_w - self.handle_size//2, scaled_y + scaled_h//2 - self.handle_size//2, 
                        self.handle_size, self.handle_size)
        return None

    def get_handle_at_pos(self, pos):
        """Get the resize handle at the given position"""
        if self.roi is None:
            return None
            
        x, y, w, h = self.roi
        handles = ['top_left', 'top', 'top_right', 'right', 'bottom_right', 
                  'bottom', 'bottom_left', 'left']
        
        # Get scroll position
        scroll_x = self.scroll_area.horizontalScrollBar().value()
        scroll_y = self.scroll_area.verticalScrollBar().value()
        
        # Adjust mouse position for scroll
        adjusted_pos = QPoint(pos.x() + scroll_x, pos.y() + scroll_y)
        
        for handle in handles:
            rect = self.get_handle_rect(x, y, w, h, handle)
            if rect and rect.contains(adjusted_pos):
                return handle
                
        # Check if point is inside ROI (for moving)
        scaled_x = int(x * self.scale_factor)
        scaled_y = int(y * self.scale_factor)
        scaled_w = int(w * self.scale_factor)
        scaled_h = int(h * self.scale_factor)
        
        roi_rect = QRect(scaled_x, scaled_y, scaled_w, scaled_h)
        if roi_rect.contains(adjusted_pos):
            return 'move'
            
        return None

    def map_to_image_coords(self, pos):
        """Convert widget coordinates to image coordinates"""
        if self.current_image is None:
            return None
            
        # Get the image label's position in the widget
        label_pos = self.image_label.mapFrom(self, pos)
        
        # Get the scroll area's scroll position
        scroll_x = self.scroll_area.horizontalScrollBar().value()
        scroll_y = self.scroll_area.verticalScrollBar().value()
        
        # Adjust for scroll position
        x = label_pos.x() + scroll_x
        y = label_pos.y() + scroll_y
        
        # Convert to image coordinates
        image_x = x / self.scale_factor
        image_y = y / self.scale_factor
        
        # Check if the point is within the image bounds
        if (0 <= image_x < self.current_image.shape[1] and 
            0 <= image_y < self.current_image.shape[0]):
            return (image_x, image_y)
        return None

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
            
            # Draw resize handles if Ctrl is pressed
            if self.is_ctrl_pressed:
                handles = ['top_left', 'top', 'top_right', 'right', 'bottom_right', 
                          'bottom', 'bottom_left', 'left']
                for handle in handles:
                    rect = self.get_handle_rect(x, y, w, h, handle)
                    if rect:
                        painter.setBrush(QColor(255, 255, 255))
                        painter.drawRect(rect)
            
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

    def scroll_area_wheel_event(self, event):
        """Handle wheel events in the scroll area"""
        if self.current_image is not None:
            # Get the amount of scroll
            delta = event.angleDelta().y()
            
            # Calculate zoom factor based on scroll direction
            factor = 1.25 if delta > 0 else 0.8
            
            # Get cursor position relative to the image
            pos = self.image_label.mapFrom(self, event.pos())
            
            # Zoom around cursor position
            self.zoom(factor, pos)
            
            # Prevent default scroll behavior
            event.accept()
        else:
            # If no image is loaded, allow normal scrolling
            QScrollArea.wheelEvent(self.scroll_area, event)

    def wheelEvent(self, event):
        """Handle wheel events in the main widget"""
        if self.current_image is not None:
            # Get the amount of scroll
            delta = event.angleDelta().y()
            
            # Calculate zoom factor based on scroll direction
            factor = 1.25 if delta > 0 else 0.8
            
            # Get cursor position relative to the image
            pos = self.image_label.mapFrom(self, event.pos())
            
            # Zoom around cursor position
            self.zoom(factor, pos)
            
            # Prevent default scroll behavior
            event.accept()
        else:
            # If no image is loaded, allow normal scrolling
            super().wheelEvent(event)

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
            
            # Calculate the position in the image before zoom
            old_x = pos.x() / old_factor
            old_y = pos.y() / old_factor
            
            # Calculate the new position after zoom
            new_x = old_x * self.scale_factor
            new_y = old_y * self.scale_factor
            
            # Adjust scroll position to keep the point under cursor fixed
            scroll_x.setValue(int(new_x - pos.x() + scroll_x.value()))
            scroll_y.setValue(int(new_y - pos.y() + scroll_y.value()))

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

    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Control:
            self.is_ctrl_pressed = True
            if self.roi is not None:
                self.setCursor(Qt.SizeAllCursor)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Handle key release events"""
        if event.key() == Qt.Key_Control:
            self.is_ctrl_pressed = False
            self.setCursor(Qt.ArrowCursor)
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press for panning and ROI resizing"""
        if event.button() == Qt.LeftButton:
            pos = self.image_label.mapFrom(self, event.pos())
            
            if self.is_ctrl_pressed:
                # Handle ROI editing when Ctrl is pressed
                handle = self.get_handle_at_pos(pos)
                if handle:
                    self.resize_handle = handle
                    self.resize_start = pos
                    self.resize_start_roi = self.roi
                    if handle == 'move':
                        self.setCursor(Qt.SizeAllCursor)
                    else:
                        self.setCursor(Qt.CrossCursor)
                else:
                    # Start new ROI creation if Ctrl is pressed and clicked outside ROI
                    if self.current_image is not None:
                        self.new_roi_start = pos
                        # Clear existing ROI
                        self.roi = None
                        self.display_image(self.current_image)
            else:
                # Only pan the image when Ctrl is not pressed
                self.pan_start = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
                self.last_pos = event.pos()

    def mouseReleaseEvent(self, event):
        """Handle mouse release for panning, ROI resizing, and new ROI creation"""
        if event.button() == Qt.LeftButton:
            if self.resize_handle:
                self.resize_handle = None
                self.resize_start = None
                self.resize_start_roi = None
                if self.is_ctrl_pressed:
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
            elif self.new_roi_start is not None:
                # Finish new ROI creation
                pos = self.image_label.mapFrom(self, event.pos())
                self.create_new_roi(self.new_roi_start, pos)
                self.new_roi_start = None
            else:
                self.pan_start = None
                self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, event):
        """Handle mouse move for panning and ROI resizing"""
        pos = self.image_label.mapFrom(self, event.pos())
        
        # Update cursor based on position and Ctrl state
        if not self.resize_handle and not self.new_roi_start:
            if self.is_ctrl_pressed:
                if self.roi is not None:
                    handle = self.get_handle_at_pos(pos)
                    if handle == 'move':
                        self.setCursor(Qt.SizeAllCursor)
                    elif handle in ['top_left', 'bottom_right']:
                        self.setCursor(Qt.SizeFDiagCursor)
                    elif handle in ['top_right', 'bottom_left']:
                        self.setCursor(Qt.SizeBDiagCursor)
                    elif handle in ['left', 'right']:
                        self.setCursor(Qt.SizeHorCursor)
                    elif handle in ['top', 'bottom']:
                        self.setCursor(Qt.SizeVerCursor)
                    else:
                        self.setCursor(Qt.ArrowCursor)
                else:
                    self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.ClosedHandCursor)
        
        if self.resize_handle and self.is_ctrl_pressed:
            if self.roi is None:
                return
                
            # Get scroll position
            scroll_x = self.scroll_area.horizontalScrollBar().value()
            scroll_y = self.scroll_area.verticalScrollBar().value()
            
            # Calculate movement in image coordinates
            dx = (pos.x() + scroll_x - self.resize_start.x() - scroll_x) / self.scale_factor
            dy = (pos.y() + scroll_y - self.resize_start.y() - scroll_y) / self.scale_factor
            
            x, y, w, h = self.resize_start_roi
            
            if self.resize_handle == 'move':
                new_x = max(0, min(x + dx, self.current_image.shape[1] - w))
                new_y = max(0, min(y + dy, self.current_image.shape[0] - h))
                self.roi = (new_x, new_y, w, h)
            else:
                new_x, new_y, new_w, new_h = x, y, w, h
                
                if 'left' in self.resize_handle:
                    new_x = max(0, min(x + dx, x + w - 10))
                    new_w = w - (new_x - x)
                if 'right' in self.resize_handle:
                    new_w = max(10, min(w + dx, self.current_image.shape[1] - x))
                if 'top' in self.resize_handle:
                    new_y = max(0, min(y + dy, y + h - 10))
                    new_h = h - (new_y - y)
                if 'bottom' in self.resize_handle:
                    new_h = max(10, min(h + dy, self.current_image.shape[0] - y))
                
                self.roi = (new_x, new_y, new_w, new_h)
            
            self.display_image(self.current_image)
            self.roi_changed.emit(self.roi)
        elif self.new_roi_start is not None and self.is_ctrl_pressed:
            # Draw new ROI while dragging
            self.draw_new_roi(self.new_roi_start, pos)
        elif self.pan_start is not None and not self.is_ctrl_pressed:
            # Calculate movement
            delta = event.pos() - self.last_pos
            self.last_pos = event.pos()
            
            # Update scroll bars
            scroll_x = self.scroll_area.horizontalScrollBar()
            scroll_y = self.scroll_area.verticalScrollBar()
            
            scroll_x.setValue(scroll_x.value() - delta.x())
            scroll_y.setValue(scroll_y.value() - delta.y())

    def create_new_roi(self, start_pos, end_pos):
        """Create a new ROI from start and end positions"""
        if self.current_image is None:
            return
        
        # Use map_to_image_coords for both positions
        start_coords = self.map_to_image_coords(start_pos)
        end_coords = self.map_to_image_coords(end_pos)
        if start_coords is None or end_coords is None:
            return
        start_x, start_y = start_coords
        end_x, end_y = end_coords
        
        # Calculate ROI coordinates as integers
        x = int(round(min(start_x, end_x)))
        y = int(round(min(start_y, end_y)))
        w = int(round(abs(end_x - start_x)))
        h = int(round(abs(end_y - start_y)))
        
        # Ensure ROI is within image bounds
        x = max(0, min(x, self.current_image.shape[1] - 1))
        y = max(0, min(y, self.current_image.shape[0] - 1))
        w = min(w, self.current_image.shape[1] - x)
        h = min(h, self.current_image.shape[0] - y)
        
        # Set minimum size
        if w >= 10 and h >= 10:
            self.roi = (x, y, w, h)
            self.display_image(self.current_image)
            self.roi_changed.emit((x, y, w, h))

    def draw_new_roi(self, start_pos, current_pos):
        """Draw the new ROI while dragging"""
        if self.current_image is None:
            return
        
        # Use map_to_image_coords for both positions
        start_coords = self.map_to_image_coords(start_pos)
        end_coords = self.map_to_image_coords(current_pos)
        if start_coords is None or end_coords is None:
            return
        start_x, start_y = [int(round(v)) for v in start_coords]
        end_x, end_y = [int(round(v)) for v in end_coords]
        
        # Create a copy of the current image
        display_image = self.display_image_data.copy()
        
        # Calculate ROI coordinates
        x = min(start_x, end_x)
        y = min(start_y, end_y)
        w = abs(end_x - start_x)
        h = abs(end_y - start_y)
        
        # Draw the temporary ROI
        if w >= 1 and h >= 1:
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Display the image with the temporary ROI
        self.display_image(display_image) 