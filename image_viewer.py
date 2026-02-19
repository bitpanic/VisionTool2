import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout, QPushButton, QDoubleSpinBox, QComboBox, QApplication, QFileDialog
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
        self.resize_start = None   # Starting position for resize (label coords)
        self.resize_start_roi = None  # ROI at start of resize
        self.resize_start_img = None  # Mouse position in image coords at start of resize/move
        self.handle_size = 8  # Size of resize handles in pixels
        self.new_roi_start = None  # Starting point for new ROI creation
        # Calibration, measurement, LUT, and view mode
        self.pixel_size = 1.0  # physical units per pixel
        self.pixel_unit = "px"
        self.measure_mode = False
        self.measure_start = None  # (x, y) in image coords
        self.measure_temp_end = None  # (x, y) in image coords for live preview
        self.measurements = []  # list of (x1, y1, x2, y2, length_px, length_phys, unit)
        self.lut_enabled = False
        self.lut_low = 0
        self.lut_high = 255
        self.view_mode = "rgb"  # rgb, gray, h, s, v
        self.hsv_s_scale = 1.0
        self.hsv_v_scale = 1.0

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

        # Export current image (without overlays)
        self.export_img_btn = QPushButton("Export Image")
        self.export_img_btn.clicked.connect(self.export_current_image)
        toolbar.addWidget(self.export_img_btn)

        # Measurement mode toggle
        self.measure_btn = QPushButton("Measure")
        self.measure_btn.setCheckable(True)
        self.measure_btn.toggled.connect(self.set_measure_mode)
        toolbar.addWidget(self.measure_btn)

        # Clear measurements
        self.clear_measure_btn = QPushButton("Clear Measurements")
        self.clear_measure_btn.clicked.connect(self.clear_measurements)
        toolbar.addWidget(self.clear_measure_btn)

        # Calibration controls
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.0001, 1e9)
        self.pixel_size_spin.setDecimals(4)
        self.pixel_size_spin.setValue(1.0)
        self.pixel_size_spin.valueChanged.connect(self.on_calibration_changed)
        toolbar.addWidget(QLabel("Units / pixel:"))
        toolbar.addWidget(self.pixel_size_spin)

        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["px", "µm", "mm"])
        self.unit_combo.setCurrentText("px")
        self.unit_combo.currentTextChanged.connect(self.on_calibration_changed)
        toolbar.addWidget(self.unit_combo)
        
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
        """Get the resize handle at the given position (label coordinates)."""
        if self.roi is None:
            return None

        x, y, w, h = self.roi
        handles = ['top_left', 'top', 'top_right', 'right', 'bottom_right',
                   'bottom', 'bottom_left', 'left']

        # Adjust mouse position for margins (centering)
        margin_x, margin_y = self._get_image_margins()
        adjusted_pos = QPoint(pos.x() - margin_x, pos.y() - margin_y)

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

    def _get_scaled_image_size(self):
        """Return the current scaled image width and height."""
        if self.current_image is None:
            return 0, 0
        img_h, img_w = self.current_image.shape[:2]
        return img_w * self.scale_factor, img_h * self.scale_factor

    def _get_image_margins(self):
        """Return horizontal and vertical margins due to centering inside the label."""
        scaled_w, scaled_h = self._get_scaled_image_size()
        margin_x = max(0, (self.image_label.width() - scaled_w) / 2)
        margin_y = max(0, (self.image_label.height() - scaled_h) / 2)
        return margin_x, margin_y

    def map_to_image_coords(self, pos):
        """Convert label coordinates to image coordinates.

        The incoming `pos` is expected to already be in `image_label` coordinates.
        Because the pixmap is centered inside the label, we must remove
        the horizontal/vertical margins before undoing the scaling.
        """
        if self.current_image is None:
            return None

        # Margins added by centering inside the label
        margin_x, margin_y = self._get_image_margins()

        # Position relative to the top-left corner of the pixmap
        rel_x = pos.x() - margin_x
        rel_y = pos.y() - margin_y

        # Convert to image coordinates by undoing scale
        image_x = rel_x / self.scale_factor
        image_y = rel_y / self.scale_factor

        # Check if the point is within the image bounds
        if (0 <= image_x < self.current_image.shape[1] and
            0 <= image_y < self.current_image.shape[0]):
            return (image_x, image_y)
        return None

    def apply_lut(self, image):
        """Apply window LUT [lut_low, lut_high] to image (on brightness).

        - Grayscale: window is applied directly to intensities.
        - Color: window is applied to the V channel in HSV, then converted back to BGR.
        """
        if not self.lut_enabled or image is None:
            return image

        low = max(0, min(255, int(self.lut_low)))
        high = max(0, min(255, int(self.lut_high)))
        if high <= low:
            return image

        if len(image.shape) == 2 or image.shape[2] == 1:
            # Grayscale image
            img = image.astype(np.float32)
            img = (img - low) * (255.0 / float(high - low))
            img = np.clip(img, 0, 255).astype(np.uint8)
            return img

        # Color image: work in HSV space on V channel
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)

        v = (v - low) * (255.0 / float(high - low))
        v = np.clip(v, 0, 255)
        hsv_merged = cv2.merge([h, s, v]).astype(np.uint8)
        result = cv2.cvtColor(hsv_merged, cv2.COLOR_HSV2BGR)
        return result

    def set_lut(self, low, high):
        """Set LUT window and refresh display."""
        self.lut_low = int(low)
        self.lut_high = int(high)
        self.lut_enabled = True
        if self.current_image is not None:
            self.display_image(self.current_image)

    def set_view_mode(self, mode):
        """Set how the image is visualized: rgb / gray / h / s / v."""
        self.view_mode = mode
        if self.current_image is not None:
            self.display_image(self.current_image)

    def set_hsv_scales(self, s_scale, v_scale):
        """Set scaling factors for HSV saturation and value."""
        self.hsv_s_scale = max(0.0, float(s_scale))
        self.hsv_v_scale = max(0.0, float(v_scale))
        if self.current_image is not None:
            self.display_image(self.current_image)

    def export_current_image(self):
        """Export the current processed image (with LUT, without overlays)."""
        if self.current_image is None:
            return

        # Apply LUT and convert to RGB for saving
        img = self.apply_lut(self.current_image)
        if img is None:
            return

        if len(img.shape) == 3:
            save_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = save_img.shape[:2]
            bytes_per_line = 3 * width
            qimg = QImage(save_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            # Grayscale
            height, width = img.shape
            bytes_per_line = width
            qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)"
        )
        if not file_name:
            return

        pixmap = QPixmap.fromImage(qimg)
        pixmap.save(file_name)

    def display_image(self, image):
        """Display the given image"""
        if image is None:
            return

        # Apply LUT before view-mode/color operations
        image = self.apply_lut(image)

        # Apply view mode and HSV scaling
        if image is not None and len(image.shape) == 3:
            bgr = image
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)

            # Apply global S/V scaling
            s *= self.hsv_s_scale
            v *= self.hsv_v_scale
            s = np.clip(s, 0, 255)
            v = np.clip(v, 0, 255)

            if self.view_mode == "gray":
                # Show value channel as grayscale
                gray = v.astype(np.uint8)
                image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            elif self.view_mode in ("h", "s", "v"):
                if self.view_mode == "h":
                    ch = h * (255.0 / 179.0)  # Hue 0-179 -> 0-255
                elif self.view_mode == "s":
                    ch = s
                else:  # "v"
                    ch = v
                ch = np.clip(ch, 0, 255).astype(np.uint8)
                image = cv2.cvtColor(ch, cv2.COLOR_GRAY2RGB)
            else:
                # Standard RGB view with adjusted S/V
                hsv_merged = cv2.merge([h, s, v]).astype(np.uint8)
                bgr_adj = cv2.cvtColor(hsv_merged, cv2.COLOR_HSV2BGR)
                image = cv2.cvtColor(bgr_adj, cv2.COLOR_BGR2RGB)
        else:
            # Grayscale image: convert once to RGB for display
            if image is not None and len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
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

        # Draw ROI, measurements and overlays
        needs_painter = (
            self.roi is not None
            or bool(self.measurements)
            or (self.measure_mode and self.measure_start is not None and self.measure_temp_end is not None)
        )

        if needs_painter:
            painter = QPainter(scaled_pixmap)

            # Draw ROI if exists
            if self.roi is not None:
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

                # Draw resize handles when Ctrl is held
                if QApplication.keyboardModifiers() & Qt.ControlModifier:
                    handles = ['top_left', 'top', 'top_right', 'right', 'bottom_right',
                               'bottom', 'bottom_left', 'left']
                    for handle in handles:
                        rect = self.get_handle_rect(x, y, w, h, handle)
                        if rect:
                            painter.setBrush(QColor(255, 255, 255))
                            painter.drawRect(rect)

            # Draw existing measurements
            if self.measurements:
                pen_line = QPen(QColor(0, 255, 0))
                pen_line.setWidth(2)
                pen_text = QPen(QColor(255, 255, 0))
                for x1, y1, x2, y2, length_px, length_phys, unit in self.measurements:
                    sx1 = int(x1 * self.scale_factor)
                    sy1 = int(y1 * self.scale_factor)
                    sx2 = int(x2 * self.scale_factor)
                    sy2 = int(y2 * self.scale_factor)
                    painter.setPen(pen_line)
                    painter.drawLine(sx1, sy1, sx2, sy2)
                    # Label near the middle of the line
                    mx = (sx1 + sx2) // 2
                    my = (sy1 + sy2) // 2
                    label = f"{length_phys:.2f} {unit}"
                    painter.setPen(pen_text)
                    painter.drawText(mx + 5, my - 5, label)

            # Draw live measurement preview
            if self.measure_mode and self.measure_start is not None and self.measure_temp_end is not None:
                x1, y1 = self.measure_start
                x2, y2 = self.measure_temp_end
                sx1 = int(x1 * self.scale_factor)
                sy1 = int(y1 * self.scale_factor)
                sx2 = int(x2 * self.scale_factor)
                sy2 = int(y2 * self.scale_factor)
                pen_preview = QPen(QColor(0, 200, 255))
                pen_preview.setWidth(2)
                painter.setPen(pen_preview)
                painter.drawLine(sx1, sy1, sx2, sy2)

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

    def mousePressEvent(self, event):
        """Handle mouse press for panning, ROI resizing, and measurements"""
        if event.button() == Qt.LeftButton:
            pos = self.image_label.mapFrom(self, event.pos())
            ctrl_pressed = bool(event.modifiers() & Qt.ControlModifier)

            # Measurement mode has priority over ROI/pan
            if self.measure_mode and self.current_image is not None:
                coords = self.map_to_image_coords(pos)
                if coords is not None:
                    self.measure_start = coords
                    self.measure_temp_end = None
                return

            if ctrl_pressed:
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
        """Handle mouse release for panning, ROI resizing, new ROI creation, and measurements"""
        if event.button() == Qt.LeftButton:
            ctrl_pressed = bool(event.modifiers() & Qt.ControlModifier)
            # Finish measurement if in measurement mode
            if self.measure_mode and self.measure_start is not None and self.current_image is not None:
                pos = self.image_label.mapFrom(self, event.pos())
                end_coords = self.map_to_image_coords(pos)
                if end_coords is not None:
                    x1, y1 = self.measure_start
                    x2, y2 = end_coords
                    dx = x2 - x1
                    dy = y2 - y1
                    length_px = float(np.hypot(dx, dy))
                    if length_px > 0.5:
                        # Convert to physical units if configured
                        if self.pixel_unit == "px":
                            length_phys = length_px
                            unit = "px"
                        else:
                            length_phys = length_px * self.pixel_size
                            unit = self.pixel_unit
                        self.measurements.append((x1, y1, x2, y2, length_px, length_phys, unit))
                        self.display_image(self.current_image)
                self.measure_start = None
                self.measure_temp_end = None
                return

            if self.resize_handle:
                # Finish ROI move/resize and emit a single change event
                self.resize_handle = None
                self.resize_start = None
                self.resize_start_roi = None
                self.resize_start_img = None
                self.roi_changed.emit(self.roi)
                if ctrl_pressed:
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
            elif self.new_roi_start is not None:
                # Finish new ROI creation and emit change event once
                pos = self.image_label.mapFrom(self, event.pos())
                self.create_new_roi(self.new_roi_start, pos)
                self.new_roi_start = None
            else:
                self.pan_start = None
                self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, event):
        """Handle mouse move for panning, ROI resizing, and measurements"""
        pos = self.image_label.mapFrom(self, event.pos())
        ctrl_pressed = bool(event.modifiers() & Qt.ControlModifier)

        # Live update for measurement mode
        if self.measure_mode and self.measure_start is not None and self.current_image is not None:
            coords = self.map_to_image_coords(pos)
            if coords is not None:
                self.measure_temp_end = coords
                self.display_image(self.current_image)
            return

        # Update cursor based on position and Ctrl state
        if not self.resize_handle and not self.new_roi_start:
            if ctrl_pressed:
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

        if self.resize_handle and ctrl_pressed:
            if self.roi is None:
                return

            # Calculate movement in image coordinates using start and current mouse positions
            current_img_pos = self.map_to_image_coords(pos)
            if current_img_pos is None:
                return

            if self.resize_start_img is None:
                self.resize_start_img = self.map_to_image_coords(self.resize_start)
                if self.resize_start_img is None:
                    return

            start_img_x, start_img_y = self.resize_start_img
            curr_img_x, curr_img_y = current_img_pos
            dx = curr_img_x - start_img_x
            dy = curr_img_y - start_img_y

            x, y, w, h = self.resize_start_roi

            if self.resize_handle == 'move':
                new_x = max(0, min(x + dx, self.current_image.shape[1] - w))
                new_y = max(0, min(y + dy, self.current_image.shape[0] - h))
                self.roi = (int(round(new_x)), int(round(new_y)), int(round(w)), int(round(h)))
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

                self.roi = (
                    int(round(new_x)),
                    int(round(new_y)),
                    int(round(new_w)),
                    int(round(new_h)),
                )

            self.display_image(self.current_image)
        elif self.new_roi_start is not None and ctrl_pressed:
            # Draw new ROI while dragging
            self.draw_new_roi(self.new_roi_start, pos)
        elif self.pan_start is not None and not ctrl_pressed:
            # Calculate movement
            delta = event.pos() - self.last_pos
            self.last_pos = event.pos()
            
            # Update scroll bars
            scroll_x = self.scroll_area.horizontalScrollBar()
            scroll_y = self.scroll_area.verticalScrollBar()
            
            scroll_x.setValue(scroll_x.value() - delta.x())
            scroll_y.setValue(scroll_y.value() - delta.y())

    def set_measure_mode(self, enabled):
        """Enable or disable measurement mode."""
        self.measure_mode = enabled
        self.measure_start = None
        self.measure_temp_end = None
        cursor = Qt.CrossCursor if enabled else Qt.ArrowCursor
        self.setCursor(cursor)
        self.image_label.setCursor(cursor)

    def clear_measurements(self):
        """Remove all stored measurements."""
        self.measurements.clear()
        if self.current_image is not None:
            self.display_image(self.current_image)

    def on_calibration_changed(self, *args):
        """Update calibration (physical size per pixel and unit)."""
        self.pixel_size = float(self.pixel_size_spin.value())
        self.pixel_unit = self.unit_combo.currentText()
        if self.current_image is not None:
            self.display_image(self.current_image)

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