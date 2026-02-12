import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QColor


class HistogramWidget(QWidget):
    """Displays a grayscale histogram with draggable low/high LUT markers."""

    lut_changed = pyqtSignal(int, int)  # low, high (0-255)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.histogram = np.zeros(256, dtype=np.float32)
        self.low = 0
        self.high = 255
        self.dragging_low = False
        self.dragging_high = False
        self.setMinimumHeight(120)

    def set_image(self, image):
        """Update histogram from a BGR or grayscale image."""
        if image is None:
            self.histogram = np.zeros(256, dtype=np.float32)
            self.update()
            return

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255))
        self.histogram = hist.astype(np.float32)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        w = rect.width()
        h = rect.height()

        # Background
        painter.fillRect(rect, QColor(30, 30, 30))

        if self.histogram.sum() > 0:
            # Normalize histogram to widget height
            hist = self.histogram / self.histogram.max()
            pen_hist = QPen(QColor(200, 200, 200))
            painter.setPen(pen_hist)

            for i in range(256):
                x = int(i / 255.0 * (w - 1))
                y = int(hist[i] * (h - 10))
                painter.drawLine(x, h - 1, x, h - 1 - y)

        # Draw low/high markers
        pen_low = QPen(QColor(0, 200, 255))
        pen_low.setWidth(2)
        pen_high = QPen(QColor(255, 200, 0))
        pen_high.setWidth(2)

        def value_to_x(v):
            return int(v / 255.0 * (w - 1))

        x_low = value_to_x(self.low)
        x_high = value_to_x(self.high)

        painter.setPen(pen_low)
        painter.drawLine(x_low, 0, x_low, h)

        painter.setPen(pen_high)
        painter.drawLine(x_high, 0, x_high, h)

        painter.end()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        x = event.pos().x()
        w = self.width()

        def x_to_value(xpos):
            xpos = max(0, min(w - 1, xpos))
            return int(round(xpos / float(max(1, w - 1)) * 255))

        v = x_to_value(x)
        # Decide which marker to drag: choose nearest
        if abs(v - self.low) <= abs(v - self.high):
            self.dragging_low = True
        else:
            self.dragging_high = True

    def mouseMoveEvent(self, event):
        if not (self.dragging_low or self.dragging_high):
            return
        x = event.pos().x()
        w = self.width()

        def x_to_value(xpos):
            xpos = max(0, min(w - 1, xpos))
            return int(round(xpos / float(max(1, w - 1)) * 255))

        v = x_to_value(x)

        if self.dragging_low:
            self.low = max(0, min(v, self.high - 1))
        elif self.dragging_high:
            self.high = min(255, max(v, self.low + 1))

        self.update()
        self.lut_changed.emit(self.low, self.high)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self.dragging_low = False
        self.dragging_high = False

