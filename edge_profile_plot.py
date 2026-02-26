from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QColor


class EdgeProfilePlot(QWidget):
    """Simple widget to plot intensity profile and its derivative."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._s = []
        self._intensity = []
        self._derivative = []
        self._peak_pos_index = None
        self._peak_neg_index = None
        self._width_indices = None

    def set_data(self, s, intensity, derivative, peak_pos_index=None, peak_neg_index=None, width_indices=None):
        self._s = list(s or [])
        self._intensity = list(intensity or [])
        self._derivative = list(derivative or [])
        self._peak_pos_index = peak_pos_index
        self._peak_neg_index = peak_neg_index
        self._width_indices = list(width_indices or []) if width_indices is not None else None
        self.update()

    def clear(self):
        self._s = []
        self._intensity = []
        self._derivative = []
        self._peak_pos_index = None
        self._peak_neg_index = None
        self._width_indices = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QColor(20, 20, 20))

        if not self._s or not self._intensity or not self._derivative:
            painter.end()
            return

        w = rect.width()
        h = rect.height()

        # Margins
        margin_left = 4
        margin_right = 4
        margin_top = 4
        margin_bottom = 4

        plot_w = max(1, w - margin_left - margin_right)
        plot_h = max(1, h - margin_top - margin_bottom)

        # X range from s
        s_min = min(self._s)
        s_max = max(self._s)
        if s_max == s_min:
            s_max = s_min + 1.0

        def sx(i):
            s_val = self._s[i]
            return margin_left + int((s_val - s_min) / (s_max - s_min) * (plot_w - 1))

        # Y ranges
        I_min = min(self._intensity)
        I_max = max(self._intensity)
        if I_max == I_min:
            I_max = I_min + 1.0

        d_min = min(self._derivative)
        d_max = max(self._derivative)
        if d_max == d_min:
            d_max = d_min + 1.0

        def sy_int(i):
            v = self._intensity[i]
            norm = (v - I_min) / (I_max - I_min)
            return margin_top + int((1.0 - norm) * (plot_h - 1))

        def sy_der(i):
            v = self._derivative[i]
            norm = (v - d_min) / (d_max - d_min)
            return margin_top + int((1.0 - norm) * (plot_h - 1))

        # Draw intensity curve
        pen_I = QPen(QColor(0, 200, 255))
        pen_I.setWidth(2)
        painter.setPen(pen_I)
        for i in range(1, len(self._s)):
            painter.drawLine(sx(i - 1), sy_int(i - 1), sx(i), sy_int(i))

        # Draw derivative curve
        pen_D = QPen(QColor(255, 180, 0))
        pen_D.setWidth(1)
        painter.setPen(pen_D)
        for i in range(1, len(self._s)):
            painter.drawLine(sx(i - 1), sy_der(i - 1), sx(i), sy_der(i))

        # Peaks
        peak_pen = QPen(QColor(0, 255, 0))
        peak_pen.setWidth(4)
        painter.setPen(peak_pen)
        if self._peak_pos_index is not None and 0 <= self._peak_pos_index < len(self._s):
            i = self._peak_pos_index
            painter.drawPoint(sx(i), sy_der(i))
        if self._peak_neg_index is not None and 0 <= self._peak_neg_index < len(self._s):
            i = self._peak_neg_index
            painter.drawPoint(sx(i), sy_der(i))

        # Edge width markers (10–90%)
        if self._width_indices:
            width_pen = QPen(QColor(200, 0, 200))
            width_pen.setWidth(3)
            painter.setPen(width_pen)
            for i in self._width_indices:
                if 0 <= i < len(self._s):
                    painter.drawPoint(sx(i), sy_int(i))

        painter.end()

