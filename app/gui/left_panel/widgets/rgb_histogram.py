import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget


class RgbHistogram(QWidget):
    """Small widget to display an RGB histogram.

    Usage:
        hist = HistogramWidget(parent)
        hist.update_from_array(img_np)  # img_np uint8 HxWxC, C in {1,3}
    """

    def __init__(self, parent: QWidget | None = None, height: int = 120) -> None:
        super().__init__(parent)
        self._bins_r: np.ndarray | None = None
        self._bins_g: np.ndarray | None = None
        self._bins_b: np.ndarray | None = None
        self._max_count: float = 1.0

        self._bg: QColor = QColor(0, 0, 0, 160)
        self._fg_grid: QColor = QColor(255, 255, 255, 40)
        self._pen_r: QPen = QPen(QColor(255, 80, 80, 220), 1)
        self._pen_g: QPen = QPen(QColor(80, 255, 80, 220), 1)
        self._pen_b: QPen = QPen(QColor(80, 160, 255, 220), 1)
        self._border_pen: QPen = QPen(QColor(220, 220, 220, 120), 1)

        self.setMinimumWidth(100)
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

    def clear(self) -> None:
        self._bins_r = self._bins_g = self._bins_b = None
        self._max_count = 1.0
        self.update()

    def update_from_array(self, img_np: np.ndarray) -> None:
        """Accept an image array (uint8 HxWxC, C=1 or 3) and compute histograms."""
        if img_np is None or img_np.size == 0:
            self.clear()
            return

        if img_np.ndim != 3 or img_np.shape[2] not in (1, 3):
            self.clear()
            return

        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8, copy=False)

        if img_np.shape[2] == 1:
            ch = img_np[:, :, 0]
            bins, _ = np.histogram(ch, bins=256, range=(0, 255))
            self._bins_r = self._bins_g = self._bins_b = bins
        else:
            r: np.ndarray = img_np[:, :, 0]
            g: np.ndarray = img_np[:, :, 1]
            b: np.ndarray = img_np[:, :, 2]
            self._bins_r, _ = np.histogram(r, bins=256, range=(0, 255))
            self._bins_g, _ = np.histogram(g, bins=256, range=(0, 255))
            self._bins_b, _ = np.histogram(b, bins=256, range=(0, 255))

        self._max_count = float(
            max(
                1.0,
                np.max(self._bins_r) if self._bins_r is not None else 1.0,
                np.max(self._bins_g) if self._bins_g is not None else 1.0,
                np.max(self._bins_b) if self._bins_b is not None else 1.0,
            )
        )
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p: QPainter = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect: QRect = self.rect()

        p.fillRect(rect, self._bg)
        p.setPen(self._border_pen)
        p.drawRect(rect.adjusted(0, 0, -1, -1))

        p.setPen(self._fg_grid)
        for i in range(1, 4):
            y: int = rect.top() + int(rect.height() * i / 4)
            p.drawLine(QPoint(rect.left() + 4, y), QPoint(rect.right() - 4, y))

        if self._bins_r is None or self._bins_g is None or self._bins_b is None:
            p.end()
            return

        # plotting area with small margins
        left = rect.left() + 4
        right = rect.right() - 4
        top = rect.top() + 4
        bottom: int = rect.bottom() - 4
        width: int = max(1, right - left)
        height: int = max(1, bottom - top)

        def build_path(bins: np.ndarray) -> QPainterPath:
            path = QPainterPath()
            # normalize bins to [0, 1]
            norm = bins.astype(np.float32) / (self._max_count or 1.0)
            # build a polyline over 256 bins
            for i in range(256):
                x: int = left + int(i * (width - 1) / 255)
                h: int = int(norm[i] * (height - 1))
                y: int = bottom - h
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            return path

        # draw R, G, B
        p.setPen(self._pen_r)
        p.drawPath(build_path(self._bins_r))
        p.setPen(self._pen_g)
        p.drawPath(build_path(self._bins_g))
        p.setPen(self._pen_b)
        p.drawPath(build_path(self._bins_b))

        p.end()
