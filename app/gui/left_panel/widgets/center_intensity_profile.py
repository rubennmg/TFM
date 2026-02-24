import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget


class CenterIntensityProfile(QWidget):
    def __init__(self, parent: QWidget | None = None, height: int = 120) -> None:
        super().__init__(parent)
        self._profile_h: np.ndarray | None = None
        self._profile_v: np.ndarray | None = None
        self._max_value: float = 1.0

        self._bg: QColor = QColor(0, 0, 0, 160)
        self._fg_grid: QColor = QColor(255, 255, 255, 40)
        self._pen_h: QPen = QPen(QColor(255, 210, 80, 220), 1)
        self._pen_v: QPen = QPen(QColor(80, 220, 255, 220), 1)
        self._border_pen: QPen = QPen(QColor(220, 220, 220, 120), 1)

        self.setMinimumWidth(100)
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

    def clear(self) -> None:
        self._profile_h = None
        self._profile_v = None
        self._max_value = 1.0
        self.update()

    def update_from_array(self, img_np: np.ndarray) -> None:
        if img_np is None or img_np.size == 0:
            self.clear()
            return

        if img_np.ndim != 3 or img_np.shape[2] not in (1, 3):
            self.clear()
            return

        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8, copy=False)

        if img_np.shape[2] == 1:
            luminance = img_np[:, :, 0].astype(np.float32, copy=False)
        else:
            rgb = img_np.astype(np.float32, copy=False)
            luminance = (
                0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
            )

        height, width = luminance.shape
        center_y = height // 2
        center_x = width // 2

        self._profile_h = luminance[center_y, :].astype(np.float32, copy=False)
        self._profile_v = luminance[:, center_x].astype(np.float32, copy=False)

        max_h = float(np.max(self._profile_h)) if self._profile_h.size > 0 else 1.0
        max_v = float(np.max(self._profile_v)) if self._profile_v.size > 0 else 1.0
        self._max_value = max(1.0, max_h, max_v)
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

        if self._profile_h is None or self._profile_v is None:
            p.end()
            return

        left = rect.left() + 4
        right = rect.right() - 4
        top = rect.top() + 4
        bottom: int = rect.bottom() - 4
        width: int = max(1, right - left)
        height: int = max(1, bottom - top)

        def build_path(profile: np.ndarray) -> QPainterPath:
            path = QPainterPath()
            if profile.size == 0:
                return path

            norm = profile / (self._max_value or 1.0)
            last_index = max(1, profile.size - 1)
            for i in range(profile.size):
                x: int = left + int(i * (width - 1) / last_index)
                h: int = int(norm[i] * (height - 1))
                y: int = bottom - h
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            return path

        p.setPen(self._pen_h)
        p.drawPath(build_path(self._profile_h))
        p.setPen(self._pen_v)
        p.drawPath(build_path(self._profile_v))

        p.end()
