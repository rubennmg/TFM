from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtGui import QPainter, QPalette, QPixmap
from PyQt6.QtWidgets import QWidget


class ImageCanvas(QWidget):
    """Widget responsible for drawing the current pixmap Image.

    Args:
        QWidget (QWidget): Widget base class.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._placeholder: str = "No image loaded"
        self._default_size: QSize = QSize(640, 480)
        self.setMinimumSize(0, 0)
        self.resize(self._default_size)

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        self._pixmap = pixmap
        target_size = pixmap.size() if pixmap is not None else self._default_size
        self.resize(target_size)
        self.updateGeometry()
        self.update()

    def clear(self) -> None:
        self.set_pixmap(None)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        if self._pixmap is not None:
            return self._pixmap.size()
        return self._default_size

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(0, 0)

    def paintEvent(self, a0) -> None:  # noqa: N802
        assert a0 is not None
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.fillRect(a0.rect(), self.palette().color(QPalette.ColorRole.Base))

        if self._pixmap is None:
            painter.setPen(self.palette().color(QPalette.ColorRole.Mid))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder
            )
            return

        pix_rect: QRect = QRect(QPoint(0, 0), self._pixmap.size())
        pix_rect.moveCenter(self.rect().center())
        painter.drawPixmap(pix_rect.topLeft(), self._pixmap)
