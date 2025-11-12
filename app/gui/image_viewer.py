import numpy as np
import torch
from PyQt6.QtCore import QEvent, QSize, Qt
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget


class ImageViewer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self._zoom: float = 0.25
        self._zoom_step: float = 1.15
        self._min_zoom: float = 0.05
        self._max_zoom: float = 10.0

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout: QVBoxLayout = QVBoxLayout()
        self.label: QLabel = QLabel("No image loaded")
        self.label.setObjectName("imageLabel")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.label)
        self.scroll_area.setWidgetResizable(False)

        try:
            self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        except Exception:
            pass

        viewport = self.scroll_area.viewport()
        if viewport is not None:
            viewport.installEventFilter(self)
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)

    def show_tensor(self, tensor: torch.Tensor) -> None:
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        elif tensor.ndim != 3:
            raise ValueError("Tensor shape must be (C,H,W) or (H,W,C)")

        tensor = (tensor * 255).round().type(torch.uint8)
        img_np: np.ndarray = tensor.cpu().numpy()

        if img_np.shape[2] == 1:
            qimg = QImage(
                img_np.tobytes(),
                img_np.shape[1],
                img_np.shape[0],
                img_np.strides[0],
                QImage.Format.Format_Grayscale8,
            )
        else:
            qimg = QImage(
                img_np.tobytes(),
                img_np.shape[1],
                img_np.shape[0],
                img_np.strides[0],
                QImage.Format.Format_RGB888,
            )

        self._pixmap = QPixmap.fromImage(qimg)
        self._update_display()

    def _update_display(self) -> None:
        if self._pixmap is None:
            self.label.setText("No image loaded")
            return

        self.label.setText("")
        pix_to_show = self._pixmap

        if self._zoom != 1.0:
            w = max(1, int(pix_to_show.width() * self._zoom))
            h = max(1, int(pix_to_show.height() * self._zoom))
            pix_to_show = pix_to_show.scaled(
                QSize(w, h),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        self.label.setPixmap(pix_to_show)
        self.label.setFixedSize(self.label.pixmap().size())

    def set_zoom(self, factor: float) -> None:
        factor = max(self._min_zoom, min(self._max_zoom, factor))
        if abs(self._zoom - factor) < 1e-6:
            return
        self._zoom = factor
        self._update_display()

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * self._zoom_step)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / self._zoom_step)

    def reset_zoom(self) -> None:
        self.set_zoom(0.25)

    def eventFilter(self, a0, a1) -> bool:
        if a1 is None:
            return False

        if a1.type() == QEvent.Type.Wheel:
            if not isinstance(a1, QWheelEvent):
                return False
            modifiers = a1.modifiers()
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                delta = a1.angleDelta().y()
                if delta > 0:
                    self.zoom_in()
                else:
                    self.zoom_out()
                return True
        return super().eventFilter(a0, a1)
