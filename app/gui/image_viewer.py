import numpy as np
import torch

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt


class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout: QVBoxLayout = QVBoxLayout()
        self.label: QLabel = QLabel("No image loaded")
        self.label.setObjectName("imageLabel")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
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
            qimg: QImage = QImage(img_np.tobytes(), img_np.shape[1], img_np.shape[0], img_np.strides[0], QImage.Format.Format_Grayscale8)
        else:
            qimg: QImage = QImage(img_np.tobytes(), img_np.shape[1], img_np.shape[0], img_np.strides[0], QImage.Format.Format_RGB888)

        pixmap: QPixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap.scaled(
            self.label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))
