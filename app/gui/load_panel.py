from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
from PyQt6.QtCore import Qt
from torch import Tensor


class LoadPanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout: QVBoxLayout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.label: QLabel = QLabel("Load image:")
        self.button_load: QPushButton = QPushButton("Open file...")
        self.button_load.clicked.connect(self._on_load_clicked)

        layout.addWidget(self.label)
        layout.addWidget(self.button_load)
        layout.addStretch()

        self.setLayout(layout)

    def _on_load_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select RAW or RGB image",
            "",
            "Images (*.raw *.png *.jpg *.jpeg *.bmp *.tiff);;All files (*)"
        )
        if not file_path:
            return

        tensor: Tensor = self.controller.load_image(file_path)
        if tensor is not None:
            self.controller.update_viewer(tensor)
