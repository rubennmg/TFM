from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QVBoxLayout, QWidget
from torch import Tensor

from .helpers.labelled_button import LabelledButton


class LoadPanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout: QVBoxLayout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.load_control: LabelledButton = LabelledButton(
            "Load image:", "Open file..."
        )
        self.load_control.clicked.connect(self._on_load_clicked)

        self.reset_control: LabelledButton = LabelledButton("Reset:", "Reset image")
        self.reset_control.clicked.connect(self._on_reset_clicked)

        layout.addWidget(self.load_control)
        layout.addWidget(self.reset_control)

        layout.addStretch()

        self.setLayout(layout)

    def _on_load_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select RAW or RGB image",
            "",
            "Images (*.raw *.RAW *.png *.PNG *.jpg *.JPG *.jpeg *.JPEG *.bmp *.BMP *.tiff *.TIFF "
            "*.nef *.NEF *.cr2 *.CR2 *.arw *.ARW *.dng *.DNG *.rw2 *.RW2 *.orf *.ORF);;"
            "All files (*)",
        )
        if not file_path:
            return

        tensor: Tensor = self.controller.load_image(file_path)
        if tensor is not None:
            self.controller.update_viewer(tensor)

    def _on_reset_clicked(self) -> None:
        tensor: Tensor = self.controller.reset_image()
        if tensor is not None:
            self.controller.update_viewer(tensor)
