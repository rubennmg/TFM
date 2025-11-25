import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QVBoxLayout, QWidget

from gui.helpers.labelled_button_widget import LabelledButtonWidget
from gui.helpers.rgb_histogram_widget import RgbHistogramWidget


class LeftPanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout: QVBoxLayout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # rgb histogram
        self.histogram: RgbHistogramWidget = RgbHistogramWidget(self, height=180)
        layout.addWidget(self.histogram)

        # load / reset controls
        self.load_control: LabelledButtonWidget = LabelledButtonWidget(
            "Load image:", "Open file..."
        )
        self.load_control.clicked.connect(self._on_load_clicked)

        self.reset_control: LabelledButtonWidget = LabelledButtonWidget(
            "Reset:", "Reset image"
        )
        self.reset_control.clicked.connect(self._on_reset_clicked)

        layout.addWidget(self.load_control)
        layout.addWidget(self.reset_control)

        layout.addStretch()

        self.setLayout(layout)

    def update_histogram(self, img_np: np.ndarray | None) -> None:
        if img_np is None or img_np.size == 0:
            self.histogram.clear()
            return
        self.histogram.update_from_array(img_np)

    def update_histogram_bins(
        self, bins_r: np.ndarray, bins_g: np.ndarray, bins_b: np.ndarray
    ) -> None:
        self.histogram.update_from_bins(bins_r, bins_g, bins_b)

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

        self.controller.load_image(file_path)

    def _on_reset_clicked(self) -> None:
        self.controller.reset_image()
