import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QVBoxLayout, QWidget

from gui.left_panel.widgets.labelled_button import LabelledButton
from gui.left_panel.widgets.operation_pipeline import OperationPipeline
from gui.left_panel.widgets.rgb_histogram import RgbHistogram


class LeftPanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout: QVBoxLayout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # rgb histogram
        self.histogram: RgbHistogram = RgbHistogram(self, height=180)
        layout.addWidget(self.histogram)

        # operation pipeline
        self.pipeline: OperationPipeline = OperationPipeline(self)
        self.pipeline.add_operation_requested.connect(
            self.controller.add_pipeline_operation
        )
        self.pipeline.remove_operation_requested.connect(
            self.controller.remove_last_operation
        )
        self.pipeline.device_changed.connect(self.controller.change_device)

        # load / reset controls
        self.load_control: LabelledButton = LabelledButton(
            "Load image:", "Open file..."
        )
        self.load_control.clicked.connect(self._on_load_clicked)

        self.reset_control: LabelledButton = LabelledButton("Reset:", "Reset image")
        self.reset_control.clicked.connect(self._on_reset_clicked)

        # operations profile
        self.operations_profile_control: LabelledButton = LabelledButton(
            "Operations profile:", "Generate profile"
        )
        self.operations_profile_control.clicked.connect(
            self._on_generate_operations_profile_clicked
        )

        # save image
        self.save_control: LabelledButton = LabelledButton(
            "Save image:", "Save file..."
        )
        self.save_control.clicked.connect(self._on_save_clicked)

        layout.addWidget(self.load_control)
        layout.addWidget(self.reset_control)
        layout.addWidget(self.pipeline, stretch=1)

        layout.addStretch()

        layout.addWidget(self.operations_profile_control)
        layout.addWidget(self.save_control)

        self.setLayout(layout)

    def update_histogram(self, img_np: np.ndarray | None) -> None:
        if img_np is None or img_np.size == 0:
            self.histogram.clear()
            return
        self.histogram.update_from_array(img_np)

    def build_loader_extensions_filter(self) -> str:
        extensions: list[str] = self.controller.get_image_extensions()
        extensions_filter: str = "Images ("
        for ext in extensions:
            extensions_filter += f"*.{ext} *.{ext.upper()} "
        extensions_filter = extensions_filter.strip() + ")"
        extensions_filter += ";;All Files (*)"
        return extensions_filter

    def clear_pipeline(self) -> None:
        self.pipeline.clear()

    def update_pipeline_full(self, operations_profile: list[dict]) -> None:
        self.pipeline.set_operations(operations_profile)

    def set_pipeline_enabled(self, enabled: bool) -> None:
        self.pipeline.set_interactions_enabled(enabled)

    def _on_load_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select RAW or RGB image",
            "",
            self.build_loader_extensions_filter(),
        )

        if not file_path:
            return

        self.controller.load_image(file_path)

    def _on_reset_clicked(self) -> None:
        self.controller.reset_image()

    def _on_generate_operations_profile_clicked(self) -> None:
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Operations Profile",
            "operations_profile.json",
            "JSON Files (*.json);;All Files (*)",
        )

        if not save_path:
            return

        self.controller.export_profile(save_path)

    def _on_save_clicked(self) -> None:
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "image.jpg",
            "JPG Files (*.jpg);;JPEG Files (*.jpeg);;All Files (*)",
        )

        if not save_path:
            return

        self.controller.save_image(save_path)
