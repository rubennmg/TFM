from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QWidget

from gui.image_viewer.image_viewer import ImageViewer
from gui.left_panel.left_panel import LeftPanel
from gui.right_panel.right_panel import RightPanel
from models.image import Image


class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("TFM - Image Processor")
        self.resize(1600, 800)

        central_widget: QWidget = QWidget()
        main_layout: QHBoxLayout = QHBoxLayout()

        self.left_panel: LeftPanel = LeftPanel(controller)
        self.viewer: ImageViewer = ImageViewer()
        self.right_panel: RightPanel = RightPanel(controller)

        main_layout.addWidget(self.left_panel, stretch=1)
        main_layout.addWidget(self.viewer, stretch=3)
        main_layout.addWidget(self.right_panel, stretch=1)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def update_image_view(self, image: Image) -> None:
        np_array = image.np_array
        self.viewer.update_image(np_array, image)
        self.left_panel.update_histogram(np_array)

    def reset_image_view(self) -> None:
        self.viewer.reset_zoom()
        self.right_panel.clear_pipeline_controls()

    def update_operation_pipeline(self, operations_profile: list[dict]) -> None:
        self.left_panel.update_pipeline_full(operations_profile)

    def clear_operation_pipeline(self) -> None:
        self.left_panel.clear_pipeline()

    def set_pipeline_enabled(self, enabled: bool) -> None:
        self.left_panel.set_pipeline_enabled(enabled)

    def append_operation_log(self, entry: str) -> None:
        self.viewer.append_log_entry(entry)

    def clear_operation_log(self) -> None:
        self.viewer.clear_log()
