from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QWidget

from .image_viewer import ImageViewer
from .left_panel import LeftPanel
from .right_panel import RightPanel


class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Image Viewer")
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
