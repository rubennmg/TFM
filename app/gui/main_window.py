from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QWidget

from .controls_panel import ControlsPanel
from .image_viewer import ImageViewer
from .load_panel import LoadPanel


class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Image Viewer")
        self.resize(1600, 800)

        central_widget = QWidget()
        main_layout = QHBoxLayout()

        self.load_panel = LoadPanel(controller)
        self.viewer = ImageViewer()
        self.controls_panel = ControlsPanel(controller)

        main_layout.addWidget(self.load_panel, stretch=1)
        main_layout.addWidget(self.viewer, stretch=6)
        main_layout.addWidget(self.controls_panel, stretch=1)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
