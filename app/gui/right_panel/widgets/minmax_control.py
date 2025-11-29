from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QPushButton, QWidget, QLabel


class MinMaxControlWidget(QWidget):
    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__(parent)
        self.controller = controller
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setToolTip("Apply Min-Max Normalization to the image")
        self.setObjectName("operationControl")

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 18, 10, 8)
        layout.setSpacing(6)

        self.title_lbl = QLabel("Min-Max Normalization")
        self.title_lbl.setObjectName("operationControlTitle")
        layout.addWidget(self.title_lbl)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._on_apply)

        layout.addWidget(self.apply_button)

        self.setLayout(layout)

    def _on_apply(self):
        self.controller.apply_operation("MinMaxNormalization")
