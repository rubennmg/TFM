from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class MinMaxControl(QWidget):
    def __init__(self, controller, operation_index: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.controller = controller
        self.operation_index = operation_index
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setToolTip("Apply Min-Max Normalization to the image")
        self.setObjectName("operationControl")

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 18, 10, 8)
        layout.setSpacing(10)

        self.title_lbl = QLabel("Min-Max Normalization")
        self.title_lbl.setObjectName("operationControlTitle")
        layout.addWidget(self.title_lbl)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._on_apply)

        layout.addWidget(self.apply_button)

        self.setLayout(layout)

    def _on_apply(self):
        self.controller.apply_operation(
            "MinMaxNormalization", operation_idx=self.operation_index
        )

    def reset(self) -> None:
        pass
