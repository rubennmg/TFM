from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class ColorControlWidget(QWidget):
    def __init__(
        self, title: str, controller, operation_name: str, parent: QWidget | None = None
    ):
        super().__init__(parent)

        self.title = title
        self.controller = controller
        self.operation_name = operation_name

        self.setObjectName("operationControl")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._title_label = QLabel(self.title)
        self._title_label.setObjectName("operationControlTitle")

        self._apply_button = QPushButton("Apply")
        self._apply_button.clicked.connect(self._on_apply_clicked)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 18, 10, 8)
        layout.setSpacing(10)
        layout.addWidget(self._title_label)
        layout.addWidget(self._apply_button)

        self.setLayout(layout)

    def _on_apply_clicked(self) -> None:
        self.controller.apply_operation(self.operation_name)

    def reset(
        self,
    ) -> None:
        pass
