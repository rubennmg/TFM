from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class NoParamOperationControl(QWidget):
    def __init__(
        self,
        controller,
        title: str,
        operation_name: str,
        operation_index: int,
        parent=None,
    ):
        super().__init__(parent)

        self.controller = controller
        self.title = title
        self.operation_name = operation_name
        self.operation_index = operation_index
        self._setup_ui()

    def _setup_ui(self):
        self.setObjectName("operationControl")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._title_label = QLabel(self.title)
        self._title_label.setObjectName("operationControlTitle")

        self._info_label = QLabel("This operation has no parameters.")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setWordWrap(True)
        self._info_label.setObjectName("operationControlInfo")

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 18, 10, 8)
        layout.setSpacing(10)

        layout.addWidget(self._title_label)
        layout.addWidget(self._info_label)

        self.setLayout(layout)

    def reset(self) -> None:
        pass
