from typing import Iterable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget

_METHODS: Iterable[tuple[str, str]] = (
    ("gray_world", "Gray World"),
    ("max_rgb", "Max RGB"),
)


class WhiteBalanceOperationControl(QWidget):
    """Compound widget with label, selector and apply button for white balance.

    Args:
        QWidget (QWidget): Base Qt widget.
    """

    def __init__(
        self,
        controller,
        operation_index: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.controller = controller
        self.operation_index = operation_index
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setToolTip("Apply white balance adjustment to the image")
        self._title = QLabel("White Balance")
        self._title.setObjectName("operationControlTitle")
        self._selector = QComboBox()
        self._apply_button = QPushButton("Apply")
        self._apply_button.clicked.connect(self._on_apply_clicked)

        container = QWidget(self)
        container.setObjectName("operationControl")

        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(10, 18, 10, 8)
        container_layout.setSpacing(10)
        container_layout.addWidget(self._title)
        container_layout.addWidget(self._selector)
        container_layout.addWidget(self._apply_button)
        container.setLayout(container_layout)

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(10)
        root_layout.addWidget(container)

        self.setLayout(root_layout)

        self.set_methods(_METHODS)

    def set_methods(self, methods: Iterable[tuple[str, str]]) -> None:
        self._selector.clear()
        for method_key, method_name in methods:
            self._selector.addItem(method_name, method_key)

    def _on_apply_clicked(self) -> None:
        method_key = self._selector.currentData()
        self.controller.apply_operation(
            "WhiteBalance", operation_idx=self.operation_index, method=method_key
        )

    def reset(self) -> None:
        self._selector.setCurrentIndex(0)
