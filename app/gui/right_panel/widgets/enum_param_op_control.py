from typing import Iterable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget


class EnumParamOperationControl(QWidget):
    """Reusable control for operations with one enum-like parameter."""

    def __init__(
        self,
        controller,
        operation_index: int,
        title: str,
        tooltip: str,
        operation_name: str,
        param_name: str,
        options: Iterable[tuple[str, str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.controller = controller
        self.operation_index = operation_index
        self._operation_name = operation_name
        self._param_name = param_name

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setToolTip(tooltip)

        self._title = QLabel(title)
        self._title.setObjectName("operationControlTitle")
        self._selector = QComboBox()

        container = QWidget(self)
        container.setObjectName("operationControl")

        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(10, 18, 10, 8)
        container_layout.setSpacing(10)
        container_layout.addWidget(self._title)
        container_layout.addWidget(self._selector)
        container.setLayout(container_layout)

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(10)
        root_layout.addWidget(container)
        self.setLayout(root_layout)

        self.set_options(options)
        self._selector.activated.connect(self._on_selection_changed)

    def set_options(self, options: Iterable[tuple[str, str]]) -> None:
        self._selector.clear()
        for key, label in options:
            self._selector.addItem(label, userData=key)

    def current_value(self) -> str | None:
        return self._selector.currentData()

    def _on_selection_changed(self, _index: int) -> None:
        value = self.current_value()
        if value is None:
            return

        self.controller.apply_operation(
            self._operation_name,
            operation_idx=self.operation_index,
            **{self._param_name: value},
        )

    def reset(self) -> None:
        if self._selector.count() > 0:
            self._selector.setCurrentIndex(0)
