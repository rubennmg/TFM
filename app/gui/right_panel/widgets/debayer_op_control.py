from typing import Iterable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget

_ALGORITHMS: Iterable[tuple[str, str]] = (
    ("debayer2x2", "Debayer 2x2"),
    ("debayer3x3", "Debayer 3x3"),
    ("debayer5x5", "Debayer 5x5"),
    ("debayersplit", "Debayer Split"),
)


class DebayerOperationControl(QWidget):
    """Compound widget with label, selector and apply button for debayering.

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
        self.setToolTip("Apply debayering to a RAW image")
        self._title = QLabel("Debayer Demosaicing")
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

        self.set_algorithms(_ALGORITHMS)

    def set_algorithms(self, algorithms: Iterable[tuple[str, str]]) -> None:
        self._selector.clear()
        for key, label in algorithms:
            self._selector.addItem(label, userData=key)

    def set_button_text(self, text: str) -> None:
        self._apply_button.setText(text)

    def current_algorithm(self) -> str | None:
        return self._selector.currentData()

    def _on_apply_clicked(self) -> None:
        key = self.current_algorithm()
        if key is None:
            return
        self.controller.apply_operation(
            "Debayer", operation_idx=self.operation_index, algorithm_name=key
        )

    def reset(self) -> None:
        if self._selector.count() > 0:
            self._selector.setCurrentIndex(0)
