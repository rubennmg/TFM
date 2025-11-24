from typing import Iterable, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget

_ALGORITHMS: Iterable[Tuple[str, str]] = (
    ("debayer2x2", "Debayer 2x2"),
    ("debayer3x3", "Debayer 3x3"),
    ("debayer5x5", "Debayer 5x5"),
    ("debayersplit", "Debayer Split"),
)


class DebayerControlWidget(QWidget):
    """Compound widget with label, selector and apply button for debayering.

    Args:
        QWidget (QWidget): Base Qt widget.
    """

    applyClicked = pyqtSignal(str)

    def __init__(
        self,
        title: str = "Debayer Demosaicing",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._title = QLabel(title)
        self._title.setObjectName("debayerControlTitle")
        self._selector = QComboBox()
        self._apply_button = QPushButton("Apply")
        self._apply_button.clicked.connect(self._emit_current_algorithm)

        container = QWidget(self)
        container.setObjectName("debayerControl")

        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(10, 8, 10, 8)
        container_layout.setSpacing(6)
        container_layout.addWidget(self._title)
        container_layout.addWidget(self._selector)
        container.setLayout(container_layout)

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(8)
        root_layout.addWidget(container)
        root_layout.addWidget(self._apply_button)
        self.setLayout(root_layout)

        self.set_algorithms(_ALGORITHMS)

    def set_algorithms(self, algorithms: Iterable[Tuple[str, str]]) -> None:
        self._selector.clear()
        for key, label in algorithms:
            self._selector.addItem(label, userData=key)

    def set_button_text(self, text: str) -> None:
        self._apply_button.setText(text)

    def current_algorithm(self) -> str | None:
        return self._selector.currentData()

    def _emit_current_algorithm(self) -> None:
        key = self.current_algorithm()
        if key is None:
            return
        self.applyClicked.emit(key)
