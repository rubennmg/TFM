from typing import Iterable, Tuple

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget


class DebayerControlWidget(QWidget):
    """Compound widget with label, selector and apply button for debayering.

    Args:
        QWidget (QWidget): Base Qt widget.
    """

    applyClicked = pyqtSignal(str)

    def __init__(
        self,
        title: str = "Debayer",
        algorithms: Iterable[Tuple[str, str]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._title = QLabel(title)
        self._selector = QComboBox()
        self._apply_button = QPushButton("Apply")
        self._apply_button.clicked.connect(self._emit_current_algorithm)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._title)
        layout.addWidget(self._selector)
        layout.addWidget(self._apply_button)
        self.setLayout(layout)

        if algorithms:
            self.set_algorithms(algorithms)

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
