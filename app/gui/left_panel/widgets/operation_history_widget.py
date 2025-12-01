from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QSizePolicy, QTextEdit, QVBoxLayout, QWidget


class OperationHistoryWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout: QVBoxLayout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QLabel("History")
        title.setObjectName("operationHistoryTitle")
        layout.addWidget(title)

        self._text: QTextEdit = QTextEdit()
        self._text.setObjectName("operationHistoryText")
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        font.setPointSize(10)
        self._text.setFont(font)
        size_policy = QSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._text.setSizePolicy(size_policy)
        self.setMinimumHeight(0)
        self._text.setMinimumHeight(0)
        self._text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(self._text)

        self.setLayout(layout)

    def clear(self) -> None:
        self._text.clear()

    def append_line(self, text: str) -> None:
        self._text.append(text)

    def set_lines(self, lines: list[str]) -> None:
        self._text.clear()
        if not lines:
            return
        self._text.setPlainText("\n".join(lines))
