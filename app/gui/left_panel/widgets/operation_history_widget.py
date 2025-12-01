from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget, QTextEdit


class OperationHistoryWidget(QWidget):
    def __init__(self, parent: QWidget | None = None, height: int = 180) -> None:
        super().__init__(parent)
        self._setup_ui(height)

    def _setup_ui(self, height: int) -> None:
        layout: QVBoxLayout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QLabel("History")
        title.setObjectName("leftPanelSectionTitle")
        layout.addWidget(title)

        self._text: QTextEdit = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        font.setPointSize(10)
        self._text.setFont(font)
        self._text.setFixedHeight(height)
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
