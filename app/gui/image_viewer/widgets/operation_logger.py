from PyQt6.QtGui import QTextOption
from PyQt6.QtWidgets import QPlainTextEdit, QWidget


class OperationLogger(QPlainTextEdit):
    """Minimal logging panel.

    Args:
        QPlainTextEdit (QWidget): PyQt6 Plain Text Edit widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("operationLogger")
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setWordWrapMode(QTextOption.WrapMode.NoWrap)

    def append_entry(self, entry: str) -> None:
        self.appendPlainText(entry)
        scrollbar = self.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def clear_entries(self) -> None:
        self.setPlainText("")
