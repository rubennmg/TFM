from PyQt6.QtGui import QTextOption
from PyQt6.QtWidgets import QPlainTextEdit, QWidget

from utils.torch import get_device_info


class DeviceLogger(QPlainTextEdit):
    """Minimal logging panel for device information.

    Args:
        QPlainTextEdit (QWidget): PyQt6 Plain Text Edit widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("generalLogger")
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        self._load_device_info()

    def _load_device_info(self) -> None:
        """Load and display device information from torch."""
        try:
            device_info_lines = get_device_info()
            for line in device_info_lines:
                self.appendPlainText(line)
        except Exception as e:
            self.appendPlainText(f"Error loading device information: {e}")

    def append_entry(self, entry: str) -> None:
        self.appendPlainText(entry)
        scrollbar = self.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def clear_entries(self) -> None:
        self.setPlainText("")
