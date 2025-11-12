from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox


def show_error_dialog(title: str, message: str, detailed: Optional[str] = None) -> None:
    box: QMessageBox = QMessageBox(None)
    box.setWindowModality(Qt.WindowModality.ApplicationModal)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle(title)
    box.setText(message)
    if detailed:
        box.setDetailedText(detailed)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()
