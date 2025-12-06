from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class LabelledButton(QWidget):
    def __init__(self, label_text: str, button_text: str, parent=None):
        super().__init__(parent)
        self.label: QLabel = QLabel(label_text)
        self.button: QPushButton = QPushButton(button_text)

        # layout
        vl: QVBoxLayout = QVBoxLayout()
        vl.setAlignment(Qt.AlignmentFlag.AlignTop)

        # widgets
        vl.addWidget(self.label)
        vl.addWidget(self.button)

        self.setLayout(vl)

    @property
    def clicked(self):
        return self.button.clicked
