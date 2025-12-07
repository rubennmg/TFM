from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


class FlipControl(QWidget):
    def __init__(self, controller, operation_index: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.controller = controller
        self.operation_index = operation_index
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setToolTip("Flip the image horizontally or vertically")
        self.setObjectName("operationControl")

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 18, 10, 8)
        layout.setSpacing(10)

        self.title_lbl = QLabel("Image flipping")
        self.title_lbl.setObjectName("operationControlTitle")
        layout.addWidget(self.title_lbl)

        self.horizontal_rb = QRadioButton("Horizontal")
        self.vertical_rb = QRadioButton("Vertical")
        self.horizontal_rb.setChecked(True)

        self.flip_group = QButtonGroup(self)
        self.flip_group.addButton(self.horizontal_rb)
        self.flip_group.addButton(self.vertical_rb)

        rb_layout = QHBoxLayout()
        rb_layout.addWidget(self.horizontal_rb)
        rb_layout.addWidget(self.vertical_rb)
        layout.addLayout(rb_layout)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._on_apply)

        layout.addWidget(self.apply_button)

        self.setLayout(layout)

    def _on_apply(self):
        if self.horizontal_rb.isChecked():
            self.controller.apply_operation(
                "Flip", operation_idx=self.operation_index, horizontal=True
            )
        else:
            self.controller.apply_operation(
                "Flip", operation_idx=self.operation_index, horizontal=False
            )

    def reset(self) -> None:
        self.horizontal_rb.setChecked(True)
        self.vertical_rb.setChecked(False)
