from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class ResizeOperationControl(QWidget):
    def __init__(self, controller, operation_index: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.controller = controller
        self.operation_index = operation_index
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("operationControl")
        self.setToolTip("Resize the image to specified width and height")

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 18, 10, 8)
        layout.setSpacing(10)

        title = QLabel("Resize")
        title.setObjectName("operationControlTitle")
        layout.addWidget(title)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setObjectName("widthSpin")
        self.width_spin.setRange(1.0, 4096.0)
        self.width_spin.setSingleStep(1.0)
        self.width_spin.setDecimals(0)
        self.width_spin.setValue(1024.0)
        self.width_spin.setToolTip("Width to resize the image to (>= 1)")
        self.width_spin.setPrefix("Width: ")
        self.width_spin.valueChanged.connect(self._on_apply)

        self.height_spin = QDoubleSpinBox()
        self.height_spin.setObjectName("heightSpin")
        self.height_spin.setRange(1.0, 4096.0)
        self.height_spin.setSingleStep(1.0)
        self.height_spin.setDecimals(0)
        self.height_spin.setValue(768.0)
        self.height_spin.setToolTip("Height to resize the image to (>= 1)")
        self.height_spin.setPrefix("Height: ")
        self.height_spin.valueChanged.connect(self._on_apply)

        controls_row.addWidget(self.width_spin)
        controls_row.addWidget(self.height_spin)
        layout.addLayout(controls_row)

        self.setLayout(layout)

    def _on_apply(self):
        width_val = int(self.width_spin.value())
        height_val = int(self.height_spin.value())

        self.controller.apply_operation(
            "Resize",
            operation_idx=self.operation_index,
            size=(height_val, width_val),
        )
