from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QSpinBox,
    QPushButton,
    QHBoxLayout,
)


class MedianFilterControlWidget(QWidget):
    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__(parent)
        self.controller = controller
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("operationControl")
        self.setToolTip("Apply Median Filter to reduce impulse noise")

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 18, 10, 8)
        layout.setSpacing(10)

        title = QLabel("Median Filter")
        title.setObjectName("operationControlTitle")
        layout.addWidget(title)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(1, 31)
        self.kernel_spin.setSingleStep(2)
        self.kernel_spin.setValue(3)
        self.kernel_spin.setToolTip("Odd kernel size (e.g. 3,5,7)")
        self.kernel_spin.setObjectName("percentileSpin")

        row.addWidget(QLabel("Kernel:"))
        row.addWidget(self.kernel_spin)
        layout.addLayout(row)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._on_apply)
        layout.addWidget(self.apply_btn)

        self.setLayout(layout)

    def _on_apply(self) -> None:
        k = int(self.kernel_spin.value())

        if k % 2 == 0:
            k = max(1, k - 1)
            self.kernel_spin.setValue(k)

        self.controller.apply_operation("MedianFilter", kernel_size=k)

    def reset(self) -> None:
        self.kernel_spin.setValue(3)
