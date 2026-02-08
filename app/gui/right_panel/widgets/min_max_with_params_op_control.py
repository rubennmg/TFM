from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class MinMaxWithParamsOperationControl(QWidget):
    def __init__(self, controller, operation_index: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.controller = controller
        self.operation_index = operation_index
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("operationControl")
        self.setToolTip("Apply Min-Max Normalization with specified min and max values")

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 18, 10, 8)
        layout.setSpacing(10)

        title = QLabel("Min-Max Normalization with Params")
        title.setObjectName("operationControlTitle")
        layout.addWidget(title)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)

        self.min_spin = QDoubleSpinBox()
        self.min_spin.setObjectName("minSpin")
        self.min_spin.setRange(0.0, 1.0)
        self.min_spin.setSingleStep(0.01)
        self.min_spin.setDecimals(2)
        self.min_spin.setValue(0.0)
        self.min_spin.setToolTip("Minimum value for normalization (>= 0.0 and < max)")
        self.min_spin.setPrefix("Min: ")
        self.min_spin.valueChanged.connect(self._on_apply)

        self.max_spin = QDoubleSpinBox()
        self.max_spin.setObjectName("maxSpin")
        self.max_spin.setRange(0.0, 1.0)
        self.max_spin.setSingleStep(0.01)
        self.max_spin.setDecimals(2)
        self.max_spin.setValue(1.0)
        self.max_spin.setToolTip("Maximum value for normalization (> min and <= 1.0)")
        self.max_spin.setPrefix("Max: ")
        self.max_spin.valueChanged.connect(self._on_apply)

        controls_row.addWidget(self.min_spin)
        controls_row.addWidget(self.max_spin)
        layout.addLayout(controls_row)

        self.setLayout(layout)

    def _on_apply(self):
        min_val = self.min_spin.value()
        max_val = self.max_spin.value()

        if min_val >= max_val:
            return

        self.controller.apply_operation(
            "MinMaxNormalizationWithParams",
            operation_idx=self.operation_index,
            min=min_val,
            max=max_val,
        )

    def reset(self) -> None:
        self.min_spin.setValue(0.0)
        self.max_spin.setValue(1.0)
