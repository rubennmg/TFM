from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MinMaxPercentileControlWidget(QWidget):
    def __init__(self, controller, parent: QWidget | None = None):
        super().__init__(parent)
        self.controller = controller
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("operationControl")
        self.setToolTip("Apply Min-Max Percentile Normalization to the image")

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 18, 10, 8)
        layout.setSpacing(6)

        title = QLabel("Min-Max Percentile Normalization")
        title.setObjectName("operationControlTitle")
        layout.addWidget(title)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(8)

        self.lower_spin = QDoubleSpinBox()
        self.lower_spin.setObjectName("percentileSpin")
        self.lower_spin.setRange(0.0, 1.0)
        self.lower_spin.setSingleStep(0.01)
        self.lower_spin.setDecimals(2)
        self.lower_spin.setValue(0.02)
        self.lower_spin.setToolTip("Lower percentile (>= 0.0 and < upper)")
        self.lower_spin.setPrefix("Low: ")

        self.upper_spin = QDoubleSpinBox()
        self.upper_spin.setObjectName("percentileSpin")
        self.upper_spin.setRange(0.0, 1.0)
        self.upper_spin.setSingleStep(0.01)
        self.upper_spin.setDecimals(2)
        self.upper_spin.setValue(0.98)
        self.upper_spin.setToolTip("Upper percentile (> lower and <= 1.0)")
        self.upper_spin.setPrefix("Up: ")

        controls_row.addWidget(self.lower_spin)
        controls_row.addWidget(self.upper_spin)
        layout.addLayout(controls_row)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._on_apply)
        layout.addWidget(self.apply_btn)

        self.lower_spin.valueChanged.connect(self._validate)
        self.upper_spin.valueChanged.connect(self._validate)
        self._validate()

        self.setLayout(layout)

    def _validate(self):
        lower = self.lower_spin.value()
        upper = self.upper_spin.value()
        valid = 0.0 <= lower < upper <= 1.0
        self.apply_btn.setEnabled(valid)
        if not valid:
            self.apply_btn.setToolTip(
                "Invalid percentiles: must satisfy 0.0 <= lower < upper <= 1.0"
            )
        else:
            self.apply_btn.setToolTip("Apply with L=%.3f U=%.3f" % (lower, upper))

    def _on_apply(self):
        lower = self.lower_spin.value()
        upper = self.upper_spin.value()

        self.controller.apply_operation(
            "MinMaxPercentileNormalization",
            lower_percentile=lower,
            upper_percentile=upper,
        )
