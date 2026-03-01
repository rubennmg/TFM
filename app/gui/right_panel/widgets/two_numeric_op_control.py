from dataclasses import dataclass
from typing import Any, Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget


@dataclass(frozen=True)
class NumericSpinConfig:
    object_name: str
    minimum: float
    maximum: float
    step: float
    decimals: int
    default: float
    prefix: str
    tooltip: str


class TwoNumericOperationControl(QWidget):
    def __init__(
        self,
        controller,
        operation_index: int,
        operation_name: str,
        title: str,
        tooltip: str,
        first_spin: NumericSpinConfig,
        second_spin: NumericSpinConfig,
        build_operation_params: Callable[[float, float], dict[str, Any]],
        validator: Callable[[float, float], bool] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.controller = controller
        self.operation_index = operation_index
        self.operation_name = operation_name
        self._build_operation_params = build_operation_params
        self._validator = validator
        self._first_default = first_spin.default
        self._second_default = second_spin.default

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("operationControl")
        self.setToolTip(tooltip)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 18, 10, 8)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setObjectName("operationControlTitle")
        layout.addWidget(title_label)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)

        self.first_spin = self._build_spin(first_spin)
        self.second_spin = self._build_spin(second_spin)

        controls_row.addWidget(self.first_spin)
        controls_row.addWidget(self.second_spin)
        layout.addLayout(controls_row)

        self.setLayout(layout)

    def _build_spin(self, config: NumericSpinConfig) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setObjectName(config.object_name)
        spin.setRange(config.minimum, config.maximum)
        spin.setSingleStep(config.step)
        spin.setDecimals(config.decimals)
        spin.setValue(config.default)
        spin.setToolTip(config.tooltip)
        spin.setPrefix(config.prefix)
        spin.valueChanged.connect(self._on_apply)
        return spin

    def _on_apply(self) -> None:
        first_value = self.first_spin.value()
        second_value = self.second_spin.value()

        if self._validator is not None and not self._validator(
            first_value, second_value
        ):
            return

        self.controller.apply_operation(
            self.operation_name,
            operation_idx=self.operation_index,
            **self._build_operation_params(first_value, second_value),
        )

    def reset(self) -> None:
        self.first_spin.setValue(self._first_default)
        self.second_spin.setValue(self._second_default)
