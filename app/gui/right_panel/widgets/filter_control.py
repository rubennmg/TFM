from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from models.float_param_spec import FloatParamSpec


class _FloatSlider(QWidget):
    """Compound widget: QLabel + QSlider (float-mapped) + QDoubleSpinBox.

    - Exposes value() / setValue(float)
    - Emits valueChanged(float) whenever slider or spinbox changes.
    """

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        step: float,
        default: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._min: float = minimum
        self._max: float = maximum
        self._step: float = max(step, 1e-9)
        self._default: float = default

        # compute integer range mapping for QSlider
        self._int_min: int = 0
        # guard: cap number of steps to avoid too many signals
        steps = int(round((self._max - self._min) / self._step))
        self._int_max: int = max(1, min(10000, steps))

        self._label: QLabel = QLabel(label)
        self._slider: QSlider = QSlider(Qt.Orientation.Horizontal)
        self._spin: QDoubleSpinBox = QDoubleSpinBox()

        self._slider.setRange(self._int_min, self._int_max)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(max(1, self._int_max // 20))
        self._slider.setTracking(True)

        self._spin.setRange(self._min, self._max)
        self._spin.setSingleStep(self._step)
        self._spin.setDecimals(min(6, self._infer_decimals(self._step)))
        self._spin.setKeyboardTracking(False)

        hl: QHBoxLayout = QHBoxLayout()
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self._label)
        hl.addWidget(self._slider, stretch=1)
        hl.addWidget(self._spin)
        self.setLayout(hl)

        self.setValue(self._default)

        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spin.valueChanged.connect(self._on_spin_changed)

    @staticmethod
    def _infer_decimals(step: float) -> int:
        s: list[str] = f"{step:.10f}".rstrip("0").split(".")
        return len(s[1]) if len(s) > 1 else 0

    def _float_to_int(self, v: float) -> int:
        ratio: float = (
            (v - self._min) / (self._max - self._min) if self._max != self._min else 0.0
        )
        return int(round(self._int_min + ratio * (self._int_max - self._int_min)))

    def _int_to_float(self, i: int) -> float:
        ratio: float = (
            (i - self._int_min) / (self._int_max - self._int_min)
            if self._int_max != self._int_min
            else 0.0
        )
        v: float = self._min + ratio * (self._max - self._min)
        # snap to step grid
        steps_from_min: int = round((v - self._min) / self._step)
        return max(self._min, min(self._max, self._min + steps_from_min * self._step))

    def _on_slider_changed(self, ival: int) -> None:
        fval: float = self._int_to_float(ival)
        # block signal loops when syncing widgets
        try:
            self._spin.blockSignals(True)
            self._spin.setValue(fval)
        finally:
            self._spin.blockSignals(False)
        self.valueChanged.emit(fval)

    def _on_spin_changed(self, fval: float) -> None:
        ival: int = self._float_to_int(float(fval))
        try:
            self._slider.blockSignals(True)
            self._slider.setValue(ival)
        finally:
            self._slider.blockSignals(False)
        self.valueChanged.emit(float(fval))

    def value(self) -> float:
        return float(self._spin.value())

    def setValue(self, v: float) -> None:  # noqa: N802 (Qt style)
        v = max(self._min, min(self._max, v))
        ival: int = self._float_to_int(v)
        self._slider.setValue(ival)
        self._spin.setValue(v)

    def reset_to_default(self) -> None:
        self.setValue(self._default)


class FilterControl(QWidget):
    """Reusable filter operation widget: title + parameters + optional extra widgets.

    - Accepts a list of FloatParamSpec instances to create parameters.
    - add_widget(QWidget) allows adding arbitrary custom controls.
    """

    def __init__(
        self,
        title: str,
        controller,
        operation_name: str,
        operation_index: int,
        params: list[FloatParamSpec] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.controller = controller
        self.operation_name = operation_name
        self.operation_index = operation_index

        self._container = QWidget()
        self._container.setObjectName("operationControl")

        self._title_lbl = QLabel(title)
        self._title_lbl.setObjectName("operationControlTitle")

        self._form = QFormLayout()
        self._form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        self._form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._sliders: dict[str, _FloatSlider] = {}

        self._suppress_emit: bool = False

        if params:
            for p in params:
                slider = _FloatSlider(
                    p.label, p.minimum, p.maximum, p.step, p.default, self
                )
                slider._slider.sliderReleased.connect(self._on_interaction_finished)
                slider._spin.editingFinished.connect(self._on_interaction_finished)
                self._sliders[p.key] = slider
                self._form.addRow(slider)

        self._extra_container = QVBoxLayout()
        self._extra_container.setContentsMargins(0, 0, 0, 0)

        inner_layout = QVBoxLayout()
        inner_layout.setContentsMargins(10, 18, 10, 8)
        inner_layout.setSpacing(10)
        inner_layout.addWidget(self._title_lbl)
        inner_layout.addLayout(self._form)
        inner_layout.addLayout(self._extra_container)

        self._container.setLayout(inner_layout)

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self._container)

        self.setLayout(root_layout)

    def add_widget(self, w: QWidget) -> None:
        self._extra_container.addWidget(w)

    def get_params(self) -> dict[str, float]:
        return {k: s.value() for k, s in self._sliders.items()}

    def set_param(self, key: str, value: float) -> None:
        if key in self._sliders:
            self._suppress_emit = True
            try:
                self._sliders[key].setValue(value)
            finally:
                self._suppress_emit = False

    def _on_interaction_finished(self) -> None:
        if self._suppress_emit:
            return
        self.controller.apply_operation(
            self.operation_name,
            operation_idx=self.operation_index,
            **self.get_params(),
        )

    def reset_controls_to_default(self) -> None:
        self._suppress_emit = True
        try:
            for slider in self._sliders.values():
                slider.reset_to_default()
        finally:
            self._suppress_emit = False

    def reset(self) -> None:
        self.reset_controls_to_default()
