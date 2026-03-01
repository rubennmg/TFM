from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.right_panel.widgets.debayer_op_control import DebayerOperationControl
from gui.right_panel.widgets.filter_op_control import FilterOperationControl
from gui.right_panel.widgets.flip_op_control import FlipOperationControl
from gui.right_panel.widgets.light_compensation_op_control import (
    LightCompensationOperationControl,
)
from gui.right_panel.widgets.no_param_op_control import NoParamOperationControl
from gui.right_panel.widgets.two_numeric_op_control import (
    NumericSpinConfig,
    TwoNumericOperationControl,
)
from gui.right_panel.widgets.white_balance_op_control import (
    WhiteBalanceOperationControl,
)
from models.operation_definition import OperationDefinition, get_operation_definitions


class RightPanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._definitions: dict[str, OperationDefinition] = {
            definition.name: definition for definition in get_operation_definitions()
        }
        self._control_stack: list[tuple[str, QWidget]] = []
        self._controls_layout: QVBoxLayout | None = None
        self._placeholder_widget: QWidget | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout: QVBoxLayout = QVBoxLayout()

        container_widget = QWidget()
        container_widget.setObjectName("operationControlsLayout")
        container_layout: QVBoxLayout = QVBoxLayout()
        container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        container_layout.setContentsMargins(4, 8, 4, 8)
        container_widget.setLayout(container_layout)

        label_title: QLabel = QLabel("Operation Controls")
        label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_title.setObjectName("operationControlsTitle")
        container_layout.addWidget(label_title)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self._controls_layout = QVBoxLayout()
        self._controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._controls_layout.setContentsMargins(12, 0, 12, 0)
        self._controls_layout.setSpacing(12)

        self._placeholder_widget = self._build_placeholder()
        self._controls_layout.addWidget(self._placeholder_widget)

        container = QWidget()
        container.setLayout(self._controls_layout)
        scroll_area.setWidget(container)

        container_layout.addWidget(scroll_area)

        layout.addWidget(container_widget)

        self.setLayout(layout)

    def _build_placeholder(self) -> QWidget:
        wrapper = QWidget()
        inner_layout = QVBoxLayout()
        inner_layout.setContentsMargins(20, 40, 20, 40)
        inner_layout.setSpacing(10)

        message = QLabel(
            "Add an operation from the pipeline to configure its parameters here."
        )
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message.setWordWrap(True)
        message.setObjectName("operationControlPlaceholderText")
        inner_layout.addWidget(message)
        wrapper.setLayout(inner_layout)
        return wrapper

    def _build_control(
        self, definition: OperationDefinition, operation_index: int
    ) -> QWidget:
        control_type = definition.control_type

        factory_map = {
            "filter": lambda: FilterOperationControl(
                title=definition.label,
                controller=self.controller,
                operation_name=definition.name,
                operation_index=operation_index,
                params=definition.params or [],
                parent=self,
            ),
            "minmax_percentile": lambda: TwoNumericOperationControl(
                controller=self.controller,
                operation_index=operation_index,
                operation_name="MinMaxPercentileNormalization",
                title="Min-Max Percentile Normalization",
                tooltip="Apply Min-Max Percentile Normalization to the image",
                first_spin=NumericSpinConfig(
                    object_name="percentileSpin",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    decimals=2,
                    default=0.02,
                    prefix="Low: ",
                    tooltip="Lower percentile (>= 0.0 and < upper)",
                ),
                second_spin=NumericSpinConfig(
                    object_name="percentileSpin",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    decimals=2,
                    default=0.98,
                    prefix="Up: ",
                    tooltip="Upper percentile (> lower and <= 1.0)",
                ),
                build_operation_params=lambda lower, upper: {
                    "lower_percentile": lower,
                    "upper_percentile": upper,
                },
                parent=self,
            ),
            "minmax_with_params": lambda: TwoNumericOperationControl(
                controller=self.controller,
                operation_index=operation_index,
                operation_name="MinMaxNormalizationWithParams",
                title="Min-Max Normalization with Params",
                tooltip="Apply Min-Max Normalization with specified min and max values",
                first_spin=NumericSpinConfig(
                    object_name="minSpin",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    decimals=2,
                    default=0.0,
                    prefix="Min: ",
                    tooltip="Minimum value for normalization (>= 0.0 and < max)",
                ),
                second_spin=NumericSpinConfig(
                    object_name="maxSpin",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    decimals=2,
                    default=1.0,
                    prefix="Max: ",
                    tooltip="Maximum value for normalization (> min and <= 1.0)",
                ),
                build_operation_params=lambda min_value, max_value: {
                    "min": min_value,
                    "max": max_value,
                },
                validator=lambda min_value, max_value: min_value < max_value,
                parent=self,
            ),
            "flip": lambda: FlipOperationControl(
                controller=self.controller,
                operation_index=operation_index,
                parent=self,
            ),
            "debayer": lambda: DebayerOperationControl(
                controller=self.controller,
                operation_index=operation_index,
                parent=self,
            ),
            "no_param": lambda: NoParamOperationControl(
                controller=self.controller,
                title=definition.label,
                operation_name=definition.name,
                operation_index=operation_index,
                parent=self,
            ),
            "resize": lambda: TwoNumericOperationControl(
                controller=self.controller,
                operation_index=operation_index,
                operation_name="Resize",
                title="Resize",
                tooltip="Resize the image to specified width and height",
                first_spin=NumericSpinConfig(
                    object_name="widthSpin",
                    minimum=1.0,
                    maximum=4096.0,
                    step=1.0,
                    decimals=0,
                    default=1024.0,
                    prefix="Width: ",
                    tooltip="Width to resize the image to (>= 1)",
                ),
                second_spin=NumericSpinConfig(
                    object_name="heightSpin",
                    minimum=1.0,
                    maximum=4096.0,
                    step=1.0,
                    decimals=0,
                    default=768.0,
                    prefix="Height: ",
                    tooltip="Height to resize the image to (>= 1)",
                ),
                build_operation_params=lambda width, height: {
                    "size": (int(height), int(width)),
                },
                parent=self,
            ),
            "white_balance": lambda: WhiteBalanceOperationControl(
                controller=self.controller,
                operation_index=operation_index,
                parent=self,
            ),
            "light_compensation": lambda: LightCompensationOperationControl(
                controller=self.controller,
                operation_index=operation_index,
                parent=self,
            ),
        }

        if control_type not in factory_map:
            raise ValueError(f"Unknown control type: {control_type}")

        return factory_map[control_type]()

    def push_operation_control(self, operation_name: str, operation_index: int) -> None:
        if self._controls_layout is None:
            return
        definition = self._definitions.get(operation_name)
        if definition is None:
            return

        widget = self._build_control(definition, operation_index)
        self._control_stack.append((operation_name, widget))
        if self._placeholder_widget is not None:
            self._placeholder_widget.hide()
        self._controls_layout.addWidget(widget)

    def pop_operation_control(self) -> None:
        if self._controls_layout is None or not self._control_stack:
            return

        _, widget = self._control_stack.pop()
        self._controls_layout.removeWidget(widget)
        widget.deleteLater()

        if not self._control_stack and self._placeholder_widget is not None:
            self._placeholder_widget.show()

    def clear_pipeline_controls(self) -> None:
        if self._controls_layout is None:
            return

        while self._control_stack:
            _, widget = self._control_stack.pop()
            self._controls_layout.removeWidget(widget)
            widget.deleteLater()

        if self._placeholder_widget is not None:
            self._placeholder_widget.show()
