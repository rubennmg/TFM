from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.right_panel.widgets.color_control import ColorControl
from gui.right_panel.widgets.debayer_control import DebayerControl
from gui.right_panel.widgets.filter_control import FilterControl
from gui.right_panel.widgets.flip_control import FlipControl
from gui.right_panel.widgets.median_filter_control import MedianFilterControl
from gui.right_panel.widgets.minmax_control import MinMaxControl
from gui.right_panel.widgets.minmax_percentile_control import (
    MinMaxPercentileControl,
)
from models.float_param_spec import FloatParamSpec
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
        if control_type == "filter":
            params: list[FloatParamSpec] | None = definition.params
            return FilterControl(
                definition.label,
                self.controller,
                definition.name,
                operation_index,
                params or [],
                self,
            )
        if control_type == "color":
            return ColorControl(
                definition.label,
                self.controller,
                definition.name,
                operation_index,
                self,
            )
        if control_type == "median":
            return MedianFilterControl(self.controller, operation_index, self)
        if control_type == "minmax":
            return MinMaxControl(self.controller, operation_index, self)
        if control_type == "minmax_percentile":
            return MinMaxPercentileControl(self.controller, operation_index, self)
        if control_type == "flip":
            return FlipControl(self.controller, operation_index, self)
        if control_type == "debayer":
            return DebayerControl(
                controller=self.controller,
                operation_index=operation_index,
                parent=self,
            )

        raise ValueError(f"Unsupported control type '{control_type}'")

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
