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
from gui.right_panel.widgets.min_max_with_params_op_control import (
    MinMaxWithParamsOperationControl,
)
from gui.right_panel.widgets.minmax_percentile_op_control import (
    MinMaxPercentileOperationControl,
)
from gui.right_panel.widgets.no_param_op_control import NoParamOperationControl
from gui.right_panel.widgets.resize_op_control import ResizeOperationControl
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
            "minmax_percentile": lambda: MinMaxPercentileOperationControl(
                controller=self.controller,
                operation_index=operation_index,
                parent=self,
            ),
            "minmax_with_params": lambda: MinMaxWithParamsOperationControl(
                controller=self.controller,
                operation_index=operation_index,
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
            "resize": lambda: ResizeOperationControl(
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
