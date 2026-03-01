from typing import Iterable

from PyQt6.QtWidgets import QWidget

from gui.right_panel.widgets.enum_param_op_control import EnumParamOperationControl

_METHODS: Iterable[tuple[str, str]] = (
    ("gray_world", "Gray World"),
    ("max_rgb", "Max RGB"),
)


class WhiteBalanceOperationControl(EnumParamOperationControl):
    """Compound widget with label, selector and apply button for white balance.

    Args:
        QWidget (QWidget): Base Qt widget.
    """

    def __init__(
        self,
        controller,
        operation_index: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            controller=controller,
            operation_index=operation_index,
            title="White Balance",
            tooltip="Apply white balance adjustment to the image",
            operation_name="WhiteBalance",
            param_name="method",
            options=_METHODS,
            parent=parent,
        )

    def set_methods(self, methods: Iterable[tuple[str, str]]) -> None:
        self.set_options(methods)
