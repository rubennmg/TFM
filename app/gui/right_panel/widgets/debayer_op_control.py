from typing import Iterable

from PyQt6.QtWidgets import QWidget

from gui.right_panel.widgets.enum_param_op_control import EnumParamOperationControl

_ALGORITHMS: Iterable[tuple[str, str]] = (
    ("debayer2x2", "Debayer 2x2"),
    ("debayer3x3", "Debayer 3x3"),
    ("debayer5x5", "Debayer 5x5"),
    ("debayersplit", "Debayer Split"),
)


class DebayerOperationControl(EnumParamOperationControl):
    """Compound widget with label, selector and apply button for debayering.

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
            title="Debayer Demosaicing",
            tooltip="Apply debayering to a RAW image",
            operation_name="Debayer",
            param_name="algorithm_name",
            options=_ALGORITHMS,
            parent=parent,
        )

    def set_algorithms(self, algorithms: Iterable[tuple[str, str]]) -> None:
        self.set_options(algorithms)

    def current_algorithm(self) -> str | None:
        return self.current_value()
