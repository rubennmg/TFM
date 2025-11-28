from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from gui.right_panel.widgets.debayer_control_widget import DebayerControlWidget
from gui.right_panel.widgets.filter_control_widget import FilterControlWidget
from gui.right_panel.widgets.flip_control_widget import FlipControlWidget
from models.float_param_spec import FloatParamSpec


class RightPanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout: QVBoxLayout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        label_title: QLabel = QLabel("TRANSFORMERS")
        label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_title.setObjectName("transformationsTitle")

        # SIGMOID CONTRAST CONTROL
        sig_params: list[FloatParamSpec] = [
            FloatParamSpec(
                key="gain",
                label="Gain",
                minimum=0.0,
                maximum=10.0,
                step=0.1,
                default=0.0,
            ),
            FloatParamSpec(
                key="cutoff",
                label="Cutoff",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                default=0,
            ),
        ]
        self.sigmoid_widget: FilterControlWidget = FilterControlWidget(
            "Sigmoid contrast",
            self.controller,
            "SigmoidContrast",
            sig_params,
            self,
        )
        self.sigmoid_widget.setToolTip("Adjust image contrast using a sigmoid function")

        # DEBAYER CONTROL
        self.debayer_widget: DebayerControlWidget = DebayerControlWidget(
            parent=self, controller=self.controller
        )

        # FLIP CONTROL
        self.flip_widget = FlipControlWidget(self.controller, self)

        # ROTATE CONTROL
        rotation_params: list[FloatParamSpec] = [
            FloatParamSpec(
                key="angle",
                label="Angle",
                minimum=-360.0,
                maximum=360.0,
                step=1.0,
                default=0.0,
            ),
        ]
        self.rotate_widget: FilterControlWidget = FilterControlWidget(
            "Image Rotation",
            self.controller,
            "Rotate",
            rotation_params,
            self,
        )
        self.rotate_widget.setToolTip("Rotate image by a specified angle")

        layout.addWidget(label_title)
        layout.addWidget(self.sigmoid_widget)
        layout.addWidget(self.flip_widget)
        layout.addWidget(self.rotate_widget)
        layout.addWidget(self.debayer_widget)

        layout.addStretch()

        self.setLayout(layout)

    def _on_sigmoid_params(self, params: dict) -> None:
        gain: float = float(params.get("gain", 10.0))
        cutoff: float = float(params.get("cutoff", 0.5))
        self.controller.apply_operation("SigmoidContrast", gain=gain, cutoff=cutoff)
