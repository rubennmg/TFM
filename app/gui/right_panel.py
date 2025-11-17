from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from gui.helpers.operation_control_widget import OperationControlWidget
from models.float_param_spec import FloatParamSpec


class RightPanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout: QVBoxLayout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        label_title: QLabel = QLabel("TRANSFORMATIONS")
        label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_title.setObjectName("transformationsTitle")
        layout.addWidget(label_title)

        # sigmoid contrast control
        sig_params: list[FloatParamSpec] = [
            FloatParamSpec(
                key="gain",
                label="Gain",
                minimum=0.0,
                maximum=50.0,
                step=0.1,
                default=0.0,
            ),
            FloatParamSpec(
                key="cutoff",
                label="Cutoff",
                minimum=-1.0,
                maximum=1.0,
                step=0.05,
                default=0,
            ),
        ]
        self.sigmoid_widget: OperationControlWidget = OperationControlWidget(
            "Sigmoid contrast", sig_params, self
        )
        self.sigmoid_widget.paramsChanged.connect(self._on_sigmoid_params)
        self.sigmoid_widget.setToolTip("Adjust image contrast using a sigmoid function")
        self.sigmoid_widget.setObjectName("operationControl")

        layout.addWidget(self.sigmoid_widget)

        # debayer 5x5 button
        self.label_debayer: QLabel = QLabel("Debayer 5x5")
        self.button_debayer: QPushButton = QPushButton("Apply Debayer 5x5")
        self.button_debayer.clicked.connect(self._on_debayer_clicked)

        layout.addWidget(self.label_debayer)
        layout.addWidget(self.button_debayer)
        layout.addStretch()

        self.setLayout(layout)

    def _on_sigmoid_params(self, params: dict) -> None:
        gain: float = float(params.get("gain", 10.0))
        cutoff: float = float(params.get("cutoff", 0.5))
        self.controller.apply_contrast(gain, cutoff)

    def _on_debayer_clicked(self) -> None:
        self.controller.apply_debayer5x5()
