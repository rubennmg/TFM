from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from gui.helpers.debayer_control_widget import DebayerControlWidget
from gui.helpers.operation_control_widget import OperationControlWidget
from models.float_param_spec import FloatParamSpec

DEBOUNCE_MS: int = 75  # miliseconds to debounce slider changes


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
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                default=0,
            ),
        ]
        self.sigmoid_widget: OperationControlWidget = OperationControlWidget(
            "Sigmoid contrast", sig_params, self
        )

        try:
            self.sigmoid_widget.set_debounce_ms(DEBOUNCE_MS)
        except Exception:
            pass

        self.sigmoid_widget.paramsChanged.connect(self._on_sigmoid_params)
        self.sigmoid_widget.setToolTip("Adjust image contrast using a sigmoid function")
        self.sigmoid_widget.setObjectName("operationControl")

        self.debayer_widget: DebayerControlWidget = DebayerControlWidget(parent=self)
        self.debayer_widget.applyClicked.connect(self._on_apply_debayer)

        layout.addWidget(self.sigmoid_widget)
        layout.addWidget(self.debayer_widget)

        layout.addStretch()

        self.setLayout(layout)

    def _on_sigmoid_params(self, params: dict) -> None:
        gain: float = float(params.get("gain", 10.0))
        cutoff: float = float(params.get("cutoff", 0.5))
        self.controller.apply_contrast(gain, cutoff)

    def _on_apply_debayer(self, algorithm_key: str) -> None:
        if not algorithm_key:
            return
        self.controller.apply_debayer(algorithm_key)
