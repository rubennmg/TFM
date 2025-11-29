from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)

from gui.right_panel.widgets.debayer_control import DebayerControlWidget
from gui.right_panel.widgets.filter_control import FilterControlWidget
from gui.right_panel.widgets.flip_control import FlipControlWidget
from gui.right_panel.widgets.minmax_control import MinMaxControlWidget
from gui.right_panel.widgets.collapsible_section import CollapsibleSection
from models.float_param_spec import FloatParamSpec


class RightPanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout: QVBoxLayout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        label_title: QLabel = QLabel("Transformers")
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

        # GAUSSIAN FILTER CONTROL
        gaussian_params: list[FloatParamSpec] = [
            FloatParamSpec(
                key="kernel_size",
                label="Kernel Size",
                minimum=1,
                maximum=21,
                step=2,
                default=5,
            ),
            FloatParamSpec(
                key="sigma",
                label="Sigma",
                minimum=0.1,
                maximum=10.0,
                step=0.1,
                default=1.0,
            ),
        ]
        self.gaussian_widget: FilterControlWidget = FilterControlWidget(
            "Gaussian Filter",
            self.controller,
            "GaussianFilter",
            gaussian_params,
            self,
        )
        self.gaussian_widget.setToolTip("Apply Gaussian filter to smooth the image")

        # MIN-MAX NORMALIZATION CONTROL
        self.minmax_widget = MinMaxControlWidget(self.controller, self)

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

        # DEBAYER CONTROL
        self.debayer_widget: DebayerControlWidget = DebayerControlWidget(
            parent=self, controller=self.controller
        )

        layout.addWidget(label_title)

        # filters
        filters_container = QWidget()
        filters_layout = QVBoxLayout()
        filters_layout.setContentsMargins(0, 0, 0, 0)
        filters_layout.setSpacing(6)
        filters_layout.addWidget(self.sigmoid_widget)
        filters_layout.addWidget(self.gaussian_widget)
        filters_container.setLayout(filters_layout)
        layout.addWidget(CollapsibleSection("Filters", filters_container, self))

        # normalization
        normalization_container = QWidget()
        normalization_layout = QVBoxLayout()
        normalization_layout.setContentsMargins(0, 0, 0, 0)
        normalization_layout.setSpacing(6)
        normalization_layout.addWidget(self.minmax_widget)
        normalization_container.setLayout(normalization_layout)
        layout.addWidget(
            CollapsibleSection("Normalization", normalization_container, self)
        )

        # geometry
        geometry_container = QWidget()
        geometry_layout = QVBoxLayout()
        geometry_layout.setContentsMargins(0, 0, 0, 0)
        geometry_layout.setSpacing(6)
        geometry_layout.addWidget(self.flip_widget)
        geometry_layout.addWidget(self.rotate_widget)
        geometry_container.setLayout(geometry_layout)
        layout.addWidget(CollapsibleSection("Geometry", geometry_container, self))

        # debayer
        bayer_container = QWidget()
        bayer_layout = QVBoxLayout()
        bayer_layout.setContentsMargins(0, 0, 0, 0)
        bayer_layout.setSpacing(6)
        bayer_layout.addWidget(self.debayer_widget)
        bayer_container.setLayout(bayer_layout)
        layout.addWidget(CollapsibleSection("Bayer", bayer_container, self))

        layout.addStretch()

        self.setLayout(layout)

    def _on_sigmoid_params(self, params: dict) -> None:
        gain: float = float(params.get("gain", 10.0))
        cutoff: float = float(params.get("cutoff", 0.5))
        self.controller.apply_operation("SigmoidContrast", gain=gain, cutoff=cutoff)
