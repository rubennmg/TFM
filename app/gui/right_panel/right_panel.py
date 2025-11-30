from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui.right_panel.widgets.collapsible_section import CollapsibleSection
from gui.right_panel.widgets.debayer_control import DebayerControlWidget
from gui.right_panel.widgets.filter_control import FilterControlWidget
from gui.right_panel.widgets.flip_control import FlipControlWidget
from gui.right_panel.widgets.minmax_control import MinMaxControlWidget
from gui.right_panel.widgets.minmax_percentile_control import (
    MinMaxPercentileControlWidget,
)
from gui.right_panel.widgets.median_filter_control import (
    MedianFilterControlWidget,
)
from gui.right_panel.widgets.color_control import ColorControlWidget
from models.float_param_spec import FloatParamSpec


class RightPanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._setup_ui()

    def _make_section(self, title: str, widgets: list[QWidget]) -> CollapsibleSection:
        container = QWidget()
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(6)

        for w in widgets:
            vlayout.addWidget(w)
        container.setLayout(vlayout)

        return CollapsibleSection(title, container, self)

    def _setup_ui(self) -> None:
        layout: QVBoxLayout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(6, 10, 6, 10)

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

        # MEDIAN FILTER CONTROL
        self.median_widget = MedianFilterControlWidget(self.controller, self)

        # COLOR TO GRAYSCALE CONTROL
        self.colorToGray_widget: ColorControlWidget = ColorControlWidget(
            "Color to Grayscale",
            self.controller,
            "ColorToGray",
            self,
        )
        self.colorToGray_widget.setToolTip("Convert color image to grayscale")

        # GRAYSCALE TO COLOR CONTROL
        self.grayToColor_widget: ColorControlWidget = ColorControlWidget(
            "Grayscale to Color",
            self.controller,
            "GrayToColor",
            self,
        )
        self.grayToColor_widget.setToolTip("Convert grayscale image to color")

        # RGB TO HSV CONTROL
        self.rgbToHsv_widget: ColorControlWidget = ColorControlWidget(
            "RGB to HSV",
            self.controller,
            "RgbToHsv",
            self,
        )

        # HSV TO RGB CONTROL
        self.hsvToRgb_widget: ColorControlWidget = ColorControlWidget(
            "HSV to RGB",
            self.controller,
            "HsvToRgb",
            self,
        )

        # MIN-MAX NORMALIZATION CONTROL
        self.minmax_widget = MinMaxControlWidget(self.controller, self)

        # MIN-MAX PERCENTILE NORMALIZATION CONTROL
        self.minmax_percentile_widget = MinMaxPercentileControlWidget(
            self.controller, self
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

        # DEBAYER CONTROL
        self.debayer_widget: DebayerControlWidget = DebayerControlWidget(
            parent=self, controller=self.controller
        )

        # Container widget for scrollable operations
        operations_container = QWidget()
        operations_layout = QVBoxLayout()
        operations_layout.setContentsMargins(0, 0, 10, 0)
        operations_layout.setSpacing(8)

        # filters
        operations_layout.addWidget(
            self._make_section(
                "Filters",
                [self.sigmoid_widget, self.gaussian_widget, self.median_widget],
            )
        )

        # color
        operations_layout.addWidget(
            self._make_section(
                "Color",
                [
                    self.colorToGray_widget,
                    self.grayToColor_widget,
                    self.rgbToHsv_widget,
                    self.hsvToRgb_widget,
                ],
            )
        )

        # normalization
        operations_layout.addWidget(
            self._make_section(
                "Normalization",
                [self.minmax_widget, self.minmax_percentile_widget],
            )
        )

        # geometry
        operations_layout.addWidget(
            self._make_section("Geometry", [self.flip_widget, self.rotate_widget])
        )

        # debayer
        operations_layout.addWidget(self._make_section("Bayer", [self.debayer_widget]))

        operations_layout.addStretch()
        operations_container.setLayout(operations_layout)

        scroll_area = QScrollArea()
        scroll_area.setObjectName("operationsScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_area.setWidget(operations_container)

        layout.addWidget(label_title)
        layout.addWidget(scroll_area)

        self.setLayout(layout)
