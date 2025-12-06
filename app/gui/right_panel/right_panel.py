from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui.right_panel.widgets.collapsible_section import CollapsibleSection
from gui.right_panel.widgets.color_control import ColorControl
from gui.right_panel.widgets.debayer_control import DebayerControl
from gui.right_panel.widgets.filter_control import FilterControl
from gui.right_panel.widgets.flip_control import FlipControl
from gui.right_panel.widgets.median_filter_control import (
    MedianFilterControl,
)
from gui.right_panel.widgets.minmax_control import MinMaxControl
from gui.right_panel.widgets.minmax_percentile_control import (
    MinMaxPercentileControl,
)
from models.float_param_spec import FloatParamSpec


class RightPanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._setup_ui()

    def _filter_widget(
        self,
        title: str,
        operation_name: str,
        params: list[FloatParamSpec],
    ) -> FilterControl:
        return FilterControl(title, self.controller, operation_name, params, self)

    def _color_widget(self, title: str, operation_name: str) -> ColorControl:
        return ColorControl(title, self.controller, operation_name, self)

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
        self.sigmoid_widget: FilterControl = self._filter_widget(
            "Sigmoid contrast", "SigmoidContrast", sig_params
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
        self.gaussian_widget: FilterControl = self._filter_widget(
            "Gaussian Filter", "GaussianFilter", gaussian_params
        )
        self.gaussian_widget.setToolTip("Apply Gaussian filter to smooth the image")

        # MEDIAN FILTER CONTROL
        self.median_widget = MedianFilterControl(self.controller, self)

        # GAMMA ADJUSTMENT CONTROL
        gamma_params: list[FloatParamSpec] = [
            FloatParamSpec(
                key="c",
                label="C",
                minimum=0.1,
                maximum=2.0,
                step=0.1,
                default=1.0,
            ),
            FloatParamSpec(
                key="gamma",
                label="Gamma",
                minimum=0.1,
                maximum=3.0,
                step=0.1,
                default=1.0,
            ),
        ]
        self.gamma_widget: FilterControl = self._filter_widget(
            "Gamma Adjustment", "GammaAdjustment", gamma_params
        )
        self.gamma_widget.setToolTip("Adjust image brightness using gamma correction")

        # COLOR TO GRAYSCALE CONTROL
        self.color_to_gray_widget: ColorControl = self._color_widget(
            "Color to Grayscale", "ColorToGray"
        )
        self.color_to_gray_widget.setToolTip("Convert color image to grayscale")

        # GRAYSCALE TO COLOR CONTROL
        self.gray_to_color_widget: ColorControl = self._color_widget(
            "Grayscale to Color", "GrayToColor"
        )
        self.gray_to_color_widget.setToolTip("Convert grayscale image to color")

        # RGB TO HSV CONTROL
        self.rgb_to_hsv_widget: ColorControl = self._color_widget(
            "RGB to HSV", "RgbToHsv"
        )

        # HSV TO RGB CONTROL
        self.hsv_to_rgb_widget: ColorControl = self._color_widget(
            "HSV to RGB", "HsvToRgb"
        )

        # MIN-MAX NORMALIZATION CONTROL
        self.minmax_widget = MinMaxControl(self.controller, self)

        # MIN-MAX PERCENTILE NORMALIZATION CONTROL
        self.minmax_percentile_widget = MinMaxPercentileControl(self.controller, self)

        # FLIP CONTROL
        self.flip_widget = FlipControl(self.controller, self)

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
        self.rotate_widget: FilterControl = self._filter_widget(
            "Image Rotation", "Rotate", rotation_params
        )
        self.rotate_widget.setToolTip("Rotate image by a specified angle")

        # REAL TO RGB8 CONTROL
        self.real_to_rgb8_widget: ColorControl = self._color_widget(
            "Real to RGB8", "RealToRGB8"
        )

        # RGB8 TO REAL CONTROL
        self.rgb8_to_real_widget: ColorControl = self._color_widget(
            "RGB8 to Real", "RGB8ToReal"
        )

        # DEBAYER CONTROL
        self.debayer_widget: DebayerControl = DebayerControl(
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
                [
                    self.sigmoid_widget,
                    self.gamma_widget,
                    self.gaussian_widget,
                    self.median_widget,
                ],
            )
        )

        # color
        operations_layout.addWidget(
            self._make_section(
                "Color",
                [
                    self.color_to_gray_widget,
                    self.gray_to_color_widget,
                    self.rgb_to_hsv_widget,
                    self.hsv_to_rgb_widget,
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

        # format
        operations_layout.addWidget(
            self._make_section(
                "Format", [self.real_to_rgb8_widget, self.rgb8_to_real_widget]
            )
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

    def reset_controls_to_default(self) -> None:
        widgets = [
            self.sigmoid_widget,
            self.gaussian_widget,
            self.median_widget,
            self.gamma_widget,
            self.color_to_gray_widget,
            self.gray_to_color_widget,
            self.rgb_to_hsv_widget,
            self.hsv_to_rgb_widget,
            self.minmax_widget,
            self.minmax_percentile_widget,
            self.flip_widget,
            self.rotate_widget,
            self.real_to_rgb8_widget,
            self.rgb8_to_real_widget,
            self.debayer_widget,
        ]
        for w in widgets:
            if hasattr(w, "reset"):
                w.reset()
