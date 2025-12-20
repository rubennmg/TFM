from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from models.float_param_spec import FloatParamSpec


@dataclass(frozen=True)
class OperationDefinition:
    name: str
    label: str
    category: str
    control_type: str
    params: Optional[List[FloatParamSpec]] = None
    default_params: Mapping[str, object] = field(default_factory=dict)


def _build_filter_params(specs: List[FloatParamSpec]) -> Dict[str, float]:
    return {spec.key: spec.default for spec in specs}


_FILTER_SIGMOID_PARAMS = [
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
        minimum=-1.0,
        maximum=1.0,
        step=0.01,
        default=0.0,
    ),
]

_FILTER_GAUSSIAN_PARAMS = [
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

_FILTER_GAMMA_PARAMS = [
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

_FILTER_ROTATE_PARAMS = [
    FloatParamSpec(
        key="angle",
        label="Angle",
        minimum=-360.0,
        maximum=360.0,
        step=1.0,
        default=0.0,
    )
]

_FILTER_CLAHE_PARAMS = [
    FloatParamSpec(
        key="clip_limit",
        label="Clip Limit",
        minimum=1.0,
        maximum=100.0,
        step=1.0,
        default=40.0,
    ),
    FloatParamSpec(
        key="grid_size",
        label="Grid Size",
        minimum=2,
        maximum=32,
        step=2,
        default=8,
    ),
]

_FILTER_UNSHARP_MASKING_PARAMS = [
    FloatParamSpec(
        key="kernel_size",
        label="Kernel Size",
        minimum=1,
        maximum=27,
        step=2,
        default=3,
    ),
    FloatParamSpec(
        key="sigma",
        label="Sigma",
        minimum=0.1,
        maximum=10.0,
        step=0.1,
        default=1.5,
    ),
]


_OPERATIONS: List[OperationDefinition] = [
    OperationDefinition(
        name="SigmoidContrast",
        label="Sigmoid Contrast",
        category="Filters",
        control_type="filter",
        params=_FILTER_SIGMOID_PARAMS,
        default_params=_build_filter_params(_FILTER_SIGMOID_PARAMS),
    ),
    OperationDefinition(
        name="GaussianFilter",
        label="Gaussian Filter",
        category="Filters",
        control_type="filter",
        params=_FILTER_GAUSSIAN_PARAMS,
        default_params=_build_filter_params(_FILTER_GAUSSIAN_PARAMS),
    ),
    OperationDefinition(
        name="MedianFilter",
        label="Median Filter",
        category="Filters",
        control_type="median",
        default_params={"kernel_size": 3},
    ),
    OperationDefinition(
        name="GammaAdjustment",
        label="Gamma Adjustment",
        category="Filters",
        control_type="filter",
        params=_FILTER_GAMMA_PARAMS,
        default_params=_build_filter_params(_FILTER_GAMMA_PARAMS),
    ),
    OperationDefinition(
        name="ColorToGray",
        label="Color to Grayscale",
        category="Color",
        control_type="no_param",
    ),
    OperationDefinition(
        name="GrayToColor",
        label="Grayscale to Color",
        category="Color",
        control_type="no_param",
    ),
    OperationDefinition(
        name="RgbToHsv",
        label="RGB to HSV",
        category="Color",
        control_type="no_param",
    ),
    OperationDefinition(
        name="HsvToRgb",
        label="HSV to RGB",
        category="Color",
        control_type="no_param",
    ),
    OperationDefinition(
        name="MinMaxNormalization",
        label="Min-Max Normalization",
        category="Normalization",
        control_type="no_param",
    ),
    OperationDefinition(
        name="MinMaxPercentileNormalization",
        label="Min-Max Percentile",
        category="Normalization",
        control_type="minmax_percentile",
        default_params={"lower_percentile": 0.02, "upper_percentile": 0.98},
    ),
    OperationDefinition(
        name="Flip",
        label="Flip",
        category="Geometry",
        control_type="flip",
        default_params={"horizontal": True},
    ),
    OperationDefinition(
        name="Rotate",
        label="Rotate",
        category="Geometry",
        control_type="filter",
        params=_FILTER_ROTATE_PARAMS,
        default_params=_build_filter_params(_FILTER_ROTATE_PARAMS),
    ),
    OperationDefinition(
        name="RealToRGB8",
        label="Real to RGB8",
        category="Format",
        control_type="no_param",
    ),
    OperationDefinition(
        name="RGB8ToReal",
        label="RGB8 to Real",
        category="Format",
        control_type="no_param",
    ),
    OperationDefinition(
        name="Debayer",
        label="Debayer",
        category="Bayer",
        control_type="debayer",
        default_params={"algorithm_name": "debayer2x2"},
    ),
    OperationDefinition(
        name="HistogramEqualization",
        label="Histogram Equalization",
        category="Filters",
        control_type="no_param",
    ),
    OperationDefinition(
        name="CLAHE",
        label="CLAHE",
        category="Filters",
        control_type="filter",
        params=_FILTER_CLAHE_PARAMS,
        default_params=_build_filter_params(_FILTER_CLAHE_PARAMS),
    ),
    OperationDefinition(
        name="UnsharpMasking",
        label="Unsharp Masking",
        category="Filters",
        control_type="filter",
        params=_FILTER_UNSHARP_MASKING_PARAMS,
        default_params=_build_filter_params(_FILTER_UNSHARP_MASKING_PARAMS),
    ),
]


def get_operation_definitions() -> List[OperationDefinition]:
    return list(_OPERATIONS)


def get_operation_definition(name: str) -> OperationDefinition | None:
    for op in _OPERATIONS:
        if op.name == name:
            return op
    return None


def get_operation_choices_by_category() -> Dict[str, List[OperationDefinition]]:
    categories: Dict[str, List[OperationDefinition]] = {}
    for op in _OPERATIONS:
        categories.setdefault(op.category, []).append(op)
    for definitions in categories.values():
        definitions.sort(key=lambda entry: entry.label)
    return categories
