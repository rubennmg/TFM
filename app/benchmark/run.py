from typing import Callable

import pytest
import torch
from torch import Tensor, dtype

_SHAPES = [
    (1, 3, 256, 256),
    (1, 3, 512, 512),
    (1, 3, 1024, 1024),
]

_DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

_DTYPES = [torch.float16, torch.float32]


def _shape_id(shape: tuple[int, int, int, int]) -> str:
    return f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}"


def _dtype_id(dtype: dtype) -> str:
    return str(dtype).replace("torch.", "")


@pytest.mark.benchmark(group="pipeline")
@pytest.mark.parametrize("dtype", _DTYPES, ids=_dtype_id)
@pytest.mark.parametrize(
    "shape",
    _SHAPES,
    ids=_shape_id,
)
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICES)
def test_benchmark_run(
    benchmark: Callable,
    synthetic_tensor: Callable,
    pipeline_factory: Callable[[str], Callable[[Tensor], Tensor]],
    device: str,
    shape: tuple[int, int, int, int],
    dtype: dtype,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    pipeline: Callable[[Tensor], Tensor] = pipeline_factory(device)

    x: Tensor = synthetic_tensor(shape=shape, dtype=dtype)

    benchmark(pipeline, x)
