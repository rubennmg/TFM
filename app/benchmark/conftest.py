import json
from typing import Callable

import pytest
import torch
from torch import Tensor, dtype

from core.image_operation import ImageOperation
from core.registry import OPERATION_REGISTRY


def pytest_addoption(parser):
    parser.addoption("--bench-profile", action="store", default=None)


def pytest_configure(config):
    config.option.benchmark_timer_unit = "ms"
    config.option.benchmark_name = "short"
    config.option.benchmark_sort = "fullname"


@pytest.fixture(scope="session")
def bench_profile(pytestconfig) -> dict:
    path = pytestconfig.getoption("--bench-profile")
    if path is None:
        pytest.exit("You must specify --bench-profile=FILE.json")
    with open(path, "r") as f:
        return json.load(f)


@pytest.fixture
def synthetic_tensor():
    def _factory(
        shape: tuple[int, int, int, int] = (1, 3, 256, 256),
        dtype: dtype = torch.float32,
    ) -> Tensor:
        if dtype.is_floating_point:
            return torch.rand(shape, dtype=dtype)
        else:
            return torch.randint(0, 255, shape, dtype=dtype)

    return _factory


@pytest.fixture
def pipeline_factory(bench_profile):
    def _factory(device_str: str) -> Callable:
        ops: list[ImageOperation] = []

        for entry in bench_profile:
            name = entry["operation"]
            params = entry.get("params", {})

            cls = OPERATION_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"Operation '{name}' not found in registry")

            ops.append(cls(**params))

        device = torch.device(device_str)

        def _run(x: Tensor) -> Tensor:
            x = x.to(device)
            for op in ops:
                x = op(x)
            return x

        return _run

    return _factory
