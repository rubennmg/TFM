import json
from typing import Any, Callable

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

    path = config.getoption("--bench-profile")
    if path:
        profile = _load_bench_profile(path)
        config._bench_profile_cache = profile
        _print_pipeline(config, profile, path)


@pytest.fixture(scope="session")
def bench_profile(pytestconfig) -> list[dict[str, Any]]:
    cached = getattr(pytestconfig, "_bench_profile_cache", None)
    if cached is not None:
        return cached

    path = pytestconfig.getoption("--bench-profile")
    if path is None:
        pytest.exit("You must specify --bench-profile=FILE.json")

    profile = _load_bench_profile(path)
    pytestconfig._bench_profile_cache = profile
    return profile


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


def _load_bench_profile(path: str) -> list[dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except OSError as exc:
        pytest.exit(f"Could not read bench profile '{path}': {exc}")


def _print_pipeline(config, profile: list[dict[str, Any]], path: str) -> None:
    lines = ["Benchmark pipeline:", f"  Profile file: {path}"]

    for idx, entry in enumerate(profile, 1):
        name = entry.get("operation", "<unknown>")
        params = entry.get("params") or {}

        if params:
            params_repr = json.dumps(params, sort_keys=True)
            lines.append(f"  {idx}. {name} {params_repr}")
        else:
            lines.append(f"  {idx}. {name}")

    reporter = config.pluginmanager.get_plugin("terminalreporter")
    writer = reporter.write_line if reporter else print
    for line in lines:
        writer(line)
