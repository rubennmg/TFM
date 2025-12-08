import pytest
import torch
from torch import Size


@pytest.fixture
def base_tensor():
    def _factory(shape: Size = Size([1, 3, 4, 4]), dtype: str = "float"):
        numel = 1
        for dim in shape:
            numel *= dim

        if dtype == "float":
            base = torch.linspace(0.0, 1.0, steps=numel, dtype=torch.float32)
            return base.view(*shape)
        elif dtype == "int":
            base = (torch.arange(numel, dtype=torch.int32) % 256).to(torch.uint8)
            return base.view(*shape)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    return _factory


@pytest.fixture
def assert_tensors():
    def _compare(a, b, atol=1e-4):
        assert isinstance(a, torch.Tensor)
        assert isinstance(b, torch.Tensor)
        assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
        assert a.dtype == b.dtype, f"Dtype mismatch: {a.dtype} vs {b.dtype}"
        assert torch.allclose(a, b, atol=atol), f"Tensors differ more than {atol}"

    return _compare
