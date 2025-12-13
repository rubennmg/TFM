import pytest
import torch
from torch import Size, Tensor


@pytest.fixture
def base_tensor():
    """Returns an image tensor with shape [B, C, H, W] and given dtype."""

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
    """Checks given tensors for shape, dtype and value equality within a tolerance."""

    def _compare(a: Tensor, b: Tensor, atol=1e-4):
        assert isinstance(a, Tensor)
        assert isinstance(b, Tensor)
        assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
        assert a.dtype == b.dtype, f"Dtype mismatch: {a.dtype} vs {b.dtype}"
        assert torch.allclose(a, b, atol=atol), f"Tensors differ more than {atol}"

    return _compare


@pytest.fixture
def assert_channels():
    """Checks that an image tensor with shape [B, C, H, W] has the expected number of channels."""

    def _assert_channels(tensor: Tensor, expected_channels: int):
        assert tensor.dim() == 4, f"Expected a 4D tensor, got {tensor.dim()}D tensor."
        actual_channels = tensor.size(1)
        assert actual_channels == expected_channels, (
            f"Expected {expected_channels} channels, got {actual_channels} channels."
        )

    return _assert_channels
