import pytest
import torch

from core.normalization.minmax import MinMaxNormalization
from core.normalization.minmax_percentile import MinMaxPercentileNormalization


class TestMinMaxNormalization:
    """Unit tests for the MinMaxNormalization operation class.

    Tests cover:
    - Basic functionality with expected values
    - Edge case: constant tensor returns zero tensor

    See Also:
        core.normalization.minmax.MinMaxNormalization: The MinMaxNormalization operation class being tested
    """

    def test_expected_values(self, assert_tensors):
        channel_0 = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        channel_1 = torch.tensor([[5.0, 5.0], [5.0, 10.0]])

        entry = torch.stack([channel_0, channel_1]).unsqueeze(0)

        op = MinMaxNormalization()
        eps = op.eps

        result = op(entry)

        x_min = entry.amin(dim=(2, 3), keepdim=True)
        x_max = entry.amax(dim=(2, 3), keepdim=True)

        expected = (entry - x_min) / (x_max - x_min + eps)

        assert_tensors(result, expected)

    def test_constant_tensor_returns_zero(self, assert_tensors):
        entry = torch.ones(1, 1, 4, 4)
        op = MinMaxNormalization()

        result = op(entry)

        expected = torch.zeros_like(entry)

        assert_tensors(result, expected)


class TestMinMaxPercentileNormalization:
    """Unit tests for the MinMaxPercentileNormalization operation class.

    Tests cover:
    - Parameter validation (invalid percentile values)
    - Basic functionality with expected percentile-based normalization

    See Also:
        core.normalization.minmax_percentile.MinMaxPercentileNormalization: The MinMaxPercentileNormalization operation class being tested
    """

    def test_parameter_validation(self):
        with pytest.raises(ValueError):
            MinMaxPercentileNormalization(lower_percentile=0.5, upper_percentile=0.5)

        with pytest.raises(ValueError):
            MinMaxPercentileNormalization(lower_percentile=-0.1, upper_percentile=0.9)

    def test_percentile_values(self, assert_tensors):
        entry = torch.tensor([[[[0.0, 1.0], [2.0, 100.0]]]])

        op = MinMaxPercentileNormalization(lower_percentile=0.25, upper_percentile=0.75)
        eps = op.eps

        result = op(entry)

        flat = entry.view(1, 1, -1)

        lower = torch.quantile(flat, 0.25, dim=2, keepdim=True)
        upper = torch.quantile(flat, 0.75, dim=2, keepdim=True)

        expected = torch.clamp((flat - lower) / (upper - lower + eps), 0.0, 1.0)
        expected = expected.view_as(entry)

        assert_tensors(result, expected)
