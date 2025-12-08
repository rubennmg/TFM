import pytest
import torch

from core.filters.sigmoid_contrast import SigmoidContrast


class TestSigmoidContrast:
    def test_instance(self):
        op = SigmoidContrast()

        assert isinstance(op, SigmoidContrast)

    def test_parameters(self):
        cutoff = 0.4
        gain = 5.0

        op = SigmoidContrast(cutoff=cutoff, gain=gain)

        assert isinstance(op.cutoff, float)
        assert op.cutoff == cutoff

        assert isinstance(op.gain, float)
        assert op.gain == gain

    def test_default_params(self, base_tensor, assert_tensors):
        entry_float = base_tensor(shape=[1, 3, 4, 4], dtype="float")
        op = SigmoidContrast()
        result_float = op(entry_float)

        assert_tensors(entry_float, result_float)

        entry_int = base_tensor(shape=[1, 3, 4, 4], dtype="int")
        op = SigmoidContrast()
        result_int = op(entry_int)

        assert_tensors(entry_int, result_int)

    @pytest.mark.parametrize(
        "gain,cutoff",
        [
            (10.0, 0.3),
            (5.0, 0.5),
            (12.0, -0.5),
        ],
    )
    def test_custom_params(self, base_tensor, assert_tensors, gain, cutoff):
        entry_float = base_tensor(shape=[1, 3, 4, 4], dtype="float")
        op = SigmoidContrast(gain=gain, cutoff=cutoff)
        result_float = op(entry_float)

        expected = expected_sigmoid_contrast(entry_float, gain=gain, cutoff=cutoff)

        assert_tensors(result_float, expected)


def expected_sigmoid_contrast(x: torch.Tensor, gain: float, cutoff: float):
    if gain == 0:
        return x

    if cutoff < 0:
        estimated = x.mean()
        t_cutoff = torch.clamp(estimated, 0.4, 0.6)
    else:
        t_cutoff = x.new_tensor(cutoff, dtype=x.dtype, device=x.device)

    sigmoid_min = torch.sigmoid(-gain * t_cutoff)
    sigmoid_max = torch.sigmoid(gain * (1 - t_cutoff))

    imgf = torch.sigmoid(gain * (x - t_cutoff))
    imgf = (imgf - sigmoid_min) / (sigmoid_max - sigmoid_min)

    return imgf
