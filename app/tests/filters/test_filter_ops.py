import pytest
import torch
from torch import Tensor
from torchvision import transforms as T

from core.filters.clahe import CLAHE
from core.filters.gamma import GammaAdjustment
from core.filters.gaussian import GaussianFilter
from core.filters.histogram_equalization import HistogramEqualization
from core.filters.median import MedianFilter
from core.filters.sigmoid_contrast import SigmoidContrast
from core.filters.unsharp_masking import UnsharpMasking


class TestSigmoidContrast:
    """Unit tests for the SigmoidContrast operation class.

    Tests cover:
    - Parameter validation (type checking)
    - Correct application of the sigmoid contrast operation

    See Also:
        core.filters.sigmoid_contrast.SigmoidContrast: The SigmoidContrast operation class being tested
    """

    def test_parameter_validation(self) -> None:
        with pytest.raises(ValueError):
            SigmoidContrast(gain=-1.0)

        with pytest.raises(ValueError):
            SigmoidContrast(cutoff=1.5)

    def test_apply_operation(self, base_tensor, assert_tensors) -> None:
        entry: Tensor = base_tensor(shape=[1, 3, 4, 4])
        op = SigmoidContrast(cutoff=1.0, gain=10.0)

        result = op(entry)

        expected = self.expected_sigmoid_contrast(entry, gain=10.0, cutoff=1.0)

        assert_tensors(result, expected)

    def expected_sigmoid_contrast(
        self, x: Tensor, gain: float, cutoff: float
    ) -> Tensor:
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


class TestGammaAdjustment:
    """Unit tests for the GammaAdjustment operation class.

    Tests cover:
    - Parameter validation (value constraints)
    - Correct application of the gamma adjustment operation

    See Also:
        core.filters.gamma.GammaAdjustment: The GammaAdjustment operation class being tested
    """

    def test_parameter_validation(self) -> None:
        with pytest.raises(ValueError):
            GammaAdjustment(c=0.0)

        with pytest.raises(ValueError):
            GammaAdjustment(gamma=0.0)

    def test_apply_operation(self, base_tensor, assert_tensors) -> None:
        entry: Tensor = base_tensor(shape=[1, 3, 4, 4])
        op = GammaAdjustment(c=2.0, gamma=0.5)

        result = op(entry)

        expected = (entry.clamp(0.0, 1.0) ** 0.5) * 2.0
        expected = expected.clamp(0.0, 1.0)

        assert_tensors(result, expected)


class TestGaussianFilter:
    """Unit tests for the GaussianFilter operation class.

    Tests cover:
    - Parameter validation (value constraints)
    - Correct application of the Gaussian filter operation

    See Also:
        core.filters.gaussian.GaussianFilter: The GaussianFilter operation class being tested
    """

    def test_parameter_validation(self) -> None:
        with pytest.raises(ValueError):
            GaussianFilter(kernel_size=4)

        with pytest.raises(ValueError):
            GaussianFilter(kernel_size=3, sigma=0.0)

    def test_apply_operation(self, base_tensor, assert_tensors) -> None:
        entry: Tensor = base_tensor(shape=[1, 3, 5, 5])
        op = GaussianFilter(kernel_size=3, sigma=1.0)

        result = op(entry)

        ref = T.GaussianBlur(kernel_size=3, sigma=1.0)
        expected = ref(entry)

        assert_tensors(result, expected)


class TestMedianFilter:
    """Unit tests for the MedianFilter operation class.

    Tests cover:
    - Parameter validation (value constraints)
    - Correct application of the median filter operation

    See Also:
        core.filters.median.MedianFilter: The MedianFilter operation class being tested
    """

    def test_parameter_validation(self) -> None:
        with pytest.raises(ValueError):
            MedianFilter(kernel_size=0)

        with pytest.raises(ValueError):
            MedianFilter(kernel_size=2)

    def test_apply_operation(self, assert_tensors) -> None:
        entry = torch.zeros(1, 1, 5, 5)
        entry[0, 0, 2, 2] = 1.0
        op = MedianFilter(kernel_size=3)

        result = op(entry)

        expected = torch.zeros_like(entry)

        assert_tensors(result, expected)


class TestNotImplementedFilters:
    def test_clahe_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            CLAHE()

    def test_histogram_equalization_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            HistogramEqualization()

    def test_unsharp_masking_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            UnsharpMasking()
