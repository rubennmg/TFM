import pytest
import torch
from torch import Tensor
from torchvision import transforms as T
from torchvision.transforms import functional as F
from kornia import enhance as K_e
from kornia import filters as K_f

from core.filters.affine_intensity_transformation import AffineIntensityTransformation
from core.filters.brightness_adjustement import BrightnessAdjustment
from core.filters.clahe import CLAHE
from core.filters.gamma import GammaAdjustment
from core.filters.gaussian import GaussianFilter
from core.filters.histogram_equalization import HistogramEqualization
from core.filters.light_compensation import LightCompensation
from core.filters.mean_contrast_adjustment import MeanContrastAdjustment
from core.filters.median import MedianFilter
from core.filters.sigmoid_contrast import SigmoidContrast
from core.filters.unsharp_masking import UnsharpMasking
from core.filters.white_balance import WhiteBalance


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
        entry: Tensor = base_tensor(shape=[3, 4, 4])
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
        entry: Tensor = base_tensor(shape=[3, 4, 4])
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
            GaussianFilter(sigma=0.0)

    def test_apply_operation(self, base_tensor, assert_tensors) -> None:
        entry: Tensor = base_tensor(shape=[3, 5, 5])
        op = GaussianFilter(sigma=1.0)

        result = op(entry)

        ref = T.GaussianBlur(kernel_size=7, sigma=1.0)
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
        entry = torch.zeros(1, 5, 5)
        entry[0, 2, 2] = 1.0
        op = MedianFilter(kernel_size=3)

        result = op(entry)

        expected = torch.zeros_like(entry)

        assert_tensors(result, expected)


class TestHistogramEqualization:
    """Unit tests for the HistogramEqualization operation class.

    Tests cover:
    - Correct application of the histogram equalization operation

    See Also:
        core.filters.histogram_equalization.HistogramEqualization: The HistogramEqualization operation class being tested
    """

    def test_apply_operation(self, base_tensor, assert_tensors) -> None:
        entry: Tensor = base_tensor(shape=[3, 4, 4])
        op = HistogramEqualization()

        result = op(entry)

        entry_rgb8 = F.convert_image_dtype(entry, dtype=torch.uint8)
        expected = F.equalize(entry_rgb8)
        expected = F.convert_image_dtype(expected, dtype=entry.dtype)

        assert_tensors(result, expected, atol=1e-2)


class TestCLAHE:
    """Unit tests for the CLAHE operation class.

    Tests cover:
    - Parameter validation (value constraints)
    - Correct application of the CLAHE operation

    See Also:
        core.filters.clahe.CLAHE: The CLAHE operation class being tested
    """

    def test_parameter_validation(self) -> None:
        with pytest.raises(ValueError):
            CLAHE(clip_limit=0.0)

        with pytest.raises(ValueError):
            CLAHE(grid_size=0)

    def test_apply_operation(self, base_tensor, assert_tensors) -> None:
        entry: Tensor = base_tensor(shape=[3, 16, 16])
        op = CLAHE(clip_limit=2.0, grid_size=8)

        result = op(entry)
        expected = K_e.equalize_clahe(entry, clip_limit=2.0, grid_size=(8, 8))

        assert_tensors(result, expected)


class TestUnsharpMasking:
    """Unit tests for the UnsharpMasking operation class.

    Tests cover:
    - Parameter validation (value constraints)
    - Correct application of the unsharp masking operation

    See Also:
        core.filters.unsharp_masking.UnsharpMasking: The UnsharpMasking operation class being tested
    """

    def test_parameter_validation(self) -> None:
        with pytest.raises(ValueError):
            UnsharpMasking(kernel_size=4)

        with pytest.raises(ValueError):
            UnsharpMasking(kernel_size=3, sigma=0.0)

    def test_apply_operation(self, base_tensor, assert_tensors) -> None:
        entry: Tensor = base_tensor(shape=[3, 5, 5])
        op = UnsharpMasking(kernel_size=3, sigma=1.0)

        result = op(entry)
        expected = K_f.unsharp_mask(
            entry.unsqueeze(0), kernel_size=3, sigma=(1.0, 1.0)
        ).squeeze(0)

        assert_tensors(result, expected)


class TestAffineIntensityTransformation:
    """Unit tests for the AffineIntensityTransformation operation class.

    Tests cover:
    - Parameter validation (type and value constraints)
    - Correct application of the affine intensity transformation operation

    See Also:
        core.filters.affine_intensity_transformation.AffineIntensityTransformation: The AffineIntensityTransformation operation class being tested
    """

    def test_parameter_validation(self) -> None:
        with pytest.raises(TypeError):
            AffineIntensityTransformation(gain="1.0")

        with pytest.raises(ValueError):
            AffineIntensityTransformation(gain=-1.0)

        with pytest.raises(TypeError):
            AffineIntensityTransformation(bias="0.0")

    def test_apply_operation(self, base_tensor, assert_tensors) -> None:
        entry: Tensor = base_tensor(shape=[3, 4, 4])
        op = AffineIntensityTransformation(gain=1.5, bias=0.1)

        result = op(entry)
        expected = entry * 1.5 + 0.1

        assert_tensors(result, expected)


class TestBrightnessAdjustment:
    """Unit tests for the BrightnessAdjustment operation class.

    Tests cover:
    - Parameter validation (type and value constraints)
    - Correct application of the brightness adjustment operation

    See Also:
        core.filters.brightness_adjustment.BrightnessAdjustment: The BrightnessAdjustment operation class being tested
    """

    def test_parameter_validation(self) -> None:
        with pytest.raises(TypeError):
            BrightnessAdjustment(alfa="1.0")

        with pytest.raises(ValueError):
            BrightnessAdjustment(alfa=-1.0)

    def test_apply_operation(self, base_tensor, assert_tensors) -> None:
        entry: Tensor = base_tensor(shape=[3, 4, 4])
        op = BrightnessAdjustment(alfa=1.25)

        result = op(entry)
        expected = entry * 1.25

        assert_tensors(result, expected)


class TestLightCompensation:
    """Unit tests for the LightCompensation operation class.

    Tests cover:
    - Parameter validation (type and value constraints)
    - Correct application of the light compensation operation

    See Also:
        core.filters.light_compensation.LightCompensation: The LightCompensation operation class being tested
    """

    def test_parameter_validation(self) -> None:
        with pytest.raises(ValueError):
            LightCompensation(strength=-0.1)

        with pytest.raises(ValueError):
            LightCompensation(light_gain_compensation=torch.ones(1, 2, 3))

    def test_apply_operation(self, base_tensor, assert_tensors) -> None:
        entry: Tensor = base_tensor(shape=[3, 2, 2])
        gain = torch.tensor([[1.0, 1.2], [0.8, 1.5]], dtype=entry.dtype)
        op = LightCompensation(light_gain_compensation=gain, strength=0.5)

        result = op(entry)
        effective_gain = 1 + (gain - 1) * 0.5
        expected = entry * effective_gain.unsqueeze(0)

        assert_tensors(result, expected)


class TestMeanContrastAdjustment:
    """Unit tests for the MeanContrastAdjustment operation class.

    Tests cover:
    - Parameter validation (type and value constraints)
    - Correct application of the mean contrast adjustment operation

    See Also:
        core.filters.mean_contrast_adjustment.MeanContrastAdjustment: The MeanContrastAdjustment operation class being tested
    """

    def test_parameter_validation(self) -> None:
        with pytest.raises(TypeError):
            MeanContrastAdjustment(beta="1.0")

        with pytest.raises(ValueError):
            MeanContrastAdjustment(beta=-1.0)

    def test_apply_operation(self, base_tensor, assert_tensors) -> None:
        entry: Tensor = base_tensor(shape=[3, 4, 4])
        op = MeanContrastAdjustment(beta=1.5)

        result = op(entry)
        mean = entry.mean()
        expected = (entry - mean) * 1.5 + mean

        assert_tensors(result, expected)


class TestWhiteBalance:
    """Unit tests for the WhiteBalance operation class.

    Tests cover:
    - Parameter validation (type and value constraints)
    - Correct application of the white balance operation

    See Also:
        core.filters.white_balance.WhiteBalance: The WhiteBalance operation class being tested
    """

    def test_parameter_validation(self) -> None:
        with pytest.raises(ValueError):
            WhiteBalance(method="unsupported")

    def test_apply_gray_world_operation(self, assert_tensors) -> None:
        entry = torch.tensor(
            [
                [[0.2, 0.4], [0.6, 0.8]],
                [[0.3, 0.5], [0.7, 0.9]],
                [[0.4, 0.6], [0.8, 1.0]],
            ],
            dtype=torch.float32,
        )
        op = WhiteBalance(method="gray_world")

        result = op(entry.clone())

        r_mean = entry[0].mean()
        g_mean = entry[1].mean()
        b_mean = entry[2].mean()
        overall_mean = (r_mean + g_mean + b_mean) / 3.0
        expected = entry.clone()
        expected[0] *= overall_mean / (r_mean + op.eps)
        expected[1] *= overall_mean / (g_mean + op.eps)
        expected[2] *= overall_mean / (b_mean + op.eps)

        assert_tensors(result, expected)

    def test_apply_max_rgb_operation(self, assert_tensors) -> None:
        entry = torch.tensor(
            [
                [[0.2, 0.4], [0.6, 0.8]],
                [[0.3, 0.5], [0.7, 0.9]],
                [[0.4, 0.6], [0.8, 1.0]],
            ],
            dtype=torch.float32,
        )
        op = WhiteBalance(method="max_rgb")

        result = op(entry.clone())

        r_max = entry[0].max()
        g_max = entry[1].max()
        b_max = entry[2].max()
        overall_max = torch.max(torch.max(r_max, g_max), b_max)
        expected = entry.clone()
        expected[0] *= overall_max / (r_max + op.eps)
        expected[1] *= overall_max / (g_max + op.eps)
        expected[2] *= overall_max / (b_max + op.eps)

        assert_tensors(result, expected)
