import pytest
import torch
from torch import Tensor

from core.color.color_to_gray import ColorToGray
from core.color.gray_to_color import GrayToColor
from core.color.hsv_to_rgb import HsvToRgb
from core.color.rgb_to_hsv import RgbToHsv


class TestColorToGray:
    """Unit tests for the ColorToGray operation class.

    Tests cover:
    - Input validation (requires three-channel input)
    - Correct application of the color to grayscale conversion operation

    See Also:
        core.color.color_to_gray.ColorToGray: The ColorToGray operation class being tested
    """

    def test_requires_three_channels(self, base_tensor) -> None:
        grayscale: Tensor = base_tensor(shape=[1, 1, 4, 4])
        op = ColorToGray()

        with pytest.raises(ValueError):
            op(grayscale)

    def test_apply_operation(
        self, base_tensor, assert_tensors, assert_channels
    ) -> None:
        rgb: Tensor = base_tensor(shape=[1, 3, 4, 4])
        op = ColorToGray()

        result = op(rgb)

        weights = rgb.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        expected = (rgb * weights).sum(dim=1, keepdim=True)

        assert_channels(result, expected_channels=1)
        assert_tensors(result, expected)


class TestGrayToColor:
    """Unit tests for the GrayToColor operation class.

    Tests cover:
    - Input validation (requires single-channel input)
    - Invalid mode handling
    - Correct application of the grayscale to color conversion operation for 'repeat' and 'heat'

    See Also:
        core.color.gray_to_color.GrayToColor: The GrayToColor operation class being tested
    """

    def test_requires_single_channel(self, base_tensor) -> None:
        rgb: Tensor = base_tensor(shape=[1, 3, 4, 4])
        op = GrayToColor(mode="repeat")

        with pytest.raises(ValueError):
            op(rgb)

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError):
            GrayToColor(mode="invalid")

    def test_repeat_mode(self, base_tensor, assert_tensors, assert_channels) -> None:
        grayscale: Tensor = base_tensor(shape=[1, 1, 4, 4])
        op = GrayToColor(mode="repeat")

        result = op(grayscale)

        expected = grayscale.repeat(1, 3, 1, 1)

        assert_tensors(result, expected)
        assert_channels(result, expected_channels=3)

    def test_heat_mode(self, base_tensor, assert_tensors, assert_channels) -> None:
        grayscale: Tensor = base_tensor(shape=[1, 1, 4, 4])
        op = GrayToColor(mode="heat")

        result = op(grayscale)

        xmin = grayscale.amin(dim=(2, 3), keepdim=True)
        xmax = grayscale.amax(dim=(2, 3), keepdim=True)
        v = (grayscale - xmin) / (xmax - xmin + 1e-8)

        expected = torch.cat([v, v.sqrt(), torch.zeros_like(v)], dim=1)

        assert_tensors(result, expected)
        assert_channels(result, expected_channels=3)


class TestColorSpaceConversions:
    """Unit tests for color space conversion operations.

    Tests cover:
    - RGB to HSV conversion
    - HSV to RGB conversion
    - Roundtrip conversions (RGB -> HSV -> RGB and HSV -> RGB -> HSV)

    See Also:
        core.color.rgb_to_hsv.RgbToHsv: The RgbToHsv operation class being tested
        core.color.hsv_to_rgb.HsvToRgb: The HsvToRgb operation class being tested
    """

    def test_rgb_to_hsv(self, assert_tensors) -> None:
        rgb = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]])
        op = RgbToHsv()

        hsv = op(rgb)

        expected = torch.tensor([[[[0.0]], [[1.0]], [[1.0]]]], dtype=rgb.dtype)

        assert_tensors(hsv, expected)

    def test_hsv_to_rgb(self, assert_tensors) -> None:
        hsv = torch.tensor([[[[1.0 / 3.0]], [[1.0]], [[1.0]]]])
        op = HsvToRgb()

        rgb = op(hsv)

        expected = torch.tensor([[[[0.0]], [[1.0]], [[0.0]]]], dtype=hsv.dtype)

        assert_tensors(rgb, expected)

    def test_roundtrip_rgb_hsv(self, base_tensor, assert_tensors) -> None:
        rgb: Tensor = base_tensor(shape=[1, 3, 4, 4])
        to_hsv = RgbToHsv()
        to_rgb = HsvToRgb()

        hsv = to_hsv(rgb)
        recon = to_rgb(hsv)

        assert_tensors(rgb, recon, atol=1e-3)

    def test_roundtrip_hsv_rgb(self, base_tensor, assert_tensors) -> None:
        hsv: Tensor = base_tensor(shape=[1, 3, 4, 4])
        to_rgb = HsvToRgb()
        to_hsv = RgbToHsv()

        rgb = to_rgb(hsv)
        recon = to_hsv(rgb)

        assert_tensors(hsv, recon, atol=1e-3)
