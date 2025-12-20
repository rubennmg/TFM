import pytest
from torch import Tensor
from torchvision.transforms.v2 import functional as F

from core.format.real_to_rgb8 import RealToRGB8
from core.format.rgb8_to_real import RGB8ToReal


class TestRealToRGB8:
    """Unit tests for the RealToRGB8 operation class.

    Tests cover:
    - Input type validation (requires real-valued input)
    - Correct application of the conversion operation

    See Also:
        core.format.real_to_rgb8.RealToRGB8: The RealToRGB8 operation class being tested
    """

    def test_requires_real_input(self, base_tensor):
        entry: Tensor = base_tensor(shape=[1, 3, 4, 4], dtype="int")
        op = RealToRGB8()

        with pytest.raises(ValueError):
            op(entry)

    def test_apply_operation(self, base_tensor, assert_tensors):
        entry: Tensor = base_tensor(shape=[1, 3, 4, 4], dtype="float")
        op = RealToRGB8()

        result = op(entry)

        expected = F.convert_image_dtype(entry, dtype=result.dtype)

        assert result.dtype == expected.dtype
        assert_tensors(result, expected)


class TestRGB8ToReal:
    """Unit tests for the RGB8ToReal operation class.

    Tests cover:
    - Input type validation (requires integer-valued input)
    - Correct application of the conversion operation

    See Also:
        core.format.rgb8_to_real.RGB8ToReal: The RGB8ToReal operation class being tested
    """

    def test_requires_integer_input(self, base_tensor):
        entry: Tensor = base_tensor(shape=[1, 3, 4, 4], dtype="float")
        op = RGB8ToReal()

        with pytest.raises(ValueError):
            op(entry)

    def test_apply_operation(self, base_tensor, assert_tensors):
        entry: Tensor = base_tensor(shape=[1, 3, 4, 4], dtype="int")
        op = RGB8ToReal()

        result = op(entry)

        expected = F.convert_image_dtype(entry, dtype=result.dtype)

        assert result.is_floating_point()
        assert_tensors(result, expected)
