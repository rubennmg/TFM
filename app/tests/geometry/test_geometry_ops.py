import pytest
from torch import Tensor
from torchvision.transforms.functional import rotate as tv_rotate

from core._tensor_utils import HEIGHT_DIM, WIDTH_DIM
from core.geometry.flip import Flip
from core.geometry.rotate import Rotate


class TestFlip:
    """Unit tests for the Flip operation class

    Tests cover:
    - Parameter validation (type checking)
    - Horizontal flip operation
    - Vertical flip operation
    - Edge case: double flip returns original image

    See Also:
        core.geometry.flip.Flip: The Flip operation class being tested
    """

    def test_parameter_validation(self):
        with pytest.raises(TypeError):
            Flip(horizontal="yes")  # type: ignore

    def test_horizontal_flip(self, base_tensor, assert_tensors):
        entry: Tensor = base_tensor(shape=[1, 1, 4, 4])
        op = Flip(horizontal=True)

        result = op(entry)
        expected = entry.flip(dims=[WIDTH_DIM])

        assert_tensors(result, expected)

    def test_vertical_flip(self, base_tensor, assert_tensors):
        entry: Tensor = base_tensor(shape=[1, 1, 4, 4])
        op = Flip(horizontal=False)

        result = op(entry)
        expected = entry.flip(dims=[HEIGHT_DIM])
        assert_tensors(result, expected)

    def test_no_flip(self, base_tensor, assert_tensors):
        entry: Tensor = base_tensor(shape=[1, 1, 4, 4])
        op = Flip(horizontal=True)

        result = op(op(entry))
        expected = entry
        assert_tensors(result, expected)


class TestRotate:
    """
    Unit tests for the Rotate operation class.

    Tests cover:
    - Parameter validation (type checking and value constraints)
    - Basic rotation operation with arbitrary angles
    - Edge case: zero rotation (no change expected)
    - Edge case: full 360° rotation (returns to original state)

    See Also:
        core.geometry.rotate.Rotate: The Rotate operation class being tested
    """

    def test_parameter_validation(self):
        with pytest.raises(TypeError):
            Rotate(angle="90")  # type: ignore

        with pytest.raises(ValueError):
            Rotate(angle=720)

    def test_apply_operation(self, base_tensor, assert_tensors):
        entry: Tensor = base_tensor(shape=[1, 3, 4, 4])
        angle = 30.0
        op = Rotate(angle=angle)

        result = op(entry)
        expected = tv_rotate(entry, angle=angle, expand=False)

        assert_tensors(result, expected, atol=1e-5)

    def test_no_rotation(self, base_tensor, assert_tensors):
        entry: Tensor = base_tensor(shape=[1, 3, 4, 4])
        op = Rotate(angle=0)

        result = op(entry)
        expected = entry

        assert_tensors(result, expected)

    def test_full_rotation(self, base_tensor, assert_tensors):
        entry: Tensor = base_tensor(shape=[1, 3, 4, 4])
        op = Rotate(angle=360)

        result = op(entry)
        expected = entry

        assert_tensors(result, expected)
