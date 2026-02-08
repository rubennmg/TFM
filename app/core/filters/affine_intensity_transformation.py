import torch
from torch import Tensor

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class AffineIntensityTransformation(ImageOperation):
    """Class to apply affine intensity transformation on image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self, gain: float = 1.0, bias: float = 0.0):
        """Class constructor.

        Args:
            gain (float): Gain factor for the affine transformation. Defaults to 1.0.
            bias (float): Bias factor for the affine transformation. Defaults to 0.0.

        Raises:
            TypeError: If gain is not a number.
            ValueError: If gain is negative.
            TypeError: If bias is not a number.
        """
        if not isinstance(gain, (int, float)):
            raise TypeError(f"Gain must be a number, got {type(gain)}")
        if gain < 0:
            raise ValueError("Gain must be non-negative")

        if not isinstance(bias, (int, float)):
            raise TypeError(f"Bias must be a number, got {type(bias)}")

        self.gain = gain
        self.bias = bias

    def apply(self, x: Tensor) -> Tensor:
        """Apply the affine intensity transformation.

        Args:
            x (Tensor): Input image tensor of shape (C, H, W).

        Returns:
            Tensor: Affine intensity transformed image tensor of shape (C, H, W).
        """
        T_u.assert_real_valued_tensor(x)

        return torch.add(torch.mul(x, self.gain), self.bias)
