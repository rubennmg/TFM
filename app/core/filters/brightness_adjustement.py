import torch
from torch import Tensor

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class BrightnessAdjustment(ImageOperation):
    """Class to apply brightness adjustment on image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self, alfa: float = 1.0):
        """Class constructor.

        Args:
            alfa (float): Brightness adjustment factor. Defaults to 1.0.

        Raises:
            TypeError: If alfa is not a number.
            ValueError: If alfa is negative.
        """
        if not isinstance(alfa, (int, float)):
            raise TypeError(f"Alfa must be a number, got {type(alfa)}")
        if alfa < 0:
            raise ValueError("Alfa must be non-negative")

        self.alfa = alfa

    def apply(self, x: Tensor) -> Tensor:
        """Apply the brightness adjustment.

        Args:
            x (Tensor): Input image tensor of shape (C, H, W).

        Returns:
            Tensor: Brightness adjusted image tensor of shape (C, H, W).
        """
        T_u.assert_real_valued_tensor(x)

        return torch.mul(x, self.alfa)
