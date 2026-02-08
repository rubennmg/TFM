import torch
from torch import Tensor

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class MeanContrastAdjustment(ImageOperation):
    """Class to apply mean contrast adjustment on image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self, beta: float = 1.0):
        """Class constructor.

        Args:
            beta (float): Contrast adjustment factor. Defaults to 1.0.

        Raises:
            TypeError: If beta is not a number.
            ValueError: If beta is negative.
        """
        if not isinstance(beta, (int, float)):
            raise TypeError(f"Beta must be a number, got {type(beta)}")
        if beta < 0:
            raise ValueError("Beta must be non-negative")

        self.beta = beta

    def apply(self, x: Tensor) -> Tensor:
        """Apply the mean contrast adjustment.

        Args:
            x (Tensor): Input image tensor of shape (C, H, W).

        Returns:
            Tensor: Mean contrast adjusted image tensor of shape (C, H, W).
        """
        T_u.assert_real_valued_tensor(x)

        mean = torch.mean(x)
        return torch.mul(x - mean, self.beta) + mean
