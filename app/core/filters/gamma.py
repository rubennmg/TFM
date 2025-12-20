from torch import Tensor
from torchvision.transforms.v2 import functional as F

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class GammaAdjustment(ImageOperation):
    """Class to perform gamma adjustment on images.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self, c: float = 1.0, gamma: float = 1.0):
        """Class constructor.

        Args:
            gamma (float): Gamma value for adjustment. Default is 1.0 (no change).
        """
        if c <= 0:
            raise ValueError("Parameter 'c' must be greater than 0.")
        if gamma <= 0:
            raise ValueError("Parameter 'gamma' must be greater than 0.")

        self.c = c
        self.gamma = gamma

    def apply(self, x: Tensor) -> Tensor:
        """Apply gamma adjustment to the input image tensor.

        Args:
            x (Tensor): Input image tensor of shape (C, H, W).

        Returns:
            Tensor: Gamma-adjusted image tensor of shape (C, H, W).
        """
        T_u.assert_real_valued_tensor(x)

        return F.adjust_gamma(x, self.gamma, self.c)
