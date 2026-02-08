from kornia import enhance as K_e
from torch import Tensor
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class MinMaxNormalizationWithParams(ImageOperation):
    """Class to apply Min-Max Normalization to image tensors with specified min and max values.
    Uses Kornia's Min-Max Normalization implementation.

    See: https://kornia.readthedocs.io/en/stable/enhance.html#kornia.enhance.normalize_min_max

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self, min: float = 0.0, max: float = 1.0):
        """Class constructor.

        Args:
            min (float): Minimum value for normalization.
            max (float): Maximum value for normalization.
        """
        if min >= max:
            raise ValueError("min must be less than max.")
        if min < 0.0 or max > 1.0:
            raise ValueError("min and max should be in the range [0.0, 1.0].")

        self.min = min
        self.max = max
        self.eps = 1e-6  # Small constant to prevent division by zero

    def apply(self, x: Tensor) -> Tensor:
        """Apply Min-Max Normalization to the image tensor using specified min and max values.

        Args:
            x (Tensor): Input image tensor of shape (C, H, W).

        Returns:
            Tensor: Normalized image tensor of shape (C, H, W).
        """
        return K_e.normalize_min_max(
            x, min_val=self.min, max_val=self.max, eps=self.eps
        )
