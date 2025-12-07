from torch import Tensor

from core._tensor_utils import HEIGHT_DIM, WIDTH_DIM
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class MinMaxNormalization(ImageOperation):
    """Class to apply Min-Max Normalization to image tensors.

    Normalization is applied **by channel** using the minimum and maximum values.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor.

        No additional parameters are required for Min-Max Normalization.
        """
        self.eps = 1e-6  # Small constant to prevent division by zero

    def apply(self, x: Tensor) -> Tensor:
        """Apply Min-Max Normalization to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Normalized image tensor of shape (B, C, H, W).
        """
        x_min = x.amin(dim=(HEIGHT_DIM, WIDTH_DIM), keepdim=True)
        x_max = x.amax(dim=(HEIGHT_DIM, WIDTH_DIM), keepdim=True)

        denom = (x_max - x_min).clamp_min(self.eps)

        x_normalized = (x - x_min) / denom

        return x_normalized
