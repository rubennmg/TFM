import torch
from torch import Tensor

from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class MinMaxNormalization(ImageOperation):
    """Class to apply Min-Max Normalization to image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    target_tensor = "tensor"
    updates_debayer_state = False

    def __init__(self):
        """Class constructor.

        No additional parameters are required for Min-Max Normalization.
        """
        pass

    def apply(self, x: Tensor) -> Tensor:
        """Apply Min-Max Normalization to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Normalized image tensor of shape (B, C, H, W).
        """
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)

        denom = x_max - x_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)

        x_normalized = (x - x_min) / denom

        return x_normalized
