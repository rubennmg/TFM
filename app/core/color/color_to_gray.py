import torch
from torch import Tensor

from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class ColorToGray(ImageOperation):
    """Class to convert color image tensors to grayscale.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    target_tensor = "tensor"
    updates_debayer_state = False

    def __init__(self):
        """Class constructor.

        This operation does not require any parameters.
        """
        pass

    def apply(self, x: Tensor) -> Tensor:
        """Apply the color to grayscale conversion to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, 3, H, W) in color format.

        Returns:
            Tensor: Converted image tensor of shape (B, 1, H, W) in grayscale format.
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                "Input tensor must have shape (B, 3, H, W) for color images."
            )

        weights = x.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)

        gray = torch.mul(x, weights).sum(dim=1, keepdim=True).to(x.dtype)

        return gray
