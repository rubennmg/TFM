import torch
from torch import Tensor

from core import _tensor_utils as T_u
from core._tensor_utils import CHANNEL_DIM, HEIGHT_DIM, WIDTH_DIM
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class GrayToColor(ImageOperation):
    """Convert grayscale images to color images."""

    def __init__(self, mode: str = "heat"):
        if mode not in {"repeat", "heat"}:
            raise ValueError("Mode should be 'repeat' or 'heat'.")
        self.mode = mode

    def apply(self, x: Tensor) -> Tensor:
        """Apply the grayscale to color conversion to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, 1, H, W) in grayscale format.

        Returns:
            Tensor: Converted image tensor of shape (B, 3, H, W) in color format.
        """
        T_u.assert_grayscale_image_tensor(x)

        if self.mode == "repeat":
            return x.repeat(1, 3, 1, 1)

        xmin = x.amin(dim=(HEIGHT_DIM, WIDTH_DIM), keepdim=True)
        xmax = x.amax(dim=(HEIGHT_DIM, WIDTH_DIM), keepdim=True)
        v = (x - xmin) / (xmax - xmin + 1e-8)

        r = v
        g = v.sqrt()
        b = torch.zeros_like(v)

        color = torch.cat([r, g, b], dim=CHANNEL_DIM)

        T_u.assert_color_image_tensor(color)

        return color
