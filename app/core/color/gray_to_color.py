from torch import Tensor
from torchvision.transforms.v2 import functional as F

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class GrayToColor(ImageOperation):
    """Convert grayscale images to color images."""

    def __init__(self):
        """Class constructor.

        This operation does not require any parameters.
        """
        pass

    def apply(self, x: Tensor) -> Tensor:
        """Apply the grayscale to color conversion to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, 1, H, W) in grayscale format.

        Returns:
            Tensor: Converted image tensor of shape (B, 3, H, W) in color format.
        """
        T_u.assert_grayscale_image_tensor(x)

        color = F.grayscale_to_rgb(x)

        T_u.assert_color_image_tensor(color)

        return color
