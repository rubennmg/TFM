from torch import Tensor
from torchvision.transforms.v2 import functional as F

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class ColorToGray(ImageOperation):
    """Class to convert color image tensors to grayscale.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor.

        This operation does not require any parameters.
        """
        pass

    def apply(self, x: Tensor) -> Tensor:
        """Apply the color to grayscale conversion to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (3, H, W) in color format.

        Returns:
            Tensor: Converted image tensor of shape (1, H, W) in grayscale format.
        """
        T_u.assert_color_image_tensor(x)

        gray = F.rgb_to_grayscale(x)

        T_u.assert_grayscale_image_tensor(gray)

        return gray
