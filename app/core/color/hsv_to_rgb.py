from kornia import color as K_c
from torch import Tensor

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class HsvToRgb(ImageOperation):
    """Class to convert HSV image tensors to RGB color space.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self):
        """Class constructor.

        This operation does not require any parameters.
        """
        pass

    def apply(self, x: Tensor) -> Tensor:
        """Apply the HSV to RGB conversion to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (3, H, W) in HSV format.

        Returns:
            Tensor: Converted image tensor of shape (3, H, W) in RGB format.
        """
        T_u.assert_color_image_tensor(x)

        rgb = K_c.hsv_to_rgb(x)

        T_u.assert_color_image_tensor(rgb)

        return rgb
