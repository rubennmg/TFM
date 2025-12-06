import torch
from torch import Tensor

from core import _tensor_utils as T_u
from core._tensor_utils import CHANNEL_DIM
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class HsvToRgb(ImageOperation):
    """Class to convert HSV image tensors to RGB color space.

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
        """Apply the HSV to RGB conversion to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, 3, H, W) in HSV format.

        Returns:
            Tensor: Converted image tensor of shape (B, 3, H, W) in RGB format.
        """
        T_u.assert_color_image_tensor(x)

        hsv_h, hsv_s, hsv_l = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        _c = hsv_l * hsv_s
        _x = _c * (-torch.abs(hsv_h * 6.0 % 2.0 - 1) + 1.0)
        _m = hsv_l - _c
        _o = torch.zeros_like(_c)

        idx = (hsv_h * 6.0).type(torch.uint8)
        idx = (idx % 6).expand(-1, 3, -1, -1)

        rgb = torch.empty_like(x)

        rgb[idx == 0] = torch.cat([_c, _x, _o], dim=CHANNEL_DIM)[idx == 0]
        rgb[idx == 1] = torch.cat([_x, _c, _o], dim=CHANNEL_DIM)[idx == 1]
        rgb[idx == 2] = torch.cat([_o, _c, _x], dim=CHANNEL_DIM)[idx == 2]
        rgb[idx == 3] = torch.cat([_o, _x, _c], dim=CHANNEL_DIM)[idx == 3]
        rgb[idx == 4] = torch.cat([_x, _o, _c], dim=CHANNEL_DIM)[idx == 4]
        rgb[idx == 5] = torch.cat([_c, _o, _x], dim=CHANNEL_DIM)[idx == 5]

        rgb += _m

        T_u.assert_color_image_tensor(rgb)

        return rgb
