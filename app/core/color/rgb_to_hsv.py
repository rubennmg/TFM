import torch
from torch import Tensor

from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class RgbToHsv(ImageOperation):
    """Class to convert RGB image tensors to HSV color space.

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
        """Apply the RGB to HSV conversion to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, 3, H, W) in RGB format.

        Returns:
            Tensor: Converted image tensor of shape (B, 3, H, W) in HSV format.
        """
        if x.shape[1] != 3:
            raise ValueError(
                "Input tensor must have 3 channels representing RGB image."
            )

        cmax, cmax_idx = torch.max(x, dim=1, keepdim=True)
        cmin = torch.min(x, dim=1, keepdim=True)[0]

        delta = cmax - cmin
        hsv_h = torch.empty_like(x[:, 0:1, :, :])

        cmax_idx[delta == 0] = 3

        hsv_h[cmax_idx == 0] = (((x[:, 1:2] - x[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsv_h[cmax_idx == 1] = (((x[:, 2:3] - x[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsv_h[cmax_idx == 2] = (((x[:, 0:1] - x[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsv_h[cmax_idx == 3] = 0.0
        hsv_h /= 6.0

        hsv_s = torch.where(cmax == 0, torch.tensor(0.0).type_as(x), delta / cmax)

        hsv_v = cmax

        return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)
