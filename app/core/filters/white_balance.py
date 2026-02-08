import torch
from torch import Tensor

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class WhiteBalance(ImageOperation):
    """Class to perform white balance adjustment on images.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self, method: str = "gray_world"):
        """Class constructor

        Args:
            method (str): 'gray_world' or 'max_rgb'. Default is 'gray_world'.

        Raises:
            ValueError: If method is not 'gray_world' or 'max_rgb'.
        """
        super().__init__()
        if method not in ("gray_world", "max_rgb"):
            raise ValueError(
                f"Method must be 'gray_world' or 'max_rgb', got '{method}'"
            )
        self.method = method
        self.eps = 1e-6  # Small constant to prevent division by zero

    def apply(self, x: Tensor) -> Tensor:
        """Apply white balance adjustment to the input image tensor.

        Args:
            x (Tensor): Input image tensor of shape (3, H, W).

        Returns:
            Tensor: White balance adjusted image tensor of shape (3, H, W).
        """
        T_u.assert_color_image_tensor(x)
        T_u.assert_real_valued_tensor(x)

        if self.method == "gray_world":
            return self._gray_world(x)
        else:
            return self._max_rgb(x)

    def _gray_world(self, x: Tensor) -> Tensor:
        """Gray World algorithm.

        Calculates mean value per channel and normalizes so that red, green, and blue
        have the same mean value.
        """
        r_mean = x[0].mean()
        g_mean = x[1].mean()
        b_mean = x[2].mean()

        overall_mean = (r_mean + g_mean + b_mean) / 3.0

        r_gain = overall_mean / (r_mean + self.eps)
        g_gain = overall_mean / (g_mean + self.eps)
        b_gain = overall_mean / (b_mean + self.eps)

        x[0] *= r_gain
        x[1] *= g_gain
        x[2] *= b_gain

        return x

    def _max_rgb(self, x: Tensor) -> Tensor:
        """Max RGB algorithm.

        Each channel is normalized so that its maximum value equals 1.0,
        relative to the maximum value across all channels.
        """
        r_max = x[0].max()
        g_max = x[1].max()
        b_max = x[2].max()

        overall_max = torch.max(torch.max(r_max, g_max), b_max)

        r_gain = overall_max / (r_max + self.eps)
        g_gain = overall_max / (g_max + self.eps)
        b_gain = overall_max / (b_max + self.eps)

        x[0] *= r_gain
        x[1] *= g_gain
        x[2] *= b_gain

        return x
