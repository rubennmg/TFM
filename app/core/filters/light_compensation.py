import torch
from torch import Tensor

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class LightCompensation(ImageOperation):
    """Class to perform light compensation on images using a gain matrix.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(
        self, light_gain_compensation: Tensor | None = None, strength: float = 1.0
    ):
        """Class constructor.

        Args:
            light_gain_compensation (Tensor | None): 2D tensor with gain values.
                If None, a matrix of ones will be used (no compensation).
            strength (float): Strength of the compensation (0.0 to 1.0). Default is 1.0.

        Raises:
            ValueError: If strength is not in [0, 1] or light_gain_compensation is not 2D.
        """
        super().__init__()

        if not (0 <= strength <= 1):
            raise ValueError(f"Strength must be in [0, 1], got {strength}")

        self.strength = strength
        self.light_gain_compensation = light_gain_compensation

        if light_gain_compensation is not None:
            if light_gain_compensation.ndim != 2:
                raise ValueError(
                    f"light_gain_compensation must be 2D, got {light_gain_compensation.ndim}D"
                )

    def apply(self, x: Tensor) -> Tensor:
        """Apply light compensation to the input image tensor.

        Args:
            x (Tensor): Input image tensor of shape (C, H, W).

        Returns:
            Tensor: Light-compensated image tensor of shape (C, H, W).
        """
        T_u.assert_real_valued_tensor(x)

        C, H_img, W_img = x.shape

        if self.light_gain_compensation is None:
            light_gain = torch.ones(H_img, W_img, device=x.device, dtype=x.dtype)
        else:
            H_gain, W_gain = self.light_gain_compensation.shape

            light_gain = self.light_gain_compensation.to(x.device, dtype=x.dtype)

            if H_gain != H_img or W_gain != W_img:  # interpolate to match image size
                light_gain = (
                    torch.nn.functional.interpolate(
                        light_gain.unsqueeze(0).unsqueeze(0),
                        size=(H_img, W_img),
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

        effective_gain = 1 + (light_gain - 1) * self.strength

        imgf = x * effective_gain.unsqueeze(0)

        return imgf
