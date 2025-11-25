import torch
from torch import Tensor

from core.image_operation import ImageOperation


class SigmoidContrast(ImageOperation):
    """Class to apply sigmoid contrast enhancement on image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self, gain: float = 1.0, cutoff: float = 0.5):
        """Class constructor.

        Args:
            gain (float): Gain factor for the sigmoid function. Defaults to 1.0.
            cutoff (float): Cutoff value for the sigmoid function. Defaults to 0.5.

        Raises:
            TypeError: If gain is not a number.
            ValueError: If gain is negative.
            TypeError: If cutoff is not a number.
            ValueError: If cutoff is not between 0 and 1.
        """
        if not isinstance(gain, (int, float)):
            raise TypeError(f"Gain must be a number, got {type(gain)}")
        if gain < 0:
            raise ValueError("Gain must be non-negative")

        if not isinstance(cutoff, (int, float)):
            raise TypeError(f"Cutoff must be a number, got {type(cutoff)}")
        if not (0 <= cutoff <= 1):
            raise ValueError("Cutoff must be between 0 and 1")

        self.gain = gain
        self.cutoff = cutoff

    def apply(self, x: Tensor) -> Tensor:
        """Apply the sigmoid contrast enhancement.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Contrast-enhanced image tensor of shape (B, C, H, W).
        """
        if self.gain == 0:
            return x

        if self.cutoff < 0:
            estimated = x.mean()
            t_cutoff = torch.clamp(estimated, 0.4, 0.6)
        else:
            t_cutoff = x.new_tensor(self.cutoff, device=x.device, dtype=x.dtype)

        sigmoid_min = torch.sigmoid(-self.gain * t_cutoff)
        sigmoid_max = torch.sigmoid(self.gain * (1 - t_cutoff))

        imgf = torch.sigmoid(self.gain * (x - t_cutoff))
        imgf = (imgf - sigmoid_min) / (sigmoid_max - sigmoid_min)

        return imgf
