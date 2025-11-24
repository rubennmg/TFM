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
        """
        assert cutoff <= 1
        assert gain >= 0

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

        t_cutoff = torch.tensor(self.cutoff).to(device=x.device, non_blocking=True)

        if self.cutoff < 0:
            min_valid_estimated_cutoff = 0.4
            max_valid_estimated_cutoff = 0.6
            estimated_cutoff = torch.mean(x).to(device=x.device, non_blocking=True)
            t_cutoff = torch.clip(
                estimated_cutoff, min_valid_estimated_cutoff, max_valid_estimated_cutoff
            )

        imgf: Tensor = torch.empty_like(x).to(device=x.device, dtype=x.dtype)

        sigmoid_min: Tensor = 1 / (1 + torch.exp(self.gain * (t_cutoff - 0)))
        sigmoid_max: Tensor = 1 / (1 + torch.exp(self.gain * (t_cutoff - 1)))

        imgf = 1 / (1 + torch.exp(self.gain * (t_cutoff - x)))
        imgf = (imgf - sigmoid_min) / (sigmoid_max - sigmoid_min)

        return imgf
