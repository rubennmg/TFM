import torch
from torch import Tensor

from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class MinMaxPercentileNormalization(ImageOperation):
    """Class to apply Min-Max Percentile Normalization to image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    target_tensor = "tensor"
    updates_debayer_state = False

    def __init__(self, lower_percentile: float = 0.02, upper_percentile: float = 0.98):
        """Class constructor."""
        if not (0.0 <= lower_percentile < upper_percentile <= 1.0):
            raise ValueError(
                "Percentiles must satisfy 0.0 <= lower_percentile < upper_percentile <= 1.0"
            )

        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def apply(self, x: Tensor) -> Tensor:
        """Apply Min-Max Percentile Normalization to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Normalized image tensor of shape (B, C, H, W).
        """
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)

        p_lower = torch.quantile(x_flat, self.lower_percentile, dim=2, keepdim=True)
        p_upper = torch.quantile(x_flat, self.upper_percentile, dim=2, keepdim=True)

        denom = p_upper - p_lower
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)

        x_normalized = (x_flat - p_lower) / denom
        x_normalized = torch.clamp(x_normalized, 0.0, 1.0)
        x_normalized = x_normalized.view(b, c, h, w)

        return x_normalized
