from kornia import enhance as K_e
from torch import Tensor

from core import _tensor_utils as T_u
from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class CLAHE(ImageOperation):
    """Class to apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to image tensors.
    Uses Kornia's CLAHE implementation.

    See: https://kornia.readthedocs.io/en/latest/enhance.html#kornia.enhance.equalize_clahe

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    def __init__(self, clip_limit: float = 40.0, grid_size: float = 8):
        """Class constructor.

        Args:
            clip_limit (float): Threshold for contrast limiting. Default is 40.0.
            grid_size (float): Size of grid for histogram equalization. Default is 8.
        """
        if clip_limit <= 0.0:
            raise ValueError("clip_limit must be a positive float.")
        if grid_size <= 0:
            raise ValueError("grid_size must be a positive integer.")

        self.clip_limit = clip_limit
        self.grid_size = (int(grid_size), int(grid_size))

    def apply(self, x: Tensor) -> Tensor:
        """Apply CLAHE to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (C, H, W).

        Returns:
            Tensor: Image tensor after applying CLAHE of shape (C, H, W).
        """
        T_u.assert_real_valued_tensor(x)

        return K_e.equalize_clahe(
            x, clip_limit=self.clip_limit, grid_size=self.grid_size
        )
