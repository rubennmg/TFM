from torch import Tensor

from core.image_operation import ImageOperation
from core.registry import register_operation
from core._tensor_utils import HEIGHT_DIM, WIDTH_DIM


@register_operation
class Flip(ImageOperation):
    """Class to flip image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    target_tensor = "tensor"
    updates_debayer_state = False

    def __init__(self, horizontal: bool = True):
        """Class constructor.

        Args:
            horizontal (bool): If True, flip horizontally; if False, flip vertically. Defaults to True.
        """
        self.horizontal = horizontal

    def apply(self, x: Tensor) -> Tensor:
        """Apply the flip to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Flipped image tensor of shape (B, C, H, W).
        """
        if self.horizontal:
            return x.flip(dims=[WIDTH_DIM])
        else:
            return x.flip(dims=[HEIGHT_DIM])
