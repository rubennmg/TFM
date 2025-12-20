from torch import Tensor

from core.image_operation import ImageOperation
from core.registry import register_operation
from torchvision.transforms import functional as F


@register_operation
class Rotate(ImageOperation):
    """Class to rotate image tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.

    Raises:
        TypeError: If angle is not a number.
        ValueError: If angle is not between -360 and 360 degrees.
    """

    def __init__(self, angle: float):
        """Class constructor.

        Args:
            angle (float): Angle in degrees to rotate the image.
        """
        if not isinstance(angle, (int, float)):
            raise TypeError(f"Angle must be a number, got {type(angle)}")
        if not (-360 <= angle <= 360):
            raise ValueError("Angle must be between -360 and 360 degrees")

        self.angle = angle

    def apply(self, x: Tensor) -> Tensor:
        """Apply the rotation to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Rotated image tensor of shape (B, C, H, W).
        """
        return F.rotate(x, angle=self.angle, expand=False)
