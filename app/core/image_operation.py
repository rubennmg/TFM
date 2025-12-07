from abc import ABC, abstractmethod

from torch import Tensor

from core import _tensor_utils as T_u


class ImageOperation(ABC):
    """
    Base class for all Image operations.

    Every Image transformation in the system must inherit from this class.
    """

    def __call__(self, x: Tensor) -> Tensor:
        """Public entry point.

        Args:
            x (Tensor): Input image tensor.

        Raises:
            TypeError: If the input is not a Tensor.
            ValueError: If the input tensor does not have 4 dimensions.
            TypeError: If the output of apply() is not a Tensor.
            ValueError: If the output tensor does not have 4 dimensions.

        Returns:
            Tensor: The processed image tensor.
        """
        T_u.assert_image_tensor(x)

        out = self.apply(x)

        T_u.assert_image_tensor(out)

        if not out.is_contiguous():
            out = out.contiguous()

        return out

    @abstractmethod
    def apply(self, x: Tensor) -> Tensor:
        """Apply the image operation.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tensor: The processed image tensor.
        """
