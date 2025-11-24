import abc

from torch import Tensor


class ImageOperation(abc.ABC):
    """
    Base class for all image operations.

    Every image transformation in the system must inherit from this class.
    """

    def __call__(self, x: Tensor) -> Tensor:
        """Public entry point.

        Args:
            x (Tensor): Input image tensor.

        Raises:
            TypeError: If the input is not a Tensor.
            TypeError: If the output of apply() is not a Tensor.

        Returns:
            Tensor: The processed image tensor.
        """
        if not isinstance(x, Tensor):
            raise TypeError(f"ImageOp expected Tensor, got {type(x)}")

        out = self.apply(x)

        if not isinstance(out, Tensor):
            raise TypeError(
                f"{self.__class__.__name__}.apply() must return Tensor, got {type(out)}"
            )

        if not out.is_contiguous():
            out = out.contiguous()

        return out

    @abc.abstractmethod
    def apply(self, x: Tensor) -> Tensor:
        """Apply the image operation.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tensor: The processed image tensor.
        """
        raise NotImplementedError
