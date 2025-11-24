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
            ValueError: If the input tensor does not have 4 dimensions.
            TypeError: If the output of apply() is not a Tensor.
            ValueError: If the output tensor does not have 4 dimensions.

        Returns:
            Tensor: The processed image tensor.
        """
        if not isinstance(x, Tensor):
            raise TypeError(f"ImageOp expected Tensor, got {type(x)}")

        if x.ndim != 4:
            raise ValueError(
                f"Expected input tensor of shape (B, C, H, W), got {x.shape}"
            )

        out = self.apply(x)

        if not isinstance(out, Tensor):
            raise TypeError(
                f"{self.__class__.__name__}.apply() must return Tensor, got {type(out)}"
            )

        if out.ndim != 4:
            raise ValueError(
                f"Expected output tensor of shape (B, C, H, W), got {out.shape}"
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
