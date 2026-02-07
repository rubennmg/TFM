from abc import ABC, abstractmethod
import time
import torch
from torch import Tensor

from core import _tensor_utils as T_u


class ImageOperation(ABC):
    """
    Base class for all Image operations.

    Every Image transformation in the system must inherit from this class.
    """

    def __init__(self):
        self.execution_time: float = 0.0

    def __call__(self, x: Tensor) -> Tensor:
        """Public entry point.

        Args:
            x (Tensor): Input image tensor.

        Raises:
            TypeError: If the input is not a Tensor.
            ValueError: If the input tensor does not have 3 dimensions.
            TypeError: If the output of apply() is not a Tensor.
            ValueError: If the output tensor does not have 3 dimensions.

        Returns:
            Tensor: The processed image tensor.
        """
        T_u.assert_image_tensor(x)

        start_time = time.perf_counter()

        out = self.apply(x)
        torch.cuda.synchronize() if x.device.type == "cuda" else None

        end_time = time.perf_counter()

        self.execution_time = end_time - start_time

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
