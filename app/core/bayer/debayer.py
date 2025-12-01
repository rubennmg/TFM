import torch
import torch.nn as nn
from torch import Tensor

from core.bayer.modules.debayer2x2 import Debayer2x2
from core.bayer.modules.debayer3x3 import Debayer3x3
from core.bayer.modules.debayer5x5 import Debayer5x5
from core.bayer.modules.debayerSplit import DebayerSplit
from core.image_operation import ImageOperation
from core.registry import register_operation
from enums.layouts import Layout

_DEBAYER_REGISTRY = {
    "debayer2x2": Debayer2x2,
    "debayer3x3": Debayer3x3,
    "debayer5x5": Debayer5x5,
    "debayersplit": DebayerSplit,
}


@register_operation
class Debayer(ImageOperation):
    """Class to apply debayer operations on RAW tensors.

    Args:
        ImageOperation (ImageOperation): Base class for image operations.
    """

    target_tensor: str = "tensor"
    updates_debayer_state: bool = True

    def __init__(self, algorithm_name: str, layout: Layout = Layout.RGGB):
        """Class constructor.

        Args:
            algorithm_name (str): Name of the debayer algorithm to use.
            layout (Layout): Layout of the RAW image. Defaults to Layout.RGGB.

        Raises:
            ValueError: If the specified algorithm_name is not recognized.
        """
        self.algorithm_name = algorithm_name.lower().strip()
        self.layout = layout

        try:
            module_cls: nn.Module = _DEBAYER_REGISTRY[self.algorithm_name]
        except KeyError:
            raise ValueError(f"Unknown debayer algorithm: {self.algorithm_name}")

        self.module: nn.Module = module_cls(layout=self.layout)

    def apply(self, x: Tensor) -> Tensor:
        """Apply the debayering operation.

        Args:
            x (Tensor): Input RAW tensor of shape (B, 1, H, W).

        Returns:
            Tensor: Debayered tensor of shape (B, 3, H, W).
        """
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"Expected input tensor of shape (B, 1, H, W), got {x.shape}"
            )

        with torch.no_grad():
            out: Tensor = self.module(x).to(device=x.device, dtype=x.dtype)

        return out
