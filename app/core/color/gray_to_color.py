from torch import Tensor
import torch

from core.image_operation import ImageOperation
from core.registry import register_operation


@register_operation
class GrayToColor(ImageOperation):
    """Convert un canal gris a 3 canales.

    Modos:
    - repeat: replica el canal (imagen sigue siendo gris)
    - heat: pseudo-color sencillo negro->rojo->amarillo

    El modo 'heat' normaliza cada imagen a [0,1] antes de aplicar el mapa.
    """

    target_tensor = "tensor"
    updates_debayer_state = False

    def __init__(self, mode: str = "heat"):
        if mode not in {"repeat", "heat"}:
            raise ValueError("mode debe ser 'repeat' o 'heat'.")
        self.mode = mode

    def apply(self, x: Tensor) -> Tensor:
        """Apply the grayscale to color conversion to the image tensor.

        Args:
            x (Tensor): Input image tensor of shape (B, 1, H, W) in grayscale format.

        Returns:
            Tensor: Converted image tensor of shape (B, 3, H, W) in color format.
        """
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                "Input tensor must have shape (B, 1, H, W) for grayscale images."
            )

        if self.mode == "repeat":
            return x.repeat(1, 3, 1, 1)

        xmin = x.amin(dim=(2, 3), keepdim=True)
        xmax = x.amax(dim=(2, 3), keepdim=True)
        v = (x - xmin) / (xmax - xmin + 1e-8)

        r = v
        g = v.sqrt()
        b = torch.zeros_like(v)

        return torch.cat([r, g, b], dim=1)
