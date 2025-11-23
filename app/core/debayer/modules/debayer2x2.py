import torch
import torch.nn
import torch.nn.functional
from torch import Tensor, device

from enums.image_formats import ImageFormat
from enums.layouts import Layout
from models.image import Image


class Debayer2x2(torch.nn.Module):
    """Fast demosaicing of Bayer images using 2x2 convolutions.

    This method uses 3 kernels of size 2x2 and stride 2. Each kernel
    corresponds to a single color RGB. For R and B the corresponding
    value from each 2x2 Bayer block is taken according to the layout.
    For G, G1 and G2 are averaged. The resulting image has half width/
    height and is upsampled by a factor of 2.
    """

    def __init__(self, layout: Layout = Layout.RGGB):
        super(Debayer2x2, self).__init__()
        self.layout = layout

        self.kernels = torch.nn.Parameter(
            self._kernels_from_layout(layout),
            requires_grad=False,
        )

    def forward(self, x):
        """Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        """
        x = torch.nn.functional.conv2d(x, self.kernels, stride=2)

        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        return x

    def _kernels_from_layout(self, layout: Layout) -> Tensor:
        v = torch.tensor(layout.value).reshape(2, 2)
        r = torch.zeros(2, 2)
        r[v == 0] = 1.0

        g = torch.zeros(2, 2)
        g[v == 1] = 0.5

        b = torch.zeros(2, 2)
        b[v == 2] = 1.0

        k = torch.stack((r, g, b), 0).unsqueeze(1)  # 3x1x2x2
        return k


def _get_cached_debayer2x2(image: Image) -> Debayer2x2:
    """Return a Debayer2x2 module cached on the `Image` instance.

    The module is stored on the image as a private attribute (`_debayer2x2`) along
    with a small key (`_debayer2x2_key`) that records the layout, device and dtype
    used. If the image moves device or dtype changes, the module is recreated
    and replaced.

    Args:
        image (Image): Image to get debayer module for.

    Raises:
        ValueError: If the image does not have raw metadata.

    Returns:
        Debayer5x5: Debayer5x5 module configured for the image.
    """
    dev: device = image.tensor.device
    dtype: torch.dtype = image.tensor.dtype
    layout: Layout = (
        image.metadata.bayer_pattern
        if image.metadata.bayer_pattern is not None
        else Layout.RGGB  # RGGB default
    )
    cur_key: tuple = (layout, dev.type, getattr(dev, "index", None), dtype)

    module: Debayer2x2 | None = getattr(image, "_debayer2x2", None)
    if getattr(image, "_debayer2x2_key", None) != cur_key or module is None:
        module = Debayer2x2(layout=layout).to(
            device=dev, dtype=dtype, non_blocking=True
        )
        setattr(image, "_debayer2x2", module)
        setattr(image, "_debayer2x2_key", cur_key)

    return module


def apply_debayer2x2(image: Image) -> None:
    """Apply Debayer2x2 on the provided image Tensor.

    Args:
        image (Image): Image to debayer. The image tensor is modified in place.

    Raises:
        ValueError: If the image is not RAW or does not have raw metadata.
    """
    if image.image_format is not ImageFormat.RAW or image.metadata is None:
        raise ValueError("Debayering can only be applied to RAW (1xHxW) images.")

    # add batch dimension if missing
    if image.tensor.ndim == 3:
        image.tensor = image.tensor.unsqueeze(0)

    debayer2x2: Debayer2x2 = _get_cached_debayer2x2(image)

    with torch.no_grad():
        out: Tensor = debayer2x2(image.tensor).squeeze(0)

    image.tensor = out.clone()
    if not image.tensor.is_contiguous():
        image.tensor = image.tensor.contiguous()

    image.debayered_tensor = out.clone()
    image.debayered = True
