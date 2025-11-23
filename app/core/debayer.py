import torch
import torch.nn
import torch.nn.functional
from torch import Tensor, device

from enums.image_formats import ImageFormat
from enums.layouts import Layout
from models.image import Image


class Debayer5x5(torch.nn.Module):
    """Demosaicing of Bayer images using Malver-He-Cutler algorithm.

    Requires BG-Bayer color filter array layout. That is,
    the image[1,1]='B', image[1,2]='G'. This corresponds
    to OpenCV naming conventions.

    Compared to Debayer2x2 this method does not use upsampling.
    Compared to Debayer3x3 the algorithm gives sharper edges and
    less chromatic effects.

    ## References
    Malvar, Henrique S., Li-wei He, and Ross Cutler.
    "High-quality linear interpolation for demosaicing of Bayer-patterned
    color images." 2004
    """

    def __init__(self, layout: Layout = Layout.RGGB):
        super(Debayer5x5, self).__init__()
        self.layout = layout
        # fmt: off
        self.kernels = torch.nn.Parameter(
            torch.tensor(
                [
                    # G at R,B locations
                    # scaled by 16
                    [ 0,  0, -2,  0,  0], # noqa
                    [ 0,  0,  4,  0,  0], # noqa
                    [-2,  4,  8,  4, -2], # noqa
                    [ 0,  0,  4,  0,  0], # noqa
                    [ 0,  0, -2,  0,  0], # noqa

                    # R,B at G in R rows
                    # scaled by 16
                    [ 0,  0,  1,  0,  0], # noqa
                    [ 0, -2,  0, -2,  0], # noqa
                    [-2,  8, 10,  8, -2], # noqa
                    [ 0, -2,  0, -2,  0], # noqa
                    [ 0,  0,  1,  0,  0], # noqa

                    # R,B at G in B rows
                    # scaled by 16
                    [ 0,  0, -2,  0,  0], # noqa
                    [ 0, -2,  8, -2,  0], # noqa
                    [ 1,  0, 10,  0,  1], # noqa
                    [ 0, -2,  8, -2,  0], # noqa
                    [ 0,  0, -2,  0,  0], # noqa

                    # R at B and B at R
                    # scaled by 16
                    [ 0,  0, -3,  0,  0], # noqa
                    [ 0,  4,  0,  4,  0], # noqa
                    [-3,  0, 12,  0, -3], # noqa
                    [ 0,  4,  0,  4,  0], # noqa
                    [ 0,  0, -3,  0,  0], # noqa

                    # R at R, B at B, G at G
                    # identity kernel not shown
                ]
            ).view(4, 1, 5, 5).float() / 16.0,
            requires_grad=False,
        )
        # fmt: on

        self.index = torch.nn.Parameter(
            # Below, note that index 4 corresponds to identity kernel
            self._index_from_layout(layout),
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
        B, C, H, W = x.shape

        # ensure kernels are on the same device/dtype as input
        kernels = self.kernels.to(device=x.device, dtype=x.dtype)
        xpad = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")
        planes = torch.nn.functional.conv2d(xpad, kernels, stride=1)
        planes = torch.cat(
            (planes, x), 1
        )  # concat with input to give identity kernel Bx5xHxW
        h2 = H // 2
        w2 = W // 2
        idx = self.index.repeat(1, 1, h2, w2).expand(B, -1, -1, -1)
        rgb = torch.gather(planes, 1, idx)
        return torch.clamp(rgb, 0, 1)

    def _index_from_layout(self, layout: Layout) -> torch.Tensor:
        """Returns a 1x3x2x2 index tensor for each color RGB in a 2x2 bayer tile.

        Note, the index corresponding to the identity kernel is 4, which will be
        correct after concatenating the convolved output with the input image.
        """
        #       ...
        # ... b g b g ...
        # ... g R G r ...
        # ... b G B g ...
        # ... g r g r ...
        #       ...
        # fmt: off
        rggb = torch.tensor(
            [
                # dest channel r
                [4, 1],  # pixel is R,G1
                [2, 3],  # pixel is G2,B
                # dest channel g
                [0, 4],  # pixel is R,G1
                [4, 0],  # pixel is G2,B
                # dest channel b
                [3, 2],  # pixel is R,G1
                [1, 4],  # pixel is G2,B
            ]
        ).view(1, 3, 2, 2)
        # fmt: on
        return {
            Layout.RGGB: rggb,
            Layout.GRBG: torch.roll(rggb, 1, -1),
            Layout.GBRG: torch.roll(rggb, 1, -2),
            Layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
        }.get(layout, rggb)


def _get_cached_debayer5x5(image: Image) -> Debayer5x5:
    """Return a Debayer5x5 module cached on the `Image` instance.

    The module is stored on the image as a private attribute (`_debayer5x5`) along
    with a small key (`_debayer5x5_key`) that records the layout, device and dtype
    used. If the image moves device or dtype changes, the module is recreated
    and replaced.

    Args:
        image (Image): Image to get debayer module for.

    Raises:
        ValueError: If the image does not have raw metadata.

    Returns:
        Debayer5x5: Debayer5x5 module configured for the image.
    """
    if image.raw_metadata is None:
        raise ValueError("Image must have raw_metadata to get a debayer module")

    dev: device = image.tensor.device
    dtype: torch.dtype = image.tensor.dtype
    layout: Layout = image.raw_metadata.bayer_pattern
    cur_key: tuple = (layout, dev.type, getattr(dev, "index", None), dtype)

    module: Debayer5x5 | None = getattr(image, "_debayer5x5", None)
    if getattr(image, "_debayer5x5_key", None) != cur_key or module is None:
        module = Debayer5x5(layout=layout).to(
            device=dev, dtype=dtype, non_blocking=True
        )
        setattr(image, "_debayer5x5", module)
        setattr(image, "_debayer5x5_key", cur_key)

    return module


def apply_debayer5x5(image: Image) -> None:
    """Apply Debayer5x5 on the provided image Tensor.

    Args:
        image (Image): Image to debayer. The image tensor is modified in place.

    Raises:
        ValueError: If the image is not RAW or does not have raw metadata.
    """
    if image.image_format is not ImageFormat.RAW or image.raw_metadata is None:
        raise ValueError("Debayering can only be applied to RAW (1xHxW) images.")

    # add batch dimension if missing
    if image.tensor.ndim == 3:
        image.tensor = image.tensor.unsqueeze(0)

    debayer5x5: Debayer5x5 = _get_cached_debayer5x5(image)

    with torch.no_grad():
        out: Tensor = debayer5x5(image.tensor)

    image.tensor = out.squeeze(0)
    if not image.tensor.is_contiguous():
        image.tensor = image.tensor.contiguous()

    image.debayered_tensor = image.tensor.clone()
    image.debayered = True
