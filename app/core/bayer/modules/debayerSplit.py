import torch
import torch.nn
import torch.nn.functional

from models.enums.layouts import Layout


class DebayerSplit(torch.nn.Module):
    """Demosaicing of Bayer images using 3x3 green convolution and red,blue upsampling.
    Requires Bayer layout `Layout.RGGB`.
    """

    def __init__(self, layout: Layout = Layout.RGGB):
        super().__init__()
        if layout != Layout.RGGB:
            raise NotImplementedError("DebayerSplit only implemented for RGGB layout.")
        self.layout = layout

        self.pad = torch.nn.ReflectionPad2d(1)
        self.kernel = torch.nn.Parameter(
            torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])[None, None] * 0.25
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
        B, _, H, W = x.shape
        red = x[:, :, ::2, ::2]
        blue = x[:, :, 1::2, 1::2]

        green = torch.nn.functional.conv2d(
            self.pad(x), self.kernel.to(device=x.device, dtype=x.dtype)
        )
        green[:, :, ::2, 1::2] = x[:, :, ::2, 1::2]
        green[:, :, 1::2, ::2] = x[:, :, 1::2, ::2]

        return torch.cat(
            (
                torch.nn.functional.interpolate(
                    red, size=(H, W), mode="bilinear", align_corners=False
                ),
                green,
                torch.nn.functional.interpolate(
                    blue, size=(H, W), mode="bilinear", align_corners=False
                ),
            ),
            dim=1,
        )
