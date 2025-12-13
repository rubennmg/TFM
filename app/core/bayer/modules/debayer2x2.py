import torch
import torch.nn
import torch.nn.functional

from models.enums.layouts import Layout


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
        x = torch.nn.functional.conv2d(x, self.kernels.to(device=x.device), stride=2)

        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        return x

    def _kernels_from_layout(self, layout: Layout) -> torch.Tensor:
        v = torch.tensor(layout.value).reshape(2, 2)
        r = torch.zeros(2, 2)
        r[v == 0] = 1.0

        g = torch.zeros(2, 2)
        g[v == 1] = 0.5

        b = torch.zeros(2, 2)
        b[v == 2] = 1.0

        k = torch.stack((r, g, b), 0).unsqueeze(1)  # 3x1x2x2
        return k
