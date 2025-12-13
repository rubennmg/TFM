import torch
import torch.nn
import torch.nn.functional

from models.enums.layouts import Layout


class Debayer3x3(torch.nn.Module):
    """Demosaicing of Bayer images using 3x3 convolutions.

    Compared to Debayer2x2 this method does not use upsampling.
    Instead, we identify five 3x3 interpolation kernels that
    are sufficient to reconstruct every color channel at every
    pixel location.

    We convolve the image with these 5 kernels using stride=1
    and a one pixel reflection padding. Finally, we gather
    the correct channel values for each pixel location. Todo so,
    we recognize that the Bayer pattern repeats horizontally and
    vertically every 2 pixels. Therefore, we define the correct
    index lookups for a 2x2 grid cell and then repeat to image
    dimensions.
    """

    def __init__(self, layout: Layout = Layout.RGGB):
        super(Debayer3x3, self).__init__()
        self.layout = layout
        # fmt: off
        self.kernels = torch.nn.Parameter(
            torch.tensor(
                [
                    [0, 0.25, 0],
                    [0.25, 0, 0.25],
                    [0, 0.25, 0],

                    [0.25, 0, 0.25],
                    [0, 0, 0],
                    [0.25, 0, 0.25],

                    [0, 0, 0],
                    [0.5, 0, 0.5],
                    [0, 0, 0],

                    [0, 0.5, 0],
                    [0, 0, 0],
                    [0, 0.5, 0],
                ]
            ).view(4, 1, 3, 3),
            requires_grad=False,
        )
        # fmt: on

        self.index = torch.nn.Parameter(
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

        xpad = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="reflect")
        c = torch.nn.functional.conv2d(xpad, self.kernels.to(device=x.device), stride=1)
        c = torch.cat((c, x), 1)  # Concat with input to give identity kernel Bx5xHxW

        rgb = torch.gather(
            c,
            1,
            self.index.to(device=x.device)
            .repeat(
                1,
                1,
                int(torch.div(H, 2, rounding_mode="floor")),
                int(torch.div(W, 2, rounding_mode="floor")),
            )
            .expand(B, -1, -1, -1),  # expand in batch is faster than repeat
        )
        return rgb

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
                [4, 2],  # pixel is R,G1
                [3, 1],  # pixel is G2,B
                # dest channel g
                [0, 4],  # pixel is R,G1
                [4, 0],  # pixel is G2,B
                # dest channel b
                [1, 3],  # pixel is R,G1
                [2, 4],  # pixel is G2,B
            ]
        ).view(1, 3, 2, 2)
        # fmt: on
        return {
            Layout.RGGB: rggb,
            Layout.GRBG: torch.roll(rggb, 1, -1),
            Layout.GBRG: torch.roll(rggb, 1, -2),
            Layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
        }.get(layout, rggb)
