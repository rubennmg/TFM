#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

# Define image dimensions
width: int = 4096
height: int = 2168

# Load raw data
with open('../raw_images/Image__2025-05-12__10-47-32.raw', 'rb') as f:
    raw_data: NDArray[np.uint16] = np.fromfile(f, dtype=np.uint16)

# Reshape the data to 2D image
image: NDArray[np.uint16] = raw_data.reshape((height, width))

# Display the image
def show_image(image: NDArray[np.uint16]) -> None:
    plt.imshow(image, cmap='gray')
    plt.title('16-bit Raw Image')
    plt.colorbar()
    plt.show()

show_image(image)
#%%
import enum
import torch
from enhance_contrast_sigmoid import enhance_contrast_torch
from typing import Literal

class Layout(enum.Enum):
    """Possible Bayer color filter array layouts.

    The value of each entry is the color index (R=0,G=1,B=2)
    within a 2x2 Bayer block.
    """

    RGGB = (0, 1, 1, 2)
    GRBG = (1, 0, 2, 1)
    GBRG = (1, 2, 0, 1)
    BGGR = (2, 1, 1, 0)


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

        xpad = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")
        planes = torch.nn.functional.conv2d(xpad, self.kernels, stride=1)
        planes = torch.cat((planes, x), 1)  # Concat with input to give identity kernel Bx5xHxW
        rgb = torch.gather(
            planes,
            1,
            self.index.repeat(
                1,
                1,
                int(torch.div(H, 2, rounding_mode="floor").item()),
                int(torch.div(W, 2, rounding_mode="floor").item()),
            ).expand(
                B, -1, -1, -1
            ),  # expand for singleton batch dimension is faster
        )
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

debayer_processor: Debayer5x5 = Debayer5x5(layout=Layout.RGGB).cuda()

# La imagen es de 12 bits, por lo que el valor máximo es 4095.
scale: int = 4095

device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
float_datatype: torch.dtype = torch.float32

img_cuda: torch.Tensor = torch.asarray(image).to(device)
scale_tensor: torch.Tensor = torch.tensor(scale, dtype=float_datatype)
img_cuda = img_cuda / scale_tensor
bayer_tensor: torch.Tensor = img_cuda.unsqueeze(0).unsqueeze(0)
img_cuda = debayer_processor(bayer_tensor)
final_img_cuda: torch.Tensor = img_cuda.squeeze().permute(1, 2, 0)

# Force contiguous memory for performance in next steps
img_res: torch.Tensor = final_img_cuda.contiguous()

# Convertir a color ubyte para visualizar
img8: torch.Tensor = (img_res * 255).round().type(torch.uint8)
img_cpu: np.ndarray = img8.cpu().numpy()
# Mostrar la imagen
plt.imshow(img_cpu)
plt.title('Debayered Image')
plt.axis('off')
plt.show()
# %%
enhanced_img: torch.Tensor = enhance_contrast_torch(img_cuda, gain=10.0, cutoff=-1.0)
final_enhanced_img: torch.Tensor = enhanced_img.squeeze().permute(1, 2, 0).contiguous()
enhanced_img8: torch.Tensor = (final_enhanced_img * 255).round().type(torch.uint8)
enhanced_img_cpu: np.ndarray = enhanced_img8.cpu().numpy()
plt.imshow(enhanced_img_cpu)
plt.title('Enhanced Contrast Image')
plt.axis('off')
plt.show()
# %%