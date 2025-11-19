from enums.image_formats import ImageFormat
from loaders.rawpy_loader import load_rawpy
from models.image import Image


def load_arw(path: str, device) -> Image:
    """Load an ARW image from the given path and return an Image dataclass.

    Args:
        path (str): Path to the ARW image file.
        device (torch.device): Device on which the tensor should be allocated.

    Returns:
        Image: Loaded image encapsulated in an Image dataclass.
    """
    return load_rawpy(path, device, ImageFormat.ARW)
