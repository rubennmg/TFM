from torch import device

from enums.image_formats import ImageFormat
from loaders.rawpy_loader import load_rawpy
from models.image import Image


def load_dng(path: str, device: device) -> Image:
    """Load a DNG

    Args:
        path (str): _description_
        device (device): _description_

    Returns:
        Image: _description_
    """
    return load_rawpy(path, device, ImageFormat.DNG)
