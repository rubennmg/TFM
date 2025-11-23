from dataclasses import dataclass

import numpy as np
from torch import Tensor

from enums.image_formats import ImageFormat
from models.raw_metadata import RawMetadata
from utils.torch import tensor_to_uint8_np


@dataclass
class Image:
    tensor: Tensor
    original_tensor: Tensor
    path: str
    name: str
    image_format: ImageFormat
    debayered: bool = False
    debayered_tensor: Tensor | None = None
    raw_metadata: RawMetadata | None = None

    @property
    def np_array(self) -> np.ndarray:
        """uint8 numpy representation of the current `tensor`.
        Used to display the image in the GUI.

        Returns:
            np.ndarray: uint8 numpy array representation of the image tensor.
        """
        return tensor_to_uint8_np(self.tensor)
