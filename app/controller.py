from __future__ import annotations

import traceback

import numpy as np
from torch import Tensor, device

from core import image_loader
from core.debayer import apply_debayer5x5
from core.transformers import enhance_contrast_torch
from gui.helpers.error_dialog import show_error_dialog
from gui.main_window import MainWindow
from models.image_model import ImageData
from utils.utils import get_device, tensor_to_uint8_np


class Controller:
    def __init__(self):
        self.current_image: ImageData | None = None
        self.window: MainWindow | None = None
        self._device: device = get_device()

    def update_viewer(self, tensor: Tensor) -> None:
        if self.window is not None:
            # show image in viewer
            self.window.viewer.show_tensor(tensor)
            # update histogram in left panel if available
            try:
                img_np: np.ndarray = tensor_to_uint8_np(tensor)
                self.window.left_panel.update_histogram(img_np)
            except Exception:
                pass

    def _show_error(self, exc: Exception) -> None:
        tb: str = traceback.format_exc()
        show_error_dialog("Error", str(exc), detailed=tb)

    def load_image(self, path: str) -> Tensor | None:
        try:
            tensor: Tensor = image_loader.load_image(path)
            self.current_image = ImageData(
                tensor=tensor, path=path, name="test", is_raw=path.endswith(".raw")
            )
            return tensor
        except Exception as e:
            self._show_error(e)
            return None

    def reset_image(self) -> Tensor | None:
        if self.current_image is None or self.current_image.path is None:
            return None

        try:
            tensor: Tensor = image_loader.load_image(self.current_image.path)
            self.current_image = ImageData(
                tensor=tensor,
                path=self.current_image.path,
                name=self.current_image.name,
                is_raw=self.current_image.is_raw,
            )
            return tensor
        except Exception as e:
            self._show_error(e)
            return None

    def apply_debayer5x5(self) -> None:
        if self.current_image is None:
            return None

        try:
            rgb: Tensor = apply_debayer5x5(self.current_image.tensor, self._device)
            self.current_image = ImageData(
                tensor=rgb,
                path=self.current_image.path,
                name=self.current_image.name,
                is_raw=False,
            )
            self.update_viewer(rgb)
        except Exception as e:
            self._show_error(e)

    def apply_contrast(self, gain: float, cutoff: float) -> None:
        if self.current_image is None:
            return None

        try:
            contrasted: Tensor = enhance_contrast_torch(
                self.current_image.tensor, gain, cutoff
            )
            self.current_image = ImageData(
                tensor=contrasted,
                path=self.current_image.path,
                name=self.current_image.name,
                is_raw=self.current_image.is_raw,
            )
            self.update_viewer(contrasted)
        except Exception as e:
            self._show_error(e)

    # more methods...
