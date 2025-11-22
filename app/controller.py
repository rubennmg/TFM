import traceback

from torch import device

import core.debayer
import core.transformers
from gui.main_window import MainWindow
from loaders import image_loader
from models.image import Image
from utils.error import show_error
from utils.torch import get_device


class Controller:
    def __init__(self):
        self.image: Image | None = None
        self.window: MainWindow
        self._device: device = get_device()

    def __update_viewer(self) -> None:
        if self.image is None:
            return

        try:
            self.window.update_image_view(self.image)
        except Exception as e:
            self.__show_error(e)

    def __show_error(self, exc: Exception) -> None:
        tb: str = traceback.format_exc()
        show_error("Error", str(exc), detailed=tb)

    def load_image(self, path: str) -> None:
        try:
            self.image = image_loader.load_image(path, self._device)
            self.window.viewer.reset_zoom()
            self.window.right_panel.sigmoid_widget.reset_controls_to_default()
            self.__update_viewer()
        except Exception as e:
            self.__show_error(e)

    def reset_image(self) -> None:
        if self.image is None:
            return

        try:
            self.image.tensor = self.image.original_tensor.clone()
            self.__update_viewer()
            self.window.right_panel.sigmoid_widget.reset_controls_to_default()
        except Exception as e:
            self.__show_error(e)

    def apply_debayer5x5(self) -> None:
        if self.image is None:
            return

        try:
            core.debayer.apply_debayer5x5(self.image)
            self.__update_viewer()
        except Exception as e:
            self.__show_error(e)

    def apply_contrast(self, gain: float, cutoff: float) -> None:
        if self.image is None:
            return

        try:
            core.transformers.enhance_contrast_torch(
                self.image, gain=gain, cutoff=cutoff
            )
            self.__update_viewer()
        except Exception as e:
            self.__show_error(e)

    # more methods...
