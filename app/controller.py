from torch import device

import core.debayer
import core.transformers
from gui.main_window import MainWindow
from loaders import image_loader
from models.image import Image
from utils.safe import safe_call
from utils.torch import get_device


class Controller:
    def __init__(self):
        self.image: Image | None = None
        self.window: MainWindow
        self._device: device = get_device()

    def __update_viewer(self) -> None:
        if self.image is None:
            return
        safe_call(self.window.update_image_view, self.image)

    def load_image(self, path: str) -> None:
        result = safe_call(image_loader.load_image, path, self._device)
        if result is None:
            return

        self.image = result
        safe_call(self.window.viewer.reset_zoom)
        safe_call(self.window.right_panel.sigmoid_widget.reset_controls_to_default)
        safe_call(self.__update_viewer)

    def reset_image(self) -> None:
        if self.image is None:
            return

        self.image.tensor = self.image.original_tensor.clone()
        safe_call(self.__update_viewer)
        safe_call(self.window.right_panel.sigmoid_widget.reset_controls_to_default)

    def apply_debayer5x5(self) -> None:
        if self.image is None:
            return
        safe_call(core.debayer.apply_debayer5x5, self.image)
        safe_call(self.__update_viewer)

    def apply_contrast(self, gain: float, cutoff: float) -> None:
        if self.image is None:
            return
        safe_call(
            core.transformers.enhance_contrast_torch,
            self.image,
            gain=gain,
            cutoff=cutoff,
        )
        safe_call(self.__update_viewer)

    # more methods...
