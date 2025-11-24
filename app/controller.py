from torch import device

from core.debayer.debayer import Debayer
from core.contrast.sigmoid import SigmoidContrast
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
            show_error("Update Viewer Error", str(e))

    def load_image(self, path: str) -> None:
        try:
            self.image = image_loader.load_image(path, self._device)
            self.window.reset_image_view()
            self.__update_viewer()
        except Exception as e:
            show_error("Load Image Error", str(e))

    def reset_image(self) -> None:
        if self.image is None:
            return

        if self.image.debayered:
            self.image.debayered = False

        try:
            self.image.tensor = self.image.original_tensor.clone()
            self.__update_viewer()
            self.window.reset_image_view()
        except Exception as e:
            show_error("Reset Error", str(e))

    def apply_debayer(self, algorithm_name: str) -> None:
        if self.image is None:
            return

        if self.image.metadata.bayer_pattern is None:
            show_error(
                "Debayer Error",
                "Image does not have a Bayer pattern or Debayering cannot be applied.",
            )
            return

        try:
            operator: Debayer = Debayer(
                algorithm_name, layout=self.image.metadata.bayer_pattern
            )
            self.image.tensor = operator(self.image.tensor)
            self.image.debayered_tensor = self.image.tensor.clone()
            self.image.debayered = True
            self.__update_viewer()
        except Exception as e:
            show_error("Debayer Error", str(e))

    def apply_contrast(self, gain: float, cutoff: float) -> None:
        if self.image is None:
            return

        try:
            operator = SigmoidContrast(gain, cutoff)

            if self.image.debayered and self.image.debayered_tensor is not None:
                self.image.tensor = operator(self.image.debayered_tensor)
            else:
                self.image.tensor = operator(self.image.original_tensor)

            self.__update_viewer()
        except Exception as e:
            show_error("Contrast Enhancement Error", str(e))

    # more methods...
