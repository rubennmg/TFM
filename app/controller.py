from torch import device

from core.registry import OPERATION_REGISTRY
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

    def get_image_extensions(self) -> list[str]:
        return image_loader.get_supported_extensions()

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

    def apply_operation(self, operation_name: str, **params) -> None:
        if self.image is None:
            return

        try:
            cls = OPERATION_REGISTRY.get(operation_name)
            if cls is None:
                raise ValueError(f"{operation_name} operation not found in registry.")

            operation = cls(**params)

            if (
                self.image.debayered
                and self.image.debayered_tensor is not None
                and operation.target_tensor == "original_tensor"
            ):
                operation.target_tensor = "debayered_tensor"

            result = operation(getattr(self.image, operation.target_tensor))

            self.image.tensor = result

            if operation.updates_debayer_state:
                self.image.debayered = True
                self.image.debayered_tensor = result.clone()

            self.__update_viewer()

        except Exception as e:
            show_error(f"{operation_name} Error", str(e))
