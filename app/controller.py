from torch import Tensor, device

from core import image_loader
from core.debayer import apply_debayer5x5
from gui.main_window import MainWindow
from models.image_model import ImageData
from utils.utils import get_device


class Controller:
    def __init__(self):
        self.current_image: ImageData | None = None
        self.window: MainWindow | None = None
        self._device: device = get_device()

    def update_viewer(self, tensor: Tensor) -> None:
        if self.window is not None:
            self.window.viewer.show_tensor(tensor)
        
    def load_image(self, path: str) -> Tensor:
        tensor: Tensor = image_loader.load_image(path)
        self.current_image = ImageData(tensor=tensor, path=path, name="test", is_raw=path.endswith(".raw"))
        return tensor
    
    def apply_debayer5x5(self) -> None:
        if self.current_image is None:
            return None
        
        rgb: Tensor = apply_debayer5x5(self.current_image.tensor, self._device)
        
        self.current_image = ImageData(
            tensor=rgb,
            path=self.current_image.path,
            name=self.current_image.name,
            is_raw=False,
        )
        self.update_viewer(rgb)
    
    def apply_contrast(self, gain: float, cutoff: float) -> Tensor | None:
        return None
    
    # more methods...