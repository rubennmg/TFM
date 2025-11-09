import sys

from PyQt6.QtWidgets import QApplication
from torch import Tensor

from core import image_loader
from core.debayer import Debayer5x5
from gui.main_window import MainWindow
from models.image_model import ImageData
from core.layouts import Layout
from utils.utils import get_device, load_stylesheet


class Controller:
    def __init__(self):
        self.current_image: ImageData | None = None
        self.window: MainWindow | None = None
        self._device = get_device()
        self._debayer5x5: Debayer5x5 = Debayer5x5(layout=Layout.RGGB).to(self._device)

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
        
        tensor_cuda: Tensor = self.current_image.tensor.to(self._device)
        rgb: Tensor = self._debayer5x5(tensor_cuda)
        rgb = rgb.squeeze().permute(1, 2, 0)
        rgb = rgb.contiguous()
        rgb = rgb.cpu()
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

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    qss = load_stylesheet("styles/main.qss")
    app.setStyleSheet(qss)
    
    controller = Controller()
    window = MainWindow(controller)
    controller.window = window

    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()