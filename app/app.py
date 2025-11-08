import sys

from PyQt6.QtWidgets import QApplication
from torch import Tensor
import torch
from core import image_loader
from gui.main_window import MainWindow
from models.image_model import ImageData

class Controller:
    def __init__(self):
        self.current_image: ImageData | None = None
        self.window: MainWindow | None = None
    def load_image(self, path: str) -> Tensor:
        tensor: Tensor = image_loader.load_image(path)
        self.current_image = ImageData(tensor=tensor, path=path, is_raw=path.endswith(".raw"))
        return tensor
    
    def update_viewer(self, tensor: Tensor) -> None:
        if self.window is not None:
            self.window.viewer.show_tensor(tensor)
    
    def apply_contrast(self, gain: float, cutoff: float) -> Tensor | None:
        return None
    
    # more methods...

def main():
    app = QApplication(sys.argv)
    controller = Controller()
    window = MainWindow(controller)
    controller.window = window

    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()