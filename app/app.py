import sys

from PyQt6.QtWidgets import QApplication

from controller import Controller
from gui.main_window import MainWindow
from utils.io import load_stylesheet


def main():
    app: QApplication = QApplication(sys.argv)
    app.setStyle("Fusion")

    qss: str = load_stylesheet("styles/main.qss")
    app.setStyleSheet(qss)

    controller: Controller = Controller()
    window: MainWindow = MainWindow(controller)
    controller.window = window

    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
