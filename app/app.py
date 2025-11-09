import sys

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow
from utils.utils import load_stylesheet
from controller import Controller


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