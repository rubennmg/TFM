import sys
import cv2
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt


class BayerDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt TFM Test")
        self.setGeometry(200, 200, 1600, 800)

        self.image_raw = None
        self.image_rgb = None

        # Image display
        self.label_image = QLabel("No image loaded")
        self.label_image.setObjectName("imageLabel")
        self.label_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Left panel
        self.btn_load = QPushButton("Load Image")
        self.btn_reset = QPushButton("Reset Image")
        self.btn_reset.setEnabled(False)

        self.btn_load.clicked.connect(self.load_image)
        self.btn_reset.clicked.connect(self.reset_image)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.btn_load)
        left_layout.addWidget(self.btn_reset)
        left_layout.addStretch()

        # Right panel
        self.btn_bayer = QPushButton("Apply Bayer Demosaicing")
        self.btn_bayer.setEnabled(False)
        self.btn_bayer.clicked.connect(self.apply_bayer)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.btn_bayer)
        right_layout.addStretch()

        # Center area
        center_layout = QVBoxLayout()
        center_layout.addWidget(self.label_image)

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(center_layout, 6)
        main_layout.addLayout(right_layout, 1)
        self.setLayout(main_layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.bmp *.tiff *.raw)"
        )
        if file_path:
            try:
                self.image_raw = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if self.image_raw is None:
                    raise ValueError("Could not read the image.")
                self.display_image(self.image_raw)
                self.btn_bayer.setEnabled(True)
                self.btn_reset.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load image:\n{e}")

    def reset_image(self):
        if self.image_raw is not None:
            self.display_image(self.image_raw)

    def apply_bayer(self):
        if self.image_raw is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
        try:
            self.image_rgb = cv2.cvtColor(self.image_raw, cv2.COLOR_BayerBG2RGB)
            self.display_image(self.image_rgb)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying Bayer demosaicing:\n{e}")

    def display_image(self, img):
        if len(img.shape) == 2:
            height, width = img.shape
            qimg = QImage(img.data, width, height, width, QImage.Format.Format_Grayscale8)
        else:
            height, width, ch = img.shape
            bytes_per_line = ch * width
            qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg).scaled(
            700, 500, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.label_image.setPixmap(pixmap)


def load_stylesheet(file_path: str) -> str:
    """Read and return the contents of a QSS file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Could not load stylesheet: {e}")
        return ""


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    qss = load_stylesheet("styles.qss")
    app.setStyleSheet(qss)

    window = BayerDemo()
    window.show()
    sys.exit(app.exec())
