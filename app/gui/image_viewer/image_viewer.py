import numpy as np
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QScrollArea, QTabWidget, QVBoxLayout, QWidget

from gui.image_viewer.widgets.image_canvas import ImageCanvas
from gui.image_viewer.widgets.image_info import ImageInfo
from gui.image_viewer.widgets.operation_logger import OperationLogger
from models.image import Image


class ImageViewer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self._qimg: QImage | None = None
        self._zoom: float = 0.25
        self._zoom_step: float = 1.15
        self._min_zoom: float = 0.05
        self._max_zoom: float = 10.0

        self.__setup_ui()

    def __setup_ui(self) -> None:
        layout: QVBoxLayout = QVBoxLayout()

        self.info_widget: ImageInfo = ImageInfo()
        self.logger_widget: OperationLogger = OperationLogger()

        self.image_canvas: ImageCanvas = ImageCanvas()

        self.scroll_area: QScrollArea = QScrollArea()
        self.scroll_area.setObjectName("imageScrollArea")
        self.scroll_area.setWidget(self.image_canvas)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        viewport: QWidget | None = self.scroll_area.viewport()
        if viewport is not None:
            viewport.installEventFilter(self)

        self.tabs: QTabWidget = QTabWidget()
        self.tabs.setObjectName("infoLoggerTabs")
        self.tabs.addTab(self.info_widget, "Image Info")
        self.tabs.addTab(self.logger_widget, "Logger")

        layout.addWidget(self.scroll_area, stretch=5)
        layout.addWidget(self.tabs, stretch=1)

        self.setLayout(layout)

    def update_image(self, np_array: np.ndarray, image: Image) -> None:
        self.info_widget.update_from_image(image)

        buf = memoryview(np_array)

        h, w, c = np_array.shape
        bytes_per_line: int = np_array.strides[0]
        img_format: QImage.Format = (
            QImage.Format.Format_Grayscale8 if c == 1 else QImage.Format.Format_RGB888
        )

        qimg = QImage(buf, w, h, bytes_per_line, img_format).copy()
        self._qimg = qimg
        self._pixmap = QPixmap.fromImage(self._qimg)
        self.__update_display()

    def __update_display(self) -> None:
        if self._pixmap is None:
            self.image_canvas.clear()
            self.info_widget.update_from_image(None)
            return

        if self._zoom == 1.0:
            pix = self._pixmap
        else:
            assert self._qimg is not None
            w = max(1, int(self._qimg.width() * self._zoom))
            h = max(1, int(self._qimg.height() * self._zoom))
            scaled_img = self._qimg.scaled(
                w,
                h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            pix = QPixmap.fromImage(scaled_img)

        self.image_canvas.set_pixmap(pix)

    def append_log_entry(self, entry: str) -> None:
        self.logger_widget.append_entry(entry)

    def clear_log(self) -> None:
        self.logger_widget.clear_entries()

    def set_zoom(self, factor: float) -> None:
        factor = max(self._min_zoom, min(self._max_zoom, factor))
        if abs(self._zoom - factor) < 1e-6:
            return
        self._zoom = factor
        self.__update_display()

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * self._zoom_step)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / self._zoom_step)

    def reset_zoom(self) -> None:
        self.set_zoom(0.25)

    def eventFilter(self, a0, a1) -> bool:
        if a1 is None:
            return False

        if a1.type() == QEvent.Type.Wheel:
            if not isinstance(a1, QWheelEvent):
                return False
            modifiers = a1.modifiers()
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                delta = a1.angleDelta().y()
                if delta > 0:
                    self.zoom_in()
                else:
                    self.zoom_out()
                return True
        elif a1.type() == QEvent.Type.Resize:
            pass
        return super().eventFilter(a0, a1)
