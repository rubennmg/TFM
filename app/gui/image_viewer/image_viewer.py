from collections import OrderedDict

import numpy as np
from PyQt6.QtCore import QPoint, QEvent, Qt
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QScrollArea, QTabWidget, QVBoxLayout, QWidget

from gui.image_viewer.widgets.device_logger import DeviceLogger
from gui.image_viewer.widgets.image_canvas import ImageCanvas
from gui.image_viewer.widgets.image_info import ImageInfo
from gui.image_viewer.widgets.operation_logger import OperationLogger
from models.image import Image


class ImageViewer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self._qimg: QImage | None = None
        self._display_np: np.ndarray | None = None
        self._display_buf: memoryview | None = None
        self._zoom: float = 0.25
        self._zoom_step: float = 1.15
        self._min_zoom: float = 0.1
        self._max_zoom: float = 128.0
        self._max_scaled_pixels: int = 1_000_000_000
        self._performance_mode: bool = True
        self._performance_max_side: int = 1920
        self._dragging: bool = False
        self._drag_start_pos: QPoint = QPoint()
        self._drag_start_h: int = 0
        self._drag_start_v: int = 0
        self._scaled_cache: OrderedDict[float, QPixmap] = OrderedDict()
        self._scaled_cache_max: int = 12

        self.__setup_ui()

    def __setup_ui(self) -> None:
        layout: QVBoxLayout = QVBoxLayout()

        self.info_widget: ImageInfo = ImageInfo()
        self.logger_widget: OperationLogger = OperationLogger()
        self.device_info_widget: DeviceLogger = DeviceLogger()

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
        self.tabs.addTab(self.device_info_widget, "Device Info")

        layout.addWidget(self.scroll_area, stretch=5)
        layout.addWidget(self.tabs, stretch=1)

        self.setLayout(layout)

    def update_image(self, np_array: np.ndarray, image: Image) -> None:
        self.info_widget.update_from_image(image)

        display_np = self.__prepare_display_array(np_array)
        self._display_np = display_np
        self._display_buf = memoryview(display_np)

        h, w, c = display_np.shape
        bytes_per_line: int = display_np.strides[0]
        img_format: QImage.Format = (
            QImage.Format.Format_Grayscale8 if c == 1 else QImage.Format.Format_RGB888
        )

        qimg = QImage(self._display_buf, w, h, bytes_per_line, img_format)
        self._qimg = qimg
        self._pixmap = QPixmap.fromImage(self._qimg)
        self._scaled_cache.clear()
        self.__update_display()

    def __effective_max_zoom(self) -> float:
        if self._qimg is None:
            return self._max_zoom

        base_pixels = max(1, self._qimg.width() * self._qimg.height())
        max_zoom_by_pixels = float(np.sqrt(self._max_scaled_pixels / base_pixels))
        return max(self._min_zoom, min(self._max_zoom, max_zoom_by_pixels))

    def __prepare_display_array(self, np_array: np.ndarray) -> np.ndarray:
        arr = np_array

        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        elif arr.ndim != 3:
            raise ValueError(f"Expected image array with 2 or 3 dims, got {arr.shape}")

        if arr.shape[2] > 3:
            arr = arr[:, :, :3]

        if self._performance_mode:
            h, w = arr.shape[:2]
            max_side = max(h, w)
            if max_side > self._performance_max_side:
                step = int(np.ceil(max_side / self._performance_max_side))
                arr = arr[::step, ::step, :]

        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)

        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        return arr

    def __update_display(self) -> None:
        if self._pixmap is None:
            self.image_canvas.clear()
            self.info_widget.update_from_image(None)
            return

        effective_max_zoom = self.__effective_max_zoom()
        if self._zoom > effective_max_zoom:
            self._zoom = effective_max_zoom

        if self._zoom == 1.0:
            pix = self._pixmap
        else:
            assert self._qimg is not None
            assert self._pixmap is not None
            cache_key = round(self._zoom, 3)
            pix = self._scaled_cache.get(cache_key)
            if pix is None:
                w = max(1, int(self._qimg.width() * self._zoom))
                h = max(1, int(self._qimg.height() * self._zoom))
                pix = self._pixmap.scaled(
                    w,
                    h,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
                self._scaled_cache[cache_key] = pix
                if len(self._scaled_cache) > self._scaled_cache_max:
                    self._scaled_cache.popitem(last=False)
            else:
                self._scaled_cache.move_to_end(cache_key)

        self.image_canvas.set_pixmap(pix)

    def append_operation_log_entry(self, entry: str) -> None:
        self.logger_widget.append_entry(entry)

    def clear_operation_log(self) -> None:
        self.logger_widget.clear_entries()

    def append_device_log_entry(self, entry: str) -> None:
        self.device_info_widget.append_entry(entry)

    def clear_device_log(self) -> None:
        self.device_info_widget.clear_entries()

    def set_zoom(self, factor: float) -> None:
        factor = max(self._min_zoom, min(self.__effective_max_zoom(), factor))
        if abs(self._zoom - factor) < 1e-6:
            return
        self._zoom = factor
        self.__update_display()

    def __zoom_at(self, viewport_pos: QPoint, factor: float) -> None:
        if self._qimg is None or self._pixmap is None:
            return

        old_size = self.image_canvas.size()
        if old_size.width() == 0 or old_size.height() == 0:
            return

        hbar = self.scroll_area.horizontalScrollBar()
        vbar = self.scroll_area.verticalScrollBar()

        if hbar is None or vbar is None:
            return

        rel_x = (hbar.value() + viewport_pos.x()) / old_size.width()
        rel_y = (vbar.value() + viewport_pos.y()) / old_size.height()

        self.set_zoom(factor)

        new_size = self.image_canvas.size()
        if new_size.width() == 0 or new_size.height() == 0:
            return

        new_x = int(rel_x * new_size.width() - viewport_pos.x())
        new_y = int(rel_y * new_size.height() - viewport_pos.y())

        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), new_x)))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), new_y)))

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
            delta = a1.angleDelta().y()
            if delta == 0:
                return False

            factor = (
                self._zoom * self._zoom_step
                if delta > 0
                else self._zoom / self._zoom_step
            )
            self.__zoom_at(a1.position().toPoint(), factor)
            return True
        elif a1.type() == QEvent.Type.MouseButtonPress:
            if not isinstance(a1, QMouseEvent):
                return False
            if a1.button() == Qt.MouseButton.LeftButton:
                hbar = self.scroll_area.horizontalScrollBar()
                vbar = self.scroll_area.verticalScrollBar()
                if hbar is None or vbar is None:
                    return False
                self._dragging = True
                self._drag_start_pos = a1.position().toPoint()
                self._drag_start_h = hbar.value()
                self._drag_start_v = vbar.value()
                return True
        elif a1.type() == QEvent.Type.MouseMove:
            if not isinstance(a1, QMouseEvent):
                return False
            if self._dragging:
                hbar = self.scroll_area.horizontalScrollBar()
                vbar = self.scroll_area.verticalScrollBar()
                if hbar is None or vbar is None:
                    return False
                delta = a1.position().toPoint() - self._drag_start_pos
                hbar.setValue(self._drag_start_h - delta.x())
                vbar.setValue(self._drag_start_v - delta.y())
                return True
        elif a1.type() == QEvent.Type.MouseButtonRelease:
            if not isinstance(a1, QMouseEvent):
                return False
            if a1.button() == Qt.MouseButton.LeftButton and self._dragging:
                self._dragging = False
                return True
        elif a1.type() == QEvent.Type.Resize:
            pass
        return super().eventFilter(a0, a1)
