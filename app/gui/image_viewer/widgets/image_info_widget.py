from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QLabel, QWidget

from models.image import Image


class ImageInfoWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("imageInfoPanel")
        self.setContentsMargins(10, 8, 10, 16)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._info_fields: dict[str, QLabel] = {}
        self.__setup_ui()
        self.update_from_image(None)

    def __setup_ui(self) -> None:
        info_layout = QGridLayout()
        info_layout.setContentsMargins(10, 8, 10, 8)
        info_layout.setHorizontalSpacing(64)
        info_layout.setVerticalSpacing(6)

        fields = [
            ("Name", "name"),
            ("Format", "format"),
            ("Color space", "color_space"),
            ("Resolution", "resolution"),
            ("Channels", "channels"),
            ("Bit depth", "bit_depth"),
            ("Bayer pattern", "bayer_pattern"),
            ("Tensor shape", "shape"),
            ("Tensor dtype", "dtype"),
        ]

        columns = 2
        rows_per_column = max(1, (len(fields) + columns - 1) // columns)

        for idx, (title, key) in enumerate(fields):
            title_label = QLabel(f"{title}:")
            title_label.setObjectName("imageInfoTitle")
            value_label = QLabel("—")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            value_label.setObjectName("imageInfoValue")
            row = idx % rows_per_column
            col_group = idx // rows_per_column
            col_offset = col_group * 2
            info_layout.addWidget(title_label, row, col_offset)
            info_layout.addWidget(value_label, row, col_offset + 1)
            self._info_fields[key] = value_label

        for col in range(columns * 2):
            info_layout.setColumnStretch(col, 1)

        self.setLayout(info_layout)

    def update_from_image(self, image: Image | None) -> None:
        if image is None:
            self.__set_info_values(
                {
                    "name": "—",
                    "format": "—",
                    "color_space": "—",
                    "resolution": "—",
                    "channels": "—",
                    "bit_depth": "—",
                    "bayer_pattern": "—",
                    "shape": "—",
                    "dtype": "—",
                }
            )
            return

        metadata = image.metadata
        bit_depth = metadata.bit_depth if metadata.bit_depth is not None else "Unknown"
        bayer_pattern = (
            metadata.bayer_pattern.name if metadata.bayer_pattern is not None else "N/A"
        )
        resolution = f"{metadata.width} x {metadata.height}"
        image_format = image.image_format.name
        color_space = image.color_space.value if image.color_space else "—"

        self.__set_info_values(
            {
                "name": image.name,
                "format": image_format,
                "color_space": color_space,
                "resolution": resolution,
                "channels": str(image.tensor.shape[1]),
                "bit_depth": str(bit_depth),
                "bayer_pattern": bayer_pattern,
                "shape": str(image.tensor.shape),
                "dtype": str(image.tensor.dtype),
            }
        )

    def __set_info_values(self, values: dict[str, str]) -> None:
        for key, label in self._info_fields.items():
            label.setText(values.get(key, "—"))
