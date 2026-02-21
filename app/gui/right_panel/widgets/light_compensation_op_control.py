import os
import numpy as np
import pickle
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.right_panel.widgets.filter_op_control import _FloatSlider
from models.float_param_spec import FloatParamSpec


class LightCompensationOperationControl(QWidget):
    """Control widget for LightCompensation operation.

    Args:
        QWidget (QWidget): Base Qt widget.
    """

    def __init__(
        self,
        controller,
        operation_index: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.controller = controller
        self.operation_index = operation_index
        self.light_gain_matrix: torch.Tensor | None = None
        self._suppress_emit: bool = False

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setToolTip("Apply light compensation using a gain matrix")

        # Title
        self._title = QLabel("Light Compensation")
        self._title.setObjectName("operationControlTitle")

        # Matrix info display
        self._matrix_info = QTextEdit()
        self._matrix_info.setReadOnly(True)
        self._matrix_info.setMaximumHeight(80)
        self._matrix_info.setMinimumHeight(80)
        self._matrix_info.setPlaceholderText("No matrix loaded")
        self._matrix_info.setObjectName("matrixInfoDisplay")
        self._matrix_info.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        # Load button
        self._load_button = QPushButton("Load Matrix File")
        self._load_button.clicked.connect(self._on_load_clicked)

        # Strength slider
        strength_param = FloatParamSpec(
            key="strength",
            label="Strength",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            default=1.0,
        )
        self._strength_slider = _FloatSlider(
            strength_param.label,
            strength_param.minimum,
            strength_param.maximum,
            strength_param.step,
            strength_param.default,
            self,
        )

        self._strength_slider._slider.sliderReleased.connect(self._apply_operation)
        self._strength_slider._spin.editingFinished.connect(self._apply_operation)
        self._strength_slider._spin.valueChanged.connect(self._on_slider_value_changed)

        container = QWidget(self)
        container.setObjectName("operationControl")

        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(10, 18, 10, 8)
        container_layout.setSpacing(10)
        container_layout.addWidget(self._title)
        container_layout.addWidget(self._matrix_info)
        container_layout.addWidget(self._load_button)
        container_layout.addWidget(self._strength_slider)
        container.setLayout(container_layout)

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(10)
        root_layout.addWidget(container)

        self.setLayout(root_layout)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def _load_matrix_from_file(self, filename: str) -> torch.Tensor:
        """Load light gain compensation matrix from file.

        Args:
            filename (str): Path to .npy or .pkl file.

        Returns:
            torch.Tensor: 2D tensor with gain values.

        Raises:
            ValueError: If file format is not supported or matrix is not 2D.
        """
        if filename.endswith(".npy"):
            light_gain_compensation = np.load(filename)
        elif filename.endswith(".pkl"):
            with open(filename, "rb") as f:
                light_gain_compensation = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filename}")

        if isinstance(light_gain_compensation, np.ndarray):
            light_gain_compensation = torch.from_numpy(light_gain_compensation).float()

        if light_gain_compensation.ndim != 2:
            raise ValueError(
                f"light_gain_compensation must be 2D, got {light_gain_compensation.ndim}D"
            )

        return light_gain_compensation

    def _on_load_clicked(self) -> None:
        """Handle load button click to open file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Light Gain Matrix",
            "",
            "Matrix Files (*.npy *.pkl);;NumPy Files (*.npy);;Pickle Files (*.pkl);;All Files (*)",
        )

        if not file_path:
            return

        try:
            self.light_gain_matrix = self._load_matrix_from_file(file_path)

            filename = os.path.basename(file_path)
            H, W = self.light_gain_matrix.shape
            min_val = self.light_gain_matrix.min().item()
            max_val = self.light_gain_matrix.max().item()

            info_text = (
                f"File: {filename}\n"
                f"Shape: {H} × {W}\n"
                f"Range: [{min_val:.4f}, {max_val:.4f}]"
            )
            self._matrix_info.setText(info_text)

            self._apply_operation()

        except Exception as e:
            self._matrix_info.setText(f"Error loading file:\n{str(e)}")
            self.light_gain_matrix = None

    def _on_slider_value_changed(self, _: float) -> None:
        """Handle slider value change."""
        if self._suppress_emit:
            return
        self._apply_operation()

    def _apply_operation(self) -> None:
        """Apply the light compensation operation with current parameters."""
        if self._suppress_emit:
            return

        strength = self._strength_slider.value()

        self.controller.apply_operation(
            "LightCompensation",
            operation_idx=self.operation_index,
            light_gain_compensation=self.light_gain_matrix,
            strength=strength,
        )

    def reset(self) -> None:
        """Reset the control to default state."""
        self._suppress_emit = True
        try:
            self.light_gain_matrix = None
            self._matrix_info.setText("")
            self._strength_slider.setValue(1.0)
        finally:
            self._suppress_emit = False
