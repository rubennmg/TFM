from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QDoubleSpinBox
from PyQt6.QtCore import Qt
from torch import Tensor


class ControlsPanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout: QVBoxLayout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        label_title: QLabel = QLabel("Transformations:")
        label_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(label_title)

        # Gain control
        self.label_gain: QLabel = QLabel("Gain:")
        self.spin_gain: QDoubleSpinBox = QDoubleSpinBox()
        self.spin_gain.setRange(0.0, 50.0)
        self.spin_gain.setValue(10.0)
        self.spin_gain.setSingleStep(0.5)

        # Cutoff control
        self.label_cutoff: QLabel = QLabel("Cutoff:")
        self.spin_cutoff: QDoubleSpinBox = QDoubleSpinBox()
        self.spin_cutoff.setRange(-1.0, 1.0)
        self.spin_cutoff.setValue(0.5)
        self.spin_cutoff.setSingleStep(0.1)

        # Apply button
        self.button_apply: QPushButton = QPushButton("Apply sigmoid contrast")
        self.button_apply.clicked.connect(self._on_apply_clicked)

        layout.addWidget(self.label_gain)
        layout.addWidget(self.spin_gain)
        layout.addWidget(self.label_cutoff)
        layout.addWidget(self.spin_cutoff)
        layout.addWidget(self.button_apply)
        layout.addStretch()

        self.setLayout(layout)

    def _on_apply_clicked(self) -> None:
        gain: float = float(self.spin_gain.value())
        cutoff: float = float(self.spin_cutoff.value())
        tensor: Tensor | None = self.controller.apply_contrast(gain, cutoff)
        if tensor is not None:
            self.controller.update_viewer(tensor)
