from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QToolButton,
    QSizePolicy,
    QFrame,
    QVBoxLayout,
    QLayout,
)


class CollapsibleSection(QWidget):
    def __init__(self, title: str, content: QWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.setObjectName("collapsibleHeader")
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self._area = QFrame()
        self._area.setFrameShape(QFrame.Shape.NoFrame)
        self._area.setVisible(False)
        self._area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._area_layout = QVBoxLayout()
        self._area_layout.setContentsMargins(0, 0, 0, 0)
        self._area_layout.setSpacing(0)
        self._area_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        self._area.setLayout(self._area_layout)

        content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._area_layout.addWidget(content)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        layout.addWidget(self._toggle)
        layout.addWidget(self._area)
        self.setLayout(layout)

        self._toggle.toggled.connect(self._on_toggled)

    def _on_toggled(self, expanded: bool) -> None:
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._area.setVisible(expanded)
