from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from models.operation_definition import (
    OperationDefinition,
    get_operation_choices_by_category,
)


class OperationPipeline(QWidget):
    add_operation_requested = pyqtSignal(str)
    remove_operation_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._operations: list[dict] = []
        self._definitions_by_name: dict[str, OperationDefinition] = {}
        self._build_definition_lookup()
        self._setup_ui()

    def _build_definition_lookup(self) -> None:
        for definitions in get_operation_choices_by_category().values():
            for definition in definitions:
                self._definitions_by_name[definition.name] = definition

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel("Pipeline")
        title.setObjectName("operationPipelineTitle")
        layout.addWidget(title)

        self._list = QListWidget()
        self._list.setObjectName("operationPipelineList")
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout.addWidget(self._list, stretch=1)

        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(2)

        self._add_button = QPushButton("+")
        self._add_button.setStyleSheet("width: 30px; height: 18px;")
        self._add_button.setToolTip("Add operation to pipeline")
        self._add_button.clicked.connect(self._on_add_clicked)
        self._add_button.setEnabled(False)

        self._remove_button = QPushButton("-")
        self._remove_button.setStyleSheet("width: 30px; height: 18px;")
        self._remove_button.setToolTip("Remove last operation")
        self._remove_button.clicked.connect(self.remove_operation_requested.emit)
        self._remove_button.setEnabled(False)

        controls_row.addStretch()
        controls_row.addWidget(self._add_button)
        controls_row.addWidget(self._remove_button)
        layout.addLayout(controls_row)

        self.setLayout(layout)

    def clear(self) -> None:
        self._operations.clear()
        self._list.clear()
        self._remove_button.setEnabled(False)

    def set_operations(self, operations: list[dict]) -> None:
        self._operations = operations.copy()
        self._list.clear()
        for idx, op in enumerate(self._operations, start=1):
            op_name = op.get("operation", "unknown")
            definition = self._definitions_by_name.get(op_name)
            label = definition.label if definition else op_name
            params = op.get("params", {})
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            if not params_str:
                params_str = "default"
            text = f"{idx}. {label} ({params_str})"
            item = QListWidgetItem(text)
            if idx == len(self._operations):
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self._list.addItem(item)

        self._remove_button.setEnabled(bool(self._operations))

    def set_interactions_enabled(self, enabled: bool) -> None:
        self._add_button.setEnabled(enabled)
        if not enabled:
            self.clear()

    def _on_add_clicked(self) -> None:
        if not self._add_button.isEnabled():
            return

        categories = get_operation_choices_by_category()
        if not categories:
            return

        menu = QMenu(self)
        has_entries = False
        for category in sorted(categories.keys()):
            definitions = categories[category]
            if len(categories) > 1:
                category_menu = menu.addMenu(category)
                if category_menu is None:
                    continue
                self._populate_menu(category_menu, definitions)
            else:
                self._populate_menu(menu, definitions)
            if definitions:
                has_entries = True

        if not has_entries:
            return

        size_hint = menu.sizeHint()
        button_top_left = self._add_button.mapToGlobal(
            self._add_button.rect().topLeft()
        )
        pos = button_top_left - QPoint(0, size_hint.height() + 4)
        action = menu.exec(pos)

        if action is None:
            return

        operation_name = action.data()
        if isinstance(operation_name, str):
            self.add_operation_requested.emit(operation_name)

    def _populate_menu(
        self, menu: QMenu, definitions: list[OperationDefinition]
    ) -> None:
        for definition in definitions:
            action = menu.addAction(definition.label)
            if action is None:
                continue
            action.setData(definition.name)
