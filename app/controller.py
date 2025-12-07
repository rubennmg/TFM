import json
from datetime import datetime

import torch
from torch import Tensor, device

from core.registry import OPERATION_REGISTRY
from enums.color_space import ColorSpace
from gui.main_window import MainWindow
from loaders import image_loader
from models.image import Image
from models.operation_definition import get_operation_definition
from utils.error import show_error
from utils.torch import get_device

CHECKPOINT_INTERVAL = 4
MAX_SNAPSHOTS = 8
SNAPSHOT_DEVICE = "cpu"
USE_PINNED_MEMORY = True


class Controller:
    def __init__(self):
        self.image: Image | None = None
        self.window: MainWindow | None = None
        self._device: device = get_device()

        self.operations_profile: list[dict] = []
        self.operation_logs: list[str] = []

        self._snapshots: list[tuple[int, Tensor, ColorSpace]] = []

    def __log_event(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.operation_logs.append(entry)
        if self.window is not None:
            try:
                self.window.append_operation_log(entry)
            except Exception:
                pass

    def __update_viewer(self) -> None:
        if self.image is None or self.window is None:
            return
        try:
            self.window.update_image_view(self.image)
            self.window.update_operation_pipeline(self.operations_profile)
        except Exception as e:
            show_error("Update Viewer Error", str(e))

    def _make_snapshot_tensor(self, t: Tensor) -> Tensor:
        if SNAPSHOT_DEVICE == "cpu":
            t_cpu = t.detach().to("cpu")
            if USE_PINNED_MEMORY:
                try:
                    t_cpu = t_cpu.pin_memory()
                except Exception:
                    pass
            return t_cpu
        else:
            return t.detach().to(SNAPSHOT_DEVICE)

    def _store_snapshot(self, op_index: int, tensor: Tensor, color: ColorSpace) -> None:
        snap_t = self._make_snapshot_tensor(tensor)
        for i, (idx, _, _) in enumerate(self._snapshots):
            if idx == op_index:
                self._snapshots[i] = (op_index, snap_t, color)
                return

        self._snapshots.append((op_index, snap_t, color))
        self._snapshots.sort(key=lambda s: s[0])

        if len(self._snapshots) > MAX_SNAPSHOTS:
            self._snapshots = self._snapshots[-MAX_SNAPSHOTS:]

    def _get_latest_snapshot_before(
        self, op_index: int
    ) -> tuple[int, Tensor, ColorSpace]:
        chosen = self._snapshots[0]
        for s in self._snapshots:
            if s[0] <= op_index:
                chosen = s
            else:
                break

        return chosen

    def _free_cuda_cache(self) -> None:
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _clear_snapshots(self) -> None:
        for _, t, _ in self._snapshots:
            try:
                del t
            except Exception:
                pass

        self._snapshots.clear()
        self._free_cuda_cache()

    def _recompute_from(self, start_op_index: int) -> None:
        if self.image is None:
            return

        total_ops = len(self.operations_profile)

        if total_ops == 0:
            self.image.operation_result_tensor = self.image.original_tensor.detach()
            self.image.tensor = self.image.operation_result_tensor.clone()
            return

        start_idx = max(0, min(start_op_index, total_ops - 1))

        snap_idx, snap_tensor, snap_color = self._get_latest_snapshot_before(start_idx)

        compute_device = self._device
        current = snap_tensor.to(compute_device).detach()
        current_color = snap_color

        op_iter_start = snap_idx
        op_count_since_snapshot = 0

        for idx in range(op_iter_start, total_ops):
            entry = self.operations_profile[idx]
            op_name = entry.get("operation")

            if op_name is None:
                raise ValueError(
                    f"Operation entry at index {idx} missing 'operation' key."
                )

            params = dict(entry.get("params", {}))

            cls = OPERATION_REGISTRY.get(op_name)

            if cls is None:
                raise ValueError(f"Operation '{op_name}' not found in registry.")

            op = cls(**params)

            with torch.no_grad():
                out = op(current)

            if op_name == "RgbToHsv":
                current_color = ColorSpace.HSV
            elif op_name == "HsvToRgb":
                current_color = ColorSpace.RGB
            elif op_name == "ColorToGray":
                current_color = ColorSpace.GRAYSCALE
            elif op_name == "GrayToColor":
                current_color = ColorSpace.RGB
            elif op_name == "Debayer":
                current_color = ColorSpace.RGB

            current = out
            op_count_since_snapshot += 1

            if ((idx + 1) % CHECKPOINT_INTERVAL) == 0 or idx == total_ops - 1:
                self._store_snapshot(idx + 1, current, current_color)
                op_count_since_snapshot = 0
                if SNAPSHOT_DEVICE == "cpu" and current.device != torch.device("cpu"):
                    pass

        self.image.operation_result_tensor = current.detach()
        self.image.tensor = self.image.operation_result_tensor.clone().to(self._device)

        self._free_cuda_cache()

    def get_image_extensions(self) -> list[str]:
        return image_loader.get_supported_extensions()

    def load_image(self, path: str) -> None:
        if self.window is None:
            return
        try:
            self.image = image_loader.load_image(path, self._device)
            self._clear_snapshots()

            base_snap = self._make_snapshot_tensor(self.image.original_tensor.detach())

            self._snapshots.append((0, base_snap, self.image.color_space))
            self.image.operation_result_tensor = self.image.original_tensor.detach()
            self.image.tensor = self.image.operation_result_tensor.clone()

            self.operations_profile.clear()
            self.window.set_pipeline_enabled(True)
            if self.window:
                self.window.reset_image_view()
                self.window.clear_operation_pipeline()

            self.__log_event(f"Loaded image '{self.image.name}' from {path}")
            self.__update_viewer()

        except Exception as e:
            show_error("Load Image Error", str(e))

    def reset_image(self) -> None:
        if self.image is None:
            return
        try:
            self.image.operation_result_tensor = self.image.original_tensor.detach()
            self.image.tensor = self.image.original_tensor.clone()
            self.operations_profile.clear()
            self._clear_snapshots()

            base_snap = self._make_snapshot_tensor(self.image.original_tensor.detach())

            self._snapshots.append((0, base_snap, self.image.color_space))
            if self.window:
                self.window.reset_image_view()
                self.window.clear_operation_pipeline()

            self.__log_event("Image reset to original state")
            self.__update_viewer()

        except Exception as e:
            show_error("Reset Error", str(e))

    def add_pipeline_operation(self, operation_name: str) -> None:
        if self.image is None:
            show_error("Operation Error", "Load an image before adding operations.")
            return

        definition = get_operation_definition(operation_name)

        if definition is None:
            show_error(
                "Operation Error", f"Operation '{operation_name}' is unavailable."
            )
            return

        default_params = dict(definition.default_params)

        self.operations_profile.append(
            {"operation": operation_name, "params": default_params}
        )
        op_idx = len(self.operations_profile) - 1

        start_idx = max(0, op_idx - CHECKPOINT_INTERVAL)

        try:
            self._recompute_from(start_idx)
        except Exception as e:
            self.operations_profile.pop()
            show_error("Operation Error", str(e))
            return

        self.__log_event(f"Added operation '{operation_name}'")
        self.__update_viewer()

        if self.window:
            self.window.right_panel.push_operation_control(operation_name, op_idx)

    def apply_operation(
        self, operation_name: str, *, operation_idx: int | None = None, **params
    ) -> None:
        if self.image is None:
            return

        total = len(self.operations_profile)

        if operation_idx is None:
            operation_idx = total - 1 if total > 0 else 0

        if operation_idx < 0 or operation_idx > total:
            show_error("Operation Error", "Invalid operation index.")
            return

        is_new = operation_idx == total

        if is_new:
            self.operations_profile.append(
                {"operation": operation_name, "params": dict(params)}
            )
            op_idx = total
        else:
            entry = self.operations_profile[operation_idx]
            if entry["operation"] != operation_name:
                show_error("Operation Error", "Operation mismatch at index.")
                return
            entry["params"] = dict(params)
            op_idx = operation_idx

        start_idx = max(0, op_idx - CHECKPOINT_INTERVAL)

        try:
            self._recompute_from(start_idx)

        except Exception as e:
            if is_new:
                self.operations_profile.pop()
            else:
                pass

            show_error("Operation Error", str(e))
            return

        self.__log_event(f"Applied {operation_name} with params {params}")
        self.__update_viewer()

    def remove_last_operation(self) -> None:
        if self.image is None or not self.operations_profile:
            return

        removed = self.operations_profile.pop()
        last_idx = max(0, len(self.operations_profile) - 1)

        self._recompute_from(last_idx)
        self.__log_event(f"Removed operation {removed.get('operation')}")

        if self.window:
            self.window.right_panel.pop_operation_control()

        self._free_cuda_cache()
        self.__update_viewer()

    def export_profile(self, path: str) -> None:
        if self.image is None:
            return
        try:
            with open(path, "w") as f:
                json.dump(self.operations_profile, f, indent=4)

            self.__log_event(f"Exported profile to {path}")

        except Exception as e:
            show_error("Export Profile Error", str(e))
