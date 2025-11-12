from pathlib import Path
from typing import Union

import numpy as np
import rawpy
import torch
from numpy.typing import NDArray
from PIL import Image

torch_float_datatype: torch.dtype = torch.float32


def load_image(path: Union[str, Path]) -> torch.Tensor:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".raw":
        width: int = 4096  # default -> CHANGE
        height: int = 2168  # default -> CHANGE
        with open(path, "rb") as f:
            raw_data: NDArray[np.uint16] = np.fromfile(f, dtype=np.uint16).reshape(
                (height, width)
            )

        t_img: torch.Tensor = (
            torch.tensor(raw_data, dtype=torch_float_datatype) / 4095.0
        )  # 12-bit depth normalization
        t_img = t_img.unsqueeze(0).unsqueeze(0)  # -> 1x1xHxW
        return t_img

    elif suffix in {".nef", ".cr2", ".arw", ".dng", ".rw2", ".orf"}:
        with rawpy.imread(str(path)) as raw:
            rgb: NDArray[np.uint16] = raw.postprocess(
                use_camera_wb=True, no_auto_bright=True, output_bps=16
            )
        arr: NDArray[np.float32] = rgb.astype(np.float32) / 65535.0
        t_img: torch.Tensor = (
            torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        )  # 1x3xHxW
        return t_img

    else:
        img: Image.Image = Image.open(path).convert("RGB")
        arr: NDArray[np.float32] = np.asarray(img, dtype=np.float32) / 255.0
        t_img: torch.Tensor = (
            torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        )  # 1x3xHxW
        return t_img
