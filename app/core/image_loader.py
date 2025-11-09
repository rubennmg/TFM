import torch
import numpy as np

from pathlib import Path
from typing import Union

torch_float_datatype: torch.dtype = torch.float32

def load_image(path: Union[str, Path]) -> torch.Tensor:
    path = Path(path)
    if path.suffix.lower() == ".raw":
        width: int = 4096 # default -- CHANGE
        height: int = 2168 # default -- CHANGE
        with open(path, "rb") as f:
            raw_data: np.typing.NDArray[np.uint16] = np.fromfile(f, dtype=np.uint16).reshape((height, width))
        t_img: torch.Tensor = torch.tensor(raw_data, dtype=torch_float_datatype) / 4095.0  # normalize assuming 12-bit depth 2^12 - 1
        t_img = t_img.unsqueeze(0).unsqueeze(0)  # 1x1xHxW
        return t_img
    else:
        from PIL import Image
        img: Image.Image = Image.open(path).convert("RGB")
        arr: np.typing.NDArray[np.float32] = np.asarray(img, dtype=np.float32) / 255.0
        return torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
