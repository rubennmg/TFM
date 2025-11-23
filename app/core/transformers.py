import torch
from torch import Tensor

from models.image import Image

float_datatype = torch.float32


def enhance_contrast_torch(image: Image, gain: float, cutoff: float = 0.5) -> None:
    assert image.tensor.dtype == float_datatype
    assert cutoff <= 1
    assert gain >= 0
    assert image.tensor.ndim > 2

    if gain == 0:
        return

    working_tensor: Tensor = (
        image.debayered_tensor
        if image.debayered and image.debayered_tensor is not None
        else image.original_tensor
    )

    t_cutoff: Tensor = torch.tensor(cutoff).to(
        device=image.tensor.device, non_blocking=True
    )

    if cutoff < 0:
        min_valid_estimated_cutoff: float = 0.4
        max_valid_estimated_cutoff: float = 0.6
        estimated_cutoff: Tensor = torch.mean(working_tensor).to(
            device=image.tensor.device, non_blocking=True
        )
        t_cutoff = torch.clip(
            estimated_cutoff, min_valid_estimated_cutoff, max_valid_estimated_cutoff
        )

    imgf: Tensor = torch.empty_like(working_tensor).to(
        device=image.tensor.device, dtype=image.tensor.dtype
    )

    sigmoid_min: Tensor = 1 / (1 + torch.exp(gain * (t_cutoff - 0)))
    sigmoid_max: Tensor = 1 / (1 + torch.exp(gain * (t_cutoff - 1)))

    imgf = 1 / (1 + torch.exp(gain * (t_cutoff - working_tensor)))

    imgf = (imgf - sigmoid_min) / (sigmoid_max - sigmoid_min)
    image.tensor = imgf
