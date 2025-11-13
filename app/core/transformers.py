from __future__ import annotations
import torch

float_datatype = torch.float32


def enhance_contrast_torch(
    img: torch.Tensor, gain: float, cutoff: float = 0.5
) -> torch.Tensor:
    assert img.dtype == float_datatype
    assert cutoff <= 1
    assert gain >= 0
    assert img.ndim > 2

    if gain == 0:
        return img

    t_cutoff: torch.Tensor = torch.tensor(cutoff)

    if cutoff < 0:
        min_valid_estimated_cutoff: float = 0.4
        max_valid_estimated_cutoff: float = 0.6
        # estimated_cutoff = torch.mean(estimate_mean_torch(img))
        estimated_cutoff: torch.Tensor = torch.mean(img)
        t_cutoff = torch.clip(
            estimated_cutoff, min_valid_estimated_cutoff, max_valid_estimated_cutoff
        )

    imgf: torch.Tensor = torch.empty_like(img)
    sigmoid_min: torch.Tensor = 1 / (1 + torch.exp(gain * (t_cutoff - 0)))
    sigmoid_max: torch.Tensor = 1 / (1 + torch.exp(gain * (t_cutoff - 1)))
    imgf = 1 / (1 + torch.exp(gain * (t_cutoff - img)))
    imgf = (imgf - sigmoid_min) / (sigmoid_max - sigmoid_min)

    return imgf


def estimate_mean_torch(img: torch.Tensor) -> torch.Tensor:
    assert img.dtype == float_datatype
    assert img.ndim > 2

    mean_per_channel = torch.mean(img, dim=[i for i in range(0, img.ndim - 2)])
    return mean_per_channel
