import torch

float_datatype = torch.float32

def enhance_contrast_torch(img, gain, cutoff=0.5):
    assert img.dtype == float_datatype
    assert cutoff <= 1
    assert gain >= 0
    assert img.ndim > 2

    if gain == 0:
        return img

    cutoff = torch.tensor(cutoff)

    if cutoff < 0:
        min_valid_estimated_cutoff = 0.4
        max_valid_estimated_cutoff = 0.6
        estimated_cutoff = torch.mean(estimate_mean_torch(img))
        cutoff = torch.clip(
            estimated_cutoff, min_valid_estimated_cutoff, max_valid_estimated_cutoff
        )

    imgf = torch.empty_like(img)
    sigmoid_min = 1 / (1 + torch.exp(gain * (cutoff - 0)))
    sigmoid_max = 1 / (1 + torch.exp(gain * (cutoff - 1)))
    imgf = 1 / (1 + torch.exp(gain * (cutoff - img)))
    imgf = (imgf - sigmoid_min) / (sigmoid_max - sigmoid_min)

    return imgf
