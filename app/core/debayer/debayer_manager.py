from typing import Dict, List, Tuple

import torch
from torch import nn

from core.debayer.modules.debayer2x2 import Debayer2x2
from core.debayer.modules.debayer3x3 import Debayer3x3
from core.debayer.modules.debayer5x5 import Debayer5x5
from core.debayer.modules.debayerSplit import DebayerSplit
from enums.image_formats import ImageFormat
from enums.layouts import Layout
from models.image import Image

_DEBAYER_REGISTRY: Dict[str, type[nn.Module]] = {
    "debayer2x2": Debayer2x2,
    "debayer3x3": Debayer3x3,
    "debayer5x5": Debayer5x5,
    "debayersplit": DebayerSplit,
}

_DEBAYER_DISPLAY_NAMES: Dict[str, str] = {
    "debayer2x2": "Debayer 2x2",
    "debayer3x3": "Debayer 3x3",
    "debayer5x5": "Debayer 5x5",
    "debayersplit": "Debayer Split",
}


_ImageDebayerCache = Dict[str, Tuple[Tuple[object, ...], nn.Module]]


def _normalise_algorithm_name(name: str) -> str:
    return name.strip().lower()


def _resolve_algorithm(name: str) -> type[nn.Module]:
    try:
        return _DEBAYER_REGISTRY[_normalise_algorithm_name(name)]
    except KeyError as exc:
        available = ", ".join(sorted(_DEBAYER_REGISTRY))
        raise ValueError(
            f"Unknown debayer algorithm '{name}'. Available: {available}"
        ) from exc


def _ensure_raw_image(image: Image) -> None:
    if image.image_format is not ImageFormat.RAW or image.metadata is None:
        raise ValueError("Debayering can only be applied to RAW (1xHxW) images.")


def list_debayer_algorithms() -> List[Tuple[str, str]]:
    """Return available debayer algorithms as ``(key, label)`` pairs."""
    return [
        (name, _DEBAYER_DISPLAY_NAMES.get(name, name))
        for name in _DEBAYER_REGISTRY.keys()
    ]


def get_cached_debayer(image: Image, algorithm_name: str) -> nn.Module:
    """Return a cached Debayer module configured for ``image``.

    The cache is stored on the ``Image`` instance to avoid repeatedly creating
    modules when toggling transformations on the same tensor.
    """
    _ensure_raw_image(image)

    module_cls = _resolve_algorithm(algorithm_name)
    dev = image.tensor.device
    dtype = image.tensor.dtype
    layout = image.metadata.bayer_pattern or Layout.RGGB
    cache_key = (
        module_cls.__name__,
        layout,
        dev.type,
        getattr(dev, "index", None),
        dtype,
    )

    cache: _ImageDebayerCache = getattr(image, "_debayer_cache", {})
    cached = cache.get(_normalise_algorithm_name(algorithm_name))
    if cached is None or cached[0] != cache_key:
        module = module_cls(layout=layout).to(
            device=dev, dtype=dtype, non_blocking=True
        )
        cache[_normalise_algorithm_name(algorithm_name)] = (cache_key, module)
        setattr(image, "_debayer_cache", cache)
        return module

    return cached[1]


def apply_debayer(image: Image, algorithm_name: str) -> None:
    """Apply the selected Debayer algorithm in-place on ``image``.

    The method mirrors the behaviour of ``apply_debayer*_`` helpers so existing
    callers can transition to this orchestrator without losing functionality.
    """
    _ensure_raw_image(image)

    module: nn.Module = get_cached_debayer(image, algorithm_name)

    input_tensor = image.tensor

    # add batch dimension to match debayer modules
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = module(input_tensor).to(
            device=image.tensor.device, dtype=image.tensor.dtype, non_blocking=True
        )

    if output.shape[0] == 1:
        output = output.squeeze(0)

    image.tensor = output.clone()
    if not image.tensor.is_contiguous():
        image.tensor = image.tensor.contiguous()

    image.debayered_tensor = output.clone()
    image.debayered = True
