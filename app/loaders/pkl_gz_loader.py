import pickle
import gzip
from loaders.base_loader import ImageLoader
from models.enums.color_space import ColorSpace
from models.enums.image_formats import ImageFormat
from models.enums.layouts import Layout
from models.image import Image
from models.metadata import Metadata
import torch
import numpy as np


class PklGzLoader(ImageLoader):
    """Loader for .pkl.gz files.

    Args:
        ImageLoader (abc): Base ImageLoader class.
    """

    @property
    def extensions(self) -> list[str]:
        return ["gz"]

    @property
    def formats(self) -> list[ImageFormat]:
        return [ImageFormat.PKL_GZ]

    def load(self, path: str, device) -> Image:
        with open(path, "rb") as f:
            with gzip.GzipFile(fileobj=f, mode="rb") as gz:
                data = pickle.load(gz)

        arr: np.ndarray = data.get("array")
        bit_depth: int | None = data.get("bit_depth")
        bayer_pattern = data.get("bayer_pattern")

        if arr is None:
            raise ValueError("Pickle file does not contain 'array' key")

        scale: float = 1.0
        if bit_depth:
            scale = 1.0 / (2 ** int(bit_depth) - 1)

        np_arr = np.asarray(arr)

        if np_arr.ndim == 2:
            t = (
                torch.from_numpy(np_arr)
                .to(dtype=torch.float32)
                .unsqueeze(0)
                .mul_(scale)
            )
        elif np_arr.ndim == 3:
            t = (
                torch.from_numpy(np_arr)
                .to(dtype=torch.float32)
                .permute(2, 0, 1)
                .mul_(scale)
            )
        else:
            raise ValueError(f"Unsupported array ndim: {np_arr.ndim}")

        t = t.to(device=device)

        if not t.is_contiguous():
            t = t.contiguous()

        layout = None
        if bayer_pattern:
            try:
                layout = Layout[bayer_pattern]
            except Exception:
                layout = None

        metadata = Metadata(
            width=int(np_arr.shape[1]),
            height=int(np_arr.shape[0]),
            bit_depth=bit_depth,
            bayer_pattern=layout,
        )

        return Image(
            tensor=t,
            original_tensor=t.clone(),
            operation_result_tensor=t.clone(),
            path=path,
            name=path.split("/")[-1],
            image_format=ImageFormat.PKL_GZ,
            color_space=ColorSpace.GRAYSCALE if t.shape[0] == 1 else ColorSpace.RGB,
            metadata=metadata,
        )
