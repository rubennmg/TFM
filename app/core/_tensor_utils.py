from torch import Tensor

# [B, C, H, W]
BATCH_DIM = 0
CHANNEL_DIM = 1
HEIGHT_DIM = 2
WIDTH_DIM = 3


def _is_tensor_a_torch_image(x: Tensor) -> bool:
    """Check if a tensor is a torch image.

    A torch image is defined as a 3D tensor with shape (C, H, W) where C is
    the number of channels (1, 3, or 4), and H and W are the height and width
    of the image respectively.
    In this context, we consider a batch of images too, so a 4D tensor with shape
    (B, C, H, W) is how we define a torch image.

    Args:
        x (Tensor): Input tensor to check.
    Returns:
        bool: True if the tensor is a torch image, False otherwise.
    """
    return x.ndim == 4


def assert_image_tensor(x: Tensor) -> None:
    """Assert that a tensor is a torch image.

    Args:
        x (Tensor): Input tensor to check.
    Raises:
        TypeError: If the input is not a tensor.
        ValueError: If the tensor does not have at least 2 dimensions.
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected input of type Tensor, got {type(x)}")
    if not _is_tensor_a_torch_image(x):
        raise TypeError("Tensor is not a Torch image.")


def assert_color_image_tensor(x: Tensor) -> None:
    """Assert that a tensor is a color torch image.

    Args:
        x (Tensor): Input tensor to check.
    Raises:
        ValueError: If the tensor does not have 3 channels.
    """
    if x.shape[CHANNEL_DIM] != 3:
        raise ValueError(
            f"Expected color image with 3 channels, got {x.shape[CHANNEL_DIM]} channels."
        )


def assert_grayscale_image_tensor(x: Tensor) -> None:
    """Assert that a tensor is a grayscale torch image.

    Args:
        x (Tensor): Input tensor to check.
    Raises:
        ValueError: If the tensor does not have 1 channel.
    """
    if x.shape[CHANNEL_DIM] != 1:
        raise ValueError(
            f"Expected grayscale image with 1 channel, got {x.shape[CHANNEL_DIM]} channels."
        )


def assert_real_valued_tensor(x: Tensor) -> None:
    """Assert that a tensor is of a real-valued type.

    Args:
        x (Tensor): Input tensor to check.
    Raises:
        ValueError: If the tensor is not of a real-valued type.
    """
    if not x.is_floating_point():
        raise ValueError("Tensor is not of a real-valued type.")


def assert_integer_valued_tensor(x: Tensor) -> None:
    """Assert that a tensor is of an integer-valued type.

    Args:
        x (Tensor): Input tensor to check.
    Raises:
        ValueError: If the tensor is not of an integer-valued type.
    """
    if x.is_floating_point():
        raise ValueError("Tensor is not of an integer-valued type.")
