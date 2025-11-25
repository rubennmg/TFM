from typing import Dict, Type, TypeVar

from core.image_operation import ImageOperation

T = TypeVar("T", bound=ImageOperation)

OPERATION_REGISTRY: Dict[str, Type[ImageOperation]] = {}


def register_operation(cls: Type[T]) -> Type[T]:
    """Register an image operation class in the operation registry.

    Args:
        cls (ImageOperation): The image operation class to register.

    Returns:
        ImageOperation: The registered image operation class.
    """
    OPERATION_REGISTRY[cls.__name__] = cls
    return cls
