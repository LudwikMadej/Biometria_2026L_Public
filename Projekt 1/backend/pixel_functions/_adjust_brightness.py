import numpy as np
from backend import convert_to_int


def adjust_brightness(image_data: np.ndarray, beta: int) -> np.ndarray:
    """
    Adjusts the brightness of an image by adding a beta value.

    Args:
        image_data: 2D, 3D (H,W,C), or 4D (N,H,W,C) array.
        beta: Value to add to each pixel (positive to brighten, negative to darken).

    Returns:
        np.ndarray: Brightness-adjusted array in float32, clipped to [0, 255].

    Raises:
        TypeError: If input is not a numpy array.
    """
    if not isinstance(image_data, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image_data)}")

    img = image_data.copy()
    img = convert_to_int(img, dtype=np.int32)  # to avoid overflow in uint8

    adjusted = np.clip(img + beta, 0, 255).astype(np.uint8)

    return adjusted
