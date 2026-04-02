import numpy as np
from backend import convert_to_int


def adjust_contrast(image_data: np.ndarray, alpha: float) -> np.ndarray:
    """
    Adjusts the contrast of an image by scaling pixel values.

    Args:
        image_data (np.ndarray): 2D, 3D (H,W,C), or 4D (N,H,W,C) array.
        alpha (float): Contrast factor (>1 for higher, <1 for lower contrast).

    Returns:
        np.ndarray: Contrast-adjusted array in float32, clipped to [0, 1].

    Raises:
        TypeError: If input is not a numpy array.
    """
    if not isinstance(image_data, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image_data)}")

    img = image_data.copy()
    img = convert_to_int(img, dtype=np.int32)  # to avoid overflow in uint8

    adjusted = np.clip((img - 128) * alpha + 128, 0, 255).astype(np.uint8)

    return adjusted
