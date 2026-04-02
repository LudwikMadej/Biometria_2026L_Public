import numpy as np
from ._convert_to_grayscale import convert_to_grayscale


def binarize_image(image_data: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Binarizes an image based on a given threshold using grayscale conversion.

    Args:
        image_data (np.ndarray): 2D, 3D (H,W,C), or 4D (N,H,W,C) array.
        threshold (int): Value [0, 255] above which pixels become white.

    Returns:
        np.ndarray: 2D binary array (0.0 or 1.0) in float32 format.

    Raises:
        TypeError: If input is not a numpy array.
    """
    if not isinstance(image_data, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image_data)}")

    gray_image = convert_to_grayscale(image_data)
    binary = (gray_image > threshold).astype(np.uint8) * 255

    return binary
