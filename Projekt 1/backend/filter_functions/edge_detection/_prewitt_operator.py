import numpy as np
from backend.pixel_functions import convert_to_grayscale


def prewitt_operator(image: np.ndarray) -> np.ndarray:
    """
    Performs Prewitt edge detection on 1, 3, or 4-channel images.

    Args:
        image (np.ndarray): Input image array (HW, HWC RGB, or HWC RGBA).

    Returns:
        np.ndarray: 2D array (uint8) representing edge magnitude.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray.")

    gray = convert_to_grayscale(image).astype(np.float32)
    p = np.pad(gray, ((1, 1), (1, 1)), mode='edge')

    # Gradient X: sum of right column - sum of left column
    gx = (p[:-2, 2:] + p[1:-1, 2:] + p[2:, 2:]) - \
         (p[:-2, :-2] + p[1:-1, :-2] + p[2:, :-2])

    # Gradient Y: sum of top row - sum of bottom row
    gy = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]) - \
         (p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])

    mag = np.sqrt(gx ** 2 + gy ** 2)

    max_val = mag.max()
    if max_val > 0:
        mag = (mag / max_val) * 255.0

    return mag.astype(np.uint8)