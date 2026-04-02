import numpy as np
from backend.pixel_functions import convert_to_grayscale


def roberts_cross(image: np.ndarray) -> np.ndarray:
    """
    Performs Roberts Cross edge detection on 1, 3, or 4-channel images.

    Args:
        image (np.ndarray): Input image array (HW, HWC RGB, or HWC RGBA).

    Returns:
        np.ndarray: 2D array (uint8) representing edge magnitude.

    Raises:
        TypeError: If input is not a numpy array.
        ValueError: If image dimensions or channels are unsupported.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray.")

    gray = convert_to_grayscale(image)
    if np.issubdtype(gray.dtype, np.integer):
        gray = gray.astype(np.float32) / 255.0
    padded = np.pad(gray, ((0, 1), (0, 1)), mode="edge")

    gx = padded[:-1, :-1] - padded[1:, 1:]
    gy = padded[:-1, 1:] - padded[1:, :-1]

    mag = np.sqrt(gx**2 + gy**2)

    max_val = mag.max()
    if max_val > 0:
        mag = (mag / max_val) * 255.0

    return mag.astype(np.uint8)
