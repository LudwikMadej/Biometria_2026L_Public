import numpy as np
from backend import convert_to_float


def log_image(img: np.ndarray):
    """
    Performs a logarithmic transformation on an image.

    Args:
        img (np.ndarray): The input image array (usually grayscale or RGB).

    Returns:
        np.ndarray: The transformed image with the specified dtype.
    """
    img_float = convert_to_float(img)

    j_max = np.max(img_float)

    img_log = np.log(1 + img_float) / np.log(1 + j_max)

    return (255.0 * img_log).astype(np.uint8)
