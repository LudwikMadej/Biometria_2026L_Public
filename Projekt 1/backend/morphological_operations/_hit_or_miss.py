import numpy as np
from ._erode import erode
from ..pixel_functions import binarize_image


def hit_or_miss(
        image: np.ndarray,
        pattern: np.ndarray = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]),
        binarization_threshold: int = 128) -> np.ndarray:
    """
    Performs Hit-or-Miss transform. If input is not binary, it binarizes it first.

    Args:
        image (np.ndarray): Input image (binary, grayscale or RGB).
        pattern (np.ndarray): Pattern matrix with values: 1 (hit), -1 (miss), 0 (ignore).
        binarization_threshold (int): Threshold used if image needs binarization.

    Returns:
        np.ndarray: Binary image (0 or 255) showing where the pattern was found.
    """
    working_image = binarize_image(image, threshold=binarization_threshold)

    kernel_hit = (pattern == 1)
    kernel_miss = (pattern == -1)

    hit_part = erode(working_image, kernel_hit)

    complement = (working_image == 0)
    miss_part = erode(complement, kernel_miss)

    result = np.logical_and(hit_part, miss_part)

    return result.astype(np.uint8) * 255