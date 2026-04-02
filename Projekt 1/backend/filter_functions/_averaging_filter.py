import numpy as np
from backend.filter_functions import apply_filter


def averaging_filter(
    image: np.ndarray, custom_middle: int = 1, for_grayscale: bool = False
) -> np.ndarray:
    """
    Applies an 3x3 averaging filter (box blur) to the image.

    The filter smooths the image by replacing each pixel with the average
    value of its 3x3 neighborhood.

    Args:
        image (np.ndarray): Input image array (HxW, HxWxC).
        custom_middle (int): Custom value in the middle position of filter.
        for_grayscale (bool): If True, converts the image to grayscale before filtering.

    Returns:
        np.ndarray: Blurred image as uint8 array.
    """
    kernel = np.ones((3, 3), dtype=np.float32)
    if custom_middle != 1:
        kernel[1, 1] = custom_middle
    kernel = kernel / (8 + custom_middle)
    return apply_filter(image, kernel, for_grayscale=for_grayscale)
