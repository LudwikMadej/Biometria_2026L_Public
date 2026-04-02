from ._apply_morphology import apply_morphology
import numpy as np


def dilate(image_data: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Performs morphological dilation on an image using a sliding window approach.

    Works for:
        - Grayscale image: shape (H, W)
        - RGB image:       shape (H, W, 3+)

    Args:
        image_data (np.ndarray): Input image array.
        kernel (np.ndarray, optional): Structuring element (boolean mask).
            If None, a 3x3 square kernel of True values is used.

    Returns:
        np.ndarray: Dilated image of the same shape and dtype as input.

    Raises:
        TypeError: If input is not a numpy array.
    """
    return apply_morphology(image_data, "dilation", kernel)
