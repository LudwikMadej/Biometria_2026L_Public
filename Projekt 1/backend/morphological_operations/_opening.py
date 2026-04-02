import numpy as np
from ._erode import erode
from ._dilate import dilate


def opening(image_data: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Performs morphological opening (erosion followed by dilation).

    Works for:
        - Grayscale image: shape (H, W)
        - RGB image:       shape (H, W, 3+)

    Args:
        image_data (np.ndarray): Input image array.
        kernel (np.ndarray, optional): Structuring element (boolean mask).

    Returns:
        np.ndarray: Opened image of the same shape and dtype as input.
    """
    img = image_data.copy()
    eroded_img = erode(img, kernel)
    return dilate(eroded_img, kernel)
