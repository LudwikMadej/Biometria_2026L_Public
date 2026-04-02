import numpy as np
from ._erode import erode
from ._dilate import dilate


def top_hat(image_data: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Performs morphological White Top-Hat transform (Original - Opening).
    Highlights small bright elements on a dark/uneven background.

    Works for:
        - Grayscale image: shape (H, W)
        - RGB image:       shape (H, W, 3+)

    Args:
        image_data (np.ndarray): Input image array.
        kernel (np.ndarray, optional): Structuring element (boolean mask).

    Returns:
        np.ndarray: Top-Hat transformed image of the same shape and dtype.
    """
    img = image_data.copy()
    eroded = erode(img, kernel)
    opened = dilate(eroded, kernel)

    # safe subtraction
    return image_data - opened
