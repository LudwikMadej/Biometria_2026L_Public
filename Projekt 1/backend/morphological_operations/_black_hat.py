import numpy as np
from ._erode import erode
from ._dilate import dilate


def black_hat(image_data: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Performs morphological Black-Hat transform (Closing - Original).
    Highlights small dark elements on a bright/uneven background.

    Works for:
        - Grayscale image: shape (H, W)
        - RGB image:       shape (H, W, 3+)

    Args:
        image_data (np.ndarray): Input image array.
        kernel (np.ndarray, optional): Structuring element (boolean mask).

    Returns:
        np.ndarray: Black-Hat transformed image of the same shape and dtype.
    """
    img = image_data.copy()
    dilated = dilate(img, kernel)
    closed = erode(dilated, kernel)

    # safe deduction
    return closed - image_data
