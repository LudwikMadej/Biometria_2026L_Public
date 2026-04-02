import numpy as np
from ._erode import erode
from ._dilate import dilate


def closing(image_data: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Performs morphological closing (dilation followed by erosion).

    Works for:
        - Grayscale image: shape (H, W)
        - RGB image:       shape (H, W, 3+)

    Args:
        image_data (np.ndarray): Input image array.
        kernel (np.ndarray, optional): Structuring element (boolean mask).

    Returns:
        np.ndarray: Closed image of the same shape and dtype as input.
    """
    img = image_data.copy()
    dilated_img = dilate(img, kernel)
    return erode(dilated_img, kernel)
