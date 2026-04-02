import numpy as np
from ._erode import erode
from ._dilate import dilate


def morphological_gradient(
    image_data: np.ndarray, kernel: np.ndarray = None
) -> np.ndarray:
    """
    Computes the morphological gradient of an image (dilation - erosion).
    Highlights the edges of objects.

    Works for:
        - Grayscale image: shape (H, W)
        - RGB image:       shape (H, W, 3+)

    Args:
        image_data (np.ndarray): Input image array.
        kernel (np.ndarray, optional): Structuring element (boolean mask).

    Returns:
        np.ndarray: Gradient image of the same shape and dtype as input.
    """
    img = image_data.copy()
    dilated_img = dilate(img, kernel)
    eroded_img = erode(img, kernel)

    # dilated_img >= eroded_img for every pixel (pixel-wise)
    return dilated_img - eroded_img
