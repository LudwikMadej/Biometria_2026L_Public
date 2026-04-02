import numpy as np
from ._erode import erode
from ._opening import opening
from ..pixel_functions import binarize_image


def skeletonize(image: np.ndarray, kernel: np.ndarray = None, binarization_threshold : int = 128) -> np.ndarray:
    """
    Performs morphological skeletonization using the Lantuejoul's algorithm.

    The skeleton is reconstructed as the union of successive 'internal parts' 
    found by subtracting the opening of an eroded image from the eroded image itself.

    Args:
        image (np.ndarray): Input binary or grayscale image.
        kernel (np.ndarray, optional): Structuring element. If None, a 3x3 cross is typical.

    Returns:
        np.ndarray: Skeletonized image (1-pixel width features).
    """
    image = binarize_image(image.copy(), binarization_threshold)
    skeleton = np.zeros_like(image)


    if kernel is None:
        kernel = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=bool)

    while True:
        opened = opening(image, kernel)

        temp = np.subtract(image, opened)

        skeleton = np.maximum(skeleton, temp)

        image = erode(image, kernel)

        if not image.any():
            break

    return skeleton