import numpy as np
from backend.filter_functions import apply_filter


def sharpening_filter(
    image: np.ndarray, weight: float = 5.0, for_grayscale: bool = False
) -> np.ndarray:
    """
    Applies a sharpening filter to an image using a center-weighted Laplacian kernel.

    The kernel structure is:
    [[ 0, -1,  0],
     [-1,  w, -1],
     [ 0, -1,  0]]

    Args:
        image (np.ndarray): Input image array (HxW or HxWxC).
        weight (float): The center value of the kernel.
                        w=5.0 gives a strong, standard sharpen.
                        w=9.0 or higher gives a subtler effect.
        for_grayscale (bool): If True, converts to grayscale before filtering.

    Returns:
        np.ndarray: Sharpened image (uint8).
    """

    kernel = np.array([[0, -1, 0], [-1, weight, -1], [0, -1, 0]], dtype=np.float32)

    divisor = np.sum(kernel)
    if np.abs(divisor) > 1e-8:
        kernel = kernel / divisor
    else:
        raise ValueError(f"Provided forbidden value for weight: {weight}")

    return apply_filter(
        image, kernel, normalize_output=False, for_grayscale=for_grayscale
    )
