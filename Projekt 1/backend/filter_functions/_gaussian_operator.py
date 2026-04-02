import numpy as np
from backend.filter_functions import apply_filter


def gaussian_operator(
    image: np.ndarray, b: int, for_grayscale: bool = False
) -> np.ndarray:
    """
    Applies a 3x3 Gaussian-like blur filter using integer approximations.

    The kernel structure is:
    [[1, b, 1],
     [b, b^2, b],
     [1, b, 1]]

    The divisor (w) is calculated as (b + 2)^2 to ensure energy conservation.

    Args:
        image (np.ndarray): Input image array (HxW or HxWxC).
        b (int): Shape parameter (typical values: 1, 2, 3, 4).
        for_grayscale (bool): If True, converts the image to grayscale before filtering.

    Returns:
        np.ndarray: Gaussian blurred image (uint8).

    Raises:
        ValueError: If b is not a positive integer.
    """
    if b <= 0:
        raise ValueError("Parameter 'b' must be a positive integer.")

    kernel = np.array([[1, b, 1], [b, b**2, b], [1, b, 1]], dtype=np.float32)

    weight = np.sum(kernel)

    gaussian_kernel = kernel / weight

    return apply_filter(image, gaussian_kernel, for_grayscale=for_grayscale)
