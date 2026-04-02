import numpy as np
from backend.pixel_functions import convert_to_grayscale
from backend import convert_to_int


def equalize_histograms_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Performs histogram equalization based on normalized histogram
    and cumulative distribution function (CDF).

    Works for 2D grayscale images.

    Args:
        image (np.ndarray): Grayscale image array (H, W).
                            Values expected in range [0, 255] or float [0, 1].

    Returns:
        np.ndarray: Histogram-equalized image (uint8).

    Raises:
        TypeError: If input is not numpy array.
        ValueError: If input is not 2D grayscale.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")

    img = image.copy()

    if img.ndim != 2:
        img = convert_to_grayscale(img)

    img = convert_to_int(img)

    L = 256
    n = img.size

    hist = np.bincount(img.ravel(), minlength=L)

    p_r = hist / n

    cdf = np.cumsum(p_r)

    s_k = np.floor((L - 1) * cdf).astype(np.uint8)

    equalized_img = s_k[img]

    return equalized_img
