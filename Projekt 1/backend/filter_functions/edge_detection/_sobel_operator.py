import numpy as np
from backend.pixel_functions import convert_to_grayscale


def sobel_operator(image: np.ndarray) -> np.ndarray:
    """
    Apply Sobel edge detection to a 2D or 3D numpy array.
    Args: image (np.ndarray): HxW or HxWxC array.
    Returns: np.ndarray: Edge magnitude as uint8.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray.")

    # Convert and pad for 3x3 kernel
    gray = convert_to_grayscale(image).astype(np.float32)
    p = np.pad(gray, ((1, 1), (1, 1)), mode='edge')

    # Manual convolution via vectorized slicing
    gx = (p[2:, 2:] + 2 * p[1:-1, 2:] + p[:-2, 2:]) - \
         (p[2:, :-2] + 2 * p[1:-1, :-2] + p[:-2, :-2])

    gy = (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:]) - \
         (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])

    # Magnitude calculation and normalization
    mag = np.sqrt(gx ** 2 + gy ** 2)
    if mag.max() > 0:
        mag = (mag / mag.max()) * 255

    return mag.astype(np.uint8)