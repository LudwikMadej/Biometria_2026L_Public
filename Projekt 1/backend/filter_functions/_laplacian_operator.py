import numpy as np
from backend.pixel_functions import convert_to_grayscale


def laplacian_operator(
    image: np.ndarray, add_original_image: bool = False
) -> np.ndarray:
    """
    Performs Laplacian edge detection on 1, 3, or 4-channel images or image sharpening using Laplacian.

    Args:
        image (np.ndarray): Input image array (HW, HWC RGB, or HWC RGBA).
        add_original_image (bool): If True, instead of edge detection performs sharpening with use of Laplacian.

    Returns:
        np.ndarray: 2D array (uint8) representing edge intensity.

    Raises:
        TypeError: If input is not a numpy array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray.")

    img = image.copy()

    if not add_original_image:
        img = convert_to_grayscale(img).astype(np.float32)
        if img.ndim == 2:
            img = img[..., np.newaxis]
    else:
        img = img.astype(np.float32)
    p = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")

    # Laplacian kernel (3x3):
    # [[ 0,  1,  0],
    #  [ 1, -4,  1],
    #  [ 0,  1,  0]]

    laplacian = (
        p[0:-2, 1:-1, :]
        + p[2:, 1:-1, :]  # Top + Bottom
        + p[1:-1, 0:-2, :]
        + p[1:-1, 2:, :]  # Left + Right
        - 4 * p[1:-1, 1:-1, :]  # Center
    )

    if add_original_image:
        res = image.copy().astype(np.float32) - laplacian
        res = np.clip(res, 0, 255)
    else:
        res = np.abs(laplacian)
        # Normalize to 0-255
        max_val = res.max()
        if max_val > 0:
            res = (res / max_val) * 255.0
        if res.shape[2] == 1:
            res = res.squeeze(axis=2)

    return res.astype(np.uint8)
