import numpy as np
from backend import convert_to_float, convert_to_int


def convert_to_grayscale(image_data: np.ndarray) -> np.ndarray:
    """
    Converts image array to a 2D grayscale array (H, W).
    Uses identity for grayscale inputs and luminosity formula for color.

    Args:
        image_data: 2D(H,W), 3D(H,W,C) or 4D(N,H,W,C) array.

    Returns:
        np.ndarray: 2D array of grayscale pixels (float32, 0-1).

    Raises:
        ValueError: For unsupported dimensions or channel counts.
        TypeError: If input is not a numpy array.
    """
    if not isinstance(image_data, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image_data)}")

    img = image_data.copy()
    img = convert_to_float(img)

    dims = len(img.shape)

    if dims == 2:
        gray = img
    elif dims == 3:
        if img.shape[2] == 1:
            gray = img.squeeze()
        elif img.shape[2] in [3, 4]:
            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        else:
            raise ValueError("Unsupported 3D channel count.")
    elif dims == 4:
        if img.shape[3] in [3, 4]:
            gray = np.dot(img[0, ..., :3], [0.299, 0.587, 0.114])
        elif img.shape[3] == 1:
            gray = img[0].squeeze()
        else:
            raise ValueError("Unsupported 4D channel count.")
    else:
        raise ValueError("Unsupported dimensions.")

    gray = convert_to_int(gray)

    return gray
