import numpy as np
from backend import convert_to_int


def invert_image(image_data: np.ndarray) -> np.ndarray:
    """
    Inverts the colors of an image (negative).

    Args:
        image_data (np.ndarray): 2D, 3D (H,W,C), or 4D (N,H,W,C) array.

    Returns:
        np.ndarray: Inverted array in float32 format [0, 1].

    Raises:
        TypeError: If input is not a numpy array.
    """
    if not isinstance(image_data, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image_data)}")

    img = image_data.copy()
    img = convert_to_int(img)

    # Dla kanału Alpha (jeśli istnieje) zazwyczaj nie stosuje się inwersji
    if img.ndim >= 3 and img.shape[-1] == 4:
        # Odwracamy tylko RGB, zachowujemy Alpha bez zmian
        img[..., :3] = 255 - img[..., :3]
        return img

    return 255 - img
