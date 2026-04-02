import numpy as np


def exponentiate_image(image_data: np.ndarray, gamma: float) -> np.ndarray:
    """
    Adjusts image brightness using gamma correction based on the formula:
    Jw(x, y) = 255 * (J(x, y) / Jmax)^gamma

    Args:
        image_data: Input numpy array (2D, 3D, or 4D).
        gamma: The power exponent.
               gamma < 1: brightens the image.
               gamma > 1: darkens the image.

    Returns:
        np.ndarray: Gamma-corrected array in uint8, clipped to [0, 255].

    Raises:
        TypeError: If input is not a numpy array.
    """
    if not isinstance(image_data, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image_data)}")

    img_float = image_data.astype(np.float32)

    j_max = 255.0

    corrected = 255.0 * np.power((img_float / j_max), gamma)

    return np.clip(corrected, 0, 255).astype(np.uint8)
