import numpy as np
from backend import convert_to_float


def stretch_brightness(img: np.ndarray, n1=0, n2=255):
    """
    Expands the brightness range of an image using a linear function.
    Formula: Jw = (J - Jmin) / (Jmax - Jmin) * (N2 - N1) + N1

    Args:
        img (np.ndarray): Input image array.
        n1 (int): Lower bound of the target range. Defaults to 0.
        n2 (int): Upper bound of the target range. Defaults to 255.

    Returns:
        np.ndarray: Image with expanded contrast.
    """
    img_float = convert_to_float(img)

    j_min = np.min(img_float)
    j_max = np.max(img_float)

    if j_max == j_min:
        return img.astype(np.uint8)

    result = ((img_float - j_min) / (j_max - j_min)) * (n2 - n1) + n1

    return np.clip(result, n1, n2).astype(np.uint8)
