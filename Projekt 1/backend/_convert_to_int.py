import numpy as np


def convert_to_int(img: np.ndarray, dtype=np.uint8):
    """
    Converts image from [0,1] scale of float32 type to [0,255] scale of uint8 type.
    """
    img = img.copy()
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img * 255.0, 0, 255).astype(dtype)
    else:
        img = img.astype(dtype)

    return img
