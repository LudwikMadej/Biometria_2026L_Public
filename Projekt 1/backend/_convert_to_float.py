import numpy as np


def convert_to_float(img: np.ndarray):
    """
    Converts image from [0,255] scale of uint8 type to [0,1] scale of float32 type.
    """
    img = img.copy()
    if np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)

    return img
