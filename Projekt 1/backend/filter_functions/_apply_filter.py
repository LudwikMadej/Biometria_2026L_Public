import numpy as np
from backend.pixel_functions import convert_to_grayscale
from backend import convert_to_float


def apply_filter(
    image: np.ndarray,
    kernel: np.ndarray,
    normalize_output: bool = True,
    for_grayscale: bool = False,
) -> np.ndarray:
    """
    Applies a 3x3 convolution filter to an image using vectorized slicing.
    Works for RGB/RGBA images.

    Args:
        image (np.ndarray): HxW grayscale or HxWxC (C=3/4) RGB/RGBA image
        kernel (np.ndarray): 3x3 filter kernel
        normalize_output: If True, scales result to [0, 255].
                          Set to True for Edge Detection (Sobel/Scharr).
                          Set to False for Blur/Sharpen.
        for_grayscale (bool): If True, converts the image to grayscale before filtering.

    Returns:
        np.ndarray: Filtered image, same shape as input (uint8)

    Raises:
        ValueError: if kernel is not 3x3
    """
    if kernel.shape != (3, 3):
        raise ValueError("Vectorized version currently supports only 3x3 kernels.")

    img = image.copy()

    if for_grayscale:
        img = convert_to_grayscale(img)

    img = convert_to_float(img)

    if img.ndim == 2:
        img = img[:, :, np.newaxis]  # HxW -> HxWx1

    _, _, C = img.shape

    # Padding
    p = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")

    output = np.zeros_like(img)

    for c in range(C):
        # convolution as adding relevant values from surrounding
        output[..., c] = (
            kernel[0, 0] * p[0:-2, 0:-2, c]
            + kernel[0, 1] * p[0:-2, 1:-1, c]
            + kernel[0, 2] * p[0:-2, 2:, c]
            + kernel[1, 0] * p[1:-1, 0:-2, c]
            + kernel[1, 1] * p[1:-1, 1:-1, c]
            + kernel[1, 2] * p[1:-1, 2:, c]
            + kernel[2, 0] * p[2:, 0:-2, c]
            + kernel[2, 1] * p[2:, 1:-1, c]
            + kernel[2, 2] * p[2:, 2:, c]
        )

    if normalize_output:
        output = np.abs(output)

        for c in range(C):
            max_val = output[..., c].max()
            if max_val > 0:
                output[..., c] = (output[..., c] / max_val) * 255.0
    else:
        output = output * 255

    output = np.clip(output, 0, 255).astype(np.uint8)

    if C == 1:
        return output[:, :, 0]
    return output
