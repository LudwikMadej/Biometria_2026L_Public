import numpy as np
from ._dilate import dilate
from ..pixel_functions import convert_to_grayscale

def morphological_reconstruction(
        image: np.ndarray,
        operation: str = 'clear_border',
        h: int = None,
        kernel: np.ndarray = None,
        binarization_threshold : int = 128
) -> np.ndarray:
    """
    Performs specific morphological reconstruction operations.

    Args:
        image (np.ndarray): Input image.
        operation (str): 'clear_border', 'fill_holes', or 'h_dome'.
        h (int, optional): Height threshold for 'h_dome' operation.
        kernel (np.ndarray, optional): Structuring element for dilation.
        binarization_threshold:

    Returns:
        np.ndarray: Processed image.

    """

    ops = {
        "clear_border": lambda img, k, **kwargs: _clear_border(img, k),
        "fill_holes": lambda img, k, **kwargs: _fill_holes(img, k),
        "h_dome": lambda img, k, **kwargs: _h_dome(img, kwargs.get('h', 10), k)
    }

    if operation not in ops:
        raise ValueError(f"Unknown operation: {operation}. Choose from {list(ops.keys())}")

    return ops[operation](image, kernel, h=h)

def _morphological_reconstruction(marker: np.ndarray, mask: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Core engine for morphological reconstruction by dilation.
    Iteratively dilates the marker within the constraints of the mask.
    """
    if kernel is None:
        kernel = np.ones((3, 3), dtype=bool)

    current_marker = np.minimum(marker, mask)
    while True:
        dilated = dilate(current_marker, kernel)
        reconstructed = np.minimum(dilated, mask)
        if not (reconstructed != current_marker).any():
            break
        current_marker = reconstructed
    return current_marker


def _clear_border(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Suppresses structures that are connected to the image border.
    """
    mask = image
    marker = np.zeros_like(image)
    marker[0, :] = image[0, :]
    marker[-1, :] = image[-1, :]
    marker[:, 0] = image[:, 0]
    marker[:, -1] = image[:, -1]

    border_objects = _morphological_reconstruction(marker, mask, kernel)
    return image - border_objects


def _fill_holes(image: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """
    Fills dark holes inside bright structures by reconstructing from the border.
    """
    max_val = image.max()
    mask = max_val - image
    marker = np.zeros_like(image)
    marker[0, :] = mask[0, :]
    marker[-1, :] = mask[-1, :]
    marker[:, 0] = mask[:, 0]
    marker[:, -1] = mask[:, -1]

    reconstructed_bg = _morphological_reconstruction(marker, mask, kernel)
    return max_val - reconstructed_bg


def _h_dome(image: np.ndarray, h: int, kernel: np.ndarray = None) -> np.ndarray:
    """
    Extracts bright structures (domes) with height greater than h.
    """
    image = convert_to_grayscale(image)

    h = h if h is not None else 10

    mask = image
    marker = np.clip(image.astype(float) - h, 0, None).astype(image.dtype)
    reconstructed = _morphological_reconstruction(marker, mask, kernel)
    return image - reconstructed


