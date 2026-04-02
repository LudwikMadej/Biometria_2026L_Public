import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from backend import convert_to_int


def apply_morphology(
    image_data: np.ndarray, operation: str, kernel: np.ndarray = None
) -> np.ndarray:
    """
    Performs morphological erosion or dilation on an image.

    Works for:
        - Grayscale image: shape (H, W)
        - RGB image:       shape (H, W, 3+)

    Args:
        image_data (np.ndarray): Input image array.
        operation (str): 'erosion' or 'dilation'.
        kernel (np.ndarray, optional): Structuring element (boolean mask).
            If None, a 3x3 square kernel of True values is used.

    Returns:
        np.ndarray: Processed image of the same shape and dtype as input.

    Raises:
        TypeError: If input is not a numpy array.
        ValueError: If operation is not 'erosion' or 'dilation'.
    """
    if not isinstance(image_data, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image_data)}")

    if operation not in ("erosion", "dilation"):
        raise ValueError("Operation must be either 'erosion' or 'dilation'.")

    # default is square 3x3
    if kernel is None:
        kernel = np.ones((3, 3), dtype=bool)
    else:
        kernel = kernel.astype(bool)

    img = image_data.copy()
    img = convert_to_int(img)

    is_2d = img.ndim == 2
    if is_2d:
        img = np.expand_dims(img, axis=-1)

    # padding size
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    out_img = np.zeros_like(img)

    for c in range(img.shape[-1]):
        channel = img[..., c]

        # mode edge - copying borders' pixels, when insterting zeros filter would find there minimum -> invalid behavior
        padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")

        # sliding windows (extending channel matrix to 4D to instantly apply filter and find min for whole channel without loops)
        windows = sliding_window_view(padded, window_shape=(k_h, k_w))

        if operation == "erosion":
            neutral_val = 255
            reduce_func = np.min
        elif operation == "dilation":
            neutral_val = 0
            reduce_func = np.max

        # inserts neutral_val where filter have False so that for sure min/max function does not take this pixel as min/max respectively
        masked_windows = np.where(kernel, windows, neutral_val)

        # applying min/max with respect to Trues in filter only, flattening back to two dims
        result_channel = reduce_func(masked_windows, axis=(-2, -1))

        out_img[..., c] = result_channel.astype(channel.dtype)

    if is_2d:
        out_img = np.squeeze(out_img, axis=-1)

    return out_img
