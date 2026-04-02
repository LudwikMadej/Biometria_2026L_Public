import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from ._convert_to_float import convert_to_float


def plot_numpy_image(image_data: np.ndarray) -> None:
    """
    Plots a NumPy array as an image in its actual pixel size,
    without borders, axes, or titles.

    Args:
        image_data (np.ndarray): 2D(H,W), 3D(H,W,C) or 4D(N,H,W,C) array.

    Raises:
        ValueError: For unsupported dimensions or invalid channel configs.
        TypeError: If input is not a numpy array.
    """
    if not isinstance(image_data, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image_data)}")

    shape = image_data.shape
    dims = len(shape)
    img = image_data.copy()

    img = convert_to_float(img)

    cmap = None
    if dims == 2:
        cmap = "gray"
        height, width = shape
    elif dims == 3:
        height, width = shape[0], shape[1]
        if shape[2] == 1:
            img = np.squeeze(img, axis=2)
            cmap = "gray"
        elif shape[2] not in [3, 4]:
            raise ValueError("Last dimension must be 1, 3, or 4.")
    elif dims == 4:
        height, width = shape[1], shape[2]
        if shape[3] not in [3, 4]:
            raise ValueError("Last dimension must be 3 or 4.")
        img = img[0]
    else:
        raise ValueError("Unsupported dimensions. Expected 2, 3, or 4.")

    dpi = 100
    figsize = width / dpi, height / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(img, cmap=cmap, aspect="equal")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.show()
