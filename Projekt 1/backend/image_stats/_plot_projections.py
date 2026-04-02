import numpy as np
from matplotlib.figure import Figure
from backend.pixel_functions import convert_to_grayscale


def plot_projections(image: np.ndarray) -> dict[str, Figure]:
    """
    Generates horizontal and vertical projections of an image.

    Returns a dictionary containing:
        - 'vertical'   : vertical projection plot
        - 'horizontal' : horizontal projection plot
        - 'combined'   : image with aligned projections
                         (vertical above, horizontal on right)

    Args:
        image (np.ndarray): Grayscale (H,W) or RGB (H,W,3) image.

    Returns:
        dict[str, Figure]

    Raises:
        TypeError: If input is not numpy array.
        ValueError: If image shape is unsupported.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")

    img = image.copy()
    img = convert_to_grayscale(img)

    if img.ndim != 2:
        raise ValueError("Unsupported image shape.")

    height, width = img.shape

    vertical_projection = np.sum(img, axis=0)
    horizontal_projection = np.sum(img, axis=1)

    figures = {}

    fig_v = Figure(figsize=(6, 3))
    ax_v = fig_v.subplots()

    x_v = np.arange(width)

    ax_v.plot(x_v, vertical_projection)
    ax_v.fill_between(x_v, vertical_projection, 0, alpha=0.3)

    ax_v.set_title("Vertical Projection")
    ax_v.set_xlabel("Column index")
    ax_v.set_ylabel("Sum of intensities")
    ax_v.set_xlim(0, width)

    fig_v.tight_layout()
    figures["vertical"] = fig_v

    fig_h = Figure(figsize=(3, 6))
    ax_h = fig_h.subplots()

    y_h = np.arange(height)

    ax_h.plot(horizontal_projection, y_h)
    ax_h.fill_betweenx(y_h, horizontal_projection, 0, alpha=0.3)

    ax_h.invert_yaxis()

    ax_h.set_title("Horizontal Projection")
    ax_h.set_xlabel("Sum of intensities")
    ax_h.set_ylabel("Row index")
    ax_h.set_ylim(height, 0)

    fig_h.tight_layout()
    figures["horizontal"] = fig_h

    fig_c = Figure(figsize=(8, 8))

    gs = fig_c.add_gridspec(
        2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=0.05, wspace=0.05
    )

    ax_top = fig_c.add_subplot(gs[0, 0])
    ax_img = fig_c.add_subplot(gs[1, 0])
    ax_right = fig_c.add_subplot(gs[1, 1])

    ax_img.imshow(img, cmap="gray", aspect="auto")
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    ax_top.plot(x_v, vertical_projection)
    ax_top.fill_between(x_v, vertical_projection, 0, alpha=0.3)
    ax_top.set_xlim(0, width)
    ax_top.set_xticks([])
    ax_top.set_ylabel("Sum")

    ax_right.plot(horizontal_projection, y_h)
    ax_right.fill_betweenx(y_h, horizontal_projection, 0, alpha=0.3)
    ax_right.set_ylim(height, 0)
    ax_right.set_yticks([])
    ax_right.set_xlabel("Sum")

    figures["combined"] = fig_c

    return figures
