import numpy as np
from matplotlib.figure import Figure
from backend.pixel_functions import convert_to_grayscale


def plot_brightness_histograms(
    image_data: np.ndarray, normalize: bool = False
) -> dict[str, Figure]:
    """
    Generates brightness histograms for color channels (if RGB)
    and combined brightness.

    Works for:
        - Grayscale image: shape (H, W)
        - RGB image:       shape (H, W, 3+)

    Args:
        image_data (np.ndarray): Input image array.
        normalize (bool): If True, normalizes histogram counts so they sum to 1.

    Returns:
        dict[str, Figure]: Dictionary of matplotlib Figure objects.
            For RGB:
                'red', 'green', 'blue', 'combined'
            For grayscale:
                'combined'

    Raises:
        TypeError: If input is not a numpy array.
    """

    if not isinstance(image_data, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image_data)}")

    img = image_data.copy()

    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    bins_range = np.arange(256)
    ylabel = "Normalized number of pixels" if normalize else "Number of pixels"
    figures = {}

    colors = ("red", "green", "blue")
    channel_names = ("Red", "Green", "Blue")

    if img.ndim == 2:
        fig = Figure(figsize=(5, 3.5))
        ax = fig.subplots()

        counts = np.bincount(img.ravel(), minlength=256)

        if normalize:
            counts = counts / counts.sum()

        ax.bar(bins_range, counts, color="gray", width=1.0)
        ax.set_title("Brightness histogram (Grayscale)")
        ax.set_xlabel("Value of pixel (0-255)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 255)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        fig.tight_layout()
        figures["combined"] = fig

        return figures

    if img.ndim == 3 and img.shape[-1] >= 3:

        for i, color in enumerate(colors):
            fig = Figure(figsize=(5, 3.5))
            ax = fig.subplots()

            channel_data = img[..., i].ravel()
            counts = np.bincount(channel_data, minlength=256)

            if normalize:
                counts = counts / counts.sum()

            ax.bar(bins_range, counts, color=color, width=1.0)
            ax.set_title(f"Histogram for channel: {channel_names[i]}")
            ax.set_xlabel("Value of pixel (0-255)")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, 255)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            fig.tight_layout()
            figures[color] = fig

        gray = convert_to_grayscale(img)

        if np.issubdtype(gray.dtype, np.floating):
            gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)

        fig_combined = Figure(figsize=(5, 3.5))
        ax_combined = fig_combined.subplots()

        counts_combined = np.bincount(gray.ravel(), minlength=256)

        if normalize:
            counts_combined = counts_combined / counts_combined.sum()

        ax_combined.bar(bins_range, counts_combined, color="gray", width=1.0)
        ax_combined.set_title("Brightness histogram (Channels combined)")
        ax_combined.set_xlabel("Value of pixel (0-255)")
        ax_combined.set_ylabel(ylabel)
        ax_combined.set_xlim(0, 255)
        ax_combined.grid(axis="y", linestyle="--", alpha=0.7)

        fig_combined.tight_layout()
        figures["combined"] = fig_combined

        return figures

    raise ValueError("Unsupported image shape. Expected (H, W) or (H, W, 3+).")
