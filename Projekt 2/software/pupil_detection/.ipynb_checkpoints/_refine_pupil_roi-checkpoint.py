import cv2
import numpy as np
import matplotlib.pyplot as plt

def refine_pupil_roi(
    img: np.ndarray,
    x_center: float,
    y_center: float,
    radius: float,
    roi_scale: float = 2.0,
    reflection_threshold: int = 200,
    blur_kernel: int = 5,
    show_plots: bool = False
) -> tuple[float, float, float]:
    """
    Refines pupil geometry via ROI-based edge reconstruction with specular reflection suppression.

    Parameters
    ----------
    img : np.ndarray
        Grayscale input image.
    x_center : float
        Initial x-coordinate.
    y_center : float
        Initial y-coordinate.
    radius : float
        Initial radius.
    roi_scale : float, optional
        ROI expansion factor relative to radius.
    reflection_threshold : int, optional
        Intensity threshold for reflection masking.
    blur_kernel : int, optional
        Gaussian kernel size (odd).
    show_plots : bool, optional
        Enables visualization.

    Returns
    -------
    tuple[float, float, float]
        Refined (x, y, r).
    """

    margin = int(roi_scale * radius)

    y1 = max(0, int(y_center - margin))
    y2 = min(img.shape[0], int(y_center + margin))
    x1 = max(0, int(x_center - margin))
    x2 = min(img.shape[1], int(x_center + margin))

    roi = img[y1:y2, x1:x2].copy()

    _, reflection_mask = cv2.threshold(roi, reflection_threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    reflection_mask = cv2.dilate(reflection_mask, kernel)

    roi_clean = cv2.inpaint(roi, reflection_mask, 3, cv2.INPAINT_TELEA)
    roi_blur = cv2.GaussianBlur(roi_clean, (blur_kernel, blur_kernel), 0)

    cx = int(x_center - x1)
    cy = int(y_center - y1)

    base = roi_blur[cy, cx]
    exit_threshold = min(base + 35, 90)

    def scan(arr):
        for i, v in enumerate(arr):
            if v > exit_threshold:
                return i
        return len(arr)

    edge_r = scan(roi_blur[cy, cx:])
    edge_l = scan(np.flip(roi_blur[cy, :cx]))
    edge_d = scan(roi_blur[cy:, cx])

    left = cx - edge_l
    right = cx + edge_r
    bottom = cy + edge_d

    new_x = (left + right) / 2.0
    new_r = (right - left) / 2.0
    new_y = bottom - new_r

    final_x = x1 + new_x
    final_y = y1 + new_y

    if show_plots:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("ROI")
        plt.imshow(roi, cmap="gray")

        plt.subplot(1, 3, 2)
        plt.title("Inpainted")
        plt.imshow(roi_blur, cmap="gray")

        plt.subplot(1, 3, 3)
        plt.title("Refined Geometry")
        plt.imshow(roi, cmap="gray")
        circle = plt.Circle((new_x, new_y), new_r, fill=False)
        plt.gca().add_patch(circle)
        plt.scatter([new_x], [new_y])
        plt.show()

    return float(final_x), float(final_y), float(new_r)