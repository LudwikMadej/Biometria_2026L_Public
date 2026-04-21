import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_pupil_geometry(
    img: np.ndarray,
    show_plots: bool = False,
    min_area: int = 100,
    circularity_threshold: float = 0.5
) -> tuple[float | None, float | None, float | None]:
    """
    Detects pupil geometry using adaptive thresholding and contour circularity analysis.

    Parameters
    ----------
    img : np.ndarray
        Grayscale input image (uint8).
    show_plots : bool, optional
        Enables diagnostic visualization.
    min_area : int, optional
        Minimum contour area constraint.
    circularity_threshold : float, optional
        Lower bound for circularity filtering.

    Returns
    -------
    tuple[float | None, float | None, float | None]
        (center_x, center_y, radius) in pixel coordinates. Returns (None, None, None) if detection fails.

    Notes
    -----
    Threshold definition:
        P = mean(I)
        T = P / log(P)

    Circularity:
        C = 4πA / P²
    """

    P = np.mean(img)
    if P <= 1:
        return None, None, None

    threshold = P / np.log(P)
    _, pupil_bin = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_main = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    temp = cv2.morphologyEx(pupil_bin, cv2.MORPH_OPEN, kernel_small)
    processed = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel_main)

    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    max_circularity = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

        if circularity > circularity_threshold and circularity > max_circularity:
            max_circularity = circularity
            best_contour = cnt

    if best_contour is None:
        return None, None, None

    (x, y), radius = cv2.minEnclosingCircle(best_contour)

    if show_plots:
        pupil_clean = np.zeros_like(pupil_bin)
        cv2.circle(pupil_clean, (int(x), int(y)), int(radius), 255, -1)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Threshold Mask")
        plt.imshow(pupil_bin, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Reconstructed Circle")
        plt.imshow(pupil_clean, cmap="gray")
        plt.axis("off")
        plt.show()

        viz = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        cv2.circle(viz, (int(x), int(y)), int(radius), (0, 255, 255), 1)
        cv2.drawMarker(viz, (int(x), int(y)), (255, 0, 0), cv2.MARKER_CROSS, 8, 1)

        plt.figure(figsize=(6, 5))
        plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
        plt.title("Detection Overlay")
        plt.axis("off")
        plt.show()

    return float(x), float(y), float(radius)