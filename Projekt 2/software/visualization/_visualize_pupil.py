import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_pupil(
    img, x_center, y_center, radius, title="Wizualizacja geometrii źrenicy"
):
    """
    Rysuje wykrytą geometrię źrenicy na obrazie.

    Kolory:
      - Zielony neon: granica źrenicy.
      - Czerwony krzyżyk: środek źrenicy.
    """

    # konwertujemy szary do rgb żeby móc rysować kolorową linią
    if len(img.shape) == 2:
        output_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
    else:
        output_img = img.copy()

    NEON_GREEN = (57, 255, 20)  # jaskrawy zielony - dobrze widoczny na ciemnej źrenicy
    RED = (255, 0, 0)

    center = (int(x_center), int(y_center))
    r = int(radius)

    # okrąg granicy źrenicy
    cv2.circle(output_img, center, r, NEON_GREEN, 1)

    # krzyżyk w centrum
    cv2.drawMarker(
        output_img, center, RED, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(output_img)
    plt.title(title)
    plt.axis("off")
    plt.show()
