import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def visualize_eye_geometry(
    img, x_p, y_p, r_p, x_i, y_i, r_i, title="Wizualizacja geometrii oka"
):
    """
    Rysuje wykrytą geometrię oka (źrenica + tęczówka) na obrazie.

    Kolory:
      - Zielony neon: granica źrenicy.
      - Pomarańczowy: granica tęczówki.
      - Czerwony krzyżyk: środek źrenicy.
      - Niebieski krzyżyk: środek tęczówki.
    """

    # konwertujemy szary do rgb żeby móc rysować kolorowymi liniami
    if len(img.shape) == 2:
        output_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
    else:
        output_img = img.copy()

    NEON_GREEN = (
        57,
        255,
        20,
    )  # granica źrenicy - jaskrawy zielony, wyraźny na ciemnej źrenicy
    AMBER = (255, 165, 0)  # granica tęczówki - pomarańczowy, wyraźny na jasnej tęczówce
    RED = (255, 0, 0)  # środek źrenicy
    BLUE = (0, 0, 255)  # środek tęczówki

    # okrąg źrenicy (grubość 1 - cienki, żeby nie zasłaniać szczegółów)
    p_center = (int(x_p), int(y_p))
    cv2.circle(output_img, p_center, int(r_p), NEON_GREEN, 1)
    cv2.drawMarker(
        output_img,
        p_center,
        RED,
        markerType=cv2.MARKER_CROSS,
        markerSize=10,
        thickness=1,
    )

    # okrąg tęczówki (grubość 2 - nieco grubszy, bo obwód jest większy i cieńsza linia ginie)
    i_center = (int(x_i), int(y_i))
    cv2.circle(output_img, i_center, int(r_i), AMBER, 2)
    cv2.drawMarker(
        output_img,
        i_center,
        BLUE,
        markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=10,
        thickness=1,
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(output_img)
    plt.title(f"{title}\nźrenica: r={r_p:.1f} | tęczówka: r={r_i:.1f}")

    # legenda przez Line2D (nie przez cv2) - matplotlib nie zna kolorów narysowanych przez cv2
    legend_elements = [
        Line2D([0], [0], color="lime", lw=2, label="granica źrenicy"),
        Line2D([0], [0], color="orange", lw=2, label="granica tęczówki"),
        Line2D(
            [0],
            [0],
            marker="+",
            color="red",
            label="środek źrenicy",
            markersize=10,
            ls="",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="blue",
            label="środek tęczówki",
            markersize=10,
            ls="",
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper right", frameon=True)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
