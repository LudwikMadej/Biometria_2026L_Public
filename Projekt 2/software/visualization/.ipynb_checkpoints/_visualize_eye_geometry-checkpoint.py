import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def visualize_eye_geometry(img, x_p, y_p, r_p, x_i, y_i, r_i, title="Eye Geometry Visualization"):
    """
    Visualizes both pupil (x_p, y_p, r_p) and iris (x_i, y_i, r_i) geometry.
    """
    # 1. Przygotowanie obrazu wyjściowego
    if len(img.shape) == 2:
        output_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
    else:
        output_img = img.copy()

    # 2. Definicja kolorów
    NEON_GREEN = (57, 255, 20)  # Granica źrenicy
    AMBER = (255, 165, 0)       # Granica tęczówki
    RED = (255, 0, 0)           # Środek źrenicy
    BLUE = (0, 0, 255)          # Środek tęczówki

    # 3. Rysowanie źrenicy (Pupil)
    p_center = (int(x_p), int(y_p))
    cv2.circle(output_img, p_center, int(r_p), NEON_GREEN, 1)
    cv2.drawMarker(output_img, p_center, RED, 
                   markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

    # 4. Rysowanie tęczówki (Iris)
    i_center = (int(x_i), int(y_i))
    cv2.circle(output_img, i_center, int(r_i), AMBER, 2)
    cv2.drawMarker(output_img, i_center, BLUE, 
                   markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=1)

    # 5. Wyświetlanie
    plt.figure(figsize=(10, 8))
    plt.imshow(output_img)
    
    plt.title(f"{title}\nPupil: R={r_p:.1f} | Iris: R={r_i:.1f}")
    
    # Poprawna legenda używająca Line2D
    legend_elements = [
        Line2D([0], [0], color='lime', lw=2, label='Pupil Boundary'),
        Line2D([0], [0], color='orange', lw=2, label='Iris Boundary'),
        Line2D([0], [0], marker='+', color='red', label='Pupil Center', markersize=10, ls=''),
        Line2D([0], [0], marker='x', color='blue', label='Iris Center', markersize=10, ls='')
    ]
    plt.legend(handles=legend_elements, loc='upper right', frameon=True)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()