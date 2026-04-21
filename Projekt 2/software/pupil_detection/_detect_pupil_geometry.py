import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_pupil_geometry(
    img: np.ndarray,
    show_plots: bool = False,
    min_area: int = 100,
    circularity_threshold: float = 0.5,
) -> tuple[float | None, float | None, float | None]:
    """
    Wykrywa geometrię źrenicy przez progowanie adaptacyjne i analizę kołowości konturów.

    Metoda:
    1. Wyznaczamy próg jasności jako P/log(P), gdzie P = średnia jasność obrazu.
       Formuła ta automatycznie daje niski próg dla ciemnych obrazów (tylko najciemniejsze
       piksele są kandydatami) i nieco wyższy dla jasnych - bez ręcznego dobierania.
    2. Progujemy odwrotnie (THRESH_BINARY_INV): piksele ciemniejsze od progu → białe.
    3. Morfologia usuwa szum i wypełnia przerwy w konturze źrenicy.
    4. Z konturów wybieramy ten o najwyższej kołowości (źrenica jest kołem).

    Returns:
        (center_x, center_y, radius) w pikselach lub (None, None, None) gdy brak detekcji.
    """

    P = np.mean(img)
    # obraz prawie całkowicie czarny - brak sensu progować
    if P <= 1:
        return None, None, None

    # próg: P/log(P) - dla typowej jasności ~100-150 daje wartość ~20-30,
    # czyli wyodrębnia tylko najciemniejsze piksele (kandydaci na źrenicę)
    threshold = P / np.log(P)
    _, pupil_bin = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)

    # mały kernel otwierający usuwa pojedyncze jasne piksele szumu wewnątrz źrenicy
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # większy kernel domykający scala pobliskie obszary w jeden spójny kontur
    kernel_main = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    temp = cv2.morphologyEx(pupil_bin, cv2.MORPH_OPEN, kernel_small)
    processed = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel_main)

    # zewnętrzne kontury - szukamy granic obszarów, nie wewnętrznych dziur
    contours, _ = cv2.findContours(
        processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_contour = None
    max_circularity = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # pomijamy mikroskopijne artefakty - źrenica musi mieć minimalną powierzchnię
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # kołowość: 4πA/P² = 1 dla idealnego koła, maleje dla nieregularnych kształtów
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

        # wybieramy kontur o najwyższej kołowości (źrenica jest kołem)
        if circularity > circularity_threshold and circularity > max_circularity:
            max_circularity = circularity
            best_contour = cnt

    if best_contour is None:
        return None, None, None

    # minimalne koło otaczające kontur → centrum i promień źrenicy
    (x, y), radius = cv2.minEnclosingCircle(best_contour)

    if show_plots:
        # wizualizacja pomocnicza - maska progowania i zrekonstruowany okrąg
        pupil_clean = np.zeros_like(pupil_bin)
        cv2.circle(pupil_clean, (int(x), int(y)), int(radius), 255, -1)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Maska progowania")
        plt.imshow(pupil_bin, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Zrekonstruowany okrąg")
        plt.imshow(pupil_clean, cmap="gray")
        plt.axis("off")
        plt.show()

        viz = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        cv2.circle(viz, (int(x), int(y)), int(radius), (0, 255, 255), 1)
        cv2.drawMarker(viz, (int(x), int(y)), (255, 0, 0), cv2.MARKER_CROSS, 8, 1)

        plt.figure(figsize=(6, 5))
        plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
        plt.title("Wykryta geometria")
        plt.axis("off")
        plt.show()

    return float(x), float(y), float(radius)
