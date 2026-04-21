import cv2
import numpy as np
import matplotlib.pyplot as plt

from ._detect_pupil_geometry import detect_pupil_geometry
from ._refine_pupil_roi import refine_pupil_roi


def _get_weight(img: np.ndarray, x: float, y: float, r: float) -> float:
    """
    Oblicza wagę detekcji na podstawie ciemności ROI.

    Dlaczego eksponencja: chcemy bardzo mocno preferować ciemne wyniki
    (prawdziwa źrenica) nad jaśniejszymi (artefakty, tęczówka). Eksponencja
    tworzy duże różnice wag nawet przy małych różnicach jasności.
    """
    if x is None or y is None or r <= 0:
        return 0.0

    # wycinamy roi wokół wykrytego centrum - szybsze niż maskowanie całego obrazu
    y_min = max(0, int(y - r))
    y_max = min(img.shape[0], int(y + r))
    x_min = max(0, int(x - r))
    x_max = min(img.shape[1], int(x + r))
    roi = img[y_min:y_max, x_min:x_max]

    if roi.size == 0:
        return 0.0

    # mediana roi: odporna na odblaski (jasne punkty mają wartości ~0.9-1.0 kwantyla
    # i nie zaburzają oceny ciemności centrum)
    dark_threshold = np.quantile(roi, 0.5)

    # "ciemność" = odwrotność mediany, przeskalowana do zakresu [0, 255]
    darkness = 255.0 - dark_threshold

    # eksponencja z promieniem w wykładniku: większy promień = wyższa waga
    # (w granicach sensowności - duże fałszywe kołowości są filtrowane wcześniej)
    return np.exp(darkness * 2 + r)


def detect_pupil(
    img: np.ndarray,
    detect_kwargs: dict | None = None,
    refine_kwargs: dict | None = None,
    return_intermediate: bool = False,
) -> tuple:
    """
    Wykrywa źrenicę w dwóch krokach: gruba detekcja → uściślenie ROI.

    Dwuetapowość: detect_pupil_geometry operuje na całym obrazie i podaje
    przybliżone centrum; refine_pupil_roi skupia się na małym wycinku
    i precyzyjnie wyznacza granicę - podział ról daje lepszą dokładność
    przy rozsądnym czasie obliczeń.

    Returns:
        (x, y, r) - uśrednione współrzędne z detekcji i uściślenia.
        Jeśli return_intermediate=True: wszystkie 9 wartości pośrednich.
    """
    detect_kwargs = detect_kwargs or {}
    refine_kwargs = refine_kwargs or {}

    # krok 1: przybliżona detekcja na całym obrazie
    x0, y0, r0 = detect_pupil_geometry(img, **detect_kwargs)

    if x0 is None:
        if return_intermediate:
            return (None,) * 9
        return None, None, None

    # krok 2: uściślenie w roi wokół znalezionego centrum
    x1, y1, r1 = refine_pupil_roi(img, x0, y0, r0, **refine_kwargs)

    # łączenie wyników
    # liczymy wagę każdego wyniku - ciemniejsza detekcja otrzymuje wyższe wagi
    w0 = _get_weight(img, x0, y0, r0)
    w1 = _get_weight(img, x1, y1, r1)

    total_w = w0 + w1

    if total_w > 0 and (abs(np.log10(w0) - np.log10(w1)) > 1):
        # jeśli jedna waga jest rząd wielkości wyższa od drugiej,
        # używamy ważonej średniej - jeden z wyników wyraźnie lepszy
        x_final = (x0 * w0 + x1 * w1) / total_w
        y_final = (y0 * w0 + y1 * w1) / total_w
        r_final = (r0 * w0 + r1 * w1) / total_w
    else:
        # gdy oba wyniki są podobnej jakości, bierzemy prostą średnią środków
        # i większy z promieni (konserwatywne - lepiej trochę zawyżyć promień
        # niż go uciąć i stracić część tęczówki w unrollu)
        x_final = (x0 + x1) / 2.0
        y_final = (y0 + y1) / 2.0
        r_final = max(r0, r1)

    if return_intermediate:
        return x0, y0, r0, x1, y1, r1, x_final, y_final, r_final

    return x_final, y_final, r_final
