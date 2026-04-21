import cv2
import numpy as np
import matplotlib.pyplot as plt

from ._detect_pupil_geometry import detect_pupil_geometry
from ._refine_pupil_roi import refine_pupil_roi

def _get_weight(img: np.ndarray, x: float, y: float, r: float) -> float:
    """
    Oblicza wagę ignorując odblaski. 
    Używamy niskiego kwantyla (np. 0.2), aby sprawdzić bazową ciemność źrenicy.
    """
    if x is None or y is None or r <= 0:
        return 0.0
    
    # 1. Wycinamy ROI (szybsze niż maskowanie całego obrazu)
    y_min, y_max = max(0, int(y-r)), min(img.shape[0], int(y+r))
    x_min, x_max = max(0, int(x-r)), min(img.shape[1], int(x+r))
    roi = img[y_min:y_max, x_min:x_max]
    
    if roi.size == 0:
        return 0.0
    
    # 2. Obliczamy kwantyl 0.2 (20% najciemniejszych pikseli)
    # Odblaski (jasne punkty) znajdą się w okolicy kwantyla 0.9-1.0 i zostaną zignorowane.
    dark_threshold = np.quantile(roi, 0.5)
    
    # 3. Czarność bazujemy na tym, jak ciemna jest ta "najciemniejsza grupa"
    darkness = 255.0 - dark_threshold
    
    # 4. Waga: im ciemniejszy jest ten dolny kwantyl, tym wyższa waga
    return np.exp(darkness * 2 + r)

def detect_pupil(
    img: np.ndarray,
    detect_kwargs: dict | None = None,
    refine_kwargs: dict | None = None,
    return_intermediate: bool = False
) -> tuple:
    detect_kwargs = detect_kwargs or {}
    refine_kwargs = refine_kwargs or {}

    # 1. Wstępna detekcja
    x0, y0, r0 = detect_pupil_geometry(img, **detect_kwargs)

    if x0 is None:
        if return_intermediate:
            return (None,) * 9
        return None, None, None

    # 2. Uściślenie (Refinement)
    x1, y1, r1 = refine_pupil_roi(img, x0, y0, r0, **refine_kwargs)

    # 3. Obliczenie wag opartych na eksponencie czarności
    w0 = _get_weight(img, x0, y0, r0)
    w1 = _get_weight(img, x1, y1, r1)
    
    total_w = w0 + w1

    if (total_w > 0) and (abs(np.log10(w0) - np.log10(w1)) > 1):
        x_final = (x0 * w0 + x1 * w1) / total_w
        y_final = (y0 * w0 + y1 * w1) / total_w
        r_final = (r0 * w0 +  r1 * w1) / total_w
        
        
    else:
        x_final = (x0 + x1) / 2.0
        y_final = (y0 + y1) / 2.0
        r_final = max(r0, r1)
        
    

    if return_intermediate:
        return x0, y0, r0, x1, y1, r1, x_final, y_final, r_final

    return x_final, y_final, r_final