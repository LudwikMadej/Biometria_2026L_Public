import cv2
import numpy as np
import matplotlib.pyplot as plt


from .iris_detection import detect_iris_geometry
from .pupil_detection import detect_pupil
from .visualization import visualize_eye_geometry


def get_eye_geometry(img, max_iters=3, scale_factor=0.95, contrast_step=1.01, show_plots=False,
                     n_start_adv=15, n_end_adv=30,
                     n_start_ref=10, n_end_ref=20):
    """
    Wersja oparta na średniej jasności. Przyciemniamy jasne obrazy, 
    żeby kontrast ich nie spalił.
    """
    # 1. ANALIZA JASNOŚCI NA WEJŚCIU
    brightness = np.mean(img)
    
    # Startujemy od oryginału
    working_img = img.copy()
    
    # 2. DECYZJA O PRZYCIEMNIANIU (jeśli "frajer" jest za jasny)
    # Jeśli średnia > 170, obraz jest obiektywnie jasny. 
    # Odejmujemy różnicę, żeby sprowadzić go do poziomu ~130-140.
    initial_beta = 0
    if brightness > 170:
        initial_beta = int(140 - brightness) # Wartość ujemna, np. -40
        if show_plots: print(f"  [Info] Jasność {brightness:.1f} > 170. Przyciemniam (beta={initial_beta})")
        working_img = cv2.convertScaleAbs(working_img, alpha=1.0, beta=initial_beta)
    else:
        # Nawet dla normalnych obrazów warto zrobić normalize, żeby zacząć od 0-255
        working_img = cv2.normalize(working_img, None, 0, 255, cv2.NORM_MINMAX)

    cumulative_scale = 1.0
    final_pupil = (0.0, 0.0, 0.0)
    final_iris = (0.0, 0.0, 0.0)

    if show_plots:
        print(f"--- Start skalowania (Rozmiar: {img.shape}) ---")

    for i in range(max_iters):
        # 3. ZWIĘKSZANIE KONTRASTU
        # Przy każdej iteracji podbijamy kontrast, ale pilnujemy jasności (beta=-2), 
        # żeby znowu nie uciec w biel.
        working_img = cv2.convertScaleAbs(working_img, alpha=contrast_step, beta=-2)
        
        # 4. DETEKCJA ŹRENICY
        res_p = detect_pupil(working_img)
        
        # Jeśli nie idzie na zmodyfikowanym, ostatnia szansa na surowym (tylko 1 it)
        if (res_p is None or any(v is None for v in res_p)) and i == 0:
            if show_plots: print("  [Backup] Próba na surowym...")
            res_p = detect_pupil(cv2.resize(img, (working_img.shape[1], working_img.shape[0])))

        if res_p is None or any(v is None for v in res_p):
            if show_plots: print(f"  [Iteracja {i+1}] Brak źrenicy.")
            break
        
        lxp, lyp, lrp = res_p

        # 5. DETEKCJA TĘCZÓWKI
        try:
            res_i = detect_iris_geometry(
                working_img, lxp, lyp, lrp,
                n_start_adv=15, n_end_adv=30,
                n_start_ref=10, n_end_ref=20
            )
            lxi, lyi, lri = res_i if res_i is not None else (lxp, lyp, lrp * 2.5)
        except:
            lxi, lyi, lri = lxp, lyp, lrp * 2.5

        # TWARDA LOGIKA
        if (lri - lrp) < 10:
            lri = lrp * 2.5
            lxi, lyi = lxp, lyp
        
        # 6. POWRÓT DO GLOBALNYCH PX
        inv_scale = 1.0 / cumulative_scale
        final_pupil = (lxp * inv_scale, lyp * inv_scale, lrp * inv_scale)
        final_iris = (lxi * inv_scale, lyi * inv_scale, lri * inv_scale)

        # 7. WIZUALIZACJA
        if show_plots:
            current_br = np.mean(working_img)
            print(f"  [Iteracja {i+1}] Skala: {cumulative_scale:.3f}, Średnia jasność: {current_br:.1f}")
            visualize_eye_geometry(working_img, lxp, lyp, lrp, lxi, lyi, lri)

        # 8. SKALOWANIE OBRAZU
        if i < max_iters - 1:
            new_w = int(working_img.shape[1] * scale_factor)
            new_h = int(working_img.shape[0] * scale_factor)
            
            if new_w > 150:
                working_img = cv2.resize(working_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cumulative_scale *= scale_factor
            else:
                break

    return final_pupil, final_iris