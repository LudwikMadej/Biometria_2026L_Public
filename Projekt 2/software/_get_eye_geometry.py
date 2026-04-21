import cv2
import numpy as np
import matplotlib.pyplot as plt

from .iris_detection import detect_iris_geometry
from .pupil_detection import detect_pupil
from .visualization import visualize_eye_geometry


def get_eye_geometry(
    img,
    max_iters=3,
    scale_factor=0.95,
    contrast_step=1.01,
    show_plots=False,
    n_start_adv=15,
    n_end_adv=30,
    n_start_ref=10,
    n_end_ref=20,
):
    """
    Wykrywa geometrię oka (źrenica + tęczówka) iteracyjnie.

    Strategia oparta na średniej jasności: zbyt jasne obrazy są przyciemniane,
    żeby kontrast nie "spalił" szczegółów. W każdej iteracji podbijamy kontrast
    i skalujemy obraz w dół - detekcja jest stabilniejsza na mniejszych obrazach.

    Returns:
        (pupil, iris): dwie krotki (x, y, r) w pikselach oryginalnego obrazu.
    """

    # ── analiza jasności ──────────────────────────────────────────────────────
    brightness = np.mean(img)
    working_img = img.copy()

    if brightness > 170:
        # jeśli obraz jest obiektywnie jasny (średnia > 170), przyciemniamy go
        # odejmując różnicę do docelowej średniej ~140 - unikamy nasycenia
        # przy późniejszym podbijaniu kontrastu
        initial_beta = int(140 - brightness)
        if show_plots:
            print(
                f"  [info] Jasność {brightness:.1f} > 170. Przyciemniam (beta={initial_beta})"
            )
        working_img = cv2.convertScaleAbs(working_img, alpha=1.0, beta=initial_beta)
    else:
        # dla normalnych obrazów rozciągamy histogram do pełnego zakresu 0-255,
        # żeby detekcja działała na spójnej skali jasności niezależnie od źródła
        working_img = cv2.normalize(working_img, None, 0, 255, cv2.NORM_MINMAX)

    # skumulowana skala potrzebna do przeliczenia wykrytych współrzędnych
    # z powrotem do układu oryginalnego obrazu
    cumulative_scale = 1.0
    final_pupil = (0.0, 0.0, 0.0)
    final_iris = (0.0, 0.0, 0.0)

    if show_plots:
        print(f"--- Start skalowania (rozmiar: {img.shape}) ---")

    for i in range(max_iters):
        # podbijanie kontrastu
        # contrast_step > 1 rozciąga histogram; beta=-2 lekko przyciemnia,
        # żeby kolejne podbijania nie doprowadziły do całkowitego nasycenia
        working_img = cv2.convertScaleAbs(working_img, alpha=contrast_step, beta=-2)

        # detekcja źrenicy
        res_p = detect_pupil(working_img)

        # backup: jeśli pierwsza próba nieudana, spróbuj na surowym obrazie
        # (tylko w pierwszej iteracji, żeby nie wydłużać zbytnio czasu)
        if (res_p is None or any(v is None for v in res_p)) and i == 0:
            if show_plots:
                print("  [backup] Próba na surowym obrazie...")
            res_p = detect_pupil(
                cv2.resize(img, (working_img.shape[1], working_img.shape[0]))
            )

        if res_p is None or any(v is None for v in res_p):
            if show_plots:
                print(f"  [iteracja {i+1}] Brak źrenicy.")
            break

        lxp, lyp, lrp = res_p

        # detekcja tęczówki
        try:
            res_i = detect_iris_geometry(
                working_img,
                lxp,
                lyp,
                lrp,
                n_start_adv=n_start_adv,
                n_end_adv=n_end_adv,
                n_start_ref=n_start_ref,
                n_end_ref=n_end_ref,
            )
            lxi, lyi, lri = res_i if res_i is not None else (lxp, lyp, lrp * 2.5)
        except Exception:
            # jeśli detekcja tęczówki rzuci wyjątek, szacujemy promień jako 2.5×źrenica
            lxi, lyi, lri = lxp, lyp, lrp * 2.5

        # twarda korekta: jeśli tęczówka jest prawie tej samej wielkości co źrenica,
        # detekcja jest nieprawidłowa - wymuszamy rozsądny stosunek promieni
        if (lri - lrp) < 10:
            lri = lrp * 2.5
            lxi, lyi = lxp, lyp

        # przeliczenie do układu oryginalnego obrazu
        # w każdej iteracji obraz był skalowany w dół o scale_factor,
        # więc wszystkie współrzędne są w "skurczonym" układzie - dzielimy przez
        # skumulowaną skalę, żeby odwrócić to przekształcenie
        inv_scale = 1.0 / cumulative_scale
        final_pupil = (lxp * inv_scale, lyp * inv_scale, lrp * inv_scale)
        final_iris = (lxi * inv_scale, lyi * inv_scale, lri * inv_scale)

        if show_plots:
            current_br = np.mean(working_img)
            print(
                f"  [iteracja {i+1}] Skala: {cumulative_scale:.3f}, średnia jasność: {current_br:.1f}"
            )
            visualize_eye_geometry(working_img, lxp, lyp, lrp, lxi, lyi, lri)

        # skalowanie w dół na następną iterację
        # mniejszy obraz → hough/kontury działają na grubszej skali → łatwiej
        # uchwycić źrenicę bez szumów; zatrzymujemy się gdy obraz staje się za mały
        if i < max_iters - 1:
            new_w = int(working_img.shape[1] * scale_factor)
            new_h = int(working_img.shape[0] * scale_factor)

            if new_w > 150:
                working_img = cv2.resize(
                    working_img, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
                cumulative_scale *= scale_factor
            else:
                break

    return final_pupil, final_iris
