import numpy as np
import itertools
from tqdm.auto import tqdm

from ._evaluator import Evaluator
from ._get_eye_geometry import get_eye_geometry


def find_optimal_eye_geometry(img, param_grid, top_k=5, trim_ratio=0.2):
    """
    Szuka najlepszej geometrii oka przez przeszukiwanie siatki parametrów (grid search).

    Dlaczego grid search: get_eye_geometry ma kilka parametrów (skala, kontrast,
    zakresy skanowania), które silnie wpływają na jakość detekcji. Zamiast
    ręcznie dobierać parametry, sprawdzamy wszystkie kombinacje i bierzemy
    uśredniony wynik z najlepszych - daje stabilniejszą geometrię niż pojedyncze uruchomienie.

    Args:
        img:        Obraz w skali szarości.
        param_grid: Słownik {nazwa_parametru: [wartości]}.
        top_k:      Ile najlepszych wyników uwzględniamy przy uśrednianiu.
        trim_ratio: Część wyników do odrzucenia od góry przed uśrednianiem
                    (zabezpieczenie przed pojedynczymi bardzo dobrymi, ale
                    niereprezentatywnymi wynikami).
    """
    evaluator = Evaluator(img)
    results = []
    seen_keys = (
        set()
    )  # zbiór do deduplikacji - ta sama geometria może pojawić się wielokrotnie

    # wszystkie kombinacje wartości parametrów z siatki
    keys, values = zip(*param_grid.items())
    combinations = list(itertools.product(*values))

    for combo in tqdm(combinations, desc="Optimizing Eye Geometry", unit="cfg"):
        p = dict(zip(keys, combo))

        # n_end obliczamy dynamicznie jako n_start + gap, żeby siatka
        # była opisana czytelnie przez dwa parametry zamiast jednego końca
        adv = (
            p.get("n_start_adv", 15),
            p.get("n_start_adv", 15) + p.get("gap_adv", 10),
        )
        ref = (
            p.get("n_start_ref", 10),
            p.get("n_start_ref", 10) + p.get("gap_ref", 10),
        )

        try:
            pupil, iris = get_eye_geometry(
                img,
                max_iters=p.get("max_iters", 3),
                scale_factor=p.get("scale_factor", 0.95),
                contrast_step=p.get("contrast_step", 1.01),
                n_start_adv=adv[0],
                n_end_adv=adv[1],
                n_start_ref=ref[0],
                n_end_ref=ref[1],
                show_plots=False,
            )

            # odrzucamy wyniki z zerowym promieniem lub tęczówką nie większą od źrenicy
            if pupil[2] < 5 or iris[2] <= pupil[2]:
                continue

            # deduplikacja: zaokrąglamy do intów - ta sama geometria (różniąca się
            # o ułamek piksela) nie powinna być liczona wielokrotnie
            geom_key = tuple(np.round([*pupil, *iris]).astype(int))
            if geom_key in seen_keys:
                continue
            seen_keys.add(geom_key)

            results.append(
                {
                    "score": evaluator.evaluate(pupil, iris),
                    "p": pupil,
                    "i": iris,
                }
            )

        except Exception:
            # ignorujemy wszelkie błędy numeryczne z konkretnej kombinacji parametrów
            continue

    if not results:
        return None, None

    # sortujemy malejąco po wyniku oceniającego i bierzemy top_k
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    # trim: odrzucamy trim_ratio najlepszych wyników - chronią przed przypadkowym
    # "outlierem" z bardzo wysokim wynikiem, który zaburzyłby uśrednianie
    keep_n = max(1, int(len(results) * (1 - trim_ratio)))
    results = results[:keep_n]

    # ważona średnia geometrii: geometrie z wyższym wynikiem mają większy wkład
    scores = np.array([r["score"] for r in results])
    weights = scores / (scores.sum() + 1e-9)

    avg_p = np.sum([np.array(r["p"]) * w for r, w in zip(results, weights)], axis=0)
    avg_i = np.sum([np.array(r["i"]) * w for r, w in zip(results, weights)], axis=0)

    return tuple(avg_p), tuple(avg_i)
