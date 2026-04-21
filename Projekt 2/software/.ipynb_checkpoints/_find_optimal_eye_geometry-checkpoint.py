import numpy as np
import itertools
from tqdm.auto import tqdm  # Automatycznie wybiera wersję konsolową lub notebookową

from ._evaluator import Evaluator
from ._get_eye_geometry import get_eye_geometry

def find_optimal_eye_geometry(img, param_grid, top_k=5, trim_ratio=0.2):
    """
    Przeprowadza grid search parametrów z paskiem postępu tqdm, 
    a następnie zwraca uśrednioną geometrię.
    """
    evaluator = Evaluator(img)
    results = []
    seen_keys = set()

    # 1. Przygotowanie kombinacji
    keys, values = zip(*param_grid.items())
    combinations = list(itertools.product(*values)) # Konwersja na listę, by znać 'total'
    
    # 2. Grid Search z tqdm
    # desc: opis paska, unit: jednostka (tu 'cfg' jako konfiguracja)
    for combo in tqdm(combinations, desc="Optimizing Eye Geometry", unit="cfg"):
        p = dict(zip(keys, combo))
        
        # Logika dynamicznych końcówek n_end
        adv = (p.get("n_start_adv", 15), p.get("n_start_adv", 15) + p.get("gap_adv", 10))
        ref = (p.get("n_start_ref", 10), p.get("n_start_ref", 10) + p.get("gap_ref", 10))

        try:
            pupil, iris = get_eye_geometry(
                img, 
                max_iters=p.get("max_iters", 3),
                scale_factor=p.get("scale_factor", 0.95),
                contrast_step=p.get("contrast_step", 1.01),
                n_start_adv=adv[0], n_end_adv=adv[1],
                n_start_ref=ref[0], n_end_ref=ref[1],
                show_plots=False
            )

            # Walidacja i Deduplikacja
            geom_key = tuple(np.round([*pupil, *iris]).astype(int))
            if pupil[2] < 5 or iris[2] <= pupil[2] or geom_key in seen_keys:
                continue
            
            seen_keys.add(geom_key)
            results.append({
                "score": evaluator.evaluate(pupil, iris), 
                "p": pupil, 
                "i": iris
            })

        except Exception:
            continue

    if not results:
        return None, None

    # 3. Sortowanie i uśrednianie (identycznie jak wcześniej)
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    keep_n = max(1, int(len(results) * (1 - trim_ratio)))
    results = results[:keep_n]

    scores = np.array([r["score"] for r in results])
    weights = scores / (scores.sum() + 1e-9)
    
    avg_p = np.sum([np.array(r["p"]) * w for r, w in zip(results, weights)], axis=0)
    avg_i = np.sum([np.array(r["i"]) * w for r, w in zip(results, weights)], axis=0)

    return tuple(avg_p), tuple(avg_i)