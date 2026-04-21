import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_iris_geometry(
    img,
    x_p,
    y_p,
    r_p,
    n_start_adv=15,
    n_end_adv=25,
    n_start_ref=10,
    n_end_ref=20,
    roi_factor=1.2,
    show_plots=False,
):
    """
    Wykrywa granicę tęczówki w dwóch krokach: globalny skan → lokalne uściślenie.

    Krok 1 (advanced): skanujemy promieniowo od centrum źrenicy i szukamy
    miejsca gdzie ciemna maska binarna "kończy się" (granica tęczówka→tło).
    Z zebranych punktów dopasowujemy okrąg metodą najmniejszych kwadratów.

    Krok 2 (refine): zawężamy obszar poszukiwań do ROI wokół okręgu z kroku 1
    i powtarzamy skan w węższym zakresie odległości - daje precyzyjniejszy wynik.
    """

    def fit_circle_lsm(points):
        """
        Dopasowuje okrąg do zbioru punktów metodą najmniejszych kwadratów.

        Algebraiczne LSM: równanie okręgu x²+y² = 2xc·x + 2yc·y + (xc²+yc²−r²)
        można przepisać jako układ liniowy Ax=B i rozwiązać lstsq.
        Szybsze i stabilniejsze niż metody iteracyjne dla małych zbiorów punktów.

        Returns:
            (xc, yc, rc) lub (None, None, None) gdy za mało punktów lub wynik numerycznie błędny.
        """
        # minimum 3 punkty do wyznaczenia okręgu (2 dają nieskończenie wiele rozwiązań)
        if len(points) < 3:
            return None, None, None

        x, y = points[:, 0], points[:, 1]
        # macierz układu: [2x, 2y, 1] · [xc, yc, (xc²+yc²-r²)] = x²+y²
        A = np.column_stack([x * 2, y * 2, np.ones(len(x))])
        B = x**2 + y**2
        res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

        xc, yc = res[0], res[1]
        # r² = res[2] + xc² + yc² - może być ujemne przy złych danych numerycznych
        r_sq = res[2] + xc**2 + yc**2
        if r_sq <= 0:
            return None, None, None

        rc = np.sqrt(r_sq)
        return xc, yc, rc

    # krok 1: globalny skan (detect_iris_advanced)

    # clahe (contrast limited adaptive histogram equalization) poprawia
    # lokalny kontrast obrazu - granica tęczówki staje się wyraźniejsza
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # próg z obrazu po wzmocnieniu kontrastu (nie oryginału) - spójność:
    # thresholdujemy enhanced używając statystyk z enhanced
    threshold_val_adv = np.quantile(enhanced, 0.2)
    _, iris_bin_adv = cv2.threshold(
        enhanced, threshold_val_adv, 255, cv2.THRESH_BINARY_INV
    )

    # morfologia zamykająca wypełnia przerwy w granicy tęczówki (np. rzęsy);
    # duży kernel (11×11) bo tęczówka jest znacznie większa niż źrenica
    kernel_adv = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cleaned_adv = cv2.morphologyEx(iris_bin_adv, cv2.MORPH_CLOSE, kernel_adv)
    cleaned_adv = cv2.morphologyEx(cleaned_adv, cv2.MORPH_OPEN, kernel_adv)

    # skanujemy wzdłuż 72 kierunków (co 5°) - dobry balans między
    # pokryciem kątowym a czasem obliczeń
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    scan_adv = []
    max_scan_adv = int(r_p * 4.5)  # maksymalna odległość: 4.5× promień źrenicy

    for angle in angles:
        dx, dy = np.cos(angle), np.sin(angle)
        # startujemy od 1.1 × r_pupil - pomijamy obszar źrenicy
        for dist in range(int(r_p * 1.1), max_scan_adv):
            curr_x = int(x_p + dx * dist)
            curr_y = int(y_p + dy * dist)
            # wychodzimy poza obraz - przerywamy ten kierunek
            if (
                curr_y >= cleaned_adv.shape[0]
                or curr_y < 0
                or curr_x >= cleaned_adv.shape[1]
                or curr_x < 0
            ):
                break
            # piksel = 0 w masce binarnej → koniec ciemnego obszaru → granica tęczówki
            if cleaned_adv[curr_y, curr_x] == 0:
                scan_adv.append((dist, curr_x, curr_y))
                break

    if len(scan_adv) < 3:
        # zbyt mało punktów - fallback: szacujemy tęczówkę jako 2× źrenica
        x_i1, y_i1, r_i1 = x_p, y_p, r_p * 2
    else:
        # sortujemy po odległości od centrum i bierzemy środkowe punkty:
        # pomijamy n_start_adv najbliższych (mogą być szumem przy źrenicy)
        # i n_end_adv najdalszych (mogą być poza tęczówką)
        scan_adv.sort(key=lambda p: p[0])
        actual_end_adv = min(n_end_adv, len(scan_adv))
        pts_adv = np.array([(p[1], p[2]) for p in scan_adv[n_start_adv:actual_end_adv]])

        result = fit_circle_lsm(pts_adv)
        if result[0] is None:
            x_i1, y_i1, r_i1 = x_p, y_p, r_p * 2
        else:
            x_i1, y_i1, r_i1 = result

    if show_plots:
        print(f"Debug: krok 1 (advanced) - r: {r_i1:.2f}")

    # krok 2: lokalne uściślenie (refine_iris_geometry)

    h, w = img.shape

    # roi wokół wyniku z kroku 1 - roi_factor=1.2 daje 20% margines
    # poza promieniem tęczówki, żeby skan mógł wykroczyć nieznacznie dalej
    roi_margin = int(r_i1 * roi_factor)
    x1 = max(0, int(x_i1 - roi_margin))
    y1 = max(0, int(y_i1 - roi_margin))
    x2 = min(w, int(x_i1 + roi_margin))
    y2 = min(h, int(y_i1 + roi_margin))

    roi = img[y1:y2, x1:x2]
    # centrum tęczówki w układzie lokalnym roi
    local_x_c = x_i1 - x1
    local_y_c = y_i1 - y1

    # mediana roi jako próg - adaptuje się do lokalnej jasności
    local_thresh = np.median(roi)
    _, local_bin = cv2.threshold(roi, local_thresh, 255, cv2.THRESH_BINARY_INV)
    kernel_ref = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_ref = cv2.morphologyEx(local_bin, cv2.MORPH_CLOSE, kernel_ref)

    scan_ref = []
    # szukamy tylko w przedziale [0.8, 1.3] × r_i1 - wąski zakres wokół
    # wyniku z kroku 1 eliminuje fałszywe krawędzie poza tęczówką
    search_range_min = int(r_i1 * 0.8)
    search_range_max = int(r_i1 * 1.3)

    for angle in angles:
        dx, dy = np.cos(angle), np.sin(angle)
        for dist in range(search_range_min, search_range_max):
            curr_x = int(local_x_c + dx * dist)
            curr_y = int(local_y_c + dy * dist)
            if (
                0 <= curr_y < cleaned_ref.shape[0]
                and 0 <= curr_x < cleaned_ref.shape[1]
            ):
                if cleaned_ref[curr_y, curr_x] == 0:
                    scan_ref.append((dist, curr_x, curr_y))
                    break

    if len(scan_ref) < 10:
        # zbyt mało punktów z lokalnego skanu - zwracamy wynik z kroku 1
        return x_i1, y_i1, r_i1

    # analogicznie jak w kroku 1: sortujemy i bierzemy środkowe punkty
    scan_ref.sort(key=lambda p: p[0])
    selected_ref = scan_ref[n_start_ref : min(n_end_ref, len(scan_ref))]
    pts_ref = np.array([(p[1], p[2]) for p in selected_ref])

    result_ref = fit_circle_lsm(pts_ref)
    if result_ref[0] is None:
        # lsm nie powiodło się - zwracamy wynik z kroku 1
        return x_i1, y_i1, r_i1

    local_xi_f, local_yi_f, refined_ri = result_ref

    # przeliczamy z układu lokalnego roi do globalnego układu obrazu
    final_xi = local_xi_f + x1
    final_yi = local_yi_f + y1

    return final_xi, final_yi, refined_ri
