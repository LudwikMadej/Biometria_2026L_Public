import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_iris_geometry(img, x_p, y_p, r_p, 
                              n_start_adv=15, n_end_adv=25, 
                              n_start_ref=10, n_end_ref=20,
                              roi_factor=1.2,
                              show_plots=False):
    """
    Pełny proces detekcji tęczówki:
    1. detect_iris_advanced (Globalny skan na bazie źrenicy)
    2. refine_iris_geometry (Lokalne dopasowanie w ROI)
    """

    # Funkcja pomocnicza do dopasowania okręgu (LSM)
    def fit_circle_lsm(points):
        x, y = points[:, 0], points[:, 1]
        A = np.column_stack([x * 2, y * 2, np.ones(len(x))])
        B = x**2 + y**2
        res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        xc, yc = res[0], res[1]
        rc = np.sqrt(res[2] + xc**2 + yc**2)
        return xc, yc, rc

    # ==========================================================
    # KROK 1: detect_iris_advanced
    # ==========================================================
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    threshold_val_adv = np.quantile(img, 0.2)
    _, iris_bin_adv = cv2.threshold(enhanced, threshold_val_adv, 255, cv2.THRESH_BINARY_INV)

    kernel_adv = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cleaned_adv = cv2.morphologyEx(iris_bin_adv, cv2.MORPH_CLOSE, kernel_adv)
    cleaned_adv = cv2.morphologyEx(cleaned_adv, cv2.MORPH_OPEN, kernel_adv)

    angles = np.linspace(0, 2*np.pi, 72, endpoint=False)
    scan_adv = [] 
    max_scan_adv = int(r_p * 4.5)
    
    for angle in angles:
        dx, dy = np.cos(angle), np.sin(angle)
        for dist in range(int(r_p * 1.1), max_scan_adv):
            curr_x, curr_y = int(x_p + dx * dist), int(y_p + dy * dist)
            if curr_y >= cleaned_adv.shape[0] or curr_y < 0 or curr_x >= cleaned_adv.shape[1] or curr_x < 0:
                break
            if cleaned_adv[curr_y, curr_x] == 0:
                scan_adv.append((dist, curr_x, curr_y))
                break
    
    if len(scan_adv) < 3:
        # Fallback do etapu 1
        x_i1, y_i1, r_i1 = x_p, y_p, r_p * 2
    else:
        scan_adv.sort(key=lambda x: x[0])
        actual_end_adv = min(n_end_adv, len(scan_adv))
        pts_adv = np.array([(p[1], p[2]) for p in scan_adv[n_start_adv:actual_end_adv]])
        x_i1, y_i1, r_i1 = fit_circle_lsm(pts_adv)

    if show_plots:
        print(f"DEBUG: Stage 1 (Advanced) - R: {r_i1:.2f}")

    # ==========================================================
    # KROK 2: refine_iris_geometry (korzysta z wyników Kroku 1)
    # ==========================================================
    h, w = img.shape
    roi_margin = int(r_i1 * roi_factor)
    x1 = max(0, int(x_i1 - roi_margin))
    y1 = max(0, int(y_i1 - roi_margin))
    x2 = min(w, int(x_i1 + roi_margin))
    y2 = min(h, int(y_i1 + roi_margin))
    
    roi = img[y1:y2, x1:x2]
    local_x_c = x_i1 - x1
    local_y_c = y_i1 - y1

    local_thresh = np.median(roi)
    _, local_bin = cv2.threshold(roi, local_thresh, 255, cv2.THRESH_BINARY_INV)
    kernel_ref = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_ref = cv2.morphologyEx(local_bin, cv2.MORPH_CLOSE, kernel_ref)

    scan_ref = []
    search_range_min = int(r_i1 * 0.8)
    search_range_max = int(r_i1 * 1.3)

    for angle in angles:
        dx, dy = np.cos(angle), np.sin(angle)
        for dist in range(search_range_min, search_range_max):
            curr_x = int(local_x_c + dx * dist)
            curr_y = int(local_y_c + dy * dist)
            if 0 <= curr_y < cleaned_ref.shape[0] and 0 <= curr_x < cleaned_ref.shape[1]:
                if cleaned_ref[curr_y, curr_x] == 0:
                    scan_ref.append((dist, curr_x, curr_y))
                    break
    
    if len(scan_ref) < 10:
        # Jeśli lokalny skan zawiedzie, zwracamy wynik z Kroku 1
        return x_i1, y_i1, r_i1

    scan_ref.sort(key=lambda x: x[0])
    selected_ref = scan_ref[n_start_ref:min(n_end_ref, len(scan_ref))]
    pts_ref = np.array([(p[1], p[2]) for p in selected_ref])

    local_xi_f, local_yi_f, refined_ri = fit_circle_lsm(pts_ref)

    # Finalna translacja współrzędnych do globalnego układu
    final_xi = local_xi_f + x1
    final_yi = local_yi_f + y1

    return final_xi, final_yi, refined_ri