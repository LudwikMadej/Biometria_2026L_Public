import numpy as np
import cv2

def polar_to_cartesian(p_params, i_params, res_r=60, res_theta=360):
    """
    Tworzy mapy odwzorowania dla modelu gumowej płachty Daugmana.
    res_r: wysokość wynikowego prostokąta (rozdzielczość radialna)
    res_theta: szerokość wynikowego prostokąta (rozdzielczość kątowa)
    """
    xp, yp, rp = p_params
    xi, yi, ri = i_params

    # Przygotowanie siatki współrzędnych dla obrazu wyjściowego
    r = np.linspace(0, 1, res_r)
    theta = np.linspace(0, 2 * np.pi, res_theta)
    r_grid, theta_grid = np.meshgrid(r, theta)

    # Obliczanie współrzędnych brzegowych dla każdego kąta theta
    # Brzeg źrenicy (x_p, y_p)
    x_p_edge = xp + rp * np.cos(theta_grid)
    y_p_edge = yp + rp * np.sin(theta_grid)

    # Brzeg tęczówki (x_i, y_i)
    x_i_edge = xi + ri * np.cos(theta_grid)
    y_i_edge = yi + ri * np.sin(theta_grid)

    # Liniowa interpolacja między brzegami (Gumowa Płachta)
    map_x = (1 - r_grid) * x_p_edge + r_grid * x_i_edge
    map_y = (1 - r_grid) * y_p_edge + r_grid * y_i_edge

    return map_x.astype(np.float32).T, map_y.astype(np.float32).T

def unroll_iris(img, p_params, i_params, width=360, height=60):
    """
    Rozwija tęczówkę do prostokąta o zadanych wymiarach.
    """
    # 1. Wygeneruj mapy mapowania
    map_x, map_y = polar_to_cartesian(p_params, i_params, res_r=height, res_theta=width)

    # 2. Wykonaj remapowanie z interpolacją bilingową
    # INTER_CUBIC daje ładniejsze wygładzenie tekstury tęczówki
    normalized_iris = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC)

    return normalized_iris

# --- Przykład użycia ---
# pupil = (100.5, 100.2, 30.0)  # x, y, r
# iris = (102.0, 99.5, 85.0)   # x, y, r
# image = cv2.imread('oko.jpg', 0)
#
# rectangle = unroll_iris(image, pupil, iris)
# cv2.imshow("Normalized Iris", rectangle)