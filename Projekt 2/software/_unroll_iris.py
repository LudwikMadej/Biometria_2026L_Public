import numpy as np
import cv2


def polar_to_cartesian(p_params, i_params, res_r=60, res_theta=360):
    """
    Tworzy mapy odwzorowania dla modelu Daugmana (rubber-sheet model).

    Pomysł: każdy punkt tęczówki opisujemy współrzędnymi (r, theta), gdzie
    r=0 to granica źrenicy, a r=1 to granica tęczówki. Dzięki temu obraz
    tęczówki staje się prostokątem niezależnym od rozmiaru źrenicy/tęczówki.

    res_r:     wysokość wynikowego prostokąta (rozdzielczość radialna)
    res_theta: szerokość wynikowego prostokąta (rozdzielczość kątowa)
    """
    xp, yp, rp = p_params
    xi, yi, ri = i_params

    # siatka współrzędnych prostokąta wyjściowego:
    # r ∈ [0, 1] - od granicy źrenicy (0) do granicy tęczówki (1)
    # theta ∈ [0, 2π] - pełny obrót wokół osi oka
    r = np.linspace(0, 1, res_r)
    theta = np.linspace(0, 2 * np.pi, res_theta)

    # meshgrid tworzy macierze 2d - dla każdego piksela (r, theta)
    # znamy jednocześnie wartość r i theta bez pętli
    r_grid, theta_grid = np.meshgrid(r, theta)

    # punkt na granicy źrenicy dla każdego kąta theta
    x_p_edge = xp + rp * np.cos(theta_grid)
    y_p_edge = yp + rp * np.sin(theta_grid)

    # punkt na granicy tęczówki dla każdego kąta theta
    x_i_edge = xi + ri * np.cos(theta_grid)
    y_i_edge = yi + ri * np.sin(theta_grid)

    # liniowa interpolacja: dla r=0 bierzemy punkt na źrenicy,
    # dla r=1 bierzemy punkt na tęczówce - stąd nazwa "gumowa płachta"
    # (obraz jest jakby rozciągany między tymi dwoma granicami)
    map_x = (1 - r_grid) * x_p_edge + r_grid * x_i_edge
    map_y = (1 - r_grid) * y_p_edge + r_grid * y_i_edge

    # ascontiguousarray zapewnia ciągłość pamięci wymaganą przez cv2.remap
    return (
        np.ascontiguousarray(map_x.astype(np.float32).T),
        np.ascontiguousarray(map_y.astype(np.float32).T),
    )


def unroll_iris(img, p_params, i_params, width=360, height=60):
    """
    Rozwija tęczówkę z obrazu biegunowego do prostokąta (width × height).
    """
    # generujemy mapy piksel → współrzędna źródłowa
    map_x, map_y = polar_to_cartesian(p_params, i_params, res_r=height, res_theta=width)

    # cv2.remap próbkuje piksele z oryginalnego obrazu według wygenerowanych map;
    # inter_cubic daje lepsze wygładzenie tekstury tęczówki niż inter_linear
    # kosztem nieznacznie większego czasu obliczeń
    normalized_iris = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC)

    return normalized_iris
