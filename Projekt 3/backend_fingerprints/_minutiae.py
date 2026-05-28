import numpy as np
from ._utils import to_foreground_mask


# kolejność ośmiu sąsiadów zgodna z definicją Crossing Number (zgodnie z ruchem
# wskazówek zegara, startując od piksela "na północ"): N, NE, E, SE, S, SW, W, NW
_NEIGHBOUR_OFFSETS = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]


def _crossing_number_map(mask: np.ndarray) -> np.ndarray:
    """
    Oblicza Crossing Number (CN) (stabilniejsza detekcja zakończeń i rozwidleń) dla każdego piksela maski zgodnie ze wzorem:

        CN(p) = 0.5 * sum_{i=1-8} |P_i - P_{i+1}|

    gdzie P_1-P_8 to osiem sąsiadów w porządku zgodnym z ruchem wskazówek
    zegara, a P_9 = P_1.

    Args:
        mask (np.ndarray): maska uint8 {0, 1}.

    Returns:
        np.ndarray: macierz int32 z wartościami CN dla każdego piksela.
    """
    # padding zerami, żeby piksele przy brzegu miały fikcyjnych sąsiadów
    padded = np.pad(mask.astype(np.int32), 1, mode="constant", constant_values=0)
    h, w = mask.shape

    # dla każdego z ośmiu kierunków wycinamy przesunięte okno odpowiadające
    # sąsiadowi P_i dla wszystkich pikseli naraz (wektoryzacja)
    neighbours = []
    for dy, dx in _NEIGHBOUR_OFFSETS:
        neighbours.append(padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w])
    # zamykamy cykl P_9 = P_1
    neighbours.append(neighbours[0])

    # sumujemy moduły różnic kolejnych sąsiadów - to liczba zmian tło<->kontur
    # w pierścieniu 3x3 wokół piksela
    diffs = np.zeros_like(mask, dtype=np.int32)
    for i in range(8):
        diffs += np.abs(neighbours[i] - neighbours[i + 1])
    # dzielenie przez 2, bo każda zmiana wliczana jest dwukrotnie (wejście + wyjście)
    return diffs // 2


def detect_minutiae(
    skeleton: np.ndarray,
    foreground: str = "dark",
    border_margin: int = 1,
):
    """
    Wyznacza lokalizacje minucji w ścienionym obrazie odcisku
    palca przy użyciu Crossing Number.

        CN = 1  ->  zakończenie linii (ridge ending),
        CN = 3  ->  bifurkacja (ridge bifurcation).

    Uwaga: metoda wymaga szkieletu o szerokości jednego piksela. Piksele leżące
    bliżej krawędzi niż `border_margin` są pomijane, bo w ich otoczeniu brakuje
    kontekstu potrzebnego do rzetelnej klasyfikacji.

    Args:
        skeleton (np.ndarray): obraz binarny 2D (0/255 lub bool)
            z jednopikselowym szkieletem linii papilarnych.
        foreground (str): "dark" jeśli krawędzie są ciemne, "light" - jasne.
        border_margin (int): szerokość marginesu ignorowanego przy brzegach.

    Returns:
        dict: słownik z kluczami:
            - "endings":     np.ndarray shape (N, 2) z (y, x) zakończeń,
            - "bifurcations": np.ndarray shape (M, 2) z (y, x) bifurkacji,
            - "cn_map":      mapa Crossing Number (np.int32).
    """
    # sprowadzamy obraz do jednolitej maski 0/1 (1 = krawędź)
    mask = to_foreground_mask(skeleton, foreground)
    # wyliczamy CN dla każdego piksela - potem filtrujemy tylko te należące do szkieletu
    cn = _crossing_number_map(mask)

    # maska pikseli, które w ogóle mogą być minucją (krawędź + odpowiedni odstęp od brzegu)
    valid = mask.astype(bool).copy()
    if border_margin > 0:
        # odrzucamy `border_margin` pikseli z każdej strony - tam szkielet
        # bywa poszarpany przez krawędź obrazu i daje fałszywe minucje
        valid[:border_margin, :] = False
        valid[-border_margin:, :] = False
        valid[:, :border_margin] = False
        valid[:, -border_margin:] = False

    # CN = 1 -> dokładnie jedno sąsiedztwo tło<->krawędź w pierścieniu -> koniec linii
    endings = np.argwhere(valid & (cn == 1))
    # CN = 3 -> trzy takie przejścia -> rozwidlenie (bifurkacja)
    bifurcations = np.argwhere(valid & (cn == 3))

    return {
        "endings": endings,
        "bifurcations": bifurcations,
        "cn_map": cn,
    }
