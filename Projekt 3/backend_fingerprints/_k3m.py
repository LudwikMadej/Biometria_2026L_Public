import numpy as np
from ._utils import (
    to_foreground_mask,
    from_foreground_mask,
    compute_neighbour_weight_map,
)


# tablice A0-A5 oraz A1pix zgodnie z opisem algorytmu K3M . Każda tablica
# to zbiór wag sąsiedztwa 0-255, dla których piksel w danej fazie jest
# kasowany (kandydaci do usunięcia w danym kroku)

# A0 - wagi oznaczające piksele-brzegi (kandydaci do sprawdzenia w fazach 1..5)
A0 = frozenset(
    {
        3,
        6,
        7,
        12,
        14,
        15,
        24,
        28,
        30,
        31,
        48,
        56,
        60,
        62,
        63,
        96,
        112,
        120,
        124,
        126,
        127,
        129,
        131,
        135,
        143,
        159,
        191,
        192,
        193,
        195,
        199,
        207,
        223,
        224,
        225,
        227,
        231,
        239,
        240,
        241,
        243,
        247,
        248,
        249,
        251,
        252,
        253,
        254,
    }
)
# A1 - usunięcie pikseli z 3 przylegającymi sąsiadami
A1 = frozenset({7, 14, 28, 56, 112, 131, 193, 224})
# A2 - 3 lub 4 przylegających sąsiadów
A2 = frozenset({7, 14, 15, 28, 30, 56, 60, 112, 120, 131, 135, 193, 195, 224, 225, 240})
# A3 - 3-5 przylegających
A3 = frozenset(
    {
        7,
        14,
        15,
        28,
        30,
        31,
        56,
        60,
        62,
        112,
        120,
        124,
        131,
        135,
        143,
        193,
        195,
        199,
        224,
        225,
        227,
        240,
        241,
        248,
    }
)
# A4 - 3-6
A4 = frozenset(
    {
        7,
        14,
        15,
        28,
        30,
        31,
        56,
        60,
        62,
        63,
        112,
        120,
        124,
        126,
        131,
        135,
        143,
        159,
        193,
        195,
        199,
        207,
        224,
        225,
        227,
        231,
        240,
        241,
        243,
        248,
        249,
        252,
    }
)
# A5 - 3-7
A5 = frozenset(
    {
        7,
        14,
        15,
        28,
        30,
        31,
        56,
        60,
        62,
        63,
        112,
        120,
        124,
        126,
        131,
        135,
        143,
        159,
        191,
        193,
        195,
        199,
        207,
        224,
        225,
        227,
        231,
        239,
        240,
        241,
        243,
        248,
        249,
        251,
        252,
        254,
    }
)
# A1pix - dodatkowa tablica dla fazy doprowadzającej szkielet do jednopikselowej
# szerokości (stosowana po zakończeniu głównych iteracji)
A1PIX = frozenset(
    {
        3,
        6,
        7,
        12,
        14,
        15,
        24,
        28,
        30,
        31,
        48,
        56,
        60,
        62,
        63,
        96,
        112,
        120,
        124,
        126,
        127,
        129,
        131,
        135,
        143,
        159,
        191,
        192,
        193,
        195,
        199,
        207,
        223,
        224,
        225,
        227,
        231,
        239,
        240,
        241,
        243,
        247,
        248,
        249,
        251,
        252,
        253,
        254,
    }
)


def _lookup_array_to_bool_table(lookup: frozenset) -> np.ndarray:
    """Zamienia zbiór wag (0-255) na tablicę boolowską o długości 256."""
    # dzięki zamianie na tablicę boolowską sprawdzenie „czy waga w tablicy"
    # jest pojedynczym dostępem indeksowanym zamiast szukania w zbiorze
    table = np.zeros(256, dtype=bool)
    for v in lookup:
        table[v] = True
    return table


_TABLE_A0 = _lookup_array_to_bool_table(A0)
_TABLE_PHASES = [_lookup_array_to_bool_table(a) for a in (A1, A2, A3, A4, A5)]
_TABLE_A1PIX = _lookup_array_to_bool_table(A1PIX)


# lista ośmiu kierunków sąsiedztwa z wagą; używana podczas lokalnego
# (sekwencyjnego) liczenia wagi pojedynczego piksela
_DIRECTIONS = (
    (-1, 0, 1),
    (-1, 1, 2),
    (0, 1, 4),
    (1, 1, 8),
    (1, 0, 16),
    (1, -1, 32),
    (0, -1, 64),
    (-1, -1, 128),
)


def _single_weight(mask: np.ndarray, y: int, x: int) -> int:
    """Waga sąsiedztwa jednego piksela liczona od razu z aktualnego stanu maski."""
    h, w = mask.shape
    weight = 0
    # sumujemy wagi kierunków, w których aktualnie znajduje się piksel
    for dy, dx, wt in _DIRECTIONS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 1:
            weight += wt
    return weight


def _delete_phase(
    mask: np.ndarray, borders: np.ndarray, phase_table: np.ndarray
) -> bool:
    """
    Usuwa piksele-brzegi, których waga sąsiedztwa znajduje się
    w tablicy `phase_table`. Modyfikuje `mask` i `borders` w miejscu.
    Zwraca True, jeżeli cokolwiek usunięto.
    """
    any_deleted = False
    coords = np.argwhere(borders)
    if coords.size == 0:
        return False

    # przechodzimy po brzegach sekwencyjnie - po skasowaniu piksela waga
    # kolejnego kandydata może się zmienić, dlatego wagę liczymy na bieżąco
    # dla aktualnej maski
    for y, x in coords:
        weights = _single_weight(mask, int(y), int(x))
        if phase_table[weights]:
            mask[y, x] = 0
            borders[y, x] = False
            any_deleted = True
    return any_deleted


def k3m(
    image: np.ndarray, foreground: str = "dark", max_iterations: int = 200
) -> np.ndarray:
    """
    Ścienia obraz binarny algorytmem K3M.

    Kroki iteracji algorytmu:
        Faza 0: oznaczenie pikseli-brzegów (tych, dla których waga sąsiedztwa
                znajduje się w tablicy A0).
        Fazy 1-5: usunięcie brzegów posiadających kolejno:
                   3 / 3 lub 4 / 3-5 / 3-6 / 3-7 przylegających sąsiadów
                   (tablice A1-A5).
        Faza 6: "odznaczenie" pozostałych brzegów - piksele wracają do zwykłego
                pierwszego planu.
        Iteracje powtarzane są dopóki w iteracji zachodzi jakakolwiek zmiana.

    Na końcu uruchamiana jest dodatkowa faza wykorzystująca tablicę A1pix,
    która sprowadza szkielet do szerokości jednego piksela.

    Args:
        image (np.ndarray): obraz binarny 2D (0/255 lub bool).
        foreground (str): "dark" jeśli linie papilarne są ciemne,
            "light" jeśli linie papilarne są jasne.
        max_iterations (int): zabezpieczenie przed nieskończoną pętlą.

    Returns:
        np.ndarray: ścieniony obraz binarny (0/255, uint8).
    """
    # pracujemy na masce 0/1, gdzie 1 oznacza ridge (linia papilarna)
    mask = to_foreground_mask(image, foreground).copy()

    for _ in range(max_iterations):
        modified = False

        # faza 0: obliczamy wagi dla całej maski i wybieramy piksele-brzegi
        # (tylko te są kandydatami do usunięcia w tej iteracji)
        weights = compute_neighbour_weight_map(mask)
        borders = (mask == 1) & _TABLE_A0[weights]

        # jeśli nie ma żadnego piksela-brzegu, szkielet już stabilny
        if not np.any(borders):
            break

        # fazy 1-5: kolejno usuwamy coraz "gęstsze" brzegi - im dalej iteracja
        # tym większa dopuszczalna liczba przylegających sąsiadów
        for phase_table in _TABLE_PHASES:
            if _delete_phase(mask, borders, phase_table):
                modified = True

        # faza 6 to tylko "odznaczenie" brzegów - w naszym modelu nie trzeba
        # nic robić, bo `borders` była lokalna tylko dla tej iteracji

        # jeżeli żaden piksel nie został usunięty - koniec głównej pętli
        if not modified:
            break

    # dodatkowa faza: redukcja do szkieletu o szerokości jednego piksela
    # (A1pix). powtarzamy aż przestaniemy cokolwiek usuwać.
    for _ in range(max_iterations):
        weights = compute_neighbour_weight_map(mask)
        candidates = (mask == 1) & _TABLE_A1PIX[weights]
        if not np.any(candidates):
            break

        any_removed = False
        for y, x in np.argwhere(candidates):
            # sprawdzamy wagę jeszcze raz na bieżącej masce, bo między
            # wybraniem kandydatów a obecną iteracją mogła się zmienić
            w = _single_weight(mask, int(y), int(x))
            if _TABLE_A1PIX[w]:
                mask[y, x] = 0
                any_removed = True
        if not any_removed:
            break

    return from_foreground_mask(mask, foreground)
