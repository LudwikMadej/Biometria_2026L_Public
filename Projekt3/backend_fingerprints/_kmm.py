import numpy as np
from ._utils import to_foreground_mask, from_foreground_mask

# tablica usuwania KMM: piksele o takiej wadze sąsiedztwa (0-255) są kasowane
# w fazach N=2 oraz N=3 (zgodnie z dokumentem).
_KMM_DELETION = np.zeros(256, dtype=bool)
for _v in (
    3,
    5,
    7,
    12,
    13,
    14,
    15,
    20,
    21,
    22,
    23,
    28,
    29,
    30,
    31,
    48,
    52,
    53,
    54,
    55,
    56,
    60,
    61,
    62,
    63,
    65,
    67,
    69,
    71,
    77,
    79,
    80,
    81,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    91,
    92,
    93,
    94,
    95,
    97,
    99,
    101,
    103,
    109,
    111,
    112,
    113,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    123,
    124,
    125,
    126,
    127,
    131,
    133,
    135,
    141,
    143,
    149,
    151,
    157,
    159,
    181,
    183,
    189,
    191,
    192,
    193,
    195,
    197,
    199,
    205,
    207,
    208,
    209,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
    227,
    229,
    231,
    237,
    239,
    240,
    241,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    251,
    252,
    253,
    254,
    255,
):
    _KMM_DELETION[_v] = True


# kierunki ośmiu sąsiadów wraz z wagą (N, NE, E, SE, S, SW, W, NW) zgodne z macierzą wag
# Porządek wpływa również na sprawdzanie, czy foreground-sąsiedzi tworzą spójny łuk wokół piksela.
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


def _weight(labels: np.ndarray, y: int, x: int) -> int:
    """
    Waga sąsiedztwa pojedynczego piksela (y, x) w macierzy etykiet.
    Za piksel pierwszego planu uważamy każdy, którego etykieta jest niezerowa
    (czyli 1, 2 lub 3 - piksele o etykiecie '4' zostały wcześniej skasowane).
    """
    h, w = labels.shape
    total = 0

    # przeglądamy ośmiu sąsiadów i dodajemy wagę tylko wtedy, gdy
    # stoi tam piksel pierwszego planu
    for dy, dx, wt in _DIRECTIONS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] != 0:
            total += wt
    return total


def _contiguous_arc_length(labels: np.ndarray, y: int, x: int) -> int:
    """
    Zwraca długość pojedynczego, spójnego łuku sąsiadów pierwszego planu
    wokół piksela (y, x) w porządku N, NE, E, SE, S, SW, W, NW.

    Jeżeli piksele pierwszego planu tworzą więcej niż jeden łuk (np. są
    "po dwóch stronach" piksela), zwraca -1. Jeśli nie ma foreground-sąsiadów -
    zwraca 0. Dla wszystkich ośmiu sąsiadów pierwszego planu zwraca 8.
    """
    h, w = labels.shape

    # zbieramy stan sąsiedztwa jako 0/1 w pierścieniu w kolejności zgodnej
    # z ruchem wskazówek zegara
    ring = []
    for dy, dx, _ in _DIRECTIONS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] != 0:
            ring.append(1)
        else:
            ring.append(0)

    total = sum(ring)
    if total == 0:
        return 0
    if total == 8:
        return 8

    # liczymy zmiany tło<->pierwszy plan w cyklu - pojedynczy spójny łuk
    # daje dokładnie dwie takie zmiany (wejście w łuk i wyjście z niego)
    # (sprawdza ile jest nieciągłości na krawędziach łuku)
    transitions = 0
    for i in range(8):
        if ring[i] != ring[(i + 1) % 8]:
            transitions += 1

    if transitions == 2:
        return total
    return -1


def _mark_borders(labels: np.ndarray) -> None:
    """
    Zaznacza krawędzie (etykieta 2) oraz narożniki (etykieta 3).
    Pikseli wewnętrznych nie rusza - zostają z etykietą 1.
    """
    h, w = labels.shape
    for y in range(h):
        for x in range(w):
            # interesują nas tylko piksele pierwszego planu, których jeszcze
            # nie skategoryzowano
            if labels[y, x] != 1:
                continue

            # "edge touch" - brak sąsiada pierwszego planu w jednym z kierunków
            # osiowych oznacza, że piksel przylega do tła krawędzią
            edge_touch = False
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w) or labels[ny, nx] == 0:
                    edge_touch = True
                    break
            if edge_touch:
                labels[y, x] = 2
                continue

            # "corner touch" - analogicznie dla kierunków po przekątnych;
            # osiowy już sprawdzono więc patrzymy tylko na rogi
            corner_touch = False
            for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w) or labels[ny, nx] == 0:
                    corner_touch = True
                    break
            if corner_touch:
                labels[y, x] = 3


def kmm(
    image: np.ndarray, foreground: str = "dark", max_iterations: int = 200
) -> np.ndarray:
    """
    Ścienia obraz binarny algorytmem KMM.

    Iteracja algorytmu wykonuje kolejno:
        1. Zaznacza wszystkie piksele pierwszego planu etykietą '1'.
        2. Piksele stykające się z tłem krawędzią (N/S/W/E) dostają etykietę '2',
           a stykające się tylko rogiem - '3'.
        3. Piksele oznaczone '2' lub '3', których foreground-sąsiedzi tworzą
           spójny łuk długości 2, 3 albo 4, otrzymują etykietę '4' i są od razu
           usuwane (redundantne narożniki / krótkie krawędzie).
        4. W porządku rastrowym dla każdego piksela o etykiecie '2' liczona jest
           waga sąsiedztwa (wg tabeli 128/1/2/64/x/4/32/16/8). Jeżeli waga znajduje
           się w tablicy usunięć, piksel jest kasowany, w przeciwnym razie wraca
           do etykiety '1'. Ten sam krok powtarzany jest dla pikseli '3'.
        5. Iteracja jest powtarzana dopóki w ciągu jednej iteracji wykonano
           choćby jedno usunięcie.

    Args:
        image (np.ndarray): obraz binarny 2D (0/255 lub bool).
        foreground (str): "dark" (linie papilarne ciemne) albo "light".
        max_iterations (int): zabezpieczenie przed nieskończoną pętlą.

    Returns:
        np.ndarray: obraz binarny (0/255, uint8) ze ścienionym szkieletem.
    """
    # etykiety: 0 = tło, 1 = pierwszy plan, 2/3 = kontury, 4 = do skasowania
    labels = to_foreground_mask(image, foreground).astype(np.int32)

    for _ in range(max_iterations):
        any_deleted = False

        # na początku każdej iteracji wszystkie piksele pierwszego planu
        # resetujemy do etykiety 1, żeby zaznaczanie konturu odbyło się
        # na czystym stanie
        labels = (labels > 0).astype(np.int32)
        _mark_borders(labels)

        h, w = labels.shape

        # krok: piksele '2'/'3' o spójnym łuku długości 2-4 oznaczamy '4'
        for y in range(h):
            for x in range(w):
                if labels[y, x] in (2, 3):
                    arc = _contiguous_arc_length(labels, y, x)
                    if 2 <= arc <= 4:
                        labels[y, x] = 4

        # krok: skasowanie wszystkich pikseli '4' naraz
        mask_four = labels == 4
        if np.any(mask_four):
            labels[mask_four] = 0
            any_deleted = True

        # krok: skanowanie rastrowe najpierw dla N=2, potem dla N=3
        # (sekwencyjne - usunięcie jednego piksela może wpłynąć na wagę
        # kolejnych w tej samej iteracji)
        for target in (2, 3):
            for y in range(h):
                for x in range(w):
                    if labels[y, x] != target:
                        continue
                    w_val = _weight(labels, y, x)
                    if _KMM_DELETION[w_val]:
                        labels[y, x] = 0
                        any_deleted = True
                    else:
                        # piksel niezbędny do zachowania ciągłości - wraca
                        # do zwykłego pierwszego planu
                        labels[y, x] = 1

        # jeżeli żaden piksel nie został usunięty w pełnej iteracji,
        # szkielet się ustabilizował
        if not any_deleted:
            break

    # finalna maska: wszystkie niezerowe etykiety to pierwszy plan
    mask = (labels > 0).astype(np.uint8)
    return from_foreground_mask(mask, foreground)
