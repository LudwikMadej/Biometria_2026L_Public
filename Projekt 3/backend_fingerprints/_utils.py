import numpy as np


# macierz wag 3x3 używana w KMM i K3M - każdy z ośmiu sąsiadów ma
# przypisaną unikalną potęgę dwójki, dzięki czemu suma wag jednoznacznie
# kodujące konfigurację sąsiedztwa mieści się w przedziale 0-255
NEIGHBOUR_WEIGHTS = np.array(
    [[128, 1, 2], [64, 0, 4], [32, 16, 8]],
    dtype=np.int32,
)


def to_foreground_mask(image: np.ndarray, foreground: str = "dark") -> np.ndarray:
    """
    Konwertuje obraz binarny (0/255 lub bool) na maskę 0/1 typu uint8,
    gdzie 1 oznacza piksele linii papilarnych (pierwszy plan).

    Args:
        image (np.ndarray): obraz binarny 2D.
        foreground (str): "dark" jeśli ridge = wartości ciemne (0),
            "light" jeśli ridge = wartości jasne (255).

    Returns:
        np.ndarray: maska uint8 z wartościami {0, 1}.
    """
    if image.ndim != 2:
        raise ValueError("Oczekiwano obrazu 2D.")

    if image.dtype == bool:
        mask = image.astype(np.uint8)
    else:
        mask = (image > 0).astype(np.uint8)

    # dla odcisków palców linie są ciemne - po binaryzacji mają wartość 0,
    # więc trzeba odwrócić maskę, żeby dostać 1 = ridge
    if foreground == "dark":
        mask = 1 - mask
    elif foreground != "light":
        raise ValueError("foreground musi być 'dark' albo 'light'.")

    return mask


def from_foreground_mask(mask: np.ndarray, foreground: str = "dark") -> np.ndarray:
    """
    Przekształca maskę 0/1 z powrotem do obrazu binarnego 0/255 (uint8),
    zachowując orientację tła/pierwszego planu z `to_foreground_mask`.
    """
    # normalizacja - zabezpieczenie na wypadek wartości innych niż 0/1
    mask = (mask > 0).astype(np.uint8)

    # przywracanie oryginalnej konwencji kolorów pierwszego planu
    if foreground == "dark":
        return ((1 - mask) * 255).astype(np.uint8)
    if foreground == "light":
        return (mask * 255).astype(np.uint8)
    raise ValueError("foreground musi być 'dark' albo 'light'.")


def compute_neighbour_weight_map(mask: np.ndarray) -> np.ndarray:
    """
    Wylicza wagę sąsiedztwa (0-255) dla każdego piksela w masce 0/1.

    Każdemu z ośmiu sąsiadów przypisywana jest waga z NEIGHBOUR_WEIGHTS,
    a waga piksela to suma wag tych sąsiadów, którzy są pierwszym planem.

    Args:
        mask (np.ndarray): maska uint8 z wartościami {0, 1}.

    Returns:
        np.ndarray: macierz wag tej samej wielkości (int32).
    """
    # padding zerami pozwala jednolicie przetwarzać piksele brzegowe - sąsiedzi
    # spoza obrazu są traktowani jak tło
    padded = np.pad(mask.astype(np.int32), 1, mode="constant", constant_values=0)

    weights = np.zeros_like(mask, dtype=np.int32)
    h, w = mask.shape

    # przesuwamy okno 3x3 po macierzy dodając wkład każdego sąsiada - dzięki
    # temu unikamy jawnej pętli po każdym pikselu
    for di in range(3):
        for dj in range(3):
            w_val = int(NEIGHBOUR_WEIGHTS[di, dj])
            if w_val == 0:
                continue
            weights += padded[di : di + h, dj : dj + w] * w_val
    return weights
