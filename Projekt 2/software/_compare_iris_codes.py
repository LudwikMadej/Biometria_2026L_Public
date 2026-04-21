import numpy as np


def compare_iris_codes(
    code1: np.ndarray,
    code2: np.ndarray,
    n_shifts: int = 10,
    n_bands: int = 8,
) -> float:
    """
    Porównuje dwa kody tęczówki za pomocą znormalizowanej odległości Hamminga.

    Dlaczego przesunięcia: podczas dwóch zdjęć oko może być nieznacznie
    obrócone. Jedno przesunięcie tablicy bitów o bits_per_step pozycji
    odpowiada dokładnie jednemu krokowi kątowemu - dzięki przeplotowemu
    układowi bitów z encode_iris. Sprawdzamy [-n_shifts, +n_shifts] i
    bierzemy minimum, bo najlepsze dopasowanie zniweluje rotację.

    Args:
        code1, code2: Binarne kody tęczówki (uint8) tej samej długości.
        n_shifts:     Maksymalne przesunięcie w jednostkach próbek kątowych.
        n_bands:      Liczba pasm radialnych - musi zgadzać się z encode_iris.

    Returns:
        Znormalizowana odległość Hamminga ∈ [0, 1].
        Wartości < ~0.32 wskazują tę samą tęczówkę,
        wartości > ~0.45 wskazują różne tęczówki.
    """
    # każda próbka kątowa wnosi n_bands bitów rzeczywistych + n_bands urojonych
    bits_per_step = n_bands * 2

    # startujemy od najgorszego możliwego wyniku - każde realne porównanie może tylko poprawić
    min_hd = 1.0

    for shift in range(-n_shifts, n_shifts + 1):
        # przesuwamy code1 o `shift` próbek kątowych względem code2,
        # co kompensuje ewentualną rotację oka między ujęciami
        rolled = np.roll(code1, shift * bits_per_step)

        # frakcja pozycji bitowych, w których kody się różnią
        hd = float(np.mean(rolled != code2))

        # zachowujemy tylko najlepsze (najmniejsze) dopasowanie
        if hd < min_hd:
            min_hd = hd

    return min_hd
