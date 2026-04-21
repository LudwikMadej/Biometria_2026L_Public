import numpy as np


def _gabor_kernels(f: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Buduje jednowymiarowe jądra falki Gabora (część rzeczywista i urojona).

    Falka Gabora to iloczyn gaussowskiej koperty i fali sinusoidalnej.
    Do ekstrakcji fazy sygnału tęczówki - znak odpowiedzi
    rzeczywistej i urojonej koduje lokalną fazę w każdym paśmie radialnym.

    Args:
        f: Częstotliwość falki w cyklach na próbkę.
    """
    # sigma gaussowskiej koperty: sigma = 1/(2πf) - wynika bezpośrednio
    # ze specyfikacji algorytmu daugmana; szeroka koperta (małe f) uchwyci
    # gruboziarnistą strukturę tęczówki, wąska (duże f) - drobne tekstury
    sigma = 0.5 * np.pi * f

    # długość połowy jądra: 3*sigma obejmuje ~99.7% energii gaussowskiej
    # (reguła 3 sigma), co minimalizuje artefakty przycięcia
    half = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-half, half + 1, dtype=np.float64)

    # koperta gaussowska - tłumi próbki daleko od centrum
    envelope = np.exp(-(x**2) / (2.0 * sigma**2))

    # część rzeczywista: kosinusoidalna modulacja (jądro parzyste)
    real = envelope * np.cos(2.0 * np.pi * f * x)

    # część urojona: sinusoidalna modulacja (jądro nieparzyste, przesunięcie fazy o 90°)
    # razem real i imag tworzą parę hilberta - reprezentacja fazorowa sygnału
    imag = envelope * np.sin(2.0 * np.pi * f * x)

    return real, imag


def _radial_gauss_weights(n: int) -> np.ndarray:
    """
    Wagi gaussowskie do uśredniania radialnego wewnątrz jednego pasma n wierszy.

    Środkowe wiersze pasma mają wyższy wkład niż brzegowe - zmniejsza to
    wpływ nieciągłości na krawędzi pasma, które mogłyby zaburzać sygnał.
    """
    # środek gaussowskiej w połowie pasma
    center = (n - 1) / 2.0
    # sigma = n/4 - gaussowska mieści się wewnątrz pasma bez twardego ucięcia
    sigma = n / 4.0
    w = np.exp(-((np.arange(n) - center) ** 2) / (2.0 * sigma**2))
    # normalizacja: wagi sumują się do 1, żeby uśrednianie było ważoną średnią
    return w / w.sum()


def _valid_columns(W: int, top_col: int, excl_half: int) -> np.ndarray:
    """
    Zwraca indeksy kolumn z wyłączeniem strefy górnej powieki.

    Górna powieka zasłania fragment tęczówki przy θ ≈ 3π/2 (góra okręgu).
    Kolumny w tej strefie zawierają piksele powieki zamiast tęczówki,
    więc zakłócałyby kodowanie - pomijamy je.

    Args:
        W:         Całkowita liczba kolumn rozwiniętego obrazu.
        top_col:   Kolumna odpowiadająca górze tęczówki (θ = 3π/2).
        excl_half: Połowa szerokości strefy wyłączenia po każdej stronie top_col.
    """
    cols = np.arange(W)
    lo = (top_col - excl_half) % W
    hi = (top_col + excl_half) % W

    # przypadek bez zawijania przez kolumnę 0
    if lo < hi:
        return cols[(cols < lo) | (cols > hi)]

    # strefa wyłączenia przechodzi przez kolumnę 0 - obsługa zawijania
    return cols[(cols > hi) & (cols < lo)]


def encode_iris(
    iris_img: np.ndarray,
    n_bands: int = 8,
    n_samples: int = 128,
    f: float = 1.5,
    top_col: int = 270,
    excl_half: int = 40,
) -> np.ndarray:
    """
    Koduje rozwinięty obraz tęczówki jako binarny iris code metodą Daugmana.

    Algorytm:
    1. Rozwinięty obraz (60x360) dzielimy na n_bands poziomych pasm.
    2. Pomijamy kolumny górnej powieki (top_col +- excl_half), z pozostałych
       wybieramy n_samples równomiernie rozmieszczonych pozycji kątowych.
    3. W każdej pozycji kątowej wiersze pasma zwijamy do jednej wartości
       ważoną średnią gaussowską (środkowe wiersze ważniejsze).
    4. Otrzymany sygnał 1D splatamy z falką Gabora - znak odpowiedzi
       rzeczywistej i urojonej to dwa bity na pozycję na pasmo.
    5. Bity są przeplotowane wg pozycji kątowej: przesunięcie o k pozycji
       kątowych = przesunięcie tablicy bitów o k * n_bands * 2 - umożliwia
       efektywne porównanie odporne na rotację.

    Args:
        iris_img:  Rozwinięty obraz tęczówki w skali szarości (60x360).
        n_bands:   Liczba pasm radialnych - więcej pasm = więcej bitów,
                   ale też większa wrażliwość na błędy geometrii.
        n_samples: Liczba próbek kątowych - więcej = więcej bitów.
        f:         Częstotliwość falki Gabora (zakres użyteczny ≈ [0.6, π]).
        top_col:   Kolumna szczytu tęczówki (θ = 3π/2 → kolumna 270 dla 360px).
        excl_half: Połowa szerokości strefy powieki do pominięcia.

    Returns:
        Tablica uint8 o długości n_bands * n_samples * 2 (domyślnie 2048 bitów).
    """
    H, W = iris_img.shape
    # pracujemy w float64 żeby uniknąć przepełnienia przy splocie
    img = iris_img.astype(np.float64)

    # wybieramy kolumny poza strefą powieki, potem redukujemy do n_samples
    # równomiernie rozmieszczonych - spójne próbkowanie niezależnie od wyłączeń
    valid = _valid_columns(W, top_col, excl_half)
    idx = np.round(np.linspace(0, len(valid) - 1, n_samples)).astype(int)
    cols = valid[idx]

    # jądra gabora obliczamy raz - wspólne dla wszystkich pasm
    gabor_r, gabor_i = _gabor_kernels(f)

    # wysokość pojedynczego pasma; ostatnie pasmo może być o 1 wiersz krótsze
    # jeśli H nie dzieli się przez n_bands bez reszty
    band_h = H // n_bands

    # akumulatory odpowiedzi gabora dla każdego pasma i próbki kątowej
    real_resp = np.zeros((n_bands, n_samples))
    imag_resp = np.zeros((n_bands, n_samples))

    for b in range(n_bands):
        # wiersze należące do pasma b
        r0 = b * band_h
        r1 = min(r0 + band_h, H)

        # wagi gaussowskie do zwinięcia pasm w pojedynczy sygnał 1d
        w = _radial_gauss_weights(r1 - r0)

        # ważona średnia radialnie: (n_samples,) - jeden sygnał na pasmo
        signal = img[r0:r1, :][:, cols].T @ w

        # normalizacja zero-mean, unit-variance: znaki odpowiedzi gabora
        # mają być zależne od lokalnej fazy, nie od bezwzględnej jasności
        signal = (signal - signal.mean()) / (signal.std() + 1e-6)

        # splot z jądrami gabora w trybie "same" - wynik tej samej długości co sygnał
        real_resp[b] = np.convolve(signal, gabor_r, mode="same")
        imag_resp[b] = np.convolve(signal, gabor_i, mode="same")

    # układ bitów: dla każdej z n_samples pozycji kątowych zapisujemy
    # blok n_bands*2 bitów [b0_real, b0_imag, b1_real, …, b7_imag].
    # dzięki temu przesunięcie o 1 próbkę kątową = przesunięcie o block bitów -
    # porównanie z rotacją sprowadza się do np.roll na tablicy bitów
    block = n_bands * 2
    bits = np.empty(n_samples * block, dtype=np.uint8)

    for b in range(n_bands):
        # bit 0/1 zależnie od znaku odpowiedzi: nieujemna → 1, ujemna → 0
        bits[b * 2 :: block] = (real_resp[b] >= 0).astype(np.uint8)
        bits[b * 2 + 1 :: block] = (imag_resp[b] >= 0).astype(np.uint8)

    return bits
