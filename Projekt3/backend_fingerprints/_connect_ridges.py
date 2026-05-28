import sys
from pathlib import Path
import numpy as np

# dodanie do path folderu z operacjami morfologicznymi z Projektu 1
_PROJEKT1_PATH = Path(__file__).resolve().parent.parent.parent / "Projekt1"
if _PROJEKT1_PATH.is_dir() and str(_PROJEKT1_PATH) not in sys.path:
    sys.path.insert(0, str(_PROJEKT1_PATH))

from backend import closing

from ._utils import to_foreground_mask, from_foreground_mask


def connect_ridges(
    image: np.ndarray,
    kernel: np.ndarray = None,
    foreground: str = "dark",
) -> np.ndarray:
    """
    Poprawa połączeń między poprzerywanymi liniami papilarnymi przy pomocy
    domknięcia morfologicznego wykonanego na masce pierwszego planu.

    Implementacja korzysta z funkcji `closing` z pakietu `backend` z Projektu 1.

    Args:
        image (np.ndarray): obraz binarny 2D.
        kernel (np.ndarray, optional): element strukturyzujący; domyślnie krzyż 3x3.
        foreground (str): "dark" (domyślnie) lub "light".

    Returns:
        np.ndarray: obraz binarny 0/255 (uint8) po zamknięciu.
    """
    # domyślny kernel
    if kernel is None:
        kernel = np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            dtype=bool,
        )
    # closing z Projektu 1 chce zbinaryzowanego zdjęcia
    kernel = kernel.astype(bool)

    # sprowadzamy obraz do maski 0/1, a następnie do konwencji 0/255,
    # którą przyjmuje implementacja z Projektu 1
    mask = to_foreground_mask(image, foreground)
    mask_255 = (mask * 255).astype(np.uint8)

    # wykonujemy domknięcie morfologiczne (dylatacja -> erozja)
    closed_255 = closing(mask_255, kernel)

    # wracamy do maski 0/1, a potem do oryginalnej konwencji kolorów
    closed_mask = (closed_255 > 0).astype(np.uint8)
    return from_foreground_mask(closed_mask, foreground)
