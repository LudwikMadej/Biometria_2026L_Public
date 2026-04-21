import cv2
import os


def load_grayscale_image(file_path):
    """
    Wczytuje obraz tęczówki z pliku .bmp.

    Args:
        file_path (str): Pełna ścieżka do pliku .bmp.

    Returns:
        numpy.ndarray: Obraz w skali szarości (uint8) lub None w razie błędu.
    """

    # sprawdzamy istnienie pliku zanim zlecimy odczyt - cv2.imread nie rzuca
    # wyjątku, tylko cicho zwraca none, co trudniej debugować
    if not os.path.exists(file_path):
        print(f"Błąd: plik nie istnieje: {file_path}")
        return None

    # cv2.imread_grayscale konwertuje obraz do jednokanałowej skali szarości
    # już przy wczytaniu - unikamy zbędnej kopii w pamięci w porównaniu
    # do wczytania rgb i późniejszej konwersji
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Błąd: nie można zdekodować obrazu.")
        return None

    return image
