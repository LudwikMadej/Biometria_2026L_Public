import cv2
import numpy as np
import matplotlib.pyplot as plt


def refine_pupil_roi(
    img: np.ndarray,
    x_center: float,
    y_center: float,
    radius: float,
    roi_scale: float = 2.0,
    reflection_threshold: int = 200,
    blur_kernel: int = 5,
    show_plots: bool = False,
) -> tuple[float, float, float]:
    """
    Uściśla geometrię źrenicy na podstawie lokalnego ROI (region of interest).

    Dlaczego uściślanie: detect_pupil_geometry podaje wynik na poziomie całego obrazu,
    co ogranicza precyzję. Tutaj wycinamy region wokół znalezionej źrenicy i
    szukamy dokładniejszej granicy przez skanowanie jasności w 3 kierunkach.

    Args:
        roi_scale:            Jak duży ROI wycinamy: roi = roi_scale * radius w każdą stronę.
        reflection_threshold: Piksele jaśniejsze od tego progu to odblaski - inpaintujemy je.
        blur_kernel:          Rozmiar kernela gaussowskiego wygładzającego obraz przed skanowaniem.
    """

    # roi musi być wystarczająco duże żeby obejmować całą źrenicę z marginesem
    margin = int(roi_scale * radius)

    y1 = max(0, int(y_center - margin))
    y2 = min(img.shape[0], int(y_center + margin))
    x1 = max(0, int(x_center - margin))
    x2 = min(img.shape[1], int(x_center + margin))

    roi = img[y1:y2, x1:x2].copy()

    # usunięcie odblasków
    # jasne plamki wewnątrz źrenicy (odblaski rogówki) zaburzają skan jasności
    # - inpaintujemy je, żeby uzyskać jednolity ciemny dysk
    _, reflection_mask = cv2.threshold(
        roi, reflection_threshold, 255, cv2.THRESH_BINARY
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # rozszerzamy maskę o 1px na każdą stronę
    reflection_mask = cv2.dilate(reflection_mask, kernel)
    roi_clean = cv2.inpaint(roi, reflection_mask, 3, cv2.INPAINT_TELEA)

    # gaussowskie wygładzenie redukuje szum przed skanowaniem jasności -
    # bez tego pojedyncze jasne piksele mogłyby przedwcześnie zatrzymać skan
    roi_blur = cv2.GaussianBlur(roi_clean, (blur_kernel, blur_kernel), 0)

    # współrzędne środka źrenicy w układzie lokalnym roi
    cx = int(x_center - x1)
    cy = int(y_center - y1)

    # próg wyjścia: jasność centrum + stały margines, max 90 -
    # szukamy miejsca gdzie jasność wyraźnie rośnie (granica źrenica→tęczówka)
    base = roi_blur[cy, cx]
    exit_threshold = min(base + 35, 90)

    def scan(arr):
        # skanujemy wartości pikseli wzdłuż linii;
        # pierwsza wartość przekraczająca próg = krawędź źrenicy
        for i, v in enumerate(arr):
            if v > exit_threshold:
                return i
        return len(arr)

    # skanujemy w 3 kierunkach: prawo, lewo, dół -
    # na tej podstawie szacujemy środek i promień przez symetrię poziomą
    edge_r = scan(roi_blur[cy, cx:])  # krawędź prawa
    edge_l = scan(
        np.flip(roi_blur[cy, :cx])
    )  # krawędź lewa (skanujemy od środka na lewo)
    edge_d = scan(roi_blur[cy:, cx])  # krawędź dolna

    # środek x: połowa między lewą a prawą krawędzią
    left = cx - edge_l
    right = cx + edge_r
    new_x = (left + right) / 2.0
    new_r = (right - left) / 2.0

    # środek y: dół - promień (zakładamy symetrię pionową)
    bottom = cy + edge_d
    new_y = bottom - new_r

    # przeliczamy z układu lokalnego roi do układu globalnego obrazu
    final_x = x1 + new_x
    final_y = y1 + new_y

    if show_plots:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("ROI (oryginał)")
        plt.imshow(roi, cmap="gray")

        plt.subplot(1, 3, 2)
        plt.title("Po inpaincie i wygładzeniu")
        plt.imshow(roi_blur, cmap="gray")

        plt.subplot(1, 3, 3)
        plt.title("Uściślona geometria")
        plt.imshow(roi, cmap="gray")
        circle = plt.Circle((new_x, new_y), new_r, fill=False, color="lime")
        plt.gca().add_patch(circle)
        plt.scatter([new_x], [new_y], color="red", s=20)
        plt.show()

    return float(final_x), float(final_y), float(new_r)
