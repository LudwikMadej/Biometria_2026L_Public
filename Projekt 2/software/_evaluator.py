import numpy as np
import cv2


class Evaluator:
    """
    Ocenia jakość wykrytej geometrii oka (źrenica + tęczówka) na podstawie
    sześciu niezależnych miar. Używana w find_optimal_eye_geometry do wyboru
    najlepszej konfiguracji parametrów z grid searcha.
    """

    def __init__(self, image):
        # konwertujemy na float32 żeby uniknąć błędów zaokrąglenia przy operacjach
        # na wartościach pikseli (gradient, interpolacja)
        self.img = image.astype(np.float32)
        self.h, self.w = image.shape

        # gradienty sobela obliczamy raz w konstruktorze - wszystkie metody oceny
        # z nich korzystają, więc wyliczenie ich z góry oszczędza czas
        self.grad_x = cv2.Sobel(self.img, cv2.CV_32F, 1, 0, ksize=3)
        self.grad_y = cv2.Sobel(self.img, cv2.CV_32F, 0, 1, ksize=3)
        # 1e-6 zapobiega dzieleniu przez zero przy normalizacji gradientu
        self.grad_mag = np.sqrt(self.grad_x**2 + self.grad_y**2) + 1e-6

    def _bilinear(self, x, y):
        # próbkowanie subpikselowe: zamiast zaokrąglać (x,y) do sąsiadującego
        # piksela, ważymy czterech sąsiadów proporcjonalnie do odległości -
        # daje ciągłą wartość nawet dla ułamkowych współrzędnych
        x0, y0 = int(x), int(y)
        x1 = min(x0 + 1, self.w - 1)
        y1 = min(y0 + 1, self.h - 1)
        dx, dy = x - x0, y - y0

        val = (
            self.img[y0, x0] * (1 - dx) * (1 - dy)
            + self.img[y0, x1] * dx * (1 - dy)
            + self.img[y1, x0] * (1 - dx) * dy
            + self.img[y1, x1] * dx * dy
        )
        return val

    def _circle_mask(self, x, y, r):
        # maska koła - używana do wyodrębnienia pikseli źrenicy
        Y, X = np.ogrid[: self.h, : self.w]
        return (X - x) ** 2 + (Y - y) ** 2 <= r**2

    def _annulus_mask(self, x, y, r1, r2):
        # maska pierścienia między promieniami r1 i r2 - wyodrębnia tęczówkę
        # (obszar między granicą źrenicy a granicą tęczówki)
        Y, X = np.ogrid[: self.h, : self.w]
        dist2 = (X - x) ** 2 + (Y - y) ** 2
        return (r1**2 <= dist2) & (dist2 <= r2**2)

    def daugman_score(self, x, y, r, samples=120):
        """
        Mierzy jak ostry jest gradient wzdłuż okręgu o promieniu r.
        Wysoki wynik → okrąg pokrywa się z prawdziwą krawędzią (źrenicy lub tęczówki).
        Idea pochodzi bezpośrednio z oryginalnego operatora Daugmana (1993).
        """
        angles = np.linspace(0, 2 * np.pi, samples)
        grads = []

        for a in angles:
            px = x + r * np.cos(a)
            py = y + r * np.sin(a)

            # pomijamy punkty poza obrazem (margin 1px na interpolację)
            if not (1 <= px < self.w - 1 and 1 <= py < self.h - 1):
                continue

            gx = self.grad_x[int(py), int(px)]
            gy = self.grad_y[int(py), int(px)]

            # gradient radialny: rzutujemy wektor gradientu na kierunek radialny;
            # na prawdziwej krawędzi gradient jest prostopadły do okręgu,
            # więc rzut radialny będzie duży
            rx, ry = np.cos(a), np.sin(a)
            radial_grad = gx * rx + gy * ry
            grads.append(radial_grad)

        if len(grads) < 10:
            return 0

        grads = np.array(grads)
        # normalizacja przez odchylenie std (jak w daugmanie): wynik jest miarą
        # spójności gradientu wzdłuż okręgu - duże mean/std → wyraźna krawędź
        score = np.mean(grads) / (np.std(grads) + 1e-6)
        return max(0, score)

    def robust_contrast(self, pupil_mask, iris_mask):
        """
        Mierzy różnicę jasności między źrenicą a tęczówką.
        Używamy mediany zamiast średniej - jest odporna na jasne odblaski
        wewnątrz źrenicy, które zawyżałyby średnią.
        """
        p = self.img[pupil_mask]
        i = self.img[iris_mask]

        if len(p) < 10 or len(i) < 10:
            return 0

        mp = np.median(p)
        mi = np.median(i)

        # znormalizowany kontrast ∈ [0, 1]: (iris - pupil) / iris
        # tęczówka powinna być jaśniejsza niż źrenica → wynik dodatni
        return np.clip((mi - mp) / (mi + 1e-6), 0, 1)

    def edge_alignment(self, x, y, r, samples=120):
        """
        Mierzy jak bardzo gradient obrazu jest wyrównany z kierunkiem radialnym
        na okręgu. Idealna krawędź ma gradient prostopadły do okręgu (cos=1).
        """
        angles = np.linspace(0, 2 * np.pi, samples)
        scores = []

        for a in angles:
            px = x + r * np.cos(a)
            py = y + r * np.sin(a)

            if not (1 <= px < self.w - 1 and 1 <= py < self.h - 1):
                continue

            gx = self.grad_x[int(py), int(px)]
            gy = self.grad_y[int(py), int(px)]

            # normalizujemy gradient do wektora jednostkowego
            grad_norm = np.sqrt(gx**2 + gy**2) + 1e-6
            gx /= grad_norm
            gy /= grad_norm

            rx, ry = np.cos(a), np.sin(a)
            # cosinus kąta między gradientem a kierunkiem radialnym;
            # abs() bo gradient może wskazywać do wewnątrz lub na zewnątrz
            alignment = abs(gx * rx + gy * ry)
            scores.append(alignment)

        if not scores:
            return 0
        return np.mean(scores)

    def geometry_score(self, xp, yp, rp, xi, yi, ri):
        """
        Sprawdza czy źrenica leży wewnątrz tęczówki (warunek geometryczny).
        Im mniejsze przesunięcie środków względem różnicy promieni,
        tym wyższy wynik - funkcja eksponencjalna szybko karze za przesunięcia.
        """
        dist = np.sqrt((xp - xi) ** 2 + (yp - yi) ** 2)

        # jeśli źrenica wychodzi poza tęczówkę → geometria niemożliwa
        if dist + rp > ri:
            return 0

        # eksponencjalne tłumienie: małe przesunięcie → wynik bliski 1
        return np.exp(-dist / (ri - rp + 1e-6))

    def leakage_score(self, xp, yp, rp):
        """
        Sprawdza czy bezpośrednio za granicą źrenicy (pierścień r+4px)
        nie ma ciemnych pikseli - jeśli są, to granica źrenicy jest
        prawdopodobnie za mała i "wycieka" w tęczówkę.
        """
        # pierścień o szerokości 4px wokół źrenicy
        ring = self._annulus_mask(xp, yp, rp, rp + 4)
        vals = self.img[ring]

        if len(vals) == 0:
            return 1

        # piksele ciemniejsze niż 20. percentyl całego obrazu traktujemy jako "ciemne"
        threshold = np.percentile(self.img, 20)
        leakage = np.mean(vals < threshold)

        # im mniej ciemnych pikseli za granicą, tym lepiej
        return 1 - leakage

    def radial_consistency(self, x, y, r_inner, r_outer, samples=50):
        """
        Mierzy zmienność jasności wzdłuż okręgów o różnych promieniach
        wewnątrz tęczówki. Wysoka spójność (niskie std) sugeruje że okrąg
        tęczówki dobrze opisuje granicę - obszar wewnątrz jest jednorodny.
        """
        radii = np.linspace(r_inner, r_outer, 10)
        angles = np.linspace(0, 2 * np.pi, samples)

        profiles = []
        for r in radii:
            vals = [
                self._bilinear(x + r * np.cos(a), y + r * np.sin(a))
                for a in angles
                if 0 <= x + r * np.cos(a) < self.w and 0 <= y + r * np.sin(a) < self.h
            ]
            if vals:
                profiles.append(np.std(vals))

        if not profiles:
            return 0

        # odwracamy: im mniejsze odchylenie → im wyższy wynik
        return 1 / (np.mean(profiles) + 1e-6)

    def evaluate(self, p_params, i_params):
        """
        Łączy wszystkie miary w jeden wynik ∈ [0, 1] za pomocą ważonej sumy.

        Wagi odzwierciedlają znaczenie każdej miary:
        - Daugman na źrenicy (0.25): najważniejszy - ostra, wyraźna granica źrenicy
        - Kontrast (0.15): źrenica musi być ciemniejsza od tęczówki
        - Wyrównanie krawędzi źrenicy (0.15): gradient musi być radialny
        - Daugman na tęczówce (0.15): ostra granica zewnętrzna
        - Geometria (0.10): źrenica wewnątrz tęczówki
        - Wyrównanie krawędzi tęczówki (0.10): gradient zewnętrzny
        - Wyciek (0.05): brak ciemnych pikseli za granicą źrenicy
        - Spójność (0.05): jednorodność tęczówki
        """
        xp, yp, rp = p_params
        xi, yi, ri = i_params

        pupil_mask = self._circle_mask(xp, yp, rp)
        iris_mask = self._annulus_mask(xi, yi, rp, ri)

        g_p = self.daugman_score(xp, yp, rp)
        g_i = self.daugman_score(xi, yi, ri)
        align_p = self.edge_alignment(xp, yp, rp)
        align_i = self.edge_alignment(xi, yi, ri)
        contrast = self.robust_contrast(pupil_mask, iris_mask)
        geo = self.geometry_score(xp, yp, rp, xi, yi, ri)
        leak = self.leakage_score(xp, yp, rp)
        radial = self.radial_consistency(xi, yi, rp, ri)

        # tanh normalizuje wynik daugmana do [0, 1] -
        # bez tego bardzo duże wartości dominowałyby ważoną sumę
        g_p = np.tanh(g_p)
        g_i = np.tanh(g_i)

        final = (
            0.25 * g_p
            + 0.15 * g_i
            + 0.15 * align_p
            + 0.10 * align_i
            + 0.15 * contrast
            + 0.10 * geo
            + 0.05 * leak
            + 0.05 * radial
        )

        return float(np.clip(final, 0, 1))
