import numpy as np
import cv2

class Evaluator:
    def __init__(self, image):
        self.img = image.astype(np.float32)
        self.h, self.w = image.shape

        # Precompute gradienty (Sobel)
        self.grad_x = cv2.Sobel(self.img, cv2.CV_32F, 1, 0, ksize=3)
        self.grad_y = cv2.Sobel(self.img, cv2.CV_32F, 0, 1, ksize=3)
        self.grad_mag = np.sqrt(self.grad_x**2 + self.grad_y**2) + 1e-6

    # -------------------------
    # INTERPOLACJA (ważne!)
    # -------------------------
    def _bilinear(self, x, y):
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, self.w - 1), min(y0 + 1, self.h - 1)

        dx, dy = x - x0, y - y0

        val = (
            self.img[y0, x0] * (1 - dx) * (1 - dy) +
            self.img[y0, x1] * dx * (1 - dy) +
            self.img[y1, x0] * (1 - dx) * dy +
            self.img[y1, x1] * dx * dy
        )
        return val

    # -------------------------
    # MASKI
    # -------------------------
    def _circle_mask(self, x, y, r):
        Y, X = np.ogrid[:self.h, :self.w]
        return (X - x)**2 + (Y - y)**2 <= r**2

    def _annulus_mask(self, x, y, r1, r2):
        Y, X = np.ogrid[:self.h, :self.w]
        dist2 = (X - x)**2 + (Y - y)**2
        return (r1**2 <= dist2) & (dist2 <= r2**2)

    # -------------------------
    # 1. DAUGMAN-LIKE SCORE
    # -------------------------
    def daugman_score(self, x, y, r, samples=120):
        angles = np.linspace(0, 2*np.pi, samples)

        vals = []
        grads = []

        for a in angles:
            px = x + r * np.cos(a)
            py = y + r * np.sin(a)

            if 1 <= px < self.w-1 and 1 <= py < self.h-1:
                val = self._bilinear(px, py)
                gx = self.grad_x[int(py), int(px)]
                gy = self.grad_y[int(py), int(px)]

                # kierunek radialny
                rx, ry = np.cos(a), np.sin(a)
                radial_grad = gx * rx + gy * ry

                vals.append(val)
                grads.append(radial_grad)

        if len(grads) < 10:
            return 0

        grads = np.array(grads)

        # normalizacja energii (jak w Daugmanie)
        score = np.mean(grads) / (np.std(grads) + 1e-6)

        return max(0, score)

    # -------------------------
    # 2. ROBUST CONTRAST
    # -------------------------
    def robust_contrast(self, pupil_mask, iris_mask):
        p = self.img[pupil_mask]
        i = self.img[iris_mask]

        if len(p) < 10 or len(i) < 10:
            return 0

        # median zamiast mean (odporność na odblaski)
        mp = np.median(p)
        mi = np.median(i)

        return np.clip((mi - mp) / (mi + 1e-6), 0, 1)

    # -------------------------
    # 3. EDGE ALIGNMENT
    # -------------------------
    def edge_alignment(self, x, y, r, samples=120):
        angles = np.linspace(0, 2*np.pi, samples)

        scores = []

        for a in angles:
            px = x + r * np.cos(a)
            py = y + r * np.sin(a)

            if 1 <= px < self.w-1 and 1 <= py < self.h-1:
                gx = self.grad_x[int(py), int(px)]
                gy = self.grad_y[int(py), int(px)]

                grad_norm = np.sqrt(gx**2 + gy**2) + 1e-6

                gx /= grad_norm
                gy /= grad_norm

                # radial vector
                rx, ry = np.cos(a), np.sin(a)

                # cos similarity
                alignment = abs(gx * rx + gy * ry)
                scores.append(alignment)

        if len(scores) == 0:
            return 0

        return np.mean(scores)

    # -------------------------
    # 4. GEOMETRY (IoU-like)
    # -------------------------
    def geometry_score(self, xp, yp, rp, xi, yi, ri):
        dist = np.sqrt((xp-xi)**2 + (yp-yi)**2)

        if dist + rp > ri:
            return 0

        return np.exp(-dist / (ri - rp + 1e-6))

    # -------------------------
    # 5. LEAKAGE (ulepszone)
    # -------------------------
    def leakage_score(self, xp, yp, rp):
        ring = self._annulus_mask(xp, yp, rp, rp+4)
        vals = self.img[ring]

        if len(vals) == 0:
            return 1

        threshold = np.percentile(self.img, 20)
        leakage = np.mean(vals < threshold)

        return 1 - leakage

    # -------------------------
    # 6. RADIAL CONSISTENCY
    # -------------------------
    def radial_consistency(self, x, y, r_inner, r_outer, samples=50):
        radii = np.linspace(r_inner, r_outer, 10)
        angles = np.linspace(0, 2*np.pi, samples)

        profiles = []

        for r in radii:
            vals = []
            for a in angles:
                px = x + r*np.cos(a)
                py = y + r*np.sin(a)

                if 0 <= px < self.w and 0 <= py < self.h:
                    vals.append(self._bilinear(px, py))

            if len(vals) > 0:
                profiles.append(np.std(vals))

        if len(profiles) == 0:
            return 0

        return 1 / (np.mean(profiles) + 1e-6)

    # -------------------------
    # FINAL SCORE
    # -------------------------
    def evaluate(self, p_params, i_params):
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

        # NORMALIZACJA
        g_p = np.tanh(g_p)
        g_i = np.tanh(g_i)

        final = (
            0.25 * g_p +
            0.15 * g_i +
            0.15 * align_p +
            0.10 * align_i +
            0.15 * contrast +
            0.10 * geo +
            0.05 * leak +
            0.05 * radial
        )

        return float(np.clip(final, 0, 1))