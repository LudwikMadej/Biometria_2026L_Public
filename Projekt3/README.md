# Fingerprint Thinning & Minutiae Detection

Comparison of three thinning algorithms on fingerprint images, followed by
minutiae detection (ridge endings and bifurcations) via the Crossing Number
method.

## Authors

- Maciej Andrzejewski
- Ludwik Madej

## What's inside

- **Morphological skeletonization** (Lantuejoul, reused from Projekt1).
- **KMM** (Saeed, Rybnik, Tabędzki) - labels 1/2/3/4 + 256-entry lookup.
- **K3M** (Saeed, Tabędzki, Rybnik, Adamski) - 6-phase extension of KMM
  with a final pass reducing the skeleton to one-pixel width.
- Morphological closing to reconnect broken ridges before thinning.
- Crossing Number detector: CN=1 -> ending, CN=3 -> bifurcation.

Morphological operations (`closing`, `skeletonize`) are not reimplemented -
the `backend` package from `Projekt1` is added to `sys.path` automatically.

## Getting started

Requires Python 3.10 and both `Projekt1` and `Projekt3` present in the repository.

```bash
cd Projekt3
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux / macOS
pip install -r requirements.txt
jupyter notebook notebooks/scienianie_detekcja_minuncji.ipynb
```

In the first cell of the notebook, set `FINGERPRINT_PATH` to a file inside
`data/` and adjust `FOREGROUND` (`"dark"`/`"light"`) if needed.

## Tech stack

| Library    | Purpose                                     |
| ---------- | ------------------------------------------- |
| NumPy      | Neighbourhood weights, lookup tables, masks |
| Matplotlib | Skeleton and minutiae visualization         |
| Pillow     | Image loading (via `backend` from Projekt1) |
| Jupyter    | Comparison notebook                         |

KMM and K3M implemented from scratch based on `docs/KMM.pdf` and `docs/K3M.pdf`.
