# Rozpoznawanie tożsamości na podstawie tęczówki oka

System biometryczny realizujący pipeline rozpoznawania tęczówki: od wczytania obrazu, przez segmentację, aż po weryfikację tożsamości z oceną skuteczności (EER, ROC).

Dane wejściowe: baza **MMU Iris Database** (45 osób, lewe i prawe oko, 5 zdjęć na oko).

---

## Jak działa pipeline

1. **Detekcja źrenicy** - progowanie adaptacyjne + operacje morfologiczne + wybór konturu o największej kolistości; opcjonalne doprecyzowanie przez usunięcie refleksów i skan jasności brzegu.
2. **Detekcja tęczówki** - wyznaczenie zewnętrznej granicy tęczówki względem wykrytej źrenicy przy użyciu metody MNK.
3. **Rozwinięcie pierścienia tęczówki** - model (Daugmana): pierścień biegunowy mapowany na prostokąt.
4. **Kodowanie** - filtr Gabora, wynik binaryzowany do kodu tęczówki.
5. **Porównanie** - znormalizowany dystans Hamminga z przesunięciami rotacyjnymi; próg wyznaczany przez EER.

---

## Schemat repozytorium

```
Projekt2/
├── data/
│   └── MMU-Iris-Database/        # baza danych (45 podkatalogów, lewe/prawe oko)
│       └── <id>/left|right/*.bmp
│
├── notebooks/                    # notebooki Jupyter (w kolejności)
│   ├── 0_loading_image.ipynb     # wczytywanie obrazów BMP
│   ├── 1_pupil_detection.ipynb   # detekcja źrenicy
│   ├── 2_iris_detection.ipynb    # detekcja tęczówki
│   ├── 3_eye_geometry_detection_test.ipynb
│   ├── 4_checkpoint.ipynb
│   ├── 5_iris_recognition.ipynb  # pełny pipeline + EER/ROC
│   └── 6_manual_test_with_vis.ipynb
│
├── software/                     # implementacja pakietu
│   ├── __init__.py               
│   ├── _load_grayscale_image.py
│   ├── _get_eye_geometry.py
│   ├── _unroll_iris.py
│   ├── _encode_iris.py
│   ├── _compare_iris_codes.py
│   ├── _find_optimal_eye_geometry.py
│   ├── _evaluator.py
│   ├── pupil_detection/          # moduły detekcji źrenicy
│   ├── iris_detection/           # moduły detekcji tęczówki
│   └── visualization/            # narzędzia wizualizacji
│
├── docs/                         # dokumenty
│   ├── BIO_2026_Projekt_2.pdf
│   ├── Kod_teczowki_uwagi.pdf
│   └── book_fragments/
│
├── Projekt 2 - Dokumentacja.pdf  # dokumentacja
└── requirements.txt
```

---

## Uruchomienie

### 1. Klonowanie repozytorium

```bash
git clone <repository-url>
cd Projekt2
```

### 2. Utworzenie środowiska wirtualnego

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Instalacja zależności

```bash
pip install -r requirements.txt
```

### 4. Uruchomienie Jupyter Notebook

```bash
jupyter notebook
```

### 5. Uruchomienie wybranych notebooków
