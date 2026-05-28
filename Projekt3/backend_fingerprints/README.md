# backend_fingerprints

Operacje morfologiczne (closing, erozja, dylatacja, szkieletyzacja Lantuejoula)
są importowane z `backend` z **Projektu 1**.

## Zawartość

| Plik                 | Co robi                                                                                   |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `_utils.py`          | Konwersje: obraz do/z maska 0/1 oraz liczenie wagi sąsiedztwa                             |
| `_kmm.py`            | Algorytm **KMM**. Iteracyjne ścienianie z etykietami 1/2/3/4                              |
| `_k3m.py`            | Algorytm **K3M**. Ulepszona wersja algorytmu KMM                                          |
| `_minutiae.py`       | Detekcja minucji metodą **Crossing Number**: CN=1-> zakończenie linii, CN=3 -> bifurkacja |
| `_connect_ridges.py` | Łączenie poprzerywanych linii papilarnych przy pomocy domknięcia morfologicznego          |
