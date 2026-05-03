# Plan projektu: Segmentacja obrazów OCT dla DME

## 1. Analiza i Przygotowanie Danych
- [x] Eksploracja zbioru danych (obrazy i maski w `data_folder/`).
- [x] Podział danych: Train/Val z zachowaniem integralności pacjentów (zapobieganie leakage).
- [x] Preprocessing: Normalizacja SegFormer, zmiana rozmiaru do 512x512, podstawowa augmentacja (HorizontalFlip, Brightness/Contrast).
- [ ] Pogłębiona analiza rozkładu klas: IRF, SRF, HRF (HRF jest zazwyczaj najtrudniejsze do wykrycia).

## 2. Wybór i Implementacja Architektury
- [x] Implementacja modelu **SegFormer (B0)** (wykorzystanie transformerów dla globalnych zależności).
- [ ] Eksperyment z większymi wariantami (np. SegFormer-B2) dla lepszej precyzji.
- [ ] Konfiguracja multi-channel: dodanie `edge_map_images` jako dodatkowego kanału wejściowego.

## 3. Funkcja Straty i Optymalizacja
- [x] Implementacja **Hybrid Loss**: $Loss = 0.5 \cdot CE + 0.5 \cdot DiceLoss$.
- [x] Dobór optymalizatora: AdamW z LR 1e-4.
- [ ] Implementacja **Focal Loss** w celu lepszej walki z przewagą tła (background imbalance).

## 4. Metryki Ewaluacji
- [x] Implementacja skryptu do obliczania ilościowych metryk.
- [x] Baseline Results (SegFormer-B0):
    - mIoU: 0.7260, Mean Dice: 0.8321
    - Class 1 (IRF) IoU: 0.5730 (Priorytet poprawy)
    - Class 2 (SRF) IoU: 0.6810
    - Class 3 (HRF) IoU: 0.6585
- [ ] HD95 (Hausdorff Distance) dla oceny precyzji obrysów.

## 5. Trenowanie i Walidacja
- [x] Uruchomienie pierwszej pętli uczenia (10 epok).
- [ ] Wdrożenie **Focal Loss** lub **Weighted Cross Entropy** aby podbić wyniki Class 1.
- [ ] Wydłużenie treningu do 30-50 epok z Learning Rate Schedulerem.


## 6. Ewaluacja i Interpretacja Wyników
- [x] Wizualizacja predykcji (`eval_results.png`) - model poprawnie lokalizuje główne skupiska płynu.
- [ ] Analiza błędów: sprawdzanie małych struktur (HRF) oraz granic (edges) między klasami.
- [ ] Porównanie wyników z obrazami odszumionymi (`denoised_images`).

## 7. Dokumentacja i Raport końcowy
- [ ] Zestawienie tabelaryczne metryk dla różnych konfiguracji modelu.
- [ ] Wnioski dotyczące stabilności modelu na danych z różnych urządzeń (jeśli dostępne).
