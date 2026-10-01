# Plan Dokumentacji i Spisu Treści Pracy Inżynierskiej

## Część I: Plan dokumentacji kodu (Python / PyTorch / MONAI)

### 1. Struktura i konwencja docstringów
Zastosujemy standard **Google Style Docstrings**.
Każda funkcja/klasa będzie zawierać:
* Krótki opis działania.
* `Args:` z typami danych i kształtami tensorów.
* `Returns:` opis i kształt zwracanych wartości.
* (Opcjonalnie) `References:` odniesienia do literatury.

### 2. Główne pliki do udokumentowania
* `dataset.py`: Klasy ładowania danych, transformacje MONAI.
* `config.py`: Hiperparametry, ścieżki, stałe.
* `main.py` / `train.py` / `eval.py`: Pętle treningowe, logowanie metryk.
* `utils.py`: Funkcje pomocnicze, liczenie metryk.
* `attention_visualizer.py`: Generowanie i rzutowanie map atencji.

### 3. Architektura dokumentacji README
* `README.md` (główny): Tytuł, wymagania, instrukcja przygotowania danych, instrukcje uruchamiania, odtwarzalność wyników.

---

## Część II: Uszczegółowiony spis treści pracy inżynierskiej (LaTeX)

**Rozdział 1. Wstęp**
1.1. Motywacja i znaczenie kliniczne segmentacji obrazów OCT
1.2. Problem badawczy
1.3. Cel i zakres pracy
1.4. Struktura pracy

**Rozdział 2. Podstawy teoretyczne**
2.1. Optyczna Koherentna Tomografia (OCT) siatkówki
2.2. Patologie płynowe w schorzeniach siatkówki oka
2.3. Segmentacja semantyczna w obrazowaniu medycznym
2.4. Konwolucyjne Sieci Neuronowe (CNN) w analizie obrazów medycznych
2.5. Ewolucja architektur: od CNN do modeli opartych o mechanizm uwagi (Transformers)

**Rozdział 3. Zbiór danych i wstępne przetwarzanie**
3.1. Charakterystyka zbioru danych RETOUCH
3.2. Wyzwania związane z danymi z różnych urządzeń skanujących
3.3. Pipeline przetwarzania obrazów (preprocessing i augmentacja z użyciem biblioteki MONAI)

**Rozdział 4. Metodologia**
4.1. Architektura bazowa (Baseline): nnUNet
4.2. Zastosowane modele transformerowe
    4.2.1. SegFormer
    4.2.2. SwinUNETR
4.3. Implementacja i środowisko eksperymentalne
4.4. Strategia uczenia modelu (fine-tuning)
4.5. Funkcje kosztu i metryki ewaluacyjne

**Rozdział 5. Wyniki eksperymentów**
5.1. Analiza ilościowa
    5.1.1. Porównanie skuteczności: CNN vs Transformery
    5.1.2. Wyniki dla poszczególnych klas płynów
5.2. Analiza wizualna i mapy atencji (Attention Maps)
    5.2.1. Wizualizacja przewidywań segmentacji
    5.2.2. Interpretacja mechanizmu uwagi w SwinUNETR/SegFormer
5.3. Zapotrzebowanie na zasoby obliczeniowe i czas wnioskowania

**Rozdział 6. Analiza jakościowa i dyskusja**
6.1. Przypadki, w których architektury transformerowe przewyższają CNN
6.2. Analiza błędów (failure cases) i ograniczenia Transformerów
6.3. Ewaluacja wyników z perspektywy użyteczności klinicznej

**Rozdział 7. Podsumowanie i wnioski**
7.1. Realizacja celu pracy
7.2. Ograniczenia przeprowadzonych badań
7.3. Potencjalne kierunki dalszych prac rozwojowych
