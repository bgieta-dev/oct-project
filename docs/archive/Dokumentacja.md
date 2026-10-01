# Automatic OCT Retina Segmentation Pipeline

Profesjonalny system end-to-end do automatycznej segmentacji struktur patologicznych (IRF, SRF, PED) w obrazowaniu OCT (Optyczna Koherentna Tomografia), oparty na architekturze Transformerów wizyjnych oraz hybrydowym wnioskowaniu eksperckim.

## 1. Przegląd Projektu (High-Level Overview)
Projekt rozwiązuje krytyczny problem w diagnostyce okulistycznej: czasochłonność i subiektywizm ręcznej analizy skanów OCT. System automatycznie identyfikuje i segmentuje trzy kluczowe typy zmian płynowych:
*   **IRF** (Intraretinal Fluid) – płyn wewnątrzsiatkówkowy.
*   **SRF** (Subretinal Fluid) – płyn podsiatkówkowy.
*   **PED** (Pigment Epithelium Detachment) – odwarstwienie nabłonka barwnikowego.

Zastosowanie zaawansowanych modeli **SegFormer** (Semantic Segmentation Transformers) pozwala na osiągnięcie wysokiej precyzji przy zachowaniu interpretowalności wyników (Attention Maps), co jest kluczowe w systemach wsparcia decyzji medycznych (Clinical Decision Support Systems).

## 2. Stack Technologiczny i Architektura
System został zaprojektowany z myślą o modularności i wydajności obliczeniowej:

| Komponent | Technologia | Uzasadnienie Architektoniczne |
| :--- | :--- | :--- |
| **Core Engine** | Python 3.10+ / PyTorch | Standard branżowy zapewniający elastyczność w budowie niestandardowych warstw i funkcji straty. |
| **Model Backbone** | SegFormer (MiT-B2 & B0) | Wybór Transformerów zamiast klasycznych UNetów ze względu na globalne pole widzenia (Global Receptive Field) przy zachowaniu wydajności. |
| **Data Processing** | OpenCV / PIL / NumPy | Wydajna manipulacja macierzami obrazów i implementacja klinicznej normalizacji (percentile clipping). |
| **Experiment Tracking** | Logging / Discord Hooks | Automatyczne monitorowanie procesów uczenia w czasie rzeczywistym na urządzeniach mobilnych. |
| **Inference Engine** | Hybrid Ensemble | Połączenie modelu wieloklasowego (B2) z binarnym modelem eksperckim (B0) w celu eliminacji False Negatives dla IRF. |

## 3. Kluczowe Funkcjonalności (Core Features)

### A. Potok Danych (dataset.py & utils.py)
*   **Patient-Aware Stratified Split:** Mechanizm w `utils.py` gwarantuje, że skany tego samego pacjenta nigdy nie trafiają jednocześnie do zbioru treningowego i testowego. Stratyfikacja odbywa się na poziomie producenta urządzenia (Cirrus, Spectralis, Topcon), co zapewnia odporność modelu na różnice w charakterystyce obrazu (Domain Generalization).
*   **Kontekst Wolumetryczny (2.5D):** W `dataset.py` zaimplementowano logikę ładowania sąsiednich plastrów ($t-1, t, t+1$). Model otrzymuje pełniejszy obraz anatomiczny, co pozwala odróżnić artefakty od rzeczywistych patologii o charakterze ciągłym.
*   **Kliniczna Normalizacja:** Zamiast standardowej normalizacji globalnej, stosujemy clipping 1-99 percentyla, co eliminuje wpływ szumów impulsowych typowych dla aparatów OCT.

### B. Zaawansowane Funkcje Straty (train.py)
W celu poprawy segmentacji małych i nieregularnych obiektów, system wykorzystuje:
*   **Focal Tversky Loss:** Optymalizacja pod kątem wysokiego *Recall* (parametr $\beta=0.9$), kluczowego w medycynie.
*   **Boundary Loss:** Wykorzystuje transformatę dystansową (SDF), aby "wymusić" dopasowanie konturów modelu do rzeczywistych granic anatomicznych, co drastycznie poprawia metrykę HD95.
*   **Dynamic Class Weights:** Automatyczne przeliczanie wag klas na podstawie aktualnego rozkładu w zbiorze treningowym (obsługa *class imbalance*).

### C. Metryki Kliniczne i Zaawansowana Ewaluacja (eval.py)
Poza standardowym IoU/Dice, system implementuje:
*   **HD95 (Hausdorff Distance 95%):** Mierzy błąd konturu w pikselach, co jest kluczowe dla oceny precyzji brzegowej zmian.
*   **Morphological Sharpening (PED):** Ponieważ upsampling w Transformerach wygładza krawędzie, zastosowano filtr ostrzący (Laplacian-based) dla klasy PED, aby przywrócić jej kliniczny, ostro zakończony wygląd.
*   **Test-Time Augmentation (TTA):** Mechanizm wnioskowania wieloskalowego (0.8x, 1.0x, 1.2x) z uśrednianiem logitów, co stabilizuje predykcje na nietypowych skanach.
*   **Clinical Thresholding:** Zamiast prostego `argmax`, system stosuje indywidualne progi ufności (np. 0.3 dla IRF), priorytetyzując wykrywanie płynu nad anatomią tła.

### D. Hybrydowe Wnioskowanie i XAI (hybrid_inference.py & attention_visualizer.py)
*   **Hybrid Expert Override:** Logika w `hybrid_inference.py` łączy predykcje dwóch modeli. Model ekspercki (B0) działa jako filtr o niskim progu aktywacji dla IRF, nadpisując wyniki modelu ogólnego.
*   **Stage 2 Self-Attention:** W przeciwieństwie do standardowych podejść wizualizujących ostatnią warstwę (16x16), nasz system wyciąga mapy uwagi z **Drugiego Etapu** (Stage 2 - 64x64). Pozwala to na uzyskanie znacznie wyższej rozdzielczości przestrzennej map XAI, co ułatwia lekarzowi precyzyjną lokalizację źródła predykcji.
*   **Morphological Separation (IRF):** Zastosowanie operacji otwarcia (Morphological Opening) w celu rozdzielania "zmostkowanych" cyst płynu, co pozwala na rzetelne zliczanie liczby zmian chorobowych (Region Counting).


## 4. Wyzwania Techniczne i Rozwiązania (Challenges & Trade-offs)

### Wyzwanie 1: Niezbalansowanie danych i rzadkie patologie
**Problem:** W skanach medycznych tło dominuje nad patologią w stosunku 100:1.
**Rozwiązanie:** Połączenie **Focal Tversky Loss** z **Expert Binary Model**. Dedykowany model dla klasy IRF uczy się wyłącznie cech tej patologii, nie będąc rozpraszanym przez anatomię SRF czy PED.

### Wyzwanie 2: Optymalizacja VRAM i Stabilność Gradientu
**Problem:** Duże modele (MiT-B2) i wejście 2.5D (3 kanały) szybko zapełniają pamięć GPU.
**Rozwiązanie:** Implementacja **Gradient Accumulation** w `train.py`. Zamiast aktualizować wagi po każdej małej paczce (np. batch=4), system akumuluje gradienty z 8 kroków, osiągając stabilność *Effective Batch Size = 32* przy minimalnym zużyciu VRAM.

## 5. Instrukcja Uruchomienia (Quick Start)

1.  **Przygotowanie:** `pip install -r requirements.txt`
2.  **Trening:** `python main.py` (uruchamia pełny potok: split -> train -> eval -> viz).
3.  **Analiza Wyników:** Folder `experiments/run_[timestamp]` zawiera logi, wykresy metryk oraz wizualizacje predykcji i map uwagi.

---
*Projekt zrealizowany w ramach pracy dyplomowej na Politechnice Poznańskiej (2026).*
*Autorzy: Maksymilian Naskręt, Eryk Naumienko.*

