# Metodologia

Opis metod zgodny z kodem w `octseg/` (wersja po reorganizacji). Zastępuje `Dokumentacja.md` i `summary.md`
(oryginały w `docs/archive/`); niezgodności z kodem zostały poprawione (patrz "Różnice względem starych opisów").

## 1. Zadanie i dane

Segmentacja semantyczna płynów w obrazach OCT (RETOUCH; urządzenia Cirrus, Spectralis, Topcon):
0 = tło, 1 = **IRF** (płyn wewnątrzsiatkówkowy), 2 = **SRF** (podsiatkówkowy), 3 = **PED** (odwarstwienie
nabłonka barwnikowego).

* Pliki: `data_folder/cropped_images/{device}_{patient}_{slice}.png` i `data_folder/cropped_masks/` (ta sama nazwa).
* **Podział** wg pacjentów, stratyfikowany po urządzeniu (`octseg/splits.py`, seed 42): 56 / 7 / 7 pacjentów
  (6922 przekroje). Podział jest zamrożony w `splits/{train,val,test}_patients.txt`; trening, ewaluacja
  i wizualizacje czytają te pliki, nie przeliczają podziału.
* **Kontekst 2.5D** (`octseg/dataset.py`): kanały = przekroje *t-1, t, t+1* (na brzegu woluminu brakujący sąsiad
  jest zastępowany przekrojem środkowym). Każdy kanał jest normalizowany osobno (clipping 1–99 percentyla).

## 2. Model i trening (`octseg/train.py`)

* SegFormer **MiT-B2** (`nvidia/mit-b2`, wagi ImageNet), 4 klasy; dropout 0,2 (hidden/attention/classifier).
  Mniejsze modele (B0) i większe (B3) wybiera się w YAML-u; rozmiar batcha wynika z backbone'u
  (B2/B3: 8 × 4 akumulacje, B0/B1: 16 × 2; efektywny batch = 32).
* Optymalizator **AdamW**, `lr = 5e-5`, `weight_decay = 5e-2`, przycinanie gradientu do normy 1,0, AMP.
* Harmonogram: **cosinusowy z rozgrzewką** (`get_cosine_schedule_with_warmup`, 15 epok rozgrzewki, krok po epoce),
  80 epok.
* Funkcja straty: `0,5 · Focal (γ = 3, wagi klas dynamiczne) + 0,5 · Tversky (α = 0,1, β = 0,9)`
  (`octseg/losses.py`). Wagi dynamiczne: odwrotność częstości pikseli klasy (wygładzanie Laplace'a),
  tło ograniczone do 0,2. **Boundary Loss jest wyłączony** (`USE_BOUNDARY_LOSS: false`) — w testach 13/15
  pogarszał HD95 i destabilizował trening.
* Augmentacje: CLAHE, RandomResizedCrop (skala 0,8–1,0), flip poziomy, obrót ±10°, elastic/grid distortion,
  jasność/kontrast, szum gaussowski.
* **Walidacja** po każdej epoce: `argmax` + usuwanie regionów < 80 px (bez otwarcia morfologicznego);
  zapisywany jest checkpoint z najlepszym mIoU. mHD95 w walidacji liczy **karę 100** za klasę pominiętą
  lub fałszywie wykrytą (polityka `penalty100`).
* Powtarzalność: `seed_everything(SEED)` (python/numpy/torch), generator i `worker_init_fn` w `DataLoader`;
  `cudnn.deterministic` celowo **nie** jest wymuszane (AMP, 12 GB VRAM).

## 3. Ewaluacja (`octseg/evaluate.py`, `postprocess.py`, `metrics.py`)

Potok wnioskowania dla jednego przekroju:

1. **TTA**: skale 0,8 / 1,0 / 1,2 × (oryginał + odbicie poziome), uśrednienie logitów (`predict_logits`).
2. softmax → **wyostrzenie PED** (filtr 3×3 `[[0,-1,0],[-1,5,-1],[0,-1,0]]` na mapie prawdopodobieństwa klasy 3).
3. **Progi kliniczne zamiast argmax** (odwrotny priorytet — IRF zapisywany jako ostatni):
   IRF 0,30, SRF 0,80, PED 0,80 (`apply_thresholds`).
4. **Czyszczenie regionów** (`clean_regions`): dla IRF otwarcie morfologiczne 3×3 (rozdziela cienkie mostki
   między cystami), następnie usunięcie składowych spójnych (8-sąsiedztwo) mniejszych niż 80 px.

Metryki (`SegmentationMetrics`): IoU i Dice z macierzy pomyłek (średnia po 4 klasach, z tłem), HD95 i ASD
(medpy) dla klas obecnych jednocześnie w predykcji i w GT, średnia liczba regionów GT/predykcja, kontrast
granicy (boundary precision), średnia powierzchnia GT.

**Polityka HD95 w raportowanych wynikach: `skip`** — przekroje, w których klasa występuje tylko w GT albo
tylko w predykcji, są pomijane (tak powstały liczby w pracy). Walidacja podczas treningu używa `penalty100`
(kara 100), więc `mHD95` z logu treningu i z końcowej ewaluacji **nie są porównywalne**. Każdy katalog ewaluacji
zawiera `metrics.yaml` z pełnymi wynikami i użytą polityką.

## 4. Wnioskowanie hybrydowe (`octseg/hybrid_inference.py`, `evaluate_hybrid.py`)

Model bazowy (B2, 4 klasy) + binarny ekspert IRF (B0, 2 klasy, 2.5D, `configs/irf_expert.yaml`:
Tversky α = 0,3 / β = 0,7, wagi klas [1; 8]). Oba modele dostają ten sam obraz 2.5D.

* tryb **soft** (domyślny): mapa IRF jest mieszana z mapą eksperta (`linear`, `geometric`, `harmonic`, `max`,
  `min`, `confidence`; waga eksperta 0,4), potem wyostrzenie PED, progi (IRF 0,25) i czyszczenie
  (minimalny region IRF 12 px, SRF/PED 80 px). Domyślnie IRF nie nadpisuje SRF/PED (`irf_override: false`);
* tryb **hard**: maska bazowa + dodanie IRF tam, gdzie ekspert przekracza próg, wyłącznie nad tłem/IRF.

Wartości domyślne: `configs/hybrid.yaml` (`HYBRID_*`). Domyślna logika w kodzie (miękkie mieszanie
prawdopodobieństw) różni się od pierwotnego planu z `IRF_EXPERT_PLAN.md` (maska eksperta zastępuje IRF bazy);
patrz `docs/ROADMAP.md`.

## 5. Interpretowalność (`octseg/attention_visualizer.py`)

Mapy uwagi to uśrednione po głowach i zapytaniach macierze samouwagi **kolejnych bloków enkodera** (plik
`..._block{i}.png`, `i` = numer bloku w całym enkoderze, nie etap SegFormera). Siatka predykcji
(`predictions.png`) pokazuje blok 5. Starsze rysunki w pracy (`docs/thesis/test*/attention_maps/…_stage*.png`)
używają starej nazwy "stage", która oznacza ten sam indeks bloku.

## 6. Różnice względem starych opisów

| Stare twierdzenie | Stan faktyczny |
|---|---|
| Soft-CRF, rafinacja dwustronna | brak kodu CRF (`USE_SOFT_CRF` usunięte) |
| Maskowanie strefy siatkówki | usunięte z kodu |
| LR 6e-5, rozgrzewka + zanik wielomianowy | `5e-5`, zanik cosinusowy z rozgrzewką |
| TTA ze skalą 1,5 | skale `[0,8; 1,0; 1,2]` |
| Boundary Loss używany | wyłączony |
| Mapy uwagi z "Stage 2 (64×64)" | indeks bloku enkodera (rozmiar siatki zależy od bloku) |
| Ekspert IRF: α = 0,05 / β = 0,95 | α = 0,3 / β = 0,7 (`configs/irf_expert.yaml`) |
| Wyodrębnianie przypadków porażek | nie ma w bieżącym `evaluate.py` (było w `experiments/test10/eval.py`) |
