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
* Opcjonalna regularyzacja geometrii funkcji straty: **SAM (Sharpness-Aware Minimization)** (Foret et al., 2021) oraz **ASAM (Adaptive SAM)** (Kwon et al., 2021) z bazowym optymalizatorem AdamW (`configs/sam_adamw.yaml`, `configs/test21.yaml`, `USE_SAM: true`). SAM minimalizuje jednocześnie wartość funkcji straty oraz jej lokalną ostrość: $\min_w \max_{\|\epsilon\|_2 \le \rho} \mathcal{L}(w + \epsilon)$. W wariancie adaptacyjnym (`SAM_ADAPTIVE: true`, $\rho = 0{,}5$, $\eta = 0{,}01$) perturbacja jest niezmiennicza względem skali wag: $T_w = |w| + \eta$, $e_w = \rho \frac{T_w^2 \odot \nabla_w \mathcal{L}}{\| T_w \odot \nabla_w \mathcal{L} \|_2}$, co chroni drobne filtry mikrotorbieli (IRF) w początkowych blokach MiT przed nadmiernym wygładzeniem gradientowym przy zachowaniu silnej regularyzacji dużych macierzy projekcji. W pełni kompatybilny z AMP GradScaler i wielokrotną akumulacją gradientów (`octseg/optim.py`).
* Harmonogram: **cosinusowy z rozgrzewką** (`get_cosine_schedule_with_warmup`, 15 epok rozgrzewki, krok po epoce),
  80 epok.
* Funkcja straty: `0,5 · Focal (γ = 3, wagi klas dynamiczne) + 0,5 · Tversky (α = 0,1, β = 0,9)`
  (`octseg/losses.py`). Wagi dynamiczne: odwrotność częstości pikseli klasy (wygładzanie Laplace'a),
  tło ograniczone do 0,2. **Boundary Loss jest wyłączony** (`USE_BOUNDARY_LOSS: false`) — w testach 13/15
  pogarszał HD95 i destabilizował trening.
* Augmentacje: CLAHE, RandomResizedCrop (skala 0,8–1,0), flip poziomy, obrót ±10°, elastic/grid distortion,
  jasność/kontrast, szum gaussowski.
* **Walidacja i post-processing**: `argmax` + klasowo zróżnicowane progi wielkości regionów (`CLASS_MIN_REGION_SIZES: {1: 20, 2: 80, 3: 80}`, pole `MIN_REGION_SIZE`), zapobiegające usuwaniu mikrotorbieli IRF ($20\text{--}80\text{ px}$) przy odrzucaniu artefaktów tła; opcjonalne wyłączenie destrukcyjnego otwarcia morfologicznego dla IRF (`IRF_OPEN: false`). Zapisywany jest checkpoint z najlepszym mIoU. mHD95 w walidacji liczy **karę 100** za klasę pominiętą lub fałszywie wykrytą (polityka `penalty100`).
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

Uwaga: migawki raportowanych przebiegów test18 i test19 (`experiments/test19/config.py`,
`git show 4657305:experiments/test18/config.py`) mają `CLASS_THRESHOLDS = {1: 0.30, 2: 0.40, 3: 0.40}`
i `MIN_REGION_SIZE = 50`, a nie SRF/PED 0,80 i 80 px jak bieżący `configs/base.yaml`.

## 4. Wnioskowanie hybrydowe (`octseg/hybrid_inference.py`, `evaluate_hybrid.py`)

Model bazowy (B2, 4 klasy) + binarny ekspert IRF (B0, 2 klasy, 2.5D, `configs/irf_expert.yaml`:
Tversky α = 0,3 / β = 0,7, wagi klas [1; 8]). Oba modele dostają ten sam obraz 2.5D.

* tryb **soft** (domyślny): mapa IRF jest mieszana z mapą eksperta (`linear`, `geometric`, `harmonic`, `max`,
  `min`, `confidence`; waga eksperta 0,4), potem wyostrzenie PED, progi (IRF 0,25) i czyszczenie
  (minimalny region IRF 12 px, SRF/PED 80 px). Domyślnie IRF nie nadpisuje SRF/PED (`irf_override: false`);
* tryb **hard**: maska bazowa + dodanie IRF tam, gdzie ekspert przekracza próg, wyłącznie nad tłem/IRF;
* tryb **replace** (logika z `IRF_EXPERT_PLAN.md`): z maski bazowej (progi bez nadpisywania przez IRF) zostają SRF
  i PED, IRF bazy jest usuwany, a IRF eksperta (prawdopodobieństwo > progu eksperta, 0,25 w
  `configs/irf_expert.yaml`) jest wpisywany tylko na tło. Planowany krok "OR przy pewności eksperta > 0,7" nie
  wymaga osobnego kodu — próg 0,25 obejmuje już każdy piksel powyżej 0,7.

Wartości domyślne: `configs/hybrid.yaml` (`HYBRID_*`). `HybridInference.segment` = `probabilities` (oba modele,
raz na przekrój) + `postprocess` (scalenie i czyszczenie; nie modyfikuje wejść), dzięki czemu wiele wariantów
scalania można ocenić na tych samych prawdopodobieństwach.

**Wybór wariantu na zbiorze walidacyjnym** (`python -m octseg.sweep_hybrid`, domyślnie `--split val`): dla każdego
przekroju liczony jest wiersz bazowy (łańcuch identyczny z `evaluate_model`) i 14 wariantów: soft `linear` z wagą
eksperta 0,3 / 0,4 / 0,5 / 0,6 × próg IRF 0,25 / 0,35; soft `confidence` i `max` (waga 0,5) × oba progi; `hard`;
`replace`. Metryki z polityką HD95 `skip`. Wybierany jest wariant o najwyższym Dice IRF spośród tych, które
przewyższają Dice IRF modelu bazowego i utrzymują Dice SRF oraz PED nie niżej niż 0,01 poniżej bazy
(`SRF_PED_TOLERANCE`); przy remisie — pierwszy w kolejności siatki. Wynik: `sweep.yaml` oraz gotowe polecenie
`evaluate_hybrid` do **jednorazowej** oceny na zbiorze testowym. Jeśli żaden wariant nie spełnia reguły,
hybryda nie jest przyjmowana i raportowany jest model bazowy.

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
| Wyodrębnianie przypadków porażek | jest: `save_failure_cases` zapisuje k najgorszych przekrojów (IoU) do `failures/` (2D i 3D) |

## 7. Model czysto transformerowy RETFound-OCT + LoRA (`ARCH: retfound`, `octseg/model.py`)

Pytanie badawcze: samonadzorowane wstępne uczenie transformera na obrazach OCT (domena docelowa) kontra MiT
uczony na ImageNet, przy identycznym potoku (podział, straty, progi, TTA, czyszczenie, metryki).

* **Enkoder**: ViT-L/16 (timm `VisionTransformer`: łata 16 px, wymiar 1024, 24 bloki, 16 głów, MLP ×4,
  LayerNorm ε = 1e-6) z wagami RETFound-MAE-OCT (Zhou i in., Nature 2023), repozytorium Hugging Face
  `YukunZhou/RETFound_mae_natureOCT` (dostęp bramkowany: akceptacja licencji + `HF_TOKEN` w `.env`). Przy
  wczytywaniu usuwane są prefiksy `backbone.` / `encoder.` i klucze dekodera MAE (`decoder*`, `mask_token`, `head`,
  `fc_norm`); zanurzenia pozycyjne są interpolowane do siatki wejścia 512×512; brak dowolnego klucza enkodera
  (poza `norm.*`) przerywa działanie (`RuntimeError`). Wejście jest dopełniane (interpolacją) do wielokrotności
  16 px, `dynamic_img_size` obsługuje skale TTA 0,8 / 1,2.
* **Adaptacja LoRA** (peft): wagi enkodera są zamrożone; adaptery rangi r = 16 (α = 32, dropout 0,1) w projekcjach
  `attn.qkv` i `attn.proj` każdego bloku. W enkoderze trenowane są wyłącznie parametry `lora_*`; enkoder używa
  gradient checkpointing. AdamW ma dwie grupy: LoRA (lr 2e-4, weight decay 0) oraz gałąź detali + dekoder
  (lr 1e-4, weight decay 0,05).
* **Gałąź detali**: dwa pierwsze etapy enkodera MiT-B0 (`nvidia/mit-b0`, wagi ImageNet; krok 4 i 8, 32 i 64
  kanały, efektywna samouwaga). Odzyskuje drobne cysty IRF, które giną w łatach 16 px ViT; trenowana w całości.
* **Dekoder All-MLP** (jak w SegFormerze): wyjścia bloków 5, 11, 17, 23 ViT (indeksy od 0,
  `RETFOUND_TAP_LAYERS`) i dwie mapy gałęzi detali → liniowa projekcja każdego tokenu do 256 kanałów →
  interpolacja biliniowa do 1/4 rozdzielczości wejścia → konkatenacja → Linear → LayerNorm → GELU → dropout 0,2
  → Linear do 4 klas. Logity mają krok 4, jak w SegFormerze. Jedyne sploty to osadzenia łat ViT i MiT.
* Normalizacja wejścia: domyślny `SegformerImageProcessor()` (512×512, średnia/odchylenie ImageNet — te same,
  których używa RETFound). Pozostałe ustawienia z `configs/base.yaml`; `configs/retfound_lora.yaml`: batch 8 × 4
  akumulacje, 80 epok, 10 epok rozgrzewki. Mapy uwagi SegFormera nie są generowane (siatka predykcji ma
  3 kolumny).
* Ewaluacja nie wymaga tokenu HF: `load_2d_model` buduje model bez pobierania wag RETFound i ściśle wczytuje
  wytrenowany checkpoint.
* **Ablacja** (planowane test20–test22, `docs/EXPERIMENTS.md`): pełny model (`retfound_lora.yaml`), bez gałęzi
  detali (`retfound_lora_nodetail.yaml`), zamrożony enkoder bez LoRA (`retfound_frozen.yaml`, r = 0).

## 8. Natywny model 3D SwinUNETR (`ARCH: swin_unetr_3d`, `volume_data.py`, `train3d.py`, `evaluate3d.py`)

* Model: MONAI 1.6.1 `SwinUNETR` (3D, 1 kanał wejściowy, 4 klasy, `feature_size` 48, drop path 0,1, gradient
  checkpointing). Opcjonalnie samonadzorowane wagi `model_swinvit.pt` (`SWIN_PRETRAINED`, w
  `configs/swin_unetr_3d.yaml`: `outputs/pretrained/model_swinvit.pt`; wymagają `feature_size` 48).
* **Wolumin** = wszystkie B-skany pacjenta uporządkowane numerycznie po indeksie przekroju. Każdy przekrój jest
  normalizowany percentylowo (1–99) w natywnej rozdzielczości, skalowany do 256×256 (`SWIN_VOLUME_SIZE`; maska
  metodą najbliższego sąsiada) i zapisywany jako uint8. Wejście jest jednokanałowe — kontekst w osi Z zapewnia
  splot 3D zamiast stosu 2.5D.
* **Trening na łatach** 32×160×160 (Z×H×W, `SWIN_ROI`; wymiary podzielne przez 32), 2 łaty na wolumin losowane
  z równymi proporcjami klas (`RandCropByLabelClassesd`). Augmentacje: odbicie w osi bocznej (W), skalowanie
  i przesunięcie intensywności (±0,1), szum gaussowski (σ = 0,02). Strata jak w 2D: Focal + Tversky
  (Tversky działa dla tensorów N-wymiarowych). AdamW (lr 1e-4, weight decay 1e-5), harmonogram cosinusowy
  z 20 epokami rozgrzewki, 300 epok, batch 1 wolumin, AMP. Walidacja co 10 epok: `argmax`, usuwanie regionów
  < 80 px bez otwarcia morfologicznego, HD95 `penalty100` (jak w 2D).
* **Wnioskowanie**: okno przesuwne z wagami gaussowskimi (nakładanie 0,5, 4 okna na partię); TTA 3D to tylko
  odbicie boczne (`TTA_SCALES` są ignorowane). Logity są interpolowane biliniowo do 512×512 (`AUG_SIZE`) przekrój
  po przekroju, potem softmax i **ten sam łańcuch co w 2D** (wyostrzenie PED, progi, otwarcie IRF, usuwanie
  regionów) oraz te same metryki (`skip`) na GT 512×512 — wyniki są porównywalne z modelami 2D.
* Brak eksperta 3D i Boundary Loss (konfiguracja zgłasza błąd). Model nie był jeszcze trenowany na danych
  RETOUCH.

## 9. Czas wnioskowania i zużycie VRAM (`octseg/benchmark.py`)

Dane testowe są wczytywane do pamięci przed pomiarem (bez odczytu z dysku); po `--warmup` wywołaniach mierzone jest
jedno wywołanie na wejście (batch 1, `torch.cuda.synchronize()` wokół pomiaru na GPU): średnia, mediana i p95 w ms
oraz szczytowa zaalokowana VRAM (`torch.cuda.max_memory_allocated`, brak na CPU). Tryby 2D: `forward` (sam model,
bez TTA), `pipeline` (TTA + softmax + wyostrzenie PED + progi + czyszczenie), `hybrid` (baza + ekspert). Tryby 3D:
`volume_forward`, `volume_pipeline`, dodatkowo `ms_per_bscan` = średni czas woluminu / średnia liczba przekrojów.
Wynik: `outputs/benchmark/<run>/benchmark.yaml`. Szczytowa VRAM treningu jest logowana na końcu każdego treningu
(`Peak training VRAM: … MB`).
