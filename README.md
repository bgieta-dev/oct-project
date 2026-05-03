# Pobierz skany
https://www.kaggle.com/datasets/saivikassingamsetty/retouch?resource=download-directory&select=retouch_processed

# Co będziemy robić?
Tematyka: Stworzenie sieci neuronowej wykonującej zadanie segmentacji obrazów OCT w diagnostyce i monitorowaniu cukrzycowego obrzęku plamki (DME). 

DME (Diabetic Macular Edema) – poważne powikłanie cukrzycy powodujące utratę wzroku wskutek uszkodzenia naczyń krwionośnych siatkówki.

IRF - Widoczny jako czarne przestrzenie (hiporefleksyjne) wewnątrz siatkówki, często przybierające postać cyst. Są to tzw. cysty śródsiatkówkowe.Mogą występować we wszystkich warstwach siatkówki, najczęściej w zewnętrznej warstwie splotowatej (Henle’s fiber layer)

SRF - Czarna przestrzeń (hiporefleksyjne) między neurosensoryczną siatkówką a nabłonkiem barwnikowym siatkówki (RPE), co prowadzi do odwarstwienia siatkówki. Występuje w przestrzeni podsiatkówkowej.

HRF - Małe, dobrze odgraniczone punkty jasne (hiperrefleksyjne). Mogą być rozproszone we wszystkich warstwach siatkówki, często wewnątrz cyst (IRF) lub w warstwach zewnętrznych. Są one uważane za predyktory stanu zapalnego lub migrujące komórki pigmentowe/lipidy.

# Możliwe modele
https://github.com/qubvel-org/segmentation_models.pytorch

CNN:
https://smp.readthedocs.io/en/latest/models.html#unetplusplus

https://smp.readthedocs.io/en/latest/models.html#manet

transformery:
https://smp.readthedocs.io/en/latest/models.html#dpt

https://smp.readthedocs.io/en/latest/models.html#segformer
CNN:
Oktay et al. (2018) – Attention U-Net: Learning Where to Look for the Pancreas (https://arxiv.org/abs/1804.03999)

Zhou et al. (2018) – UNet++: A Nested U-Net Architecture (https://arxiv.org/abs/1807.10165)

Lee et al. (2022) – Recent Advanced Deep Learning Architectures for Retinal Fluid Segmentation (MDPI Sensors) (https://www.mdpi.com/1424-8220/22/8/3055)


Transformer:
Vaswani et al. (2017) – Attention Is All You Need (mechanizm uwagi, podstawa transformerów) (https://arxiv.org/abs/1706.03762)

Dosovitskiy et al. (2020) – ViT: An Image is Worth 16×16 Words (Vision Transformer)(https://arxiv.org/abs/2010.11929)

Xie et al. (2021) – SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers (https://arxiv.org/abs/2105.15203)

Tang et al. (2022) – SwinUNETR: Swin Transformers for Semantic Segmentation (MICCAI 2022) (https://arxiv.org/abs/2201.01266)


# Zastosowane metody

hybrid Loss: Loss=a \* DiceLoss + b \* FocalLoss

Focal Loss: Została stworzona właśnie po to, by radzić sobie z ogromną przewagą tła. Skupia się na „trudnych” pikselach (tych, które sieć klasyfikuje błędnie), a ignoruje „łatwe” piksele tła.
Generalized Dice Loss (GDL): W przeciwieństwie do zwykłego Dice’a, GDL waży każdą klasę odwrotnością jej wielkości. Dzięki temu mikro-punkty HRF będą dla sieci tak samo ważne, jak wielkie obszary SRF.

Metryka:
Precision & Recall    
Kluczowe dla lekarza. Recall (Sensitivity) mówi nam, ile % faktycznych zmian chorobowych wykryliśmy. Precision mówi, jak często sieć "panikuje" bez powodu.

IoU (Jaccard Index):
Bardziej rygorystyczny niż Dice. Pozwala lepiej wyłapać błędy w małych strukturach (IRF). IoU jest bardziej czułe na błędy pojedynczych pikseli w małych obiektach niż Dice.

HD95 (Hausdorff Distance):  
Zamiast MAE. Mierzy największą odległość między konturem predykcji a konturem wzorcowym. Jest odporna na pojedyncze błędy (outliery), ale świetnie ocenia precyzję obrysu cysty.


Zastosowanie Focal Loss w miejsce standardowej Cross-Entropy umożliwia skuteczną detekcję mikro-zmian (HRF), które często są pomijane ze względu na dużą dysproporcję między tłem a obiektem. Z kolei wprowadzenie metryki HD95 pozwala na precyzyjną ocenę morfologii zmian (kształt cyst IRF/SRF), co stanowi istotną przewagę nad tradycyjnymi metodami opartymi wyłącznie na pomiarze grubości warstw siatkówki.
