import numpy as np
from PIL import Image

from octseg.volume_data import VolumeRecord, group_volumes, load_volume


def test_group_volumes_numeric_ordering_and_patient_separation(tmp_path):
    img_dir = tmp_path / "img"
    mask_dir = tmp_path / "mask"
    img_dir.mkdir()
    mask_dir.mkdir()

    # Create files out of order
    names = [
        "Cirrus_PatA_011.png",
        "Cirrus_PatB_002.png",
        "Cirrus_PatA_009.png",
        "Cirrus_PatB_001.png",
        "Cirrus_PatA_010.png",
    ]
    img_paths = []
    mask_paths = []
    for name in names:
        ip = img_dir / name
        mp = mask_dir / name
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(ip)
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(mp)
        img_paths.append(str(ip))
        mask_paths.append(str(mp))

    records = group_volumes(img_paths, mask_paths)
    assert len(records) == 2
    assert records[0].patient == "Cirrus_PatA"
    assert records[1].patient == "Cirrus_PatB"

    pata_stems = [p.split("_")[-1].split(".")[0] for p in records[0].image_paths]
    assert pata_stems == ["009", "010", "011"]

    patb_stems = [p.split("_")[-1].split(".")[0] for p in records[1].image_paths]
    assert patb_stems == ["001", "002"]


def test_load_volume_shapes_and_nearest_mask_interpolation(tmp_path):
    img_dir = tmp_path / "img"
    mask_dir = tmp_path / "mask"
    img_dir.mkdir()
    mask_dir.mkdir()

    img_paths = []
    mask_paths = []
    rng = np.random.default_rng(42)

    for i in range(3):
        ip = img_dir / f"Spectralis_TRAIN001_{i:03d}.png"
        mp = mask_dir / f"Spectralis_TRAIN001_{i:03d}.png"
        # Source image 30x40
        raw_img = rng.integers(10, 200, (30, 40), dtype=np.uint8)
        # Source mask with distinct integer labels: 0, 1, 3
        raw_mask = np.zeros((30, 40), dtype=np.uint8)
        raw_mask[:10, :10] = 1
        raw_mask[15:25, 20:30] = 3

        Image.fromarray(raw_img).save(ip)
        Image.fromarray(raw_mask).save(mp)
        img_paths.append(str(ip))
        mask_paths.append(str(mp))

    record = VolumeRecord(patient="Spectralis_TRAIN001", image_paths=img_paths, mask_paths=mask_paths)
    target_size = (64, 48)  # H, W
    imgs, masks = load_volume(record, target_size)

    assert imgs.shape == (3, 64, 48)
    assert imgs.dtype == np.uint8
    assert masks.shape == (3, 64, 48)
    assert masks.dtype == np.uint8

    # Mask labels must only be from {0, 1, 3} (nearest neighbor, no interpolation artifacts)
    assert set(np.unique(masks)).issubset({0, 1, 3})


def test_predict_volume_and_validate_3d():
    from octseg.config import Config
    from octseg.evaluate3d import predict_volume
    from octseg.model import build_swin_unetr
    from octseg.train3d import validate_3d

    cfg = Config(
        ARCH="swin_unetr_3d",
        SWIN_FEATURE_SIZE=12,
        SWIN_ROI=(32, 64, 64),
        SWIN_SW_BATCH=1,
        SWIN_SW_OVERLAP=0.25,
        AUG_SIZE=(64, 64),
        USE_AMP=False,
    )
    model = build_swin_unetr(cfg, pretrained=False)
    vol_u8 = np.random.randint(0, 255, (32, 64, 64), dtype=np.uint8)
    mask_u8 = np.random.randint(0, 4, (32, 64, 64), dtype=np.uint8)

    probs = predict_volume(model, vol_u8, cfg, tta=True)
    assert probs.shape == (32, 4, 64, 64)
    assert probs.dtype == np.float32
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    miou, mhd95 = validate_3d(model, [(vol_u8, mask_u8)], cfg)
    assert 0.0 <= miou <= 1.0
    assert mhd95 >= 0.0
