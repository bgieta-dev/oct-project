import os

import pytest

from octseg.config import Config
from octseg.splits import get_stratified_splits, load_splits, patient_of, write_splits


def _files():
    devices = {"Cirrus": 10, "Spectralis": 10, "Topcon": 10}
    return [f"{d}_TRAIN{i:03d}_{s:03d}.png" for d, n in devices.items() for i in range(n) for s in range(3)]


def test_patient_disjoint_and_complete():
    files = _files()
    tr, va, te = get_stratified_splits(files)
    sets = [set(tr), set(va), set(te)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set().union(*sets) == {patient_of(f) for f in files}
    assert {p.split("_")[0] for s in sets for p in s} == {"Cirrus", "Spectralis", "Topcon"}
    for s in sets[1:]:
        assert {p.split("_")[0] for p in s} == {"Cirrus", "Spectralis", "Topcon"}  # stratified


def test_deterministic_and_seed_dependent():
    files = _files()
    a, b = get_stratified_splits(files), get_stratified_splits(files)
    assert all(list(x) == list(y) for x, y in zip(a, b))
    assert list(get_stratified_splits(files, seed=1)[2]) != list(a[2])


def test_write_load_roundtrip(tmp_path):
    parts = get_stratified_splits(_files())
    write_splits(tmp_path, *parts)
    assert load_splits(tmp_path) == tuple([str(p) for p in x] for x in parts)


def test_single_patient_device_rejected():
    with pytest.raises(ValueError):
        get_stratified_splits(["A_P1_001.png", "B_P1_001.png", "B_P2_001.png"])


def test_frozen_splits_match_current_data():
    cfg = Config()
    if not os.path.isdir(cfg.IMG_DIR):
        pytest.skip("data_folder not present")
    parts = get_stratified_splits(sorted(os.listdir(cfg.IMG_DIR)))
    assert load_splits(cfg.SPLIT_DIR) == tuple([str(p) for p in x] for x in parts)
