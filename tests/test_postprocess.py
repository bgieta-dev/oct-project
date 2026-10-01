import numpy as np

from octseg.postprocess import apply_thresholds, clean_regions, sharpen_ped


def _probs(c=4, size=8):
    p = np.zeros((c, size, size), dtype=np.float32)
    p[0] = 1.0
    return p


def test_reverse_priority_irf_written_last():
    p = _probs()
    p[1, :4], p[2, :4] = 0.9, 0.9  # both above threshold on the same pixels
    m = apply_thresholds(p, {1: 0.5, 2: 0.5, 3: 0.5}, irf_override=True)
    assert (m[:4] == 1).all() and (m[4:] == 0).all()


def test_irf_override_false_keeps_srf_and_ped():
    p = _probs()
    p[1] = 0.9
    p[2, :2] = 0.9
    p[3, 2:4] = 0.9
    m = apply_thresholds(p, {1: 0.5, 2: 0.5, 3: 0.5}, irf_override=False)
    assert (m[:2] == 2).all() and (m[2:4] == 3).all() and (m[4:] == 1).all()


def test_per_class_thresholds_and_batch_shape():
    p = np.stack([_probs(), _probs()])
    p[:, 1] = 0.4
    m = apply_thresholds(p, {1: 0.3, 2: 0.8, 3: 0.8})
    assert m.shape == (2, 8, 8) and (m == 1).all()
    assert (apply_thresholds(p, {1: 0.5}) == 0).all()


def test_sharpen_ped_only_touches_ped_and_stays_in_range():
    rng = np.random.default_rng(0)
    p = rng.random((4, 16, 16), dtype=np.float32)
    before = p.copy()
    sharpen_ped(p)
    assert np.array_equal(p[:3], before[:3]) and not np.array_equal(p[3], before[3])
    assert p.min() >= 0 and p.max() <= 1
    binary = rng.random((2, 16, 16), dtype=np.float32)
    assert sharpen_ped(binary.copy()).shape == binary.shape  # <= 3 classes: untouched
    assert np.array_equal(sharpen_ped(binary.copy()), binary)


def test_clean_regions_drops_small_blobs_with_per_class_sizes():
    m = np.zeros((40, 40), np.uint8)
    m[0:3, 0:3] = 2   # 9 px
    m[10:20, 10:20] = 2  # 100 px
    m[30:34, 30:34] = 3  # 16 px
    out = clean_regions(m, {2: 50, 3: 10}, irf_open=False)
    assert (out[10:20, 10:20] == 2).all() and (out[0:3, 0:3] == 0).all() and (out[30:34, 30:34] == 3).all()
    assert (clean_regions(m, 20, irf_open=False)[30:34, 30:34] == 0).all()


def test_irf_opening_breaks_thin_bridges_only_for_irf():
    m = np.zeros((30, 30), np.uint8)
    m[5:15, 5:15] = 1
    m[9, 15:25] = 1  # one-pixel bridge/spur
    assert clean_regions(m, 1, irf_open=True)[9, 20] == 0
    assert clean_regions(m, 1, irf_open=False)[9, 20] == 1
    m2 = np.where(m == 1, 2, 0).astype(np.uint8)
    assert clean_regions(m2, 1, irf_open=True)[9, 20] == 2  # SRF is never opened
