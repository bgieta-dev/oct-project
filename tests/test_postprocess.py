import numpy as np

from octseg.postprocess import (
    apply_thresholds,
    clean_regions,
    sharpen_ped,
    volumetric_consistency_filter,
)


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


def test_sharpen_ped_factor():
    rng = np.random.default_rng(42)
    p = rng.random((4, 16, 16), dtype=np.float32)
    p_orig = p.copy()

    p_f0 = sharpen_ped(p.copy(), factor=0.0)
    assert np.array_equal(p_f0, p_orig)

    p_f1 = sharpen_ped(p.copy(), factor=1.0)
    assert not np.array_equal(p_f1[3], p_orig[3])

    p_f05 = sharpen_ped(p.copy(), factor=0.5)
    expected = np.clip(0.5 * p_orig[3] + 0.5 * p_f1[3], 0, 1)
    assert np.allclose(p_f05[3], expected, atol=1e-6)
    assert np.array_equal(p_f05[:3], p_orig[:3])


def test_volumetric_consistency_filter():
    vol = np.zeros((5, 10, 10), dtype=np.uint8)
    # Isolated 1-slice component (class 1) at z=1
    vol[1, 2:5, 2:5] = 1
    # 2-slice component (class 2) at z=2, 3
    vol[2:4, 6:9, 6:9] = 2
    # 3-slice component (class 3) at z=1, 2, 3
    vol[1:4, 0:2, 0:2] = 3

    filtered = volumetric_consistency_filter(vol, min_slices=2)
    # 1-slice component removed
    assert (filtered[1, 2:5, 2:5] == 0).all()
    # 2-slice and 3-slice components preserved
    assert (filtered[2:4, 6:9, 6:9] == 2).all()
    assert (filtered[1:4, 0:2, 0:2] == 3).all()

    # Test 4D batch
    batch_vol = np.stack([vol, vol])
    batch_filtered = volumetric_consistency_filter(batch_vol, min_slices=2)
    assert batch_filtered.shape == (2, 5, 10, 10)
    assert (batch_filtered[:, 1, 2:5, 2:5] == 0).all()
    assert (batch_filtered[:, 2:4, 6:9, 6:9] == 2).all()
    assert (batch_filtered[:, 1:4, 0:2, 0:2] == 3).all()

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
