import numpy as np
import pytest

from octseg.metrics import SegmentationMetrics


def _mask(*regions, size=32):
    m = np.zeros((size, size), np.uint8)
    for c, (r0, r1, c0, c1) in regions:
        m[r0:r1, c0:c1] = c
    return m


def test_iou_dice_from_confusion_matrix():
    gt = _mask((1, (0, 4, 0, 4)))
    pred = _mask((1, (0, 4, 0, 2)))
    m = SegmentationMetrics(2)
    m.update(gt, pred)
    ious, dices = m.ious_dices()
    assert ious[1] == pytest.approx(8 / 16, abs=1e-5)
    assert dices[1] == pytest.approx(2 * 8 / (8 + 16), abs=1e-5)


def test_class_absent_everywhere_gives_zero_not_nan():
    m = SegmentationMetrics(3)
    m.update(np.zeros((8, 8), np.uint8), np.zeros((8, 8), np.uint8))
    res = m.compute()
    assert res["class_ious"][2] == 0.0 and res["class_dices"][1] == 0.0
    assert res["mHD95"] == 0.0 and res["class_hd95"][1] == 0.0


def test_hd95_skip_vs_penalty_policy():
    gt = _mask((1, (0, 4, 0, 4)))
    empty = np.zeros_like(gt)
    for policy, expected in (("skip", 0.0), ("penalty100", 100.0)):
        m = SegmentationMetrics(2, hd95_missing=policy)
        m.update(gt, empty)       # missed class
        m.update(empty, gt)       # false class
        assert m.compute()["class_hd95"][1] == expected
    m = SegmentationMetrics(2, hd95_missing="skip")
    m.update(gt, gt)
    m.update(gt, empty)
    assert m.compute()["class_hd95"][1] == 0.0  # perfect overlap counted, miss skipped


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        SegmentationMetrics(2, hd95_missing="nope")


def test_region_counts_and_area():
    gt = _mask((1, (0, 3, 0, 3)), (1, (10, 13, 10, 13)))
    pred = _mask((1, (0, 3, 0, 3)))
    m = SegmentationMetrics(2)
    m.update(gt, pred, np.zeros((32, 32), np.uint8))
    res = m.compute()
    assert res["class_avg_regions_gt"][1] == 2 and res["class_avg_regions_pred"][1] == 1
    assert res["class_avg_pixel_area"][1] == 18
