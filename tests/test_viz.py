import numpy as np
from PIL import Image

from octseg.config import Config
from octseg.viz import (
    plot_history,
    save_failure_cases,
    save_predictions_grid,
    select_vis_indices,
)


def test_select_vis_indices_picks_highest_class_count(tmp_path):
    masks = []
    # Slice 0: 5 IRF pixels
    m0 = np.zeros((10, 10), dtype=np.uint8)
    m0[:5, 0] = 1
    p0 = tmp_path / "m0.png"
    Image.fromarray(m0).save(p0)
    masks.append(str(p0))

    # Slice 1: 20 IRF pixels, 3 SRF pixels
    m1 = np.zeros((10, 10), dtype=np.uint8)
    m1[:4, :5] = 1
    m1[5:8, 0] = 2
    p1 = tmp_path / "m1.png"
    Image.fromarray(m1).save(p1)
    masks.append(str(p1))

    vis_idx, max_cnt = select_vis_indices(masks, classes=[1, 2, 3])
    assert vis_idx[1] == 1
    assert max_cnt[1] == 20
    assert vis_idx[2] == 1
    assert max_cnt[2] == 3
    assert vis_idx[3] == -1
    assert max_cnt[3] == 0


def test_select_vis_indices_target_class_remapping(tmp_path):
    masks = []
    m0 = np.zeros((10, 10), dtype=np.uint8)
    m0[:3, :3] = 2  # target class = 2, so this becomes class 1 in binary mask
    p0 = tmp_path / "m0.png"
    Image.fromarray(m0).save(p0)
    masks.append(str(p0))

    vis_idx, max_cnt = select_vis_indices(masks, classes=[1], target_class=2)
    assert vis_idx[1] == 0
    assert max_cnt[1] == 9


def test_save_predictions_grid_empty_and_normal(tmp_path):
    cfg = Config()
    out_png = tmp_path / "preds.png"

    # Empty vis_data should exit cleanly without creating file
    save_predictions_grid({}, {}, cfg, str(out_png))
    assert not out_png.exists()

    # Normal single-class entry
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    gt = np.zeros((16, 16), dtype=np.uint8)
    gt[2:6, 2:6] = 1
    pred = np.zeros((16, 16), dtype=np.uint8)
    pred[2:5, 2:5] = 1
    att = np.ones((8, 8), dtype=np.float32)

    vis_data = {1: (img, gt, pred, att)}
    counts = {1: 16}
    save_predictions_grid(vis_data, counts, cfg, str(out_png))
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_plot_history(tmp_path):
    history = {"loss": [0.5, 0.4, 0.3], "miou": [0.6, 0.7, 0.8], "mhd95": [80.0, 70.0, 60.0]}
    out_png = tmp_path / "metrics.png"
    plot_history(history, str(out_png))
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_save_failure_cases(tmp_path):
    cfg = Config()
    out_dir = tmp_path / "eval_out"

    # Empty records: no failures directory or files created
    save_failure_cases([], str(out_dir), cfg, top_k=5)
    assert not (out_dir / "failures").exists()

    # Non-empty records
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    gt = np.zeros((16, 16), dtype=np.uint8)
    gt[2:6, 2:6] = 1
    prd = np.zeros((16, 16), dtype=np.uint8)

    records = [
        (0.40, 10, img, gt, prd),
        (0.10, 42, img, gt, prd),
        (0.25, 7, img, gt, prd),
        (0.80, 15, img, gt, prd),
    ]

    save_failure_cases(records, str(out_dir), cfg, top_k=2)

    f1 = out_dir / "failures" / "failure_1_idx_42.png"
    f2 = out_dir / "failures" / "failure_2_idx_7.png"
    f3 = out_dir / "failures" / "failure_3_idx_10.png"

    assert f1.exists() and f1.stat().st_size > 0
    assert f2.exists() and f2.stat().st_size > 0
    assert not f3.exists()
