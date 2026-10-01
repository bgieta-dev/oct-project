import types

import numpy as np
import torch
from PIL import Image

from octseg.config import Config
from octseg.dataset import OCTDataset


class _Processor:
    """Returns the raw uint8 image as a CHW tensor so tests can inspect channels."""
    def __call__(self, images, return_tensors):
        return types.SimpleNamespace(pixel_values=torch.from_numpy(images.copy()).permute(2, 0, 1)[None].float())


def _volume(tmp_path, n=3, size=16):
    img_dir, mask_dir = tmp_path / "img", tmp_path / "mask"
    img_dir.mkdir(), mask_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (size, size), dtype=np.uint8)).save(img_dir / f"Cirrus_TRAIN001_{i:03d}.png")
        mask = np.zeros((size, size), np.uint8)
        mask[:4, :4], mask[8:12, 8:12] = 1, 2
        Image.fromarray(mask).save(mask_dir / f"Cirrus_TRAIN001_{i:03d}.png")
    names = [f"Cirrus_TRAIN001_{i:03d}.png" for i in range(n)]
    return [str(img_dir / f) for f in names], [str(mask_dir / f) for f in names]


def _cfg(**kw):
    return Config(AUG_SIZE=(16, 16), **kw)


def test_25d_stacks_neighbours_and_falls_back_at_volume_edges(tmp_path):
    imgs, masks = _volume(tmp_path)
    ds = OCTDataset(imgs, masks, _Processor(), cfg=_cfg(USE_25D=True))
    first = ds[0]["orig_img"]   # no t-1: falls back to the centre slice
    mid = ds[1]["orig_img"]
    last = ds[2]["orig_img"]    # no t+1
    assert np.array_equal(first[..., 0], first[..., 1]) and not np.array_equal(first[..., 1], first[..., 2])
    assert not np.array_equal(mid[..., 0], mid[..., 2])
    assert np.array_equal(last[..., 1], last[..., 2])
    assert np.array_equal(mid[..., 0], first[..., 1])  # t-1 of slice 1 is slice 0


def test_use_25d_false_is_honored(tmp_path):
    """Regression for B1: with USE_25D=False the channels must not be neighbouring slices."""
    imgs, masks = _volume(tmp_path)
    item = OCTDataset(imgs, masks, _Processor(), cfg=_cfg(USE_25D=False, USE_MULTIMODAL=False))[1]["orig_img"]
    assert np.array_equal(item[..., 0], item[..., 1]) and np.array_equal(item[..., 1], item[..., 2])


def test_target_class_is_remapped_to_binary(tmp_path):
    imgs, masks = _volume(tmp_path)
    labels = OCTDataset(imgs, masks, _Processor(), cfg=_cfg(TARGET_CLASS=2, NUM_LABELS=2))[0]["labels"]
    assert set(labels.unique().tolist()) == {0, 1} and int(labels.sum()) == 16
    labels = OCTDataset(imgs, masks, _Processor(), cfg=_cfg())[0]["labels"]
    assert set(labels.unique().tolist()) == {0, 1, 2}
