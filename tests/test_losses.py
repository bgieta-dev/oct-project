import torch
import torch.nn.functional as F

from octseg.losses import FocalLoss, TverskyLoss


def test_tversky_zero_at_perfect_prediction():
    target = torch.randint(0, 4, (2, 16, 16))
    logits = F.one_hot(target, 4).permute(0, 3, 1, 2).float() * 50
    assert TverskyLoss(0.1, 0.9, 4)(logits, target).item() < 1e-4


def test_tversky_penalises_wrong_prediction():
    target = torch.randint(1, 4, (2, 16, 16))
    wrong = F.one_hot((target % 3) + 1, 4).permute(0, 3, 1, 2).float() * 50
    assert TverskyLoss(0.1, 0.9, 4)(wrong, target).item() > 0.9


def test_focal_reduces_to_cross_entropy_when_gamma_zero():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 4, 8, 8, generator=g)
    y = torch.randint(0, 4, (2, 8, 8), generator=g)
    assert torch.allclose(FocalLoss(gamma=0)(x, y), F.cross_entropy(x, y), atol=1e-6)


def test_tversky_nd_matches_4d():
    loss_fn = TverskyLoss(0.1, 0.9, 4)
    g = torch.Generator().manual_seed(42)
    pred_4d = torch.randn(2, 4, 16, 16, generator=g)
    target_4d = torch.randint(0, 4, (2, 16, 16), generator=g)
    pred_5d = pred_4d.unsqueeze(2)  # [B, C, 1, H, W]
    target_5d = target_4d.unsqueeze(1)  # [B, 1, H, W]
    loss_4d = loss_fn(pred_4d, target_4d)
    loss_5d = loss_fn(pred_5d, target_5d)
    assert torch.allclose(loss_4d, loss_5d)
