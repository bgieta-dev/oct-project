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
