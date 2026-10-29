"""Unit tests for SAM optimizer and optimizer builders."""
import pytest
import torch
import torch.nn as nn

from octseg.config import CONFIG_DIR, Config, load_config
from octseg.optim import SAM, build_optimizer


def test_sam_invalid_rho_and_eta():
    model = nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="rho"):
        SAM(opt, rho=-0.01)
    with pytest.raises(ValueError, match="eta"):
        SAM(opt, rho=0.05, eta=-0.01)


def test_sam_step_synthetic_cpu():
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    initial_w = model.weight.data.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sam = SAM(optimizer, rho=0.05, adaptive=False)

    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])
    criterion = nn.CrossEntropyLoss()

    # Pass 1
    optimizer.zero_grad()
    loss1 = criterion(model(x), y)
    loss1.backward()

    # First step perturbs weights
    sam.first_step(zero_grad=True)
    perturbed_w = model.weight.data.clone()
    assert not torch.allclose(initial_w, perturbed_w)

    # Pass 2
    loss2 = criterion(model(x), y)
    loss2.backward()

    # Restore returns to initial_w
    sam.restore()
    restored_w = model.weight.data.clone()
    assert torch.allclose(initial_w, restored_w)

    # Base step updates weights
    optimizer.step()
    final_w = model.weight.data.clone()
    assert not torch.allclose(initial_w, final_w)


def test_sam_gradient_accumulation():
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sam = SAM(optimizer, rho=0.05)
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    # Step 1 of 2
    x1, y1 = torch.randn(2, 4), torch.tensor([0, 1])
    loss1_a = criterion(model(x1), y1)
    loss1_a.backward()
    sam.first_step(zero_grad=True)

    loss1_b = criterion(model(x1), y1) / 2.0
    loss1_b.backward()
    sam.restore()
    sam.accumulate_pass2()

    # Step 2 of 2
    optimizer.zero_grad()
    x2, y2 = torch.randn(2, 4), torch.tensor([1, 0])
    loss2_a = criterion(model(x2), y2)
    loss2_a.backward()
    sam.first_step(zero_grad=True)

    loss2_b = criterion(model(x2), y2) / 2.0
    loss2_b.backward()
    sam.restore()
    sam.accumulate_pass2()

    # Flush accumulated pass 2 grads and step
    sam.flush_accumulated()
    optimizer.step()
    optimizer.zero_grad()

    # Verify optimizer state is clean
    assert len(sam._grad_acc) == 0


def test_build_optimizer_segformer():
    cfg = Config(ARCH="segformer", USE_SAM=False)
    model = nn.Linear(10, 4)
    opt, sam = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)
    assert sam is None

    cfg_sam = Config(ARCH="segformer", USE_SAM=True, SAM_RHO=0.05)
    opt_sam, sam_instance = build_optimizer(model, cfg_sam)
    assert isinstance(opt_sam, torch.optim.AdamW)
    assert isinstance(sam_instance, SAM)
    assert sam_instance.rho == 0.05


def test_config_sam_validation():
    with pytest.raises(ValueError, match="SAM_RHO"):
        Config(SAM_RHO=-0.1)

    sam_cfg = load_config(CONFIG_DIR / "sam_adamw.yaml")
    assert sam_cfg.USE_SAM is True
    assert sam_cfg.SAM_RHO == 0.05
    assert sam_cfg.SAM_ADAPTIVE is False

def test_asam_step_synthetic_cpu():
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    initial_w = model.weight.data.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sam = SAM(optimizer, rho=0.5, adaptive=True, eta=0.01)

    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()

    sam.first_step(zero_grad=True)
    perturbed_w = model.weight.data.clone()
    assert not torch.allclose(initial_w, perturbed_w)

    sam.restore()
    assert torch.allclose(initial_w, model.weight.data)


def test_config_test21():
    cfg = load_config(CONFIG_DIR / "test21.yaml")
    assert cfg.USE_SAM is True
    assert cfg.SAM_ADAPTIVE is True
    assert cfg.SAM_RHO == 0.5
    assert cfg.SAM_ETA == 0.01
    assert cfg.FOCAL_GAMMA == 2.0
    assert cfg.TVERSKY_ALPHA == 0.2
    assert cfg.TVERSKY_BETA == 0.8
    assert cfg.CLASS_MIN_REGION_SIZES == {1: 20, 2: 80, 3: 80}
    assert cfg.IRF_OPEN is False
    assert cfg.USE_DYNAMIC_WEIGHTS is False
    assert cfg.CLASS_WEIGHTS == [0.2, 5.0, 2.0, 2.0]
    assert cfg.CLASS_THRESHOLDS[1] == 0.25
