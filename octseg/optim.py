"""Optimizer builders and Sharpness-Aware Minimization (SAM) wrapper.

Reference:
    Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization",
    ICLR 2021.
"""
from collections import defaultdict
from typing import Optional, Tuple

import torch
import torch.nn as nn

from octseg.model import optimizer_param_groups


class SAM:
    """Sharpness-Aware Minimization (SAM) optimizer wrapper.

    Minimizes loss and loss sharpness simultaneously by seeking parameters whose
    entire neighborhood has uniformly low training loss:
        min_w max_{||e||_2 <= rho} L(w + e)

    Fully compatible with PyTorch AMP GradScaler, gradient accumulation, and
    parameter groups.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, rho: float = 0.05, adaptive: bool = False):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        self.optimizer = optimizer
        self.rho = rho
        self.adaptive = adaptive
        self.state = defaultdict(dict)
        self._grad_acc = {}

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def defaults(self):
        return self.optimizer.defaults

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        """Calculates adversarial perturbation e_w and perturbs parameters: w <- w + e_w."""
        grad_norm = self._grad_norm()
        if not torch.isfinite(grad_norm) or grad_norm == 0:
            if zero_grad:
                self.zero_grad()
            return

        scale = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def restore(self):
        """Restores original unperturbed parameters: w <- old_w."""
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state and "old_p" in self.state[p]:
                    p.data.copy_(self.state[p].pop("old_p"))

    @torch.no_grad()
    def accumulate_pass2(self):
        """Accumulates pass 2 gradients across accumulation micro-batches."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if p not in self._grad_acc:
                        self._grad_acc[p] = p.grad.clone()
                    else:
                        self._grad_acc[p].add_(p.grad)

    @torch.no_grad()
    def flush_accumulated(self):
        """Copies accumulated pass 2 gradients back into parameter grad attributes."""
        if not self._grad_acc:
            return
        for group in self.param_groups:
            for p in group["params"]:
                if p in self._grad_acc:
                    if p.grad is None:
                        p.grad = self._grad_acc[p].clone()
                    else:
                        p.grad.copy_(self._grad_acc[p])
        self._grad_acc.clear()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Restores parameters and performs base optimizer step."""
        self.restore()
        self.optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    g = p.grad
                    if self.adaptive:
                        g = g * torch.abs(p)
                    norms.append(torch.norm(g, p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)


def build_optimizer(model: nn.Module, cfg) -> Tuple[torch.optim.Optimizer, Optional[SAM]]:
    """Builds base optimizer (AdamW) and optional SAM wrapper according to `cfg`.

    Args:
        model: PyTorch model module.
        cfg: configuration with ARCH, LR, WEIGHT_DECAY, USE_SAM, SAM_RHO, SAM_ADAPTIVE.

    Returns:
        (optimizer, sam_wrapper) tuple. sam_wrapper is None if cfg.USE_SAM is False.
    """
    if cfg.ARCH in ("segformer", "retfound"):
        param_groups = optimizer_param_groups(model, cfg)
        optimizer = torch.optim.AdamW(param_groups, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.ARCH == "swin_unetr_3d":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unsupported ARCH {cfg.ARCH!r} for optimizer")

    sam: Optional[SAM] = None
    if cfg.USE_SAM:
        sam = SAM(optimizer, rho=cfg.SAM_RHO, adaptive=cfg.SAM_ADAPTIVE)

    return optimizer, sam
