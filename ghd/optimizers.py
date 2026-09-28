"""Optimizers for Experiment 1: SGD, AdamW, LARS and LNGD (diagonal Fisher)."""

from __future__ import annotations

import torch
from torch.optim import Optimizer


def _trust_ratio(p: torch.Tensor, u: torch.Tensor, eta: float, lam: float, eps: float) -> float:
    """eta*||theta|| / (||u|| + lambda*||theta|| + eps), in float64.

    On vanishing layers the ratio reaches ~1e6 and the squared norms underflow
    in float32, so both norms are accumulated in double precision.
    """
    w_norm = torch.linalg.vector_norm(p, dtype=torch.float64)
    u_norm = torch.linalg.vector_norm(u, dtype=torch.float64)
    if w_norm.item() == 0.0 or u_norm.item() == 0.0:
        return 1.0  # zero-initialised biases / no gradient: plain lr
    return float(eta * w_norm / (u_norm + lam * w_norm + eps))


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling (You et al., 2017)."""

    def __init__(self, params, lr: float = 0.1, momentum: float = 0.9, eta: float = 0.001,
                 weight_decay: float = 1e-4, eps: float = 1e-8) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, eta=eta, weight_decay=weight_decay, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lam = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                phi = _trust_ratio(p, p.grad, group["eta"], lam, group["eps"])
                update = p.grad.add(p, alpha=lam) if lam else p.grad.clone()
                update.mul_(group["lr"] * phi)
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = update
                else:
                    buf.mul_(group["momentum"]).add_(update)
                p.sub_(buf)
        return loss


class LNGD(Optimizer):
    """Layer-wise natural gradient with a diagonal Fisher (Liu et al., NeurIPS 2024),
    per the Experiment 1 specification."""

    def __init__(self, params, lr: float = 0.01, beta2: float = 0.999, eta: float = 0.001,
                 weight_decay: float = 1e-4, eps: float = 1e-8) -> None:
        super().__init__(params, dict(lr=lr, beta2=beta2, eta=eta, weight_decay=weight_decay, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta2, eps = group["beta2"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                v = state["v"]
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                v_hat = v / (1 - beta2 ** state["step"])
                g_nat = g / (v_hat.sqrt() + eps)
                phi = _trust_ratio(p, g_nat, group["eta"], group["weight_decay"], eps)
                p.add_(g_nat, alpha=-group["lr"] * phi)
        return loss


OPTIMIZER_LR = {"sgd": 0.1, "adamw": 1e-3, "lars": 1.0, "lngd": 0.1}


def make_optimizer(name: str, params, lr: float | None = None) -> Optimizer:
    lr = OPTIMIZER_LR[name] if lr is None else lr
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)
    if name == "lars":
        return LARS(params, lr=lr, momentum=0.9, eta=0.001, weight_decay=1e-4)
    if name == "lngd":
        return LNGD(params, lr=lr, beta2=0.999, eta=0.001, weight_decay=1e-4)
    raise ValueError(f"Unknown optimizer: {name}")
