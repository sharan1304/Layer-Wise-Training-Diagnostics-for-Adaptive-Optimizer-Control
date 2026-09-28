"""Research baseline optimizers for comparison experiments.

Important: these are clearly labeled approximations for local benchmarking.
They are not official NOVAK or LNGD implementations.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
from torch.optim import Optimizer


class NOVAKInspired(Optimizer):
    """NOVAK-inspired hybrid adaptive optimizer.

    Combines common components described in hybrid optimizer literature:
    Adam-style moments, RAdam-style variance rectification, decoupled weight
    decay, Nesterov-like lookahead gradient, and lightweight lookahead sync.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        lookahead_k: int = 5,
        lookahead_alpha: float = 0.5,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            lookahead_k=lookahead_k,
            lookahead_alpha=lookahead_alpha,
        )
        super().__init__(params, defaults)
        self._slow_weights: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            k = group["lookahead_k"]
            la_alpha = group["lookahead_alpha"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    self._slow_weights[id(p)] = p.detach().clone()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Nesterov-style direction: current gradient plus accumulated momentum.
                nesterov_m = beta1 * exp_avg + (1.0 - beta1) * grad

                beta2_t = beta2 ** t
                rho_inf = 2.0 / (1.0 - beta2) - 1.0
                rho_t = rho_inf - 2.0 * t * beta2_t / max(1.0 - beta2_t, 1e-16)
                bias_correction1 = 1.0 - beta1 ** t

                if rho_t > 5.0:
                    r_t = math.sqrt(
                        ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf)
                        / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
                    )
                    denom = exp_avg_sq.sqrt().add_(eps)
                    update = r_t * nesterov_m / (bias_correction1 * denom)
                else:
                    update = nesterov_m / bias_correction1

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)

                if k > 0 and t % k == 0:
                    slow = self._slow_weights[id(p)]
                    slow.add_(p - slow, alpha=la_alpha)
                    p.copy_(slow)

        return loss


class LNGDLite(Optimizer):
    """LNGD-lite diagonal layer-wise natural-gradient proxy.

    This is not the NeurIPS LNGD algorithm. It is a low-cost layer-wise
    preconditioned baseline that approximates second-order behavior with a
    diagonal Fisher/RMS statistic and an adaptive per-tensor learning-rate scale.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        beta: float = 0.95,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        max_layer_scale: float = 3.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            weight_decay=weight_decay,
            max_layer_scale=max_layer_scale,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            wd = group["weight_decay"]
            max_scale = group["max_layer_scale"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                state = self.state[p]
                if len(state) == 0:
                    state["fisher_diag"] = torch.zeros_like(p)

                fisher = state["fisher_diag"]
                fisher.mul_(beta).addcmul_(grad, grad, value=1.0 - beta)
                precond = grad / (fisher.sqrt() + eps)

                grad_norm = grad.norm().clamp_min(eps)
                precond_norm = precond.norm().clamp_min(eps)
                layer_scale = torch.clamp(grad_norm / precond_norm, 1.0 / max_scale, max_scale)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(precond, alpha=-lr * float(layer_scale.item()))

        return loss
