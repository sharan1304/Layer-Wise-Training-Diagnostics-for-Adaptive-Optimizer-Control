"""GHDHook: measures gradient health per layer and corrects each layer's update.

Usage around the optimizer step:

    loss.backward()
    hook.pre_step(loss.item())   # signals + detection from the gradients; snapshots params
    optimizer.step()             # the optimizer runs untouched
    hook.post_step()             # delta = param - param_before; correct delta; param = before + delta'

Correcting the *update* rather than ``weight.grad`` means the intervention is not
cancelled by optimizers that normalise the gradient (AdamW, LARS, LNGD).

Modes:
    'passive' - observe and log only (rule detector runs, nothing is modified)
    'rules'   - rule detector + rule confidence scores drive corrections
    'rules_vanish_only' - as 'rules', but only vanishing/recovering corrections are applied (ablation)
    'ai'      - GHDController drives corrections (falls back to rules if no checkpoint)
"""

from __future__ import annotations

import math
import warnings
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .controller import build_features, load_controller
from .optimizers import LARS, LNGD, _trust_ratio
from .core import ALL_MODES, EPS, MODES, WARMUP_STEPS, LayerSignals, LayerState, rule_strength


CORRECTING_MODES = {"vanishing", "recovering", "exploding", "oscillating", "noisy"}
VANISH_MODES = {"vanishing", "recovering"}
RESCALE_MAX = 1000.0
RECOVERY_DECAY = 0.01


def _r(x: float) -> float:
    """Round for compact JSON logs (6 significant digits), keeping it finite."""
    if not math.isfinite(x):
        return 0.0 if math.isnan(x) else math.copysign(1e30, x)
    return float(f"{x:.6g}")


class GHDHook:
    def __init__(
        self,
        model: nn.Module,
        mode: str = "rules",
        controller_path: str | Path = "results/ghd_controller.pt",
        warmup: int = WARMUP_STEPS,
        enable_noisy: bool = False,
        log_every: int = 10,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        if mode not in ("passive", "rules", "rules_vanish_only", "ai"):
            raise ValueError(f"Unknown GHD mode: {mode}")
        # Monitored layers in registration order (assumed to match forward order); a conv weight
        # is treated as one flat gradient tensor, like a linear weight.
        self.layers = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
        self.layer_params = [[p for p in m.parameters()] for m in self.layers]
        self.optimizer = optimizer
        self._group_of = {id(p): g for g in optimizer.param_groups for p in g["params"]} if optimizer else {}
        self.names = [f"L{i + 1}" for i in range(len(self.layers))]
        self.sqrt_p = [math.sqrt(m.weight.numel()) for m in self.layers]
        self.states = [LayerState() for _ in self.layers]
        self.mode = mode
        self.correcting = VANISH_MODES if mode == "rules_vanish_only" else CORRECTING_MODES
        self.warmup = warmup
        self.enable_noisy = enable_noisy
        self.log_every = log_every

        self.controller = None
        self.controller_status = "n/a"
        if mode == "ai":
            self.controller = load_controller(controller_path)
            if self.controller is None:
                warnings.warn(f"No GHD controller at {controller_path}; GHD-AI falls back to rules.")
                self.controller_status = "fallback_rules"
            else:
                self.controller_status = "mlp"

        n = len(self.layers)
        self.prev_grad: list[torch.Tensor | None] = [None] * n
        self.params_before: list[list[torch.Tensor]] | None = None
        self.prev_delta: list[list[torch.Tensor] | None] = [None] * n
        self.ema_delta: list[list[torch.Tensor] | None] = [None] * n
        self.last_vanish_strength = [0.0] * n
        self.t = 0
        self._pending: dict | None = None

        # Instability safeguard.
        self.ema_loss: float | None = None
        self.loss_hist: deque[float] = deque(maxlen=50)
        self.suspend_until = 0
        self.suspensions = 0

        # Accounting.
        self.logs: list[dict] = []
        self.total_interventions = 0
        self.false_alarm_interventions = 0
        self.layer_steps = 0
        self.mode_counts = {m: 0 for m in ALL_MODES}
        self.rule_counts: Counter[str] = Counter()

    # ------------------------------------------------------------------ safeguard
    def _safeguard(self, loss: float) -> bool:
        """Returns True if corrections are suspended at this step."""
        if not math.isfinite(loss):
            self.suspend_until = self.t + 10
            self.suspensions += 1
            return True
        if self.ema_loss is not None and len(self.loss_hist) >= 10:
            if loss > self.ema_loss + 5.0 * float(np.std(self.loss_hist)) and self.t >= self.suspend_until:
                self.suspend_until = self.t + 10
                self.suspensions += 1
        self.ema_loss = loss if self.ema_loss is None else 0.95 * self.ema_loss + 0.05 * loss
        self.loss_hist.append(loss)
        return self.t < self.suspend_until

    # ------------------------------------------------------------------ before optimizer.step()
    @torch.no_grad()
    def pre_step(self, loss: float) -> None:
        """Measure gradient health and decide a mode per layer; snapshot params."""
        self.t += 1
        t = self.t
        self._pending = None
        suspended = self._safeguard(loss)
        grads = [m.weight.grad for m in self.layers]
        if any(g is None for g in grads):
            return
        n = len(grads)

        # One device sync for all scalar measurements.
        nan = torch.tensor(float("nan"), dtype=torch.float64, device=grads[0].device)
        scalars = []
        for i, g in enumerate(grads):
            scalars.append(torch.linalg.vector_norm(g, dtype=torch.float64))
            pg = self.prev_grad[i]
            if pg is None:
                scalars.append(nan)
            else:
                dot = (g.double() * pg.double()).sum()
                scalars.append(dot / (scalars[-1] * torch.linalg.vector_norm(pg, dtype=torch.float64) + EPS))
        vals = torch.stack(scalars).tolist()
        s1 = [vals[2 * i] for i in range(n)]
        cos = [None if math.isnan(vals[2 * i + 1]) else vals[2 * i + 1] for i in range(n)]

        per_param = [s1[i] / self.sqrt_p[i] for i in range(n)]
        signals: list[LayerSignals] = []
        for i in range(n):
            last = i == n - 1
            s3 = 1.0 if last else per_param[i] / (per_param[i + 1] + EPS)
            s_depth = 1.0 if last else per_param[i] / (per_param[-1] + EPS)
            signals.append(self.states[i].compute_signals(s1[i], s3, s_depth, cos[i]))

        active = t >= self.warmup
        modes = ["warmup"] * n
        rules = ["warmup"] * n
        strengths = [0.0] * n
        if active:
            for i in range(n):
                self.states[i].update_counters(signals[i])
                modes[i], rules[i] = self.states[i].detect(signals[i], t, self.enable_noisy)
                strengths[i] = rule_strength(modes[i], signals[i], rules[i])
            if self.mode == "ai" and self.controller is not None:
                modes, strengths = self._ai_decisions(signals, t)

        for i, g in enumerate(grads):
            self.prev_grad[i] = g.detach().clone()
            if not active:
                continue
            state = self.states[i]
            if modes[i] == "recovering":
                # Keep boosting a layer that just left vanishing, with decaying confidence.
                strengths[i] = self.last_vanish_strength[i]
            if modes[i] == "vanishing":
                state.last_vanish_step = t
                self.last_vanish_strength[i] = strengths[i]
            self.layer_steps += 1
            self.mode_counts[modes[i]] += 1
            self.rule_counts[rules[i]] += 1

        if t % self.log_every == 0:
            self.logs.append({"step": t, "layers": {
                self.names[i]: {
                    "s1": _r(sig.s1), "s2": _r(sig.s2), "s3": _r(sig.s3), "s4": _r(sig.s4),
                    "s5": _r(sig.s5), "s6": _r(sig.s6), "s7": _r(sig.s7), "s_depth": _r(sig.s_depth),
                    "mode": modes[i], "strength": _r(strengths[i]),
                    "persist_v": st.p_vanish, "persist_o": st.p_oscil, "persist_n": st.p_noisy,
                }
                for i, (sig, st) in enumerate(zip(signals, self.states))
            }})

        if self.mode != "passive":
            self.params_before = [[p.detach().clone() for p in ps] for ps in self.layer_params]
            self._pending = dict(modes=modes, strengths=strengths, signals=signals, active=active, suspended=suspended)

    # ------------------------------------------------------------------ after optimizer.step()
    @torch.no_grad()
    def post_step(self) -> None:
        """Correct each layer's update: param = param_before + corrected(param - param_before)."""
        pending, before = self._pending, self.params_before
        self._pending = None
        if pending is None or before is None:
            return
        n = len(self.layers)
        deltas = [[p - b for p, b in zip(ps, bs)] for ps, bs in zip(self.layer_params, before)]
        norms = torch.stack([torch.linalg.vector_norm(d[0], dtype=torch.float64) for d in deltas]).tolist()
        output_norm = norms[-1]

        for i in range(n):
            mode, strength, sig = pending["modes"][i], pending["strengths"][i], pending["signals"][i]
            raw = deltas[i]
            if pending["active"] and not pending["suspended"] and mode in self.correcting:
                decay = [self._decay_component(p, b) for p, b in zip(self.layer_params[i], before[i])]
                corrected = self._correct(i, raw, mode, strength, sig, norms[i], output_norm, decay)
                if corrected is not None:
                    for p, b, d in zip(self.layer_params[i], before[i], corrected):
                        p.copy_(b + d)
                    self.total_interventions += 1
                    if sig.s7 > 0.70:
                        self.false_alarm_interventions += 1
            if self.enable_noisy:
                ema = self.ema_delta[i]
                self.ema_delta[i] = [d.clone() for d in raw] if ema is None else [0.9 * e + 0.1 * d for e, d in zip(ema, raw)]
            self.prev_delta[i] = raw
        self.params_before = None

    def _decay_component(self, p: torch.Tensor, before: torch.Tensor) -> torch.Tensor | None:
        """The weight-decay part of this step's update, so corrections can leave it unscaled.

        SGD/AdamW: -lr * wd * theta. LARS: -lr * phi * wd * theta (decay goes through the trust
        ratio). LNGD: none (its weight_decay is only the trust-ratio lambda).
        Momentum-carried decay from earlier steps is not separable and stays in the gradient part.
        """
        group = self._group_of.get(id(p))
        if group is None or not group.get("weight_decay") or isinstance(self.optimizer, LNGD):
            return None
        lr, wd = group["lr"], group["weight_decay"]
        if isinstance(self.optimizer, LARS):
            phi = _trust_ratio(before, p.grad, group["eta"], wd, group["eps"]) if p.grad is not None else 1.0
            return before * (-lr * phi * wd)
        return before * (-lr * wd)

    # ------------------------------------------------------------------ AI decisions
    def _ai_decisions(self, signals: list[LayerSignals], t: int) -> tuple[list[str], list[float]]:
        n = len(signals)
        feats = [
            build_features(vars(sig), st.p_vanish, st.p_oscil, st.p_noisy, i, n, t)
            for i, (sig, st) in enumerate(zip(signals, self.states))
        ]
        x = torch.nan_to_num(torch.tensor(feats, dtype=torch.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        probs, strength = self.controller.predict(x)
        modes, strengths = [], []
        for i, k in enumerate(probs.argmax(-1).tolist()):
            mode = MODES[k]
            if mode == "exploding" and self.states[i].recently_vanishing(t):
                mode = "recovering"
            if mode == "noisy" and not self.enable_noisy:
                mode = "healthy"
            modes.append(mode)
            strengths.append(float(strength[i]))
        return modes, strengths

    # ------------------------------------------------------------------ corrections (on the update)
    def _correct(
        self,
        i: int,
        delta: list[torch.Tensor],
        mode: str,
        strength: float,
        sig: LayerSignals,
        delta_norm: float,
        output_norm: float,
        decay: list[torch.Tensor | None] | None = None,
    ) -> list[torch.Tensor] | None:
        if mode in ("vanishing", "recovering"):
            ratio = min(max(output_norm / (delta_norm + 1e-8), 1.0), RESCALE_MAX)
            if mode == "recovering":
                since = self.t - (self.states[i].last_vanish_step or self.t)
                conf_eff = strength * math.exp(-RECOVERY_DECAY * since)
            else:
                conf_eff = strength  # no decay while still vanishing
            # A vanishing correction only ever amplifies; strength 0 means "leave the update alone".
            mult = max(1.0, ratio * conf_eff)
            if mult == 1.0:
                return None
            # Amplify only the gradient-driven part; weight decay stays at its natural size.
            decay = decay or [None] * len(delta)
            return [d * mult if wd is None else (d - wd) * mult + wd for d, wd in zip(delta, decay)]
        if mode == "exploding":
            ratio = sig.s1 / (sig.median + 1e-8)
            delta_e = max(0.0, min(0.15 * math.log10(max(ratio, 1e-12)), 0.90))
            if delta_e <= 0.0:
                return None
            return [d * (1.0 - delta_e) for d in delta]
        if mode == "oscillating":
            prev = self.prev_delta[i]
            if prev is None:
                return None
            alpha = 1.0 - 0.30 * strength
            return [alpha * d + (1.0 - alpha) * p for d, p in zip(delta, prev)]
        if mode == "noisy":
            ema = self.ema_delta[i]
            if ema is None:
                return None
            beta = 1.0 - 0.25 * strength
            return [beta * d + (1.0 - beta) * e for d, e in zip(delta, ema)]
        return None

    # ------------------------------------------------------------------ reporting
    def summary(self) -> dict:
        early = [
            rec["s_depth"]
            for entry in self.logs if entry["step"] > self.warmup
            for name, rec in entry["layers"].items() if int(name[1:]) <= 5
        ]
        return {
            "total_interventions": self.total_interventions,
            "mode_counts": dict(self.mode_counts),
            "mean_sdepth_early": float(np.mean(early)) if early else None,
            "false_alarm_rate": self.false_alarm_interventions / max(self.layer_steps, 1),
            "rule_counts": dict(self.rule_counts),
            "safeguard_suspensions": self.suspensions,
            "controller": self.controller_status,
        }
