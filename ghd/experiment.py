"""Experiment 1: 10-layer sigmoid MLP on MNIST, optimizer x GHD convergence race."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hook import GHDHook
from .optimizers import OPTIMIZER_LR, make_optimizer


SEEDS = [42, 123, 7]
N_STEPS = 5000
# Deep sigmoid nets train slowly under SGD-family optimizers; AdamW converges by ~2000.
N_STEPS_BY_OPT = {"sgd": 10000, "lars": 10000, "lngd": 10000, "adamw": 5000}
# Linear lr warmup from lr/10 to lr over the first steps (SGD and LARS only).
LR_WARMUP_STEPS = 200
LR_WARMUP_OPTS = {"sgd", "lars"}
EVAL_EVERY = 50
BATCH_SIZE = 128
LOG_EVERY = 10
THRESHOLDS = [50, 60, 70, 80, 90, 95]
LOSS_CHECKPOINTS = [500, 1000, 2000, 5000]
SPLIT_SEED = 0  # fixed so every run sees the same 50k/10k train/val split

# (optimizer, ghd_mode) in display order.
CONFIGS = [
    ("sgd", "none"), ("sgd", "rules"), ("sgd", "ai"),
    ("adamw", "none"), ("adamw", "rules"), ("adamw", "ai"),
    ("lars", "none"), ("lars", "ai"),
    ("lngd", "none"), ("lngd", "ai"),
]
PHASE_OF_GHD = {"none": 1, "rules": 1, "ai": 2}
OPT_LABEL = {"sgd": "SGD", "adamw": "AdamW", "lars": "LARS", "lngd": "LNGD"}
GHD_LABEL = {"none": "", "rules": " + GHD-Rules", "ai": " + GHD-AI"}


def config_label(opt: str, ghd: str) -> str:
    return OPT_LABEL[opt] + GHD_LABEL[ghd]


def result_path(results_dir: str | Path, opt: str, ghd: str, seed: int) -> Path:
    return Path(results_dir) / f"mlp_mnist_{opt}_{ghd}_seed{seed}.json"


INIT_GAIN = 4.0  # xavier gain; 1.0 collapses the forward signal ~4x per sigmoid layer


class SigmoidMLP(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, num_layers: int = 10, num_classes: int = 10,
                 init_gain: float = INIT_GAIN) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [num_classes]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers))
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=init_gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.layers[:-1]:
            x = torch.sigmoid(layer(x))
        return self.layers[-1](x)


_DATA_CACHE: dict[str, tuple[torch.Tensor, ...]] = {}


def load_mnist(root: str = "data", device: str = "cpu") -> tuple[torch.Tensor, ...]:
    """Normalised MNIST as in-memory tensors: (x_train, y_train, x_val, y_val, x_test, y_test)."""
    if device in _DATA_CACHE:
        return _DATA_CACHE[device]
    from torchvision import datasets

    def prep(ds):
        x = (ds.data.float() / 255.0 - 0.1307) / 0.3081
        return x.unsqueeze(1), ds.targets.clone()

    x_full, y_full = prep(datasets.MNIST(root, train=True, download=True))
    x_test, y_test = prep(datasets.MNIST(root, train=False, download=True))
    perm = torch.randperm(len(x_full), generator=torch.Generator().manual_seed(SPLIT_SEED))
    tr, va = perm[:50000], perm[50000:60000]
    data = tuple(t.to(device) for t in (x_full[tr], y_full[tr], x_full[va], y_full[va], x_test, y_test))
    _DATA_CACHE[device] = data
    return data


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, chunk: int = 2500) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(x), chunk):
        correct += (model(x[i:i + chunk]).argmax(-1) == y[i:i + chunk]).sum().item()
    model.train()
    return 100.0 * correct / len(x)


def steps_to_threshold(val_acc: list[float], val_steps: list[int], thresholds=THRESHOLDS) -> dict[str, int | None]:
    out = {}
    for th in thresholds:
        out[str(th)] = next((s for a, s in zip(val_acc, val_steps) if a >= th), None)
    return out


def curve_auc(val_acc: list[float]) -> float:
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(val_acc) / len(val_acc)) if val_acc else 0.0


def loss_at_checkpoints(losses: list[float], checkpoints=LOSS_CHECKPOINTS, window: int = 10) -> dict[str, float | None]:
    """Mean training loss over the `window` steps ending at each checkpoint step."""
    out = {}
    for c in checkpoints:
        if c <= len(losses):
            out[str(c)] = float(np.mean(losses[max(0, c - window):c]))
        else:
            out[str(c)] = None
    return out


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(payload, f, allow_nan=False)
    tmp.replace(path)


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


# ---------------------------------------------------------------- learning rates
LR_SWEEP_FILE = "lr_sweep.json"
LR_SWEEP_GRID = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
LR_SWEEP_OPTS = ["sgd", "lars", "lngd"]
LR_SWEEP_STEPS = 500
LR_SWEEP_TARGET = 50.0


def pick_lr(sweep: list[dict]) -> tuple[float, bool]:
    """Best val acc at step 500 among lrs that pass 50%; if none pass, best overall. Returns (lr, passed)."""
    passing = [r for r in sweep if r["val_acc_500"] > LR_SWEEP_TARGET]
    pool = passing or sweep
    best = max(pool, key=lambda r: (r["val_acc_500"], -r["lr"]))
    return best["lr"], bool(passing)


def lr_sweep(results_dir: str | Path = "results", opts=LR_SWEEP_OPTS, seed: int = 42, verbose: bool = True) -> dict:
    chosen, runs = {}, []
    for opt in opts:
        sweep = []
        for lr in LR_SWEEP_GRID:
            r = train_run(opt, "none", seed, n_steps=LR_SWEEP_STEPS, lr=lr, keep_logs=False)
            acc = r["val_acc_curve"][-1]
            sweep.append({"optimizer": opt, "lr": lr, "val_acc_500": acc, "diverged_at_step": r["diverged_at_step"]})
            if verbose:
                print(f"  {OPT_LABEL[opt]:5s} lr={lr:<6g} val_acc@500 = {acc:5.1f}%")
        lr, passed = pick_lr(sweep)
        chosen[opt] = lr
        if verbose:
            note = "" if passed else f"  (WARNING: no lr reached {LR_SWEEP_TARGET:.0f}% by step {LR_SWEEP_STEPS})"
            print(f"  -> {OPT_LABEL[opt]} lr = {lr}{note}")
        runs += sweep
    payload = {"chosen": chosen, "grid": LR_SWEEP_GRID, "steps": LR_SWEEP_STEPS, "seed": seed,
               "target_val_acc": LR_SWEEP_TARGET, "runs": runs}
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    _write_json_atomic(Path(results_dir) / LR_SWEEP_FILE, payload)
    return payload


# ---------------------------------------------------------------- one training run
def train_run(
    opt: str,
    ghd: str,
    seed: int,
    n_steps: int | None = None,
    eval_every: int = EVAL_EVERY,
    lr: float | None = None,
    controller_path: str | Path = "results/ghd_controller.pt",
    keep_logs: bool = True,
) -> dict:
    """Train one (optimizer, ghd_mode, seed) configuration and return its result dict."""
    n_steps = N_STEPS_BY_OPT[opt] if n_steps is None else n_steps
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_mnist(device=device)
    lr = OPTIMIZER_LR[opt] if lr is None else lr
    warmup_steps = LR_WARMUP_STEPS if opt in LR_WARMUP_OPTS else 0

    model = SigmoidMLP().to(device)
    optimizer = make_optimizer(opt, model.parameters(), lr=lr)
    # Baselines get a passive (observe-only) hook so their gradient health is logged too.
    hook = GHDHook(model, mode="passive" if ghd == "none" else ghd, controller_path=controller_path,
                   log_every=LOG_EVERY, optimizer=optimizer)
    gen = torch.Generator().manual_seed(seed)

    losses: list[float] = []
    train_loss_curve: list[float | None] = []
    val_acc_curve: list[float] = []
    val_steps: list[int] = []
    order = torch.randperm(len(x_tr), generator=gen)
    cursor = 0
    diverged_at = None
    start = time.time()

    for step in range(1, n_steps + 1):
        if cursor + BATCH_SIZE > len(order):
            order = torch.randperm(len(x_tr), generator=gen)
            cursor = 0
        idx = order[cursor:cursor + BATCH_SIZE].to(device)
        cursor += BATCH_SIZE

        loss = F.cross_entropy(model(x_tr[idx]), y_tr[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        loss_val = loss.item()
        hook.pre_step(loss_val)
        if step <= warmup_steps:
            current_lr = lr / 10 + (lr - lr / 10) * ((step - 1) / warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
        elif step == warmup_steps + 1:
            for group in optimizer.param_groups:
                group["lr"] = lr
        optimizer.step()
        hook.post_step()

        losses.append(loss_val)
        if diverged_at is None and not math.isfinite(loss_val):
            diverged_at = step
        if step % LOG_EVERY == 0:
            train_loss_curve.append(_finite(loss_val))
        if step % eval_every == 0:
            val_acc_curve.append(evaluate(model, x_va, y_va))
            val_steps.append(step)

    test_acc = evaluate(model, x_te, y_te)
    elapsed = time.time() - start
    ghd_summary = hook.summary()
    if ghd == "none":
        ghd_summary["total_interventions"] = 0

    return {
        "config": {
            "model": f"SigmoidMLP(784-256x9-10, sigmoid, xavier_uniform gain={INIT_GAIN})",
            "init_gain": INIT_GAIN,
            "dataset": "MNIST (50k train / 10k val / 10k test)",
            "optimizer": opt,
            "ghd_mode": ghd,
            "seed": seed,
            "n_steps": n_steps,
            "lr": lr,
            "lr_warmup_steps": warmup_steps,
            "batch_size": BATCH_SIZE,
            "eval_every": eval_every,
            "device": device,
            "ghd_observer": "passive" if ghd == "none" else ghd,
            "intervention": "update (param delta after optimizer.step)",
            "controller": ghd_summary["controller"],
        },
        "val_acc_curve": val_acc_curve,
        "val_steps": val_steps,
        "train_loss_curve": train_loss_curve,
        "train_loss_steps": list(range(LOG_EVERY, n_steps + 1, LOG_EVERY)),
        "test_acc": test_acc,
        "elapsed_seconds": elapsed,
        "diverged_at_step": diverged_at,
        "metrics": {
            "steps_to_threshold": steps_to_threshold(val_acc_curve, val_steps),
            "auc": curve_auc(val_acc_curve),
            "loss_at_checkpoints": {k: (_finite(v) if v is not None else None)
                                    for k, v in loss_at_checkpoints(losses).items()},
            "final_val_acc": val_acc_curve[-1] if val_acc_curve else None,
            "final_test_acc": test_acc,
        },
        "ghd_summary": ghd_summary,
        "ghd_logs": hook.logs if keep_logs else [],
    }


RUNNING_DIR = ".running"


def running_marker(results_dir: str | Path, opt: str, ghd: str, seed: int) -> Path:
    return Path(results_dir) / RUNNING_DIR / result_path(results_dir, opt, ghd, seed).name


def run_one(
    opt: str,
    ghd: str,
    seed: int,
    n_steps: int | None = None,
    eval_every: int = EVAL_EVERY,
    results_dir: str | Path = "results",
    controller_path: str | Path = "results/ghd_controller.pt",
    lr: float | None = None,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Train one run and save its result JSON. Resumable: an existing JSON is loaded instead."""
    path = result_path(results_dir, opt, ghd, seed)
    if path.exists() and not force:
        if verbose:
            print(f"[skip] {path.name} exists")
        with open(path) as f:
            return json.load(f)

    marker = running_marker(results_dir, opt, ghd, seed)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    try:
        lr = OPTIMIZER_LR[opt] if lr is None else lr
        result = train_run(opt, ghd, seed, n_steps, eval_every, lr=lr, controller_path=controller_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(path, result)
    finally:
        marker.unlink(missing_ok=True)
    if verbose:
        m = result["metrics"]["steps_to_threshold"]
        print(f"[done] {config_label(opt, ghd):22s} seed {seed:<4d} lr {lr:<6g} "
              f"70%@{m['70']}  80%@{m['80']}  90%@{m['90']}  test {result['test_acc']:.2f}%  "
              f"({result['elapsed_seconds']:.0f}s)", flush=True)
    return result
