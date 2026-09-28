"""GHD-AI: the MLP controller and its trainer (distilled from GHD-Rules logs)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import MODES
from .paths import result_files


FEATURE_DIM = 13
STEP_HORIZON = 5000
# Log modes -> controller class. 'recovering' means "no correction", same as healthy.
LABEL_OF_MODE = {"healthy": 0, "recovering": 0, "vanishing": 1, "exploding": 2, "oscillating": 3, "noisy": 4}


def build_features(
    s: dict[str, float],
    persist_v: int,
    persist_o: int,
    persist_n: int,
    layer_idx: int,
    total_layers: int,
    step: int,
) -> list[float]:
    return [
        s["s1"], s["s2"], s["s3"], s["s4"], s["s5"], s["s6"], s["s7"], s["s_depth"],
        float(persist_v), float(persist_o), float(persist_n),
        layer_idx / total_layers,
        min(step / STEP_HORIZON, 1.0),
    ]


class GHDController(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 64, n_modes: int = len(MODES)) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
        )
        self.mode_head = nn.Linear(hidden, n_modes)  # softmax applied in predict(); CE takes logits
        self.strength_head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk((x - self.feat_mean) / self.feat_std)
        return self.mode_head(h), self.strength_head(h).squeeze(-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, strength = self(x)
        return F.softmax(logits, dim=-1), strength


def load_controller(path: str | Path) -> GHDController | None:
    """Returns the trained controller, or None if no usable checkpoint exists."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = GHDController()
        model.load_state_dict(ckpt["state_dict"])
    except Exception:
        return None
    model.eval()
    return model


def checkpoint_sources(path: str | Path) -> list[str] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return list(torch.load(path, map_location="cpu", weights_only=False).get("sources", []))
    except Exception:
        return None


def rules_log_files(log_dir: str | Path, prefix: str | None = None) -> list[Path]:
    """Result files from ghd_mode='rules' runs only (of one experiment spec if `prefix` is given)."""
    files = []
    for path in result_files(log_dir, prefix, ghd="rules"):
        try:
            with open(path) as f:
                if json.load(f)["config"]["ghd_mode"] == "rules":
                    files.append(path)
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return files


class ControllerTrainer:
    def __init__(self, seed: int = 0, batch_size: int = 256, lr: float = 1e-3, val_fraction: float = 0.1) -> None:
        self.seed = seed
        self.batch_size = batch_size
        self.lr = lr
        self.val_fraction = val_fraction
        self.model = GHDController()
        self.X = np.zeros((0, FEATURE_DIM), dtype=np.float32)
        self.y_mode = np.zeros(0, dtype=np.int64)
        self.y_strength = np.zeros(0, dtype=np.float32)
        self.sources: list[str] = []
        self.best_val_loss = float("inf")
        self.history: list[dict[str, float]] = []

    def load_logs(self, log_dir: str | Path, prefix: str | None = None) -> int:
        """Extract (13-d signals, mode label, strength label) per logged layer-step."""
        X, y_mode, y_strength = [], [], []
        files = rules_log_files(log_dir, prefix)
        for path in files:
            with open(path) as f:
                logs = json.load(f)["ghd_logs"]
            for entry in logs:
                names = list(entry["layers"].keys())
                for idx, name in enumerate(names):
                    rec = entry["layers"][name]
                    if rec["mode"] not in LABEL_OF_MODE:  # 'warmup'
                        continue
                    X.append(build_features(
                        rec, rec["persist_v"], rec["persist_o"], rec["persist_n"], idx, len(names), entry["step"]
                    ))
                    y_mode.append(LABEL_OF_MODE[rec["mode"]])
                    y_strength.append(rec["strength"])
        self.sources = [p.name for p in files]
        if X:
            self.X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
            self.y_mode = np.asarray(y_mode, dtype=np.int64)
            self.y_strength = np.asarray(y_strength, dtype=np.float32)
        return len(X)

    def train(self, epochs: int = 50, verbose: bool = True) -> dict[str, float]:
        if len(self.X) == 0:
            raise RuntimeError("No GHD-Rules logs loaded; run phase 1 first.")
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        perm = rng.permutation(len(self.X))
        n_val = max(1, int(len(perm) * self.val_fraction))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        X = torch.from_numpy(self.X)
        ym = torch.from_numpy(self.y_mode)
        ys = torch.from_numpy(self.y_strength)
        mean = X[train_idx].mean(0)
        std = X[train_idx].std(0).clamp_min(1e-6)
        self.model.feat_mean.copy_(mean)
        self.model.feat_std.copy_(std)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_state = None
        for epoch in range(1, epochs + 1):
            self.model.train()
            order = torch.from_numpy(rng.permutation(train_idx))
            train_loss = 0.0
            for start in range(0, len(order), self.batch_size):
                b = order[start:start + self.batch_size]
                logits, strength = self.model(X[b])
                loss = F.cross_entropy(logits, ym[b]) + 0.5 * F.mse_loss(strength, ys[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item() * len(b)
            train_loss /= len(train_idx)

            self.model.eval()
            with torch.no_grad():
                vb = torch.from_numpy(val_idx)
                logits, strength = self.model(X[vb])
                val_loss = (F.cross_entropy(logits, ym[vb]) + 0.5 * F.mse_loss(strength, ys[vb])).item()
                val_acc = (logits.argmax(-1) == ym[vb]).float().mean().item()
            self.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  val_mode_acc {val_acc:.3f}")

        self.model.load_state_dict(best_state)
        self.model.eval()
        return {"best_val_loss": self.best_val_loss, "n_train": len(train_idx), "n_val": n_val}

    def class_counts(self) -> dict[str, int]:
        counts = np.bincount(self.y_mode, minlength=len(MODES))
        return {m: int(c) for m, c in zip(MODES, counts)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "modes": MODES,
                "best_val_loss": self.best_val_loss,
                "sources": self.sources,
                "class_counts": self.class_counts(),
                "history": self.history,
            },
            path,
        )
