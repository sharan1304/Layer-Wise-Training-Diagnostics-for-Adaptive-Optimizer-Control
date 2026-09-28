"""Result file naming, shared by the runner, the summary and the controller trainer.

Experiment 1 keeps its original names (``mlp_mnist_{opt}_{ghd}_seed{n}.json``) so its
results stay resumable; later experiments encode the model spec in the prefix:
``{model}_{dataset}_l{layers}_g{gain}_{opt}_{ghd}_seed{n}.json``.
"""

from __future__ import annotations

import re
from pathlib import Path


EXP1_PREFIX = "mlp_mnist"
_OPTS = "sgd|adamw|lars|lngd"
_GHDS = "none|rules_vanish_only|rules|ai"


def run_prefix(model: str, dataset: str, layers: int, gain: float) -> str:
    return f"{model}_{dataset}_l{layers}_g{gain:g}"


def result_path(results_dir: str | Path, opt: str, ghd: str, seed: int, prefix: str = EXP1_PREFIX) -> Path:
    return Path(results_dir) / f"{prefix}_{opt}_{ghd}_seed{seed}.json"


def result_files(results_dir: str | Path, prefix: str | None = None, ghd: str | None = None) -> list[Path]:
    """Result JSONs in `results_dir`, optionally only those with exactly `prefix` and/or `ghd` mode.

    Matching the whole name keeps 'mlp_mnist' from also picking up 'mlp_mnist_l7_g2.5_...'.
    """
    head = re.escape(prefix) if prefix is not None else ".+"
    pattern = re.compile(rf"^{head}_({_OPTS})_({re.escape(ghd) if ghd else _GHDS})_seed\d+\.json$")
    return sorted(p for p in Path(results_dir).glob("*_seed*.json") if pattern.match(p.name))
