"""Load Experiment 1 result JSONs and aggregate them (mean +/- std across seeds)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import CONFIGS, config_label


TABLE_THRESHOLDS = ["70", "80", "90"]


def load_results(results_dir: str | Path = "results", with_logs: bool = False) -> list[dict]:
    out = []
    for path in sorted(Path(results_dir).glob("mlp_mnist_*_seed*.json")):
        try:
            with open(path) as f:
                r = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not with_logs:
            r.pop("ghd_logs", None)
        r["_file"] = path.name
        out.append(r)
    return out


def aggregate(results: list[dict]) -> pd.DataFrame:
    """One row per (optimizer, ghd_mode) with mean/std over seeds."""
    rows = []
    for opt, ghd in CONFIGS:
        runs = [r for r in results if r["config"]["optimizer"] == opt and r["config"]["ghd_mode"] == ghd]
        if not runs:
            continue
        row = {"config": config_label(opt, ghd), "optimizer": opt, "ghd_mode": ghd, "n_seeds": len(runs),
               "seeds": sorted(r["config"]["seed"] for r in runs)}
        for th in TABLE_THRESHOLDS:
            vals = [r["metrics"]["steps_to_threshold"].get(th) for r in runs]
            hit = [v for v in vals if v is not None]
            row[f"s{th}_mean"] = float(np.mean(hit)) if hit else np.nan
            row[f"s{th}_std"] = float(np.std(hit)) if hit else np.nan
            row[f"s{th}_reached"] = len(hit)
        for key, getter in [
            ("auc", lambda r: r["metrics"]["auc"]),
            ("final_val", lambda r: r["metrics"]["final_val_acc"]),
            ("test", lambda r: r["metrics"]["final_test_acc"]),
            ("interventions", lambda r: (r.get("ghd_summary") or {}).get("total_interventions", 0)),
            ("elapsed", lambda r: r["elapsed_seconds"]),
        ]:
            vals = np.asarray([getter(r) for r in runs], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(vals))
            row[f"{key}_std"] = float(np.nanstd(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_steps(row: pd.Series, th: str) -> str:
    reached, n = int(row[f"s{th}_reached"]), int(row["n_seeds"])
    if reached == 0:
        return "not reached"
    txt = f"{row[f's{th}_mean']:.0f} ± {row[f's{th}_std']:.0f}"
    return txt if reached == n else f"{txt} ({reached}/{n})"


def best_indices(df: pd.DataFrame) -> dict[str, int | None]:
    """Index of the best row per column. Steps: lowest mean among configs that reached it on every seed."""
    best: dict[str, int | None] = {}
    for th in TABLE_THRESHOLDS:
        full = df[df[f"s{th}_reached"] == df["n_seeds"]]
        best[th] = int(full[f"s{th}_mean"].idxmin()) if not full.empty else None
    best["auc"] = int(df["auc_mean"].idxmax()) if not df.empty else None
    best["final"] = int(df["final_val_mean"].idxmax()) if not df.empty else None
    return best


def table_rows(df: pd.DataFrame, bold: str = "**") -> list[list[str]]:
    best = best_indices(df)
    rows = []
    for idx, row in df.iterrows():
        cells = [row["config"]]
        for th in TABLE_THRESHOLDS:
            s = fmt_steps(row, th)
            cells.append(f"{bold}{s}{bold}" if best[th] == idx else s)
        for key, col in [("auc", "auc"), ("final", "final_val")]:
            s = f"{row[f'{col}_mean']:.2f} ± {row[f'{col}_std']:.2f}"
            cells.append(f"{bold}{s}{bold}" if best[key] == idx else s)
        rows.append(cells)
    return rows


TABLE_HEADER = ["Optimizer", "Steps→70%", "Steps→80%", "Steps→90%", "AUC", "Final Acc (%)"]


def print_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("No Experiment 1 results found.")
        return
    header = TABLE_HEADER + ["Interventions"]
    rows = [r + [f"{row['interventions_mean']:.0f} ± {row['interventions_std']:.0f}"]
            for r, (_, row) in zip(table_rows(df, bold="*"), df.iterrows())]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    print("* = best in column. Steps are mean ± std over seeds that reached the threshold; (k/n) = seeds that reached it.")
