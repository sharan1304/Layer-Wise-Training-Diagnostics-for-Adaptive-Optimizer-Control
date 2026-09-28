"""Experiment metrics for the adaptive optimizer project."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


EPS = 1e-12


def layer_mean_gradients(grad_norm_history: list[list[float]]) -> list[float]:
    """Mean gradient norm for each layer across all recorded steps."""
    if not grad_norm_history:
        return []
    width = min(len(row) for row in grad_norm_history if row)
    if width == 0:
        return []
    arr = np.asarray([row[:width] for row in grad_norm_history if len(row) >= width], dtype=float)
    return list(np.mean(arr, axis=0))


def late_to_early_gradient_ratio(grad_norms: list[float], n_layers: int = 5) -> float:
    """How much larger late-layer gradients are than early-layer gradients.

    Lower is better for gradient flow. A very high value means early layers are
    receiving far weaker gradients than late layers.
    """
    if len(grad_norms) < 2:
        return float("nan")
    n = min(n_layers, max(len(grad_norms) // 3, 1))
    early = max(float(np.mean(grad_norms[:n])), EPS)
    late = max(float(np.mean(grad_norms[-n:])), EPS)
    return late / early


def log10_vanishing_ratio(grad_norms: list[float], n_layers: int = 5) -> float:
    ratio = late_to_early_gradient_ratio(grad_norms, n_layers=n_layers)
    if not np.isfinite(ratio) or ratio <= 0:
        return float("nan")
    return float(np.log10(ratio))


def early_gradient_mean(grad_norm_history: list[list[float]], n_layers: int = 5) -> float:
    values = [
        float(np.mean(row[:n_layers]))
        for row in grad_norm_history
        if len(row) >= n_layers
    ]
    return float(np.mean(values)) if values else float("nan")


def gradient_stability(grad_norm_history: list[list[float]], n_layers: int = 5) -> float:
    """1 - coefficient of variation for early-layer gradient norms."""
    values = [
        float(np.mean(row[:n_layers]))
        for row in grad_norm_history
        if len(row) >= n_layers
    ]
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    score = 1.0 - float(np.std(arr) / (np.mean(arr) + EPS))
    return max(-1.0, min(1.0, score))


def convergence_step(loss_history: list[float], relative_improvement: float = 0.05) -> int:
    """Step where loss first improves by a relative percentage from start."""
    if not loss_history:
        return 0
    target = loss_history[0] * (1.0 - relative_improvement)
    for i, loss in enumerate(loss_history, start=1):
        if loss <= target:
            return i
    return len(loss_history)


def trust_region_activations(layer_log: list[dict[str, Any]]) -> int:
    return int(sum(1 for entry in layer_log if entry.get("trust_clipped", False)))


def trust_clip_utilization(layer_log: list[dict[str, Any]]) -> tuple[float, float]:
    ratios = []
    for entry in layer_log:
        pre = entry.get("pre_clip_update_norm")
        radius = entry.get("effective_trust", entry.get("trust"))
        try:
            pre_f = float(pre)
            radius_f = float(radius)
        except (TypeError, ValueError):
            continue
        if radius_f > EPS and np.isfinite(pre_f) and np.isfinite(radius_f):
            ratios.append(pre_f / radius_f)
    if not ratios:
        return 0.0, 0.0
    arr = np.asarray(ratios, dtype=float)
    return float(np.max(arr)), float(np.mean(arr))


def starvation_events(layer_log: list[dict[str, Any]]) -> int:
    return int(sum(1 for entry in layer_log if entry.get("starved", False)))


def starvation_recoveries(layer_log: list[dict[str, Any]]) -> int:
    return int(sum(1 for entry in layer_log if entry.get("recovered", False)))


def recovery_success_rate(layer_log: list[dict[str, Any]]) -> float:
    events = starvation_events(layer_log)
    if events == 0:
        return 0.0
    return starvation_recoveries(layer_log) / events


def starvation_activation_rate(layer_log: list[dict[str, Any]]) -> float:
    if not layer_log:
        return 0.0
    return starvation_events(layer_log) / len(layer_log)


def _safe_gain(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < EPS:
        return float("nan")
    return numerator / denominator


def full_report(results: dict[str, dict], ai_layer_log: list[dict[str, Any]] | None = None) -> dict[str, dict]:
    report = {}
    adam_history = results.get("Adam", {}).get("grad_norms", [])
    sgd_history = results.get("SGD", {}).get("grad_norms", [])
    adam_early = early_gradient_mean(adam_history) if adam_history else float("nan")
    sgd_early = early_gradient_mean(sgd_history) if sgd_history else float("nan")
    adam_ratio = late_to_early_gradient_ratio(layer_mean_gradients(adam_history)) if adam_history else float("nan")
    sgd_ratio = late_to_early_gradient_ratio(layer_mean_gradients(sgd_history)) if sgd_history else float("nan")
    for name, data in results.items():
        grad_history = data.get("grad_norms", [])
        mean_by_layer = layer_mean_gradients(grad_history)
        row_layer_log = data.get("layer_log", ai_layer_log or [])
        clip_util_max, clip_util_mean = trust_clip_utilization(row_layer_log)
        report[name] = {
            "final_loss": data["loss"][-1],
            "final_accuracy": data["accuracy"][-1],
            "early_grad_mean": early_gradient_mean(grad_history),
            "vanishing_ratio": late_to_early_gradient_ratio(mean_by_layer),
            "log10_vanishing_ratio": log10_vanishing_ratio(mean_by_layer),
            "gradient_stability": gradient_stability(grad_history),
            "convergence_step": convergence_step(data["loss"]),
            "early_gain_vs_adam": _safe_gain(early_gradient_mean(grad_history), adam_early),
            "early_gain_vs_sgd": _safe_gain(early_gradient_mean(grad_history), sgd_early),
            "vanishing_gain_vs_adam": _safe_gain(adam_ratio, late_to_early_gradient_ratio(mean_by_layer)),
            "vanishing_gain_vs_sgd": _safe_gain(sgd_ratio, late_to_early_gradient_ratio(mean_by_layer)),
            "trust_activations": trust_region_activations(row_layer_log)
            if name.startswith("AI")
            else "N/A",
            "trust_clip_utilization_max": clip_util_max
            if name.startswith("AI")
            else "N/A",
            "trust_clip_utilization_mean": clip_util_mean
            if name.startswith("AI")
            else "N/A",
            "starvation_events": starvation_events(row_layer_log)
            if name.startswith("AI")
            else "N/A",
            "starvation_recoveries": starvation_recoveries(row_layer_log)
            if name.startswith("AI")
            else "N/A",
            "starvation_activation_rate": starvation_activation_rate(row_layer_log)
            if name.startswith("AI")
            else "N/A",
            "recovery_success_rate": recovery_success_rate(row_layer_log)
            if name.startswith("AI")
            else "N/A",
        }
    return report


def format_float(value: float, precision: int = 4) -> str:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return f"{value:.{precision}f}"
