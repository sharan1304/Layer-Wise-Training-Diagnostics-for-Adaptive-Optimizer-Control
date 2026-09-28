# Layer-Wise Training Diagnostics for Adaptive Optimizer Control

> GHD is a plug-and-play diagnostic module that attaches to any optimizer, detects per-layer gradient failure modes using eight health signals, and applies targeted corrections. On a 7-layer sigmoid MLP, it reduces steps to 80% accuracy by 37% using only 197 vanishing corrections, while having no effect under AdamW and LARS.

---

## What This Project Does

Deep neural networks trained with sigmoid activations suffer from vanishing gradients — the gradient signal reaching early layers becomes hundreds of thousands of times smaller than at the output. Standard optimizers (SGD, AdamW, LARS, LNGD) apply the same update rule to every layer without any awareness of which layers are struggling.

**GHD (Gradient Health Diagnostic)** attaches to any existing optimizer without replacing it. At every training step it:
1. Computes 8 per-layer gradient health signals
2. Classifies each layer into a failure mode (vanishing, exploding, oscillating) — noisy detection exists but is disabled by default
3. Applies a targeted post-normalisation correction to struggling layers
4. Logs diagnostic information for visualisation

A two-head MLP controller learns to reproduce rule-based diagnoses, matching GHD-Rules convergence speed on SGD (300 vs 313 steps to 80%) — the primary evidence of controller quality. The 99.9% mode-classification accuracy reflects ~98% healthy examples in training data and should not be cited as standalone evidence.

---

## Key Results

| Experiment | Model | Dataset | Result |
|---|---|---|---|
| Exp 1 | 10-layer Sigmoid MLP | MNIST (gain 1.0) | S_depth detects vanishing spanning 6 orders of magnitude |
| Exp 2a | 7-layer Sigmoid MLP | MNIST (gain 1.5) | SGD+GHD reaches 80% in **300 steps vs 473** (37% faster) |
| Exp 2b | 7-layer Sigmoid MLP | MNIST (gain 1.0) | GHD rescues a network plain SGD cannot train (9.99% → 95.56%) |
| Exp 3 | 6-layer Sigmoid CNN (4 conv + 2 linear) | CIFAR-10 (gain 1.5) | Gain sweep complete (1000 steps, 1 seed). Full run pending. |

**Ablation:** vanishing correction alone (197 interventions) produces the full 37% speed-up.
Oscillation corrections (950 additional interventions) add nothing measurable.

**Why AdamW and LARS are unaffected:** Under AdamW, GHD's detector flagged no vanishing
steps — every layer was healthy throughout training, producing zero interventions. Under LARS,
vanishing steps were detected but produced no corrections. GHD does not improve convergence
for these optimizers in this setting.

---

## Architecture

```
Training loop
     ↓
  backward()
     ↓
┌─────────────────────────────────────┐
│  GHD Hook (per layer, per step)     │
│                                     │
│  8 Signals:                         │
│    S1  gradient norm                │
│    S2  relative norm (self-calib)   │
│    S3  inter-layer ratio            │
│    S4  direction consistency        │
│    S5  noise (CV of norms)          │
│    S6  oscillation (mean gradient   │
│        cosine, last 20 steps)       │
│    S7  stability composite          │
│    S_depth  vs output layer ★       │
│                                     │
│  Detection (priority order):        │
│    Structural Vanishing →           │
│    Exploding → Vanishing →          │
│    Oscillating → Noisy → Healthy    │
│                                     │
│  ┌─────────────┬───────────────┐    │
│  │ GHD-Rules   │  GHD-AI       │    │
│  │ fixed conf. │  MLP(13→64→64)│    │
│  │ scores      │  mode+strength│    │
│  └─────────────┴───────────────┘    │
│                                     │
│  Adapter (post-normalisation):      │
│    Vanishing  → rescale grad only   │
│    Exploding  → log-scale clip      │
│    Oscillating→ temporal blend      │
└─────────────────────────────────────┘
     ↓
  optimizer.step()
     ↓
  weights updated
```

★ S_depth is the key signal — compares each layer directly to the output layer. It can detect
vanishing present from initialisation, where S2 and S3 — which rely on recent history or adjacent
layers — may not flag the problem immediately.

---

## The 8 Diagnostic Signals

| Signal | Formula | What it measures |
|---|---|---|
| S1 | ‖g‖_F | Gradient magnitude |
| S2 | ‖g‖ / (median₂₀ + ε) | Weakness vs own history |
| S3 | q_l / q_{l+1} where q = ‖g‖/√P | Signal loss at layer boundary |
| S4 | (cos(g_t, g_{t-1}) + 1) / 2 | Direction consistency |
| S5 | clip(std₂₀ / mean₂₀, 0, 1) | Gradient noise |
| S6 | mean of cos(g_t, g_{t-1}) over last 20 steps | Sustained oscillation |
| S7 | clip(1 − S5 − max(0,−S6), 0, 1) | Stability composite |
| S_depth | q_l / q_L | Signal vs output layer |

---

## Project Structure

```
adaptive optimizer/
├── ghd/
│   ├── core.py          # signals, detection, confidence scores
│   ├── hook.py          # GHDHook — attaches to any optimizer
│   ├── controller.py    # GHDController (two-head MLP) + ControllerTrainer
│   ├── optimizers.py    # LARS (float64), LNGD, make_optimizer
│   ├── experiment.py    # SigmoidMLP, SigmoidCNN, MNIST/CIFAR-10 loaders,
│   │                    # train loop, metrics, run_one
│   ├── summary.py       # aggregates result JSONs into tables
│   └── paths.py         # result file naming per experiment spec
├── main.py              # CLI entry point (all experiments)
├── app.py               # Streamlit dashboard entry point
├── streamlit_app.py     # main page (Exp 1 + Layer-Aware Optimizer tabs)
├── exp1_dashboard.py    # Experiment 1 dashboard panels
├── ghd_results_page.py  # GHD Results page (Exp 1a / 1b tabs)
├── pages/
│   └── 1_📊_GHD_Results.py
├── test_ghd.py          # 18 GHD tests
├── test_optimizer.py    # 17 older optimizer tests
├── results/
│   ├── exp2/            # Exp 2 result JSONs (39 files: 30 Setting A,
│   │                    #   6 Setting B, 3 ablation)
│   └── archive_exp1a_gain1/  # Exp 1a results (detection evidence)
└── requirements.txt
```

---

## Results Data

Result JSON files are not tracked in this repository (they total ~150 files and are excluded by .gitignore).
To reproduce the results, run the experiments as described below. The summary CSV files for
Experiment 2 are tracked and show the aggregated numbers without re-running.

---

## How to Run

**Install:**
```bash
pip install -r requirements.txt
```

**Quick test (SGD, 500 steps, seed 42):**
```bash
python main.py --quick
```

**Experiment 2 — 7-layer MLP, MNIST, gain 1.5 (main result):**
```bash
# Phase 1: rule-based runs
python main.py --exp 2 --layers 7 --gain 1.5 --phase 1 --seeds 42 123 7 --workers 4

# Train MLP controller
python main.py --exp 2 --layers 7 --gain 1.5 --train_controller --results_dir results/exp2/

# Phase 2: AI controller runs
python main.py --exp 2 --layers 7 --gain 1.5 --phase 2 --seeds 42 123 7 --workers 4

# Vanishing-only ablation
python main.py --exp 2 --layers 7 --gain 1.5 --opt sgd --ghd_mode rules_vanish_only --seeds 42 123 7
```

**Experiment 3 — 6-layer Sigmoid CNN (4 conv + 2 linear), CIFAR-10, gain 1.5:**
```bash
python main.py --exp 3 --model cnn --dataset cifar10 --gain 1.5 --phase 1 --workers 4
```

**Dashboard:**
```bash
streamlit run app.py
```

**Run all tests:**
```bash
python -m pytest test_ghd.py test_optimizer.py -v
```

---

## Optimizers

| Optimizer | Year | Notes |
|---|---|---|
| SGD | Classic | momentum=0.9, nesterov, wd=1e-4 |
| AdamW | 2017 | betas=(0.9,0.999), wd=1e-2 |
| LARS | 2017 | layer-wise trust ratio, float64 to avoid overflow |
| LNGD | NeurIPS 2024 | diagonal Fisher + LARS trust ratio |

GHD attaches to all four. Only SGD benefits from corrections in the tested regime.

---

## Why Xavier Gain Matters

| Gain | S_depth Layer 1 | Trainable without GHD | GHD helps? |
|---|---|---|---|
| 1.0 (10-layer) | ~1.7×10⁻⁶ | No — stuck at chance | No — also stuck at chance (Exp 1a) |
| 1.0 (7-layer) | ~1.1×10⁻⁴ | No — stuck at chance | Rescues the network (9.99% → 95.56%) |
| 1.5 (7-layer) | ~0.001 | Yes — slowly | 37% faster convergence |
| 4.0 (10-layer) | ~0.057 | Yes — fast | No improvement needed |

GHD's 1000× correction cap closes gaps of up to three orders of magnitude. Exp 2b shows GHD
rescuing a network with S_depth ~1×10⁻⁴, outside the typical detection threshold.

---

## Citation

```bibtex
@article{sharan2026ghd,
  title   = {Layer-Wise Training Diagnostics for Adaptive Optimizer Control},
  author  = {Sharan V},
  journal = {Under review},
  year    = {2026},
  note    = {Rajalakshmi Engineering College, Chennai}
}
```

---

## Related Work

- **Paper 1 (base):** A Layer-Aware AI-Guided Adaptive Optimizer for Vanishing-Gradient Mitigation — Sharan V, Rajalakshmi Engineering College
- **LARS:** You et al. (2017) arXiv:1708.03888
- **LNGD:** Liu et al. NeurIPS 2024
- **AdamW:** Loshchilov & Hutter, ICLR 2019

---

*M.Tech Data Science — Rajalakshmi Engineering College, Chennai*
*Contact: sharan.ai.613@gmail.com*

