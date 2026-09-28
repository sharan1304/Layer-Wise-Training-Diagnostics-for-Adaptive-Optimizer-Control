# Layer-Wise Training Diagnostics for Adaptive Optimizer Control

> GHD is a plug-and-play diagnostic module that attaches to any optimizer, detects per-layer gradient failure modes using eight health signals, and applies targeted corrections. On a 7-layer sigmoid MLP, it reduces steps to 80% accuracy by 37% using only 197 vanishing corrections, while leaving AdamW and LARS unchanged by design.

---

## What This Project Does

Deep neural networks trained with sigmoid activations suffer from vanishing gradients — the gradient signal reaching early layers becomes millions of times smaller than at the output. Standard optimizers (SGD, AdamW, LARS, LNGD) apply the same update rule to every layer without any awareness of which layers are struggling.

**GHD (Gradient Health Diagnostic)** attaches to any existing optimizer without replacing it. At every training step it:
1. Computes 8 per-layer gradient health signals
2. Classifies each layer into a failure mode (vanishing, exploding, oscillating, noisy)
3. Applies a targeted post-normalisation correction to struggling layers
4. Logs diagnostic information for visualisation

A two-head MLP controller learns to reproduce rule-based diagnoses, achieving 99.9% mode-classification accuracy.

---

## Key Results

| Experiment | Model | Dataset | Result |
|---|---|---|---|
| Exp 1 | 10-layer Sigmoid MLP | MNIST (gain 1.0) | S_depth detects vanishing spanning 7 orders of magnitude |
| Exp 2a | 7-layer Sigmoid MLP | MNIST (gain 1.5) | SGD+GHD reaches 80% in **300 steps vs 473** (37% faster) |
| Exp 2b | 7-layer Sigmoid MLP | MNIST (gain 1.0) | GHD rescues a network plain SGD cannot train (9.99% → 95.56%) |
| Exp 3 | 4-layer Sigmoid CNN | CIFAR-10 (gain 1.5) | Architecture transfer — same signals, no redesign |

**Ablation:** vanishing correction alone (197 interventions) produces the full 37% speed-up.
Oscillation corrections (950 additional interventions) add nothing measurable.

**Why AdamW and LARS are unaffected:** their built-in magnitude normalisation
cancels GHD's vanishing correction. GHD correctly makes zero interventions on these
optimizers — this is the expected analytical result, not a failure.

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
│    S6  oscillation (update cosine)  │
│    S7  stability composite          │
│    S_depth  vs output layer ★       │
│                                     │
│  Detection (priority order):        │
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

★ S_depth is the key signal — compares each layer to the output layer and detects
structural vanishing from the very first diagnosed step. S2 and S3 cannot do this.

---

## The 8 Diagnostic Signals

| Signal | Formula | What it measures |
|---|---|---|
| S1 | ‖g‖_F | Gradient magnitude |
| S2 | ‖g‖ / (median₂₀ + ε) | Weakness vs own history |
| S3 | q_l / q_{l+1} where q = ‖g‖/√P | Signal loss at layer boundary |
| S4 | (cos(g_t, g_{t-1}) + 1) / 2 | Direction consistency |
| S5 | clip(std₂₀ / mean₂₀, 0, 1) | Gradient noise |
| S6 | mean of update cosines (20 steps) | Sustained oscillation |
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
│   ├── exp2/            # Exp 2 result JSONs (30 runs)
│   └── archive_exp1a_gain1/  # Exp 1a results (detection evidence)
└── requirements.txt
```

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
python main.py --exp 2 --layers 7 --gain 1.5 --train_controller

# Phase 2: AI controller runs
python main.py --exp 2 --layers 7 --gain 1.5 --phase 2 --seeds 42 123 7 --workers 4

# Vanishing-only ablation
python main.py --exp 2 --layers 7 --gain 1.5 --opt sgd --ghd_mode rules_vanish_only --seeds 42 123 7
```

**Experiment 3 — 4-layer CNN, CIFAR-10, gain 1.5:**
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
| 1.0 (10-layer) | ~10⁻⁷ | No — stuck at chance | Rescues the network |
| 1.5 (7-layer) | ~0.001 | Yes — slowly | 37% faster convergence |
| 4.0 (10-layer) | ~0.002 | Yes — fast | No improvement needed |

The correctable range is S_depth ∈ [0.001, 0.01]. GHD's 1000× correction cap
closes gaps up to three orders of magnitude.

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
