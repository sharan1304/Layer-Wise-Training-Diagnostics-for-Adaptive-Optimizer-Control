# AI Optimizer Algorithm Documentation

## 1. Project Overview

This project proposes a **layer-aware AI-guided adaptive optimizer** for deep neural networks.

The optimizer is designed to detect unhealthy gradient behavior, especially vanishing gradients, and respond with safe adaptive updates.

The key idea is:

```text
Do not apply the same update rule to every layer.
Analyze each layer, remember its history, generate candidate updates, fuse them, bound the result, then learn from the loss response.
```

The optimizer is built as a closed feedback loop:

```text
Backpropagation
       |
       v
Gradient State Analyzer
       |
       v
Layer State Memory
       |
       v
Adaptive Control Policy
       |
       v
Candidate Update Generator
       |
       v
Update Fusion Engine
       |
       v
Trust Region Controller
       |
       v
Parameter Update
       |
       v
Loss Response Monitor
       |
       v
Layer State Memory
       ^
       |
       +---- feedback loop
```

---

## 2. Problem: Vanishing Gradient

In deep neural networks, gradients can shrink as they propagate backward.

This causes early layers to receive extremely small updates:

```text
||g_early|| << ||g_late||
```

When this happens:

```text
early layers learn slowly
feature extraction remains weak
loss may plateau
standard optimizers may continue updating later layers only
```

Adam, AdamW, RMSProp, and SGD with Momentum use fixed mathematical update rules. They adapt to gradient scale, but they do not explicitly classify layer states such as:

```text
starved
noisy
oscillating
exploding
stable
```

This optimizer adds explicit gradient-state analysis and a layer-wise control policy.

---

## 3. Standard Adam Update

Adam uses moving averages:

```text
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
theta_t = theta_{t-1} - alpha * m_t / (sqrt(v_t) + epsilon)
```

Line by line:

```text
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
```

This stores momentum, or the recent average gradient direction.

```text
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
```

This stores recent squared gradient magnitude.

```text
theta_t = theta_{t-1} - alpha * m_t / (sqrt(v_t) + epsilon)
```

This updates the parameter using the momentum scaled by the estimated gradient magnitude.

Adam is strong, but it does not explicitly ask whether a layer is starved, noisy, oscillating, or responding well to updates.

---

## 4. Main Optimizer Architecture

### 4.1 Backpropagation

Backpropagation computes gradients for every layer.

For layer `l` at step `t`:

```text
g_l,t = gradient of layer l at step t
```

These gradients are sent to the Gradient State Analyzer.

---

### 4.2 Gradient State Analyzer

The Gradient State Analyzer computes diagnostic signals:

```text
gradient_norm_l,t
relative_gradient_norm_l,t
gradient_ratio_l,t
gradient_consistency_l,t
gradient_noise_l,t
oscillation_score_l,t
explosion_score_l,t
stability_score_l,t
starvation_score_l,t
```

It contains:

```text
Starvation Detector
Noise Detector
Oscillation Detector
Explosion Detector
Stability Detector
```

The analyzer does not update parameters. It describes the condition of each layer.

---

### 4.3 Layer State Memory

Layer State Memory stores recent history for each layer:

```text
gradient_history_l
update_history_l
loss_response_history_l
success_rate_l
amplification_state_l
trust_radius_l
```

Recommended history windows:

```text
gradient_history_window = 20 steps
update_history_window   = 20 steps
loss_plateau_window     = 50 steps
success_ema_decay       = 0.9
```

The optimizer should store summary statistics rather than full gradients whenever possible to reduce memory cost.

Recommended stored summaries:

```text
gradient norm
update norm
cosine similarity
noise estimate
loss response
success flag
```

---

### 4.4 Adaptive Control Policy

The Adaptive Control Policy decides how the layer should be updated.

It produces:

```text
alpha_l,t          = layer-wise learning rate
conf_l,t           = confidence score
amplify_l,t        = starvation amplification factor
tau_l,t            = trust radius
w_l,t              = update fusion weights
```

The project uses two implementation phases:

```text
Phase 1: Rule-based controller
Phase 2: Shared MLP controller
```

The rule-based controller makes the optimizer testable before training the MLP.

The MLP later replaces the rule formulas as a drop-in controller.

---

### 4.5 Candidate Update Generator

The optimizer generates several candidate updates:

```text
Delta_AdamW_l,t
Delta_RMSProp_l,t
Delta_SGD_Momentum_l,t
Delta_Normalized_l,t
```

Each candidate represents a different update strategy.

AdamW is useful for general adaptive training.

RMSProp is useful when gradient magnitudes vary strongly.

SGD with Momentum is useful when gradients are stable.

The normalized-gradient candidate is useful for starved layers:

```text
Delta_Normalized_l,t = -alpha_l,t * normalize(g_l,t) * conf_l,t * amplify_l,t
```

---

### 4.6 Update Fusion Engine

The Update Fusion Engine combines candidate updates:

```text
Delta_fused_l,t =
    w1_l,t * Delta_AdamW_l,t
  + w2_l,t * Delta_RMSProp_l,t
  + w3_l,t * Delta_SGD_Momentum_l,t
  + w4_l,t * Delta_Normalized_l,t
```

The weights satisfy:

```text
w1_l,t + w2_l,t + w3_l,t + w4_l,t = 1
w_i_l,t >= 0
```

This avoids unstable hard switching between optimizers.

---

### 4.7 Trust Region Controller

The Trust Region Controller bounds the update norm:

```text
if ||Delta_fused_l,t|| > tau_l,t:
    Delta_final_l,t = tau_l,t * Delta_fused_l,t / ||Delta_fused_l,t||
else:
    Delta_final_l,t = Delta_fused_l,t
```

This preserves update direction while limiting update size.

---

### 4.8 Parameter Update

The final bounded update is applied:

```text
theta_l,t = theta_l,t-1 + Delta_final_l,t
```

The update is added because `Delta_final_l,t` already includes the negative gradient direction.

---

### 4.9 Loss Response Monitor

After the update, the optimizer records whether training improved:

```text
loss_change_t = loss_t - loss_t-1
success_t = 1 if loss_t < loss_t-1 else 0
```

The result is fed back into Layer State Memory.

This feedback loop allows the optimizer to adapt over time.

---

## 5. Warm-Up and Blend Schedule

The optimizer must handle cold start. At step `t = 0`, there is no useful history.

The optimizer therefore uses three stages.

### Stage 1: Warm-Up

For the first `W` steps:

```text
if t < W:
    use pure AdamW
    collect gradient history
    collect update history
    collect loss history
    do not activate starvation correction
```

Recommended:

```text
W = 50 steps
```

This gives the optimizer enough history to estimate gradient norms, noise, consistency, and loss plateau.

---

### Stage 2: Blend Phase

After warm-up, gradually blend AdamW with the adaptive optimizer:

```text
if W <= t < W + blend_steps:
    blend = (t - W) / blend_steps
    Delta_l,t = (1 - blend) * Delta_AdamW_l,t + blend * Delta_AI_l,t
```

Recommended:

```text
blend_steps = 50
```

This prevents sudden behavior changes.

---

### Stage 3: Full Adaptive Optimizer

After warm-up and blending:

```text
if t >= W + blend_steps:
    use full AI optimizer
```

---

## 6. Starvation Detection

The optimizer should not use a fixed absolute threshold for every layer.

Instead, starvation is detected relative to each layer's own historical gradient scale.

### 6.1 Gradient Norm

```text
grad_norm_l,t = ||g_l,t||
```

This is the current gradient size for layer `l`.

---

### 6.2 Relative Gradient Threshold

Define the layer's historical gradient scale:

```text
grad_median_l,t = median(||g_l,1||, ||g_l,2||, ..., ||g_l,t||)
```

Then define relative starvation:

```text
relative_starved_l,t =
    ||g_l,t|| < epsilon_rel * grad_median_l,t
```

Recommended:

```text
epsilon_rel = 0.1
```

This means the current gradient is less than 10 percent of the layer's typical gradient magnitude.

---

### 6.3 Loss Plateau

The loss is considered plateaued when:

```text
loss_improvement_over_window < min_loss_improvement
```

Recommended:

```text
loss_plateau_window = 50
min_loss_improvement = 0.001
```

---

### 6.4 Update Effect

The update effect is low when recent updates do not improve the loss:

```text
update_effect_l,t = EMA(success_l,t, decay = 0.9)
```

where:

```text
success_l,t = 1 if loss_t < loss_t-1 else 0
success_l,t = 0 otherwise
```

Low update effect:

```text
update_effect_l,t < success_threshold
```

Recommended:

```text
success_threshold = 0.4
```

---

### 6.5 Inter-Layer Gradient Ratio

Vanishing gradients are a propagation problem, so the optimizer also compares neighboring layers:

```text
gradient_ratio_l,t = ||g_l,t|| / (||g_l+1,t|| + epsilon_num)
```

If this ratio is very small, layer `l` is receiving much weaker gradients than the next deeper layer.

Recommended signal:

```text
inter_layer_starved_l,t = gradient_ratio_l,t < ratio_threshold
```

Recommended:

```text
ratio_threshold = 0.2
```

---

### 6.6 Final Starvation Condition

The layer is considered starved when:

```text
starved_l,t =
    relative_starved_l,t
    AND plateau(loss, window = 50)
    AND update_effect_l,t < success_threshold
    AND inter_layer_starved_l,t
```

This avoids treating every tiny gradient as a vanishing-gradient problem.

---

## 7. Rule-Based Control Policy

The rule-based controller is the first implementation of `f_AI`.

It computes:

```text
alpha_l,t
conf_l,t
amplify_l,t
tau_l,t
w_l,t
```

using explicit formulas.

---

### 7.1 Adaptive Learning Rate

```text
alpha_l,t = alpha_base * depth_factor_l * starvation_factor_l * stability_factor_l
```

Line by line:

```text
alpha_base
```

The base learning rate.

```text
depth_factor_l
```

Adjusts learning rate based on layer depth.

Example:

```text
depth_factor_l = 1 + depth_boost * (1 - depth_l)
```

where:

```text
depth_l = layer_index / total_layers
```

Early layers have smaller `depth_l`, so they can receive a slightly higher factor.

Recommended:

```text
depth_boost = 0.5
```

```text
starvation_factor_l
```

Increases learning rate for starved layers:

```text
starvation_factor_l = 2.0 if starved_l,t else 1.0
```

```text
stability_factor_l
```

Reduces learning rate when noise or oscillation is high:

```text
stability_factor_l = clip(1 - 0.5 * noise_l,t - 0.5 * oscillation_l,t, 0.25, 1.0)
```

---

### 7.2 Confidence Formula

Confidence must be bounded.

Use:

```text
conf_l,t = sigmoid(
    0.25 * consistency_l,t
  + 0.25 * (1 - normalized_noise_l,t)
  + 0.25 * success_rate_l,t
  - 0.25 * oscillation_score_l,t
)
```

The output is:

```text
0 <= conf_l,t <= 1
```

Meanings:

```text
high consistency      -> higher confidence
low noise             -> higher confidence
high success rate     -> higher confidence
high oscillation      -> lower confidence
```

---

### 7.3 Amplification Formula

Amplification is used only for starved layers.

Base formula:

```text
amp_raw_l,t = 1 / (relative_grad_norm_l,t + epsilon_num)
```

where:

```text
relative_grad_norm_l,t = ||g_l,t|| / (grad_median_l,t + epsilon_num)
```

Bounded amplification:

```text
amp_candidate_l,t = clip(amp_raw_l,t, 1.0, max_amp)
```

Recommended:

```text
max_amp = 5.0
```

Final amplification:

```text
if starved_l,t:
    amplify_l,t = amp_candidate_l,t
else:
    amplify_l,t = 1.0
```

---

### 7.4 Amplification Decay Schedule

Amplification should not keep increasing if it does not help.

Use:

```text
if starved_l,t AND loss improved since last amplification:
    amplify_state_l,t = min(amplify_state_l,t-1 * growth_rate, max_amp)

elif starved_l,t AND loss did not improve:
    amplify_state_l,t = max(amplify_state_l,t-1 * decay_rate, 1.0)

else:
    amplify_state_l,t = 1.0
```

Recommended:

```text
growth_rate = 1.2
decay_rate  = 0.8
max_amp     = 5.0
```

Final amplification can combine the candidate and state:

```text
amplify_l,t = min(amp_candidate_l,t, amplify_state_l,t)
```

This prevents endless amplification when the layer does not respond.

---

### 7.5 Trust Radius Update Rule

The trust radius controls the maximum update norm.

Initial value:

```text
tau_l,0 = 0.1
```

Bounds:

```text
tau_min = 0.001
tau_max = 1.0
```

Update rule:

```text
if loss improved after last update:
    tau_l,t = min(tau_l,t-1 * 1.2, tau_max)

elif loss worsened after last update:
    tau_l,t = max(tau_l,t-1 * 0.5, tau_min)

else:
    tau_l,t = tau_l,t-1
```

Trust grows slowly and shrinks quickly.

This is intentional because unsafe behavior should be penalized more strongly than successful behavior is rewarded.

---

### 7.6 Rule-Based Fusion Weights

The fusion weights choose how much each update candidate contributes.

Weights:

```text
w_l,t = [w_AdamW, w_RMSProp, w_SGD, w_Normalized]
```

Rules:

```text
if starved_l,t:
    w_l,t = [0.20, 0.20, 0.10, 0.50]

elif noisy_l,t:
    w_l,t = [0.45, 0.35, 0.15, 0.05]

elif oscillating_l,t:
    w_l,t = [0.50, 0.30, 0.15, 0.05]

elif stable_l,t:
    w_l,t = [0.25, 0.10, 0.60, 0.05]

else:
    w_l,t = [0.50, 0.25, 0.20, 0.05]
```

These values are starting points and should be tuned experimentally.

---

## 8. MLP Control Policy

The MLP replaces the rule-based formulas after the baseline optimizer is working.

Use one shared MLP for all layers:

```text
z_l,t = MLP(x_l,t; phi)
```

where:

```text
phi = shared MLP parameters
x_l,t = normalized feature vector for layer l at step t
```

The MLP is shared across layers to reduce parameter count and improve generalization.

Layer identity is provided through normalized depth:

```text
depth_l = layer_index / total_layers
```

---

### 8.1 MLP Feature Vector

Recommended feature vector:

```text
x_l,t = [
    depth_l,
    normalized_step_t,
    log_grad_norm_l,t,
    log_update_norm_l,t-1,
    relative_grad_norm_l,t,
    gradient_ratio_l,t,
    consistency_l,t,
    normalized_noise_l,t,
    oscillation_score_l,t,
    success_rate_l,t,
    plateau_score_t,
    starved_l,t,
    exploding_l,t,
    stable_l,t
]
```

---

### 8.2 Feature Normalization

Raw gradient values can span many orders of magnitude, so inputs must be normalized.

Use:

```text
log_grad_norm_l,t = log(||g_l,t|| + epsilon_num)
```

```text
log_update_norm_l,t-1 = log(||Delta theta_l,t-1|| + epsilon_num)
```

```text
depth_l = layer_index / total_layers
```

```text
normalized_step_t = min(t / max_training_steps, 1.0)
```

Values already in `[0, 1]` can be used directly:

```text
success_rate_l,t
normalized_noise_l,t
plateau_score_t
starved_l,t
exploding_l,t
stable_l,t
```

Cosine similarity should be mapped from `[-1, 1]` to `[0, 1]`:

```text
consistency_l,t = (cosine(g_l,t, g_l,t-1) + 1) / 2
```

---

### 8.3 MLP Outputs

The MLP outputs:

```text
z_l,t = [
    z_alpha,
    z_conf,
    z_amp,
    z_tau,
    z_w1,
    z_w2,
    z_w3,
    z_w4
]
```

Convert raw outputs into bounded control values:

```text
alpha_l,t = alpha_base * alpha_max_scale * sigmoid(z_alpha)
```

```text
conf_l,t = sigmoid(z_conf)
```

```text
amplify_l,t = 1 + (max_amp - 1) * sigmoid(z_amp)
```

```text
tau_l,t = tau_min + (tau_max - tau_min) * sigmoid(z_tau)
```

```text
[w1, w2, w3, w4] = softmax([z_w1, z_w2, z_w3, z_w4])
```

This ensures:

```text
0 <= conf_l,t <= 1
1 <= amplify_l,t <= max_amp
tau_min <= tau_l,t <= tau_max
w1 + w2 + w3 + w4 = 1
```

---

## 9. Full Algorithm

For each training step `t`:

```text
1. Run forward pass.

2. Compute loss_t.

3. Run backpropagation to get gradients g_l,t for each layer.

4. If t < W:
       use AdamW update
       store gradient, update, and loss statistics
       continue

5. Compute gradient-state features for each layer:
       gradient norm
       relative gradient norm
       inter-layer gradient ratio
       noise
       consistency
       oscillation
       stability
       starvation

6. For each layer l:
       compute alpha_l,t
       compute conf_l,t
       compute amplify_l,t
       compute tau_l,t
       compute fusion weights w_l,t

7. Generate candidate updates:
       Delta_AdamW_l,t
       Delta_RMSProp_l,t
       Delta_SGD_Momentum_l,t
       Delta_Normalized_l,t

8. Fuse candidate updates:
       Delta_fused_l,t =
           w1 * Delta_AdamW_l,t
         + w2 * Delta_RMSProp_l,t
         + w3 * Delta_SGD_Momentum_l,t
         + w4 * Delta_Normalized_l,t

9. If in blend phase:
       blend = (t - W) / blend_steps
       Delta_fused_l,t =
           (1 - blend) * Delta_AdamW_l,t
         + blend * Delta_fused_l,t

10. Apply trust region:
       if ||Delta_fused_l,t|| > tau_l,t:
           Delta_final_l,t =
               tau_l,t * Delta_fused_l,t / ||Delta_fused_l,t||
       else:
           Delta_final_l,t = Delta_fused_l,t

11. Apply parameter update:
       theta_l,t = theta_l,t-1 + Delta_final_l,t

12. Monitor next loss response.

13. Update Layer State Memory:
       gradient history
       update history
       success rate
       trust radius
       amplification state
```

---

## 10. Stability Safety Check

A bad adaptive update can destabilize training.

The optimizer should monitor sudden loss jumps:

```text
if loss_t > loss_t-1 * instability_threshold:
    mark previous adaptive update as failed
    shrink tau_l,t aggressively
    reduce confidence
    use AdamW on the next step
```

Recommended:

```text
instability_threshold = 1.5
```

Meaning:

```text
if loss jumps by more than 50 percent, activate safety behavior
```

If checkpoint rollback is available, the optimizer may restore the previous parameters and reapply a safer AdamW update.

If rollback is not available, the optimizer should respond on the next step by shrinking trust radius and disabling amplification temporarily.

---

## 11. Computational Cost

The optimizer adds overhead because it stores history and computes layer statistics.

To reduce cost:

```text
store layer summaries, not full gradient tensors
use short history windows
use one shared MLP controller
compute features per layer, not per parameter
update controller features once per step
disable expensive diagnostics during warm-up if needed
```

Recommended stored values per layer:

```text
last 20 gradient norms
last 20 update norms
last 20 cosine similarities
EMA success rate
current trust radius
current amplification state
```

This keeps memory overhead small compared with storing complete gradient histories.

---

## 12. Failure Modes

### 12.1 False Starvation

A layer may appear starved even when it is near a good solution.

Mitigation:

```text
require loss plateau
require low update effect
require inter-layer gradient ratio
use confidence score
bound amplification
```

---

### 12.2 Noisy Gradients

Small-batch training can make gradients noisy.

Mitigation:

```text
detect gradient noise
reduce confidence under noise
avoid amplification when consistency is low
consider batch-size-aware noise thresholds
```

---

### 12.3 Trust Radius Collapse

The trust radius may shrink too much and stall learning.

Mitigation:

```text
enforce tau_min
allow slow recovery when loss improves
fall back to AdamW if adaptive updates fail repeatedly
```

---

### 12.4 Endless Amplification

A starved layer may keep receiving amplification without improving loss.

Mitigation:

```text
use amplification decay
cap max_amp
reduce amplification when loss does not improve
```

---

### 12.5 MLP Overfitting

The MLP controller may overfit to one training setup.

Mitigation:

```text
use shared weights
normalize inputs
train across multiple tasks
compare against rule-based baseline
keep trust region active
```

---

## 13. Stability and Convergence Discussion

This optimizer is more complex than AdamW, so formal convergence analysis is non-trivial.

The current stability mechanisms are:

```text
bounded confidence
bounded amplification
bounded learning-rate scale
bounded trust radius
norm-based update clipping
warm-up with AdamW
blend phase
fallback after instability
```

The trust region does not prove convergence by itself, but it limits the maximum step size and reduces the chance of catastrophic updates.

A formal convergence proof would require assumptions such as:

```text
bounded gradients
bounded update norms
smooth loss landscape
controlled stochastic noise
non-increasing trust radius after failed updates
```

Formal proof is future work. The first goal is empirical stability.

---

## 14. Testing Plan

### Test 1: Warm-Up Behavior

Goal:

```text
verify that the optimizer uses AdamW during t < W
```

Expected:

```text
no starvation correction during warm-up
history buffers are filled
AdamW updates are applied normally
```

Pass condition:

```text
adaptive controller is inactive before W steps
```

---

### Test 2: Starvation Detection

Goal:

```text
verify that starved early layers are detected
```

Setup:

```text
deep MLP with sigmoid or tanh activations
MNIST or synthetic classification
20 or more layers
```

Track:

```text
gradient norm per layer
relative gradient norm
inter-layer gradient ratio
loss plateau
starvation flag
```

Pass condition:

```text
early layers are marked starved only when loss is plateaued and gradient ratio is low
```

---

### Test 3: False Starvation Protection

Goal:

```text
verify that tiny gradients alone do not trigger correction
```

Expected:

```text
if gradient is small but loss is improving, starved_l,t = false
```

Pass condition:

```text
no amplification when loss is not plateaued
```

---

### Test 4: Noise Protection

Goal:

```text
verify that noisy gradients reduce confidence
```

Setup:

```text
small batch training
high stochastic gradient variance
```

Expected:

```text
normalized_noise_l,t increases
conf_l,t decreases
amplification is suppressed
```

Pass condition:

```text
optimizer does not amplify random tiny gradients
```

---

### Test 5: Trust Region Safety

Goal:

```text
verify that update norm never exceeds tau_l,t
```

Track:

```text
||Delta_fused_l,t||
||Delta_final_l,t||
tau_l,t
```

Pass condition:

```text
||Delta_final_l,t|| <= tau_l,t for every layer and step
```

---

### Test 6: Amplification Decay

Goal:

```text
verify that amplification does not continue when loss fails to improve
```

Expected:

```text
if starvation correction fails, amplify_state_l,t decreases
```

Pass condition:

```text
amplification returns toward 1.0 after repeated failure
```

---

### Test 7: Fusion Weight Behavior

Goal:

```text
verify that different layer states produce different fusion weights
```

Expected:

```text
starved layers increase normalized-gradient weight
noisy layers increase AdamW/RMSProp weight
stable layers increase SGD+Momentum weight
```

Pass condition:

```text
the optimizer does not use identical weights for all layers
```

---

### Test 8: Optimizer Comparison

Goal:

```text
compare against standard optimizers
```

Controlled architecture:

```text
20-layer MLP on MNIST
sigmoid or tanh activations to create vanishing gradients
same initialization
same batch size
same training budget
```

Compare:

```text
Adam
AdamW
RMSProp
SGD+Momentum
Rule-Based AI Optimizer
MLP AI Optimizer
```

Metrics:

```text
final loss
training speed
early-layer gradient norm
early-layer update norm
loss stability
accuracy
```

Pass condition:

```text
AI optimizer improves early-layer learning without instability
```

---

### Test 9: Ablation Study

Goal:

```text
measure which modules matter
```

Ablations:

```text
without starvation detector
without confidence score
without amplification decay
without trust region
without inter-layer gradient ratio
without warm-up
without update fusion
without layer memory
```

Expected:

```text
without trust region -> more instability
without confidence -> noisy gradients amplified
without warm-up -> poor early decisions
without inter-layer ratio -> weaker starvation detection
without amplification decay -> repeated failed amplification
```

---

## 15. Paper Roadmap

This project can support multiple papers if each paper has a different problem statement, contribution, and experiment design.

### Paper 1: Gradient State Analysis

Focus:

```text
detecting layer-wise gradient health states
```

Contribution:

```text
starvation detection
noise detection
oscillation detection
inter-layer gradient ratio
```

Experiments:

```text
detection accuracy
correlation with training failure
early warning before loss plateau
```

---

### Paper 2: Update Fusion Engine

Focus:

```text
adaptive fusion of optimizer update candidates
```

Contribution:

```text
AdamW + RMSProp + SGD+Momentum + normalized-gradient fusion
adaptive fusion weights
```

Experiments:

```text
fixed optimizer vs fused optimizer
hard switching vs soft fusion
per-layer fusion weight analysis
```

---

### Paper 3: Layer State Memory

Focus:

```text
historical memory for optimizer decisions
```

Contribution:

```text
gradient history
update history
loss response feedback
success-rate EMA
```

Experiments:

```text
no memory vs gradient memory vs gradient/update/loss memory
impact on false starvation and convergence
```

---

### Paper 4: Trust Region Control

Focus:

```text
safe adaptive optimizer updates
```

Contribution:

```text
dynamic trust radius
bounded amplification
stability fallback
```

Experiments:

```text
with vs without trust region
fixed clipping vs dynamic trust radius
loss spike frequency
maximum update norm
```

---

## 16. Final Summary

The updated optimizer is an implementable hybrid system.

It uses:

```text
AdamW warm-up
rule-based adaptive control
optional shared MLP controller
relative starvation detection
inter-layer gradient ratio
bounded confidence
bounded amplification
amplification decay
adaptive update fusion
dynamic trust region
loss-response feedback
```

The central principle is:

```text
Detect the state of each layer, choose an update strategy, bound the risk, observe the result, and adapt the next decision.
```

This turns the original idea from a design sketch into a concrete optimizer specification that can be implemented, tested, and expanded into research papers.
