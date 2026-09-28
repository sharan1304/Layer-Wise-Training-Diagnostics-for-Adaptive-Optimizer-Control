"""Layer-aware AI-guided adaptive optimizer, Phase 1.

This is a drop-in PyTorch optimizer:

    optimizer = AIOptimizer(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(loss=loss.item())
"""

from __future__ import annotations

from collections import deque
from typing import Callable, List, Literal, Optional, Tuple

import torch
from torch.optim import Optimizer

from candidates import CandidateGenerator
from config import DetectorMode, OptimizerConfig, OptimizerMode
from control_policy import RuleBasedController, blend_signals
from fusion import UpdateFusionEngine
from grad_analyzer import GradientStateAnalyzer
from layer_memory import LayerStateMemory
from mlp_policy import MLPController, build_current_feature_vector
from trust_region import TrustRegionController

ControllerMode = Literal["rule", "mlp"]


class AIOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        warmup_steps: int | None = None,
        blend_steps: int | None = None,
        instability_threshold: float = 1.5,
        weight_decay: float = 0.01,
        trust_warmup: bool = False,
        mode: OptimizerMode = "balanced",
        detector: DetectorMode = "hybrid",
        config: OptimizerConfig | None = None,
        controller: ControllerMode = "rule",
        mlp_checkpoint: str | None = None,
    ) -> None:
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.config = config or OptimizerConfig(mode=mode, detector=detector)
        self.lr = lr
        self.warmup_steps = warmup_steps if warmup_steps is not None else self.config.warmup_steps
        self.blend_steps = blend_steps if blend_steps is not None else self.config.blend_steps
        self.instability_threshold = instability_threshold
        self.weight_decay = weight_decay
        self.trust_warmup = trust_warmup
        self.controller = controller

        self.memory = LayerStateMemory()
        self.analyzer = GradientStateAnalyzer(self.config)
        # The rule controller always runs, as the teacher for distillation
        # and as the safety baseline the MLP is confidence-blended against.
        self.rule_policy = RuleBasedController(self.config)
        if controller == "mlp" and mlp_checkpoint is None and not self.config.allow_untrained_mlp:
            raise ValueError(
                "controller=\"mlp\" requires mlp_checkpoint unless "
                "OptimizerConfig.allow_untrained_mlp=True is set for debugging."
            )
        self.mlp_policy = (
            MLPController(self.config, checkpoint=mlp_checkpoint) if controller == "mlp" else None
        )
        self.mlp_blend_base = self.config.mlp_blend_base
        self.gen = CandidateGenerator()
        self.fusion = UpdateFusionEngine()
        self.trust = TrustRegionController()

        self.global_step = 0
        self.loss_history: deque[float] = deque(maxlen=100)
        self.prev_loss: Optional[float] = None
        self.in_emergency = False
        self.layer_log: List[dict] = []
        # Optional hook for collecting (features, state, signals) samples,
        # e.g. to distill a RuleBasedController run into an MLPController.
        self.record_callback: Optional[Callable] = None

    def _collect_all_params(self) -> List[Tuple[int, torch.Tensor]]:
        result: List[Tuple[int, torch.Tensor]] = []
        idx = 0
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    result.append((idx, param))
                    idx += 1
        return result

    def _collect_weight_like_gradients(self) -> dict[int, torch.Tensor]:
        grads: dict[int, torch.Tensor] = {}
        idx = 0
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.ndim >= 2:
                    grads[idx] = param.grad.detach()
                idx += 1
        return grads

    def _next_weight_like_grad(
        self,
        layer_idx: int,
        all_weight_grads: dict[int, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        for next_idx in sorted(k for k in all_weight_grads if k > layer_idx):
            return all_weight_grads[next_idx]
        return None

    @torch.no_grad()
    def step(self, closure=None, loss: Optional[float] = None):
        loss_val = None
        if closure is not None:
            with torch.enable_grad():
                closure_loss = closure()
            loss_val = float(closure_loss.item() if hasattr(closure_loss, "item") else closure_loss)
        if loss is not None:
            loss_val = float(loss)

        if loss_val is not None:
            self.loss_history.append(loss_val)

        self.global_step += 1
        all_params = self._collect_all_params()
        all_layer_ids = [layer_idx for layer_idx, _ in all_params]

        instability = (
            self.prev_loss is not None
            and loss_val is not None
            and self.trust.check_instability(
                loss_val,
                self.prev_loss,
                self.instability_threshold,
            )
        )
        if instability:
            self.trust.emergency_response(self.memory, all_layer_ids)
            self.in_emergency = True

        loss_plateau = self.memory.get_loss_plateau(self.loss_history)
        all_weight_grads = self._collect_weight_like_gradients()
        weight_like_count = max(len(all_weight_grads), 1)

        for layer_idx, param in all_params:
            g = param.grad.detach()
            state = self.memory.get_or_create(layer_idx)
            improved = (
                loss_val is not None
                and self.prev_loss is not None
                and loss_val < self.prev_loss
            )

            warmup_active = self.global_step <= self.warmup_steps
            emergency_active = self.in_emergency or state.emergency_mode
            weight_like = param.ndim >= 2

            if warmup_active or emergency_active or not weight_like:
                delta = self.gen.adamw_update(
                    g,
                    param,
                    state,
                    self.lr,
                    weight_decay=self.weight_decay,
                )
                if self.trust_warmup or emergency_active:
                    delta = self.trust.apply(delta, state.trust_radius)

                param.add_(delta)
                grad_norm = float(g.norm().item())
                cosine = self.analyzer.compute_consistency(g, state.prev_grad)
                self.memory.update_after_step(
                    layer_idx,
                    grad_norm=grad_norm,
                    update_norm=float(delta.norm().item()),
                    cosine_sim=cosine,
                    success_flag=float(improved),
                )
                if not warmup_active and weight_like:
                    self.memory.update_trust_radius(layer_idx, improved)
                state.prev_grad = g.detach().clone()
                self.memory.reset_emergency(layer_idx)
                continue

            next_grad = self._next_weight_like_grad(layer_idx, all_weight_grads)
            features = self.analyzer.compute_all(
                layer_idx,
                g,
                state,
                next_grad,
                loss_plateau,
            )
            effective_layer_idx = sum(1 for idx in all_weight_grads if idx <= layer_idx)
            depth_l = max(effective_layer_idx - 1, 0) / max(weight_like_count - 1, 1)
            policy_layer_idx = max(effective_layer_idx - 1, 0)

            rule_signals = self.rule_policy.compute_all(
                features,
                state,
                policy_layer_idx,
                weight_like_count,
                self.lr,
                loss_plateau=loss_plateau,
                global_step=self.global_step,
            )
            depth_trust_factor = 1.0 + self.config.depth_trust_boost * (1.0 - depth_l)
            if features.starved:
                depth_trust_factor += 0.05 * min(state.starved_steps, 8)
            rule_effective_trust = min(rule_signals.trust_radius * depth_trust_factor, self.config.tau_max)

            if self.mlp_policy is not None:
                mlp_signals = self.mlp_policy.compute_all(
                    features,
                    state,
                    policy_layer_idx,
                    weight_like_count,
                    self.lr,
                    loss_plateau=loss_plateau,
                    global_step=self.global_step,
                )
                # The MLP's trust-radius output is distilled directly from the
                # rule-based effective trust radius, so it's already on the
                # same scale as rule_effective_trust (no extra depth scaling).
                mlp_effective_trust = mlp_signals.trust_radius
                blend = self.mlp_blend_base * mlp_signals.confidence
                signals = blend_signals(rule_signals, mlp_signals, blend)
                effective_trust_radius = (
                    (1.0 - blend) * rule_effective_trust + blend * mlp_effective_trust
                )
            else:
                signals = rule_signals
                effective_trust_radius = rule_effective_trust

            current_feature_vector = build_current_feature_vector(
                features,
                state,
                policy_layer_idx,
                weight_like_count,
                self.global_step,
                self.config.max_training_steps,
                loss_plateau,
                self.lr,
            )

            if self.record_callback is not None:
                # Always recorded from the pure rule signals, so a
                # distillation run stays valid even with controller="mlp".
                self.record_callback(
                    features=features,
                    state=state,
                    layer_idx=policy_layer_idx,
                    total_layers=weight_like_count,
                    alpha_base=self.lr,
                    loss_plateau=loss_plateau,
                    global_step=self.global_step,
                    signals=rule_signals,
                    effective_trust_radius=rule_effective_trust,
                )

            candidates = self.gen.generate_all(
                g,
                param,
                state,
                signals,
                weight_decay=self.weight_decay,
            )
            delta_ai = self.fusion.fuse(candidates, signals.fusion_weights)
            # Use the already-generated AdamW candidate. Recomputing it would
            # mutate Adam moments twice in one step.
            delta_fused = self.fusion.blend(
                delta_ai,
                candidates.adamw,
                self.global_step,
                self.warmup_steps,
                self.blend_steps,
            )
            delta_final, pre_clip_norm, trust_clipped = self.trust.apply_with_info(
                delta_fused,
                effective_trust_radius,
            )
            param.add_(delta_final)

            stable_enough = state.success_ema >= 0.45
            self.memory.update_after_step(
                layer_idx,
                grad_norm=features.grad_norm,
                update_norm=float(delta_final.norm().item()),
                cosine_sim=features.consistency,
                success_flag=float(improved),
            )
            recovered = state.starved_steps > 0 and not features.starved
            self.memory.update_trust_radius(layer_idx, improved or stable_enough or features.starved)
            self.memory.update_amp_state(layer_idx, features.starved, improved)
            self.memory.update_recovery_state(
                layer_idx,
                starved=features.starved,
                improved=improved,
                recovered=recovered,
                grow=self.config.recovery_grow,
                shrink=self.config.recovery_shrink,
                min_scale=self.config.recovery_min,
                max_scale=self.config.recovery_max,
            )
            state.prev_grad = g.detach().clone()
            self.memory.append_feature_vector(
                layer_idx,
                current_feature_vector,
                maxlen=self.config.mlp_history_window,
            )

            if len(self.layer_log) < 10000:
                self.layer_log.append(
                    {
                        "step": self.global_step,
                        "layer": layer_idx,
                        "grad_norm": features.grad_norm,
                        "relative_norm": features.relative_norm,
                        "inter_layer_ratio": features.inter_layer_ratio,
                        "confidence": signals.confidence,
                        "update_norm": float(delta_final.norm().item()),
                        "pre_clip_update_norm": pre_clip_norm,
                        "trust_clipped": trust_clipped,
                        "starved": features.starved,
                        "recovered": recovered,
                        "trust": state.trust_radius,
                        "effective_trust": effective_trust_radius,
                        "amp": state.amp_state,
                        "recovery_scale": state.recovery_scale,
                        "starved_steps": state.starved_steps,
                        "weights": signals.fusion_weights,
                    }
                )

        self.in_emergency = False
        if loss_val is not None:
            self.prev_loss = loss_val
        return loss_val
