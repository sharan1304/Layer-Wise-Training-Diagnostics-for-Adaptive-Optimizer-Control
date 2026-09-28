"""Unit tests for the Phase 1 AI optimizer."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

from ai_optimizer import AIOptimizer
from control_policy import RuleBasedController
from mlp_policy import FEATURE_DIM, ControlMLP, build_feature_vector, checkpoint_is_compatible
from config import OptimizerConfig
from grad_analyzer import GradientFeatures, GradientStateAnalyzer
from layer_memory import LayerStateMemory
from trust_region import TrustRegionController


class TestAIOptimizer(unittest.TestCase):
    def test_1_warmup_uses_adamw(self) -> None:
        model = nn.Linear(10, 5)
        opt = AIOptimizer(model.parameters(), lr=1e-3, warmup_steps=50)
        x = torch.randn(4, 10)
        y = torch.randint(0, 5, (4,))

        for _ in range(30):
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            opt.step(loss=loss.item())

        self.assertEqual(len(opt.layer_log), 0)
        for state in opt.memory.states.values():
            self.assertEqual(state.amp_state, 1.0)

    def test_2_starvation_detection_requires_all_conditions(self) -> None:
        analyzer = GradientStateAnalyzer(OptimizerConfig(detector="strict"))
        self.assertFalse(
            analyzer.detect_starvation(
                relative_norm=0.05,
                inter_layer_ratio=0.1,
                success_ema=0.3,
                loss_plateau=False,
            )
        )
        self.assertTrue(
            analyzer.detect_starvation(
                relative_norm=0.05,
                inter_layer_ratio=0.1,
                success_ema=0.3,
                loss_plateau=True,
            )
        )

    def test_3_false_starvation_protection(self) -> None:
        analyzer = GradientStateAnalyzer(OptimizerConfig(detector="strict"))
        self.assertFalse(
            analyzer.detect_starvation(
                relative_norm=0.001,
                inter_layer_ratio=0.05,
                success_ema=0.3,
                loss_plateau=False,
            )
        )

    def test_4_noise_reduces_confidence(self) -> None:
        policy = RuleBasedController()
        noisy = GradientFeatures(
            grad_norm=0.1,
            relative_norm=0.5,
            inter_layer_ratio=0.5,
            consistency=0.5,
            noise=0.8,
            oscillation=0.3,
            stability=0.1,
            starved=False,
            exploding=False,
            noisy=True,
            oscillating=False,
            stable=False,
        )
        clean = GradientFeatures(
            grad_norm=0.1,
            relative_norm=0.5,
            inter_layer_ratio=0.5,
            consistency=0.9,
            noise=0.05,
            oscillation=0.05,
            stability=0.9,
            starved=False,
            exploding=False,
            noisy=False,
            oscillating=False,
            stable=True,
        )
        self.assertLess(policy.compute_confidence(noisy, 0.5), policy.compute_confidence(clean, 0.5))

    def test_5_trust_region_bounds_update(self) -> None:
        trust = TrustRegionController()
        for tau in [0.001, 0.01, 0.1, 0.5, 1.0]:
            for _ in range(20):
                delta = torch.randn(50) * 10.0
                clipped = trust.apply(delta, tau)
                self.assertLessEqual(float(clipped.norm().item()), tau + 1e-5)

    def test_6_amplification_decay(self) -> None:
        memory = LayerStateMemory()
        layer_id = 0
        memory.get_or_create(layer_id).amp_state = 3.0
        for _ in range(10):
            memory.update_amp_state(layer_id, starved=True, improved=False)
        final_amp = memory.states[layer_id].amp_state
        self.assertLess(final_amp, 3.0)
        self.assertGreaterEqual(final_amp, 1.0)

    def test_7_fusion_weights_vary_by_state(self) -> None:
        policy = RuleBasedController()

        def make_features(**kwargs) -> GradientFeatures:
            defaults = dict(
                grad_norm=0.1,
                relative_norm=0.5,
                inter_layer_ratio=0.5,
                consistency=0.5,
                noise=0.1,
                oscillation=0.1,
                stability=0.8,
                starved=False,
                exploding=False,
                noisy=False,
                oscillating=False,
                stable=False,
            )
            defaults.update(kwargs)
            return GradientFeatures(**defaults)

        w_starved = policy.compute_fusion_weights(make_features(starved=True))
        w_noisy = policy.compute_fusion_weights(make_features(noisy=True))
        w_stable = policy.compute_fusion_weights(make_features(stable=True))
        self.assertGreater(w_starved[3], 0.4)
        self.assertGreater(w_noisy[0] + w_noisy[1], 0.7)
        self.assertGreater(w_stable[2], 0.5)
        for weights in [w_starved, w_noisy, w_stable]:
            self.assertAlmostEqual(sum(weights), 1.0, places=4)

    def test_8_adamw_decay_scaled_by_lr(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        opt = AIOptimizer(model.parameters(), lr=1e-3, warmup_steps=1, weight_decay=0.01)
        before = model.weight.detach().clone()
        model.weight.grad = torch.zeros_like(model.weight)
        opt.step(loss=1.0)
        expected = before - 1e-3 * 0.01 * before
        self.assertTrue(torch.allclose(model.weight, expected, atol=1e-7))

    def test_9_optimizer_runs_after_warmup(self) -> None:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
        opt = AIOptimizer(model.parameters(), lr=1e-3, warmup_steps=2, blend_steps=2)
        x = torch.randn(16, 4)
        y = torch.randint(0, 2, (16,))

        for _ in range(6):
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            opt.step(loss=loss.item())

        self.assertGreater(len(opt.layer_log), 0)
        self.assertTrue(np.isfinite(loss.item()))

    def test_10_hybrid_detector_accepts_early_signal(self) -> None:
        strict = GradientStateAnalyzer(OptimizerConfig(detector="strict"))
        hybrid = GradientStateAnalyzer(OptimizerConfig(detector="hybrid"))

        kwargs = dict(
            relative_norm=0.2,
            inter_layer_ratio=0.3,
            success_ema=0.5,
            loss_plateau=False,
        )
        self.assertFalse(strict.detect_starvation(**kwargs))
        self.assertTrue(hybrid.detect_starvation(**kwargs))

    def test_11_optimizer_mode_presets(self) -> None:
        expected = {
            "safe": [0.20, 0.20, 0.10, 0.50],
            "balanced": [0.10, 0.10, 0.10, 0.70],
            "aggressive": [0.05, 0.05, 0.05, 0.85],
            "extreme": [0.00, 0.00, 0.00, 1.00],
        }
        for mode, weights in expected.items():
            self.assertEqual(OptimizerConfig(mode=mode).starved_fusion_weights, weights)

    def test_12_recovery_scale_grows_and_shrinks(self) -> None:
        memory = LayerStateMemory()
        layer_id = 0
        state = memory.get_or_create(layer_id)
        self.assertEqual(state.recovery_scale, 1.0)

        memory.update_recovery_state(
            layer_id,
            starved=True,
            improved=True,
            recovered=False,
            grow=1.2,
            shrink=0.7,
            min_scale=1.0,
            max_scale=10.0,
        )
        self.assertGreater(memory.states[layer_id].recovery_scale, 1.0)
        self.assertEqual(memory.states[layer_id].starved_steps, 1)

        grown = memory.states[layer_id].recovery_scale
        memory.update_recovery_state(
            layer_id,
            starved=False,
            improved=True,
            recovered=True,
            grow=1.2,
            shrink=0.7,
            min_scale=1.0,
            max_scale=10.0,
        )
        self.assertLess(memory.states[layer_id].recovery_scale, grown)
        self.assertEqual(memory.states[layer_id].starved_steps, 0)
        self.assertEqual(memory.states[layer_id].recovery_successes, 1)

    def test_13_config_drives_alpha_and_cold_start_amp(self) -> None:
        features = GradientFeatures(
            grad_norm=0.01,
            relative_norm=0.4,
            inter_layer_ratio=0.2,
            consistency=0.8,
            noise=0.0,
            oscillation=0.0,
            stability=1.0,
            starved=True,
            exploding=False,
            noisy=False,
            oscillating=False,
            stable=True,
        )
        config = OptimizerConfig(
            depth_boost=1.0,
            starvation_lr_multiplier=4.0,
            cold_start_amp=2.0,
        )
        policy = RuleBasedController(config)
        alpha = policy.compute_alpha(features, layer_idx=0, total_layers=3, alpha_base=1e-3)
        self.assertAlmostEqual(alpha, 0.008, places=8)

        state = LayerStateMemory().get_or_create(0)
        amp = policy.compute_amplification(features, state)
        self.assertGreaterEqual(amp, 2.0)

    def test_14_depth_trust_logged_for_early_layer(self) -> None:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
        opt = AIOptimizer(model.parameters(), lr=1e-3, warmup_steps=1, blend_steps=1)
        x = torch.randn(16, 4)
        y = torch.randint(0, 2, (16,))

        for _ in range(4):
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            opt.step(loss=loss.item())

        first_weight_logs = [row for row in opt.layer_log if row["layer"] == 0]
        self.assertTrue(first_weight_logs)
        self.assertGreaterEqual(first_weight_logs[-1]["effective_trust"], first_weight_logs[-1]["trust"])

    def test_15_mlp_requires_checkpoint_unless_explicitly_allowed(self) -> None:
        model = nn.Linear(4, 2)
        with self.assertRaises(ValueError):
            AIOptimizer(model.parameters(), controller="mlp")

        model_allowed = nn.Linear(4, 2)
        opt = AIOptimizer(
            model_allowed.parameters(),
            controller="mlp",
            config=OptimizerConfig(allow_untrained_mlp=True),
            warmup_steps=1,
        )
        self.assertIsNotNone(opt.mlp_policy)

    def test_16_mlp_temporal_feature_vector_has_expected_size(self) -> None:
        memory = LayerStateMemory()
        state = memory.get_or_create(0)
        features = GradientFeatures(
            grad_norm=0.01,
            relative_norm=0.2,
            inter_layer_ratio=0.3,
            consistency=0.8,
            noise=0.1,
            oscillation=0.05,
            stability=0.85,
            starved=True,
            exploding=False,
            noisy=False,
            oscillating=False,
            stable=True,
        )
        x = build_feature_vector(
            features,
            state,
            layer_idx=0,
            total_layers=4,
            global_step=3,
            max_training_steps=100,
            loss_plateau=True,
            alpha_base=1e-3,
            history_window=5,
        )
        self.assertEqual(x.numel(), FEATURE_DIM)
        self.assertTrue(torch.isfinite(x).all())

    def test_17_mlp_checkpoint_compatibility_detects_current_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/mlp.pt"
            torch.save(ControlMLP().state_dict(), path)
            self.assertTrue(checkpoint_is_compatible(path))

            bad_path = f"{tmpdir}/bad.pt"
            torch.save({"net.0.weight": torch.zeros(64, 14), "net.4.weight": torch.zeros(8, 64)}, bad_path)
            self.assertFalse(checkpoint_is_compatible(bad_path))


if __name__ == "__main__":
    unittest.main()
