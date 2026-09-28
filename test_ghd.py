"""Unit tests for the Experiment 1 GHD components."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from ghd.controller import ControllerTrainer, GHDController, load_controller
from ghd.core import (
    LayerSignals, LayerState, confidence_noise, confidence_oscil, confidence_vanish, confidence_vanish_from_s2_s3,
    rule_strength, sigmoid_clipped,
)
from ghd.experiment import SigmoidMLP, curve_auc, run_one, steps_to_threshold
from ghd.hook import GHDHook
from ghd.optimizers import LARS, LNGD, _trust_ratio
from ghd.paths import result_files, run_prefix


def _sig(**kw) -> LayerSignals:
    base = dict(s1=1.0, s2=1.0, s3=1.0, s4=0.5, s5=0.0, s6=0.0, s7=1.0, s_depth=1.0, median=1.0, n_prior=20)
    base.update(kw)
    return LayerSignals(**base)


class TestCore(unittest.TestCase):
    def test_confidence_bounds(self) -> None:
        for x in (-1e9, -3, 0, 3, 1e9):
            self.assertTrue(0.047 <= sigmoid_clipped(x) <= 0.953)
        self.assertEqual(confidence_vanish(0.0), 1.0)
        self.assertAlmostEqual(confidence_vanish(0.005), 0.5)
        self.assertEqual(confidence_vanish(0.02), 0.0)
        self.assertTrue(0.047 <= confidence_oscil(0.0, -1.0) <= 0.953)
        self.assertTrue(0.047 <= confidence_noise(1.0) <= 0.953)

    def test_persistence_counters_are_independent(self) -> None:
        st = LayerState()
        for _ in range(4):
            st.update_counters(_sig(s6=-0.5))
        self.assertEqual((st.p_vanish, st.p_oscil, st.p_noisy), (0, 4, 0))
        st.update_counters(_sig(s_depth=0.001))  # vanishing only: oscillation counter resets, vanish starts
        self.assertEqual((st.p_vanish, st.p_oscil, st.p_noisy), (1, 0, 0))

    def test_structural_vanishing_beats_exploding(self) -> None:
        st = LayerState()
        st.p_vanish = 5
        mode, rule = st.detect(_sig(s_depth=0.001, s1=100.0, median=1.0), step=200)
        self.assertEqual((mode, rule), ("vanishing", "structural_vanishing"))

    def test_spike_after_vanishing_is_recovering(self) -> None:
        st = LayerState()
        st.last_vanish_step = 195
        self.assertEqual(st.detect(_sig(s1=100.0), step=200)[0], "recovering")
        self.assertEqual(st.detect(_sig(s1=100.0), step=300)[0], "exploding")

    def test_noisy_disabled_by_default(self) -> None:
        st = LayerState()
        st.p_noisy = 10
        self.assertEqual(st.detect(_sig(s5=0.9), step=200)[0], "healthy")
        self.assertEqual(st.detect(_sig(s5=0.9), step=200, enable_noisy=True)[0], "noisy")

    def test_non_structural_vanishing_has_nonzero_strength(self) -> None:
        sig = _sig(s2=0.05, s3=0.1, s_depth=0.5)  # S_depth >= 0.01: confidence_vanish would be 0
        self.assertEqual(confidence_vanish(sig.s_depth), 0.0)
        self.assertAlmostEqual(confidence_vanish_from_s2_s3(0.10, 0.20), 0.5)
        for rule in ("vanishing_strict", "vanishing_early", "vanishing_s3"):
            self.assertGreater(rule_strength("vanishing", sig, rule), 0.5, rule)
        self.assertEqual(rule_strength("vanishing", sig, "structural_vanishing"), 0.0)

    def test_s4_masked_without_previous_gradient(self) -> None:
        self.assertEqual(LayerState().compute_signals(1.0, 1.0, 1.0, None).s4, 0.5)


class TestOptimizers(unittest.TestCase):
    def test_trust_ratio_float64_survives_tiny_gradients(self) -> None:
        p = torch.ones(1000)
        g = torch.full((1000,), 1e-25)  # squares underflow to 0 in float32
        phi = _trust_ratio(p, g, eta=0.001, lam=0.0, eps=0.0)
        self.assertGreater(phi, 1e15)

    def test_lars_and_lngd_reduce_loss(self) -> None:
        for cls, kw in ((LARS, dict(lr=1.0)), (LNGD, dict(lr=1.0))):
            torch.manual_seed(0)
            model = nn.Linear(5, 1)
            nn.init.normal_(model.weight)
            x, y = torch.randn(64, 5), torch.randn(64, 1)
            opt = cls(model.parameters(), **kw)
            first = None
            for _ in range(50):
                loss = ((model(x) - y) ** 2).mean()
                first = first if first is not None else loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.assertLess(loss.item(), first, cls.__name__)


class TestHook(unittest.TestCase):
    def _batch(self):
        return torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,))

    def test_passive_hook_never_modifies_gradients(self) -> None:
        torch.manual_seed(0)
        model = SigmoidMLP(num_layers=4, hidden_dim=16)
        hook = GHDHook(model, mode="passive", warmup=2, log_every=1)
        for _ in range(10):
            x, y = self._batch()
            model.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            before = [m.weight.grad.clone() for m in hook.layers]
            weights = [m.weight.detach().clone() for m in hook.layers]
            hook.pre_step(1.0)
            hook.post_step()
            for b, w, m in zip(before, weights, hook.layers):
                self.assertTrue(torch.equal(b, m.weight.grad))
                self.assertTrue(torch.equal(w, m.weight))
        self.assertEqual(hook.total_interventions, 0)
        self.assertEqual(len(hook.logs), 10)
        self.assertEqual(hook.logs[0]["layers"]["L4"]["s_depth"], 1.0)

    def test_vanishing_correction_amplifies_the_sgd_update(self) -> None:
        torch.manual_seed(0)
        model = SigmoidMLP(num_layers=10, hidden_dim=32)
        hook = GHDHook(model, mode="rules", warmup=1)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        first = hook.layers[0]
        amplified = False
        for _ in range(12):
            x, y = self._batch()
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            hook.pre_step(2.3)
            w0 = first.weight.detach().clone()
            opt.step()
            natural = (first.weight - w0).norm().item()
            hook.post_step()
            applied = (first.weight - w0).norm().item()
            self.assertGreaterEqual(applied, natural * (1 - 1e-5))
            amplified |= applied > 1.5 * natural
        self.assertTrue(amplified)
        self.assertGreater(hook.total_interventions, 0)

    def test_conv_layers_are_monitored(self) -> None:
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1), nn.Sigmoid(), nn.Conv2d(4, 4, 3, padding=1), nn.Sigmoid(),
            nn.Flatten(), nn.Linear(4 * 28 * 28, 10),
        )
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        hook = GHDHook(model, mode="rules", warmup=2, log_every=1, optimizer=opt)
        self.assertEqual([type(m) for m in hook.layers], [nn.Conv2d, nn.Conv2d, nn.Linear])
        for _ in range(5):
            x, y = self._batch()
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            hook.pre_step(2.3)
            opt.step()
            hook.post_step()
        self.assertEqual(list(hook.logs[-1]["layers"]), ["L1", "L2", "L3"])
        self.assertGreater(hook.logs[-1]["layers"]["L1"]["s1"], 0.0)

    def test_ai_mode_falls_back_without_checkpoint(self) -> None:
        with self.assertWarns(UserWarning):
            hook = GHDHook(SigmoidMLP(num_layers=3, hidden_dim=8), mode="ai", controller_path="/nonexistent.pt")
        self.assertEqual(hook.controller_status, "fallback_rules")


class TestExperiment(unittest.TestCase):
    def test_metrics(self) -> None:
        acc, steps = [40.0, 72.0, 85.0, 91.0], [50, 100, 150, 200]
        s = steps_to_threshold(acc, steps)
        self.assertEqual((s["50"], s["70"], s["80"], s["90"], s["95"]), (100, 100, 150, 200, None))
        self.assertAlmostEqual(curve_auc([0.0, 100.0]), 25.0)

    def test_short_run_end_to_end_and_controller(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r = run_one("sgd", "rules", 42, n_steps=150, results_dir=tmp, verbose=False)
            self.assertEqual(len(r["val_acc_curve"]), 3)
            self.assertTrue(Path(tmp, "mlp_mnist_sgd_rules_seed42.json").exists())
            again = run_one("sgd", "rules", 42, n_steps=150, results_dir=tmp, verbose=False)  # resumed
            self.assertEqual(again["elapsed_seconds"], r["elapsed_seconds"])

            trainer = ControllerTrainer()
            self.assertGreater(trainer.load_logs(tmp), 0)
            trainer.train(epochs=2, verbose=False)
            ckpt = Path(tmp, "ghd_controller.pt")
            trainer.save(ckpt)
            self.assertIsInstance(load_controller(ckpt), GHDController)
            ai = run_one("sgd", "ai", 42, n_steps=150, results_dir=tmp, controller_path=ckpt, verbose=False)
            self.assertEqual(ai["ghd_summary"]["controller"], "mlp")

    def test_exp2_spec_naming_and_eval_every(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = run_prefix("mlp", "mnist", 7, 2.5)
            self.assertEqual(prefix, "mlp_mnist_l7_g2.5")
            r = run_one("sgd", "none", 42, n_steps=30, eval_every=10, results_dir=tmp, verbose=False,
                        prefix=prefix, num_layers=7, init_gain=2.5)
            self.assertTrue(Path(tmp, "mlp_mnist_l7_g2.5_sgd_none_seed42.json").exists())
            self.assertEqual(r["val_steps"], [10, 20, 30])
            self.assertEqual((r["config"]["num_layers"], r["config"]["init_gain"]), (7, 2.5))
            Path(tmp, "mlp_mnist_sgd_none_seed42.json").write_text("{}")  # an Exp 1 name in the same folder
            self.assertEqual([p.name for p in result_files(tmp, prefix)], ["mlp_mnist_l7_g2.5_sgd_none_seed42.json"])
            self.assertEqual([p.name for p in result_files(tmp, "mlp_mnist")], ["mlp_mnist_sgd_none_seed42.json"])
            with self.assertRaises(NotImplementedError):
                run_one("sgd", "none", 42, n_steps=10, results_dir=tmp, verbose=False, model_name="cnn",
                        dataset="cifar10", prefix="cnn_cifar10_l4_g2.5")


if __name__ == "__main__":
    unittest.main()
