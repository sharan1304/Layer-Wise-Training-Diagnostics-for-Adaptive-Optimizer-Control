"""Optimizers x GHD CLI. Experiment 1: 10-layer sigmoid MLP on MNIST, gain 4.0.

    python main.py --exp 1                        # phase 1 -> train controller -> phase 2
    python main.py --exp 1 --phase 1              # baselines + GHD-Rules
    python main.py --train_controller             # distil GHDController from GHD-Rules logs
    python main.py --exp 1 --phase 2              # GHD-AI configs
    python main.py --exp 1 --opt sgd --seeds 42   # one optimizer family, one seed
    python main.py --quick                        # 500 steps, seed 42, SGD only (results/quick/)
    python main.py --lr_sweep                     # 500-step lr sweep for SGD/LARS/LNGD (diagnostic only)
    python main.py --exp 1 --opt sgd --steps 2000 --seeds 42 --quick_check
                                                  # SGD sanity check into results/quick_check/
    python main.py --summary                      # print the results table

    python main.py --exp 2 --layers 7 --gain 2.5 --phase 1 --workers 4
                                                  # Experiment 2 into results/exp2/, eval every 10 steps

Runs are resumable: an existing result JSON is skipped. Experiment 1 files are
results/mlp_mnist_{opt}_{ghd}_seed{seed}.json; Experiment 2+ files are
results/exp{N}/{model}_{dataset}_l{layers}_g{gain}_{opt}_{ghd}_seed{seed}.json.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ghd.controller import ControllerTrainer, checkpoint_sources, rules_log_files
from ghd.experiment import (
    CONFIGS, DATASETS, EVAL_EVERY, EVAL_EVERY_NEW, INIT_GAIN, MODELS, N_STEPS_BY_OPT, NUM_LAYERS, PHASE_OF_GHD,
    SEEDS, RUNNING_DIR, check_supported, config_label, lr_sweep, run_one,
)
from ghd.paths import EXP1_PREFIX, result_path, run_prefix
from ghd.optimizers import OPTIMIZER_LR
from ghd.summary import aggregate, load_results, print_table


def train_controller(results_dir: Path, ckpt: Path, epochs: int, prefix: str | None = None) -> bool:
    trainer = ControllerTrainer()
    n = trainer.load_logs(results_dir, prefix)
    if n == 0:
        print(f"No GHD-Rules logs in {results_dir}; run phase 1 first.")
        return False
    print(f"Training GHDController on {n} layer-steps from {len(trainer.sources)} GHD-Rules runs")
    print(f"  class counts: {trainer.class_counts()}")
    stats = trainer.train(epochs=epochs)
    trainer.save(ckpt)
    print(f"Saved {ckpt} (best val loss {stats['best_val_loss']:.4f})")
    return True


def controller_is_stale(results_dir: Path, ckpt: Path, prefix: str | None = None) -> bool:
    """Retrain if the checkpoint is missing, was trained on other files, or any rules log is newer."""
    used = checkpoint_sources(ckpt)
    logs = rules_log_files(results_dir, prefix)
    if used is None:
        return True
    if logs and sorted(used) != sorted(p.name for p in logs):
        return True
    return any(p.stat().st_mtime > ckpt.stat().st_mtime for p in logs)


def run_many(jobs: list[tuple[str, str, int]], args, results_dir: Path, ckpt: Path) -> None:
    todo = [j for j in jobs if args.force or not result_path(results_dir, *j, prefix=args.prefix).exists()]
    skipped = len(jobs) - len(todo)
    if skipped:
        print(f"Skipping {skipped} run(s) whose result JSON already exists.")
    if not todo:
        return
    steps = args.steps or "default"
    print(f"Running {len(todo)} run(s) x {steps} steps with {args.workers} worker(s)...")
    kwargs = dict(n_steps=args.steps, eval_every=args.eval_every, results_dir=results_dir, controller_path=ckpt,
                  force=args.force, prefix=args.prefix, model_name=args.model, dataset=args.dataset,
                  num_layers=args.layers, init_gain=args.gain)
    if args.workers <= 1:
        for opt, ghd, seed in todo:
            run_one(opt, ghd, seed, **kwargs)
        return
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(run_one, opt, ghd, seed, **kwargs): (opt, ghd, seed) for opt, ghd, seed in todo}
        for fut in as_completed(futures):
            opt, ghd, seed = futures[fut]
            try:
                fut.result()
            except Exception as exc:  # keep the other runs going
                print(f"[fail] {config_label(opt, ghd)} seed {seed}: {exc!r}")


def report_quick_check(results: list[dict], step: int = 1000, target: float = 50.0) -> None:
    print(f"\n=== Quick check: validation accuracy at step {step} (target > {target:.0f}%) ===")
    for r in results:
        c = r["config"]
        acc = next((a for s, a in zip(r["val_steps"], r["val_acc_curve"]) if s == step), None)
        verdict = "n/a" if acc is None else "PASS" if acc > target else "FAIL"
        print(f"  {config_label(c['optimizer'], c['ghd_mode']):18s} seed {c['seed']:<4d} lr {c['lr']:<6g} "
              f"val@{step} = {'—' if acc is None else f'{acc:.1f}%'}  final = {r['val_acc_curve'][-1]:.1f}%  {verdict}")


def select_jobs(opts, ghds, phases, seeds) -> list[tuple[str, str, int]]:
    return [
        (opt, ghd, seed)
        for opt, ghd in CONFIGS
        if (not opts or opt in opts) and (not ghds or ghd in ghds) and PHASE_OF_GHD[ghd] in phases
        for seed in seeds
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", type=int, choices=[1, 2, 3],
                   help="1 = fixed Exp 1 spec (10 layers, gain 4.0); 2/3 = spec from --model/--dataset/--layers/--gain")
    p.add_argument("--phase", type=int, choices=[1, 2])
    p.add_argument("--train_controller", action="store_true")
    p.add_argument("--lr_sweep", action="store_true", help="re-run the SGD/LARS/LNGD learning-rate sweep")
    p.add_argument("--opt", nargs="+", choices=["sgd", "adamw", "lars", "lngd"])
    p.add_argument("--ghd", nargs="+", choices=["none", "rules", "ai"], help="filter GHD modes")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--steps", type=int, default=None,
                   help=f"training steps (default per optimizer: {N_STEPS_BY_OPT})")
    p.add_argument("--workers", type=int, default=1, help="parallel runs (each uses 1 thread)")
    p.add_argument("--model", choices=MODELS, default="mlp")
    p.add_argument("--dataset", choices=DATASETS, default="mnist")
    p.add_argument("--layers", type=int, default=NUM_LAYERS, help="number of MLP layers")
    p.add_argument("--gain", type=float, default=INIT_GAIN, help="xavier init gain")
    p.add_argument("--results_dir", default=None, help="default: results/ for Exp 1, results/exp{N}/ for Exp 2+")
    p.add_argument("--epochs", type=int, default=50, help="controller training epochs")
    p.add_argument("--quick", action="store_true", help="500 steps, seed 42, SGD only, into results/quick/")
    p.add_argument("--force", action="store_true", help="re-run even if the result JSON exists")
    p.add_argument("--summary", action="store_true", help="print the results table and exit")
    p.add_argument("--quick_check", action="store_true",
                   help="baseline + GHD-Rules only, into results/quick_check/; reports val acc at step 1000")
    args = p.parse_args()

    if args.quick:
        args.steps, args.seeds, args.opt = 500, [42], ["sgd"]
        args.results_dir = str(Path(args.results_dir or "results") / "quick")
        args.exp = 1
    if args.quick_check:
        args.exp, args.phase, args.ghd = args.exp or 1, 1, ["none", "rules"]

    spec = (args.model, args.dataset, args.layers, args.gain)
    if args.exp == 1 and spec != ("mlp", "mnist", NUM_LAYERS, INIT_GAIN):
        p.error(f"Experiment 1 is fixed at mlp/mnist, {NUM_LAYERS} layers, gain {INIT_GAIN:g}; use --exp 2 for other specs")
    if args.exp is not None and args.exp >= 2:
        try:
            check_supported(args.model, args.dataset)
        except NotImplementedError as exc:
            p.error(str(exc))
    # Exp 1 keeps its original file names and 50-step eval; later experiments encode the spec and eval every 10.
    # Without --exp (e.g. --train_controller / --summary alone), every result file in the folder is used.
    args.prefix = EXP1_PREFIX if args.exp == 1 else run_prefix(*spec) if args.exp else None
    args.eval_every = EVAL_EVERY if args.exp in (None, 1) else EVAL_EVERY_NEW
    if args.results_dir is None:
        args.results_dir = f"results/exp{args.exp}" if args.exp and args.exp >= 2 else "results"
    if args.quick_check:
        args.results_dir = str(Path(args.results_dir) / "quick_check")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt = results_dir / "ghd_controller.pt"

    if args.summary:
        print_table(aggregate(load_results(results_dir, prefix=args.prefix)))
        return
    if args.lr_sweep:
        print("=== Learning-rate sweep (500 steps, seed 42) ===")
        lr_sweep(results_dir)
    if args.train_controller:
        train_controller(results_dir, ckpt, args.epochs, args.prefix)
    if args.exp is None:
        if not (args.train_controller or args.lr_sweep):
            p.print_help()
        return

    lock = results_dir / RUNNING_DIR / "experiment.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.touch()
    try:
        run_experiment(args, results_dir, ckpt)
    finally:
        lock.unlink(missing_ok=True)


def run_experiment(args, results_dir: Path, ckpt: Path) -> None:
    phases = [args.phase] if args.phase else [1, 2]
    if args.exp != 1:
        print(f"Experiment {args.exp}: {args.model}/{args.dataset}, {args.layers} layers, gain {args.gain:g}, "
              f"eval every {args.eval_every} steps -> {results_dir}/{args.prefix}_*.json")
    print("Learning rates: " + ", ".join(f"{k}={v:g}" for k, v in OPTIMIZER_LR.items())
          + " | steps: " + (str(args.steps) if args.steps else ", ".join(f"{k}={v}" for k, v in N_STEPS_BY_OPT.items())))
    if 1 in phases:
        print("=== Phase 1: baselines + GHD-Rules ===")
        run_many(select_jobs(args.opt, args.ghd, [1], args.seeds), args, results_dir, ckpt)
    if 2 in phases:
        jobs = select_jobs(args.opt, args.ghd, [2], args.seeds)
        if jobs and args.phase is None and controller_is_stale(results_dir, ckpt, args.prefix):
            print("=== Training GHDController from GHD-Rules logs ===")
            train_controller(results_dir, ckpt, args.epochs, args.prefix)
        if jobs and not ckpt.exists():
            print(f"WARNING: {ckpt} not found; GHD-AI runs will fall back to rule-based decisions.")
        print("=== Phase 2: GHD-AI ===")
        run_many(jobs, args, results_dir, ckpt)

    results = load_results(results_dir, prefix=args.prefix)
    if args.quick_check:
        report_quick_check(results)
    df = aggregate(results)
    print()
    print_table(df)
    if not df.empty:
        df.drop(columns=["seeds"]).to_csv(results_dir / "exp1_summary.csv", index=False)
        print(f"\nSummary written to {results_dir / 'exp1_summary.csv'}")


if __name__ == "__main__":
    main()
