# Limit per-process BLAS threads BEFORE importing torch, so each pool worker
# uses a single thread. We get parallelism from running many processes at once
# (one per (strategy, seed) job), not from intra-op threads — which avoids
# oversubscribing the CPU when `workers` processes run concurrently.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import argparse
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from parallel_bo import (
    create_objective,
    run_qlogei_optimization,
    run_async_simulation,
    run_single_point_longer,
    run_random_search,
)

STRATEGY_FNS = {
    "qlogei": run_qlogei_optimization,
    "async_simulation": run_async_simulation,
    "single_point_longer": run_single_point_longer,
    "random_search": run_random_search,
}

STRATEGY_NAMES = {
    "qlogei": "qLogEI",
    "async_simulation": "AsyncSimulation",
    "single_point_longer": "SinglePointLonger",
    "random_search": "RandomSearch",
}

# Strategies run by default (mirrors the previous hardcoded set).
DEFAULT_STRATEGIES = ["qlogei", "async_simulation", "single_point_longer"]


def _init_worker():
    """Pin each pool worker to a single torch thread."""
    torch.set_num_threads(1)


def _run_single_job(strategy, dim, n_init, n_iterations, batch_size, run_idx, seed, save_dir_str):
    """Run one (strategy, seed) job in a worker process.

    The verbose per-iteration output is redirected to a per-run log file so the
    parent's console feed stays clean. The run's JSON is written into the
    parent-assigned save_dir (one dir per strategy), avoiding timestamp races.
    """
    save_dir = Path(save_dir_str)
    log_dir = save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{run_idx:03d}.log"

    fn = STRATEGY_FNS[strategy]
    objective_fn = create_objective(dim)

    with open(log_file, "w") as lf, redirect_stdout(lf):
        print(f"=== {STRATEGY_NAMES[strategy]} run={run_idx} seed={seed} "
              f"(dim={dim}, q={batch_size}) ===", flush=True)
        x_all, y_all, best_value = fn(
            objective_fn=objective_fn,
            dim=dim,
            n_init=n_init,
            n_iterations=n_iterations,
            batch_size=batch_size,
            seed=seed,
        )

    run_result = {
        "run_idx": run_idx,
        "seed": seed,
        "best_value": best_value,
        "n_evaluations": len(x_all),
        "x_all": x_all.cpu().tolist(),
        "y_all": y_all.cpu().tolist(),
    }
    run_file = save_dir / f"run_{run_idx:03d}.json"
    with open(run_file, "w") as f:
        json.dump(run_result, f, indent=2)

    return {
        "strategy": strategy,
        "run_idx": run_idx,
        "seed": seed,
        "best_value": best_value,
        "n_evaluations": len(x_all),
    }


def _write_summary(strategy, save_dir, results, dim, batch_size, n_init, n_iterations, n_runs, timestamp):
    """Aggregate a strategy's per-run results into summary.json (parent side)."""
    results = sorted(results, key=lambda r: r["run_idx"])
    best_values = [r["best_value"] for r in results]
    strategy_name = STRATEGY_NAMES[strategy]

    if not best_values:
        print(f"WARNING: no successful runs for {strategy_name}; skipping summary.")
        return None

    mean_best = sum(best_values) / len(best_values)
    std_best = (sum((x - mean_best) ** 2 for x in best_values) / len(best_values)) ** 0.5

    summary = {
        "strategy": strategy_name,
        "dim": dim,
        "batch_size": batch_size,
        "n_init": n_init,
        "n_iterations": n_iterations,
        "effective_n_iterations": n_iterations * batch_size if strategy == "single_point_longer" else n_iterations,
        "n_runs": n_runs,
        "n_successful_runs": len(best_values),
        "timestamp": timestamp,
        "best_values": best_values,
        "mean_best": mean_best,
        "std_best": std_best,
        "min_best": min(best_values),
        "max_best": max(best_values),
    }

    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{strategy_name} (d={dim}, q={batch_size}): "
          f"mean best = {mean_best:.4f} ± {std_best:.4f} "
          f"[{min(best_values):.4f}, {max(best_values):.4f}] over {len(best_values)} runs")
    print(f"  -> {save_dir}")
    return summary


def run_sweep(strategies, dim, batch_size, n_init, n_iterations, n_runs, workers):
    """Run all (strategy, seed) jobs across a process pool and aggregate."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parent assigns one save_dir per strategy up front (single timestamp), so
    # parallel workers write into a shared, stable directory per strategy.
    save_dirs = {}
    for strategy in strategies:
        d = Path(f"data/{STRATEGY_NAMES[strategy]}/dim={dim}/q={batch_size}/{timestamp}")
        d.mkdir(parents=True, exist_ok=True)
        save_dirs[strategy] = d

    jobs = [
        (strategy, run_idx, 42 + run_idx)
        for strategy in strategies
        for run_idx in range(n_runs)
    ]
    total = len(jobs)

    print(f"Launching {total} jobs ({len(strategies)} strategies x {n_runs} runs) "
          f"across {workers} workers (1 thread each).")
    print(f"Timestamp: {timestamp} | per-run logs under data/.../{timestamp}/logs/\n")

    results_by_strategy = {strategy: [] for strategy in strategies}
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        futures = {
            ex.submit(_run_single_job, strategy, dim, n_init, n_iterations,
                      batch_size, run_idx, seed, str(save_dirs[strategy])):
                (strategy, run_idx, seed)
            for (strategy, run_idx, seed) in jobs
        }
        for fut in as_completed(futures):
            strategy, run_idx, seed = futures[fut]
            done += 1
            try:
                res = fut.result()
                results_by_strategy[strategy].append(res)
                print(f"[{done}/{total}] {STRATEGY_NAMES[strategy]} "
                      f"run={run_idx} seed={seed} -> best={res['best_value']:.4f}", flush=True)
            except Exception as e:
                print(f"[{done}/{total}] {STRATEGY_NAMES[strategy]} "
                      f"run={run_idx} seed={seed} FAILED: {e!r}", flush=True)

    print(f"\n{'='*70}\nSummaries:")
    for strategy in strategies:
        _write_summary(strategy, save_dirs[strategy], results_by_strategy[strategy],
                       dim, batch_size, n_init, n_iterations, n_runs, timestamp)
    print(f"{'='*70}\nAll experiments completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Bayesian Optimization experiments (parallel)')
    parser.add_argument('--dim', '-d', type=int, default=1, help='Dimension of the problem')
    parser.add_argument('--batch_size', '-q', type=int, default=50, help='Batch size (default: 50)')
    parser.add_argument('--n_runs', '-r', type=int, default=30, help='Number of runs (default: 30)')
    parser.add_argument('--n_iterations', '-i', type=int, default=20, help='Number of iterations (default: 20)')
    parser.add_argument('--workers', '-w', type=int, default=os.cpu_count(),
                        help='Parallel worker processes (default: all logical CPUs)')
    parser.add_argument('--strategies', '-s', nargs='+', default=DEFAULT_STRATEGIES,
                        choices=list(STRATEGY_FNS.keys()),
                        help=f'Strategies to run (default: {DEFAULT_STRATEGIES})')

    args = parser.parse_args()

    dim = args.dim
    q = args.batch_size
    n_init = 3 * dim

    print(f"\nStarting experiments: d={dim}, q={q}, n_runs={args.n_runs}, "
          f"n_iterations={args.n_iterations}, workers={args.workers}")
    print("=" * 70)

    run_sweep(
        strategies=args.strategies,
        dim=dim,
        batch_size=q,
        n_init=n_init,
        n_iterations=args.n_iterations,
        n_runs=args.n_runs,
        workers=args.workers,
    )
