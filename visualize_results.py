"""Visualize BO experiment results saved by run_experiments.py.

Produces two plots per (dim, q):
  - aggregated_dim{d}_q{q}.png   : mean +/- std of best-value-so-far vs iteration
  - individual_runs_dim{d}_q{q}.png : per-run best-so-far grid (one subplot per run)

"Best so far at iteration k" is the running max over the evaluation budget:
iteration 0 = best of the n_init initial points, iteration k = best over the
init points plus the first k batches of q evaluations. This maps every strategy
(batched, async, or single-point) onto a common 0..n_iterations axis.

Usage:
    python visualize_results.py --data-dir data_from_vm --dim 8 --q 50
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Display style per strategy dir name (order defines plotting order).
STRATEGY_STYLE = {
    "qLogEI":            {"label": "qLogEI",      "color": "#1f77b4", "marker": "o"},
    "JointSequential":   {"label": "JointSeq",    "color": "#ff7f0e", "marker": "v"},
    "KrigingBeliever":   {"label": "KrigingBel",  "color": "#9467bd", "marker": "P"},
    "AsyncSimulation":   {"label": "AsyncSim",    "color": "#d62728", "marker": "s"},
    "SinglePointLonger": {"label": "SinglePoint", "color": "#2ca02c", "marker": "^"},
    "RandomSearch":      {"label": "Random",      "color": "#7f7f7f", "marker": "D"},
}


def find_run_dir(data_dir: Path, strategy: str, dim: int, q: int) -> Path | None:
    """Return the latest timestamp dir for a strategy, or None if absent."""
    base = data_dir / strategy / f"dim={dim}" / f"q={q}"
    if not base.is_dir():
        return None
    timestamps = sorted(p for p in base.iterdir() if p.is_dir())
    return timestamps[-1] if timestamps else None


def best_so_far(y_all, n_init: int, q: int) -> np.ndarray:
    """Running max at iteration 0..n_iterations (0 = init block, then +q each)."""
    y = np.asarray(y_all, dtype=float).ravel()
    n_iters = (len(y) - n_init) // q
    cuts = [n_init + k * q for k in range(n_iters + 1)]
    return np.array([y[:c].max() for c in cuts])


def load_strategy(run_dir: Path):
    """Load summary + per-run curves. Returns (summary, list of (seed, curve))."""
    summary = json.loads((run_dir / "summary.json").read_text())
    n_init, q = summary["n_init"], summary["batch_size"]

    curves = []
    for run_file in sorted(run_dir.glob("run_*.json")):
        run = json.loads(run_file.read_text())
        curves.append((run["run_idx"], run["seed"], best_so_far(run["y_all"], n_init, q)))
    curves.sort(key=lambda t: t[0])
    return summary, curves


def plot_aggregated(loaded, dim, q, n_runs, out_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    for strategy, (summary, curves) in loaded.items():
        style = STRATEGY_STYLE[strategy]
        M = np.vstack([c for _, _, c in curves])          # (n_runs, T)
        x = np.arange(M.shape[1])
        mean, std = M.mean(axis=0), M.std(axis=0)
        ax.plot(x, mean, color=style["color"], marker=style["marker"],
                markersize=6, linewidth=2, label=style["label"])
        ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.2)

    ax.set_xlabel("Iteration", fontweight="bold", fontsize=12)
    ax.set_ylabel("Best Value (mean ± std)", fontweight="bold", fontsize=12)
    ax.set_title(f"Aggregated Results: d={dim}, q={q}, N_runs={n_runs}",
                 fontweight="bold", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_individual(loaded, dim, q, n_runs, out_path):
    cols = 3
    rows = (n_runs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 3.5), squeeze=False)
    axes = axes.ravel()

    # y-limits shared across subplots for comparability
    all_vals = np.concatenate([c for _, curves in loaded.values() for _, _, c in curves])
    ymin, ymax = all_vals.min(), all_vals.max()
    pad = 0.05 * (ymax - ymin)

    seeds_by_run = {}
    for run_idx in range(n_runs):
        ax = axes[run_idx]
        for strategy, (summary, curves) in loaded.items():
            style = STRATEGY_STYLE[strategy]
            match = [(seed, c) for ri, seed, c in curves if ri == run_idx]
            if not match:
                continue
            seed, curve = match[0]
            seeds_by_run[run_idx] = seed
            ax.plot(np.arange(len(curve)), curve, color=style["color"],
                    marker=style["marker"], markersize=4, linewidth=1.5, label=style["label"])
        ax.set_title(f"Run {run_idx + 1} (seed={seeds_by_run.get(run_idx, '?')})", fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Value")
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    for ax in axes[n_runs:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize BO experiment results")
    parser.add_argument("--data-dir", default="data_from_vm", type=Path,
                        help="Root results dir (default: data_from_vm)")
    parser.add_argument("--dim", "-d", type=int, required=True)
    parser.add_argument("--q", "-q", type=int, required=True)
    parser.add_argument("--out-dir", default="plots", type=Path,
                        help="Where to save PNGs (default: plots)")
    parser.add_argument("--strategies", "-s", nargs="+", default=None,
                        help="Subset of strategy dir names to plot (default: all present)")
    args = parser.parse_args()

    candidates = args.strategies or list(STRATEGY_STYLE.keys())
    loaded = {}
    for strategy in candidates:
        run_dir = find_run_dir(args.data_dir, strategy, args.dim, args.q)
        if run_dir is None:
            if args.strategies:
                print(f"WARNING: no data for {strategy} at dim={args.dim}, q={args.q}")
            continue
        summary, curves = load_strategy(run_dir)
        loaded[strategy] = (summary, curves)
        print(f"Loaded {strategy}: {len(curves)} runs from {run_dir}")

    if not loaded:
        raise SystemExit(f"No results found under {args.data_dir} for dim={args.dim}, q={args.q}")

    n_runs = max(len(curves) for _, curves in loaded.values())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    agg_path = args.out_dir / f"aggregated_dim{args.dim}_q{args.q}.png"
    ind_path = args.out_dir / f"individual_runs_dim{args.dim}_q{args.q}.png"
    plot_aggregated(loaded, args.dim, args.q, n_runs, agg_path)
    plot_individual(loaded, args.dim, args.q, n_runs, ind_path)
    print(f"Saved:\n  {agg_path}\n  {ind_path}")


if __name__ == "__main__":
    main()
