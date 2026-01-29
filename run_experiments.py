import torch
import json
from pathlib import Path
from datetime import datetime
from parallel_bo import (
    create_objective,
    run_qlogei_optimization,
    run_async_simulation,
    run_random_search
)

def run_experiment(
    strategy: str,
    dim: int,
    batch_size: int,
    n_init: int,
    n_iterations: int,
    n_runs: int
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    strategy_map = {
        "qlogei": "qLogEI",
        "async_simulation": "AsyncSimulation",
        "random_search": "RandomSearch"
    }
    strategy_name = strategy_map.get(strategy, strategy)
    save_dir = Path(f"data/{strategy_name}/dim={dim}/q={batch_size}/{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Running {strategy_name} with d={dim}, q={batch_size}")
    print(f"Saving results to: {save_dir}")
    print(f"{'='*70}\n")

    all_results = []

    objective_fn = create_objective(dim)

    for run_idx in range(n_runs):
        seed = 42 + run_idx
        print(f"\n--- Run {run_idx + 1}/{n_runs} (seed={seed}) ---")

        if strategy == "qlogei":
            x_all, y_all, best_value = run_qlogei_optimization(
                objective_fn=objective_fn,
                dim=dim,
                n_init=n_init,
                n_iterations=n_iterations,
                batch_size=batch_size,
                seed=seed
            )
        elif strategy == "async_simulation":
            x_all, y_all, best_value = run_async_simulation(
                objective_fn=objective_fn,
                dim=dim,
                n_init=n_init,
                n_iterations=n_iterations,
                batch_size=batch_size,
                seed=seed
            )
        elif strategy == "random_search":
            x_all, y_all, best_value = run_random_search(
                objective_fn=objective_fn,
                dim=dim,
                n_init=n_init,
                n_iterations=n_iterations,
                batch_size=batch_size,
                seed=seed
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        run_result = {
            "run_idx": run_idx,
            "seed": seed,
            "best_value": best_value,
            "n_evaluations": len(x_all),
            "x_all": x_all.cpu().tolist(),
            "y_all": y_all.cpu().tolist()
        }

        run_file = save_dir / f"run_{run_idx:03d}.json"
        with open(run_file, 'w') as f:
            json.dump(run_result, f, indent=2)

        all_results.append(run_result)
        print(f"Best value: {best_value:.4f}")

    best_values = [r["best_value"] for r in all_results]
    summary = {
        "strategy": strategy_name,
        "dim": dim,
        "batch_size": batch_size,
        "n_init": n_init,
        "n_iterations": n_iterations,
        "n_runs": n_runs,
        "timestamp": timestamp,
        "best_values": best_values,
        "mean_best": sum(best_values) / len(best_values),
        "std_best": (sum((x - sum(best_values)/len(best_values))**2 for x in best_values) / len(best_values))**0.5,
        "min_best": min(best_values),
        "max_best": max(best_values)
    }

    summary_file = save_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Summary for {strategy_name} (d={dim}, q={batch_size}):")
    print(f"Mean best value: {summary['mean_best']:.4f} ± {summary['std_best']:.4f}")
    print(f"Range: [{summary['min_best']:.4f}, {summary['max_best']:.4f}]")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*70}\n")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Bayesian Optimization experiments')
    parser.add_argument('--dim', '-d', type=int, default=1, help='Dimension of the problem')
    parser.add_argument('--batch_size', '-q', type=int, default=50, help='Batch size (default: 50)')
    parser.add_argument('--n_runs', '-r', type=int, default=30, help='Number of runs (default: 30)')
    parser.add_argument('--n_iterations', '-i', type=int, default=20, help='Number of iterations (default: 20)')

    args = parser.parse_args()

    dim = args.dim
    q = args.batch_size
    n_init = 3 * q

    print(f"\nStarting experiments with d={dim}, q={q}, n_runs={args.n_runs}, n_iterations={args.n_iterations}")
    print("="*70)

    run_experiment(
        strategy="qlogei",
        dim=dim,
        batch_size=q,
        n_init=n_init,
        n_iterations=args.n_iterations,
        n_runs=args.n_runs
    )

    run_experiment(
        strategy="async_simulation",
        dim=dim,
        batch_size=q,
        n_init=n_init,
        n_iterations=args.n_iterations,
        n_runs=args.n_runs
    )

    print("\nAll experiments completed!")
