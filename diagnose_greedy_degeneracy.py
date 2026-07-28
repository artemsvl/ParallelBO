"""Locate the real cause of L-BFGS failures in greedy (sequential=True) batch qLogEI.

`optimize_acqf(..., sequential=True)` is a loop: pick candidate k by maximizing the
acquisition with candidates 0..k-1 attached as `X_pending`, then append and repeat.
This script reimplements that loop explicitly so every subproblem can be instrumented,
then reports, per candidate index k, the L-BFGS retry count and achieved log-EI.

Variants compared (each on an identical dataset / seed):
  torch_fit   - repo default: fit_gpytorch_mll_torch (Adam)
  scipy_fit   - BoTorch default: fit_gpytorch_mll (L-BFGS-B, multi-retry)
  scipy_qlognei - scipy fit + qLogNoisyEI (X_baseline instead of best_f)

Usage:
    python diagnose_greedy_degeneracy.py --dim 8 --q 50
"""
import argparse
import time
import warnings

import torch
from botorch.acquisition import (
    LogExpectedImprovement,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from parallel_bo import create_objective, draw_init_points, unit_bounds

dtype = torch.double


def build_model(x, y, fitter):
    model = SingleTaskGP(x, y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fitter(mll)
    return model


def model_report(model, y):
    """Fitted hyperparameters, on the model's internal (standardized) scale."""
    ls = model.covar_module.lengthscale.detach().flatten()
    return {
        "noise": model.likelihood.noise.item(),
        "outputscale": getattr(model.covar_module, "outputscale", torch.tensor(float("nan"))).item()
        if hasattr(model.covar_module, "outputscale") else float("nan"),
        "ls_min": ls.min().item(),
        "ls_med": ls.median().item(),
        "ls_max": ls.max().item(),
        "mll_per_pt": None,
    }


def greedy_loop(x_all, y_all, dim, q, fitter, acqf_name, tau_max=0.01, log_every=10):
    """Explicit reimplementation of optimize_acqf(sequential=True), instrumented.

    acqf_name == "kb" is Kriging Believer: instead of attaching the chosen point as
    X_pending on a q-point MC acquisition, condition the GP on its posterior *mean*
    and keep using the analytic 1-point LogEI. No smooth-max over a growing batch,
    so no gradient shadowing.
    """
    model = build_model(x_all, y_all, fitter)
    hp = model_report(model, y_all)

    bounds = unit_bounds(dim)
    chosen = torch.empty(0, dim, dtype=dtype)
    per_k = []
    kb_model, kb_best = model, y_all.max().item()

    for k in range(q):
        pending = chosen if k > 0 else None
        if acqf_name == "qlogei":
            acqf = qLogExpectedImprovement(
                model=model, best_f=y_all.max().item(), X_pending=pending,
                tau_max=tau_max,
            )
        elif acqf_name == "kb":
            acqf = LogExpectedImprovement(model=kb_model, best_f=kb_best)
        else:
            acqf = qLogNoisyExpectedImprovement(
                model=model, X_baseline=x_all, X_pending=pending, prune_baseline=True,
                tau_max=tau_max,
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            cand, acq_val = optimize_acqf(
                acq_function=acqf, bounds=bounds, q=1,
                num_restarts=10, raw_samples=1024,
            )
            dt = time.perf_counter() - t0

        retries = sum("Trying again" in str(w.message) or "second try" in str(w.message)
                      for w in caught)
        per_k.append({"k": k, "retries": retries, "logei": acq_val.item(), "sec": dt})
        chosen = torch.cat([chosen, cand], dim=0)

        if acqf_name == "kb":
            # Fantasize: treat the posterior mean at `cand` as if observed.
            with torch.no_grad():
                y_fant = kb_model.posterior(cand).mean
            kb_model = kb_model.condition_on_observations(cand, y_fant)
            kb_best = max(kb_best, y_fant.max().item())

        if (k + 1) % log_every == 0 or k == 0:
            print(f"    k={k:3d}  p_pending={k:3d}  logEI={acq_val.item():10.3f}  "
                  f"retries={retries}  {dt:6.2f}s")

    return hp, per_k


def summarize(name, hp, per_k, total_sec):
    n_fail = sum(r["retries"] > 0 for r in per_k)
    tot_retry = sum(r["retries"] for r in per_k)
    logeis = [r["logei"] for r in per_k]
    # onset: first k where a retry occurs
    onset = next((r["k"] for r in per_k if r["retries"] > 0), None)
    print(f"\n  [{name}]  {total_sec:.1f}s total")
    print(f"    fitted: noise={hp['noise']:.3e} outputscale={hp['outputscale']:.3f} "
          f"lengthscale min/med/max={hp['ls_min']:.3f}/{hp['ls_med']:.3f}/{hp['ls_max']:.3f}")
    print(f"    subproblems with >=1 retry: {n_fail}/{len(per_k)}   total retries: {tot_retry}")
    print(f"    first failing k: {onset}")
    print(f"    logEI  first={logeis[0]:.3f}  median={sorted(logeis)[len(logeis)//2]:.3f}  "
          f"last={logeis[-1]:.3f}  min={min(logeis):.3f}")
    return {"name": name, "sec": total_sec, "n_fail": n_fail, "retries": tot_retry,
            "onset": onset, "logei_last": logeis[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", "-d", type=int, default=8)
    ap.add_argument("--q", "-q", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", nargs="+",
                    default=["torch_fit", "scipy_fit", "scipy_qlognei"])
    args = ap.parse_args()

    torch.set_num_threads(1)
    objective = create_objective(args.dim)
    n_init = 3 * args.dim
    x_all = draw_init_points(n_init, args.dim, args.seed)
    y_all = objective(x_all).unsqueeze(-1)
    print(f"dim={args.dim} q={args.q} n_init={n_init}  "
          f"y range [{y_all.min():.3f}, {y_all.max():.3f}] std={y_all.std():.3f}\n")

    specs = {
        # name              fitter                   acqf       tau_max
        "torch_fit":     (fit_gpytorch_mll_torch, "qlogei",  0.01),
        "scipy_fit":     (fit_gpytorch_mll,       "qlogei",  0.01),
        "scipy_qlognei": (fit_gpytorch_mll,       "qlognei", 0.01),
        "tau_0.1":       (fit_gpytorch_mll_torch, "qlogei",  0.1),
        "tau_1.0":       (fit_gpytorch_mll_torch, "qlogei",  1.0),
        "kriging_believer": (fit_gpytorch_mll_torch, "kb",   0.01),
    }

    rows = []
    for name in args.variants:
        fitter, acqf_name, tau = specs[name]
        print(f"--- {name} (tau_max={tau}) ---")
        t0 = time.perf_counter()
        hp, per_k = greedy_loop(x_all, y_all, args.dim, args.q, fitter, acqf_name, tau_max=tau)
        rows.append(summarize(name, hp, per_k, time.perf_counter() - t0))
        print()

    print("=" * 76)
    print(f"{'variant':16s} {'sec':>8s} {'fail/q':>9s} {'retries':>8s} {'onset k':>8s} {'logEI[-1]':>10s}")
    for r in rows:
        print(f"{r['name']:16s} {r['sec']:8.1f} {r['n_fail']:5d}/{args.q:<3d} "
              f"{r['retries']:8d} {str(r['onset']):>8s} {r['logei_last']:10.3f}")


if __name__ == "__main__":
    main()
