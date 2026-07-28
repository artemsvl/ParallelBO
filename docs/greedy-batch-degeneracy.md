# Greedy batch construction at large q: what breaks, and what to use instead

Scope: `dim=8`, `q=50`, Ackley (negated), BoTorch 0.16.1 / GPyTorch 1.15.2 / torch 2.11.0.
All numbers measured 2026-07-28. Reproduce with `diagnose_greedy_degeneracy.py`.

## TL;DR

1. `sequential=True` does **not** rescue the joint qLogEI optimizer — it hits the same
   L-BFGS-B failure (129 retries in iteration 1 alone, 15 seeds, on the VM).
2. Output standardization is **already on by default** in BoTorch 0.16.1, so "add
   `Standardize`" is a no-op, and the `e^-45` flat-surface story does not hold here
   (iteration-1 log-EI is -1.8).
3. The gradient-shadowing explanation for the failures is **measured and refuted**;
   `tau_max` is not the lever.
4. The failures are **platform-specific** — 0 retries across 5 seeds on ARM/Accelerate,
   ~17% on x86 — so `joint_sequential`'s wall-clock is not portable.
5. Use **`run_kriging_believer`** instead: ~21x cheaper for the same batch, 0 retries, and
   it isolates the study's variable of interest more cleanly than `joint_sequential` does.
   Caveats: at the *run* level the shared GP fit dilutes that to ~3-4x, and KB visibly
   over-exploits (plateaus after iteration 5 on seed 42).

## Why there is a `joint_sequential` arm at all

The study's open question is why `AsyncSimulation` beats joint `qLogEI` batches. Async
differs from the joint arm in *two* ways at once, and they are confounded:

1. it builds the batch **sequentially** (point k is chosen knowing points 0..k-1), and
2. it conditions on the **true evaluated** y of those earlier points.

`joint_sequential` was added to isolate (1): same acquisition, same budget, but the
q x d joint problem replaced by q d-dimensional ones via `optimize_acqf(..., sequential=True)`,
which attaches each accepted candidate as `X_pending`. If sequential construction alone
explained async's edge, this arm should close most of the gap.

It does not cleanly answer the question, for the reasons below.

## Finding 1 — `sequential=True` does not rescue the joint optimizer

The earlier diagnostic (`diagnose_joint_optimizer.py`, 2026-07-24) established **Case A** at
`dim=8`, `q=8/16`: the joint optimizer *numerically* fails at later rounds — an
async-prefix batch scores several log-units higher under joint qLogEI's *own* acquisition
than the batch the joint optimizer found. Warm-starting from that prefix recovered nearly
the whole gap, so the failure is initialization/basin, not raw budget.

The natural hope was that `sequential=True` sidesteps this by never posing the
high-dimensional joint problem. **It does not.** In the very first BO iteration of the
`q=50, d=8` cloud sweep (15 seeds), the parent log recorded:

- **129** `ABNORMAL_TERMINATION_IN_LNSRCH` retries from `scipy.optimize.minimize`
- of which **68** *also* failed on the retry ("Optimization failed on the second try,
  after generating a new set of initial conditions")

That is ~17% of the 750 (15 seeds x 50 candidates) subproblems in a single iteration.
So replacing one q*d-dimensional problem with q d-dimensional ones does **not** escape the
conditioning problem. Whatever is degenerate about this acquisition surface is degenerate
for the greedy formulation too — which is the substantive point for the writeup: the
Case A result should not be read as "the joint optimizer is bad, use greedy instead."

## Finding 2 — output standardization is a red herring

An earlier note in this project claimed the models are unstandardized, with late-round
acquisition values at `e^-10..e^-45`, and that standardizing would fix the flat surface.
**That is not true for the installed BoTorch.** In 0.16.1, `SingleTaskGP.__init__` has
`outcome_transform=DEFAULT`, which resolves to `Standardize(m=1)`:

```
raw Y mean/std:                          -20.646 / 0.470
model.train_targets mean/std (internal):   2.5e-15 / 1.000
```

Verified identical on the VM (same library versions). Passing `outcome_transform=Standardize(m=1)`
explicitly is a **no-op**. Correspondingly the acquisition values are *not* degenerate at
iteration 1 — the first subproblem's log-EI is **-1.795**, and over 50 greedy steps it rises
monotonically toward 0 (`-1.795 -> -0.817 -> -0.667 -> -0.603`), because batch EI grows as
members are added. There is no `e^-45` to fix here.

## Finding 3 — the flat-surface / gradient-shadowing hypothesis is also wrong

The plausible mechanism was gradient shadowing: qLogEI smooth-maxes over batch members with
temperature `tau_max=0.01`, so with 49 competitors already pending, the one free candidate's
share of that max could underflow, flattening the surface in x. Measured directly
(median gradient norm over 64 probe points, `dim=8`):

| pending p | median ‖∇ₓ qLogEI‖ (tau=0.01) | fraction < 1e-12 | median (tau=1.0) |
|---|---|---|---|
| 0  | 3.98 | 0.0% | 6.44 |
| 10 | 0.82 | 0.0% | 1.64 |
| 30 | 0.50 | 0.0% | 0.55 |
| 49 | 0.58 | 0.0% | 0.75 |

Gradients stay O(0.5) out to p=49 and **no** probe underflows; `tau_max` barely matters.
Hypothesis refuted. `tau_max` is still exposed on `run_joint_sequential` so the sweep is
cheap to redo, but it is not the lever.

Relatedly, per-subproblem cost does **not** blow up with p: ~4.8 s at p=9, 19 and 29. (Later
readings drift to ~8-19 s, but that is contention from concurrent benchmark processes, not p.)
`joint_sequential`'s ~4-6 min first iteration is simply 50 subproblems of a few seconds each —
there is no super-linear degeneracy in p to fix.

## Finding 4 — the retry storm is platform-specific, not seed-specific

Running the full 50-candidate greedy batch on the laptop, `dim=8 q=50`, seeds 42-46:

| seed | wall | subproblems with >=1 retry | total retries |
|---|---|---|---|
| 42 | 364.7 s | **0** / 50 | **0** |
| 43 | 234.7 s | **0** / 50 | **0** |
| 44 | 378.2 s | **0** / 50 | **0** |
| 46 | 315.5 s | **0** / 50 | **0** |

(Walls vary because these ran concurrently; seed 45 likewise showed 0 through k=39.)
So **zero** failures on Apple/Accelerate across five seeds and 250 subproblems, against
~17% on the VM with the *same* botorch 0.16.1 / gpytorch 1.15.2 /
torch 2.11.0 and the same `Standardize` default. The VM figure kept climbing — **268** retries
by 1h52m — and in that time it completed **0 of 300** iterations across its 15 runs, because
every retry re-runs `raw_samples=1024` plus a fresh L-BFGS.

The remaining differences are BLAS (Accelerate/ARM vs MKL-family/x86) and the scipy build.
`ABNORMAL_TERMINATION_IN_LNSRCH` is precisely the failure that flips on rounding, so a
platform-dependent line-search failure is the leading explanation — but it is **not proven**.

Two consequences:

- **Do not state a mechanism for this in the thesis text yet.**
- More importantly, **wall-clock numbers for `joint_sequential` are not portable across
  machines.** Any runtime claim about it has to name the platform, which makes it a poor
  arm to build a performance argument on.

## The fix: Kriging Believer (`run_kriging_believer`)

Rather than harden a formulation whose failure mode is not understood, use the greedy
construction that has no MC batch acquisition at all. Same sequential structure, different
bookkeeping for already-chosen points:

| | accounts for chosen points via | acquisition | per-iteration GP fits |
|---|---|---|---|
| `joint_sequential` | `X_pending` on q-point MC qLogEI | qLogEI (MC) | 1 |
| `kriging_believer` | `condition_on_observations(x, posterior mean)` | LogEI (**analytic**) | 1 |
| `async_simulation` | `condition on true evaluated y` (refits) | LogEI (analytic) | **q** |

Conditioning is exact — feeding back the raw-scale posterior mean leaves it unchanged and
shrinks the posterior sd as it should (verified: `mu` -18.0085 -> -18.0085 exactly,
`sd` 0.508 -> 0.099, train set 20 -> 21), so the outcome transform is handled correctly.

Measured on one laptop core, `dim=8 q=50`, identical dataset and seed — one full
50-candidate batch:

| arm | one batch of q=50 | retries |
|---|---|---|
| `joint_sequential` (`X_pending` + MC qLogEI) | 364.7 s | 0 (laptop) / ~17% (VM) |
| `kriging_believer` (fantasy + analytic LogEI) | **17.2 s** | **0** |

**~21x cheaper for the same batch**, with no line-search failures on either platform. Its
log-EI trace also has the right sign of behaviour: it *decreases* as fantasies accumulate
(`-1.795 -> -2.363 -> -2.666`), i.e. each believed point genuinely consumes remaining
improvement potential, whereas the `X_pending` formulation's value *rises* toward 0
(`-1.795 -> -0.817 -> -0.603 -> -0.532`) because batch EI grows with membership.

**The 21x does not carry over to a whole run.** Per-iteration cost grows steeply with N
because the single GP fit — which both arms pay equally — comes to dominate:

| iteration | N | wall |
|---|---|---|
| 1 | 74 | 19 s |
| 6 | 324 | 36 s |
| 13 | 674 | 96 s |
| 15 | 774 | 184 s |
| 17 | 874 | 148 s |

By N~800 the fit is ~180 s of a ~185 s iteration and the acquisition work KB saves is
almost free anyway. So the honest framing is: **KB removes the acquisition-side cost
(21x on that component) but leaves the fit-side cost untouched**, giving roughly 3-4x on a
full 20-iteration run rather than 21x. The batch-level number should not be quoted as a
run-level speedup.

That in turn identifies the *next* bottleneck: `fit_gpytorch_mll_torch` (Adam) is the repo's
fitter, where BoTorch's default is `fit_gpytorch_mll` (scipy L-BFGS-B). At n=24 the two
produce near-identical hyperparameters, so swapping is likely safe and is the obvious next
thing to benchmark at large N — it would speed up *every* arm, not just this one.

Optimisation quality, seed 42: it reaches **-3.93 by iteration 5** — already past `qLogEI`'s
*final* 20-iteration mean of -4.55 — and then **plateaus there for at least 12 more
iterations** with no improvement at all. Fast early progress followed by a hard stall is the
textbook signature of Kriging Believer's known over-exploitation: believing the posterior
mean discards predictive variance, so the fantasy batch collapses toward the current
incumbent and stops exploring.

This is a *modelling* limitation, not a numerical defect, and it is the main thing to weigh
before adopting KB as the headline arm. Constant Liar (believe a pessimistic constant instead
of the mean) or `qLogNEI` are the standard remedies and are cheap to add as companion arms.
One seed is not a result — run the 15-seed sweep before drawing conclusions.

### Why this is the better scientific instrument

Kriging Believer is the *clean* isolation the study wanted. Compare it against
`async_simulation`: both build the batch sequentially with analytic LogEI and one
hyperparameter fit; they differ **only** in whether the conditioning value is the GP's
predicted mean or the true objective evaluation. So a gap between them measures the value of
**real information** with nothing else confounded — which is precisely the question
`joint_sequential` was meant to answer but cannot, because it also changes the acquisition
(MC qLogEI vs analytic LogEI) and inherits an unexplained numerical failure.

Caveat to state in the writeup: Kriging Believer is known to be over-confident — believing
the mean ignores predictive variance and tends to over-exploit, which is the classic argument
for Constant Liar or for `qLogEI`/`qLogNEI`. That is a *modelling* limitation to discuss, not
a numerical defect.

## What is still open

Why x86/MKL trips the line search and ARM/Accelerate does not. The cheap next probe is to run
`diagnose_greedy_degeneracy.py` directly on the VM (it reports per-candidate retries, so the
onset in k is immediately visible) and to try the `scipy_fit` variant there, since it is the
acquisition optimizer — not the GP fit — that is failing.

Also unresolved, and independent of all the above: `run_async_simulation` refits GP
hyperparameters inside its per-worker loop (q refits per iteration instead of 1), which is
what makes it ~10 h per run at q=50. That is an implementation artifact, not a property of
asynchronous BO, and `run_kriging_believer` shows the same sequential structure costs minutes
when the fit is hoisted. Async's wall-clock should not be reported as-is.

## Reproducing

```bash
source .venv/bin/activate

# mechanism diagnostics: per-candidate retries, log-EI, timing
python diagnose_greedy_degeneracy.py --dim 8 --q 50 \
    --variants torch_fit scipy_fit kriging_believer tau_1.0

# the new arm, alongside the existing ones
python run_experiments.py -d 8 -q 50 -r 15 -i 20 -s kriging_believer
```
