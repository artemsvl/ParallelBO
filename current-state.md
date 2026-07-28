# Batch Bayesian Optimization via Asynchronous Simulation — Current State

## 1. Topic & why we care

Bayesian Optimization (BO) is the standard tool for optimizing expensive black-box
functions. When we have parallel compute, we want to evaluate a **batch of `q` points
per round** instead of one. The classic recipe (q-EI / q-UCB) selects the batch by
jointly maximizing a single acquisition function over all `q` points at once — this
turns a `d`-dimensional optimization into a `q × d`-dimensional one. That joint problem
is hard: it scales badly and tends to get stuck in suboptimal local maxima.

We investigate a simpler alternative — **Asynchronous Simulation**. Instead of solving
one `q × d` problem, we solve `q` cheap `d`-dimensional problems in sequence, where each
new point is chosen conditioned on the **actual evaluated values** of earlier points in
the batch. Unlike greedy batch methods (Kriging Believer / fantasy-based, Wu & Frazier),
we never plug in imputed "fantasy" values — we use ground-truth history, which should
remove the noise those heuristics introduce. The framing follows Riegler et al. (2025),
"Standard Acquisition is All You Need" for async BO.

**The question that makes this a thesis:** *do we actually need joint batch acquisition
functions at all, or does a cheaper sequential/async scheme match or beat them?*

## 2. What we learned so far

**Setup.** Ackley benchmark, unit cube `[0,1]^d` unnormalized to `[-20,20]^d`,
BoTorch/GPyTorch, `n_init = 3·d`. Four strategies implemented and compared
(`parallel_bo.py`): joint batch **qLogEI**, **Async Simulation** (proposed),
**Single-Point** (LogEI sequential, matched evaluation budget), and **Random Search**
baseline. Runs collected over `d ∈ {2,4,8,16}` and `q ∈ {2,3,4,50}`, 15 seeds each.

- **Async ≥ joint batch at large `q`.** On 4-D Ackley with `q=50`, Async Simulation is
  consistently at least as good as joint qLogEI, and **per-run** it is clearly dominant
  even when the mean best-value is comparable. Both strongly beat random search.
- **Spatial convergence differs.** Using the distance-to-optimum metric for acquired
  points (`batch_points_dist`, inspired by "We Still Don't Understand HD-BO"), the two
  methods show distinct spatial patterns — the async batch spreads/converges differently
  than the joint batch.
- **Single-Point as the "north star".** The fully-sequential single-point method (each
  point chosen with complete knowledge of all previous evaluations, matched budget of
  `n_iter × q` evaluations) is the ceiling our async approach is trying to reach — async
  is a parallel-friendly approximation of it. It gives us the *potential maximum* of the
  async idea.
- **Infra:** stood up a cloud cluster to run `d`/`q` sweeps unattended, and parallelized
  runs across a process pool.
- **Reading (scoped out for now):** *Vanilla BO Performs Great in High Dimensions*
  (prior scaling — interesting, out of scope) and *We Still Don't Understand HD-BO*
  (spherical linear transform — out of scope, but we borrowed its distance metrics).

## 3. Questions we want to answer next

1. **Where is the critical point?** Sweep `q` (and `d`) to characterize the regime
   boundary: *for small `q`, async is no worse than joint batch; for large `q`, async is
   usually better.* If that holds, we can pose the sharp question — **do we actually need
   batch acquisition functions?**
2. **Performance under a memory budget.** Joint batch quality depends on `raw_samples`;
   fewer samples → less likely to find the acqf optimum, which is realistic since memory
   is finite. Plan: report **best value after N iterations vs. available memory
   (8/16/32… GB)** for each method — a practically meaningful axis for the thesis.
3. **Why does this happen?** Explain the results, at least via intuition/heuristics and
   implicit evidence (e.g. the spatial-distance patterns), even if not a full proof.
4. **Truly parallel imitation (not the current loop).** Today the async simulation is
   implemented as a sequential `for`-loop over workers. Set up a genuinely parallelized
   harness — batch acqf vs. async with real concurrent workers — to gather proper
   statistics on the trade-off **performance vs. wall-clock time vs. memory**, rather than
   only best-value. This is what lets us make quantitative time/memory claims.
5. **Broaden reading:** *Maximizing Acquisition Functions for BO*, *Standard GP Is All
   You Need for High-Dimensional BO*.
6. **Admin:** officially register the thesis topic with the university.
