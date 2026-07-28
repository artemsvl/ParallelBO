import torch
from botorch.acquisition import qLogExpectedImprovement, LogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def unit_bounds(dim):
    bounds = torch.tensor([[0] * dim, [1] * dim], dtype=dtype, device=device)

    return bounds

def create_objective(dim: int):
    ackley = Ackley(dim=dim, negate=True)
    bounds = torch.tensor([[-20] * dim, [20] * dim])

    def scaled_ackley(x):
        x_scaled = unnormalize(x, bounds)
        return ackley(x_scaled)

    return scaled_ackley

def draw_init_points(n: int, dim: int, seed: int = 42):
    torch.manual_seed(seed)
    return torch.rand(n, dim, dtype=dtype, device=device)

def run_qlogei_optimization(
    objective_fn,
    dim: int,
    n_init: int,
    n_iterations: int,
    batch_size: int,
    seed: int = 42
):
    x_init = draw_init_points(n_init, dim, seed)
    y_init = objective_fn(x_init).unsqueeze(-1)

    x_all = x_init.clone()
    y_all = y_init.clone()

    for iteration in range(n_iterations):
        model = SingleTaskGP(x_all, y_all)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll_torch(mll)

        qlogei = qLogExpectedImprovement(
            model=model,
            best_f=y_all.max().item()
        )

        candidates, acq_value = optimize_acqf(
            acq_function=qlogei,
            bounds=unit_bounds(dim),
            q=batch_size,
            num_restarts=10,
            raw_samples=1024,
        )

        y_new = objective_fn(candidates).unsqueeze(-1)

        x_all = torch.cat([x_all, candidates], dim=0)
        y_all = torch.cat([y_all, y_new], dim=0)

        print(f"Iteration {iteration + 1}/{n_iterations}: Best value = {y_all.max().item():.4f}")

    best_value = y_all.max().item()

    return x_all, y_all, best_value


def run_joint_sequential(
    objective_fn,
    dim: int,
    n_init: int,
    n_iterations: int,
    batch_size: int,
    seed: int = 42,
    tau_max: float = 0.01,
):
    """Greedy batch BO: qLogEI optimized one candidate at a time.

    Same acquisition and evaluation budget as `run_qlogei_optimization`, but the
    q x d joint problem is replaced by q sequential d-dimensional problems via
    `optimize_acqf(..., sequential=True)`. BoTorch appends each accepted
    candidate to `X_pending`, so later candidates account for earlier ones
    through the model's fantasized (not evaluated) values.

    NOTE: at large q this arm is numerically unreliable, and `sequential=True` does
    NOT avoid the joint optimizer's conditioning problem. On the dim=8 / q=50 cloud
    sweep, iteration 1 alone produced 129 ABNORMAL_TERMINATION_IN_LNSRCH retries
    across 15 seeds (68 of which also failed on the retry) — ~17% of subproblems.
    The root cause is not yet established: it does not reproduce on Apple/Accelerate
    across seeds 42-46, so it is likely platform/BLAS dependent. `tau_max` (qLogEI's
    smooth-max temperature) is exposed for sweeping, but measured gradients stay
    healthy out to 49 pending points, so it is probably not the lever. See
    docs/greedy-batch-degeneracy.md; prefer `run_kriging_believer`. Kept unchanged
    by default so it remains an honest "naive sequential=True" baseline.
    """
    x_init = draw_init_points(n_init, dim, seed)
    y_init = objective_fn(x_init).unsqueeze(-1)

    x_all = x_init.clone()
    y_all = y_init.clone()

    for iteration in range(n_iterations):
        model = SingleTaskGP(x_all, y_all)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll_torch(mll)

        qlogei = qLogExpectedImprovement(
            model=model,
            best_f=y_all.max().item(),
            tau_max=tau_max,
        )

        candidates, acq_value = optimize_acqf(
            acq_function=qlogei,
            bounds=unit_bounds(dim),
            q=batch_size,
            num_restarts=10,
            raw_samples=1024,
            sequential=True,
        )

        y_new = objective_fn(candidates).unsqueeze(-1)

        x_all = torch.cat([x_all, candidates], dim=0)
        y_all = torch.cat([y_all, y_new], dim=0)

        print(f"Iteration {iteration + 1}/{n_iterations}: Best value = {y_all.max().item():.4f}, "
              f"Total points = {len(x_all)}")

    best_value = y_all.max().item()

    return x_all, y_all, best_value


def run_kriging_believer(
    objective_fn,
    dim: int,
    n_init: int,
    n_iterations: int,
    batch_size: int,
    seed: int = 42
):
    """Greedy batch BO via Kriging Believer — the numerically sound sequential arm.

    Like `run_joint_sequential` this builds the batch one point at a time, but it
    swaps the mechanism used to account for already-chosen points:

      joint_sequential : attach them as `X_pending` on a q-point MC qLogEI
      kriging_believer : condition the GP on their posterior *mean* and keep
                         using the analytic 1-point LogEI

    Three consequences, all of them wins:

    1. Each subproblem is an ordinary 1-point analytic LogEI on a growing model —
       no MC acquisition over a q-point batch, so none of the numerical trouble
       `run_joint_sequential` hits at large q (0 retries observed).
    2. Analytic LogEI has a closed form, no MC sampling, so a subproblem costs
       ~0.1-0.2s against 1.2-7.3s for a q-point qLogEI evaluation.
    3. Hyperparameters are fitted once per iteration and reused via
       `condition_on_observations`, exactly as in `run_async_simulation`, but
       without that function's 50 redundant refits per iteration.

    Against `run_async_simulation` this isolates the variable the study cares
    about: both construct the batch sequentially, but async conditions on the
    *true evaluated* y while this conditions on the *predicted* y. Any performance
    gap between the two is therefore attributable to information content, not to
    sequential-vs-joint construction.
    """
    x_init = draw_init_points(n_init, dim, seed)
    y_init = objective_fn(x_init).unsqueeze(-1)

    x_all = x_init.clone()
    y_all = y_init.clone()

    for iteration in range(n_iterations):
        model = SingleTaskGP(x_all, y_all)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll_torch(mll)

        # Fantasy model: grows by one point per candidate, hyperparameters frozen.
        fantasy = model
        best_f = y_all.max().item()
        candidates = []

        for _ in range(batch_size):
            logei = LogExpectedImprovement(model=fantasy, best_f=best_f)

            candidate, acq_value = optimize_acqf(
                acq_function=logei,
                bounds=unit_bounds(dim),
                q=1,
                num_restarts=10,
                raw_samples=1024,
            )

            # Believe the posterior mean, then condition on it as a pseudo-observation.
            with torch.no_grad():
                y_believed = fantasy.posterior(candidate).mean
            fantasy = fantasy.condition_on_observations(candidate, y_believed)
            best_f = max(best_f, y_believed.max().item())

            candidates.append(candidate)

        candidates = torch.cat(candidates, dim=0)

        # Only now are the real evaluations spent — one batch of q, same budget.
        y_new = objective_fn(candidates).unsqueeze(-1)

        x_all = torch.cat([x_all, candidates], dim=0)
        y_all = torch.cat([y_all, y_new], dim=0)

        print(f"Iteration {iteration + 1}/{n_iterations}: Best value = {y_all.max().item():.4f}, "
              f"Total points = {len(x_all)}")

    best_value = y_all.max().item()

    return x_all, y_all, best_value


def run_async_simulation(
    objective_fn,
    dim: int,
    n_init: int,
    n_iterations: int,
    batch_size: int,
    seed: int = 42
):
    x_init = draw_init_points(n_init, dim, seed)
    y_init = objective_fn(x_init).unsqueeze(-1)

    x_all = x_init.clone()
    y_all = y_init.clone()

    for iteration in range(n_iterations):
        iteration_x = []
        iteration_y = []

        # First iteration: use batch acquisition like qLogEI
        if iteration == 0:
            model = SingleTaskGP(x_all, y_all)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll_torch(mll)

            qlogei = qLogExpectedImprovement(
                model=model,
                best_f=y_all.max().item()
            )

            candidates, acq_value = optimize_acqf(
                acq_function=qlogei,
                bounds=unit_bounds(dim),
                q=batch_size,
                num_restarts=10,
                raw_samples=1024,
            )

            y_new = objective_fn(candidates).unsqueeze(-1)

            x_all = torch.cat([x_all, candidates], dim=0)
            y_all = torch.cat([y_all, y_new], dim=0)

            print(f"Iteration 1/{n_iterations}: Best value = {y_all.max().item():.4f}, "
                  f"Total points = {len(x_all)}")

        # Subsequent iterations: async simulation
        else:
            n_at_iteration_start = len(x_all)
            for worker_idx in range(batch_size):
                # Worker k sees: full dataset before last iteration + k results from previous iteration
                n_points_to_use = n_at_iteration_start - batch_size + worker_idx + 1

                x_train = x_all[:n_points_to_use]
                y_train = y_all[:n_points_to_use]

                model = SingleTaskGP(x_train, y_train)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll_torch(mll)

                logei = LogExpectedImprovement(
                    model=model,
                    best_f=y_train.max().item()
                )

                candidate, acq_value = optimize_acqf(
                    acq_function=logei,
                    bounds=unit_bounds(dim),
                    q=1,
                    num_restarts=10,
                    raw_samples=1024,
                )

                y_new = objective_fn(candidate).unsqueeze(-1)

                x_all = torch.cat([x_all, candidate], dim=0)
                y_all = torch.cat([y_all, y_new], dim=0)

                iteration_x.append(candidate)
                iteration_y.append(y_new)

            print(f"Iteration {iteration + 1}/{n_iterations}: Best value = {y_all.max().item():.4f}, "
                  f"Total points = {len(x_all)}")

    best_value = y_all.max().item()

    return x_all, y_all, best_value


def run_single_point_longer(
    objective_fn,
    dim: int,
    n_init: int,
    n_iterations: int,
    batch_size: int,
    seed: int = 42
):
    # Match the evaluation budget of batched methods: q points per iteration.
    total_iterations = n_iterations * batch_size

    x_init = draw_init_points(n_init, dim, seed)
    y_init = objective_fn(x_init).unsqueeze(-1)

    x_all = x_init.clone()
    y_all = y_init.clone()

    for iteration in range(total_iterations):
        model = SingleTaskGP(x_all, y_all)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll_torch(mll)

        qlogei = qLogExpectedImprovement(
            model=model,
            best_f=y_all.max().item()
        )

        candidate, acq_value = optimize_acqf(
            acq_function=qlogei,
            bounds=unit_bounds(dim),
            q=1,
            num_restarts=10,
            raw_samples=1024,
        )

        y_new = objective_fn(candidate).unsqueeze(-1)
        x_all = torch.cat([x_all, candidate], dim=0)
        y_all = torch.cat([y_all, y_new], dim=0)

        if iteration % batch_size == 0:
            print(
                f"Iteration {iteration + 1}/{total_iterations}: "
                f"Best value = {y_all.max().item():.4f}, Total points = {len(x_all)}"
            )

    best_value = y_all.max().item()

    return x_all, y_all, best_value


def run_random_search(
    objective_fn,
    dim: int,
    n_init: int,
    n_iterations: int,
    batch_size: int,
    seed: int = 42
):
    torch.manual_seed(seed)

    x_init = draw_init_points(n_init, dim, seed)
    y_init = objective_fn(x_init).unsqueeze(-1)

    x_all = x_init.clone()
    y_all = y_init.clone()

    for iteration in range(n_iterations):
        # Sample random points in [0, 1]^d
        candidates = torch.rand(batch_size, dim, dtype=dtype, device=device)
        y_new = objective_fn(candidates).unsqueeze(-1)

        x_all = torch.cat([x_all, candidates], dim=0)
        y_all = torch.cat([y_all, y_new], dim=0)

        print(f"Iteration {iteration + 1}/{n_iterations}: Best value = {y_all.max().item():.4f}, "
              f"Total points = {len(x_all)}")

    best_value = y_all.max().item()

    return x_all, y_all, best_value
