import torch
from botorch.acquisition import qLogExpectedImprovement, LogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
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
        fit_gpytorch_mll(mll)

        qlogei = qLogExpectedImprovement(
            model=model,
            best_f=y_all.max().item()
        )

        candidates, acq_value = optimize_acqf(
            acq_function=qlogei,
            bounds=unit_bounds(dim),
            q=batch_size,
            num_restarts=10,
            raw_samples=1000 * dim,
        )

        y_new = objective_fn(candidates).unsqueeze(-1)

        x_all = torch.cat([x_all, candidates], dim=0)
        y_all = torch.cat([y_all, y_new], dim=0)

        print(f"Iteration {iteration + 1}/{n_iterations}: Best value = {y_all.max().item():.4f}")

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
            fit_gpytorch_mll(mll) # TODO: there is also ..._torch version of that

            qlogei = qLogExpectedImprovement(
                model=model,
                best_f=y_all.max().item()
            )

            candidates, acq_value = optimize_acqf(
                acq_function=qlogei,
                bounds=unit_bounds(dim),
                q=batch_size,
                num_restarts=10,
                raw_samples=1000 * dim,
            )

            y_new = objective_fn(candidates).unsqueeze(-1)

            x_all = torch.cat([x_all, candidates], dim=0)
            y_all = torch.cat([y_all, y_new], dim=0)

        # Subsequent iterations: async simulation
        else:
            for worker_idx in range(batch_size):
                current_n = len(x_all)
                # Dynamically expose evaluated points to workers
                n_points_to_use = current_n - batch_size + worker_idx + 1

                x_train = x_all[:n_points_to_use]
                y_train = y_all[:n_points_to_use]

                model = SingleTaskGP(x_train, y_train)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)

                logei = LogExpectedImprovement(
                    model=model,
                    best_f=y_train.max().item()
                )

                candidate, acq_value = optimize_acqf(
                    acq_function=logei,
                    bounds=unit_bounds(dim),
                    q=1,
                    num_restarts=10,
                    raw_samples=1000 * dim,
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
