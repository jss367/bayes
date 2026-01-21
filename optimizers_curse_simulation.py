"""
Visual demonstrations of the optimizer's curse and a Bayesian shrinkage correction.

This script simulates decisions among multiple alternatives when the analyst only has
noisy estimates of each option's expected value. The option with the highest estimated
expected value is chosen, which creates an optimism bias: the selected estimate is more
likely to be an overestimate than an underestimate. The script also shows how a simple
Bayesian update shrinks extreme estimates back toward a prior and reduces the expected
"postdecision surprise" (the realized value minus the estimate).

Key scenarios covered:
- "Zero-centered" decisions where true expected values cluster around zero.
- "Positive-shifted" decisions where the average true value is above zero.

Running the script saves multiple figures into the ``figures`` directory that visualize
how the curse intensifies with more alternatives and how Bayesian shrinkage helps.
"""

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")


@dataclass
class SimulationConfig:
    """Configuration for a single scenario simulation."""

    n_trials: int
    n_options: int
    true_mean: float
    true_sd: float
    estimation_sd: float
    outcome_sd: float
    prior_mean: float
    prior_sd: float
    random_seed: int = 0
    scenario: str = ""

    def posterior_means(self, estimates: np.ndarray) -> np.ndarray:
        """Compute posterior means given estimates under a normal-normal model.

        Prior: mu ~ N(prior_mean, prior_sd^2)
        Likelihood: estimate | mu ~ N(mu, estimation_sd^2)
        Posterior mean formula is analytic for conjugate normals.
        """

        prior_precision = 1 / (self.prior_sd**2)
        likelihood_precision = 1 / (self.estimation_sd**2)
        posterior_variance = 1 / (prior_precision + likelihood_precision)
        return posterior_variance * (prior_precision * self.prior_mean + likelihood_precision * estimates)


@dataclass
class SimulationResult:
    method: str
    scenario: str
    n_options: int
    chosen_true: np.ndarray
    chosen_estimate: np.ndarray
    chosen_realized: np.ndarray
    chosen_posterior: np.ndarray
    forecast_value: np.ndarray
    best_true: np.ndarray

    @property
    def postdecision_surprise(self) -> np.ndarray:
        return self.chosen_realized - self.forecast_value

    @property
    def optimism(self) -> np.ndarray:
        return self.forecast_value - self.chosen_true

    @property
    def regret(self) -> np.ndarray:
        return self.best_true - self.chosen_true


# ---- Simulation helpers ----------------------------------------------------


def simulate_decisions(cfg: SimulationConfig) -> tuple[SimulationResult, SimulationResult]:
    """Simulate repeated decisions and return outcomes for naïve and Bayesian choices."""

    rng = np.random.default_rng(cfg.random_seed)

    # True expected values for each trial/option.
    mu = rng.normal(cfg.true_mean, cfg.true_sd, size=(cfg.n_trials, cfg.n_options))

    # Noisy estimates from the limited analysis and noisy realized outcomes.
    estimates = mu + rng.normal(0, cfg.estimation_sd, size=mu.shape)
    realized = mu + rng.normal(0, cfg.outcome_sd, size=mu.shape)

    posterior = cfg.posterior_means(estimates)

    # Determine chosen indices.
    naive_choice = np.argmax(estimates, axis=1)
    bayes_choice = np.argmax(posterior, axis=1)

    trials = np.arange(cfg.n_trials)
    best_true = np.max(mu, axis=1)

    def gather(indices: np.ndarray, method: str) -> SimulationResult:
        chosen_true = mu[trials, indices]
        chosen_estimate = estimates[trials, indices]
        chosen_realized = realized[trials, indices]
        chosen_posterior = posterior[trials, indices]
        forecast_value = chosen_estimate if method == "Naive EV" else chosen_posterior
        return SimulationResult(
            method=method,
            scenario=cfg.scenario,
            n_options=cfg.n_options,
            chosen_true=chosen_true,
            chosen_estimate=chosen_estimate,
            chosen_realized=chosen_realized,
            chosen_posterior=chosen_posterior,
            forecast_value=forecast_value,
            best_true=best_true,
        )

    return gather(naive_choice, "Naive EV"), gather(bayes_choice, "Bayesian shrinkage")


def aggregate_bias_by_k(
    ks: Iterable[int], base_cfg: SimulationConfig, n_trials: int
) -> list[tuple[int, float, float, float, float]]:
    """Run a sweep over option counts and capture mean surprise/regret.

    Returns a list of tuples ``(k, surprise_naive, surprise_bayes, regret_naive, regret_bayes)``.
    """

    records: list[tuple[int, float, float, float, float]] = []
    for k in ks:
        cfg = SimulationConfig(
            n_trials=n_trials,
            n_options=k,
            true_mean=base_cfg.true_mean,
            true_sd=base_cfg.true_sd,
            estimation_sd=base_cfg.estimation_sd,
            outcome_sd=base_cfg.outcome_sd,
            prior_mean=base_cfg.prior_mean,
            prior_sd=base_cfg.prior_sd,
            random_seed=base_cfg.random_seed,
            scenario=base_cfg.scenario,
        )
        naive, bayes = simulate_decisions(cfg)
        records.append(
            (
                k,
                float(np.mean(naive.postdecision_surprise)),
                float(np.mean(bayes.postdecision_surprise)),
                float(np.mean(naive.regret)),
                float(np.mean(bayes.regret)),
            )
        )
    return records


# ---- Plotting helpers ------------------------------------------------------


def plot_surprise_distribution(naive: SimulationResult, bayes: SimulationResult, title: str, path: str) -> None:
    plt.figure(figsize=(8, 5))
    bins = 40
    plt.hist(
        naive.postdecision_surprise,
        bins=bins,
        alpha=0.6,
        density=True,
        label="Naive EV",
        color="#d95f02",
    )
    plt.hist(
        bayes.postdecision_surprise,
        bins=bins,
        alpha=0.6,
        density=True,
        label="Bayesian shrinkage",
        color="#1b9e77",
    )
    plt.axvline(np.mean(naive.postdecision_surprise), color="#d95f02", linestyle="--")
    plt.axvline(np.mean(bayes.postdecision_surprise), color="#1b9e77", linestyle="--")
    plt.xlabel("Realized value minus estimated value (postdecision surprise)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_estimate_vs_true(
    naive: SimulationResult, bayes: SimulationResult, title: str, path: str, sample: int = 4000
) -> None:
    rng = np.random.default_rng(42)
    idxs_naive = rng.choice(len(naive.chosen_true), size=min(sample, len(naive.chosen_true)), replace=False)
    idxs_bayes = rng.choice(len(bayes.chosen_true), size=min(sample, len(bayes.chosen_true)), replace=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(
        naive.forecast_value[idxs_naive],
        naive.chosen_true[idxs_naive],
        s=10,
        alpha=0.35,
        label="Naive EV",
        color="#d95f02",
    )
    plt.scatter(
        bayes.forecast_value[idxs_bayes],
        bayes.chosen_true[idxs_bayes],
        s=10,
        alpha=0.35,
        label="Bayesian shrinkage",
        color="#1b9e77",
    )
    lims = plt.axis()
    mn = min(lims[0], lims[2])
    mx = max(lims[1], lims[3])
    plt.plot([mn, mx], [mn, mx], color="black", linestyle=":", linewidth=1)
    plt.axis([mn, mx, mn, mx])
    plt.xlabel("Estimated expected value of chosen option")
    plt.ylabel("True expected value of chosen option")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_bias_curves(
    records_zero: list[tuple[int, float, float, float, float]],
    records_shift: list[tuple[int, float, float, float, float]],
    path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    panels = [
        (records_zero, "Zero-centered true values"),
        (records_shift, "Positive true values"),
    ]

    for ax, (records, title) in zip(axes, panels):
        ks = [r[0] for r in records]
        surprise_naive = [r[1] for r in records]
        surprise_bayes = [r[2] for r in records]
        regret_naive = [r[3] for r in records]
        regret_bayes = [r[4] for r in records]

        ax.plot(ks, surprise_naive, marker="o", label="Surprise (naive)", color="#d95f02")
        ax.plot(ks, surprise_bayes, marker="o", label="Surprise (Bayes)", color="#1b9e77")
        ax.plot(ks, regret_naive, marker="s", linestyle="--", label="Regret (naive)", color="#7570b3")
        ax.plot(ks, regret_bayes, marker="s", linestyle="--", label="Regret (Bayes)", color="#e7298a")
        ax.set_xlabel("Number of alternatives evaluated")
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.8)

    axes[0].set_ylabel("Mean surprise/regret")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Optimizer's curse intensifies with more options; Bayesian shrinkage softens it")
    fig.tight_layout(rect=[0, 0, 0.86, 0.95])
    fig.savefig(path)
    plt.close(fig)


def describe_results(result: SimulationResult) -> str:
    return (
        f"Method: {result.method} | Options: {result.n_options}\n"
        f"Mean surprise: {np.mean(result.postdecision_surprise):.3f}\n"
        f"Mean optimism (estimate - truth): {np.mean(result.optimism):.3f}\n"
        f"Mean regret vs. best: {np.mean(result.regret):.3f}\n"
    )


# ---- Main script -----------------------------------------------------------


def main() -> None:
    os.makedirs("figures", exist_ok=True)

    zero_centered = SimulationConfig(
        n_trials=20000,
        n_options=8,
        true_mean=0.0,
        true_sd=1.0,
        estimation_sd=1.0,
        outcome_sd=1.0,
        prior_mean=0.0,
        prior_sd=1.0,
        random_seed=123,
        scenario="Zero-centered",
    )

    positive_shift = SimulationConfig(
        n_trials=20000,
        n_options=8,
        true_mean=1.0,
        true_sd=1.0,
        estimation_sd=1.0,
        outcome_sd=1.0,
        prior_mean=0.5,
        prior_sd=1.0,
        random_seed=456,
        scenario="Positive shift",
    )

    scenarios = [zero_centered, positive_shift]

    for cfg in scenarios:
        naive, bayes = simulate_decisions(cfg)
        print(f"\nScenario: {cfg.scenario}")
        print(describe_results(naive))
        print(describe_results(bayes))

        plot_surprise_distribution(
            naive,
            bayes,
            title=f"Postdecision surprise — {cfg.scenario}",
            path=os.path.join("figures", f"surprise_hist_{cfg.scenario.replace(' ', '_').lower()}.png"),
        )
        plot_estimate_vs_true(
            naive,
            bayes,
            title=f"Chosen option: estimate vs. truth — {cfg.scenario}",
            path=os.path.join("figures", f"estimate_vs_true_{cfg.scenario.replace(' ', '_').lower()}.png"),
        )

    # How the curse grows with more options.
    ks = [2, 3, 5, 8, 12, 20, 30]
    zero_records = aggregate_bias_by_k(ks, zero_centered, n_trials=8000)
    shift_records = aggregate_bias_by_k(ks, positive_shift, n_trials=8000)

    plot_bias_curves(
        zero_records,
        shift_records,
        path=os.path.join("figures", "bias_vs_k.png"),
    )


if __name__ == "__main__":
    main()
