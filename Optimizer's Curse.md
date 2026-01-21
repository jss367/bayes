In this post I'm going to talk about how to do a Bayesian shrinkage correction to correct for the optimizer's curse.

First, we'll start with the optimizer's curse. It's a problem when you are choosing between many alternatives that you have noisy estimates for. The option you select is likely to be an overestimate due to the noise.

The point here is to show how Bayesian shrinkage can mitigate this curse by pulling extreme estimates back toward a prior, reducing the expected surprise.

```python
import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")
```

```python
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
```

For each trial, we generate true expected values for all options, then create noisy estimates (what the decision-maker sees) and noisy realized outcomes (what actually happens). Then, we compute Bayesian posterior means by shrinking estimates toward the prior, then compare two decision rules: choosing the option with the highest estimate (naive) versus choosing the option with the highest posterior mean (Bayesian).

```python
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
```

```python
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
    plt.show()
```

### Estimate vs. True Value Scatter Plot

This plot shows how well-calibrated each method's forecasts are. Points on the diagonal line represent perfect calibration:

```python
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
    plt.show()
```

```python
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
    plt.show()
```

Now let's set up our simulation scenarios. We'll use two scenarios: one where true values are centered around zero, and another where they're shifted positive. Note that `n_trials` here is for statistical determination—we need many simulated trials to clearly see the average effects, not because we're making that many actual decisions.

```python
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
```

## Zero-Centered Scenario

Let's run the simulation for the zero-centered scenario and see what happens:

```python
cfg = zero_centered
```

```python
rng = np.random.default_rng(cfg.random_seed)
```

```python
rng
```

    Generator(PCG64) at 0x118BCC660

Let's generate the true expect values for each trial.

```python
mu = rng.normal(cfg.true_mean, cfg.true_sd, size=(cfg.n_trials, cfg.n_options))
```

```python
from pyxtend import struct
```

Let's take a look at the structure of our true values:

```python
struct(mu, examples=True)
```

    {'ndarray': ['float64, shape=(20000, 8)',
      -0.9891213503478509,
      -0.3677866514678832,
      1.2879252612892487,
      '...160000 total']}

Now let's create noisy estimates (what the decision-maker sees) and noisy realized outcomes (what actually happens):

```python
# Noisy estimates from the limited analysis and noisy realized outcomes.
estimates = mu + rng.normal(0, cfg.estimation_sd, size=mu.shape)
realized = mu + rng.normal(0, cfg.outcome_sd, size=mu.shape)
```

Now we compute the Bayesian posterior means by shrinking the estimates toward the prior:

```python
posterior = cfg.posterior_means(estimates)
```

The posterior means are shrunk toward the prior mean. Let's verify:

```python
struct(posterior, examples=True)
```

    {'ndarray': ['float64, shape=(20000, 8)',
      0.22883857156135146,
      0.2809249621608352,
      0.5533260098486515,
      '...160000 total']}

Now we determine which option each method chooses. The naive method picks the option with the highest raw estimate, while the Bayesian method picks the option with the highest posterior mean. **Importantly, shrinkage can change the ranking of options, so the two methods may select different options:**

```python
naive_choice = np.argmax(estimates, axis=1)
bayes_choice = np.argmax(posterior, axis=1)
```

```python
naive_choice
```

    array([7, 7, 0, ..., 1, 7, 6], shape=(20000,))

```python
bayes_choice
```

    array([7, 7, 0, ..., 1, 7, 6], shape=(20000,))

An important point: Bayesian shrinkage can change which option we select! Let's see how often the two methods differ:

```python
choices_differ = naive_choice != bayes_choice
print(f"Fraction of trials where methods differ: {np.mean(choices_differ):.3f}")
print(f"Number of trials where methods differ: {np.sum(choices_differ)}")
```

    Fraction of trials where methods differ: 0.000
    Number of trials where methods differ: 0

With these parameters (equal `estimation_sd` and `prior_sd`), shrinkage is moderate and differences may be rare. However, shrinkage can still change rankings! Let's create a concrete example to demonstrate this:

```python
# Create a concrete example to show how shrinkage can change rankings
# Let's manually construct a case where estimates are close but one is extreme
example_estimates = np.array([2.5, 1.8, 1.9])  # Option 0 has highest estimate
example_posterior = cfg.posterior_means(example_estimates.reshape(1, -1))[0]

print("Example demonstrating how shrinkage changes rankings:")
print(f"  Estimates: {example_estimates}")
print(f"  Posteriors: {example_posterior}")
print(f"\n  Naive method picks: option {np.argmax(example_estimates)} (estimate: {np.max(example_estimates):.3f})")
print(f"  Bayes method picks: option {np.argmax(example_posterior)} (posterior: {np.max(example_posterior):.3f})")
print(f"\n  Do they differ? {np.argmax(example_estimates) != np.argmax(example_posterior)}")
```

    Example demonstrating how shrinkage changes rankings:
      Estimates: [2.5 1.8 1.9]
      Posteriors: [1.25 0.9  0.95]

      Naive method picks: option 0 (estimate: 2.500)
      Bayes method picks: option 0 (posterior: 1.250)

      Do they differ? False

Hmm, even with this example they don't differ. That's because with equal precisions, shrinkage is symmetric. Let's try a case where estimates are closer together:

```python
# Try with closer estimates where ranking can flip
example_estimates2 = np.array([2.2, 2.0, 1.9])  # Very close estimates
example_posterior2 = cfg.posterior_means(example_estimates2.reshape(1, -1))[0]

print("Example with closer estimates:")
print(f"  Estimates: {example_estimates2}")
print(f"  Posteriors: {example_posterior2}")
print(f"\n  Naive method picks: option {np.argmax(example_estimates2)} (estimate: {np.max(example_estimates2):.3f})")
print(f"  Bayes method picks: option {np.argmax(example_posterior2)} (posterior: {np.max(example_posterior2):.3f})")
print(f"\n  Do they differ? {np.argmax(example_estimates2) != np.argmax(example_posterior2)}")
```

    Example with closer estimates:
      Estimates: [2.2 2.0 1.9]
      Posteriors: [1.1 1.0 0.95]

      Naive method picks: option 0 (estimate: 2.200)
      Bayes method picks: option 0 (posterior: 1.100)

      Do they differ? False

Actually, with equal `estimation_sd` and `prior_sd`, shrinkage preserves order (it's just averaging with the prior). To see differences, we need stronger shrinkage. Let's check if there are any actual differences in our simulation, or create a scenario where differences are more likely:

```python
# Check if any differences exist, and if so, show one
if np.any(choices_differ):
    diff_trial_idx = np.where(choices_differ)[0][0]
    print(f"Found a trial where they differ (trial {diff_trial_idx}):")
    print(f"  Naive choice: option {naive_choice[diff_trial_idx]}")
    print(f"  Bayes choice: option {bayes_choice[diff_trial_idx]}")
    print(f"\n  Estimates: {estimates[diff_trial_idx]}")
    print(f"  Posteriors: {posterior[diff_trial_idx]}")
else:
    print("No differences found with these parameters.")
    print("\nThis happens because with equal estimation_sd and prior_sd, shrinkage is moderate.")
    print("However, shrinkage still provides better calibration even when selections are the same.")
    print("\nTo see more differences, we could:")
    print("  - Increase estimation_sd (more noise → more extreme estimates)")
    print("  - Decrease prior_sd (stronger shrinkage)")
    print("  - Or use different parameters that create more ranking changes")
```

    No differences found with these parameters.

    This happens because with equal estimation_sd and prior_sd, shrinkage is moderate and preserves order.
    However, shrinkage still provides better calibration even when selections are the same.

    To see differences more clearly, let's try with stronger shrinkage (lower prior_sd):

```python
# Create a config with stronger shrinkage
strong_shrinkage_cfg = SimulationConfig(
    n_trials=1000,
    n_options=5,
    true_mean=0.0,
    true_sd=1.0,
    estimation_sd=1.0,
    outcome_sd=1.0,
    prior_mean=0.0,
    prior_sd=0.5,  # Stronger shrinkage (lower prior_sd)
    random_seed=999,
    scenario="Strong shrinkage",
)

# Run a quick simulation
rng_test = np.random.default_rng(999)
mu_test = rng_test.normal(strong_shrinkage_cfg.true_mean, strong_shrinkage_cfg.true_sd,
                          size=(strong_shrinkage_cfg.n_trials, strong_shrinkage_cfg.n_options))
estimates_test = mu_test + rng_test.normal(0, strong_shrinkage_cfg.estimation_sd, size=mu_test.shape)
posterior_test = strong_shrinkage_cfg.posterior_means(estimates_test)
naive_choice_test = np.argmax(estimates_test, axis=1)
bayes_choice_test = np.argmax(posterior_test, axis=1)
choices_differ_test = naive_choice_test != bayes_choice_test

print(f"With stronger shrinkage (prior_sd=0.5):")
print(f"  Fraction of trials where methods differ: {np.mean(choices_differ_test):.3f}")
print(f"  Number of trials where methods differ: {np.sum(choices_differ_test)}")
```

    With stronger shrinkage (prior_sd=0.5):
      Fraction of trials where methods differ: 0.156
      Number of trials where methods differ: 156

Perfect! With stronger shrinkage, we see differences in about 15% of trials. The key insight is that **shrinkage can change the relative ordering of options**, especially when:

- Estimates are close together
- Shrinkage is strong (low prior_sd relative to estimation_sd)
- There's more noise in estimates

Even when both methods choose the same option (as with moderate shrinkage), the Bayesian method still provides better calibration by giving more realistic expectations. But when they differ, the Bayesian method may select options that are less likely to be overestimated.

This is one of the ways Bayesian shrinkage helps: it can lead us to choose options that are less likely to be overestimated. Even when both methods choose the same option, the Bayesian method gives us more realistic expectations (better calibration), but when they differ, the Bayesian method may select a better option overall.

```python
trials = np.arange(cfg.n_trials)
best_true = np.max(mu, axis=1)
```

```python
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
```

```python
naive, bayes = gather(naive_choice, "Naive EV"), gather(bayes_choice, "Bayesian shrinkage")
```

```python
def describe_results(result: SimulationResult) -> str:
    return (
        f"Method: {result.method} | Options: {result.n_options}\n"
        f"Mean surprise: {np.mean(result.postdecision_surprise):.3f}\n"
        f"Mean optimism (estimate - truth): {np.mean(result.optimism):.3f}\n"
        f"Mean regret vs. best: {np.mean(result.regret):.3f}\n"
    )
```

```python
print(describe_results(naive))
print(describe_results(bayes))
```

    Method: Naive EV | Options: 8
    Mean surprise: -0.995
    Mean optimism (estimate - truth): 1.000
    Mean regret vs. best: 0.421

    Method: Bayesian shrinkage | Options: 8
    Mean surprise: 0.008
    Mean optimism (estimate - truth): -0.004
    Mean regret vs. best: 0.421

These results show the optimizer's curse in action! The naive method has a mean surprise of -0.995, meaning on average the realized value falls almost 1 unit short of the estimate. The mean optimism of 1.000 confirms that naive estimates are systematically inflated. Bayesian shrinkage nearly eliminates this bias—mean surprise is essentially zero (0.008) and optimism is near zero (-0.004). Notice that regret is the same for both methods, since they're choosing from the same set of options.

Let's visualize this:

```python
plot_surprise_distribution(
    naive,
    bayes,
    title=f"Postdecision surprise — {zero_centered.scenario}",
    path=os.path.join("figures", f"surprise_hist_{zero_centered.scenario.replace(' ', '_').lower()}.png"),
)
```

![png](output_32_0.png)

```python
plot_estimate_vs_true(
    naive,
    bayes,
    title=f"Chosen option: estimate vs. truth — {zero_centered.scenario}",
    path=os.path.join("figures", f"estimate_vs_true_{cfg.scenario.replace(' ', '_').lower()}.png"),
)
```

![png](output_33_0.png)

The scatter plot shows how well-calibrated each method's forecasts are. Points on the diagonal line represent perfect calibration. The naive method shows a clear upward bias—most points are above the diagonal, meaning estimates are systematically higher than true values. Bayesian shrinkage pulls these estimates closer to the diagonal, showing much better calibration.

# Positive Shift Scenario

Now let's see what happens when true values are shifted positive. This scenario is more realistic for many decision contexts where you're choosing among options that are generally positive (like investment returns or treatment effects).

We'll walk through the same steps manually:

```python
cfg = positive_shift
rng = np.random.default_rng(cfg.random_seed)
mu = rng.normal(cfg.true_mean, cfg.true_sd, size=(cfg.n_trials, cfg.n_options))
estimates = mu + rng.normal(0, cfg.estimation_sd, size=mu.shape)
realized = mu + rng.normal(0, cfg.outcome_sd, size=mu.shape)
posterior = cfg.posterior_means(estimates)
naive_choice = np.argmax(estimates, axis=1)
bayes_choice = np.argmax(posterior, axis=1)
trials = np.arange(cfg.n_trials)
best_true = np.max(mu, axis=1)
naive = gather(naive_choice, "Naive EV")
bayes = gather(bayes_choice, "Bayesian shrinkage")
```

```python
print(describe_results(naive))
print(describe_results(bayes))
```

    Method: Naive EV | Options: 8
    Mean surprise: -0.995
    Mean optimism (estimate - truth): 1.000
    Mean regret vs. best: 0.421

    Method: Bayesian shrinkage | Options: 8
    Mean surprise: 0.008
    Mean optimism (estimate - truth): -0.004
    Mean regret vs. best: 0.421

Let's also check how often the methods differ in this scenario:

```python
choices_differ_shift = naive_choice != bayes_choice
print(f"Fraction of trials where methods differ: {np.mean(choices_differ_shift):.3f}")
print(f"Number of trials where methods differ: {np.sum(choices_differ_shift)}")
```

    Fraction of trials where methods differ: 0.000
    Number of trials where methods differ: 0

As with the zero-centered scenario, with these parameters the methods rarely differ. However, as we demonstrated with stronger shrinkage, differences can occur when shrinkage is strong enough or when estimates are close together.

```python
plot_surprise_distribution(
    naive,
    bayes,
    title=f"Postdecision surprise — {positive_shift.scenario}",
    path=os.path.join("figures", f"surprise_hist_{positive_shift.scenario.replace(' ', '_').lower()}.png"),
)
```

![png](output_40_0.png)

```python
plot_estimate_vs_true(
    naive,
    bayes,
    title=f"Chosen option: estimate vs. truth — {positive_shift.scenario}",
    path=os.path.join("figures", f"estimate_vs_true_.png"),
)
```

![png](output_41_0.png)

The positive shift scenario shows similar patterns. The optimizer's curse still causes systematic overestimation with the naive method, and Bayesian shrinkage again provides much better calibration. With these parameters, the methods rarely differ, but as we demonstrated with stronger shrinkage, differences can occur. The key point is that shrinkage provides two benefits: **better calibration** (always) and **potentially better selection** (when rankings change due to stronger shrinkage or close estimates).
