This post shows how Bayesian updating works when we observe new evidence. We compare four different _prior_ beliefs about the probability of success of some event. Priors vary along two dimensions:

1.  **_Location_** – what probability do we believe _a priori_?
2.  **_Strength_** – how strongly do we believe it? (number of pseudo-observations encoded in the Beta prior)

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
```

Let's define some priors. We use the Beta–Binomial model, so a prior is Beta(α, β). Observing `successes` and `failures` yields a posterior Beta(α + successes, β + failures).

```python
priors: list[dict[str, object]] = [
    {
        "label": "Weak, sceptical 10%",  # believe success is ~10 %, but weakly held
        "alpha": 1,
        "beta": 9,
        "color": "tab:red",
    },
    {
        "label": "Strong, sceptical 10%",  # same mean, but 100 pseudo-trials (!)
        "alpha": 10,
        "beta": 90,
        "color": "firebrick",
    },
    {
        "label": "Weak, neutral 50%",  # flat prior
        "alpha": 1,
        "beta": 1,
        "color": "tab:blue",
    },
    {
        "label": "Strong, neutral 50%",  # 100 pseudo-trials centred at 0.5
        "alpha": 50,
        "beta": 50,
        "color": "navy",
    },
]
```

Let's go ahead and plot our priors.

```python
x = np.linspace(0, 1, 1000)

fig_prior, ax_prior = plt.subplots(figsize=(10, 6))
ax_prior.set_title("Step 1 – Prior beliefs about success probability")
ax_prior.set_xlabel("Probability of success p")
ax_prior.set_ylabel("Density")

for prior in priors:
    pdf = stats.beta(prior["alpha"], prior["beta"]).pdf(x)
    ax_prior.plot(x, pdf, label=prior["label"], color=prior["color"])

ax_prior.legend()
ax_prior.grid(alpha=0.3)
fig_prior.tight_layout()
```

![png](output_5_0.png)

Now, let's imagine we gather some new evidence. Let's say we run 20 trials, getting 18 successes and 2 failures.

```python
SUCCESS_COUNT = 18
FAIL_COUNT = 2
```

Let's update our priors with this new information.

```python
for prior in priors:
    # Posterior parameters
    prior["alpha_post"] = prior["alpha"] + SUCCESS_COUNT
    prior["beta_post"] = prior["beta"] + FAIL_COUNT
```

Now, we can plot them.

```python
fig_post, ax_post = plt.subplots(figsize=(10, 6))
ax_post.set_title("Priors and updated posteriors after new information")
ax_post.set_xlabel("Probability of success p")
ax_post.set_ylabel("Density")

for prior in priors:
    # Plot prior (dashed)
    pdf_prior = stats.beta(prior["alpha"], prior["beta"]).pdf(x)
    ax_post.plot(
        x,
        pdf_prior,
        linestyle="--",
        color=prior["color"],
        alpha=0.6,
    )
    # Plot posterior (solid)
    pdf_post = stats.beta(prior["alpha_post"], prior["beta_post"]).pdf(x)
    ax_post.plot(x, pdf_post, linestyle="-", color=prior["color"], label=prior["label"])

ax_post.legend(title="Solid = posterior, dashed = prior")
ax_post.grid(alpha=0.3)
fig_post.tight_layout()
```

![png](output_11_0.png)

```python
# ---------- 6. Visualise mean shift ---------------------------------------------
# For a blog it is often illustrative to highlight how the expected value moves.
# We create a simple arrow plot (or bars) showing prior mean → posterior mean.

fig_shift, ax_shift = plt.subplots(figsize=(10, 4))
ax_shift.set_title("Step 3 – Shift in mean probability after observing data")
ax_shift.set_ylim(0, 1)
ax_shift.set_ylabel("Mean p")
ax_shift.set_xticks(range(len(priors)))
ax_shift.set_xticklabels([p["label"] for p in priors], rotation=20, ha="right")

for idx, prior in enumerate(priors):
    mean_prior = prior["alpha"] / (prior["alpha"] + prior["beta"])
    mean_post = prior["alpha_post"] / (prior["alpha_post"] + prior["beta_post"])

    # Plot prior mean as a dot
    ax_shift.plot(idx, mean_prior, "o", color=prior["color"], markersize=8)
    # Arrow to posterior mean
    ax_shift.annotate(
        "",
        xy=(idx, mean_post),
        xytext=(idx, mean_prior),
        arrowprops=dict(arrowstyle="->", color=prior["color"], lw=2),
    )
    # Posterior mean dot
    ax_shift.plot(idx, mean_post, "s", color=prior["color"], markersize=8)

ax_shift.grid(axis="y", alpha=0.3)
fig_shift.tight_layout()
```

![png](output_12_0.png)

```python
# ---------- 7. Show all figures --------------------------------------------------
plt.show()

# ------------------------------------------------------------------------------
# Blog-post talking points
# ------------------------------------------------------------------------------
# • **Weak vs strong priors** – Notice how the weak priors (α + β small) move much
#   more in response to the 9/10 successes than the strong priors.
# • **Sceptical vs neutral** – Compare the sceptical 10 % priors with the neutral
#   50 % priors: the direction of movement is the same (towards high p), but the
#   starting point is different.
# • **Posterior means** – The final means still differ: strong priors drag the
#   posterior back toward the prior belief, illustrating how information is
#   balanced in Bayesian inference.
```

```python

```
