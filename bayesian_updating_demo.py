# Bayesian Updating Demonstration
# -------------------------------------------------------------
# This standalone script shows, step-by-step, how Bayesian updating
# works when we observe new evidence.  We compare four different
# *prior* beliefs about the probability of success of some event.
#
# Priors vary along two dimensions:
#   1. ***Location***  – what probability do we believe *a priori*?
#   2. ***Strength***  – how strongly do we believe it? (number of
#                        pseudo-observations encoded in the Beta prior)
#
# We use the Beta–Binomial model, so a prior is Beta(α, β).  Observing
# `successes` and `failures` yields a posterior Beta(α + successes,
# β + failures).
#
# Blog-post-friendly features
# --------------------------
# • Lots of inline comments explaining each step.
# • Multiple figures so you can show them one-by-one:
#     1. Priors only.
#     2. Priors *and* posteriors overlaid.
#     3. A bar/arrow plot highlighting how the posterior means moved.
# • No external dependencies beyond matplotlib, numpy and scipy.
# -------------------------------------------------------------

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# ---------- 1. Define Our Priors ------------------------------------------------
# Each prior is a dict with human-readable label, Beta parameters and colour.
# "strength" is roughly α + β – the number of pseudo-trials you assume *a priori*.
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
    {
        "label": "Strong, optimistic 95%",  # believe success is ~95 %, stronger than the data
        "alpha": 95,
        "beta": 5,
        "color": "tab:green",
    },
]

# ---------- 2. New Evidence ------------------------------------------------------
# Imagine we run 10 trials and see 9 successes – very strong evidence success
# probability is *high*.
SUCCESS_COUNT = 9
FAIL_COUNT = 1  # 10 – SUCCESS_COUNT

# ---------- 3. Plot priors -------------------------------------------------------
# We plot the density of each prior Beta distribution.

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

# ---------- 4. Compute posteriors ------------------------------------------------
# Update each prior with the new data.
for prior in priors:
    # Posterior parameters
    prior["alpha_post"] = prior["alpha"] + SUCCESS_COUNT
    prior["beta_post"] = prior["beta"] + FAIL_COUNT

# ---------- 5. Plot priors + posteriors overlay ---------------------------------
fig_post, ax_post = plt.subplots(figsize=(10, 6))
ax_post.set_title("Step 2 – Priors and updated posteriors after 9/10 successes")
ax_post.set_xlabel("Probability of success p")
ax_post.set_ylabel("Density")

# Vertical reference line at the observed success rate
observed_rate = SUCCESS_COUNT / (SUCCESS_COUNT + FAIL_COUNT)
ax_post.axvline(
    observed_rate,
    linestyle=":",
    color="gray",
    linewidth=1.5,
    label=f"Observed rate = {observed_rate:.0%}",
)

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

# ---------- 6. Visualise mean shift ---------------------------------------------
# For a blog it is often illustrative to highlight how the expected value moves.
# We create a simple arrow plot (or bars) showing prior mean → posterior mean.

fig_shift, ax_shift = plt.subplots(figsize=(10, 4))
ax_shift.set_title("Step 3 – Shift in mean probability after observing data")
ax_shift.set_ylim(0, 1)
ax_shift.set_ylabel("Mean p")
ax_shift.set_xticks(range(len(priors)))
ax_shift.set_xticklabels([p["label"] for p in priors], rotation=25, ha="right")

# Horizontal reference line at the observed success rate
observed_rate = SUCCESS_COUNT / (SUCCESS_COUNT + FAIL_COUNT)
ax_shift.axhline(
    observed_rate,
    linestyle=":",
    color="gray",
    linewidth=1.5,
    label=f"Observed rate = {observed_rate:.0%}",
)

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
ax_shift.legend(loc="upper left")
fig_shift.tight_layout()

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
# • **Optimistic prior** – The new 95 % prior *overshoots* the observed 90 %
#   success rate, so its posterior is pulled *downwards* by the data, illustrating
#   the symmetry of Bayesian learning: evidence moves beliefs whether it points
#   up or down.
# • **Posterior means** – The final means still differ: strong priors drag the
#   posterior back toward the prior belief, illustrating how information is
#   balanced in Bayesian inference.
