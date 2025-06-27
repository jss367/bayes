import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def plot_ecmo_posteriors() -> None:
    """Plot posterior distributions for ECMO effectiveness under several prior beliefs."""
    # Set up the figure
    plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1])

    # Set a consistent style
    plt.style.use('seaborn-v0_8-pastel')

    # Data from the first ECMO trial
    conventional_n = 1
    conventional_survived = 0
    ecmo_n = 11
    ecmo_survived = 11

    # Prior probabilities for ECMO superiority
    prior_percentages = [5, 10, 50]
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    # Range of values for survival probability
    p = np.linspace(0.001, 0.999, 1000)

    # Create first subplot - posterior distributions
    ax1 = plt.subplot(gs[0])

    # For each prior belief
    results = []
    for i, prior_percent in enumerate(prior_percentages):
        # Convert prior percentage to beta parameters
        # For the skeptical priors (5% and 10%), we want strong belief that treatments are similar
        if prior_percent == 5:
            # Very skeptical prior
            conv_prior_alpha = 9
            conv_prior_beta = 1
            ecmo_prior_alpha = 1
            ecmo_prior_beta = 4
        elif prior_percent == 10:
            # Moderately skeptical prior
            conv_prior_alpha = 9
            conv_prior_beta = 1
            ecmo_prior_alpha = 1
            ecmo_prior_beta = 2
        else:  # 50%
            # Neutral prior
            conv_prior_alpha = 1
            conv_prior_beta = 1
            ecmo_prior_alpha = 1
            ecmo_prior_beta = 1

        # Calculate posterior parameters after the first trial
        conv_posterior_alpha = conv_prior_alpha + conventional_survived
        conv_posterior_beta = conv_prior_beta + (conventional_n - conventional_survived)

        ecmo_posterior_alpha = ecmo_prior_alpha + ecmo_survived
        ecmo_posterior_beta = ecmo_prior_beta + (ecmo_n - ecmo_survived)

        # Calculate posterior distributions
        conv_posterior = stats.beta(conv_posterior_alpha, conv_posterior_beta).pdf(p)
        ecmo_posterior = stats.beta(ecmo_posterior_alpha, ecmo_posterior_beta).pdf(p)

        # Plot the posteriors
        ax1.plot(p, conv_posterior, '--', color=colors[i], alpha=0.5, label=f'Conv. treatment ({prior_percent}% prior)')
        ax1.plot(p, ecmo_posterior, '-', color=colors[i], linewidth=2, label=f'ECMO treatment ({prior_percent}% prior)')

        # Calculate probability that ECMO is better
        samples = 100000
        conv_samples = stats.beta(conv_posterior_alpha, conv_posterior_beta).rvs(samples)
        ecmo_samples = stats.beta(ecmo_posterior_alpha, ecmo_posterior_beta).rvs(samples)
        prob_ecmo_better = np.mean(ecmo_samples > conv_samples)
        results.append((prior_percent, prob_ecmo_better))

        # Add text annotation for this prior
        y_pos = 5 - i * 1.2  # Position text at different heights
        ax1.text(
            0.5,
            y_pos,
            f'Starting with {prior_percent}% belief in ECMO superiority:\n'
            f'After trial: {prob_ecmo_better:.1%} probability ECMO is better',
            ha='center',
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round', color=colors[i]),
        )

    ax1.set_xlabel('Probability of Survival')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Posterior Distributions with Different Prior Beliefs About ECMO', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 7)

    # Second subplot - prior to posterior visualization
    ax2 = plt.subplot(gs[1])

    # Set up bar positions
    bar_positions = np.arange(len(prior_percentages))
    bar_width = 0.35

    # Create bars for priors
    prior_bars = ax2.bar(
        bar_positions - bar_width / 2,
        [p / 100 for p in prior_percentages],
        bar_width,
        label='Prior Belief',
        color=[c for c in colors],
        alpha=0.5,
    )

    # Create bars for posteriors
    posterior_bars = ax2.bar(
        bar_positions + bar_width / 2,
        [r[1] for r in results],
        bar_width,
        label='Posterior Belief',
        color=[c for c in colors],
    )

    # Add percentage labels
    for i, (prior_bar, post_bar) in enumerate(zip(prior_bars, posterior_bars)):
        ax2.text(
            prior_bar.get_x() + prior_bar.get_width() / 2,
            prior_bar.get_height() + 0.02,
            f'{prior_percentages[i]}%',
            ha='center',
        )
        ax2.text(
            post_bar.get_x() + post_bar.get_width() / 2,
            post_bar.get_height() + 0.02,
            f'{results[i][1]:.1%}',
            ha='center',
        )

    # Customize the plot
    ax2.set_xticks(bar_positions)
    ax2.set_xticklabels([f'{p}% Prior' for p in prior_percentages])
    ax2.set_ylabel('Probability ECMO is Superior')
    ax2.set_title('From Prior to Posterior: Update in Beliefs After ECMO Trial Data', fontsize=14)
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_axisbelow(True)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_ecmo_posteriors()
