import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats

# Set global style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams.update({'font.size': 11})

# Dataset from the text
# Study 1: 1 conventional (died), 11 ECMO (all survived)
# After Study 1: 2 conventional (both died), 8 ECMO (all survived)
# Study 2: 9 conventional (4 died, 5 survived), 9 ECMO (all survived)
# Study 2 Phase 2: 20 ECMO (19 survived, 1 died)

# Define the data
conventional_total = 1 + 2 + 9  # Total conventional treatment
conventional_deaths = 1 + 2 + 4  # Total deaths from conventional treatment
conventional_survived = conventional_total - conventional_deaths

ecmo_total = 11 + 8 + 9 + 20  # Total ECMO treatment
ecmo_deaths = 0 + 0 + 0 + 1  # Total deaths from ECMO treatment
ecmo_survived = ecmo_total - ecmo_deaths

# Sequential outcomes for plotting the adaptive trial process
treatments = [
    'Conv',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'Conv',
    'Conv',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'Conv',
    'Conv',
    'Conv',
    'Conv',
    'Conv',
    'Conv',
    'Conv',
    'Conv',
    'Conv',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
    'ECMO',
]

outcomes = [
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
]  # 1 = survived, 0 = died


def plot_raw_results():
    """Plot 1: Raw Results Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    treatments_labels = ['Conventional', 'ECMO']
    deaths = [conventional_deaths, ecmo_deaths]
    survivals = [conventional_survived, ecmo_survived]

    bar_width = 0.6
    ax.bar(treatments_labels, deaths, bar_width, color='firebrick', label='Deaths')
    ax.bar(treatments_labels, survivals, bar_width, bottom=deaths, color='mediumseagreen', label='Survivals')

    # Add the counts as text on the bars
    for i, v in enumerate(deaths):
        if v > 0:
            ax.text(i, v / 2, str(v), ha='center', color='white', fontweight='bold')
    for i, v in enumerate(survivals):
        if v > 0:
            ax.text(i, deaths[i] + v / 2, str(v), ha='center', color='white', fontweight='bold')

    # Add total numbers at the top of each bar
    for i in range(len(treatments_labels)):
        ax.text(
            i,
            deaths[i] + survivals[i] + 0.5,
            f'n = {deaths[i] + survivals[i]}',
            ha='center',
            va='bottom',
            fontweight='bold',
        )

    ax.set_ylabel('Number of Patients')
    ax.set_title('Raw Trial Results: Survival vs. Death by Treatment Group', fontsize=14)
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig


def plot_frequentist_perspective():
    """Plot 2: Frequentist Perspective"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate Fisher's exact test
    contingency_table = [[conventional_survived, conventional_deaths], [ecmo_survived, ecmo_deaths]]
    odds_ratio, p_value = stats.fisher_exact(contingency_table)

    # Calculate survival rates
    conventional_rate = conventional_survived / conventional_total * 100
    ecmo_rate = ecmo_survived / ecmo_total * 100

    treatments_labels = ['Conventional', 'ECMO']
    bar_width = 0.6

    # Create the bar chart for survival rates
    ax.bar(treatments_labels, [conventional_rate, ecmo_rate], color=['lightcoral', 'lightgreen'], width=bar_width)

    # Add the percentages as text on the bars
    ax.text(0, conventional_rate / 2, f"{conventional_rate:.1f}%", ha='center', fontweight='bold')
    ax.text(1, ecmo_rate / 2, f"{ecmo_rate:.1f}%", ha='center', fontweight='bold')

    # Add sample sizes beneath the bars
    ax.text(0, -5, f"n = {conventional_total}", ha='center')
    ax.text(1, -5, f"n = {ecmo_total}", ha='center')

    # Add p-value information
    p_value_text = f"Fisher's Exact Test: p = {p_value:.6f}"
    if p_value < 0.001:
        p_value_text = "Fisher's Exact Test: p < 0.001"
    ax.text(0.5, 105, p_value_text, ha='center', fontweight='bold')
    ax.text(0.5, 95, f"Odds Ratio: {odds_ratio:.2f}", ha='center')

    ax.set_ylabel('Survival Rate (%)')
    ax.set_title('Frequentist Perspective: Statistical Significance of Difference', fontsize=14)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    return fig


def beta_params(mean, std):
    """Convert mean and std to alpha and beta parameters of Beta distribution"""
    variance = std**2
    alpha = mean * (mean * (1 - mean) / variance - 1)
    beta = (1 - mean) * (mean * (1 - mean) / variance - 1)
    return alpha, beta


def plot_bayesian_perspective():
    """Plot 3: Bayesian Perspective"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prior (weak, assuming ~50% chance of success with some uncertainty)
    prior_mean = 0.5
    prior_std = 0.2
    prior_alpha, prior_beta = beta_params(prior_mean, prior_std)

    # Likelihood from data
    conv_alpha = conventional_survived + 1  # +1 for prior
    conv_beta = conventional_deaths + 1  # +1 for prior
    ecmo_alpha = ecmo_survived + 1  # +1 for prior
    ecmo_beta = ecmo_deaths + 1  # +1 for prior

    # Calculate posterior distributions
    x = np.linspace(0, 1, 1000)
    conv_prior = stats.beta.pdf(x, prior_alpha, prior_beta)
    ecmo_prior = stats.beta.pdf(x, prior_alpha, prior_beta)

    conv_posterior = stats.beta.pdf(x, conv_alpha, conv_beta)
    ecmo_posterior = stats.beta.pdf(x, ecmo_alpha, ecmo_beta)

    # Plot the Bayesian analysis
    ax.plot(x, conv_prior, 'lightcoral', linestyle='--', label='Prior (both treatments)')
    ax.plot(x, conv_posterior, 'firebrick', label='Posterior: Conventional')
    ax.plot(x, ecmo_posterior, 'forestgreen', label='Posterior: ECMO')

    # Find the modes of the posteriors for labeling
    conv_mode = x[np.argmax(conv_posterior)]
    ecmo_mode = x[np.argmax(ecmo_posterior)]

    # Add vertical lines at the modes
    ax.axvline(conv_mode, color='firebrick', linestyle=':')
    ax.axvline(ecmo_mode, color='forestgreen', linestyle=':')

    # Add text labels for the modes
    ax.text(conv_mode, max(conv_posterior) * 0.3, f"Mode: {conv_mode:.2f}", ha='center', color='firebrick')
    ax.text(ecmo_mode, max(ecmo_posterior) * 0.6, f"Mode: {ecmo_mode:.2f}", ha='center', color='forestgreen')

    # Calculate the probability that ECMO is better than conventional
    ecmo_better_prob = np.mean(
        [
            1 if e > c else 0
            for e, c in zip(
                np.random.beta(ecmo_alpha, ecmo_beta, 100000), np.random.beta(conv_alpha, conv_beta, 100000)
            )
        ]
    )

    # Add the probability text
    ax.text(
        0.5,
        max(ecmo_posterior) * 0.9,
        f"P(ECMO better than Conv) = {ecmo_better_prob:.4f}",
        ha='center',
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8),
    )

    ax.set_xlabel('Probability of Survival')
    ax.set_ylabel('Probability Density')
    ax.set_title('Bayesian Perspective: Posterior Probability Distributions', fontsize=14)
    ax.legend(loc='upper left')
    plt.tight_layout()
    return fig


def plot_adaptive_trial_perspective():
    """Plot 4: Adaptive Trial Perspective"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Simulate the adaptive trial process
    ecmo_wins = 0
    conv_wins = 0
    ecmo_prob = []
    patient_num = []

    for i, (treat, outcome) in enumerate(zip(treatments, outcomes)):
        if treat == 'ECMO' and outcome == 1:
            ecmo_wins += 1
        elif treat == 'ECMO' and outcome == 0:
            pass  # ECMO loss, no adjustment
        elif treat == 'Conv' and outcome == 1:
            conv_wins += 1
        elif treat == 'Conv' and outcome == 0:
            pass  # Conv loss, no adjustment

        total_trials = i + 1
        ecmo_count = treatments[: i + 1].count('ECMO')
        conv_count = treatments[: i + 1].count('Conv')

        if ecmo_count > 0:
            ecmo_success_rate = ecmo_wins / ecmo_count
        else:
            ecmo_success_rate = 0

        if conv_count > 0:
            conv_success_rate = conv_wins / conv_count
        else:
            conv_success_rate = 0

        # Calculate probability of selecting ECMO as better (simplified)
        if ecmo_count > 0 and conv_count > 0:
            p_ecmo_better = np.mean(
                [
                    1 if e > c else 0
                    for e, c in zip(
                        np.random.beta(ecmo_wins + 1, ecmo_count - ecmo_wins + 1, 1000),
                        np.random.beta(conv_wins + 1, conv_count - conv_wins + 1, 1000),
                    )
                ]
            )
        else:
            p_ecmo_better = 0.5  # Start with no preference

        ecmo_prob.append(p_ecmo_better)
        patient_num.append(i + 1)

    # Plot the evolution of belief in ECMO superiority
    ax.plot(patient_num, ecmo_prob, 'b-', linewidth=2)
    ax.axhline(y=0.95, color='green', linestyle='--', label='95% Confidence Threshold')
    ax.axhline(y=0.5, color='gray', linestyle=':', label='No Preference (50%)')

    # Find where the probability first exceeds 95%
    if any(p >= 0.95 for p in ecmo_prob):
        cross_idx = next(i for i, p in enumerate(ecmo_prob) if p >= 0.95)
        cross_patient = patient_num[cross_idx]
        ax.axvline(x=cross_patient, color='red', linestyle='--')
        ax.text(
            cross_patient + 1, 0.6, f"95% confidence reached\nafter {cross_patient} patients", ha='left', color='red'
        )

    ax.set_xlabel('Number of Patients')
    ax.set_ylabel('P(ECMO superior to Conventional)')
    ax.set_title('Adaptive Trial Perspective: Evolution of Belief Over Time', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, len(patient_num))
    plt.tight_layout()
    return fig


def plot_bandit_perspective():
    """Plot 5: Multi-Armed Bandit Perspective"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Simulate a Win-Stay, Lose-Shift bandit algorithm
    balls_in_urn = {'ECMO': 1, 'Conv': 1}
    prob_ecmo = []
    treatment_choices = []
    cumulative_lives_saved = []
    lives_saved = 0
    lives_that_could_be_saved = 0

    # Track the selections for visualization
    ecmo_selections = []
    conv_selections = []

    # Run the simulation
    np.random.seed(42)  # For reproducibility
    for i in range(60):  # Simulate 60 patients
        # Current probability of selection
        p_ecmo = balls_in_urn['ECMO'] / (balls_in_urn['ECMO'] + balls_in_urn['Conv'])
        prob_ecmo.append(p_ecmo)

        # Select treatment based on current probabilities
        if np.random.random() < p_ecmo:
            treatment = 'ECMO'
            ecmo_selections.append(i)
        else:
            treatment = 'Conv'
            conv_selections.append(i)

        treatment_choices.append(treatment)

        # Determine outcome based on real-world probabilities
        ecmo_survival_rate = ecmo_survived / ecmo_total
        conv_survival_rate = conventional_survived / conventional_total

        if treatment == 'ECMO':
            outcome = 1 if np.random.random() < ecmo_survival_rate else 0
        else:
            outcome = 1 if np.random.random() < conv_survival_rate else 0

        # Update based on Win-Stay, Lose-Shift logic
        if outcome == 1:  # Success
            balls_in_urn[treatment] += 1
            lives_saved += 1
        else:  # Failure
            other_treatment = 'Conv' if treatment == 'ECMO' else 'ECMO'
            balls_in_urn[other_treatment] += 1

        lives_that_could_be_saved += 1 if np.random.random() < ecmo_survival_rate else 0
        cumulative_lives_saved.append(lives_saved)

    # Calculate the regret
    optimal_strategy = np.random.binomial(1, ecmo_survival_rate, size=60).cumsum()
    regret = optimal_strategy - np.array(cumulative_lives_saved)

    # Plot the bandit simulation results
    ax.plot(range(1, len(prob_ecmo) + 1), prob_ecmo, 'b-', linewidth=2, label='P(Select ECMO)')

    # Mark the selections on the x-axis
    ax.scatter(
        np.array(ecmo_selections) + 1,
        [0] * len(ecmo_selections),
        marker='^',
        color='green',
        s=80,
        label='ECMO Selected',
    )
    ax.scatter(
        np.array(conv_selections) + 1,
        [0] * len(conv_selections),
        marker='v',
        color='red',
        s=80,
        label='Conventional Selected',
    )

    # Plot the regret
    ax_twin = ax.twinx()
    ax_twin.plot(range(1, len(regret) + 1), regret, 'r--', linewidth=2, label='Cumulative Regret')
    ax_twin.set_ylabel('Cumulative Regret (Expected Lives Lost)', color='red')
    ax_twin.tick_params(axis='y', labelcolor='red')

    ax.set_xlabel('Patient Number')
    ax.set_ylabel('Probability of Selecting ECMO Treatment')
    ax.set_title('Multi-Armed Bandit Perspective: Treatment Selection & Regret Over Time', fontsize=14)

    # Create combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    return fig


def main():
    """Main function to run any of the plots independently"""
    # Dictionary mapping plot names to their functions
    plots = {
        'raw': plot_raw_results,
        'frequentist': plot_frequentist_perspective,
        'bayesian': plot_bayesian_perspective,
        'adaptive': plot_adaptive_trial_perspective,
        'bandit': plot_bandit_perspective,
    }

    # Example usage:
    # To run a specific plot:
    # plot_name = 'raw'  # or 'frequentist', 'bayesian', 'adaptive', 'bandit'
    # fig = plots[plot_name]()
    # plt.show()

    # To run all plots:
    for name, plot_func in plots.items():
        fig = plot_func()
        plt.savefig(f'ecmo_study_{name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
