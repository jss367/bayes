import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

# Define the study data in a more structured format
study_data = [
    {"name": "Study 1", "conventional": {"total": 1, "survived": 0}, "ecmo": {"total": 11, "survived": 11}},
    {"name": "Study 2", "conventional": {"total": 10, "survived": 6}, "ecmo": {"total": 29, "survived": 28}},
]


def get_study_data(study_indices=None):
    """
    Get data for specified studies or all studies if none specified.

    Args:
        study_indices: List of indices of studies to include (0-indexed)

    Returns:
        study_data_subset, treatments, outcomes
    """
    if study_indices is None:
        study_indices = list(range(len(study_data)))

    # Get the requested subset of studies
    study_data_subset = [study_data[i] for i in study_indices]

    # Generate treatments and outcomes lists
    treatments = []
    outcomes = []

    # Function to add patients to the lists
    def add_patients(treatment_type, total, survived):
        for i in range(total):
            treatments.append(treatment_type)
            # 1 = survived, 0 = died
            outcomes.append(1 if i < survived else 0)

    # Add patients from each study in sequence
    for study in study_data_subset:
        # Add conventional patients
        add_patients('Conv', study["conventional"]["total"], study["conventional"]["survived"])
        # Add ECMO patients
        add_patients('ECMO', study["ecmo"]["total"], study["ecmo"]["survived"])

    return study_data_subset, treatments, outcomes


def beta_params(mean, std):
    """Convert mean and std to alpha and beta parameters of Beta distribution"""
    variance = std**2
    alpha = mean * (mean * (1 - mean) / variance - 1)
    beta = (1 - mean) * (mean * (1 - mean) / variance - 1)
    return alpha, beta


def plot_raw_results(study_indices=None):
    """
    Plot 1: Raw Results Comparison

    Args:
        study_indices: List of indices of studies to include (0-indexed)
    """
    study_data_subset, _, _ = get_study_data(study_indices)

    # Calculate totals from the structured data
    conventional_total = sum(study["conventional"]["total"] for study in study_data_subset)
    conventional_survived = sum(study["conventional"]["survived"] for study in study_data_subset)
    conventional_deaths = conventional_total - conventional_survived

    ecmo_total = sum(study["ecmo"]["total"] for study in study_data_subset)
    ecmo_survived = sum(study["ecmo"]["survived"] for study in study_data_subset)
    ecmo_deaths = ecmo_total - ecmo_survived

    # Create the plot
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

    # Add title with included studies
    title = 'Raw Trial Results: Survival vs. Death by Treatment Group'
    if study_indices is not None:
        included_studies = [study_data[i]["name"] for i in study_indices]
        title += f"\nIncluded Studies: {', '.join(included_studies)}"

    ax.set_ylabel('Number of Patients')
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig


def plot_frequentist_perspective(study_indices=None):
    """
    Plot 2: Frequentist Perspective

    Args:
        study_indices: List of indices of studies to include (0-indexed)
    """
    study_data_subset, _, _ = get_study_data(study_indices)

    # Calculate totals from the structured data
    conventional_total = sum(study["conventional"]["total"] for study in study_data_subset)
    conventional_survived = sum(study["conventional"]["survived"] for study in study_data_subset)
    conventional_deaths = conventional_total - conventional_survived

    ecmo_total = sum(study["ecmo"]["total"] for study in study_data_subset)
    ecmo_survived = sum(study["ecmo"]["survived"] for study in study_data_subset)
    ecmo_deaths = ecmo_total - ecmo_survived

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate Fisher's exact test
    contingency_table = [[conventional_survived, conventional_deaths], [ecmo_survived, ecmo_deaths]]
    odds_ratio, p_value = stats.fisher_exact(contingency_table)

    # Calculate survival rates
    conventional_rate = conventional_survived / conventional_total * 100 if conventional_total > 0 else 0
    ecmo_rate = ecmo_survived / ecmo_total * 100 if ecmo_total > 0 else 0

    treatments_labels = ['Conventional', 'ECMO']
    bar_width = 0.6

    # Create the bar chart for survival rates
    ax.bar(treatments_labels, [conventional_rate, ecmo_rate], color=['lightcoral', 'lightgreen'], width=bar_width)

    # Add the percentages as text on the bars
    if conventional_rate > 0:
        ax.text(0, conventional_rate / 2, f"{conventional_rate:.1f}%", ha='center', fontweight='bold')
    if ecmo_rate > 0:
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

    # Add title with included studies
    title = 'Frequentist Perspective: Statistical Significance of Difference'
    if study_indices is not None:
        included_studies = [study_data[i]["name"] for i in study_indices]
        title += f"\nIncluded Studies: {', '.join(included_studies)}"

    ax.set_ylabel('Survival Rate (%)')
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    return fig


def plot_bayesian_perspective(study_indices=None):
    """
    Plot 3: Bayesian Perspective

    Args:
        study_indices: List of indices of studies to include (0-indexed)
    """
    study_data_subset, _, _ = get_study_data(study_indices)

    # Calculate totals from the structured data
    conventional_total = sum(study["conventional"]["total"] for study in study_data_subset)
    conventional_survived = sum(study["conventional"]["survived"] for study in study_data_subset)
    conventional_deaths = conventional_total - conventional_survived

    ecmo_total = sum(study["ecmo"]["total"] for study in study_data_subset)
    ecmo_survived = sum(study["ecmo"]["survived"] for study in study_data_subset)
    ecmo_deaths = ecmo_total - ecmo_survived

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
    prior_dist = stats.beta.pdf(x, prior_alpha, prior_beta)

    conv_posterior = stats.beta.pdf(x, conv_alpha, conv_beta)
    ecmo_posterior = stats.beta.pdf(x, ecmo_alpha, ecmo_beta)

    # Plot the Bayesian analysis
    ax.plot(x, prior_dist, 'lightcoral', linestyle='--', label='Prior (both treatments)')
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

    # Add title with included studies
    title = 'Bayesian Perspective: Posterior Probability Distributions'
    if study_indices is not None:
        included_studies = [study_data[i]["name"] for i in study_indices]
        title += f"\nIncluded Studies: {', '.join(included_studies)}"

    ax.set_xlabel('Probability of Survival')
    ax.set_ylabel('Probability Density')
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left')
    plt.tight_layout()
    return fig


def plot_adaptive_trial_perspective(study_indices=None):
    """
    Plot 4: Adaptive Trial Perspective

    Args:
        study_indices: List of indices of studies to include (0-indexed)
    """
    _, treatments, outcomes = get_study_data(study_indices)

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

        ecmo_count = treatments[: i + 1].count('ECMO')
        conv_count = treatments[: i + 1].count('Conv')

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

    # Add title with included studies
    title = 'Adaptive Trial Perspective: Evolution of Belief Over Time'
    if study_indices is not None:
        included_studies = [study_data[i]["name"] for i in study_indices]
        title += f"\nIncluded Studies: {', '.join(included_studies)}"

    ax.set_xlabel('Number of Patients')
    ax.set_ylabel('P(ECMO superior to Conventional)')
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, len(patient_num))
    plt.tight_layout()
    return fig


def plot_bandit_perspective(study_indices=None):
    """
    Plot 5: Multi-Armed Bandit Perspective

    Args:
        study_indices: List of indices of studies to include (0-indexed)
    """
    study_data_subset, _, _ = get_study_data(study_indices)

    # Calculate survival rates for the selected subset
    conventional_total = sum(study["conventional"]["total"] for study in study_data_subset)
    conventional_survived = sum(study["conventional"]["survived"] for study in study_data_subset)
    conventional_survival_rate = conventional_survived / conventional_total if conventional_total > 0 else 0

    ecmo_total = sum(study["ecmo"]["total"] for study in study_data_subset)
    ecmo_survived = sum(study["ecmo"]["survived"] for study in study_data_subset)
    ecmo_survival_rate = ecmo_survived / ecmo_total if ecmo_total > 0 else 0

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
        if treatment == 'ECMO':
            outcome = 1 if np.random.random() < ecmo_survival_rate else 0
        else:
            outcome = 1 if np.random.random() < conventional_survival_rate else 0

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

    # Add title with included studies
    title = 'Multi-Armed Bandit Perspective: Treatment Selection & Regret Over Time'
    if study_indices is not None:
        included_studies = [study_data[i]["name"] for i in study_indices]
        title += f"\nIncluded Studies: {', '.join(included_studies)}"

    ax.set_xlabel('Patient Number')
    ax.set_ylabel('Probability of Selecting ECMO Treatment')
    ax.set_title(title, fontsize=14)

    # Create combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    return fig


def create_statistical_summary(study_indices=None):
    """
    Create a statistical summary table figure for the specified studies

    Args:
        study_indices: List of indices of studies to include (0-indexed)
    """
    study_data_subset, _, _ = get_study_data(study_indices)

    # Calculate totals from the structured data
    conventional_total = sum(study["conventional"]["total"] for study in study_data_subset)
    conventional_survived = sum(study["conventional"]["survived"] for study in study_data_subset)
    conventional_deaths = conventional_total - conventional_survived

    ecmo_total = sum(study["ecmo"]["total"] for study in study_data_subset)
    ecmo_survived = sum(study["ecmo"]["survived"] for study in study_data_subset)
    ecmo_deaths = ecmo_total - ecmo_survived

    # Calculate mortality rates
    ecmo_mortality = (ecmo_deaths / ecmo_total * 100) if ecmo_total > 0 else 0
    conventional_mortality = (conventional_deaths / conventional_total * 100) if conventional_total > 0 else 0

    # Calculate risk metrics
    absolute_risk_reduction = (
        conventional_mortality - ecmo_mortality if conventional_total > 0 and ecmo_total > 0 else 0
    )
    relative_risk_reduction = (
        (absolute_risk_reduction / conventional_mortality * 100) if conventional_mortality > 0 else 0
    )
    nnt = 100 / absolute_risk_reduction if absolute_risk_reduction > 0 else float('inf')

    # Chi-square test
    if conventional_total > 0 and ecmo_total > 0:
        contingency_table = [[conventional_survived, conventional_deaths], [ecmo_survived, ecmo_deaths]]
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    else:
        chi2, p_value = 0, 1

    # Create figure with gridspec
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    # Left side: Bar chart (similar to plot_raw_results)
    ax1 = fig.add_subplot(gs[0, 0])

    treatments_labels = ['Conventional', 'ECMO']
    deaths = [conventional_deaths, ecmo_deaths]
    survivals = [conventional_survived, ecmo_survived]

    bar_width = 0.6
    ax1.bar(treatments_labels, deaths, bar_width, color='firebrick', label='Deaths')
    ax1.bar(treatments_labels, survivals, bar_width, bottom=deaths, color='mediumseagreen', label='Survivals')

    # Add the counts as text on the bars
    for i, v in enumerate(deaths):
        if v > 0:
            ax1.text(i, v / 2, str(v), ha='center', color='white', fontweight='bold')
    for i, v in enumerate(survivals):
        if v > 0:
            ax1.text(i, deaths[i] + v / 2, str(v), ha='center', color='white', fontweight='bold')

    # Add total numbers at the top of each bar
    for i in range(len(treatments_labels)):
        ax1.text(
            i,
            deaths[i] + survivals[i] + 0.5,
            f'n = {deaths[i] + survivals[i]}',
            ha='center',
            va='bottom',
            fontweight='bold',
        )

    ax1.set_ylabel('Number of Patients')
    ax1.set_title('Raw Trial Results', fontsize=14)
    ax1.legend(loc='upper right')

    # Right side: Statistical summary table
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('tight')
    ax2.axis('off')

    table_data = [
        ["", "ECMO", "Conventional"],
        ["Deaths", f"{ecmo_deaths} ({ecmo_mortality:.1f}%)", f"{conventional_deaths} ({conventional_mortality:.1f}%)"],
        [
            "Survivors",
            f"{ecmo_survived} ({100-ecmo_mortality:.1f}%)",
            f"{conventional_survived} ({100-conventional_mortality:.1f}%)",
        ],
        ["Total", f"{ecmo_total}", f"{conventional_total}"],
        ["", "", ""],
        ["Absolute Risk Reduction", f"{absolute_risk_reduction:.1f}%", ""],
        ["Relative Risk Reduction", f"{relative_risk_reduction:.1f}%", ""],
        ["Number Needed to Treat", f"{nnt:.1f}" if nnt != float('inf') else "∞", ""],
        ["Chi-square value", f"{chi2:.2f}", ""],
        ["P-value", f"{p_value:.6f}" if p_value >= 0.001 else "<0.001", ""],
    ]

    table = ax2.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Highlight p-value
    p_value_cell = table._cells[(9, 1)]
    p_value_cell.set_facecolor('#ffcccc')
    p_value_cell.set_text_props(weight='bold', color='red')

    # Title for table and overall figure
    ax2.set_title('Statistical Summary', fontweight='bold', pad=20)

    # Add title with included studies
    if study_indices is not None:
        included_studies = [study_data[i]["name"] for i in study_indices]
        fig.suptitle(f"Statistical Analysis\nIncluded Studies: {', '.join(included_studies)}", fontsize=16)
    else:
        fig.suptitle("Statistical Analysis (All Studies)", fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    return fig


def plot_perspective_comparison(study_indices=None):
    """
    Plot 6: Perspective Comparison - showing Bayesian posteriors and different statistical perspectives

    Args:
        study_indices: List of indices of studies to include (0-indexed)
    """
    # Get the data for the specified studies
    if study_indices is None or len(study_indices) == 0:
        # Default to just the first ECMO trial
        study_indices = [0]

    study_data_subset = [study_data[i] for i in study_indices]

    # Calculate totals from the specified studies
    conventional_total = sum(study["conventional"]["total"] for study in study_data_subset)
    conventional_survived = sum(study["conventional"]["survived"] for study in study_data_subset)

    ecmo_total = sum(study["ecmo"]["total"] for study in study_data_subset)
    ecmo_survived = sum(study["ecmo"]["survived"] for study in study_data_subset)

    # Set up the figure with two subplots
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5])

    # Set a consistent style
    plt.style.use('seaborn-v0_8-pastel')

    # Prior parameters (assuming a Beta(1,1) prior which is uniform)
    prior_alpha = 1
    prior_beta = 1

    # Posterior parameters
    conventional_posterior_alpha = prior_alpha + conventional_survived
    conventional_posterior_beta = prior_beta + (conventional_total - conventional_survived)

    ecmo_posterior_alpha = prior_alpha + ecmo_survived
    ecmo_posterior_beta = prior_beta + (ecmo_total - ecmo_survived)

    # Range of values for survival probability
    p = np.linspace(0.001, 0.999, 1000)

    # Calculate prior and posterior distributions
    prior = stats.beta(prior_alpha, prior_beta).pdf(p)
    conventional_posterior = stats.beta(conventional_posterior_alpha, conventional_posterior_beta).pdf(p)
    ecmo_posterior = stats.beta(ecmo_posterior_alpha, ecmo_posterior_beta).pdf(p)

    # Calculate probability that ECMO is better than conventional treatment
    samples = 100000
    conventional_samples = stats.beta(conventional_posterior_alpha, conventional_posterior_beta).rvs(samples)
    ecmo_samples = stats.beta(ecmo_posterior_alpha, ecmo_posterior_beta).rvs(samples)
    prob_ecmo_better = np.mean(ecmo_samples > conventional_samples)

    # First subplot - Posterior distributions
    ax1 = plt.subplot(gs[0])
    ax1.plot(p, prior, '--', color='gray', alpha=0.7, label='Prior belief (uninformative)')
    ax1.plot(
        p,
        conventional_posterior,
        color='#FF5733',
        linewidth=2,
        label=f'Conventional treatment posterior\n({conventional_survived}/{conventional_total} survivals)',
    )
    ax1.plot(
        p,
        ecmo_posterior,
        color='#33A1FF',
        linewidth=2,
        label=f'ECMO treatment posterior\n({ecmo_survived}/{ecmo_total} survivals)',
    )

    # Calculate posterior means
    conv_mean = conventional_posterior_alpha / (conventional_posterior_alpha + conventional_posterior_beta)
    ecmo_mean = ecmo_posterior_alpha / (ecmo_posterior_alpha + ecmo_posterior_beta)

    # Add vertical lines at the means
    ax1.axvline(conv_mean, color='#FF5733', linestyle=':', alpha=0.7)
    ax1.axvline(ecmo_mean, color='#33A1FF', linestyle=':', alpha=0.7)

    # Add text annotations
    ax1.text(conv_mean, 1.0, f'Mean: {conv_mean:.2f}', color='#FF5733', ha='center')
    ax1.text(ecmo_mean, 2.0, f'Mean: {ecmo_mean:.2f}', color='#33A1FF', ha='center')

    # Add text showing probability that ECMO is better
    ax1.text(
        0.5,
        3.5,
        f'P(ECMO better than conventional) = {prob_ecmo_better:.4f} ≈ {prob_ecmo_better:.2%}',
        ha='center',
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
    )

    ax1.set_xlabel('Probability of Survival')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Bayesian Posterior Distributions After ECMO Trial', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 5)

    # Second subplot - Different statistical perspectives
    perspectives = ['Frequentist', 'Bayesian', 'Adaptive Trial', 'Medical Ethics']

    # Calculate a statistical significance p-value for frequentist perspective
    if conventional_total > 0 and ecmo_total > 0:
        # Create contingency table
        conv_deaths = conventional_total - conventional_survived
        ecmo_deaths = ecmo_total - ecmo_survived
        contingency_table = [[conventional_survived, conv_deaths], [ecmo_survived, ecmo_deaths]]
        _, p_value = stats.fisher_exact(contingency_table)
    else:
        p_value = 1.0

    # Calculate confidence values dynamically based on the data
    # 1. Frequentist confidence based on p-value and sample sizes
    if p_value < 0.01:
        freq_confidence = 0.95  # Very significant
    elif p_value < 0.05:
        freq_confidence = 0.85  # Significant
    elif p_value < 0.1:
        freq_confidence = 0.7  # Marginally significant
    else:
        # Adjust for small sample sizes
        if conventional_total < 5:
            # Very small control group
            freq_confidence = 0.35
        elif conventional_total < 10:
            # Small control group
            freq_confidence = 0.45
        else:
            # Adequate sample size but not significant
            freq_confidence = 0.55

    # 2. Bayesian confidence is directly from the posterior probability
    bayes_confidence = prob_ecmo_better

    # 3. Adaptive trial confidence - based on the strength of evidence and sample size
    adaptive_confidence = min(0.95, 0.6 + (prob_ecmo_better - 0.5) * 1.2)

    # 4. Medical ethics confidence - based on survival rate differences and sample size
    conv_survival_rate = conventional_survived / conventional_total if conventional_total > 0 else 0
    ecmo_survival_rate = ecmo_survived / ecmo_total if ecmo_total > 0 else 0

    # If ECMO is clearly better, ethical concerns rise
    survival_diff = ecmo_survival_rate - conv_survival_rate
    if survival_diff > 0.5:
        ethics_confidence = 0.95  # Very clear difference
    elif survival_diff > 0.3:
        ethics_confidence = 0.85  # Clear difference
    else:
        ethics_confidence = 0.7  # Moderate difference

    # Adjust based on sample sizes
    if conventional_total < 5 and ecmo_total < 10:
        # Too small to be confident
        ethics_confidence = min(ethics_confidence, 0.8)

    # Final confidence values
    confidence = [freq_confidence, bayes_confidence, adaptive_confidence, ethics_confidence]

    # Generate conclusions based on the data
    conclusions = [
        f'{"Significant" if p_value < 0.05 else "Insufficient evidence"} with p={p_value:.3f}\n'
        f'Based on {conventional_total + ecmo_total} total patients',
        f'{prob_ecmo_better:.1%} probability that\nECMO is better than conventional treatment',
        f'{"Strong" if adaptive_confidence > 0.8 else "Moderate"} signal to '
        f'{"favor ECMO" if adaptive_confidence > 0.7 else "continue testing"}',
        f'{"High" if ethics_confidence > 0.8 else "Some"} ethical concerns about\n'
        f'continuing to randomize to conventional',
    ]

    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']

    ax2 = plt.subplot(gs[1])

    # Create bars
    bars = ax2.barh(perspectives, confidence, color=colors, alpha=0.7)

    # Add percentage labels
    for i, (bar, conf) in enumerate(zip(bars, confidence)):
        ax2.text(conf + 0.02, i, f'{conf:.0%}', va='center')

        # Add conclusion text
        ax2.text(
            0.5,
            i,
            conclusions[i],
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'),
        )

    # Customize the plot
    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel('Confidence that ECMO is superior')
    ax2.set_title('Different Perspectives on ECMO Trial Results', fontsize=14)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_axisbelow(True)
    ax2.grid(axis='x', alpha=0.3)

    # Add title with included studies
    title = 'Perspective Comparison on ECMO Trial Results'
    if study_indices is not None:
        included_studies = [study_data[i]["name"] for i in study_indices]
        fig.suptitle(title + "\nIncluded Studies: " + ', '.join(included_studies), fontsize=16)
    else:
        fig.suptitle(title, fontsize=16)

    # Add annotations for context
    study_text = (
        f'Note: The data includes {conventional_total} conventional treatment patients '
        f'({conventional_survived} survived) and {ecmo_total} ECMO treatment patients ({ecmo_survived} survived).'
    )
    plt.figtext(0.5, 0.01, study_text, ha='center', fontsize=10, bbox=dict(facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.08)

    return fig


def main():
    """Main function to run any of the plots with specified data subsets"""
    # Dictionary mapping plot names to their functions
    plots = {
        'raw': plot_raw_results,
        'frequentist': plot_frequentist_perspective,
        'bayesian': plot_bayesian_perspective,
        'adaptive': plot_adaptive_trial_perspective,
        'bandit': plot_bandit_perspective,
        'summary': create_statistical_summary,
        'perspectives': plot_perspective_comparison,  # Add the new plot
    }

    # Run all plots with all data (original behavior)
    for name, plot_func in plots.items():
        fig = plot_func()
        plt.savefig(f'ecmo_study_{name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Create perspective comparison plots for various study combinations
    study_combinations = [
        {'indices': [0], 'name': 'study1_only'},
        {'indices': [1], 'name': 'study2_only'},
        {'indices': [0, 1], 'name': 'all_studies'},
    ]

    for combo in study_combinations:
        fig = plot_perspective_comparison(combo['indices'])
        plt.savefig('ecmo_perspectives_' + combo['name'] + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
