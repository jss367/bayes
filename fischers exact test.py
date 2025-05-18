import matplotlib.pyplot as plt
import scipy.stats as stats

# Define the data in a more self-documenting way
# Format: [survived, died]
treatment_group = [10, 0]  # 10 survived, 0 died
control_group = [0, 1]  # 0 survived, 1 died

# Create the contingency table for Fisher's Exact Test
contingency_table = [
    treatment_group,  # [survived, died] for treatment group
    control_group,  # [survived, died] for control group
]

# Perform Fisher's Exact Test
oddsratio, p_value = stats.fisher_exact(contingency_table, alternative='greater')

print(f"Odds Ratio: {oddsratio:.2f}")
print(f"P-value: {p_value:.5f}")

# Visual: bar chart of survival outcomes
groups = ['Treatment', 'Control']
survived = [treatment_group[0], control_group[0]]
died = [treatment_group[1], control_group[1]]

bar_width = 0.35
x = range(len(groups))

plt.bar(x, survived, bar_width, label='Survived')
plt.bar(x, died, bar_width, bottom=survived, label='Died')

plt.xticks(x, groups)
plt.ylabel('Count')
plt.title('Survival by Group')
plt.legend()
plt.tight_layout()
plt.show()
