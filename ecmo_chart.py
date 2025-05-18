import pandas as pd
from IPython.display import HTML, display

# Create the data for the table
data = {
    'Trial (Year)': ['Bartlett et al., 1985 (USA)', 'O\'Rourke et al., 1989 (USA)', 'UK Collaborative Trial, 1996'],
    'Design': [
        'Randomized play-the-winner (adaptive probability)',
        'Sequential RCT with planned early stopping',
        'Multicenter RCT (conventional randomization; DMC stopped early)',
    ],
    'Patients (ECMO vs CMT)': [
        '12 total (ECMO 11, CMT 1)',
        '29 ECMO vs 10 CMT in randomized phase (39 total; additional 20 ECMO in phase II)',
        '185 total (ECMO 93, CMT 92)',
    ],
    'Outcome: Survival ECMO vs CMT': [
        '100% (11/11) vs 0% (0/1) survival',
        '97% (28/29) vs 60% (6/10) survival',
        '68% (63/93) vs 41% (38/92) survival to discharge (RR ~0.55, p=0.0005)',
    ],
    'Key Points': [
        'Trial stopped after 1 control death; dramatic survival difference. Criticized for extremely small control sample. Set precedent for adaptive trial design in critical care.',
        'Randomization halted after 4 CMT deaths as pre-specified. Highly significant mortality reduction (p<0.05). Addressed some criticisms by including 10 controls.',
        'Stopped at interim due to clear benefit. First trial to assess 1-year outcomes: improved survival without higher severe disability. Established ECMO as standard of care.',
    ],
}

# Create DataFrame
df = pd.DataFrame(data)


# Define a function to style the table with bold headers
def style_table(df):
    # Create the styled HTML
    styled_html = """
    <style>
    table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 20px;
    }
    th {
        background-color: #f2f2f2;
        font-weight: bold;
        text-align: left;
        padding: 8px;
        border: 1px solid #ddd;
    }
    td {
        padding: 8px;
        border: 1px solid #ddd;
        vertical-align: top;
    }
    caption {
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 10px;
        text-align: center;
    }
    </style>
    """

    # Convert DataFrame to HTML with caption
    table_html = df.to_html(index=False, escape=False)
    table_html = table_html.replace('<table', '<table border="1" class="dataframe"')

    # Add the caption/title
    table_html = (
        f'<caption><strong>Table 1. Key Neonatal ECMO Trials and Designs (1980s–1990s)</strong></caption>{table_html}'
    )

    return styled_html + table_html


# Display the styled table
display(HTML(style_table(df)))

# If you're working in a script without IPython/Jupyter, you can save to an HTML file instead
with open('ecmo_trials_table.html', 'w') as f:
    f.write(style_table(df))

print("Table has been created. If you're running this in a script, check the ecmo_trials_table.html file.")
