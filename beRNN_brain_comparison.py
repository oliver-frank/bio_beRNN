import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, ks_2samp

# Participants
participants_beRNN_models = ['beRNN_01'] # ['beRNN_02', 'beRNN_03', 'beRNN_05'] #
participants_brain_models = ['SNIPKPB84'] # ['SNIPYL4AS', 'SNIP6IECX', 'SNIP96WID'] #

# Topological markers
top_markers = ["degree", "betweenness", "assortativity"]

# Months for Model 1 (Model 2 has no months)
months = ["3", "4", "5"]

# Output directory
destination_dir = "W:\\group_csp\\analyses\\oliver.frank\\brainModels\\topMarkerComparisons_beRNN_brain"
os.makedirs(destination_dir, exist_ok=True)

def load_bootstrap_distributions(directory, participant, top_marker, months=months):
    """
    Load bootstrap distributions for a given participant and topological marker.
    """
    distributions = {}

    if 'beRNNmodels' in directory:
        # beRNN model (Has months)
        for month in months:
            file_path = os.path.join(directory, f"{top_marker}List_{month}.npy")
            if os.path.exists(file_path):
                distributions[month] = np.load(file_path, allow_pickle=True)
            else:
                print(f"Missing file: {file_path}")

    else:
        # brain model (No months, so duplicate data across months)
        file_path = os.path.join(directory, f"{participant}_bootstrap_{top_marker}.npy")
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True)
            distributions = {month: data for month in months}  # Replicate for 3 months
        else:
            print(f"Missing file: {file_path}")

    return distributions

def compare_models(beRNN_model_distributions, brain_model_distributions, participant_indice):
    """
    Compare Model Class 1 and Model Class 2 distributions for each topological marker across months.
    """
    p_values = {marker: {} for marker in top_markers}

    for marker in top_markers:
        for month in months:
            if month in beRNN_model_distributions[marker] and month in brain_model_distributions[marker]:
                data_model1 = beRNN_model_distributions[marker][month]
                data_model2 = brain_model_distributions[marker][month]

                if len(data_model1) > 1 and len(data_model2) > 1:
                    t_stat, p_ttest = ttest_ind(data_model1, data_model2, equal_var=False)
                    ks_stat, p_ks = ks_2samp(data_model1, data_model2)

                    p_values[marker][month] = min(p_ttest, p_ks)  # Store the smaller p-value
                else:
                    p_values[marker][month] = 1.0  # No valid comparison

    # Convert to DataFrame for visualization
    p_df = pd.DataFrame(p_values).T  # Transpose so markers are rows, months are columns

    # Prepare text annotations with significance levels
    def format_p_value(p):
        if p < 0.001:
            return f"$\\bf{{{p:.3f}}}$***"  # Bold + ***
        elif p < 0.01:
            return f"$\\bf{{{p:.3f}}}$**"   # Bold + **
        elif p < 0.05:
            return f"$\\bf{{{p:.3f}}}$*"    # Bold + *
        else:
            return f"{p:.3f}"               # No bold

    annotations = p_df.applymap(format_p_value)

    # Plot heatmap of p-values
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        p_df.astype(float),
        annot=annotations,
        fmt="",
        cmap="magma",
        vmin=0.001,
        vmax=1.0,
        center=0.05,
        cbar_kws={"shrink": 1.0},
        annot_kws={"fontsize": 10, "color": "white"},
    )

    plt.title(f"Statistical Comparison: beRNN_model vs brain_model - {participants_beRNN_models[participant_indice]}")
    plt.xlabel("Months")
    plt.ylabel("Topological Markers")

    # Save and show the plot
    plot_path = os.path.join(destination_dir, f"topMarkerComparison_{participants_beRNN_models[participant_indice]}_beRNN_model_vs_brain_model.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

# Process each participant
for participant_indice in range(0,len(participants_beRNN_models)):
    print(f"Processing participant: {participants_beRNN_models[participant_indice]}")
    print(f"Processing participant: {participants_brain_models[participant_indice]}")

    # Directories for the two model classes
    beRNN_model_dir = f"C:\\Users\\oliver.frank\\Desktop\\BackUp\\beRNNmodels\\2025_03_2\\{participants_beRNN_models[participant_indice]}\\overviews\\distributions"  # 9 Distributions per participant
    brain_model_dir = "W:\\group_csp\\analyses\\oliver.frank\\brainModels\\topologicalMarkers_threshold_0.4\\topologicalMarkers_bootstrap"  # 3 Distributions per participant

    # Load distributions
    beRNN_model_distributions = {marker: load_bootstrap_distributions(beRNN_model_dir, participants_beRNN_models[participant_indice], marker, months) for marker in top_markers}
    brain_model_distributions = {marker: load_bootstrap_distributions(brain_model_dir, participants_brain_models[participant_indice], marker) for marker in top_markers}

    # Compare and visualize
    compare_models(beRNN_model_distributions, brain_model_distributions, participant_indice)

print("All comparisons completed!")
