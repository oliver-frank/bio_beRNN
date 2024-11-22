import os
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
import nibabel as nib
import networkx as nx

# Define ROIs of interest (Schaefer 200 Atlas)
roi_indices = [
    # Left Hemisphere (LH)
    1, 2, 3, 4, 5, 6, 7, 8,  # Default_PFCd, PFCv, ACC
    9, 10, 11, 12, 13, 14, 15, 16,  # PCC, Precuneus
    17, 18, 19, 20, 21, 22, 23, 24,  # Limbic_OFC
    25, 26, 27, 28, 29, 30, 31, 32,  # SalVentAttn_Ins
    33, 34, 35, 36, 37, 38, 39, 40,  # DorsAttn_IPS, SPL
    41, 42, 43, 44, 45, 46, 47, 48,  # Frontal Eye Fields
    49, 50, 51, 52, 53, 54, 55, 56,  # Control Network
    57, 58, 59, 60, 61, 62, 63, 64,  # Temporal Regions

    # Right Hemisphere (RH)
    101, 102, 103, 104, 105, 106, 107, 108,  # Default_PFCd, PFCv, ACC
    109, 110, 111, 112, 113, 114, 115, 116,  # PCC, Precuneus
    117, 118, 119, 120, 121, 122, 123, 124,  # Limbic_OFC
    125, 126, 127, 128, 129, 130, 131, 132,  # SalVentAttn_Ins
    133, 134, 135, 136, 137, 138, 139, 140,  # DorsAttn_IPS, SPL
    141, 142, 143, 144, 145, 146, 147, 148,  # Frontal Eye Fields
    149, 150, 151, 152, 153, 154, 155, 156,  # Control Network
    157, 158, 159, 160, 161, 162, 163, 164   # Temporal Regions
]


# Relevant labels for visualization (match indices)
roi_labels = [
    # Left Hemisphere (LH)
    "LH_Region_1", "LH_Region_2", "LH_Region_3", "LH_Region_4", "LH_Region_5", "LH_Region_6", "LH_Region_7", "LH_Region_8",
    "LH_Region_9", "LH_Region_10", "LH_Region_11", "LH_Region_12", "LH_Region_13", "LH_Region_14", "LH_Region_15", "LH_Region_16",
    "LH_Region_17", "LH_Region_18", "LH_Region_19", "LH_Region_20", "LH_Region_21", "LH_Region_22", "LH_Region_23", "LH_Region_24",
    "LH_Region_25", "LH_Region_26", "LH_Region_27", "LH_Region_28", "LH_Region_29", "LH_Region_30", "LH_Region_31", "LH_Region_32",
    "LH_Region_33", "LH_Region_34", "LH_Region_35", "LH_Region_36", "LH_Region_37", "LH_Region_38", "LH_Region_39", "LH_Region_40",
    "LH_Region_41", "LH_Region_42", "LH_Region_43", "LH_Region_44", "LH_Region_45", "LH_Region_46", "LH_Region_47", "LH_Region_48",
    "LH_Region_49", "LH_Region_50", "LH_Region_51", "LH_Region_52", "LH_Region_53", "LH_Region_54", "LH_Region_55", "LH_Region_56",
    "LH_Region_57", "LH_Region_58", "LH_Region_59", "LH_Region_60", "LH_Region_61", "LH_Region_62", "LH_Region_63", "LH_Region_64",

    # Right Hemisphere (RH)
    "RH_Region_1", "RH_Region_2", "RH_Region_3", "RH_Region_4", "RH_Region_5", "RH_Region_6", "RH_Region_7", "RH_Region_8",
    "RH_Region_9", "RH_Region_10", "RH_Region_11", "RH_Region_12", "RH_Region_13", "RH_Region_14", "RH_Region_15", "RH_Region_16",
    "RH_Region_17", "RH_Region_18", "RH_Region_19", "RH_Region_20", "RH_Region_21", "RH_Region_22", "RH_Region_23", "RH_Region_24",
    "RH_Region_25", "RH_Region_26", "RH_Region_27", "RH_Region_28", "RH_Region_29", "RH_Region_30", "RH_Region_31", "RH_Region_32",
    "RH_Region_33", "RH_Region_34", "RH_Region_35", "RH_Region_36", "RH_Region_37", "RH_Region_38", "RH_Region_39", "RH_Region_40",
    "RH_Region_41", "RH_Region_42", "RH_Region_43", "RH_Region_44", "RH_Region_45", "RH_Region_46", "RH_Region_47", "RH_Region_48",
    "RH_Region_49", "RH_Region_50", "RH_Region_51", "RH_Region_52", "RH_Region_53", "RH_Region_54", "RH_Region_55", "RH_Region_56",
    "RH_Region_57", "RH_Region_58", "RH_Region_59", "RH_Region_60", "RH_Region_61", "RH_Region_62", "RH_Region_63", "RH_Region_64"
]

def apply_threshold(matrix, threshold):
    # Set all values below the threshold to zero
    matrix_thresholded = np.where(np.abs(matrix) > threshold, matrix, 0)
    return matrix_thresholded

assert len(roi_labels) == len(roi_indices), "Mismatch between ROI labels and indices!"

# Directories and configuration
participants = ['sub-SNIP6IECX', 'sub-SNIP96WID', 'sub-SNIPKPB84', 'sub-SNIPYL4AS']
recordings = ['01', '02', '03', '04']
tasks = ['faces', 'flanker', 'nback', 'rest', 'reward']
data_dir = 'C:\\Users\\oliver.frank\\Desktop\\BackUp\\bio_BeRNN'
directory = 'W:\\group_csp\\analyses\\oliver.frank\\brainModels'
mode = 'SCHAEFER'  # Using Schaefer Atlas
n_rois = 400
threshold = 0.2  # threshold defining sparsity in created graph

# Load Schaefer atlas
schaefer_atlas = datasets.fetch_atlas_schaefer_2018(data_dir=data_dir, n_rois=n_rois, yeo_networks=7)
atlas_filename = schaefer_atlas.maps
labels = schaefer_atlas.labels

# Define masker for extracting time series
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, detrend=True, memory='nilearn_cache')


# Function to extract and process correlation matrices
def process_participant(participant, recording, tasks):
    subject_directory = os.path.join(directory, f'{participant + recording}', 'func')
    correlation_matrix_list = []

    for task in tasks:
        nifti_file = os.path.join(subject_directory, f'{participant + recording}_task-{task}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')

        if not os.path.exists(nifti_file):
            print(f"File not found: {nifti_file}")
            continue

        # Load fMRI data
        fmri_img = nib.load(nifti_file)

        # Extract time series
        time_series = masker.fit_transform(fmri_img)

        # Select relevant regions
        # filtered_time_series = time_series[:, roi_indices]
        filtered_time_series = time_series[:, [i - 1 for i in roi_indices]]

        # Compute correlation matrix
        correlation_matrix = np.corrcoef(filtered_time_series.T)
        correlation_matrix_name = f'{participant + recording}' + '_' + task
        correlation_matrix_list.append(correlation_matrix)
        # Save the correlation matrix as a .npy file
        corrMatrixPath = os.path.join(subject_directory, f'npy_corrMatrices_{mode}_filtered_rois_{participant + recording}')

        if not os.path.exists(corrMatrixPath):
            # If it doesn't exist, create the directory
            os.makedirs(corrMatrixPath)
            print(f"Directory created: {corrMatrixPath}")
        else:
            print(f"Directory already exists: {corrMatrixPath}")

        np.save(os.path.join(corrMatrixPath, correlation_matrix_name), correlation_matrix)

        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label="Correlation Coefficient")
        plt.title(f"Correlation Matrix ({mode} Atlas) - {participant + recording} - {task} - filtered_rois")
        plt.xticks(ticks=np.arange(len(roi_labels)), labels=roi_labels, rotation=90, fontsize=8)
        plt.yticks(ticks=np.arange(len(roi_labels)), labels=roi_labels, fontsize=8)
        plt.tight_layout()

        # Save figure
        fname = f'Correlation Matrix ({mode} Atlas) - {participant + recording} - {task} - filtered_rois' + '.png'
        fpath = f'{directory}\\visuals\\Correlation_fMRI\\{mode}\\{participant + recording}'

        if not os.path.exists(fpath):
            # If it doesn't exist, create the directory
            os.makedirs(fpath)
            print(f"Directory created: {fpath}")
        else:
            print(f"Directory already exists: {fpath}")

        # Save the plot
        plt.savefig(os.path.join(fpath, fname), format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    # Average correlation matrix across tasks
    if correlation_matrix_list:
        averaged_correlation_matrix = np.mean(correlation_matrix_list, axis=0)
        averaged_correlation_matrix_name = f'{participant + recording}' + '_' + 'averagedTask - filtered_rois'

        # Plot averaged correlation matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(averaged_correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label="Correlation Coefficient")
        plt.title(f'Correlation Matrix ({mode} Atlas) - {participant + recording} - averaged tasks - filtered_rois')
        plt.xticks(ticks=np.arange(len(roi_labels)), labels=roi_labels, rotation=90, fontsize=8)
        plt.yticks(ticks=np.arange(len(roi_labels)), labels=roi_labels, fontsize=8)
        plt.tight_layout()

        # Save averaged matrix plot
        plt.savefig(os.path.join(fpath, averaged_correlation_matrix_name), dpi=300)
        plt.close()

        # Save the averaged correlation matrix
        np.save(os.path.join(corrMatrixPath, averaged_correlation_matrix_name), averaged_correlation_matrix)



        ########################################################################################################################
        # Topological Marker Analysis - Brain
        ########################################################################################################################
        # for idx, average_correlationMatrix in enumerate(average_correlationMatrixList):
        # Define a threshold (you can experiment with this value)
        averaged_correlation_matrix_thresholded = apply_threshold(averaged_correlation_matrix, threshold)

        # Function to apply a threshold to the matrix
        G = nx.from_numpy_array(averaged_correlation_matrix_thresholded)

        # The degree of a node in a graph is the count of edges connected to that node. For each node, it represents the number of
        # direct connections (or neighbors) it has within the graph.
        degrees = nx.degree(G)  # For calculating node degrees. attention: transform into graph then apply stuff
        # Betweenness centrality quantifies the importance of a node based on its position within the shortest paths between other nodes.
        betweenness = nx.betweenness_centrality(G)  # For betweenness centrality.
        # Closeness centrality measures the average distance from a node to all other nodes in the network.
        closeness = nx.closeness_centrality(G)  # For closeness centrality.
        # average_path_length = nx.average_shortest_path_length(G) # For average path length. attention: Graph is disconnected after thresholding therefore not possible

        # Optionally calculate averages of node-based metrics
        avg_degree = np.mean(list(dict(G.degree()).values()))
        avg_betweenness = np.mean(list(betweenness.values()))
        avg_closeness = np.mean(list(closeness.values()))

        # Graph Density: Measures how many edges exist compared to the maximum possible. This can give you a sense of how interconnected the network is.
        # Max 1; Min 0 Everything or Nothing is connected
        density = nx.density(G)
        # Assortativity: Measures the tendency of nodes to connect to other nodes with similar degrees.
        # A positive value means that high-degree nodes tend to connect to other high-degree nodes. Around 0 no relevant correlation
        assortativity = nx.degree_assortativity_coefficient(G)
        # Transitivity (Global Clustering Coefficient): Measures the likelihood that the adjacent nodes of a node are connected.
        # It’s an indicator of local clustering. 0-1 ; 1 every node that has two neighbours are also connected to each other
        transitivity = nx.transitivity(G)
        # Average Clustering Coefficient: Provides the average of the clustering coefficient (local) for all nodes.
        # It gives you a sense of how well nodes tend to form clusters. Similar to Transitivity
        avg_clustering = nx.average_clustering(G)
        # Largest Connected Component: The size (number of nodes) in the largest connected subgraph of the network.
        # This is particularly important in sparse graphs after thresholding.
        largest_cc = len(max(nx.connected_components(G), key=len))

        metrics = {
            "degrees": degrees,
            "betweenness": betweenness,
            "closeness": closeness,
            "avg_degree": avg_degree,
            "avg_betweenness": avg_betweenness,
            "avg_closeness": avg_closeness,
            "density": density,
            "assortativity": assortativity,
            "transitivity": transitivity,
            "avg_clustering": avg_clustering,
            "largest_cc": largest_cc
        }

        # Define the output directory and file name
        output_directory = os.path.join(directory, 'topologicalMarkers_threshold_' + str(threshold))

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Save the metrics dictionary as a .npy file
        output_file = os.path.join(output_directory,
                                   f'topologicalMarkers_threshold_{threshold}_{mode}_filtered_rois_{participant + recording}.npy')
        np.save(output_file, metrics, allow_pickle=True)

        print(f"Network measures saved to: {output_file}")


# Run pipeline for each participant and recording
for participant in participants:
    for recording in recordings:
        process_participant(participant, recording, tasks)




import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind

# Directory containing .npy files
participant = 'BeRNN_05'
directory = f'W:\\group_csp\\analyses\\oliver.frank\\beRNNmodels\\barnaModels\\topMarkerDistributions_monthWise_{participant}'

# Get all .npy files
file_names = sorted([f for f in os.listdir(directory) if f.endswith('.npy')])

# Dynamically group files based on a common identifier (e.g., part of the filename before "List")
groups = {}
for file in file_names:
    key = file.split('List')[0]  # Extract the dynamic group key from filename
    if key not in groups:
        groups[key] = []
    groups[key].append(os.path.join(directory, file))

# Create a figure with as many rows as groups and 9 columns
num_rows = len(groups)
fig, axes = plt.subplots(num_rows, 9, figsize=(25, 5 * num_rows), sharex=False, sharey=False)

# Ensure axes is always a 2D array for consistency
if num_rows == 1:
    axes = [axes]

# Process and plot distributions for each group
for row, (group, files) in enumerate(groups.items()):
    # Determine row-specific x and y limits
    row_x_min, row_x_max = float('inf'), float('-inf')
    row_y_min, row_y_max = float('inf'), float('-inf')

    # First pass: Calculate row-specific axis limits
    for file in files[:9]:  # Ensure max 9 plots per group
        try:
            data = np.load(file)  # Load the list of values
            counts, bins = np.histogram(data, bins=20, density=True)
            row_x_min = min(row_x_min, bins.min())
            row_x_max = max(row_x_max, bins.max())
            row_y_min = min(row_y_min, counts.min())
            row_y_max = max(row_y_max, counts.max())
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # Second pass: Plot with row-specific limits
    for col, file in enumerate(files[:9]):  # Ensure max 9 plots per group
        try:
            data = np.load(file)  # Load the list of values
            mean = np.mean(data)
            variance = np.var(data)

            ax = axes[row][col]
            ax.hist(data, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(mean, color='red', linestyle='dashed', linewidth=1.5,
                       label=f'Mean: {mean:.2f}\nVar: {variance:.2f}')
            ax.set_title(os.path.basename(file).split('Values_')[-1].split('.')[0], fontsize=18)
            ax.legend(fontsize=8)

            # Set row-specific axis limits
            ax.set_xlim(row_x_min, row_x_max)
            ax.set_ylim(row_y_min, row_y_max)

            # T-Test: Compare current distribution with the next
            if col < len(files) - 1:
                next_data = np.load(files[col + 1])
                t_stat, p_value = ttest_ind(data, next_data, equal_var=False)

                # Add significance annotation between consecutive plots
                if p_value < 0.05:
                    ax.annotate(
                        f"* p={p_value:.2e}",
                        xy=(0.5, 0.85), xycoords='axes fraction',
                        fontsize=14, fontweight='bold', ha='center', color='green'
                    )
                else:
                    ax.annotate(
                        f"n.s. p={p_value:.2e}",
                        xy=(0.5, 0.85), xycoords='axes fraction',
                        fontsize=14, fontweight='bold', ha='center', color='gray'
                    )

        except Exception as e:
            print(f"Error processing file {file}: {e}")

# Adjust layout and save plot
plt.tight_layout()
plt.suptitle(f"Distributions Grouped Dynamically by Key - {participant}", fontsize=16, fontweight='bold', y=1.02)
plt.savefig(os.path.join(directory, f'ALLtopMarkerDistributions_monthWise_{participant}.png'),
            format='png', dpi=300, bbox_inches='tight')
plt.show()


