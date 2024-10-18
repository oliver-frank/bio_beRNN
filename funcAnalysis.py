########################################################################################################################
from nilearn import datasets
from nilearn.image import resample_to_img
import nibabel as nib
import os
import numpy as np
import pandas as pd
from nilearn.image import load_img
from nilearn.maskers import NiftiMapsMasker
from sklearn.covariance import GraphicalLassoCV
from sklearn.metrics.pairwise import cosine_similarity
from nilearn import plotting
import matplotlib.pyplot as plt


# info: Average Cosine similiarity matrix over the different tasks subject-wise ########################################
participants = ['sub-SNIP6IECX02'] # , 'sub-SNIP96WID01', 'sub-SNIPDKHPB01', 'sub-SNIPKPB8401', 'sub-SNIPYL4AS01'
for participant in participants:

    # Step 4: Iterate over each task to calculate the covariance matrix
    for task in tasks:

    # Step 5: Compute the average correlation matrix across all tasks
    average_correlationMatrix = np.mean(correlationMatrices, axis=0)


# info: AAL
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, masking, image
from nilearn.input_data import NiftiLabelsMasker

# Define the data directory where the AAL atlas will be downloaded
data_dir = 'C:\\Users\\oliver.frank\\Desktop\\BackUp\\bio_BeRNN'

# Step 1: Fetch the AAL atlas
aal_atlas = datasets.fetch_atlas_aal(data_dir=data_dir)
atlas_filename = aal_atlas.maps
labels = aal_atlas.labels

# Step 2: Load your fMRI data
participant = 'sub-SNIP6IECX02'
task = 'flanker'

subject_directory = os.path.join('C:\\Users\\oliver.frank\\Desktop\\BackUp\\Brain_models', f'{participant}', 'func')
nifti_file = os.path.join(subject_directory, f'{participant}_task-{task}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')

fmri_img = nib.load(nifti_file)

# Step 3: Use the AAL atlas to extract time series data from each region
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, memory='nilearn_cache')
time_series = masker.fit_transform(fmri_img)

# Step 4: Compute the correlation matrix
correlation_matrix = np.corrcoef(time_series.T)

# Step 5: Plot the correlation matrix
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, vmin=-1, vmax=1, cmap='coolwarm')
plt.colorbar(label='Correlation coefficient')
plt.title('Correlation Matrix (AAL Atlas)')
plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90, fontsize=6)
plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=6)
plt.tight_layout()
plt.show()


# info: MSDL
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker

# Define the data directory where the MSDL atlas will be downloaded
data_dir = 'C:\\Users\\oliver.frank\\Desktop\\BackUp\\bio_BeRNN'

# Step 1: Fetch the MSDL atlas
msdl_atlas = datasets.fetch_atlas_msdl(data_dir=data_dir)
atlas_filename = msdl_atlas.maps
labels = msdl_atlas.labels  # These are the names of the regions

# Step 2: Load your fMRI data
participant = 'sub-SNIP6IECX02'
task = 'flanker'
subject_directory = os.path.join('C:\\Users\\oliver.frank\\Desktop\\BackUp\\Brain_models', f'{participant}', 'func')
nifti_file = os.path.join(subject_directory, f'{participant}_task-{task}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')

fmri_img = nib.load(nifti_file)

# Step 3: Use the MSDL atlas to extract time series data from each region
masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, memory='nilearn_cache')
time_series = masker.fit_transform(fmri_img)

# Step 4: Compute the correlation matrix
correlation_matrix = np.corrcoef(time_series.T)

# Step 5: Plot the correlation matrix with region labels
num_regions = len(labels)  # This should match the number of regions in the time series

plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, vmin=-1, vmax=1, cmap='coolwarm')
plt.colorbar(label='Correlation coefficient')
plt.title('Correlation Matrix (MSDL Atlas)')

# Label the axes with the region names
plt.xticks(ticks=np.arange(num_regions), labels=labels, rotation=90, fontsize=6)
plt.yticks(ticks=np.arange(num_regions), labels=labels, fontsize=6)
plt.tight_layout()
plt.show()


# info: SCHAEFER
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker

# Define the data directory where the Schaefer atlas will be downloaded
data_dir = 'C:\\Users\\oliver.frank\\Desktop\\BackUp\\bio_BeRNN'

# Step 1: Fetch the Schaefer 2018 atlas with 200 regions and 7 networks
schaefer_atlas = datasets.fetch_atlas_schaefer_2018(data_dir=data_dir, n_rois=200, yeo_networks=7)
atlas_filename = schaefer_atlas.maps
labels = schaefer_atlas.labels  # These are the names of the regions

# Step 2: Load your fMRI data
participant = 'sub-SNIP6IECX02'
task = 'flanker'
subject_directory = os.path.join('C:\\Users\\oliver.frank\\Desktop\\BackUp\\Brain_models', f'{participant}', 'func')
nifti_file = os.path.join(subject_directory, f'{participant}_task-{task}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')

fmri_img = nib.load(nifti_file)

# Step 3: Use the Schaefer atlas to extract time series data from each region
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, memory='nilearn_cache')
time_series = masker.fit_transform(fmri_img)

# Step 4: Compute the correlation matrix
correlation_matrix = np.corrcoef(time_series.T)

# Step 5: Plot the correlation matrix with region labels
num_regions = len(labels)

plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, vmin=-1, vmax=1, cmap='coolwarm')
plt.colorbar(label='Correlation coefficient')
plt.title('Correlation Matrix (Schaefer 200x7 Atlas)')

# Label the axes with the region names (truncated for display clarity if too long)
plt.xticks(ticks=np.arange(num_regions), labels=[label[:15] for label in labels], rotation=90, fontsize=6)
plt.yticks(ticks=np.arange(num_regions), labels=[label[:15] for label in labels], fontsize=6)
plt.tight_layout()
plt.show()


########################################################################################################################
# Topological Marker Analysis - Brain
########################################################################################################################
import networkx as nx
import numpy as np

def apply_threshold(matrix, threshold):
    # Set all values below the threshold to zero
    matrix_thresholded = np.where(np.abs(matrix) > threshold, matrix, 0)
    return matrix_thresholded

# Define a threshold (you can experiment with this value)
threshold = 0.2  # Example threshold
average_correlationMatrix_thresholded = apply_threshold(average_correlationMatrix, threshold)

# Function to apply a threshold to the matrix
G = nx.from_numpy_array(average_correlationMatrix_thresholded)

degrees = nx.degree(G) # For calculating node degrees. attention: transform into graph then apply stuff
betweenness = nx.betweenness_centrality(G) # For betweenness centrality.
closeness = nx.closeness_centrality(G) # For closeness centrality.
# average_path_length = nx.average_shortest_path_length(G) # For average path length. attention: Graph is disconnected after thresholding therefore not possible

# Optionally calculate averages of node-based metrics
avg_degree = np.mean(list(dict(G.degree()).values()))
avg_betweenness = np.mean(list(betweenness.values()))
avg_closeness = np.mean(list(closeness.values()))

# # Print or further process these metrics
# print("Node Degrees:", degrees)
# print("Betweenness Centrality:", betweenness)
# print("Closeness Centrality:", closeness)
# # print("Average Shortest Path Length:", average_path_length)

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

# Print or analyze these metrics
print(f"Graph Density: {density}")
print(f"Degree Assortativity: {assortativity}")
print(f"Transitivity: {transitivity}")
print(f"Average Clustering Coefficient: {avg_clustering}")
print(f"Average Degree: {avg_degree}")
print(f"Average Betweenness Centrality: {avg_betweenness}")
print(f"Average Closeness Centrality: {avg_closeness}")

# Graph Edit Distance: Measures the similarity between two networks based on the number of changes (insertions, deletions, substitutions)
# required to transform one graph into another.
# graph_edit_dist = nx.graph_edit_distance(G1, G2)




# # info: Covariance matrix for individual tasks subject-wise ############################################################
# # Step 1: Fetch the atlas data
# data_dir = 'Z:\\Desktop\\ZI\\PycharmProjects\\bio_BeRNN'
# atlas = datasets.fetch_atlas_msdl(data_dir=data_dir)
#
# # Load atlas images and labels
# atlas_filename = atlas["maps"]
# labels = atlas["labels"]
#
# # Step 2: Define paths for your data and confounds
# subject_directory = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\BrainModels\\derivatives\\fmriprep', 'sub-SNIP6IECX01', 'func')
#
# subjectDictionary = {
#     'maps': atlas["maps"],
#     'labels': atlas["labels"],
#     'data': os.path.join(subject_directory, 'sub-SNIP6IECX01_task-nback_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'),
#     'confounds': os.path.join(subject_directory, 'sub-SNIP6IECX01_task-nback_desc-confounds_timeseries.tsv')
# }
#
# # Step 3: Load and check the fMRI data
# func_img_BeRNN = load_img(subjectDictionary['data'])
#
# # Step 4: Load and inspect the confounds file
# confounds_df_BeRNN = pd.read_csv(subjectDictionary['confounds'], sep='\t')
# confounds_df_BeRNN_cleaned = confounds_df_BeRNN.fillna(0) # info: could also be filled with te mean .fillna(confounds_df_BeRNN.mean())
#
# # Check for NaN, Inf, -Inf values
# print("Are there NaN values?", confounds_df_BeRNN_cleaned.isnull().values.any())
# print("Are there Inf values?", (confounds_df_BeRNN_cleaned == float('inf')).values.any())
# print("Are there -Inf values?", (confounds_df_BeRNN_cleaned == float('-inf')).values.any())
# print(confounds_df_BeRNN_cleaned.describe())
#
# # Step 5: Use NiftiMapsMasker with the corrected confounds DataFrame
# masker = NiftiMapsMasker(
#     maps_img=subjectDictionary['maps'],
#     standardize='zscore',
#     standardize_confounds='zscore',
#     memory="nilearn_cache",
#     verbose=5,
# )
#
# # Step 6: Fit-transform the fMRI data using the cleaned confounds
# time_series = masker.fit_transform(subjectDictionary['data'], confounds=confounds_df_BeRNN_cleaned)
#
# # Step 7: Apply GraphicalLassoCV to estimate the covariance matrix
# estimator = GraphicalLassoCV()
# estimator.fit(time_series)
#
# # Step 8: Plot the covariance matrix
# plotting.plot_matrix(
#     estimator.covariance_,
#     labels=labels,
#     figure=(9, 7),
#     vmax=1,
#     vmin=-1,
#     title="",
# )
#
# save_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\BrainModels\\derivatives\\fmriprep',\
#                          'visuals\\CovarianceMatrix', subject_directory.split('\\')[-2],subjectDictionary['data'].split('_')[-5]+'.png')
# plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
# # Show the plot
# plt.show()
#
#
# # info: Average Covariance matrix over the different tasks subject-wise ################################################
# # Step 1: Fetch the atlas data
# data_dir = 'Z:\\Desktop\\ZI\\PycharmProjects\\bio_BeRNN'
# atlas = datasets.fetch_atlas_msdl(data_dir=data_dir)
# subject = 'sub-SNIPYL4AS01'
#
# # Load atlas images and labels
# atlas_filename = atlas["maps"]
# labels = atlas["labels"]
#
# # Step 2: Define the subject directory and tasks
# subject_directory = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\BrainModels\\derivatives\\fmriprep', subject, 'func')
# tasks = ['flanker', 'rest', 'nback', 'faces']
#
# # Step 3: Initialize an empty list to store covariance matrices
# covariances = []
#
# # Step 4: Iterate over each task to calculate the covariance matrix
# for task in tasks:
#     # Define the file paths for the current task
#     subject_dict = {
#         'maps': atlas["maps"],
#         'labels': atlas["labels"],
#         'data': os.path.join(subject_directory, f'{subject}_task-{task}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'),
#         'confounds': os.path.join(subject_directory, f'{subject}_task-{task}_desc-confounds_timeseries.tsv')
#     }
#
#     # Load the fMRI data
#     func_img_BeRNN = load_img(subject_dict['data'])
#
#     # Load and clean the confounds file
#     confounds_df_BeRNN = pd.read_csv(subject_dict['confounds'], sep='\t')
#     confounds_df_BeRNN_cleaned = confounds_df_BeRNN.fillna(0)  # Fill NaN values
#
#     # Initialize NiftiMapsMasker
#     masker = NiftiMapsMasker(
#         maps_img=subject_dict['maps'],
#         standardize='zscore',
#         standardize_confounds='zscore',
#         memory="nilearn_cache",
#         verbose=5,
#     )
#
#     # Fit-transform the fMRI data using the cleaned confounds
#     time_series = masker.fit_transform(subject_dict['data'], confounds=confounds_df_BeRNN_cleaned)
#
#     # Apply GraphicalLassoCV to estimate the covariance matrix
#     estimator = GraphicalLassoCV()
#     estimator.fit(time_series)
#
#     # Store the covariance matrix
#     covariances.append(estimator.covariance_)
#
# # Step 5: Compute the average covariance matrix across all tasks
# average_covariance = np.mean(covariances, axis=0)
#
# # Step 6: Plot the average covariance matrix
# plotting.plot_matrix(
#     average_covariance,
#     labels=labels,
#     figure=(9, 7),
#     vmax=1,
#     vmin=-1,
#     title="",
# )
#
# # Save the plot
# save_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\BrainModels\\derivatives\\fmriprep',\
#                          'visuals\\CovarianceMatrix',subject_directory.split('\\')[-2],'average_covariance_matrix_across_tasks.png')
# plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
#
# # Show the plot
# plt.show()


