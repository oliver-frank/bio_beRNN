########################################################################################################################
from nilearn import datasets
import os
import numpy as np
import pandas as pd
from nilearn.image import load_img
from nilearn.maskers import NiftiMapsMasker
from sklearn.covariance import GraphicalLassoCV
from nilearn import plotting
import matplotlib.pyplot as plt


# Covariance matrix for individual tasks subject-wise ##################################################################
# Step 1: Fetch the atlas data
data_dir = 'Z:\\Desktop\\ZI\\PycharmProjects\\bio_BeRNN'
atlas = datasets.fetch_atlas_msdl(data_dir=data_dir)

# Load atlas images and labels
atlas_filename = atlas["maps"]
labels = atlas["labels"]

# Step 2: Define paths for your data and confounds
subject_directory = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\BrainModels\\derivatives\\fmriprep', 'sub-SNIP6IECX01', 'func')

subjectDictionary = {
    'maps': atlas["maps"],
    'labels': atlas["labels"],
    'data': os.path.join(subject_directory, 'sub-SNIP6IECX01_task-nback_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'),
    'confounds': os.path.join(subject_directory, 'sub-SNIP6IECX01_task-nback_desc-confounds_timeseries.tsv')
}

# Step 3: Load and check the fMRI data
func_img_BeRNN = load_img(subjectDictionary['data'])

# Step 4: Load and inspect the confounds file
confounds_df_BeRNN = pd.read_csv(subjectDictionary['confounds'], sep='\t')
confounds_df_BeRNN_cleaned = confounds_df_BeRNN.fillna(0) # info: could also be filled with te mean .fillna(confounds_df_BeRNN.mean())

# Check for NaN, Inf, -Inf values
print("Are there NaN values?", confounds_df_BeRNN_cleaned.isnull().values.any())
print("Are there Inf values?", (confounds_df_BeRNN_cleaned == float('inf')).values.any())
print("Are there -Inf values?", (confounds_df_BeRNN_cleaned == float('-inf')).values.any())
print(confounds_df_BeRNN_cleaned.describe())

# Step 5: Use NiftiMapsMasker with the corrected confounds DataFrame
masker = NiftiMapsMasker(
    maps_img=subjectDictionary['maps'],
    standardize='zscore',
    standardize_confounds='zscore',
    memory="nilearn_cache",
    verbose=5,
)

# Step 6: Fit-transform the fMRI data using the cleaned confounds
time_series = masker.fit_transform(subjectDictionary['data'], confounds=confounds_df_BeRNN_cleaned)

# Step 7: Apply GraphicalLassoCV to estimate the covariance matrix
estimator = GraphicalLassoCV()
estimator.fit(time_series)

# Step 8: Plot the covariance matrix
plotting.plot_matrix(
    estimator.covariance_,
    labels=labels,
    figure=(9, 7),
    vmax=1,
    vmin=-1,
    title="",
)

save_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\BrainModels\\derivatives\\fmriprep',\
                         'visuals\\CovarianceMatrix', subject_directory.split('\\')[-2],subjectDictionary['data'].split('_')[-5]+'.png')
plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
# Show the plot
plt.show()





########################################################################################################################
from nilearn import datasets
import os
import numpy as np
import pandas as pd
from nilearn.image import load_img
from nilearn.maskers import NiftiMapsMasker
from sklearn.covariance import GraphicalLassoCV
from nilearn import plotting
import matplotlib.pyplot as plt
# Average Covariance matrix over the different tasks subject-wise ######################################################
# Step 1: Fetch the atlas data
data_dir = 'Z:\\Desktop\\ZI\\PycharmProjects\\bio_BeRNN'
atlas = datasets.fetch_atlas_msdl(data_dir=data_dir)
subject = 'sub-SNIPYL4AS01'

# Load atlas images and labels
atlas_filename = atlas["maps"]
labels = atlas["labels"]

# Step 2: Define the subject directory and tasks
subject_directory = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\BrainModels\\derivatives\\fmriprep', subject, 'func')
tasks = ['flanker', 'rest', 'nback', 'faces']

# Step 3: Initialize an empty list to store covariance matrices
covariances = []

# Step 4: Iterate over each task to calculate the covariance matrix
for task in tasks:
    # Define the file paths for the current task
    subject_dict = {
        'maps': atlas["maps"],
        'labels': atlas["labels"],
        'data': os.path.join(subject_directory, f'{subject}_task-{task}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'),
        'confounds': os.path.join(subject_directory, f'{subject}_task-{task}_desc-confounds_timeseries.tsv')
    }

    # Load the fMRI data
    func_img_BeRNN = load_img(subject_dict['data'])

    # Load and clean the confounds file
    confounds_df_BeRNN = pd.read_csv(subject_dict['confounds'], sep='\t')
    confounds_df_BeRNN_cleaned = confounds_df_BeRNN.fillna(0)  # Fill NaN values

    # Initialize NiftiMapsMasker
    masker = NiftiMapsMasker(
        maps_img=subject_dict['maps'],
        standardize='zscore',
        standardize_confounds='zscore',
        memory="nilearn_cache",
        verbose=5,
    )

    # Fit-transform the fMRI data using the cleaned confounds
    time_series = masker.fit_transform(subject_dict['data'], confounds=confounds_df_BeRNN_cleaned)

    # Apply GraphicalLassoCV to estimate the covariance matrix
    estimator = GraphicalLassoCV()
    estimator.fit(time_series)

    # Store the covariance matrix
    covariances.append(estimator.covariance_)

# Step 5: Compute the average covariance matrix across all tasks
average_covariance = np.mean(covariances, axis=0)

# Step 6: Plot the average covariance matrix
plotting.plot_matrix(
    average_covariance,
    labels=labels,
    figure=(9, 7),
    vmax=1,
    vmin=-1,
    title="",
)

# Save the plot
save_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\BrainModels\\derivatives\\fmriprep',\
                         'visuals\\CovarianceMatrix',subject_directory.split('\\')[-2],'average_covariance_matrix_across_tasks.png')
plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()


