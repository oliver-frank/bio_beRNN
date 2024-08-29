########################################################################################################################
from nilearn import datasets
import os
import pandas as pd
from nilearn.image import load_img
from nilearn.maskers import NiftiMapsMasker
from sklearn.covariance import GraphicalLassoCV
from nilearn import plotting
import matplotlib.pyplot as plt

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
    'data': os.path.join(subject_directory, 'sub-SNIP6IECX01_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'),
    'confounds': os.path.join(subject_directory, 'sub-SNIP6IECX01_task-rest_desc-confounds_timeseries.tsv')
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

# Show the plot
plt.show()