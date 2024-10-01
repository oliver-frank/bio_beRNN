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

import networkx as nx # info: Need to extract an graph first from your matrices

def convert_atlas_to_4d(atlas_img):
    """
    Convert a 3D atlas to a 4D image where each region in the atlas is represented
    as a separate binary mask in the fourth dimension.

    Parameters:
    - atlas_img: Nibabel Nifti1Image, the 3D atlas image.

    Returns:
    - nib.Nifti1Image: A 4D Nifti image where the fourth dimension corresponds to each region.
    """
    atlas_data = atlas_img.get_fdata()
    unique_labels = np.unique(atlas_data)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background
    atlas_4d = np.zeros((*atlas_data.shape, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        atlas_4d[..., i] = (atlas_data == label).astype(np.float32)

    atlas_4d_img = nib.Nifti1Image(atlas_4d, affine=atlas_img.affine)
    return atlas_4d_img



# info: Cosine similiarity matrix for individual tasks subject-wise ####################################################
participants = ['sub-SNIP6IECX02'] # , 'sub-SNIP96WID02', 'sub-SNIPDKHPB02', 'sub-SNIPKPB8402', 'sub-SNIPYL4AS02'
# participants = ['sub-SNIP6IECX01']
for participant in participants:
    tasks = ['flanker'] # , 'rest', 'nback', 'faces'
    # tasks = ['flanker']
    for task in tasks:
        # Step 1: Fetch the atlas data
        data_dir = 'Z:\\Desktop\\ZI\\PycharmProjects\\bio_BeRNN'
        '''
        The atlas consists of 39 components or regions of interest (ROIs). 
        Each component is a spatial map representing a brain region that tends to be co-activated with other regions in 
        the same network during resting state.
        
        You should take a different one when analyzing brain networks representing activity during certain tasks.
        '''
        # atlas, mode  = datasets.fetch_atlas_msdl(data_dir=data_dir), 'msdl' # 39 regions - broader cognitive function like default mode network, visual and motor; based on resting state
        atlas, mode = datasets.fetch_atlas_aal(data_dir=data_dir), 'aal' # 116 to 120 regions - Anatomical studies - anatomy based
        # atlas, mode = datasets.fetch_atlas_yeo_2011(data_dir=data_dir), 'yeo' # 7 to 17 subnetworks - studies involving broad functional networks # fix: not implemented yet
        # atlas, mode = datasets.fetch_atlas_basc_multiscale_2015(data_dir=data_dir), 'basc' # 64 to 444 regions - network analysis - resting state based

        # Load atlas images and labels
        if mode == 'msdl' or mode == 'aal':
            atlas_filename = atlas["maps"]
            labels = atlas["labels"]
        elif mode == 'basc':
            map = '036'
            atlas_filename = atlas[f"scale{map}"] # valid parcallations are {007, 012, 020, 036, 064, 122, 197, 325, 444}
            labels = None

        # Step 2: Define paths for your data and confounds
        subject_directory = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\Brain_models\\derivatives\\fmriprep', f'{participant}', 'func')

        subjectDictionary = {
            'maps': atlas_filename,
            'labels': labels,
            'data': os.path.join(subject_directory, f'{participant}_task-{task}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'),
            'confounds': os.path.join(subject_directory, f'{participant}_task-{task}_desc-confounds_timeseries.tsv')
        }

        # attention ####################################################################################################
        if mode == 'aal' or mode == 'basc':
            # info: Overwrite old with new 4d atlas maps
            # Load the atlas and fMRI data
            atlas_img = nib.load(subjectDictionary['maps'])
            func_img = nib.load(subjectDictionary['data'])
            # Convert the 3D atlas to a 4D atlas
            atlas_4d_img = convert_atlas_to_4d(atlas_img)

            resampled_atlas_4d_img = resample_to_img(
                atlas_4d_img,  # The atlas image (e.g., scale444)
                func_img,  # The fMRI data image to match
                interpolation='nearest'  # Use nearest-neighbor interpolation
            )

            if mode == 'aal':
                # Save the 4D atlas if needed
                fourDimFile = os.path.join('\\'.join(subjectDictionary['maps'].split('\\')[:-1]), 'AAL_4DIM.nii.gz')
                nib.save(atlas_4d_img, fourDimFile)
                subjectDictionary['maps'] = os.path.join('\\'.join(subjectDictionary['maps'].split('\\')[:-1]), 'AAL_4DIM.nii.gz')
            elif mode == 'basc':
                # Save the 4D atlas if needed
                fourDimFile = os.path.join('\\'.join(subjectDictionary['maps'].split('\\')[:-1]), f'template_cambridge_basc_multiscale_sym_scale{map}_4DIM.nii.gz')
                nib.save(atlas_4d_img, fourDimFile)
                subjectDictionary['maps'] = os.path.join('\\'.join(subjectDictionary['maps'].split('\\')[:-1]), f'template_cambridge_basc_multiscale_sym_scale{map}_4DIM.nii.gz')

            # Check if their affine matrices (spatial transformations) are similar
            print("Atlas affine matrix:\n", atlas_img.affine)
            print("Atlas 4D affine matrix:\n", atlas_4d_img.affine)
            print("fMRI affine matrix:\n", func_img.affine)

            # Check their shapes
            print("Atlas shape:", atlas_img.shape)
            print("Atlas 4D shape:", atlas_4d_img.shape)
            print("fMRI shape:", func_img.shape)
        # attention ####################################################################################################

        # Step 4: Load and inspect the confounds file
        confounds_df_BeRNN = pd.read_csv(subjectDictionary['confounds'], sep='\t')
        confounds_df_BeRNN_cleaned = confounds_df_BeRNN.fillna(0) # info: could also be filled with the mean .fillna(confounds_df_BeRNN.mean())

        # attention ####################################################################################################
        # Check for NaN, Inf, -Inf values
        print("Are there NaN values?", confounds_df_BeRNN_cleaned.isnull().values.any())
        print("Are there Inf values?", (confounds_df_BeRNN_cleaned == float('inf')).values.any())
        print("Are there -Inf values?", (confounds_df_BeRNN_cleaned == float('-inf')).values.any())
        print(confounds_df_BeRNN_cleaned.describe())
        # attention ####################################################################################################

        # Step 5: Use NiftiMapsMasker with the corrected confounds DataFrame
        masker = NiftiMapsMasker(
            maps_img=subjectDictionary['maps'],
            standardize=True, # 'zscore'
            standardize_confounds=True, # 'zscore'
            resampling_target='data', # info: Important for mode = 'basc' as provided atlas map doesn't naturally have time dimension
            memory="nilearn_cache",
            verbose=5,
        )

        # Step 6: Fit-transform the fMRI data using the cleaned confounds
        time_series = masker.fit_transform(subjectDictionary['data'], confounds=confounds_df_BeRNN_cleaned)

        # attention ####################################################################################################
        # Check the shape and some statistics of the time series
        print(f"Shape of time series: {time_series.shape}")  # Should be (time_points, regions)
        print(f"Mean: {time_series.mean()}, Std: {time_series.std()}")  # Check for variance

        # Optionally plot a few time series to visually inspect
        import matplotlib.pyplot as plt

        plt.plot(time_series[:, :5])  # Plot first 5 time series
        plt.title('First 5 time series from BASC Atlas')
        plt.show()
        # attention ####################################################################################################

        # Step 7: Compute cosine similarity matrix
        similarity = cosine_similarity(time_series.T)

        # Step 8: Plot the cosine similarity matrix
        # Set up the figure
        fig = plt.figure(figsize=(10, 10))

        # Create the main similarity matrix plot
        matrix_left = 0.1
        matrix_bottom = 0.3
        matrix_width = 0.6
        matrix_height = 0.6

        ax_matrix = fig.add_axes([matrix_left, matrix_bottom, matrix_width, matrix_height])
        im = ax_matrix.imshow(similarity, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)

        # Add title
        ax_matrix.set_title(f'Functional Cosine Similiarity - Brain - {task}', fontsize=22, pad=20)
        if mode == 'msdl' or mode == 'aal':
            # Add x-axis and y-axis labels with MSDL region labels
            ax_matrix.set_xticks(range(len(labels)))  # Set the number of ticks to match the number of labels
            ax_matrix.set_yticks(range(len(labels)))  # Set the number of ticks to match the number of labels
            ax_matrix.set_xticklabels(labels, rotation=90, fontsize=10)  # Rotate the x-axis labels for readability
            ax_matrix.set_yticklabels(labels, fontsize=10)
        else:
            # Add x-axis and y-axis labels
            ax_matrix.set_xlabel('Brain regions', fontsize=16, labelpad=15)
            ax_matrix.set_ylabel('Brain regions', fontsize=16, labelpad=15)
            # Remove x and y ticks
            ax_matrix.set_xticks([])  # Disable x-ticks
            ax_matrix.set_yticks([])  # Disable y-ticks

        # Create the colorbar on the right side, aligned with the matrix
        colorbar_left = matrix_left + matrix_width + 0.02
        colorbar_width = 0.03

        ax_cb = fig.add_axes([colorbar_left, matrix_bottom, colorbar_width, matrix_height])
        cb = plt.colorbar(im, cax=ax_cb)
        cb.set_ticks([-1, 1])
        cb.outline.set_linewidth(0.5)
        cb.set_label('Similarity', fontsize=18, labelpad=0)

        # # Set the title above the similarity matrix, centered
        # if mode == 'Training':
        #     title = '_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TRAINING'
        # elif mode == 'Evaluation':
        #     title = '_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TEST'

        # ax_matrix.set_title(title, fontsize=14, pad=20)

        # Step 9: Save the figure
        save_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\Brain_models\\derivatives\\fmriprep',\
                                 'visuals\\Similiarity_fMRI',mode, subject_directory.split('\\')[-2], subjectDictionary['data'].split('_')[-5]+'.png')
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

        # Step 10: Show the plot
        plt.show()











# info: Average Cosine similiarity matrix over the different tasks subject-wise ########################################
participants = ['sub-SNIP6IECX01', 'sub-SNIP96WID01', 'sub-SNIPDKHPB01', 'sub-SNIPKPB8401', 'sub-SNIPYL4AS01']
for participant in participants:
    # Step 1: Fetch the atlas data
    data_dir = 'Z:\\Desktop\\ZI\\PycharmProjects\\bio_BeRNN'
    atlas = datasets.fetch_atlas_msdl(data_dir=data_dir)

    # Load atlas images and labels
    atlas_filename = atlas["maps"]
    labels = atlas["labels"]

    # Step 2: Define the subject directory and tasks
    subject_directory = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\BrainModels\\derivatives\\fmriprep', participant, 'func')
    tasks = ['flanker', 'rest', 'nback', 'faces']

    # Step 3: Initialize an empty list to store covariance matrices
    cosineSimiliarities = []

    # Step 4: Iterate over each task to calculate the covariance matrix
    for task in tasks:
        # Define the file paths for the current task
        subject_dict = {
            'maps': atlas["maps"],
            'labels': atlas["labels"],
            'data': os.path.join(subject_directory, f'{participant}_task-{task}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'),
            'confounds': os.path.join(subject_directory, f'{participant}_task-{task}_desc-confounds_timeseries.tsv')
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

        # Step 5: Compute cosine similarity matrix
        similarity = cosine_similarity(time_series.T)

        # Store the covariance matrix
        cosineSimiliarities.append(similarity)

    # Step 5: Compute the average covariance matrix across all tasks
    average_cosineSimiliarity = np.mean(cosineSimiliarities, axis=0)

    # Step 8: Plot the cosine similarity matrix
    # Set up the figure
    fig = plt.figure(figsize=(10, 10))

    # Create the main similarity matrix plot
    matrix_left = 0.1
    matrix_bottom = 0.3
    matrix_width = 0.6
    matrix_height = 0.6

    ax_matrix = fig.add_axes([matrix_left, matrix_bottom, matrix_width, matrix_height])
    im = ax_matrix.imshow(similarity, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)

    # Add title
    ax_matrix.set_title('Average Functional Cosine Similiarity - Brain', fontsize=22, pad=20)
    # Add x-axis and y-axis labels
    ax_matrix.set_xlabel('Brain regions', fontsize=16, labelpad=15)
    ax_matrix.set_ylabel('Brain regions', fontsize=16, labelpad=15)

    # Remove x and y ticks
    ax_matrix.set_xticks([])  # Disable x-ticks
    ax_matrix.set_yticks([])  # Disable y-ticks

    # Create the colorbar on the right side, aligned with the matrix
    colorbar_left = matrix_left + matrix_width + 0.02
    colorbar_width = 0.03

    ax_cb = fig.add_axes([colorbar_left, matrix_bottom, colorbar_width, matrix_height])
    cb = plt.colorbar(im, cax=ax_cb)
    cb.set_ticks([-1, 1])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Similarity', fontsize=18, labelpad=0)

    # # Set the title above the similarity matrix, centered
    # if mode == 'Training':
    #     title = '_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TRAINING'
    # elif mode == 'Evaluation':
    #     title = '_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TEST'

    # ax_matrix.set_title(title, fontsize=14, pad=20)

    # Step 9: Save the figure
    save_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\BrainModels\\derivatives\\fmriprep', \
                             'visuals\\Similiarity_fMRI',subject_directory.split('\\')[-2],'average_SimiliarityCosine' + '.png')
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

    # Step 10: Show the plot
    plt.show()














# info: Covariance matrix for individual tasks subject-wise ############################################################
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


# info: Average Covariance matrix over the different tasks subject-wise ################################################
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


