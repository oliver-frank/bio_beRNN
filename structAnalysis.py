import nibabel as nib
from nilearn import plotting
from nilearn import datasets #, masking
from nilearn.input_data import NiftiLabelsMasker
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity



# # info: T1-weighted Image native - Use this file for individual-level structural analyses.
# # Load the preprocessed T1-weighted image (in native space)
# t1_img_native = nib.load('W:\\group_csp\\analyses\\oliver.frank\\Brain_models\\derivatives\\fmriprep\\sub-SNIP6IECX02\\anat\\sub-SNIP6IECX02_desc-preproc_T1w.nii.gz')
#
# # Plot the image using Nilearn for visualization
# plotting.plot_anat(t1_img_native, title="Preprocessed T1-weighted Image - native space")
# plt.show()
#
# # info: Brain Mask native - Use this file to isolate brain voxels from non-brain tissue.
# # Load the brain mask
# brain_mask_img_native = nib.load('W:\\group_csp\\analyses\\oliver.frank\\Brain_models\\derivatives\\fmriprep\\sub-SNIP6IECX02\\anat\\sub-SNIP6IECX02_desc-brain_mask.nii.gz')
#
# # Mask the T1-weighted image (apply the brain mask)
# t1_data_native = t1_img_native.get_fdata()
# mask_data_native = brain_mask_img_native.get_fdata()



# info: T1-weighted Image MNI - Use this file for individual-level structural analyses.
# Load the preprocessed T1-weighted image (in native space)
t1_img_MNI = nib.load('W:\\group_csp\\analyses\\oliver.frank\\Brain_models\\derivatives\\fmriprep\\sub-SNIP6IECX02\\anat\\sub-SNIP6IECX02_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz')

# Plot the image using Nilearn for visualization
plotting.plot_anat(t1_img_MNI, title="Preprocessed T1-weighted Image - MNI space")
plt.show()

# info: Brain Mask MNI - Use this file to isolate brain voxels from non-brain tissue.
# Load the brain mask
brain_mask_img_MNI = nib.load('W:\\group_csp\\analyses\\oliver.frank\\Brain_models\\derivatives\\fmriprep\\sub-SNIP6IECX02\\anat\\sub-SNIP6IECX02_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz')

# Mask the T1-weighted image (apply the brain mask)
t1_data_MNI = t1_img_MNI.get_fdata()
mask_data_MNI = brain_mask_img_MNI.get_fdata()

# info: Apply the mask (only keep voxels inside the brain mask)
masked_t1_data_MNI = t1_data_MNI * mask_data_MNI

# Create a new Nifti image for the masked data
masked_t1_img_MNI = nib.Nifti1Image(masked_t1_data_MNI, affine=t1_img_MNI.affine)

# Visualize the masked T1-weighted image
plotting.plot_anat(masked_t1_img_MNI, title="Masked T1-weighted Image - MNI space")
plt.show()

# info: Apply atlas regions on MNI voxels
# Load the AAL atlas (116 or 120 regions)
aal_atlas = datasets.fetch_atlas_aal(version='SPM12')

masker = NiftiLabelsMasker(labels_img=aal_atlas.maps, mask_img=brain_mask_img_MNI, standardize=True)

roi_time_series = masker.fit_transform(t1_img_MNI)

cosine_sim_matrix = cosine_similarity(roi_time_series.T)

# Visualize the cosine similarity matrix
plt.figure(figsize=(10, 8))
plt.imshow(cosine_sim_matrix, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Cosine Similarity Matrix of AAL ROIs')
plt.xlabel('ROIs')
plt.ylabel('ROIs')
plt.show()

# Visualize the AAL atlas on the T1-weighted image to ensure correct alignment
plotting.plot_roi(aal_atlas.maps, bg_img=t1_img_MNI, title="AAL Atlas over T1-weighted Image")
plt.show()



# # info: T1-weighted Image in MNI Space - Use this file for group-level analyses, where images are aligned to a common template
# # Load the T1-weighted image in MNI space (2mm resolution)
# t1_mni_img = nib.load('W:\\group_csp\\analyses\\oliver.frank\\Brain_models\\derivatives\\fmriprep\\sub-SNIP6IECX02\\anat\\sub-SNIP6IECX02_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz')
#
# # Visualize the MNI-space T1-weighted image
# plotting.plot_anat(t1_mni_img, title="T1-weighted Image in MNI Space (2mm)")
# plt.show()
#
#
#
# # Extract the voxel values from the masked T1-weighted image (brain only)
# brain_voxels = masked_t1_data[mask_data > 0]
#
# # Print basic statistics
# print(f"Number of brain voxels: {len(brain_voxels)}")
# print(f"Mean voxel intensity: {np.mean(brain_voxels)}")
# print(f"Standard deviation: {np.std(brain_voxels)}")



# fix: Where to apply the topological markers on?
# fix: Cosine similarity good for structure? And function?


