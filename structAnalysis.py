import nibabel as nib
from nilearn import plotting

import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import cosine_similarity

participant = 'sub-SNIP6IECX02'

# info: T1-weighted Image native
# Load the preprocessed T1-weighted image (in native space)
t1_img_native = nib.load(f'W:\\group_csp\\analyses\\oliver.frank\\brainModels\\{participant}\\anat\\{participant}_desc-preproc_T1w.nii.gz')
# Plot the image using Nilearn for visualization
plotting.plot_anat(t1_img_native, title="Preprocessed T1-weighted Image - native space")
plt.show()
# Load the brain mask in native
brain_mask_img_native = nib.load(f'W:\\group_csp\\analyses\\oliver.frank\\brainModels\\{participant}\\anat\\{participant}_desc-brain_mask.nii.gz')
# Mask the T1-weighted image (apply the brain mask)
t1_data_native = t1_img_native.get_fdata()
mask_data_native = brain_mask_img_native.get_fdata()
masked_t1_data_MNI = t1_data_native * mask_data_native
# Create a new Nifti image for the masked data
masked_t1_img_MNI = nib.Nifti1Image(masked_t1_data_MNI, affine=t1_img_native.affine)
# Visualize the masked T1-weighted image
plotting.plot_anat(masked_t1_img_MNI, title="Masked T1-weighted Image - native space")
plt.show()

# info: T1-weighted Image MNI
# Load the preprocessed T1-weighted image (in native space)
t1_img_MNI = nib.load(f'W:\\group_csp\\analyses\\oliver.frank\\brainModels\\{participant}\\anat\\{participant}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz')
# Plot the image using nilearn for visualization
plotting.plot_anat(t1_img_MNI, title="Preprocessed T1-weighted Image - MNI space")
plt.show()
# Load the brain mask in MNI
brain_mask_img_MNI = nib.load(f'W:\\group_csp\\analyses\\oliver.frank\\brainModels\\{participant}\\anat\\{participant}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz')
# Mask the T1-weighted image (apply the brain mask)
t1_data_MNI = t1_img_MNI.get_fdata()
mask_data_MNI = brain_mask_img_MNI.get_fdata()
# Apply the mask (only keep voxels inside the brain mask)
masked_t1_data_MNI = t1_data_MNI * mask_data_MNI
# Create a new Nifti image for the masked data
masked_t1_img_MNI = nib.Nifti1Image(masked_t1_data_MNI, affine=t1_img_MNI.affine)
# Visualize the masked T1-weighted image
plotting.plot_anat(masked_t1_img_MNI, title="Masked T1-weighted Image - MNI space")
plt.show()



