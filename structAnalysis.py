import nibabel as nib
from nilearn import plotting
import numpy as np
import matplotlib.pyplot as plt



# info: T1-weighted Image - Use this file for individual-level structural analyses.
# Load the preprocessed T1-weighted image (in native space)
t1_img = nib.load('W:\\group_csp\\analyses\\oliver.frank\\Brain_models\\derivatives\\fmriprep\\sub-SNIP6IECX02\\anat\\sub-SNIP6IECX02_desc-preproc_T1w.nii.gz')

# Plot the image using Nilearn for visualization
plotting.plot_anat(t1_img, title="Preprocessed T1-weighted Image")
plt.show()



# info: Brain Mask - Use this file to isolate brain voxels from non-brain tissue.
# Load the brain mask
brain_mask_img = nib.load('W:\\group_csp\\analyses\\oliver.frank\\Brain_models\\derivatives\\fmriprep\\sub-SNIP6IECX02\\anat\\sub-SNIP6IECX02_desc-brain_mask.nii.gz')

# Mask the T1-weighted image (apply the brain mask)
t1_data = t1_img.get_fdata()
mask_data = brain_mask_img.get_fdata()

# Apply the mask (only keep voxels inside the brain mask)
masked_t1_data = t1_data * mask_data

# Create a new Nifti image for the masked data
masked_t1_img = nib.Nifti1Image(masked_t1_data, affine=t1_img.affine)

# Visualize the masked T1-weighted image
plotting.plot_anat(masked_t1_img, title="Masked T1-weighted Image")
plt.show()



# info: T1-weighted Image in MNI Space - Use this file for group-level analyses, where images are aligned to a common template
# Load the T1-weighted image in MNI space (2mm resolution)
t1_mni_img = nib.load('W:\\group_csp\\analyses\\oliver.frank\\Brain_models\\derivatives\\fmriprep\\sub-SNIP6IECX02\\anat\\sub-SNIP6IECX02_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz')

# Visualize the MNI-space T1-weighted image
plotting.plot_anat(t1_mni_img, title="T1-weighted Image in MNI Space (2mm)")
plt.show()



# Extract the voxel values from the masked T1-weighted image (brain only)
brain_voxels = masked_t1_data[mask_data > 0]

# Print basic statistics
print(f"Number of brain voxels: {len(brain_voxels)}")
print(f"Mean voxel intensity: {np.mean(brain_voxels)}")
print(f"Standard deviation: {np.std(brain_voxels)}")



# fix: Where to apply the topological markers on?
# fix: Cosine similarity good for structure? And function?


