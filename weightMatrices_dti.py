import os
import numpy as np
from dipy.io.image import load_nifti
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.tracking.utils import seeds_from_mask, connectivity_matrix
from dipy.tracking.streamline import transform_streamlines, Streamlines
from sklearn.cluster import KMeans
from scipy.ndimage import zoom

def run_pipeline(base_path, participant):
    dwi_nii = os.path.join(base_path, 'dwi', f'025_sub-{participant}_diff_PA_257_eddy_corrected.nii.gz')
    bval = os.path.join(base_path, 'dwi', f'025_sub-{participant}_diff_PA_257.bval')
    bvec = os.path.join(base_path, 'dwi', f'025_sub-{participant}_diff_PA_257_eddy_corrected.eddy_rotated_bvecs')
    atlas_nii = os.path.join(base_path, 'atlas', 'atlas_resampled_to_dwi.nii.gz')
    out_path = os.path.join(base_path, 'output')
    os.makedirs(out_path, exist_ok=True)

    print("Loading DWI and gradient data...")
    data, affine = load_nifti(dwi_nii)
    bvals = np.loadtxt(bval)
    bvecs = np.loadtxt(bvec).T
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)

    print("Fitting tensor model...")
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data)
    fa = fractional_anisotropy(tenfit.evals)
    fa[np.isnan(fa)] = 0
    mask = fa > 0.25
    seeds = seeds_from_mask(mask, affine=affine, density=1)
    evecs = tenfit.evecs[..., 0]

    print("Generating streamlines...")
    streamlines = []
    for seed in seeds:
        streamline = []
        pos = np.array(seed)
        for _ in range(100):  # max steps
            idx = np.round(pos).astype(int)
            if np.any(idx < 0) or np.any(idx >= data.shape[:3]) or fa[tuple(idx)] < 0.2:
                break
            direction = evecs[tuple(idx)]
            if np.linalg.norm(direction) == 0:
                break
            pos += direction
            streamline.append(pos.copy())
        if len(streamline) > 10:
            streamlines.append(np.array(streamline))

    print(f"{len(streamlines)} streamlines generated.")

    print("Filtering streamlines that leave the brain volume...")
    valid_streamlines = Streamlines([
        s for s in streamlines
        if np.all((s >= 0) & (s < data.shape[:3]))
    ])
    print(f"{len(valid_streamlines)} valid streamlines remain.")

    print("Loading atlas...")
    atlas_img, _ = load_nifti(atlas_nii)

    print("Transforming streamlines to atlas voxel space...")
    lin_T = np.linalg.inv(affine)
    transformed_streamlines = list(transform_streamlines(valid_streamlines, lin_T))

    # Clip coordinates to valid range
    atlas_shape = np.array(atlas_img.shape)

    clipped_streamlines = Streamlines([
        np.clip(s, [0, 0, 0], atlas_shape - 1) for s in transformed_streamlines
    ])
    print(f"{len(clipped_streamlines)} streamlines remain after transformation and filtering.")

    print("🔗 Constructing 300×300 connectivity matrix...")
    conn_matrix, _ = connectivity_matrix(
        clipped_streamlines, affine=np.eye(4), label_volume=atlas_img.astype(int),
        return_mapping=True, mapping_as_streamlines=False
    )
    conn_matrix = conn_matrix.astype(float)
    conn_matrix = (conn_matrix + conn_matrix.T) / 2
    conn_matrix /= conn_matrix.max()
    np.save(os.path.join(out_path, f'connectome_{participantDictionary[participant]}_300.npy'), conn_matrix)

    # for res in [256, 128, 64, 32]:
    #     print(f"Downsampling to {res}×{res}...")
    #     # Get unique atlas labels (excluding 0 = background)
    #     unique_labels = np.unique(atlas_img)
    #     unique_labels = unique_labels[unique_labels > 0]
    #
    #     # Cluster the unique label indices (not voxel values)
    #     label_indices = np.arange(len(unique_labels)).reshape(-1, 1)
    #     kmeans = KMeans(n_clusters=res, random_state=42).fit(label_indices)
    #     clustered_labels = kmeans.labels_
    #     downsampled = np.zeros((res, res))
    #
    #     for i in range(res):
    #         idx_i = np.where(clustered_labels == i)[0]
    #         for j in range(res):
    #             idx_j = np.where(clustered_labels == j)[0]
    #             downsampled[i, j] = np.mean(conn_matrix[np.ix_(idx_i, idx_j)])
    #     downsampled = (downsampled + downsampled.T) / 2
    #     downsampled /= downsampled.max()
    #     np.save(os.path.join(out_path, f'connectome_{participantDictionary[participant]}_{res}.npy'), downsampled)

    # Create one upsampled connectome weigth matrix
    # conn_matrix is 300x300
    upsampled_512 = zoom(conn_matrix, (512 / 300, 512 / 300), order=1)  # bilinear interpolation
    upsampled_512 = (upsampled_512 + upsampled_512.T) / 2  # ensure symmetry
    upsampled_512 = upsampled_512[:512, :512]  # crop to exact size
    upsampled_512 /= upsampled_512.max()

    np.save(os.path.join(out_path, f'connectome_{participantDictionary[participant]}_512.npy'),
            upsampled_512.astype(np.float32))

    print("All connectome matrices saved.")


participantDictionary = {
    'SNIPKPB8401': 'beRNN_01',
    'SNIPYL4AS01': 'beRNN_02',
    'SNIP6IECX01': 'beRNN_03',
    'SNIPDKHPB01': 'beRNN_04',
    'SNIP96WID01': 'beRNN_05'
}

# Run the pipeline:
participant = 'SNIPKPB8401'
run_pipeline(rf"C:\Users\oliver.frank\Desktop\PyProjects\bio_BeRNN\weightMatrices_dwi\{participantDictionary[participant]}\ses-01", participant)


