import os
import subprocess
import nibabel as nib
from pathlib import Path
import numpy as np
import glob
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm



# (MHH) an alias for more easier nii utils
def nii__print_scan_header(scan_path):
    print(nib.load(scan_path).header)


#%% play with os.system() & invoking the WSL bash shell through it

go = '"' + r'C:\Program Files\WindowsApps\CanonicalGroupLimited.Ubuntu18.04onWindows_1804.2020.824.0_x64__79rhkp1fndgsc' \
           r'\ubuntu1804.exe' + '"' + ' ' \
          + 'run' + ' ' + './invoke_FSL.sh'
           # + '-help'
# os.system(go)
subprocess.run(go)

#%% s4_2 -> separate the scan to a "3d image from the middle of range of motion (will be S)", & "the rest (will be the dynamic seq D)"
dataset__path = Path(r'S:\datasets\s4_2')
scan_name = '471_MK_UTE_passive_6_iso_E1_1k2_iso'
time_dim = 3

scan = nib.load(f'{dataset__path}/{scan_name}.nii')
nda = scan.get_fdata(dtype=scan.get_data_dtype())
middle_im_idx = np.int(np.floor(nda.shape[time_dim]/2))
nda_middle = nda[:,:,:,middle_im_idx]
# deleted_frames = middle_im_idx
deleted_frames = [*list(range(0, (middle_im_idx-4))), middle_im_idx, *list(range(middle_im_idx+5, nda.shape[time_dim]))]  # optional: further reduce the size of the dynamic set (from the start & end of it) in order to test Makki's code
nda_rest = np.delete(nda, deleted_frames, time_dim)
# nib.Nifti1Image(nda_middle, scan.affine).to_filename(f'{dataset__path}\im_middle.nii')
nib.Nifti1Image(nda_rest, scan.affine).to_filename(f'{dataset__path}\dynSeq_shorter_2.nii')

#%% s4_2 -> combine masks of "im_middle" into one
dataset__path = Path(r'S:\datasets\s4_2')
bones = ('femur', 'tibia', 'patella')
arb_scan = nib.load(f"{dataset__path}/im_middle.nii")
cmbnd_mask_nda = np.zeros(arb_scan.get_fdata(dtype=np.float32).shape)
for bone in bones:
    cmbnd_mask_nda += nib.load(f"{dataset__path}/masks/im_middle.labels_{bone}.nii").get_fdata(dtype=np.float32)
nib.Nifti1Image(cmbnd_mask_nda, arb_scan.affine).to_filename(f'{dataset__path}/masks/im_middle.labels_allBones.nii')


#%% s4_2 --> combine results of propagating masks, then overlay them on corresponding dyn frames
dataset__path = Path(r'S:\datasets\s4_2')
results_folder = 'results-motionEst--at-d-full_seq__modified_rotational_srch'     #{results-motionEst--at-d-full_seq__modified_rotational_srch}
n_bones = 3
n_frms = 32

im_middle_nib = nib.load(f"{dataset__path}/im_middle.nii")      # just an arbitrary scan to get some params
affine = im_middle_nib.affine
im_shape = im_middle_nib.get_data().shape

# os.mkdir(f"{dataset__path}/{results_folder}/combined-masks-overlayed-on-dyn-frames")

cmbnd_wrpd_frms__ndarr = np.zeros([*im_shape,n_frms])
for t in range(n_frms):
    for i in range(n_bones):
        cmbnd_wrpd_frms__ndarr[:,:,:,t] += nib.load(f"{dataset__path}/{results_folder}/propagation/output_path_component{i}/final_results/"
                                            f"mask_dyn{str(t).zfill(4)}_component_{i}.nii.gz").get_fdata(dtype=im_middle_nib.get_data_dtype())
    #
    # ## overlay combined masks on the corresponding time-frame
    # # reread ims using sitk to be in the right format for using a sitk filter (overlay)
    # vol_dyn_sitk = sitk.ReadImage(f"{dataset__path}/{results_folder}/dyn{str(t).zfill(4)}.nii.gz")
    # mask_combnd_sitk = sitk.ReadImage(f'{dataset__path}/{results_folder}/cmbnd_masks_frame_{t}.nii.gz', sitk.sitkUInt8)
    #
    # im_ovrlay_sitk = sitk.LabelOverlay(vol_dyn_sitk, mask_combnd_sitk)  # creates an RGB image (RGB -> colormap assigned to mask)
    #
    # ## write using nib (Amira will interpret the RGB as a time dimesion) (Amira can't read RGB ims generated by sitk!!)
    # nib.Nifti1Image(sitk.GetArrayFromImage(im_ovrlay_sitk), affine).to_filename(
    #     f"{dataset__path}/{results_folder}/combined-masks-overlayed-on-dyn-frames/ovrlay_{t}.nii.gz")

nib.Nifti1Image(cmbnd_wrpd_frms__ndarr, affine).to_filename(f'{dataset__path}/{results_folder}/cmbnd_warpd_masks_4D.nii.gz')





#%% s4_2 --> after "run_transformFusion.sh" --> combine results of warping im_middle to dynSeq in a single 4d dynamic nii file (to generate a vid in Amira easily)
dataset__path = Path(r'S:\datasets\s4_2')
results__folder = "results-transFusion--at-d-full_seq__modified_rotational_srch--input-bonesMask"
dynSeq__file = 'dynSeq.nii'      # {dynSeq.nii, padded_dynSeq.nii}

dynSeq__scan = nib.load(f"{dataset__path}/{dynSeq__file}")
affine = dynSeq__scan.affine
dynSeq__ndarr = dynSeq__scan.get_fdata(dtype=np.float32)
im_shape = dynSeq__ndarr.shape
n_frms = im_shape[3]

cmbnd_wrpd_frms__ndarr = np.zeros(im_shape)
for t in tqdm(range(n_frms)):
    cmbnd_wrpd_frms__ndarr[:, :, :, -(t + 1)] += nib.load(f"{dataset__path}/{results__folder}/result-warping-S-to-dyn{str(t).zfill(4)}/"
                                            f"warped_dyn_frame_dyn{str(t).zfill(4)}.nii.gz").get_fdata(dtype=np.float32)
    # TODO: why reversed? seems a simple issue (an index reversed somewhere in the pipeline)
    # cmbnd_wrpd_frms__ndarr[:,:,:,t] += nib.load(f"{dataset__path}/result-warping-S-to-dyn{str(t).zfill(4)}/"
    #                                         f"warped_dyn_frame.nii.gz").get_fdata(dtype=im_middle_nib.get_data_dtype())

nib.Nifti1Image(cmbnd_wrpd_frms__ndarr, affine).to_filename(f'{dataset__path}/{results__folder}/result--cmbnd_wrpd_frms.nii')

# generate difference image
diff_im__ndarr = dynSeq__ndarr - cmbnd_wrpd_frms__ndarr
nib.Nifti1Image(diff_im__ndarr, affine).to_filename(f'{dataset__path}/{results__folder}/result--diff_im.nii')




#%% s4_2 -> pad dyn seq at both ends (temporally by duplicating first & last 3d frames) to test if that reduces warping error at distal tibia of "transformFusion" at these ends
dataset__path = Path(r'S:\datasets\s4_2')
n_padding = 11

dynSeq_scan = nib.load(f"{dataset__path}/dynSeq.nii")
padded_dynSeq_ndarray = np.pad(dynSeq_scan.get_fdata(dtype=np.float32),
                               ((0,0),(0,0),(0,0),(n_padding,n_padding)),   # only pad in time
                               mode='edge')
nib.Nifti1Image(padded_dynSeq_ndarray, dynSeq_scan.affine).to_filename(f"{dataset__path}/padded_dynSeq.nii")
















