import os
import subprocess
import nibabel as nib
from pathlib import Path
import numpy as np
import glob
from pathlib import Path
import SimpleITK as sitk



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
dataset_path = Path(r'S:\datasets\s4_2')
scan_name = '471_MK_UTE_passive_6_iso_E1_1k2_iso'
time_dim = 3

scan = nib.load(f'{dataset_path}/{scan_name}.nii')
nda = scan.get_fdata(dtype=scan.get_data_dtype())
middle_im_idx = np.int(np.floor(nda.shape[time_dim]/2))
nda_middle = nda[:,:,:,middle_im_idx]
# deleted_frames = middle_im_idx
deleted_frames = [*list(range(0, (middle_im_idx-4))), middle_im_idx, *list(range(middle_im_idx+5, nda.shape[time_dim]))]  # optional: further reduce the size of the dynamic set (from the start & end of it) in order to test Makki's code
nda_rest = np.delete(nda, deleted_frames, time_dim)
# nib.Nifti1Image(nda_middle, scan.affine).to_filename(f'{dataset_path}\im_middle.nii')
nib.Nifti1Image(nda_rest, scan.affine).to_filename(f'{dataset_path}\im_rest_shorter_2.nii')

#%% s4_2 --> combine results of propagating masks, then overlay them on corresponding dyn frames
dataset_path = Path(r'S:\datasets\s4_2')
results_folder = 'results_when_-d_is_im_rest'
n_bones = 3
n_frms = 32

arb_scan = nib.load(f"{dataset_path}/im_middle.nii")      # just an arbitrary scan to get some params
affine = arb_scan.affine
im_shape = arb_scan.get_data().shape

cmbnd_masks = np.zeros([n_frms, *im_shape])
for t in range(n_frms):
    for i in range(n_bones):
        cmbnd_masks[t, :, :, :] += nib.load(f"{dataset_path}/{results_folder}/propagation/output_path_component{i}/final_results/"
                                            f"mask_dyn{str(t).zfill(4)}_component_{i}.nii.gz").get_fdata(dtype=arb_scan.get_data_dtype())
    # nib.Nifti1Image(cmbnd_masks[t], affine).to_filename(f'{dataset_path}/{results_folder}/cmbnd_masks_frame_{t}.nii.gz')

    ## overlay combined masks on the corresponding time-frame
    # reread ims using sitk to be in the right format for using a sitk filter (overlay)
    vol_dyn_sitk = sitk.ReadImage(f"{dataset_path}/{results_folder}/dyn{str(t).zfill(4)}.nii.gz")
    mask_combnd_sitk = sitk.ReadImage(f'{dataset_path}/{results_folder}/cmbnd_masks_frame_{t}.nii.gz', sitk.sitkUInt8)

    im_ovrlay_sitk = sitk.LabelOverlay(vol_dyn_sitk, mask_combnd_sitk)  # creates an RGB image (RGB -> colormap assigned to mask)

    ## write using nib (Amira will interpret the RGB as a time dimesion) (Amira can't read RGB ims generated by sitk!!)
    nib.Nifti1Image(sitk.GetArrayFromImage(im_ovrlay_sitk), affine).to_filename(
        f"{dataset_path}/{results_folder}/combined-masks-overlayed-on-dyn-frames/ovrlay_{t}.nii.gz")
