#!/bin/bash

# shellcheck disable=SC2164
#cd /mnt/s/datasets/s4_2/

#scp /mnt/c/Users/bzfmuham/OneDrive/Knee-Kinematics/Code_and_utils/dynMRI/transformFusion.py bzfmuham@vsl4:/local/bzfmuham/Code_and_utils/dynMRI/

ssh bzfmuham@vsl4 echo "hello world"

#for i in {0..9}
#do
#  echo "warping static image to dyn_000${i}"
#  python3 /mnt/c/Users/bzfmuham/OneDrive/Knee-Kinematics/Code_and_utils/dynMRI/transformfusion.py -in im_middle.nii -refweight ./results-at-d-full_seq/propagation/output_path_component0/final_results/mask_dyn000${i}_component_0.nii.gz -refweight ./results-at-d-full_seq/propagation/output_path_component1/final_results/mask_dyn000${i}_component_1.nii.gz -refweight ./results-at-d-full_seq/propagation/output_path_component2/final_results/mask_dyn000${i}_component_2.nii.gz -t ./results-at-d-full_seq/propagation/output_path_component0/final_results/direct_static_on_dyn000${i}_component_0.mat -t ./results-at-d-full_seq/propagation/output_path_component1/final_results/direct_static_on_dyn000${i}_component_1.mat -t ./results-at-d-full_seq/propagation/output_path_component2/final_results/direct_static_on_dyn000${i}_component_2.mat -o ./result-warping-S-to-dyn000${i}/ -warped_image warped_dyn_frame_dyn000${i}.nii.gz -def_field def_fld_dyn000${i}
#done
#
#for i in {10..31}
#do
#  echo "warping static image to dyn_00${i}"
#  python3 /mnt/c/Users/bzfmuham/OneDrive/Knee-Kinematics/Code_and_utils/dynMRI/transformfusion.py -in im_middle.nii -refweight ./results-at-d-full_seq/propagation/output_path_component0/final_results/mask_dyn00${i}_component_0.nii.gz -refweight ./results-at-d-full_seq/propagation/output_path_component1/final_results/mask_dyn00${i}_component_1.nii.gz -refweight ./results-at-d-full_seq/propagation/output_path_component2/final_results/mask_dyn00${i}_component_2.nii.gz -t ./results-at-d-full_seq/propagation/output_path_component0/final_results/direct_static_on_dyn00${i}_component_0.mat -t ./results-at-d-full_seq/propagation/output_path_component1/final_results/direct_static_on_dyn00${i}_component_1.mat -t ./results-at-d-full_seq/propagation/output_path_component2/final_results/direct_static_on_dyn00${i}_component_2.mat -o ./result-warping-S-to-dyn00${i}/ -warped_image warped_dyn_frame_dyn000${i}.nii.gz -def_field def_fld_dyn000${i}
#done
