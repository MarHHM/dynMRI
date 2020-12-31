#!/bin/bash

#TODO: NEXT TIME RUN IN WINDOWS (not WSL) (as transformFusion.py doesn't depend on FLIRT)!!

n_frms=53
dataSet__path=/mnt/s/datasets/s4_2
input__path=im_middle.nii
#input__path=masks/im_middle.labels_allBones.nii     # inputing combined mask instead
motionEstimation_output__folderName=results--motEst--for_input_padded_dynSeq
#motionEstimation_output__folderName=results-motionEst--at-d-full_seq__modified_rotational_srch
transformFusion_output__folderName=results--transFusion--for_input_padded_dynSeq
#transformFusion_output__folderName=results-transFusion--at-d-full_seq__modified_rotational_srch--input-bonesMask

motionEstimation_output__path=${dataSet__path}/${motionEstimation_output__folderName}/propagation

for ((i=1; i<=9; i++))
do
  echo "warping static image to dyn_000${i}"
  python3 transformFusion.py -in ${dataSet__path}/${input__path} -refweight ${motionEstimation_output__path}/output_path_component0/final_results/mask_dyn000${i}_component_0.nii.gz -refweight ${motionEstimation_output__path}/output_path_component1/final_results/mask_dyn000${i}_component_1.nii.gz -refweight ${motionEstimation_output__path}/output_path_component2/final_results/mask_dyn000${i}_component_2.nii.gz -t ${motionEstimation_output__path}/output_path_component0/final_results/direct_static_on_dyn000${i}_component_0.mat -t ${motionEstimation_output__path}/output_path_component1/final_results/direct_static_on_dyn000${i}_component_1.mat -t ${motionEstimation_output__path}/output_path_component2/final_results/direct_static_on_dyn000${i}_component_2.mat -o ${dataSet__path}/${transformFusion_output__folderName}/result-warping-S-to-dyn000${i}/ -warped_image warped_dyn_frame_dyn000${i}.nii.gz -def_field def_fld_dyn000${i}

  ## only tibia mask as input
#  python3 transformFusion.py -in ${dataSet__path}/${input__path} -refweight ${motionEstimation_output__path}/output_path_component1/final_results/mask_dyn000${i}_component_1.nii.gz -t ${motionEstimation_output__path}/output_path_component1/final_results/direct_static_on_dyn000${i}_component_1.mat -o ${dataSet__path}/${transformFusion_output__folderName}/result-warping-S-to-dyn000${i}/ -warped_image warped_dyn_frame_dyn000${i}.nii.gz -def_field def_fld_dyn000${i}

  echo "-----------------------------------------------------------------------------"
done

for ((i=10; i<=$n_frms; i++))
do
  echo "warping static image to dyn_00${i}"
  python3 transformFusion.py -in ${dataSet__path}/${input__path} -refweight ${motionEstimation_output__path}/output_path_component0/final_results/mask_dyn00${i}_component_0.nii.gz -refweight ${motionEstimation_output__path}/output_path_component1/final_results/mask_dyn00${i}_component_1.nii.gz -refweight ${motionEstimation_output__path}/output_path_component2/final_results/mask_dyn00${i}_component_2.nii.gz -t ${motionEstimation_output__path}/output_path_component0/final_results/direct_static_on_dyn00${i}_component_0.mat -t ${motionEstimation_output__path}/output_path_component1/final_results/direct_static_on_dyn00${i}_component_1.mat -t ${motionEstimation_output__path}/output_path_component2/final_results/direct_static_on_dyn00${i}_component_2.mat -o ${dataSet__path}/${transformFusion_output__folderName}/result-warping-S-to-dyn00${i}/ -warped_image warped_dyn_frame_dyn00${i}.nii.gz -def_field def_fld_dyn00${i}

  ## only tibia mask as input
#  python3 transformFusion.py -in ${dataSet__path}/${input__path} -refweight ${motionEstimation_output__path}/output_path_component1/final_results/mask_dyn00${i}_component_1.nii.gz -t ${motionEstimation_output__path}/output_path_component1/final_results/direct_static_on_dyn00${i}_component_1.mat -o ${dataSet__path}/${transformFusion_output__folderName}/result-warping-S-to-dyn00${i}/ -warped_image warped_dyn_frame_dyn00${i}.nii.gz -def_field def_fld_dyn00${i}

  echo "-----------------------------------------------------------------------------"
done
