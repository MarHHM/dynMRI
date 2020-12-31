#!/bin/bash

dataset_path=/mnt/s/datasets/s4_2
dynSeq__fileName=padded_dynSeq 	# {dynSeq, padded_dynSeq, dynSeq_shorter_8_frms} 
output__folderName=results--motEst--for_input_padded_dynSeq

python3 motionEstimation.py -s ${dataset_path}/im_middle.nii -d ${dataset_path}/${dynSeq__fileName}.nii -m ${dataset_path}/masks/im_middle.labels_femur.nii -m ${dataset_path}/masks/im_middle.labels_tibia.nii -m ${dataset_path}/masks/im_middle.labels_patella.nii -o ${dataset_path}/${output__folderName}/

