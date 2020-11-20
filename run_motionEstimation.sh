#!/bin/bash

cd /mnt/s/datasets/s4_2/

python3 /mnt/c/Users/bzfmuham/OneDrive/Knee-Kinematics/Code_and_utils/dynMRI/motionEstimation.py -s im_middle.nii -d im_rest.nii -m masks/im_middle.labels_femur.nii -m masks/im_middle.labels_tibia.nii -m masks/im_middle.labels_patella.nii -o results-at-d-full_seq/