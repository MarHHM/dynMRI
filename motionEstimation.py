## [MHH] This code corresponds to sec. 2.3.1 in the [Makki 2019] (rigid motion estimation of each bone between S & each frame D_k), generating rigid transform matrices per bone per frame D_k. (i.e. propagating bone segmentations from S to each frame in D)
## - I dind't have access to Makki's data, so I tried it on a dataset of ours (s4_2)
## - I call this script from the WSL bash (as the FSL library was installed there - it has no binaries for windows)

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
  © IMT Atlantique - LATIM-INSERM UMR 1101

  Author(s): Karim Makki (karim.makki@imt-atlantique.fr)

  This software is governed by the CeCILL-B license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL-B
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.

  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.

  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL-B license and that you accept its terms.

"""

import os
import os.path
import glob
import numpy as np
from numpy import matrix
from numpy.linalg import inv
import nibabel as nib
import nipype.algorithms.metrics as nipalg
import argparse
import multiprocessing
import subprocess
from timeit import default_timer as timer
from tqdm import tqdm           # for printing time-progress bar for loops




# Read from text file and store in matrix
#
# Parameters
# ----------
# filename : text filename (.mat)
#
# Returns matrix of floats
# -------
# output :
# 4*4 transformation matrix
def Text_file_to_matrix(filename):
   T = np.loadtxt(str(filename), dtype='f')
   return np.mat(T)

# save matrix as a text file
#
# Parameters
# ----------
# matrix : the matrix to be saved as text file
# text_filename : desired name of the text file (.mat)
def Matrix_to_text_file(matrix, text_filename):
    np.savetxt(text_filename, matrix, delimiter='  ')

# compute the dice score between two binary masks
#
# Parameters
# ----------
# rfile : reference image filename
# ifile : input image filename
#
# Returns
# -------
# output : scalar
# dice score between the two masks
def Bin_dice(rfile,ifile):

    tmp=nib.load(rfile) #segmentation
    d1=tmp.get_data()
    tmp=nib.load(ifile) #segmentation
    d2=tmp.get_data()
    d1_2=np.zeros(np.shape(tmp.get_data()))

    d1_2=len(np.where(d1*d2 != 0)[0])+0.0
    d1=len(np.where(d1 != 0)[0])+0.0
    d2=len(np.where(d2 != 0)[0])+0.0

    if d1==0:
        print ('ERROR: reference image is empty')
        dice=0
    elif d2==0:
        print ('ERROR: input image is empty')
        dice=0
    else:
        dice=2*d1_2/(d1+d2)

    return dice


def nifti_to_array(filename):

    nii = nib.load(filename)

    return (nii.get_data())


# compute the dice score between two blurry masks
#
# Parameters
# ----------
# rfile : reference image filename
# ifile : input image filename
#
# Returns
# -------
# output : scalar
# overlap between the two blurry masks
def Fuzzy_dice(rfile, ifile):

    overlap = nipalg.FuzzyOverlap()
    overlap.inputs.in_ref = rfile
    overlap.inputs.in_tst = ifile
    overlap.inputs.weighting = 'volume'
    res = overlap.run()

    return res.outputs.dice

def Binarize_fuzzy_mask(fuzzy_mask, binary_mask, threshold):

    nii = nib.load(fuzzy_mask)
    data = nii.get_data()
    output = np.zeros(data.shape)
    output[np.where(data>threshold)]= 1
    s = nib.Nifti1Image(output, nii.affine)
    nib.save(s,binary_mask)

    return 0


######################################################################################
######################################################################################
######################################################################################

if __name__ == '__main__':
    ## global FLIRT params
    fnl_intrpl = 'nearestneighbour'       # {trilinear, nearestneighbour (org in this script),sinc,spline}


    ##
    print(".\n.\n.")
    parser = argparse.ArgumentParser(prog='dynMRI')

    # before running this script, add these args through "Run -> Edit Configs"
    parser.add_argument('-s', '--static', help='High-res Static 3D input image', type=str, required = True)
    parser.add_argument('-d', '--dyn', help='Low-res Dynamic 4D input sequence', type=str, required = True)
    parser.add_argument('-m', '--mask', help='binary mask (segmentation) of a bone in the high-res image', type=str, required = True,action='append')
    parser.add_argument('-o', '--output', help='Output directory', type=str, required = True)
    parser.add_argument('-os', '--OperatingSystem', help='Operating system: 0 if Linux and 1 if Mac Os', type=int, default = 1) # using 1 as it seems that the FSL installed inside the wsl has the same function names as the Mac OS

    args = parser.parse_args()

    if (args.OperatingSystem == 1):
        call_flirt = 'flirt'
        call_fslsplit = 'fslsplit'
    else:
        call_flirt = 'fsl5.0-flirt'
        call_fslsplit = 'fsl5.0-fslsplit'

    ## (MHH) to run Linux & also specifically FSL commands inside windows (or actually WSL) -> invoke the WSL bash in the "run" mode
    # wsl_bash_run_lnx_cmds = '"' + r'C:\Program Files\WindowsApps\CanonicalGroupLimited.Ubuntu18.04onWindows_1804.2020.824.0_x64__79rhkp1fndgsc\ubuntu1804.exe' + '"' + ' ' + 'run' + ' '
    # wsl_bash_run_fsl_cmds = wsl_bash_run_lnx_cmds + '/usr/local/fsl/etc/fslconf/fsl.sh' + ' && ' + ' echo $FSLOUTPUTTYPE ' + ' && ' + '/usr/local/fsl/bin/'       # concatenate the path where the FSL bins were installed by fslinstaller.py
    # wsl_bash_run_fsl_cmds = wsl_bash_run_lnx_cmds + 'FSLOUTPUTTYPE=NIFTI_GZ' + ' && ' + '/usr/local/fsl/bin/'       # concatenate the path where the FSL bins were installed by fslinstaller.py --> ERROR: didn't run. So, I resorted to calling this script from the bash itself inside WSL

#######################Output path creation##########################

    outputpath=args.output
    if not os.path.exists(outputpath):
       os.makedirs(outputpath)

############################## Data #################################

    High_resolution_static = args.static # 3D image
    dynamic = args.dyn  # 4D volume

############# split the 4D file into lots of 3D files###############

    output_basename = 'dyn'
    # go = wsl_bash_run_fsl_cmds + call_fslsplit + ' ' + dynamic + ' ' + outputpath + output_basename
    # go = call_fslsplit + ' ' + dynamic + ' ' + outputpath + output_basename
    print("-> split the 4D file into lots of 3D files..")
    go = [call_fslsplit, dynamic, outputpath+output_basename]
    subprocess.run(go)
    print("-> (end) split the 4D file into lots of 3D files.\n")

############ Get the sorted set of 3D time frames ###################

    dynamicSet = glob.glob(outputpath+'/'+output_basename+'*.nii.gz')
    dynamicSet.sort()

################# Automated folders creation ########################

    outputpath_bone=outputpath+'propagation'
    if not os.path.exists(outputpath_bone):
        os.makedirs(outputpath_bone)


    for i in range(0, len(args.mask)):
        component_outputpath=outputpath_bone+'/output_path_component'+str(i)
        if not os.path.exists(component_outputpath):
            os.makedirs(component_outputpath)

    outputpath_boneSet=glob.glob(outputpath_bone+'/*')
    outputpath_boneSet.sort()

########################### Notations ################################
# t describes the time
# i describes the the component or the bone
######################################################################

####################### Define basenames #############################

    global_mask_basename = 'global_mask'
    global_image_basename = 'flirt_global_static'
    global_matrix_basename = 'global_static_on'
    local_matrix_basename = 'transform_dyn'
    mask_basename = 'mask_dyn'
    direct_transform_basename = 'direct_static_on'
    propagation_matrix_basename = 'matrix_flirt'
    propagation_image_basename = 'flirt_dyn'

######## Global rigid registration of the static on each time frame (sec 2.3.1->step1->First) #########

    movimage= High_resolution_static
    print("-> Global rigid registration of the static on each time frame (using FLIRT)..")
    # start = timer()
    for t in tqdm(range(0, len(dynamicSet))):
        # print(f"\tframe {t}..", end='\r')
        refimage = dynamicSet[t]
        prefix = dynamicSet[t].split('/')[-1].split('.')[0]
        global_outputimage = outputpath+'flirt_global_static_on_'+prefix+'.nii.gz'
        global_outputmat = outputpath+'global_static_on_'+prefix+'.mat'
        # go_init = wsl_bash_run_fsl_cmds + call_flirt+' -noresampblur -searchrx -40 40 -searchry -40 40 -searchrz -40 40 -cost mutualinfo  -dof 6 -ref '+refimage+' -in '+movimage+' -out '+global_outputimage+' -omat '+global_outputmat
        go_init = [call_flirt,  '-noresampblur',
                                '-searchrx', '-40', '40',
                                '-searchry', '-40', '40',
                                '-searchrz', '-40', '40',
                                '-cost', 'mutualinfo',
                                '-dof', '6',
                                '-ref', refimage,
                                '-in', movimage,
                                '-out', global_outputimage,
                                '-omat', global_outputmat]
        subprocess.run(go_init)
    print("-> (end) Global rigid registration of the static on each time frame (using FLIRT).\n")
    # end = timer()
    # print(f"time elapsed: {(end-start)/60} min.\n")

#########################################################################

    global_matrixSet=glob.glob(outputpath+global_matrix_basename+'*.mat')
    global_matrixSet.sort()

    global_imageSet=glob.glob(outputpath+global_image_basename+'*.nii.gz')
    global_imageSet.sort()

##### Propagate manual segmentations into the low_resolution domain #####
############# using the estimated global transformations ################
    print("-> Propagate manual segmentations into the low_resolution domain using the estimated global transformations (using FLIRT)..")
    # start = timer()
    for i in tqdm(range(0,len(outputpath_boneSet))):
        # print(f"\tbone {i}")
        for t in range(0, len(global_imageSet)):
            # print(f"\t\tframe {t}..", end='\r')
            prefix = dynamicSet[t].split('/')[-1].split('.')[0]
            global_mask= outputpath_boneSet[i]+'/global_mask_'+prefix+'_component_'+str(i)+'.nii.gz'
            # go_propagation = wsl_bash_run_fsl_cmds + call_flirt +' -applyxfm -noresampblur -ref '+global_imageSet[t]+' -in ' + args.mask[i] + ' -init '+ global_matrixSet[t] + ' -out ' + global_mask  + ' -interp nearestneighbour '
            go_propagation = [call_flirt,   '-applyxfm', '-noresampblur',
                                            '-ref', global_imageSet[t],
                                            '-in', args.mask[i],
                                            '-init', global_matrixSet[t],
                                            '-out', global_mask,
                                            '-interp', fnl_intrpl]
            subprocess.run(go_propagation)
            Binarize_fuzzy_mask(global_mask, global_mask, 0.4)
    print("-> (end) Propagate manual segmentations into the low_resolution domain using the estimated global transformations (using FLIRT).\n")
    # end = timer()
    # print(f"time elapsed: {(end-start)/60} min.\n")

##########################################################################

##############Local Rigid registration of bones###########################
    print("-> Local Rigid registration of bones (using FLIRT)..")
    # start = timer()
    for i in tqdm(range(0,len(outputpath_boneSet))):
        # print(f"\tbone {i}..")
        global_maskSet=glob.glob(outputpath_boneSet[i]+'/'+global_mask_basename+'*.nii.gz')
        global_maskSet.sort()

        for t in range(0, len(dynamicSet)):
            # print(f"\t\tframe {t}..", end='\r')
            prefix = dynamicSet[t].split('/')[-1].split('.')[0]
            refimage = dynamicSet[t]
            local_outputimage = outputpath_boneSet[i]+'/flirt_'+prefix+'_on_global_component_'+str(i)+'.nii.gz'
            local_outputmat = outputpath_boneSet[i]+'/transform_'+prefix+'_on_global_component_'+str(i)+'.mat'
            #go_init = 'flirt -searchrx -40 40 -searchry -40 40 -searchrz -40 40  -dof 6 -anglerep quaternion  -in '+refimage+' -ref '+global_imageSet[t]+' -out '+local_outputimage+' -omat '+local_outputmat +' -refweight '+ global_maskSet[t]
            # go_init = wsl_bash_run_fsl_cmds + call_flirt +' -searchrx -40 40 -searchry -40 40 -searchrz -40 40 -dof 6 -in '+global_imageSet[t]+' -ref '+refimage+' -out '+local_outputimage+' -omat '+local_outputmat +' -inweight '+ global_maskSet[t]
            go_init = [call_flirt,  '-searchrx', '-40', '40',
                                    '-searchry', '-40', '40',
                                    '-searchrz', '-40', '40',
                                    '-dof', '6',
                                    '-in', global_imageSet[t],
                                    '-ref', refimage,
                                    '-out', local_outputimage,
                                    '-omat', local_outputmat,
                                    '-inweight', global_maskSet[t]]

###################### Talus registration (Makki used (-cost normcorr) for this case specifically ) ################################
            # # (MHH) i commented this part
            # if(i==1):
            #     go_init = wsl_bash_run_fsl_cmds + call_flirt +' -searchrx -40 40 -searchry -40 40 -searchrz -40 40 -cost normcorr -dof 6 -in '+global_imageSet[t]+' -ref '+refimage+' -out '+local_outputimage+' -omat '+local_outputmat +' -inweight '+ global_maskSet[t]

            subprocess.run(go_init)
    print("-> (end) Local Rigid registration of bones (using FLIRT).\n")
    # end = timer()
    # print(f"time elapsed: {(end-start)/60} min.\n")

#### Compute composed transformations from static to each time frame (fast) #####
    print("-> Compute composed transformations from static to each time frame..")
    for i in tqdm(range(0,len(outputpath_boneSet))):
        local_matrixSet = glob.glob(outputpath_boneSet[i]+'/'+local_matrix_basename+'*.mat')
        local_matrixSet.sort()

        for t in range(0, len(dynamicSet)):
            prefix = dynamicSet[t].split('/')[-1].split('.')[0]
            #static_to_dyn_matrix = np.dot(np.linalg.inv(Text_file_to_matrix(local_matrixSet[t])), Text_file_to_matrix(global_matrixSet[t]))
            static_to_dyn_matrix = np.dot(Text_file_to_matrix(local_matrixSet[t]), Text_file_to_matrix(global_matrixSet[t]))

            save_path= outputpath_boneSet[i]+'/direct_static_on_'+prefix+'_component_'+str(i)+'.mat'
            np.savetxt(save_path, static_to_dyn_matrix, delimiter='  ')
    print("-> (end) Compute composed transformations from static to each time frame.\n")

###### Propagate the high_resolution segmentations into time frames #######
####### using the estimated composed transformations from static ##########
########################## to each time frame #############################
    print("-> Propagate the high_resolution segmentations into time frames"
          " using the estimated composed transformations from static to each time frame (using FLIRT)..")
    # start = timer()
    for i in tqdm(range(0,len(outputpath_boneSet))):
        # print(f"\tbone {i}..")
        init_matrixSet = glob.glob(outputpath_boneSet[i]+'/'+direct_transform_basename+'*.mat')
        init_matrixSet.sort()

        for t in range(0, len(dynamicSet)):
            # print(f"\t\tframe {t}..", end='\r')
            prefix = dynamicSet[t].split('/')[-1].split('.')[0]
            low_resolution_mask = outputpath_boneSet[i]+'/mask_'+prefix+'_component_'+str(i)+'.nii.gz'
            # go_init = wsl_bash_run_fsl_cmds + call_flirt + ' -applyxfm -noresampblur -ref '+ dynamicSet[t] + ' -in '+ args.mask[i] + ' -out '+ low_resolution_mask + ' -init '+init_matrixSet[t]+ ' -interp nearestneighbour '
            go_init = [call_flirt,  '-applyxfm', '-noresampblur',
                                    '-ref', dynamicSet[t],
                                    '-in', args.mask[i],
                                    '-out', low_resolution_mask,
                                    '-init', init_matrixSet[t],
                                    '-interp', fnl_intrpl]
            subprocess.run(go_init)
            Binarize_fuzzy_mask(low_resolution_mask, low_resolution_mask, 0.4)
    print("-> (end) Propagate the high_resolution segmentations into time frames"
          " using the estimated composed transformations from static to each time frame (using FLIRT).\n")
    # end = timer()
    # print(f"time elapsed: {(end-start)/60} min.\n")

######### Finding the time frame that best align with static image (D_{k^*} in Makki 2019) ########
    print("-> Finding the time frame that best align with static image (D_{k^*})..")
    # start = timer()
    dice_evaluation_array=np.zeros((len(outputpath_boneSet), len(dynamicSet)))

    #Hausdroff_evaluation_array=np.zeros((len(outputpath_boneSet), len(dynamicSet)))

    for i in tqdm(range(0,len(outputpath_boneSet))):
        # print(f"\tbone {i}..")
        global_maskSet=glob.glob(outputpath_boneSet[i]+'/'+global_mask_basename+'*.nii.gz')
        global_maskSet.sort()
        maskSet=glob.glob(outputpath_boneSet[i]+'/'+mask_basename+'*.nii.gz')
        maskSet.sort()
        direct_static_on_dynSet=glob.glob(outputpath_boneSet[i]+'/'+direct_transform_basename+'*.nii.gz')
        direct_static_on_dynSet.sort()

        for t in range(0, len(dynamicSet)):
            # print(f"\t\tframe {t}..", end='\r')
            dice_evaluation_array[i][t] = Bin_dice(maskSet[t],global_maskSet[t])
            #Hausdroff_evaluation_array[i][t] = directed_hausdorff(nifti_to_array(maskSet[t]),nifti_to_array(global_maskSet[t]))[0]

    linked_time_frame=np.prod(dice_evaluation_array, axis=0)
    #linked_time_frame=np.prod(Hausdroff_evaluation_array, axis=0)

    t__closest_to_S = np.argmax(linked_time_frame) ### the main idea here is to detect the time frame the most closest to the static scan ("no-motion" detection)
    #t__closest_to_S=np.argmin(linked_time_frame) ### the main idea here is to detect the time frame the most closest to the static scan ("no-motion" detection)
    print(f"**time frame that best align with static image is frame {t__closest_to_S}**")
    print("-> (end) Finding the time frame that best align with static image (D_{k^*}).\n")
    # end = timer()
    # print(f"time elapsed: {(end-start)/60} min.\n")

########## copy the best "static/dynamic" registration results to the final outputs folder ############
    print('-> copying the best "static/dynamic" registration results to the final outputs folder..')
    # start = timer()
    for i in tqdm(range(0,len(outputpath_boneSet))):
        # print(f"\tbone {i}..")
        maskSet=glob.glob(outputpath_boneSet[i]+'/'+mask_basename+'*.nii.gz')
        maskSet.sort()
        transformationSet=glob.glob(outputpath_boneSet[i]+'/'+direct_transform_basename+'*.mat')
        transformationSet.sort()

        output_results=outputpath_boneSet[i]+'/final_results'
        if not os.path.exists(output_results):
            os.makedirs(output_results)

        # copy= wsl_bash_run_lnx_cmds + 'cp '+maskSet[t__closest_to_S]+ '  '+ output_results
        copy = ['cp', maskSet[t__closest_to_S], output_results]
        subprocess.run(copy) ######copy mask(t) to the final_results folder

        # copy= wsl_bash_run_lnx_cmds + 'cp '+transformationSet[t__closest_to_S]+ '  '+ output_results
        copy= ['cp', transformationSet[t__closest_to_S], output_results]
        subprocess.run(copy) ######copy the most accurate estimated transformation in the "final_results" folder

        direct_static_on_dynSet=glob.glob(output_results+'/'+direct_transform_basename+'*.mat')
        direct_static_on_dynSet.sort()
        final_refweightSet=glob.glob(output_results+'/'+mask_basename+'*.nii.gz')
        final_refweightSet.sort()

######################## Backward propagation (starting from D_{k^*} to the begin of D) ##############################
        print("\t-> Backward propagation (using FLIRT)..")
        t = t__closest_to_S
        # while(t>0):           # I changed it to include 0. Why did he excluded 0 (i.e. first time frams in the low-res dynamic sequence)?
        while(t>=0):
            # print(f"\t\tframe {t}..", end='\r')
            final_refweightSet.sort()
            movimage = dynamicSet[t-1]
            refimage = dynamicSet[t]

            prefix1 = dynamicSet[t].split('/')[-1].split('.')[0]
            prefix2 = dynamicSet[t-1].split('/')[-1].split('.')[0]
            outputimage = output_results+'/flirt_'+prefix2+'_on_'+prefix1+'.nii.gz'
            outputmat   = output_results+'/matrix_flirt_'+prefix2+'_on_'+prefix1+'.mat'
            # go = wsl_bash_run_fsl_cmds + go_init +' -in '+movimage+' -out '+outputimage+ ' -omat '+outputmat + ' -refweight ' +  final_refweightSet[0]
            go = [call_flirt,   '-searchrx', '-40', '40',
                                '-searchry', '-40', '40',
                                '-searchrz', '-40', '40',
                                ## TODO: should i put it back? (with my data causes the problem of "Parameters do not form a valid axis - greater than unity")
                                # '-anglerep', 'quaternion',
                                '-dof', '6',
                                '-ref', refimage,
                                '-in', movimage,
                                '-out', outputimage,
                                '-omat', outputmat,
                                '-refweight', final_refweightSet[0]]
            # print("\t\t\tgo")
            subprocess.run(go)

            final_refweightSet=glob.glob(output_results+'/'+mask_basename+'*.nii.gz')
            final_refweightSet.sort()
            direct_transform = output_results+'/direct_static_on_'+prefix2+'_component_'+str(i)+'.mat'
            direct_static_on_dyn=np.dot(inv(Text_file_to_matrix(outputmat)), Text_file_to_matrix(direct_static_on_dynSet[0]))
            Matrix_to_text_file(direct_static_on_dyn, direct_transform)
            direct_static_on_dynSet=glob.glob(output_results+'/'+direct_transform_basename+'*.mat')
            direct_static_on_dynSet.sort()
            out_refweight= output_results+'/mask_'+prefix2+'_component_'+str(i)+'.nii.gz'
            # go_propagation = wsl_bash_run_fsl_cmds + call_flirt + ' -applyxfm -noresampblur -ref '+dynamicSet[t-1]+' -in ' + args.mask[i] + ' -init '+ direct_static_on_dynSet[0] + ' -out ' +out_refweight  + ' -interp nearestneighbour '
            go_propagation = [call_flirt,   '-applyxfm',
                                            '-noresampblur',
                                            '-ref', dynamicSet[t-1],
                                            '-in', args.mask[i],
                                            '-init', direct_static_on_dynSet[0],
                                            '-out', out_refweight,
                                            '-interp', fnl_intrpl]
            # print("\t\t\tgo_propagation")
            subprocess.run(go_propagation)

            t-=1
        print("\t-> (end) Backward propagation (using FLIRT).\n")

######################### Forward propagation (starting from D_{k^*} to the end of D)##############################
        print("\t-> Forward propagation (using FLIRT).." )
        direct_static_on_dynSet=glob.glob(output_results+'/'+direct_transform_basename+'*.mat')
        direct_static_on_dynSet.sort()
        final_refweightSet=glob.glob(output_results+'/'+mask_basename+'*.nii.gz')
        final_refweightSet.sort()
        direct_static_on_dynSet.sort()
        final_refweightSet.sort()

        t = t__closest_to_S
        while(t<len(dynamicSet)-1):
            # print(f"\t\tframe {t}..", end='\r')
            final_refweightSet.sort()
            movimage = dynamicSet[t+1]
            refimage = dynamicSet[t]
            # go_init = call_flirt + ' -searchrx -40 40 -searchry -40 40  -searchrz -40 40  -anglerep quaternion  -dof 6 -ref '+refimage
            prefix1 = dynamicSet[t].split('/')[-1].split('.')[0]
            prefix2 = dynamicSet[t+1].split('/')[-1].split('.')[0]
            outputimage = output_results+'/flirt_'+prefix2+'_on_'+prefix1+'.nii.gz'
            outputmat   = output_results+'/matrix_flirt_'+prefix2+'_on_'+prefix1+'.mat'
            # go = wsl_bash_run_fsl_cmds + go_init +' -in '+movimage+' -out '+outputimage+ ' -omat '+outputmat + ' -refweight ' +  final_refweightSet[t]
            go = [call_flirt,   '-searchrx', '-40', '40',
                                '-searchry', '-40', '40',
                                '-searchrz', '-40', '40',
                                ## TODO: should i put it back? (with my data causes the problem of "Parameters do not form a valid axis - greater than unity")
                                # '-anglerep', 'quaternion',
                                '-dof', '6',
                                '-ref', refimage,
                                '-in', movimage,
                                '-out', outputimage,
                                '-omat', outputmat,
                                '-refweight', final_refweightSet[t]]
            # print("\t\t\tgo")
            subprocess.run(go)

            direct_transform = output_results+'/direct_static_on_'+prefix2+'_component_'+str(i)+'.mat'
            direct_static_on_dyn=np.dot(inv(Text_file_to_matrix(outputmat)), Text_file_to_matrix(direct_static_on_dynSet[t]))
            Matrix_to_text_file(direct_static_on_dyn, direct_transform)
            direct_static_on_dynSet=glob.glob(output_results+'/'+direct_transform_basename+'*.mat')
            direct_static_on_dynSet.sort()
            out_refweight= output_results+'/mask_'+prefix2+'_component_'+str(i)+'.nii.gz'
            # go_propagation = wsl_bash_run_fsl_cmds + call_flirt + ' -applyxfm -noresampblur -ref '+dynamicSet[t+1]+' -in ' + args.mask[i] + ' -init '+ direct_static_on_dynSet[t+1] + ' -out ' +out_refweight  + ' -interp nearestneighbour '
            go_propagation = [call_flirt,   '-applyxfm', '-noresampblur',
                                            '-ref', dynamicSet[t+1],
                                            '-in', args.mask[i],
                                            '-init', direct_static_on_dynSet[t+1],
                                            '-out', out_refweight,
                                            '-interp', fnl_intrpl]
            # print("\t\t\tgo_propagation")
            subprocess.run(go_propagation)

            final_refweightSet=glob.glob(output_results+'/'+mask_basename+'*.nii.gz')
            final_refweightSet.sort()

            t+=1
        print("\t-> (end) Forward propagation (using FLIRT).\n")
    print('-> (end) copying the best "static/dynamic" registration results to the final outputs folder.\n')
    # end = timer()
    # print(f"time elapsed: {(end-start)/60} min.\n")

########## Clean "final_results" folders inorder to only keep the low_resolution automated segmentations and the direct transformations from the static scan to each low-resolution time frame ########################
####################################  move all intermediate results from the "final_results" folder to the "outputpath_boneSet[i]" folder #############################################################################
    print('-> Cleaning (move all intermediate results from the "final_results" folder to the "outputpath_boneSet[i]" folder)..')
    # start = timer()
    for i in tqdm(range(0,len(outputpath_boneSet))):
        # print(f"\tbone {i}..")
        output_results=outputpath_boneSet[i]+'/final_results'

        propagation_matrixSet=glob.glob(output_results+'/'+propagation_matrix_basename+'*.mat')
        propagation_matrixSet.sort()

        propagation_imageSet=glob.glob(output_results+'/'+propagation_image_basename+'*.nii.gz')
        propagation_imageSet.sort()

        for t in range(0, len(propagation_matrixSet)-1):
            # print(f"\t\tframe {t}..", end='\r')
            prefix1 = dynamicSet[t].split('/')[-1].split('.')[0]
            prefix2 = dynamicSet[t+1].split('/')[-1].split('.')[0]

            # go1= wsl_bash_run_lnx_cmds + 'mv '+ propagation_matrixSet[t] +' '+ outputpath_boneSet[i]+'/matrix_flirt_'+prefix1+'_on_'+prefix2+'.mat'
            # go2= wsl_bash_run_lnx_cmds + 'mv '+ propagation_imageSet[t] +' '+ outputpath_boneSet[i]+'/flirt_'+prefix1+'_on_'+prefix2+'.nii.gz'
            go1= ['mv', propagation_matrixSet[t], outputpath_boneSet[i]+'/matrix_flirt_'+prefix1+'_on_'+prefix2+'.mat']
            go2= ['mv', propagation_imageSet[t], outputpath_boneSet[i]+'/flirt_'+prefix1+'_on_'+prefix2+'.nii.gz']
            subprocess.run(go1)
            subprocess.run(go2)
    print('-> (end) Cleaning (move all intermediate results from the "final_results" folder to the "outputpath_boneSet[i]" folder).\n')
    # end = timer()
    # print(f"time elapsed: {(end-start)/60} min.\n")

