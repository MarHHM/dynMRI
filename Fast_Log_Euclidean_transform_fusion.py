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


from scipy import ndimage
import nibabel as nib
import numpy as np
import argparse
import os
import scipy.linalg as la
import glob
from numpy.linalg import det
from numpy import newaxis
import itertools
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
#from scipy.ndimage.morphology import binary_erosion
import multiprocessing

from numpy import *



def Text_file_to_matrix(filename):

   T = np.loadtxt(str(filename), dtype='f')

   return np.mat(T)


def nifti_to_array(filename):

    nii = nib.load(filename)

    return (nii.get_data())


def nifti_image_shape(filename):

    nii = nib.load(filename)
    #nii.get_data_dtype() == np.dtype(np.float32)
    data = nii.get_data()


    return (data.shape)


def get_header_from_nifti_file(filename):

    nii = nib.load(filename)

    return nii.header

'''
Define the sigmoid function,  which smooths the  slope of the  weight map near the wire.
Parameters
----------
x : N dimensional array
Returns
-------
output : array
  N_dimensional array containing sigmoid function result

'''

def sigmoid(x):
  return 1 / (1 + np.exp(np.negative(x)))

"""
compute the  associated weighting function to a binary mask (a region in the reference image)
Parameters
----------
component : array of data (binary mask)
Returns
-------
output : array
  N_dimensional array containing the weighting function value for each voxel according to the entered mask, convolved with a Gaussian kernel
  with a standard deviation set to 2 voxels inorder to take into account the partial volume effect due to anisotropy of the image resolution
"""

def component_weighting_function(data):

    np.subtract(np.max(data), data, data)
    return 1/(1+0.5*ndimage.distance_transform_edt(data)**2)
    #return gaussian_filter(1/(1+ndimage.distance_transform_edt(data)), sigma=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-in', '--floating', help='floating input image', type=str, required = True)
    parser.add_argument('-refweight', '--component', help='', type=str, required = True,action='append')
    parser.add_argument('-t', '--transform', help='', type=str, required = True,action='append')
    parser.add_argument('-o', '--output', help='Output directory', type=str, required = True)
    parser.add_argument('-warped_image', '--outputimage', help='Output image name', type=str, required = True)
    parser.add_argument('-def_field', '--deformation_field', help='Deformation field image name', type=str, required = True)

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    normalized_weighting_function_path = args.output+'/normalized_weighting_function/'
    if not os.path.exists(normalized_weighting_function_path):
        os.makedirs(normalized_weighting_function_path)


######################compute the normalized weighting function of each component #########################
    nii = nib.load(args.component[0])
    data_shape = nifti_image_shape(args.component[0])
    dim0, dim1, dim2 = nifti_image_shape(args.floating)

    sum_of_weighting_functions = np.zeros((data_shape))
    Normalized_weighting_function = np.zeros((data_shape))

    for i in range(0, len(args.component)):

        sum_of_weighting_functions += component_weighting_function(nifti_to_array(args.component[i]))

    for i in range(0, len(args.component)):

        np.divide(component_weighting_function(nifti_to_array(args.component[i])), sum_of_weighting_functions, Normalized_weighting_function)

        k = nib.Nifti1Image(Normalized_weighting_function, nii.affine)
        save_path = normalized_weighting_function_path+'Normalized_weighting_function_component'+str(i)+'.nii.gz'
        nib.save(k, save_path)

###############################################################################################################
    del sum_of_weighting_functions
    del Normalized_weighting_function

###### set of computed normalized weighting functions #######

    Normalized_weighting_functionSet = glob.glob(normalized_weighting_function_path+'*.nii.gz')
    Normalized_weighting_functionSet.sort()

##########################Test#####################################

##### create an array of matrices: final_log_transform(x,y,z)= -∑i  w_norm(i)[x,y,z]*log(T(i)) ########
    final_transform = np.zeros((dim0, dim1, dim2, 4, 4))

    #fg= np.argwhere(HR_data > 0) #Foreground voxels
    #bg= np.argwhere(HR_data == 0) #Background voxels

    for i in range(0, len(args.transform)):

        np.subtract(final_transform, np.multiply(la.logm(Text_file_to_matrix(args.transform[i])).real , nifti_to_array(Normalized_weighting_functionSet[i])[:,:,:,newaxis,newaxis]), final_transform)

    print("log part successsfully computed")


##### compute the exponential of each matrix in the final_log_transform array of matrices using Eigen decomposition   ############
##############################  final_exp_transform(x,y,z)= exp(-∑i  w_norm(i)[x,y,z]*log(T(i))) #################################

    d, Y = np.linalg.eig(final_transform)  #returns an array of vectors with the eigenvalues (d[dim0,dim1,dim2,4]) and an array of matrices (Y[dim0,dim1,dim2,(4,4)]) with
    print("eigenvalues and eigen vectors were successsfully computed")

    Yinv = np.linalg.inv(Y)

    print("Yinv is successfully computed")

    d = np.exp(d)  # exp(final_transform) = Y*exp(d)*inv(Y).  exp (d) is much more easy to calculate than exp(final_transform)
    #since, for a diagonal matrix, we just need to exponentiate the diagonal elements.

    print("exponentiate the diagonal elements (eigenvalues): done")

    final_transform[:,:,:,3,3]= 1
    #first row
    final_transform[:,:,:,0,0]= Y[:,:,:,0,0]*Yinv[:,:,:,0,0]*d[:,:,:,0] + Y[:,:,:,0,1]*Yinv[:,:,:,1,0]*d[:,:,:,1] + Y[:,:,:,0,2]*Yinv[:,:,:,2,0]*d[:,:,:,2] + Y[:,:,:,0,3]*Yinv[:,:,:,3,0]*d[:,:,:,3]
    final_transform[:,:,:,0,1]= Y[:,:,:,0,0]*Yinv[:,:,:,0,1]*d[:,:,:,0] + Y[:,:,:,0,1]*Yinv[:,:,:,1,1]*d[:,:,:,1] + Y[:,:,:,0,2]*Yinv[:,:,:,2,1]*d[:,:,:,2] + Y[:,:,:,0,3]*Yinv[:,:,:,3,1]*d[:,:,:,3]
    final_transform[:,:,:,0,2]= Y[:,:,:,0,0]*Yinv[:,:,:,0,2]*d[:,:,:,0] + Y[:,:,:,0,1]*Yinv[:,:,:,1,2]*d[:,:,:,1] + Y[:,:,:,0,2]*Yinv[:,:,:,2,2]*d[:,:,:,2] + Y[:,:,:,0,3]*Yinv[:,:,:,3,2]*d[:,:,:,3]
    final_transform[:,:,:,0,3]= Y[:,:,:,0,0]*Yinv[:,:,:,0,3]*d[:,:,:,0] + Y[:,:,:,0,1]*Yinv[:,:,:,1,3]*d[:,:,:,1] + Y[:,:,:,0,2]*Yinv[:,:,:,2,3]*d[:,:,:,2] + Y[:,:,:,0,3]*Yinv[:,:,:,3,3]*d[:,:,:,3]

    final_transform[:,:,:,1,0]= Y[:,:,:,1,0]*Yinv[:,:,:,0,0]*d[:,:,:,0] + Y[:,:,:,1,1]*Yinv[:,:,:,1,0]*d[:,:,:,1] + Y[:,:,:,1,2]*Yinv[:,:,:,2,0]*d[:,:,:,2] + Y[:,:,:,1,3]*Yinv[:,:,:,3,0]*d[:,:,:,3]
    final_transform[:,:,:,1,1]= Y[:,:,:,1,0]*Yinv[:,:,:,0,1]*d[:,:,:,0] + Y[:,:,:,1,1]*Yinv[:,:,:,1,1]*d[:,:,:,1] + Y[:,:,:,1,2]*Yinv[:,:,:,2,1]*d[:,:,:,2] + Y[:,:,:,1,3]*Yinv[:,:,:,3,1]*d[:,:,:,3]
    final_transform[:,:,:,1,2]= Y[:,:,:,1,0]*Yinv[:,:,:,0,2]*d[:,:,:,0] + Y[:,:,:,1,1]*Yinv[:,:,:,1,2]*d[:,:,:,1] + Y[:,:,:,1,2]*Yinv[:,:,:,2,2]*d[:,:,:,2] + Y[:,:,:,1,3]*Yinv[:,:,:,3,2]*d[:,:,:,3]
    final_transform[:,:,:,1,3]= Y[:,:,:,1,0]*Yinv[:,:,:,0,3]*d[:,:,:,0] + Y[:,:,:,1,1]*Yinv[:,:,:,1,3]*d[:,:,:,1] + Y[:,:,:,1,2]*Yinv[:,:,:,2,3]*d[:,:,:,2] + Y[:,:,:,1,3]*Yinv[:,:,:,3,3]*d[:,:,:,3]

    final_transform[:,:,:,2,0]= Y[:,:,:,2,0]*Yinv[:,:,:,0,0]*d[:,:,:,0] + Y[:,:,:,2,1]*Yinv[:,:,:,1,0]*d[:,:,:,1] + Y[:,:,:,2,2]*Yinv[:,:,:,2,0]*d[:,:,:,2] + Y[:,:,:,2,3]*Yinv[:,:,:,3,0]*d[:,:,:,3]
    final_transform[:,:,:,2,1]= Y[:,:,:,2,0]*Yinv[:,:,:,0,1]*d[:,:,:,0] + Y[:,:,:,2,1]*Yinv[:,:,:,1,1]*d[:,:,:,1] + Y[:,:,:,2,2]*Yinv[:,:,:,2,1]*d[:,:,:,2] + Y[:,:,:,2,3]*Yinv[:,:,:,3,1]*d[:,:,:,3]
    final_transform[:,:,:,2,2]= Y[:,:,:,2,0]*Yinv[:,:,:,0,2]*d[:,:,:,0] + Y[:,:,:,2,1]*Yinv[:,:,:,1,2]*d[:,:,:,1] + Y[:,:,:,2,2]*Yinv[:,:,:,2,2]*d[:,:,:,2] + Y[:,:,:,2,3]*Yinv[:,:,:,3,2]*d[:,:,:,3]
    final_transform[:,:,:,2,3]= Y[:,:,:,2,0]*Yinv[:,:,:,0,3]*d[:,:,:,0] + Y[:,:,:,2,1]*Yinv[:,:,:,1,3]*d[:,:,:,1] + Y[:,:,:,2,2]*Yinv[:,:,:,2,3]*d[:,:,:,2] + Y[:,:,:,2,3]*Yinv[:,:,:,3,3]*d[:,:,:,3]

    ### free memory
    del Y
    del Yinv
    del d

    print("final transform exponential computed")


######## compute the warped image #################################

    input_header = get_header_from_nifti_file(args.floating)
    reference_header = get_header_from_nifti_file(args.floating)

## Compute coordinates in the input image

    coords = np.zeros((3,dim0, dim1, dim2), dtype='float32')
    coords[0,...] = np.arange(dim0)[:,np.newaxis,np.newaxis]
    coords[1,...] = np.arange(dim1)[np.newaxis,:,np.newaxis]
    coords[2,...] = np.arange(dim2)[np.newaxis,np.newaxis,:]

# flip [x,y,z] if necessary (based on the sign of the sform or qform determinant)

    if np.sign(det(input_header.get_qform())) == 1:
        coords[0,...] = input_header.get_data_shape()[0]-1-coords[0,...]
        coords[1,...] = input_header.get_data_shape()[1]-1-coords[1,...]
        coords[2,...] = input_header.get_data_shape()[2]-1-coords[2,...]

#scale the values by multiplying by the corresponding voxel sizes (in mm)

    np.multiply(input_header.get_zooms()[0], coords[0,...], coords[0,...])
    np.multiply(input_header.get_zooms()[1], coords[1,...], coords[1,...])
    np.multiply(input_header.get_zooms()[2], coords[2,...], coords[2,...])

# Apply the FLIRT matrix for each voxel to map to the reference space

    coords_ref = np.zeros((3,dim0, dim1, dim2),dtype='float32')
    coords_ref[0,...] = final_transform[:,:,:,0,0]*coords[0,...] + final_transform[:,:,:,0,1]*coords[1,...] + final_transform[:,:,:,0,2]* coords[2,...] +  final_transform[:,:,:,0,3]
    coords_ref[1,...] = final_transform[:,:,:,1,0]*coords[0,...] + final_transform[:,:,:,1,1]*coords[1,...] + final_transform[:,:,:,1,2]* coords[2,...] +  final_transform[:,:,:,1,3]
    coords_ref[2,...] = final_transform[:,:,:,2,0]*coords[0,...] + final_transform[:,:,:,2,1]*coords[1,...] + final_transform[:,:,:,2,2]* coords[2,...] +  final_transform[:,:,:,2,3]

# removing final transforms from the computer RAM after the exponentiation step

    del final_transform

#divide by the corresponding voxel sizes (in mm, of the reference image this time)

    np.divide(coords_ref[0,...], reference_header.get_zooms()[0], coords[0,...])
    np.divide(coords_ref[1,...], reference_header.get_zooms()[1], coords[1,...])
    np.divide(coords_ref[2,...], reference_header.get_zooms()[2], coords[2,...])

    del coords_ref

#flip the [x,y,z] coordinates (based on the sign of the sform or qform determinant, of the reference image this time)

    if np.sign(det(reference_header.get_qform())) == 1:
        coords[0,...] = reference_header.get_data_shape()[0]-1-coords[0,...]
        coords[1,...] = reference_header.get_data_shape()[1]-1-coords[1,...]
        coords[2,...] = reference_header.get_data_shape()[2]-1-coords[2,...]

    print("warped image is successfully computed")

#create index for the reference space

    i = np.arange(0,dim0)
    j = np.arange(0,dim1)
    k = np.arange(0,dim2)
    iv,jv,kv = np.meshgrid(i,j,k,indexing='ij')

    iv = np.reshape(iv,(-1))
    jv = np.reshape(jv,(-1))
    kv = np.reshape(kv,(-1))

#reshape the warped coordinates

    pointset = np.zeros((3,iv.shape[0]))
    pointset[0,:] = iv
    pointset[1,:] = jv
    pointset[2,:] = kv

    coords = np.reshape(coords, pointset.shape)
    val = np.zeros(iv.shape)

#### Interpolation:  mapping output data into the reference image space####

    map_coordinates(nifti_to_array(args.floating),[coords[0,:],coords[1,:],coords[2,:]],output=val,order=1, mode='nearest')

    del coords

    output_data = np.reshape(val,nifti_image_shape(args.floating))


#######writing and saving warped image ###

    i = nib.Nifti1Image(output_data, nii.affine)
    save_path = args.output + args.outputimage
    nib.save(i, save_path)