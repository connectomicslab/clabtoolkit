import os
import sys
import subprocess
from pathlib import Path
import numpy as np
from scipy.spatial import distance
from scipy.ndimage import convolve
from glob import glob
import nibabel as nib
from typing import Union
import clabtoolkit.misctools as cltmisc
import clabtoolkit.bidstools as cltbids
import clabtoolkit.imagetools as cltimg
import clabtoolkit.parcellationtools as cltparc


def abased_parcellation(t1: str,
                            t1_temp: str,
                            atlas: str, 
                            out_parc: str, 
                            xfm_output: str, 
                            atlas_type: str ='spam',
                            interp: str = 'Linear',
                            cont_tech: str = 'local',
                            cont_image: str = None,
                            force: bool = False):
    """
    Compute atlas-based parcellation using ANTs. 
    
    Parameters:
    ----------
    t1 : str
        T1 image file
        
    atlas: str
        Atlas image.
        
    atlas_type : str
        Atlas type (e.g. spam or maxprob)
        
    out_parc : str
        Output parcellation file
        
    xfm_output : str
        Output name for the affine spatial transformation. It can include the path.
        
    interp : str
        Interpolation method (e.g. NearestNeighbor, Linear, BSpline)
        
    cont_tech : str
        Container technology (e.g. singularity, docker or local).
    
    cont_image : str
        Container image.
    
    force : bool
        Overwrite the results.
        
    Returns:
    --------
        
    """
    
    ######## -- Registration to the template space  ------------ #
    # Creating spatial transformation folder
    stransf_dir  = Path(os.path.dirname(xfm_output))
    stransf_name = os.path.basename(xfm_output)
    out_parc_dir = Path(os.path.dirname(out_parc))
    
    # If the directory does not exist create the directory and if it fails because it does not have write access send an error
    try:
        stransf_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print("The directory to store the spatial transformations does not have write access.")
        sys.exit()
        
    try:
        out_parc_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print("The directory to store the parcellation does not have write access.")
        sys.exit()
    
    # Spatial transformation files (Temporal).
    tmp_xfm_basename  = os.path.join(stransf_dir, 'temp')    
    temp_xfm_affine   = tmp_xfm_basename + '_0GenericAffine.mat'
    temp_xfm_nl       = tmp_xfm_basename + '_1Warp.nii.gz'
    temp_xfm_invnl    = tmp_xfm_basename + '_1InverseWarp.nii.gz'
    temp_xfm_invnlw   = tmp_xfm_basename + '_InverseWarped.nii.gz'
    temp_xfm_nlw      = tmp_xfm_basename + '_Warped.nii.gz'
    
    if stransf_name.endswith('.nii.gz'):
        stransf_name = stransf_name[:-7]
    elif stransf_name.endswith('.nii') or stransf_name.endswith('.mat'):
        stransf_name = stransf_name[:-4]
    
    if stransf_name.endswith('_xfm'):
        stransf_name = stransf_name[:-4]
    
    if "_desc-" in stransf_name:
        affine_name = cltbids.replace_entity_value(stransf_name, {'desc':'affine'})
        nl_name = cltbids.replace_entity_value(stransf_name, {'desc':'warp'})
        invnl_name = cltbids.replace_entity_value(stransf_name, {'desc':'iwarp'})
    else:
        affine_name = stransf_name + '_desc-affine'
        nl_name = stransf_name + '_desc-warp'
        invnl_name = stransf_name + '_desc-iwarp'
    
    
    # Affine transformation filename
    xfm_affine = os.path.join(stransf_dir, affine_name + '_xfm.mat')
    
    # Non-linear transformation filename
    xfm_nl= os.path.join(stransf_dir, nl_name + '_xfm.nii.gz')
    
    # Filename for the inverse of the Non-linear transformation 
    xfm_invnl= os.path.join(stransf_dir, invnl_name + '_xfm.nii.gz')
    
    if not os.path.isfile(xfm_invnl) or force:
        # Registration to MNI template
        
        cmd_bashargs = ['antsRegistrationSyN.sh', '-d', '3', '-f', t1_temp, '-m', t1, '-t', 's',
                        '-o', tmp_xfm_basename + '_']
        
        cmd_cont = cltmisc.generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
        
        # Changing the names
        cmd_bashargs = ['mv', temp_xfm_affine, xfm_affine]
        cmd_cont = cltmisc.generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command

        cmd_bashargs = ['mv', temp_xfm_nl, xfm_nl]
        cmd_cont = cltmisc.generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
        
        cmd_bashargs = ['mv', temp_xfm_invnl, xfm_invnl]
        cmd_cont = cltmisc.generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
        
        # Deleting the warped images
        cmd_bashargs = ['rm', tmp_xfm_basename + '*Warped.nii.gz']
        warped_imgs = glob(tmp_xfm_basename + '*Warped.nii.gz')
        if len(warped_imgs) > 0:
            for w_img in warped_imgs:
                os.remove(w_img)
        
    # Applying spatial transform to the atlas
    if not os.path.isfile(out_parc):
        
        if atlas_type == 'spam':
            # Applying spatial transform
            cmd_bashargs = ['antsApplyTransforms', '-d', '3', '-e', '3', '-i', atlas,
                            '-o', out_parc, '-r', t1, '-t', xfm_invnl,
                            '-t','[' + xfm_affine + ',1]', '-n', interp]

            cmd_cont = cltmisc.generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
            subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
            
        elif atlas_type == 'maxprob':
            # Applying spatial transform
            cmd_bashargs = ['antsApplyTransforms', '-d', '3', '-i', atlas,
                            '-o', out_parc, '-r', t1, '-t', xfm_invnl,
                            '-t','[' + xfm_affine + ',1]', '-n', 'NearestNeighbor']

            cmd_cont = cltmisc.generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
            subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
            
            tmp_parc = cltparc.Parcellation(parc_file=out_parc)
            tmp_parc.save_parcellation(out_file= out_parc, affine=tmp_parc.affine, save_lut=False, save_tsv=False)
            
        # Removing the Warped images
        if os.path.isfile(temp_xfm_invnlw):
            os.remove(temp_xfm_invnlw)
            
        if os.path.isfile(temp_xfm_nlw):
            os.remove(temp_xfm_nlw)
        
    return out_parc

def spams2maxprob(spam_image:str,
                    prob_thresh:float=0.05,
                    vol_indexes:np.array=None,
                    maxp_name:str=None):
    """
    This method converts a SPAMs image into a maximum probability image.
    
    Parameters:
    -----------
    
    spam_image : str
        SPAMs image filename.
        
    prob_thresh : float
        Threshold value.
        
    vol_indexes : np.array
        Volume indexes.
        
    maxp_name : str 
        Output maximum probability image filename. 
        If None, the image will not be saved and the function will return the image as a nunmpy array.
        
    Returns:
    --------
    
    maxp_name : str 
        Output maximum probability image filename. 
        If None, the image will not be saved and the function will return the image as a nunmpy array.

    """
    
    spam_img           = nib.load(spam_image)
    affine             = spam_img.affine
    spam_vol           = spam_img.get_fdata()

    spam_vol[spam_vol < prob_thresh] = 0
    spam_vol[spam_vol > 1]      = 1
    
    if vol_indexes is not None:   
        # Creating the maxprob
        
        # I want to find the complementary indexes to vol_indexes
        all_indexes = np.arange(0, spam_vol.shape[3])
        set1 = set(all_indexes)
        set2 = set(vol_indexes)

        # Find the symmetric difference
        diff_elements = set1.symmetric_difference(set2)

        # Convert the result back to a NumPy array if needed
        diff_array = np.array(list(diff_elements))
        spam_vol[:,:,:,diff_array] = 0
        # array_data = np.delete(spam_vol, diff_array, 3)
        
    
    ind = np.where(np.sum(spam_vol, axis=3) == 0)
    maxprob_thl      = spam_vol.argmax(axis=3) + 1
    maxprob_thl[ind] = 0
    
    if maxp_name is not None:
        # Save the image
        imgcoll          = nib.Nifti1Image(maxprob_thl.astype('int16'), affine)
        nib.save(imgcoll, maxp_name)
    else:
        maxp_name = maxprob_thl
    
    return maxp_name

def region_growing(iparc: np.ndarray,
                    mask: Union[np.ndarray, np.bool_], 
                    neighborhood='26'):
    """
    Region growing algorithm for parcellation correction. It labels the unlabeled voxels of 
    the image mask with the most frequent label among the neighbors of each voxel.
    
    Parameters:
    -----------
    iparc : np.ndarray
        Parcellation image.
        
    mask : np.ndarray
        Mask image.
        
        neighborhood : str
        Neighborhood type (e.g. 6, 18, 26).
        
    Returns:
    --------
    
    
    """

    # Create a binary array where labeled voxels are marked as 1
    binary_labels = (iparc > 0).astype(int)

    # Convolve with the kernel to count labeled neighbors for each voxel
    kernel = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    labeled_neighbor_count = convolve(binary_labels, kernel, mode='constant', cval=0)

    # Mask for voxels that have at least one labeled neighbor
    mask_with_labeled_neighbors = (labeled_neighbor_count > 0) & (iparc == 0)
    ind = np.argwhere((mask_with_labeled_neighbors != 0)& (binary_labels == 0)& (mask))
    ind_orig = ind.copy() * 0
    
    # Loop until no more voxels could be labeled or all the voxels are labeled
    while (len(ind) > 0)& (np.array_equal(ind, ind_orig) == False):
        ind_orig = ind.copy()
        # Process each unlabeled voxel
        for coord in ind:
            x, y, z = coord
            
            # Detecting the neighbors
            neighbors = cltimg.get_vox_neighbors(coord=coord, neighborhood='26', dims= '3')
            # Remove from motion the coordinates out of the bounding box
            neighbors = neighbors[(neighbors[:, 0] >= 0) & (neighbors[:, 0] < iparc.shape[0]) &
                            (neighbors[:, 1] >= 0) & (neighbors[:, 1] < iparc.shape[1]) &
                            (neighbors[:, 2] >= 0) & (neighbors[:, 2] < iparc.shape[2])]
                        
            # Labels of the neighbors
            neigh_lab = iparc[neighbors[:,0],neighbors[:,1],neighbors[:,2]]
            
            if len(np.argwhere(neigh_lab > 0)) > 2:
                
                # Remove the neighbors that are not labeled
                neighbors = neighbors[neigh_lab > 0]
                neigh_lab = neigh_lab[neigh_lab > 0]
                
                unique_labels, counts = np.unique(neigh_lab, return_counts=True)
                max_count = counts.max()
                max_labels = unique_labels[counts == max_count]
                
                if len(max_labels) == 1:
                    iparc[x, y, z] = max_labels[0]
                    
                else:
                    # In case of tie, choose the label of the closest neighbor
                    distances = [distance.euclidean(coord, (dx, dy, dz))
                                for (dx, dy, dz), lbl in zip(neighbors, neigh_lab) if lbl in max_labels]
                    closest_label = max_labels[np.argmin(distances)]
                    iparc[x, y, z] = closest_label
                # most_frequent_label = np.bincount(neigh_lab[neigh_lab != 0]).argmax()

        
        # Create a binary array where labeled voxels are marked as 1
        binary_labels = (iparc > 0).astype(int)

        # Convolve with the kernel to count labeled neighbors for each voxel
        labeled_neighbor_count = convolve(binary_labels, kernel, mode='constant', cval=0)

        # Mask for voxels that have at least one labeled neighbor
        mask_with_labeled_neighbors = (labeled_neighbor_count > 0) & (iparc == 0)
        ind = np.argwhere((mask_with_labeled_neighbors != 0)& (binary_labels == 0)& (mask))

    return iparc * mask