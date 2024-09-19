
import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import nibabel as nib
import clabtoolkit.misctools as cltmisc



def _abased_parcellation(t1: str,
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
    elif stransf_name.endswith('.nii'):
        stransf_name = stransf_name[:-4]
    
    # Affine transformation filename
    xfm_affine = os.path.join(stransf_dir, stransf_name + '_desc-affine_xfm.mat')
    
    # Non-linear transformation filename
    xfm_nl= os.path.join(stransf_dir, stransf_name + '_desc-warp_xfm.mat')
    
    # Filename for the inverse of the Non-linear transformation 
    xfm_invnl= os.path.join(stransf_dir, stransf_name + '_desc-iwarp_xfm.mat')
    
    if not os.path.isfile(xfm_invnl) or force:
        # Registration to MNI template
        
        cmd_bashargs = ['antsRegistrationSyNQuick.sh', '-d', '3', '-f', t1_temp, '-m', t1, '-t', 's',
                        '-o', tmp_xfm_basename + '_']
        
        cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
        
        # Changing the names
        cmd_bashargs = ['mv', temp_xfm_affine, xfm_affine]
        cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command

        cmd_bashargs = ['mv', temp_xfm_nl, xfm_nl]
        cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
        
        cmd_bashargs = ['mv', temp_xfm_invnl, xfm_invnl]
        cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
        

    if not os.path.isfile(out_parc):
        
        if atlas_type == 'spam':
            # Applying spatial transform
            cmd_bashargs = ['antsApplyTransforms', '-d', '3', '-e', '3', '-i', atlas,
                            '-o', out_parc, '-r', t1, '-t', xfm_invnl,
                            '-t','[' + xfm_affine + ',1]', '-n', interp]
    
        elif atlas_type == 'maxprob':
            # Applying spatial transform
            cmd_bashargs = ['antsApplyTransforms', '-d', '3', '-e', '3', '-i', atlas,
                            '-o', out_parc, '-r', t1, '-t', xfm_invnl,
                            '-t','[' + xfm_affine + ',1]', '-n', 'NearestNeighbor']

        cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
        
        # Removing the Warped images
        if os.path.isfile(temp_xfm_invnlw):
            os.remove(temp_xfm_invnlw)
            
        if os.path.isfile(temp_xfm_nlw):
            os.remove(temp_xfm_nlw)
        
    return out_parc

def _spams2maxprob(spam_image:str,
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