import os
import sys
import nibabel as nib
import numpy as np
import subprocess
from pathlib import Path
from typing import Union

import clabtoolkit.misctools as cltmisc
import clabtoolkit.bidstools as cltbids


def crop_image_from_mask(in_image: str, 
                            mask: Union[str, np.ndarray], 
                            out_image: str,
                            st_codes:Union[list, np.ndarray] = None):
    """
    Crops an image using a mask. This mask can be a binary mask or a mask with multiple structures. 
    The function will crop the image to the minimum bounding box that contains all the structures in the mask.
    The mask could be an image file path or a numpy array. If the mask is a numpy array, the function will use it directly.

    Parameters
    ----------
    in_image : str
        Image file path.
    mask : str or np.ndarray
        Mask file path or numpy array.
    st_codes : list or np.ndarray
        List of structures codes to be cropped.
    out_image : str
        Output image file path.
        
    Raises
    ------
    ValueError
        If the in_image is not a string.
        If the mask file does not exist if the mask variable is a string.
        If the mask parameter is not a string or a numpy array.
        

    Returns
    -------
    None

    Examples
    --------
    >>> _crop_image_from_mask(in_image='/path/to/image.nii.gz', mask='/path/to/mask.nii.gz', st_codes = ['3:6', 22, 9-10], out_image='/path/to/out_image.nii.gz')
    
    """
    
    if isinstance(in_image, str) == False:
        raise ValueError("The 'image' parameter must be a string.")
    
    if isinstance(mask, str) :
        if not os.path.exists(mask):
            raise ValueError("The 'mask' parameter must be a string.")
        else:
            mask = nib.load(mask)
            mask_data = mask.get_fdata()
    elif isinstance(mask, np.ndarray):
        mask_data = mask
    else:
        raise ValueError("The 'mask' parameter must be a string or a numpy array.")
    
    if st_codes is None:
        st_codes = np.unique(mask_data)
        st_codes = st_codes[st_codes != 0]
    
    st_codes = cltmisc.build_indexes(st_codes)
    st_codes = np.array(st_codes)
    
    
    # Create the output directory if it does not exist
    out_pth = os.path.dirname(out_image)
    if os.path.exists(out_pth) == False:
        Path(out_pth).mkdir(parents=True, exist_ok=True)

    # Loading both images
    img1 = nib.load(in_image)  # Original MRI image

    # Get data and affine matrices
    img1_affine = img1.affine
    
    # Get the destination shape
    img1_data = img1.get_fdata()
    img1_shape = img1_data.shape
    
    # Finding the minimum and maximum indexes for the mask
    tmask = np.isin(mask_data, st_codes)
    tmp_var = np.argwhere(tmask)
    
    # Minimum and maximum indexes for X axis
    i_start = np.min(tmp_var[:,0])
    i_end = np.max(tmp_var[:,0])
    
    # Minimum and maximum indexes for Y axis
    j_start = np.min(tmp_var[:,1])
    j_end = np.max(tmp_var[:,1])
    
    # Minimum and maximum indexes for Z axis
    k_start = np.min(tmp_var[:,2])
    k_end = np.max(tmp_var[:,2])
    
    # If img1_data is a 4D array we need to multiply it by the mask in the last dimension only. If not, we multiply it by the mask
    # Applying the mask 
    if len(img1_data.shape) == 4:
        masked_data = img1_data * tmask[..., np.newaxis]
    else:
        masked_data = img1_data * tmask
    
    # Creating a new Nifti image with the same affine and header as img1
    array_img = nib.Nifti1Image(masked_data, img1_affine)

    # Cropping the masked data
    if len(img1_data.shape) == 4:
        cropped_img = array_img.slicer[i_start:i_end, j_start:j_end, k_start:k_end, :]
    else:
        cropped_img = array_img.slicer[i_start:i_end, j_start:j_end, k_start:k_end]
        
    # Saving the cropped image
    nib.save(cropped_img, out_image)
    
    return out_image    

def cropped_to_native(in_image: str, native_image: str, out_image: str):
    """
    Restores a cropped image to the dimensions of a reference image.

    Parameters
    ----------
    in_image : str
        Cropped image file path.
    native_image : str
        Reference image file path.
    out_image : str
        Output image file path.
        
    Raises
    ------
    ValueError
        If the 'index' or 'name' attributes are missing when writing a TSV file.

    Returns
    -------
    None

    Examples
    --------
    >>> _cropped_to_native(in_image='/path/to/cropped_image.nii.gz', native_image='/path/to/native_image.nii.gz', out_image='/path/to/out_image.nii.gz')
    
    """
    
    if isinstance(in_image, str) == False:
        raise ValueError("The 'in_image' parameter must be a string.")
    
    if isinstance(native_image, str) == False:
        raise ValueError("The 'native_image' parameter must be a string.")
    
    # Create the output directory if it does not exist
    out_pth = os.path.dirname(out_image)
    if os.path.exists(out_pth) == False:
        Path(out_pth).mkdir(parents=True, exist_ok=True)

    
    # Loading both images
    img1 = nib.load(native_image)  # Original MRI image
    img2 = nib.load(in_image)  # Cropped image

    # Get data and affine matrices
    img1_affine = img1.affine
    img2_affine = img2.affine
    
    # Get the destination shape
    img1_data = img1.get_fdata()
    img1_shape = img1_data.shape
    
    # Get data from IM2
    img2_data = img2.get_fdata()
    img2_shape = img2_data.shape
    
    
    # Multiply the inverse of the affine matrix of img1 by the affine matrix of img2
    affine_mult = np.linalg.inv(img1_affine) @ img2_affine
    
    # If the img2 is a 4D add the forth dimension to the shape of the img1
    if len(img2_shape) == 4:
        img1_shape = (img1_shape[0], img1_shape[1], img1_shape[2], img2_shape[3])

        # Create an empty array with the same dimensions as IM1
        new_data = np.zeros(img1_shape, dtype=img2_data.dtype)
        
        for vol in range(img2_data.shape[-1]):
            # Find the coordinates in voxels of the voxels different from 0 on the img2
            indices = np.argwhere(img2_data[..., vol] != 0)

            # Apply the affine transformation to the coordinates of the voxels different from 0 on img2
            new_coords = np.round(affine_mult @ np.concatenate((indices.T, np.ones((1, indices.shape[0]))), axis=0)).astype(int)

            # Fill the new image with the values of the voxels different from 0 on img2
            new_data[new_coords[0], new_coords[1], new_coords[2], vol] = img2_data[indices[:, 0], indices[:, 1], indices[:, 2], vol]
            
    elif len(img2_shape) == 3:
        # Create an empty array with the same dimensions as IM1
        new_data = np.zeros(img1_shape, dtype=img2_data.dtype)
        
        # Find the coordinates in voxels of the voxels different from 0 on the img2
        indices = np.argwhere(img2_data != 0)

        # Apply the affine transformation to the coordinates of the voxels different from 0 on img2
        new_coords = np.round(affine_mult @ np.concatenate((indices.T, np.ones((1, indices.shape[0]))), axis=0)).astype(int)

        # Fill the new image with the values of the voxels different from 0 on img2
        new_data[new_coords[0], new_coords[1], new_coords[2]] = img2_data[indices[:, 0], indices[:, 1], indices[:, 2]]

    # Create a new Nifti image with the same affine and header as IM1
    new_img2 = nib.Nifti1Image(new_data, affine=img1_affine, header=img1.header)

    # Save the new image
    nib.save(new_img2, out_image)
    
    return out_image

def apply_multi_transf(in_image:str, 
                        out_image:str, 
                        ref_image:str, 
                        xfm_output, 
                        interp_order:int=0, 
                        invert:bool=False,
                        cont_tech:str="local", 
                        cont_image:str = None,
                        force:bool=False ):
    """
    This function applies an ANTs transformation to an image.

    Parameters
    ----------
    in_image : str
        Input image file path.
    out_image : str
        Output image file path.
    ref_image : str
        Reference image file path.
    xfm_output : str
        Spatial transformation file path.
    interp_order : int
        Interpolation order. Default is 0 (NearestNeighbor). Options are: 0 (NearestNeighbor), 1 (Linear), 2 (BSpline[3]), 3 (CosineWindowedSinc), 4 (WelchWindowedSinc), 5 (HammingWindowedSinc), 6 (LanczosWindowedSinc), 7 (Welch).
    invert : bool
        Invert the transformation. Default is False.
    cont_tech : str
        Containerization technology. Default is 'local'. Options are: 'local', 'singularity', 'docker'.
    cont_image : str
        Container image. Default is None.
    force : bool
        Force the computation. Default is False.
        
    Raises
    ------
    ValueError
        If the 'interp_order' is not an integer.
        If the 'interp_order' is not between 0 and 7.
        If the 'invert' is not a boolean.
        If the 'cont_tech' is not a string.
        If the 'cont_image' is not a string.
        If the 'force' is not a boolean.
        If the 'in_image' is not a string.
        If the 'in_image' does not exist.
        If the 'out_image' is not a string.
        If the 'ref_image' is not a string.
        If the 'ref_image' does not exist.
        If the 'xfm_output' is not a string.
        
    Examples
    --------
    >>> apply_multi_transf(in_image='/path/to/in_image.nii.gz', out_image='/path/to/out_image.nii.gz', ref_image='/path/to/ref_image.nii.gz', xfm_output='/path/to/xfm_output.nii.gz', interp_order=0, invert=False, cont_tech='local', cont_image=None, force=False)
    
    """

    # Check if the path of out_basename exists
    out_path = os.path.dirname(out_image)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    if interp_order == 0:
        interp_cad = "NearestNeighbor"
    elif interp_order == 1:
        interp_cad = "Linear"
    elif interp_order == 2:
        interp_cad = "BSpline[3]"
    elif interp_order == 3:
        interp_cad = "CosineWindowedSinc"
    elif interp_order == 4:
        interp_cad = "WelchWindowedSinc"
    elif interp_order == 5:
        interp_cad = "HammingWindowedSinc"
    elif interp_order == 6:
        interp_cad = "LanczosWindowedSinc"
    elif interp_order == 7:
        interp_cad = "Welch"

    ######## -- Registration to the template space  ------------ #
    # Creating spatial transformation folder
    stransf_dir  = Path(os.path.dirname(xfm_output))
    stransf_name = os.path.basename(xfm_output)
    
    
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
    
    affine_transf = os.path.join(stransf_dir, affine_name + '_xfm.mat')
    nl_transf = os.path.join(stransf_dir, nl_name + '_xfm.nii.gz')
    invnl_transf = os.path.join(stransf_dir, invnl_name + '_xfm.nii.gz')

    # Check if out_image is not computed and force is True
    if not os.path.isfile(out_image) or force:

        if not os.path.isfile(affine_transf):
            print("The spatial transformation file does not exist.")
            sys.exit()
        
        if os.path.isfile(invnl_transf) and os.path.isfile(nl_transf):
            if invert:
                bashargs_transforms = ["-t", invnl_transf, "-t", "[" + affine_transf + ",1]"]
            else:
                bashargs_transforms = ["-t", nl_transf, "-t", affine_transf]
        else:
            if invert:
                bashargs_transforms = ["-t", "[" + affine_transf + ",1]"]
            else:
                bashargs_transforms = ["-t", affine_transf]
        
        
        # Creating the command
        cmd_bashargs = ["antsApplyTransforms", "-e", "3", "-i" , in_image, "-r", ref_image, "-o", out_image, "-n", interp_cad]
        cmd_bashargs.extend(bashargs_transforms)

        # Running containerization
        cmd_cont = cltmisc.generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        out_cmd = subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True)

def get_vox_neighbors(coord: np.ndarray, 
                    neighborhood: str = '26', 
                    dims: str = '3', 
                    order: int = 1):
    """
    Get the neighborhood of a voxel.
    
    Parameters:
    -----------
    
    coord : np.ndarray
        Coordinates of the voxel.
    
    neighborhood : str
        Neighborhood type (e.g. 6, 18, 26).
        
    dims : str
        Number of dimensions (e.g. 2, 3).
        
    Returns:
    --------
    
    neighbors : list
        List of neighbors.
    
    Raises:
    -------
    
    ValueError
        If the number of dimensions is not supported.
        
    Examples:
    ---------
        
        >>> neigh = get_vox_neighbors(neighborhood = '6', dims = '3')
    
    """
    
    # Check if the number of dimensions in coord supported by the supplied coordinates
    if len(coord) != int(dims):
        raise ValueError("The number of dimensions in the coordinates is not supported.")
    
    # Check if the number of dimensions is supported
    if dims == '3':
        
        # Check if it is a valid neighborhood
        if neighborhood not in ['6', '18', '26']:
            raise ValueError("The neighborhood type is not supported.")

        # Constructing the neighborhood
        if neighborhood == '6':
            neighbors = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])

        elif neighborhood == '12':
            neighbors = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                                    [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 0, 1], [-1, 0, -1]])
            
        elif neighborhood == '18':
            neighbors = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                                    [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 0, 1], [-1, 0, -1],
                                    [1, 0, -1], [-1, 0, 1], [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]])
            
        elif neighborhood == '26':
            neighbors = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                                    [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 0, 1], [-1, 0, -1],
                                    [1, 0, -1], [-1, 0, 1], [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
                                    [1, 1, 1], [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1], [-1, -1, 1],
                                    [1, -1, 1], [-1, 1, 1]])
    elif dims == '2':
        
        if neighborhood not in ['4', '8']:
            raise ValueError("The neighborhood type is not supported.")

        if neighborhood == '4':
            neighbors = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        elif neighborhood == '8':
            neighbors = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                                    [1, 1], [-1, -1], [1, -1], [-1, 1]])
        
    else:
        raise ValueError("The number of dimensions is not supported.")
    
    neighbors = np.array([coord + n for n in neighbors])
            
    # # Loop around the order of the neighborhood
    # if order > 1:
    #     if dims == '3':
    #         neighbors = np.vstack([np.array([x, y, z]) for x, y, z in neighbors])
    #         neighbors = np.unique(neighbors, axis=0)
    #         neighbors = np.vstack([get_vox_neighbors(neighborhood, dims, order - 1) for x, y, z in neighbors])    
    #     elif dims == '2':
    #         neighbors = np.vstack([np.array([x, y]) for x, y in neighbors])
    #         neighbors = np.unique(neighbors, axis=0)
    #         neighbors = np.vstack([get_vox_neighbors(neighborhood, dims, order - 1) for x, y in neighbors])
        
    return neighbors