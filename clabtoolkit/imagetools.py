import os
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Union

import clabtoolkit.misctools as cltmisc


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
    
    st_codes = cltmisc._build_indexes(st_codes)
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
    
    # If the img2 is a 4D add the forth dimension to the shape of the img1
    if len(img2.shape) == 4:
        img1_shape = (img1_shape[0], img1_shape[1], img1_shape[2], img2.shape[3])

    # Get data from IM2
    img2_data = img2.get_fdata()

    # Create an empty array with the same dimensions as IM1
    new_data = np.zeros(img1_shape)
    
    for vol in range(img2_data.shape[-1]):
        # Find the coordinates in voxels of the voxels different from 0 on the img2
        indices = np.argwhere(img2_data[..., vol] != 0)

        # Multiply the inverse of the affine matrix of img1 by the affine matrix of img2
        affine_mult = np.linalg.inv(img1_affine) @ img2_affine

        # Apply the affine transformation to the coordinates of the voxels different from 0 on img2
        new_coords = np.round(affine_mult @ np.concatenate((indices.T, np.ones((1, indices.shape[0]))), axis=0)).astype(int)

        # Fill the new image with the values of the voxels different from 0 on img2
        new_data[new_coords[0], new_coords[1], new_coords[2], vol] = img2_data[indices[:, 0], indices[:, 1], indices[:, 2], vol]

    # Create a new Nifti image with the same affine and header as IM1
    new_img2 = nib.Nifti1Image(new_data, affine=img1_affine, header=img1.header)

    # Save the new image
    nib.save(new_img2, out_image)
    
    return out_image