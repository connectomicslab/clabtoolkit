import os
import sys
import nibabel as nib
import numpy as np
import subprocess
from pathlib import Path
from typing import Union

# Importing local modules
from . import misctools as cltmisc
from . import bidstools as cltbids

def get_voxel_size(affine: np.ndarray):
    """
    This method will compute the voxel size from an affine matrix.
    
    Parameters
    ----------
    affine: 4x4 affine transformation matrix from NIfTI header
    
    Returns
    -------
    tuple: (voxel_x, voxel_y, voxel_z) sizes in mm


    """
    # Extract voxel sizes as the magnitude of each column vector
    voxel_x = np.linalg.norm(affine[:3, 0])
    voxel_y = np.linalg.norm(affine[:3, 1])
    voxel_z = np.linalg.norm(affine[:3, 2])
    return (voxel_x, voxel_y, voxel_z)

def get_voxel_volume(affine: np.ndarray):
    """
    This method will compute the voxel volume from an affine matrix.
    
    Parameters
    ----------
    affine: 4x4 affine transformation matrix from NIfTI header
    
    Returns
    -------
    float: voxel volume in mm^3

    """
    voxel_x, voxel_y, voxel_z = get_voxel_size(affine)
    return voxel_x * voxel_y * voxel_z

def get_center(affine: np.ndarray):
    """
    Extract the center/origin from the affine matrix.

    Parameters
    ----------
    affine: 4x4 affine transformation matrix from NIfTI header

    Returns
    -------
    tuple: (center_x, center_y, center_z) translation from 4th column

    """
    return (affine[0, 3], affine[1, 3], affine[2, 3])

def get_rotation_matrix(affine: np.ndarray):
    """
    Extract the rotation matrix from the affine matrix.

    Parameters
    ----------
    affine: 4x4 affine transformation matrix from NIfTI header

    Returns
    -------
    np.ndarray: 3x3 rotation matrix (normalized, without scaling)
    
    """
    # Extract 3x3 rotation/scaling matrix
    rot_scale = affine[:3, :3]
    # Normalize each column to remove scaling and keep only rotation
    rotation = np.zeros_like(rot_scale)
    for i in range(3):
        rotation[:, i] = rot_scale[:, i] / np.linalg.norm(rot_scale[:, i])
    return rotation

def crop_image_from_mask(
    in_image: str,
    mask: Union[str, np.ndarray],
    out_image: str,
    st_codes: Union[list, np.ndarray] = None,
):
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

    if isinstance(mask, str):
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

    st_codes = cltmisc.build_indices(st_codes)
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
    i_start = np.min(tmp_var[:, 0])
    i_end = np.max(tmp_var[:, 0])

    # Minimum and maximum indexes for Y axis
    j_start = np.min(tmp_var[:, 1])
    j_end = np.max(tmp_var[:, 1])

    # Minimum and maximum indexes for Z axis
    k_start = np.min(tmp_var[:, 2])
    k_end = np.max(tmp_var[:, 2])

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
            new_coords = np.round(
                affine_mult
                @ np.concatenate((indices.T, np.ones((1, indices.shape[0]))), axis=0)
            ).astype(int)

            # Fill the new image with the values of the voxels different from 0 on img2
            new_data[new_coords[0], new_coords[1], new_coords[2], vol] = img2_data[
                indices[:, 0], indices[:, 1], indices[:, 2], vol
            ]

    elif len(img2_shape) == 3:
        # Create an empty array with the same dimensions as IM1
        new_data = np.zeros(img1_shape, dtype=img2_data.dtype)

        # Find the coordinates in voxels of the voxels different from 0 on the img2
        indices = np.argwhere(img2_data != 0)

        # Apply the affine transformation to the coordinates of the voxels different from 0 on img2
        new_coords = np.round(
            affine_mult
            @ np.concatenate((indices.T, np.ones((1, indices.shape[0]))), axis=0)
        ).astype(int)

        # Fill the new image with the values of the voxels different from 0 on img2
        new_data[new_coords[0], new_coords[1], new_coords[2]] = img2_data[
            indices[:, 0], indices[:, 1], indices[:, 2]
        ]

    # Create a new Nifti image with the same affine and header as IM1
    new_img2 = nib.Nifti1Image(new_data, affine=img1_affine, header=img1.header)

    # Save the new image
    nib.save(new_img2, out_image)

    return out_image


def apply_multi_transf(
    in_image: str,
    out_image: str,
    ref_image: str,
    xfm_output,
    interp_order: int = 0,
    invert: bool = False,
    cont_tech: str = "local",
    cont_image: str = None,
    force: bool = False,
):
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
    stransf_dir = Path(os.path.dirname(xfm_output))
    stransf_name = os.path.basename(xfm_output)

    if stransf_name.endswith(".nii.gz"):
        stransf_name = stransf_name[:-7]
    elif stransf_name.endswith(".nii") or stransf_name.endswith(".mat"):
        stransf_name = stransf_name[:-4]

    if stransf_name.endswith("_xfm"):
        stransf_name = stransf_name[:-4]

    if "_desc-" in stransf_name:
        affine_name = cltbids.replace_entity_value(stransf_name, {"desc": "affine"})
        nl_name = cltbids.replace_entity_value(stransf_name, {"desc": "warp"})
        invnl_name = cltbids.replace_entity_value(stransf_name, {"desc": "iwarp"})
    else:
        affine_name = stransf_name + "_desc-affine"
        nl_name = stransf_name + "_desc-warp"
        invnl_name = stransf_name + "_desc-iwarp"

    affine_transf = os.path.join(stransf_dir, affine_name + "_xfm.mat")
    nl_transf = os.path.join(stransf_dir, nl_name + "_xfm.nii.gz")
    invnl_transf = os.path.join(stransf_dir, invnl_name + "_xfm.nii.gz")

    # Check if out_image is not computed and force is True
    if not os.path.isfile(out_image) or force:

        if not os.path.isfile(affine_transf):
            print("The spatial transformation file does not exist.")
            sys.exit()

        if os.path.isfile(invnl_transf) and os.path.isfile(nl_transf):
            if invert:
                bashargs_transforms = [
                    "-t",
                    invnl_transf,
                    "-t",
                    "[" + affine_transf + ",1]",
                ]
            else:
                bashargs_transforms = ["-t", nl_transf, "-t", affine_transf]
        else:
            if invert:
                bashargs_transforms = ["-t", "[" + affine_transf + ",1]"]
            else:
                bashargs_transforms = ["-t", affine_transf]

        # Creating the command
        cmd_bashargs = [
            "antsApplyTransforms",
            "-e",
            "3",
            "-i",
            in_image,
            "-r",
            ref_image,
            "-o",
            out_image,
            "-n",
            interp_cad,
        ]
        cmd_bashargs.extend(bashargs_transforms)

        # Running containerization
        cmd_cont = cltmisc.generate_container_command(
            cmd_bashargs, cont_tech, cont_image
        )  # Generating container command
        out_cmd = subprocess.run(
            cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
        )


def get_vox_neighbors(
    coord: np.ndarray, neighborhood: str = "26", dims: str = "3", order: int = 1
):
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
        raise ValueError(
            "The number of dimensions in the coordinates is not supported."
        )

    # Check if the number of dimensions is supported
    if dims == "3":

        # Check if it is a valid neighborhood
        if neighborhood not in ["6", "18", "26"]:
            raise ValueError("The neighborhood type is not supported.")

        # Constructing the neighborhood
        if neighborhood == "6":
            neighbors = np.array(
                [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
            )

        elif neighborhood == "12":
            neighbors = np.array(
                [
                    [1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                    [0, 0, -1],
                    [1, 1, 0],
                    [-1, -1, 0],
                    [1, -1, 0],
                    [-1, 1, 0],
                    [1, 0, 1],
                    [-1, 0, -1],
                ]
            )

        elif neighborhood == "18":
            neighbors = np.array(
                [
                    [1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                    [0, 0, -1],
                    [1, 1, 0],
                    [-1, -1, 0],
                    [1, -1, 0],
                    [-1, 1, 0],
                    [1, 0, 1],
                    [-1, 0, -1],
                    [1, 0, -1],
                    [-1, 0, 1],
                    [0, 1, 1],
                    [0, -1, -1],
                    [0, 1, -1],
                    [0, -1, 1],
                ]
            )

        elif neighborhood == "26":
            neighbors = np.array(
                [
                    [1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                    [0, 0, -1],
                    [1, 1, 0],
                    [-1, -1, 0],
                    [1, -1, 0],
                    [-1, 1, 0],
                    [1, 0, 1],
                    [-1, 0, -1],
                    [1, 0, -1],
                    [-1, 0, 1],
                    [0, 1, 1],
                    [0, -1, -1],
                    [0, 1, -1],
                    [0, -1, 1],
                    [1, 1, 1],
                    [-1, -1, -1],
                    [1, -1, -1],
                    [-1, 1, -1],
                    [1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [-1, 1, 1],
                ]
            )
    elif dims == "2":

        if neighborhood not in ["4", "8"]:
            raise ValueError("The neighborhood type is not supported.")

        if neighborhood == "4":
            neighbors = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        elif neighborhood == "8":
            neighbors = np.array(
                [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]
            )

    else:
        raise ValueError("The number of dimensions is not supported.")

    neighbors = np.array([coord + n for n in neighbors])

    return neighbors


# Moving coordinates from voxel to mm
def vox2mm(vox_coords, affine):
    """
    Convert voxel coordinates to mm coordinates. The input matrix must have 3 columns.

    Parameters
    ----------
    vox_coords : numpy array
        Matrix with the voxel coordinates. The matrix must have 3 columns.
    affine : numpy array
        Affine matrix of the image.

    Returns
    -------
    mm_coords : numpy array
        Matrix with the mm coordinates. The matrix has the same number of rows as the input matrix, and 3 columns.

    Raises:
    -------

    ValueError : The number of columns of the input matrix must be 3

    Examples
    --------
    >>> vox2mm(np.array([[1,2,3]]), np.eye(4))
    array([[1, 2, 3]])

    """

    # Detect if the number of rows is bigger than the number of columns. If not, transpose the matrix
    nrows = np.shape(vox_coords)[0]
    ncols = np.shape(vox_coords)[1]
    if (nrows < ncols) and (ncols > 3):
        vox_coords = np.transpose(vox_coords)

    if np.shape(vox_coords)[1] == 3:
        npoints = np.shape(vox_coords)
        vox_coords = np.c_[ vox_coords, np.full(npoints[0], 1)]

        
        mm_coords = np.matmul(affine, vox_coords.T)
        mm_coords = np.transpose(mm_coords)
        mm_coords = mm_coords[:, :3]

    else:
        # Launch an error if the number of columns is different from 3
        raise ValueError("The number of columns of the input matrix must be 3")

    return mm_coords


def mm2vox(mm_coords, affine):
    """
    Convert mm coordinates to voxel coordinates. The input matrix must have 3 columns.

    Parameters
    ----------
    mm_coords : numpy array
        Matrix with the mm coordinates. The matrix must have 3 columns.
    affine : numpy array
        Affine matrix of the image.

    Returns
    -------
    vox_coords : numpy array
        Matrix with the voxel coordinates. The matrix has the same number of rows as the input matrix, and 3 columns.

    Raises:
    -------

    ValueError : The number of columns of the input matrix must be 3

    Examples
    --------
    >>> mm2vox(np.array([[1,2,3]]), np.eye(4))
    array([[1, 2, 3]])

    """

    # Detect if the number of rows is bigger than the number of columns. If not, transpose the matrix
    nrows = np.shape(mm_coords)[0]
    ncols = np.shape(mm_coords)[1]
    if (nrows < ncols) and (ncols > 3):
        mm_coords = np.transpose(mm_coords)

    if np.shape(mm_coords)[1] == 3:
        npoints = np.shape(mm_coords)
        mm_coords = np.c_[ mm_coords, np.full(npoints[0], 1)]

        vox_coords = np.matmul(affine, mm_coords.T)
        vox_coords = np.transpose(vox_coords)
        vox_coords = vox_coords[:, :3]

    else:
        # Launch an error if the number of columns is different from 3
        raise ValueError("The number of columns of the input matrix must be 3")

    return vox_coords
class MorphologicalOperations:
    """
    A class to perform morphological operations on binary arrays.
    Works with both 2D and 3D arrays using scipy.ndimage.
    """
    
    def __init__(self):
        """Initialize the morphological operations class."""
        pass
    
    def create_structuring_element(self, shape='cube', size=3, dimensions=None):
        """
        Create a structuring element for morphological operations.
        
        Parameters:
        - shape: str, 'cube'/'square' (box), 'ball'/'disk' (sphere/circle), 'cross'
        - size: int, size of the structuring element
        - dimensions: int, number of dimensions (auto-detected from input if None)
        
        Returns:
        - numpy array representing the structuring element
        """
        if dimensions is None:
            dimensions = 3  # default to 3D
        
        if shape in ['cube', 'square']:
            # Create cubic/square structuring element
            return np.ones((size,) * dimensions, dtype=bool)
        
        elif shape in ['ball', 'disk']:
            # Create spherical/circular structuring element
            radius = size // 2
            if dimensions == 2:
                y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
                return x**2 + y**2 <= radius**2
            elif dimensions == 3:
                z, y, x = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
                return x**2 + y**2 + z**2 <= radius**2
            else:
                raise ValueError("Ball/disk only supported for 2D and 3D")
        
        elif shape == 'cross':
            # Create cross-shaped structuring element
            if dimensions == 2:
                cross = np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]], dtype=bool)
                return cross
            elif dimensions == 3:
                cross = np.zeros((3, 3, 3), dtype=bool)
                cross[1, 1, :] = True  # x-axis
                cross[1, :, 1] = True  # y-axis
                cross[:, 1, 1] = True  # z-axis
                return cross
            else:
                raise ValueError("Cross only supported for 2D and 3D")
        
        else:
            raise ValueError("Shape must be 'cube', 'ball', 'cross', 'square', or 'disk'")
    
    def erode(self, binary_array, structure=None, iterations=1):
        """
        Perform binary erosion - shrinks objects, removes small noise.
        
        Parameters:
        - binary_array: binary numpy array (2D or 3D)
        - structure: structuring element (None for default 3x3 or 3x3x3 cube)
        - iterations: number of erosion iterations
        """
        binary_array = self._ensure_binary(binary_array)
        
        if structure is None:
            structure = self.create_structuring_element('cube', 3, binary_array.ndim)
        
        return binary_erosion(binary_array, structure=structure, iterations=iterations)
    
    def dilate(self, binary_array, structure=None, iterations=1):
        """
        Perform binary dilation - expands objects, fills small gaps.
        
        Parameters:
        - binary_array: binary numpy array (2D or 3D)
        - structure: structuring element (None for default 3x3 or 3x3x3 cube)
        - iterations: number of dilation iterations
        """
        binary_array = self._ensure_binary(binary_array)
        
        if structure is None:
            structure = self.create_structuring_element('cube', 3, binary_array.ndim)
        
        return binary_dilation(binary_array, structure=structure, iterations=iterations)
    
    def opening(self, binary_array, structure=None, iterations=1):
        """
        Perform morphological opening (erosion followed by dilation).
        Removes small objects and noise while preserving larger structures.
        
        Parameters:
        - binary_array: binary numpy array (2D or 3D)
        - structure: structuring element
        - iterations: number of iterations
        """
        binary_array = self._ensure_binary(binary_array)
        
        if structure is None:
            structure = self.create_structuring_element('cube', 3, binary_array.ndim)
        
        return binary_opening(binary_array, structure=structure, iterations=iterations)
    
    def closing(self, binary_array, structure=None, iterations=1):
        """
        Perform morphological closing (dilation followed by erosion).
        Fills small holes and gaps while preserving object size.
        
        Parameters:
        - binary_array: binary numpy array (2D or 3D)
        - structure: structuring element
        - iterations: number of iterations
        """
        binary_array = self._ensure_binary(binary_array)
        
        if structure is None:
            structure = self.create_structuring_element('cube', 3, binary_array.ndim)
        
        return binary_closing(binary_array, structure=structure, iterations=iterations)
    
    def fill_holes(self, binary_array, structure=None):
        """
        Fill holes in binary objects.
        Works on both 2D and 3D arrays.
        
        Parameters:
        - binary_array: binary numpy array (2D or 3D)
        - structure: structuring element for connectivity
        """
        binary_array = self._ensure_binary(binary_array)
        return binary_fill_holes(binary_array, structure=structure)
    
    def remove_small_objects(self, binary_array, min_size=50):
        """
        Remove connected components smaller than min_size.
        
        Parameters:
        - binary_array: binary numpy array (2D or 3D)
        - min_size: minimum size of objects to keep (in voxels/pixels)
        """
        binary_array = self._ensure_binary(binary_array)
        labeled_array, num_labels = label(binary_array)
        
        # Count voxels/pixels in each connected component
        label_sizes = np.bincount(labeled_array.ravel())
        
        # Create mask for objects to keep (size >= min_size)
        keep_labels = label_sizes >= min_size
        keep_labels[0] = False  # Always remove background (label 0)
        
        # Create final result
        result = np.zeros_like(binary_array, dtype=bool)
        for label_idx in np.where(keep_labels)[0]:
            result[labeled_array == label_idx] = True
        
        return result
    
    def gradient(self, binary_array, structure=None):
        """
        Morphological gradient (dilation - erosion).
        Highlights object boundaries.
        
        Parameters:
        - binary_array: binary numpy array (2D or 3D)
        - structure: structuring element
        """
        binary_array = self._ensure_binary(binary_array)
        
        if structure is None:
            structure = self.create_structuring_element('cube', 3, binary_array.ndim)
        
        dilated = self.dilate(binary_array, structure)
        eroded = self.erode(binary_array, structure)
        
        return (dilated & ~eroded)  # Return as boolean
    
    def tophat(self, binary_array, structure=None):
        """
        White top-hat transform (original - opening).
        Extracts small bright structures.
        
        Parameters:
        - binary_array: binary numpy array (2D or 3D)
        - structure: structuring element
        """
        binary_array = self._ensure_binary(binary_array)
        opened = self.opening(binary_array, structure)
        return (binary_array & ~opened)  # Return as boolean
    
    def blackhat(self, binary_array, structure=None):
        """
        Black top-hat transform (closing - original).
        Extracts small dark structures (holes).
        
        Parameters:
        - binary_array: binary numpy array (2D or 3D)
        - structure: structuring element
        """
        binary_array = self._ensure_binary(binary_array)
        closed = self.closing(binary_array, structure)
        return (closed & ~binary_array)  # Return as boolean
    
    def _ensure_binary(self, array):
        """Ensure the array is binary (boolean type)."""
        if array.dtype != bool:
            return (array != 0)
        return array


# Convenience function for quick operations
def quick_morphology(binary_array, operation, **kwargs):
    """
    Quick access to morphological operations.
    
    Parameters:
    - binary_array: binary numpy array (2D or 3D)
    - operation: str, 'erode', 'dilate', 'opening', 'closing', 'fill_holes', 
                 'remove_small', 'gradient', 'tophat', 'blackhat'
    - **kwargs: additional arguments for the operation
    """
    morph = MorphologicalOperations()
    
    operation_map = {
        'erode': morph.erode,
        'dilate': morph.dilate,
        'opening': morph.opening,
        'closing': morph.closing,
        'fill_holes': morph.fill_holes,
        'remove_small': morph.remove_small_objects,
        'gradient': morph.gradient,
        'tophat': morph.tophat,
        'blackhat': morph.blackhat
    }
    
    if operation not in operation_map:
        raise ValueError(f"Operation must be one of: {list(operation_map.keys())}")
    
    return operation_map[operation](binary_array, **kwargs)

def extract_mesh_from_volume(volume_array: np.ndarray, 
                                gaussian_smooth:bool=True, 
                                sigma:float=1.0,
                                fill_holes: bool = True,
                                smooth_iterations: int = 10,
                                affine: np.ndarray = None,
                                closing_iterations: int = 1,
                                vertex_value: np.float32 = 1.0

    ) -> pv.PolyData:

    """    
    Extracts a mesh from a 3D volume array using marching cubes algorithm.
    Parameters
    ----------
    volume_array : np.ndarray
        3D numpy array representing the volume data.

    gaussian_smooth : bool, optional
        Whether to apply Gaussian smoothing to the volume data before extraction (default: True).

    sigma : float, optional
        Standard deviation for Gaussian smoothing (default: 1.0).

    fill_holes : bool, optional
        Whether to fill holes in the extracted mesh (default: True).

    smooth_iterations : int, optional
        Number of smoothing iterations to apply to the mesh (default: 10).

    affine : np.ndarray, optional
        Affine transformation matrix to convert vertices from voxel space to mm space (default: None).

    closing_iterations : int, optional
        Number of iterations for morphological closing operation to fill small gaps in the binary mask (default:

    vertex_value : np.float32, optional
        Value to assign to the vertices in the mesh (default: 1.0).

    Returns
    -------
    tuple
        A tuple containing vertices, faces, normals, and values of the extracted mesh.
    """

    # Binary mask for the specified value
    if not isinstance(volume_array, np.ndarray):
        raise TypeError("The volume_array must be a numpy ndarray.")
    
    if volume_array.ndim != 3:
        raise ValueError("The volume_array must be a 3D numpy ndarray.")
    
    # Everything that is different from 0 is set to 1
    volume_array = (volume_array != 0).astype(np.float32)

    if closing_iterations > 0:
        volume_array = quick_morphology(volume_array, 
                                            'closing',
                                            iterations=closing_iterations)

    # Apply Gaussian smoothing to reduce noise and fill small gaps
    if gaussian_smooth:

        # Apply Gaussian smoothing
        tmp_volume_array = gaussian_filter(volume_array, sigma=sigma)
        # Re-threshold after smoothing
        tmp_volume_array = (tmp_volume_array > 0).astype(int)

        if tmp_volume_array.max() == 0:
            tmp_volume_array = copy.deepcopy(volume_array)
    else:
        tmp_volume_array = copy.deepcopy(volume_array)
        
    # Check if the code exists in the data
    # Extract surface using marching cubes
    vertices, faces, normals, values = measure.marching_cubes(
        volume_array, 
        level=0.5, 
        gradient_direction='ascent'
    )
    if len(faces) == 0:
        raise ValueError(f"No surface extracted for value. The volume may not contain sufficient data.")
    
    # Move vertices to mm space and the apply affine transformation if the affine is provided
    # and it is a 4x4  numpy array
    # If the affine is not provided, the vertices will remain in voxel space
    # Convert vertices to mm space
    if affine is not None and isinstance(affine, np.ndarray) and affine.shape == (4, 4):
        vertices = vox2mm(vertices, affine=affine)

    # Add column with 3's to faces array for PyVista
    faces = np.c_[np.full(len(faces), 3), faces]

    mesh = pv.PolyData(vertices, faces)

    # Mesh processing pipeline for better quality
    # 1. Clean the mesh (remove duplicate points, unused points, degenerate cells)
    mesh = mesh.clean()
    
    # 2. Fill holes if requested
    if fill_holes:
        mesh = mesh.fill_holes(1000)  # Fill holes with max 1000 triangles
    
    # 3. Apply Taubin smoothing (preserves features better than Laplacian)
    if smooth_iterations > 0:
        mesh = mesh.smooth_taubin(n_iter=smooth_iterations, 
                                pass_band=0.1)
    
    # 4. Clean again after smoothing
    mesh = mesh.clean()
    
    # 5. Compute normals for better shading
    mesh = mesh.compute_normals(split_vertices=True)

    mesh.point_data["surface"] = (
        np.ones((len(mesh.points), 1), dtype=np.float32) * vertex_value
    )

    return mesh