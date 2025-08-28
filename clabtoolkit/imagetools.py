import os
import sys
import copy
import warnings

import nibabel as nib
import numpy as np
import subprocess
from pathlib import Path
from typing import Union, Optional, List, Tuple
import pyvista as pv

from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, convolve
from scipy.ndimage import binary_fill_holes, label, binary_closing, gaussian_filter
from scipy.spatial import distance

from skimage import measure

# Importing local modules
from . import misctools as cltmisc
from . import bidstools as cltbids


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############ Section 1: Class and methods to perform morphological operations on images ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
class MorphologicalOperations:
    """
    A class to perform morphological operations on binary arrays.

    Provides methods for common morphological operations including erosion,
    dilation, opening, closing, and hole filling on 2D and 3D binary images.
    """

    def __init__(self):
        """Initialize the morphological operations class."""
        pass

    ########################################################################################################
    def create_structuring_element(self, shape="cube", size=3, dimensions=None):
        """
        Create a structuring element for morphological operations.

        Parameters
        ----------
        shape : str, optional
            Element shape: 'cube'/'square', 'ball'/'disk', or 'cross'. Default is 'cube'.

        size : int, optional
            Size of the structuring element. Default is 3.

        dimensions : int, optional
            Number of dimensions (2 or 3). If None, defaults to 3. Default is None.

        Returns
        -------
        np.ndarray
            Boolean array representing the structuring element.

        Raises
        ------
        ValueError
            If shape is not supported or dimensions are invalid.

        Examples
        --------
        >>> morph = MorphologicalOperations()
        >>> cube_elem = morph.create_structuring_element('cube', size=5)
        >>> ball_elem = morph.create_structuring_element('ball', size=7, dimensions=3)
        """

        if dimensions is None:
            dimensions = 3  # default to 3D

        if shape in ["cube", "square"]:
            # Create cubic/square structuring element
            return np.ones((size,) * dimensions, dtype=bool)

        elif shape in ["ball", "disk"]:
            # Create spherical/circular structuring element
            radius = size // 2
            if dimensions == 2:
                y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
                return x**2 + y**2 <= radius**2
            elif dimensions == 3:
                z, y, x = np.ogrid[
                    -radius : radius + 1, -radius : radius + 1, -radius : radius + 1
                ]
                return x**2 + y**2 + z**2 <= radius**2
            else:
                raise ValueError("Ball/disk only supported for 2D and 3D")

        elif shape == "cross":
            # Create cross-shaped structuring element
            if dimensions == 2:
                cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
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
            raise ValueError(
                "Shape must be 'cube', 'ball', 'cross', 'square', or 'disk'"
            )

    ########################################################################################################
    def erode(self, binary_array, structure=None, iterations=1):
        """
        Perform binary erosion to shrink objects and remove small noise.

        Parameters
        ----------
        binary_array : np.ndarray
            Binary numpy array (2D or 3D).

        structure : np.ndarray, optional
            Structuring element. If None, uses default 3x3 or 3x3x3 cube. Default is None.

        iterations : int, optional
            Number of erosion iterations. Default is 1.

        Returns
        -------
        np.ndarray
            Eroded binary array.

        Examples
        --------
        >>> eroded = morph.erode(binary_image, iterations=2)
        """

        binary_array = self._ensure_binary(binary_array)

        if structure is None:
            structure = self.create_structuring_element("cube", 3, binary_array.ndim)

        return binary_erosion(binary_array, structure=structure, iterations=iterations)

    ########################################################################################################
    def dilate(self, binary_array, structure=None, iterations=1):
        """
        Perform binary dilation to expand objects and fill small gaps.

        Parameters
        ----------
        binary_array : np.ndarray
            Binary numpy array (2D or 3D).

        structure : np.ndarray, optional
            Structuring element. If None, uses default 3x3 or 3x3x3 cube. Default is None.

        iterations : int, optional
            Number of dilation iterations. Default is 1.

        Returns
        -------
        np.ndarray
            Dilated binary array.

        Examples
        --------
        >>> dilated = morph.dilate(binary_image, iterations=3)
        """

        binary_array = self._ensure_binary(binary_array)

        if structure is None:
            structure = self.create_structuring_element("cube", 3, binary_array.ndim)

        return binary_dilation(binary_array, structure=structure, iterations=iterations)

    ########################################################################################################
    def opening(self, binary_array, structure=None, iterations=1):
        """
        Perform morphological opening (erosion followed by dilation).

        Removes small objects and noise while preserving larger structures.

        Parameters
        ----------
        binary_array : np.ndarray
            Binary numpy array (2D or 3D).

        structure : np.ndarray, optional
            Structuring element. Default is None.

        iterations : int, optional
            Number of iterations. Default is 1.

        Returns
        -------
        np.ndarray
            Opened binary array.

        Examples
        --------
        >>> cleaned = morph.opening(noisy_image, iterations=2)
        """

        binary_array = self._ensure_binary(binary_array)

        if structure is None:
            structure = self.create_structuring_element("cube", 3, binary_array.ndim)

        return binary_opening(binary_array, structure=structure, iterations=iterations)

    ########################################################################################################
    def closing(self, binary_array, structure=None, iterations=1):
        """
        Perform morphological closing (dilation followed by erosion).

        Fills small holes and gaps while preserving object size.

        Parameters
        ----------
        binary_array : np.ndarray
            Binary numpy array (2D or 3D).

        structure : np.ndarray, optional
            Structuring element. Default is None.

        iterations : int, optional
            Number of iterations. Default is 1.

        Returns
        -------
        np.ndarray
            Closed binary array.

        Examples
        --------
        >>> filled = morph.closing(image_with_holes, iterations=1)
        """

        binary_array = self._ensure_binary(binary_array)

        if structure is None:
            structure = self.create_structuring_element("cube", 3, binary_array.ndim)

        return binary_closing(binary_array, structure=structure, iterations=iterations)

    ########################################################################################################
    def fill_holes(self, binary_array, structure=None):
        """
        Fill holes in binary objects.

        Parameters
        ----------
        binary_array : np.ndarray
            Binary numpy array (2D or 3D).

        structure : np.ndarray, optional
            Structuring element for connectivity. Default is None.

        Returns
        -------
        np.ndarray
            Binary array with filled holes.

        Examples
        --------
        >>> filled = morph.fill_holes(binary_mask)
        """

        binary_array = self._ensure_binary(binary_array)
        return binary_fill_holes(binary_array, structure=structure)

    ########################################################################################################
    def remove_small_objects(self, binary_array, min_size=50):
        """
        Remove connected components smaller than specified size.

        Parameters
        ----------
        binary_array : np.ndarray
            Binary numpy array (2D or 3D).

        min_size : int, optional
            Minimum size of objects to keep in voxels/pixels. Default is 50.

        Returns
        -------
        np.ndarray
            Binary array with small objects removed.

        Examples
        --------
        >>> cleaned = morph.remove_small_objects(binary_image, min_size=100)
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

    ########################################################################################################
    def gradient(self, binary_array, structure=None):
        """
        Morphological gradient (dilation - erosion) to highlight object boundaries.

        Parameters
        ----------
        binary_array : np.ndarray
            Binary numpy array (2D or 3D).

        structure : np.ndarray, optional
            Structuring element. Default is None.

        Returns
        -------
        np.ndarray
            Binary array containing object boundaries.

        Examples
        --------
        >>> edges = morph.gradient(binary_object)
        """

        binary_array = self._ensure_binary(binary_array)

        if structure is None:
            structure = self.create_structuring_element("cube", 3, binary_array.ndim)

        dilated = self.dilate(binary_array, structure)
        eroded = self.erode(binary_array, structure)

        return dilated & ~eroded  # Return as boolean

    ########################################################################################################
    def tophat(self, binary_array, structure=None):
        """
        White top-hat transform (original - opening) to extract small bright structures.

        Parameters
        ----------
        binary_array : np.ndarray
            Binary numpy array (2D or 3D).

        structure : np.ndarray, optional
            Structuring element. Default is None.

        Returns
        -------
        np.ndarray
            Binary array containing small bright structures.

        Examples
        --------
        >>> small_objects = morph.tophat(binary_image)
        """

        binary_array = self._ensure_binary(binary_array)
        opened = self.opening(binary_array, structure)
        return binary_array & ~opened  # Return as boolean

    ########################################################################################################
    def blackhat(self, binary_array, structure=None):
        """
        Black top-hat transform (closing - original) to extract small dark structures.

        Parameters
        ----------
        binary_array : np.ndarray
            Binary numpy array (2D or 3D).

        structure : np.ndarray, optional
            Structuring element. Default is None.

        Returns
        -------
        np.ndarray
            Binary array containing small dark structures (holes).

        Examples
        --------
        >>> small_holes = morph.blackhat(binary_image)
        """

        binary_array = self._ensure_binary(binary_array)
        closed = self.closing(binary_array, structure)
        return closed & ~binary_array  # Return as boolean

    def _ensure_binary(self, array):
        """Ensure the array is binary (boolean type)."""
        if array.dtype != bool:
            return array != 0
        return array


#####################################################################################################
# Convenience function for quick operations
def quick_morphology(binary_array, operation, **kwargs):
    """
    Quick access to morphological operations without creating class instance.

    Parameters
    ----------
    binary_array : np.ndarray
        Binary numpy array (2D or 3D).

    operation : str
        Operation name: 'erode', 'dilate', 'opening', 'closing', 'fill_holes',
        'remove_small', 'gradient', 'tophat', 'blackhat'.

    **kwargs
        Additional arguments for the specific operation.

    Returns
    -------
    np.ndarray
        Result of the morphological operation.

    Raises
    ------
    ValueError
        If operation is not supported.

    Examples
    --------
    >>> # Quick erosion
    >>> eroded = quick_morphology(binary_image, 'erode', iterations=2)
    >>>
    >>> # Quick hole filling
    >>> filled = quick_morphology(binary_mask, 'fill_holes')
    """
    morph = MorphologicalOperations()

    operation_map = {
        "erode": morph.erode,
        "dilate": morph.dilate,
        "opening": morph.opening,
        "closing": morph.closing,
        "fill_holes": morph.fill_holes,
        "remove_small": morph.remove_small_objects,
        "gradient": morph.gradient,
        "tophat": morph.tophat,
        "blackhat": morph.blackhat,
    }

    if operation not in operation_map:
        raise ValueError(f"Operation must be one of: {list(operation_map.keys())}")

    return operation_map[operation](binary_array, **kwargs)


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############          Section 2: Methods to get attributes from the images              ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def get_voxel_size(affine: np.ndarray):
    """
    Compute voxel dimensions from NIfTI affine matrix.

    Parameters
    ----------
    affine : np.ndarray
        4x4 affine transformation matrix from NIfTI header.

    Returns
    -------
    tuple
        Voxel sizes (voxel_x, voxel_y, voxel_z) in mm.

    Examples
    --------
    >>> img = nib.load('image.nii.gz')
    >>> vox_x, vox_y, vox_z = get_voxel_size(img.affine)
    >>> print(f"Voxel size: {vox_x:.2f} x {vox_y:.2f} x {vox_z:.2f} mm")
    """

    # Extract voxel sizes as the magnitude of each column vector
    voxel_x = np.linalg.norm(affine[:3, 0])
    voxel_y = np.linalg.norm(affine[:3, 1])
    voxel_z = np.linalg.norm(affine[:3, 2])
    return (voxel_x, voxel_y, voxel_z)


####################################################################################################
def get_voxel_volume(affine: np.ndarray) -> float:
    """
    Compute voxel dimensions from an affine matrix.

    Parameters
    ----------
    affine : np.ndarray
        4x4 affine transformation matrix from NIfTI header.

    Returns
    -------
    tuple
        Voxel sizes (voxel_x, voxel_y, voxel_z) in mm.

    Examples
    --------
    >>> img = nib.load('image.nii.gz')
    >>> vox_x, vox_y, vox_z = get_voxel_size(img.affine)
    >>> print(f"Voxel size: {vox_x:.2f} x {vox_y:.2f} x {vox_z:.2f} mm")
    """

    voxel_x, voxel_y, voxel_z = get_voxel_size(affine)
    return voxel_x * voxel_y * voxel_z


####################################################################################################
def get_center(affine: np.ndarray) -> tuple:
    """
    Compute voxel volume from NIfTI affine matrix.

    Parameters
    ----------
    affine : np.ndarray
        4x4 affine transformation matrix from NIfTI header.

    Returns
    -------
    float
        Voxel volume in mm³.

    Examples
    --------
    >>> img = nib.load('image.nii.gz')
    >>> volume = get_voxel_volume(img.affine)
    >>> print(f"Voxel volume: {volume:.3f} mm³")
    """
    return (affine[0, 3], affine[1, 3], affine[2, 3])


####################################################################################################
def get_rotation_matrix(affine: np.ndarray) -> np.ndarray:
    """
    Extract normalized rotation matrix from affine matrix.

    Parameters
    ----------
    affine : np.ndarray
        4x4 affine transformation matrix from NIfTI header.

    Returns
    -------
    np.ndarray
        3x3 normalized rotation matrix (without scaling).

    Examples
    --------
    >>> rotation = get_rotation_matrix(img.affine)
    >>> print(f"Rotation matrix shape: {rotation.shape}")
    """
    # Extract 3x3 rotation/scaling matrix
    rot_scale = affine[:3, :3]
    # Normalize each column to remove scaling and keep only rotation
    rotation = np.zeros_like(rot_scale)
    for i in range(3):
        rotation[:, i] = rot_scale[:, i] / np.linalg.norm(rot_scale[:, i])
    return rotation


####################################################################################################
def get_vox_neighbors(
    coord: np.ndarray, neighborhood: str = "26", dims: str = "3", order: int = 1
) -> np.ndarray:
    """
    Get neighborhood coordinates for a voxel.

    Parameters
    ----------
    coord : np.ndarray
        Coordinates of the center voxel.

    neighborhood : str, optional
        Neighborhood type: '6', '18', '26' for 3D or '4', '8' for 2D. Default is '26'.

    dims : str, optional
        Number of dimensions: '2' or '3'. Default is '3'.

    order : int, optional
        Order parameter (currently unused). Default is 1.

    Returns
    -------
    np.ndarray
        Array of neighbor coordinates.

    Raises
    ------
    ValueError
        If dimensions don't match coordinates or neighborhood type is invalid.

    Examples
    --------
    >>> # Get 26-connected neighbors in 3D
    >>> center = np.array([10, 15, 20])
    >>> neighbors = get_vox_neighbors(center, neighborhood='26', dims='3')
    >>> print(f"Found {len(neighbors)} neighbors")
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


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############           Section 3: Methods to operate over images (e.g. crop)            ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def crop_image_from_mask(
    in_image: str,
    mask: Union[str, np.ndarray],
    out_image: str,
    st_codes: Union[list, np.ndarray] = None,
) -> str:
    """
    Crop image using a mask to minimum bounding box containing specified structures.

    Parameters
    ----------
    in_image : str
        Path to input image file.

    mask : str or np.ndarray
        Path to mask file or mask array. Can be binary or multi-label.

    out_image : str
        Path for output cropped image.

    st_codes : list or np.ndarray, optional
        Structure codes to include in cropping. If None, uses all non-zero values.
        Default is None.

    Returns
    -------
    str
        Path to the created output image.

    Raises
    ------
    ValueError
        If input parameters are invalid or files don't exist.

    Examples
    --------
    >>> # Crop using binary mask
    >>> output = crop_image_from_mask(
    ...     'brain.nii.gz', 'mask.nii.gz', 'cropped_brain.nii.gz'
    ... )
    >>>
    >>> # Crop specific structures
    >>> output = crop_image_from_mask(
    ...     'image.nii.gz', 'segmentation.nii.gz', 'cropped.nii.gz',
    ...     st_codes=[1, 2, 3]
    ... )
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


####################################################################################################
def cropped_to_native(in_image: str, native_image: str, out_image: str) -> str:
    """
    Restore cropped image to dimensions of reference native image.

    Parameters
    ----------
    in_image : str
        Path to cropped image file.

    native_image : str
        Path to reference image defining target dimensions.

    out_image : str
        Path for output restored image.

    Returns
    -------
    str
        Path to the created output image.

    Raises
    ------
    ValueError
        If input parameters are not strings.

    Examples
    --------
    >>> # Restore cropped image to original dimensions
    >>> restored = cropped_to_native(
    ...     'cropped_result.nii.gz',
    ...     'original.nii.gz',
    ...     'restored_result.nii.gz'
    ... )
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


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############      Section 4: Methods to apply transformations or changes of space       ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
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
) -> None:
    """
    Apply ANTs transformation to image with support for multiple transform types.

    Parameters
    ----------
    in_image : str
        Path to input image.

    out_image : str
        Path for transformed output image.

    ref_image : str
        Path to reference image defining target space.

    xfm_output : str
        Path to transformation files (supports affine and nonlinear).

    interp_order : int, optional
        Interpolation method: 0=NearestNeighbor, 1=Linear, 2=BSpline, etc.
        Default is 0.

    invert : bool, optional
        Whether to invert the transformation. Default is False.

    cont_tech : str, optional
        Container technology: 'local', 'singularity', 'docker'. Default is 'local'.

    cont_image : str, optional
        Container image specification. Default is None.

    force : bool, optional
        Force recomputation if output exists. Default is False.

    Examples
    --------
    >>> # Apply transformation with nearest neighbor interpolation
    >>> apply_multi_transf(
    ...     'input.nii.gz', 'output.nii.gz', 'template.nii.gz',
    ...     'transform_prefix', interp_order=0
    ... )
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


####################################################################################################
def vox2mm(vox_coords, affine) -> np.ndarray:
    """
    Convert voxel coordinates to millimeter coordinates using affine matrix.

    Parameters
    ----------
    vox_coords : np.ndarray
        Matrix with voxel coordinates (N x 3).

    affine : np.ndarray
        4x4 affine transformation matrix.

    Returns
    -------
    np.ndarray
        Matrix with millimeter coordinates (N x 3).

    Raises
    ------
    ValueError
        If input matrix doesn't have 3 columns.

    Examples
    --------
    >>> # Convert voxel coordinates to mm
    >>> vox_coords = np.array([[10, 20, 30], [15, 25, 35]])
    >>> mm_coords = vox2mm(vox_coords, img.affine)
    >>> print(f"MM coordinates: {mm_coords}")
    """

    # Detect if the number of rows is bigger than the number of columns. If not, transpose the matrix
    nrows = np.shape(vox_coords)[0]
    ncols = np.shape(vox_coords)[1]
    if (nrows < ncols) and (ncols > 3):
        vox_coords = np.transpose(vox_coords)

    if np.shape(vox_coords)[1] == 3:
        npoints = np.shape(vox_coords)
        vox_coords = np.c_[vox_coords, np.full(npoints[0], 1)]

        mm_coords = np.matmul(affine, vox_coords.T)
        mm_coords = np.transpose(mm_coords)
        mm_coords = mm_coords[:, :3]

    else:
        # Launch an error if the number of columns is different from 3
        raise ValueError("The number of columns of the input matrix must be 3")

    return mm_coords


####################################################################################################
def mm2vox(mm_coords, affine) -> np.ndarray:
    """
    Convert millimeter coordinates to voxel coordinates using affine matrix.

    Parameters
    ----------
    mm_coords : np.ndarray
        Matrix with millimeter coordinates (N x 3).

    affine : np.ndarray
        4x4 affine transformation matrix.

    Returns
    -------
    np.ndarray
        Matrix with voxel coordinates (N x 3).

    Raises
    ------
    ValueError
        If input matrix doesn't have 3 columns.

    Examples
    --------
    >>> # Convert mm coordinates to voxels
    >>> mm_coords = np.array([[45.5, -12.3, 78.9]])
    >>> vox_coords = mm2vox(mm_coords, img.affine)
    >>> print(f"Voxel coordinates: {vox_coords}")
    """

    # Detect if the number of rows is bigger than the number of columns. If not, transpose the matrix
    nrows = np.shape(mm_coords)[0]
    ncols = np.shape(mm_coords)[1]
    if (nrows < ncols) and (ncols > 3):
        mm_coords = np.transpose(mm_coords)

    if np.shape(mm_coords)[1] == 3:
        npoints = np.shape(mm_coords)
        mm_coords = np.c_[mm_coords, np.full(npoints[0], 1)]

        vox_coords = np.matmul(affine, mm_coords.T)
        vox_coords = np.transpose(vox_coords)
        vox_coords = vox_coords[:, :3]

    else:
        # Launch an error if the number of columns is different from 3
        raise ValueError("The number of columns of the input matrix must be 3")

    return vox_coords


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############                Section 5: Methods to work with 4D Images                   ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def spams2maxprob(
    spam_image: str,
    prob_thresh: float = 0.05,
    vol_indexes: np.array = None,
    maxp_name: str = None,
):
    """
    Convert SPAM probability maps to maximum probability parcellation.

    Transforms 4D spatial probability maps into discrete 3D parcellation by
    selecting the most probable label at each voxel, with optional thresholding
    and volume selection.

    Parameters
    ----------
    spam_image : str
        Path to 4D SPAM image file with probability maps for each region.

    prob_thresh : float, optional
        Minimum probability threshold. Values below this are set to zero.
        Default is 0.05.

    vol_indexes : np.ndarray, optional
        Indices of volumes (regions) to include. If None, uses all volumes.
        Default is None.

    maxp_name : str, optional
        Output file path for maximum probability image. If None, returns
        array without saving. Default is None.

    Returns
    -------
    str or np.ndarray
        Output file path if maxp_name provided, otherwise numpy array
        with maximum probability labels.

    Notes
    -----
    The conversion process:
    1. Applies probability threshold to remove low-confidence voxels
    2. Optionally filters to specific volume indices
    3. Finds maximum probability across volumes for each voxel
    4. Assigns winner-take-all labels (1-indexed)
    5. Sets background where no probabilities exceed threshold

    Useful for converting probabilistic atlases to discrete parcellations
    for analysis requiring discrete labels.

    Examples
    --------
    >>> # Convert SPAM to discrete parcellation
    >>> maxprob_file = spams2maxprob(
    ...     'AAL_SPAM.nii.gz',
    ...     prob_thresh=0.1,
    ...     maxp_name='AAL_discrete.nii.gz'
    ... )
    >>>
    >>> # Get array without saving, specific regions only
    >>> selected_regions = np.array([0, 1, 2, 5, 6])  # Volume indices
    >>> maxprob_array = spams2maxprob(
    ...     'probabilistic_atlas.nii.gz',
    ...     vol_indexes=selected_regions,
    ...     prob_thresh=0.2
    ... )
    >>> print(f"Max label: {maxprob_array.max()}")
    """

    spam_img = nib.load(spam_image)
    affine = spam_img.affine
    spam_vol = spam_img.get_fdata()

    spam_vol[spam_vol < prob_thresh] = 0
    spam_vol[spam_vol > 1] = 1

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
        spam_vol[:, :, :, diff_array] = 0
        # array_data = np.delete(spam_vol, diff_array, 3)

    ind = np.where(np.sum(spam_vol, axis=3) == 0)
    maxprob_thl = spam_vol.argmax(axis=3) + 1
    maxprob_thl[ind] = 0

    if maxp_name is not None:
        # Save the image
        imgcoll = nib.Nifti1Image(maxprob_thl.astype("int16"), affine)
        nib.save(imgcoll, maxp_name)
    else:
        maxp_name = maxprob_thl

    return maxp_name


#####################################################################################################
def simulate_image(
    input_image: Union[str, nib.Nifti1Image],
    simulated_image: str = None,
    n_volumes: int = 3,
    distribution: str = "normal",
    random_seed: Optional[int] = None,
    **dist_params,
) -> nib.Nifti1Image:
    """
    Generate a simulated image with random values at non-zero voxel positions.

    This function creates a new NIfTI image where non-zero voxels from the input
    image are filled with random values following a specified statistical distribution.
    The output preserves the spatial dimensions, affine transformation, and header
    information from the input image.

    This is useful for simulating functional or structural images based on a mask
    or anatomical image, allowing for controlled random value generation in specific
    regions of interest.

    Parameters
    ----------
    input_image : str or nibabel.Nifti1Image
        Input image used as a mask. Can be either a file path to a NIfTI image
        or a nibabel Nifti1Image object.

    simulated_image : str
        Output file path where the simulated image will be saved. Must include
        the .nii or .nii.gz extension. If None, the function will generate a
        default name based on the input image.

    n_volumes : int, default=3
        Number of volumes in the output image:
        - If n_volumes == 1: creates a 3D image
        - If n_volumes > 1: creates a 4D image with n_volumes timepoints

    distribution : str, default='normal'
        Statistical distribution for random value generation. Supported options:
        - 'normal': Normal (Gaussian) distribution
        - 'uniform': Uniform distribution
        - 'exponential': Exponential distribution

    random_seed : int, optional
        Random seed for reproducible results. If None, uses system time.

    **dist_params : dict
        Distribution-specific parameters:
        - For 'normal': loc (mean, default=0), scale (std, default=1)
        - For 'uniform': low (default=0), high (default=1)
        - For 'exponential': scale (default=1)

    Returns
    -------
    nibabel.Nifti1Image
        The simulated image object with the same spatial properties as input.

    Raises
    ------
    FileNotFoundError
        If input_image path does not exist.

    ValueError
        If input_image is not a valid type, distribution is unsupported,
        n_volumes is invalid, or output directory doesn't exist.

    RuntimeError
        If no non-zero voxels are found in the input image.

    Examples
    --------
    >>> # Create 3D simulation with normal distribution
    >>> sim_img = simulate_image(
    ...     'brain_mask.nii.gz',
    ...     'output_3d.nii.gz',
    ...     n_volumes=1,
    ...     distribution='normal',
    ...     loc=0,
    ...     scale=1,
    ...     random_seed=42
    ... )

    >>> # Create 4D simulation with uniform distribution
    >>> sim_img = simulate_image(
    ...     'brain_mask.nii.gz',
    ...     'output_4d.nii.gz',
    ...     n_volumes=10,
    ...     distribution='uniform',
    ...     low=0,
    ...     high=100
    ... )

    Notes
    -----
    - Only voxels with non-zero values in the input image will contain random values
    - All other voxels remain zero in the output
    - The function preserves the input image's affine transformation and header
    - Output file extension should be .nii or .nii.gz
    """

    # Input validation
    if not isinstance(n_volumes, int) or n_volumes < 1:
        raise ValueError("n_volumes must be a positive integer")

    if distribution not in ["normal", "uniform", "exponential"]:
        raise ValueError(
            f"Unsupported distribution '{distribution}'. "
            "Supported: 'normal', 'uniform', 'exponential'"
        )

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Load and validate input image
    if isinstance(input_image, str):
        if not os.path.exists(input_image):
            raise FileNotFoundError(f"Input image file not found: {input_image}")
        try:
            input_img = nib.load(input_image)
        except Exception as e:
            raise ValueError(f"Failed to load input image: {e}")

    elif isinstance(input_image, nib.Nifti1Image):
        input_img = input_image
    else:
        raise ValueError(
            "input_image must be a file path (str) or nibabel.Nifti1Image object"
        )

    # Validate simulated_image path. If None, create a temporary filename
    if simulated_image is None:
        simulated_image = cltmisc.create_temporary_filename()

    # Validate output path
    output_dir = os.path.dirname(os.path.abspath(simulated_image))
    if output_dir and not os.path.exists(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")

    if not simulated_image.endswith((".nii", ".nii.gz")):
        warnings.warn(
            "Output filename should have .nii or .nii.gz extension", UserWarning
        )

    # Extract image properties
    input_data = input_img.get_fdata()
    affine = input_img.affine.copy()
    header = input_img.header.copy()
    original_shape = input_data.shape

    # Create mask for non-zero voxels
    mask = input_data != 0
    n_nonzero_voxels = np.sum(mask)

    if n_nonzero_voxels == 0:
        raise RuntimeError("No non-zero voxels found in input image")

    # Distribution parameter validation and random value generation
    def _generate_random_values(size: int) -> np.ndarray:
        """Generate random values based on specified distribution."""
        try:
            if distribution == "normal":
                loc = dist_params.get("loc", 0.0)
                scale = dist_params.get("scale", 1.0)
                if scale <= 0:
                    raise ValueError(
                        "Scale parameter for normal distribution must be positive"
                    )
                return np.random.normal(loc, scale, size)

            elif distribution == "uniform":
                low = dist_params.get("low", 0.0)
                high = dist_params.get("high", 1.0)
                if low >= high:
                    raise ValueError(
                        "Low parameter must be less than high parameter for uniform distribution"
                    )
                return np.random.uniform(low, high, size)

            elif distribution == "exponential":
                scale = dist_params.get("scale", 1.0)
                if scale <= 0:
                    raise ValueError(
                        "Scale parameter for exponential distribution must be positive"
                    )
                return np.random.exponential(scale, size)

        except Exception as e:
            raise ValueError(f"Error generating random values: {e}")

    # Create simulated data array
    if n_volumes == 1:
        # 3D output
        simulated_data = np.zeros(original_shape, dtype=np.float32)
        simulated_data[mask] = _generate_random_values(n_nonzero_voxels)
        output_shape = original_shape

    else:
        # 4D output
        spatial_shape = (
            original_shape[:3] if len(original_shape) >= 3 else original_shape
        )
        output_shape = spatial_shape + (n_volumes,)
        simulated_data = np.zeros(output_shape, dtype=np.float32)

        # Fill each volume with independent random values
        for volume_idx in range(n_volumes):
            simulated_data[..., volume_idx][mask] = _generate_random_values(
                n_nonzero_voxels
            )

    # Update header for correct dimensionality
    header_copy = header.copy()

    if n_volumes == 1 and len(original_shape) > 3:
        # Convert from 4D+ to 3D
        header_copy["dim"][0] = 3
        header_copy["dim"][4] = 1

    elif n_volumes > 1:
        # Ensure 4D header
        header_copy["dim"][0] = 4
        header_copy["dim"][4] = n_volumes

        # Set appropriate time units if not already set
        if header_copy.get("xyzt_units", 0) == 0:
            header_copy["xyzt_units"] = 10  # mm + sec

    # Create output image
    try:
        simulated_img = nib.Nifti1Image(simulated_data, affine, header_copy)
        nib.save(simulated_img, simulated_image)

    except Exception as e:
        raise RuntimeError(f"Failed to create or save simulated image: {e}")

    # Log summary information
    print(f"Successfully created simulated image:")
    print(f"  Input shape: {original_shape}")
    print(f"  Output shape: {output_shape}")
    print(f"  Non-zero voxels: {n_nonzero_voxels:,}")
    print(f"  Distribution: {distribution}")
    print(f"  Saved to: {simulated_image}")

    return simulated_img


#####################################################################################################
def delete_volumes_from_4D_images(
    in_image: str,
    out_image: str,
    vols_to_delete: List[Union[int, tuple, list, str, np.ndarray]] = None,
    overwrite: bool = False,
) -> Tuple[str, List[int]]:
    """
    Remove specific volumes from a 4D neuroimaging file.

    This function removes specified volumes from 4D NIfTI images (such as fMRI time series,
    DTI volumes, or other 4D datasets) and saves the result to a new file. It supports
    flexible volume specification including individual indices, ranges, and complex
    string-based patterns.

    Parameters
    ----------
    in_image : str
        Path to the input 4D NIfTI image file (.nii or .nii.gz).
        The file must exist and be a valid 4D image.

    out_image : str
        Path where the output 4D image will be saved. The directory must exist.
        If the file already exists, use `overwrite=True` to replace it.

    vols_to_delete : list of int, tuple, list, np.ndarray, or str
        Specification of volumes to remove from the 4D image. Supports multiple formats:

        **Individual integers:**
            Single volume indices to remove (0-based indexing).

        **Tuples of 2 integers:**
            Ranges specified as (start, end) - both endpoints are included.
            Example: (5, 8) removes volumes [5, 6, 7, 8]

        **Lists or numpy arrays:**
            Collections of volume indices, automatically flattened.

        **Strings (flexible syntax):**
            Powerful string-based specification supporting:

            - Single numbers: "5" → [5]
            - Hyphen ranges: "8-10" → [8, 9, 10]
            - Colon ranges: "11:13" → [11, 12, 13]
            - Step ranges: "14:2:22" → [14, 16, 18, 20, 22]
            - Comma-separated: "1, 2, 3" → [1, 2, 3]
            - Mixed combinations: "0-2, 5, 10:2:14, 20" → [0, 1, 2, 5, 10, 12, 14, 20]

        **Note:** All formats can be mixed in a single list.

    overwrite : bool, default=False
        Whether to overwrite the output file if it already exists.
        If False and the output file exists, raises FileExistsError.

    Returns
    -------
    out_image : str
        Path to the successfully created output image file.

    vols_removed : list of int
        Sorted list of all volume indices that were removed from the original image.
        Useful for verification and logging purposes.

    Raises
    ------
    FileNotFoundError
        - If the input image file does not exist
        - If the output directory does not exist

    ValueError
        - If vols_to_delete is None or empty after parsing
        - If the input image is not 4D (wrong number of dimensions)
        - If any volume indices are out of range (negative or >= number of volumes)

    FileExistsError
        If the output file already exists and overwrite=False.

    RuntimeError
        If attempting to delete all volumes (would result in empty image).

    Notes
    -----
    - Volume indexing is 0-based (first volume is index 0)
    - Duplicate volume indices are automatically removed
    - The function preserves the original image's affine transformation and header
    - Memory usage scales with image size; very large 4D images may require substantial RAM
    - The function validates all volume indices before processing to prevent partial failures

    **Performance considerations:**
    - Loading and processing large 4D images may take significant time and memory
    - Consider processing in chunks for very large datasets

    **File format support:**
    - Input: .nii and .nii.gz files
    - Output: Format determined by output filename extension

    Examples
    --------
    **Basic usage - Remove specific volumes:**

    >>> delete_volumes_from_4D_images(
    ...     'fmri_data.nii.gz',
    ...     'fmri_cleaned.nii.gz',
    ...     vols_to_delete=[0, 1, 2, 99],
    ...     overwrite=True
    ... )

    **Remove ranges using tuples:**

    >>> delete_volumes_from_4D_images(
    ...     'dti_data.nii.gz',
    ...     'dti_subset.nii.gz',
    ...     vols_to_delete=[(0, 4), (95, 99)],  # Remove first 5 and last 5 volumes
    ...     overwrite=True
    ... )

    **String-based range specification:**

    >>> delete_volumes_from_4D_images(
    ...     'timeseries.nii.gz',
    ...     'timeseries_clean.nii.gz',
    ...     vols_to_delete=["0-4", "95:99"],  # Same as above using strings
    ...     overwrite=True
    ... )

    **Complex mixed specification:**

    >>> delete_volumes_from_4D_images(
    ...     'bold_data.nii.gz',
    ...     'bold_processed.nii.gz',
    ...     vols_to_delete=[
    ...         "0-2",           # Remove first 3 volumes (motion artifacts)
    ...         (147, 149),      # Remove volumes 147-149 (spike artifacts)
    ...         "200:5:220",     # Remove every 5th volume from 200-220
    ...         [300, 301, 302], # Remove specific outlier volumes
    ...         "450"            # Remove final volume
    ...     ],
    ...     overwrite=True
    ... )

    **Advanced string patterns:**

    >>> # Remove multiple ranges and individual volumes in one string
    >>> delete_volumes_from_4D_images(
    ...     'input.nii.gz',
    ...     'output.nii.gz',
    ...     vols_to_delete=["0-2, 5, 10:2:14, 20-22, 50"],
    ...     overwrite=True
    ... )
    >>> # This removes: [0,1,2,5,10,12,14,20,21,22,50]

    """

    # Check if input file exists
    if not os.path.isfile(in_image):
        raise FileNotFoundError(f"File {in_image} not found.")

    # Check if volumes to delete is specified
    if vols_to_delete is None:
        raise ValueError(
            "vols_to_delete parameter is required. Please specify which volumes to remove."
        )

    # Ensure vols_to_delete is a list
    if not isinstance(vols_to_delete, list):
        vols_to_delete = [vols_to_delete]

    # Convert vols_to_delete to a flat list of integers
    vols_to_delete = cltmisc.build_indices(vols_to_delete, nonzeros=False)

    # Check if vols_to_delete is not empty
    if len(vols_to_delete) == 0:
        print("No volumes to delete. The volumes to delete list is empty.")
        return in_image, []

    # Create output filename if not specified
    out_path = os.path.dirname(out_image)

    # Check if the output path exists otherwise give an error
    if out_path and not os.path.exists(out_path):
        raise FileNotFoundError(f"Output directory {out_path} does not exist.")

    if os.path.exists(out_image) and not overwrite:
        raise FileExistsError(
            f"Output file {out_image} already exists. Use 'overwrite=True' to replace it."
        )

    # Load the 4D image
    img = nib.load(in_image)

    # Get the dimensions of the image
    dim = img.shape

    # Check if the image is 4D
    if len(dim) != 4:
        raise ValueError(
            f"Image {in_image} is not a 4D image. It has {len(dim)} dimensions."
        )

    # Get the number of volumes
    nvols = dim[3]

    # Check if trying to delete all volumes
    if len(vols_to_delete) == nvols:
        print(
            "Number of volumes to delete is equal to the total number of volumes. No volumes will be deleted."
        )
        return in_image, []

    # Check if volumes to delete are in valid range
    if np.max(vols_to_delete) >= nvols:
        vols_to_delete_array = np.array(vols_to_delete)
        out_of_range = np.where(vols_to_delete_array >= nvols)[0]
        raise ValueError(
            f"Volumes out of range: {vols_to_delete_array[out_of_range]}. "
            f"Values should be between 0 and {nvols-1}."
        )

    if np.min(vols_to_delete) < 0:
        raise ValueError(
            f"Volumes to delete {vols_to_delete} contain negative indices. "
            f"Values should be between 0 and {nvols-1}."
        )

    # Get volumes to keep and remove
    vols_to_remove = np.array(vols_to_delete)
    vols_to_keep = np.where(np.isin(np.arange(nvols), vols_to_remove, invert=True))[0]

    # Load image data
    img_data = img.get_fdata()
    affine = img.affine
    header = img.header

    # Remove the specified volumes
    filtered_data = img_data[:, :, :, vols_to_keep]

    # Create new image with filtered data
    new_img = nib.Nifti1Image(filtered_data, affine, header)

    # Save the new image
    nib.save(new_img, out_image)

    print(f"Successfully removed {len(vols_to_remove)} volumes from {in_image}")
    print(f"Output saved to: {out_image}")
    print(f"Volumes removed: {vols_to_remove.tolist()}")

    return out_image, vols_to_remove.tolist()


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############        Section 6: Methods to perform operations from 3D or 4D arrays       ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def extract_mesh_from_volume(
    volume_array: np.ndarray,
    gaussian_smooth: bool = True,
    sigma: float = 1.0,
    fill_holes: bool = True,
    smooth_iterations: int = 10,
    affine: np.ndarray = None,
    closing_iterations: int = 1,
    vertex_value: np.float32 = 1.0,
) -> pv.PolyData:
    """
    Extract surface mesh from 3D volume using marching cubes algorithm.

    Creates high-quality surface mesh with optional smoothing, hole filling,
    and coordinate transformation to millimeter space.

    Parameters
    ----------
    volume_array : np.ndarray
        3D binary volume array for mesh extraction.

    gaussian_smooth : bool, optional
        Whether to apply Gaussian smoothing before extraction. Default is True.

    sigma : float, optional
        Standard deviation for Gaussian smoothing. Default is 1.0.

    fill_holes : bool, optional
        Whether to fill holes in extracted mesh. Default is True.

    smooth_iterations : int, optional
        Number of Taubin smoothing iterations. Default is 10.

    affine : np.ndarray, optional
        4x4 affine matrix to transform vertices to mm space. Default is None.

    closing_iterations : int, optional
        Morphological closing iterations before extraction. Default is 1.

    vertex_value : float, optional
        Scalar value assigned to mesh vertices. Default is 1.0.

    Returns
    -------
    pv.PolyData
        PyVista mesh with vertices in mm coordinates (if affine provided),
        computed normals, and scalar values.

    Raises
    ------
    TypeError
        If volume_array is not a numpy array.

    ValueError
        If volume_array is not 3D or no surface can be extracted.

    Notes
    -----
    The extraction pipeline includes:
    1. Morphological closing to fill small gaps
    2. Optional Gaussian smoothing for noise reduction
    3. Marching cubes surface extraction
    4. Mesh cleaning and hole filling
    5. Taubin smoothing for feature preservation
    6. Normal computation for proper shading

    Examples
    --------
    >>> # Basic mesh extraction
    >>> mesh = extract_mesh_from_volume(binary_volume)
    >>> print(f"Mesh has {mesh.n_points} vertices and {mesh.n_cells} faces")
    >>>
    >>> # High-quality mesh with coordinate transformation
    >>> mesh = extract_mesh_from_volume(
    ...     binary_volume,
    ...     affine=img.affine,
    ...     smooth_iterations=20,
    ...     fill_holes=True
    ... )
    >>>
    >>> # Save mesh
    >>> mesh.save('surface.ply')
    """

    # Binary mask for the specified value
    if not isinstance(volume_array, np.ndarray):
        raise TypeError("The volume_array must be a numpy ndarray.")

    if volume_array.ndim != 3:
        raise ValueError("The volume_array must be a 3D numpy ndarray.")

    # Everything that is different from 0 is set to 1
    volume_array = (volume_array != 0).astype(np.float32)

    if closing_iterations > 0:
        volume_array = quick_morphology(
            volume_array, "closing", iterations=closing_iterations
        )

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
        volume_array, level=0.5, gradient_direction="ascent"
    )
    if len(faces) == 0:
        raise ValueError(
            f"No surface extracted for value. The volume may not contain sufficient data."
        )

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
        mesh = mesh.smooth_taubin(n_iter=smooth_iterations, pass_band=0.1)

    # 4. Clean again after smoothing
    mesh = mesh.clean()

    # 5. Compute normals for better shading
    mesh = mesh.compute_normals(split_vertices=True)

    mesh.point_data["surface"] = (
        np.ones((len(mesh.points), 1), dtype=np.uint32) * vertex_value
    )

    return mesh


#####################################################################################################
def extract_centroid_from_volume(
    volume_array: np.ndarray,
    gaussian_smooth: bool = True,
    sigma: float = 1.0,
    closing_iterations: int = 1,
) -> tuple:
    """
    Extract centroid and voxel count from a 3D binary volume.
    Computes the centroid of the non-zero region in the volume and counts the number of voxels.
    Optionally applies Gaussian smoothing and morphological closing to improve the region definition.

    Parameters
    ----------
    volume_array : np.ndarray
        3D binary volume array where non-zero values represent the region of interest.

    gaussian_smooth : bool, optional
        Whether to apply Gaussian smoothing before centroid calculation. Default is True.

    sigma : float, optional
        Standard deviation for Gaussian smoothing. Default is 1.0.

    closing_iterations : int, optional
        Number of morphological closing iterations to apply before centroid calculation. Default is 1.

    Returns
    -------
    tuple
        A tuple containing:
        - Centroid coordinates as a numpy array of shape (3,) in mm space.
        - Voxel count as an integer.

    Raises
    ------
    TypeError
        If volume_array is not a numpy ndarray.

    ValueError
        If volume_array is not 3D or if no region is found in the volume.

    ValueError
        If the volume does not contain sufficient data to compute a centroid.


    Notes
    The function processes the input volume as follows:
    1. Converts non-zero values to 1 to create a binary mask.
    2. Applies morphological closing to fill small gaps in the region.
    3. Optionally applies Gaussian smoothing to reduce noise.
    4. Computes the centroid of the non-zero region.
    5. Counts the number of voxels in the region.
    6. Returns the centroid coordinates and voxel count as a numpy array.

    Examples
    --------
    >>> # Basic centroid extraction
    >>> centroid_info = extract_centroid_from_volume(binary_volume)
    >>> print(f"Centroid: {centroid_info[:3]}, Voxel Count: {centroid_info[3]}")
    >>>
    >>> # With Gaussian smoothing and morphological closing
    >>> centroid_info = extract_centroid_from_volume(binary_volume, gaussian_smooth=True, sigma=1.5, closing_iterations=2)
    >>> print(f"Centroid: {centroid_info[:3]}, Voxel Count: {centroid_info[3]}")
    """

    # Binary mask for the specified value
    if not isinstance(volume_array, np.ndarray):
        raise TypeError("The volume_array must be a numpy ndarray.")

    if volume_array.ndim != 3:
        raise ValueError("The volume_array must be a 3D numpy ndarray.")

    # Everything that is different from 0 is set to 1
    volume_array = (volume_array != 0).astype(np.float32)

    if closing_iterations > 0:
        volume_array = quick_morphology(
            volume_array, "closing", iterations=closing_iterations
        )

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

    # Create mask for current region
    region_x, region_y, region_z = np.where(tmp_volume_array != 0)

    # Skip if region doesn't exist in the data
    if len(region_x) == 0 and len(region_y) == 0 and len(region_z) == 0:
        return (np.array([None, None, None], dtype=np.float32), 0)

    else:
        # Compute centroid
        centroid_x = np.mean(region_x)
        centroid_y = np.mean(region_y)
        centroid_z = np.mean(region_z)

        # Count voxels and compute volume
        voxel_count = len(region_x)

        return (
            np.array([centroid_x, centroid_y, centroid_z], dtype=np.float32),
            int(voxel_count),
        )


####################################################################################################
def compute_statistics_at_nonzero_voxels(
    mask_array: np.ndarray, data_array: np.ndarray, metric: str = "mean"
) -> np.ndarray:
    """
    Compute the value of certain statistic from data_array at positions where mask_array is non-zero.

    Parameters
    -----------
    mask_array : numpy.ndarray
        3D array with dimensions (N, M, P) - used to find non-zero positions
        Non-zero positions in this array indicate where to compute the statistic in data_array.

    data_array : numpy.ndarray
        3D array (N, M, P) or 4D array (N, M, P, T) - data to compute mean from
        If 3D, computes a single mean value.
        If 4D, computes mean across non-zero positions for each time point T.

    metric : str, optional
        Metric to compute at non-zero positions. Default is "mean".
        Supported metrics: "mean", "std", "var", "median", "sum", "max", "min".

    Returns
    --------
    numpy.ndarray or float
        - If data_array is 3D: returns a single float value
        - If data_array is 4D: returns a 1D array of length T

    Raises
    -------
    ValueError
        If data_array is not 3D or 4D.

    ValueError
        If metric is not one of the supported metrics.

    Notes
    -----
    - The function extracts values from data_array at positions where mask_array is non-zero.
    - For 3D data_array, it computes the mean of all selected values.
    - For 4D data_array, it computes the mean across non-zero positions for each time point,
    returning a 1D array with the mean for each time point.
    - If mask_array has no non-zero positions, the function will return NaN for 3D data_array
    or an array of NaNs for 4D data_array.

    Examples
    ---------
    >>> # Example with 3D data_array
    >>> mask = np.array([[[0, 1], [0, 0]],
    ...                  [[1, 0], [0, 1]]])
    >>> data = np.array([[[1, 2], [3, 4]],
    ...                  [[5, 6], [7, 8]]])
    >>> mean_value = compute_statistics_at_nonzero_voxels(mask, data)
    >>> print(mean_value)  # Output: 4.5 (mean of 2, 5, 6, 8)
    >>>
    >>> # Example with 4D data_array
    >>> mask_4d = np.array([[[0, 1], [0, 0]],
    ...                     [[1, 0], [0, 1]]])
    >>> data_4d = np.array([[[[1, 2], [3, 4]],
    ...                      [[5, 6], [7, 8]]],
    ...                     [[[9, 10], [11, 12]],
    ...                      [[13, 14], [15, 16]]]])
    >>> mean_values_4d = compute_statistics_at_nonzero_voxels(mask_4d, data_4d)
    >>> print(mean_values_4d)  # Output: [ 6.  8. 10. 12.] (mean for each time point
    1, 2, 3, 4)

    """

    # Validate metric
    metric = metric.lower()
    valid_metrics = ["mean", "std", "var", "median", "sum", "max", "min"]
    if metric not in valid_metrics:
        raise ValueError(
            f"Invalid metric '{metric}'. Supported metrics: {', '.join(valid_metrics)}"
        )

    # Find indices where mask_array is non-zero
    nonzero_mask = mask_array != 0

    # Check if data_array is 3D or 4D
    if data_array.ndim == 3:
        # 3D case: extract values at non-zero positions and compute mean
        selected_values = data_array[nonzero_mask]
        if len(selected_values) == 0:
            return np.nan

        if metric == "mean":
            return np.mean(selected_values)

        elif metric == "std":
            return np.std(selected_values)

        elif metric == "var":
            return np.var(selected_values)

        elif metric == "median":
            return np.median(selected_values)

        elif metric == "sum":
            return np.sum(selected_values)

        elif metric == "max":
            return np.max(selected_values)

        elif metric == "min":
            return np.min(selected_values)

    elif data_array.ndim == 4:
        # 4D case: extract values at non-zero positions for each time point
        selected_values = data_array[nonzero_mask, :]  # Shape: (num_nonzero_voxels, T)

        if metric == "mean":
            return np.mean(selected_values, axis=0)  # Mean across voxels, returns (T,)
        elif metric == "std":
            return np.std(selected_values, axis=0)  # Std across voxels, returns (T,)
        elif metric == "var":
            return np.var(
                selected_values, axis=0
            )  # Variance across voxels, returns (T,)
        elif metric == "median":
            return np.median(
                selected_values, axis=0
            )  # Median across voxels, returns (T,)
        elif metric == "sum":
            return np.sum(selected_values, axis=0)  # Sum across voxels, returns (T,)
        elif metric == "max":
            return np.max(selected_values, axis=0)  # Max across voxels, returns (T,)
        elif metric == "min":
            return np.min(selected_values, axis=0)  # Min across voxels, returns (T,)

    else:
        raise ValueError("data_array must be 3D or 4D")


#####################################################################################################
def region_growing(
    iparc: np.ndarray, mask: Union[np.ndarray, np.bool_], neighborhood="26"
) -> np.ndarray:
    """
    Fill gaps in parcellation using region growing algorithm.

    Labels unlabeled voxels within the mask by assigning the most frequent
    label among their labeled neighbors, iteratively until convergence or
    no more voxels can be labeled.

    Parameters
    ----------
    iparc : np.ndarray
        3D parcellation array with labeled (>0) and unlabeled (0) voxels.

    mask : np.ndarray or np.bool_
        3D binary mask defining the region where growing should occur.

    neighborhood : str, optional
        Neighborhood connectivity: '6', '18', or '26' for 3D. Default is '26'.

    Returns
    -------
    np.ndarray
        Updated parcellation array with gaps filled, masked to input mask.

    Notes
    -----
    The algorithm works iteratively:
    1. Identifies unlabeled voxels with at least one labeled neighbor
    2. For each candidate voxel, finds most frequent label among neighbors
    3. In case of ties, selects label from spatially closest neighbor
    4. Repeats until no more voxels can be labeled or convergence

    Particularly useful for:
    - Filling gaps in atlas-based parcellations
    - Completing partial segmentations
    - Correcting registration artifacts

    Examples
    --------
    >>> # Fill gaps in parcellation
    >>> filled_parc = region_growing(parcellation_array, brain_mask)
    >>> print(f"Filled {np.sum(filled_parc > 0) - np.sum(parcellation_array > 0)} voxels")
    >>>
    >>> # Use 6-connectivity for more conservative growing
    >>> conservative_fill = region_growing(
    ...     incomplete_labels,
    ...     region_mask,
    ...     neighborhood='6'
    ... )
    """

    # Create a binary array where labeled voxels are marked as 1
    binary_labels = (iparc > 0).astype(int)

    # Convolve with the kernel to count labeled neighbors for each voxel
    kernel = np.array(
        [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        ]
    )
    labeled_neighbor_count = convolve(binary_labels, kernel, mode="constant", cval=0)

    # Mask for voxels that have at least one labeled neighbor
    mask_with_labeled_neighbors = (labeled_neighbor_count > 0) & (iparc == 0)
    ind = np.argwhere(
        (mask_with_labeled_neighbors != 0) & (binary_labels == 0) & (mask)
    )
    ind_orig = ind.copy() * 0

    # Loop until no more voxels could be labeled or all the voxels are labeled
    while (len(ind) > 0) & (np.array_equal(ind, ind_orig) == False):
        ind_orig = ind.copy()
        # Process each unlabeled voxel
        for coord in ind:
            x, y, z = coord

            # Detecting the neighbors
            neighbors = get_vox_neighbors(coord=coord, neighborhood="26", dims="3")
            # Remove from motion the coordinates out of the bounding box
            neighbors = neighbors[
                (neighbors[:, 0] >= 0)
                & (neighbors[:, 0] < iparc.shape[0])
                & (neighbors[:, 1] >= 0)
                & (neighbors[:, 1] < iparc.shape[1])
                & (neighbors[:, 2] >= 0)
                & (neighbors[:, 2] < iparc.shape[2])
            ]

            # Labels of the neighbors
            neigh_lab = iparc[neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]]

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
                    distances = [
                        distance.euclidean(coord, (dx, dy, dz))
                        for (dx, dy, dz), lbl in zip(neighbors, neigh_lab)
                        if lbl in max_labels
                    ]
                    closest_label = max_labels[np.argmin(distances)]
                    iparc[x, y, z] = closest_label
                # most_frequent_label = np.bincount(neigh_lab[neigh_lab != 0]).argmax()

        # Create a binary array where labeled voxels are marked as 1
        binary_labels = (iparc > 0).astype(int)

        # Convolve with the kernel to count labeled neighbors for each voxel
        labeled_neighbor_count = convolve(
            binary_labels, kernel, mode="constant", cval=0
        )

        # Mask for voxels that have at least one labeled neighbor
        mask_with_labeled_neighbors = (labeled_neighbor_count > 0) & (iparc == 0)
        ind = np.argwhere(
            (mask_with_labeled_neighbors != 0) & (binary_labels == 0) & (mask)
        )

    return iparc * mask


#####################################################################################################
def simulate_array(
    mask_array: np.ndarray,
    n_volumes: int = 1,
    distribution: str = "normal",
    random_seed: Optional[int] = None,
    **dist_params,
) -> np.ndarray:
    """
    Generate a simulated array with random values at non-zero voxel positions.

    This function creates a new array where non-zero voxels from the mask array
    are filled with random values following a specified statistical distribution.

    Parameters
    ----------
    mask_array : numpy.ndarray
        3D array used as a mask to determine where to place random values.
        Shape: (N, M, P)

    n_volumes : int, default=1
        Number of volumes in the output array:
        - If n_volumes == 1: creates a 3D array (N, M, P)
        - If n_volumes > 1: creates a 4D array (N, M, P, n_volumes)

    distribution : str, default='normal'
        Statistical distribution for random value generation. Supported options:
        - 'normal': Normal (Gaussian) distribution
        - 'uniform': Uniform distribution
        - 'exponential': Exponential distribution

    random_seed : int, optional
        Random seed for reproducible results. If None, uses system time.

    **dist_params : dict
        Distribution-specific parameters:
        - For 'normal': loc (mean, default=0), scale (std, default=1)
        - For 'uniform': low (default=0), high (default=1)
        - For 'exponential': scale (default=1)

    Returns
    -------
    numpy.ndarray
        Simulated array with random values at non-zero mask positions:
        - 3D array (N, M, P) if n_volumes == 1
        - 4D array (N, M, P, n_volumes) if n_volumes > 1

    Raises
    ------
    ValueError
        If mask_array is not 3D, distribution is unsupported,
        n_volumes is invalid, or distribution parameters are invalid.

    RuntimeError
        If no non-zero voxels are found in the mask array.

    Examples
    --------
    >>> # Create 3D simulation with normal distribution
    >>> mask = np.random.randint(0, 2, (64, 64, 30))
    >>> sim_3d = simulate_array(
    ...     mask,
    ...     n_volumes=1,
    ...     distribution='normal',
    ...     loc=0,
    ...     scale=1,
    ...     random_seed=42
    ... )

    >>> # Create 4D simulation with uniform distribution
    >>> sim_4d = simulate_array(
    ...     mask,
    ...     n_volumes=10,
    ...     distribution='uniform',
    ...     low=0,
    ...     high=100,
    ...     random_seed=42
    ... )

    Notes
    -----
    - Only voxels with non-zero values in the mask array will contain random values
    - All other voxels remain zero in the output
    - For 4D outputs, each volume gets independent random values
    """

    # Input validation
    if not isinstance(mask_array, np.ndarray):
        raise ValueError("mask_array must be a numpy array")

    if mask_array.ndim != 3:
        raise ValueError(f"mask_array must be 3D, got {mask_array.ndim}D")

    if not isinstance(n_volumes, int) or n_volumes < 1:
        raise ValueError("n_volumes must be a positive integer")

    if distribution not in ["normal", "uniform", "exponential"]:
        raise ValueError(
            f"Unsupported distribution '{distribution}'. "
            "Supported: 'normal', 'uniform', 'exponential'"
        )

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create mask for non-zero voxels
    mask = mask_array != 0
    n_nonzero_voxels = np.sum(mask)

    if n_nonzero_voxels == 0:
        raise RuntimeError("No non-zero voxels found in mask array")

    # Distribution parameter validation and random value generation
    def _generate_random_values(size: int) -> np.ndarray:
        """Generate random values based on specified distribution."""
        try:
            if distribution == "normal":
                loc = dist_params.get("loc", 0.0)
                scale = dist_params.get("scale", 1.0)
                if scale <= 0:
                    raise ValueError(
                        "Scale parameter for normal distribution must be positive"
                    )
                return np.random.normal(loc, scale, size).astype(np.float32)

            elif distribution == "uniform":
                low = dist_params.get("low", 0.0)
                high = dist_params.get("high", 1.0)
                if low >= high:
                    raise ValueError(
                        "Low parameter must be less than high parameter for uniform distribution"
                    )
                return np.random.uniform(low, high, size).astype(np.float32)

            elif distribution == "exponential":
                scale = dist_params.get("scale", 1.0)
                if scale <= 0:
                    raise ValueError(
                        "Scale parameter for exponential distribution must be positive"
                    )
                return np.random.exponential(scale, size).astype(np.float32)

        except Exception as e:
            raise ValueError(f"Error generating random values: {e}")

    # Get mask shape
    mask_shape = mask_array.shape

    # Create simulated data array
    if n_volumes == 1:
        # 3D output
        simulated_data = np.zeros(mask_shape, dtype=np.float32)
        simulated_data[mask] = _generate_random_values(n_nonzero_voxels)
        output_shape = mask_shape

    else:
        # 4D output
        output_shape = mask_shape + (n_volumes,)
        simulated_data = np.zeros(output_shape, dtype=np.float32)

        # Fill each volume with independent random values
        for volume_idx in range(n_volumes):
            simulated_data[..., volume_idx][mask] = _generate_random_values(
                n_nonzero_voxels
            )

    return simulated_data


#####################################################################################################
def delete_volumes_from_4D_array(
    in_array: np.ndarray,
    vols_to_delete: List[Union[int, tuple, list, str, np.ndarray]] = None,
) -> Tuple[str, List[int]]:
    """
    Remove specific volumes from a 4D array.

    This function allows you to delete specified volumes from a 4D numpy array,
    which is commonly used in neuroimaging data (e.g., fMRI, DTI). It supports
    various formats for specifying which volumes to remove, including individual indices,
    ranges, lists, and flexible string-based specifications.
    The function returns the modified array and a list of removed volume indices.
    It is designed to handle large 4D arrays efficiently and ensures that the
    integrity of the remaining data is preserved.
    4D arrays are expected to have the shape (X, Y, Z, T), where T is the number of volumes.

    Parameters
    ----------
    in_array : np.ndarray
        Input 4D numpy array from which volumes will be removed.

    vols_to_delete : list of int, tuple, list, np.ndarray, or str
        Specification of volumes to remove from the 4D image. Supports multiple formats:

        **Individual integers:**
            Single volume indices to remove (0-based indexing).

        **Tuples of 2 integers:**
            Ranges specified as (start, end) - both endpoints are included.
            Example: (5, 8) removes volumes [5, 6, 7, 8]

        **Lists or numpy arrays:**
            Collections of volume indices, automatically flattened.

        **Strings (flexible syntax):**
            Powerful string-based specification supporting:

            - Single numbers: "5" → [5]
            - Hyphen ranges: "8-10" → [8, 9, 10]
            - Colon ranges: "11:13" → [11, 12, 13]
            - Step ranges: "14:2:22" → [14, 16, 18, 20, 22]
            - Comma-separated: "1, 2, 3" → [1, 2, 3]
            - Mixed combinations: "0-2, 5, 10:2:14, 20" → [0, 1, 2, 5, 10, 12, 14, 20]

        **Note:** All formats can be mixed in a single list.

    Returns
    -------
    out_array : np.ndarray
        The modified 4D numpy array with specified volumes removed.

    vols_removed : list of int
        Sorted list of all volume indices that were removed from the original image.
        Useful for verification and logging purposes.

    Raises
    ------
    TypeError
        If `in_array` is not a numpy ndarray or if `vols_to_delete` is not a list or convertible to a list.

    ValueError
        If `in_array` is not a 4D array, if `vols_to_delete` is empty,
        or if any specified volume indices are out of range.

    Notes
    - The function checks for valid input types and dimensions.
    - It handles both individual volume indices and ranges specified as tuples or strings.
    - The output array retains the original shape minus the removed volumes.
    - If all volumes are specified for deletion, it returns the original array unchanged.


    Examples
    --------
    **Basic usage - Remove specific volumes:**

    >>> delete_volumes_from_4D_array(
    ...     in_array=np.random.rand(64, 64, 30, 100),  # Example 4D array
    ...     vols_to_delete=[0, 1, 2],  # Remove first 3 volumes
    ... )

    (array with shape (64, 64, 30, 97), [0, 1, 2])

    **Remove a range of volumes:**
    >>> delete_volumes_from_4D_array(
    ...     in_array=np.random.rand(64, 64, 30, 100),  # Example 4D array
    ...     vols_to_delete=[(5, 8)],  # Remove volumes 5 to 8 (inclusive)
    ... )
    (array with shape (64, 64, 30, 96), [5, 6, 7, 8])

    **Remove volumes using a string specification:**
    >>> delete_volumes_from_4D_array(
    ...     in_array=np.random.rand(64, 64, 30, 100),  # Example 4D array
    ...     vols_to_delete=["0-2", "5", "10:2:14", "20"]  # Mixed specification
    ... )
    (array with shape (64, 64, 30, 92), [0, 1, 2, 5, 10, 12, 14, 20])

    """

    # Ensure input is a numpy array
    if not isinstance(in_array, np.ndarray):
        raise TypeError("Input in_array must be a numpy ndarray.")

    # Check if it is a 4D array
    if in_array.ndim != 4:
        raise ValueError(
            f"Input in_array must be a 4D array. It has {in_array.ndim} dimensions."
        )

    # Ensure vols_to_delete is a list
    if not isinstance(vols_to_delete, list):
        vols_to_delete = [vols_to_delete]

    # Convert vols_to_delete to a flat list of integers
    vols_to_delete = cltmisc.build_indices(vols_to_delete, nonzeros=False)

    # Check if vols_to_delete is not empty
    if len(vols_to_delete) == 0:
        print("No volumes to delete. The volumes to delete list is empty.")
        return in_array, []

    # Get the dimensions of the array
    dim = in_array.shape

    # Get the number of volumes
    nvols = dim[3]

    # Check if trying to delete all volumes
    if len(vols_to_delete) == nvols:
        print(
            "Number of volumes to delete is equal to the total number of volumes. No volumes will be deleted."
        )
        return in_array, []

    # Check if volumes to delete are in valid range
    if np.max(vols_to_delete) >= nvols:
        vols_to_delete_array = np.array(vols_to_delete)
        out_of_range = np.where(vols_to_delete_array >= nvols)[0]
        raise ValueError(
            f"Volumes out of range: {vols_to_delete_array[out_of_range]}. "
            f"Values should be between 0 and {nvols-1}."
        )

    if np.min(vols_to_delete) < 0:
        raise ValueError(
            f"Volumes to delete {vols_to_delete} contain negative indices. "
            f"Values should be between 0 and {nvols-1}."
        )

    # Get volumes to keep and remove
    vols_to_remove = np.array(vols_to_delete)
    vols_to_keep = np.where(np.isin(np.arange(nvols), vols_to_remove, invert=True))[0]

    # Remove the specified volumes from the input array
    out_array = in_array[:, :, :, vols_to_keep]

    return out_array, list(vols_to_remove.tolist())
