import os
import sys
import copy

import nibabel as nib
import numpy as np
import subprocess
from pathlib import Path
from typing import Union
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
############    Section 5: Methods to perform, or work with, tesselations from images   ############
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
        np.ones((len(mesh.points), 1), dtype=np.float32) * vertex_value
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

        return (np.array([centroid_x, centroid_y, centroid_z], dtype=np.float32), int(voxel_count))

#####################################################################################################
def region_growing(
    iparc: np.ndarray, mask: Union[np.ndarray, np.bool_], neighborhood="26"
):
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
