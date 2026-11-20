"""The imagetools module provides advanced neuroimaging operations, morphological
processing, and image manipulation capabilities specifically designed for brain
imaging data.
"""

import json
import os
import subprocess
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    binary_opening,
    convolve,
    gaussian_filter,
    label,
)
from skimage import measure

from . import bidstools as cltbids
from . import colorstools as cltcol
from . import dwitools_utils as dwiutils
from . import imagetools_utils as imgutils

# Importing local modules
from . import misctools as cltmisc
from . import parcellationtools as cltparc


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

    ############################################################################################
    def dilate_mm(
        self,
        array: np.ndarray,
        affine: np.ndarray,
        shape: str = "cube",
        dilation_mm: float = 1,
    ) -> np.ndarray:
        """
        Binarize a 3D array and dilate by a given distance in millimeters.

        Parameters
        ----------
        array       : 3D numpy array with values and zeros
        affine      : 4x4 affine matrix (voxel → mm)
        shape       : 'cube' for a box or 'ball' for an ellipsoid structuring element
        dilation_mm : dilation radius in millimeters

        Returns
        -------
        dilated : binary 3D numpy array (bool)
        """

        binary_array = self._ensure_binary(array)
        voxel_sizes = get_voxel_size(affine)

        # Per-axis radii in voxels
        radii_vox = np.array([dilation_mm / vs for vs in voxel_sizes])
        r = np.ceil(radii_vox).astype(int)

        if shape is None:
            shape = "cube"

        # If voxels are isotropic, reuse create_structuring_element directly
        if np.allclose(r, r[0]):
            size = 2 * r[0] + 1
            structure = self.create_structuring_element(
                shape=shape, size=size, dimensions=binary_array.ndim
            )

        # Anisotropic voxels: build the structure manually per axis
        else:
            if shape in ["cube", "square"]:
                # Non-square box sized independently per axis
                structure = np.ones([2 * ri + 1 for ri in r], dtype=bool)

            elif shape in ["ball", "disk"]:
                # True ellipsoid in mm space
                xi = np.arange(-r[0], r[0] + 1)
                yi = np.arange(-r[1], r[1] + 1)
                zi = np.arange(-r[2], r[2] + 1)
                xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
                structure = (
                    (xx / radii_vox[0]) ** 2
                    + (yy / radii_vox[1]) ** 2
                    + (zz / radii_vox[2]) ** 2
                ) <= 1.0

            else:
                raise ValueError("Shape must be 'cube', 'ball', 'square', or 'disk'")

        return binary_dilation(binary_array, structure=structure)

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

        # scipy treats everything outside the array as background, so the erosion
        # step removes foreground voxels touching the border and closing stops being
        # extensive. Padding by the reach of the dilation avoids that artifact.
        pad = max(s // 2 for s in structure.shape) * iterations
        padded = np.pad(binary_array, pad, mode="constant", constant_values=False)
        closed = binary_closing(padded, structure=structure, iterations=iterations)

        return closed[tuple(slice(pad, pad + s) for s in binary_array.shape)]

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
    def detect_communities(self, binary_array, structure=None):
        """
        Detect connected components (communities) in a binary array.

        Parameters
        ----------
        binary_array : np.ndarray
            Binary numpy array (2D or 3D).

        structure : np.ndarray, optional
            Structuring element for connectivity. Default is None.

        Returns
        -------
        labeled_array : np.ndarray
            Array where each connected component has a unique label.
        num_labels : int
            Number of connected components detected.

        Examples
        --------
        >>> labeled, num = morph.detect_communities(binary_image)
        """

        binary_array = self._ensure_binary(binary_array)

        if structure is None:
            structure = self.create_structuring_element("cube", 3, binary_array.ndim)

        labeled_array, num_labels = label(binary_array, structure=structure)
        return labeled_array, num_labels

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
def get_orientation(affine: np.ndarray) -> str:
    """
    Get the orientation code of the voxel axes from an affine matrix.

    Parameters
    ----------
    affine : np.ndarray
        4x4 affine transformation matrix.

    Returns
    -------
    str
        Three-letter orientation code (e.g. 'RAS', 'LAS', 'LPI'). Each letter is
        the anatomical direction towards which the corresponding voxel axis
        increases.

    Examples
    --------
    >>> img = nib.load('image.nii.gz')
    >>> get_orientation(img.affine)
    'LAS'
    """

    return "".join(nib.aff2axcodes(affine))


####################################################################################################
def get_voxel_volume(affine: np.ndarray) -> float:
    """
    Compute voxel volume from an affine matrix.

    Parameters
    ----------
    affine : np.ndarray
        4x4 affine transformation matrix from NIfTI header.

    Returns
    -------
    float
        Voxel volume in mm³ (the product of the three voxel dimensions).

    Examples
    --------
    >>> img = nib.load('image.nii.gz')
    >>> volume = get_voxel_volume(img.affine)
    >>> print(f"Voxel volume: {volume:.3f} mm³")
    """

    voxel_x, voxel_y, voxel_z = get_voxel_size(affine)
    return voxel_x * voxel_y * voxel_z


####################################################################################################
def get_center(affine: np.ndarray) -> tuple:
    """
    Get the world-space origin (translation component) from a NIfTI affine matrix.

    Parameters
    ----------
    affine : np.ndarray
        4x4 affine transformation matrix from NIfTI header.

    Returns
    -------
    tuple
        (x, y, z) coordinates in mm of the world-space origin encoded in the
        affine's translation column (i.e. the mm location of voxel (0, 0, 0)).

    Examples
    --------
    >>> img = nib.load('image.nii.gz')
    >>> origin = get_center(img.affine)
    >>> print(f"Origin: {origin}")
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
        Neighborhood type: '6', '12', '18', '26' for 3D or '4', '8' for 2D. Default is '26'.

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
        if neighborhood not in ["6", "12", "18", "26"]:
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
    mask: str | np.ndarray,
    out_image: str,
    st_codes: list | np.ndarray = None,
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
        If input parameters are invalid, files don't exist, or the mask's
        spatial shape doesn't match the input image's shape.

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

    if not isinstance(in_image, str):
        raise ValueError("The 'image' parameter must be a string.")

    if isinstance(mask, str):
        if not os.path.exists(mask):
            raise ValueError(f"The mask file does not exist: {mask}")
        else:
            mask = nib.load(mask)
            # Stored dtype (e.g. int16 labels) instead of a float64 copy
            mask_data = np.asanyarray(mask.dataobj)
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
    if not os.path.exists(out_pth):
        Path(out_pth).mkdir(parents=True, exist_ok=True)

    # Loading the image lazily: only the cropped block is ever read into memory
    img1 = nib.load(in_image)  # Original MRI image

    # The mask's spatial dimensions must match the image's - otherwise the
    # masking below either raises an opaque broadcasting error or, worse,
    # silently produces a wrong-sized/garbage result if the shapes happen to be
    # broadcastable.
    if mask_data.shape[:3] != img1.shape[:3]:
        raise ValueError(
            f"Mask spatial shape {mask_data.shape[:3]} does not match "
            f"image spatial shape {img1.shape[:3]}."
        )

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
    del tmp_var

    # Crop first, then mask: the full-size image is never loaded or multiplied.
    # The end indexes are inclusive, so +1 keeps the last slice of the mask.
    # SpatialImage.slicer also updates the affine so the crop keeps its
    # position in mm.
    crop_slices = (
        slice(i_start, i_end + 1),
        slice(j_start, j_end + 1),
        slice(k_start, k_end + 1),
    )
    cropped_ref = img1.slicer[crop_slices]
    cropped_data = cropped_ref.get_fdata()
    cropped_mask = tmask[crop_slices]

    # If the image is 4D the mask is applied to every volume. Not in place:
    # get_fdata may return a read-only buffer for float64 images.
    if cropped_data.ndim == 4:
        cropped_data = cropped_data * cropped_mask[..., np.newaxis]
    else:
        cropped_data = cropped_data * cropped_mask

    # Saving the cropped image
    cropped_img = nib.Nifti1Image(cropped_data, cropped_ref.affine)
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

    if not isinstance(in_image, str):
        raise ValueError("The 'in_image' parameter must be a string.")

    if not isinstance(native_image, str):
        raise ValueError("The 'native_image' parameter must be a string.")

    # Create the output directory if it does not exist
    out_pth = os.path.dirname(out_image)
    if not os.path.exists(out_pth):
        Path(out_pth).mkdir(parents=True, exist_ok=True)

    # Loading both images lazily. Only the native image's shape and affine are
    # needed, so its voxels are never read. keep_file_open=True lets a 4D .nii.gz
    # be streamed volume by volume with a single decompression pass.
    img1 = nib.load(native_image)  # Original MRI image
    img2 = nib.load(in_image, keep_file_open=True)  # Cropped image

    # Get the affine matrices and the destination shape
    img1_affine = img1.affine
    img2_affine = img2.affine
    native_shape = tuple(img1.shape[:3])
    img2_shape = img2.shape

    if len(img2_shape) not in (3, 4):
        raise ValueError(
            f"'{in_image}' must be a 3D or 4D image, got {len(img2_shape)}D."
        )

    # Multiply the inverse of the affine matrix of img1 by the affine matrix of img2
    affine_mult = np.linalg.inv(img1_affine) @ img2_affine

    def _paste_nonzero(src: np.ndarray, dst: np.ndarray, vol_msg: str) -> None:
        """Copy the non-zero voxels of the 3D ``src`` into the 3D ``dst``."""
        # Find the coordinates in voxels of the voxels different from 0 on the img2
        indices = np.argwhere(src != 0)

        # Apply the affine transformation to the coordinates of the voxels different from 0 on img2
        new_coords = np.round(
            affine_mult
            @ np.concatenate((indices.T, np.ones((1, indices.shape[0]))), axis=0)
        ).astype(int)

        # Guard against transformed coordinates that fall outside img1's
        # bounds - without this, mismatched affines/FOVs either raise a
        # raw IndexError or, with negative indices, silently wrap around
        # and corrupt the output.
        in_bounds = (
            (new_coords[0] >= 0)
            & (new_coords[0] < native_shape[0])
            & (new_coords[1] >= 0)
            & (new_coords[1] < native_shape[1])
            & (new_coords[2] >= 0)
            & (new_coords[2] < native_shape[2])
        )
        n_dropped = int(np.size(in_bounds) - np.sum(in_bounds))
        if n_dropped > 0:
            warnings.warn(
                f"{n_dropped} voxel(s){vol_msg} mapped outside the native image "
                f"bounds and were dropped. Check that '{in_image}' and "
                f"'{native_image}' share a consistent field of view.",
                stacklevel=3,
            )

        # Fill the new image with the values of the voxels different from 0 on img2
        dst[
            new_coords[0, in_bounds], new_coords[1, in_bounds], new_coords[2, in_bounds]
        ] = src[indices[in_bounds, 0], indices[in_bounds, 1], indices[in_bounds, 2]]

    # The data are read in their stored type (e.g. int16 labels, float32
    # probabilities) rather than as float64, which halves or quarters the size of
    # the native-space output array.
    if len(img2_shape) == 4:
        n_vols = img2_shape[3]
        new_data = None
        for vol in range(n_vols):
            vol_data = np.asanyarray(img2.dataobj[..., vol])
            if new_data is None:
                # Create an empty array with the same dimensions as IM1
                new_data = np.zeros(native_shape + (n_vols,), dtype=vol_data.dtype)

            # new_data[..., vol] is a view, so pasting into it fills new_data
            _paste_nonzero(vol_data, new_data[..., vol], f" in volume {vol}")
            del vol_data

    else:
        img2_data = np.asanyarray(img2.dataobj)

        # Create an empty array with the same dimensions as IM1
        new_data = np.zeros(native_shape, dtype=img2_data.dtype)
        _paste_nonzero(img2_data, new_data, "")
        del img2_data

    # Release the input file handle (kept open for streaming) before writing,
    # since callers may write the output over the input file.
    del img2

    # Create a new Nifti image with the same affine and header as IM1
    new_img2 = nib.Nifti1Image(new_data, affine=img1_affine, header=img1.header)

    # Save the new image
    nib.save(new_img2, out_image)

    return out_image


####################################################################################################
def reorient_image(
    in_image: str | Path | nib.spatialimages.SpatialImage,
    out_image: str | Path = None,
    orientation: str = "RAS",
    overwrite: bool = False,
) -> str | nib.spatialimages.SpatialImage:
    """
    Reorient an image to a given voxel orientation without resampling.

    The voxel array is only permuted and flipped (no interpolation), and the
    affine is updated accordingly, so the world coordinates of every voxel are
    preserved and the result overlays exactly on the input. For NIfTI images the
    stored values, the data type and the intensity scaling (scl_slope and
    scl_inter) are carried over unchanged, and the qform and the sform are
    reoriented independently.

    Parameters
    ----------
    in_image : str, Path or nibabel image
        Path to the input image, or an already loaded nibabel image.

    out_image : str or Path, optional
        Path for the reoriented image. If None, nothing is saved and the
        reoriented image object is returned. Default is None.

    orientation : str, optional
        Three-letter target orientation code, with one letter from each of the
        pairs L/R, A/P and S/I (e.g. 'RAS', 'LAS', 'LPI'). Each letter is the
        direction towards which the corresponding voxel axis increases.
        Default is 'RAS' (nibabel, FreeSurfer and MNI convention). Use 'LAS' for
        the FSL standard orientation (as fslreorient2std on MNI-like images).

    Returns
    -------
    str or nibabel image
        Path to the saved image if ``out_image`` is given, otherwise the
        reoriented nibabel image.

    Raises
    ------
    ValueError
        If the input file does not exist, the image has fewer than three
        dimensions, or the orientation code is not valid.

    Notes
    -----
    For oblique acquisitions the voxel axes are matched to the closest
    anatomical directions; the image stays oblique because it is not resampled.

    Examples
    --------
    >>> # Reorient to RAS and save
    >>> output = reorient_image('image.nii.gz', 'image_ras.nii.gz')
    >>>
    >>> # FSL-style orientation, working with image objects
    >>> las_img = reorient_image(nib.load('image.nii.gz'), orientation='LAS')
    >>> get_orientation(las_img.affine)
    'LAS'
    """

    # Checking the orientation code
    if not isinstance(orientation, str):
        raise ValueError("The 'orientation' parameter must be a string, e.g. 'RAS'.")

    target = tuple(orientation.upper())
    axis_pairs = ({"L", "R"}, {"A", "P"}, {"S", "I"})
    if len(target) != 3 or any(
        sum(code in pair for code in target) != 1 for pair in axis_pairs
    ):
        raise ValueError(
            f"Invalid orientation code '{orientation}'. It must have three letters, "
            "one from each of the pairs L/R, A/P and S/I (e.g. 'RAS', 'LPI')."
        )

    # Loading the image
    if isinstance(in_image, (str, Path)):
        if not os.path.exists(in_image):
            raise ValueError(f"The image file does not exist: {in_image}")
        img = nib.load(str(in_image))
    elif isinstance(in_image, nib.spatialimages.SpatialImage):
        img = in_image
    else:
        raise ValueError(
            "The 'in_image' parameter must be a path or a nibabel image object."
        )

    if len(img.shape) < 3:
        raise ValueError(
            f"The image must have at least 3 dimensions, got {len(img.shape)}D."
        )

    # Transformation between the current and the target orientations
    src_ornt = nib.orientations.io_orientation(img.affine)
    dst_ornt = nib.orientations.axcodes2ornt(target)
    xfm = nib.orientations.ornt_transform(src_ornt, dst_ornt)

    if not isinstance(img, nib.Nifti1Image):
        # Other formats (e.g. MGH): no stored scaling or qform/sform to preserve
        out = img.as_reoriented(xfm)
    else:
        hdr_in = img.header

        # Work on the stored values so they are not re-quantized: the original
        # scl_slope / scl_inter are carried over unchanged (bit-exact).
        if nib.is_proxy(img.dataobj) and hasattr(img.dataobj, "get_unscaled"):
            raw = np.asanyarray(img.dataobj.get_unscaled())
            # The scaling lives in the proxy: nibabel clears it from the
            # header of a loaded image
            slope, inter = img.dataobj.slope, img.dataobj.inter
        else:
            raw = np.asanyarray(img.dataobj)
            slope, inter = None, None
        scaled = slope is not None and (slope != 1 or (inter or 0) != 0)

        # as_reoriented updates data, affine and dim_info consistently
        raw_img = img.__class__(raw, img.affine, hdr_in.copy())
        out = raw_img.as_reoriented(xfm)
        out.set_data_dtype(img.get_data_dtype())

        # Reorient the qform and the sform independently, keeping their codes.
        # A transform that is not set takes the reoriented affine with code 1,
        # so that the orientation is not lost.
        reorient_aff = nib.orientations.inv_ornt_aff(xfm, img.shape[:3])
        for getter, setter, code_key in (
            (img.get_qform, out.set_qform, "qform_code"),
            (img.get_sform, out.set_sform, "sform_code"),
        ):
            code = int(hdr_in[code_key])
            if code > 0:
                setter(getter() @ reorient_aff, code)
            else:
                setter(out.affine, 1)

        # Carry the exact original voxel sizes through the axis permutation
        zooms = np.array(hdr_in.get_zooms(), dtype=float)
        zooms[:3] = zooms[:3][np.argsort(xfm[:, 0])]
        out.header.set_zooms(zooms)

        if scaled:
            # The array holds the stored values: the scaling in the header is
            # applied when the image is read back.
            out.header.set_slope_inter(slope, inter)
            if out_image is None:
                # Round trip through memory so the returned object applies it
                out = out.__class__.from_bytes(out.to_bytes())

    if out_image is None:
        return out

    # Create the output directory if it does not exist
    out_image = str(out_image)
    out_pth = os.path.dirname(out_image)
    if out_pth and not os.path.exists(out_pth):
        Path(out_pth).mkdir(parents=True, exist_ok=True)
    elif os.path.exists(out_image) and not overwrite:
        raise FileExistsError(
            f"Output image {out_image} already exists and overwrite is set to False."
        )

    nib.save(out, out_image)

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
    overwrite: bool = False,
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
        Interpolation method: 0=NearestNeighbor, 1=Linear, 2=BSpline, 3=CosineWindowedSinc,
        4=WelchWindowedSinc, 5=HammingWindowedSinc, 6=LanczosWindowedSinc, 7=Welch.
        Default is 0.

    invert : bool, optional
        Whether to invert the transformation. Default is False.

    cont_tech : str, optional
        Container technology: 'local', 'singularity', 'docker'. Default is 'local'.

    cont_image : str, optional
        Container image specification. Default is None.

    overwrite : bool, optional
        overwrite recomputation if output exists. Default is False.

    Raises
    ------
    ValueError
        If interp_order is not one of the supported values (0-7).

    FileNotFoundError
        If the required spatial transformation file does not exist.

    Examples
    --------
    >>> # Apply transformation with nearest neighbor interpolation
    >>> apply_multi_transf(
    ...     'input.nii.gz', 'output.nii.gz', 'template.nii.gz',
    ...     'transform_prefix', interp_order=0
    ... )
    """

    interp_map = {
        0: "NearestNeighbor",
        1: "Linear",
        2: "BSpline[3]",
        3: "CosineWindowedSinc",
        4: "WelchWindowedSinc",
        5: "HammingWindowedSinc",
        6: "LanczosWindowedSinc",
        7: "Welch",
    }
    if interp_order not in interp_map:
        raise ValueError(
            f"interp_order must be one of {sorted(interp_map)}, got {interp_order}."
        )
    interp_cad = interp_map[interp_order]

    # Check if the path of out_basename exists. os.path.dirname('') happens
    # when out_image is a bare filename with no directory component, and
    # os.makedirs('') raises FileNotFoundError - resolve to an absolute path
    # first so out_path is never empty.
    out_path = os.path.dirname(os.path.abspath(out_image))
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

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

    # Check if out_image is not computed and overwrite is True
    if not os.path.isfile(out_image) or overwrite:

        if not os.path.isfile(affine_transf):
            # A bare sys.exit() here would kill the whole process/kernel and
            # can't be caught by a caller's try/except - raise instead so
            # this behaves like a normal library function.
            raise FileNotFoundError(
                f"The spatial transformation file does not exist: {affine_transf}"
            )

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
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, text=True)


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

    # Convert to homogeneous coordinates (add column of ones)
    n_points = mm_coords.shape[0]
    coords_homogeneous = np.column_stack([mm_coords, np.ones(n_points)])

    # Apply inverse affine transformation
    # (to go from world coordinates to voxel coordinates)
    affine_inv = np.linalg.inv(affine)
    coords_vox_homogeneous = coords_homogeneous @ affine_inv.T

    # Remove the homogeneous coordinate (last column)
    coords_vox = coords_vox_homogeneous[:, :3]

    return coords_vox


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############                Section 5: Methods to work with 4D Images                   ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def merge_to_4d(
    file_paths: str | Path | list[str | Path],
    output_path: str | Path | None = None,
    check_affine: bool = True,
) -> tuple[nib.Nifti1Image, dict | None]:
    """
    Merge multiple 3D or 4D NIfTI files into a single 4D NIfTI file.

    Accepts any mix of 3D (x, y, z) and 4D (x, y, z, n) volumes and
    concatenates them along the 4th dimension. Sidecar files are handled
    as follows:

    - **.json**: merged whenever *all* inputs have an associated JSON file,
      regardless of modality. Fields identical across files are kept as
      scalars; differing fields use the value from the first file.
    - **.bvec / .bval**: concatenated when *all* inputs have both files
      (DWI detection). The number of directions is cross-checked against
      the volume count of each NIfTI.

    Parameters
    ----------
    file_paths : str, Path, or list of str/Path
        Path(s) to 3D or 4D NIfTI files to be merged. All files must share
        the same spatial dimensions (x, y, z).

    output_path : str or Path, optional
        Path where the merged 4D NIfTI file should be saved. Sidecar files
        (.bvec, .bval, .json) are written to the same directory using the
        same stem when applicable. If None, nothing is written to disk.

    check_affine : bool, default True
        Whether to verify that all inputs share the same affine matrix.
        Raises ValueError on mismatch when True.

    Returns
    -------
    merged_img : nibabel.Nifti1Image
        A 4D NIfTI image where the 4th dimension is the concatenation of all
        input volumes in the order provided. Returned in both cases,
        regardless of whether output_path was given.

    metadata : dict or None
        Populated whenever any sidecar is detected; None otherwise. May
        contain:

        - ``"bvecs"`` : np.ndarray, shape (3, N) — concatenated gradient
          directions (DWI only).
        - ``"bvals"`` : np.ndarray, shape (N,) — concatenated b-values
          (DWI only).
        - ``"json"``  : dict — merged JSON sidecar (all modalities with
          JSON sidecars).

    Raises
    ------
    ValueError
        If spatial dimensions or affines are inconsistent, if only some inputs
        have sidecar files, or if bvec/bval direction counts do not match the
        volume count of the corresponding NIfTI.

    FileNotFoundError
        If any input NIfTI file cannot be found.

    IOError
        If any input file cannot be read.

    Examples
    --------
    Mix of 3D and 4D volumes (no sidecars):

    >>> merged_img, meta = merge_to_4d(['run1.nii.gz', 'run2.nii.gz'])
    >>> meta is None
    True

    fMRI runs — JSON sidecars merged, no bvec/bval:

    >>> merged_img, meta = merge_to_4d(
    ...     ['sub-01_run-1_bold.nii.gz', 'sub-01_run-2_bold.nii.gz'],
    ...     output_path='sub-01_bold.nii.gz',
    ... )
    >>> 'bvecs' in meta
    False
    >>> 'json' in meta
    True

    DWI runs — bvec, bval, and JSON all concatenated:

    >>> merged_img, meta = merge_to_4d(
    ...     ['sub-01_run-1_dwi.nii.gz', 'sub-01_run-2_dwi.nii.gz'],
    ...     output_path='sub-01_dwi.nii.gz',
    ... )
    >>> meta['bvecs'].shape   # (3, total_directions)
    >>> meta['bvals'].shape   # (total_directions,)
    """

    # ------------------------------------------------------------------ #
    #  Input normalisation                                                 #
    # ------------------------------------------------------------------ #
    if not file_paths:
        raise ValueError("file_paths cannot be empty")

    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]

    file_paths = [Path(fp) for fp in file_paths]

    if output_path is not None:
        output_path = Path(output_path)
        if not output_path.parent.exists():
            raise ValueError(f"Output directory does not exist: {output_path.parent}")

    # ------------------------------------------------------------------ #
    #  Existence checks                                                    #
    # ------------------------------------------------------------------ #
    for fp in file_paths:
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {fp}")

    # ------------------------------------------------------------------ #
    #  Sidecar detection                                                   #
    # ------------------------------------------------------------------ #
    sidecars = [imgutils.get_sidecars_files(fp) for fp in file_paths]
    has_bvec = [s["bvec"] is not None for s in sidecars]
    has_bval = [s["bval"] is not None for s in sidecars]
    has_json = [s["json"] is not None for s in sidecars]

    is_dwi = all(has_bvec) and all(has_bval)
    has_all_json = all(has_json)

    partial = (
        (any(has_bvec) and not all(has_bvec))
        or (any(has_bval) and not all(has_bval))
        or (any(has_json) and not all(has_json))
    )
    if partial:
        missing = [
            fp
            for fp, bvec, bval, js in zip(
                file_paths, has_bvec, has_bval, has_json, strict=False
            )
            if not (bvec and bval and js)
        ]
        raise ValueError(
            "Sidecar files (.bvec/.bval/.json) were found for some—but not "
            "all—input volumes. Either all inputs must have sidecars or none "
            "should. Volumes missing complete sidecars:\n"
            + "\n".join(f"  {fp}" for fp in missing)
        )

    if is_dwi:
        print(
            f"DWI sidecars detected for all {len(file_paths)} volumes. "
            "Will concatenate .bvec and .bval files."
        )
    if has_all_json:
        print(f"JSON sidecars detected for all {len(file_paths)} volumes. Will merge.")

    # ------------------------------------------------------------------ #
    #  Load NIfTI volumes — accept 3D or 4D                               #
    # ------------------------------------------------------------------ #
    print(f"Loading {len(file_paths)} NIfTI files...")

    try:
        first_img = nib.load(file_paths[0])
        first_data = first_img.get_fdata()

        if first_data.ndim not in (3, 4):
            raise ValueError(
                f"Expected 3D or 4D data, got {first_data.ndim}D in {file_paths[0]}"
            )

        reference_spatial = first_data.shape[:3]
        reference_affine = first_img.affine
        reference_header = first_img.header.copy()

        if first_data.ndim == 3:
            first_data = first_data[..., np.newaxis]

        data_arrays = [first_data]

    except Exception as e:
        raise OSError(f"Error loading first file {file_paths[0]}: {e}") from e

    for fp in file_paths[1:]:
        try:
            img = nib.load(fp)
            data = img.get_fdata()

            if data.ndim not in (3, 4):
                raise ValueError(f"Expected 3D or 4D data, got {data.ndim}D in {fp}")

            if data.shape[:3] != reference_spatial:
                raise ValueError(
                    f"Spatial shape mismatch: {fp} has shape {data.shape[:3]}, "
                    f"expected {reference_spatial}"
                )

            if check_affine and not np.allclose(
                img.affine, reference_affine, atol=1e-6
            ):
                raise ValueError(
                    f"Affine matrix mismatch in {fp}. "
                    "Set check_affine=False to ignore this check."
                )

            if data.ndim == 3:
                data = data[..., np.newaxis]

            data_arrays.append(data)

        except Exception as e:
            raise OSError(f"Error loading file {fp}: {e}") from e

    # ------------------------------------------------------------------ #
    #  Concatenate along 4th dimension                                     #
    # ------------------------------------------------------------------ #
    print("Merging volumes...")
    merged_data = np.concatenate(data_arrays, axis=3)

    new_header = reference_header.copy()
    new_header.set_data_shape(merged_data.shape)
    merged_img = nib.Nifti1Image(merged_data, reference_affine, new_header)

    print(
        f"Successfully merged {len(file_paths)} files into 4D image "
        f"with shape {merged_data.shape}"
    )

    # ------------------------------------------------------------------ #
    #  Load and concatenate sidecars                                       #
    # ------------------------------------------------------------------ #
    metadata = None

    if is_dwi or has_all_json:
        metadata = {}

    if is_dwi:
        print("Loading and concatenating DWI sidecars...")

        all_bvecs = [dwiutils.load_bvecs(s["bvec"]) for s in sidecars]
        all_bvals = [dwiutils.load_bvals(s["bval"]) for s in sidecars]

        for fp, arr, bvecs, bvals in zip(
            file_paths, data_arrays, all_bvecs, all_bvals, strict=False
        ):
            n_vols = arr.shape[3]
            if bvecs.shape[1] != n_vols:
                raise ValueError(
                    f"bvec columns ({bvecs.shape[1]}) do not match volume "
                    f"count ({n_vols}) in {fp}"
                )
            if bvals.shape[0] != n_vols:
                raise ValueError(
                    f"bval entries ({bvals.shape[0]}) do not match volume "
                    f"count ({n_vols}) in {fp}"
                )

        metadata["bvecs"] = np.concatenate(all_bvecs, axis=1)  # (3, total_N)
        metadata["bvals"] = np.concatenate(all_bvals, axis=0)  # (total_N,)

        print(
            f"DWI concatenation complete — "
            f"{metadata['bvecs'].shape[1]} total directions, "
            f"b-value range: [{metadata['bvals'].min():.0f}, {metadata['bvals'].max():.0f}]"
        )

    if has_all_json:
        print("Merging JSON sidecars...")
        metadata["json"] = imgutils.merge_json_files([s["json"] for s in sidecars])

    # ------------------------------------------------------------------ #
    #  Save outputs                                                        #
    # ------------------------------------------------------------------ #
    if output_path is not None:
        print(f"Saving merged 4D volume to: {output_path}")
        nib.save(merged_img, output_path)

        if metadata:
            stem = output_path.name
            for suffix in (".nii.gz", ".nii"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            out_dir = output_path.parent

            if is_dwi:
                bvec_out = out_dir / f"{stem}.bvec"
                bval_out = out_dir / f"{stem}.bval"
                dwiutils.save_bvecs(metadata["bvecs"], bvec_out)
                dwiutils.save_bvals(metadata["bvals"], bval_out)
                print(f"Saved DWI sidecars:\n  {bvec_out}\n  {bval_out}")

            if has_all_json:
                json_out = out_dir / f"{stem}.json"
                with open(json_out, "w") as f:
                    json.dump(metadata["json"], f, indent=2)
                print(f"Saved merged JSON sidecar:\n  {json_out}")

    # The docstring promises (merged_img: nibabel.Nifti1Image, metadata) in
    # both cases - `nib.Nifti1Image` has no `.data` attribute, so the old
    # `output_path is None` branch (`return merged_img.data, metadata`) was
    # a guaranteed AttributeError. Return the image object itself, always.
    return merged_img, metadata


#####################################################################################################
def create_spams(
    in_parcs: list[Path] | list[str],
    out_spams: str | Path,
    lut_table: str | Path | dict,
    save_color_spams: bool = False,
    rgb255: bool = True,
):
    """
    Create SPAM probability maps from multiple parcellation images.
    Generates spatial probability maps (SPAMs) from a list of parcellation
    images, using a lookup table (LUT) for region definitions and colors.

    Parameters
    ----------
    in_parcs : list of str or Path
        List of file paths to input parcellation images.

    out_spams : str or Path
        File path for the output SPAM image.

    lut_table : str, Path, or dict
        Path to LUT file (TSV or LUT format) or a dictionary with 'index',
        'name', and 'color' keys defining regions.

    save_color_spams : bool, optional
        Whether to save a color-coded SPAM image. Default is False.

    rgb255 : bool, optional
        Whether to scale RGB colors to 0-255 range. Default is True.

    Raises
    ------
    ValueError
        If input parameters are invalid.

    Examples
    --------
    >>> # Create SPAMs from parcellations with LUT file
    >>> create_spams(
    ...     in_parcs=['subj1_parc.nii.gz', 'subj2_parc.nii.gz'],
    ...     out_spams='group_spams.nii.gz
    ...     lut_table='parc_lut.tsv',
    ...     save_color_spams=True,
    ...     rgb255=True
    ... )

    """

    # Check if the in_parcs is a list
    if not isinstance(in_parcs, list):
        raise ValueError("The 'in_parcs' parameter must be a list of file paths.")

    # Unify and check if all the file exist
    in_parcs_checked = []
    for parc in in_parcs:
        if isinstance(parc, str):
            parc_path = Path(parc)
        elif isinstance(parc, Path):
            parc_path = parc
        else:
            raise ValueError("Each item in 'in_parcs' must be a string or Path object.")

        if not parc_path.exists():
            raise FileNotFoundError(f"The file {parc_path} does not exist.")

        in_parcs_checked.append(parc_path)

    # Load the image with nibabel to check
    for i in range(len(in_parcs_checked)):

        # Load the image with nibabel to check
        img = nib.load(str(in_parcs[i]))
        data = img.get_fdata()

        # Create a 4D volume with 0s with the same shape as the original and number of subjects as 4th dimension
        if i == 0:
            all_subj = np.zeros(data.shape + (len(in_parcs_checked),))

        all_subj[..., i] = data

    # Read the LUT file
    # Check if str or Path and if it exists
    if lut_table is None:
        sts_ids = np.unique(all_subj)
        lut_dict = cltcol.create_lut_dictionary(sts_ids)

    else:
        if isinstance(lut_table, dict):

            # Check if the keys are correct
            required_keys = {"index", "name", "color"}
            if not required_keys.issubset(lut_table.keys()):
                raise ValueError(
                    f"The LUT dictionary must contain the keys: {required_keys}"
                )
            lut_dict = lut_table

        else:
            if isinstance(lut_table, (str, Path)):
                lut_table = Path(lut_table)

                if not lut_table.exists():
                    lut_dict = cltcol.create_lut_dictionary(np.unique(all_subj))

                else:
                    with open(lut_table) as f:
                        first_line = f.readline().strip()

                        # TSV format has tab-separated header: index, name, color
                        if "\t" in first_line and "index" in first_line.lower():
                            lut_dict = cltparc.Parcellation.read_tsvtable(
                                in_file=str(lut_table)
                            )
                        else:
                            lut_dict = cltcol.ColorTableLoader.read_luttable(
                                str(lut_table)
                            )

    sts_ids = lut_dict["index"]
    sts_colors = lut_dict["color"]
    sts_colors = cltcol.multi_hex2rgb(sts_colors)

    if rgb255:
        sts_colors = cltcol.harmonize_colors(sts_colors, output_format="rgb")

    else:
        sts_colors = cltcol.harmonize_colors(sts_colors, output_format="rgbnorm")

    # Create the SPAM image
    spam_image = create_spams_from_volume(all_subj, sts_ids)

    # Save the SPAM image
    # Check if the output directory exists
    if isinstance(out_spams, str):
        out_spams = Path(out_spams)

    # If the directory does not exist, lunch an error
    out_dir = out_spams.parent
    if not out_dir.exists():
        raise ValueError(f"The output directory {out_dir} does not exist.")

    spam_nifti = nib.Nifti1Image(spam_image, img.affine)
    nib.save(spam_nifti, str(out_spams))

    # Save the color SPAM image if required
    if save_color_spams:
        spams_dim = spam_image.shape
        color_spam_image = np.zeros((spams_dim[0], spams_dim[1], spams_dim[2], 3))

        # Take the same name and and colored after the original basename
        spam_stem = out_spams.name
        for suffix in (".nii.gz", ".nii"):
            if spam_stem.endswith(suffix):
                spam_stem = spam_stem[: -len(suffix)]
                break
        colored_spam_name = spam_stem + "_colored.nii.gz"

        for i, _vol_index in enumerate(sts_ids):
            color = sts_colors[i]
            color_spam_image[..., 0] = (
                color_spam_image[..., 0] + spam_image[..., i] * color[0]
            )
            color_spam_image[..., 1] = (
                color_spam_image[..., 1] + spam_image[..., i] * color[1]
            )
            color_spam_image[..., 2] = (
                color_spam_image[..., 2] + spam_image[..., i] * color[2]
            )

        color_spam_nifti = nib.Nifti1Image(color_spam_image, img.affine)
        color_spam_path = os.path.join(out_dir, colored_spam_name)
        nib.save(color_spam_nifti, str(color_spam_path))


####################################################################################################
def spams2maxprob(
    spam_image: str | Path,
    prob_thresh: float = 0.05,
    vol_indexes: np.ndarray | list | str = None,
    maxp_name: str = None,
    lut_table: str | Path | dict = None,
    out_ctab: str | Path = None,
    ctab_format: str = "lut",
    overwrite: bool = False,
):
    """
    Convert SPAM probability maps to maximum probability parcellation.

    Transforms 4D spatial probability maps into discrete 3D parcellation by
    selecting the most probable label at each voxel, with optional thresholding
    and volume selection. Optionally, a color table describing the SPAM volumes
    can be supplied to write a matching color table for the maximum probability
    image.

    Parameters
    ----------
    spam_image : str or Path
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

    lut_table : str, Path or dict, optional
        Color table describing the SPAM volumes, in any format accepted by
        ``ColorTableLoader`` (FreeSurfer LUT, TSV, or a dictionary with at
        least an 'index' key). It must contain exactly one entry per SPAM
        volume, in volume order: the n-th entry gives the name, color and
        opacity of the n-th volume. The indices stored in the table are
        ignored; the volume position is used instead. Default is None.

    out_ctab : str or Path, optional
        Output path for the color table of the maximum probability image.
        Only used if ``lut_table`` is given. If None and ``maxp_name`` is
        given, the table is written next to the image, with the same base
        name and a '.tsv' (for 'tsv' and 'nilearn') or '.lut' extension. If
        both ``out_ctab`` and ``maxp_name`` are None, no table is written.
        Default is None.

    ctab_format : str, optional
        Format of the output color table: 'lut', 'tsv', 'fsl' or 'nilearn'.
        Default is 'lut'.

    overwrite : bool, optional
        If True, replace the maximum probability image and its color table if
        they already exist. If False, an existing output raises a
        ``FileExistsError`` before any volume is read. Default is False.

    Returns
    -------
    str or np.ndarray
        Output file path if maxp_name provided, otherwise numpy array
        with maximum probability labels.

    Raises
    ------
    FileNotFoundError
        If the SPAM image does not exist.

    FileExistsError
        If ``maxp_name`` or the output color table already exists and
        ``overwrite`` is False.

    ValueError
        If the SPAM image is not 4D, if the number of entries in ``lut_table``
        differs from the number of SPAM volumes, or if ``ctab_format`` is not
        supported.

    IndexError
        If ``vol_indexes`` contains indices outside the SPAM volume range.

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

    The label of volume ``i`` (0-based) in the maximum probability image is
    ``i + 1``, so the output color table assigns index ``i + 1`` to the
    ``i``-th entry of ``lut_table``. When ``vol_indexes`` is given, only the
    selected volumes are written to the output color table.

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
    >>>
    >>> # Name and color the volumes, writing 'AAL_discrete.tsv' alongside
    >>> maxprob_file = spams2maxprob(
    ...     'AAL_SPAM.nii.gz',
    ...     maxp_name='AAL_discrete.nii.gz',
    ...     lut_table='AAL_SPAM.lut',
    ...     ctab_format='tsv'
    ... )
    """

    # Check if the spam_image exists
    if isinstance(spam_image, Path):
        spam_image = str(spam_image)

    if not os.path.exists(spam_image):
        raise FileNotFoundError(f"SPAM image file not found: {spam_image}")

    # Load the SPAM image lazily. The 4D array is never materialised: the volumes
    # are streamed one at a time while a running (max probability, label) pair is
    # kept, so peak memory is a few 3D volumes instead of the whole 4D array in
    # float64. keep_file_open=True matters for .nii.gz files: reading the volumes
    # in ascending order then only seeks forward in the gzip stream, so the file
    # is decompressed once rather than once per volume.
    spam_img = nib.load(spam_image, keep_file_open=True)
    affine = spam_img.affine

    if len(spam_img.shape) != 4:
        raise ValueError(
            f"The SPAM image must be 4D, got shape {spam_img.shape}: {spam_image}"
        )
    n_vols = spam_img.shape[3]

    if vol_indexes is not None:
        # nonzeros=False: 0 is the first volume here, not a background label
        vol_indexes = cltmisc.build_indices(vol_indexes, nonzeros=False)
        vols2use = sorted({int(v) for v in vol_indexes})
        if vols2use and (vols2use[0] < 0 or vols2use[-1] >= n_vols):
            raise IndexError(
                f"vol_indexes must be in [0, {n_vols - 1}] for {spam_image}, "
                f"got {vols2use}"
            )
    else:
        vols2use = range(n_vols)

    # Load and validate the color table before the (expensive) volume loop
    ctab = None
    if lut_table is not None:
        supported_formats = ("lut", "tsv", "fsl", "nilearn")
        ctab_format = str(ctab_format).lower()
        if ctab_format not in supported_formats:
            raise ValueError(
                f"Unsupported ctab_format: {ctab_format}. "
                f"Supported formats are {supported_formats}."
            )

        ctab = cltcol.ColorTableLoader(lut_table)
        if len(ctab.name) != n_vols:
            raise ValueError(
                f"The color table has {len(ctab.name)} entries but the SPAM image "
                f"has {n_vols} volumes: {spam_image}"
            )

        if out_ctab is None and maxp_name is not None:
            maxp_path = str(maxp_name)
            for ext in (".nii.gz", ".nii"):
                if maxp_path.endswith(ext):
                    maxp_path = maxp_path[: -len(ext)]
                    break
            ctab_ext = ".tsv" if ctab_format in ("tsv", "nilearn") else ".lut"
            out_ctab = maxp_path + ctab_ext

    # Refuse to replace existing outputs before the (expensive) volume loop
    if not overwrite:
        outputs = [maxp_name]
        if ctab is not None:
            outputs.append(out_ctab)
        for out_file in outputs:
            if out_file is not None and os.path.exists(out_file):
                raise FileExistsError(
                    f"Output file {out_file} already exists. "
                    "Use 'overwrite=True' to replace it."
                )

    max_prob = np.zeros(spam_img.shape[:3], dtype=np.float32)
    maxprob_thl = np.zeros(spam_img.shape[:3], dtype=np.int32)

    for vol in vols2use:
        # np.array (not asarray): the proxy may hand back a read-only buffer
        prob = np.array(spam_img.dataobj[..., vol], dtype=np.float32)

        # Apply probability threshold and clamp values between 0 and 1
        prob[prob < prob_thresh] = 0
        np.minimum(prob, 1, out=prob)

        # Strict '>' keeps the first maximum on ties, matching np.argmax. Voxels
        # where every selected volume is 0 are never updated and stay background.
        winners = prob > max_prob
        max_prob[winners] = prob[winners]
        maxprob_thl[winners] = vol + 1
        del prob, winners

    del max_prob

    if ctab is not None and out_ctab is not None:
        # The table indices are ignored: volume i gets the label i + 1, which is
        # the value it takes in the maximum probability image
        vols2use = list(vols2use)
        colors = cltcol.harmonize_colors(ctab.color, output_format="hex")
        maxp_ctab = cltcol.ColorTableLoader(
            {
                "index": [vol + 1 for vol in vols2use],
                "name": [ctab.name[vol] for vol in vols2use],
                "color": [colors[vol] for vol in vols2use],
                "opacity": [ctab.opacity[vol] for vol in vols2use],
                "headerlines": [],
            }
        )
        maxp_ctab.export(out_ctab, out_format=ctab_format, overwrite=True)

    if maxp_name is not None:
        # Save the image
        imgcoll = nib.Nifti1Image(maxprob_thl.astype("int16"), affine)
        nib.save(imgcoll, maxp_name)
    else:
        maxp_name = maxprob_thl

    return maxp_name


#####################################################################################################
def simulate_image(
    reference: "str | Path | nib.Nifti1Image | cltparc.Parcellation | tuple | list | np.ndarray",
    simulated_image: str = None,
    affine: np.ndarray | None = None,
    n_volumes: int | None = None,
    distribution: str = "normal",
    random_seed: int | None = None,
    **dist_params,
) -> nib.Nifti1Image:
    """
    Generate a simulated image with random values at non-zero voxel positions.

    This function creates a new NIfTI image where non-zero voxels from the
    reference are filled with random values following a specified statistical
    distribution. The output preserves the spatial dimensions, affine
    transformation, and header information from the reference.

    This is useful for simulating functional or structural images based on a mask,
    parcellation, anatomical image, or bare geometry, allowing for controlled
    random value generation in specific regions of interest.

    Parameters
    ----------
    reference : str, Path, nibabel.Nifti1Image, Parcellation, tuple, list, or np.ndarray
        Source used to define the volume's mask, shape, and (unless overridden by
        `affine`) affine. Can be:

        - str or Path: file path to a NIfTI image, loaded and used as a mask.
        - nibabel.Nifti1Image: used directly as a mask.
        - Parcellation: its `.data` is used as the mask and its `.affine` as the
          affine (unless `affine` is explicitly given).
        - tuple, list, or 1-D np.ndarray of length 3 or 4: a volume *shape*
          rather than real data - (dimx, dimy, dimz) or (dimx, dimy, dimz,
          n_volumes). Since there is no real anatomy to mask, every voxel is
          treated as "in mask". A length-4 shape implies `n_volumes` from its
          last value unless `n_volumes` is explicitly given (see below).

        If the reference itself is 4D (a real 4D file, Nifti1Image, or
        Parcellation whose `.data` happens to be 4D), the mask used is
        spatial/3D: a voxel counts as "in mask" if it is non-zero in ANY of
        the reference's volumes. This 3D mask is then applied uniformly to
        every volume of the *output*, whose number of volumes is controlled
        solely by `n_volumes` (see below) - independent of how many volumes
        the reference itself had.

    simulated_image : str
        Output file path where the simulated image will be saved. Must include
        the .nii or .nii.gz extension. If None, the function will generate a
        default name based on the reference.

    affine : np.ndarray, optional
        4x4 affine transformation matrix. If given, it always takes priority
        over any affine implied by `reference` (a loaded file's affine, a
        Parcellation's affine, or the default centered affine built for a
        shape reference). If None and `reference` is a shape, a diagonal
        affine centered on the volume is built automatically.

    n_volumes : int, optional
        Number of volumes in the output image:

        - If n_volumes == 1: creates a 3D image, always collapsed to pure
          spatial dimensions even if the reference itself was 4D.
        - If n_volumes > 1: creates a 4D image with n_volumes timepoints.
        - If None (default): uses 1, unless `reference` is a length-4 shape,
          in which case its last value is used instead. If both a length-4
          shape and an explicit n_volumes are given and they disagree, the
          explicit n_volumes wins and a warning is raised.

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
        The simulated image object with the same spatial (3D) properties as
        the reference, and n_volumes volumes.

    Raises
    ------
    FileNotFoundError
        If reference is a path that does not exist.

    ValueError
        If reference is not a valid type/shape, distribution is unsupported,
        n_volumes is invalid, or output directory doesn't exist.

    RuntimeError
        If no non-zero voxels are found in the reference.

    Examples
    --------
    >>> # Create 3D simulation with normal distribution, from a real file
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

    >>> # From a Parcellation object, using its own affine
    >>> sim_img = simulate_image(parc, 'output.nii.gz', n_volumes=5)

    >>> # From a shape only - no real anatomy, every voxel gets random values
    >>> sim_img = simulate_image((64, 64, 40), 'output.nii.gz', n_volumes=20)

    >>> # A length-4 shape implies n_volumes from its last value
    >>> sim_img = simulate_image((64, 64, 40, 20), 'output.nii.gz')

    >>> # Shape with a custom affine
    >>> sim_img = simulate_image(
    ...     (64, 64, 40), 'output.nii.gz', affine=my_affine, n_volumes=1
    ... )

    >>> # A real 4D reference, but requesting a single 3D output volume -
    >>> # the mask is "non-zero in any of the reference's volumes"
    >>> sim_img = simulate_image('fmri_4d.nii.gz', 'output_3d.nii.gz', n_volumes=1)

    Notes
    -----
    - Only voxels with non-zero values in the reference will contain random values
    - All other voxels remain zero in the output
    - When reference is a real image/Parcellation, the function preserves its
      affine transformation and header, unless `affine` is explicitly given
    - When reference is a shape, there is no real header/anatomy to preserve;
      a default header is generated and every voxel is treated as "in mask"
    - The output is always shaped from reference's spatial (x, y, z) dimensions
      plus n_volumes - a 4D reference does not force a 4D output, and a 3D
      reference does not prevent a multi-volume output
    - Output file extension should be .nii or .nii.gz
    """

    # Distribution validation up front (independent of reference type)
    if distribution not in ["normal", "uniform", "exponential"]:
        raise ValueError(
            f"Unsupported distribution '{distribution}'. "
            "Supported: 'normal', 'uniform', 'exponential'"
        )

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # --- Resolve reference into a working nib.Nifti1Image, and (for a
    #     shape reference) any n_volumes implied by a 4th shape element ---
    implied_n_volumes = None

    if isinstance(reference, (str, Path)):
        reference = str(reference)
        if not os.path.exists(reference):
            raise FileNotFoundError(f"Reference image file not found: {reference}")
        try:
            input_img = nib.load(reference)
        except Exception as e:
            raise ValueError(f"Failed to load reference image: {e}") from e

    elif isinstance(reference, nib.Nifti1Image):
        input_img = reference

    elif isinstance(reference, cltparc.Parcellation):
        input_img = nib.Nifti1Image(reference.data.astype(np.float32), reference.affine)

    elif isinstance(reference, (tuple, list)) or (
        isinstance(reference, np.ndarray) and reference.ndim == 1
    ):
        shape = tuple(int(s) for s in reference)
        if len(shape) not in (3, 4):
            raise ValueError(
                f"When given as a shape, reference must have length 3 "
                f"(dimx, dimy, dimz) or 4 (dimx, dimy, dimz, n_volumes), "
                f"got length {len(shape)}: {shape}."
            )
        if not all(s > 0 for s in shape):
            raise ValueError(
                f"All shape values must be positive integers, got {shape}."
            )

        spatial_shape_in = shape[:3]
        if len(shape) == 4:
            implied_n_volumes = shape[3]

        # No real anatomy to derive a mask from - every voxel is "in mask".
        default_affine = np.eye(4)
        center = np.array(spatial_shape_in) // 2
        default_affine[:3, 3] = -center

        input_img = nib.Nifti1Image(
            np.ones(spatial_shape_in, dtype=np.float32),
            affine if affine is not None else default_affine,
        )

    else:
        raise ValueError(
            "reference must be a file path (str/Path), nibabel.Nifti1Image, "
            "Parcellation, or a shape given as a tuple/list/np.ndarray of "
            f"length 3 or 4 (got {type(reference)})."
        )

    # --- Explicit affine always wins over whatever reference implied ---
    if affine is not None and not np.array_equal(input_img.affine, affine):
        input_img = nib.Nifti1Image(input_img.get_fdata(), affine, input_img.header)

    # --- Resolve n_volumes: explicit > shape-implied > default (1) ---
    if n_volumes is None:
        n_volumes = implied_n_volumes if implied_n_volumes is not None else 1
    elif implied_n_volumes is not None and n_volumes != implied_n_volumes:
        warnings.warn(
            f"reference's shape implies n_volumes={implied_n_volumes}, but "
            f"n_volumes={n_volumes} was explicitly given. Using the explicit "
            f"value.",
            stacklevel=2,
        )

    if not isinstance(n_volumes, int) or n_volumes < 1:
        raise ValueError("n_volumes must be a positive integer")

    # Validate simulated_image path. If None, create a temporary filename
    if simulated_image is None:
        simulated_image = cltmisc.create_temporary_filename()

    # Validate output path
    output_dir = os.path.dirname(os.path.abspath(simulated_image))
    if output_dir and not os.path.exists(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")

    if not simulated_image.endswith((".nii", ".nii.gz")):
        warnings.warn(
            "Output filename should have .nii or .nii.gz extension",
            UserWarning,
            stacklevel=2,
        )

    # Extract image properties
    input_data = input_img.get_fdata()
    affine = input_img.affine.copy()
    header = input_img.header.copy()
    original_shape = input_data.shape
    spatial_shape = original_shape[:3] if len(original_shape) >= 3 else original_shape

    # Create mask for non-zero voxels. The mask must always be spatial/3D so
    # it can be applied uniformly to every generated output volume below -
    # if the reference itself is 4D, a voxel counts as "in mask" if it is
    # non-zero in ANY of the reference's volumes. (Previously, a 4D
    # reference's mask stayed 4D, which broke n_volumes>1 outputs via a
    # shape-mismatched boolean index, and n_volumes==1 outputs silently
    # stayed 4D instead of collapsing to pure spatial dimensions.)
    if input_data.ndim == 4:
        mask = np.any(input_data != 0, axis=3)
    else:
        mask = input_data != 0
    n_nonzero_voxels = np.sum(mask)

    if n_nonzero_voxels == 0:
        raise RuntimeError("No non-zero voxels found in reference image")

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
            raise ValueError(f"Error generating random values: {e}") from e

    # Create simulated data array - always built from spatial_shape (3D), so
    # the output's dimensionality is governed purely by n_volumes rather
    # than by whatever dimensionality the reference itself happened to have.
    if n_volumes == 1:
        # 3D output
        simulated_data = np.zeros(spatial_shape, dtype=np.float32)
        simulated_data[mask] = _generate_random_values(n_nonzero_voxels)
        output_shape = spatial_shape

    else:
        # 4D output
        output_shape = spatial_shape + (n_volumes,)
        simulated_data = np.zeros(output_shape, dtype=np.float32)

        # Fill each volume with independent random values
        for volume_idx in range(n_volumes):
            simulated_data[..., volume_idx][mask] = _generate_random_values(
                n_nonzero_voxels
            )

    # Update header for correct dimensionality. Note: nib.Nifti1Image below
    # recomputes the header's dim fields from the array actually passed to
    # it, so this is mostly a defensive no-op - kept for header metadata
    # consistency in case any downstream code reads header_copy directly.
    header_copy = header.copy()
    key_value = next(
        k for k in ["dim", "dims"] if k in header_copy.structarr.dtype.names
    )

    if n_volumes == 1:
        header_copy.structarr[key_value][4] = 1
    else:
        header_copy.structarr[key_value][3] = n_volumes

    # Create output image
    try:
        simulated_img = nib.Nifti1Image(simulated_data, affine, header_copy)
        nib.save(simulated_img, simulated_image)

    except Exception as e:
        raise RuntimeError(f"Failed to create or save simulated image: {e}") from e

    # Log summary information
    print("Successfully created simulated image:")
    print(f"  Reference shape: {original_shape}")
    print(f"  Output shape: {output_shape}")
    print(f"  Non-zero voxels: {n_nonzero_voxels:,}")
    print(f"  Distribution: {distribution}")
    print(f"  Saved to: {simulated_image}")

    return simulated_img


#####################################################################################################
def delete_volumes_from_4D_images(
    in_image: str | Path,
    out_image: str | Path,
    vols_to_delete: list[int | tuple | list | str | np.ndarray] = None,
    overwrite: bool = False,
) -> tuple[str, list[int]]:
    """
    Remove specific volumes from a 4D neuroimaging file.

    This function removes specified volumes from 4D NIfTI images (such as fMRI time series,
    DTI volumes, or other 4D datasets) and saves the result to a new file. It supports
    flexible volume specification including individual indices, ranges, and complex
    string-based patterns.

    Parameters
    ----------
    in_image : str | Path
        Path to the input 4D NIfTI image file (.nii or .nii.gz).
        The file must exist and be a valid 4D image.

    out_image : str | Path
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

    if isinstance(in_image, Path):
        in_image = str(in_image)

    if isinstance(out_image, Path):
        out_image = str(out_image)

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

        # Apply Gaussian smoothing. The closing above returns a boolean array and
        # gaussian_filter keeps the input dtype, so it must be cast to float first.
        tmp_volume_array = gaussian_filter(volume_array.astype(np.float32), sigma=sigma)
        # Re-threshold after smoothing
        tmp_volume_array = (tmp_volume_array > 0.5).astype(np.float32)

        if tmp_volume_array.max() == 0:
            tmp_volume_array = volume_array.astype(np.float32)
    else:
        tmp_volume_array = volume_array.astype(np.float32)

    # Check if the code exists in the data
    # Extract surface using marching cubes
    vertices, faces, normals, values = measure.marching_cubes(
        tmp_volume_array, level=0.5, gradient_direction="ascent"
    )
    if len(faces) == 0:
        raise ValueError(
            "No surface extracted for value. The volume may not contain sufficient data."
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

    mesh.point_data["default"] = (
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

        - Centroid coordinates as a numpy array of shape (3,) in voxel space
          (NaN if the volume is empty). Use vox2mm to convert them to mm.
        - Number of non-zero voxels in the original volume (before closing
          and smoothing) as an integer.

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
    4. Computes the centroid of the processed region.
    5. Counts the number of voxels in the original (unprocessed) region.
    6. Returns the centroid coordinates and voxel count as a tuple.

    Examples
    --------
    >>> # Basic centroid extraction
    >>> centroid, voxel_count = extract_centroid_from_volume(binary_volume)
    >>> print(f"Centroid: {centroid}, Voxel Count: {voxel_count}")
    >>>
    >>> # With Gaussian smoothing and morphological closing
    >>> centroid, voxel_count = extract_centroid_from_volume(
    ...     binary_volume, gaussian_smooth=True, sigma=1.5, closing_iterations=2
    ... )
    """

    # Binary mask for the specified value
    if not isinstance(volume_array, np.ndarray):
        raise TypeError("The volume_array must be a numpy ndarray.")

    if volume_array.ndim != 3:
        raise ValueError("The volume_array must be a 3D numpy ndarray.")

    # Everything that is different from 0 is set to 1
    volume_array = (volume_array != 0).astype(np.float32)

    # The voxel count is taken from the original region: closing and smoothing
    # can spill into neighboring voxels (e.g. other regions of a parcellation)
    voxel_count = int(np.count_nonzero(volume_array))

    if closing_iterations > 0:
        volume_array = quick_morphology(
            volume_array, "closing", iterations=closing_iterations
        )

    # Apply Gaussian smoothing to reduce noise and fill small gaps
    if gaussian_smooth:

        # Apply Gaussian smoothing. The closing above returns a boolean array and
        # gaussian_filter keeps the input dtype, so it must be cast to float first.
        tmp_volume_array = gaussian_filter(volume_array.astype(np.float32), sigma=sigma)
        # Re-threshold after smoothing
        tmp_volume_array = (tmp_volume_array > 0.5).astype(np.float32)

        if tmp_volume_array.max() == 0:
            tmp_volume_array = volume_array.astype(np.float32)
    else:
        tmp_volume_array = volume_array.astype(np.float32)

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

        return (
            np.array([centroid_x, centroid_y, centroid_z], dtype=np.float32),
            voxel_count,
        )


####################################################################################################
def create_spams_from_volume(
    indiv_parc: np.ndarray, sts_ids: list[int] | np.ndarray
) -> np.ndarray:
    """
    Create SPAMs (Spatial Probability Maps) from individual parcellation volumes.

    Parameters
    ----------
    indiv_parc : numpy.ndarray
        4D array with dimensions (X, Y, Z, N) where N is the number of subjects.
        Each voxel contains integer labels representing different structures.

    sts_ids : list or numpy.ndarray
        List or array of integer structure IDs for which to create SPAMs.

    Returns
    -------
    numpy.ndarray
        4D array with dimensions (X, Y, Z, M) where M is the number of structure IDs.
        Each voxel contains the proportion of subjects that have the corresponding
        structure ID at that voxel.

    Raises
    ------
    ValueError
        If indiv_parc is not a 4D numpy array.
    ValueError
        If sts_ids is not a list or numpy array of integers.

    Notes
    -----
    - The function computes the proportion of subjects that have each specified
        structure ID at each voxel.
    - The output SPAMs can be used for group-level analyses in neuroimaging studies.

    Examples
    --------
    >>> # Example usage
    >>> indiv_parc = np.random.randint(0, 5, size=(64, 64, 64, 10))  # 10 subjects with labels 0-4
    >>> sts_ids = [1, 2, 3]
    >>> spams = create_spams_from_volume(indiv_parc, sts_ids)
    >>> print(spams.shape)  # Output shape will be (64, 64, 64, 3)
    """

    # Validate inputs
    if not isinstance(indiv_parc, np.ndarray) or indiv_parc.ndim != 4:
        raise ValueError("indiv_parc must be a 4D numpy array (X, Y, Z, N).")

    # np.int64/np.int32 (e.g. from np.unique(...), a common source of sts_ids)
    # are NOT instances of Python's built-in int, so isinstance(id, int)
    # alone would wrongly reject perfectly valid numpy-integer input.
    if not isinstance(sts_ids, (list, np.ndarray)) or not all(
        isinstance(id, (int, np.integer)) for id in sts_ids
    ):
        raise ValueError("sts_ids must be a list or numpy array of integers.")

    # Creating the SPAMs
    # Get the dimensions of the data
    data_shape = indiv_parc.shape
    spam_image = np.zeros(data_shape[0:3] + (len(sts_ids),), dtype=np.float32)
    for cont, sts_id in enumerate(sts_ids):
        tmp_sts_img = indiv_parc == sts_id

        # Convert to logical and sum across subjects
        tmp_sts_img = tmp_sts_img.astype(np.int16)
        sum_img = np.sum(tmp_sts_img, axis=3)

        # Divide by number of subjects to get proportion
        spam_image[:, :, :, cont] = sum_img / data_shape[3]

    return spam_image


#####################################################################################################
def spams2maxprob_from_volume(
    spam_vol: np.ndarray,
    prob_thresh: float = 0.0,
    vol_indexes: list[int] | np.ndarray = None,
) -> np.ndarray:
    """
    Convert SPAMs (Spatial Probability Maps) to a maximum probability parcellation volume.

    Parameters
    ----------
    spam_vol : numpy.ndarray
        4D array with dimensions (X, Y, Z, M) where M is the number of structures.
        Each voxel contains the probability of belonging to each structure.

    prob_thresh : float, optional
        Probability threshold to apply before determining maximum probability.
        Voxels with probabilities below this threshold will be set to zero. Default is 0.0.

    vol_indexes : list or numpy.ndarray, optional
        List or array of integer structure indexes to consider for maximum probability.
        If provided, only these structures will be considered. Default is None.

    Returns
    -------
    numpy.ndarray
        3D array with dimensions (X, Y, Z) where each voxel contains the index of the structure
        with the maximum probability, or 0 if below the threshold.

    Raises
    ------
    ValueError
        If spam_vol is not a 4D numpy array.

    Notes
    -----
    - The function applies a probability threshold to the SPAMs and then determines
        the structure with the maximum probability at each voxel.
    - Voxels with probabilities below the threshold are assigned a value of 0 in the output
    - If vol_indexes is provided, only those structures are considered for maximum probability.

    Examples
    --------
    >>> # Example usage
    >>> spam_vol = np.random.rand(64, 64, 64, 5)  # SPAMs for 5 structures
    >>> maxprob_vol = spams2maxprob_from_volume(spam_vol, prob_thresh=0.2)
    >>> print(maxprob_vol.shape)  # Output shape will be (64, 64, 64)
    """

    # Validate inputs
    if not isinstance(spam_vol, np.ndarray) or spam_vol.ndim != 4:
        raise ValueError("spam_vol must be a 4D numpy array (X, Y, Z, M).")

    # Work on a copy so the caller's SPAMs are not modified
    spam_vol = spam_vol.copy()
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
        diff_array = np.array(list(diff_elements), dtype=int)
        spam_vol[:, :, :, diff_array] = 0
        # array_data = np.delete(spam_vol, diff_array, 3)

    ind = np.where(np.sum(spam_vol, axis=3) == 0)
    maxprob_thl = spam_vol.argmax(axis=3) + 1
    maxprob_thl[ind] = 0

    return maxprob_thl


####################################################################################################
def compute_statistics_at_nonzero_voxels(
    mask_array: np.ndarray, data_array: np.ndarray, metric: str = "mean"
) -> np.ndarray:
    """
    Compute the value of certain statistic from data_array at positions where mask_array is non-zero.

    Parameters
    ----------
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
    -------
    numpy.ndarray or float

        - If data_array is 3D: returns a single float value
        - If data_array is 4D: returns a 1D array of length T

    Raises
    ------
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
    --------
    >>> # Example with 3D data_array
    >>> mask = np.array([[[0, 1], [0, 0]],
    ...                  [[1, 0], [0, 1]]])
    >>> data = np.array([[[1, 2], [3, 4]],
    ...                  [[5, 6], [7, 8]]])
    >>> mean_value = compute_statistics_at_nonzero_voxels(mask, data)
    >>> print(mean_value)  # Output: 5.0 (mean of 2, 5, 8)
    >>>
    >>> # Example with 4D data_array
    >>> mask_4d = np.array([[[0, 1], [0, 0]],
    ...                     [[1, 0], [0, 1]]])
    >>> data_4d = np.array([[[[1, 2], [3, 4]],
    ...                      [[5, 6], [7, 8]]],
    ...                     [[[9, 10], [11, 12]],
    ...                      [[13, 14], [15, 16]]]])
    >>> mean_values_4d = compute_statistics_at_nonzero_voxels(mask_4d, data_4d)
    >>> print(mean_values_4d)  # Output: [ 9. 10.] (mean of [3, 9, 15] and [4, 10, 16])

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
def interpolate(
    scalar_data: np.ndarray, vertices_vox: np.ndarray, interp_method: str = "linear"
) -> np.ndarray:
    """
    Interpolate 3D or 4D scalar data at specified voxel coordinates using regular grid interpolation.
    Parameters
    ----------
    scalar_data : np.ndarray
        3D or 4D array of scalar values to interpolate from.

    vertices_vox : np.ndarray
        Nx3 array of voxel coordinates where interpolation is desired.

    interp_method : str, optional
        Interpolation method: 'linear', 'nearest', or 'slinear'. Default is 'linear'.

    Returns
    -------
    np.ndarray
        Interpolated scalar values at the specified voxel coordinates.

    Raises
    ------
    ValueError
        If scalar_data is not 3D or 4D, or if vertices_vox is not Nx3.
        If interp_method is not one of the supported methods.

    Notes
    -----
    - Uses scipy's RegularGridInterpolator for interpolation.
    - Supports both 3D and 4D scalar data.
    - Interpolates each volume separately if scalar_data is 4D.
    - Out-of-bounds coordinates are filled with 0.

    """

    # Validate inputs
    if scalar_data.ndim not in [3, 4]:
        raise ValueError("scalar_data must be a 3D or 4D numpy array.")

    if vertices_vox.ndim != 2 or vertices_vox.shape[1] != 3:
        raise ValueError("vertices_vox must be a Nx3 numpy array.")

    if interp_method not in ["linear", "nearest", "slinear"]:
        raise ValueError("interp_method must be 'linear', 'nearest', or 'slinear'.")

    # Define grid points based on scalar_data shape    # Creating interpolation function
    x = np.arange(scalar_data.shape[0])
    y = np.arange(scalar_data.shape[1])
    z = np.arange(scalar_data.shape[2])

    # Interpolate scalar data at specified voxel coordinates
    if scalar_data.ndim == 3:
        my_interpolating_scalmap = RegularGridInterpolator(
            (x, y, z),
            scalar_data,  # Removed .data
            method=interp_method,
            bounds_error=False,
            fill_value=0,
        )

        interpolated_data = my_interpolating_scalmap(vertices_vox)

    elif scalar_data.ndim == 4:
        # If 4D, interpolate each volume separately and stack results
        interpolated_data = np.zeros((vertices_vox.shape[0], scalar_data.shape[3]))
        for t in range(scalar_data.shape[3]):
            my_interpolating_scalmap = RegularGridInterpolator(
                (x, y, z),
                scalar_data[:, :, :, t],  # Removed .data
                method=interp_method,
                bounds_error=False,
                fill_value=0,
            )
            interpolated_data[:, t] = my_interpolating_scalmap(vertices_vox)

    return interpolated_data


#####################################################################################################
def region_growing(
    iparc: np.ndarray,
    mask: np.ndarray | np.bool_,
    neighborhood: str = "26",
    min_neighbors: int = 1,
) -> np.ndarray:
    """
    Fill gaps in parcellation using region growing algorithm.

    Labels the unlabeled voxels of the mask with the labels of the
    parcellation, growing the labels from the labeled voxels outwards, one
    layer of voxels at a time, until every voxel of the mask that can be
    reached from a labeled voxel is labeled.

    Parameters
    ----------
    iparc : np.ndarray
        3D parcellation array with labeled (>0) and unlabeled (0) voxels.

    mask : np.ndarray or bool array
        3D mask defining the region to label. Any non-zero value is part of
        the mask, so boolean, integer and float arrays (such as the output of
        ``nibabel``'s ``get_fdata``) can be given. Must have the shape of
        ``iparc``.

    neighborhood : str, optional
        Neighborhood connectivity: '6', '12', '18' or '26'. It defines the
        neighbors that are counted and that vote for the label.
        Default is '26'.

    min_neighbors : int, optional
        Minimum number of labeled neighbors an unlabeled voxel needs to be
        labeled in an iteration. With the default of 1, every voxel of the
        mask that is connected to a labeled voxel is labeled. Higher values
        make the growing more conservative (thin spurs poking into a
        neighboring region need more support), but voxels that never reach
        the threshold are left unlabeled (0). Must be between 1 and the size
        of the neighborhood. Default is 1.

    Returns
    -------
    np.ndarray
        Parcellation with the same shape and type as ``iparc``, with the gaps
        of the mask filled and every voxel outside the mask set to 0.

    Raises
    ------
    ValueError
        If ``iparc`` is not 3D, if ``mask`` does not have its shape, if
        ``neighborhood`` is not supported or if ``min_neighbors`` is out of
        range.

    Warns
    -----
    UserWarning
        If some voxels of the mask could not be labeled: they are not
        connected to any labeled voxel or never reach ``min_neighbors``.

    Notes
    -----
    The algorithm works iteratively:

    1. Finds the front: the unlabeled voxels of the mask with at least
       `min_neighbors` labeled neighbors.
    2. Gives each voxel of the front the most frequent label among its
       labeled neighbors. Ties go to the label of the closest neighbor
       (face, then edge, then corner) and then to the smallest label.
    3. Labels the whole front at once, so the labels grow one layer per
       iteration at the same speed in every direction, and repeats until
       the front is empty.

    Labeled voxels outside the mask also act as seeds for the voxels of the
    mask next to them. The input parcellation is not modified.

    Particularly useful for:

    - Filling gaps in atlas-based parcellations
    - Completing partial segmentations
    - Correcting registration artifacts

    Examples
    --------
    >>> # Label every voxel of a brain mask from a partial parcellation
    >>> filled_parc = region_growing(parcellation_array, brain_mask)
    >>> print(f"Filled {np.sum(filled_parc > 0) - np.sum(parcellation_array > 0)} voxels")
    >>>
    >>> # Mask read with nibabel (float values) and conservative growing
    >>> mask = nib.load('mask.nii.gz').get_fdata()
    >>> conservative_fill = region_growing(
    ...     incomplete_labels,
    ...     mask,
    ...     neighborhood='6',
    ...     min_neighbors=2,
    ... )
    """

    iparc = np.asarray(iparc)
    if iparc.ndim != 3:
        raise ValueError(f"iparc must be a 3D array, got shape {iparc.shape}")

    # Any non-zero value is part of the mask (get_fdata returns floats)
    mask = np.asarray(mask) != 0
    if mask.shape != iparc.shape:
        raise ValueError(
            f"The mask shape {mask.shape} does not match the parcellation shape "
            f"{iparc.shape}"
        )

    # Offsets of the neighbors and their distances to the center voxel
    offsets = get_vox_neighbors(
        coord=np.zeros(3, dtype=int), neighborhood=neighborhood, dims="3"
    )
    n_neigh = len(offsets)
    if not 1 <= min_neighbors <= n_neigh:
        raise ValueError(
            f"min_neighbors must be between 1 and {n_neigh} for a "
            f"{neighborhood}-neighborhood, got {min_neighbors}"
        )
    off_dist = np.linalg.norm(offsets, axis=1)

    kernel = np.zeros((3, 3, 3), dtype=np.uint8)
    kernel[tuple((offsets + 1).T)] = 1
    # The kernel is used with correlate semantics; flip it for convolve so
    # asymmetric neighborhoods ('12') count the right voxels
    kernel = kernel[::-1, ::-1, ::-1]

    # Work on a copy so the caller's parcellation is not modified
    out = iparc.copy()

    if not mask.any():
        return np.zeros_like(out)

    # Restrict the work to the bounding box of the mask plus one voxel, the
    # only labels that can reach it
    nz = np.argwhere(mask)
    lo = np.maximum(nz.min(axis=0) - 1, 0)
    hi = np.minimum(nz.max(axis=0) + 2, mask.shape)
    del nz
    box = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
    sub = out[box]  # view: labels written here go to out
    sub_mask = mask[box]
    sub_shape = np.array(sub.shape)

    labeled = sub > 0

    while True:
        unlabeled = sub_mask & ~labeled
        if not unlabeled.any():
            break

        # Number of labeled neighbors of every voxel
        n_labeled = convolve(labeled.astype(np.uint8), kernel, mode="constant", cval=0)
        front = np.argwhere(unlabeled & (n_labeled >= min_neighbors))
        del n_labeled
        if len(front) == 0:
            break

        # Labels of the neighbors of the front, read before any of them is
        # labeled so the whole front grows at once (no scan-order bias)
        neigh = front[:, None, :] + offsets[None, :, :]
        inside = np.all((neigh >= 0) & (neigh < sub_shape), axis=2)
        neigh = np.where(inside[..., None], neigh, 0)
        neigh_lab = np.where(
            inside, sub[neigh[..., 0], neigh[..., 1], neigh[..., 2]], 0
        )
        del neigh, inside

        # One entry per labeled neighbor: (front voxel, label, distance)
        rows = np.repeat(np.arange(len(front)), n_neigh)
        labs = neigh_lab.ravel()
        dists = np.tile(off_dist, len(front))
        keep = labs > 0
        rows, labs, dists = rows[keep], labs[keep], dists[keep]

        # Group by (front voxel, label): count and distance of the closest
        order = np.lexsort((dists, labs, rows))
        rows, labs, dists = rows[order], labs[order], dists[order]
        starts = np.flatnonzero(
            np.r_[True, (rows[1:] != rows[:-1]) | (labs[1:] != labs[:-1])]
        )
        g_rows, g_labs, g_dist = rows[starts], labs[starts], dists[starts]
        g_count = np.diff(np.r_[starts, len(rows)])

        # Winner of each front voxel: most votes, then closest, then smallest
        order = np.lexsort((g_labs, g_dist, -g_count, g_rows))
        g_rows, g_labs = g_rows[order], g_labs[order]
        first = np.r_[True, g_rows[1:] != g_rows[:-1]]
        win_rows, win_labs = g_rows[first], g_labs[first]

        tgt = front[win_rows]
        sub[tgt[:, 0], tgt[:, 1], tgt[:, 2]] = win_labs
        labeled[tgt[:, 0], tgt[:, 1], tgt[:, 2]] = True

    n_left = int((sub_mask & ~labeled).sum())
    if n_left > 0:
        warnings.warn(
            f"{n_left} voxels of the mask could not be labeled: they are not "
            "connected to any labeled voxel"
            + (
                f" or have fewer than {min_neighbors} labeled neighbors"
                if min_neighbors > 1
                else ""
            )
            + ". They are left as 0.",
            stacklevel=2,
        )

    out[~mask] = 0
    return out


#####################################################################################################
def simulate_array(
    mask_array: np.ndarray,
    n_volumes: int = 1,
    distribution: str = "normal",
    random_seed: int | None = None,
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
            raise ValueError(f"Error generating random values: {e}") from e

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
    vols_to_delete: list[int | tuple | list | str | np.ndarray] = None,
) -> tuple[str, list[int]]:
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


######################################################################################################
def dilate_mm(
    array: np.ndarray, affine: np.ndarray, shape: str = "cube", dilation_mm: float = 1
) -> np.ndarray:
    """
    Binarize a 3D array and dilate by a given distance in millimeters.

    Parameters
    ----------
    array       : 3D numpy array with values and zeros
    affine      : 4x4 affine matrix (voxel → mm)
    shape       : 'cube' for a box or 'sphere' for an ellipsoid structuring element
    dilation_mm : dilation radius in millimeters

    Returns
    -------
    dilated : binary 3D numpy array (bool)
    """

    binary_array = array != 0
    voxel_sizes = get_voxel_size(affine)

    # Per-axis radii in voxels
    radii_vox = np.array([dilation_mm / vs for vs in voxel_sizes])
    r = np.ceil(radii_vox).astype(int)

    if shape is None or shape == "cube":
        # Box: include every voxel within ±r[i] along each axis
        structure = np.ones([2 * ri + 1 for ri in r], dtype=bool)

    else:
        # Ellipsoid: anisotropic sphere in mm space
        xi = np.arange(-r[0], r[0] + 1)
        yi = np.arange(-r[1], r[1] + 1)
        zi = np.arange(-r[2], r[2] + 1)
        xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
        structure = (
            (xx / radii_vox[0]) ** 2
            + (yy / radii_vox[1]) ** 2
            + (zz / radii_vox[2]) ** 2
        ) <= 1.0

    dilated = binary_dilation(binary_array, structure=structure)

    return dilated.astype(bool)
