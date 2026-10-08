import copy
import os
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from scipy import stats
from scipy.linalg import pinv

# Importing local modules
from . import bidstools as cltbids
from . import colorstools as cltcol
from . import connectivitytools as cltcon
from . import freesurfertools as cltfree
from . import imagetools as cltimg
from . import misctools as cltmisc
from . import surfacetools as cltsurf
from .pointstools import PointCloud


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############      Section 1: Class dedicated to work with parcellation images           ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
class Parcellation:
    """
    Comprehensive class for working with brain parcellation data.

    Provides tools for loading, manipulating, and analyzing brain parcellation
    files with associated lookup tables. Supports filtering, masking, grouping,
    volume calculations, and various export formats for neuroimaging workflows.
    """

    ####################################################################################################
    def __init__(
        self,
        parc_file: str | Path | np.ndarray = None,
        color_table: str | Path | dict | None = None,
        affine: np.ndarray | None = None,
        parc_id: str | None = None,
        space_id: str | None = "unknown",
    ):
        """
        Initialize Parcellation object from file or array.

        Parameters
        ----------
        parc_file : str, Path, or np.ndarray, optional
            Path to parcellation file (NIfTI format) or numpy array containing
            parcellation data. If string/Path, loads from file and attempts to
            find associated TSV/LUT files. Default is None.

        color_table : str, Path, or dict, optional
            Color lookup table for parcellation regions. Can be:
            - Path to TSV/LUT file with columns: index, name, R, G, B, A (and optionally opacity)
            - Dictionary with required keys 'index', 'name', 'color' and optional keys
            'opacity', 'headerlines'
            If None, color table is auto-generated or loaded from sidecar files.
            Default is None.

        affine : np.ndarray, optional
            4x4 affine transformation matrix. If None and parc_file is array,
            creates identity matrix centered on data. Default is None.

        parc_id : str, optional
            Unique identifier for the parcellation. If None, generated from
            file name or set to 'numpy_array' for array input. Default is None.

        space_id : str, optional
            Identifier for the space in which the parcellation is defined
            (e.g., 'MNI152NLin6Asym', 'native'). Default is "unknown".

        Attributes
        ----------
        data : np.ndarray
            3D parcellation data array (integer labels).
        affine : np.ndarray
            4x4 affine transformation matrix.
        index : list of int
            List of region codes present in parcellation (excluding 0).
        name : list of str
            List of region names corresponding to codes.
        color : list of str
            List of colors (hex format) for each region.
        opacity : list of float
            List of opacity values (0-1) for each region. Default is 1.0 for all.
        headerlines : list
            List of header lines for the color table. Default is [].
        parc_file : str
            Path to parcellation file or 'numpy_array'.
        id : str
            Unique identifier for the parcellation.
        space : str
            Space identifier.
        dim : tuple
            Dimensions of the parcellation data.
        voxel_size : float
            Volume of a single voxel in mm³.
        dtype : np.dtype
            Data type of the parcellation.

        Raises
        ------
        ValueError
            If parcellation file does not exist or parc_file is None.
        FileNotFoundError
            If specified color_table file does not exist.

        Examples
        --------
        >>> # Load from file with automatic color table detection
        >>> parc = Parcellation('parcellation.nii.gz')
        >>>
        >>> # Load with explicit color table
        >>> parc = Parcellation('parcellation.nii.gz', color_table='colors.tsv')
        >>>
        >>> # Create from array with custom affine
        >>> parc = Parcellation(label_array, affine=img.affine, parc_id='custom')
        >>>
        >>> # Create from array with full color dictionary
        >>> color_dict = {
        ...     'index': [1, 2, 3],
        ...     'name': ['region1', 'region2', 'region3'],
        ...     'color': ['#FF0000', '#00FF00', '#0000FF'],
        ...     'opacity': [1.0, 0.8, 0.6],  # optional
        ...     'headerlines': ['# My custom parcellation']  # optional
        ... }
        >>> parc = Parcellation(label_array, color_table=color_dict)
        >>>
        >>> # Create with minimal color dictionary (opacity defaults to 1.0)
        >>> color_dict = {
        ...     'index': [1, 2, 3],
        ...     'name': ['region1', 'region2', 'region3'],
        ...     'color': ['#FF0000', '#00FF00', '#0000FF']
        ... }
        >>> parc = Parcellation(label_array, color_table=color_dict)
        """

        if parc_file is None:
            raise ValueError(
                "parc_file cannot be None. Provide a file path or numpy array."
            )

        # Handle file path input
        if isinstance(parc_file, (str, Path)):
            parc_file = str(parc_file)

            if not os.path.exists(parc_file):
                raise ValueError(f"The parcellation file does not exist: {parc_file}")

            self.parc_file = parc_file

            # Set parcellation ID
            if parc_id is not None:
                self.id = parc_id
            else:
                self.get_parcellation_id()

            # Set space ID
            self.set_space_id()

            # Load the parcellation data
            temp_iparc = nib.load(parc_file)
            self.affine = temp_iparc.affine
            self.data = temp_iparc.get_fdata().astype(np.int32)
            self.dtype = temp_iparc.get_data_dtype()

            # Determine color table to load
            lut_2_load = self._determine_color_table_file(parc_file, color_table)

            # Load or create color table
            if lut_2_load is not None:
                self.load_colortable(lut_file=lut_2_load)
            elif isinstance(color_table, dict):
                self._load_colortable_from_dict(color_table)
            else:
                # Auto-generate color table
                self._create_default_colortable()

        # Handle numpy array input
        elif isinstance(parc_file, np.ndarray):
            self.parc_file = "numpy_array"
            self.id = parc_id if parc_id is not None else "numpy_array"
            self.space = space_id
            self.data = parc_file.astype(np.int32)
            self.dtype = self.data.dtype

            # Create affine matrix if not provided
            if affine is None:
                affine = np.eye(4)
                center = np.array(self.data.shape) // 2
                affine[:3, 3] = -center

            self.affine = affine

            # Handle color table
            if isinstance(color_table, dict):
                self._load_colortable_from_dict(color_table)
            elif isinstance(color_table, (str, Path)) and os.path.exists(
                str(color_table)
            ):
                self.load_colortable(lut_file=str(color_table))
            else:
                # Auto-generate color table
                self._create_default_colortable()

        else:
            raise TypeError(
                f"parc_file must be str, Path, or np.ndarray, got {type(parc_file)}"
            )

        # Ensure required attributes exist and adjust to data
        self._ensure_attributes()
        self.adjust_values()

        # Set dimensional properties
        self.dim = self.data.shape
        self.voxel_size = cltimg.get_voxel_size(self.affine)

        # Set Voxel volume
        self.voxel_volume = cltimg.get_voxel_volume(self.affine)

        # Detect label range
        self.parc_range()

    def _determine_color_table_file(
        self, parc_file: str, color_table: str | Path | None
    ) -> str | None:
        """
        Determine which color table file to load.

        Priority: explicit color_table > .tsv sidecar > .lut sidecar

        Parameters
        ----------
        parc_file : str
            Path to parcellation file.
        color_table : str, Path, or None
            Explicitly provided color table path.

        Returns
        -------
        str or None
            Path to color table file, or None if none found.
        """
        # Check for explicit color_table
        if color_table is not None and isinstance(color_table, (str, Path)):
            color_table = str(color_table)
            if os.path.isfile(color_table):
                return color_table
            else:
                raise FileNotFoundError(
                    f"Specified color_table does not exist: {color_table}"
                )

        # Determine base name for sidecar files
        if parc_file.endswith(".nii.gz"):
            base = parc_file[:-7]  # Remove .nii.gz
        elif parc_file.endswith(".nii"):
            base = parc_file[:-4]  # Remove .nii
        else:
            return None

        # Check for sidecar files
        tsv_file = base + ".tsv"
        lut_file = base + ".lut"

        if os.path.isfile(tsv_file):
            return tsv_file
        elif os.path.isfile(lut_file):
            return lut_file

        return None

    def _load_colortable_from_dict(self, color_dict: dict) -> None:
        """
        Load color table from dictionary.

        Parameters
        ----------
        color_dict : dict
            Dictionary with required keys:
            - 'index': list of int (region codes)
            - 'name': list of str (region names)
            And optional keys:
            - 'color': list of str (hex colors)
            - 'opacity': list of float (0-1 range, defaults to 1.0 for all)
            - 'headerlines': list of str (defaults to [])

        Raises
        ------
        ValueError
            If required keys are missing or lists have mismatched lengths.
        """
        required_keys = {"index", "name"}
        if not required_keys.issubset(color_dict.keys()):
            raise ValueError(
                f"color_table dict must contain keys: {required_keys}. "
                f"Got: {set(color_dict.keys())}"
            )

        index = color_dict["index"]
        name = color_dict["name"]
        color = color_dict["color"]

        if not (len(index) == len(name)):
            raise ValueError(
                f"All required lists in color_table dict must have same length. "
                f"Got: index={len(index)}, name={len(name)}, color={len(color)}"
            )

        # Convert to appropriate types
        self.index = [int(x) for x in index]
        self.name = list(name)

        if "color" in color_dict:
            color = color_dict["color"]
            if len(color) != len(index):
                raise ValueError(
                    f"If provided, color list must have same length as index. "
                    f"Got: color={len(color)}, index={len(index)}"
                )

        else:
            color = cltcol.create_distinguishable_colors(
                len(self.index), output_format="hex"
            )

        # Force the colors to be in hex format
        color = cltcol.harmonize_colors(color, output_format="hex")
        self.color = list(color)

        # Handle optional opacity
        if "opacity" in color_dict:
            opacity = color_dict["opacity"]
            if len(opacity) != len(index):
                raise ValueError(
                    f"If provided, opacity list must have same length as index. "
                    f"Got: opacity={len(opacity)}, index={len(index)}"
                )
            self.opacity = [float(x) for x in opacity]
        else:
            # Default opacity: 1.0 for all regions
            self.opacity = [1.0] * len(self.index)

        # Handle optional headerlines
        if "headerlines" in color_dict:
            self.headerlines = list(color_dict["headerlines"])
        else:
            # Default: empty list
            self.headerlines = []

    def _create_default_colortable(self) -> None:
        """
        Create default color table from unique values in data.

        Generates region indices, automatic names, distinguishable colors,
        default opacity (1.0), and empty headerlines.
        """
        # Get unique non-zero values
        unique_vals = np.unique(self.data)
        unique_vals = unique_vals[unique_vals != 0]

        self.index = [int(x) for x in unique_vals]
        self.name = cltmisc.create_names_from_indices(self.index)

        if len(self.index) > 0:
            self.color = cltcol.create_distinguishable_colors(
                len(self.index), output_format="hex"
            )
            # Default opacity: 1.0 for all regions
            self.opacity = [1.0] * len(self.index)
        else:
            self.color = []
            self.opacity = []

        # Default: empty headerlines
        self.headerlines = []

    def _ensure_attributes(self) -> None:
        """
        Ensure all required attributes exist with valid values.

        Creates default values for index, name, color, opacity, and headerlines
        if not present.
        """
        # Ensure index exists
        if not hasattr(self, "index") or self.index is None:
            unique_vals = np.unique(self.data)
            unique_vals = unique_vals[unique_vals != 0]
            self.index = [int(x) for x in unique_vals]

        # Force index to int
        self.index = [int(x) for x in self.index]

        # Ensure name exists
        if not hasattr(self, "name") or self.name is None:
            self.name = cltmisc.create_names_from_indices(self.index)

        # Ensure color exists
        if not hasattr(self, "color") or self.color is None:
            if len(self.index) > 0:
                self.color = cltcol.create_distinguishable_colors(
                    len(self.index), output_format="hex"
                )
            else:
                self.color = []

        # Ensure opacity exists (default to 1.0 for all regions)
        if not hasattr(self, "opacity") or self.opacity is None:
            self.opacity = [1.0] * len(self.index)

        # Ensure opacity is iterable (guard against scalar float)
        if not hasattr(self.opacity, "__len__"):
            self.opacity = [float(self.opacity)] * len(self.index)

        # Ensure opacity is a flat 1-D list of floats with the correct length
        opacity_arr = np.array(self.opacity)
        if opacity_arr.ndim != 1 or len(opacity_arr) != len(self.index):
            self.opacity = [1.0] * len(self.index)
        else:
            self.opacity = [float(x) for x in opacity_arr.tolist()]

        # Ensure headerlines exists (default to empty list)
        if not hasattr(self, "headerlines") or self.headerlines is None:
            self.headerlines = []

        # Detect minimum and maximum labels
        self.parc_range()

    #####################################################################################################
    @classmethod
    def simulate_parcellation(
        cls,
        n_regions: int,
        dimensions: tuple[int, int, int] = (128, 128, 100),
        voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        affine: np.ndarray | None = None,
        seed: int | None = None,
    ) -> "Parcellation":
        """
        Simulate a random, brain-like parcellation using Voronoi tessellation.

        Scatters `n_regions` seed voxels across a volume of the requested
        `dimensions` and assigns every other voxel to the label of its nearest
        seed, producing contiguous, space-filling regions (no background/zero
        voxels). Useful for generating quick test data for methods that need a
        labeled volume with a known number of regions.

        Parameters
        ----------
        n_regions : int
            Number of regions to simulate. Must be a positive integer. The
            resulting labels are 1, 2, ..., n_regions.

        dimensions : tuple of int, optional
            Volume dimensions (dimx, dimy, dimz) in voxels. A single integer
            is broadcast to all three axes. Default is (128, 128, 100).

        voxel_size : tuple of float, optional
            Voxel size (vx, vy, vz) in mm. A single number is broadcast to
            all three axes. Default is (1.0, 1.0, 1.0). Used to build the
            default affine (when affine is None) and to compute voxel_volume.

        affine : np.ndarray, optional
            4x4 affine matrix. If None (default), a diagonal affine is built
            from voxel_size, with the translation set so the geometric center
            of the volume sits at the world-space origin.

        seed : int, optional
            Seed for the random number generator, for reproducible simulations.

        Returns
        -------
        Parcellation
            A new Parcellation with `n_regions` contiguous regions, an
            auto-generated color table, and space_id='simulated'.

        Raises
        ------
        ValueError
            If n_regions, dimensions, voxel_size, or affine are invalid.

        Examples
        --------
        >>> parc = Parcellation.simulate_parcellation(n_regions=10)
        >>> parc = Parcellation.simulate_parcellation(
        ...     n_regions=50, dimensions=(64, 64, 64), voxel_size=2.0, seed=42
        ... )
        """
        # --- Validate n_regions ---
        if not isinstance(n_regions, (int, np.integer)) or n_regions < 1:
            raise ValueError(
                f"n_regions must be a positive integer, got {n_regions!r}."
            )
        n_regions = int(n_regions)

        # --- Validate/normalize dimensions ---
        if isinstance(dimensions, (int, np.integer)):
            dimensions = (int(dimensions),) * 3
        dimensions = tuple(dimensions)
        if len(dimensions) != 3 or not all(
            isinstance(d, (int, np.integer)) and d > 0 for d in dimensions
        ):
            raise ValueError(
                f"dimensions must be a tuple of 3 positive integers (or a single "
                f"integer), got {dimensions!r}."
            )
        dimensions = tuple(int(d) for d in dimensions)

        # --- Validate/normalize voxel_size ---
        if isinstance(voxel_size, (int, float, np.integer, np.floating)):
            voxel_size = (float(voxel_size),) * 3
        voxel_size = tuple(voxel_size)
        if len(voxel_size) != 3 or not all(
            isinstance(v, (int, float, np.integer, np.floating)) and v > 0
            for v in voxel_size
        ):
            raise ValueError(
                f"voxel_size must be a tuple of 3 positive numbers (or a single "
                f"number), got {voxel_size!r}."
            )
        voxel_size = tuple(float(v) for v in voxel_size)

        # --- Validate/build affine ---
        if affine is None:
            affine = np.eye(4)
            affine[0, 0], affine[1, 1], affine[2, 2] = voxel_size
            center_vox = np.array(dimensions, dtype=float) / 2.0
            center_mm = center_vox * np.array(voxel_size)
            affine[:3, 3] = -center_mm
        else:
            affine = np.asarray(affine, dtype=float)
            if affine.shape != (4, 4):
                raise ValueError(
                    f"affine must be a 4x4 numpy array, got shape {affine.shape}."
                )

            # Voxel size implied by the affine's column norms (robust to rotated affines)
            affine_voxel_size = tuple(np.linalg.norm(affine[:3, :3], axis=0))
            if voxel_size != (1.0, 1.0, 1.0) and not np.allclose(
                affine_voxel_size, voxel_size, atol=1e-6
            ):
                warnings.warn(
                    f"voxel_size={voxel_size} was provided together with an explicit "
                    f"affine matrix (whose implied voxel size is "
                    f"{tuple(round(v, 3) for v in affine_voxel_size)}). Voxel size is "
                    f"always derived from the affine, not set independently — the "
                    f"provided voxel_size is ignored.",
                    stacklevel=2,
                )
        # --- Simulate a Voronoi-style parcellation ---
        from scipy.spatial import cKDTree

        rng = np.random.default_rng(seed)
        seed_coords = rng.integers(low=[0, 0, 0], high=dimensions, size=(n_regions, 3))

        grid = np.indices(dimensions).reshape(3, -1).T  # (n_voxels, 3)
        tree = cKDTree(seed_coords)
        _, nearest_seed = tree.query(grid)

        data = (nearest_seed + 1).reshape(dimensions).astype(np.int32)

        parc = cls(
            data,
            affine=affine,
            parc_id=f"simulated_{n_regions}regions",
            space_id="simulated",
        )

        # Use the seed for the region colors too, so the whole simulation is reproducible
        if len(parc.index) > 0:
            parc.color = cltcol.create_distinguishable_colors(
                len(parc.index), output_format="hex", random_seed=seed
            )

        return parc

    #####################################################################################################
    def get_space_id(self) -> str:
        """
        Infer the space identifier from the parcellation filename if space is not yet set.

        Returns
        -------
        space_id : str
            The space identifier extracted from the BIDS filename, or "unknown" if not found.

        Notes
        -----
        This method only attempts to extract the space entity from BIDS-compliant
        filenames. It does not modify the `space` attribute.

        Examples
        --------
        >>> # Extract from BIDS filename
        >>> parc = Parcellation('sub-01_space-t1_atlas-xxx.nii.gz')
        >>> space_id = parc.get_space_id()
        >>> print(space_id)
        't1'

        >>> # Non-BIDS filename
        >>> parc = Parcellation('custom_parcellation.nii.gz')
        >>> space_id = parc.get_space_id()
        >>> print(space_id)
        'unknown'

        >>> # No file set
        >>> parc = Parcellation()
        >>> space_id = parc.get_space_id()
        >>> print(space_id)
        'unknown'
        """

        # If space is already set, return it
        if hasattr(self, "space"):
            return self.space

        # Otherwise, infer from filename
        if not hasattr(self, "parc_file") or not self.parc_file:
            return "unknown"

        parc_file_name = os.path.basename(self.parc_file)

        # Check if the filename follows BIDS naming conventions
        if cltbids.is_bids_filename(parc_file_name):
            # Extract entities from the filename
            name_ent_dict = cltbids.str2entity(parc_file_name)

            # Return space entity if present
            if "space" in name_ent_dict:
                return name_ent_dict["space"]

        return "unknown"

    #####################################################################################################
    def set_space_id(self, space_id: str | None = None) -> str:
        """
        Set the space identifier for the parcellation.

        Parameters
        ----------
        space_id : str, optional
            Identifier for the space. If None, attempts to infer from filename
            using get_space_id().

        Returns
        -------
        space_id : str
            The space identifier that was set.

        Notes
        -----
        Priority order for setting space:
        1. Provided space_id parameter (if not None)
        2. Inferred from filename via get_space_id() (returns "unknown" if not found)

        This method sets the `space` attribute of the Parcellation object.

        Examples
        --------
        >>> # Explicitly set space
        >>> parc = Parcellation('sub-01_atlas-xxx.nii.gz')
        >>> parc.set_space_id('mni152')
        'mni152'

        >>> # Infer from filename
        >>> parc = Parcellation('sub-01_space-t1_atlas-xxx.nii.gz')
        >>> parc.set_space_id()
        't1'

        >>> # Fallback to unknown
        >>> parc = Parcellation('custom_parcellation.nii.gz')
        >>> parc.set_space_id()
        'unknown'
        """

        # Priority 1: Use explicitly provided space_id
        if space_id is not None:
            final_space_id = space_id
        else:
            # Priority 2: Infer from filename (returns "unknown" if not found)
            final_space_id = self.get_space_id()

        # Set the space attribute
        self.space = final_space_id

        return final_space_id

    ####################################################################################################
    def get_parcellation_id(self) -> str:
        """
        Generate a unique identifier for the parcellation based on its filename. If the filename
        follows BIDS naming conventions, it extracts relevant entities to form the ID.
        If the filename does not follow BIDS conventions, it uses the filename without extension.

        Returns
        -------
        str
            Unique identifier for the parcellation, formatted as 'atlas-<atlas_name>_seg-<seg_name>_scale-<scale_value>_desc-<description>'.
            If no entities are found, it returns the filename without extension.

        Raises
        ------
        ValueError
            If the parcellation file is not set.

        Notes
        This method is useful for identifying and categorizing parcellation files based on their naming conventions.
        It can be used to easily retrieve or reference specific parcellations in analyses or reports.

        Examples
        --------
        >>> parc = Parcellation('sub-01_ses-01_acq-mprage_space-t1_atlas-xxx_seg-yyy_scale-1_desc-test.nii.gz')
        >>> parc_id = parc.get_parcellation_id()
        >>> print(parc_id)
        'atlas-xxx_seg-yyy_scale-1_desc-test'
        >>> parc = Parcellation('custom_parcellation.nii.gz')
        >>> parc_id = parc.get_parcellation_id()
        >>> print(parc_id)
        'custom_parcellation'

        """
        # Check if the parcellation file is set
        if not hasattr(self, "parc_file"):
            raise ValueError(
                "The parcellation file is not set. Please load a parcellation file first."
            )

        # Initialize parc_fullid as an empty string
        parc_fullid = ""

        # Get the base name of the parcellation file
        parc_file_name = os.path.basename(self.parc_file)

        # Check if the parcellation file name follows BIDS naming conventions
        if cltbids.is_bids_filename(parc_file_name):

            # Extract entities from the parcellation file name
            name_ent_dict = cltbids.str2entity(parc_file_name)
            ent_names_list = list(name_ent_dict.keys())

            # Create parc_fullid based on the entities present in the parcellation file name
            parc_fullid = ""
            if "atlas" in ent_names_list:
                parc_fullid = "atlas-" + name_ent_dict["atlas"]

            if "seg" in ent_names_list:
                parc_fullid += "_seg-" + name_ent_dict["seg"]

            if "scale" in ent_names_list:
                parc_fullid += "_scale-" + name_ent_dict["scale"]

            if "desc" in ent_names_list:
                parc_fullid += "_desc-" + name_ent_dict["desc"]

            # Remove the _ if the parc_fullid starts with it
            if parc_fullid.startswith("_"):
                parc_fullid = parc_fullid[1:]

        else:

            # Remove the file extension if it exists
            if parc_file_name.endswith(".nii.gz"):
                parc_fullid = parc_file_name[:-7]
            else:
                parc_fullid = parc_file_name[:-4]

        self.id = parc_fullid

        return parc_fullid

    ######################################################################################################
    def get_info(self, verbose: bool = True) -> dict:
        """
        Display and return comprehensive information about the Parcellation object.

        Provides a formatted overview of the parcellation including identification,
        image properties, color-table statistics, and label consistency checks.
        Useful for quick inspection and validation of volumetric parcellation data.

        The method displays:
            - Basic identification (ID, space, file path)
            - Image properties (dimensions, voxel size, data type, affine matrix)
            - Color-table properties (number of regions, label range, opacity range)
            - Label consistency (codes in data missing from table, and table entries
              absent from data)

        Parameters
        ----------
        verbose : bool, optional
            If True, prints the information in a formatted table to stdout.
            If False, only returns the info dictionary without printing.
            Default is True.

        Returns
        -------
        info : dict
            Dictionary containing the following keys:

            - 'id' : str
                Parcellation identifier string.

            - 'space' : str
                Space identifier (e.g. 'MNI152NLin6Asym', 'native').

            - 'parc_file' : str
                Full path to the parcellation file, or 'numpy_array'.

            - 'dim' : tuple
                Spatial dimensions of the parcellation volume (x, y, z).

            - 'voxel_size' : float
                Volume of a single voxel in mm³.

            - 'dtype' : str
                NumPy data type of the parcellation array.

            - 'affine' : np.ndarray or None
                4x4 affine transformation matrix, or None if not set.

            - 'n_regions' : int
                Number of regions defined in the color table (len of index).

            - 'min_label' : int or None
                Minimum non-zero label value present in the data.

            - 'max_label' : int or None
                Maximum label value present in the data.

            - 'opacity_range' : tuple or None
                (min_opacity, max_opacity) across all defined regions,
                or None if opacity is not set.

            - 'labels_not_in_table' : list of int
                Label values present in the data but absent from ``self.index``.
                Ideally empty.

            - 'regions_not_in_data' : list of str
                Region names whose index codes do not appear in the data.
                These are defined but unused regions.

        Examples
        --------
        >>> parc = Parcellation('sub-01_space-MNI_atlas-Schaefer_desc-400Parcels.nii.gz')
        >>> info = parc.get_info()
        ╔════════════════════════════════════════════════════════════════╗
        ║                    PARCELLATION INFO                           ║
        ╠════════════════════════════════════════════════════════════════╣
        ║  ID     : atlas-Schaefer_desc-400Parcels                       ║
        ║  Space  : MNI                                                  ║
        ║  File   : sub-01_space-MNI_atlas-Schaefer_desc-400Parcels...   ║
        ╠════════════════════════════════════════════════════════════════╣
        ║  IMAGE PROPERTIES                                              ║
        ║    Dimensions  :   182 x 218 x 182                            ║
        ║    Voxel size  :         1.000 mm³                            ║
        ║    Data type   :              int32                            ║
        ╠════════════════════════════════════════════════════════════════╣
        ║  COLOR TABLE                                                   ║
        ║    Regions     :               400                             ║
        ║    Label range :         1  →  400                            ║
        ║    Opacity     :      1.00  →  1.00                           ║
        ╠════════════════════════════════════════════════════════════════╣
        ║  LABEL CONSISTENCY                                             ║
        ║    Labels not in table   :    0                                ║
        ║    Regions not in data   :    0                                ║
        ╚════════════════════════════════════════════════════════════════╝
        """

        WIDTH = 64  # inner content width (between the ║ borders)

        def _row(content: str) -> None:
            """Print a single bordered row, left-justified."""
            print(f"║{content.ljust(WIDTH)}║")

        # ── Gather information ────────────────────────────────────────────────────

        info: dict = {
            "id": getattr(self, "id", None),
            "space": getattr(self, "space", None),
            "parc_file": getattr(self, "parc_file", None),
            "dim": getattr(self, "dim", None),
            "voxel_size": getattr(self, "voxel_size", None),
            "voxel_volume": getattr(self, "voxel_volume", None),
            "dtype": str(getattr(self, "dtype", None)),
            "affine": getattr(self, "affine", None),
            "n_regions": None,
            "min_label": getattr(self, "minlab", None),
            "max_label": getattr(self, "maxlab", None),
            "opacity_range": None,
            "labels_not_in_table": [],
            "regions_not_in_data": [],
        }

        # Color-table stats
        if hasattr(self, "index") and self.index is not None:
            info["n_regions"] = len(self.index)

        if hasattr(self, "opacity") and self.opacity is not None:
            try:
                op_arr = np.array(self.opacity, dtype=float)
                info["opacity_range"] = (float(op_arr.min()), float(op_arr.max()))
            except Exception:
                pass

        # Label consistency checks (only when both data and index exist)
        if (
            hasattr(self, "data")
            and self.data is not None
            and hasattr(self, "index")
            and self.index is not None
        ):
            data_codes = {int(v) for v in np.unique(self.data) if v != 0}
            table_codes = {int(v) for v in self.index}

            info["labels_not_in_table"] = sorted(data_codes - table_codes)

            missing_in_data = sorted(table_codes - data_codes)
            if hasattr(self, "name") and self.name is not None:
                idx_to_name = {
                    int(c): n for c, n in zip(self.index, self.name, strict=False)
                }
                info["regions_not_in_data"] = [
                    idx_to_name.get(c, str(c)) for c in missing_in_data
                ]
            else:
                info["regions_not_in_data"] = missing_in_data

        # ── Print ─────────────────────────────────────────────────────────────────
        if verbose:

            # Helper: truncate a string to fit within a column width
            def _trunc(s: str, max_w: int) -> str:
                return s if len(s) <= max_w else "..." + s[-(max_w - 3) :]

            print("╔" + "═" * WIDTH + "╗")
            _row("  PARCELLATION INFO".center(WIDTH))
            print("╠" + "═" * WIDTH + "╣")

            # — Identification ————————————————————————————————————————————————
            id_val = info["id"] or "N/A"
            space_val = info["space"] or "N/A"
            file_val = _trunc(info["parc_file"] or "N/A", WIDTH - 12)

            _row(f"  ID     : {id_val}")
            _row(f"  Space  : {space_val}")
            _row(f"  File   : {file_val}")

            # — Image properties ——————————————————————————————————————————————
            print("╠" + "═" * WIDTH + "╣")
            _row("  IMAGE PROPERTIES")

            if info["dim"] is not None:
                dim_str = " x ".join(str(d) for d in info["dim"])
                _row(f"    Dimensions  : {dim_str:>20}")
            else:
                _row("    Dimensions  :          N/A")

            if info["voxel_size"] is not None:
                dim_str = " x ".join(str(d) for d in info["voxel_size"])
                _row(f"    Voxel size  : {dim_str:>20} mm")
            else:
                _row("    Voxel size  :          N/A")

            if info["voxel_volume"] is not None:
                _row(f"    Voxel volume: {info['voxel_volume']:>16.3f} mm³")
            else:
                _row("    Voxel volume  :          N/A")

            _row(f"    Data type   : {info['dtype']:>20}")

            # — Color table ———————————————————————————————————————————————————
            print("╠" + "═" * WIDTH + "╣")
            _row("  COLOR TABLE")

            if info["n_regions"] is not None:
                _row(f"    Regions     : {info['n_regions']:>20,}")
            else:
                _row("    Regions     :          N/A")

            if info["min_label"] is not None and info["max_label"] is not None:
                range_str = f"{info['min_label']}  →  {info['max_label']}"
                _row(f"    Label range : {range_str:>20}")
            else:
                _row("    Label range :          N/A")

            if info["opacity_range"] is not None:
                op_str = (
                    f"{info['opacity_range'][0]:.2f}  →  {info['opacity_range'][1]:.2f}"
                )
                _row(f"    Opacity     : {op_str:>20}")
            else:
                _row("    Opacity     :          N/A")

            # — Label consistency ——————————————————————————————————————————————
            print("╠" + "═" * WIDTH + "╣")
            _row("  LABEL CONSISTENCY")

            n_missing_table = len(info["labels_not_in_table"])
            n_missing_data = len(info["regions_not_in_data"])

            _row(f"    Labels not in table   : {n_missing_table:>4}")
            if n_missing_table > 0:
                codes_str = ", ".join(str(v) for v in info["labels_not_in_table"])
                _row(f"      ↳ codes : {_trunc(codes_str, WIDTH - 14)}")

            _row(f"    Regions not in data   : {n_missing_data:>4}")
            if n_missing_data > 0:
                names_str = ", ".join(str(v) for v in info["regions_not_in_data"])
                _row(f"      ↳ names : {_trunc(names_str, WIDTH - 14)}")

            print("╚" + "═" * WIDTH + "╝")

        return info

    ######################################################################################################
    def get_regions_info(
        self,
        region_labels: int | list[int] | np.ndarray = None,
        region_names: str | list[str] = None,
        output_format: str = "dataframe",
    ) -> "pd.DataFrame | dict":
        """
        Get summary information (name, color, voxel count, and volume) for the
        specified regions.

        Parameters
        ----------
        region_labels : int, list of int, or np.ndarray, optional
            Region labels to include. If both region_labels and region_names
            are specified, region_labels takes priority and region_names is
            ignored. Default is None.

        region_names : str or list of str, optional
            Region name(s) (substring match) to include. Ignored if
            region_labels is also specified. Default is None.

        output_format : str, optional
            'dataframe' (default) returns a pandas.DataFrame with one row per
            region. 'dict' returns a dictionary keyed by region label, e.g.
            {label: {'name': ..., 'color': ..., 'nvoxels': ..., 'volume': ...}}.

        Returns
        -------
        pd.DataFrame or dict
            Regions' index, name, color, number of voxels ('nvoxels'), and
            volume in mm^3 ('volume'). If neither region_labels nor
            region_names is given, all regions currently in the parcellation
            are returned.

        Raises
        ------
        ValueError
            If none of the requested labels/names match a region in the
            parcellation, or if output_format is not 'dataframe' or 'dict'.

        Examples
        --------
        >>> # Info for every region currently in the parcellation
        >>> df = parc.get_regions_info()
        >>>
        >>> # Info for specific labels
        >>> df = parc.get_regions_info(region_labels=[1, 2, 3])
        >>>
        >>> # Info by name (substring match)
        >>> df = parc.get_regions_info(region_names=['hippocampus', 'amygdala'])
        >>>
        >>> # region_labels takes priority when both are given
        >>> df = parc.get_regions_info(region_labels=[1, 2], region_names=['hippocampus'])
        >>>
        >>> # As a dictionary keyed by region label instead of a DataFrame
        >>> info = parc.get_regions_info(region_labels=[1, 2], output_format='dict')
        >>> info[1]['volume']
        """

        if output_format not in ("dataframe", "dict"):
            raise ValueError(
                f"output_format must be 'dataframe' or 'dict', got '{output_format}'."
            )

        if region_labels is not None and region_names is not None:
            print(
                "Both region_labels and region_names were specified. Ignoring "
                "region_names and using region_labels for region selection."
            )
            region_names = None

        # Determine which labels to summarize
        if region_labels is not None:
            if isinstance(region_labels, (int, np.integer)):
                region_labels = [region_labels]
            elif isinstance(region_labels, np.ndarray):
                region_labels = region_labels.tolist()
            region_labels = cltmisc.build_indices(region_labels)

            present_labels = set(self.index)
            selected_labels = [lb for lb in region_labels if lb in present_labels]

            if len(selected_labels) == 0:
                raise ValueError(
                    f"None of the requested region_labels were found in the "
                    f"parcellation: {region_labels}"
                )

        elif region_names is not None:
            if isinstance(region_names, str):
                region_names = [region_names]

            indexes = cltmisc.get_indexes_by_substring(
                input_list=self.name,
                or_filter=region_names,
                invert=False,
                bool_case=False,
            )

            if len(indexes) == 0:
                raise ValueError(
                    f"None of the requested region_names matched any region in "
                    f"the parcellation: {region_names}"
                )

            selected_labels = [self.index[i] for i in indexes]

        else:
            # No filter specified - summarize every region currently in the parcellation
            selected_labels = list(self.index)

        # Voxel volume (mm^3) implied by the affine
        voxel_volume = cltimg.get_voxel_volume(self.affine)

        # Count voxels per label directly from the data, so counts stay correct
        # even if self.index/self.data were ever out of sync
        unique_data_labels, voxel_counts = np.unique(self.data, return_counts=True)
        voxel_count_map = dict(
            zip(unique_data_labels.tolist(), voxel_counts.tolist(), strict=False)
        )

        info: dict = {}
        for label in selected_labels:
            pos = self.index.index(label)
            nvox = voxel_count_map.get(label, 0)

            info[int(label)] = {
                "name": self.name[pos],
                "color": self.color[pos],
                "nvoxels": int(nvox),
                "volume": float(nvox) * voxel_volume,
            }

        if output_format == "dict":
            return info

        df = pd.DataFrame(
            {
                "index": list(info.keys()),
                "name": [v["name"] for v in info.values()],
                "color": [v["color"] for v in info.values()],
                "nvoxels": [v["nvoxels"] for v in info.values()],
                "volume": [v["volume"] for v in info.values()],
            }
        )
        return df

    ##########################################################################################################
    def get_data(self) -> np.ndarray:
        """
        Get the parcellation data as a numpy array.

        Returns
        -------
        np.ndarray
            The parcellation data array.

        Raises
        ------
        ValueError
            If the parcellation data is not set.

        Notes
        -----
        This method provides direct access to the underlying parcellation data.
        It is useful for analyses or manipulations that require the raw label values.

        Examples
        --------
        >>> parc = Parcellation('parcellation.nii.gz')
        >>> data_array = parc.get_data()
        >>> print(data_array.shape)
        (182, 218, 182)
        """
        if not hasattr(self, "data"):
            raise ValueError(
                "The parcellation data is not set. Please load a parcellation file first."
            )
        return self.data

    ############################################################################################################
    def get_affine(self) -> np.ndarray:
        """
        Get the affine transformation matrix of the parcellation.

        Returns
        -------
        np.ndarray
            The 4x4 affine transformation matrix.

        Raises
        ------
        ValueError
            If the affine matrix is not set.

        Notes
        -----
        The affine matrix defines the spatial orientation and voxel size of the parcellation.
        It is essential for spatial transformations and alignment with other neuroimaging data.

        Examples
        --------
        >>> parc = Parcellation('parcellation.nii.gz')
        >>> affine_matrix = parc.get_affine()
        >>> print(affine_matrix)
        [[-1.   0.   0.  90.]
         [ 0.   1.   0. -126.]
         [ 0.   0.   1. -72.]
         [ 0.   0.   0.   1.]]
        """
        if not hasattr(self, "affine"):
            raise ValueError(
                "The affine matrix is not set. Please load a parcellation file first."
            )
        return self.affine

    ####################################################################################################
    def get_index(self) -> list[int]:
        """
        Get the list of region indices (codes) defined in the parcellation.

        Returns
        -------
        List[int]
            List of region indices.

        Raises
        ------
        ValueError
            If the index is not set.

        Notes
        -----
        The index represents the unique integer codes assigned to each region in the parcellation.
        It is useful for mapping label values to region names and colors.

        Examples
        --------
        >>> parc = Parcellation('parcellation.nii.gz')
        >>> indices = parc.get_index()
        >>> print(indices)
        [1, 2, 3, 4, 5]
        """
        if not hasattr(self, "index"):
            raise ValueError(
                "The index is not set. Please load a parcellation file first."
            )
        return self.index

    ####################################################################################################
    def get_names(self) -> list[str]:
        """
        Get the list of region names defined in the parcellation.

        Returns
        -------
        List[str]
            List of region names.

        Raises
        ------
        ValueError
            If the names are not set.

        Notes
        -----
        The names correspond to the human-readable labels for each region in the parcellation.
        They are useful for reporting and visualization purposes.

        Examples
        --------
        >>> parc = Parcellation('parcellation.nii.gz')
        >>> names = parc.get_names()
        >>> print(names)
        ['Region 1', 'Region 2', 'Region 3', 'Region 4', 'Region 5']
        """
        if not hasattr(self, "name"):
            raise ValueError(
                "The names are not set. Please load a parcellation file first."
            )
        return self.name

    ####################################################################################################
    def get_colors(self) -> list[str]:
        """
        Get the list of region colors defined in the parcellation.

        Returns
        -------
        List[str]
            List of region colors in hex format.

        Raises
        ------
        ValueError
            If the colors are not set.

        Notes
        -----
        The colors correspond to the visual representation of each region in the parcellation.
        They are useful for visualization and plotting purposes.

        Examples
        --------
        >>> parc = Parcellation('parcellation.nii.gz')
        >>> colors = parc.get_colors()
        >>> print(colors)
        ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']
        """
        if not hasattr(self, "color"):
            raise ValueError(
                "The colors are not set. Please load a parcellation file first."
            )

        colors = cltcol.harmonize_colors(self.color, output_format="hex")

        return colors

    ####################################################################################################
    def set_data(self, data: np.ndarray) -> None:
        """
        Set the parcellation data.

        Parameters
        ----------
        data : np.ndarray
            The parcellation data array to set.

        Raises
        ------
        TypeError
            If the provided data is not a numpy array.

        Notes
        -----
        This method allows direct assignment of the underlying parcellation data.
        Use with caution, as it does not automatically update dependent attributes
        such as the index, names, or colors.

        Examples
        --------
        >>> parc = Parcellation('parcellation.nii.gz')
        >>> parc.set_data(new_data_array)
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("The data must be a numpy array.")
        self.data = data

    ####################################################################################################
    def set_affine(self, affine: np.ndarray) -> None:
        """
        Set the affine transformation matrix of the parcellation.

        Parameters
        ----------
        affine : np.ndarray
            The 4x4 affine transformation matrix to set.

        Raises
        ------
        TypeError
            If the provided affine is not a numpy array.
        ValueError
            If the affine matrix does not have shape (4, 4).

        Notes
        -----
        The affine matrix defines the spatial orientation and voxel size of the
        parcellation. It is essential for spatial transformations and alignment
        with other neuroimaging data.

        Examples
        --------
        >>> parc = Parcellation('parcellation.nii.gz')
        >>> parc.set_affine(new_affine_matrix)
        """
        if not isinstance(affine, np.ndarray):
            raise TypeError("The affine matrix must be a numpy array.")
        if affine.shape != (4, 4):
            raise ValueError("The affine matrix must have shape (4, 4).")
        self.affine = affine

    ####################################################################################################
    def set_index(self, index: list[int]) -> None:
        """
        Set the list of region indices (codes) defined in the parcellation.

        Parameters
        ----------
        index : List[int]
            List of region indices to set.

        Raises
        ------
        TypeError
            If the provided index is not a list of integers.

        Notes
        -----
        The index represents the unique integer codes assigned to each region
        in the parcellation. It is useful for mapping label values to region
        names and colors.

        Examples
        --------
        >>> parc = Parcellation('parcellation.nii.gz')
        >>> parc.set_index([1, 2, 3, 4, 5])
        """
        if not isinstance(index, list) or not all(
            isinstance(i, (int, np.integer)) for i in index
        ):
            raise TypeError("The index must be a list of integers.")
        self.index = index

    ####################################################################################################
    def set_names(self, names: list[str]) -> None:
        """
        Set the list of region names defined in the parcellation.

        Parameters
        ----------
        names : List[str]
            List of region names to set.

        Raises
        ------
        TypeError
            If the provided names is not a list of strings.

        Notes
        -----
        The names correspond to the human-readable labels for each region in
        the parcellation. They are useful for reporting and visualization
        purposes.

        Examples
        --------
        >>> parc = Parcellation('parcellation.nii.gz')
        >>> parc.set_names(['Region 1', 'Region 2', 'Region 3'])
        """
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise TypeError("The names must be a list of strings.")
        self.name = names

    ####################################################################################################
    def set_colors(self, colors: list[str]) -> None:
        """
        Set the list of region colors defined in the parcellation.

        Parameters
        ----------
        colors : List[str]
            List of region colors (e.g., hex strings) to set.

        Raises
        ------
        TypeError
            If the provided colors is not a list.

        Notes
        -----
        The colors correspond to the visual representation of each region in
        the parcellation. They are harmonized to a consistent hex format
        before being stored.

        Examples
        --------
        >>> parc = Parcellation('parcellation.nii.gz')
        >>> parc.set_colors(['#FF0000', '#00FF00', '#0000FF'])
        """
        if not isinstance(colors, list):
            raise TypeError("The colors must be a list.")
        self.color = cltcol.harmonize_colors(colors, output_format="hex")

    ####################################################################################################
    def export_summary_to_hdf5(self, out_file: str, overwrite: bool = False):
        """
        Export parcellation summary to HDF5 file.

        Parameters
        ----------
        out_file : str
            Path to output HDF5 file.

        Raises
        ------
        ValueError
            If the parcellation data is not set.

        Notes
        -----
        This method saves the parcellation data, index, name, and color attributes to an HDF5 file.
        It is useful for archiving and sharing parcellation information in a structured format.

        Examples
        --------
        >>> parc.export_summary_to_hdf5('parcellation_summary.h5')
        """

        out_path = Path(out_file)

        # Check if the output directory exists, if not create raise an error
        if not out_path.parent.exists():
            raise ValueError(
                f"The output directory {out_path.parent} does not exist. Please create it first."
            )

        # Check if the output file already exists
        if out_path.exists() and not overwrite:
            raise ValueError(
                f"The output file {out_path} already exists. Use overwrite=True to overwrite it."
            )

        # Check if the parcellation data is set
        if not hasattr(self, "data"):
            raise ValueError(
                "The parcellation data is not set. Please load a parcellation file first."
            )

        # Check if the attributes parcellation_id and space_id are set
        if not hasattr(self, "id"):
            self.get_parcellation_id()

        if not hasattr(self, "space"):
            self.get_space_id(space_id="unknown")

        parc_id = self.id
        space_id = self.space
        base_cad = f"parcellation_{parc_id}/space-{space_id}"

        # Create the hf file
        hf = h5py.File(out_file, "w")

        # Save the filename
        if hasattr(self, "parc_file"):
            hf.create_dataset(f"{base_cad}/header/file_path", data=self.parc_file)

        # Save the LUT file pathname if it exists
        if hasattr(self, "lut_file"):
            hf.create_dataset(f"{base_cad}/header/lut_file", data=self.lut_file)

        # Save the parcellation id
        if hasattr(self, "id"):
            hf.create_dataset(f"{base_cad}/header/id", data=self.id)

        # Save the space id
        if hasattr(self, "space"):
            hf.create_dataset(f"{base_cad}/header/space", data=self.space)

        # Save the parcellation dimension
        if hasattr(self, "dim"):
            hf.create_dataset(f"{base_cad}/header/dim", data=self.dim)

        # Save the parcellation voxel size
        if hasattr(self, "voxel_size"):
            hf.create_dataset(f"{base_cad}/header/voxel_size", data=self.voxel_size)

        # Save the parcellation affine
        if hasattr(self, "affine"):
            hf.create_dataset(f"{base_cad}/header/affine", data=self.affine)

        # Save the number of regions
        if hasattr(self, "index"):
            hf.create_dataset(f"{base_cad}/header/num_regions", data=len(self.index))

        else:
            # If index is not set, calculate the number of regions from the data
            regions = np.unique(self.data)
            n_regions = len(regions[regions != 0])
            hf.create_dataset(f"{base_cad}/header/num_regions", data=n_regions)

        # Save the minimum label
        if hasattr(self, "min_label"):
            hf.create_dataset(f"{base_cad}/header/min_label", data=self.min_label)
        else:
            # If min_label and max_label are not set, calculate them from the data
            regions = np.unique(self.data)
            regions = regions[regions != 0]
            hf.create_dataset(f"{base_cad}/header/min_label", data=np.min(regions))

            # Save the maximum label
        if hasattr(self, "max_label"):
            hf.create_dataset(f"{base_cad}/header/max_label", data=self.max_label)
        else:
            # If min_label and max_label are not set, calculate them from the data
            regions = np.unique(self.data)
            regions = regions[regions != 0]
            hf.create_dataset(f"{base_cad}/header/max_label", data=np.max(regions))

        # Save the index of the regions
        if hasattr(self, "index"):
            hf.create_dataset(f"{base_cad}/regions_indices", data=self.index)

        # Save the region names
        if hasattr(self, "name"):
            hf.create_dataset(f"{base_cad}/regions_names", data=self.name)

        # Save the region colors
        if hasattr(self, "color"):
            hf.create_dataset(f"{base_cad}/regions_colors", data=self.color)

        # Save the parcellation centroids
        if hasattr(self, "centroids"):
            hf.create_dataset(f"{base_cad}/regions_centroids", data=self.centroids)

        # Save the timeseries if they exist
        if hasattr(self, "timeseries"):
            hf.create_dataset(f"{base_cad}/time_series", data=self.timeseries)

        # Close the file
        hf.close()

        # Save the morphometry DataFrame if it exists
        if hasattr(self, "morphometry"):
            cltmisc.save_morphometry_hdf5(
                out_file, "{base_cad}/morphometry", self.morphometry, mode="w"
            )

    ####################################################################################################
    def prepare_for_connectomics(
        self,
        mergectx: bool = True,
        output_file: str | Path | None = None,
        overwrite: bool = True,
    ) -> "Parcellation":
        """
        Prepare parcellation for connectomics analysis by merging cortical white matter
        labels to their corresponding cortical gray matter values.

        Converts white matter labels (>=3000) to corresponding gray matter labels
        by subtracting 3000, and removes other structures labels (>=5000). Useful for
        tractography/connectomics applications where the parcellation was generated
        using Chimera (https://github.com/connectomicslab/chimera), or any parcellation
        following the same labeling scheme:

        - Gray matter regions: 1-2999
        - White matter regions: 3000-4999
            - 3000: generic white matter label
            - 3000 + X: FreeSurfer cortical WM label corresponding to gray matter label X
        - Other structures: 5000-5008 (removed)
        - Corpus callosum: 5009-5013 (merged into the generic WM label, 3000)

        Parameters
        ----------
        mergectx : bool, default True
            If True, cortical WM labels are merged into their corresponding cortical
            GM labels via `merge_ctx_wm`. If False, WM voxels (>=3000) are zeroed out
            instead of merged.

        output_file : str or Path, optional
            If provided, saves the modified parcellation to this file. Must be a valid
            path to a writable location. If the file already exists, it will be overwritten.

        overwrite : bool, default True
            If True, allows overwriting the output file if it already exists. Ignored if
            `output_file` is None.

        Returns
        -------
        Parcellation
            self, to allow method chaining.

        Examples
        --------
        >>> parc.prepare_for_connectomics()
        >>> print(f"Max label after prep: {parc.data.max()}")
        """

        # Merge corpus callosum labels (5009-5013) into the generic WM label
        self.data[self.data >= 5009] = 3000

        # Remove any remaining "other structures" labels (5000-5008) through the
        # official API so color tables / name lookups stay in sync with self.data
        other_codes = (
            np.unique(self.data[(self.data >= 5000) & (self.data < 5009)])
            .astype(int)
            .tolist()
        )
        if other_codes:
            self.remove_by_code(codes2remove=other_codes)

        if mergectx:
            self.merge_ctx_wm()
            self.data[self.data == 3000] = 0
        else:
            self.data[self.data >= 3000] = 0

        self.adjust_values()

        if self.data.max() == 0:
            warnings.warn(
                "prepare_for_connectomics produced an empty parcellation — "
                "check that the input follows the expected Chimera labeling scheme.",
                stacklevel=2,
            )

        if output_file is not None:
            if not isinstance(output_file, (str, Path)):
                raise TypeError(
                    f"output_file must be a string or Path, got {type(output_file)}"
                )
            output_file = Path(output_file)
            if not output_file.parent.exists():
                raise FileNotFoundError(
                    f"Output directory does not exist: {output_file.parent}"
                )
            self.save_parcellation(out_file=output_file, overwrite=overwrite)

        return self

    ####################################################################################################
    def keep_by_name(self, names2keep: list | str, rearrange: bool = False):
        """
        Filter parcellation to keep only regions with specified names.

        Parameters
        ----------
        names2keep : str or list
            Name substring(s) to search for in region names.

        rearrange : bool, optional
            Whether to rearrange labels starting from 1. Default is False.

        Examples
        --------
        >>> # Keep only hippocampal regions
        >>> parc.keep_by_name('hippocampus')
        >>>
        >>> # Keep multiple regions and rearrange
        >>> parc.keep_by_name(['frontal', 'parietal'], rearrange=True)
        """

        if isinstance(names2keep, str):
            names2keep = [names2keep]

        if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
            # Find the indexes of the names that contain the substring
            indexes = cltmisc.get_indexes_by_substring(
                input_list=self.name,
                or_filter=names2keep,
                invert=False,
                bool_case=False,
            )

            if len(indexes) > 0:
                sel_st_codes = [self.index[i] for i in indexes]
                self.keep_by_code(codes2keep=sel_st_codes, rearrange=rearrange)
            else:
                print("The names were not found in the parcellation")

    #####################################################################################################
    def keep_by_code(
        self, codes2keep: str | list | np.ndarray, rearrange: bool = False
    ):
        """
        Filter parcellation to keep only specified region codes.

        Parameters
        ----------
        codes2keep : list or np.ndarray
            Region codes to retain in parcellation.

        rearrange : bool, optional
            Whether to rearrange labels consecutively from 1. Default is False.

        Raises
        ------
        ValueError
            If codes2keep is empty or contains invalid codes.

        Examples
        --------
        >>> # Keep specific regions
        >>> parc.keep_by_code([1, 2, 5, 10])
        >>>
        >>> # Keep and rearrange
        >>> parc.keep_by_code([100, 200, 300], rearrange=True)
        """

        # Validate codes2keep
        if isinstance(codes2keep, str):
            codes2keep = [codes2keep]

        # Convert codes2keep to numpy array
        if isinstance(codes2keep, list):
            codes2keep = cltmisc.build_indices(codes2keep)
            codes2keep = np.array(codes2keep)

        # Create a boolean mask for voxels to keep
        mask = np.isin(self.data, codes2keep)

        # Set elements to zero if they are not in the retain list
        self.data[~mask] = 0

        # Get the actual codes present in the filtered data (excluding 0)
        remaining_codes = np.unique(self.data)
        remaining_codes = remaining_codes[remaining_codes != 0]

        # Filter metadata arrays to match remaining codes
        if hasattr(self, "index"):
            temp_index = np.array(self.index)
            metadata_mask = np.isin(temp_index, remaining_codes)
            self.index = temp_index[metadata_mask].tolist()

            # Apply the same mask to other metadata arrays
            if hasattr(self, "name"):
                self.name = np.array(self.name)[metadata_mask].tolist()

            if hasattr(self, "color"):
                self.color = np.array(self.color)[metadata_mask].tolist()

            if hasattr(self, "opacity"):
                opacity_arr = np.array(self.opacity)
                # Guard: opacity must be a 1-D array with length matching the index
                if opacity_arr.ndim == 1 and len(opacity_arr) == len(metadata_mask):
                    self.opacity = opacity_arr[metadata_mask].tolist()
                else:
                    # Fallback: default opacity for the remaining regions
                    self.opacity = [1.0] * int(metadata_mask.sum())

        # If rearrange is True, the parcellation will be rearranged starting from 1
        if rearrange:
            self.rearrange()

        # Detect minimum and maximum labels
        self.parc_range()

    #####################################################################################################
    def names_to_labels(self, names: str | list[str]) -> list[int]:
        """
        Convert region names to their corresponding labels.

        Parameters
        ----------
        names : str or list of str
            Region names to convert to labels.

        Returns
        -------
        list of int
            Corresponding labels for the given region names.

        Examples
        --------
        >>> parc.names_to_label('ctx-lh-bankssts')
        [1]
        >>> parc.names_to_label(['ctx-lh-bankssts', 'ctx-rh-bankssts'])
        [1, 2]
        """
        if isinstance(names, str):
            names = [names]

        indexes = cltmisc.get_indexes_by_substring(self.name, names)

        return [self.index[i] for i in indexes]

    #####################################################################################################
    def labels_to_names(
        self, labels: int | list[int] | np.ndarray | str | list[str]
    ) -> list[str]:
        """
        Convert region labels to their corresponding names.
        The input labels can be integers, lists of integers, numpy arrays, strings, or lists of strings.
        Strings will be converted to their corresponding labels using the parcellation's name-to-label mapping.

        Parameters
        ----------
        labels : int, list of int, np.ndarray, str, or list of str
            Region labels or names to convert to names. Strings will be converted
            to their corresponding labels using the parcellation's name-to-label mapping.

        Returns
        -------
        list of str
            Corresponding region names for the given labels or names.

        Examples
        --------
        >>> parc.labels_to_names(1)
        ['ctx-lh-bankssts']
        >>> parc.labels_to_names([1, 2])
        ['ctx-lh-bankssts', 'ctx-rh-bankssts']
        """

        # Ensure code is a list of indices
        labels = cltmisc.build_indices(labels)

        missing = [label for label in labels if label not in self.index]
        if missing:
            raise ValueError(
                f"The following labels are not present in the parcellation: {missing}. "
                f"Available labels range from {self.minlab} to {self.maxlab}."
            )

        indexes = [self.index.index(label) for label in labels]

        return [self.name[i] for i in indexes]

    #####################################################################################################
    def get_voxels_by_code(
        self,
        labels: int | list[int] | np.ndarray | str | list[str],
        all_voxels: bool = True,
    ) -> np.ndarray | dict[int, np.ndarray]:
        """
        Get voxels corresponding to the specified region code(s).

        Parameters
        ----------
        labels : int | list[int] | np.ndarray | str | list[str]
            Region code(s) or name(s) to retrieve voxels for.

        all_voxels : bool, optional
            If True (default), return a single flat array containing the label
            value of every voxel matching any of the requested codes (voxels
            from different codes are merged together — the original behavior).
            If False, return a dictionary mapping each requested code that is
            actually present in the data to an (N, 3) array of its voxel
            coordinates (i, j, k). Codes not present in the data are simply
            omitted from the dictionary — no error is raised for missing codes.

        Returns
        -------
        np.ndarray or dict of {int : np.ndarray}
            If all_voxels is True: 1-D array of label values for every matching
            voxel. If all_voxels is False: dictionary of {code: voxel_coords}
            for the codes that exist, where voxel_coords has shape
            (n_voxels_for_that_code, 3).

        Examples
        --------
        >>> # Flat array of matching voxel values (original behavior)
        >>> parc.get_voxels_by_code([1, 2])
        array([1, 1, 1, ..., 2, 2, 2])
        >>>
        >>> # Per-region voxel coordinates, existing codes only
        >>> parc.get_voxels_by_code([1, 2, 999], all_voxels=False)
        {1: array([[10, 20, 15], ...]), 2: array([[30, 40, 25], ...])}
        """

        # Ensure code is a list of indices
        labels = cltmisc.build_indices(labels)

        if all_voxels:
            return self.data[np.isin(self.data, labels)]

        present_lables = set(np.unique(self.data).tolist())
        existing_labels = [c for c in labels if c in present_lables]

        return {c: np.argwhere(self.data == c) for c in existing_labels}

    #####################################################################################################
    def get_voxels_by_name(
        self,
        names: str | list[str],
        all_voxels: bool = True,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """
        Get voxels corresponding to the specified region name(s).

        Parameters
        ----------
        names : str or list of str
            Region name(s) to retrieve voxels for.

        all_voxels : bool, optional
            If True (default), return a single flat array containing the label
            value of every voxel matching any of the specified names (voxels
            from different regions are merged together — the original behavior).
            If False, return a dictionary mapping each matched region name to
            an (N, 3) array of its voxel coordinates (i, j, k). Only names that
            actually match a region present in the data are included — no error
            is raised for names that don't match anything.

        Returns
        -------
        np.ndarray or dict of {str : np.ndarray}
            If all_voxels is True: 1-D array of label values for every matching
            voxel. If all_voxels is False: dictionary of {name: voxel_coords}
            for the region names that exist, where voxel_coords has shape
            (n_voxels_for_that_region, 3).

        Examples
        --------
        >>> # Flat array of matching voxel values (original behavior)
        >>> parc.get_voxels_by_name(['bankssts'])
        array([1, 1, 1, ...])
        >>>
        >>> # Per-region voxel coordinates, existing names only
        >>> parc.get_voxels_by_name(['bankssts', 'not-a-real-region'], all_voxels=False)
        {'ctx-lh-bankssts': array([[10, 20, 15], ...])}
        """
        if isinstance(names, str):
            names = [names]

        indexes = cltmisc.get_indexes_by_substring(self.name, names)
        codes = [self.index[i] for i in indexes]
        matched_names = [self.name[i] for i in indexes]

        if all_voxels:
            return self.get_voxels_by_code(codes)

        voxel_dict_by_code = self.get_voxels_by_code(codes, all_voxels=False)

        return {
            name: voxel_dict_by_code[code]
            for name, code in zip(matched_names, codes, strict=False)
            if code in voxel_dict_by_code
        }

    #####################################################################################################
    def remove_by_code(
        self, codes2remove: str | list | np.ndarray, rearrange: bool = False
    ):
        """
        Remove regions with specified codes from parcellation.

        Parameters
        ----------
        codes2remove : list or np.ndarray
            Region codes to remove from parcellation.

        rearrange : bool, optional
            Whether to rearrange remaining labels from 1. Default is False.

        Examples
        --------
        >>> # Remove specific regions
        >>> parc.remove_by_code([1, 5, 10])
        >>>
        >>> # Remove and rearrange
        >>> parc.remove_by_code([100, 200], rearrange=True)
        """
        # Validate codes2remove
        if isinstance(codes2remove, str):
            codes2remove = [codes2remove]

        # Convert codes2remove to numpy array
        if isinstance(codes2remove, list):
            codes2remove = cltmisc.build_indices(codes2remove)
            codes2remove = np.array(codes2remove)

        # Set voxels with codes to remove to 0
        self.data[np.isin(self.data, codes2remove)] = 0

        # Get remaining codes (excluding 0)
        remaining_codes = np.unique(self.data)
        remaining_codes = remaining_codes[remaining_codes != 0]

        # Use keep_by_code to clean up metadata
        # (parc_range is called by keep_by_code, so no need to call it again)
        self.keep_by_code(codes2keep=remaining_codes, rearrange=rearrange)

    #####################################################################################################
    def remove_by_name(self, names2remove: list | str, rearrange: bool = False):
        """
        Remove regions with specified names from parcellation.

        Parameters
        ----------
        names2remove : str or list
            Name substring(s) to search for removal.

        rearrange : bool, optional
            Whether to rearrange remaining labels from 1. Default is False.

        Examples
        --------
        >>> # Remove ventricles
        >>> parc.remove_by_name('ventricle')
        >>>
        >>> # Remove multiple structures
        >>> parc.remove_by_name(['csf', 'unknown'], rearrange=True)
        """

        # Convert single string to list
        if isinstance(names2remove, str):
            names2remove = [names2remove]

        # Check required attributes
        if not (hasattr(self, "name") and hasattr(self, "index")):
            raise AttributeError(
                "Parcellation must have 'name' and 'index' attributes to remove by name"
            )

        # Get indexes of regions whose names contain the substrings to remove
        indexes_to_remove = cltmisc.get_indexes_by_substring(
            input_list=self.name, or_filter=names2remove, invert=False, bool_case=False
        )

        if len(indexes_to_remove) == 0:
            print(f"No regions found matching: {names2remove}")
            return

        # Get the codes corresponding to the regions to remove
        codes_to_remove = [self.index[i] for i in indexes_to_remove]

        # Remove the regions using remove_by_code
        # (parc_range is called by remove_by_code, so no need to call it again)
        self.remove_by_code(codes2remove=codes_to_remove, rearrange=rearrange)

    #####################################################################################################
    def apply_mask(
        self,
        image_mask: "str | Path | np.ndarray | Parcellation",
        mask_codes: str | list | np.ndarray = None,
        invert: bool = False,
        fill: bool = False,
    ):
        """
        Apply spatial mask to restrict parcellation to specific regions.

        Parameters
        ----------
        image_mask : str, Path, np.ndarray, or Parcellation
            3D mask array, parcellation object, or path to mask file.
            Can be binary mask (0/1) or labeled image with region codes.

        mask_codes : list or np.ndarray, optional
            Specific codes in the mask image to use for masking.
            If None, uses all non-zero values in mask. Default is None.

        invert : bool, optional
            If False, keep only voxels where mask has specified codes.
            If True, remove voxels where mask has specified codes.
            Default is False.

        fill : bool, optional
            Whether to grow regions to fill mask using region growing.
            Default is False.

        Raises
        ------
        ValueError
            If mask file doesn't exist or shapes don't match.

        Examples
        --------
        >>> # Apply binary cortical mask
        >>> parc.apply_mask(cortex_mask)
        >>>
        >>> # Mask using specific regions from another parcellation
        >>> parc.apply_mask(roi_parc, mask_codes=[1, 2, 3])
        >>>
        >>> # Inverse masking with region growing
        >>> parc.apply_mask(exclusion_mask, invert=True, fill=True)
        """

        # Load mask data
        if isinstance(image_mask, (str, Path)):
            image_mask = str(image_mask)
            if not os.path.exists(image_mask):
                raise ValueError(f"Mask file does not exist: {image_mask}")

            temp_mask = nib.load(image_mask)
            mask_data = temp_mask.get_fdata()

        elif isinstance(image_mask, np.ndarray):
            mask_data = image_mask

        elif isinstance(image_mask, Parcellation):
            mask_data = image_mask.data

        else:
            raise ValueError(
                "image_mask must be a file path, numpy array, or Parcellation object"
            )

        # Validate shape compatibility
        if mask_data.shape != self.data.shape:
            raise ValueError(
                f"Mask shape {mask_data.shape} doesn't match parcellation shape {self.data.shape}"
            )

        # Determine which codes in the mask to use
        if mask_codes is None:
            # Use all non-zero values in the mask
            mask_codes = np.unique(mask_data)
            mask_codes = mask_codes[mask_codes != 0]
        else:
            # Convert to standardized format
            if isinstance(mask_codes, str):
                mask_codes = [mask_codes]
            mask_codes = cltmisc.build_indices(mask_codes)
            mask_codes = np.array(mask_codes)

        # Create boolean mask for regions to keep
        bool_mask = np.isin(mask_data, mask_codes)

        # Apply masking
        if invert:
            # Remove voxels where mask contains specified codes. If filling, the
            # area that was just excised is exactly what needs new labels grown
            # into it, so keep it (not its complement) as the fill target.
            fill_target = bool_mask.copy()
            self.data[bool_mask] = 0
        else:
            # Keep only voxels where mask contains specified codes. If filling,
            # any label-less gaps inside that ROI are what needs growing.
            self.data[~bool_mask] = 0
            fill_target = bool_mask

        # Optional region growing to fill the mask
        if fill:
            self.data = cltimg.region_growing(self.data, fill_target)

        # Adjust parcellation values
        self.adjust_values()

        # Update parcellation range
        self.parc_range()

    ####################################################################################################
    def mask_image(
        self,
        image_2mask: str | Path | list[str | Path] | np.ndarray,
        masked_image: str | Path | list | None = None,
        region_labels: str | list | np.ndarray = None,
        region_names: str | list = None,
        invert: bool = False,
    ) -> np.ndarray | list:
        """
        Mask external images using parcellation as binary mask.

        Parameters
        ----------
        image_2mask : str, Path, list of str/Path, or np.ndarray
            Image(s) to mask using parcellation. Can be file path(s) or a numpy
            array. Arrays may be 3D (matching the parcellation's spatial shape)
            or 4D (e.g. an fMRI time series) - for 4D input, every volume is
            masked identically using the parcellation's 3D mask.

        masked_image : str, Path, list, optional
            Output path(s) for masked images. Required when image_2mask is path(s).
            Ignored when image_2mask is numpy array. Default is None.

        region_labels : str, list or np.ndarray, optional
            Region codes to use for masking. Default is None (all non-zero regions).

        region_names : str or list, optional
            Region names to use for masking. Default is None.

        invert : bool, optional
            If False, keep only voxels within specified regions.
            If True, remove voxels within specified regions.
            Default is False.

        Returns
        -------
        np.ndarray or list
            If image_2mask is numpy array, returns masked array.
            If image_2mask is path(s), returns list of output paths.

        Raises
        ------
        ValueError
            If both region_labels and region_names are specified, if none of the
            requested region_labels/region_names match a region in the parcellation,
            if image_2mask is an empty list, if output paths don't match input
            paths in length, or if files don't exist, or shapes don't match.

        Examples
        --------
        >>> # Mask T1 image with all parcellation regions
        >>> parc.mask_image('T1w.nii.gz', 'T1w_masked.nii.gz')
        ['T1w_masked.nii.gz']

        >>> # Mask with specific region codes
        >>> parc.mask_image('fmri.nii.gz', 'fmri_masked.nii.gz', region_labels=[1, 2, 3])
        ['fmri_masked.nii.gz']

        >>> # Mask with specific region names
        >>> parc.mask_image('dwi.nii.gz', 'dwi_masked.nii.gz', region_names=['cortex', 'hippocampus'])
        ['dwi_masked.nii.gz']

        >>> # Inverted masking (remove specific regions)
        >>> parc.mask_image('dwi.nii.gz', 'dwi_masked.nii.gz', region_labels=[5, 6], invert=True)
        ['dwi_masked.nii.gz']

        >>> # Mask numpy array
        >>> masked_data = parc.mask_image(img_array, region_labels=[10, 20])
        """

        # Normalize image_2mask to list
        if isinstance(image_2mask, (str, Path)):
            image_2mask = [image_2mask]

        if isinstance(image_2mask, list) and len(image_2mask) == 0:
            raise ValueError("image_2mask cannot be an empty list")

        is_file_input = isinstance(image_2mask, list) and isinstance(
            image_2mask[0], (str, Path)
        )

        # Handle masked_image paths
        if is_file_input:
            if masked_image is None:
                raise ValueError(
                    "masked_image output path(s) required when image_2mask is file path(s)"
                )

            if isinstance(masked_image, (str, Path)):
                masked_image = [masked_image]

            # Convert all paths to strings
            image_2mask = [str(p) for p in image_2mask]
            masked_image = [str(p) for p in masked_image]

            if len(masked_image) != len(image_2mask):
                raise ValueError(
                    f"Number of output paths ({len(masked_image)}) must match "
                    f"number of input images ({len(image_2mask)})"
                )

        # Check if both inclusion criteria are specified
        if region_labels is not None and region_names is not None:
            # If both are specified, prioritize region_labels and ignore region_names
            region_names = None

        # Determine which codes to use for masking
        if region_labels is not None:
            # Use specified codes
            if isinstance(region_labels, str):
                region_labels = [region_labels]
            codes_to_use = cltmisc.build_indices(region_labels)
            codes_to_use = np.array(codes_to_use)

            # Unlike the region_names branch below, np.isin silently matches nothing
            # if none of these codes exist in the data - with invert=False (the
            # default) that would zero out the ENTIRE image with no warning, so
            # check explicitly rather than letting it fail silently.
            present_codes = set(np.unique(self.data).tolist())
            if not any(c in present_codes for c in codes_to_use.tolist()):
                raise ValueError(
                    f"None of the requested region_labels were found in the "
                    f"parcellation: {region_labels}"
                )

        elif region_names is not None:
            # Get codes from names
            if isinstance(region_names, str):
                region_names = [region_names]

            if not hasattr(self, "name") or not hasattr(self, "index"):
                raise ValueError(
                    "Parcellation must have 'name' and 'index' attributes to use region_names"
                )

            # Find indexes of matching names
            indexes = cltmisc.get_indexes_by_substring(
                input_list=self.name,
                or_filter=region_names,
                invert=False,
                bool_case=False,
            )

            if len(indexes) == 0:
                raise ValueError(f"No regions found matching names: {region_names}")

            codes_to_use = np.array([self.index[i] for i in indexes])

        else:
            # Use all non-zero codes
            codes_to_use = np.unique(self.data)
            codes_to_use = codes_to_use[codes_to_use != 0]

        # Create boolean mask of voxels to zero out
        if invert:
            # Remove voxels with specified codes
            voxels_to_zero = np.isin(self.data, codes_to_use)
        else:
            # Keep only voxels with specified codes
            voxels_to_zero = ~np.isin(self.data, codes_to_use)

        # Process file inputs
        if is_file_input:
            output_paths = []

            for img_path, out_path in zip(image_2mask, masked_image, strict=False):
                if not os.path.exists(img_path):
                    raise ValueError(f"Image file does not exist: {img_path}")

                # Load image
                temp_img = nib.load(img_path)
                img_data = temp_img.get_fdata()

                # Validate shape
                if img_data.shape[:3] != self.data.shape:
                    raise ValueError(
                        f"Image shape {img_data.shape[:3]} doesn't match "
                        f"parcellation shape {self.data.shape}"
                    )

                # Apply mask. voxels_to_zero is always 3D; for a 4D img_data
                # (e.g. an fMRI series) numpy's boolean-indexing rule zeroes
                # every volume at each matching voxel, which is the intended
                # behavior for both 3D and 4D inputs.
                img_data[voxels_to_zero] = 0

                # Save masked image
                out_img = nib.Nifti1Image(img_data, temp_img.affine, temp_img.header)
                nib.save(out_img, out_path)
                output_paths.append(out_path)

            return output_paths

        # Process numpy array input
        elif isinstance(image_2mask, np.ndarray):
            # Validate shape
            if image_2mask.shape[:3] != self.data.shape:
                raise ValueError(
                    f"Image shape {image_2mask.shape[:3]} doesn't match "
                    f"parcellation shape {self.data.shape}"
                )

            # Create copy to avoid modifying input
            img_data = image_2mask.copy()
            img_data[voxels_to_zero] = 0

            return img_data

        else:
            raise ValueError(
                "image_2mask must be a file path, Path object, list of file "
                "paths/Path objects, or numpy array"
            )

    #####################################################################################################
    def compute_region_adjacency(
        self,
        region_labels: list[int] | np.ndarray = None,
        region_names: list[str] | str = None,
        rearrange: bool = False,
        weighted: bool = False,
        name: str = None,
    ) -> tuple["cltcon.Connectome", dict, dict]:
        """
        Computes the region adjacency (neighbor) matrix for the parcellation.

        Two regions are considered neighbors when the dilation of one of them by a
        single voxel reaches the other. The resulting matrix, binary or weighted, is
        returned as a Connectome object whose nodes are labeled by the parcellation
        codes: the matrix is N x N, with N the maximum label present in the image, so
        the row and the column of a region are always its code minus one. Labels that
        are absent from the image keep an empty row and column, which makes the
        matrices of different subjects directly comparable as long as they share the
        same labeling scheme.

        The names, the colors and the centroids of the regions that remain in the
        parcellation are attached to the Connectome object, so it carries everything
        needed to plot the matrix or the network.

        Parameters
        ----------
        region_labels : list or np.ndarray, optional
            Specific region codes to include. Default is None (all regions).

        region_names : list or str, optional
            Specific region names to include. Default is None.

        rearrange : bool, optional
            Whether to rearrange the parcellation labels before computing the
            adjacency. If True, the regions kept are relabeled from 1 to the number
            of regions, which gives the most compact matrix. Default is False.

        weighted : bool, optional
            If False, the matrix is binary (1 for neighboring regions). If True, the
            weight of a pair of regions A and B is the number of interface voxels:
            the voxels of B that fall in the one-voxel dilation shell of A, plus the
            voxels of A that fall in the dilation shell of B. The matrix is symmetric,
            and binarizing it gives the same matrix as weighted=False.
            Default is False.

        name : str, optional
            Name of the resulting Connectome object. If None, it is derived from the
            parcellation identifier. Default is None.

        Returns
        -------
        adjacency : cltcon.Connectome
            Connectome object with the binary or weighted adjacency matrix (N x N, N
            being the maximum label present in the image), the region names, colors,
            centroid coordinates in mm and the affine of the parcellation. Its
            connectivity type is set to "adjacency" or "adjacency-weighted".

        source : dict
            Dictionary with the row indices ('idxs'), the codes ('codes'), the names
            ('names') of the first region of every neighboring pair, and the number
            of interface voxels of the pair ('weights'). The weights are reported
            even when weighted=False.

        target : dict
            Row indices ('idxs'), codes ('codes') and names ('names') of the second
            region of every neighboring pair. Each pair is reported once, even though
            the matrix is symmetric.

        Raises
        ------
        ValueError
            If both region_labels and region_names are specified, or if the parcellation
            does not contain any labeled voxel.

        Notes
        -----
        The centroids are the centers of mass of the regions, converted to mm with
        the affine of the parcellation. They are computed in a single pass, unlike
        `compute_centroids`, which refines them region by region.

        The interface voxel counts depend on the connectivity of the structuring
        element used by the dilation. With a 26-connected element, voxels that touch
        another region only through an edge or a corner are also counted. Larger
        regions naturally share more interface voxels, so normalize the weights if
        pairs of regions with very different sizes have to be compared.

        The size of the matrix follows the labeling scheme, not the number of
        regions. Parcellations with sparse codes, such as the FreeSurfer ones, can
        therefore produce large matrices, and a warning is issued when the matrix
        needs more than 256 MB. Use rearrange=True to obtain a compact matrix.

        Examples
        --------
        >>> # Binary adjacency of the whole parcellation
        >>> adjacency, source, target = parc.compute_region_adjacency()
        >>> adjacency.plot_matrix(figsize=(5, 4))
        >>>
        >>> # Weighted adjacency: number of interface voxels between regions
        >>> adjacency_w, source, target = parc.compute_region_adjacency(weighted=True)
        >>> adjacency_w.plot_matrix(figsize=(5, 4), log_scale=True)
        >>>
        >>> # Compact matrix, with one row per region kept
        >>> adjacency, source, target = parc.compute_region_adjacency(
        ...     region_names="ctx", rearrange=True, weighted=True
        ... )
        >>> for s, t, w in zip(source["names"], target["names"], source["weights"]):
        ...     print(f"{s} - {t}: {w} voxels")
        """

        from scipy import ndimage

        from .imagetools import MorphologicalOperations

        # Check if both inclusion criteria are specified
        if region_labels is not None and region_names is not None:
            raise ValueError(
                "Cannot specify both region_labels and region_names. Please choose one."
            )

        # Work on a copy to avoid modifying original
        temp_parc = copy.deepcopy(self)

        # Apply filtering if specified
        if region_labels is not None:
            temp_parc.keep_by_code(codes2keep=region_labels, rearrange=rearrange)

        if region_names is not None:
            temp_parc.keep_by_name(names2keep=region_names, rearrange=rearrange)

        data = temp_parc.data

        # Integer view of the volume, needed to locate and group the labels
        if np.issubdtype(data.dtype, np.integer):
            label_volume = data
        else:
            label_volume = data.astype(np.int32)

        # Codes that are really present in the image. They define the size of the
        # matrix, which is the maximum label found in the image.
        present_codes = np.unique(label_volume[label_volume > 0]).astype(int)

        if present_codes.size == 0:
            raise ValueError(
                "The parcellation does not contain any labeled voxel. "
                "The adjacency matrix cannot be computed."
            )

        n_nodes = int(present_codes.max())

        # The Connectome object stores the matrix as float64. The count matrix used
        # below is int64, so the peak memory is roughly three times this size.
        matrix_size = n_nodes * n_nodes * 8
        if matrix_size > 256 * 1024**2:
            warnings.warn(
                f"The maximum label in the image ({n_nodes}) leads to a matrix of "
                f"{matrix_size / 1024**3:.1f} GB. Use rearrange=True to relabel the "
                "regions and obtain a compact matrix.",
                stacklevel=2,
            )

        # counts[i, j]: number of voxels of region j+1 inside the one-voxel dilation
        # shell of region i+1. This matrix is not symmetric.
        counts = np.zeros((n_nodes, n_nodes), dtype=np.int64)

        # Bounding box of every label. The shell of a region never reaches further
        # than one voxel outside its bounding box, so dilating the whole volume for
        # every region is unnecessary and, on a large image, very expensive.
        bounding_boxes = ndimage.find_objects(label_volume)

        morph = MorphologicalOperations()
        for region_code in present_codes:

            # Sub-volume holding the region and the voxels its dilation can reach
            bbox = bounding_boxes[region_code - 1]
            sub_slices = tuple(
                slice(max(axis_slice.start - 1, 0), min(axis_slice.stop + 1, dim))
                for axis_slice, dim in zip(bbox, label_volume.shape, strict=False)
            )
            sub_volume = label_volume[sub_slices]

            # Binary mask of the region of interest
            region_mask = sub_volume == region_code

            # Dilate the region by 1 voxel
            dilated_mask = morph.dilate(region_mask.astype(int), iterations=1)

            # Boundary shell: dilated area minus the original region
            boundary = (dilated_mask == 1) & (region_mask == 0)

            # Count the voxels of every label in the shell. Index 0 is background,
            # and the region itself cannot appear because the shell excludes it.
            shell_labels = sub_volume[boundary].astype(np.int64)
            label_counts = np.bincount(shell_labels, minlength=n_nodes + 1)

            counts[region_code - 1, :] = label_counts[1 : n_nodes + 1]

        # Symmetric interface size: voxels of both regions touching each other
        interface_voxels = counts + counts.T
        np.fill_diagonal(interface_voxels, 0)
        del counts

        # Unique neighboring pairs (upper triangle, smaller code first). np.nonzero
        # returns them in row-major order, sorted by the first and then the second code.
        rows, cols = np.nonzero(np.triu(interface_voxels, k=1))
        unique_pairs = np.column_stack((rows + 1, cols + 1))

        # Names and colors of the regions still in the parcellation. The rows of the
        # labels that are not in the table are filled with placeholders.
        node_codes = np.arange(1, n_nodes + 1, dtype=int)
        node_names = cltmisc.create_names_from_indices(
            node_codes, prefix="unused-label"
        )
        node_colors = ["#000000"] * n_nodes

        for i, region_code in enumerate(temp_parc.index):
            region_code = int(region_code)
            if 1 <= region_code <= n_nodes:
                node_names[region_code - 1] = temp_parc.name[i]
                node_colors[region_code - 1] = temp_parc.color[i]

        # Centroids, as centers of mass of every region present in the image. They
        # are computed in a single pass and converted to mm.
        node_coords = np.full((n_nodes, 3), np.nan)
        centroids_vox = np.array(
            ndimage.center_of_mass(
                label_volume > 0, labels=label_volume, index=present_codes
            )
        )
        node_coords[present_codes - 1, :] = cltimg.vox2mm(
            centroids_vox, temp_parc.affine
        )

        # Adjacency matrix. The row of a region is its code minus one.
        if weighted:
            neighb_matrix = interface_voxels.astype(np.float64)
        else:
            neighb_matrix = (interface_voxels > 0).astype(np.float64)

        # Create source and target dictionaries
        source = {"idxs": [], "codes": [], "names": [], "weights": []}
        target = {"idxs": [], "codes": [], "names": []}

        for code1, code2 in unique_pairs:
            code1, code2 = int(code1), int(code2)

            # Row indices of both regions in the matrix
            roi_idx1 = code1 - 1
            roi_idx2 = code2 - 1

            # Store the pair (only once, not symmetric)
            source["idxs"].append(roi_idx1)
            source["codes"].append(code1)
            source["names"].append(node_names[roi_idx1])
            source["weights"].append(int(interface_voxels[roi_idx1, roi_idx2]))

            target["idxs"].append(roi_idx2)
            target["codes"].append(code2)
            target["names"].append(node_names[roi_idx2])

        # Name of the resulting Connectome object
        if name is None:
            parc_id = getattr(self, "id", None)
            suffix = "adjacency-weighted" if weighted else "adjacency"
            name = f"{parc_id}-{suffix}" if parc_id else f"region-{suffix}"

        adjacency = cltcon.Connectome(
            matrix=neighb_matrix,
            name=name,
            region_coords=node_coords,
            region_names=node_names,
            region_index=node_codes,
            region_colors=node_colors,
            modality="adjacency-weighted" if weighted else "adjacency",
            affine=temp_parc.affine,
        )

        return adjacency, source, target

    ######################################################################################################
    def compute_centroids(
        self,
        region_labels: list[int] | np.ndarray = None,
        region_names: list[str] | str = None,
        gaussian_smooth: bool = True,
        sigma: float = 1.0,
        closing_iterations: int = 2,
        centroid_table: str | Path | None = None,
        output_format: str = "dataframe",  # Options: "dataframe", "pointcloud", "none"
    ) -> pd.DataFrame | PointCloud | None:
        """
        Compute region centroids, voxel counts, and volumes.

        Parameters
        ----------
        region_labels : list or np.ndarray, optional
            Specific region codes to include. Default is None (all regions).

        region_names : list or str, optional
            Specific region names to include. Default is None.

        gaussian_smooth : bool, optional
            Whether to apply Gaussian smoothing before centroid calculation. Default is True.

        sigma : float, optional
            Standard deviation for Gaussian smoothing. Default is 1.0.

        closing_iterations : int, optional
            Number of morphological closing iterations. Default is 2.

        centroid_table : str or Path, optional
            Path to save results as TSV file. Default is None.

        Returns
        -------
        pd.DataFrame or PointCloud or None
            DataFrame with columns: index, name, color, x_vox, y_vox, z_vox,
            x_mm, y_mm, z_mm, nvoxels, volume.
            If output_format is "pointcloud", returns a PointCloud object instead.

        Raises
        ------
        ValueError
            If both region_labels and region_names are specified.

        Notes
        -----
        This method sets the `centroids` attribute (Nx3 array in mm or voxel coordinates).

        Examples
        --------
        >>> # Compute all centroids
        >>> centroids_df = parc.compute_centroids()
        >>>
        >>> # Specific regions with file output
        >>> df = parc.compute_centroids(
        ...     region_labels=[1, 2, 3],
        ...     centroid_table='centroids.tsv'
        ... )
        >>> # Specific regions by name
        >>> df = parc.compute_centroids(
        ...     region_names=['hippocampus', 'amygdala'],
        ...     centroid_table='centroids.tsv'
        ... )
        """

        # Check if both inclusion criteria are specified
        if region_labels is not None and region_names is not None:
            raise ValueError(
                "Cannot specify both region_labels and region_names. Please choose one."
            )

        # Work on a copy to avoid modifying original
        temp_parc = copy.deepcopy(self)

        # Apply filtering if specified
        if region_labels is not None:
            temp_parc.keep_by_code(codes2keep=region_labels)

        if region_names is not None:
            temp_parc.keep_by_name(names2keep=region_names)

        # Get region information
        region_codes = np.array(temp_parc.index)
        len(region_codes)

        # Initialize result lists
        codes = []
        names = []
        colors = []
        x_coords_vox = []
        y_coords_vox = []
        z_coords_vox = []
        num_voxels = []
        volumes = []

        # Get voxel size
        voxel_volume = cltimg.get_voxel_volume(temp_parc.affine)

        # Iterate over regions - indices align with temp_parc.index
        for i, region_code in enumerate(region_codes):
            # Extract centroid and voxel count for this region
            centroid_vox, voxel_count = cltimg.extract_centroid_from_volume(
                temp_parc.data == region_code,
                gaussian_smooth=gaussian_smooth,
                sigma=sigma,
                closing_iterations=closing_iterations,
            )

            # Calculate total volume
            total_volume = voxel_count * voxel_volume

            # Store results
            codes.append(int(region_code))
            names.append(temp_parc.name[i])
            colors.append(temp_parc.color[i])
            x_coords_vox.append(centroid_vox[0])
            y_coords_vox.append(centroid_vox[1])
            z_coords_vox.append(centroid_vox[2])
            num_voxels.append(voxel_count)
            volumes.append(total_volume)

        # Convert voxel coordinates to mm
        coords_vox = np.stack(
            (np.array(x_coords_vox), np.array(y_coords_vox), np.array(z_coords_vox)),
            axis=-1,
        )
        coords_mm = cltimg.vox2mm(coords_vox, self.affine)

        # Store centroids as attribute (in mm)
        self.centroids = coords_mm.astype(float)

        # Extract mm coordinates
        x_coords_mm = coords_mm[:, 0].tolist()
        y_coords_mm = coords_mm[:, 1].tolist()
        z_coords_mm = coords_mm[:, 2].tolist()

        # Create DataFrame
        df = pd.DataFrame(
            {
                "index": codes,
                "name": names,
                "color": colors,
                "x_vox": x_coords_vox,
                "y_vox": y_coords_vox,
                "z_vox": z_coords_vox,
                "x_mm": x_coords_mm,
                "y_mm": y_coords_mm,
                "z_mm": z_coords_mm,
                "nvoxels": num_voxels,
                "volume": volumes,
            }
        )

        # Save to TSV file if path is provided
        if centroid_table is not None:
            centroid_table = str(centroid_table)  # Convert Path to str

            # Check if directory exists
            directory = os.path.dirname(centroid_table)
            if directory and not os.path.exists(directory):
                print(f"Directory does not exist: {directory}.")
                return df

            try:
                df.to_csv(centroid_table, sep="\t", index=False)
                print(f"Centroid table saved to: {centroid_table}")
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to save centroid table: {e}", UserWarning, stacklevel=2
                )
        if output_format == "dataframe":
            return df
        elif output_format == "pointcloud":
            return PointCloud(
                points=coords_mm,
                region_colors=colors,
                region_names=names,
                name="centroids",
                affine=self.affine,
            )

    ######################################################################################################
    def get_regionwise_timeseries(
        self,
        time_series_data: str | np.ndarray,
        vols_to_delete: list[int] | np.ndarray = None,
        method: str = "nilearn",
        metric: str = "mean",
        region_labels: list[int] | np.ndarray = None,
        region_names: list[str] | str = None,
    ) -> np.ndarray:
        """
        Compute region-wise time series.

        Parameters
        ----------
        time_series_data : str or np.ndarray
            Path to time series file or numpy array with shape (dimx X dimy X dimZ x Timepoints).

        region_labels : list or np.ndarray, optional
            Specific region codes to include. Default is None (all regions).

        region_names : list or str, optional
            Specific region names to include. Default is None.

        ouput_h5file : str, optional
            Path to save results as HDF5 file. Default is None.

        Returns
        -------
        np.ndarray
            Region-wise time series array with shape (n_regions x timepoints).

        Raises
        ------
        ValueError
            If both region_labels and region_names are specified.

        Examples
        --------
        >>> # Compute region-wise time series from file
        >>> region_ts = parc.get_regionwise_timeseries('timeseries.nii.gz')
        >>>
        # Compute from numpy array
        >>> region_ts = parc.get_regionwise_timeseries(time_series_data=np.random.rand(64, 64, 64, 100))
        >>>
        # Compute with specific regions using codes
        >>> region_ts = parc.get_regionwise_timeseries(
        ...     time_series_data='timeseries.nii.gz',
        ...     region_labels=[1, 2, 3])

        """

        # Check if include_by_code and include_by_name are different from None at the same time
        if region_labels is not None and region_names is not None:
            region_labels = None
            print(
                "Both region_labels and region_names were specified. Ignoring region_labels and using region_names for region selection."
            )

        temp_parc = copy.deepcopy(self)

        # Apply inclusion if specified
        if region_labels is not None:
            temp_parc.keep_by_code(codes2keep=region_labels)

        if region_names is not None:
            temp_parc.keep_by_name(names2keep=region_names)

        # Delete volumes if specified
        if vols_to_delete is not None:

            # Check if the time_series_data is a string
            if isinstance(time_series_data, str):
                # Generating a temporary file to save the 4D data
                tmp_image = cltmisc.create_temporary_filename(
                    prefix="temp_timeseries",
                    extension=".nii.gz",
                )
                if method == "nilearn":
                    # Deleting the volumes from the 4D image
                    cltimg.delete_volumes_from_4D_images(
                        in_image=time_series_data,
                        out_image=tmp_image,
                        vols_to_delete=vols_to_delete,
                    )
                    time_series_data_tmp = tmp_image
                elif method == "clabtoolkit":
                    # Deleting the volumes from the 4D image and loading it as a numpy array

                    # Load the 4D image
                    img = nib.load(time_series_data)

                    # Get the dimensions of the image

                    time_series_data_tmp, _ = cltimg.delete_volumes_from_4D_array(
                        in_array=img.get_fdata(), vols_to_delete=vols_to_delete
                    )

            elif isinstance(time_series_data, np.ndarray):
                time_series_data_tmp, _ = cltimg.delete_volumes_from_4D_array(
                    in_array=time_series_data, vols_to_delete=vols_to_delete
                )
        else:
            time_series_data_tmp = time_series_data

        if method == "nilearn":

            if isinstance(time_series_data_tmp, str):

                # Check if the file exists
                try:
                    from nilearn.maskers import NiftiLabelsMasker
                except Exception as err:
                    raise ImportError(
                        "nilearn is not installed. Please install it to use this method."
                    ) from err

                # Generating a temporary parcellation file
                tmp_parc_image = cltmisc.create_temporary_filename(
                    prefix="temp_parcellation", extension=".nii.gz"
                )
                tmp_basename = cltmisc.get_real_basename(tmp_parc_image)
                tmp_parc_image_nilearnlut = os.path.join(
                    tempfile.gettempdir(), f"{tmp_basename}_nilearnlut.txt"
                )
                temp_parc.save_parcellation(
                    out_file=tmp_parc_image,
                    lut_file=tmp_parc_image_nilearnlut,
                    lut_type="nilearn",
                    overwrite=True,
                )

                # Generating the masker
                masker = NiftiLabelsMasker(
                    labels_img=tmp_parc_image,
                    lut=tmp_parc_image_nilearnlut,
                    standardize="zscore_sample",
                    standardize_confounds=True,
                    memory="nilearn_cache",
                    verbose=1,
                )

                # Check if the parcellation is a numpy array
                region_time_series = masker.fit_transform(time_series_data_tmp).T

                # Delete the temporary files
                os.remove(tmp_parc_image)
                os.remove(tmp_parc_image_nilearnlut)

            elif isinstance(time_series_data_tmp, np.ndarray):
                print(
                    "Using nilearn method requires a file path. Please provide a valid file path."
                )
                print(
                    "Computing region-wise timeseries without using nilearn. This may take longer."
                )
                method = "clabtoolkit"

        if method.lower() != "nilearn":

            # Get unique region values
            unique_regions = np.array(temp_parc.index)

            # Load time series data
            if isinstance(time_series_data_tmp, str):
                if os.path.exists(time_series_data_tmp):
                    time_series = nib.load(time_series_data_tmp).get_fdata()

                else:
                    raise ValueError("The time series file does not exist")

            elif isinstance(time_series_data_tmp, np.ndarray):
                time_series = time_series_data_tmp

            else:
                raise ValueError(
                    "time_series_data must be a string (file path) or a numpy array"
                )

            # Check if time series has 4 dimensions
            if time_series.ndim != 4:
                raise ValueError(
                    "Time series data must have 4 dimensions (dimx, dimy, dimz, timepoints)"
                )
            # Check if time series dimensions match parcellation dimensions
            if time_series.shape[:3] != temp_parc.data.shape:
                raise ValueError(
                    "Time series dimensions do not match parcellation dimensions"
                )

            # Detect the number of time points
            num_timepoints = time_series.shape[-1]

            # Initialize array to hold region-wise time series
            region_time_series = np.zeros((len(unique_regions), num_timepoints))

            # Fixed loop - iterate over regions and find their index
            for i, region_label in enumerate(unique_regions):
                # Find the index of this region in parc.index
                region_idx = np.where(np.array(temp_parc.index) == region_label)[0]
                if len(region_idx) == 0:
                    continue
                region_idx = region_idx[0]  # Get the first (should be only) match

                # Computing the mean time series at non-zero voxels for this region
                ts_values = cltimg.compute_statistics_at_nonzero_voxels(
                    temp_parc.data == region_label, time_series, metric=metric
                )

                region_time_series[i, :] = ts_values

        # Create an attribute to hold the time series
        region_time_series = RegionTimeSeries(
            region_time_series,
            method=method,
            region_names=temp_parc.name,
            region_colors=temp_parc.color,
        )

        return region_time_series

    ######################################################################################################
    def surface_extraction(
        self,
        region_labels: list[int] | np.ndarray = None,
        region_names: list[str] | str = None,
        gaussian_smooth: bool = True,
        smooth_iterations: int = 10,
        fill_holes: bool = True,
        sigma: float = 1.0,
        closing_iterations: int = 1,
        out_filename: str = None,
        merge_surfaces: bool = True,
        out_format: str = "freesurfer",
        save_annotation: bool = True,
        overwrite: bool = False,
    ):
        """
        Extract 3D surface meshes from parcellation regions.

        Uses marching cubes algorithm with optional smoothing and hole filling
        to create high-quality surface meshes for visualization or analysis.

        Parameters
        ----------
        region_labels : list or np.ndarray, optional
            Region codes to extract surfaces for. Default is None (all regions).

        region_names : list or str, optional
            Region names to extract surfaces for. Default is None.

        gaussian_smooth : bool, optional
            Whether to apply Gaussian smoothing to volume. Default is True.

        smooth_iterations : int, optional
            Number of Taubin smoothing iterations. Default is 10.

        fill_holes : bool, optional
            Whether to fill holes in extracted meshes. Default is True.

        sigma : float, optional
            Standard deviation for Gaussian smoothing. Default is 1.0.

        closing_iterations : int, optional
            Morphological closing iterations before extraction. Default is 1.

        out_filename : str, optional
            Output file path for merged surface. Default is None.

        merge_surfaces : bool, optional
            Whether to merge all region surfaces into one. Default is True.

        out_format : str, optional
            Output format: 'freesurfer', 'vtk', 'ply', 'stl', 'obj'. Default is 'freesurfer'.

        save_annotation : bool, optional
            Whether to save annotation file with surface. Default is True.

        overwrite : bool, optional
            Whether to overwrite existing files. Default is False.

        Returns
        -------
        Surface
            Merged surface object containing all extracted regions or list of surfaces.

        Raises
        ------
        ValueError
            If both region_labels and region_names are specified.
        FileNotFoundError
            If output directory doesn't exist.
        FileExistsError
            If output file exists and overwrite=False.

        Examples
        --------
        >>> # Extract all surfaces
        >>> surface = parc.surface_extraction()
        >>>
        >>> # Extract specific regions with high quality
        >>> surface = parc.surface_extraction(
        ...     region_labels=[1, 2, 3],
        ...     smooth_iterations=20,
        ...     out_filename='regions.surf'
        ... )
        """

        # Check if include_by_code and include_by_name are different from None at the same time
        if region_labels is not None and region_names is not None:
            raise ValueError(
                "You cannot specify both include_by_code and include_by_name at the same time. Please choose one of them."
            )

        temp_parc = copy.deepcopy(self)

        # Apply inclusion if specified
        if region_labels is not None:
            temp_parc.keep_by_code(codes2keep=region_labels)

        if region_names is not None:
            temp_parc.keep_by_name(names2keep=region_names)

        # Get unique region values
        unique_regions = np.array(temp_parc.index)

        color_table = cltfree.colors2colortable(temp_parc.color)
        color_table, log, corresp_dict = cltfree.resolve_colortable_duplicates(
            color_table
        )

        # Surface objects store their colortables with the colors in the range
        # [0, 1] and an explicit opacity, while the FreeSurfer table uses the
        # range [0, 255] and leaves the alpha channel at 0. The last column, used
        # as the vertex value of every mesh, is kept untouched.
        surf_color_table = color_table.astype(float)
        surf_color_table[:, :3] = surf_color_table[:, :3] / 255
        surf_color_table[:, 3] = 1.0

        table_dict = {
            "names": temp_parc.name,
            "color_table": surf_color_table,
            "lookup_table": None,
        }

        surfaces_list = []

        # Add Rich progress bar around the main loop
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            expand=True,
        ) as progress:

            task = progress.add_task("Mesh extraction", total=len(unique_regions))

            for i, code in enumerate(unique_regions):
                struct_name = temp_parc.name[i]
                progress.update(
                    task,
                    description=f"Mesh extraction (Code {code}: {struct_name})",
                    completed=i + 1,
                )

                # Create binary mask for current code
                st_parc_temp = copy.deepcopy(self)
                st_parc_temp.keep_by_code(codes2keep=[code], rearrange=True)

                mesh = cltimg.extract_mesh_from_volume(
                    st_parc_temp.data,
                    gaussian_smooth=gaussian_smooth,
                    sigma=sigma,
                    fill_holes=fill_holes,
                    smooth_iterations=smooth_iterations,
                    affine=st_parc_temp.affine,
                    closing_iterations=closing_iterations,
                    vertex_value=color_table[i, 4],
                )

                surf_temp = cltsurf.Surface(name=struct_name)
                surf_temp.mesh = copy.deepcopy(mesh)

                # The mesh is attached directly, so the default colortable must be
                # replaced by the one of the region it was extracted from. Its
                # value column matches the vertex value assigned to the mesh.
                surf_temp.colortables["default"] = {
                    "names": [struct_name],
                    "color_table": surf_color_table[i : i + 1, :].copy(),
                    "lookup_table": None,
                }

                surfaces_list.append(surf_temp)
                # Update progress to show completion of this region

        if not merge_surfaces:
            return surfaces_list

        # surf_orig.merge_surfaces(surfaces_list)
        merged_surf = cltsurf.merge_surfaces(surfaces_list)
        merged_surf.colortables["default"] = table_dict

        if out_filename is not None:
            # Check if the directory exists, if not, gives an error
            path_dir = os.path.dirname(out_filename)

            if not os.path.exists(path_dir):
                raise FileNotFoundError(
                    f"The directory {path_dir} does not exist. Please create it before saving the surface."
                )

            # Check if the file exists, if it does check if overwrite is True
            if os.path.exists(out_filename) and not overwrite:
                raise FileExistsError(
                    f"The file {out_filename} already exists. Please set overwrite=True to overwrite it."
                )

            if save_annotation:
                save_path = os.path.dirname(out_filename)
                save_name = os.path.basename(out_filename)

                # Replace the file extension with .annot
                save_name = os.path.splitext(save_name)[0] + ".annot"
                annot_filename = os.path.join(save_path, save_name)

                merged_surf.save_surface(
                    filename=out_filename,
                    format=out_format,
                    map_name="default",
                    save_annotation=annot_filename,
                    overwrite=overwrite,
                )
            else:
                merged_surf.save_surface(
                    filename=out_filename,
                    format=out_format,
                    map_name="default",
                    overwrite=overwrite,
                )

        return merged_surf

    ######################################################################################################
    def adjust_values(self):
        """
        Synchronize index, name, and color attributes with data contents.

        Removes entries for codes not present in data, sorts the remaining
        entries by index value, and updates the min/max label range.
        """
        attrs = [a for a in ("index", "name", "color", "opacity") if hasattr(self, a)]

        if "index" not in attrs:
            raise AttributeError("The object has no 'index' attribute to adjust.")

        lengths = {a: len(getattr(self, a)) for a in attrs}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"index, name, color and opacity must have the same length. Got {lengths}"
            )

        st_codes = np.unique(self.data)
        unique_codes = st_codes[st_codes != 0]

        index_arr = np.asarray(self.index)
        keep = np.where(np.isin(index_arr, unique_codes))[0]

        # Order the kept positions by their index value (stable, so ties keep order)
        order = keep[np.argsort(index_arr[keep], kind="stable")]

        self.index = [int(x) for x in index_arr[order]]
        for attr in ("name", "color", "opacity"):
            if attr in attrs:
                values = getattr(self, attr)
                setattr(self, attr, [values[i] for i in order])

        self.parc_range()

    ######################################################################################################
    def group_by_codes(
        self, group_dict: dict, keep_ungrouped: bool = False
    ) -> tuple[np.ndarray, dict]:
        """
        Group array values and create color table for new groups.

        Structures not included in any group will remain unchanged with their original
        properties in both the array and color table.

        Parameters:
        -----------

        group_dict : dict
            {new_id: {'index': [old_ids], 'name': str, 'color': str, 'opacity': float}}
            Index values can be integers, strings with ranges ("11:12", "50-52"), or mixed.
            Name, color, and opacity are optional.

        keep_ungrouped : bool, optional
            Whether to keep structures not included in any group. Default is False.

        Returns:
        --------
        tuple : (modified_array, color_table)
            modified_array : numpy.ndarray
                Array with grouped values replaced by new IDs. Ungrouped structures remain unchanged.

            color_table : dict
                Color table with 'index', 'name', 'color', 'opacity', 'headerlines' keys.
                Includes both grouped structures and ungrouped structures with original properties.

        Examples:
        ---------
        >>> import numpy as np
        >>> array = np.random.randint(0, 60, (100, 100, 100))
        >>>
        >>> # Define groups with mixed specifications
        >>> group_dict = {
        ...     3: {'index': ["11:12", "50-52", 13]},  # Auto-generated name and color
        ...     4: {'index': [10, 49], 'name': 'Thalamus', 'color': '#33FF57', 'opacity': 0.8},
        ...     5: {'index': [17, 53, 18, 54], 'name': 'LimbicSystem', 'color': '#3357FF'},
        ...     6: {'index': [8, 47], 'name': 'Cerebellum', 'color': '#F1C40F', 'opacity': 0.8}
        ... }
        >>>
        >>> grouped_array, color_table = parc.group_by_codes(group_dict)
        >>> print(color_table['index'])  # [3, 4, 5, 6, ...ungrouped codes...]
        >>> print(color_table['name'])   # ['group_1', 'Thalamus', 'LimbicSystem', 'Cerebellum', ...original names...]
        """

        # Expand all groups up front
        groups = []
        for new_id, params in group_dict.items():
            old_ids = cltmisc.build_indices(params["index"])
            groups.append((int(new_id), [int(c) for c in old_ids], params))

        new_ids = [g[0] for g in groups]
        if len(set(new_ids)) != len(new_ids):
            raise ValueError("Duplicate new IDs in group_dict.")

        # A code assigned to two groups is ambiguous
        seen = {}
        for new_id, old_ids, _ in groups:
            for c in old_ids:
                if c in seen and seen[c] != new_id:
                    raise ValueError(
                        f"Code {c} is assigned to groups {seen[c]} and {new_id}."
                    )
                seen[c] = new_id

        if not keep_ungrouped:
            self.keep_by_code(codes2keep=sorted(seen))

        # Snapshot the original data and metadata BEFORE modifying anything
        orig = self.data.copy()
        orig_meta = {
            int(c): (n, col, op)
            for c, n, col, op in zip(
                self.index, self.name, self.color, self.opacity, strict=False
            )
        }

        grouped_mask = np.isin(orig, list(seen))
        ungrouped_codes = [int(c) for c in np.unique(orig[~grouped_mask]) if c != 0]

        clash = set(ungrouped_codes) & set(new_ids)
        if clash:
            raise ValueError(
                f"New group IDs {sorted(clash)} collide with ungrouped region codes. "
                "Choose IDs outside the existing label range or set keep_ungrouped=False."
            )

        new_data = orig.copy()
        def_colors = cltcol.create_distinguishable_colors(
            len(groups), output_format="hex"
        )
        def_names = cltmisc.create_names_from_indices(
            np.arange(1, len(groups) + 1), prefix="group"
        )

        color_table = {
            "index": [],
            "name": [],
            "color": [],
            "opacity": [],
            "headerlines": [],
        }
        for i, (new_id, old_ids, params) in enumerate(groups):
            new_data[np.isin(orig, old_ids)] = new_id  # mask from ORIGINAL data
            color_table["index"].append(new_id)
            color_table["name"].append(params.get("name", def_names[i]))
            color_table["color"].append(params.get("color", def_colors[i]))
            color_table["opacity"].append(float(params.get("opacity", 1.0)))

        for code in ungrouped_codes:
            n, col, op = orig_meta.get(code, (f"region_{code}", "#ffffff", 1.0))
            color_table["index"].append(code)
            color_table["name"].append(n)
            color_table["color"].append(col)
            color_table["opacity"].append(op)

        self.data = new_data
        self.index = color_table["index"]
        self.name = color_table["name"]
        self.color = cltcol.harmonize_colors(color_table["color"], output_format="hex")
        self.opacity = color_table["opacity"]
        self.adjust_values()

        return self.data, color_table

    ######################################################################################################
    def group_by_names(
        self,
        group_dict: dict,
        keep_ungrouped: bool = True,
        bool_case: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """
        Group regions by name substrings and create a color table for the new groups.

        Every region whose name contains one of a group's substrings is relabeled
        with that group's ID. Masks are built from the original data, so groups
        cannot contaminate each other.

        Parameters
        ----------
        group_dict : dict
            {group_name: {'names': str | list[str], 'index': int,
                        'color': (R, G, B) | hex str, 'opacity': float}}
            'names' is required: substring(s) matched against region names.
            'index' is the new label; defaults to the group's position (1, 2, ...).
            'color' and 'opacity' are optional (opacity defaults to 1.0).

        keep_ungrouped : bool, optional
            Keep regions not matched by any group, with their original code, name,
            and color. Default is True.

        bool_case : bool, optional
            Case-sensitive substring matching. Default is False.

        Returns
        -------
        tuple : (modified_array, color_table)
            modified_array : np.ndarray
                The grouped parcellation data (same object as self.data).
            color_table : dict
                Keys 'index', 'name', 'color', 'opacity', 'headerlines'.

        Raises
        ------
        ValueError
            If group_dict is empty, a group lacks 'names', new IDs are duplicated,
            a region matches more than one group, no group matches any region, or a
            new ID collides with an ungrouped region code.

        Examples
        --------
        >>> group_dict = {
        ...     'Thalamus':    {'names': 'thal-',  'index': 1, 'color': '#2CB746', 'opacity': 0.8},
        ...     'Hippocampus': {'names': 'hipp-',  'index': 2, 'color': (241, 196, 15)},
        ...     'Cerebellum':  {'names': ['cer-'], 'index': 3},
        ... }
        >>> grouped_array, color_table = parc.group_by_names(group_dict)
        """
        if not isinstance(group_dict, dict) or len(group_dict) == 0:
            raise ValueError("group_dict must be a non-empty dictionary.")

        code_groups: dict[int, dict] = {}
        claimed: dict[int, str] = {}  # region code -> group name that claimed it

        for pos, (group_name, params) in enumerate(group_dict.items()):
            if "names" not in params:
                raise ValueError(f"Group '{group_name}' has no 'names' key.")

            filters = params["names"]
            if isinstance(filters, str):
                filters = [filters]

            new_id = int(params.get("index", pos + 1))
            if new_id in code_groups:
                raise ValueError(
                    f"New ID {new_id} is used by more than one group "
                    f"('{code_groups[new_id]['name']}' and '{group_name}')."
                )

            matches = cltmisc.get_indexes_by_substring(
                input_list=self.name,
                or_filter=filters,
                invert=False,
                bool_case=bool_case,
            )
            codes = [int(self.index[k]) for k in matches]

            if len(codes) == 0:
                warnings.warn(
                    f"Group '{group_name}' ({filters}) matched no region and is skipped.",
                    stacklevel=2,
                )
                continue

            # A region matched by two groups is ambiguous
            for k, code in zip(matches, codes, strict=False):
                if code in claimed:
                    raise ValueError(
                        f"Region '{self.name[k]}' (code {code}) matches both "
                        f"'{claimed[code]}' and '{group_name}'. Use more specific substrings."
                    )
                claimed[code] = group_name

            entry = {
                "index": codes,
                "name": group_name,
                "opacity": float(params.get("opacity", 1.0)),
            }
            if "color" in params:
                entry["color"] = cltcol.harmonize_colors(
                    [params["color"]], output_format="hex"
                )[0]

            code_groups[new_id] = entry

        if len(code_groups) == 0:
            raise ValueError(
                "None of the groups matched any region in the parcellation."
            )

        # Masking from the original data, ungrouped handling, collision checks,
        # and metadata updates are all done by group_by_codes
        return self.group_by_codes(code_groups, keep_ungrouped=keep_ungrouped)

    ######################################################################################################
    def relabel_regions(self, relabel_dict: dict, rearrange: bool = False):
        """
        Relabel regions in the parcellation.

        Parameters
        ----------
        relabel_dict : dict
            Mapping {old_label: new_label}. Only labels present as keys are
            changed; everything else is left untouched.
        rearrange : bool, optional
            If True, after relabeling, renumber all labels present in the
            parcellation into a contiguous sequence (1, 2, 3, ...), ordered
            by the sorted value of the (already relabeled) labels.
            Default is False.

        Returns
        -------
        self
            Updates self.data and self.index in place, and returns self so
            calls can be chained.
        """
        if not isinstance(relabel_dict, dict):
            raise TypeError("relabel_dict must be a dict of {old_label: new_label}")

        # Warn if a label appears both as a key and as a value (e.g. a swap
        # like {1: 2, 2: 1}, or a chain like {1: 2, 2: 3}). This is not an
        # error - the mapping still resolves correctly since we always read
        # from the original data - but it's worth flagging since it's an
        # easy source of unintended results.
        overlap = set(relabel_dict.keys()) & set(relabel_dict.values())
        if overlap:
            warnings.warn(
                f"relabel_dict has label(s) that are both an old_label and "
                f"a new_label: {sorted(overlap)}. This is handled correctly "
                f"(all lookups use the original labels), but double-check "
                f"this is intentional.",
                UserWarning,
                stacklevel=2,
            )

        # Work from the ORIGINAL data so simultaneous/overlapping mappings
        # (e.g. swapping 1 <-> 2) don't cascade into each other.
        old_data = self.data
        new_data = old_data.copy()
        for old_val, new_val in relabel_dict.items():
            new_data[old_data == old_val] = new_val
        self.data = new_data

        # Mirror the same mapping onto the index (labels not in the dict
        # pass through unchanged).
        if self.index is not None:
            old_index = np.asarray(self.index)
            new_index = np.array([relabel_dict.get(val, val) for val in old_index])
            self.index = new_index

        # Adjust values and update parcellation range after relabeling
        self.adjust_values()
        self.parc_range()

        if rearrange:
            self.rearrange()

        return self

    ######################################################################################################
    def rename_regions(self, rename_dict: dict):
        """
        Rename regions in the parcellation.

        Parameters
        ----------
        rename_dict : dict
            Mapping {old_name: new_name}. Only names present as keys are
            changed; everything else is left untouched.

        Returns
        -------
        self
            Updates self.name in place, and returns self so calls can be
            chained.
        """
        if not isinstance(rename_dict, dict):
            raise TypeError("rename_dict must be a dict of {old_name: new_name}")

        # Warn if a name appears both as a key and as a value (e.g. a swap
        # like {"A": "B", "B": "A"}, or a chain like {"A": "B", "B": "C"}).
        # This is not an error - the mapping still resolves correctly since
        # we always read from the original names - but it's worth flagging
        # since it's an easy source of unintended results.
        overlap = set(rename_dict.keys()) & set(rename_dict.values())
        if overlap:
            warnings.warn(
                f"rename_dict has name(s) that are both an old_name and a "
                f"new_name: {sorted(overlap)}. This is handled correctly "
                f"(all lookups use the original names), but double-check "
                f"this is intentional.",
                UserWarning,
                stacklevel=2,
            )

        # Work from the ORIGINAL names so simultaneous/overlapping mappings
        # (e.g. swapping name A <-> name B) don't cascade into each other.
        old_names = list(self.name)
        new_names = [rename_dict.get(name, name) for name in old_names]
        self.name = new_names

        # Adjust values and update parcellation range after renaming
        self.adjust_values()
        self.parc_range()

        return self

    ######################################################################################################
    def rearrange(self, offset: int = 0):
        """
        Rearrange parcellation labels to consecutive integers.

        Parameters
        ----------
        offset : int, optional
            Starting value for rearranged labels. Default is 0 (starts from 1).

        Examples
        --------
        >>> # Rearrange to 1, 2, 3, ...
        >>> parc.rearrange()
        >>>
        >>> # Start from 100
        >>> parc.rearrange(offset=99)
        """

        # First, adjust values to ensure index, name, color align with data
        self.adjust_values()

        # Get unique structure codes in data (excluding background/0)
        st_codes = np.unique(self.data)
        st_codes = st_codes[st_codes != 0]

        # Leave the index if it is present in stcodes
        new_parc = np.zeros_like(self.data)
        new_index = []
        new_name = []
        new_color = []
        new_opacity = []
        for i, index in enumerate(self.index):
            if index in st_codes:
                new_index.append(i + 1 + offset)
                new_name.append(self.name[i])
                new_color.append(self.color[i])
                if hasattr(self, "opacity"):
                    new_opacity.append(self.opacity[i])

                new_parc[self.data == index] = i + 1 + offset

        # Update data with rearranged labels
        self.data = new_parc
        self.index = new_index
        self.name = new_name
        self.color = new_color
        if hasattr(self, "opacity"):
            self.opacity = new_opacity

        # Update parcellation range
        self.parc_range()

    ######################################################################################################
    def harmonize(self):
        """
        Harmonize parcellation attributes with data contents.

        Ensures index, name, color and opacity attributes align in type and length
        with actual data values, removing unused entries and updating min/max label range.

        Examples
        --------
        >>> parc.harmonize()
        >>> print(f"Regions in data: {len(parc.index)}")
        """

        # Get unique codes present in data (excluding background/0)
        st_codes = np.unique(self.data)
        unique_codes = st_codes[st_codes != 0]

        # Find which indices are actually present in the data
        if hasattr(self, "index"):
            mask = np.isin(self.index, unique_codes)
            indexes = np.where(mask)[0]

            # Filter index to only present labels
            temp_index = np.array(self.index)
            index_new = temp_index[mask]
            self.index = [int(x) for x in index_new.tolist()]

            # Filter name if present
            if hasattr(self, "name"):
                self.name = [self.name[i] for i in indexes]

            # Filter color if present
            if hasattr(self, "color"):
                self.color = [self.color[i] for i in indexes]

            # Filter opacity if present
            if hasattr(self, "opacity"):
                self.opacity = [self.opacity[i] for i in indexes]

        # Harmonize colors to consistent format (after filtering)
        if hasattr(self, "color"):
            self.color = cltcol.harmonize_colors(self.color)

        # Harmonize opacity to list of floats (after filtering)
        if hasattr(self, "opacity"):
            if isinstance(self.opacity, np.ndarray):
                self.opacity = [float(x) for x in self.opacity.tolist()]
            elif not hasattr(self.opacity, "__len__"):
                # scalar: broadcast to all regions
                self.opacity = [float(self.opacity)] * len(self.index)
            elif not isinstance(self.opacity, list):
                self.opacity = [float(x) for x in self.opacity]
            else:
                self.opacity = [float(x) for x in self.opacity]

        # Update parcellation range
        self.parc_range()

    ######################################################################################################
    def add_parcellation(self, parc2add, append: bool = False):
        """
        Combine another parcellation into current object.

        Parameters
        ----------
        parc2add : Parcellation or list
            Parcellation object(s) to add.

        append : bool, optional
            If True, adds new labels by offsetting. If False, overlays directly. Default is False.

        Examples
        --------
        >>> # Overlay parcellations
        >>> parc1.add_parcellation(parc2, append=False)
        >>>
        >>> # Append with new labels
        >>> parc1.add_parcellation(parc2, append=True)
        """

        # Harmonize current parcellation
        self.harmonize()

        # Convert single parcellation to list
        if isinstance(parc2add, Parcellation):
            parc2add = [parc2add]

        if not isinstance(parc2add, list):
            raise TypeError(
                "parc2add must be a Parcellation object or list of Parcellation objects"
            )

        if len(parc2add) == 0:
            raise ValueError("The parcellation list is empty")

        # Process each parcellation to add
        for parc in parc2add:
            if not isinstance(parc, Parcellation):
                raise TypeError("All elements must be Parcellation objects")

            # Deep copy and harmonize
            tmp_parc = copy.deepcopy(parc)
            tmp_parc.harmonize()

            # Get non-zero indices
            ind = np.where(tmp_parc.data != 0)

            # Adjust labels if appending
            if append:
                tmp_parc.data[ind] = tmp_parc.data[ind] + self.maxlab
                if hasattr(tmp_parc, "index"):
                    tmp_parc.index = [int(x + self.maxlab) for x in tmp_parc.index]

            # Check if both parcellations have lookup tables
            has_lut = all(
                hasattr(tmp_parc, attr) for attr in ["index", "name", "color"]
            )
            self_has_lut = all(
                hasattr(self, attr) for attr in ["index", "name", "color"]
            )

            if has_lut and self_has_lut:
                # After harmonize(), all attributes are lists, so simple concatenation
                self.index = self.index + tmp_parc.index
                self.name = self.name + tmp_parc.name
                self.color = self.color + tmp_parc.color

                # Handle opacity if present in either parcellation
                if hasattr(tmp_parc, "opacity"):
                    if hasattr(self, "opacity"):
                        self.opacity = self.opacity + tmp_parc.opacity
                    else:
                        # Create default opacity for existing labels
                        self.opacity = [1.0] * len(self.index) + tmp_parc.opacity
                elif hasattr(self, "opacity"):
                    # Extend opacity with defaults for new labels
                    self.opacity = self.opacity + [1.0] * len(tmp_parc.index)

            elif has_lut and np.sum(self.data) == 0:
                # Self is empty, copy all attributes from tmp_parc
                self.index = tmp_parc.index
                self.name = tmp_parc.name
                self.color = tmp_parc.color
                if hasattr(tmp_parc, "opacity"):
                    self.opacity = tmp_parc.opacity

            # Update parcellation data
            self.data[ind] = tmp_parc.data[ind]

        # Final harmonization
        self.harmonize()

        # Detect minimum and maximum labels
        self.parc_range()

    ######################################################################################################
    def save_parcellation(
        self,
        out_file: str | Path,
        affine: np.float64 = None,
        headerlines: list | str = None,
        lut_file: str | Path | list[str] | list[Path] = None,
        lut_type: str | list[str] = "lut",
        overwrite: bool = True,
    ):
        """
        Save parcellation to NIfTI file with optional lookup tables.

        Parameters
        ----------
        out_file : str
            Output file path.

        affine : np.ndarray, optional
            Affine transformation matrix. If None, uses object's affine.

        headerlines : list, str, or None, optional
            Header lines for LUT format. If None, uses object's headerlines.

        lut_file : str, Path, list of str/Path, or None, optional
            Path(s) for lookup table file(s). If None, paths are auto-generated
            from out_file using the appropriate extension for each lut_type.
            If a list, must match the length of lut_type.

        lut_type : str or list of str, optional
            Lookup table format(s): 'lut', 'tsv', 'fsl', or 'nilearn'.
            Can be a list to export multiple formats simultaneously,
            e.g. ['lut', 'tsv']. Default is 'lut'.

        overwrite : bool, optional
            Whether to overwrite existing files. Default is True.

        Raises
        ------
        ValueError
            If lut_file is a list whose length does not match lut_type,
            or if an unrecognised lut_type is given.

        Examples
        --------
        >>> # Save with a single LUT format
        >>> parc.save_parcellation('output.nii.gz', lut_type='tsv')

        >>> # Save with multiple LUT formats (auto-generated paths)
        >>> parc.save_parcellation('output.nii.gz', lut_type=['lut', 'tsv'])

        >>> # Save with multiple LUT formats and explicit paths
        >>> parc.save_parcellation(
        ...     'output.nii.gz',
        ...     lut_type=['lut', 'tsv'],
        ...     lut_file=['custom.lut', 'custom.tsv']
        ... )
        """

        # Mapping from lut_type to file extension
        _EXT_MAP = {
            "lut": ".lut",
            "tsv": ".tsv",
            "fsl": ".fsllut",
            "nilearn": ".nilearnlut",
        }

        # Handle affine
        if affine is None:
            affine = self.affine

        # Handle headerlines
        if headerlines is None:
            headerlines = self.headerlines

        elif isinstance(headerlines, str):
            headerlines = [headerlines]

            # Add an empty line at the end if not present
            headerlines = (
                headerlines + "\n"
                if not headerlines[-1].endswith("\n")
                else headerlines
            )

        # Normalise out_file to str
        if isinstance(out_file, Path):
            out_file = str(out_file)

        if not overwrite and os.path.exists(out_file):
            raise FileExistsError(
                f"File {out_file} already exists. Set overwrite=True to overwrite."
            )

        # Save NIfTI file with proper data type
        data_to_save = self.data.astype(np.int32)
        out_atlas = nib.Nifti1Image(data_to_save, affine)
        nib.save(out_atlas, out_file)

        if lut_type is None:
            return

        # --- Normalise lut_type to a list ---
        if isinstance(lut_type, str):
            lut_type = [lut_type]

        # Validate all requested formats
        for lt in lut_type:
            if lt.lower() not in _EXT_MAP:
                raise ValueError(
                    f"Unrecognised lut_type '{lt}'. Must be one of: {list(_EXT_MAP.keys())}"
                )

        # --- Normalise lut_file to a list of matching length ---
        if lut_file is None:
            # Auto-generate one path per format
            base_name = cltmisc.get_real_basename(os.path.basename(out_file))
            out_dir = os.path.dirname(out_file)
            lut_file = [
                os.path.join(out_dir, base_name + _EXT_MAP[lt.lower()])
                for lt in lut_type
            ]

        elif isinstance(lut_file, (str, Path)):
            # Single explicit path — only valid when a single format is requested
            if len(lut_type) > 1:
                raise ValueError(
                    f"A single lut_file was provided but lut_type contains "
                    f"{len(lut_type)} formats. Provide a list of paths or set "
                    f"lut_file=None to auto-generate them."
                )
            lut_file = [str(lut_file)]

        elif isinstance(lut_file, list):
            lut_file = [str(f) for f in lut_file]
            if len(lut_file) != len(lut_type):
                raise ValueError(
                    f"lut_file list length ({len(lut_file)}) must match "
                    f"lut_type list length ({len(lut_type)})."
                )
        else:
            raise TypeError(
                f"lut_file must be a str, Path, list, or None; got {type(lut_file)}."
            )

        # --- Export one colortable per (lut_file, lut_type) pair ---
        for lf, lt in zip(lut_file, lut_type, strict=False):
            self.export_colortable(
                out_file=lf,
                lut_type=lt.lower(),
                overwrite=overwrite,
                headerlines=headerlines,
            )

    ######################################################################################################
    def load_colortable(self, lut_file: str | Path | dict = None):
        """
        Load lookup table to associate codes with names and colors.

        Parameters
        ----------
        lut_file : str or dict, optional
            Path to LUT file or dictionary with index/name/color keys. Default is None.


        Examples
        --------
        >>> # Load FreeSurfer LUT
        >>> parc.load_colortable('FreeSurferColorLUT.txt')
        >>>
        >>> # Load TSV table
        >>> parc.load_colortable('regions.tsv')
        """

        if lut_file is None:
            # Get the enviroment variable of $FREESURFER_HOME
            freesurfer_home = os.getenv("FREESURFER_HOME")
            lut_file = os.path.join(freesurfer_home, "FreeSurferColorLUT.txt")

        if isinstance(lut_file, (str, Path)):
            if os.path.exists(lut_file):
                self.lut_file = lut_file

                col_dict = cltcol.ColorTableLoader.load_colortable(lut_file)

            else:
                raise ValueError("The lut file does not exist")

        elif isinstance(lut_file, dict):
            self.lut_file = None

            col_dict = copy.deepcopy(lut_file)

        if "index" in col_dict.keys() and "name" in col_dict.keys():
            self.index = col_dict["index"]
            self.name = col_dict["name"]
        else:
            raise ValueError("The dictionary must contain the keys 'index' and 'name'")

        if "color" in col_dict.keys():
            self.color = cltcol.harmonize_colors(col_dict["color"], output_format="hex")

        else:
            self.color = cltcol.create_distinguishable_colors(
                len(self.index), output_format="hex"
            )

        if "opacity" in col_dict.keys():
            self.opacity = col_dict["opacity"]
        else:
            self.opacity = [1.0] * len(self.index)

        if "headerlines" in col_dict.keys():
            self.headerlines = col_dict["headerlines"]
        else:
            self.headerlines = []

        self.adjust_values()
        self.parc_range()

    ######################################################################################################
    def sort_index(self):
        """
        Sort index, name, and color attributes by index values.

        Examples
        --------
        >>> parc.sort_index()
        >>> print(f"First region: {parc.name[0]} (code: {parc.index[0]})")
        """

        # Sort the all_index and apply the order to all_name and all_color
        sort_index = np.argsort(self.index)
        self.index = [self.index[i] for i in sort_index]
        self.name = [self.name[i] for i in sort_index]
        self.color = [self.color[i] for i in sort_index]
        self.opacity = [self.opacity[i] for i in sort_index]

    ######################################################################################################
    def export_colortable(
        self,
        out_file: str,
        lut_type: str = "lut",
        headerlines: list | str = None,
        overwrite: bool = True,
    ):
        """
        Export lookup table to file.

        Parameters
        ----------
        out_file : str
            Output file path.

        lut_type : str, optional
            Output format: 'lut' or 'tsv'. Default is 'lut'.

        headerlines : list or str, optional
            Header lines for LUT format. Default is None.

        overwrite : bool, optional
            Whether to overwrite existing files. Default is True.

        Examples
        --------
        >>> # Export FreeSurfer LUT
        >>> parc.export_colortable('regions.lut', lut_type='lut')
        >>>
        >>> # Export TSV
        >>> parc.export_colortable('regions.tsv', lut_type='tsv')
        """

        if headerlines is None:
            headerlines = self.headerlines
        else:
            if isinstance(headerlines, str):
                headerlines = [headerlines]

        if len(headerlines) == 0:
            headerlines = self.headerlines

        if (
            not hasattr(self, "index")
            or not hasattr(self, "name")
            or not hasattr(self, "color")
        ):
            raise ValueError(
                "The parcellation does not contain a color table. The index, name and color attributes must be present"
            )

        # Adjusting the colortable to the values in the parcellation
        array_3d = self.data
        unique_codes = np.unique(array_3d)
        unique_codes = unique_codes[unique_codes != 0]

        mask = np.isin(self.index, unique_codes)
        indexes = np.where(mask)[0]

        temp_index = np.array(self.index)
        index_new = temp_index[mask]

        if hasattr(self, "index"):
            self.index = index_new

        # If name is an attribute of self
        if hasattr(self, "name"):
            self.name = [self.name[i] for i in indexes]

        # If color is an attribute of self
        if hasattr(self, "color"):
            self.color = [self.color[i] for i in indexes]

        if hasattr(self, "opacity"):
            self.opacity = [self.opacity[i] for i in indexes]

        # Create color dictionary
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

        if len(headerlines) == 0:
            headerlines = [f"# $Id: {out_file} {date_time} \n"]

            if os.path.isfile(self.parc_file):
                headerlines.append(f"# Corresponding parcellation: {self.parc_file} \n")

        if lut_type == "lut":

            now = datetime.now()
            date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

            if len(headerlines) == 0:
                headerlines = [f"# $Id: {out_file} {date_time} \n"]

                if os.path.isfile(self.parc_file):
                    headerlines.append(
                        f"# Corresponding parcellation: {self.parc_file} \n"
                    )

        elif lut_type == "tsv":

            if self.index is None or self.name is None:
                raise ValueError(
                    "The parcellation does not contain a color table. The index and name attributes must be present"
                )

            tsv_df = pd.DataFrame({"index": np.asarray(self.index), "name": self.name})
            # Add color if it is present
            if self.color is not None:

                if isinstance(self.color, list):
                    if isinstance(self.color[0], str):
                        if self.color[0][0] != "#":
                            raise ValueError("The colors must be in hexadecimal format")
                        else:
                            tsv_df["color"] = self.color
                    else:
                        tsv_df["color"] = cltcol.multi_rgb2hex(self.color)

                elif isinstance(self.color, np.ndarray):
                    tsv_df["color"] = cltcol.multi_rgb2hex(self.color)

        col_dict = {
            "index": self.index,
            "name": self.name,
            "color": self.color,
            "opacity": self.opacity,
            "headerlines": headerlines,
        }

        col_obj = cltcol.ColorTableLoader(col_dict)
        col_obj.export(
            out_file, out_format=lut_type, overwrite=overwrite, headerlines=headerlines
        )

    #########################################################################################################
    def merge_ctx_wm(
        self,
        ctx_wm_offset: int = 3000,
        output_file: str | Path | None = None,
        overwrite: bool = True,
    ) -> "Parcellation":
        """
        Merge cortical gray matter (GM) and adjacent white matter (WM) into a single tissue type.

        This method modifies the parcellation in place, combining cortical GM parcels with their
        corresponding WM parcels. The WM parcels are identified by adding CTX_WM_OFFSET (3000) to
        the cortical GM parcel codes. After merging, the parcellation's index, name, color, and
        opacity attributes are updated accordingly.

        Parameters
        ----------
        ctx_wm_offset : int, optional
            The offset used to identify WM parcels corresponding to cortical GM parcels. Default is 3000

        output_file : str or Path, optional
            Path to save the modified parcellation. If None, the parcellation is not saved to disk.

        overwrite : bool, optional
            Whether to overwrite the output file if it already exists. Default is True.

        Returns
        -------
        Parcellation
            self, to allow method chaining.

        Examples
        --------
        >>> # Merge cortical GM and adjacent WM in the parcellation
        >>> parc.merge_ctx_wm()
        """

        ind_codes = cltmisc.get_indexes_by_substring(
            [n.lower() for n in self.name], ["wm-lh", "wm-rh"]
        )
        ctx_wm_codes = [self.index[i] for i in ind_codes]

        if not ctx_wm_codes:
            warnings.warn(
                f"No WM parcel has a matching cortical code "
                f"(WM code = cortical code + {ctx_wm_offset}). No merging was performed.",
                stacklevel=2,
            )
            return self

        ind_wm_ctx_vox = np.isin(self.data, ctx_wm_codes)
        self.data[ind_wm_ctx_vox] = (
            self.data[ind_wm_ctx_vox] - ctx_wm_offset
        )  # Reassign WM voxels to their corresponding GM tissue code

        self.adjust_values()  # Update index, name, color, opacity after modification

        if output_file is not None:
            if not isinstance(output_file, (str, Path)):
                raise TypeError(
                    f"output_file must be a string or Path, got {type(output_file)}"
                )
            output_file = Path(output_file)
            if not output_file.parent.exists():
                raise FileNotFoundError(
                    f"Output directory does not exist: {output_file.parent}"
                )
            self.save_parcellation(out_file=output_file, overwrite=overwrite)
            warnings.warn(f"Saved merged parcellation to {output_file}", stacklevel=2)

        return self

    #########################################################################################################
    def create_5tt(
        self, output_file: str | Path = None, mergectx: bool = False
    ) -> np.ndarray:
        """
        Create a 5-tissue-type (5TT) image from a parcellation following the
        MRtrix3 convention: [cortical GM, subcortical GM, WM, CSF, pathology].

        Parameters
        ----------
        output_file : str or Path, optional
            Path to save the 5TT image. If None, the image is not saved.

        mergectx : bool, optional
            If True, merges cortical GM and the WM next to the cortex into a single tissue type.
            Default is False.

        Returns
        -------
        five_tt_image : np.ndarray
            4D array with shape (X, Y, Z, 5) representing the five tissue types.

        Raises
        ------
        ValueError
            If no supra-regions match the expected tissue groups, indicating that
            the parcellation does not follow the Chimera naming convention.

        Examples
        --------
        >>> # Create 5TT image and save to file
        >>> five_tt = parc.create_5tt(output_file='5tt_image.nii.gz')

        """

        tissue_groups = {
            "cortical_gm": ["ctx", "cer"],
            "subcortical_gm": ["subc", "thal", "hipp", "amygd", "hypo", "vdc"],
            "wm": ["wm", "bstem"],
            "csf": ["vent", "csf"],
        }

        # Work on a copy when merging so the caller's parcellation is left untouched.
        work = copy.deepcopy(self) if mergectx else self

        if mergectx:
            work.merge_ctx_wm()

        masks = []
        for tissue, names in tissue_groups.items():
            indexes = cltmisc.get_indexes_by_substring(
                input_list=work.name, or_filter=names, bool_case=False
            )
            if indexes:
                mask = np.isin(work.data, [work.index[i] for i in indexes])
            else:
                mask = np.zeros(work.data.shape, dtype=bool)

            if not mask.any():
                warnings.warn(
                    f"No voxels found for 5TT tissue '{tissue}' "
                    f"(names: {names}). Check the Chimera naming convention.",
                    stacklevel=2,
                )
            masks.append(mask)

        if not any(m.any() for m in masks):
            raise ValueError(
                "No supra-regions matched. The parcellation does not appear to "
                "follow the Chimera naming convention."
            )

        # Pathology volume is empty (no lesion model here).
        masks.append(np.zeros(work.data.shape, dtype=bool))

        # Enforce mutual exclusivity by priority so each voxel sums to <= 1.
        assigned = np.zeros(work.data.shape, dtype=bool)
        for i, mask in enumerate(masks):
            masks[i] = mask & ~assigned
            assigned |= masks[i]

        five_tt_image = np.stack(masks, axis=-1).astype(np.float32)

        if output_file is not None:
            if not isinstance(output_file, (str, Path)):
                raise TypeError(
                    f"output_file must be a string or Path, got {type(output_file)}"
                )
            output_file = Path(output_file)
            if not output_file.parent.exists():
                raise FileNotFoundError(
                    f"Output directory does not exist: {output_file.parent}"
                )
            nib.save(
                nib.Nifti1Image(five_tt_image, affine=self.affine), str(output_file)
            )
            print(f"Saved 5TT image to {output_file}")

        return five_tt_image

    ######################################################################################################
    def replace_labels(
        self,
        codes2rep: int | str | list[int | str | list[int]] | np.ndarray | dict,
        new_codes: int | list[int] | np.ndarray | None = None,
    ) -> "Parcellation":
        """
        Replace region codes with new values, supporting group replacements.

        All masks are built from the ORIGINAL data, so chained or swapping
        mappings (e.g. {1: 2, 2: 1}) never cascade.

        Parameters
        ----------
        codes2rep : int, str, list, np.ndarray, or dict
            Codes to replace. Accepted forms:
            - dict: {old: new} or {(old1, old2): new}; keys may also be range
            strings ("11:12", "50-52"). new_codes is ignored.
            - int or str: a single code or range (one group).
            - list of int/str: each entry is one group, paired with one new code.
            - list of lists: every code in a group gets the same new code.
            - 1-D np.ndarray: same as a list of int.

        new_codes : int, list of int, or np.ndarray, optional
            New codes, paired positionally with the groups in codes2rep.
            Required unless codes2rep is a dict. Each entry must resolve to
            exactly one code.

        Returns
        -------
        Parcellation
            self, to allow method chaining.

        Notes
        -----
        Metadata (name, color, opacity) of the resulting regions:
        - A region whose code is not replaced keeps its own metadata. If replaced
        codes are merged into it, its metadata wins and a warning is issued.
        - When several codes are merged into a new code, the metadata of the
        lowest code in the group is used.
        - If none of the merged codes had a color-table entry, a default entry
        (region_<code>, white, opacity 1.0) is created.

        Raises
        ------
        AttributeError
            If the object has no 'data' attribute.
        TypeError
            If the inputs have unsupported types.
        ValueError
            If new_codes is missing, the numbers of groups and new codes differ,
            a new-code entry resolves to more than one code, or a code is assigned
            to different new codes.

        Examples
        --------
        >>> parc.replace_labels({1: 9, 2: 8})
        >>> parc.replace_labels([1, 2], [20, 10])
        >>> parc.replace_labels([[1, 2], [3]], [10, 20])          # merge 1 and 2
        >>> parc.replace_labels({(11, 12, 13): 100, "50-52": 200})
        >>> parc.replace_labels({1: 2, 2: 1})                      # swap
        """
        if not hasattr(self, "data"):
            raise AttributeError("Object must have 'data' attribute")

        _scalar = (int, np.integer, str)

        # ------------------------------------------------------------------
        # Normalize inputs into groups of old codes and a list of new codes
        # ------------------------------------------------------------------
        if isinstance(codes2rep, dict):
            if new_codes is not None:
                warnings.warn(
                    "codes2rep is a dict; new_codes is ignored.", stacklevel=2
                )
            raw_groups, raw_new = [], []
            for old, new in codes2rep.items():
                raw_groups.append(
                    list(old) if isinstance(old, (list, tuple)) else [old]
                )
                raw_new.append(new)

        else:
            if isinstance(codes2rep, _scalar):
                raw_groups = [[codes2rep]]
            elif isinstance(codes2rep, np.ndarray):
                if codes2rep.ndim != 1:
                    raise TypeError("Unsupported numpy array shape for codes2rep")
                raw_groups = [[int(x)] for x in codes2rep.tolist()]
            elif isinstance(codes2rep, (list, tuple)):
                if len(codes2rep) == 0:
                    raise ValueError("codes2rep cannot be empty")
                if all(isinstance(x, _scalar) for x in codes2rep):
                    raw_groups = [[x] for x in codes2rep]
                elif all(isinstance(x, (list, tuple)) for x in codes2rep):
                    raw_groups = [list(x) for x in codes2rep]
                else:
                    raise TypeError(
                        "codes2rep must be a list of codes or a list of lists of codes"
                    )
            else:
                raise TypeError(
                    f"codes2rep must be int, str, list, numpy array or dict, "
                    f"got {type(codes2rep)}"
                )

            if new_codes is None:
                raise ValueError("new_codes is required unless codes2rep is a dict.")
            if isinstance(new_codes, _scalar):
                raw_new = [new_codes]
            elif isinstance(new_codes, np.ndarray):
                raw_new = new_codes.ravel().tolist()
            elif isinstance(new_codes, (list, tuple)):
                raw_new = list(new_codes)
            else:
                raise TypeError(
                    f"new_codes must be int, list or numpy array, got {type(new_codes)}"
                )

        # Expand every group on its own, so build_indices() sorting cannot re-pair them
        groups = []
        for g in raw_groups:
            expanded = [int(c) for c in cltmisc.build_indices(g, nonzeros=False)]
            if len(expanded) == 0:
                raise ValueError(f"Group {g} does not contain any code.")
            groups.append(expanded)

        new_list = []
        for entry in raw_new:
            expanded = cltmisc.build_indices([entry], nonzeros=False)
            if len(expanded) != 1:
                raise ValueError(
                    f"Each new code must resolve to a single label; '{entry}' "
                    f"resolved to {len(expanded)} labels."
                )
            new_list.append(int(expanded[0]))

        if len(new_list) != len(groups):
            raise ValueError(
                f"Number of new codes ({len(new_list)}) must equal "
                f"number of groups ({len(groups)}) to be replaced"
            )

        # A code assigned to two different targets is ambiguous
        code_to_new: dict[int, int] = {}
        for group, new in zip(groups, new_list, strict=False):
            for c in group:
                if c in code_to_new and code_to_new[c] != new:
                    raise ValueError(
                        f"Code {c} is assigned to both {code_to_new[c]} and {new}."
                    )
                code_to_new[c] = new

        # ------------------------------------------------------------------
        # Relabel the data, building every mask from the original volume
        # ------------------------------------------------------------------
        orig = self.data
        new_data = orig.copy()
        present = set(np.unique(orig).tolist())

        for group, new in zip(groups, new_list, strict=False):
            if not any(c in present for c in group):
                warnings.warn(
                    f"None of the codes {group} is present in the data.", stacklevel=2
                )
                continue
            new_data[np.isin(orig, group)] = new

        # ------------------------------------------------------------------
        # Rebuild the color table with one entry per final code
        # ------------------------------------------------------------------
        if getattr(self, "index", None) is not None:
            old_index = [int(c) for c in self.index]
            n_entries = len(old_index)

            names = list(
                getattr(self, "name", None) or [f"region_{c}" for c in old_index]
            )
            colors = list(getattr(self, "color", None) or ["#ffffff"] * n_entries)
            opac = getattr(self, "opacity", None)
            opacities = (
                [float(x) for x in opac]
                if opac is not None
                and hasattr(opac, "__len__")
                and len(opac) == n_entries
                else [1.0] * n_entries
            )

            # First position of every code in the original table
            pos_of: dict[int, int] = {}
            for pos, c in enumerate(old_index):
                pos_of.setdefault(c, pos)

            final: dict[int, tuple] = {}  # final code -> (name, color, opacity)

            # 1) Entries whose code is not replaced keep their own metadata
            for c, pos in pos_of.items():
                if c not in code_to_new:
                    final[c] = (names[pos], colors[pos], opacities[pos])

            # 2) Replaced codes: metadata of the lowest code in the group
            for group, new in zip(groups, new_list, strict=False):
                if not any(c in present for c in group):
                    continue
                if new in final:
                    if new not in code_to_new:  # existing, untouched region
                        warnings.warn(
                            f"Codes {group} were merged into the existing region "
                            f"{new} ('{final[new][0]}'), whose metadata is kept.",
                            stacklevel=2,
                        )
                    continue
                donor = next((c for c in group if c in pos_of), None)
                if donor is not None:
                    p = pos_of[donor]
                    final[new] = (names[p], colors[p], opacities[p])
                else:
                    final[new] = (f"region_{new}", "#ffffff", 1.0)

            self.index = list(final.keys())
            self.name = [v[0] for v in final.values()]
            self.color = cltcol.harmonize_colors(
                [v[1] for v in final.values()], output_format="hex"
            )
            self.opacity = [v[2] for v in final.values()]

            self.data = new_data
            self.adjust_values()  # drops codes absent from the data and sorts
        else:
            self.data = new_data

        self.parc_range()
        return self

    ######################################################################################################
    def replace_names(
        self,
        names2rep: str | list[str | list[str]] | np.ndarray | dict,
        new_names: str | list[str] | np.ndarray | None = None,
        match: str = "exact",
        bool_case: bool = True,
    ) -> "Parcellation":
        """
        Replace region names, supporting group replacements.

        Only the names change; labels, colors, and opacities are untouched.
        All matches are resolved against the ORIGINAL names, so overlapping or
        swapping mappings (e.g. {"lh": "rh", "rh": "lh"}) never cascade.

        Parameters
        ----------
        names2rep : str, list, np.ndarray, or dict
            Names to replace. Accepted forms:
            - dict: {old: new} or {(old1, old2): new}. new_names is ignored.
            - str: a single name (new_names must be a single name).
            - list of str: each entry is paired with one entry of new_names.
            - list of lists of str: every name in a group gets the same new name.
            - 1-D np.ndarray of str: same as a list of str.

        new_names : str, list of str, or np.ndarray, optional
            Replacement names, paired positionally with the groups in names2rep.
            Required unless names2rep is a dict.

        match : {"exact", "contains", "substring"}, optional
            - "exact": a region matches if its whole name equals an old name.
            - "contains": a region matches if its name contains an old name;
            the WHOLE name is replaced by the new name.
            - "substring": every occurrence of an old name inside a region name
            is replaced by the new text, in a single pass
            (e.g. "ctx-lh-" -> "ctx-left-").
            Default is "exact".

        bool_case : bool, optional
            Case-sensitive matching. Default is True.

        Returns
        -------
        Parcellation
            self, to allow method chaining.

        Raises
        ------
        TypeError
            If the inputs have unsupported types.
        ValueError
            If new_names is missing, the number of new names does not match the
            number of groups, match is invalid, an old name is empty, a region
            matches more than one group ("exact"/"contains"), or the same old
            text maps to different new texts ("substring").

        Warns
        -----
        UserWarning
            If a group matches no region, or the replacement creates duplicate names.

        Examples
        --------
        >>> # Dictionary, exact names
        >>> parc.replace_names({"ctx-lh-bankssts": "L_BSTS", "ctx-rh-bankssts": "R_BSTS"})
        >>>
        >>> # Paired lists
        >>> parc.replace_names(["ctx-lh-bankssts", "ctx-rh-bankssts"], ["L_BSTS", "R_BSTS"])
        >>>
        >>> # Group: several regions get the same name
        >>> parc.replace_names([["thal-lh-VA", "thal-lh-VL"]], ["thal-lh-ventral"])
        >>>
        >>> # Rename every region containing a substring
        >>> parc.replace_names("hipp-lh", "Left-Hippocampus", match="contains")
        >>>
        >>> # Swap hemisphere tags inside names, without cascading
        >>> parc.replace_names({"-lh-": "-rh-", "-rh-": "-lh-"}, match="substring")
        """
        import re
        from collections import Counter

        if not hasattr(self, "name") or self.name is None:
            raise AttributeError("Object must have a 'name' attribute")

        match = match.lower().strip()
        if match not in ("exact", "contains", "substring"):
            raise ValueError(
                f"match must be 'exact', 'contains' or 'substring', got '{match}'."
            )

        # ------------------------------------------------------------------
        # Normalize names2rep into groups of old names
        # ------------------------------------------------------------------
        if isinstance(names2rep, dict):
            if new_names is not None:
                warnings.warn(
                    "names2rep is a dict; new_names is ignored.", stacklevel=2
                )
            old_groups, new_list = [], []
            for old, new in names2rep.items():
                old_groups.append([old] if isinstance(old, str) else list(old))
                new_list.append(new)

        else:
            if isinstance(names2rep, str):
                old_groups = [[names2rep]]
            elif isinstance(names2rep, np.ndarray):
                if names2rep.ndim != 1:
                    raise TypeError("Unsupported numpy array shape for names2rep")
                old_groups = [[str(x)] for x in names2rep.tolist()]
            elif isinstance(names2rep, (list, tuple)):
                if len(names2rep) == 0:
                    raise ValueError("names2rep cannot be empty")
                if all(isinstance(x, str) for x in names2rep):
                    old_groups = [[x] for x in names2rep]
                elif all(isinstance(x, (list, tuple)) for x in names2rep):
                    old_groups = [list(x) for x in names2rep]
                else:
                    raise TypeError(
                        "names2rep must be a list of str or a list of lists of str"
                    )
            else:
                raise TypeError(
                    f"names2rep must be str, list, numpy array or dict, got {type(names2rep)}"
                )

            # Normalize new_names
            if new_names is None:
                raise ValueError("new_names is required unless names2rep is a dict.")
            if isinstance(new_names, str):
                new_list = [new_names]
            elif isinstance(new_names, (list, tuple, np.ndarray)):
                new_list = list(new_names)
            else:
                raise TypeError(
                    f"new_names must be str, list or numpy array, got {type(new_names)}"
                )

        # Validate contents
        for group in old_groups:
            if len(group) == 0 or not all(
                isinstance(o, str) and o != "" for o in group
            ):
                raise ValueError("Every old name must be a non-empty string.")
        if not all(isinstance(n, (str, np.str_)) for n in new_list):
            raise TypeError("All new names must be strings.")
        new_list = [str(n) for n in new_list]

        if len(new_list) != len(old_groups):
            raise ValueError(
                f"Number of new names ({len(new_list)}) must equal "
                f"number of groups ({len(old_groups)}) to be replaced"
            )

        original = list(self.name)
        updated = list(original)

        # ------------------------------------------------------------------
        # "exact" and "contains": replace the whole name
        # ------------------------------------------------------------------
        if match in ("exact", "contains"):
            claimed: dict[int, int] = {}  # region position -> group position

            for g, (olds, new) in enumerate(zip(old_groups, new_list, strict=False)):
                if match == "exact":
                    if bool_case:
                        olds_set = set(olds)
                        hits = [k for k, n in enumerate(original) if n in olds_set]
                    else:
                        olds_set = {o.lower() for o in olds}
                        hits = [
                            k for k, n in enumerate(original) if n.lower() in olds_set
                        ]
                else:
                    hits = cltmisc.get_indexes_by_substring(
                        input_list=original,
                        or_filter=olds,
                        invert=False,
                        bool_case=bool_case,
                    )

                if len(hits) == 0:
                    warnings.warn(f"No region matched {olds}; skipped.", stacklevel=2)
                    continue

                for k in hits:
                    if k in claimed and claimed[k] != g:
                        raise ValueError(
                            f"Region '{original[k]}' matches both {old_groups[claimed[k]]} "
                            f"and {olds}. Use more specific names."
                        )
                    claimed[k] = g
                    updated[k] = new

        # ------------------------------------------------------------------
        # "substring": replace text inside names, single pass
        # ------------------------------------------------------------------
        else:
            lookup: dict[str, str] = {}
            for olds, new in zip(old_groups, new_list, strict=False):
                for o in olds:
                    key = o if bool_case else o.lower()
                    if key in lookup and lookup[key] != new:
                        raise ValueError(
                            f"'{o}' is mapped to both '{lookup[key]}' and '{new}'."
                        )
                    lookup[key] = new

            # Longest first, so "ctx-lh-" wins over "lh"
            olds_sorted = sorted(lookup, key=len, reverse=True)
            flags = 0 if bool_case else re.IGNORECASE
            pattern = re.compile("|".join(re.escape(o) for o in olds_sorted), flags)

            def _sub(m):
                text = m.group(0)
                return lookup[text if bool_case else text.lower()]

            updated = [pattern.sub(_sub, n) for n in original]

            for o in olds_sorted:
                if not any(re.search(re.escape(o), n, flags) for n in original):
                    warnings.warn(f"No region name contains '{o}'.", stacklevel=2)

        # ------------------------------------------------------------------
        # Warn about duplicates introduced by the replacement
        # ------------------------------------------------------------------
        before = Counter(original)
        after = Counter(updated)
        new_dups = sorted(n for n, c in after.items() if c > 1 and before.get(n, 0) < c)
        if new_dups:
            warnings.warn(
                f"Replacement produced duplicate region names: {new_dups}. "
                "Name-based methods will treat these regions as one.",
                stacklevel=2,
            )

        self.name = updated
        return self

    ######################################################################################################
    def parc_range(self) -> None:
        """
        Update minimum and maximum label values in parcellation.

        Sets minlab and maxlab attributes based on non-zero values in data.

        Returns
        -------
        tuple : (minlab, maxlab)
            minlab : int
                Minimum label value (excluding zero).

            maxlab : int
                Maximum label value.

        Examples
        --------
        >>> parc.parc_range()
        >>> print(f"Label range: {parc.minlab} - {parc.maxlab}")
        """

        # Get unique non-zero elements
        unique_codes = np.unique(self.data)
        nonzero_codes = unique_codes[unique_codes != 0]

        if nonzero_codes.size > 0:
            self.minlab = np.min(nonzero_codes)
            self.maxlab = np.max(nonzero_codes)
        else:
            self.minlab = 0
            self.maxlab = 0

        return self.minlab, self.maxlab

    #######################################################################################################
    def compute_morphometry_table(
        self,
        output_table: str | Path = None,
        add_bids_entities: bool = False,
        map_files: str | Path | list = None,
        map_ids: str | list = None,
        units: str | list = "unknown",
        exclude_by_code: list | np.ndarray = None,
        exclude_by_name: list | str = None,
        include_by_code: list | np.ndarray = None,
        include_by_name: list | str = None,
        include_global: bool = True,
    ) -> pd.DataFrame:
        """
        Compute morphometry table for all regions in parcellation.

        Computes region volumes and, optionally, regional statistics of additional
        maps. The result is stored in the `morphometry` attribute and returned.

        Parameters
        ----------
        output_table : str or Path, optional
            Path to save the output table as CSV. If None, the table is not saved.
            Default is None.

        add_bids_entities : bool, optional
            Whether to add BIDS entities to the output. Default is False.

        map_files : str, Path, or list of str/Path, optional
            Paths to additional map files. Regional statistics are computed for
            each existing file. If None, only volumes are computed. Default is None.

        map_ids : str or list of str, optional
            IDs for the additional maps, one per map file. If None, the file names
            (without extension) are used. A single string is only valid when a
            single map file is given. Default is None.

        units : str or list of str, optional
            Units for the additional maps. A single string is applied to all maps;
            a list must have one entry per map file. If None, "unknown" is used.
            Default is "unknown".

        exclude_by_code : list or np.ndarray, optional
            Region codes to exclude. Default is None.

        exclude_by_name : list or str, optional
            Region names to exclude. Default is None.

        include_by_code : list or np.ndarray, optional
            Region codes to include. If None, all regions are included. Default is None.

        include_by_name : list or str, optional
            Region names to include. If None, all regions are included. Default is None.

        include_global : bool, optional
            Whether to include global morphometry metrics. Default is True.

        Returns
        -------
        pd.DataFrame
            Morphometry table with the volumes and the statistics of every map
            that was processed successfully.

        Raises
        ------
        TypeError
            If output_table, map_files, map_ids or units have incorrect types.

        FileNotFoundError
            If the output directory does not exist.

        ValueError
            If the number of map_ids or units does not match the number of map_files,
            or if a single map_id is given for several map files.

        Warns
        -----
        UserWarning
            If a map file does not exist or a map fails to be processed. Missing and
            failed maps are skipped; the remaining ones are still computed.

        Examples
        --------
        >>> # Volumes only
        >>> parc.compute_morphometry_table(output_table='morphometry_base.csv')
        >>>
        >>> # Volumes plus two maps
        >>> parc.compute_morphometry_table(
        ...     output_table='morphometry.csv',
        ...     add_bids_entities=True,
        ...     map_files=['map1.nii.gz', 'map2.nii.gz'],
        ...     map_ids=['map1', 'map2'],
        ...     units=['mm^3', 'unknown']
        ... )
        >>>
        >>> # Single map given as Path
        >>> parc.compute_morphometry_table(
        ...     output_table=Path('morphometry_single.csv'),
        ...     map_files=Path('single_map.nii.gz'),
        ...     map_ids='single_map',
        ...     units='mm^3'
        ... )
        """

        from rich.markup import escape

        from . import morphometrytools as cltmorpho

        # Loading the default configuration file
        cwd = os.path.dirname(os.path.abspath(__file__))

        # Default to the standard configuration file
        def_config_file = os.path.join(cwd, "config", "config.json")

        # Read the config file in order to get default settings and units
        config = cltmisc.load_json(def_config_file)
        vol_units = config["metrics_units"]["volume"]

        # ------------------------------------------------------------------
        # Validate the output path first, so a bad path fails before computing
        # ------------------------------------------------------------------
        if output_table is not None:
            if not isinstance(output_table, (str, Path)):
                raise TypeError(
                    f"output_table must be a string or Path, got {type(output_table)}"
                )
            output_table = Path(output_table)
            if not output_table.parent.exists():
                raise FileNotFoundError(
                    f"Output directory does not exist: {output_table.parent}"
                )

        # ------------------------------------------------------------------
        # Normalize map_files, map_ids and units
        # ------------------------------------------------------------------
        fin_maps, fin_map_ids, fin_units = [], [], []

        if map_files is not None:
            if isinstance(map_files, (str, Path)):
                map_files = [str(map_files)]
            elif isinstance(map_files, (list, tuple)):
                if not all(isinstance(f, (str, Path)) for f in map_files):
                    raise TypeError(
                        "All items in map_files must be strings or Path objects"
                    )
                map_files = [str(f) for f in map_files]
            else:
                raise TypeError(
                    f"map_files must be str, Path, or list, got {type(map_files)}"
                )

            n_maps = len(map_files)

            # map_ids
            if map_ids is None:
                map_ids = [cltmisc.get_real_basename(f) for f in map_files]
            elif isinstance(map_ids, str):
                if n_maps != 1:
                    raise ValueError(
                        f"A single map_id was given for {n_maps} map files. "
                        "Provide one ID per map file, or set map_ids=None."
                    )
                map_ids = [map_ids]
            elif isinstance(map_ids, (list, tuple)):
                if not all(isinstance(mid, str) for mid in map_ids):
                    raise TypeError("All items in map_ids must be strings")
                if len(map_ids) != n_maps:
                    raise ValueError(
                        f"Number of map_ids ({len(map_ids)}) does not match "
                        f"number of map_files ({n_maps})."
                    )
                map_ids = list(map_ids)
            else:
                raise TypeError(f"map_ids must be str or list, got {type(map_ids)}")

            # units
            if units is None:
                units = ["unknown"] * n_maps
            elif isinstance(units, str):
                units = [units] * n_maps
            elif isinstance(units, (list, tuple)):
                if not all(isinstance(u, str) for u in units):
                    raise TypeError("All items in units must be strings")
                if len(units) != n_maps:
                    raise ValueError(
                        f"Number of units ({len(units)}) does not match "
                        f"number of map_files ({n_maps})."
                    )
                units = list(units)
            else:
                raise TypeError(f"units must be str or list, got {type(units)}")

            # Keep only the existing files
            for map_file, map_id, unit in zip(map_files, map_ids, units, strict=True):
                if os.path.exists(map_file):
                    fin_maps.append(map_file)
                    fin_map_ids.append(map_id)
                    fin_units.append(unit)
                else:
                    warnings.warn(
                        f"Map file not found, skipped: {map_file}", stacklevel=2
                    )

            if len(fin_maps) == 0:
                warnings.warn(
                    "No valid map files found. Computing only base morphometry.",
                    stacklevel=2,
                )

        n_valid_maps = len(fin_maps)
        failed_maps = []

        # ------------------------------------------------------------------
        # Compute, with one progress step per table
        # ------------------------------------------------------------------
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            expand=True,
        ) as progress:

            task = progress.add_task(
                "[bold green]Computing base morphometry: volume[/bold green] "
                f"([yellow]{vol_units}[/yellow])",
                total=n_valid_maps,
            )

            # --- Step 1: base morphometry ---
            morphometry_table, *_ = cltmorpho.compute_reg_volume_fromparcellation(
                self,
                add_bids_entities=add_bids_entities,
                include_by_code=include_by_code,
                include_by_name=include_by_name,
                exclude_by_code=exclude_by_code,
                exclude_by_name=exclude_by_name,
                include_global=include_global,
            )
            progress.advance(task)

            # --- Step 2: additional maps ---
            for i, (map_file, map_id, unit) in enumerate(
                zip(fin_maps, fin_map_ids, fin_units, strict=True), start=1
            ):
                # Describe the step before running it, advance after
                progress.update(
                    task,
                    description=(
                        f"[bold green]Processing map {i}/{n_valid_maps}[/bold green]"
                        f" • {escape(map_id)} ({escape(unit)})"
                    ),
                )
                try:
                    df, _, _ = cltmorpho.compute_reg_val_fromparcellation(
                        map_file,
                        self,
                        add_bids_entities=add_bids_entities,
                        metric=map_id,
                        units=unit,
                        include_by_code=include_by_code,
                        include_by_name=include_by_name,
                        exclude_by_code=exclude_by_code,
                        exclude_by_name=exclude_by_name,
                        include_global=include_global,
                    )
                    morphometry_table = pd.concat([morphometry_table, df], axis=0)

                except Exception as e:
                    failed_maps.append(map_id)
                    progress.console.print(
                        f"[yellow]WARNING:[/yellow] Map {escape(map_id)} "
                        f"failed with: {escape(str(e))}"
                    )
                finally:
                    progress.advance(task)

            # Final summary
            if n_valid_maps == 0:
                final_msg = "[bold green]✓[/bold green] Completed base morphometry"
            elif failed_maps:
                n_ok = n_valid_maps - len(failed_maps)
                final_msg = (
                    f"[bold yellow]✓[/bold yellow] Completed: 1 base + "
                    f"{n_ok}/{n_valid_maps} map(s) ({len(failed_maps)} failed)"
                )
            else:
                final_msg = (
                    f"[bold green]✓[/bold green] Completed: 1 base + "
                    f"{n_valid_maps} map(s)"
                )

            progress.update(task, description=final_msg)

        # Warn outside the progress context, so callers can catch it
        if failed_maps:
            warnings.warn(
                f"{len(failed_maps)} map(s) failed and are missing from the table: "
                f"{failed_maps}",
                stacklevel=2,
            )

        self.morphometry = morphometry_table

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        if output_table is not None:
            morphometry_table.to_csv(output_table, index=False)
            print(f"Saved morphometry table to {output_table}")

        return morphometry_table

    ######################################################################################################
    def compute_volume_table(
        self,
        exclude_by_code: list | np.ndarray = None,
        exclude_by_name: list | str = None,
        include_by_code: list | np.ndarray = None,
        include_by_name: list | str = None,
        include_global: bool = True,
        add_bids_entities: bool = False,
        output_table: str | Path = None,
    ):
        """
        Compute volume table for all regions in parcellation.

        Sets volumetable attribute containing region volumes and statistics.

        exclude_by_code : list or np.ndarray, optional
            Region codes to exclude from the analysis. If None, no regions are excluded by code.
            Useful for excluding regions like ventricles or non-brain tissue.

        exclude_by_name : list or str, optional
            Region names to exclude from the analysis. If None, no regions are excluded by name.
            Example: ["Ventricles", "White-Matter"] to focus only on gray matter regions.

        include_by_code : list or np.ndarray, optional
            Region codes to include in the analysis. If None, all regions are included.
            Useful for focusing on specific regions of interest.

        include_by_name : list or str, optional
            Region names to include in the analysis. If None, all regions are included.
            Example: ["Cortex", "Hippocampus"] to focus on specific structures.

        add_bids_entities : bool, default=False
            Whether to include BIDS entities as columns in the resulting DataFrame.
            This extracts subject, session, and other metadata from the filename.

        include_global : bool, default=True
            Whether to include a the total volume in the output table.
            If True, adds a row for the total volume calculated from the parcellation.

        output_table : str or Path, optional
            Path to save the resulting volume table. If None, the table is not saved to disk.

        Examples
        --------
        >>> parc.compute_volume_table()
        >>> volume_df, _ = parc.volumetable
        >>> print(volume_df.head())
        """

        from . import morphometrytools as cltmorpho

        volume_table = cltmorpho.compute_reg_volume_fromparcellation(
            self,
            exclude_by_code=exclude_by_code,
            exclude_by_name=exclude_by_name,
            include_by_code=include_by_code,
            include_by_name=include_by_name,
            include_global=include_global,
            output_table=output_table,
            add_bids_entities=add_bids_entities,
        )

        return volume_table

    ######################################################################################################
    def print_properties(self):
        """
        Print all attributes and methods of the parcellation object.

        Displays non-private attributes and methods for object inspection.

        Examples
        --------
        >>> parc.print_properties()
        Attributes:
        data
        affine
        index
        ...
        Methods:
        keep_by_code
        save_parcellation
        ...
        """

        # Get and print attributes and methods
        attributes_and_methods = [
            attr for attr in dir(self) if not callable(getattr(self, attr))
        ]
        methods = [method for method in dir(self) if callable(getattr(self, method))]

        print("Attributes:")
        for attribute in attributes_and_methods:
            if not attribute.startswith("__"):
                print(attribute)

        print("\nMethods:")
        for method in methods:
            if not method.startswith("__"):
                print(method)

    #######################################################################################################
    def compute_fc_matrix(
        self,
        data: str | Path | np.ndarray,
        method: str = "pearson",
        *,
        z_transform: bool = False,
        absolute: bool = False,
        threshold: float | None = None,
        normalize_rows: bool = False,
        vols_to_delete: str | list | np.ndarray = None,
        ts_method: str = "nilearn",
        region_labels: list[int] | np.ndarray = None,
        region_names: list[str] | str = None,
    ) -> cltcon.Connectome:
        """Compute a functional connectivity (FC) matrix from a ROI × time series or 4-D NIfTI file.

        Each entry ``FC[i, j]`` reflects the pairwise association between the
        time series of ROI *i* and ROI *j*.  The result is always a symmetric
        square matrix of shape ``(n_rois, n_rois)`` with ones on the diagonal
        (except for ``"partial"`` and ``"mutual_info"``).

        Parameters
        ----------
        data : str, Path, or np.ndarray
            Either a path to a NIfTI file, a pre-loaded 2-D ROI × time matrix,
            or a 4-D NIfTI array (handled via ``get_regionwise_timeseries``).

        method : str, default ``"pearson"``
            Correlation / association method.  Supported values:

            ``"pearson"``
                Standard Pearson *r*.  Fast; assumes linearity.

            ``"spearman"``
                Rank-based Spearman *ρ*.  Robust to monotone non-linearities
                and mild outliers.

            ``"kendall"``
                Kendall *τ-b*.  More robust than Spearman but *O(n²)* in time
                points — avoid for very long series.

            ``"partial"``
                Partial correlation via the precision matrix (inverse of the
                covariance).  Controls for the linear influence of all other
                ROIs.  Requires ``n_timepoints > n_rois``.

            ``"mutual_info"``
                Normalised mutual information (scikit-learn).  Captures
                non-linear dependencies; values in ``[0, 1]``.

        z_transform : bool, default False
            Apply Fisher's *r*-to-*z* transform: ``z = arctanh(r)``.
            Useful before group-level statistics.  Not applied for
            ``"mutual_info"``.

        absolute : bool, default False
            Return ``|FC|`` instead of signed values.  Useful when only
            connection *strength* matters, not sign.

        threshold : float, optional
            Zero out all entries whose absolute value is below *threshold*
            after all other transforms.

        normalize_rows : bool, default False
            Z-score each row of *data* before computing the FC matrix.
            Equivalent to ``create_carpet_plot``'s ``normalize_rows``.

        vols_to_delete : str, list, or np.ndarray, optional
            Volume indices to discard before computing the FC matrix.
            Passed directly to ``get_regionwise_timeseries``.

        ts_method : str, default ``"nilearn"``
            Time-series extraction backend passed to ``get_regionwise_timeseries``.

        Returns
        -------
        fc_connectome : cltcon.Connectome
            A Connectome object containing the FC matrix and metadata.

        Raises
        ------
        ValueError
            On unsupported method, wrong data dimensionality, or insufficient
            time points for partial correlation.

        Examples
        --------
        >>> import numpy as np
        >>> data = np.random.randn(90, 200)      # 90 ROIs, 200 time points

        >>> fc_r   = compute_fc_matrix(data, method="pearson")
        >>> fc_rho = compute_fc_matrix(data, method="spearman", z_transform=True)
        >>> fc_par = compute_fc_matrix(data, method="partial")
        >>> fc_mi  = compute_fc_matrix(data, method="mutual_info")
        """

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        SUPPORTED = {"pearson", "spearman", "kendall", "partial", "mutual_info"}

        if isinstance(data, RegionTimeSeries):
            fc_connectome = data.compute_fc_matrix(
                method=method,
                z_transform=z_transform,
                absolute=absolute,
                threshold=threshold,
                region_labels=region_labels,
                region_names=region_names,
                normalize_rows=normalize_rows,
            )
            return fc_connectome

        # Check if include_by_code and include_by_name are different from None at the same time
        if region_labels is not None and region_names is not None:
            region_labels = None
            print(
                "Both region_labels and region_names were specified. Ignoring region_labels and using region_names for region selection."
            )

        temp_parc = copy.deepcopy(self)

        # Apply inclusion if specified
        if region_labels is not None:
            temp_parc.keep_by_code(codes2keep=region_labels)

        if region_names is not None:
            temp_parc.keep_by_name(names2keep=region_names)

        if vols_to_delete is not None:
            # Ensure vols_to_delete is a list
            if not isinstance(vols_to_delete, list):
                vols_to_delete = [vols_to_delete]

            # Convert vols_to_delete to a flat list of integers
            vols_to_delete = cltmisc.build_indices(vols_to_delete, nonzeros=False)

            # Check if vols_to_delete is not empty
            if len(vols_to_delete) == 0:
                vols_to_delete = None  # Reset to None for get_regionwise_timeseries

        if isinstance(data, (str, Path)):
            if os.path.exists(data):
                if isinstance(data, str):
                    data = temp_parc.get_regionwise_timeseries(
                        data, vols_to_delete=vols_to_delete, method=ts_method
                    )
                    region_names = data.region_names
                    data = data.data

            else:
                raise ValueError(f"Data file does not exist: {data}")

        elif isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=float)
            if data.ndim == 4:
                data = temp_parc.get_regionwise_timeseries(
                    data, vols_to_delete=vols_to_delete, method=ts_method
                )
                data = data.data
            elif data.ndim == 2:
                if vols_to_delete is not None:
                    if max(vols_to_delete) >= data.shape[1]:
                        raise ValueError(
                            f"vols_to_delete contains indices that exceed the number of time points ({data.shape[1]})."
                        )
                    data = np.delete(data, vols_to_delete, axis=1)
            else:

                raise ValueError(
                    f"Data array must be 2-D (n_rois × n_timepoints) or 4-D (n_x × n_y × n_z × n_timepoints), got shape {data.shape}."
                )

        method = method.lower().strip()
        if method not in SUPPORTED:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: {sorted(SUPPORTED)}."
            )

        n_rois, n_timepoints = data.shape

        if n_rois != len(temp_parc.index):
            raise ValueError(
                f"Number of ROIs in data ({n_rois}) does not match number of regions in parcellation ({len(temp_parc.index)})."
            )

        # ------------------------------------------------------------------
        # Optional row-wise z-scoring
        # ------------------------------------------------------------------
        if normalize_rows:
            mu = data.mean(axis=1, keepdims=True)
            sigma = data.std(axis=1, keepdims=True)
            sigma[sigma == 0] = 1.0
            data = (data - mu) / sigma

        # ------------------------------------------------------------------
        # Compute FC
        # ------------------------------------------------------------------
        if method == "pearson":
            fc = np.corrcoef(data)  # (n_rois, n_rois), fast vectorised

        elif method == "spearman":
            # Rank each row, then run Pearson on the ranks — identical to
            # scipy.stats.spearmanr but avoids the slow Python loop.
            ranked = np.apply_along_axis(stats.rankdata, axis=1, arr=data)
            fc = np.corrcoef(ranked)

        elif method == "kendall":
            fc = np.eye(n_rois)
            for i in range(n_rois):
                for j in range(i + 1, n_rois):
                    tau, _ = stats.kendalltau(data[i], data[j])
                    fc[i, j] = fc[j, i] = tau

        elif method == "partial":
            if n_timepoints <= n_rois:
                raise ValueError(
                    f"Partial correlation requires n_timepoints ({n_timepoints}) "
                    f"> n_rois ({n_rois})."
                )
            cov = np.cov(data)  # (n_rois, n_rois)
            prec = pinv(cov)  # precision matrix
            # Normalise to correlation scale: pcor[i,j] = -prec[i,j] / sqrt(prec[i,i]*prec[j,j])
            d = np.sqrt(np.diag(prec))
            fc = -prec / np.outer(d, d)
            np.fill_diagonal(fc, 1.0)

        elif method == "mutual_info":
            from sklearn.metrics import mutual_info_score
            from sklearn.preprocessing import KBinsDiscretizer

            # Discretise each time series into bins for MI estimation
            n_bins = max(10, int(np.sqrt(n_timepoints)))
            est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
            data_disc = est.fit_transform(data.T).T.astype(
                int
            )  # (n_rois, n_timepoints)

            mi_raw = np.zeros((n_rois, n_rois))
            for i in range(n_rois):
                for j in range(i, n_rois):
                    mi = mutual_info_score(data_disc[i], data_disc[j])
                    mi_raw[i, j] = mi_raw[j, i] = mi

            # Normalise: NMI(i,j) = MI(i,j) / sqrt(H(i)*H(j))  ∈ [0, 1]
            entropies = np.diag(mi_raw)  # MI(x, x) == H(x)
            denom = np.sqrt(np.outer(entropies, entropies))
            denom[denom == 0] = 1.0
            fc = mi_raw / denom

        # ------------------------------------------------------------------
        # Post-processing
        # ------------------------------------------------------------------
        # Clip numerical noise outside [-1, 1] for correlation-based methods
        if method not in {"mutual_info"}:
            fc = np.clip(fc, -1.0, 1.0)

        if z_transform and method != "mutual_info":
            # arctanh is undefined at ±1; clip diagonal / extreme values
            fc_clip = np.clip(fc, -0.9999, 0.9999)
            fc = np.arctanh(fc_clip)

        if absolute:
            fc = np.abs(fc)

        if threshold is not None:
            fc[np.abs(fc) < threshold] = 0.0

        fc_connectome = cltcon.Connectome(
            fc,
            region_names=temp_parc.name,
            region_index=temp_parc.index,
            region_colors=temp_parc.color,
            modality="functional",
        )

        return fc_connectome


########################################################################################################
class RegionTimeSeries:
    """
    Class for handling region-wise time series extracted from parcellations.

    Attributes
    ----------
    data : np.ndarray
        2-D array of shape (n_regions, n_timepoints) containing the time series for each region.

    region_names : list of str
        List of region names corresponding to the rows of `data`.

    region_colors : list of tuples or list of str
        List of region colors corresponding to the rows of `data`. Colors can be in RGB tuples or hex string format.

    """

    def __init__(
        self,
        data: np.ndarray,
        region_names: list[str] = None,
        region_colors: list[tuple[float, float, float] | str] = None,
        method: str = "clabtoolkit",
    ):
        """
        Initialize the RegionTimeSeries object.

        Parameters
        ----------
        data : np.ndarray
            2-D array of shape (n_regions, n_timepoints) containing the time series for each region.

        region_names : list of str, optional
            List of region names corresponding to the rows of `data`. If None, regions will be named "Region 1", "Region 2", etc.

        region_colors : list of tuples or list of str, optional
            List of region colors corresponding to the rows of `data`. Colors can be in RGB tuples (e.g., (255, 0, 0)) or hex string format (e.g., "#FF0000"). If None, default colors will be assigned.

        method : str, optional
            Method to create time series. Default is "clabtoolkit". Supported values:
            - "clabtoolkit": Use mean time series for each region.
            - "nilearn": Use Nilearn's NiftiLabelsMasker to extract time series. Requires Nilearn to be installed.
        """

        self.data = data

        n_regions = data.shape[0]
        n_timepoints = data.shape[1]
        self._n_regions = n_regions
        self._n_timepoints = n_timepoints

        # Validate and assign region names and colors
        if region_names is None:
            indexes = np.arange(1, n_regions + 1)
            region_names = cltmisc.create_names_from_indices(indexes)

        if region_colors is None:
            region_colors = cltcol.create_distinguishable_colors(
                n_regions, output_format="hex"
            )

        if region_names is not None:
            if len(region_names) != n_regions:
                raise ValueError(
                    f"Length of region_names ({len(region_names)}) must match number of regions in data ({n_regions})."
                )
            self.region_names = region_names

        if region_colors is not None:
            if len(region_colors) != n_regions:
                raise ValueError(
                    f"Length of region_colors ({len(region_colors)}) must match number of regions in data ({n_regions})."
                )
            self.region_colors = region_colors

    ##############################################################################################
    def compute_fc_matrix(
        self,
        method: str = "pearson",
        *,
        z_transform: bool = False,
        absolute: bool = False,
        threshold: float | None = None,
        normalize_rows: bool = False,
        vols_to_delete: str | list | np.ndarray = None,
        region_names: list[str] | str = None,
    ) -> cltcon.Connectome:
        """Compute a functional connectivity (FC) matrix.

        Each entry ``FC[i, j]`` reflects the pairwise association between the
        time series of ROI *i* and ROI *j*.  The result is always a symmetric
        square matrix of shape ``(n_rois, n_rois)`` with ones on the diagonal
        (except for ``"partial"`` and ``"mutual_info"``).

        Parameters
        ----------
        method : str, default ``"pearson"``
            Correlation / association method.  Supported values:

            ``"pearson"``
                Standard Pearson *r*.  Fast; assumes linearity.

            ``"spearman"``
                Rank-based Spearman *ρ*.  Robust to monotone non-linearities
                and mild outliers.

            ``"kendall"``
                Kendall *τ-b*.  More robust than Spearman but *O(n²)* in time
                points — avoid for very long series.

            ``"partial"``
                Partial correlation via the precision matrix (inverse of the
                covariance).  Controls for the linear influence of all other
                ROIs.  Requires ``n_timepoints > n_rois``.

            ``"mutual_info"``
                Normalised mutual information (scikit-learn).  Captures
                non-linear dependencies; values in ``[0, 1]``.

        z_transform : bool, default False
            Apply Fisher's *r*-to-*z* transform: ``z = arctanh(r)``.
            Useful before group-level statistics.  Not applied for
            ``"mutual_info"``.

        absolute : bool, default False
            Return ``|FC|`` instead of signed values.  Useful when only
            connection *strength* matters, not sign.

        threshold : float, optional
            Zero out all entries whose absolute value is below *threshold*
            after all other transforms.

        normalize_rows : bool, default False
            Z-score each row of *data* before computing the FC matrix.
            Equivalent to ``create_carpet_plot``'s ``normalize_rows``.

        vols_to_delete : str, list, or np.ndarray, optional
            Volume indices to discard before computing the FC matrix.
            Passed directly to ``get_regionwise_timeseries``.

        Returns
        -------
        fc_connectome : cltcon.Connectome
            A Connectome object containing the FC matrix and metadata.

        Raises
        ------
        ValueError
            On unsupported method, wrong data dimensionality, or insufficient
            time points for partial correlation.

        Examples
        --------
        >>> import numpy as np
        >>> data = np.random.randn(90, 200)      # 90 ROIs, 200 time points

        >>> fc_r   = compute_fc_matrix(data, method="pearson")
        >>> fc_rho = compute_fc_matrix(data, method="spearman", z_transform=True)
        >>> fc_par = compute_fc_matrix(data, method="partial")
        >>> fc_mi  = compute_fc_matrix(data, method="mutual_info")
        """

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        SUPPORTED = {"pearson", "spearman", "kendall", "partial", "mutual_info"}

        if region_names is not None:
            if isinstance(region_names, str):
                region_names = [region_names]
            elif isinstance(region_names, list):
                if not all(isinstance(name, str) for name in region_names):
                    raise TypeError("All items in region_names must be strings")
            else:
                raise TypeError(
                    f"region_names must be str or list, got {type(region_names)}"
                )

            indices = cltmisc.get_indexes_by_substring(self.region_names, region_names)
            data = self.data[indices, :]
            region_names = [self.region_names[i] for i in indices]
            region_colors = [self.region_colors[i] for i in indices]
        else:
            data = self.data
            region_names = self.region_names
            region_colors = self.region_colors

        if vols_to_delete is not None:
            # Ensure vols_to_delete is a list
            if not isinstance(vols_to_delete, list):
                vols_to_delete = [vols_to_delete]

            # Convert vols_to_delete to a flat list of integers
            vols_to_delete = cltmisc.build_indices(vols_to_delete, nonzeros=False)

            # Check if vols_to_delete is not empty
            if len(vols_to_delete) == 0:
                vols_to_delete = None  # Reset to None for get_regionwise_timeseries

            if max(vols_to_delete) >= data.shape[1]:
                raise ValueError(
                    f"vols_to_delete contains indices that exceed the number of time points ({data.shape[1]})."
                )
            data = np.delete(data, vols_to_delete, axis=1)

        method = method.lower().strip()
        if method not in SUPPORTED:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: {sorted(SUPPORTED)}."
            )

        n_rois, n_timepoints = data.shape

        # ------------------------------------------------------------------
        # Optional row-wise z-scoring
        # ------------------------------------------------------------------
        if normalize_rows:
            mu = data.mean(axis=1, keepdims=True)
            sigma = data.std(axis=1, keepdims=True)
            sigma[sigma == 0] = 1.0
            data = (data - mu) / sigma

        # ------------------------------------------------------------------
        # Compute FC
        # ------------------------------------------------------------------
        if method == "pearson":
            fc = np.corrcoef(data)  # (n_rois, n_rois), fast vectorised

        elif method == "spearman":
            # Rank each row, then run Pearson on the ranks — identical to
            # scipy.stats.spearmanr but avoids the slow Python loop.
            ranked = np.apply_along_axis(stats.rankdata, axis=1, arr=data)
            fc = np.corrcoef(ranked)

        elif method == "kendall":
            fc = np.eye(n_rois)
            for i in range(n_rois):
                for j in range(i + 1, n_rois):
                    tau, _ = stats.kendalltau(data[i], data[j])
                    fc[i, j] = fc[j, i] = tau

        elif method == "partial":
            if n_timepoints <= n_rois:
                raise ValueError(
                    f"Partial correlation requires n_timepoints ({n_timepoints}) "
                    f"> n_rois ({n_rois})."
                )
            cov = np.cov(data)  # (n_rois, n_rois)
            prec = pinv(cov)  # precision matrix
            # Normalise to correlation scale: pcor[i,j] = -prec[i,j] / sqrt(prec[i,i]*prec[j,j])
            d = np.sqrt(np.diag(prec))
            fc = -prec / np.outer(d, d)
            np.fill_diagonal(fc, 1.0)

        elif method == "mutual_info":
            from sklearn.metrics import mutual_info_score
            from sklearn.preprocessing import KBinsDiscretizer

            # Discretise each time series into bins for MI estimation
            n_bins = max(10, int(np.sqrt(n_timepoints)))
            est = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
            data_disc = est.fit_transform(data.T).T.astype(
                int
            )  # (n_rois, n_timepoints)

            mi_raw = np.zeros((n_rois, n_rois))
            for i in range(n_rois):
                for j in range(i, n_rois):
                    mi = mutual_info_score(data_disc[i], data_disc[j])
                    mi_raw[i, j] = mi_raw[j, i] = mi

            # Normalise: NMI(i,j) = MI(i,j) / sqrt(H(i)*H(j))  ∈ [0, 1]
            entropies = np.diag(mi_raw)  # MI(x, x) == H(x)
            denom = np.sqrt(np.outer(entropies, entropies))
            denom[denom == 0] = 1.0
            fc = mi_raw / denom

        # ------------------------------------------------------------------
        # Post-processing
        # ------------------------------------------------------------------
        # Clip numerical noise outside [-1, 1] for correlation-based methods
        if method not in {"mutual_info"}:
            fc = np.clip(fc, -1.0, 1.0)

        if z_transform and method != "mutual_info":
            # arctanh is undefined at ±1; clip diagonal / extreme values
            fc_clip = np.clip(fc, -0.9999, 0.9999)
            fc = np.arctanh(fc_clip)

        if absolute:
            fc = np.abs(fc)

        if threshold is not None:
            fc[np.abs(fc) < threshold] = 0.0

        fc_connectome = cltcon.Connectome(
            fc,
            region_names=region_names,
            region_index=list(range(1, n_rois + 1)),
            region_colors=region_colors,
        )

        return fc_connectome

    #########################################################################################################
    def get_info(self) -> None:
        """
        Display a formatted summary of the RegionTimeSeries object.

        Shows data shape, dtype, basic statistics, and a preview of the
        region names / colours (up to ``_MAX_SHOWN`` rows; the rest are
        summarised).

        Examples
        --------
        >>> rts.get_info()
        """
        _MAX_SHOWN = 10
        _NAME_W = 38  # max chars for a region name column
        _COLOR_W = 18  # max chars for a colour column

        # Inner content width: index(6) + space(1) + name + space(1) + color
        _INNER_W = 4 + 1 + _NAME_W + 1 + _COLOR_W  # = 64
        _WIDTH = _INNER_W + 2  # +2 for the leading "  " indent

        def _border(left="╠", mid="═", right="╣") -> None:
            print(f"{left}{mid * _WIDTH}{right}")

        def _row(content: str) -> None:
            # Pad / truncate so the closing '║' always lands in the same column.
            print(f"║{content[:_WIDTH]:<{_WIDTH}}║")

        def _trunc(s: str, max_w: int) -> str:
            return s if len(s) <= max_w else s[: max_w - 1] + "…"

        # ------------------------------------------------------------------ #
        # Gather fields (graceful fallbacks if somehow unset)
        # ------------------------------------------------------------------ #
        n_regions = getattr(self, "_n_regions", self.data.shape[0])
        n_timepoints = getattr(self, "_n_timepoints", self.data.shape[1])
        region_names = getattr(self, "region_names", [])
        region_colors = getattr(self, "region_colors", [])

        # ------------------------------------------------------------------ #
        # Header
        # ------------------------------------------------------------------ #
        _border("╔", "═", "╗")
        _row("  REGION TIME SERIES".center(_WIDTH))
        _border()

        # ------------------------------------------------------------------ #
        # Data block
        # ------------------------------------------------------------------ #
        _row("  DATA")
        _row(f"    Shape      : {n_regions} regions  ×  {n_timepoints} timepoints")
        _row(f"    Dtype      : {self.data.dtype}")
        try:
            _row(f"    Min / Max  : {self.data.min():.4g}  /  {self.data.max():.4g}")
            _row(f"    Mean / Std : {self.data.mean():.4g}  /  {self.data.std():.4g}")
        except (ValueError, TypeError):
            _row("    Statistics : unavailable")

        # ------------------------------------------------------------------ #
        # Regions block
        # ------------------------------------------------------------------ #
        _border()
        n_names = len(region_names)
        _row(f"  REGIONS  ({n_names})")

        if n_names == 0:
            _row("    (no region names stored)")
        else:
            # Column header
            idx_hdr = f"{'#':>4}"
            name_hdr = f"{'Name':<{_NAME_W}}"
            color_hdr = f"{'Color':<{_COLOR_W}}"
            _row(f"  {idx_hdr} {name_hdr} {color_hdr}")
            _border("╟", "─", "╢")

            n_show = min(n_names, _MAX_SHOWN)
            for i in range(n_show):
                idx_str = f"{i + 1:>4}"
                name_str = _trunc(str(region_names[i]), _NAME_W)
                color_str = _trunc(
                    str(region_colors[i]) if i < len(region_colors) else "—", _COLOR_W
                )
                _row(f"  {idx_str} {name_str:<{_NAME_W}} {color_str:<{_COLOR_W}}")

            if n_names > _MAX_SHOWN:
                _border("╟", "─", "╢")
                _row(f"    … {n_names - _MAX_SHOWN} more region(s) not shown")

        # ------------------------------------------------------------------ #
        # Footer
        # ------------------------------------------------------------------ #
        _border("╚", "═", "╝")

    ###########################################################################################
    def show_content(
        self,
        show_private=False,
        show_dunder=False,
        show_methods=True,
        show_properties=True,
        show_attributes=True,
    ):
        """
        Alias for get_info() to display the content of the RegionTimeSeries object.

        Examples
        --------
        >>> rts.show_content()
        """
        cltmisc.show_object_content(
            self,
            show_private=show_private,
            show_dunder=show_dunder,
            show_methods=show_methods,
            show_properties=show_properties,
            show_attributes=show_attributes,
        )
