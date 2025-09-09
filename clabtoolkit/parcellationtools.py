import os
from datetime import datetime
import copy
import h5py

import numpy as np
import pandas as pd
import nibabel as nib
import pyvista as pv
from pathlib import Path


from typing import Union, List, Optional
from scipy.ndimage import gaussian_filter
from skimage import measure

from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    TextColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)

# Importing local modules
from . import misctools as cltmisc
from . import imagetools as cltimg
from . import segmentationtools as cltseg
from . import freesurfertools as cltfree
from . import surfacetools as cltsurf
from . import bidstools as cltbids


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
        parc_file: Union[str, np.uint] = None,
        affine: np.float64 = None,
        parc_id: Optional[str] = None,
        space_id: Optional[str] = "unknown",
    ):
        """
        Initialize Parcellation object from file or array.

        Parameters
        ----------
        parc_file : str or np.ndarray, optional
            Path to parcellation file or numpy array. If string, loads from file
            and attempts to find associated TSV/LUT files. Default is None.

        affine : np.ndarray, optional
            4x4 affine transformation matrix. If None and parc_file is array,
            creates identity matrix. Default is None.

        parc_id : str, optional
            Unique identifier for the parcellation. If None, generated from file name.
            Default is None.

        space_id : str, optional
            Identifier for the space in which the parcellation is defined. Default is "unknown".

        Attributes
        ----------
        data : np.ndarray
            3D parcellation data array.

        affine : np.ndarray
            4x4 affine transformation matrix.

        index : list
            List of region codes present in parcellation.

        name : list
            List of region names corresponding to codes.

        color : list
            List of colors (hex format) for each region.

        Examples
        --------
        >>> # Load from file
        >>> parc = Parcellation('parcellation.nii.gz')
        >>>
        >>> # Create from array
        >>> parc = Parcellation(label_array, affine=img.affine)
        """

        if parc_file is not None:
            if isinstance(parc_file, str):
                if os.path.exists(parc_file):
                    self.parc_file = parc_file
                    self.get_parcellation_id()
                    self.get_space_id(space_id=space_id)
                    temp_iparc = nib.load(parc_file)
                    affine = temp_iparc.affine
                    self.data = temp_iparc.get_fdata()
                    self.data.astype(np.int32)

                    self.affine = affine
                    self.dtype = temp_iparc.get_data_dtype()

                    if parc_file.endswith(".nii.gz"):
                        tsv_file = parc_file.replace(".nii.gz", ".tsv")
                        lut_file = parc_file.replace(".nii.gz", ".lut")

                        if os.path.isfile(tsv_file):
                            self.load_colortable(lut_file=tsv_file, lut_type="tsv")

                        elif not os.path.isfile(tsv_file) and os.path.isfile(lut_file):
                            self.load_colortable(lut_file=lut_file, lut_type="lut")

                    elif parc_file.endswith(".nii"):
                        tsv_file = parc_file.replace(".nii", ".tsv")
                        lut_file = parc_file.replace(".nii", ".lut")

                        if os.path.isfile(tsv_file):
                            self.load_colortable(lut_file=tsv_file, lut_type="tsv")

                        elif not os.path.isfile(tsv_file) and os.path.isfile(lut_file):
                            self.load_colortable(lut_file=lut_file, lut_type="lut")

                    # Adding index, name and color attributes
                    if not hasattr(self, "index"):
                        self.index = np.unique(self.data)
                        self.index = self.index[self.index != 0].tolist()
                        self.index = [int(x) for x in self.index]

                    if not hasattr(self, "name"):
                        # create a list with the names of the regions. I would like a format for the names similar to this supra-side-000001
                        self.name = cltmisc.create_names_from_indices(self.index)

                    if not hasattr(self, "color"):
                        self.color = cltmisc.create_random_colors(
                            len(self.index), output_format="hex"
                        )

                else:
                    raise ValueError("The parcellation file does not exist")

            # If the parcellation is a numpy array
            elif isinstance(parc_file, np.ndarray):
                self.parc_file = "numpy_array"

                if parc_id is None:
                    self.id = "numpy_array"

                self.space = space_id
                self.data = parc_file

                # Creating a new affine matrix if the affine matrix is None
                if affine is None:
                    affine = np.eye(4)

                    center = np.array(self.data.shape) // 2
                    affine[:3, 3] = -center

                self.affine = affine

                # Create a list with all the values different from 0
                st_codes = np.unique(self.data)
                st_codes = st_codes[st_codes != 0]

                self.index = st_codes.tolist()
                self.index = [int(x) for x in self.index]
                self.name = cltmisc.create_names_from_indices(self.index)

                if len(self.index) > 0:
                    # Generate the colors
                    self.color = cltmisc.create_random_colors(
                        len(self.index), output_format="hex"
                    )
                else:
                    self.color = []

            # Adjust values to the ones present in the parcellation

            # Force index to be int
            if hasattr(self, "index"):
                self.index = [int(x) for x in self.index]

            if (
                hasattr(self, "index")
                and hasattr(self, "name")
                and hasattr(self, "color")
            ):
                self.adjust_values()

            # Dimensions of the parcellation
            self.dim = self.data.shape

            # Get voxel size
            self.voxel_size = cltimg.get_voxel_volume(self.affine)

            # Detect minimum and maximum labels
            self.parc_range()

    #####################################################################################################
    def get_space_id(self, space_id: Optional[str] = "unknown"):
        """
        Set the space identifier for the parcellation.

        Parameters
        ----------
        space_id : str, optional
            Identifier for the space in which the parcellation is defined. Default is "unknown".

        Raises
        ------
        ValueError
            If the parcellation file is not set.

        Returns
        -------
        space_id : str
            The space identifier for the parcellation, formatted as 'space-<space_id>'.

        Notes
        -----
        This method sets the `space` attribute of the Parcellation object.
        It is useful for tracking the spatial context of the parcellation data.

        Examples
        --------
        >>> parc = Parcellation('sub-01_ses-01_acq-mprage_space-t1_atlas-xxx_seg-yyy_scale-1_desc-test.nii.gz')
        >>> space_id = parc.get_space_id()
        >>> print(space_id)
        't1'

        >>> parc = Parcellation('custom_parcellation.nii.gz')
        >>> space_id = parc.get_space_id(space_id='custom_space')
        >>> print(space_id)
        'custom_space'
        >>> parc = Parcellation()
        >>> space_id = parc.get_space_id()
        'unknown'

        """

        # Check if the parcellation file is set
        if not hasattr(self, "parc_file"):
            raise ValueError(
                "The parcellation file is not set. Please load a parcellation file first."
            )
            # Get the base name of the parcellation file
        parc_file_name = os.path.basename(self.parc_file)

        # Check if the parcellation file name follows BIDS naming conventions
        if cltbids.is_bids_filename(parc_file_name):

            # Extract entities from the parcellation file name
            name_ent_dict = cltbids.str2entity(parc_file_name)
            ent_names_list = list(name_ent_dict.keys())

            # Create space_id based on the entities present in the parcellation file name
            if "space" in ent_names_list:
                if space_id != "unknown":

                    # If space_id is provided, use it
                    space_id = name_ent_dict["space"]

        # Assign the space_id to the object
        self.space = space_id

        return space_id

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
    def prepare_for_tracking(self):
        """
        Prepare parcellation for fiber tracking by merging cortical white matter labels
        to their corresponding cortical gray matter values.

        Converts white matter labels (>=3000) to corresponding gray matter labels
        by subtracting 3000, and removes other structures labels (>=5000).

        Examples
        --------
        >>> parc.prepare_for_tracking()
        >>> print(f"Max label after prep: {parc.data.max()}")
        """

        # Unique of non-zero values
        sts_vals = np.unique(self.data)

        # sts_vals as integers
        sts_vals = sts_vals.astype(int)

        # get the values of sts_vals that are bigger or equaal to 5000 and create a list with them
        indexes = [x for x in sts_vals if x >= 5000]

        self.remove_by_code(codes2remove=indexes)

        # Get the labeled wm values
        ind = np.argwhere(self.data >= 3000)

        # Add the wm voxels to the gm label
        self.data[ind[:, 0], ind[:, 1], ind[:, 2]] = (
            self.data[ind[:, 0], ind[:, 1], ind[:, 2]] - 3000
        )

        # Adjust the values
        self.adjust_values()

    ####################################################################################################
    def keep_by_name(self, names2look: Union[list, str], rearrange: bool = False):
        """
        Filter parcellation to keep only regions with specified names.

        Parameters
        ----------
        names2look : str or list
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

        if isinstance(names2look, str):
            names2look = [names2look]

        if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
            # Find the indexes of the names that contain the substring
            indexes = cltmisc.get_indexes_by_substring(
                input_list=self.name, substr=names2look, invert=False, bool_case=False
            )

            if len(indexes) > 0:
                sel_st_codes = [self.index[i] for i in indexes]
                self.keep_by_code(codes2keep=sel_st_codes, rearrange=rearrange)
            else:
                print("The names were not found in the parcellation")

    #####################################################################################################
    def keep_by_code(
        self, codes2keep: Union[list, np.ndarray], rearrange: bool = False
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

        # Convert the codes2keep to a numpy array
        if isinstance(codes2keep, list):
            codes2keep = cltmisc.build_indices(codes2keep)
            codes2keep = np.array(codes2keep)

        # Create a boolean mask where elements are True if they are in the retain list
        mask = np.isin(self.data, codes2keep)

        # Set elements to zero if they are not in the retain list
        self.data[~mask] = 0

        # Remove the elements from retain_list that are not present in the data
        img_tmp_codes = np.unique(self.data)

        # Codes to look is img_tmp_codes without the 0
        codes2keep = img_tmp_codes[img_tmp_codes != 0]

        if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
            sts = np.unique(self.data)
            sts = sts[sts != 0]
            temp_index = np.array(self.index)
            mask = np.isin(temp_index, sts)
            self.index = temp_index[mask].tolist()
            self.name = np.array(self.name)[mask].tolist()
            self.color = np.array(self.color)[mask].tolist()

        # If rearrange is True, the parcellation will be rearranged starting from 1
        if rearrange:
            self.rearrange_parc()

        # Detect minimum and maximum labels
        self.parc_range()

    #####################################################################################################
    def remove_by_code(
        self, codes2remove: Union[list, np.ndarray], rearrange: bool = False
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

        if isinstance(codes2remove, list):
            codes2remove = cltmisc.build_indices(codes2remove)
            codes2remove = np.array(codes2remove)

        self.data[np.isin(self.data, codes2remove)] = 0

        st_codes = np.unique(self.data)
        st_codes = st_codes[st_codes != 0]

        # If rearrange is True, the parcellation will be rearranged starting from 1
        if rearrange:
            self.keep_by_code(codes2keep=st_codes, rearrange=True)
        else:
            self.keep_by_code(codes2keep=st_codes, rearrange=False)

        # Detect minimum and maximum labels
        self.parc_range()

    #####################################################################################################
    def remove_by_name(self, names2remove: Union[list, str], rearrange: bool = False):
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

        if isinstance(names2remove, str):
            names2remove = [names2remove]

        if hasattr(self, "name") and hasattr(self, "index") and hasattr(self, "color"):

            indexes = cltmisc.get_indexes_by_substring(
                input_list=self.name, substr=names2remove, invert=True, bool_case=False
            )

            if len(indexes) > 0:
                sel_st_codes = [self.index[i] for i in indexes]
                self.keep_by_code(codes2keep=sel_st_codes, rearrange=rearrange)

            else:
                print("The names were not found in the parcellation")
        else:
            print(
                "The parcellation does not contain the attributes name, index and color"
            )

        # Detect minimum and maximum labels
        self.parc_range()

    #####################################################################################################
    def apply_mask(
        self,
        image_mask,
        codes2mask: Union[list, np.ndarray] = None,
        mask_type: str = "upright",
        fill: bool = False,
    ):
        """
        Apply spatial mask to restrict parcellation to specific regions.

        Parameters
        ----------
        image_mask : np.ndarray, Parcellation, or str
            3D mask array, parcellation object, or path to mask file.

        codes2mask : list or np.ndarray, optional
            Specific region codes to mask. If None, masks all regions. Default is None.

        mask_type : str, optional
            'upright' to keep masked regions, 'inverted' to remove them. Default is 'upright'.

        fill : bool, optional
            Whether to grow regions to fill mask using region growing. Default is False.

        Examples
        --------
        >>> # Apply cortical mask
        >>> parc.apply_mask(cortex_mask, mask_type='upright')
        >>>
        >>> # Mask specific regions with filling
        >>> parc.apply_mask(roi_mask, codes2mask=[1, 2, 3], fill=True)
        """

        if isinstance(image_mask, str):
            if os.path.exists(image_mask):
                temp_mask = nib.load(image_mask)
                mask_data = temp_mask.get_fdata()
            else:
                raise ValueError("The mask file does not exist")

        elif isinstance(image_mask, np.ndarray):
            mask_data = image_mask

        elif isinstance(image_mask, Parcellation):
            mask_data = image_mask.data

        mask_type.lower()
        if mask_type not in ["upright", "inverted"]:
            raise ValueError("The mask_type must be 'upright' or 'inverted'")

        if codes2mask is None:
            codes2mask = np.unique(self.data)
            codes2mask = codes2mask[codes2mask != 0]

        if isinstance(codes2mask, list):
            codes2mask = cltmisc.build_indices(codes2mask)
            codes2mask = np.array(codes2mask)

        if mask_type == "inverted":
            self.data[np.isin(mask_data, codes2mask) == True] = 0
            bool_mask = np.isin(mask_data, codes2mask) == False

        else:
            self.data[np.isin(mask_data, codes2mask) == False] = 0
            bool_mask = np.isin(mask_data, codes2mask) == True

        if fill:

            # Refilling the unlabeled voxels according to a supplied mask
            self.data = cltimg.region_growing(self.data, bool_mask)

        if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
            self.adjust_values()

        # Detect minimum and maximum labels
        self.parc_range()

    def mask_image(
        self,
        image_2mask: Union[str, list, np.ndarray],
        masked_image: Union[str, list, np.ndarray] = None,
        codes2mask: Union[str, list, np.ndarray] = None,
        mask_type: str = "upright",
    ):
        """
        Mask external images using parcellation as binary mask.

        Parameters
        ----------
        image_2mask : str, list, or np.ndarray
            Image(s) to mask using parcellation.

        masked_image : str or list, optional
            Output path(s) for masked images. Default is None.

        codes2mask : list or np.ndarray, optional
            Region codes to use for masking. Default is None (all regions).

        mask_type : str, optional
            'upright' uses specified codes, 'inverted' uses other codes. Default is 'upright'.

        Examples
        --------
        >>> # Mask T1 image with parcellation
        >>> parc.mask_image('T1w.nii.gz', 'T1w_masked.nii.gz')
        >>>
        >>> # Mask with specific regions
        >>> parc.mask_image('fmri.nii.gz', codes2mask=[1, 2, 3])
        """

        if isinstance(image_2mask, str):
            image_2mask = [image_2mask]

        if isinstance(masked_image, str):
            masked_image = [masked_image]

        if isinstance(masked_image, list) and isinstance(image_2mask, list):
            if len(masked_image) != len(image_2mask):
                raise ValueError(
                    "The number of images to mask must be equal to the number of images to be saved"
                )

        if codes2mask is None:
            # Get the indexes of all values different from zero
            codes2mask = np.unique(self.data)
            codes2mask = codes2mask[codes2mask != 0]

        if isinstance(codes2mask, list):
            codes2mask = cltmisc.build_indices(codes2mask)
            codes2mask = np.array(codes2mask)

        if mask_type == "inverted":
            ind2rem = np.isin(self.data, codes2mask) == True

        else:
            ind2rem = np.isin(self.data, codes2mask) == False

        if isinstance(image_2mask, list):
            if isinstance(image_2mask[0], str):
                for cont, img in enumerate(image_2mask):
                    if os.path.exists(img):
                        temp_img = nib.load(img)
                        img_data = temp_img.get_fdata()
                        img_data[ind2rem] = 0

                        # Save the masked image
                        out_img = nib.Nifti1Image(img_data, temp_img.affine)
                        nib.save(out_img, masked_image[cont])

                    else:
                        raise ValueError("The image file does not exist")
            else:
                raise ValueError(
                    "The image_2mask must be a list of strings containing the paths to the images"
                )

        elif isinstance(image_2mask, np.ndarray):
            img_data = image_2mask
            img_data[ind2rem] = 0

            return img_data

    ######################################################################################################
    def compute_centroids(
        self,
        struct_codes: Union[List[int], np.ndarray] = None,
        struct_names: Union[List[str], str] = None,
        gaussian_smooth: bool = True,
        sigma: float = 1.0,
        closing_iterations: int = 2,
        centroid_table: str = None,
    ) -> pd.DataFrame:
        """
        Compute region centroids, voxel counts, and volumes.

        Parameters
        ----------
        struct_codes : list or np.ndarray, optional
            Specific region codes to include. Default is None (all regions).

        struct_names : list or str, optional
            Specific region names to include. Default is None.

        gaussian_smooth : bool, optional
            Whether to apply Gaussian smoothing to the volume before centroid calculation. Default is True.

        sigma : float, optional
            Standard deviation for Gaussian smoothing. Default is 1.0.

        closing_iterations : int, optional
            Number of morphological closing iterations before centroid extraction. Default is 2.

        centroid_table : str, optional
            Path to save results as TSV file. Default is None.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: index, name, color, X, Y, Z (mm), nvoxels, volume.

        Raises
        ------
        ValueError
            If both struct_codes and struct_names are specified.

        Examples
        --------
        >>> # Compute all centroids
        >>> centroids_df = parc.compute_centroids()
        >>>
        >>> # Specific regions with file output
        >>> df = parc.compute_centroids(
        ...     struct_codes=[1, 2, 3],
        ...     centroid_table='centroids.tsv'
        ... )
        >>> # Specific regions by name
        >>> df = parc.compute_centroids(
        ...     struct_names=['hippocampus', 'amygdala'],
        ...     centroid_table='centroids.tsv'
        ... )
        """

        # Check if include_by_code and include_by_name are different from None at the same time
        if struct_codes is not None and struct_names is not None:
            raise ValueError(
                "You cannot specify both include_by_code and include_by_name at the same time. Please choose one of them."
            )

        temp_parc = copy.deepcopy(self)

        # Apply inclusion if specified
        if struct_codes is not None:
            temp_parc.keep_by_code(codes2keep=struct_codes)

        if struct_names is not None:
            temp_parc.keep_by_name(names2look=struct_names)

        # Get unique region values
        unique_regions = np.array(temp_parc.index)

        # Lists to store results
        codes = []
        names = []
        colors = []
        x_coords = []
        y_coords = []
        z_coords = []
        num_voxels = []
        volumes = []

        # Get voxel size
        voxel_volume = cltimg.get_voxel_volume(temp_parc.affine)

        # Fixed loop - iterate over regions and find their index
        for region_label in unique_regions:
            # Find the index of this region in parc.index
            region_idx = np.where(np.array(temp_parc.index) == region_label)[0]
            if len(region_idx) == 0:
                continue
            region_idx = region_idx[0]  # Get the first (should be only) match

            # Extract centroid and voxel count
            centroid, voxel_count = cltimg.extract_centroid_from_volume(
                temp_parc.data == region_label,
                gaussian_smooth=gaussian_smooth,
                sigma=sigma,
                closing_iterations=closing_iterations,
            )

            centroid_x, centroid_y, centroid_z = centroid[0], centroid[1], centroid[2]

            # Calculate total volume
            total_volume = voxel_count * voxel_volume

            # Store results
            codes.append(int(region_label))
            names.append(temp_parc.name[region_idx])
            colors.append(temp_parc.color[region_idx])
            x_coords.append(centroid_x)
            y_coords.append(centroid_y)
            z_coords.append(centroid_z)
            num_voxels.append(voxel_count)
            volumes.append(total_volume)

        # Convert coordinates to mm
        coords_vox = np.stack(
            (np.array(x_coords), np.array(y_coords), np.array(z_coords)), axis=-1
        )
        coords_mm = cltimg.vox2mm(coords_vox, self.affine)

        # Add centroid coordinates in mm
        self.centroids = coords_mm.astype(float)

        x_coords_mm = coords_mm[:, 0]
        y_coords_mm = coords_mm[:, 1]
        z_coords_mm = coords_mm[:, 2]

        # Convert to list
        x_coords_mm = x_coords_mm.tolist()
        y_coords_mm = y_coords_mm.tolist()
        z_coords_mm = z_coords_mm.tolist()

        # Create DataFrame
        df = pd.DataFrame(
            {
                "index": codes,
                "name": names,
                "color": colors,
                "Xvox": x_coords,
                "Yvox": y_coords,
                "Zvox": z_coords,
                "Xmm": x_coords_mm,
                "Ymm": y_coords_mm,
                "Zmm": z_coords_mm,
                "nvoxels": num_voxels,
                "volume": volumes,
            }
        )

        # Save to TSV file if path is provided
        if centroid_table is not None:
            try:
                # Check if the directory exists
                directory = os.path.dirname(centroid_table)
                if directory and not os.path.exists(directory):
                    print(
                        f"Warning: Directory '{directory}' does not exist. Cannot save file."
                    )
                else:
                    # Save as TSV file
                    df.to_csv(centroid_table, sep="\t", index=False)
                    print(f"Centroid table saved to: {centroid_table}")
            except Exception as e:
                print(f"Error saving centroid table: {e}")

        return df

    ######################################################################################################
    def get_regionwise_timeseries(
        self,
        time_series_data: Union[str, np.ndarray],
        vols_to_delete: Union[List[int], np.ndarray] = None,
        method: str = "nilearn",
        metric: str = "mean",
        struct_codes: Union[List[int], np.ndarray] = None,
        struct_names: Union[List[str], str] = None,
    ) -> pd.DataFrame:
        """
        Compute region-wise time series.

        Parameters
        ----------
        time_series_data : str or np.ndarray
            Path to time series file or numpy array with shape (dimx X dimy X dimZ x Timepoints).

        struct_codes : list or np.ndarray, optional
            Specific region codes to include. Default is None (all regions).

        struct_names : list or str, optional
            Specific region names to include. Default is None.

        ouput_h5file : str, optional
            Path to save results as HDF5 file. Default is None.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: index, name, color, X, Y, Z (mm), nvoxels, volume.

        Raises
        ------
        ValueError
            If both struct_codes and struct_names are specified.

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
        ...     struct_codes=[1, 2, 3])

        """

        # Check if include_by_code and include_by_name are different from None at the same time
        if struct_codes is not None and struct_names is not None:
            raise ValueError(
                "You cannot specify both include_by_code and include_by_name at the same time. Please choose one of them."
            )

        temp_parc = copy.deepcopy(self)

        # Apply inclusion if specified
        if struct_codes is not None:
            temp_parc.keep_by_code(codes2keep=struct_codes)

        if struct_names is not None:
            temp_parc.keep_by_name(names2look=struct_names)

        # Delete volumes if specified
        if vols_to_delete is not None:

            # Check if the time_series_data is a string
            if isinstance(time_series_data, str):
                # Generating a temporary file to save the 4D data
                tmp_image = cltmisc.create_temporary_filename(
                    prefix="temp_timeseries",
                    extension=".nii.gz",
                    tmp_dir="/tmp",
                )

                # Deleting the volumes from the 4D image
                del_img = cltimg.delete_volumes_from_4D_images(
                    in_image=time_series_data,
                    out_image=tmp_image,
                    vols_to_delete=vols_to_delete,
                )
                time_series_data_tmp = tmp_image

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
                except:
                    raise ImportError(
                        "nilearn is not installed. Please install it to use this method."
                    )

                # Generating a temporary parcellation file
                tmp_parc_image = cltmisc.create_temporary_filename(
                    prefix="temp_parcellation", extension=".nii.gz", tmp_dir="/tmp"
                )

                temp_parc.save_parcellation(out_file=tmp_parc_image, save_lut=True)
                tmp_parc_image_lut = tmp_parc_image.replace(".nii.gz", ".lut")
                tmp_parc_image_nilearnlut = tmp_parc_image.replace(
                    ".nii.gz", "_nilearn.lut"
                )

                # Converting the parcellation to a nilearn LUT format
                temp_parc.lut_to_nilearnlut(
                    tmp_parc_image_lut, tmp_parc_image_nilearnlut, overwrite=True
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
                os.remove(tmp_parc_image_lut)
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

            elif isinstance(time_series_data, np.ndarray):
                time_series = time_series_data

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
        temp_parc.seriesextractionmethod = method
        temp_parc.timeseries = region_time_series

        return region_time_series

    ######################################################################################################
    def surface_extraction(
        self,
        struct_codes: Union[List[int], np.ndarray] = None,
        struct_names: Union[List[str], str] = None,
        gaussian_smooth: bool = True,
        smooth_iterations: int = 10,
        fill_holes: bool = True,
        sigma: float = 1.0,
        closing_iterations: int = 1,
        out_filename: str = None,
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
        struct_codes : list or np.ndarray, optional
            Region codes to extract surfaces for. Default is None (all regions).

        struct_names : list or str, optional
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

        out_format : str, optional
            Output format: 'freesurfer', 'vtk', 'ply', 'stl', 'obj'. Default is 'freesurfer'.

        save_annotation : bool, optional
            Whether to save annotation file with surface. Default is True.

        overwrite : bool, optional
            Whether to overwrite existing files. Default is False.

        Returns
        -------
        Surface
            Merged surface object containing all extracted regions.

        Raises
        ------
        ValueError
            If both struct_codes and struct_names are specified.
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
        ...     struct_codes=[1, 2, 3],
        ...     smooth_iterations=20,
        ...     out_filename='regions.surf'
        ... )
        """

        # Check if include_by_code and include_by_name are different from None at the same time
        if struct_codes is not None and struct_names is not None:
            raise ValueError(
                "You cannot specify both include_by_code and include_by_name at the same time. Please choose one of them."
            )

        temp_parc = copy.deepcopy(self)

        # Apply inclusion if specified
        if struct_codes is not None:
            temp_parc.keep_by_code(codes2keep=struct_codes)

        if struct_names is not None:
            temp_parc.keep_by_name(names2look=struct_names)

        # Get unique region values
        unique_regions = np.array(temp_parc.index)

        color_table = cltfree.colors2colortable(temp_parc.color)
        color_table, log, corresp_dict = cltfree.resolve_colortable_duplicates(
            color_table
        )

        table_dict = {
            "struct_names": temp_parc.name,
            "color_table": color_table,
            "lookup_table": None,
        }
        color_tables = {
            "surface": table_dict,
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

                surf_temp = cltsurf.Surface()
                surf_temp.mesh = copy.deepcopy(mesh)
                # surf_temp.load_from_mesh(mesh, hemi="lh")
                surfaces_list.append(surf_temp)
                # Update progress to show completion of this region

        # surf_orig.merge_surfaces(surfaces_list)
        merged_surf = cltsurf.merge_surfaces_list(surfaces_list)
        merged_surf.colortables = color_tables

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
                    map_name="surface",
                    save_annotation=annot_filename,
                    overwrite=overwrite,
                )
            else:
                merged_surf.save_surface(
                    filename=out_filename,
                    format=out_format,
                    map_name="surface",
                    overwrite=overwrite,
                )

        return merged_surf

    ######################################################################################################
    def adjust_values(self):
        """
        Synchronize index, name, and color attributes with data contents.

        Removes entries for codes not present in data and updates
        min/max label range.

        Examples
        --------
        >>> parc.adjust_values()
        >>> print(f"Regions in data: {len(parc.index)}")
        """

        st_codes = np.unique(self.data)
        unique_codes = st_codes[st_codes != 0]

        mask = np.isin(self.index, unique_codes)
        indexes = np.where(mask)[0]

        temp_index = np.array(self.index)
        index_new = temp_index[mask]

        if hasattr(self, "index"):
            self.index = [int(x) for x in index_new.tolist()]

        # If name is an attribute of self
        if hasattr(self, "name"):
            self.name = [self.name[i] for i in indexes]

        # If color is an attribute of self
        if hasattr(self, "color"):
            self.color = [self.color[i] for i in indexes]

        self.parc_range()

    ######################################################################################################
    def group_by_code(
        self,
        codes2group: Union[list, np.ndarray],
        new_codes: Union[list, np.ndarray] = None,
        new_names: Union[list, str] = None,
        new_colors: Union[list, np.ndarray] = None,
    ):
        """
        Group regions by combining specified codes into new regions.

        Parameters
        ----------
        codes2group : list or np.ndarray
            List of codes or list of code lists to group.

        new_codes : list or np.ndarray, optional
            New codes for groups. If None, uses sequential numbering. Default is None.

        new_names : list or str, optional
            New names for groups. Default is None.

        new_colors : list or np.ndarray, optional
            New colors for groups. Default is None.

        Examples
        --------
        >>> # Group bilateral regions
        >>> parc.group_by_code(
        ...     [[1, 2], [3, 4]],  # Left/right pairs
        ...     new_names=['region1', 'region2']
        ... )
        """

        # if all the  elements in codes2group are numeric then convert codes2group to a numpy array
        if all(isinstance(x, (int, np.integer, float)) for x in codes2group):
            codes2group = np.array(codes2group)

        # Detect thecodes2group is a list of list
        if isinstance(codes2group, list):
            if isinstance(codes2group[0], list):
                n_groups = len(codes2group)

            elif isinstance(codes2group[0], (str, np.integer, int, tuple)):
                codes2group = [codes2group]
                n_groups = 1

        elif isinstance(codes2group, np.ndarray):
            codes2group = [codes2group.tolist()]
            n_groups = 1

        for i, v in enumerate(codes2group):
            if isinstance(v, list):
                codes2group[i] = cltmisc.build_indices(v)

        # Convert the new_codes to a numpy array
        if new_codes is not None:
            if isinstance(new_codes, list):
                new_codes = cltmisc.build_indices(new_codes)
                new_codes = np.array(new_codes)
            elif isinstance(new_codes, (str, np.integer, int)):
                new_codes = np.array([new_codes])

        else:
            new_codes = np.arange(1, n_groups + 1)

        if len(new_codes) != n_groups:
            raise ValueError(
                "The number of new codes must be equal to the number of groups that will be created"
            )

        # Convert the new_names to a list
        if new_names is not None:
            if isinstance(new_names, str):
                new_names = [new_names]

            if len(new_names) != n_groups:
                raise ValueError(
                    "The number of new names must be equal to the number of groups that will be created"
                )

        # Convert the new_colors to a numpy array
        if new_colors is not None:
            if isinstance(new_colors, list):

                if isinstance(new_colors[0], str):
                    new_colors = cltmisc.multi_hex2rgb(new_colors)

                elif isinstance(new_colors[0], np.ndarray):
                    new_colors = np.array(new_colors)

                else:
                    raise ValueError(
                        "If new_colors is a list, it must be a list of hexadecimal colors or a list of rgb colors"
                    )

            elif isinstance(new_colors, np.ndarray):
                pass

            else:
                raise ValueError(
                    "The new_colors must be a list of colors or a numpy array"
                )

            new_colors = cltmisc.readjust_colors(new_colors)

            if new_colors.shape[0] != n_groups:
                raise ValueError(
                    "The number of new colors must be equal to the number of groups that will be created"
                )

        # Creating the grouped parcellation
        out_atlas = np.zeros_like(self.data, dtype="int16")
        for i in range(n_groups):
            code2look = np.array(codes2group[i])

            if new_codes is not None:
                out_atlas[np.isin(self.data, code2look) == True] = new_codes[i]
            else:
                out_atlas[np.isin(self.data, code2look) == True] = i + 1

        self.data = out_atlas

        if new_codes is not None:
            self.index = new_codes.tolist()

        if new_names is not None:
            self.name = new_names
        else:
            # If new_names is not provided, the names will be created
            self.name = ["group_{}".format(i) for i in new_codes]

        if new_colors is not None:
            self.color = new_colors
        else:
            # If new_colors is not provided, the colors will be created
            self.color = cltmisc.create_random_colors(n_groups)

        # Detect minimum and maximum labels
        self.parc_range()

    ######################################################################################################
    def group_by_name(
        self,
        names2group: Union[List[list], List[str]],
        new_codes: Union[list, np.ndarray] = None,
        new_names: Union[list, str] = None,
        new_colors: Union[list, np.ndarray] = None,
    ):
        """
        Group regions by combining regions with specified name patterns.

        Parameters
        ----------
        names2group : list
            List of name patterns or list of pattern lists to group.

        new_codes : list or np.ndarray, optional
            New codes for groups. Default is None.

        new_names : list or str, optional
            New names for groups. Default is None.

        new_colors : list or np.ndarray, optional
            New colors for groups. Default is None.

        Examples
        --------
        >>> # Group by anatomical regions
        >>> parc.group_by_name(
        ...     [['frontal'], ['parietal'], ['temporal']],
        ...     new_names=['frontal_lobe', 'parietal_lobe', 'temporal_lobe']
        ... )
        """

        # Detect thecodes2group is a list of list
        if isinstance(names2group, list):
            if isinstance(names2group[0], list):
                n_groups = len(names2group)

            elif isinstance(codes2group[0], (str)):
                codes2group = [codes2group]
                n_groups = 1

        for i, v in enumerate(codes2group):
            if isinstance(v, list):
                codes2group[i] = cltmisc.build_indices(v)

        # Convert the new_codes to a numpy array
        if new_codes is not None:
            if isinstance(new_codes, list):
                new_codes = cltmisc.build_indices(new_codes)
                new_codes = np.array(new_codes)
            elif isinstance(new_codes, (str, np.integer, int)):
                new_codes = np.array([new_codes])

        else:
            new_codes = np.arange(1, n_groups + 1)

        if len(new_codes) != n_groups:
            raise ValueError(
                "The number of new codes must be equal to the number of groups that will be created"
            )

        # Convert the new_names to a list
        if new_names is not None:
            if isinstance(new_names, str):
                new_names = [new_names]

            if len(new_names) != n_groups:
                raise ValueError(
                    "The number of new names must be equal to the number of groups that will be created"
                )

        # Convert the new_colors to a numpy array
        if new_colors is not None:
            if isinstance(new_colors, list):

                if isinstance(new_colors[0], str):
                    new_colors = cltmisc.multi_hex2rgb(new_colors)

                elif isinstance(new_colors[0], np.ndarray):
                    new_colors = np.array(new_colors)

                else:
                    raise ValueError(
                        "If new_colors is a list, it must be a list of hexadecimal colors or a list of rgb colors"
                    )

            elif isinstance(new_colors, np.ndarray):
                pass

            else:
                raise ValueError(
                    "The new_colors must be a list of colors or a numpy array"
                )

            new_colors = cltmisc.readjust_colors(new_colors)

            if new_colors.shape[0] != n_groups:
                raise ValueError(
                    "The number of new colors must be equal to the number of groups that will be created"
                )

        # Creating the grouped parcellation
        out_atlas = np.zeros_like(self.data, dtype="int16")

        for i in range(n_groups):
            indexes = cltmisc.get_indexes_by_substring(
                input_list=self.name, substr=names2group[i]
            )
            code2look = np.array(indexes) + 1

            if new_codes is not None:
                out_atlas[np.isin(self.data, code2look) == True] = new_codes[i]
            else:
                out_atlas[np.isin(self.data, code2look) == True] = i + 1

        self.data = out_atlas

        if new_codes is not None:
            self.index = new_codes.tolist()

        if new_names is not None:
            self.name = new_names
        else:
            # If new_names is not provided, the names will be created
            self.name = ["group_{}".format(i) for i in new_codes]

        if new_colors is not None:
            self.color = new_colors
        else:
            # If new_colors is not provided, the colors will be created
            self.color = cltmisc.create_random_colors(n_groups)

        # Detect minimum and maximum labels
        self.parc_range()

    ######################################################################################################
    def rearrange_parc(self, offset: int = 0):
        """
        Rearrange parcellation labels to consecutive integers.

        Parameters
        ----------
        offset : int, optional
            Starting value for rearranged labels. Default is 0 (starts from 1).

        Examples
        --------
        >>> # Rearrange to 1, 2, 3, ...
        >>> parc.rearrange_parc()
        >>>
        >>> # Start from 100
        >>> parc.rearrange_parc(offset=99)
        """

        st_codes = np.unique(self.data)
        st_codes = st_codes[st_codes != 0]

        # Parcellation with values starting from 1 or starting from the offset
        new_parc = np.zeros_like(self.data, dtype="int16")
        if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
            if len(self.index) > 0:
                for i, code in enumerate(self.index):
                    new_parc[self.data == code] = i + 1 + offset

            else:
                for i, code in enumerate(st_codes):
                    new_parc[self.data == code] = i + 1 + offset

            self.data = new_parc
        else:

            for i, code in enumerate(st_codes):
                new_parc[self.data == code] = i + 1 + offset
            self.data = new_parc

        if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
            temp_index = np.unique(self.data)
            temp_index = temp_index[temp_index != 0]
            self.index = temp_index.tolist()

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

        if isinstance(parc2add, Parcellation):
            parc2add = [parc2add]

        if isinstance(parc2add, list):
            if len(parc2add) > 0:
                for parc in parc2add:
                    tmp_parc_obj = copy.deepcopy(parc)
                    if isinstance(parc, Parcellation):
                        ind = np.where(tmp_parc_obj.data != 0)
                        if append:
                            tmp_parc_obj.data[ind] = (
                                tmp_parc_obj.data[ind] + self.maxlab
                            )

                        if (
                            hasattr(parc, "index")
                            and hasattr(parc, "name")
                            and hasattr(parc, "color")
                        ):
                            if (
                                hasattr(self, "index")
                                and hasattr(self, "name")
                                and hasattr(self, "color")
                            ):

                                if append:
                                    # Adjust the values of the index
                                    tmp_parc_obj.index = [
                                        int(x + self.maxlab) for x in tmp_parc_obj.index
                                    ]

                                if isinstance(tmp_parc_obj.index, list) and isinstance(
                                    self.index, list
                                ):
                                    self.index = self.index + tmp_parc_obj.index

                                elif isinstance(
                                    tmp_parc_obj.index, np.ndarray
                                ) and isinstance(self.index, np.ndarray):
                                    self.index = np.concatenate(
                                        (self.index, tmp_parc_obj.index), axis=0
                                    ).tolist()

                                elif isinstance(
                                    tmp_parc_obj.index, list
                                ) and isinstance(self.index, np.ndarray):
                                    self.index = (
                                        tmp_parc_obj.index + self.index.tolist()
                                    )

                                elif isinstance(
                                    tmp_parc_obj.index, np.ndarray
                                ) and isinstance(self.index, list):
                                    self.index = (
                                        self.index + tmp_parc_obj.index.tolist()
                                    )

                                self.name = self.name + tmp_parc_obj.name

                                if isinstance(tmp_parc_obj.color, list) and isinstance(
                                    self.color, list
                                ):
                                    self.color = self.color + tmp_parc_obj.color

                                elif isinstance(
                                    tmp_parc_obj.color, np.ndarray
                                ) and isinstance(self.color, np.ndarray):
                                    self.color = np.concatenate(
                                        (self.color, tmp_parc_obj.color), axis=0
                                    )

                                elif isinstance(
                                    tmp_parc_obj.color, list
                                ) and isinstance(self.color, np.ndarray):
                                    temp_color = cltmisc.readjust_colors(self.color)
                                    temp_color = cltmisc.multi_rgb2hex(temp_color)

                                    self.color = temp_color + tmp_parc_obj.color
                                elif isinstance(
                                    tmp_parc_obj.color, np.ndarray
                                ) and isinstance(self.color, list):
                                    temp_color = cltmisc.readjust_colors(
                                        tmp_parc_obj.color
                                    )
                                    temp_color = cltmisc.multi_rgb2hex(temp_color)

                                    self.color = self.color + temp_color

                            # If the parcellation self.data is all zeros
                            elif np.sum(self.data) == 0:
                                self.index = tmp_parc_obj.index
                                self.name = tmp_parc_obj.name
                                self.color = tmp_parc_obj.color

                        # Concatenating the parcellation data
                        self.data[ind] = tmp_parc_obj.data[ind]

            else:
                raise ValueError("The list is empty")

        if hasattr(self, "color"):
            self.color = cltmisc.harmonize_colors(self.color)

        # Detect minimum and maximum labels
        self.parc_range()

    ######################################################################################################
    def save_parcellation(
        self,
        out_file: str,
        affine: np.float64 = None,
        headerlines: Union[list, str] = None,
        save_lut: bool = False,
        save_tsv: bool = False,
    ):
        """
        Save parcellation to NIfTI file with optional lookup tables.

        Parameters
        ----------
        out_file : str
            Output file path.

        affine : np.ndarray, optional
            Affine matrix. If None, uses object's affine. Default is None.

        headerlines : list or str, optional
            Header lines for LUT file. Default is None.

        save_lut : bool, optional
            Whether to save FreeSurfer LUT file. Default is False.

        save_tsv : bool, optional
            Whether to save TSV lookup table. Default is False.

        Examples
        --------
        >>> # Save with lookup tables
        >>> parc.save_parcellation('output.nii.gz', save_lut=True, save_tsv=True)
        """

        if affine is None:
            affine = self.affine

        if headerlines is not None:
            if isinstance(headerlines, str):
                headerlines = [headerlines]

        self.data.astype(np.int32)
        out_atlas = nib.Nifti1Image(self.data, affine)
        nib.save(out_atlas, out_file)

        if save_lut:
            if (
                hasattr(self, "index")
                and hasattr(self, "name")
                and hasattr(self, "color")
            ):
                self.export_colortable(
                    out_file=out_file.replace(".nii.gz", ".lut"),
                    headerlines=headerlines,
                )
            else:
                print(
                    "Warning: The parcellation does not contain a color table. The lut file will not be saved"
                )

        if save_tsv:
            if (
                hasattr(self, "index")
                and hasattr(self, "name")
                and hasattr(self, "color")
            ):
                self.export_colortable(
                    out_file=out_file.replace(".nii.gz", ".tsv"), lut_type="tsv"
                )
            else:
                print(
                    "Warning: The parcellation does not contain a color table. The tsv file will not be saved"
                )

    ######################################################################################################
    def load_colortable(self, lut_file: Union[str, dict] = None, lut_type: str = "lut"):
        """
        Load lookup table to associate codes with names and colors.

        Parameters
        ----------
        lut_file : str or dict, optional
            Path to LUT file or dictionary with index/name/color keys. Default is None.

        lut_type : str, optional
            File format: 'lut' or 'tsv'. Default is 'lut'.

        Examples
        --------
        >>> # Load FreeSurfer LUT
        >>> parc.load_colortable('FreeSurferColorLUT.txt', lut_type='lut')
        >>>
        >>> # Load TSV table
        >>> parc.load_colortable('regions.tsv', lut_type='tsv')
        """

        if lut_file is None:
            # Get the enviroment variable of $FREESURFER_HOME
            freesurfer_home = os.getenv("FREESURFER_HOME")
            lut_file = os.path.join(freesurfer_home, "FreeSurferColorLUT.txt")

        if isinstance(lut_file, str):
            if os.path.exists(lut_file):
                self.lut_file = lut_file

                if lut_type == "lut":
                    col_dict = self.read_luttable(in_file=lut_file)

                elif lut_type == "tsv":
                    col_dict = self.read_tsvtable(in_file=lut_file)

                else:
                    raise ValueError("The lut_type must be 'lut' or 'tsv'")

                if "index" in col_dict.keys() and "name" in col_dict.keys():
                    st_codes = col_dict["index"]
                    st_names = col_dict["name"]
                else:
                    raise ValueError(
                        "The dictionary must contain the keys 'index' and 'name'"
                    )

                if "color" in col_dict.keys():
                    st_colors = col_dict["color"]
                else:
                    st_colors = None

                self.index = st_codes
                self.name = st_names
                self.color = st_colors

            else:
                raise ValueError("The lut file does not exist")

        elif isinstance(lut_file, dict):
            self.lut_file = None

            if "index" not in lut_file.keys() or "name" not in lut_file.keys():
                raise ValueError(
                    "The dictionary must contain the keys 'index' and 'name'"
                )

            self.index = lut_file["index"]
            self.name = lut_file["name"]

            if "color" not in lut_file.keys():
                self.color = None
            else:
                self.color = lut_file["color"]

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

    ######################################################################################################
    def export_colortable(
        self,
        out_file: str,
        lut_type: str = "lut",
        headerlines: Union[list, str] = None,
        force: bool = True,
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

        force : bool, optional
            Whether to overwrite existing files. Default is True.

        Examples
        --------
        >>> # Export FreeSurfer LUT
        >>> parc.export_colortable('regions.lut', lut_type='lut')
        >>>
        >>> # Export TSV
        >>> parc.export_colortable('regions.tsv', lut_type='tsv')
        """

        if headerlines is not None:
            if isinstance(headerlines, str):
                headerlines = [headerlines]

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

        if lut_type == "lut":

            now = datetime.now()
            date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

            if headerlines is None:
                headerlines = ["# $Id: {} {} \n".format(out_file, date_time)]

                if os.path.isfile(self.parc_file):
                    headerlines.append(
                        "# Corresponding parcellation: {} \n".format(self.parc_file)
                    )

                headerlines.append(
                    "{:<4} {:<50} {:>3} {:>3} {:>3} {:>3}".format(
                        "#No.", "Label Name:", "R", "G", "B", "A"
                    )
                )

            self.write_luttable(
                self.index, self.name, self.color, out_file, headerlines=headerlines
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
                        tsv_df["color"] = cltmisc.multi_rgb2hex(self.color)

                elif isinstance(self.color, np.ndarray):
                    tsv_df["color"] = cltmisc.multi_rgb2hex(self.color)

            self.write_tsvtable(tsv_df, out_file, force=force)
        else:
            raise ValueError("The lut_type must be 'lut' or 'tsv'")

    ######################################################################################################
    def replace_values(
        self,
        codes2rep: Union[List[Union[int, List[int]]], np.ndarray],
        new_codes: Union[int, List[int], np.ndarray],
    ) -> None:
        """
        Replace region codes with new values, supporting group replacements.

        Parameters
        ----------
        codes2rep : list or np.ndarray
            Codes to replace. Can be flat list for individual replacement
            or list of lists for group replacement.

        new_codes : int, list, or np.ndarray
            New codes to replace with. Must match number of groups.

        Raises
        ------
        ValueError
            If number of new codes doesn't match number of groups.

        Examples
        --------
        >>> # Replace individual codes
        >>> parc.replace_values([1, 2, 3], [10, 20, 30])
        >>>
        >>> # Group replacement
        >>> parc.replace_values([[1, 2], [3, 4]], [100, 200])
        """

        # Input validation
        if not hasattr(self, "data"):
            raise AttributeError("Object must have 'data' attribute")

        # Handle single integer new_codes
        if isinstance(new_codes, (int, np.integer)):
            new_codes = [np.int32(new_codes)]

        # Process codes2rep to determine structure and number of groups
        if isinstance(codes2rep, list):
            if len(codes2rep) == 0:
                raise ValueError("codes2rep cannot be empty")

            # Detect whether it's a flat list of ints or a list of lists
            if all(isinstance(x, (int, np.integer)) for x in codes2rep):
                # Interpret as individual values -> multiple groups
                codes2rep = [[x] for x in codes2rep]
            elif all(isinstance(x, list) for x in codes2rep):
                pass  # Already in group form
            else:
                raise TypeError(
                    "codes2rep must be a list of ints or a list of lists of ints"
                )
            n_groups = len(codes2rep)

        elif isinstance(codes2rep, np.ndarray):
            if codes2rep.ndim == 1:
                codes2rep = [[int(x)] for x in codes2rep.tolist()]
            else:
                raise TypeError("Unsupported numpy array shape for codes2rep")
            n_groups = len(codes2rep)
        else:
            raise TypeError(
                f"codes2rep must be list or numpy array, got {type(codes2rep)}"
            )

        # Optionally convert codes using cltmisc.build_indices if available
        for i, group in enumerate(codes2rep):
            codes2rep[i] = cltmisc.build_indices(group, nonzeros=False)

        # Process new_codes
        if isinstance(new_codes, list):
            new_codes = cltmisc.build_indices(new_codes, nonzeros=False)
            new_codes = np.array(new_codes, dtype=np.int32)

        elif isinstance(new_codes, (int, np.integer)):
            new_codes = np.array([new_codes], dtype=np.int32)
        else:
            new_codes = np.array(new_codes, dtype=np.int32)

        # Validate matching lengths
        if len(new_codes) != n_groups:
            raise ValueError(
                f"Number of new codes ({len(new_codes)}) must equal "
                f"number of groups ({n_groups}) to be replaced"
            )

        # Perform replacements
        for group_idx in range(n_groups):
            codes_to_replace = np.array(codes2rep[group_idx])
            mask = np.isin(self.data, codes_to_replace)
            self.data[mask] = new_codes[group_idx]

        # Optional post-processing
        if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
            if hasattr(self, "adjust_values"):
                self.adjust_values()

        if hasattr(self, "parc_range"):
            self.parc_range()

    ######################################################################################################
    def parc_range(self) -> None:
        """
        Update minimum and maximum label values in parcellation.

        Sets minlab and maxlab attributes based on non-zero values in data.

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

    #######################################################################################################
    def compute_morphometry_table(
        self,
        output_table: str = None,
        add_bids_entities: bool = False,
        map_files: Union[str, list] = None,
        map_ids: Union[str, list] = None,
        units: Union[str, list] = "unknown",
    ):
        """
        Compute morphometry table for all regions in parcellation.
        Sets morphometry containing region volumes and statistics.

        Parameters
        ----------
        output_table : str, optional
            Path to save the output table. If None, does not save. Default is None.

        add_bids_entities : bool, optional
            Whether to add BIDS entities to the output. Default is False.

        map_files : str or list, optional
            Paths to additional map files for morphometry. If None, only base morphometry is computed.
            Default is None. This method will compute morphometry for each map file provided.

        map_ids : str or list, optional
            IDs for the additional maps. If None, uses filenames as IDs. Default is None.

        units : str or list, optional
            Units for the additional maps. If None, uses "unknown". Default is "unknown".

        Raises
        ------
        TypeError
            If output_table is not a string path.

        FileNotFoundError
            If the output directory does not exist.


        Examples
        --------
        >>> # Compute morphometry table and save to CSV
        >>> parc.compute_morphometry_table(
        ...     output_table='morphometry.csv',
        ...     add_bids_entities=True,
        ...     map_files=['map1.nii.gz', 'map2.nii.gz'],
        ...     map_ids=['map1', 'map2'],
        ...     units=['mm^3', 'unknown']
        ... )
        >>> # Compute morphometry without additional maps
        >>> parc.compute_morphometry_table(
        ...     output_table='morphometry_base.csv',
        ...     add_bids_entities=False
        ... )
        >>> # Compute morphometry with a single map file
        >>> parc.compute_morphometry_table(
        ...     output_table='morphometry_single.csv',
        ...     add_bids_entities=True,
        ...     map_files='single_map.nii.gz',
        ...     map_ids='single_map',
        ...     units='mm^3'
        ... )
        """

        from . import morphometrytools as cltmorpho

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

            cont_maps = 0

            if map_files is not None:
                if isinstance(map_files, str):
                    map_files = [map_files]

                # Handle map_ids: only replace if None or wrong length
                if map_ids is None:
                    map_ids = []
                    for f in map_files:
                        map_ids.append(cltmisc.get_real_basename(f))
                elif isinstance(map_ids, str):
                    map_ids = [map_ids]
                elif len(map_ids) != len(map_files):
                    # If lengths don't match, generate new ones
                    map_ids = []
                    for f in map_files:
                        map_ids.append(cltmisc.get_real_basename(f))

                # Handle units: same logic as map_ids
                if units is None:
                    units = ["unknown"] * len(map_files)
                elif isinstance(units, str):
                    units = [units]
                elif len(units) != len(map_files):
                    units = ["unknown"] * len(map_files)

                fin_maps = []
                fin_map_ids = []
                fin_units = []

                # Check if each map file exists
                for map_file, map_id, unit in zip(map_files, map_ids, units):
                    if os.path.exists(map_file):
                        cont_maps += 1
                        fin_maps.append(map_file)
                        fin_map_ids.append(map_id)
                        fin_units.append(unit)
                    else:
                        print(f"Warning: Map file not found: {map_file}")

            # Fix: Calculate correct total (base morphometry + number of additional maps)
            total_tasks = 1 + cont_maps

            # Adding a task to the progress bar with correct total
            task = progress.add_task("Computing base morphometry", total=total_tasks)

            progress.update(
                task,
                description=f"Computing: [bold green]volume[/bold green] ([yellow]cm3[/yellow])",
                completed=1,
            )

            # Computing the volume table
            morphometry_table, *_ = cltmorpho.compute_reg_volume_fromparcellation(
                self, add_bids_entities=add_bids_entities
            )

            # If there are additional maps, compute morphometry for each map file
            if cont_maps > 0:
                # Compute morphometry for each map file
                for i, (map_file, map_id, unit) in enumerate(
                    zip(fin_maps, fin_map_ids, fin_units)
                ):
                    # Update progress description to show current map info
                    progress.update(
                        task,
                        description=f"Processing map: [bold green]{map_id}[/bold green] ([yellow]{unit}[/yellow])",
                        completed=i
                        + 2,  # +2 because we already completed the base (1) + current index
                    )

                    # Compute morphometry for the current map file
                    df, _, _ = cltmorpho.compute_reg_val_fromparcellation(
                        map_file,
                        self,
                        add_bids_entities=add_bids_entities,
                        metric=map_id,
                        units=unit,
                    )
                    morphometry_table = pd.concat([morphometry_table, df], axis=0)

            # Final update
            progress.update(
                task,
                description=f"[bold blue]Completed[/bold blue] morphometry for {cont_maps} maps",
                completed=total_tasks,
            )

            self.morphometry = morphometry_table

            # Saving the morphometry table if output_table is provided
            if output_table is not None:
                if isinstance(output_table, str):
                    output_table = Path(output_table)
                else:
                    raise TypeError("output_table must be a string path")

                # If the directory does not exist, create raise an error
                if not output_table.parent.exists():
                    raise FileNotFoundError(
                        f"Output directory does not exist: {output_table.parent}"
                    )

                # Save the DataFrame to CSV
                morphometry_table.to_csv(output_table, index=False)
                print(f"Saved morphometry table to {output_table}")

    ######################################################################################################
    def compute_volume_table(self):
        """
        Compute volume table for all regions in parcellation.

        Sets volumetable attribute containing region volumes and statistics.

        Examples
        --------
        >>> parc.compute_volume_table()
        >>> volume_df, _ = parc.volumetable
        >>> print(volume_df.head())
        """

        from . import morphometrytools as cltmorpho

        volume_table = cltmorpho.compute_reg_volume_fromparcellation(self)
        self.volumetable = volume_table

        return volume_table

    ######################################################################################################
    @staticmethod
    def lut_to_fsllut(lut_file_fs: str, lut_file_fsl: str):
        """
        Convert FreeSurfer LUT file to FSL format.

        Parameters
        ----------
        lut_file_fs : str
            Path to FreeSurfer LUT file.

        lut_file_fsl : str
            Path for output FSL LUT file.

        Examples
        --------
        >>> Parcellation.lut_to_fsllut('FreeSurferColorLUT.txt', 'fsl_colors.lut')
        """

        # Reading FreeSurfer color lut
        lut_dict = Parcellation.read_luttable(lut_file_fs)
        st_codes_lut = lut_dict["index"]
        st_names_lut = lut_dict["name"]
        st_colors_lut = lut_dict["color"]

        st_colors_lut = cltmisc.multi_hex2rgb(st_colors_lut)

        lut_lines = []
        for roi_pos, st_code in enumerate(st_codes_lut):
            st_name = st_names_lut[roi_pos]
            lut_lines.append(
                "{:<4} {:>3.5f} {:>3.5f} {:>3.5f} {:<40} ".format(
                    st_code,
                    st_colors_lut[roi_pos, 0] / 255,
                    st_colors_lut[roi_pos, 1] / 255,
                    st_colors_lut[roi_pos, 2] / 255,
                    st_name,
                )
            )

        with open(lut_file_fsl, "w") as colorLUT_f:
            colorLUT_f.write("\n".join(lut_lines))

    ######################################################################################################
    @staticmethod
    def lut_to_nilearnlut(
        input_lut_path: Union[str, Path],
        output_lut_path: Union[str, Path],
        overwrite: bool = False,
    ) -> str:
        """
        Convert a FreeSurfer-style color lookup table to nilearn-compatible format.

        This function reads a color lookup table (LUT) file with the format:
        "#No. Label Name: R G B A" and converts it to a pandas-readable format
        that nilearn's NiftiLabelsMasker can use.

        Parameters
        ----------
        input_lut_path : str or Path
            Path to the input LUT file. Must exist.
            Expected format: "label name R G B A" (will be converted to 'index' for nilearn)

        output_lut_path : str or Path
            Path for the output file. Directory must exist.

        overwrite : bool, optional
            Whether to overwrite the output file if it already exists.
            Default: False

        Returns
        -------
        str
            Path to the created nilearn-compatible LUT file

        Raises
        ------
        FileNotFoundError
            If input_lut_path doesn't exist or output directory doesn't exist
        FileExistsError
            If output file exists and overwrite=False
        ValueError
            If no valid data lines are found in the input file

        Examples
        --------
        >>> # Convert LUT file
        >>> nilearn_lut = convert_colorlut_to_nilearn(
        ...     'parcellation.lut',
        ...     'parcellation_nilearn.lut'
        ... )

        >>> # Convert with overwrite enabled
        >>> nilearn_lut = convert_colorlut_to_nilearn(
        ...     'parcellation.lut',
        ...     'parcellation_nilearn.lut',
        ...     overwrite=True
        ... )

        >>> # Use with NiftiLabelsMasker
        >>> from nilearn.maskers import NiftiLabelsMasker
        >>> masker = NiftiLabelsMasker(labels_img='parcellation.nii.gz', lut=nilearn_lut)
        """
        # Convert to Path objects
        input_path = Path(input_lut_path)
        output_path = Path(output_lut_path)

        # Check if input file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Input LUT file not found: {input_path}")

        # Check if output directory exists
        if not output_path.parent.exists():
            raise FileNotFoundError(
                f"Output directory does not exist: {output_path.parent}"
            )

        # Check if output file exists and handle overwrite
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_path}. Use overwrite=True to overwrite."
            )

        print(f"Converting: {input_path} -> {output_path}")

        # Parse the input file
        data_rows = []

        with open(input_path, "r") as file:
            for line in file:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Split line and extract data
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        # Format: index name R G B A (nilearn expects 'index' column)
                        label = int(parts[0])
                        r, g, b, a = map(int, parts[-4:])  # Last 4 are RGBA
                        name_parts = parts[1:-4]  # Everything between label and RGBA
                        name = "-".join(name_parts) if name_parts else f"region_{label}"

                        data_rows.append(
                            {
                                "index": label,  # nilearn expects 'index' not 'label'
                                "name": name,
                                "R": r,
                                "G": g,
                                "B": b,
                                "A": a,
                            }
                        )
                    except ValueError:
                        print(f"Warning: Skipping invalid line: {line}")
                        continue

        # Check if we found any data
        if not data_rows:
            raise ValueError(f"No valid data lines found in {input_path}")

        # Create DataFrame and save
        df = pd.DataFrame(data_rows)
        df = df.sort_values("index").reset_index(drop=True)
        df.to_csv(output_path, sep=" ", index=False)

        print(f"✓ Converted {len(data_rows)} regions")
        print(f"✓ Saved to: {output_path}")

        return str(output_path)

    ######################################################################################################
    @staticmethod
    def read_luttable(
        in_file: str, filter_by_name: Union[str, List[str]] = None
    ) -> dict:
        """
        Read FreeSurfer lookup table file.

        Parameters
        ----------
        in_file : str
            Path to LUT file.

        filter_by_name : str or list, optional
            Filter regions by name substring(s). Default is None.

        Returns
        -------
        dict
            Dictionary with 'index', 'name', and 'color' keys.

        Examples
        --------
        >>> lut_dict = Parcellation.read_luttable('FreeSurferColorLUT.txt')
        >>> print(f"Found {len(lut_dict['index'])} regions")
        >>>
        >>> # Filter for hippocampus
        >>> hippo_dict = Parcellation.read_luttable(
        ...     'FreeSurferColorLUT.txt',
        ...     filter_by_name='hippocampus'
        ... )
        """

        # Read the LUT file content
        with open(in_file, "r", encoding="utf-8") as f:
            lut_content = f.readlines()

        # Initialize lists to store parsed data
        region_codes = []
        region_names = []
        region_colors_rgb = []

        # Parse each non-comment line in the file
        for line in lut_content:
            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("\\\\"):
                continue

            # Split line into components
            parts = line.split()
            if len(parts) < 5:  # Need at least code, name, R, G, B
                continue

            # Extract data
            try:
                code = int(parts[0])  # Using Python's built-in int, not numpy.int32
                name = parts[1]
                r, g, b = int(parts[2]), int(parts[3]), int(parts[4])

                region_codes.append(code)
                region_names.append(name)
                region_colors_rgb.append([r, g, b])
            except (ValueError, IndexError):
                # Skip malformed lines
                continue

        # Convert RGB colors to hex format
        try:
            # Use the existing multi_rgb2hex function if available
            region_colors_hex = cltmisc.multi_rgb2hex(np.array(region_colors_rgb))
        except (NameError, AttributeError):
            # Fallback to direct conversion if the function isn't available
            region_colors_hex = [
                f"#{r:02x}{g:02x}{b:02x}" for r, g, b in region_colors_rgb
            ]
        if filter_by_name is not None:
            if isinstance(filter_by_name, str):
                filter_by_name = [filter_by_name]

            filtered_indices = cltmisc.get_indexes_by_substring(
                region_names, filter_by_name
            )

            # Filter the LUT based on the provided names
            # filtered_indices = [
            #     i for i, name in enumerate(region_names) if name in filter_by_name
            # ]
            region_codes = [region_codes[i] for i in filtered_indices]
            region_names = [region_names[i] for i in filtered_indices]
            region_colors_hex = [region_colors_hex[i] for i in filtered_indices]

        # Create and return the result dictionary
        return {
            "index": region_codes,  # Now contains standard Python integers
            "name": region_names,
            "color": region_colors_hex,
        }

    ######################################################################################################
    @staticmethod
    def read_tsvtable(
        in_file: str, filter_by_name: Union[str, List[str]] = None
    ) -> dict:
        """
        Read TSV lookup table file.

        Parameters
        ----------
        in_file : str
            Path to TSV file.

        filter_by_name : str or list, optional
            Filter regions by name substring(s). Default is None.

        Returns
        -------
        dict
            Dictionary with column names as keys.

        Raises
        ------
        ValueError
            If required columns 'index' and 'name' are missing.

        Examples
        --------
        >>> tsv_dict = Parcellation.read_tsvtable('regions.tsv')
        >>> print(f"Columns: {list(tsv_dict.keys())}")
        """

        # Check if file exists
        if not os.path.exists(in_file):
            raise FileNotFoundError(f"TSV file not found: {in_file}")

        try:
            # Read the TSV file into a pandas DataFrame
            tsv_df = pd.read_csv(in_file, sep="\t")

            # Check for required columns
            required_columns = ["index", "name"]
            missing_columns = [
                col for col in required_columns if col not in tsv_df.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"TSV file missing required columns: {', '.join(missing_columns)}"
                )

            # Convert DataFrame to dictionary
            tsv_dict = tsv_df.to_dict(orient="list")

            # Convert index values to integers
            if "index" in tsv_dict:
                tsv_dict["index"] = [int(x) for x in tsv_dict["index"]]

            if filter_by_name is not None:
                if isinstance(filter_by_name, str):
                    filter_by_name = [filter_by_name]

                filtered_indices = cltmisc.get_indexes_by_substring(
                    tsv_dict["name"], filter_by_name
                )

                # Filter the TSV based on the provided names
                tsv_dict = {
                    key: [tsv_dict[key][i] for i in filtered_indices]
                    for key in tsv_dict.keys()
                }

            return tsv_dict

        except pd.errors.EmptyDataError:
            raise ValueError("The TSV file is empty or improperly formatted")
        except pd.errors.ParserError:
            raise ValueError("The TSV file could not be parsed correctly")
        except Exception as e:
            raise ValueError(f"Error reading TSV file: {str(e)}")

    #######################################################################################################
    @staticmethod
    def write_luttable(
        codes: list,
        names: list,
        colors: Union[list, np.ndarray],
        out_file: str = None,
        headerlines: Union[list, str] = None,
        boolappend: bool = False,
        force: bool = True,
    ):
        """
        Write FreeSurfer format lookup table file.

        Parameters
        ----------
        codes : list
            Region codes.

        names : list
            Region names.

        colors : list or np.ndarray
            Region colors (RGB or hex).

        out_file : str, optional
            Output file path. Default is None.

        headerlines : list or str, optional
            Header lines. Default is None.

        boolappend : bool, optional
            Whether to append to existing file. Default is False.

        force : bool, optional
            Whether to overwrite existing files. Default is True.

        Returns
        -------
        list
            List of formatted LUT lines.

        Examples
        --------
        >>> Parcellation.write_luttable(
        ...     [1, 2, 3],
        ...     ['region1', 'region2', 'region3'],
        ...     ['#FF0000', '#00FF00', '#0000FF'],
        ...     'output.lut'
        ... )
        """

        # Check if the file already exists and if the force parameter is False
        if out_file is not None:
            if os.path.exists(out_file) and not force:
                print("Warning: The file already exists. It will be overwritten.")

            out_dir = os.path.dirname(out_file)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        happend_bool = True  # Boolean to append the headerlines
        if headerlines is None:
            happend_bool = (
                False  # Only add this if it is the first time the file is created
            )
            now = datetime.now()
            date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
            headerlines = [
                "# $Id: {} {} \n".format(out_file, date_time),
                "{:<4} {:<50} {:>3} {:>3} {:>3} {:>3}".format(
                    "#No.", "Label Name:", "R", "G", "B", "A"
                ),
            ]

        elif isinstance(headerlines, str):
            headerlines = [headerlines]

        elif isinstance(headerlines, list):
            pass

        else:
            raise ValueError("The headerlines parameter must be a list or a string")

        if boolappend:
            if not os.path.exists(out_file):
                raise ValueError("The file does not exist")
            else:
                with open(out_file, "r") as file:
                    luttable = file.readlines()

                luttable = [l.strip("\n\r") for l in luttable]
                luttable = ["\n" if element == "" else element for element in luttable]

                if happend_bool:
                    luttable = luttable + headerlines

        else:
            luttable = headerlines

        if isinstance(colors, list):
            if isinstance(colors[0], str):
                colors = cltmisc.harmonize_colors(colors)
                colors = cltmisc.multi_hex2rgb(colors)
            elif isinstance(colors[0], list):
                colors = np.array(colors)
            elif isinstance(colors[0], np.ndarray):
                colors = np.vstack(colors)

        # Table for parcellation
        for roi_pos, roi_name in enumerate(names):

            if roi_pos == 0:
                luttable.append("\n")

            luttable.append(
                "{:<4} {:<50} {:>3} {:>3} {:>3} {:>3}".format(
                    codes[roi_pos],
                    names[roi_pos],
                    colors[roi_pos, 0],
                    colors[roi_pos, 1],
                    colors[roi_pos, 2],
                    0,
                )
            )
        luttable.append("\n")

        if out_file is not None:
            if os.path.isfile(out_file) and force:
                # Save the lut table
                with open(out_file, "w") as colorLUT_f:
                    colorLUT_f.write("\n".join(luttable))
            elif not os.path.isfile(out_file):
                # Save the lut table
                with open(out_file, "w") as colorLUT_f:
                    colorLUT_f.write("\n".join(luttable))

        return luttable

    #######################################################################################################
    @staticmethod
    def write_tsvtable(
        tsv_df: Union[pd.DataFrame, dict],
        out_file: str,
        boolappend: bool = False,
        force: bool = False,
    ):
        """
        Write TSV format lookup table file.

        Parameters
        ----------
        tsv_df : pd.DataFrame or dict
            Data to write with index/name/color information.

        out_file : str
            Output file path.

        boolappend : bool, optional
            Whether to append to existing file. Default is False.

        force : bool, optional
            Whether to overwrite existing files. Default is False.

        Returns
        -------
        str
            Output file path.

        Examples
        --------
        >>> data = {'index': [1, 2], 'name': ['region1', 'region2']}
        >>> Parcellation.write_tsvtable(data, 'regions.tsv', force=True)
        """

        # Check if the file already exists and if the force parameter is False
        if os.path.exists(out_file) and not force:
            print("Warning: The TSV file already exists. It will be overwritten.")

        out_dir = os.path.dirname(out_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Table for parcellation
        # 1. Converting colors to hexidecimal string

        if isinstance(tsv_df, pd.DataFrame):
            tsv_dict = tsv_df.to_dict(orient="list")
        else:
            tsv_dict = tsv_df

        if "name" not in tsv_dict.keys() or "index" not in tsv_dict.keys():
            raise ValueError("The dictionary must contain the keys 'index' and 'name'")

        codes = tsv_dict["index"]
        names = tsv_dict["name"]

        if "color" in tsv_dict.keys():
            temp_colors = tsv_dict["color"]

            if isinstance(temp_colors, list):
                if isinstance(temp_colors[0], str):
                    if temp_colors[0][0] != "#":
                        raise ValueError("The colors must be in hexadecimal format")

                elif isinstance(temp_colors[0], list):
                    colors = np.array(temp_colors)
                    seg_hexcol = cltmisc.multi_rgb2hex(colors)
                    tsv_dict["color"] = seg_hexcol

            elif isinstance(temp_colors, np.ndarray):
                seg_hexcol = cltmisc.multi_rgb2hex(temp_colors)
                tsv_dict["color"] = seg_hexcol

        if boolappend:
            if not os.path.exists(out_file):
                raise ValueError("The file does not exist")
            else:
                tsv_orig = Parcellation.read_tsvtable(in_file=out_file)

                # Create a list with the common keys between tsv_orig and tsv_dict
                common_keys = list(set(tsv_orig.keys()) & set(tsv_dict.keys()))

                # List all the keys for both dictionaries
                all_keys = list(set(tsv_orig.keys()) | set(tsv_dict.keys()))

                # Concatenate values for those keys and the rest of the keys that are in tsv_orig add white space
                for key in common_keys:
                    tsv_orig[key] = tsv_orig[key] + tsv_dict[key]

                for key in all_keys:
                    if key not in common_keys:
                        if key in tsv_orig.keys():
                            tsv_orig[key] = tsv_orig[key] + [""] * len(tsv_dict["name"])
                        elif key in tsv_dict.keys():
                            tsv_orig[key] = [""] * len(tsv_orig["name"]) + tsv_dict[key]
        else:
            tsv_orig = tsv_dict

        # Dictionary to dataframe
        tsv_df = pd.DataFrame(tsv_orig)

        if os.path.isfile(out_file) and force:

            # Save the tsv table
            with open(out_file, "w+") as tsv_file:
                tsv_file.write(tsv_df.to_csv(sep="\t", index=False))

        elif not os.path.isfile(out_file):
            # Save the tsv table
            with open(out_file, "w+") as tsv_file:
                tsv_file.write(tsv_df.to_csv(sep="\t", index=False))

        return out_file

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
