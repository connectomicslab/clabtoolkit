import os
from datetime import datetime
import copy

import numpy as np
import pandas as pd
import nibabel as nib
from typing import Union, List

# Importing local modules
from . import misctools as cltmisc
from . import segmentationtools as cltseg


class Parcellation:
    """
    The `Parcellation` class provides comprehensive tools for working with brain parcellation data.
    It enables loading, manipulating, and analyzing parcellation files (typically in NIfTI format)
    along with their associated lookup tables. This class supports various operations including
    filtering regions by name or code, applying masks, grouping regions, calculating volumes, and
    exporting results. It handles both the volumetric data and the associated metadata (region
    indices, names, and colors), making it a complete solution for neuroimaging parcellation analysis
    workflows. The class is designed to work seamlessly with annotation files that define brain
    regions and supports various common file formats in neuroimaging research.

    """

    def __init__(
        self, parc_file: Union[str, np.uint] = None, affine: np.float64 = None
    ):

        if parc_file is not None:
            if isinstance(parc_file, str):
                if os.path.exists(parc_file):
                    self.parc_file = parc_file
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

                self.data = parc_file
                self.parc_file = "numpy_array"
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

                # Generate the colors
                self.color = cltmisc.create_random_colors(
                    len(self.index), output_format="hex"
                )

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

            # Detect minimum and maximum labels
            self.parc_range()

    def prepare_for_tracking(self):
        """
        Prepare the parcellation for fibre tracking. It will add the parcellated wm voxels to its
        corresponding gm label. It also puts to zero the voxels that are not in the gm.

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

    def keep_by_name(self, names2look: Union[list, str], rearrange: bool = False):
        """
        Filter the parcellation by a list of names or just a a substring that could be included in the name.
        It will keep only the structures with names containing the strings specified in the list.
        @params:
            names2look     - Required  : List or string of names to look for. It can be a list of strings or just a string.
            rearrange      - Required  : If True, the parcellation will be rearranged starting from 1. Default = False
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

    def keep_by_code(
        self, codes2keep: Union[list, np.ndarray], rearrange: bool = False
    ):
        """
        Filter the parcellation by a list of codes. It will keep only the structures with codes specified in the list.

        Parameters
        ----------
        codes2keep : list or np.ndarray
            List of codes to look for. Only structures with these codes will be retained.

        rearrange : bool, optional
            If True, the parcellation will be rearranged starting from 1.
            Default is False.

        Returns
        -------
        self or new_instance
            Returns the filtered parcellation object with only the specified codes retained.

        Raises
        ------
        ValueError
            If codes2keep is empty or contains invalid codes.
        TypeError
            If codes2keep is not a list or numpy array.

        Examples
        --------
        >>> # Keep only specific brain regions
        >>> parcellation.keep_by_code([1, 2, 5, 10])

        >>> # Keep regions and rearrange codes starting from 1
        >>> parcellation.keep_by_code([100, 200, 300], rearrange=True)

        >>> # Using numpy array of codes
        >>> import numpy as np
        >>> codes = np.array([1, 3, 7, 12, 15])
        >>> parcellation.keep_by_code(codes, rearrange=False)
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

    def remove_by_code(
        self, codes2remove: Union[list, np.ndarray], rearrange: bool = False
    ):
        """
        Remove the structures with the codes specified in the list.
        @params:
            codes2remove     - Required  : List of codes to remove:
            rearrange        - Required  : If True, the parcellation will be rearranged starting from 1. Default = False
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

    def remove_by_name(self, names2remove: Union[list, str], rearrange: bool = False):
        """
        Remove the structures with the names specified in the list.

        Parameters
        ----------
        names2remove : list, str
            List of names to remove. It can be a list of strings or just a string.

        rearrange : bool
            If True, the parcellation will be rearranged starting from 1. Default = False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the names were not found in the parcellation.
        ValueError
            If the parcellation does not contain the attributes name, index and color.
        ValueError
            If the names2remove is not a list or string.
        ValueError
            If the names2remove is a list of strings and the parcellation does not contain the attributes name, index and color.

        Examples
        --------
        >>> parcellation = Parcellation("parc_file.nii")
        >>> parcellation.remove_by_name(["name1", "name2"], rearrange=True)
        This will produce a new object with the parcellation data, index, name and color without the regions with names equal to name1 or name2.
        The new labels will go now from 1 to N-2, where N is the original number of regions in that parcellation

        >>> parcellation.remove_by_name("name3", rearrange=False)
        This will produce a new object with the parcellation data, index, name and color without the regions with names equal to name3
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

    def apply_mask(
        self,
        image_mask,
        codes2mask: Union[list, np.ndarray] = None,
        mask_type: str = "upright",
        fill: bool = False,
    ):
        """
        Applies a mask to the parcellation data, restricting the spatial extension of the
        parcellation with values equal codes2mask only to the voxels included in the image_mask.

        This method modifies the 3D parcellation data (`self.data`) by applying a given
        3D mask array. All voxels where the mask has a value of zero will be set to
        zero in the parcellation data, effectively excluding those regions
        from further analysis.

        Parameters
        ----------
        image_mask : np.ndarray, Parcellation or str
            A 3D numpy array with the same shape as `self.data`. The mask indicates
            which voxels should be retained (non-zero values) and which should be set
            to `mask_value` (zero values).

        codes2mask : int, list, np.ndarray
            The codes of the regions that will be masked. If None, all regions with
            non-zero values will be masked. Default is None.

        mask_type : str
            The type of mask to apply. If 'upright', the mask will be applied to the
            regions with the codes specified in `codes2mask`. If 'inverted', the mask
            will be applied to the regions with codes different from those specified
            in `codes2mask`. Default is 'upright'.

        fill : bool
            If True, the regions will grow until the fill the provided mask. Default is False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the mask shape does not match the shape of `self.data`.

        Example
        -------
        >>> parcellation = Parcellation("parc_file.nii")
        >>> mask = np.array([...])  # A 3D mask array of the same shape as `data`
        >>> parcellation.apply_mask(mask, codes2mask=[1, 2, 3], mask_type='upright')

        This will apply the mask to the regions with codes 1, 2, and 3 in the parcellation.

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
            self.data = cltseg.region_growing(self.data, bool_mask)

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
        Masks the images specified in `image_2mask` with a binary mask created from the parcellation data.
        It will use the regions with the codes specified in `codes2mask` to create a binary mask. The mask can
        be created with all the regions in the parcellation if `codes2mask` is None. The mask can be applied
        to the regions with the codes specified in `codes2mask` or to the regions with codes different from
        those specified in `codes2mask`. The masked images will be saved in the paths specified in `masked_image`.

        Parameters
        ----------
        image_2mask: str, list, np.ndarray
            The path to the image file or a list of paths to the images that will be masked. It can also be a
            3D numpy array with the same shape as `self.data`.

        masked_image: str, list
            The path to the image file or a list of paths to the images where the masked images will be saved.
            If None, the masked images will not be saved. Default is None.

        codes2mask: int, list, np.ndarray
            The codes of the regions that will be masked. If None, all regions with non-zero values will be masked.
            Default is None.

        mask_type: str
            The type of mask to apply. If 'upright', the mask will use the regions with the codes specified in
            `codes2mask` to create the binary mask. If 'inverted', the mask will use the regions with codes different
            from those specified in `codes2mask` to create the binary mask. Default is 'upright'.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the number of images to mask is different from the number of images to be saved.

        Example
        -------
        >>> parcellation = Parcellation("parc_file.nii")
        >>> image = "image.nii"
        >>> masked_image = "masked_image.nii"
        >>> parcellation.mask_image(image_2mask=image, masked_image=masked_image, codes2mask=[1, 2, 3], mask_type='upright')

        This will mask the image with the parcellation, using the regions with codes 1, 2, and 3 to create the binary mask.
        The masked image will be saved in the path specified in `masked_image`.


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

    def adjust_values(self):
        """
        Adjust the codes, indexes, names and colors to the values present on the parcellation

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

    def group_by_code(
        self,
        codes2group: Union[list, np.ndarray],
        new_codes: Union[list, np.ndarray] = None,
        new_names: Union[list, str] = None,
        new_colors: Union[list, np.ndarray] = None,
    ):
        """
        Group the structures with the codes specified in the list or array codes2group.
        @params:
            codes2group      - Required  : List, numpy array or list of list of codes to group:
            new_codes        - Optional  : New codes for the groups. It can assign new codes
                                            otherwise it will assign the codes from 1 to number of groups:
            new_names        - Optional  : New names for the groups:
            new_colors       - Optional  : New colors for the groups:

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

    def group_by_name(
        self,
        names2group: Union[List[list], List[str]],
        new_codes: Union[list, np.ndarray] = None,
        new_names: Union[list, str] = None,
        new_colors: Union[list, np.ndarray] = None,
    ):
        """
        Group the structures with the names specified in the list or array names2group.
        @params:
            names2group      - Required  : List or list of list of names to group:
            new_codes        - Optional  : New codes for the groups. It can assign new codes
                                            otherwise it will assign the codes from 1 to number of groups:
            new_names        - Optional  : New names for the groups:
            new_colors       - Optional  : New colors for the groups:

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

    def rearrange_parc(self, offset: int = 0):
        """
        Rearrange parcellation labels to consecutive integers starting from offset + 1.

        This method remaps all non-zero parcellation codes to consecutive integers
        starting from the specified offset, eliminating gaps in the labeling scheme.
        Zero values (background) are preserved unchanged.

        Parameters
        ----------
        offset : int, optional
            Starting value for the rearranged parcellation labels. Default is 1.

        Notes
        -----
        - Modifies self.data in-place
        - Zero values are treated as background and remain unchanged
        - Original parcellation codes are sorted before reassignment
        - If index, name, and color attributes exist, the index attribute is updated
        to reflect the new consecutive labeling scheme
        - Calls self.parc_range() after rearrangement

        Examples
        --------
        Original parcellation with codes [5, 10, 15]:
        - With offset=1: becomes [2, 3, 4]
        - With offset=10: becomes [11, 12, 13]

        Returns
        -------
        None
            Method modifies the object in-place.
        """

        st_codes = np.unique(self.data)
        st_codes = st_codes[st_codes != 0]

        # Parcellation with values starting from 1 or starting from the offset
        new_parc = np.zeros_like(self.data, dtype="int16")
        for i, code in enumerate(st_codes):
            new_parc[self.data == code] = i + 1 + offset
        self.data = new_parc

        if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
            temp_index = np.unique(self.data)
            temp_index = temp_index[temp_index != 0]
            self.index = temp_index.tolist()

        self.parc_range()

    def add_parcellation(self, parc2add, append: bool = False):
        """
        Combines another parcellation object into the current parcellation.

        This method appends the regions of another parcellation into the
        current object. The behavior of the combination depends on the `append`:

        - "True": Adds the new regions with new labels by adding the maximum label of the
        current parcellation to the data of the other parcellation.

        - "False": Integrates the data of the other parcellation, keeping the labels of the
        current parcellation.

        Parameters
        ----------
        parc2add : Parcellation
                Another instance of the `Parcellation` class to be combined with the current
                parcellation.
        append : bool
                If True, the new regions will be added with new labels. If False, the labels
                of the current parcellation will be kept.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `other` is not an instance of the `Parcellation` class.
        ValueError
            If `merge_method` is not one of the supported values ("append" or "merge").

        Example
        -------
        >>> parcellation1 = Parcellation(parc1.nii.gz)
        >>> parcellation2 = Parcellation(parc2.nii.gz)
        >>> parcellation1.add_parcellation(parcellation2, append=False)
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

    def save_parcellation(
        self,
        out_file: str,
        affine: np.float64 = None,
        headerlines: Union[list, str] = None,
        save_lut: bool = False,
        save_tsv: bool = False,
    ):
        """
        Save the parcellation to a file
        @params:
            out_file     - Required  : Output file:
            affine       - Optional  : Affine matrix. Default = None
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

    def load_colortable(self, lut_file: Union[str, dict] = None, lut_type: str = "lut"):
        """
        Add a lookup table to the parcellation
        @params:
            lut_file     - Required  : Lookup table file. It can be a string with the path to the
                                        file or a dictionary containing the keys 'index', 'color' and 'name':
            lut_type     - Optional  : Type of the lut file: 'lut' or 'tsv'. Default = 'lut'
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

    def sort_index(self):
        """
        This method sorts the index, name and color attributes of the parcellation according to the index
        """

        # Sort the all_index and apply the order to all_name and all_color
        sort_index = np.argsort(self.index)
        self.index = [self.index[i] for i in sort_index]
        self.name = [self.name[i] for i in sort_index]
        self.color = [self.color[i] for i in sort_index]

    def export_colortable(
        self,
        out_file: str,
        lut_type: str = "lut",
        headerlines: Union[list, str] = None,
        force: bool = True,
    ):
        """
        Export the lookup table to a file
        @params:
            out_file     - Required  : Lookup table file:
            lut_type     - Optional  : Type of the lut file: 'lut' or 'tsv'. Default = 'lut'
            force        - Optional  : If True, it will overwrite the file. Default = True
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

    def replace_values(
        self,
        codes2rep: Union[List[Union[int, List[int]]], np.ndarray],
        new_codes: Union[int, List[int], np.ndarray],
    ) -> None:
        """
        Replace groups of values in the 3D array/image with new codes.

        This method allows for flexible replacement of pixel/voxel values, supporting
        both single value replacements and group replacements where multiple original
        values are mapped to a single new value.

        Parameters
        ----------
        codes2rep : list, np.ndarray, or list of lists
            Values or groups of values to be replaced. Can be:
            - Single list: [1, 2, 3] - replace values 1, 2, 3 individually
            - List of lists: [[1, 2], [3, 4]] - replace groups (1,2)->new_code[0], (3,4)->new_code[1]
            - numpy array: np.array([1, 2, 3]) - replace values 1, 2, 3 individually
        new_codes : int, list, or np.ndarray
            New values to replace with. Must match the number of groups in codes2rep.
            - Single int: used when codes2rep is a single group
            - List/array: length must equal number of groups in codes2rep

        Returns
        -------
        None
            Modifies self.data in-place

        Raises
        ------
        ValueError
            If the number of new codes doesn't match the number of groups to replace
        TypeError
            If input types are not supported

        Examples
        --------
        >>> # Single value replacement
        >>> image.replace_values([1], [100])  # Replace all 1s with 100s

        >>> # Multiple individual replacements
        >>> image.replace_values([1, 2, 3], [10, 20, 30])  # 1->10, 2->20, 3->30

        >>> # Group replacements
        >>> image.replace_values([[1, 2], [3, 4]], [100, 200])  # (1,2)->100, (3,4)->200

        >>> # Mixed usage with numpy array
        >>> image.replace_values(np.array([5, 6]), 500)  # Replace 5s and 6s with 500
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

    def parc_range(self) -> None:
        """
        Detect and update the range of non-zero labels in the data.

        Updates the minlab and maxlab attributes based on unique non-zero values
        in self.data.

        Returns
        -------
        None
            Updates self.minlab and self.maxlab attributes in-place
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

    def compute_volume_table(self):
        """
        Compute the volume table of the parcellation.
        This method computes the volume of each region in the parcellation and stores it in the attribute called volumetable.

        """
        from . import morphometrytools as cltmorpho

        volume_table = cltmorpho.compute_reg_volume_fromparcellation(self)
        self.volumetable = volume_table

    @staticmethod
    def lut_to_fsllut(lut_file_fs: str, lut_file_fsl: str):
        """
        Convert FreeSurfer lut file to FSL lut file
        @params:
            lut_file_fs     - Required  : FreeSurfer color lut:
            lut_file_fsl      - Required  : FSL color lut:
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

    @staticmethod
    def read_luttable(
        in_file: str, filter_by_name: Union[str, List[str]] = None
    ) -> dict:
        """
        Read and parse a FreeSurfer Color Lookup Table (LUT) file.

        This method reads a FreeSurfer color lookup table file and parses its contents into
        a structured dictionary containing region codes, names, and colors. The LUT file format
        follows FreeSurfer's standard format where each non-comment line contains a region code,
        name, and RGB color values.

        Parameters
        ----------
        in_file : str
            Path to the color lookup table file.

        filter_by_name: Union[str, List[str]], optional
            If provided, filter the regions by name. Only regions containing this substring will be included.
            Can be a single string or a list of strings.

        Returns
        -------
        dict
            Dictionary with the following keys:
            - 'index': List of integer region codes (standard Python integers)
            - 'name': List of region name strings
            - 'color': List of hex color codes (format: '#RRGGBB')

        Examples
        --------
        # Setup common variables
        >>> fs_dir = os.environ.get('FREESURFER_HOME')
        >>> lut_file = os.path.join(fs_dir, 'FreeSurferColorLUT.txt')

        >>> lut_dict = read_luttable(lut_file)
        >>> lut_dict['index'][:3]  # First three region codes
        [0, 1, 2]
        >>> lut_dict['name'][:3]  # First three region names
        ['Unknown', 'Left-Cerebral-Exterior', 'Left-Cerebral-White-Matter']
        >>> lut_dict['color'][:3]  # First three colors in hex format
        ['#000000', '#4682b4', '#f5f5f5']

        >>> # Get information for a specific region (e.g., hippocampus)
        >>> lut_dict = cltparc.Parcellation.read_luttable(lut_file, filter_by_name='hippocampus')
        >>> print(f"Index: {lut_dict["index"]}...")
        Index: [17, 53, 106, 115]...
        >>> print(f"Label: {lut_dict["name"]}...")
        Label: ['Left-Hippocampus', 'Right-Hippocampus', 'Left-hippocampus-intensity-abnormality', 'Right-hippocampus-intensity-abnormality']...
        >>> print(f"Color: {lut_dict["color"]}...")
        Color: ['#dcd814', '#dcd814', '#7c8fb2', '#7c8fb2']...

        Notes
        -----
        - Comment lines (starting with '#') in the LUT file are ignored
        - Each non-comment line should have at least 5 elements: code, name, R, G, B
        - The returned region codes are standard Python integers, not numpy objects
        - If the multi_rgb2hex function is not available, the color conversion will need
        to be modified
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

    @staticmethod
    def read_tsvtable(
        in_file: str, filter_by_name: Union[str, List[str]] = None
    ) -> dict:
        """
        Read and parse a TSV (Tab-Separated Values) lookup table file.

        This method reads a TSV file containing parcellation information and returns a dictionary
        with the data. The TSV file must contain at least 'index' and 'name' columns. If a 'color'
        column is present, it will be included in the returned dictionary.

        Parameters
        ----------
        in_file : str
            Path to the TSV lookup table file

        filter_by_name: Union[str, List[str]], optional
            If provided, filter the regions by name. Only regions containing this substring will be included.
            Can be a single string or a list of strings.

        Returns
        -------
        dict
            Dictionary with keys corresponding to column names in the TSV file.
            Must include at least:
            - 'index': List of integer region codes (standard Python integers)
            - 'name': List of region name strings
            May also include:
            - 'color': List of color codes if present in the TSV file
            - Any other columns present in the TSV file

        Examples
        --------
        >>> tsv_dict = read_tsvtable('regions.tsv')
        >>> tsv_dict['index'][:3]  # First three region codes
        [0, 1, 2]
        >>> tsv_dict['name'][:3]  # First three region names
        ['Unknown', 'Left-Cerebral-Exterior', 'Left-Cerebral-White-Matter']

        >>> # Check if color information is available
        >>> if 'color' in tsv_dict:
        ...     print(f"Color for region {tsv_dict['name'][0]}: {tsv_dict['color'][0]}")

        Raises
        ------
        FileNotFoundError
            If the specified TSV file does not exist
        ValueError
            If the TSV file does not contain required 'index' and 'name' columns

        Notes
        -----
        - The 'index' column values are converted to standard Python integers
        - All other columns are preserved in their original format
        - If the file cannot be parsed as a TSV, pandas exceptions may be raised
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
        Function to create a lut table for parcellation

        Parameters
        ----------
        codes : list
            List of codes for the parcellation
        names : list
            List of names for the parcellation
        colors : list
            List of colors for the parcellation
        lut_filename : str
            Name of the lut file
        headerlines : list or str
            List of strings for the header lines

        Returns
        -------
        out_file: file
            Lut file with the table

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

    @staticmethod
    def write_tsvtable(
        tsv_df: Union[pd.DataFrame, dict],
        out_file: str,
        boolappend: bool = False,
        force: bool = False,
    ):
        """
        Function to create a tsv table for parcellation

        Parameters
        ----------
        codes : list
            List of codes for the parcellation
        names : list
            List of names for the parcellation
        colors : list
            List of colors for the parcellation
        tsv_filename : str
            Name of the tsv file

        Returns
        -------
        tsv_file: file
            Tsv file with the table

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

    @staticmethod
    def tissue_seg_table(tsv_filename):
        """
        Function to create a tsv table for tissue segmentation

        Parameters
        ----------
        tsv_filename : str
            Name of the tsv file

        Returns
        -------
        seg_df: pandas DataFrame
            DataFrame with the tsv table

        """

        # Table for tissue segmentation
        # 1. Default values for tissues segmentation table
        seg_rgbcol = np.array([[172, 0, 0], [0, 153, 76], [0, 102, 204]])
        seg_codes = np.array([1, 2, 3])
        seg_names = ["cerebro_spinal_fluid", "gray_matter", "white_matter"]
        seg_acron = ["CSF", "GM", "WM"]

        # 2. Converting colors to hexidecimal string
        seg_hexcol = []
        nrows, ncols = seg_rgbcol.shape
        for i in np.arange(0, nrows):
            seg_hexcol.append(
                cltmisc.rgb2hex(seg_rgbcol[i, 0], seg_rgbcol[i, 1], seg_rgbcol[i, 2])
            )

        seg_df = pd.DataFrame(
            {
                "index": seg_codes,
                "name": seg_names,
                "abbreviation": seg_acron,
                "color": seg_hexcol,
            }
        )
        # Save the tsv table
        with open(tsv_filename, "w+") as tsv_file:
            tsv_file.write(seg_df.to_csv(sep="\t", index=False))

        return seg_df

    def print_properties(self):
        """
        Print the properties of the parcellation
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
