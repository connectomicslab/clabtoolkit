import os
from datetime import datetime
import copy

import numpy as np
import pandas as pd
import nibabel as nib
from typing import Union, List
import clabtoolkit.misctools as cltmisc
import clabtoolkit.segmentationtools as cltseg

class Parcellation:

    def __init__(self, 
                    parc_file: Union[str, np.uint] = None, 
                    affine:np.float64 = None):
        
        self.parc_file = parc_file
        
        if parc_file is not None:
            if isinstance(parc_file, str):
                if os.path.exists(parc_file):

                    temp_iparc = nib.load(parc_file)
                    affine = temp_iparc.affine
                    self.data = temp_iparc.get_fdata()
                    self.affine = affine
                    self.dtype = temp_iparc.get_data_dtype()
                    
                    if parc_file.endswith('.nii.gz'):
                        tsv_file = parc_file.replace('.nii.gz', '.tsv')
                        lut_file = parc_file.replace('.nii.gz', '.lut')
                        
                        if os.path.isfile(tsv_file):
                            self.load_colortable(lut_file=tsv_file, lut_type='tsv')
                            
                        elif not os.path.isfile(tsv_file) and os.path.isfile(lut_file):
                            self.load_colortable(lut_file=lut_file, lut_type='lut')
                            
                    elif parc_file.endswith('.nii'):
                        tsv_file = parc_file.replace('.nii', '.tsv')
                        lut_file = parc_file.replace('.nii', '.lut')
                    
                        if os.path.isfile(tsv_file):
                            self.load_colortable(lut_file=tsv_file, lut_type='tsv')
                            
                        elif not os.path.isfile(tsv_file) and os.path.isfile(lut_file):
                            self.load_colortable(lut_file=lut_file, lut_type='lut')
                
            elif isinstance(parc_file, np.ndarray):
                self.data = parc_file
                self.affine = affine

            # Adjust values to the ones present in the parcellation
            
            if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
                self.adjust_values()
            
            # Detect minimum and maximum labels
            self.parc_range()


    def keep_by_name(self, 
                            names2look: Union[list, str], 
                            rearrange: bool = False):
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
            indexes = cltmisc.get_indexes_by_substring(list1=self.name, 
                                substr=names2look, 
                                invert=False, 
                                boolcase=False)
            
            if len(indexes) > 0:
                sel_st_codes = [self.index[i] for i in indexes]
                self.keep_by_code(codes2look=sel_st_codes, rearrange=rearrange)
            else:
                print("The names were not found in the parcellation")


    def keep_by_code(self, 
                            codes2look: Union[list, np.ndarray], 
                            rearrange: bool = False):
        """
        Filter the parcellation by a list of codes. It will keep only the structures with codes specified in the list.
        @params:
            codes2look     - Required  : List of codes to look for:
            rearrange      - Required  : If True, the parcellation will be rearranged starting from 1. Default = False
        """

        # Convert the codes2look to a numpy array
        if isinstance(codes2look, list):
            codes2look = cltmisc.build_indexes(codes2look)
            codes2look = np.array(codes2look)

        # Create 
        dims = np.shape(self.data)
        out_atlas = np.zeros((dims[0], dims[1], dims[2]), dtype='int32') 

        array_3d = self.data

        # Create a boolean mask where elements are True if they are in the retain list
        mask = np.isin(array_3d, codes2look)

        # Set elements to zero if they are not in the retain list
        array_3d[~mask] = 0

        # Remove the elements from retain_list that are not present in the data
        img_tmp_codes = np.unique(array_3d)

        maskc = np.isin(codes2look, img_tmp_codes)

        # Set elements to zero if they are not in the retain list
        codes2look[~maskc] = 0

        if hasattr(self, "index"):
            temp_index = np.array(self.index) 
            index_new = []
            indexes = []

        # Rearrange the array_3d of the data to start from 1
        for i, code in enumerate(codes2look):
            out_atlas[array_3d == code] = i + 1
            
            if hasattr(self, "index"):
                # Find the element in self.index that is equal to v
                ind = np.where(temp_index == code)[0]

                if len(ind) > 0:
                    indexes.append(ind[0])
                    if rearrange:
                        index_new.append(i+1)
                    else:
                        index_new.append(self.index[ind[0]])
        if rearrange:
            self.data = out_atlas
        else:
            self.data = array_3d
        
        if hasattr(self, "index"):                       
            self.index = index_new

        # If name is an attribute of self
        if hasattr(self, "name"):
            self.name = [self.name[i] for i in indexes]

        # If color is an attribute of self
        if hasattr(self, "color"):
            self.color = [self.color[i] for i in indexes]
            
        # Detect minimum and maximum labels
        self.parc_range()

    
    def remove_by_code(self,
                            codes2remove: Union[list, np.ndarray],
                            rearrange: bool = False):
        """
        Remove the structures with the codes specified in the list.
        @params:
            codes2remove     - Required  : List of codes to remove:
            rearrange        - Required  : If True, the parcellation will be rearranged starting from 1. Default = False
        """

        if isinstance(codes2remove, list):
            codes2remove = cltmisc.build_indexes(codes2remove)
            codes2remove = np.array(codes2remove)

        for i, v in enumerate(codes2remove):
            # Find the elements in the data that are equal to v
            result = np.where(self.data == v)

            if len(result[0]) > 0:
                self.data[result[0], result[1], result[2]] = 0

        st_codes = np.unique(self.data)
        st_codes = st_codes[st_codes != 0]

        # If rearrange is True, the parcellation will be rearranged starting from 1
        if rearrange:
            self.keep_by_code(codes2look=st_codes, rearrange=True)
        else:
            self.keep_by_code(codes2look=st_codes, rearrange=False)

        # Detect minimum and maximum labels
        self.parc_range()
        
    def remove_by_name(self,
                            names2remove: Union[list, str],
                            rearrange: bool = False):
        """
        Remove the structures with the names specified in the list.
        @params:
            names2remove     - Required  : List of the names of the structures that will be removed:
            rearrange        - Required  : If True, the parcellation will be rearranged starting from 1. Default = False
        """

        
        if isinstance(names2remove, str):
            names2remove = [names2remove]
        
        if hasattr(self, "name") and hasattr(self, "index") and hasattr(self, "color"):
        
            indexes = cltmisc.get_indexes_by_substring(list1=self.name, 
                                substr=names2remove, 
                                invert=True, 
                                boolcase=False)
            
            if len(indexes) > 0:
                sel_st_codes = [self.index[i] for i in indexes]
                self.keep_by_code(codes2look=sel_st_codes, rearrange=rearrange)
                
            else:
                print("The names were not found in the parcellation")
        else:
            print("The parcellation does not contain the attributes name, index and color")
            
        # Detect minimum and maximum labels
        self.parc_range()
    
    
    def apply_mask(self, image_mask,
                        codes2mask: Union[list, np.ndarray] = None,
                        mask_type: str = 'upright',
                        fill: bool = False
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
        if mask_type not in ['upright', 'inverted']:
            raise ValueError("The mask_type must be 'upright' or 'inverted'")
        
        if codes2mask is None:
            codes2mask = np.unique(self.data)
            codes2mask = codes2mask[codes2mask != 0]
        
        if isinstance(codes2mask, list):
            codes2mask = cltmisc.build_indexes(codes2mask)
            codes2mask = np.array(codes2mask)
        
        if mask_type == 'inverted':
            self.data[np.isin(mask_data, codes2mask)==True] = 0
            bool_mask = np.isin(mask_data, codes2mask)==False

        else:
            self.data[np.isin(mask_data, codes2mask)==False] = 0
            bool_mask = np.isin(mask_data, codes2mask)==True
            
        if fill:
            
            # Refilling the unlabeled voxels according to a supplied mask
            self.data = cltseg.region_growing(self.data, bool_mask)

        if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
            self.adjust_values()
        
        # Detect minimum and maximum labels
        self.parc_range()
    
    def mask_image(self, 
                        image_2mask: Union[str, list, np.ndarray],
                        masked_image: Union[str, list, np.ndarray] = None,
                        codes2mask: Union[str, list, np.ndarray] = None,
                        mask_type: str = 'upright'
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
                raise ValueError("The number of images to mask must be equal to the number of images to be saved")
        
        if codes2mask is None:
            # Get the indexes of all values different from zero
            codes2mask = np.unique(self.data)
            codes2mask = codes2mask[codes2mask != 0]

        if isinstance(codes2mask, list):
            codes2mask = cltmisc.build_indexes(codes2mask)
            codes2mask = np.array(codes2mask)
        
        if mask_type == 'inverted':
            ind2rem = np.isin(self.data, codes2mask)==True

        else:
            ind2rem = np.isin(self.data, codes2mask)==False
        
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
                raise ValueError("The image_2mask must be a list of strings containing the paths to the images")
            
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
            self.index = index_new

        # If name is an attribute of self
        if hasattr(self, "name"):
            self.name = [self.name[i] for i in indexes]

        # If color is an attribute of self
        if hasattr(self, "color"):
            self.color = [self.color[i] for i in indexes]
        
        self.parc_range()
        
    def group_by_code(self,
                        codes2group: Union[list, np.ndarray],
                        new_codes: Union[list, np.ndarray] = None,
                        new_names: Union[list, str] = None,
                        new_colors: Union[list, np.ndarray] = None):
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
                codes2group[i] = cltmisc.build_indexes(v)
        
        # Convert the new_codes to a numpy array
        if new_codes is not None:
            if isinstance(new_codes, list):
                new_codes = cltmisc.build_indexes(new_codes)
                new_codes = np.array(new_codes)
            elif isinstance(new_codes, (str, np.integer, int)):
                new_codes = np.array([new_codes])

        else:
            new_codes = np.arange(1, n_groups + 1)

        if len(new_codes) != n_groups:
            raise ValueError("The number of new codes must be equal to the number of groups that will be created")
        
        # Convert the new_names to a list
        if new_names is not None:
            if isinstance(new_names, str):
                new_names = [new_names]

            if len(new_names) != n_groups:
                raise ValueError("The number of new names must be equal to the number of groups that will be created")
        
        # Convert the new_colors to a numpy array
        if new_colors is not None:
            if isinstance(new_colors, list):

                if isinstance(new_colors[0], str):
                    new_colors = cltmisc.multi_hex2rgb(new_colors)

                elif isinstance(new_colors[0], np.ndarray):
                    new_colors = np.array(new_colors)

                else:
                    raise ValueError("If new_colors is a list, it must be a list of hexadecimal colors or a list of rgb colors")
                
            elif isinstance(new_colors, np.ndarray):
                pass

            else:
                raise ValueError("The new_colors must be a list of colors or a numpy array")

            new_colors = cltmisc.readjust_colors(new_colors)

            if new_colors.shape[0] != n_groups:
                raise ValueError("The number of new colors must be equal to the number of groups that will be created")
        
        # Creating the grouped parcellation
        out_atlas = np.zeros_like(self.data, dtype='int16')
        for i in range(n_groups):
            code2look = np.array(codes2group[i])

            if new_codes is not None:
                out_atlas[np.isin(self.data, code2look)==True] = new_codes[i]
            else:
                out_atlas[np.isin(self.data, code2look)==True] = i + 1

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

    def group_by_name(self,
                        names2group: Union[List[list], List[str]],
                        new_codes: Union[list, np.ndarray] = None,
                        new_names: Union[list, str] = None,
                        new_colors: Union[list, np.ndarray] = None):
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
                codes2group[i] = cltmisc.build_indexes(v)
        
        # Convert the new_codes to a numpy array
        if new_codes is not None:
            if isinstance(new_codes, list):
                new_codes = cltmisc.build_indexes(new_codes)
                new_codes = np.array(new_codes)
            elif isinstance(new_codes, (str, np.integer, int)):
                new_codes = np.array([new_codes])

        else:
            new_codes = np.arange(1, n_groups + 1)

        if len(new_codes) != n_groups:
            raise ValueError("The number of new codes must be equal to the number of groups that will be created")
        
        # Convert the new_names to a list
        if new_names is not None:
            if isinstance(new_names, str):
                new_names = [new_names]

            if len(new_names) != n_groups:
                raise ValueError("The number of new names must be equal to the number of groups that will be created")
        
        # Convert the new_colors to a numpy array
        if new_colors is not None:
            if isinstance(new_colors, list):

                if isinstance(new_colors[0], str):
                    new_colors = cltmisc.multi_hex2rgb(new_colors)

                elif isinstance(new_colors[0], np.ndarray):
                    new_colors = np.array(new_colors)

                else:
                    raise ValueError("If new_colors is a list, it must be a list of hexadecimal colors or a list of rgb colors")
                
            elif isinstance(new_colors, np.ndarray):
                pass

            else:
                raise ValueError("The new_colors must be a list of colors or a numpy array")

            new_colors = cltmisc.readjust_colors(new_colors)

            if new_colors.shape[0] != n_groups:
                raise ValueError("The number of new colors must be equal to the number of groups that will be created")
        
        # Creating the grouped parcellation
        out_atlas = np.zeros_like(self.data, dtype='int16')
        
        for i in range(n_groups):
            indexes = cltmisc.get_indexes_by_substring(list1=self.name, 
                                    substr=names2group[i])
            code2look = np.array(indexes) + 1

            if new_codes is not None:
                out_atlas[np.isin(self.data, code2look)==True] = new_codes[i]
            else:
                out_atlas[np.isin(self.data, code2look)==True] = i + 1

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

    def rearange_parc(self, offset: int = 0):
        """
        Rearrange the parcellation starting from 1
        @params:
            offset     - Optional  : Offset to start the rearrangement. Default = 0
        """

        st_codes = np.unique(self.data)
        st_codes = st_codes[st_codes != 0]
        self.keep_by_code(codes2look=st_codes, rearrange=True)
        
        ind = np.where(self.data != 0)
        self.data[ind] = self.data[ind] + offset
        

        if offset != 0:
            self.index = [x + offset for x in self.index]

        self.parc_range()

    def add_parcellation(self,
                parc2add, 
                append: bool = False):
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
                            tmp_parc_obj.data[ind] = tmp_parc_obj.data[ind] + self.maxlab

                        if hasattr(parc, "index") and hasattr(parc, "name") and hasattr(parc, "color"):
                            if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
                                
                                if append:
                                    tmp_parc_obj.index = [x + self.maxlab for x in tmp_parc_obj.index]
                                
                                if isinstance(tmp_parc_obj.index, list) and isinstance(self.index, list):
                                    self.index = self.index + tmp_parc_obj.index
                                
                                elif isinstance(tmp_parc_obj.index, np.ndarray) and isinstance(self.index, np.ndarray):    
                                    self.index = np.concatenate((self.index, tmp_parc_obj.index), axis=0).tolist()
                                
                                elif isinstance(tmp_parc_obj.index, list) and isinstance(self.index, np.ndarray):
                                    self.index = tmp_parc_obj.index + self.index.tolist()
                                
                                elif isinstance(tmp_parc_obj.index, np.ndarray) and isinstance(self.index, list):
                                    self.index = self.index + tmp_parc_obj.index.tolist()
                                
                                self.name = self.name + tmp_parc_obj.name
                                
                                if isinstance(tmp_parc_obj.color, list) and isinstance(self.color, list):
                                    self.color = self.color + tmp_parc_obj.color
                                
                                elif isinstance(tmp_parc_obj.color, np.ndarray) and isinstance(self.color, np.ndarray):
                                    self.color = np.concatenate((self.color, tmp_parc_obj.color), axis=0)
                                    
                                elif isinstance(tmp_parc_obj.color, list) and isinstance(self.color, np.ndarray):
                                    temp_color = cltmisc.readjust_colors(self.color)
                                    temp_color = cltmisc.multi_rgb2hex(temp_color)
                                    
                                    self.color = temp_color + tmp_parc_obj.color
                                elif isinstance(tmp_parc_obj.color, np.ndarray) and isinstance(self.color, list):
                                    temp_color = cltmisc.readjust_colors(tmp_parc_obj.color)
                                    temp_color = cltmisc.multi_rgb2hex(temp_color)
                                    
                                    self.color = self.color + temp_color
                            
                            # If the parcellation self.data is all zeros  
                            elif np.sum(self.data) == 0:
                                self.index = tmp_parc_obj.index
                                self.name  = tmp_parc_obj.name
                                self.color = tmp_parc_obj.color  
                        
                        # Concatenating the parcellation data
                        self.data[ind] = tmp_parc_obj.data[ind]  
                                    
            else:
                raise ValueError("The list is empty")
        
        if hasattr(self, "color"):
            self.color = cltmisc.harmonize_colors(self.color)
        
        # Detect minimum and maximum labels
        self.parc_range()

    def save_parcellation(self,
                            out_file: str,
                            affine: np.float64 = None,
                            headerlines: Union[list, str] = None,
                            save_lut: bool = False,
                            save_tsv: bool = False):
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
        
        self.data = np.int32(self.data)

        out_atlas = nib.Nifti1Image(self.data, affine)
        nib.save(out_atlas, out_file)

        if save_lut:
            if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
                self.export_colortable(out_file=out_file.replace(".nii.gz", ".lut"), headerlines=headerlines)
            else:
                print("Warning: The parcellation does not contain a color table. The lut file will not be saved")
        
        if save_tsv:
            if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
                self.export_colortable(out_file=out_file.replace(".nii.gz", ".tsv"), lut_type="tsv")
            else:
                print("Warning: The parcellation does not contain a color table. The tsv file will not be saved")   
                
    def load_colortable(self, 
                    lut_file: Union[str, dict] = None, 
                    lut_type: str = "lut"):
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
                    raise ValueError("The dictionary must contain the keys 'index' and 'name'")
                
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
                raise ValueError("The dictionary must contain the keys 'index' and 'name'")
            
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
    
    def export_colortable(self, 
                            out_file: str, 
                            lut_type: str = "lut",
                            headerlines: Union[list, str] = None,
                            force: bool = True):
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
        
        if not hasattr(self, "index") or not hasattr(self, "name") or not hasattr(self, "color"):
            raise ValueError("The parcellation does not contain a color table. The index, name and color attributes must be present")
        
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

            now              = datetime.now()
            date_time        = now.strftime("%m/%d/%Y, %H:%M:%S")
            
            if headerlines is None:
                headerlines      = ['# $Id: {} {} \n'.format(out_file, date_time)]
                
                if os.path.isfile(self.parc_file):
                    headerlines.append('# Corresponding parcellation: {} \n'.format(self.parc_file))

                headerlines.append('{:<4} {:<50} {:>3} {:>3} {:>3} {:>3}'.format("#No.", "Label Name:", "R", "G", "B", "A"))

            self.write_luttable(
                self.index, self.name, self.color, out_file, headerlines=headerlines
            )
        elif lut_type == "tsv":
            
            if self.index is None or self.name is None:
                raise ValueError("The parcellation does not contain a color table. The index and name attributes must be present")
            
            tsv_df = pd.DataFrame(
                {"index": np.asarray(self.index), "name": self.name}
            )
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
            
            
            self.write_tsvtable(
                tsv_df, out_file, force = force
            )
        else:
            raise ValueError("The lut_type must be 'lut' or 'tsv'")
    
    def replace_values(self,
                        codes2rep: Union[list, np.ndarray],
                        new_codes: Union[list, np.ndarray]
                        ):
        """
        Replace groups of values of the image with the new codes.
        @params:
            codes2rep        - Required  : List, numpy array or list of list of codes to be replaced:
            new_codes        - Optional  : New codes:

        """
        
        # Correcting if new_codes is an integer
        if isinstance(new_codes, int):
            new_codes = [np.int32(new_codes)]

        # Detect thecodes2group is a list of list
        if isinstance(codes2rep, list):
            if isinstance(codes2rep[0], list):
                n_groups = len(codes2rep)
            
            elif isinstance(codes2rep[0], (str, np.integer, tuple)):
                codes2rep = [codes2rep]
                n_groups = 1
            
        elif isinstance(codes2rep, np.ndarray):
            codes2rep = codes2rep.tolist()
            n_groups = 1

        for i, v in enumerate(codes2rep):
            if isinstance(v, list):
                codes2rep[i] = cltmisc.build_indexes(v, nonzeros=False)
        
        # Convert the new_codes to a numpy array
        if isinstance(new_codes, list):
            new_codes = cltmisc.build_indexes(new_codes, nonzeros=False)
            new_codes = np.array(new_codes)
        elif isinstance(new_codes, np.integer):
            new_codes = np.array([new_codes])

        if len(new_codes) != n_groups:
            raise ValueError("The number of new codes must be equal to the number of groups of values that will be replaced")
        
        for ng in np.arange(n_groups):
            code2look = np.array(codes2rep[ng])
            mask = np.isin(self.data, code2look)
            self.data[mask] = new_codes[ng]
        
        if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
            self.adjust_values()
            
        self.parc_range()
        
        
    def parc_range(self):
        """
        Detect the range of labels

        """
        # Detecting the unique elements in the parcellation different from zero
        st_codes = np.unique(self.data)
        st_codes = st_codes[st_codes != 0]
        if np.size(st_codes) > 0:
            self.minlab = np.min(st_codes)
            self.maxlab = np.max(st_codes)
        else:
            self.minlab = 0
            self.maxlab = 0

    @staticmethod
    def write_fslcolortable(lut_file_fs: str, 
                                    lut_file_fsl: str):
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
    def read_luttable(in_file: str):
        """
        Reading freesurfer lut file
        @params:
            in_file     - Required  : FreeSurfer color lut:
        
        Returns
        -------
        st_codes: list
            List of codes for the parcellation
        st_names: list
            List of names for the parcellation
        st_colors: list
            List of colors for the parcellation
        
        """

        # Readind a color LUT file
        fid = open(in_file)
        LUT = fid.readlines()
        fid.close()

        # Make dictionary of labels
        LUT = [row.split() for row in LUT]
        st_names = []
        st_codes = []
        cont = 0
        for row in LUT:
            if (
                len(row) > 1 and row[0][0] != "#" and row[0][0] != "\\\\"
            ):  # Get rid of the comments
                st_codes.append(int(row[0]))
                st_names.append(row[1])
                if cont == 0:
                    st_colors = np.array([[int(row[2]), int(row[3]), int(row[4])]])
                else:
                    ctemp = np.array([[int(row[2]), int(row[3]), int(row[4])]])
                    st_colors = np.append(st_colors, ctemp, axis=0)
                cont = cont + 1
        
        # Convert the elements to integer 32 bits
        st_codes = [np.int32(x) for x in st_codes]
        
        # Converting colors to hexadecimal format
        st_colors = cltmisc.multi_rgb2hex(st_colors)
        
        # Create the dictionary
        lut_dict = {"index": st_codes, "name": st_names, "color": st_colors}

        return lut_dict

    @staticmethod
    def read_tsvtable(in_file: str):
        """
        Reading tsv table
        @params:
            in_file     - Required  : TSV file:
            cl_format   - Optional  : Color format: 'rgb' or 'hex'. Default = 'rgb'
        
        Returns
        -------
        tsv_dict: dict
            Dictionary with the tsv table
            
        
        """

        # Read the tsv file
        if not os.path.exists(in_file):
            raise ValueError("The file does not exist")
        
        tsv_df = pd.read_csv(in_file, sep="\t")
        
        # Convert to dictionary
        tsv_dict = tsv_df.to_dict(orient="list")
        
        if "index" in tsv_dict.keys():
            # Convert the elements to integer 32 bits
            tsv_dict["index"] = [np.int32(x) for x in tsv_dict["index"]]
            
        # Test if index and name are keys
        if "index" not in tsv_dict.keys() or "name" not in tsv_dict.keys():
            raise ValueError("The tsv file must contain the columns 'index' and 'name'")
        
        if "color" in tsv_dict.keys():
            temp_colors = tsv_dict["color"]

        return tsv_dict
    
    @staticmethod
    def write_luttable(codes:list, 
                        names:list, 
                        colors:Union[list, np.ndarray],
                        out_file:str = None, 
                        headerlines: Union[list, str] = None,
                        boolappend: bool = False,
                        force: bool = True):
        
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
        
        happend_bool = True # Boolean to append the headerlines
        if headerlines is None:
            happend_bool = False # Only add this if it is the first time the file is created
            now              = datetime.now()
            date_time        = now.strftime("%m/%d/%Y, %H:%M:%S")
            headerlines      = ['# $Id: {} {} \n'.format(out_file, date_time),
                                '{:<4} {:<50} {:>3} {:>3} {:>3} {:>3}'.format("#No.", "Label Name:", "R", "G", "B", "A")] 
        
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

                luttable = [l.strip('\n\r') for l in luttable]
                luttable = ["\n" if element == "" else element for element in luttable]


                if happend_bool:
                    luttable  = luttable + headerlines
                
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
                luttable.append('\n')

            luttable.append('{:<4} {:<50} {:>3} {:>3} {:>3} {:>3}'.format(codes[roi_pos], 
                                                                        names[roi_pos], 
                                                                        colors[roi_pos,0], 
                                                                        colors[roi_pos,1], 
                                                                        colors[roi_pos,2], 0))
        luttable.append('\n')
        
        if out_file is not None:
            if os.path.isfile(out_file) and force:
                # Save the lut table
                with open(out_file, 'w') as colorLUT_f:
                    colorLUT_f.write('\n'.join(luttable))
            elif not os.path.isfile(out_file):
                # Save the lut table
                with open(out_file, 'w') as colorLUT_f:
                    colorLUT_f.write('\n'.join(luttable))

        return luttable

    @staticmethod
    def write_tsvtable(tsv_df: Union[pd.DataFrame, dict],
                        out_file:str,
                        boolappend: bool = False,
                        force: bool = False):
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
                            tsv_orig[key] = tsv_orig[key] + [""]*len(tsv_dict["name"])
                        elif key in tsv_dict.keys():
                            tsv_orig[key] =  [""]*len(tsv_orig["name"]) + tsv_dict[key]
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
        attributes_and_methods = [attr for attr in dir(self) if not callable(getattr(self, attr))]
        methods = [method for method in dir(self) if callable(getattr(self, method))]

        print("Attributes:")
        for attribute in attributes_and_methods:
            if not attribute.startswith("__"):
                print(attribute)

        print("\nMethods:")
        for method in methods:
            if not method.startswith("__"):
                print(method)
