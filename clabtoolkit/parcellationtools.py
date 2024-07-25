import os
from datetime import datetime

import numpy as np
import pandas as pd
import nibabel as nib
from typing import Union
import clabtoolkit.misctools as cltmisc

class Parcellation:

    def __init__(self, 
                    parc_file: Union[str, np.uint] = None, 
                    affine:np.float_ = None):
        
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
                            self._load_colortable(lut_file=tsv_file, lut_type='tsv')
                            
                        elif not os.path.isfile(tsv_file) and os.path.isfile(lut_file):
                            self._load_colortable(lut_file=lut_file, lut_type='lut')
                            
                    elif parc_file.endswith('.nii'):
                        tsv_file = parc_file.replace('.nii', '.tsv')
                        lut_file = parc_file.replace('.nii', '.lut')
                    
                        if os.path.isfile(tsv_file):
                            self._load_colortable(lut_file=tsv_file, lut_type='tsv')
                            
                        elif not os.path.isfile(tsv_file) and os.path.isfile(lut_file):
                            self._load_colortable(lut_file=lut_file, lut_type='lut')
                
            elif isinstance(parc_file, np.ndarray):
                self.data = parc_file
                self.affine = affine

            # Adjust values to the ones present in the parcellation
            
            if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
                self._adjust_values()
            
            # Detect minimum and maximum labels
            self._parc_range()


    def _keep_by_name(self, 
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
            names2look = np.array(names2look)

        if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
            # Find the indexes of the names that contain the substring
            indexes = cltmisc._get_indexes_by_substring(list1=self.name, 
                                substr=names2look, 
                                invert=False, 
                                boolcase=False)
            
            if len(indexes) > 0:
                self._keep_by_code(codes2look=self.index[indexes], rearrange=rearrange)
            else:
                print("The names were not found in the parcellation")


    def _keep_by_code(self, 
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
            codes2look = cltmisc._build_indexes(codes2look)
            codes2look = np.array(codes2look)

        # Create 
        dims = np.shape(self.data)
        out_atlas = np.zeros((dims[0], dims[1], dims[2]), dtype='int16') 

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
        self._parc_range()

    
    def _remove_by_code(self,
                            codes2remove: Union[list, np.ndarray],
                            rearrange: bool = False):
        """
        Remove the structures with the codes specified in the list.
        @params:
            codes2remove     - Required  : List of codes to remove:
            rearrange        - Required  : If True, the parcellation will be rearranged starting from 1. Default = False
        """

        if isinstance(codes2remove, list):
            codes2look = cltmisc._build_indexes(codes2look)
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
            self._keep_by_code(codes2look=st_codes, rearrange=True)
        else:
            self._keep_by_code(codes2look=st_codes, rearrange=False)

        # Detect minimum and maximum labels
        self._parc_range()
    
    def _apply_mask(self, image_mask,
                        codes2mask: Union[list, np.ndarray] = None,
                        mask_type: str = 'upright'
                        ):
        """
        Mask the structures with the codes specified in the list or array codes2mask.
        @params:
            image_mask     - Required  : Image mask:
            codes2mask     - Optional  : List of codes to mask:
            mask_type      - Optional  : Mask type: 'upright' or 'inverted'. Default = upright
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
            codes2mask = cltmisc._build_indexes(codes2mask)
            codes2mask = np.array(codes2mask)
        
        if mask_type == 'inverted':
            self.data[np.isin(mask_data, codes2mask)==True] = 0

        else:
            self.data[np.isin(mask_data, codes2mask)==False] = 0
        
        if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
            self._adjust_values()
        
        # Detect minimum and maximum labels
        self._parc_range()
    
    def _adjust_values(self):
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
        
        self._parc_range()
        
    def _group_by_code(self,
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

        # Detect thecodes2group is a list of list
        if isinstance(codes2group, list):
            if isinstance(codes2group[0], list):
                n_groups = len(codes2group)
            
            elif isinstance(codes2group[0], (str, np.integer, tuple)):
                codes2group = [codes2group]
                n_groups = 1
            
        elif isinstance(codes2group, np.ndarray):
            codes2group = codes2group.tolist()
            n_groups = 1

        for i, v in enumerate(codes2group):
            if isinstance(v, list):
                codes2group[i] = cltmisc._build_indexes(v)
        
        # Convert the new_codes to a numpy array
        if new_codes is not None:
            if isinstance(new_codes, list):
                new_codes = cltmisc._build_indexes(new_codes)
                new_codes = np.array(new_codes)
        elif isinstance(new_codes, np.integer):
            new_codes = np.array([new_codes])

        elif new_codes is None:
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
                    new_colors = cltmisc._multi_hex2rgb(new_colors)

                elif isinstance(new_colors[0], np.ndarray):
                    new_colors = np.array(new_colors)

                else:
                    raise ValueError("If new_colors is a list, it must be a list of hexadecimal colors or a list of rgb colors")
                
            elif isinstance(new_colors, np.ndarray):
                pass

            else:
                raise ValueError("The new_colors must be a list of colors or a numpy array")

            new_colors = cltmisc._readjust_colors(new_colors)

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
            self.color = cltmisc._create_random_colors(n_groups)

            
        # Detect minimum and maximum labels
        self._parc_range()

    def _rearange_parc(self, offset: int = 0):
        """
        Rearrange the parcellation starting from 1
        @params:
            offset     - Optional  : Offset to start the rearrangement. Default = 0
        """

        st_codes = np.unique(self.data)
        st_codes = st_codes[st_codes != 0]
        self._keep_by_code(codes2look=st_codes, rearrange=True)

        if offset != 0:
            self.index = [x + offset for x in self.index]

        self._parc_range()

    def _add_parcellation(self,
                parc2add, 
                append: bool = False):
        """
        Add a parcellation 
        @params:
            parc2add     - Required  : Parcellation to add:
            append       - Optional  : If True, the parcellation will be appended. The labels will be 
                                    added to the maximum label of the current parcellation. Default = False
        """
        if isinstance(parc2add, Parcellation):
            parc2add = [parc2add]

        if isinstance(parc2add, list):
            if len(parc2add) > 0:
                for parc in parc2add:
                    if isinstance(parc, Parcellation):
                        ind = np.where(parc.data != 0)
                        if append:
                            parc.data[ind] = parc.data[ind] + self.maxlab

                        if hasattr(parc, "index") and hasattr(parc, "name") and hasattr(parc, "color"):
                            if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
                                
                                if append:
                                    parc.index = [x + self.maxlab for x in parc.index]
                                
                                if isinstance(parc.index, list) and isinstance(self.index, list):
                                    self.index = self.index + parc.index
                                
                                elif isinstance(parc.index, np.ndarray) and isinstance(self.index, np.ndarray):    
                                    self.index = np.concatenate((self.index, parc.index), axis=0).tolist()
                                
                                elif isinstance(parc.index, list) and isinstance(self.index, np.ndarray):
                                    self.index = parc.index + self.index.tolist()
                                
                                elif isinstance(parc.index, np.ndarray) and isinstance(self.index, list):
                                    self.index = self.index + parc.index.tolist()
                                
                                self.name = self.name + parc.name
                                
                                if isinstance(parc.color, list) and isinstance(self.color, list):
                                    self.color = self.color + parc.color
                                
                                elif isinstance(parc.color, np.ndarray) and isinstance(self.color, np.ndarray):
                                    self.color = np.concatenate((self.color, parc.color), axis=0)
                                    
                                elif isinstance(parc.color, list) and isinstance(self.color, np.ndarray):
                                    temp_color = cltmisc._readjust_colors(self.color)
                                    temp_color = cltmisc._multi_rgb2hex(temp_color)
                                    
                                    self.color = temp_color + parc.color
                                elif isinstance(parc.color, np.ndarray) and isinstance(self.color, list):
                                    temp_color = cltmisc._readjust_colors(parc.color)
                                    temp_color = cltmisc._multi_rgb2hex(temp_color)
                                    
                                    self.color = self.color + temp_color
                            
                            # If the parcellation self.data is all zeros  
                            elif np.sum(self.data) == 0:
                                self.index = parc.index
                                self.name  = parc.name
                                self.color = parc.color  
                        
                        # Concatenating the parcellation data
                        self.data[ind] = parc.data[ind]  
                                    
            else:
                raise ValueError("The list is empty")
        
        # Detect minimum and maximum labels
        self._parc_range()

    def _save_parcellation(self,
                            out_file: str,
                            affine: np.float_ = None, 
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
            
        self.data = np.int32(self.data)

        out_atlas = nib.Nifti1Image(self.data, affine)
        nib.save(out_atlas, out_file)

        if save_lut:
            if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
                self._export_colortable(out_file=out_file.replace(".nii.gz", ".lut"))
            else:
                print("Warning: The parcellation does not contain a color table. The lut file will not be saved")
        
        if save_tsv:
            if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
                self._export_colortable(out_file=out_file.replace(".nii.gz", ".tsv"), lut_type="tsv")
            else:
                print("Warning: The parcellation does not contain a color table. The tsv file will not be saved")   
                
    def _load_colortable(self, 
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
                    st_codes, st_names, st_colors = self.read_luttable(in_file=lut_file)

                elif lut_type == "tsv":
                    
                    tsv_dict = self.read_tsvtable(in_file=lut_file)
                    if "index" in tsv_dict.keys() and "name" in tsv_dict.keys():
                        st_codes = tsv_dict["index"]
                        st_names = tsv_dict["name"]
                    else: 
                        raise ValueError("The dictionary must contain the keys 'index' and 'name'")
                    
                    if "color" in tsv_dict.keys():
                        st_colors = tsv_dict["color"]
                        
                        if isinstance(st_colors[0], str):
                            st_colors = cltmisc._multi_hex2rgb(st_colors)

                        elif isinstance(st_colors[0], list):
                            st_colors = np.array(st_colors)
                    else:
                        st_colors = None

                else:
                    raise ValueError("The lut_type must be 'lut' or 'tsv'")
                    
                self.index = st_codes
                self.name = st_names
                self.color = st_colors

            else:
                raise ValueError("The lut file does not exist")

        elif isinstance(lut_file, dict):
            self.lut_file = None

            if "index" not in lut_file.keys() or "name" not in lut_file.keys():
                raise ValueError("The dictionary must contain the keys 'index' and 'name'")
            
            if "color" not in lut_file.keys():
                st_colors = None
            else:
                
                st_colors = lut_file["color"]
                if isinstance(st_colors[0], str):
                    st_colors = cltmisc._multi_hex2rgb(st_colors)

                elif isinstance(st_colors[0], list):
                    st_colors = np.array(st_colors)

            self.index = lut_file["index"]
            self.color = st_colors
            self.name = lut_file["name"]
    
    def _export_colortable(self, 
                            out_file: str, 
                            lut_type: str = "lut",
                            force: bool = True):
        """
        Export the lookup table to a file
        @params:
            out_file     - Required  : Lookup table file:
            lut_type     - Optional  : Type of the lut file: 'lut' or 'tsv'. Default = 'lut'
            force        - Optional  : If True, it will overwrite the file. Default = True
        """

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
                        tsv_df["color"] = cltmisc._multi_rgb2hex(self.color)
                        
                elif isinstance(self.color, np.ndarray):
                    tsv_df["color"] = cltmisc._multi_rgb2hex(self.color)
            
            
            self.write_tsvtable(
                tsv_df, out_file, force = force
            )
        else:
            raise ValueError("The lut_type must be 'lut' or 'tsv'")
    
    def _replace_values(self,
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
                codes2rep[i] = cltmisc._build_indexes(v, nonzeros=False)
        
        # Convert the new_codes to a numpy array
        if isinstance(new_codes, list):
            new_codes = cltmisc._build_indexes(new_codes, nonzeros=False)
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
            self._adjust_values()
            
        self._parc_range()
        
        
    def _parc_range(self):
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
        st_codes_lut, st_names_lut, st_colors_lut = Parcellation.read_luttable(lut_file_fs)
        
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

        return st_codes, st_names, st_colors

    @staticmethod
    def read_tsvtable(in_file: str, 
                        cl_format: str = "rgb"):
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

            if cl_format == "rgb":
                st_colors = cltmisc._multi_hex2rgb(temp_colors)
            elif cl_format == "hex":
                st_colors = temp_colors
                
            tsv_dict["color"] = st_colors

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
                colors = cltmisc._multi_hex2rgb(colors)
            elif isinstance(colors[0], list):
                colors = np.array(colors)
        
        # Table for parcellation      
        for roi_pos, roi_name in enumerate(names):
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
                    
            elif os.path.isfile(out_file) and not force:
                raise ValueError("The file already exists. Please use the 'force' flag to overwrite the LUT file")
            
                

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
                    seg_hexcol = cltmisc._multi_rgb2hex(colors)
                    tsv_dict["color"] = seg_hexcol  
                    
            elif isinstance(temp_colors, np.ndarray):
                seg_hexcol = cltmisc._multi_rgb2hex(temp_colors)
                tsv_dict["color"] = seg_hexcol 
                
        
        if boolappend:
            if not os.path.exists(out_file):
                raise ValueError("The file does not exist")
            else:
                tsv_orig = Parcellation.read_tsvtable(in_file=out_file, cl_format="hex")
                
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
                
        elif os.path.isfile(out_file) and not force:
            raise ValueError("The file already exists. Please use the 'force' flag to overwrite the TSV file")

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
                cltmisc._rgb2hex(seg_rgbcol[i, 0], seg_rgbcol[i, 1], seg_rgbcol[i, 2])
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
    

    def _print_properties(self):
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
