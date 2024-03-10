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

            elif isinstance(parc_file, np.ndarray):
                self.data = parc_file
                self.affine = affine

        # Detect minimum and maximum labels
        self._parc_range()

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

        if hasattr(self, "index"):
            temp_index = np.array(self.index) 
            index_new = []
            indexes = []

        cont = 0

        for i, v in enumerate(codes2look):
            
            # Find the elements in the data that are equal to v
            result = np.where(self.data == v)

            if len(result[0]) > 0:
                cont = cont + 1

                if hasattr(self, "index"):
                    # Find the element in self.index that is equal to v
                    ind = np.where(temp_index == v)[0]

                    if len(ind) > 0:
                        indexes.append(ind[0])
                        if rearrange:
                            index_new.append(cont)
                        else:
                            index_new.append(self.index[ind[0]])

                if rearrange:
                    out_atlas[result[0], result[1], result[2]] = cont

                else:
                    out_atlas[result[0], result[1], result[2]] = v

        
        # Find the indexes of the elements in a that are also in b
        # If index is an attribute of self
        if hasattr(self, "index"):                       
            self.index = index_new

        # If name is an attribute of self
        if hasattr(self, "name"):
            self.name = [self.name[i] for i in indexes]

        # If color is an attribute of self
        if hasattr(self, "color"):
            self.color = self.color[indexes]

        self.data = out_atlas

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
    
    def _mask_by_code(self,
                        codes2mask: Union[list, np.ndarray],
                        mask_type: str = 'upright'
                        ):
        """
        Mask the structures with the codes specified in the list or array codes2mask.
        @params:
            codes2mask     - Required  : List of codes to mask:
            mask_type      - Optional  : Mask type: 'upright' or 'inverted'. Default = upright
        """
        mask_type.lower()
        if mask_type not in ['upright', 'inverted']:
            raise ValueError("The mask_type must be 'upright' or 'inverted'")
        
        if isinstance(codes2mask, list):
            codes2look = cltmisc._build_indexes(codes2look)
            codes2mask = np.array(codes2mask)
        
        if mask_type == 'inverted':
            self.data[np.isin(self.data, codes2mask)==True] = 0

        else:
            self.data[np.isin(self.data, codes2mask)==False] = 0
        
        # Detect minimum and maximum labels
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
            
            elif isinstance(codes2group[0], str) or isinstance(codes2group[0], int) or isinstance(codes2group[0], tuple):
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
        elif isinstance(new_codes, int):
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
                parc2add):
        """
        Add a parcellation 
        @params:
            parc2add     - Required  : Parcellation to add:
        """
        if isinstance(parc2add, Parcellation):
            parc2add = [parc2add]

        if isinstance(parc2add, list):
            if len(parc2add) > 0:
                for parc in parc2add:
                    if isinstance(parc, Parcellation):
                        ind = np.where(parc.data != 0)
                        parc.data[ind] = parc.data[ind] + self.maxlab
                        self.data[ind] = self.data[ind] +  parc.data[ind]
                        
                        if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
                            parc.index = [x + self.maxlab for x in parc.index]

                            self.index = self.index + parc.index
                            self.name = self.name + parc.name
                            self.color = np.concatenate((self.color, parc.color), axis=0)
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

        out_atlas = nib.Nifti1Image(self.data, affine)
        nib.save(out_atlas, out_file)

        if save_lut:
            if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
                self._export_colortable(out_file=out_file.replace(".nii.gz", ".lut"))
            else:
                raise ValueError("The parcellation does not contain a color table")
        
        if save_tsv:
            if hasattr(self, "index") and hasattr(self, "name") and hasattr(self, "color"):
                self._export_colortable(out_file=out_file.replace(".nii.gz", ".tsv"), lut_type="tsv")
            else:
                raise ValueError("The parcellation does not contain a color table")
            
    def _load_colortable(self, 
                    lut_file: Union[str, dict], 
                    lut_type: str = "lut"):
        """
        Add a lookup table to the parcellation
        @params:
            lut_file     - Required  : Lookup table file. It can be a string with the path to the 
                                        file or a dictionary containing the keys 'index', 'color' and 'name':
            lut_type     - Optional  : Type of the lut file: 'lut' or 'tsv'. Default = 'lut'
        """

        if isinstance(lut_file, str):
            if os.path.exists(lut_file):
                self.lut_file = lut_file

                if lut_type == "lut":
                    st_codes, st_names, st_colors = self.read_luttable(in_file=lut_file)

                elif lut_type == "tsv":
                    st_codes, st_names, st_colors = self.read_tsvtable(in_file=lut_file)
                    
                else:
                    raise ValueError("The lut_type must be 'lut' or 'tsv'")
                    
                self.index = st_codes
                self.name = st_names
                self.color = st_colors

            else:
                raise ValueError("The lut file does not exist")

        elif isinstance(lut_file, dict):
            self.lut_file = None

            if "index" not in lut_file.keys() or "color" not in lut_file.keys() or "name" not in lut_file.keys():
                raise ValueError("The dictionary must contain the keys 'index', 'color' and 'name'")

            colors = lut_file["color"]
            if isinstance(colors[0], str):
                colors = cltmisc._multi_hex2rgb(colors)

            elif isinstance(colors[0], list):
                colors = np.array(colors)

            self.index = lut_file["index"]
            self.color = colors
            self.name = lut_file["name"]
    
    def _export_colortable(self, 
                            out_file: str, 
                            lut_type: str = "lut"):
        """
        Export the lookup table to a file
        @params:
            out_file     - Required  : Lookup table file:
            lut_type     - Optional  : Type of the lut file: 'lut' or 'tsv'. Default = 'lut'
        """

        if not hasattr(self, "index") or not hasattr(self, "name") or not hasattr(self, "color"):
            raise ValueError("The parcellation does not contain a color table. The index, name and color attributes must be present")
        

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
            self.write_tsvtable(
                self.index, self.name, self.color, out_file
            )
        else:
            raise ValueError("The lut_type must be 'lut' or 'tsv'")
        
    def _parc_range(self):
        """
        Detect the range of labels

        """
        # Detecting the unique elements in the parcellation different from zero
        st_codes = np.unique(self.data)
        st_codes = st_codes[st_codes != 0]
        self.minlab = np.min(st_codes)
        self.maxlab = np.max(st_codes)

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
        st_codes: list
            List of codes for the parcellation
        st_names: list
            List of names for the parcellation
        st_colors: list
            List of colors for the parcellation
        
        """

        # Read the tsv file
        if not os.path.exists(in_file):
            raise ValueError("The file does not exist")
        
        tsv_df = pd.read_csv(in_file, sep="\t")

        st_codes = tsv_df["index"].values
        st_names = tsv_df["name"].values
        temp_colors = tsv_df["color"].values.tolist()

        if cl_format == "rgb":
            st_colors = cltmisc._multi_hex2rgb(temp_colors)
        elif cl_format == "hex":
            st_colors = temp_colors

        return st_codes, st_names, st_colors
    
    @staticmethod
    def write_luttable(codes:list, 
                        names:list, 
                        colors:Union[list, np.ndarray],
                        out_file:str, 
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
        if os.path.exists(out_file) and not force:
            raise ValueError("The file already exists")
        
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
        
        # Table for parcellation      
        for roi_pos, roi_name in enumerate(names):
            luttable.append('{:<4} {:<50} {:>3} {:>3} {:>3} {:>3}'.format(codes[roi_pos], 
                                                                        names[roi_pos], 
                                                                        colors[roi_pos,0], 
                                                                        colors[roi_pos,1], 
                                                                        colors[roi_pos,2], 0))
        luttable.append('\n')

        # Save the lut table
        with open(out_file, 'w') as colorLUT_f:
            colorLUT_f.write('\n'.join(luttable))

        return out_file

    @staticmethod
    def write_tsvtable(codes:list, 
                        names:list, 
                        colors:Union[list, np.ndarray],
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
            raise ValueError("The TSV file already exists")
        
        out_dir = os.path.dirname(out_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        # Table for parcellation
        # 1. Converting colors to hexidecimal string
        seg_hexcol = cltmisc._multi_rgb2hex(colors)
        
        if boolappend:
            if not os.path.exists(out_file):
                raise ValueError("The file does not exist")
            else:
                tsv_df = pd.read_csv(out_file, sep="\t")
                tsv_df = tsv_df.append(
                    {"index": np.asarray(codes), "name": names, "color": seg_hexcol}
                )
        else:
            tsv_df = pd.DataFrame(
                {"index": np.asarray(codes), "name": names, "color": seg_hexcol}
        )
        # Save the tsv table
        with open(out_file, "w+") as tsv_file:
            tsv_file.write(tsv_df.to_csv(sep="\t", index=False))

        return tsv_file
    
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
