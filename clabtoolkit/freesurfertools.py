import os
import time
import subprocess
import sys
from glob import glob
from typing import Union
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import clabtoolkit.misctools as cltmisc
import clabtoolkit.parcellationtools as cltparc


class AnnotParcellation:
    """
    This class contains methods to work with FreeSurfer annot files

    # Implemented methods:
    # - Correct the parcellation by refilling the vertices from the cortex label file that do not have a label in the annotation file
    # - Convert FreeSurfer annot files to gcs files
    
    # Methods to be implemented:
    # Grouping regions to create a coarser parcellation
    # Removing regions from the parcellation
    # Correct parcellations by removing small clusters of vertices labeled inside another region
    
    """

    def __init__(self, parc_file: str,
                    ref_surf: str = None, 
                    cont_tech: str = "local", 
                    cont_image: str = "local"):
        """
        Initialize the AnnotParcellation object
        
        Parameters
        ----------
        parc_file     - Required  : Parcellation filename:
        ref_surf      - Optional  : Reference surface. Default is the white surface of the fsaverage subject:   
        cont_tech     - Optional  : Container technology. Default is local:
        cont_image    - Optional  : Container image. Default is local:
        
        """
        booldel = False
        self.filename = parc_file
        
        # Verify if the file exists
        if not os.path.exists(self.filename):
            raise ValueError("The parcellation file does not exist")
        
        # Extracting the filename, folder and name
        self.path = os.path.dirname(self.filename)
        self.name = os.path.basename(self.filename)

        # Detecting the hemisphere
        temp_name = self.name.lower()
        
        # Find in the string annot_name if it is lh. or rh.
        hemi = _detect_hemi(self.name)

        self.hemi = hemi

        # If the file is a .gii file, then convert it to a .annot file
        if self.name.endswith(".gii"):
            
            annot_file = AnnotParcellation.gii2annot(self.filename, 
                                                        ref_surf=ref_surf,  annot_file=self.filename.replace(".gii", ".annot"), cont_tech=cont_tech, cont_image= cont_image)
            booldel = True
            
        elif self.name.endswith(".annot"):
            annot_file = self.filename
            
        elif self.name.endswith(".gcs"):
            annot_file = AnnotParcellation.gcs2annot(self.filename, annot_file=self.filename.replace(".gcs", ".annot"))
            booldel = True
        
        # Read the annot file using nibabel
        codes, reg_table, reg_names = nib.freesurfer.io.read_annot(annot_file)
        
        if booldel:
            os.remove(annot_file)
        
        # Correcting region names
        reg_names = [name.decode("utf-8") for name in reg_names]

        # Storing the codes, colors and names in the object
        self.codes = codes
        self.regtable = reg_table
        self.regnames = reg_names

    
    def _save_annotation(self, out_file: str = None):
        """
        Save the annotation file
        @params:
            out_file     - Required  : Output annotation file:
        """

        if out_file is None:
            out_file = os.path.join(self.path, self.name)

        # If the directory does not exist then create it
        temp_dir = os.path.dirname(out_file)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Save the annotation file
        nib.freesurfer.io.write_annot(
            out_file, self.codes, self.regtable, self.regnames
        )

    def _fill_parcellation(
        self, 
        label_file: str, 
        surf_file: str, 
        corr_annot: str = None
    ):
        """
        Correct the parcellation by refilling the vertices from the cortex label file that do not have a label in the annotation file.
        @params:
            label_file     - Required  : Label file:
            surf_file      - Required  : Surface file:
            corr_annot     - Optional  : Corrected annotation file. If not provided, it will be saved with the same filename as the original annotation file:

        Returns
        -------
        corr_annot: str
            Corrected annotation file

        """

        # Auxiliary variables for the progress bar
        # LINE_UP = '\033[1A'
        # LINE_CLEAR = '\x1b[2K'

        # Get the vertices from the cortex label file that do not have a label in the annotation file

        # If the surface file does not exist, raise an error, otherwise load the surface
        if os.path.isfile(surf_file):
            vertices, faces = nib.freesurfer.read_geometry(surf_file)
        else:
            raise ValueError(
                "Surface file not found. Annotation, surface and cortex label files are mandatory to correct the parcellation."
            )

        # If the cortex label file does not exist, raise an error, otherwise load the cortex label
        if os.path.isfile(label_file):
            cortex_label = nib.freesurfer.read_label(label_file)
        else:
            raise ValueError(
                "Cortex label file not found. Annotation, surface and cortex label files are mandatory to correct the parcellation."
            )

        vert_lab = self.codes
        vert_lab[vert_lab == -1] = 0

        reg_ctable = self.regtable
        reg_names = self.regnames

        ctx_lab = vert_lab[cortex_label].astype(
            int
        )  # Vertices from the cortex label file that have a label in the annotation file

        bool_bound = vert_lab[faces] != 0

        # Boolean variable to check the faces that contain at least two vertices that are different from 0 and at least one vertex that is not 0 (Faces containing the boundary of the parcellation)
        bool_a = np.sum(bool_bound, axis=1) < 3
        bool_b = np.sum(bool_bound, axis=1) > 0
        bool_bound = bool_a & bool_b

        faces_bound = faces[bool_bound, :]
        bound_vert = np.ndarray.flatten(faces_bound)

        vert_lab_bound = vert_lab[bound_vert]

        # Delete from the array bound_vert the vertices that contain the vert_lab_bound different from 0
        bound_vert = np.delete(bound_vert, np.where(vert_lab_bound != 0)[0])
        bound_vert = np.unique(bound_vert)

        # Detect which vertices from bound_vert are in the  cortex_label array
        bound_vert = bound_vert[np.isin(bound_vert, cortex_label)]

        bound_vert_orig = np.zeros(len(bound_vert))
        # Create a while loop to fill the vertices that are in the boundary of the parcellation
        # The loop will end when the array bound_vert is empty or when bound_vert is equal bound_vert_orig

        # Detect if the array bound_vert is equal to bound_vert_orig
        bound = np.array_equal(bound_vert, bound_vert_orig)
        it_count = 0
        while len(bound_vert) > 0:

            if not bound:
                # it_count = it_count + 1
                # cad2print = "Interation number: {} - Vertices to fill: {}".format(
                #     it_count, len(bound_vert))
                # print(cad2print)
                # time.sleep(.5)
                # print(LINE_UP, end=LINE_CLEAR)

                bound_vert_orig = np.copy(bound_vert)
                temp_Tri = np.zeros((len(bound_vert), 100))
                for pos, i in enumerate(bound_vert):
                    # Get the neighbors of the vertex
                    neighbors = np.unique(faces[np.where(faces == i)[0], :])
                    neighbors = np.delete(neighbors, np.where(neighbors == i)[0])
                    temp_Tri[pos, 0 : len(neighbors)] = neighbors
                temp_Tri = temp_Tri.astype(int)
                index_zero = np.where(temp_Tri == 0)
                labels_Tri = vert_lab[temp_Tri]
                labels_Tri[index_zero] = 0

                for pos, i in enumerate(bound_vert):

                    # Get the labels of the neighbors
                    labels = labels_Tri[pos, :]
                    # Get the most frequent label different from 0
                    most_frequent_label = np.bincount(labels[labels != 0]).argmax()

                    # Assign the most frequent label to the vertex
                    vert_lab[i] = most_frequent_label

                ctx_lab = vert_lab[cortex_label].astype(
                    int
                )  # Vertices from the cortex label file that have a label in the annotation file

                bool_bound = vert_lab[faces] != 0

                # Boolean variable to check the faces that contain at least one vertex that is 0 and at least one vertex that is not 0 (Faces containing the boundary of the parcellation)
                bool_a = np.sum(bool_bound, axis=1) < 3
                bool_b = np.sum(bool_bound, axis=1) > 0
                bool_bound = bool_a & bool_b

                faces_bound = faces[bool_bound, :]
                bound_vert = np.ndarray.flatten(faces_bound)

                vert_lab_bound = vert_lab[bound_vert]

                # Delete from the array bound_vert the vertices that contain the vert_lab_bound different from 0
                bound_vert = np.delete(bound_vert, np.where(vert_lab_bound != 0)[0])
                bound_vert = np.unique(bound_vert)

                # Detect which vertices from bound_vert are in the  cortex_label array
                bound_vert = bound_vert[np.isin(bound_vert, cortex_label)]

                bound = np.array_equal(bound_vert, bound_vert_orig)

        # Save the annotation file
        if corr_annot is not None:
            if os.path.isfile(corr_annot):
                os.remove(corr_annot)

            # Create folder if it does not exist
            os.makedirs(os.path.dirname(corr_annot), exist_ok=True)
            nib.freesurfer.write_annot(corr_annot, vert_lab, reg_ctable, reg_names)
        else:
            nib.freesurfer.write_annot(self.filename, vert_lab, reg_ctable, reg_names)
            corr_annot = self.filename

        return corr_annot, vert_lab, reg_ctable, reg_names
    
    def _export_to_tsv(self, 
                        prefix2add: str = None,
                        reg_offset: int = 1000,
                        tsv_file: str = None):
        
        """
        Export the table of the parcellation to a tsv file. It will contain the index, the annotation id, 
        the parcellation id, the name and the color of the regions. 
        If a prefix is provided, it will be added to the names of the regions.
        If a tsv file is provided, it will be saved in the specified path. 
        Otherwise it will only return the pandas dataframe.
        
        Parameters
        ----------
        prefix2add     - Optional  : Prefix to add to the names of the regions:
        reg_offset     - Optional  : Offset to add to the parcellation id. Default is 1000:
        tsv_file       - Optional  : Output tsv file:
        
        Returns
        -------
        tsv_df: pandas dataframe : Table of the parcellation
        tsv_file: str : Tsv filename
        
        """
    
        # Creating the hexadecimal colors for the regions
        parc_hexcolor = cltmisc._multi_rgb2hex(self.regtable[:, 0:3])

        # Creating the region names
        parc_names = self.regnames
        if prefix2add is not None:
            parc_names = cltmisc._correct_names(parc_names, prefix=prefix2add)
        
        parc_index = np.arange(0, len(parc_names))
        
        # Selecting the Id in the annotation file
        annot_id = self.regtable[:, 4]
        
        
        parc_id = reg_offset + parc_index
            
        
        # Creating the dictionary for the tsv files
        tsv_df = pd.DataFrame(
                {"index": np.asarray(parc_index), 
                "annotid": np.asarray(annot_id),
                "parcid": np.asarray(parc_id),
                "name": parc_names, 
                "color": parc_hexcolor}
                )


        # Save the tsv table
        if tsv_file is not None:
            tsv_path = os.path.dirname(tsv_file)
            
                    # Create the directory if it does not exist using the library Path
            tsv_path = Path(tsv_path)
            
            # If the directory does not exist create the directory and if it fails because it does not have write access send an error
            try:
                tsv_path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                print("The TemplateFlow directory does not have write access.")
                sys.exit()

            with open(tsv_file, "w+") as tsv_f:
                tsv_f.write(tsv_df.to_csv(sep="\t", index=False))
        
        return tsv_df, tsv_file
    
    
    @staticmethod
    def gii2annot(gii_file: str,
                    ref_surf: str = None,
                    annot_file: str = None, 
                    cont_tech: str = "local",
                    cont_image: str = "local"):
        """
        Function to convert FreeSurfer gifti files to annot files using mris_convert
        
        Parameters
        ----------
        gii_file       - Required  : Gii filename:
        ref_surf       - Optional  : Reference surface. Default is the white surface of the fsaverage subject:
        annot_file     - Optional  : Annot filename:
        cont_tech      - Optional  : Container technology. Default is local:
        cont_image     - Optional  : Container image. Default is local:
        
        Output
        ------
        gii_file: str : Gii filename
        
        """
        
        if not os.path.exists(gii_file):
            raise ValueError("The gii file does not exist")
        
        if ref_surf is None:
            
            # Get freesurfer directory
            if "FREESURFER_HOME" in os.environ:
                freesurfer_dir = os.path.join(os.environ["FREESURFER_HOME"], 'subjects')
                subj_id = "fsaverage"

                hemi = _detect_hemi(gii_file)
                ref_surf = os.path.join(freesurfer_dir, subj_id, "surf", hemi + ".white")
            else:
                raise ValueError("Impossible to set the reference surface file. Please provide it as an argument")
            
        else:              
            if not os.path.exists(ref_surf):
                raise ValueError("The reference surface file does not exist")
        
        if annot_file is None:
            annot_file = os.path.join(os.path.dirname(gii_file), os.path.basename(gii_file).replace(".gii", ".annot"))

        # Generating the bash command
        cmd_bashargs = ['mris_convert', '--annot', gii_file, ref_surf, annot_file]
        
        cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
        
        return annot_file

    @staticmethod
    def annot2gii(annot_file: str,
                    ref_surf: str = None,
                    gii_file: str = None, 
                    cont_tech: str = "local",
                    cont_image: str = "local"):
        """
        Function to convert FreeSurfer annot files to gii files using mris_convert
        
        Parameters
        ----------
        annot_file     - Required  : Annot filename:
        ref_surf       - Optional  : Reference surface.  Default is the white surface of the fsaverage subject:
        gii_file       - Optional  : Gii filename:
        cont_tech      - Optional  : Container technology. Default is local:
        cont_image     - Optional  : Container image. Default is local:
        
        Output
        ------
        gii_file: str : Gii filename
        
        """
        
        if not os.path.exists(annot_file):
            raise ValueError("The annot file does not exist")
        
        if ref_surf is None:
            
            # Get freesurfer directory
            if "FREESURFER_HOME" in os.environ:
                freesurfer_dir = os.path.join(os.environ["FREESURFER_HOME"], 'subjects')
                subj_id = "fsaverage"

                hemi = _detect_hemi(gii_file)
                ref_surf = os.path.join(freesurfer_dir, subj_id, "surf", hemi + ".white")
            else:
                raise ValueError("Impossible to set the reference surface file. Please provide it as an argument")
            
        else:              
            if not os.path.exists(ref_surf):
                raise ValueError("The reference surface file does not exist")
        
        if gii_file is None:
            gii_file = os.path.join(os.path.dirname(annot_file), os.path.basename(annot_file).replace(".annot", ".gii"))
        
        if not os.path.exists(annot_file):
            raise ValueError("The annot file does not exist")
        
        if not os.path.exists(ref_surf):
            raise ValueError("The reference surface file does not exist")
        
        
        # Generating the bash command
        cmd_bashargs = ['mris_convert', '--annot', annot_file, ref_surf, gii_file]
        
        cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command

    @staticmethod
    def gcs2annot(gcs_file: str, 
                    annot_file: str = None, 
                    freesurfer_dir: str = None, 
                    ref_id: str = "fsaverage", 
                    cont_tech: str = "local",
                    cont_image: str = "local"):
        
        """
        Function to convert gcs files to FreeSurfer annot files
        
        Parameters
        ----------
        gcs_file     - Required  : GCS filename:
        annot_file    - Required  : Annot filename:
        freesurfer_dir - Optional  : FreeSurfer directory. Default is the $SUBJECTS_DIR environment variable:
        ref_id       - Optional  : Reference subject id. Default is fsaverage:
        cont_tech    - Optional  : Container technology. Default is local:
        cont_image   - Optional  : Container image. Default is local:
        
        Output
        ------
        annot_file: str : Annot filename
                
        """
        
        if not os.path.exists(gcs_file):
            raise ValueError("The gcs file does not exist")
        
        
        # Set the FreeSurfer directory
        if freesurfer_dir is not None:
            if not os.path.isdir(freesurfer_dir):
                
                # Create the directory if it does not exist
                freesurfer_dir = Path(freesurfer_dir)
                freesurfer_dir.mkdir(parents=True, exist_ok=True)
                os.environ["SUBJECTS_DIR"] = str(freesurfer_dir)
                
        else:
            if "SUBJECTS_DIR" not in os.environ:
                raise ValueError(
                    "The FreeSurfer directory must be set in the environment variables or passed as an argument"
                )
            else:
                freesurfer_dir = os.environ["SUBJECTS_DIR"]
                
                if not os.path.isdir(freesurfer_dir):
                    
                    # Create the directory if it does not exist
                    freesurfer_dir = Path(freesurfer_dir)
                    freesurfer_dir.mkdir(parents=True, exist_ok=True)

        freesurfer_dir = str(freesurfer_dir)
        
        if not os.path.isdir(freesurfer_dir):
            
            # Take the default FreeSurfer directory
            if "FREESURFER_HOME" in os.environ:
                freesurfer_dir = os.path.join(os.environ["FREESURFER_HOME"], 'subjects')
                ref_id = "fsaverage"
            else:
                raise ValueError("The FreeSurfer directory must be set in the environment variables or passed as an argument")
        
        # Set freesurfer directory as subjects directory
        os.environ["SUBJECTS_DIR"] = freesurfer_dir
            
        hemi_cad = _detect_hemi(gcs_file)
        
        if annot_file is None:
            annot_file = os.path.join(os.path.dirname(gcs_file), os.path.basename(gcs_file).replace(".gcs", ".annot"))
        
        ctx_label = os.path.join(freesurfer_dir, ref_id, "label", hemi_cad + ".cortex.label")
        aseg_presurf = os.path.join(freesurfer_dir, ref_id, "mri", "aseg.mgz")
        sphere_reg = os.path.join(freesurfer_dir, ref_id, "surf", hemi_cad + ".sphere.reg")
        
        cmd_bashargs = ['mris_ca_label', '-l', ctx_label, '-aseg', aseg_presurf, ref_id,
                        hemi_cad, sphere_reg, gcs_file, annot_file]
        
        cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
        
        return annot_file
    
    def _annot2tsv(self, tsv_file: str = None):
        """
        Save the annotation file as a tsv file
        @params:
            tsv_file     - Required  : Output tsv file:
        """

        if tsv_file is None:
            tsv_file = os.path.join(self.path, self.name.replace(".annot", ".tsv"))

        # If the directory does not exist then create it
        temp_dir = os.path.dirname(tsv_file)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Save the annotation file
        np.savetxt(tsv_file, self.codes, fmt="%d", delimiter="\t")

        return tsv_file
    
    def _annot2gcs(
        self,
        gcs_file: str = None,
        freesurfer_dir: str = None,
        fssubj_id: str = None,
        hemi: str = None,
        cont_tech: str = "local", 
        cont_image: str = "local"
    ):
        """
        Convert FreeSurfer annot files to gcs files
        @params:
            annot_file       - Required  : Annot filename:
            gcs_file         - Optional  : GCS filename. If not provided, it will be saved in the same folder as the annot file:
            freesurfer_dir   - Optional  : FreeSurfer directory. Default is the $SUBJECTS_DIR environment variable:
            fssubj_id        - Optional  : FreeSurfer subject id. Default is fsaverage:
            hemi             - Optional  : Hemisphere (lh or rh). If not provided, it will be extracted from the annot filename:
        """

        if gcs_file is None:
            gcs_name = self.name.replace(".annot", ".gcs")

            # Create te gcs folder if it does not exist
            if gcs_folder is None:
                gcs_folder = self.path

            gcs_file = os.path.join(gcs_folder, gcs_name)

        else:
            gcs_name = os.path.basename(gcs_file)
            gcs_folder = os.path.dirname(gcs_file)

        if not os.path.exists(gcs_folder):
            os.makedirs(gcs_folder)

        # Read the colors from annot
        reg_colors = self.regtable[:, 0:3]

        # Create the lookup table for the right hemisphere
        luttable = []
        for roi_pos, roi_name in enumerate(self.regnames):

            luttable.append(
                "{:<4} {:<40} {:>3} {:>3} {:>3} {:>3}".format(
                    roi_pos + 1,
                    roi_name,
                    reg_colors[roi_pos, 0],
                    reg_colors[roi_pos, 1],
                    reg_colors[roi_pos, 2],
                    0,
                )
            )

        # Set the FreeSurfer directory
        if freesurfer_dir is not None:
            if not os.path.isdir(freesurfer_dir):
                
                # Create the directory if it does not exist
                freesurfer_dir = Path(freesurfer_dir)
                freesurfer_dir.mkdir(parents=True, exist_ok=True)
                os.environ["SUBJECTS_DIR"] = str(freesurfer_dir)
                
        else:
            if "SUBJECTS_DIR" not in os.environ:
                raise ValueError(
                    "The FreeSurfer directory must be set in the environment variables or passed as an argument"
                )
            else:
                freesurfer_dir = os.environ["SUBJECTS_DIR"]
                
                if not os.path.isdir(freesurfer_dir):
                    
                    # Create the directory if it does not exist
                    freesurfer_dir = Path(freesurfer_dir)
                    freesurfer_dir.mkdir(parents=True, exist_ok=True)

        # Set the FreeSurfer subject id
        if fssubj_id is None:
            raise ValueError(
                    "Please supply a valid subject ID."
                )
        
        # If the freesurfer subject directory does not exist, raise an error
        if not os.path.isdir(os.path.join(freesurfer_dir, fssubj_id)):
            raise ValueError(
                "The FreeSurfer subject directory for {} does not exist".format(fssubj_id)
            )
        
        if not os.path.isfile(os.path.join(freesurfer_dir, fssubj_id, "surf", "sphere.reg")):
            raise ValueError(
                "The FreeSurfer subject directory for {} does not contain the sphere.reg file".format(fssubj_id)
            )

        # Save the lookup table for the left hemisphere
        ctab_file = os.path.join(gcs_folder, self.name + ".ctab")
        with open(ctab_file, "w") as colorLUT_f:
            colorLUT_f.write("\n".join(luttable))

        # Detecting the hemisphere
        if hemi is None:
            hemi = self.hemi
            if hemi is None:
                raise ValueError(
                    "The hemisphere could not be extracted from the annot filename. Please provide it as an argument"
                )

        cmd_bashargs = [
            "mris_ca_train",
            "-n",
            "2",
            "-t",
            ctab_file,
            hemi,
            "sphere.reg",
            self.filename,
            fssubj_id,
            gcs_file,
        ]
        
        cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command

        # Delete the ctab file
        os.remove(ctab_file)

        return gcs_name

class FreeSurferSubject():
    """
    This class contains methods to work with FreeSurfer subjects.
    
    """

    def __init__(self, subj_id:str, subjs_dir: str = None):
        """
        This method initializes the FreeSurferSubject object according to the subject id and the subjects directory.
        It reads the FreeSurfer files and stores them in a dictionary.
                
        Parameters
        ----------
        
        subj_id: str     - Required  : FreeSurfer subject id:
        subjs_dir: str   - Optional  : FreeSurfer subjects directory. Default is the $SUBJECTS_DIR environment variable:
        
        Output
        ------
        fs_files: dict : Dictionary with the FreeSurfer files
        
        
        """
        
        if subjs_dir is None:
            self.subjs_dir = os.environ.get('SUBJECTS_DIR')
        else:
            if not os.path.exists(subjs_dir):
                raise FileNotFoundError(f"Directory {subjs_dir} does not exist")
            self.subjs_dir = subjs_dir
        
        subj_dir= os.path.join(self.subjs_dir, subj_id)
        if not os.path.exists(subj_dir):
            raise FileNotFoundError(f"Directory {subj_id} does not exist in {self.subjs_dir}")
        
        self.subj_id = subj_id

        # Generate a dictionary of the FreeSurfer files
        self.fs_files = {}
        mri_dict = {}
        mri_dict['orig'] = os.path.join(subj_dir, 'mri', 'orig.mgz')
        mri_dict['brainmask'] = os.path.join(subj_dir, 'mri', 'brainmask.mgz')
        mri_dict['aseg'] = os.path.join(subj_dir, 'mri', 'aseg.mgz')
        mri_dict['desikan+aseg'] = os.path.join(subj_dir, 'mri', 'aparc+aseg.mgz')
        mri_dict['destrieux+aseg'] = os.path.join(subj_dir, 'mri', 'aparc.a2009s+aseg.mgz')
        mri_dict['dkt+aseg'] = os.path.join(subj_dir, 'mri', 'aparc.DKTatlas+aseg.mgz')
        mri_dict['T1'] = os.path.join(subj_dir, 'mri', 'T1.mgz')
        mri_dict['talairach'] = os.path.join(subj_dir, 'mri', 'transforms', 'talairach.lta')
        mri_dict['ribbon'] = os.path.join(subj_dir, 'mri', 'ribbon.mgz')
        mri_dict['wm'] = os.path.join(subj_dir, 'mri', 'wm.mgz')
        mri_dict['wmparc'] = os.path.join(subj_dir, 'mri', 'wmparc.mgz')
        self.fs_files['mri'] = mri_dict

        # Creating the Surf dictionary
        surf_dict = {}

        lh_s_dict, lh_t_dict = self._get_hemi_dicts(subj_dir=subj_dir, hemi='lh')
        rh_s_dict, rh_t_dict = self._get_hemi_dicts(subj_dir=subj_dir, hemi='rh')

        surf_dict['lh'] = lh_s_dict
        surf_dict['rh'] = rh_s_dict
        self.fs_files['surf'] = surf_dict

        # Creating the Stats dictionary
        stats_dict = {}
        global_dict = {}
        global_dict['aseg'] = os.path.join(subj_dir, 'stats', 'aseg.stats')
        global_dict['wmparc'] = os.path.join(subj_dir, 'stats', 'wmparc.stats')
        global_dict['brainvol'] = os.path.join(subj_dir, 'stats', 'brainvol.stats')
        stats_dict['global'] = global_dict
        stats_dict['lh'] = lh_t_dict
        stats_dict['rh'] = rh_t_dict

        self.fs_files['stats'] = stats_dict


    def _get_hemi_dicts(self, subj_dir:str, hemi:str):
        """
        This method creates the dictionaries for the hemisphere files.
        
        Parameters
        ----------
        subj_dir: str     - Required  : FreeSurfer subject ID:
        
        hemi: str        - Required  : Hemisphere (lh or rh):
        
        
        """
        
        # Surface dictionary
        s_dict = {}
        s_dict['pial'] = os.path.join(subj_dir, 'surf', hemi + '.pial')
        s_dict['white'] = os.path.join(subj_dir, 'surf', hemi + '.white')
        s_dict['inflated'] = os.path.join(subj_dir, 'surf', hemi + '.inflated')
        s_dict['sphere'] = os.path.join(subj_dir, 'surf', hemi + '.sphere')
        s_dict['curv'] = os.path.join(subj_dir, 'surf', hemi + '.curv')
        s_dict['sulc'] = os.path.join(subj_dir, 'surf', hemi + '.sulc')
        s_dict['thickness'] = os.path.join(subj_dir, 'surf', hemi + '.thickness')
        s_dict['area'] = os.path.join(subj_dir, 'surf', hemi + '.area')
        s_dict['volume'] = os.path.join(subj_dir, 'surf', hemi + '.volume')
        s_dict['lgi'] = os.path.join(subj_dir, 'surf', hemi + '.pial_lgi')
        s_dict['desikan'] = os.path.join(subj_dir, 'label', hemi + '.aparc.annot')
        s_dict['destrieux'] = os.path.join(subj_dir, 'label', hemi + '.aparc.a2009s.annot')
        s_dict['dkt'] = os.path.join(subj_dir, 'label', hemi + '.aparc.DKTatlas.annot')
        
        # Statistics dictionary
        t_dict = {}
        t_dict['desikan'] = os.path.join(subj_dir, 'stats', hemi + '.aparc.stats')
        t_dict['destrieux'] = os.path.join(subj_dir, 'stats', hemi + '.aparc.a2009s.stats')
        t_dict['dkt'] = os.path.join(subj_dir, 'stats', hemi + '.aparc.DKTatlas.stats')
        t_dict['curv'] = os.path.join(subj_dir, 'stats', hemi + '.curv.stats')

        return s_dict, t_dict
    
    def _get_proc_status(self):
        """
        This method checks the processing status
            
        Parameters
        ----------
        self: object : FreeSurferSubject object
            
        Returns
        -------
        pstatus: str : Processing status (all, autorecon1, autorecon2, unprocessed)
            
        """
        
        # Check if the FreeSurfer subject id exists
        if not os.path.isdir(os.path.join(self.subjs_dir, self.subj_id)):
            pstatus = 'unprocessed'
        else:
            arecon1_files = [self.fs_files["mri"]['T1'], self.fs_files["mri"]['brainmask'], 
                            self.fs_files["mri"]['orig']]

            arecon2_files = [self.fs_files["mri"]['talairach'], self.fs_files["mri"]['wm'],
                            self.fs_files["surf"]['lh']['pial'], self.fs_files["surf"]['rh']['pial'],
                            self.fs_files["surf"]['lh']['white'], self.fs_files["surf"]['rh']['white'],
                            self.fs_files["surf"]['lh']['inflated'], self.fs_files["surf"]['rh']['inflated'], 
                            self.fs_files["surf"]['lh']['curv'], self.fs_files["surf"]['rh']['curv'], 
                            self.fs_files["stats"]['lh']['curv'], self.fs_files["stats"]['rh']['curv'], 
                            self.fs_files["surf"]['lh']['sulc'], self.fs_files["surf"]['rh']['sulc']]
            
            arecon3_files = [self.fs_files["mri"]['aseg'], self.fs_files["mri"]['desikan+aseg'],
                            self.fs_files["mri"]['destrieux+aseg'], self.fs_files["mri"]['dkt+aseg'],
                            self.fs_files["mri"]['wmparc'], self.fs_files["mri"]['ribbon'],
                            self.fs_files["surf"]['lh']['sphere'], self.fs_files["surf"]['rh']['sphere'],
                            self.fs_files["surf"]['lh']['thickness'], self.fs_files["surf"]['rh']['thickness'],
                            self.fs_files["surf"]['lh']['area'], self.fs_files["surf"]['rh']['area'],
                            self.fs_files["surf"]['lh']['volume'], self.fs_files["surf"]['rh']['volume'],
                            self.fs_files["surf"]['lh']['desikan'], self.fs_files["surf"]['rh']['desikan'],
                            self.fs_files["surf"]['lh']['destrieux'], self.fs_files["surf"]['rh']['destrieux'],
                            self.fs_files["surf"]['lh']['dkt'], self.fs_files["surf"]['rh']['dkt'],
                            self.fs_files["stats"]['lh']['desikan'], self.fs_files["stats"]['rh']['desikan'],
                            self.fs_files["stats"]['lh']['destrieux'], self.fs_files["stats"]['rh']['destrieux'],
                            self.fs_files["stats"]['lh']['dkt'], self.fs_files["stats"]['rh']['dkt']]
            
            # Check if the files exist in the FreeSurfer subject directory for auto-recon1 
            if all([os.path.exists(f) for f in arecon1_files]):
                arecon1_bool = True
            else:
                arecon1_bool = False

            # Check if the files exist in the FreeSurfer subject directory for auto-recon2
            if all([os.path.exists(f) for f in arecon2_files]):
                arecon2_bool = True
            else:
                arecon2_bool = False

            # Check if the files exist in the FreeSurfer subject directory for auto-recon3
            if all([os.path.exists(f) for f in arecon3_files]):
                arecon3_bool = True
            else:
                arecon3_bool = False

            # Check the processing status
            if arecon3_bool and arecon2_bool and arecon1_bool:
                pstatus = 'processed'
            elif arecon2_bool and arecon1_bool and not arecon3_bool:
                pstatus = 'autorecon2'
            elif arecon1_bool and not arecon2_bool and not arecon3_bool:
                pstatus = 'autorecon1'
            else:
                pstatus = 'unprocessed'

        self.pstatus = pstatus

    
    def _launch_freesurfer(self, t1w_img:str = None, 
                            proc_stage: Union[str, list] = 'all',
                            extra_proc: Union[str, list] = None,
                            cont_tech: str = "local",
                            cont_image: str = "local",
                            force = False):
        """
        Function to launch recon-all command with different options
        
        Parameters
        ----------
        t1w_img       - Mandatory : T1w image filename:
        proc_stage    - Optional  : Processing stage. Default is all:
                                    Valid options are: all, autorecon1, autorecon2, autorecon3
        extra_proc    - Optional  : Extra processing stages. Default is None:
                                    Valid options are: lgi, thalamus, brainstem, hippocampus, amygdala, hypothalamus
                                    Some A few freesurfer modules, like subfield/nuclei segmentation tools, require 
                                    the matlab runtime package (MCR). 
                                    Please go to https://surfer.nmr.mgh.harvard.edu/fswiki/MatlabRuntime
                                    to download the appropriate version of MCR for your system.
        cont_tech    - Optional  : Container technology. Default is local:
        cont_image   - Optional  : Container image. Default is local:
        force        - Optional  : Force the processing. Default is False:
        
        Output
        ------
        proc_stage: str : Processing stage
        
        """

        # Set the FreeSurfer directory
        if self.subjs_dir is not None:
            
            if not os.path.isdir(self.subjs_dir):
                
                # Create the directory if it does not exist
                self.subjs_dir = Path(self.subjs_dir)
                self.subjs_dir.mkdir(parents=True, exist_ok=True)
                os.environ["SUBJECTS_DIR"] = str(self.subjs_dir)
                
        else:
            if "SUBJECTS_DIR" not in os.environ:
                raise ValueError(
                    "The FreeSurfer directory must be set in the environment variables or passed as an argument"
                )
            else:
                self.subjs_dir = os.environ["SUBJECTS_DIR"]
                
                if not os.path.isdir(self.subjs_dir):
                    
                    # Create the directory if it does not exist
                    self.subjs_dir = Path(self.subjs_dir)
                    self.subjs_dir.mkdir(parents=True, exist_ok=True)
        
        # Getting the freesurfer version
        ver_cad = get_version(cont_tech = cont_tech,cont_image = cont_image)
        ver_ent = ver_cad.split('.')
        vert_int = int(''.join(ver_ent))

        if not hasattr(self, 'pstatus'):
            self._get_proc_status() 
        proc_status = self.pstatus

        # Check if the processing stage is valid
        val_stages = ['all', 'autorecon1', 'autorecon2', 'autorecon3']
        
        if isinstance(proc_stage, str):
            proc_stage = [proc_stage]
        
        proc_stage = [stage.lower() for stage in proc_stage]

        for stage in proc_stage:
            if stage not in val_stages:
                raise ValueError(f"Stage {stage} is not valid")
        
        if 'all' in proc_stage:
            proc_stage = ['all']
        
        # Check if the extra processing stages are valid
        val_extra_stages = ['lgi', 'thalamus', 'brainstem', 'hippocampus', 'amygdala', 'hypothalamus']
        if extra_proc is not None:
            if isinstance(extra_proc, str):
                extra_proc = [extra_proc]

            # Put the extra processing stages in lower case
            extra_proc = [stage.lower() for stage in extra_proc]
            
            # If hippocampus and amygdala are in the list, remove amygdala from the list
            if 'hippocampus' in extra_proc and 'amygdala' in extra_proc:
                extra_proc.remove('amygdala')

            for stage in extra_proc:
                if stage not in val_extra_stages:
                    raise ValueError(f"Stage {stage} is not valid")

        
        if force:

            if t1w_img is None:
                if os.path.isdir(os.path.join(self.subjs_dir, self.subj_id)) and os.path.isfile(self.fs_files['mri']['orig']):
                    for st in proc_stage:
                        cmd_bashargs = ['recon-all', '-subjid', self.subj_id, '-' + st]
                        cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image)
                        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
            else:
                if os.path.isfile(t1w_img):
                    for st in proc_stage:
                        cmd_bashargs = ['recon-all', '-subjid', self.subj_id, '-i', t1w_img,'-' + st]
                        cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
                        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
                else:
                    raise ValueError("The T1w image does not exist")
        else:
            if proc_status == 'unprocessed':
                if t1w_img is None:
                    if os.path.isdir(os.path.join(self.subjs_dir, self.subj_id)) and os.path.isfile(self.fs_files['mri']['orig']):
                        cmd_bashargs = ['recon-all', '-subjid', self.subj_id, '-all']
                else:
                    if os.path.isfile(t1w_img):
                        cmd_bashargs = ['recon-all', '-subjid', self.subj_id, '-i', t1w_img, '-all']
                    else:
                        raise ValueError("The T1w image does not exist")

                cmd_bashargs = ['recon-all', '-subjid', '-i', t1w_img, self.subj_id, '-all']
                cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
                subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
            elif proc_status == 'autorecon1':
                cmd_bashargs = ['recon-all', '-subjid', self.subj_id, '-autorecon2']
                cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
                subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command

                cmd_bashargs = ['recon-all', '-subjid', self.subj_id, '-autorecon3']
                cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
                subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command

            elif proc_status == 'autorecon2':
                cmd_bashargs = ['recon-all', '-subjid', self.subj_id, '-autorecon3']
                cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image)
                subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command

        self._get_proc_status() 
        proc_status = self.pstatus

        # Processing extra stages
        if extra_proc is not None:
            if isinstance(extra_proc, str):
                extra_proc = [extra_proc]

            cmd_list = []
            for stage in extra_proc:
                if stage in val_extra_stages:
                    if stage == 'lgi': # Compute the local gyrification index

                        if ((not os.path.isfile(self.fs_files['surf']['lh']['lgi']) and not os.path.isfile(self.fs_files['surf']['rh']['lgi'])) or force == True):
                            cmd_bashargs = ['recon-all', '-subjid', self.subj_id, '-lgi']
                            cmd_list.append(cmd_bashargs)

                    elif stage == 'thalamus': # Segment the thalamic nuclei using the thalamic nuclei segmentation tool
                        
                        th_files = glob(os.path.join(self.subjs_dir, self.subj_id, 'mri', 'ThalamicNuclei.*'))

                        if len(th_files) != 3 or force == True:
                            if vert_int < 730:
                                cmd_bashargs = ['segmentThalamicNuclei.sh', self.subj_id, self.subjs_dir]
                            else:
                                cmd_bashargs =  ['segment_subregions', 'thalamus', '--cross', self.subj_id]

                            cmd_list.append(cmd_bashargs)

                    elif stage == 'brainstem': # Segment the brainstem structures

                        bs_files = glob(os.path.join(self.subjs_dir, self.subj_id, 'mri', 'brainstemS*'))

                        if len(bs_files) != 3 or force == True:
                            os.system("WRITE_POSTERIORS=1")
                            if vert_int < 730:
                                cmd_bashargs = ['segmentBS.sh', self.subj_id, self.subjs_dir]
                            else:
                                cmd_bashargs = ['segment_subregions', 'brainstem', '--cross', self.subj_id]

                            cmd_list.append(cmd_bashargs)
                    
                    elif stage == 'hippocampus' or stage == 'amygdala': # Segment the hippocampal subfields

                        ha_files = glob(os.path.join(self.subjs_dir, self.subj_id, 'mri', '*hippoAmygLabels.*'))

                        if len(ha_files) != 16 or force == True:
                            if vert_int < 730: # Use the FreeSurfer script for versions below 7.2.0
                                cmd_bashargs = ['segmentHA_T1.sh', self.subj_id, self.subjs_dir]
                            else:
                                cmd_bashargs = ['segment_subregions', 'hippo-amygdala', '--cross', self.subj_id]  
                            
                            cmd_list.append(cmd_bashargs)

                    elif stage == 'hypothalamus': # Segment the hypothalamic subunits

                        os.system("WRITE_POSTERIORS=1")
                        if len(bs_files) != 3 or force == True:
                            cmd_bashargs = ['mri_segment_hypothalamic_subunits', '--s', self.subj_id, '--sd', self.subjs_dir, '--write_posteriors']
                            cmd_list.append(cmd_bashargs)

            if len(cmd_list) > 0:
                for cmd_bashargs in cmd_list:
                    cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image)
                    subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command

        return proc_status

    @staticmethod
    def _set_freesurfer_directory(fs_dir: str = None):
        """
        Function to set up the FreeSurfer directory
        
        Parameters
        ----------
        fs_dir       - Optional  : FreeSurfer directory. Default is None:
                                If not provided, it will be extracted from the 
                                $SUBJECTS_DIR environment variable. If it does not exist,
                                it will be created.
        
        Output
        ------
        fs_dir: str : FreeSurfer directory
        """

        # Set the FreeSurfer directory
        if fs_dir is None:
        
            if "SUBJECTS_DIR" not in os.environ:
                raise ValueError(
                    "The FreeSurfer directory must be set in the environment variables or passed as an argument"
                )
            else:
                fs_dir = os.environ["SUBJECTS_DIR"]
                
        # Create the directory if it does not exist
        fs_dir= Path(fs_dir)
        fs_dir.mkdir(parents=True, exist_ok=True)
        os.environ["SUBJECTS_DIR"] = str(fs_dir)

    
    def _annot2ind(self, 
                    ref_id:str, 
                    hemi:str, 
                    fs_annot:str, 
                    ind_annot:str, 
                    cont_tech:str="local", 
                    cont_image:str = None,
                    force = False):
        """
        Map ANNOT parcellation files to individual space.
        
        Parameters:
        ----------
        ref_id : str
            FreeSurfer ID for the reference subject
        
        hemi : str
            Hemisphere id ("lh" or "rh")
            
        fs_annot : str
            FreeSurfer GCS parcellation file
            
        ind_annot : str
            Annotation file in individual space
            
        cont_tech : str
            Container technology ("singularity", "docker", "local")
            
        cont_image: str
            Container image to use
        
        force : bool
            Force the processing    
            
        """
        
        if not os.path.isfile(fs_annot) and not os.path.isfile(os.path.join(self.subjs_dir, ref_id, 'label', hemi + '.' + fs_annot + '.annot')):
            raise FileNotFoundError(f"Files {fs_annot} or {os.path.join(self.subjs_dir, ref_id, 'label', hemi + '.' + fs_annot + '.annot')} do not exist")
        
        if fs_annot.endswith('.gii'):
            tmp_annot = fs_annot.replace(".gii", ".annot")
            tmp_refsurf = os.path.join(self.subjs_dir, ref_id, 'surf', hemi + '.inflated')
            
            AnnotParcellation.gii2annot(gii_file = fs_annot,
                                            ref_surf = tmp_refsurf,
                                            annot_file = tmp_annot, 
                                            cont_tech = cont_tech, 
                                            cont_image = cont_image)
            fs_annot = tmp_annot
    

        if not os.path.isfile(ind_annot) or force:
            
            FreeSurferSubject._set_freesurfer_directory(self.subjs_dir)
            
            # Create the folder if it does not exist
            temp_dir = os.path.dirname(ind_annot)
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Moving the Annot to individual space
            cmd_bashargs = ['mri_surf2surf', '--srcsubject', ref_id, '--trgsubject', self.subj_id,
                                    '--hemi', hemi, '--sval-annot', fs_annot,
                                    '--tval', ind_annot]
            cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
            subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
            
            # Correcting the parcellation file in order to refill the parcellation with the correct labels    
            cort_parc = AnnotParcellation(parc_file=ind_annot)
            label_file = os.path.join(self.subjs_dir, self.subj_id, 'label', hemi + '.cortex.label')
            surf_file = os.path.join(self.subjs_dir, self.subj_id, 'surf', hemi + '.inflated')
            cort_parc._fill_parcellation(corr_annot=ind_annot, label_file=label_file, surf_file=surf_file)
            
        elif os.path.isfile(ind_annot) and not force:
            # Print a message
            print(f"File {ind_annot} already exists. Use force=True to overwrite it")
            
        return ind_annot
    
    def _gcs2ind(self,
                    fs_gcs: str, 
                    ind_annot: str, 
                    hemi: str, 
                    cont_tech: str = "local", 
                    cont_image: str = None,
                    force = False):
        """
        Map GCS parcellation files to individual space.
        
        Parameters:
        ----------
        fs_gcs : str
            FreeSurfer GCS parcellation file
            
        ind_annot : str
            Individual space annotation file
            
        hemi : str
            Hemisphere id ("lh" or "rh")

        cont_tech : str
            Container technology ("singularity", "docker", "local")
            
        cont_image: str
            Container image to use    
                
        force : bool
            Force the processing  
            
        """

        if not os.path.isfile(ind_annot) or force:
            
            # Create the folder if it does not exist
            temp_dir = os.path.dirname(ind_annot)
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Moving the GCS to individual space
            cort_file = os.path.join(self.subjs_dir, self.subj_id, 'label', hemi + '.cortex.label')
            sph_file  = os.path.join(self.subjs_dir, self.subj_id, 'surf', hemi + '.sphere.reg')
            
            cmd_bashargs = ['mris_ca_label', '-l', cort_file, self.subj_id, hemi, sph_file,
                            fs_gcs, ind_annot]
            
            cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
            subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
            
            # Correcting the parcellation file in order to refill the parcellation with the correct labels    
            cort_parc = AnnotParcellation(parc_file=ind_annot)
            label_file = os.path.join(self.subjs_dir, self.subj_id, 'label', hemi + '.cortex.label')
            surf_file = os.path.join(self.subjs_dir, self.subj_id, 'surf', hemi + '.inflated')
            cort_parc._fill_parcellation(corr_annot=ind_annot, label_file=label_file, surf_file=surf_file)
            
        elif os.path.isfile(ind_annot) and not force:
                    # Print a message
                    print(f"File {ind_annot} already exists. Use force=True to overwrite it")

        return ind_annot
    
    def _surf2vol(self, 
                        atlas: str,
                        out_vol: str, 
                        gm_grow: Union[int, str] = '0', 
                        bool_native: bool = False,
                        cont_tech: str = "local", 
                        cont_image: str = None,
                        force: bool = False):
    
        """
        Create volumetric parcellation from annot files.
        
        Parameters:
        ----------
        atlas : str
            Atlas ID   
        
        out_vol: str
            Output volumetric parcellation file
            
        gm_grow : list or str
            Amount of milimiters to grow the GM labels
        
        bool_native: bool
            If True, the parcellation will be in native space. The parcellation in native space 
            will be saved in Nifti-1 format.
            
        cont_tech : str
            Container technology ("singularity", "docker", "local")
            
        cont_image: str
            Container image to use
        
        force : bool
            Force the processing.  
            
        """
        
        FreeSurferSubject._set_freesurfer_directory(self.subjs_dir)
        
        if isinstance(gm_grow, int):
            gm_grow = str(gm_grow)
        
        # Creating the volumetric parcellation using the annot files
        if out_vol.endswith('.mgz'):
            out_nat = out_vol.replace('.mgz', '.nii.gz')
        elif out_vol.endswith('.nii.gz'):
            out_nat = out_vol
                
        if gm_grow == '0':

            if not os.path.isfile(out_nat) or force:
                cmd_bashargs = ['mri_aparc2aseg', '--s', self.subj_id, '--annot', atlas,
                                '--hypo-as-wm', '--new-ribbon', '--o', out_nat]
                cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
                subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
                
                if bool_native:
                    
                    # Moving the resulting parcellation from conform space to native
                    self._conform2native(mgz_conform = out_vol, nii_native = out_nat, force=force)
                    
            elif os.path.isfile(out_nat) and not force:
                # Print a message
                print(f"File {out_nat} already exists. Use force=True to overwrite it")

        else:
            if not os.path.isfile(out_nat) or force:
                # Creating the volumetric parcellation using the annot files
                cmd_bashargs = ['mri_aparc2aseg', '--s', self.subj_id, '--annot', atlas, '--wmparc-dmax', gm_grow, '--labelwm',
                                '--hypo-as-wm', '--new-ribbon', '--o', out_nat]
                cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
                subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
                
                if bool_native:
                    
                    # Moving the resulting parcellation from conform space to native
                    self._conform2native(mgz_conform = out_vol, nii_native = out_nat, force=force)
                    
            elif os.path.isfile(out_nat) and not force:
                # Print a message
                print(f"File {out_nat} already exists. Use force=True to overwrite it")
                    

        return out_nat

    def _conform2native(self, 
                        mgz_conform: str, 
                        nii_native: str, 
                        interp_method: str = "nearest", 
                        cont_tech: str = "local", 
                        cont_image: str = None,
                        force: bool = False):
        """
        Moving image in comform space to native space
        
        Parameters:
        ----------
        mgz_conform : str
            Image in conform space

        nii_native : str
            Image in native space
            
        fssubj_dir : str
            FreeSurfer subjects directory
        
        fullid : str
            FreeSurfer ID   
        
        interp_method: str
            Interpolation method ("nearest", "trilinear", "cubic")
        
        cont_tech : str
            Container technology ("singularity", "docker", "local")
            
        cont_image: str
            Container image to use
        
        force : bool
            Force the processing
            
        """
        
        if not os.path.isfile(nii_native) or force:
            # Moving the resulting parcellation from conform space to native
            raw_vol = os.path.join(self.subjs_dir, self.subj_id, 'mri', 'rawavg.mgz')

            cmd_bashargs = ['mri_vol2vol', '--mov', mgz_conform, '--targ', raw_vol,
                            '--regheader', '--o', nii_native, '--no-save-reg', '--interp', interp_method]
            cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
            subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
        
        elif os.path.isfile(nii_native) and not force:
            # Print a message
            print(f"File {nii_native} already exists. Use force=True to overwrite it")
    
                
def _create_fsaverage_links(
    fssubj_dir: str, fsavg_dir: str = None, refsubj_name: str = None
):
    """
    Create the links to the fsaverage folder

    Parameters
    ----------
        fssubj_dir     - Required  : FreeSurfer subjects directory. It does not have to match the $SUBJECTS_DIR environment variable:
        fsavg_dir      - Optional  : FreeSurfer fsaverage directory. If not provided, it will be extracted from the $FREESURFER_HOME environment variable:
        refsubj_name   - Optional  : Reference subject name. Default is None:

    Returns
    -------
    link_folder: str
        Path to the linked folder

    """

    # Verify if the FreeSurfer directory exists
    if not os.path.isdir(fssubj_dir):
        raise ValueError("The selected FreeSurfer directory does not exist")

    # Creating and veryfying the freesurfer directory for the reference name
    if fsavg_dir is None:
        if refsubj_name is None:
            fsavg_dir = os.path.join(
                os.environ["FREESURFER_HOME"], "subjects", "fsaverage"
            )
        else:
            fsavg_dir = os.path.join(
                os.environ["FREESURFER_HOME"], "subjects", refsubj_name
            )
    else:
        if fsavg_dir.endswith(os.path.sep):
            fsavg_dir = fsavg_dir[0:-1]

        if refsubj_name is not None:
            if not fsavg_dir.endswith(refsubj_name):
                fsavg_dir = os.path.join(fsavg_dir, refsubj_name)

    if not os.path.isdir(fsavg_dir):
        raise ValueError("The selected fsaverage directory does not exist")

    # Taking into account that the fsaverage folder could not be named fsaverage
    refsubj_name = os.path.basename(fsavg_dir)

    # Create the link to the fsaverage folder
    link_folder = os.path.join(fssubj_dir, refsubj_name)

    if not os.path.isdir(link_folder):
        process = subprocess.run(
            ["ln", "-s", fsavg_dir, fssubj_dir],
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

    return link_folder


def _remove_fsaverage_links(linkavg_folder: str):
    """
    Remove the links to the average folder
    @params:
        linkavg_folder     - Required  : FreeSurfer average directory.
                                        It does not have to match the $SUBJECTS_DIR environment variable.
                                        If it is a link and do not match with the original fsaverage folder
                                        then it will be removed:
    """

    # FreeSurfer subjects directory
    fssubj_dir_orig = os.path.join(
        os.environ["FREESURFER_HOME"], "subjects", "fsaverage"
    )

    # if linkavg_folder is a link then remove it
    if (
        os.path.islink(linkavg_folder)
        and os.path.realpath(linkavg_folder) != fssubj_dir_orig
    ):
        os.remove(linkavg_folder)
        
def _detect_hemi(file_name: str):
    """
    Detect the hemisphere from the filename
    
    Parameters
    ----------
        file_name     - Required  : Filename:
        
    Returns
    -------
    hemi_cad: str : Hemisphere name
        
    """
    
    
    # Detecting the hemisphere
    file_name = file_name.lower()
    
    # Find in the string annot_name if it is lh. or rh.
    if "lh." in file_name:
        hemi = "lh"
    elif "rh." in file_name:
        hemi = "rh"
    elif "hemi-l" in file_name:
        hemi = "lh"
    elif "hemi-r" in file_name:
        hemi = "rh"
    else:
        hemi = None
        raise ValueError(
            "The hemisphere could not be extracted from the annot filename. Please provide it as an argument"
        )

    return hemi

def get_version(cont_tech: str = "local", 
                cont_image: str = "local"):
    
    """
    Function to get the FreeSurfer version.
    
    Parameters
    ----------
    cont_tech    - Optional  : Container technology. Default is local:
    cont_image   - Optional  : Container image. Default is local:
    
    Output
    ------
    vers_cad: str : FreeSurfer version number 
    
    """
    
    # Running the version command
    cmd_bashargs = ['recon-all', '-version']
    cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
    subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
    out_cmd = subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True)
    
    vers_cad = out_cmd.stdout.split('-')[3]
    
    return vers_cad
    
    
            


def _launch_surf2vol(fssubj_dir: str, 
                        out_dir: str, 
                        fullid: str, 
                        atlas: str, 
                        gm_grow: str, 
                        cont_tech: str = "local", 
                        cont_image: str = "local"):
    
    """
    Create volumetric parcellation from annot files.
    
    Parameters:
    ----------
    fssubj_dir : str
        FreeSurfer subjects directory

    out_dir : str
        Output directory
        
    fullid : str
        FreeSurfer ID
    
    atlas : str
        Atlas ID   
        
    gm_grow : str
        Amount of milimiters to grow the GM labels
        
    cont_tech : str
        Container technology ("singularity", "docker", "local")
        
    cont_image: str
        Container image to use
        
    """

    if 'desc' not in atlas:
        atlas_str = atlas + '_desc-'
    else:
        atlas_str = atlas

    if atlas == "aparc":
        atlas_str = "atlas-desikan_desc-aparc"
    elif atlas == "aparc.a2009s":
        atlas_str = "atlas-destrieux_desc-a2009s"

    out_parc = []
    for g in gm_grow:
        out_vol = os.path.join(out_dir, fullid + '_space-orig_' + atlas_str + 'grow' + g + 'mm_dseg.nii.gz')

        if g == '0':
            # Creating the volumetric parcellation using the annot files

            cmd_bashargs = ['mri_aparc2aseg', '--s', fullid, '--annot', atlas,
                            '--hypo-as-wm', '--new-ribbon', '--o', out_vol]
            cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
            subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command

        else:
            # Creating the volumetric parcellation using the annot files
            cmd_bashargs = ['mri_aparc2aseg', '--s', fullid, '--annot', atlas, '--wmparc-dmax', g, '--labelwm',
                            '--hypo-as-wm', '--new-ribbon', '--o', out_vol]
            cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
            subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command

        # Moving the resulting parcellation from conform space to native
        raw_vol = os.path.join(fssubj_dir, fullid, 'mri', 'rawavg.mgz')
        
        cmd_bashargs = ['mri_vol2vol', '--mov', out_vol, '--targ', raw_vol,
                        '--regheader', '--o', out_vol, '--no-save-reg', '--interp', 'nearest']
        cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
        
        out_parc.append(out_vol)

    return out_parc

def _conform2native(cform_mgz: str, 
                            nat_nii: str, 
                            fssubj_dir: str, 
                            fullid: str,
                            interp_method: str = "nearest", 
                            cont_tech: str = "local", 
                            cont_image: str = "local"):
    """
    Moving image in comform space to native space
    
    Parameters:
    ----------
    cform_mgz : str
        Image in conform space

    nat_nii : str
        Image in native space
        
    fssubj_dir : str
        FreeSurfer subjects directory
    
    fullid : str
        FreeSurfer ID   
    
    interp_method: str
        Interpolation method ("nearest", "trilinear", "cubic")
    
    cont_tech : str
        Container technology ("singularity", "docker", "local")
        
    cont_image: str
        Container image to use
        
    """
    
    # Moving the resulting parcellation from conform space to native
    raw_vol = os.path.join(fssubj_dir, fullid, 'mri', 'rawavg.mgz')

    cmd_bashargs = ['mri_vol2vol', '--mov', cform_mgz, '--targ', raw_vol,
                    '--regheader', '--o', nat_nii, '--no-save-reg', '--interp', interp_method]
    cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
    subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command


def _fs_addon_parcellations(vol_tparc, fullid, fssubj_dir, parcid, out_str,                             
                            cont_tech: str = "local", 
                            cont_image: str = "local"):
    
    
    
    

    volatlas_dir = os.path.dirname(vol_tparc)

    # Creating ouput directory
    if not os.path.isdir(volatlas_dir):
        try:
            os.makedirs(volatlas_dir)
        except OSError:
            print("Failed to make nested output directory")

    if parcid == 'thalamus':
    # Running Thalamic parcellation
        process = subprocess.run(
            ['segmentThalamicNuclei.sh', fullid, fssubj_dir],
            stdout=subprocess.PIPE, universal_newlines=True)
        
        
        cmd_bashargs = ['segmentThalamicNuclei.sh', fullid, fssubj_dir]
        cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
        

        thal_mgz = os.path.join(fssubj_dir, fullid, 'mri', 'ThalamicNuclei.v12.T1.mgz')

        # Moving Thalamic parcellation to native space
        _conform2native(thal_mgz, vol_tparc, fssubj_dir, fullid)

        out_parc = [vol_tparc]

    elif parcid == 'amygdala' or  parcid == 'hippocampus':
        # Running Hippocampal and Amygdala parcellation
        
        cmd_bashargs = ['segmentHA_T1.sh', fullid, fssubj_dir]
        cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command

        # Moving Hippocampal and amygdala parcellation to native space
        lh_mgz = os.path.join(fssubj_dir, fullid, 'mri', 'lh.hippoAmygLabels-T1.v21.mgz')
        lh_gz = os.path.join(volatlas_dir, fullid + '_space-orig_hemi-L_desc-' + out_str + '_dseg.nii.gz')
        _conform2native(lh_mgz, lh_gz, fssubj_dir, fullid)

        rh_mgz = os.path.join(fssubj_dir, fullid, 'mri', 'rh.hippoAmygLabels-T1.v21.mgz')
        rh_gz = os.path.join(volatlas_dir, fullid + '_space-orig_hemi-R_desc-' + out_str + '_dseg.nii.gz')
        _conform2native(rh_mgz, rh_gz, fssubj_dir, fullid)
        out_parc = [lh_gz, rh_gz]

    elif parcid == 'hypothalamus':

    # Running Hypothalamus parcellation
        os.system("WRITE_POSTERIORS=1")
        cmd_bashargs = ['mri_segment_hypothalamic_subunits', '--s', fullid, '--sd', fssubj_dir, '--write_posteriors']
        cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
        
        # Moving Hypothalamus to native space
        hypo_mgz = os.path.join(fssubj_dir, fullid, 'mri', 'hypothalamic_subunits_seg.v1.mgz')
        hypo_gz = os.path.join(volatlas_dir, fullid + '_space-orig_desc-' + out_str + '_dseg.nii.gz')
        _conform2native(hypo_mgz, hypo_gz, fssubj_dir, fullid)
        out_parc = [hypo_gz]

    elif parcid == 'brainstem':

        # Running Brainstem parcellation
        # os.environ["WRITE_POSTERIORS"] = 1
        os.system("WRITE_POSTERIORS=1")
        
        cmd_bashargs = ['segmentBS.sh', fullid, fssubj_dir]
        cmd_cont = cltmisc._generate_container_command(cmd_bashargs, cont_tech, cont_image) # Generating container command
        subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True) # Running container command
        
        # Moving Hypothalamus to native space
        bs_mgz = os.path.join(fssubj_dir, fullid, 'mri', 'brainstemSsLabels.v12.mgz')
        bs_gz = os.path.join(volatlas_dir, fullid + '_space-orig_desc-' + out_str + '_dseg.nii.gz')
        _conform2native(bs_mgz, bs_gz, fssubj_dir, fullid)
        out_parc = [bs_gz]

    return out_parc