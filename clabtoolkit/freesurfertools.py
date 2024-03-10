import os
import time
import subprocess

import numpy as np
import nibabel as nib


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

    def __init__(self, annot_file: str):


        self.annotfile = annot_file
        
        # Verify if the annot file exists
        if not os.path.exists(self.annotfile):
            raise ValueError("The annot file does not exist")
        
        # Extracting the filename, folder and name
        self.annotfolder = os.path.dirname(self.annotfile)
        self.annotname = os.path.basename(self.annotfile)

        # Detecting the hemisphere
        annot_name = self.annotname.lower()
        temp_name = annot_name.replace(".annot", "").lower()

        # Find in the string annot_name if it is lh. or rh.
        if "lh." in temp_name:
            hemi = "lh"
        elif "rh." in temp_name:
            hemi = "rh"
        elif "hemi-l" in temp_name:
            hemi = "lh"
        elif "hemi-r" in temp_name:
            hemi = "rh"
        else:
            hemi = None
            raise ValueError(
                "The hemisphere could not be extracted from the annot filename. Please provide it as an argument"
            )

        self.hemi = hemi

        # Read the annot file using nibabel
        codes, reg_table, reg_names = nib.freesurfer.io.read_annot(self.annotfile)

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
            out_file = os.path.join(self.annotfolder, self.annotname)

        # If the directory does not exist then create it
        temp_dir = os.path.dirname(out_file)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Save the annotation file
        nib.freesurfer.io.write_annot(
            out_file, self.codes, self.regtable, self.regnames
        )

    def _fill_parcellation(
        self, label_file: str, surf_file: str, corr_annot: str = None
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
            nib.freesurfer.write_annot(self.annotfile, vert_lab, reg_ctable, reg_names)
            corr_annot = self.annotfile

        return corr_annot, vert_lab, reg_ctable, reg_names

    def _annot2gcs(
        self,
        gcs_file: str = None,
        freesurfer_dir: str = None,
        fssubj_id: str = None,
        hemi: str = None,
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
            gcs_name = self.annotname.replace(".annot", ".gcs")

            # Create te gcs folder if it does not exist
            if gcs_folder is None:
                gcs_folder = self.annotfolder

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
            os.environ["SUBJECTS_DIR"] = freesurfer_dir
        else:
            if "SUBJECTS_DIR" not in os.environ:
                raise ValueError(
                    "The FreeSurfer directory must be set in the environment variables or passed as an argument"
                )
            else:
                freesurfer_dir = os.environ["SUBJECTS_DIR"]

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
        ctab_file = os.path.join(gcs_folder, self.annotname + ".ctab")
        with open(ctab_file, "w") as colorLUT_f:
            colorLUT_f.write("\n".join(luttable))

        # Detecting the hemisphere
        if hemi is None:
            hemi = self.hemi
            if hemi is None:
                raise ValueError(
                    "The hemisphere could not be extracted from the annot filename. Please provide it as an argument"
                )

        cmd_cont = [
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

        echo_var = " ".join(cmd_cont)
        print(echo_var)
        subprocess.run(
            cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
        )  # Running container command

        # Delete the ctab file
        os.remove(ctab_file)

        return gcs_name


def _create_fsaverage_links(
    fssubj_dir: str, fsavg_dir: str = None, refsubj_name: str = None
):
    """
    Create the links to the fsaverage folder
    @params:
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






