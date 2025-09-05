import os
import time
import subprocess
import sys
from glob import glob
from typing import List, Union
from pathlib import Path
from datetime import datetime
import warnings
import shutil
import json
import uuid
import numpy as np
import nibabel as nib
import pandas as pd
from collections import defaultdict

from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    TextColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)
from rich.console import Console
from rich.panel import Panel

# Importing local modules
from . import misctools as cltmisc
from . import bidstools as cltbids
from . import pipelinetools as cltpipe


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############              Section 1: Class to work with Annotation Files                ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
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

    ####################################################################################################
    def __init__(
        self,
        parc_file: str = None,
        annot_id: str = None,
        ref_surf: str = None,
        cont_tech: str = "local",
        cont_image: str = None,
    ):
        """
        Initialize the AnnotParcellation object

        Parameters
        ----------
        parc_file : str, optional
            Path to the parcellation file (annot, gii, or gcs format). If None, the object is initialized without loading any file.

        annot_id : str, optional
            Annotation ID to use for the parcellation. If None, uses the file name without extension.

        ref_surf : str, optional
            Reference surface for conversion. If None, uses the default FreeSurfer white surface.

        cont_tech : str, optional
            Container technology to use (e.g., 'local', 'docker'). Default is 'local'.

        cont_image : str, optional
            Container image to use. If None, uses the default for the specified container technology.


        """

        # If parc_file is provided, load it
        if parc_file is not None:
            if annot_id is None:
                annot_id = os.path.splitext(os.path.basename(parc_file))[0]

            self.filename = parc_file
            self.load_from_file(parc_file, annot_id, ref_surf, cont_tech, cont_image)

        else:
            self.id = annot_id
            self.filename = None
            self.path = None
            self.name = None
            self.hemi = None
            self.codes = None
            self.regtable = None
            self.regnames = None

    ####################################################################################################
    def load_from_file(
        self,
        parc_file: str,
        annot_id: str = None,
        ref_surf: str = None,
        cont_tech: str = "local",
        cont_image: str = None,
    ):
        """
        Load parcellation data from file

        Parameters
        ----------
        parc_file : str
            Path to the parcellation file (annot, gii, or gcs format)

        annot_id : str, optional
            Annotation ID to use for the parcellation. If None, uses the file name.

        ref_surf : str, optional
            Reference surface for conversion. If None, uses the default FreeSurfer white surface.

        cont_tech : str, optional
            Container technology to use (e.g., 'local', 'docker'). Default is 'local'.

        cont_image : str, optional
            Container image to use. If None, uses the default for the specified container technology.

        """
        booldel = False
        self.filename = parc_file

        # Verify if the file exists
        if not os.path.exists(self.filename):
            raise ValueError("The parcellation file does not exist")

        # Extracting the filename, folder and name
        self.path = os.path.dirname(self.filename)
        self.name = os.path.basename(self.filename)

        # If annot_id is provided, use it as the name
        if annot_id is not None:
            self.id = annot_id
        else:
            # If annot_id is not provided, use the file name without extension
            self.id = os.path.splitext(self.name)[0]

        # Detecting the hemisphere
        temp_name = self.name.lower()
        # Find in the string annot_name if it is lh. or rh.
        hemi = detect_hemi(temp_name)
        self.hemi = hemi

        # If the file is a .gii file, then convert it to a .annot file
        if self.name.endswith(".gii"):
            annot_file = AnnotParcellation.gii2annot(
                self.filename,
                ref_surf=ref_surf,
                annot_file=self.filename.replace(".gii", ".annot"),
                cont_tech=cont_tech,
                cont_image=cont_image,
            )
            booldel = True
        elif self.name.endswith(".annot"):
            annot_file = self.filename
        elif self.name.endswith(".gcs"):
            annot_file = AnnotParcellation.gcs2annot(
                self.filename, annot_file=self.filename.replace(".gcs", ".annot")
            )
            booldel = True

        # Read the annot file using nibabel
        codes, reg_table, reg_names = nib.freesurfer.read_annot(
            annot_file, orig_ids=True
        )

        if booldel:
            os.remove(annot_file)

        # Correcting region names
        reg_names = [name.decode("utf-8") for name in reg_names]

        # Detect the codes in the table that are not in the vertex wise data
        # Find the indexes where the codes are not in the vertex wise data
        tmp_ind = np.where(np.isin(reg_table[:, 4], np.unique(codes)) == False)[0]

        # If there are codes that are not in the vertex wise data, then remove them from the table
        if tmp_ind.size > 0:
            reg_table = np.delete(reg_table, tmp_ind, axis=0)
            reg_names = np.delete(reg_names, tmp_ind).tolist()

        # Storing the codes, colors and names in the object
        self.codes = codes
        self.regtable = reg_table
        self.regnames = reg_names

    ####################################################################################################
    def is_loaded(self):
        """Check if parcellation data has been loaded"""
        return self.codes is not None

    ####################################################################################################
    def create_from_data(
        self, codes, regtable, regnames, annot_id, hemi="lh", filename=None
    ):
        """
        Create parcellation from existing data arrays

        Parameters
        ----------
        codes : array-like
            Vertex-wise region codes

        regtable : array-like
            Region table with colors and codes

        regnames : list
            List of region names

        annot_id : str
            Annotation ID for the parcellation

        hemi : str, optional
            Hemisphere ('lh' or 'rh')

        filename : str, optional
            Associated filename
        """
        self.id = annot_id
        self.codes = codes
        self.regtable = regtable
        self.regnames = regnames
        self.hemi = hemi
        self.filename = filename
        if filename:
            self.path = os.path.dirname(filename)
            self.name = os.path.basename(filename)

    ####################################################################################################
    def save_annotation(self, out_file: str = None, force: bool = True):
        """
        Save the annotation file. If the file already exists, it will be overwritten.

        Parameters
        ----------
        out_file     - Optional  : Output annotation file:
        force        - Optional  : Force to overwrite the annotation file. Default is True:

        Returns
        -------



        """

        if out_file is None:
            out_file = os.path.join(self.path, self.name)

        # If the directory does not exist then create it
        temp_dir = os.path.dirname(out_file)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        if os.path.exists(out_file) and not force:
            raise ValueError(
                "The annotation file already exists. Set force to True to overwrite it."
            )
        elif os.path.exists(out_file) and force:
            os.remove(out_file)

        # Restructuring the codes to consecutive numbers
        new_codes = np.full_like(self.codes, -1, dtype=int)  # Specify dtype=int
        for i, code in enumerate(self.regtable[:, 4]):
            new_codes[self.codes == code] = i

        # Save the annotation file
        nib.freesurfer.io.write_annot(out_file, new_codes, self.regtable, self.regnames)

    ####################################################################################################
    def fill_parcellation(
        self, label_file: str, surf_file: str, corr_annot: str = None
    ):
        """
        Correct the parcellation by refilling vertices from the cortex label file
        that do not have a label in the annotation file.

        This method iteratively fills unlabeled vertices (those with value -1 or 0)
        by assigning the most frequent label from their neighboring vertices. The
        algorithm focuses on boundary vertices - those at the edges of the
        parcellation - and continues until all vertices within the cortex have
        been assigned labels or no further progress can be made.

        Parameters
        ----------
        label_file : str
            Path to the cortex label file containing vertex indices that define
            the cortical surface. This file typically has a .label extension and
            is used to mask vertices to only those within the cortex.

        surf_file : str
            Path to the surface geometry file containing vertex coordinates and
            face connectivity information. This is typically a FreeSurfer surface
            file (e.g., lh.pial, rh.white) that defines the mesh topology.

        corr_annot : str, optional
            Path where the corrected annotation file should be saved. If None,
            the original annotation filename will be used. The directory will be
            created if it doesn't exist. Default is None.

        Returns
        -------
        corr_annot : str
            Path to the corrected annotation file that was saved.

        vert_lab : numpy.ndarray
            Array of vertex labels after correction, where each element corresponds
            to the label assigned to that vertex index.

        reg_ctable : numpy.ndarray
            Region color table containing RGB color values for each region in the
            parcellation.

        reg_names : list
            List of region names corresponding to each label in the parcellation.
            May include "unknown" as the first element if unlabeled vertices were
            initially present.

        Raises
        ------
        ValueError
            If the surface file does not exist. Both annotation and surface files
            are mandatory for the parcellation correction process.

        ValueError
            If the cortex label file does not exist. The cortex label file is
            required to define which vertices should be considered for labeling.

        Notes
        -----
        The algorithm works by:

        1. Loading surface geometry and cortex label information
        2. Identifying vertices with missing labels (value -1, converted to 0)
        3. Finding boundary faces that contain both labeled and unlabeled vertices
        4. Iteratively assigning labels to unlabeled boundary vertices based on
           the most frequent label among their neighbors
        5. Repeating until no more vertices can be labeled or all vertices are filled

        The method modifies the internal state of the annotation object and
        optionally saves the result to a file.

        Examples
        --------
        Basic usage with an annotation object:

        >>> # Assuming 'annot' is an annotation object
        >>> label_file = '/path/to/lh.cortex.label'
        >>> surf_file = '/path/to/lh.pial'
        >>> output_file = '/path/to/lh.corrected.annot'
        >>>
        >>> corrected_path, labels, ctable, names = annot.fill_parcellation(
        ...     label_file=label_file,
        ...     surf_file=surf_file,
        ...     corr_annot=output_file
        ... )
        >>>
        >>> print(f"Corrected annotation saved to: {corrected_path}")
        >>> print(f"Number of regions: {len(names)}")

        Usage without specifying output file (uses original filename):

        >>> corrected_path, labels, ctable, names = annot.fill_parcellation(
        ...     label_file=label_file,
        ...     surf_file=surf_file
        ... )

        See Also
        --------
        nibabel.freesurfer.read_geometry : For reading surface files
        nibabel.freesurfer.read_label : For reading label files
        save_annotation : For saving annotation files
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
        # Find the indexes where vert_lab = -1
        tmp_ind = np.where(vert_lab == -1)[0]
        if tmp_ind.size > 0:
            addreg = True
            vert_lab[tmp_ind] = 0
        else:
            addreg = False

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

        if addreg and len(reg_names) != len(np.unique(vert_lab)):
            reg_names = ["unknown"] + reg_names

        # Save the annotation file
        if corr_annot is not None:
            if os.path.isfile(corr_annot):
                os.remove(corr_annot)

            # Create folder if it does not exist
            os.makedirs(os.path.dirname(corr_annot), exist_ok=True)
            self.filename = corr_annot
            self.codes = vert_lab
            self.regtable = reg_ctable
            self.regnames = reg_names

            self.save_annotation(out_file=corr_annot)
        else:
            corr_annot = self.filename

        return corr_annot, vert_lab, reg_ctable, reg_names

    ####################################################################################################
    def export_to_tsv(
        self, prefix2add: str = None, reg_offset: int = 1000, tsv_file: str = None
    ):
        """
        Export the parcellation table to a TSV file containing region information.

        Creates a comprehensive table with region indices, annotation IDs, parcellation IDs,
        names, and hexadecimal color codes. This is useful for creating lookup tables,
        documentation, or interfacing with other neuroimaging tools that require
        parcellation metadata in tabular format.

        Parameters
        ----------
        prefix2add : str, optional
            Prefix string to prepend to all region names. This is useful for
            distinguishing regions from different hemispheres (e.g., 'lh_' or 'rh_')
            or different atlases. If None, original region names are preserved.
            Default is None.

        reg_offset : int, optional
            Integer offset to add to the parcellation IDs. This helps avoid ID
            conflicts when combining multiple parcellations or when specific ID
            ranges are required. The offset is added to sequential indices starting
            from 0. Default is 1000.

        tsv_file : str, optional
            Path where the TSV file should be saved. If provided, the DataFrame
            will be written to this file with tab separation. The parent directory
            will be created if it doesn't exist. If None, no file is saved and
            only the DataFrame is returned. Default is None.

        Returns
        -------
        tsv_df : pandas.DataFrame
            DataFrame containing the parcellation table with the following columns:

            - 'index': Sequential indices starting from 0 for each region
            - 'annotid': Original annotation IDs from the FreeSurfer annotation file
            - 'parcid': Parcellation IDs (index + reg_offset)
            - 'name': Region names, optionally prefixed
            - 'color': Hexadecimal color codes (e.g., '#FF0000' for red)

        tsv_file : str
            Path to the saved TSV file if tsv_file parameter was provided,
            otherwise returns the input tsv_file parameter (which may be None).

        Raises
        ------
        PermissionError
            If the specified directory for the TSV file cannot be created due to
            insufficient write permissions. The method will print an error message
            and exit the program.

        Notes
        -----
        The method processes the internal parcellation data as follows:

        1. Converts RGB color values to hexadecimal format using the first 3 columns
        of the region table
        2. Applies optional prefix to region names using a name correction function
        3. Creates sequential indices and applies the specified offset to parcellation IDs
        4. Combines all information into a structured DataFrame
        5. Optionally saves to TSV format with tab separation and no row indices

        The annotation IDs correspond to the original FreeSurfer annotation values,
        while parcellation IDs provide a more standardized numbering scheme.

        Examples
        --------
        Basic usage - return DataFrame only:

        >>> # Assuming 'annot' is an annotation object
        >>> df, filename = annot.export_to_tsv()
        >>> print(df.head())
            index  annotid  parcid           name     color
        0        0     1234    1000    unknown   #000000
        1        1     5678    1001    precentral #FF0000
        2        2     9012    1002    postcentral #00FF00

        Add prefix to region names:

        >>> df, filename = annot.export_to_tsv(prefix2add='lh_')
        >>> print(df['name'].head())
        0        lh_unknown
        1        lh_precentral
        2        lh_postcentral

        Save to file with custom offset:

        >>> output_file = '/path/to/parcellation_table.tsv'
        >>> df, saved_path = annot.export_to_tsv(
        ...     prefix2add='rh_',
        ...     reg_offset=2000,
        ...     tsv_file=output_file
        ... )
        >>> print(f"Table saved to: {saved_path}")
        >>> print(f"Parcellation IDs range: {df['parcid'].min()}-{df['parcid'].max()}")

        Create hemisphere-specific tables:

        >>> # Left hemisphere
        >>> lh_df, lh_file = lh_annot.export_to_tsv(
        ...     prefix2add='lh_',
        ...     reg_offset=1000,
        ...     tsv_file='lh_parcellation.tsv'
        ... )
        >>>
        >>> # Right hemisphere
        >>> rh_df, rh_file = rh_annot.export_to_tsv(
        ...     prefix2add='rh_',
        ...     reg_offset=2000,
        ...     tsv_file='rh_parcellation.tsv'
        ... )
        >>>
        >>> # Combine both hemispheres
        >>> combined_df = pd.concat([lh_df, rh_df], ignore_index=True)

        See Also
        --------
        pandas.DataFrame.to_csv : For saving DataFrames in various formats
        cltmisc.multi_rgb2hex : For converting RGB values to hexadecimal
        cltmisc.correct_names : For applying prefixes to region names
        """

        # Creating the hexadecimal colors for the regions
        parc_hexcolor = cltmisc.multi_rgb2hex(self.regtable[:, 0:3])

        # Creating the region names
        parc_names = self.regnames
        if prefix2add is not None:
            parc_names = cltmisc.correct_names(parc_names, prefix=prefix2add)

        parc_index = np.arange(0, len(parc_names))

        # Selecting the Id in the annotation file
        annot_id = self.regtable[:, 4]

        parc_id = reg_offset + parc_index

        # Creating the dictionary for the tsv files
        tsv_df = pd.DataFrame(
            {
                "index": np.asarray(parc_index),
                "annotid": np.asarray(annot_id),
                "parcid": np.asarray(parc_id),
                "name": parc_names,
                "color": parc_hexcolor,
            }
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

    ####################################################################################################
    def map_values(
        self,
        regional_values: Union[str, pd.DataFrame, np.ndarray],
        is_dataframe: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Map regional values to vertex-wise values using the parcellation codes and region table.

        This method transforms region-level data (e.g., statistical measures, anatomical
        properties, or functional metrics computed per brain region) into vertex-level
        data for surface visualization. Each vertex is assigned the value corresponding
        to its parcellation region, enabling surface-based visualization of regional data.

        Parameters
        ----------
        regional_values : str or pandas.DataFrame or numpy.ndarray
            Regional values to map to vertices. Can be provided in three formats:

            - **str**: Path to a text/CSV file containing regional values. The file
            should have one row per region matching the parcellation. Single column
            files are read as arrays, while multi-column CSV files can optionally
            be read with headers when is_dataframe=True.

            - **pandas.DataFrame**: DataFrame with one row per region and columns
            representing different measures. Must have numeric values and column
            names. The number of rows must match the number of regions in the
            parcellation.

            - **numpy.ndarray**: Array with shape (n_regions, n_measures) where
            n_regions matches the number of regions in the parcellation. Can be
            1D for single measures or 2D for multiple measures.

        is_dataframe : bool, optional
            Whether to read text files as DataFrames with column headers. Only
            applies when regional_values is a file path (str). If True, the first
            row is treated as column names and preserved in the output. If False,
            the file is read without headers. Default is False.

        Returns
        -------
        vertex_wise_values : numpy.ndarray or pandas.DataFrame
            Vertex-wise values mapped from the regional values. The output format
            depends on the input:

            - **numpy.ndarray**: Returned when input is a numpy array or a file
            read without headers (is_dataframe=False). Shape is (n_vertices,)
            for single measures or (n_vertices, n_measures) for multiple measures.

            - **pandas.DataFrame**: Returned when input is a DataFrame or a file
            read with headers (is_dataframe=True). Has one row per vertex and
            preserves original column names.

        Raises
        ------
        ValueError
            If the regional values file does not exist (when input is a file path).

        ValueError
            If the number of rows in regional_values does not match the number of
            regions in the parcellation. This ensures proper mapping between regions
            and values.

        ValueError
            If the DataFrame contains non-numeric values. All regional values must
            be numeric for proper mapping to vertices.

        ValueError
            If regional_values is not one of the supported types (str, DataFrame,
            or ndarray).

        Notes
        -----
        The mapping process works as follows:

        1. **Input validation**: Checks that the input format is supported and that
        dimensions match the parcellation structure.

        2. **Format standardization**: Converts all input types to numpy arrays
        while preserving column information when applicable.

        3. **Vertex assignment**: Uses the internal `region_to_vertexwise` function
        to assign each vertex the value from its corresponding region based on
        the parcellation codes.

        4. **Output formatting**: Returns data in the same structural format as the
        input (array vs DataFrame) to maintain consistency.

        The parcellation codes (self.codes) define which region each vertex belongs
        to, and the region table (self.regtable) provides the mapping structure.
        Vertices with undefined regions (typically background or unknown areas)
        may receive special handling depending on the implementation of
        `region_to_vertexwise`.

        Examples
        --------
        Map single measure from numpy array:

        >>> import numpy as np
        >>> # Regional thickness values for each region
        >>> thickness_values = np.array([2.5, 3.1, 2.8, 3.2, 2.9])  # 5 regions
        >>> vertex_values = annot.map_values(thickness_values)
        >>> print(f"Mapped {len(vertex_values)} vertices")
        >>> print(f"Value range: {vertex_values.min():.2f} - {vertex_values.max():.2f}")

        Map multiple measures from DataFrame:

        >>> import pandas as pd
        >>> # Multiple regional measures
        >>> regional_data = pd.DataFrame({
        ...     'thickness': [2.5, 3.1, 2.8, 3.2, 2.9],
        ...     'volume': [1200, 1500, 1100, 1400, 1300],
        ...     'surface_area': [850, 920, 780, 900, 840]
        ... })
        >>> vertex_df = annot.map_values(regional_data)
        >>> print(f"Output columns: {list(vertex_df.columns)}")
        >>> print(f"Vertex data shape: {vertex_df.shape}")

        Read from CSV file without headers:

        >>> # File contains single column of regional values
        >>> vertex_values = annot.map_values('regional_thickness.txt')
        >>> print(f"Mapped values type: {type(vertex_values)}")

        Read from CSV file with headers:

        >>> # File contains multiple columns with headers
        >>> vertex_df = annot.map_values(
        ...     'regional_measures.csv',
        ...     is_dataframe=True
        ... )
        >>> print(f"Preserved columns: {list(vertex_df.columns)}")

        Typical neuroimaging workflow:

        >>> # Load regional statistics from analysis
        >>> regional_stats = pd.read_csv('region_statistics.csv')
        >>>
        >>> # Map to surface for visualization
        >>> surface_data = annot.map_values(regional_stats)
        >>>
        >>> # Save for visualization software
        >>> if isinstance(surface_data, pd.DataFrame):
        ...     surface_data.to_csv('surface_values.csv', index=False)
        ... else:
        ...     np.savetxt('surface_values.txt', surface_data)

        Error handling example:

        >>> try:
        ...     # This will fail if dimensions don't match
        ...     wrong_size = np.array([1, 2, 3])  # Only 3 values for 5 regions
        ...     vertex_values = annot.map_values(wrong_size)
        ... except ValueError as e:
        ...     print(f"Dimension mismatch: {e}")

        See Also
        --------
        region_to_vertexwise : Underlying function that performs the mapping
        pandas.read_csv : For reading CSV files with various options
        numpy.loadtxt : Alternative for reading simple numeric files
        """

        # Check if the regional values are a string (txt file)
        if isinstance(regional_values, str):
            # Check if the file exists
            if not os.path.exists(regional_values):
                raise ValueError("The regional values file does not exist")
            # Read the regional values as a pandas dataframe
            if is_dataframe:
                regional_values = pd.read_csv(
                    regional_values,
                    header=0,
                )
                col_names = regional_values.columns.tolist()
            else:
                regional_values = pd.read_csv(
                    regional_values,
                    header=None,
                )
                col_names = None
            regional_values = regional_values.to_numpy()

        elif isinstance(regional_values, pd.DataFrame):
            # Check if the number of rows matches the number of regions in the parcellation
            if regional_values.shape[0] != len(self.regtable):
                raise ValueError(
                    "The number of rows in the regional values does not match the number of regions in the parcellation"
                )
            else:
                # Check if the columns are numeric
                if not np.issubdtype(regional_values.dtypes[0], np.number):
                    raise ValueError("The regional values should be numeric")
                else:
                    # Convert the pandas dataframe to a numpy array
                    col_names = regional_values.columns.tolist()
                    regional_values = regional_values.to_numpy()
                    is_df = True

        elif isinstance(regional_values, np.ndarray):
            col_names = None
            # Check if the number of rows matches the number of regions in the parcellation
            if regional_values.shape[0] != len(self.regtable):
                raise ValueError(
                    "The number of rows in the regional values does not match the number of regions in the parcellation"
                )
        else:
            raise ValueError(
                "The regional values should be a pandas dataframe, a numpy array or a txt file"
            )

        vertex_wise_values = region_to_vertexwise(
            regional_values, self.codes, self.regtable
        )

        if col_names is not None:
            # If the regional values are a pandas dataframe, then create a dictionary with the column names and the vertex wise values
            vertex_wise_values = pd.DataFrame(vertex_wise_values, columns=col_names)
        else:
            # If the regional values are a numpy array, then create a numpy array with the vertex wise values
            vertex_wise_values = np.array(vertex_wise_values)

        return vertex_wise_values

    ####################################################################################################
    @staticmethod
    def gii2annot(
        gii_file: str,
        ref_surf: str = None,
        annot_file: str = None,
        cont_tech: str = "local",
        cont_image: str = None,
    ):
        """
        Convert FreeSurfer GIFTI parcellation files to annotation files using mris_convert.

        This method converts GIFTI label files (commonly used in neuroimaging pipelines
        like HCP, fMRIPrep) to FreeSurfer's native annotation format. The conversion
        enables use of external parcellations within FreeSurfer's ecosystem and
        visualization tools.

        Parameters
        ----------
        gii_file : str
            Path to the input GIFTI label file (.gii). The file should contain
            parcellation labels corresponding to surface vertices. The hemisphere
            is automatically detected from the filename.

        ref_surf : str, optional
            Path to the reference surface file used for the conversion. This surface
            should match the geometric space of the GIFTI file. If None, defaults
            to the white surface of the fsaverage subject from FREESURFER_HOME.
            Default is None.

        annot_file : str, optional
            Path for the output annotation file. If None, the output file is created
            in the same directory as the input file with the extension changed from
            .gii to .annot. Default is None.

        cont_tech : str, optional
            Container technology for running FreeSurfer commands. Options include
            'local' (run directly), 'singularity', 'docker', or other supported
            containerization methods. Default is 'local'.

        cont_image : str, optional
            Container image specification when using containerized execution.
            Required when cont_tech is not 'local'. Should specify the FreeSurfer
            container image (e.g., 'freesurfer/freesurfer:7.2.0'). Default is None.

        Returns
        -------
        annot_file : str
            Path to the created annotation file.

        Raises
        ------
        ValueError
            If the input GIFTI file does not exist.

        ValueError
            If FREESURFER_HOME environment variable is not set and no reference
            surface is provided.

        ValueError
            If the provided reference surface file does not exist.

        Notes
        -----
        This method uses FreeSurfer's `mris_convert` command with the `--annot` flag
        to perform the conversion. The command structure is:

        .. code-block:: bash

            mris_convert --annot input.gii reference_surface.surf output.annot

        The reference surface is crucial for proper conversion as it defines the
        vertex correspondence between the GIFTI labels and the annotation format.
        The method automatically detects the hemisphere from the GIFTI filename
        and selects the appropriate reference surface.

        Container execution allows running FreeSurfer tools without a local
        installation, which is useful in cloud environments or when FreeSurfer
        is not available locally.

        Examples
        --------
        Basic conversion with automatic reference surface:

        >>> # Convert HCP-style parcellation to FreeSurfer format
        >>> gii_file = '/path/to/lh.Schaefer2018_400Parcels.label.gii'
        >>> annot_file = AnnotParcellation.gii2annot(gii_file)
        >>> print(f"Created annotation: {annot_file}")

        Specify custom reference surface:

        >>> # Use subject-specific surface for conversion
        >>> gii_file = '/path/to/rh.custom_parcellation.gii'
        >>> ref_surf = '/path/to/subject/surf/rh.pial'
        >>> output_file = '/path/to/output/rh.custom.annot'
        >>>
        >>> result = AnnotParcellation.gii2annot(
        ...     gii_file=gii_file,
        ...     ref_surf=ref_surf,
        ...     annot_file=output_file
        ... )

        Using Docker container:

        >>> # Run conversion in FreeSurfer Docker container
        >>> annot_file = AnnotParcellation.gii2annot(
        ...     gii_file='parcellation.gii',
        ...     cont_tech='docker',
        ...     cont_image='freesurfer/freesurfer:7.2.0'
        ... )

        See Also
        --------
        annot2gii : Convert annotation files to GIFTI format
        mris_convert : FreeSurfer command-line tool for surface file conversion
        """

        if not os.path.exists(gii_file):
            raise ValueError("The gii file does not exist")

        if ref_surf is None:

            # Get freesurfer directory
            if "FREESURFER_HOME" in os.environ:
                freesurfer_dir = os.path.join(os.environ["FREESURFER_HOME"], "subjects")
                subj_id = "fsaverage"

                hemi = detect_hemi(gii_file)
                ref_surf = os.path.join(
                    freesurfer_dir, subj_id, "surf", hemi + ".white"
                )
            else:
                raise ValueError(
                    "Impossible to set the reference surface file. Please provide it as an argument"
                )

        else:
            if not os.path.exists(ref_surf):
                raise ValueError("The reference surface file does not exist")

        if annot_file is None:
            annot_file = os.path.join(
                os.path.dirname(gii_file),
                os.path.basename(gii_file).replace(".gii", ".annot"),
            )

        # Generating the bash command
        cmd_bashargs = ["mris_convert", "--annot", gii_file, ref_surf, annot_file]

        cmd_cont = cltmisc.generate_container_command(
            cmd_bashargs, cont_tech, cont_image
        )  # Generating container command
        subprocess.run(
            cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
        )  # Running container command

        return annot_file

    ####################################################################################################
    @staticmethod
    def annot2gii(
        annot_file: str,
        ref_surf: str = None,
        gii_file: str = None,
        cont_tech: str = "local",
        cont_image: str = None,
    ):
        """
        Convert FreeSurfer annotation files to GIFTI format using mris_convert.

        This method converts FreeSurfer's native annotation format to GIFTI label
        files, enabling use of FreeSurfer parcellations in other neuroimaging
        software and analysis pipelines that support GIFTI format (e.g., Connectome
        Workbench, CIFTI-based analyses).

        Parameters
        ----------
        annot_file : str
            Path to the input FreeSurfer annotation file (.annot). This file
            contains the parcellation labels and associated color/name information
            in FreeSurfer's binary format.

        ref_surf : str, optional
            Path to the reference surface file that corresponds to the annotation.
            This surface defines the vertex coordinates and topology. If None,
            defaults to the white surface of the fsaverage subject from
            FREESURFER_HOME. Default is None.

        gii_file : str, optional
            Path for the output GIFTI file. If None, the output file is created
            in the same directory as the input file with the extension changed
            from .annot to .gii. Default is None.

        cont_tech : str, optional
            Container technology for running FreeSurfer commands. Options include
            'local' (run directly), 'singularity', 'docker', or other supported
            containerization methods. Default is 'local'.

        cont_image : str, optional
            Container image specification when using containerized execution.
            Required when cont_tech is not 'local'. Should specify the FreeSurfer
            container image (e.g., 'freesurfer/freesurfer:7.2.0'). Default is None.

        Returns
        -------
        gii_file : str
            Path to the created GIFTI file.

        Raises
        ------
        ValueError
            If the input annotation file does not exist.

        ValueError
            If FREESURFER_HOME environment variable is not set and no reference
            surface is provided.

        ValueError
            If the provided reference surface file does not exist.

        Notes
        -----
        This method uses FreeSurfer's `mris_convert` command with the `--annot` flag
        to perform the conversion. The command structure is:

        .. code-block:: bash

            mris_convert --annot input.annot reference_surface.surf output.gii

        The GIFTI format is more widely supported across neuroimaging software
        packages and is part of the CIFTI specification used in large-scale
        neuroimaging projects. The conversion preserves the parcellation labels
        but may not retain all FreeSurfer-specific metadata.

        The hemisphere is automatically detected from the annotation filename
        to select the appropriate reference surface when using the default
        fsaverage surfaces.

        Examples
        --------
        Basic conversion with automatic reference surface:

        >>> # Convert FreeSurfer parcellation to GIFTI format
        >>> annot_file = '/path/to/lh.aparc.annot'
        >>> gii_file = AnnotParcellation.annot2gii(annot_file)
        >>> print(f"Created GIFTI: {gii_file}")

        Specify custom output location:

        >>> # Convert with custom output path
        >>> annot_file = '/path/to/rh.Destrieux.annot'
        >>> output_file = '/output/rh.Destrieux.label.gii'
        >>>
        >>> result = AnnotParcellation.annot2gii(
        ...     annot_file=annot_file,
        ...     gii_file=output_file
        ... )

        Using subject-specific surface:

        >>> # Use individual subject's surface
        >>> annot_file = '/subjects/sub001/label/lh.aparc.a2009s.annot'
        >>> ref_surf = '/subjects/sub001/surf/lh.white'
        >>>
        >>> gii_file = AnnotParcellation.annot2gii(
        ...     annot_file=annot_file,
        ...     ref_surf=ref_surf
        ... )

        Using Singularity container:

        >>> # Run conversion in Singularity container
        >>> gii_file = AnnotParcellation.annot2gii(
        ...     annot_file='parcellation.annot',
        ...     cont_tech='singularity',
        ...     cont_image='/path/to/freesurfer.sif'
        ... )

        See Also
        --------
        gii2annot : Convert GIFTI files to annotation format
        mris_convert : FreeSurfer command-line tool for surface file conversion
        """

        # Check if the annot file exists
        if not os.path.exists(annot_file):
            raise ValueError("The annot file does not exist")

        if ref_surf is None:

            # Get freesurfer directory
            if "FREESURFER_HOME" in os.environ:
                freesurfer_dir = os.path.join(os.environ["FREESURFER_HOME"], "subjects")
                subj_id = "fsaverage"

                hemi = detect_hemi(gii_file)
                ref_surf = os.path.join(
                    freesurfer_dir, subj_id, "surf", hemi + ".white"
                )
            else:
                raise ValueError(
                    "Impossible to set the reference surface file. Please provide it as an argument"
                )

        else:
            if not os.path.exists(ref_surf):
                raise ValueError("The reference surface file does not exist")

        if gii_file is None:
            gii_file = os.path.join(
                os.path.dirname(annot_file),
                os.path.basename(annot_file).replace(".annot", ".gii"),
            )

        if not os.path.exists(annot_file):
            raise ValueError("The annot file does not exist")

        if not os.path.exists(ref_surf):
            raise ValueError("The reference surface file does not exist")

        # Generating the bash command
        cmd_bashargs = ["mris_convert", "--annot", annot_file, ref_surf, gii_file]

        cmd_cont = cltmisc.generate_container_command(
            cmd_bashargs, cont_tech, cont_image
        )  # Generating container command
        subprocess.run(
            cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
        )  # Running container command

    ####################################################################################################
    @staticmethod
    def gcs2annot(
        gcs_file: str,
        annot_file: str = None,
        freesurfer_dir: str = None,
        ref_id: str = "fsaverage",
        cont_tech: str = "local",
        cont_image: str = None,
    ):
        """
        Convert FreeSurfer GCS (Gaussian Classifier Surface) files to annotation files.
        
        This method applies a trained GCS classifier to generate subject-specific 
        parcellations. GCS files contain statistical models trained on manual 
        parcellations that can be applied to new subjects to automatically generate 
        anatomically consistent region labels.
        
        Parameters
        ----------
        gcs_file : str
            Path to the input GCS classifier file. These files contain trained 
            Gaussian classifiers for automatic parcellation (e.g., aparc.gcs, 
            aparc.a2009s.gcs). The hemisphere is automatically detected from 
            the filename.
            
        annot_file : str, optional
            Path for the output annotation file. If None, the output file is 
            created in the same directory as the GCS file with the extension 
            changed from .gcs to .annot. Default is None.
            
        freesurfer_dir : str, optional
            Path to the FreeSurfer subjects directory containing the reference 
            subject data. If None, uses the SUBJECTS_DIR environment variable. 
            The directory will be created if it doesn't exist. Default is None.
            
        ref_id : str, optional
            Subject ID of the reference subject containing the surface files 
            needed for classification (sphere.reg, cortex.label, aseg.mgz). 
            Typically 'fsaverage' for standard space analysis. Default is 'fsaverage'.
            
        cont_tech : str, optional
            Container technology for running FreeSurfer commands. Options include 
            'local' (run directly), 'singularity', 'docker', or other supported 
            containerization methods. Default is 'local'.
            
        cont_image : str, optional
            Container image specification when using containerized execution. 
            Required when cont_tech is not 'local'. Should specify the FreeSurfer 
            container image (e.g., 'freesurfer/freesurfer:7.2.0'). Default is None.
        
        Returns
        -------
        annot_file : str
            Path to the created annotation file containing the classified 
            parcellation labels.
        
        Raises
        ------
        ValueError
            If the input GCS file does not exist.
            
        ValueError
            If neither freesurfer_dir is provided nor SUBJECTS_DIR environment 
            variable is set, and FREESURFER_HOME is also not available.
        
        Notes
        -----
        This method uses FreeSurfer's `mris_ca_label` command to apply the GCS 
        classifier. The command structure is:
        
        .. code-block:: bash
        
            mris_ca_label -l cortex.label -aseg aseg.mgz subject_id hemisphere \\
                        sphere.reg classifier.gcs output.annot
        
        The classification process requires several FreeSurfer files from the 
        reference subject:
        
        - **cortex.label**: Defines the cortical vertices to be labeled
        - **aseg.mgz**: Volumetric segmentation for spatial context
        - **sphere.reg**: Spherical surface registration for spatial normalization
        
        The method automatically manages the FreeSurfer environment by setting 
        the SUBJECTS_DIR variable and ensuring the necessary directory structure 
        exists. The hemisphere is detected from the GCS filename.
        
        GCS-based parcellation is particularly useful for:
        
        - Applying consistent parcellation schemes across subjects
        - Automated processing pipelines
        - Reproducing published parcellation protocols
        - Cross-study standardization
        
        Examples
        --------
        Basic GCS application with default settings:
        
        >>> # Apply Desikan-Killiany parcellation
        >>> gcs_file = '/path/to/lh.aparc.gcs'
        >>> annot_file = AnnotParcellation.gcs2annot(gcs_file)
        >>> print(f"Created parcellation: {annot_file}")
        
        Specify custom FreeSurfer directory:
        
        >>> # Use custom subjects directory
        >>> gcs_file = '/atlases/rh.aparc.a2009s.gcs'
        >>> fs_dir = '/data/freesurfer_subjects'
        >>> 
        >>> result = AnnotParcellation.gcs2annot(
        ...     gcs_file=gcs_file,
        ...     freesurfer_dir=fs_dir,
        ...     ref_id='fsaverage'
        ... )
        
        Apply to individual subject space:
        
        >>> # Use subject-specific reference
        >>> gcs_file = '/atlases/lh.aparc.gcs'
        >>> output_file = '/subjects/sub001/label/lh.aparc.annot'
        >>> 
        >>> annot_file = AnnotParcellation.gcs2annot(
        ...     gcs_file=gcs_file,
        ...     annot_file=output_file,
        ...     ref_id='sub001'  # Use subject's own surfaces
        ... )
        
        Using Docker for processing:
        
        >>> # Run classification in container
        >>> result = AnnotParcellation.gcs2annot(
        ...     gcs_file='parcellation.gcs',
        ...     freesurfer_dir='/data/subjects',
        ...     cont_tech='docker',
        ...     cont_image='freesurfer/freesurfer:7.2.0'
        ... )
        
        Batch processing multiple GCS files:
        
        >>> import glob
        >>> gcs_files = glob.glob('/atlases/*.gcs')
        >>> 
        >>> for gcs_file in gcs_files:
        ...     output_dir = '/parcellations'
        ...     basename = os.path.basename(gcs_file).replace('.gcs', '.annot')
        ...     output_file = os.path.join(output_dir, basename)
        ...     
        ...     AnnotParcellation.gcs2annot(
        ...         gcs_file=gcs_file,
        ...         annot_file=output_file,
        ...         freesurfer_dir='/data/freesurfer'
        ...     )
        
        See Also
        --------
        mris_ca_label : FreeSurfer command for applying GCS classifiers
        gii2annot : Convert GIFTI parcellations to annotation format
        annot2gii : Convert annotations to GIFTI format
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
                freesurfer_dir = os.path.join(os.environ["FREESURFER_HOME"], "subjects")
                ref_id = "fsaverage"
            else:
                raise ValueError(
                    "The FreeSurfer directory must be set in the environment variables or passed as an argument"
                )

        # Set freesurfer directory as subjects directory
        os.environ["SUBJECTS_DIR"] = freesurfer_dir

        hemi_cad = detect_hemi(gcs_file)

        if annot_file is None:
            annot_file = os.path.join(
                os.path.dirname(gcs_file),
                os.path.basename(gcs_file).replace(".gcs", ".annot"),
            )

        ctx_label = os.path.join(
            freesurfer_dir, ref_id, "label", hemi_cad + ".cortex.label"
        )
        aseg_presurf = os.path.join(freesurfer_dir, ref_id, "mri", "aseg.mgz")
        sphere_reg = os.path.join(
            freesurfer_dir, ref_id, "surf", hemi_cad + ".sphere.reg"
        )

        cmd_bashargs = [
            "mris_ca_label",
            "-l",
            ctx_label,
            "-aseg",
            aseg_presurf,
            ref_id,
            hemi_cad,
            sphere_reg,
            gcs_file,
            annot_file,
        ]

        cmd_cont = cltmisc.generate_container_command(
            cmd_bashargs, cont_tech, cont_image
        )  # Generating container command
        subprocess.run(
            cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
        )  # Running container command

        return annot_file

    ####################################################################################################
    def annot2tsv(self, tsv_file: str = None):
        """
        Save the annotation colort able a tab-separated values (TSV) file.

        This method exports the region-wise parcellation labels to a simple text
        format that can be easily read by other software packages or analysis
        scripts. Each line contains the label ID for the corresponding region.

        Parameters
        ----------
        tsv_file : str, optional
            Path for the output TSV file. If None, the file is saved in the same
            directory as the annotation file with the extension changed from
            .annot to .tsv. The directory will be created if it doesn't exist.
            Default is None.

        Returns
        -------
        tsv_file : str
            Path to the created TSV file.

        Notes
        -----
        The output TSV file contains one integer per line, where each line
        corresponds to a region index and the value represents the parcellation
        label assigned to that vertices of that region.

        This simple format is useful for:

        - Importing labels into custom analysis scripts
        - Interfacing with non-FreeSurfer neuroimaging software
        - Creating lightweight label files for data sharing
        - Debugging and manual inspection of parcellation assignments

        The method uses tab separation and integer formatting to ensure
        compatibility across different systems and software packages.

        Examples
        --------
        Save to default location:

        >>> # Annotation file: /data/lh.aparc.annot
        >>> # Output will be: /data/lh.aparc.tsv
        >>> tsv_path = AnnotParcellation.annot2tsv()
        >>> print(f"Labels saved to: {tsv_path}")

        Specify custom output path:

        >>> # Save to specific location
        >>> output_file = '/analysis/vertex_labels.tsv'
        >>> tsv_path = AnnotParcellation.annot2tsv(tsv_file=output_file)
        >>>
        >>> # Verify the output
        >>> import numpy as np
        >>> labels = np.loadtxt(tsv_path, dtype=int)
        >>> print(f"Loaded {len(labels)} vertex labels")
        >>> print(f"Unique labels: {np.unique(labels)}")

        Use in analysis pipeline:

        >>> # Convert annotation to TSV for external analysis
        >>> tsv_file = AnnotParcellation.annot2tsv()
        >>>
        >>> # Read in external software (e.g., R, MATLAB)
        >>> # R: labels <- read.table(tsv_file, header=FALSE)
        >>> # MATLAB: labels = readtable(tsv_file);

        Batch processing multiple annotations:

        >>> import glob
        >>> annot_files = glob.glob('/subjects/*/label/*.annot')
        >>>
        >>> for annot_file in annot_files:
        ...     annot = AnnotParcellation(annot_file)
        ...     tsv_path = AnnotParcellation.annot2tsv()
        ...     print(f"Converted: {annot_file} -> {tsv_path}")

        See Also
        --------
        numpy.savetxt : Function used internally for saving arrays
        export_to_tsv : Export comprehensive parcellation table with metadata
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

    ####################################################################################################
    def annot2gcs(
        self,
        gcs_file: str = None,
        freesurfer_dir: str = None,
        fssubj_id: str = None,
        hemi: str = None,
        cont_tech: str = "local",
        cont_image: str = None,
    ):
        """
        Convert FreeSurfer annotation files to GCS (Gaussian Classifier Surface) files.
        
        This method creates a trained Gaussian classifier from an existing manual 
        or semi-manual parcellation. The resulting GCS file can be applied to new 
        subjects to automatically generate parcellations with the same regional 
        definitions and boundaries as the training annotation.
        
        Parameters
        ----------
        gcs_file : str, optional
            Path for the output GCS classifier file. If None, the file is saved 
            in the same directory as the annotation file with the extension 
            changed from .annot to .gcs. Default is None.
            
        freesurfer_dir : str, optional
            Path to the FreeSurfer subjects directory containing the training 
            subject data. If None, uses the SUBJECTS_DIR environment variable. 
            The directory will be created if it doesn't exist. Default is None.
            
        fssubj_id : str, required
            Subject ID of the FreeSurfer subject to use for training the classifier. 
            This subject must have the required surface files (sphere.reg) and 
            should be the same subject from which the annotation was derived. 
            No default value - must be provided.
            
        hemi : str, optional
            Hemisphere specification ('lh' or 'rh'). If None, the hemisphere is 
            automatically detected from the annotation filename. Default is None.
            
        cont_tech : str, optional
            Container technology for running FreeSurfer commands. Options include 
            'local' (run directly), 'singularity', 'docker', or other supported 
            containerization methods. Default is 'local'.
            
        cont_image : str, optional
            Container image specification when using containerized execution. 
            Required when cont_tech is not 'local'. Should specify the FreeSurfer 
            container image (e.g., 'freesurfer/freesurfer:7.2.0'). Default is None.
        
        Returns
        -------
        gcs_name : str
            Filename (not full path) of the created GCS classifier file.
        
        Raises
        ------
        ValueError
            If SUBJECTS_DIR environment variable is not set and freesurfer_dir 
            is not provided.
            
        ValueError
            If fssubj_id is not provided (required parameter).
            
        ValueError
            If the FreeSurfer subject directory does not exist.
            
        ValueError
            If the required sphere.reg file is not found in the subject directory.
            
        ValueError
            If the hemisphere cannot be determined from the filename and is not 
            provided as a parameter.
        
        Notes
        -----
        This method uses FreeSurfer's `mris_ca_train` command to create the GCS 
        classifier. The process involves:
        
        1. **Color table creation**: Generates a temporary .ctab file with region 
        names and RGB color values from the annotation.
        
        2. **Classifier training**: Uses spherical surface registration and the 
        annotation labels to train Gaussian classifiers for each region.
        
        3. **Model output**: Creates a .gcs file containing the trained statistical 
        models that can be applied to new subjects.
        
        The command structure is:
        
        .. code-block:: bash
        
            mris_ca_train -n 2 -t color_table.ctab hemisphere sphere.reg \\
                        annotation.annot subject_id output.gcs
        
        Required FreeSurfer files for the training subject:
        
        - **sphere.reg**: Spherical surface registration for spatial normalization
        - **Proper directory structure**: Standard FreeSurfer subject organization
        
        The resulting GCS file can be used with the `gcs2annot` method to apply 
        the same parcellation scheme to new subjects automatically.
        
        Examples
        --------
        Basic GCS creation with required subject ID:
        
        >>> # Train classifier from manual parcellation
        >>> annot = Annotation('/data/sub001/label/lh.manual.annot')
        >>> gcs_name = AnnotParcellation.annot2gcs(fssubj_id='sub001')
        >>> print(f"Created classifier: {gcs_name}")
        
        Specify custom output location:
        
        >>> # Save GCS file to specific location
        >>> output_file = '/atlases/custom_parcellation.gcs'
        >>> gcs_name = AnnotParcellation.annot2gcs(
        ...     gcs_file=output_file,
        ...     fssubj_id='fsaverage',
        ...     freesurfer_dir='/data/freesurfer'
        ... )
        
        Create classifier from template subject:
        
        >>> # Use fsaverage as training template
        >>> annot = Annotation('/templates/fsaverage/label/rh.custom.annot')
        >>> gcs_name = AnnotParcellation.annot2gcs(
        ...     fssubj_id='fsaverage',
        ...     hemi='rh',
        ...     freesurfer_dir='/usr/local/freesurfer/subjects'
        ... )
        
        Using Docker for training:
        
        >>> # Train classifier in container environment
        >>> gcs_name = AnnotParcellation.annot2gcs(
        ...     fssubj_id='training_subject',
        ...     cont_tech='docker',
        ...     cont_image='freesurfer/freesurfer:7.2.0'
        ... )
        
        Complete workflow - train and apply:
        
        >>> # Step 1: Create GCS from manual annotation
        >>> manual_annot = AnnotParcellation('/manual/lh.expert_labels.annot')
        >>> gcs_file = '/classifiers/expert_parcellation.gcs'
        >>> 
        >>> gcs_name = manual_annot.annot2gcs(
        ...     gcs_file=gcs_file,
        ...     fssubj_id='template_subject'
        ... )
        >>> 
        >>> # Step 2: Apply to new subjects
        >>> for subject in ['sub002', 'sub003', 'sub004']:
        ...     output_annot = f'/results/{subject}/lh.expert_auto.annot'
        ...     AnnotParcellation.gcs2annot(
        ...         gcs_file=gcs_file,
        ...         annot_file=output_annot,
        ...         ref_id=subject
        ...     )
        
        Quality control after training:
        
        >>> # Verify the trained classifier works
        >>> test_output = '/tmp/test_application.annot'
        >>> AnnotParcellation.gcs2annot(
        ...     gcs_file='/atlases/new_classifier.gcs',
        ...     annot_file=test_output,
        ...     ref_id='fsaverage'
        ... )
        >>> 
        >>> # Compare with original
        >>> original = AnnotParcellation('/original/annotation.annot')
        >>> test_result = AnnotParcellation(test_output)
        >>> # Implement comparison logic...
        
        See Also
        --------
        gcs2annot : Apply GCS classifiers to generate annotations
        mris_ca_train : FreeSurfer command for training surface classifiers
        export_to_tsv : Export parcellation metadata for analysis
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
            raise ValueError("Please supply a valid subject ID.")

        # If the freesurfer subject directory does not exist, raise an error
        if not os.path.isdir(os.path.join(freesurfer_dir, fssubj_id)):
            raise ValueError(
                "The FreeSurfer subject directory for {} does not exist".format(
                    fssubj_id
                )
            )

        if not os.path.isfile(
            os.path.join(freesurfer_dir, fssubj_id, "surf", "sphere.reg")
        ):
            raise ValueError(
                "The FreeSurfer subject directory for {} does not contain the sphere.reg file".format(
                    fssubj_id
                )
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

        cmd_cont = cltmisc.generate_container_command(
            cmd_bashargs, cont_tech, cont_image
        )  # Generating container command
        subprocess.run(
            cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
        )  # Running container command

        # Delete the ctab file
        os.remove(ctab_file)

        return gcs_name

    ####################################################################################################
    def group_into_lobes(
        self,
        grouping: str = "desikan",
        lobes_json: str = None,
        out_annot: str = None,
        ctxprefix: str = None,
        force: bool = False,
    ):
        """
        Group parcellation regions into anatomical lobes for coarser-grained analysis.

        This method combines fine-grained brain regions into larger anatomical units
        (lobes) based on predefined or custom grouping schemes. This is useful for
        reducing dimensionality in analyses, creating simplified visualizations,
        or studying brain function at the lobar level.

        Parameters
        ----------
        grouping : str, optional
            Grouping method or scheme name to use for lobar organization. Built-in
            options include 'desikan' for Desikan-Killiany atlas grouping. Custom
            groupings can be specified when providing a lobes_json file.
            Default is 'desikan'.

        lobes_json : str, optional
            Path to a JSON file containing custom lobe definitions and region
            mappings. If None, uses the default grouping scheme called lobes.json.
            This file is located in the clabtoolkit package directory
            (e.g., clabtoolkit/config/lobes.json).

            Default is None.

        out_annot : str, optional
            Path where the new lobar annotation file should be saved. If None,
            the parcellation is only returned as an object without saving to disk.
            The output directory will be created if it doesn't exist. Default is None.

        ctxprefix : str, optional
            Prefix string to prepend to the lobe names. This is useful for
            distinguishing between hemispheres (e.g., 'lh_' or 'rh_') or different
            analysis contexts. If None, no prefix is added. Default is None.

        force : bool, optional
            Whether to overwrite existing output files. If False and the output
            file already exists, an error will be raised. If True, existing files
            will be overwritten without warning. Default is False.

        Returns
        -------
        lobar_parcellation : AnnotParcellation
            New AnnotParcellation object containing the lobar parcellation where original
            regions have been grouped into larger anatomical units. The object
            contains updated region names, codes, and color tables corresponding
            to the lobes.

        Raises
        ------
        FileExistsError
            If the output annotation file already exists and force=False.

        FileNotFoundError
            If the specified lobes_json file does not exist.

        ValueError
            If the JSON file format is invalid or missing required keys.

        KeyError
            If regions specified in the JSON file are not found in the current
            parcellation.

        Notes
        -----
        The grouping process works as follows:

        1. **Region mapping**: Original parcellation regions are mapped to their
        corresponding lobes based on the grouping scheme.

        2. **Label reassignment**: Vertex labels are updated to reflect the new
        lobar assignments rather than fine-grained regional assignments.

        3. **Color assignment**: New color table is created for the lobes, either
        from the JSON file specification or using default colors.

        4. **Metadata update**: Region names and identifiers are updated to
        reflect the lobar structure.

        The method is particularly useful for:

        - Simplifying complex parcellations for visualization
        - Reducing multiple comparisons in statistical analyses
        - Creating anatomically meaningful ROI groups
        - Cross-study comparisons at the lobar level
        - Educational and clinical applications

        Examples
        --------
        Basic lobar grouping with default scheme:

        >>> # Group Desikan-Killiany regions into standard lobes
        >>> lobar_parc = parc.group_into_lobes(grouping='desikan')
        >>> print(f"Original regions: {len(parc.regnames)}")
        >>> print(f"Lobar regions: {len(lobar_parc.regnames)}")
        >>> print(f"Lobe names: {lobar_parc.regnames}")

        Save lobar parcellation to file:

        >>> # Create and save lobar parcellation
        >>> output_file = '/results/lh.desikan_lobes.annot'
        >>> lobar_parc = parc.group_into_lobes(
        ...     grouping='desikan',
        ...     out_annot=output_file
        ... )
        >>> print(f"Lobar parcellation saved to: {output_file}")

        Use custom JSON grouping file:

        >>> # Create custom lobe definitions
        >>> custom_json = '/configs/custom_lobes.json'
        >>> lobar_parc = parc.group_into_lobes(
        ...     grouping='mylobes',
        ...     lobes_json=custom_json
        ... )


        Force overwrite existing files:

        >>> # Overwrite existing lobar parcellation
        >>> lobar_parc = parc.group_into_lobes(
        ...     grouping='desikan',
        ...     out_annot='/existing/file.annot',
        ...     force=True
        ... )


        See Also
        --------
        export_to_tsv : Export parcellation tables for external analysis
        map_values : Map regional values to surface vertices
        AnnotParcellation : Main class for parcellation handling
        """

        lobes_dict = load_lobes_json(lobes_json)

        if "lobes" not in lobes_dict.keys():
            lobes_dict = lobes_dict[grouping]

        # Lobes names
        lobe_names = list(lobes_dict["lobes"].keys())

        # Create the new parcellation
        new_codes = np.zeros_like(self.codes)
        orig_codes = np.zeros_like(self.codes)

        reg_codes = self.regtable[:, 4]

        # Create an empty numpy array to store the new table
        rgb = np.array([250, 250, 250])
        vert_val = rgb[0] + rgb[1] * 2**8 + rgb[2] * 2**16
        orig_codes += vert_val

        new_table = np.array([[rgb[0], rgb[1], rgb[2], 0, vert_val]])

        for i, lobe in enumerate(lobe_names):
            lobe_regions = lobes_dict["lobes"][lobe]
            lobe_colors = lobes_dict["colors"][lobe]

            rgb = cltmisc.hex2rgb(lobe_colors)

            # Detect the codes of the regions that belong to the lobe
            reg_indexes = cltmisc.get_indexes_by_substring(self.regnames, lobe_regions)

            if len(reg_indexes) != 0:
                reg_values = reg_codes[reg_indexes]
                vert_val = rgb[0] + rgb[1] * 2**8 + rgb[2] * 2**16
                orig_codes[np.isin(self.codes, reg_values) == True] = vert_val
                new_codes[np.isin(self.codes, reg_values) == True] = i + 1

                # Concatenate the new table
                new_table = np.concatenate(
                    (new_table, np.array([[rgb[0], rgb[1], rgb[2], 0, vert_val]])),
                    axis=0,
                )

        # Remove the first row
        # new_table = new_table[1:, :]
        self.codes = new_codes
        if ctxprefix is not None:
            self.regnames = ["unknown"] + cltmisc.correct_names(
                lobe_names, prefix=ctxprefix
            )
        else:
            self.regnames = ["unknown"] + lobe_names

        self.regtable = new_table
        self.name = ""
        self.path = ""

        # Saving the annot file
        if out_annot is not None:
            self.name = os.path.basename(out_annot)
            self.path = os.path.dirname(out_annot)

            if not os.path.exists(self.path):
                os.makedirs(self.path, exist_ok=True)

            if os.path.exists(out_annot) and not force:
                raise ValueError(
                    "The output annotation file already exists. Use the force option to overwrite it."
                )
            elif os.path.exists(out_annot) and force:
                os.remove(out_annot)

            # Save the annotation file
            self.save_annotation(out_file=out_annot)

        else:
            self.codes = orig_codes


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############          Section 2: Class to work with FreeSurfer subjects                 ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
class FreeSurferSubject:
    """
    A comprehensive class for managing and analyzing FreeSurfer subject data.

    This class provides methods to work with FreeSurfer subjects, including
    initialization of file structures, processing status checking, launching
    FreeSurfer commands, and extracting morphometric statistics.
    """

    ####################################################################################################
    def __init__(self, subj_id: str, subjs_dir: str = None):
        """
        Initialize the FreeSurferSubject object with subject ID and subjects directory.

        Creates organized dictionaries containing paths to all standard FreeSurfer
        outputs including MRI volumes, surface files, parcellations, and statistics.

        Parameters
        ----------
        subj_id : str
            FreeSurfer subject identifier matching the directory name in the
            FreeSurfer subjects directory.

        subjs_dir : str, optional
            Path to the FreeSurfer subjects directory. If None, uses the
            SUBJECTS_DIR environment variable. Directory will be created
            if it doesn't exist. Default is None.

        Attributes
        ----------
        subj_id : str
            The subject identifier.

        subjs_dir : str
            Path to the FreeSurfer subjects directory.

        fs_files : dict
            Nested dictionary containing organized paths to all FreeSurfer files
            organized by data type (mri, surf, stats) and hemisphere.

        Examples
        --------
        >>> subject = FreeSurferSubject('sub-001')
        >>> print(subject.fs_files['mri']['T1'])
        >>> lh_thickness = subject.fs_files['surf']['lh']['map']['thickness']
        """

        if subjs_dir is None:
            self.subjs_dir = os.environ.get("SUBJECTS_DIR")
        else:

            if not os.path.exists(subjs_dir):
                # Create the folder
                os.makedirs(subjs_dir, exist_ok=True)

                print(
                    f"Warning: Directory {subjs_dir} does not exist. It will be created."
                )

            self.subjs_dir = subjs_dir

        subj_dir = os.path.join(self.subjs_dir, subj_id)
        self.subj_id = subj_id

        # Generate a dictionary of the FreeSurfer files
        self.fs_files = {}
        mri_dict = {}
        mri_dict["orig"] = os.path.join(subj_dir, "mri", "orig.mgz")
        mri_dict["brainmask"] = os.path.join(subj_dir, "mri", "brainmask.mgz")
        mri_dict["T1"] = os.path.join(subj_dir, "mri", "T1.mgz")
        mri_dict["talairach"] = os.path.join(
            subj_dir, "mri", "transforms", "talairach.lta"
        )
        vol_parc_dict = {}
        vol_parc_dict["aseg"] = os.path.join(subj_dir, "mri", "aseg.mgz")
        vol_parc_dict["desikan+aseg"] = os.path.join(subj_dir, "mri", "aparc+aseg.mgz")
        vol_parc_dict["destrieux+aseg"] = os.path.join(
            subj_dir, "mri", "aparc.a2009s+aseg.mgz"
        )
        vol_parc_dict["dkt+aseg"] = os.path.join(
            subj_dir, "mri", "aparc.DKTatlas+aseg.mgz"
        )

        vol_parc_dict["ribbon"] = os.path.join(subj_dir, "mri", "ribbon.mgz")
        vol_parc_dict["wm"] = os.path.join(subj_dir, "mri", "wm.mgz")
        vol_parc_dict["wmparc"] = os.path.join(subj_dir, "mri", "wmparc.mgz")

        self.fs_files["mri"] = mri_dict
        self.fs_files["mri"]["parc"] = vol_parc_dict

        # Creating the Surf dictionary
        surf_dict = {}

        lh_s_dict, lh_m_dict, lh_p_dict, lh_t_dict = self.get_hemi_dicts(
            subj_dir=subj_dir, hemi="lh"
        )
        rh_s_dict, rh_m_dict, rh_p_dict, rh_t_dict = self.get_hemi_dicts(
            subj_dir=subj_dir, hemi="rh"
        )

        surf_dict["lh"] = {}
        surf_dict["lh"]["mesh"] = lh_s_dict
        surf_dict["lh"]["map"] = lh_m_dict
        surf_dict["lh"]["parc"] = lh_p_dict

        surf_dict["rh"] = {}
        surf_dict["rh"]["mesh"] = rh_s_dict
        surf_dict["rh"]["map"] = rh_m_dict
        surf_dict["rh"]["parc"] = rh_p_dict

        self.fs_files["surf"] = surf_dict

        # Creating the Stats dictionary
        stats_dict = {}
        global_dict = {}
        global_dict["aseg"] = os.path.join(subj_dir, "stats", "aseg.stats")
        global_dict["wmparc"] = os.path.join(subj_dir, "stats", "wmparc.stats")
        global_dict["brainvol"] = os.path.join(subj_dir, "stats", "brainvol.stats")
        stats_dict["global"] = global_dict
        stats_dict["lh"] = lh_t_dict
        stats_dict["rh"] = rh_t_dict

        self.fs_files["stats"] = stats_dict

    ####################################################################################################
    def get_hemi_dicts(self, subj_dir: str, hemi: str):
        """
        Create organized dictionaries containing hemisphere-specific FreeSurfer file paths.

        Helper method that constructs structured dictionaries for surface meshes,
        morphometric maps, parcellations, and statistics files for a specified hemisphere.

        Parameters
        ----------
        subj_dir : str
            Path to the FreeSurfer subject directory.

        hemi : str
            Hemisphere identifier ('lh' or 'rh').

        Returns
        -------
        s_dict : dict
            Surface mesh file paths (pial, white, inflated, sphere).

        m_dict : dict
            Morphometric map file paths (curv, sulc, thickness, area, volume, lgi).

        p_dict : dict
            Parcellation annotation file paths (desikan, destrieux, dkt).

        t_dict : dict
            Statistics file paths for each parcellation and curvature.

        Examples
        --------
        >>> surf, maps, parc, stats = subject.get_hemi_dicts(subj_dir, 'lh')
        >>> print(surf['pial'])
        """

        # Surface dictionary
        s_dict = {}
        s_dict["pial"] = os.path.join(subj_dir, "surf", hemi + ".pial")
        s_dict["white"] = os.path.join(subj_dir, "surf", hemi + ".white")
        s_dict["inflated"] = os.path.join(subj_dir, "surf", hemi + ".inflated")
        s_dict["sphere"] = os.path.join(subj_dir, "surf", hemi + ".sphere")
        m_dict = {}
        m_dict["curv"] = os.path.join(subj_dir, "surf", hemi + ".curv")
        m_dict["sulc"] = os.path.join(subj_dir, "surf", hemi + ".sulc")
        m_dict["thickness"] = os.path.join(subj_dir, "surf", hemi + ".thickness")
        m_dict["area"] = os.path.join(subj_dir, "surf", hemi + ".area")
        m_dict["volume"] = os.path.join(subj_dir, "surf", hemi + ".volume")
        m_dict["lgi"] = os.path.join(subj_dir, "surf", hemi + ".pial_lgi")
        p_dict = {}
        p_dict["desikan"] = os.path.join(subj_dir, "label", hemi + ".aparc.annot")
        p_dict["destrieux"] = os.path.join(
            subj_dir, "label", hemi + ".aparc.a2009s.annot"
        )
        p_dict["dkt"] = os.path.join(subj_dir, "label", hemi + ".aparc.DKTatlas.annot")

        # Statistics dictionary
        t_dict = {}
        t_dict["desikan"] = os.path.join(subj_dir, "stats", hemi + ".aparc.stats")
        t_dict["destrieux"] = os.path.join(
            subj_dir, "stats", hemi + ".aparc.a2009s.stats"
        )
        t_dict["dkt"] = os.path.join(subj_dir, "stats", hemi + ".aparc.DKTatlas.stats")
        t_dict["curv"] = os.path.join(subj_dir, "stats", hemi + ".curv.stats")

        return s_dict, m_dict, p_dict, t_dict

    ####################################################################################################
    def get_proc_status(self):
        """
        Check the FreeSurfer processing status for this subject.

        Evaluates which FreeSurfer processing stages have been completed by
        checking for the existence of key output files. Handles missing pial
        surface files by copying from pial.T1 files when needed.

        Attributes Set
        --------------
        pstatus : str
            Processing status: 'unprocessed', 'autorecon1', 'autorecon2', or 'processed'.

        Notes
        -----
        - 'unprocessed': No processing done
        - 'autorecon1': Basic preprocessing completed
        - 'autorecon2': Surface reconstruction completed
        - 'processed': Full processing including parcellation completed

        Examples
        --------
        >>> subject.get_proc_status()
        >>> print(f"Status: {subject.pstatus}")
        >>> if subject.pstatus == 'processed':
        ...     print("Ready for analysis")
        """

        # Check if the FreeSurfer subject id exists
        if not os.path.isdir(os.path.join(self.subjs_dir, self.subj_id)):
            pstatus = "unprocessed"
        else:

            # Check if the pial files exist because this file is missing in some FreeSurfer versions
            lh_pial = os.path.join(self.subjs_dir, self.subj_id, "surf", "lh.pial")
            lh_pial_t1 = os.path.join(
                self.subjs_dir, self.subj_id, "surf", "lh.pial.T1"
            )
            rh_pial = os.path.join(self.subjs_dir, self.subj_id, "surf", "rh.pial")
            rh_pial_t1 = os.path.join(
                self.subjs_dir, self.subj_id, "surf", "rh.pial.T1"
            )

            if os.path.isfile(lh_pial_t1) and not os.path.isfile(lh_pial):
                # Copy the lh.pial.T1 to lh.pial
                shutil.copy(lh_pial_t1, lh_pial)

            if os.path.isfile(rh_pial_t1) and not os.path.isfile(rh_pial):
                # Copy the rh.pial.T1 to rh.pial
                shutil.copy(rh_pial_t1, rh_pial)

            # Check the FreeSurfer files
            arecon1_files = [
                self.fs_files["mri"]["T1"],
                self.fs_files["mri"]["brainmask"],
                self.fs_files["mri"]["orig"],
            ]

            arecon2_files = [
                self.fs_files["mri"]["talairach"],
                self.fs_files["mri"]["parc"]["wm"],
                self.fs_files["surf"]["lh"]["mesh"]["pial"],
                self.fs_files["surf"]["rh"]["mesh"]["pial"],
                self.fs_files["surf"]["lh"]["mesh"]["white"],
                self.fs_files["surf"]["rh"]["mesh"]["white"],
                self.fs_files["surf"]["lh"]["mesh"]["inflated"],
                self.fs_files["surf"]["rh"]["mesh"]["inflated"],
                self.fs_files["surf"]["lh"]["map"]["curv"],
                self.fs_files["surf"]["rh"]["map"]["curv"],
                self.fs_files["surf"]["lh"]["map"]["sulc"],
                self.fs_files["surf"]["rh"]["map"]["sulc"],
                self.fs_files["stats"]["lh"]["curv"],
                self.fs_files["stats"]["rh"]["curv"],
            ]

            arecon3_files = [
                self.fs_files["mri"]["parc"]["aseg"],
                self.fs_files["mri"]["parc"]["desikan+aseg"],
                self.fs_files["mri"]["parc"]["destrieux+aseg"],
                self.fs_files["mri"]["parc"]["dkt+aseg"],
                self.fs_files["mri"]["parc"]["wmparc"],
                self.fs_files["mri"]["parc"]["ribbon"],
                self.fs_files["surf"]["lh"]["mesh"]["sphere"],
                self.fs_files["surf"]["rh"]["mesh"]["sphere"],
                self.fs_files["surf"]["lh"]["map"]["thickness"],
                self.fs_files["surf"]["rh"]["map"]["thickness"],
                self.fs_files["surf"]["lh"]["map"]["area"],
                self.fs_files["surf"]["rh"]["map"]["area"],
                self.fs_files["surf"]["lh"]["map"]["volume"],
                self.fs_files["surf"]["rh"]["map"]["volume"],
                self.fs_files["surf"]["lh"]["parc"]["desikan"],
                self.fs_files["surf"]["rh"]["parc"]["desikan"],
                self.fs_files["surf"]["lh"]["parc"]["destrieux"],
                self.fs_files["surf"]["rh"]["parc"]["destrieux"],
                self.fs_files["surf"]["lh"]["parc"]["dkt"],
                self.fs_files["surf"]["rh"]["parc"]["dkt"],
                self.fs_files["stats"]["lh"]["desikan"],
                self.fs_files["stats"]["rh"]["desikan"],
                self.fs_files["stats"]["lh"]["destrieux"],
                self.fs_files["stats"]["rh"]["destrieux"],
                self.fs_files["stats"]["lh"]["dkt"],
                self.fs_files["stats"]["rh"]["dkt"],
            ]

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
                pstatus = "processed"
            elif arecon2_bool and arecon1_bool and not arecon3_bool:
                pstatus = "autorecon2"
            elif arecon1_bool and not arecon2_bool and not arecon3_bool:
                pstatus = "autorecon1"
            else:
                pstatus = "unprocessed"

        self.pstatus = pstatus

    ####################################################################################################
    def launch_freesurfer(
        self,
        t1w_img: str = None,
        proc_stage: Union[str, list] = "all",
        extra_proc: Union[str, list] = None,
        cont_tech: str = "local",
        cont_image: str = None,
        fs_license: str = None,
        force=False,
    ):
        """
        Launch FreeSurfer recon-all processing with flexible options and containerization support.

        Provides interface for running FreeSurfer's recon-all pipeline with support
        for containerized execution, incremental processing, and additional modules.

        Parameters
        ----------
        t1w_img : str, optional
            Path to input T1-weighted MRI image. Required for unprocessed subjects.

        proc_stage : str or list, optional
            Processing stage(s): 'all', 'autorecon1', 'autorecon2', 'autorecon3',
            or list of stages. Default is 'all'.

        extra_proc : str or list, optional
            Additional modules: 'lgi', 'thalamus', 'brainstem', 'hippocampus',
            'amygdala', 'hypothalamus'. Default is None.

        cont_tech : str, optional
            Container technology: 'local', 'docker', 'singularity'. Default is 'local'.

        cont_image : str, optional
            Container image specification when using containerization.

        fs_license : str, optional
            Path to FreeSurfer license file for containers.

        force : bool, optional
            Force reprocessing even if outputs exist. Default is False.

        Returns
        -------
        proc_status : str
            Updated processing status after completion.

        Raises
        ------
        ValueError
            If invalid processing stages or missing required files.

        Examples
        --------
        >>> # Basic processing
        >>> status = subject.launch_freesurfer(t1w_img='/data/T1w.nii.gz')
        >>>
        >>> # With extra modules
        >>> status = subject.launch_freesurfer(
        ...     t1w_img='/data/T1w.nii.gz',
        ...     extra_proc=['lgi', 'hippocampus']
        ... )
        >>>
        >>> # Using Docker
        >>> status = subject.launch_freesurfer(
        ...     t1w_img='/data/T1w.nii.gz',
        ...     cont_tech='docker',
        ...     cont_image='freesurfer/freesurfer:7.2.0'
        ... )
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

        # For containerization
        mount_dirs = []
        if cont_tech == "singularity" or cont_tech == "docker":

            # Detecting the Subjects directorycontainer
            cmd_bashargs = ["echo", "$SUBJECTS_DIR"]
            cmd_cont = cltmisc.generate_container_command(
                cmd_bashargs, cont_tech, cont_image
            )  # Generating container command
            out_cmd = subprocess.run(
                cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
            )
            cont_subjs_dir = out_cmd.stdout.split("\n")[0]
            if cont_tech == "singularity":
                mount_dirs.append("--bind")
            elif cont_tech == "docker":
                mount_dirs.append("-v")
            mount_dirs.append(self.subjs_dir + ":" + cont_subjs_dir)

            # Detecting the Subjects directorycontainer
            cmd_bashargs = ["echo", "$FREESURFER_HOME"]
            cmd_cont = cltmisc.generate_container_command(
                cmd_bashargs, cont_tech, cont_image
            )  # Generating container command
            out_cmd = subprocess.run(
                cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
            )
            cont_license = os.path.join(out_cmd.stdout.split("\n")[0], "license.txt")
            if fs_license is not None:
                if cont_tech == "singularity":
                    mount_dirs.append("--bind")
                elif cont_tech == "docker":
                    mount_dirs.append("-v")
                mount_dirs.append(fs_license + ":" + cont_license)

        # Getting the freesurfer version
        ver_cad = get_version(cont_tech=cont_tech, cont_image=cont_image)
        ver_ent = ver_cad.split(".")
        vert_int = int("".join(ver_ent))

        if not hasattr(self, "pstatus"):
            self.get_proc_status()
        proc_status = self.pstatus

        # Check if the processing stage is valid
        val_stages = ["all", "autorecon1", "autorecon2", "autorecon3"]

        if isinstance(proc_stage, str):
            proc_stage = [proc_stage]

        proc_stage = [stage.lower() for stage in proc_stage]

        for stage in proc_stage:
            if stage not in val_stages:
                raise ValueError(f"Stage {stage} is not valid")

        if "all" in proc_stage:
            proc_stage = ["all"]

        # Check if the extra processing stages are valid
        val_extra_stages = [
            "lgi",
            "thalamus",
            "brainstem",
            "hippocampus",
            "amygdala",
            "hypothalamus",
        ]
        if extra_proc is not None:
            if isinstance(extra_proc, str):
                extra_proc = [extra_proc]

            # Put the extra processing stages in lower case
            extra_proc = [stage.lower() for stage in extra_proc]

            # If hippocampus and amygdala are in the list, remove amygdala from the list
            if "hippocampus" in extra_proc and "amygdala" in extra_proc:
                extra_proc.remove("amygdala")

            for stage in extra_proc:
                if stage not in val_extra_stages:
                    raise ValueError(f"Stage {stage} is not valid")

        if force:

            if t1w_img is None:
                if os.path.isdir(
                    os.path.join(self.subjs_dir, self.subj_id)
                ) and os.path.isfile(self.fs_files["mri"]["orig"]):
                    for st in proc_stage:
                        cmd_bashargs = ["recon-all", "-subjid", self.subj_id, "-" + st]
                        cmd_cont = cltmisc.generate_container_command(
                            cmd_bashargs, cont_tech, cont_image
                        )
                        subprocess.run(
                            cmd_cont,
                            stderr=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,
                            universal_newlines=True,
                        )  # Running container command
            else:
                if os.path.isfile(t1w_img):
                    for st in proc_stage:
                        cmd_bashargs = [
                            "recon-all",
                            "-subjid",
                            self.subj_id,
                            "-i",
                            t1w_img,
                            "-" + st,
                        ]
                        cmd_cont = cltmisc.generate_container_command(
                            cmd_bashargs, cont_tech, cont_image
                        )  # Generating container command
                        subprocess.run(
                            cmd_cont,
                            stderr=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,
                            universal_newlines=True,
                        )  # Running container command
                else:
                    raise ValueError("The T1w image does not exist")
        else:
            if proc_status == "unprocessed":
                if t1w_img is None:
                    if os.path.isdir(
                        os.path.join(self.subjs_dir, self.subj_id)
                    ) and os.path.isfile(self.fs_files["mri"]["orig"]):
                        cmd_bashargs = ["recon-all", "-subjid", self.subj_id, "-all"]
                else:
                    if os.path.isfile(t1w_img):
                        cmd_bashargs = [
                            "recon-all",
                            "-subjid",
                            self.subj_id,
                            "-i",
                            t1w_img,
                            "-all",
                        ]
                    else:
                        raise ValueError("The T1w image does not exist")

                cmd_bashargs = [
                    "recon-all",
                    "-i",
                    t1w_img,
                    "-subjid",
                    self.subj_id,
                    "-all",
                ]
                cmd_cont = cltmisc.generate_container_command(
                    cmd_bashargs, cont_tech, cont_image
                )  # Generating container command
                subprocess.run(
                    cmd_cont,
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    universal_newlines=True,
                )  # Running container command
            elif proc_status == "autorecon1":
                cmd_bashargs = ["recon-all", "-subjid", self.subj_id, "-autorecon2"]
                cmd_cont = cltmisc.generate_container_command(
                    cmd_bashargs, cont_tech, cont_image
                )  # Generating container command
                subprocess.run(
                    cmd_cont,
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    universal_newlines=True,
                )  # Running container command

                cmd_bashargs = ["recon-all", "-subjid", self.subj_id, "-autorecon3"]
                cmd_cont = cltmisc.generate_container_command(
                    cmd_bashargs, cont_tech, cont_image
                )  # Generating container command
                subprocess.run(
                    cmd_cont,
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    universal_newlines=True,
                )  # Running container command

            elif proc_status == "autorecon2":
                cmd_bashargs = ["recon-all", "-subjid", self.subj_id, "-autorecon3"]
                cmd_cont = cltmisc.generate_container_command(
                    cmd_bashargs, cont_tech, cont_image
                )
                subprocess.run(
                    cmd_cont,
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    universal_newlines=True,
                )  # Running container command

        self.get_proc_status()
        proc_status = self.pstatus

        # Processing extra stages
        if extra_proc is not None:
            if isinstance(extra_proc, str):
                extra_proc = [extra_proc]

            cmd_list = []
            for stage in extra_proc:
                if stage in val_extra_stages:
                    if stage == "lgi":  # Compute the local gyrification index

                        if (
                            not os.path.isfile(
                                self.fs_files["surf"]["lh"]["map"]["lgi"]
                            )
                            and not os.path.isfile(
                                self.fs_files["surf"]["rh"]["map"]["lgi"]
                            )
                        ) or force == True:
                            cmd_bashargs = [
                                "recon-all",
                                "-subjid",
                                self.subj_id,
                                "-lgi",
                            ]
                            cmd_list.append(cmd_bashargs)

                    elif (
                        stage == "thalamus"
                    ):  # Segment the thalamic nuclei using the thalamic nuclei segmentation tool

                        th_files = glob(
                            os.path.join(
                                self.subjs_dir, self.subj_id, "mri", "ThalamicNuclei.*"
                            )
                        )

                        if len(th_files) != 3 or force == True:
                            if vert_int < 730:
                                cmd_bashargs = [
                                    "segmentThalamicNuclei.sh",
                                    self.subj_id,
                                    self.subjs_dir,
                                ]
                            else:
                                cmd_bashargs = [
                                    "segment_subregions",
                                    "thalamus",
                                    "--cross",
                                    self.subj_id,
                                ]

                            cmd_list.append(cmd_bashargs)

                    elif stage == "brainstem":  # Segment the brainstem structures

                        bs_files = glob(
                            os.path.join(
                                self.subjs_dir, self.subj_id, "mri", "brainstemS*"
                            )
                        )

                        if len(bs_files) != 3 or force == True:
                            os.system("WRITE_POSTERIORS=1")
                            if vert_int < 730:
                                cmd_bashargs = [
                                    "segmentBS.sh",
                                    self.subj_id,
                                    self.subjs_dir,
                                ]
                            else:
                                cmd_bashargs = [
                                    "segment_subregions",
                                    "brainstem",
                                    "--cross",
                                    self.subj_id,
                                ]

                            cmd_list.append(cmd_bashargs)

                    elif (
                        stage == "hippocampus" or stage == "amygdala"
                    ):  # Segment the hippocampal subfields

                        ha_files = glob(
                            os.path.join(
                                self.subjs_dir,
                                self.subj_id,
                                "mri",
                                "*hippoAmygLabels.*",
                            )
                        )

                        if len(ha_files) != 16 or force == True:
                            if (
                                vert_int < 730
                            ):  # Use the FreeSurfer script for versions below 7.2.0
                                cmd_bashargs = [
                                    "segmentHA_T1.sh",
                                    self.subj_id,
                                    self.subjs_dir,
                                ]
                            else:
                                cmd_bashargs = [
                                    "segment_subregions",
                                    "hippo-amygdala",
                                    "--cross",
                                    self.subj_id,
                                ]

                            cmd_list.append(cmd_bashargs)

                    elif stage == "hypothalamus":  # Segment the hypothalamic subunits

                        hy_files = glob(
                            os.path.join(
                                self.subjs_dir,
                                self.subj_id,
                                "mri",
                                "hypothalamic_subunits*",
                            )
                        )
                        os.system("WRITE_POSTERIORS=1")
                        if len(hy_files) != 3 or force == True:
                            cmd_bashargs = [
                                "mri_segment_hypothalamic_subunits",
                                "--s",
                                self.subj_id,
                                "--sd",
                                self.subjs_dir,
                                "--write_posteriors",
                            ]
                            cmd_list.append(cmd_bashargs)

            if len(cmd_list) > 0:

                for cmd_bashargs in cmd_list:
                    cmd_cont = cltmisc.generate_container_command(
                        cmd_bashargs, cont_tech, cont_image
                    )
                    cmd_cont = cmd_cont[:2] + mount_dirs + cmd_cont[2:]

                    subprocess.run(
                        cmd_cont,
                        stderr=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        universal_newlines=True,
                    )  # Running container command

        return proc_status

    ####################################################################################################
    def create_stats_table(
        self,
        lobes_grouping: str = "desikan",
        add_bids_entities: bool = False,
        output_file: str = None,
    ) -> pd.DataFrame:
        """
        Generate comprehensive FreeSurfer statistics table combining morphometric measurements.

        Extracts and organizes cortical and volumetric measurements from FreeSurfer
        outputs into a structured DataFrame suitable for analysis.

        Parameters
        ----------
        lobes_grouping : str, optional
            Parcellation grouping method for lobar regions: 'desikan' or
            'desikan+cingulate'. Default is 'desikan'.

        add_bids_entities : bool, optional
            Whether to extract BIDS entities from subject ID. Default is False.

        output_file : str, optional
            Path to save the DataFrame as CSV. If None, not saved. Default is None.

        Returns
        -------
        pd.DataFrame
            Comprehensive statistics table with surface-based and volumetric
            measurements across multiple parcellation schemes.

        Attributes Set
        --------------
        stats_table : pd.DataFrame
            The generated statistics table stored as object attribute.

        Examples
        --------
        >>> # Generate basic stats table
        >>> stats_df = subject.create_stats_table()
        >>> print(f"Generated {len(stats_df)} measurements")
        >>>
        >>> # Save to file with BIDS entities
        >>> stats_df = subject.create_stats_table(
        ...     add_bids_entities=True,
        ...     output_file='/results/subject_stats.csv'
        ... )
        """

        from . import morphometrytools as morpho

        # Compute morphometric statistics
        lh_surf_df = self.surface_hemi_morpho(hemi="lh", lobes_grouping=lobes_grouping)
        rh_surf_df = self.surface_hemi_morpho(hemi="rh", lobes_grouping=lobes_grouping)
        vol_df = self.volume_morpho()

        # Adding the volumes extracted by FreeSurfer and stored at aseg.stats
        stats_df, _ = morpho.parse_freesurfer_stats_fromaseg(
            self.fs_files["stats"]["global"]["aseg"], add_bids_entities=False
        )
        stats_df.insert(4, "Atlas", "")

        # Parsing global metrics from aseg.mgz
        global_df, _ = morpho.parse_freesurfer_global_fromaseg(
            self.fs_files["stats"]["global"]["aseg"], add_bids_entities=False
        )
        global_df.insert(4, "Atlas", "")

        # Combine all data into a single DataFrame
        stats_table = pd.concat(
            [global_df, stats_df, vol_df, lh_surf_df, rh_surf_df], axis=0
        )

        # Adding the entities related to BIDs
        if add_bids_entities:
            ent_list = cltbids.entities4table(selected_entities=self.subj_id)

            df_add = cltbids.entities_to_table(
                filepath=self.subj_id, entities_to_extract=ent_list
            )

            stats_table = cltmisc.expand_and_concatenate(df_add, stats_table)
        else:
            # Expand a first dataframe and concatenate with the second dataframe
            stats_table.insert(0, "Participant", self.subj_id)

        # Adding the table as an attribute
        self.stats_table = stats_table

        # Save the DataFrame to a file if an output path is specified
        if output_file:
            stats_table.to_csv(output_file, index=False)
            print(f"Statistics table saved to: {output_file}")

        return stats_table

    ####################################################################################################
    def volume_morpho(
        self,
        parcellations: list = ["desikan+aseg", "destrieux+aseg", "dkt+aseg"],
        lobes_grouping: str = "desikan",
    ) -> pd.DataFrame:
        """
        Compute volume measurements from FreeSurfer volumetric parcellations.

        Extracts volumetric measurements from specified FreeSurfer parcellations
        and returns organized DataFrame with volume values per region.

        Parameters
        ----------
        parcellations : list, optional
            List of parcellation names to compute volumes from. Default includes
            Desikan-Killiany, Destrieux, and DKT atlases combined with subcortical
            segmentation.

        lobes_grouping : str, optional
            Grouping method for lobar segmentation: 'desikan' or 'desikan+cingulate'.
            Default is 'desikan'.

        Returns
        -------
        pd.DataFrame
            DataFrame with volume measurements organized by parcellation atlas
            and brain region.

        Examples
        --------
        >>> # Compute volumes for default parcellations
        >>> vol_df = subject.volume_morpho()
        >>> print(f"Computed volumes for {len(vol_df)} regions")
        >>>
        >>> # Custom parcellations
        >>> vol_df = subject.volume_morpho(
        ...     parcellations=['desikan+aseg', 'dkt+aseg']
        ... )
        """

        from . import parcellationtools as parc

        # Initialize an empty DataFrame for results
        df_vol = pd.DataFrame()

        # Iterate over each specified parcellation
        for volparc in parcellations:
            parc_file = self.fs_files["mri"]["parc"].get(volparc, None)
            if not parc_file or not os.path.isfile(parc_file):
                continue  # Skip missing parcellations

            # Load parcellation and compute volume table
            vol_parc = parc.Parcellation(parc_file=parc_file)
            vol_parc.load_colortable()
            vol_parc.compute_volume_table()
            df, _ = vol_parc.volumetable

            # Add identifying columns
            df.insert(4, "Atlas", volparc)

            # Concatenate results
            df_vol = pd.concat([df_vol, df], axis=0)

        nrows = df_vol.shape[0]

        return df_vol

    ####################################################################################################
    def surface_hemi_morpho(
        self, hemi: str = "lh", lobes_grouping: str = "desikan", verbose: bool = False
    ) -> pd.DataFrame:
        """
        Compute morphometric metrics for a hemisphere using cortical surface data.

        Extracts morphometric properties like thickness, surface area, and curvature
        from cortical surfaces, maps, and parcellations for the specified hemisphere.

        Parameters
        ----------
        hemi : str, optional
            Hemisphere to process: 'lh' or 'rh'. Default is 'lh'.

        lobes_grouping : str, optional
            Grouping method for lobar segmentation: 'desikan' or 'desikan+cingulate'.
            Default is 'desikan'.

        verbose : bool, optional
            Whether to print processing progress. Default is False.

        Returns
        -------
        pd.DataFrame
            DataFrame with morphometric measurements including thickness, area,
            curvature, and Euler characteristics organized by parcellation and region.

        Examples
        --------
        >>> # Process left hemisphere
        >>> lh_df = subject.surface_hemi_morpho(hemi='lh')
        >>> print(f"Extracted {len(lh_df)} morphometric measurements")
        >>>
        >>> # Process with verbose output
        >>> rh_df = subject.surface_hemi_morpho(
        ...     hemi='rh',
        ...     verbose=True
        ... )
        """

        from . import morphometrytools as morpho

        # Retrieve relevant FreeSurfer files
        parc_files_dict = self.fs_files["surf"][hemi]["parc"]
        metric_files_dict = self.fs_files["surf"][hemi]["map"]
        pial_surf = self.fs_files["surf"][hemi]["mesh"]["pial"]
        white_surf = self.fs_files["surf"][hemi]["mesh"]["white"]

        # Initialize DataFrame for results
        df_hemi = pd.DataFrame()

        # Process lobar parcellation
        desikan_parc = parc_files_dict.get("desikan", None)
        include_lobes = os.path.isfile(desikan_parc) if desikan_parc else False
        if include_lobes:
            # Print  Step 0: Grouping into lobes
            if verbose:
                print(" ")
                print("Step 0: Grouping into lobes")
                start_time = time.time()

            lobar_obj = AnnotParcellation(parc_file=desikan_parc)
            lobar_obj.group_into_lobes(grouping=lobes_grouping)

            # Compute the elapsed time and print it
            if verbose:
                elapsed_time = time.time() - start_time
                print(f"Elapsed time: {elapsed_time:.2f} seconds")

        # Iterate over parcellations
        if verbose:
            print(" ")
            print(
                f"Step 1: Computing morphometric metrics for the {hemi.upper()} hemisphere"
            )
        n_parc = len(parc_files_dict)
        cont_parc = 0
        for parc_name, parc_file in parc_files_dict.items():
            cont_parc += 1
            if not os.path.isfile(parc_file):
                continue  # Skip missing parcellations

            # Extract base name without extensions
            parc_base_name = ".".join(os.path.basename(parc_file).split(".")[1:-1])
            if parc_base_name == "aparc":
                parc_base_name = "desikan"
            elif parc_base_name == "aparc.a2009s":
                parc_base_name = "destrieux"
            elif parc_base_name == "aparc.DKTatlas":
                parc_base_name = "dkt"

            if verbose:
                print(" ")
                print(
                    f"    - Step 1.1: Surface-based metrics (Parcellation: {parc_base_name} [{cont_parc}/{n_parc}])"
                )

            df_metric = pd.DataFrame()

            # Compute mean thickness per region
            n_metrics = len(metric_files_dict)
            cont_metric = 0
            for metric_name, metric_file in metric_files_dict.items():
                if not os.path.isfile(metric_file):
                    continue

                cont_metric += 1
                if verbose:
                    print(
                        f"        - Metric: {metric_name} [{cont_metric}/{n_metrics+1}]"
                    )
                    start_time = time.time()

                # Compute lobar and regional metrics
                df_lobes, _, _ = morpho.compute_reg_val_fromannot(
                    metric_file,
                    lobar_obj,
                    hemi,
                    metric=metric_name,
                    add_bids_entities=False,
                )
                df_lobes.insert(4, "Atlas", f"lobes_{lobes_grouping}")

                df_region, _, _ = morpho.compute_reg_val_fromannot(
                    metric_file,
                    parc_file,
                    hemi,
                    metric=metric_name,
                    include_global=False,
                    add_bids_entities=False,
                )
                df_region.insert(4, "Atlas", parc_base_name)

                # Concatenate results
                df_metric = pd.concat([df_metric, df_lobes, df_region], axis=0)
                if verbose:
                    elapsed_time = time.time() - start_time
                    print(f"        Elapsed time: {elapsed_time:.2f} seconds")

            # Compute surface area and Euler characteristic for both pial and white surfaces
            df_parc = pd.DataFrame()
            df_e = pd.DataFrame()
            cont_metric += 1
            cont_euler = cont_metric + 1
            for surface, source_label in zip(
                [pial_surf, white_surf], ["pial", "white"]
            ):
                start_time = time.time()
                df_area_region, _ = morpho.compute_reg_area_fromsurf(
                    surface,
                    parc_file,
                    hemi,
                    include_global=False,
                    add_bids_entities=False,
                    surf_type=source_label,
                )

                df_area_region.insert(4, "Atlas", parc_base_name)

                df_area_lobes, _ = morpho.compute_reg_area_fromsurf(
                    surface,
                    lobar_obj,
                    hemi,
                    surf_type=source_label,
                    add_bids_entities=False,
                )
                df_area_lobes.insert(4, "Atlas", f"lobes_{lobes_grouping}")

                if verbose:
                    print(
                        f"        - Metric: {surface.capitalize()} Surface Area [{cont_metric}/{n_metrics+1}]"
                    )
                    elapsed_time = time.time() - start_time
                    print(f"        Elapsed time: {elapsed_time:.2f} seconds")

                # Compute Euler characteristic
                start_time = time.time()
                df_euler, _ = morpho.compute_euler_fromsurf(
                    surface, hemi, surf_type=source_label, add_bids_entities=False
                )

                if verbose:
                    print(
                        f"        - Metric: Euler Characteristic for {surface.capitalize()} Surface [{cont_euler}/{n_metrics+1}]"
                    )
                    elapsed_time = time.time() - start_time
                    print(f"        Elapsed time: {elapsed_time:.2f} seconds")

                df_euler.insert(4, "Atlas", "")

                # Concatenate all the results
                df_parc = pd.concat(
                    [df_parc, df_area_lobes, df_area_region, df_euler], axis=0
                )

            if verbose:
                print(" ")
                print(
                    f"    - Step 1.2: Metrics from Stats files (Parcellation: {parc_base_name})"
                )
                start_time = time.time()

            # Read the stats file
            stat_file = self.fs_files["stats"][hemi][parc_name]

            df_stats_cortex, _ = morpho.parse_freesurfer_cortex_stats(
                stat_file, add_bids_entities=False
            )
            if verbose:
                elapsed_time = time.time() - start_time
                print(f"        Elapsed time: {elapsed_time:.2f} seconds")

            df_stats_cortex.insert(4, "Atlas", parc_base_name)

            # Merge morphometric and area metrics
            if not df_metric.empty:
                df_parc = pd.concat([df_parc, df_metric], axis=0)

            if not df_parc.empty:
                df_hemi = pd.concat([df_hemi, df_parc, df_stats_cortex], axis=0)

        return df_hemi

    ####################################################################################################
    @staticmethod
    def set_freesurfer_directory(fs_dir: str = None):
        """
        Set up the FreeSurfer subjects directory and configure environment variables.

        Creates the FreeSurfer directory if it doesn't exist and sets the SUBJECTS_DIR
        environment variable. Used to ensure proper FreeSurfer environment setup.

        Parameters
        ----------
        fs_dir : str, optional
            Path to FreeSurfer subjects directory. If None, extracts from SUBJECTS_DIR
            environment variable. Directory will be created if it doesn't exist.
            Default is None.

        Returns
        -------
        None
            The directory path is set in the SUBJECTS_DIR environment variable.

        Raises
        ------
        ValueError
            If fs_dir is None and SUBJECTS_DIR environment variable is not set.

        Examples
        --------
        >>> # Use environment variable
        >>> FreeSurferSubject.set_freesurfer_directory()
        >>>
        >>> # Set custom directory
        >>> FreeSurferSubject.set_freesurfer_directory('/data/freesurfer')
        >>> print(os.environ['SUBJECTS_DIR'])
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
        fs_dir = Path(fs_dir)
        fs_dir.mkdir(parents=True, exist_ok=True)
        os.environ["SUBJECTS_DIR"] = str(fs_dir)

    ####################################################################################################
    def annot2ind(
        self,
        ref_id: str,
        hemi: str,
        fs_annot: str,
        ind_annot: str,
        cont_tech: str = "local",
        cont_image: str = None,
        force=False,
        verbose=False,
    ):
        """
        Map annotation parcellation from reference space to individual subject space.

        Uses FreeSurfer's mri_surf2surf to transfer parcellation labels from a reference
        subject to the individual subject's surface, followed by gap-filling to ensure
        complete cortical coverage.

        Parameters
        ----------
        ref_id : str
            FreeSurfer subject ID of the reference subject containing the source
            annotation file.

        hemi : str
            Hemisphere identifier: 'lh' or 'rh'.

        fs_annot : str
            Path to source annotation file or basename. Can be full path or just
            the annotation name (e.g., 'aparc'). Also accepts GIFTI files (.gii).

        ind_annot : str
            Path for output annotation file in individual subject space.

        cont_tech : str, optional
            Container technology: 'local', 'docker', 'singularity'. Default is 'local'.

        cont_image : str, optional
            Container image specification when using containerization. Default is None.

        force : bool, optional
            Force processing even if output file exists. Default is False.

        verbose : bool, optional
            Print verbose messages about processing status. Default is False.

        Returns
        -------
        ind_annot : str
            Path to the created individual space annotation file.

        Raises
        ------
        FileNotFoundError
            If the source annotation file cannot be found in expected locations.

        Notes
        -----
        The method performs the following steps:
        1. Converts GIFTI to annotation format if needed
        2. Uses mri_surf2surf for surface-to-surface mapping
        3. Applies gap-filling to ensure complete cortical labeling
        4. Handles containerized execution with proper volume mounting

        Examples
        --------
        >>> # Map Desikan-Killiany parcellation
        >>> output_file = subject.annot2ind(
        ...     ref_id='fsaverage',
        ...     hemi='lh',
        ...     fs_annot='aparc',
        ...     ind_annot='/output/lh.aparc.individual.annot'
        ... )
        >>>
        >>> # Using Docker container
        >>> output_file = subject.annot2ind(
        ...     ref_id='fsaverage',
        ...     hemi='rh',
        ...     fs_annot='/path/to/custom.annot',
        ...     ind_annot='/output/rh.custom.individual.annot',
        ...     cont_tech='docker',
        ...     cont_image='freesurfer/freesurfer:7.2.0'
        ... )
        """

        if not os.path.isfile(fs_annot) and not os.path.isfile(
            os.path.join(
                self.subjs_dir, ref_id, "label", hemi + "." + fs_annot + ".annot"
            )
        ):
            raise FileNotFoundError(
                f"Files {fs_annot} or {os.path.join(self.subjs_dir, ref_id, 'label', hemi + '.' + fs_annot + '.annot')} do not exist"
            )

        if fs_annot.endswith(".gii"):
            tmp_annot = fs_annot.replace(".gii", ".annot")
            tmp_refsurf = os.path.join(
                self.subjs_dir, ref_id, "surf", hemi + ".inflated"
            )

            AnnotParcellation.gii2annot(
                gii_file=fs_annot,
                ref_surf=tmp_refsurf,
                annot_file=tmp_annot,
                cont_tech=cont_tech,
                cont_image=cont_image,
            )
            fs_annot = tmp_annot

        if not os.path.isfile(ind_annot) or force:

            FreeSurferSubject.set_freesurfer_directory(self.subjs_dir)

            # Create the folder if it does not exist
            temp_dir = os.path.dirname(ind_annot)
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)

            if cont_tech != "local":
                cmd_bashargs = ["echo", "$SUBJECTS_DIR"]
                cmd_cont = cltmisc.generate_container_command(
                    cmd_bashargs, cont_tech, cont_image
                )  # Generating container command
                out_cmd = subprocess.run(
                    cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
                )
                subjs_dir_cont = out_cmd.stdout.split("\n")[0]
                dir_cad = self.subjs_dir + ":" + subjs_dir_cont

            # Moving the Annot to individual space
            cmd_bashargs = [
                "mri_surf2surf",
                "--srcsubject",
                ref_id,
                "--trgsubject",
                self.subj_id,
                "--hemi",
                hemi,
                "--sval-annot",
                fs_annot,
                "--tval",
                ind_annot,
            ]
            cmd_cont = cltmisc.generate_container_command(
                cmd_bashargs, cont_tech, cont_image
            )  # Generating container command

            # Bind the FreeSurfer subjects directory
            if cont_tech == "singularity":
                cmd_cont.insert(2, "--bind")
                cmd_cont.insert(3, dir_cad)
            elif cont_tech == "docker":
                cmd_cont.insert(2, "-v")
                cmd_cont.insert(3, dir_cad)

            subprocess.run(
                cmd_cont,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                universal_newlines=True,
            )  # Running container command

            # Correcting the parcellation file in order to refill the parcellation with the correct labels
            cort_parc = AnnotParcellation(parc_file=ind_annot)
            label_file = os.path.join(
                self.subjs_dir, self.subj_id, "label", hemi + ".cortex.label"
            )
            surf_file = os.path.join(
                self.subjs_dir, self.subj_id, "surf", hemi + ".inflated"
            )
            cort_parc.fill_parcellation(
                corr_annot=ind_annot, label_file=label_file, surf_file=surf_file
            )

        elif os.path.isfile(ind_annot) and not force:
            # Print a message
            if verbose:
                print(
                    f"File {ind_annot} already exists. Use force=True to overwrite it"
                )

        return ind_annot

    ####################################################################################################
    def gcs2ind(
        self,
        fs_gcs: str,
        ind_annot: str,
        hemi: str,
        cont_tech: str = "local",
        cont_image: str = None,
        force=False,
        verbose=False,
    ):
        """
        Apply GCS classifier to generate individual subject parcellation.

        Uses FreeSurfer's mris_ca_label to apply a trained Gaussian Classifier Surface
        (GCS) file to the individual subject, creating subject-specific parcellation
        followed by gap-filling for complete coverage.

        Parameters
        ----------
        fs_gcs : str
            Path to the FreeSurfer GCS (Gaussian Classifier Surface) file containing
            the trained classifier model.

        ind_annot : str
            Path for output annotation file in individual subject space.

        hemi : str
            Hemisphere identifier: 'lh' or 'rh'.

        cont_tech : str, optional
            Container technology: 'local', 'docker', 'singularity'. Default is 'local'.

        cont_image : str, optional
            Container image specification when using containerization. Default is None.

        force : bool, optional
            Force processing even if output file exists. Default is False.

        verbose : bool, optional
            Print verbose messages about processing status. Default is False.

        Returns
        -------
        ind_annot : str
            Path to the created individual space annotation file.

        Notes
        -----
        The method performs the following steps:
        1. Uses mris_ca_label with cortex label and sphere registration
        2. Applies the GCS classifier to generate parcellation labels
        3. Performs gap-filling to ensure complete cortical coverage
        4. Handles containerized execution with proper volume mounting

        Requires the following FreeSurfer files for the individual subject:
        - cortex.label: Defines cortical vertices
        - sphere.reg: Spherical surface registration
        - Standard FreeSurfer directory structure

        Examples
        --------
        >>> # Apply Desikan-Killiany GCS classifier
        >>> output_file = subject.gcs2ind(
        ...     fs_gcs='/atlases/lh.aparc.gcs',
        ...     ind_annot='/output/lh.aparc.individual.annot',
        ...     hemi='lh'
        ... )
        >>>
        >>> # Force reprocessing with custom classifier
        >>> output_file = subject.gcs2ind(
        ...     fs_gcs='/custom/rh.custom_atlas.gcs',
        ...     ind_annot='/output/rh.custom.individual.annot',
        ...     hemi='rh',
        ...     force=True,
        ...     verbose=True
        ... )
        """

        if not os.path.isfile(ind_annot) or force:

            FreeSurferSubject.set_freesurfer_directory(self.subjs_dir)

            # Create the folder if it does not exist
            temp_dir = os.path.dirname(ind_annot)
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)

            if cont_tech != "local":
                cmd_bashargs = ["echo", "$SUBJECTS_DIR"]
                cmd_cont = cltmisc.generate_container_command(
                    cmd_bashargs, cont_tech, cont_image
                )  # Generating container command
                out_cmd = subprocess.run(
                    cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
                )
                subjs_dir_cont = out_cmd.stdout.split("\n")[0]
                dir_cad = self.subjs_dir + ":" + subjs_dir_cont

            # Moving the GCS to individual space
            cort_file = os.path.join(
                self.subjs_dir, self.subj_id, "label", hemi + ".cortex.label"
            )
            sph_file = os.path.join(
                self.subjs_dir, self.subj_id, "surf", hemi + ".sphere.reg"
            )

            cmd_bashargs = [
                "mris_ca_label",
                "-l",
                cort_file,
                self.subj_id,
                hemi,
                sph_file,
                fs_gcs,
                ind_annot,
            ]

            cmd_cont = cltmisc.generate_container_command(
                cmd_bashargs, cont_tech, cont_image
            )  # Generating container command

            # Bind the FreeSurfer subjects directory
            if cont_tech == "singularity":
                cmd_cont.insert(2, "--bind")
                cmd_cont.insert(3, dir_cad)
            elif cont_tech == "docker":
                cmd_cont.insert(2, "-v")
                cmd_cont.insert(3, dir_cad)

            subprocess.run(
                cmd_cont,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                universal_newlines=True,
            )  # Running container command

            # Correcting the parcellation file in order to refill the parcellation with the correct labels
            cort_parc = AnnotParcellation(parc_file=ind_annot)
            label_file = os.path.join(
                self.subjs_dir, self.subj_id, "label", hemi + ".cortex.label"
            )
            surf_file = os.path.join(
                self.subjs_dir, self.subj_id, "surf", hemi + ".inflated"
            )
            cort_parc.fill_parcellation(
                corr_annot=ind_annot, label_file=label_file, surf_file=surf_file
            )

        elif os.path.isfile(ind_annot) and not force:
            # Print a message
            if verbose:
                print(
                    f"File {ind_annot} already exists. Use force=True to overwrite it"
                )

        return ind_annot

    ####################################################################################################
    def surf2vol(
        self,
        atlas: str,
        out_vol: str,
        gm_grow: Union[int, str] = "0",
        color_table: Union[list, str] = None,
        bool_native: bool = False,
        bool_mixwm: bool = False,
        cont_tech: str = "local",
        cont_image: str = None,
        force: bool = False,
        verbose: bool = False,
    ):
        """
        Create volumetric parcellation from surface annotation files.

        Converts surface-based parcellation annotations to volumetric format,
        with options for label growing, white matter mixing, and coordinate
        space selection.

        Parameters
        ----------
        atlas : str
            Atlas identifier or name for the surface parcellation to convert.
            Should correspond to available annotation files for both hemispheres.

        out_vol : str
            Path for output volumetric parcellation file.

        gm_grow : int or str, optional
            Amount in millimeters to grow gray matter labels into surrounding
            tissue. Can help fill gaps between surface and volume. Default is "0".

        color_table : list or str, optional
            Format(s) for saving color lookup table: 'tsv', 'lut', or list
            of both ['tsv', 'lut']. Default is None (no color table saved).

        bool_native : bool, optional
            If True, output parcellation in native subject space. If False,
            uses FreeSurfer's standard space. Default is False.

        bool_mixwm : bool, optional
            Mix cortical white matter growing with gray matter labels.
            Extends cortical labels into white matter regions. Default is False.

        cont_tech : str, optional
            Container technology: 'local', 'docker', 'singularity'. Default is 'local'.

        cont_image : str, optional
            Container image specification when using containerization. Default is None.

        force : bool, optional
            Force processing even if output file exists. Default is False.

        verbose : bool, optional
            Print verbose messages about processing progress. Default is False.

        Returns
        -------
        out_vol : str
            Path to the created volumetric parcellation file.

        Notes
        -----
        This method is particularly useful for:
        - Creating volumetric ROIs from surface parcellations
        - Bridging surface-based and volume-based analyses
        - Generating masks for volume-based connectivity analysis
        - Converting surface atlases to volume format for other software

        The process typically involves projecting surface labels onto the
        volumetric space, with optional growing and white matter integration
        to ensure comprehensive tissue coverage.

        Examples
        --------
        >>> # Basic surface-to-volume conversion
        >>> vol_file = subject.surf2vol(
        ...     atlas='aparc',
        ...     out_vol='/output/aparc_volume.mgz'
        ... )
        >>>
        >>> # With gray matter growing and color table
        >>> vol_file = subject.surf2vol(
        ...     atlas='aparc.a2009s',
        ...     out_vol='/output/destrieux_volume.mgz',
        ...     gm_grow="2",
        ...     color_table=['tsv', 'lut'],
        ...     bool_mixwm=True
        ... )
        >>>
        >>> # Native space output
        >>> vol_file = subject.surf2vol(
        ...     atlas='custom_atlas',
        ...     out_vol='/output/custom_native.nii.gz',
        ...     bool_native=True,
        ...     force=True
        ... )
        """
        from . import parcellationtools as cltparc

        FreeSurferSubject.set_freesurfer_directory(self.subjs_dir)

        if cont_tech != "local":
            cmd_bashargs = ["echo", "$SUBJECTS_DIR"]
            cmd_cont = cltmisc.generate_container_command(
                cmd_bashargs, cont_tech, cont_image
            )  # Generating container command
            out_cmd = subprocess.run(
                cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
            )
            subjs_dir_cont = out_cmd.stdout.split("\n")[0]
            dir_cad = self.subjs_dir + ":" + subjs_dir_cont

        if isinstance(gm_grow, int):
            gm_grow = str(gm_grow)

        if color_table is not None:
            if isinstance(color_table, str):
                color_table = [color_table]

            if not isinstance(color_table, list):
                raise ValueError(
                    "color_table must be a list or a string with its elements equal to tsv or lut"
                )

            # Check if the elements of the list are tsv or lut. If the elements are not tsv or lut delete them
            # Lower all the elements in the list
            color_table = cltmisc.filter_by_substring(
                color_table, ["tsv", "lut"], bool_case=False
            )

            # If the list is empty set its value to None
            if len(color_table) == 0:
                color_table = ["lut"]

            if cont_tech != "local":
                cmd_bashargs = ["echo", "$FREESURFER_HOME"]
                cmd_cont = cltmisc.generate_container_command(
                    cmd_bashargs, cont_tech, cont_image
                )  # Generating container command
                out_cmd = subprocess.run(
                    cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
                )
                fslut_file_cont = os.path.join(
                    out_cmd.stdout.split("\n")[0], "FreeSurferColorLUT.txt"
                )
                tmp_name = str(uuid.uuid4())
                cmd_bashargs = ["cp", "replace_cad", "/tmp/" + tmp_name]
                cmd_cont = cltmisc.generate_container_command(
                    cmd_bashargs, cont_tech, cont_image
                )

                # Replace the element of the list equal to replace_cad by the path of the lut file
                cmd_cont = [w.replace("replace_cad", fslut_file_cont) for w in cmd_cont]
                subprocess.run(
                    cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
                )
                fslut_file = os.path.join("/tmp", tmp_name)

                lut_dict = cltparc.Parcellation.read_luttable(in_file=fslut_file)

                # Remove the temporary file
                os.remove(fslut_file)

            else:

                fslut_file = os.path.join(
                    os.environ.get("FREESURFER_HOME"), "FreeSurferColorLUT.txt"
                )
                lut_dict = cltparc.Parcellation.read_luttable(in_file=fslut_file)

            fs_codes = lut_dict["index"]
            fs_names = lut_dict["name"]
            fs_colors = lut_dict["color"]

        # Create the folder if it does not exist
        temp_dir = os.path.dirname(out_vol)
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        if not os.path.isfile(out_vol) or force:

            if gm_grow == "0":

                cmd_bashargs = [
                    "mri_aparc2aseg",
                    "--s",
                    self.subj_id,
                    "--annot",
                    atlas,
                    "--hypo-as-wm",
                    "--new-ribbon",
                    "--o",
                    out_vol,
                ]

            elif gm_grow == "wm":
                cmd_bashargs = [
                    "mri_aparc2aseg",
                    "--s",
                    self.subj_id,
                    "--annot",
                    atlas,
                    "--labelwm",
                    "--hypo-as-wm",
                    "--new-ribbon",
                    "--o",
                    out_vol,
                ]

            else:
                # Creating the volumetric parcellation using the annot files
                cmd_bashargs = [
                    "mri_aparc2aseg",
                    "--s",
                    self.subj_id,
                    "--annot",
                    atlas,
                    "--wmparc-dmax",
                    gm_grow,
                    "--labelwm",
                    "--hypo-as-wm",
                    "--new-ribbon",
                    "--o",
                    out_vol,
                ]

            cmd_cont = cltmisc.generate_container_command(
                cmd_bashargs, cont_tech, cont_image
            )  # Generating container command

            # Bind the FreeSurfer subjects directory
            if cont_tech == "singularity":
                cmd_cont.insert(2, "--bind")
                cmd_cont.insert(3, dir_cad)
            elif cont_tech == "docker":
                cmd_cont.insert(2, "-v")
                cmd_cont.insert(3, dir_cad)

            subprocess.run(
                cmd_cont,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                universal_newlines=True,
            )  # Running container command

            if bool_native:

                # Moving the resulting parcellation from conform space to native
                self.conform2native(
                    mgz_conform=out_vol, nii_native=out_vol, force=force
                )

            if bool_mixwm:

                # Substracting 2000 to the WM labels in order to mix them with the GM labels
                parc = cltparc.Parcellation(parc_file=out_vol)
                parc_vol = parc.data

                # Detect the values in parc_vol that are bigger than 3000 and smaller than 5000
                # and substrac 2000 from them
                mask = np.logical_and(parc_vol >= 3000, parc_vol < 5000)
                parc_vol[mask] = parc_vol[mask] - 2000
                parc.data = parc_vol
                parc.adjust_values
                parc.save_parcellation(out_file=out_vol)

            if color_table is not None:
                temp_iparc = nib.load(out_vol)

                # unique the values
                unique_vals = np.unique(temp_iparc.get_fdata())

                # Select only the values different from 0 that are lower than 1000 or higher than 5000
                unique_vals = unique_vals[unique_vals != 0]
                unique_vals = unique_vals[(unique_vals < 1000) | (unique_vals > 5000)]

                # print them as integer numbers
                unique_vals = unique_vals.astype(int)

                fs_colortable = os.path.join(
                    os.environ.get("FREESURFER_HOME"), "FreeSurferColorLUT.txt"
                )
                lut_dict = cltparc.Parcellation.read_luttable(in_file=fs_colortable)
                fs_codes = lut_dict["index"]
                fs_names = lut_dict["name"]
                fs_colors = lut_dict["color"]

                values, idx = cltmisc.ismember_from_list(fs_codes, unique_vals.tolist())

                # select the fs_names and fs_colors in the indexes idx
                selected_fs_code = [fs_codes[i] for i in idx]
                selected_fs_name = [fs_names[i] for i in idx]
                selected_fs_color = [fs_colors[i] for i in idx]

                selected_fs_color = cltmisc.multi_rgb2hex(selected_fs_color)

                lh_ctx_parc = os.path.join(
                    self.subjs_dir, self.subj_id, "label", "lh." + atlas + ".annot"
                )
                rh_ctx_parc = os.path.join(
                    self.subjs_dir, self.subj_id, "label", "rh." + atlas + ".annot"
                )

                lh_obj = AnnotParcellation(parc_file=lh_ctx_parc)
                rh_obj = AnnotParcellation(parc_file=rh_ctx_parc)

                df_lh, out_tsv = lh_obj.export_to_tsv(
                    prefix2add="ctx-lh-", reg_offset=1000
                )
                df_rh, out_tsv = rh_obj.export_to_tsv(
                    prefix2add="ctx-rh-", reg_offset=2000
                )

                # Convert the column name of the dataframe to a list
                lh_ctx_code = df_lh["parcid"].tolist()
                rh_ctx_code = df_rh["parcid"].tolist()

                # Convert the column name of the dataframe to a list
                lh_ctx_name = df_lh["name"].tolist()
                rh_ctx_name = df_rh["name"].tolist()

                # Convert the column color of the dataframe to a list
                lh_ctx_color = df_lh["color"].tolist()
                rh_ctx_color = df_rh["color"].tolist()

                if gm_grow == "0" or bool_mixwm:
                    all_codes = selected_fs_code + lh_ctx_code + rh_ctx_code
                    all_names = selected_fs_name + lh_ctx_name + rh_ctx_name
                    all_colors = selected_fs_color + lh_ctx_color + rh_ctx_color

                else:

                    lh_wm_name = cltmisc.correct_names(
                        lh_ctx_name, replace=["ctx-lh-", "wm-lh-"]
                    )
                    # Add 2000 to each element of the list lh_ctx_code to create the WM code
                    lh_wm_code = [x + 2000 for x in lh_ctx_code]

                    rh_wm_name = cltmisc.correct_names(
                        rh_ctx_name, replace=["ctx-rh-", "wm-rh-"]
                    )
                    # Add 2000 to each element of the list lh_ctx_code to create the WM code
                    rh_wm_code = [x + 2000 for x in rh_ctx_code]

                    # Invert the colors lh_wm_color and rh_wm_color
                    ilh_wm_color = cltmisc.invert_colors(lh_ctx_color)
                    irh_wm_color = cltmisc.invert_colors(rh_ctx_color)

                    all_codes = (
                        selected_fs_code
                        + lh_ctx_code
                        + rh_ctx_code
                        + lh_wm_code
                        + rh_wm_code
                    )
                    all_names = (
                        selected_fs_name
                        + lh_ctx_name
                        + rh_ctx_name
                        + lh_wm_name
                        + rh_wm_name
                    )
                    all_colors = (
                        selected_fs_color
                        + lh_ctx_color
                        + rh_ctx_color
                        + ilh_wm_color
                        + irh_wm_color
                    )

                # Save the color table
                tsv_df = pd.DataFrame(
                    {
                        "index": np.asarray(all_codes),
                        "name": all_names,
                        "color": all_colors,
                    }
                )

                if "tsv" in color_table:
                    out_file = out_vol.replace(".nii.gz", ".tsv")
                    cltparc.Parcellation.write_tsvtable(tsv_df, out_file, force=force)
                if "lut" in color_table:
                    out_file = out_vol.replace(".nii.gz", ".lut")

                    now = datetime.now()
                    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
                    headerlines = [
                        "# $Id: {} {} \n".format(out_vol, date_time),
                        "{:<4} {:<50} {:>3} {:>3} {:>3} {:>3}".format(
                            "#No.", "Label Name:", "R", "G", "B", "A"
                        ),
                    ]

                    cltparc.Parcellation.write_luttable(
                        tsv_df["index"].tolist(),
                        tsv_df["name"].tolist(),
                        tsv_df["color"].tolist(),
                        out_file,
                        headerlines=headerlines,
                        force=force,
                    )

        elif os.path.isfile(out_vol) and not force:
            # Print a message
            if verbose:
                print(f"File {out_vol} already exists. Use force=True to overwrite it")

        return out_vol

    ####################################################################################################
    def conform2native(
        self,
        mgz_conform: str,
        nii_native: str,
        interp_method: str = "nearest",
        cont_tech: str = "local",
        cont_image: str = None,
        force: bool = False,
    ):
        """
        Transform image from FreeSurfer conform space to native acquisition space.

        Uses FreeSurfer's mri_vol2vol to transform images from the standardized
        conform space (256³ isotropic) back to the original native acquisition
        space and dimensions.

        Parameters
        ----------
        mgz_conform : str
            Path to input image in FreeSurfer conform space (typically .mgz format).

        nii_native : str
            Path for output image in native acquisition space.

        interp_method : str, optional
            Interpolation method for resampling: 'nearest', 'trilinear', or 'cubic'.
            Use 'nearest' for label/segmentation images. Default is 'nearest'.

        cont_tech : str, optional
            Container technology: 'local', 'docker', 'singularity'. Default is 'local'.

        cont_image : str, optional
            Container image specification when using containerization. Default is None.

        force : bool, optional
            Force processing even if output exists. If False, checks dimensions
            before deciding whether to reprocess. Default is False.

        Returns
        -------
        None
            Output file is created at the specified nii_native path.

        Raises
        ------
        FileNotFoundError
            If the required rawavg.mgz file (native space reference) doesn't exist.

        Notes
        -----
        This method is essential for bringing FreeSurfer processing results back
        to native space for integration with other analyses or visualization in
        original acquisition coordinates. The transformation uses the header
        information from rawavg.mgz as the target native space reference.

        Examples
        --------
        >>> # Convert parcellation to native space
        >>> subject.conform2native(
        ...     mgz_conform='/fs/mri/aparc+aseg.mgz',
        ...     nii_native='/output/aparc_native.nii.gz'
        ... )
        >>>
        >>> # Convert with trilinear interpolation
        >>> subject.conform2native(
        ...     mgz_conform='/fs/mri/T1.mgz',
        ...     nii_native='/output/T1_native.nii.gz',
        ...     interp_method='trilinear'
        ... )
        """

        raw_vol = os.path.join(self.subjs_dir, self.subj_id, "mri", "rawavg.mgz")
        tmp_raw = os.path.join(self.subjs_dir, self.subj_id, "tmp", "rawavg.nii.gz")

        # Get image dimensions
        if not os.path.isfile(raw_vol):
            raise FileNotFoundError(f"File {raw_vol} does not exist")

        cmd_bashargs = ["mri_convert", "-i", raw_vol, "-o", tmp_raw]
        cmd_cont = cltmisc.generate_container_command(
            cmd_bashargs, cont_tech, cont_image
        )  # Generating container command
        subprocess.run(
            cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
        )  # Running container command

        img = nib.load(tmp_raw)
        tmp_raw_hd = img.header["dim"]

        # Remove tmp_raw
        os.remove(tmp_raw)

        if not os.path.isfile(nii_native) or force:
            # Moving the resulting parcellation from conform space to native
            raw_vol = os.path.join(self.subjs_dir, self.subj_id, "mri", "rawavg.mgz")

            cmd_bashargs = [
                "mri_vol2vol",
                "--mov",
                mgz_conform,
                "--targ",
                raw_vol,
                "--regheader",
                "--o",
                nii_native,
                "--no-save-reg",
                "--interp",
                interp_method,
            ]
            cmd_cont = cltmisc.generate_container_command(
                cmd_bashargs, cont_tech, cont_image
            )  # Generating container command
            subprocess.run(
                cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
            )  # Running container command

        elif os.path.isfile(nii_native) and not force:
            # Print a message

            img = nib.load(nii_native)
            tmp_nii_hd = img.header["dim"]

            # Look if the dimensions are the same
            if all(tmp_raw_hd == tmp_nii_hd):
                print(f"File {nii_native} already exists and has the same dimensions")
                print(
                    f"File {nii_native} already exists. Use force=True to overwrite it"
                )

            else:
                cmd_bashargs = [
                    "mri_vol2vol",
                    "--mov",
                    mgz_conform,
                    "--targ",
                    raw_vol,
                    "--regheader",
                    "--o",
                    nii_native,
                    "--no-save-reg",
                    "--interp",
                    interp_method,
                ]
                cmd_cont = cltmisc.generate_container_command(
                    cmd_bashargs, cont_tech, cont_image
                )  # Generating container command
                subprocess.run(
                    cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
                )  # Running container command

    #####################################################################################################
    def get_surface(self, hemi: str, surf_type: str):
        """
        Get the file path for a specific surface mesh.

        Returns the path to surface files organized in the fs_files structure
        based on hemisphere and surface type specifications.

        Parameters
        ----------
        hemi : str
            Hemisphere identifier: 'lh' or 'rh'.

        surf_type : str
            Surface type: 'pial', 'white', 'inflated', or 'sphere'.

        Returns
        -------
        surf_file : str
            Full path to the requested surface file.

        Raises
        ------
        ValueError
            If hemisphere is not 'lh' or 'rh'.

        ValueError
            If surf_type is not one of the valid surface types.

        Examples
        --------
        >>> # Get left hemisphere pial surface
        >>> pial_surf = subject.get_surface('lh', 'pial')
        >>> print(pial_surf)
        >>>
        >>> # Get right hemisphere white matter surface
        >>> white_surf = subject.get_surface('rh', 'white')
        """

        if hemi not in ["lh", "rh"]:
            raise ValueError("The hemisphere must be lh or rh")

        if surf_type not in ["pial", "white", "inflated", "sphere"]:
            raise ValueError("The surface type must be pial, white, inflated or sphere")

        surf_file = self.fs_files["surf"][hemi]["mesh"][surf_type]

        return surf_file

    ####################################################################################################
    def get_vertexwise_map(self, hemi: str, map_type: str):
        """
        Get the file path for a specific vertex-wise morphometric map.

        Returns the path to morphometric map files organized in the fs_files
        structure based on hemisphere and map type specifications.

        Parameters
        ----------
        hemi : str
            Hemisphere identifier: 'lh' or 'rh'.

        map_type : str
            Map type: 'curv', 'sulc', 'thickness', 'area', or 'volume'.

        Returns
        -------
        map_file : str
            Full path to the requested morphometric map file.

        Raises
        ------
        ValueError
            If hemisphere is not 'lh' or 'rh'.

        ValueError
            If map_type is not one of the valid morphometric map types.

        Examples
        --------
        >>> # Get left hemisphere cortical thickness
        >>> thickness_map = subject.get_vertexwise_map('lh', 'thickness')
        >>> print(thickness_map)
        >>>
        >>> # Get right hemisphere curvature
        >>> curv_map = subject.get_vertexwise_map('rh', 'curv')
        """

        if hemi not in ["lh", "rh"]:
            raise ValueError("The hemisphere must be lh or rh")

        if map_type not in ["curv", "sulc", "thickness", "area", "volume"]:
            raise ValueError(
                "The map type must be curv, sulc, thickness, area or volume"
            )

        map_file = self.fs_files["surf"][hemi]["map"][map_type]

        return map_file

    ####################################################################################################
    def get_annotation(self, hemi: str, annot_type: str):
        """
        Get the file path for a specific parcellation annotation.

        Returns the path to annotation files organized in the fs_files structure
        based on hemisphere and annotation type specifications.

        Parameters
        ----------
        hemi : str
            Hemisphere identifier: 'lh' or 'rh'.

        annot_type : str
            Annotation type: 'desikan', 'destrieux', or 'dkt'.

        Returns
        -------
        annot_file : str
            Full path to the requested annotation file.

        Raises
        ------
        ValueError
            If hemisphere is not 'lh' or 'rh'.

        ValueError
            If annot_type is not one of the valid annotation types.

        Examples
        --------
        >>> # Get left hemisphere Desikan-Killiany parcellation
        >>> aparc_file = subject.get_annotation('lh', 'desikan')
        >>> print(aparc_file)
        >>>
        >>> # Get right hemisphere Destrieux parcellation
        >>> destrieux_file = subject.get_annotation('rh', 'destrieux')
        """

        if hemi not in ["lh", "rh"]:
            raise ValueError("The hemisphere must be lh or rh")

        if annot_type not in ["desikan", "destrieux", "dkt"]:
            raise ValueError("The annotation type must be desikan, destrieux or dkt")

        annot_file = self.fs_files["surf"][hemi]["parc"][annot_type]

        return annot_file


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############   Section 3: Other methods to work with FreeSurfer-based data structure    ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def create_individual_freesurfer_table(
    subj_id: str,
    subjs_dir: str = None,
    out_tab_file: str = None,
    add_bids_entities: bool = False,
) -> pd.DataFrame:
    """
    Create a dataset table from the subjects in the given directory.
    Parameters
    ----------
    subjs_dir : str, optional
        The directory containing the FreeSurfer subjects. If None, uses the
        FREESURFER_HOME environment variable to locate the subjects directory.

    subj_id : str
        The subject ID for which to create the dataset table.

    out_tab_file : str, optional
        If provided, the path to save the dataset table as a TSV file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the morphometry table for the subject.
    """
    # Check if subjs_dir is provided, otherwise use the default
    if subjs_dir is None:
        subjs_dir = os.environ.get("SUBJECTS_DIR")

    else:
        if not os.path.isdir(subjs_dir):
            raise ValueError(
                f"The provided subjs_dir '{subjs_dir}' is not a valid directory."
            )

    if out_tab_file is not None:
        if isinstance(out_tab_file, str):
            out_dir = os.path.dirname(out_tab_file)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)
        else:
            raise ValueError(
                "out_tab_file should be a string representing the file path."
            )

    if not isinstance(subj_id, str):
        raise ValueError("subj_id should be a string representing the subject ID.")
    else:
        # Check if the subject directory exists
        subj_path = os.path.join(
            subjs_dir,
            subj_id,
        )
        if not os.path.isdir(subj_path):
            raise ValueError(
                f"The subject ID '{subj_id}' does not exist in the directory '{subjs_dir}'."
            )
        else:
            if not os.path.isdir(os.path.join(subj_path, "mri")):
                raise ValueError(
                    f"The subject ID '{subj_id}' is not a valid FreeSurfer directory."
                )

    # Load the Subject object
    subject = FreeSurferSubject(subj_id, subjs_dir)
    df = subject.create_stats_table(
        output_file=out_tab_file, add_bids_entities=add_bids_entities
    )
    return df


#####################################################################################################
def process_subject(fs_fullid: str, fs_subject_dir: str, out_folder: str) -> tuple:
    """
    Process a single subject and return the result.

    Parameters
    ----------
    fs_fullid : str
        The full subject ID

    fs_subject_dir : str
        The FreeSurfer subjects directory

    out_folder : str
        The output folder for stats tables

    Returns
    -------
    tuple
        (subject_id, success_status, error_message_if_any)
    """
    try:
        sub_entity = cltbids.str2entity(fs_fullid)
        file_name = fs_fullid + "_desc-statstable_morphometry.csv"
        out_flder = os.path.join(
            out_folder,
            "freesurfer",
            "sub-" + sub_entity["sub"],
            "ses-" + sub_entity["ses"],
            "stats",
        )
        out_tab_file = os.path.join(out_flder, file_name)

        if not os.path.isfile(out_tab_file):
            df = create_individual_freesurfer_table(
                fs_fullid,
                fs_subject_dir,
                out_tab_file=out_tab_file,
                add_bids_entities=True,
            )
            return (fs_fullid, True, None)
        else:
            return (fs_fullid, True, "File already exists - skipped")

    except Exception as e:
        return (fs_fullid, False, str(e))


#####################################################################################################
def create_freesurfer_table(
    out_folder: str,
    ids_file: Union[str, List[str]] = None,
    fs_subject_dir: str = None,
    max_workers: int = 1,
):
    """
    Create FreeSurfer stats tables for multiple subjects in parallel.

    Parameters
    ----------
    out_folder : str
        The output folder for stats tables.

    ids_file : str or List[str], optional
        A text file containing subject IDs (one per line) or a list of subject IDs.
        If None, an error is raised.

    fs_subject_dir : str, optional
        The FreeSurfer subjects directory. If None, uses the SUBJECTS_DIR environment variable.

    max_workers : int, optional
        The maximum number of worker threads to use for parallel processing. Default is 1 (non-parallel).

    Returns
    -------
    None

    Notes
    -----
    This function processes each subject in parallel using ThreadPoolExecutor and displays a progress bar.
    It handles errors gracefully and provides a summary of completed and failed subjects.

    Example
    -------
    >>> create_freesurfer_table("/path/to/output", "/path/to/ids.txt", "/path/to/freesurfer/subjects", max_workers=4)

    """

    if ids_file is None:
        subject_ids = []
        subj_dirs = os.listdir(fs_subject_dir)

        subject_ids = []
        for subj_dir in subj_dirs:
            if subj_dir.startswith("sub-"):
                subject_ids.append(subj_dir)

    else:
        # Use cltpipe.get_ids2process to handle both list and file input
        subject_ids = cltpipe.get_ids2process(ids_file)

    # Initialize console and display summary
    console = Console()
    console.print(
        Panel.fit(
            f"[bold green]FreeSurfer Stats Processing (Parallel)[/bold green]\n"
            f"[blue]Total subjects to process: {len(subject_ids)}[/blue]\n"
            f"[blue]Max workers: {max_workers}[/blue]",
            style="cyan",
        )
    )

    # Configure progress bar with custom columns
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="green", finished_style="bright_green"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=10,
    )

    # Process subjects with ThreadPoolExecutor and progress bar
    completed_subjects = []
    failed_subjects = []

    with progress:
        task = progress.add_task("Processing subjects", total=len(subject_ids))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_subject = {
                executor.submit(
                    process_subject, fs_fullid, fs_subject_dir, out_folder
                ): fs_fullid
                for fs_fullid in subject_ids
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_subject):
                subject_id = future_to_subject[future]
                try:
                    result = future.result()
                    subject_id, success, error_msg = result

                    if success:
                        completed_subjects.append(subject_id)
                        if error_msg:
                            progress.update(
                                task,
                                description=f"Skipped: [yellow]{subject_id}[/yellow] ({error_msg})",
                            )
                        else:
                            progress.update(
                                task,
                                description=f"Completed: [green]{subject_id}[/green]",
                            )
                    else:
                        failed_subjects.append((subject_id, error_msg))
                        progress.update(
                            task, description=f"Failed: [red]{subject_id}[/red]"
                        )
                        console.print(
                            f"[red]Error processing {subject_id}: {error_msg}[/red]"
                        )

                except Exception as exc:
                    failed_subjects.append((subject_id, str(exc)))
                    progress.update(
                        task, description=f"Failed: [red]{subject_id}[/red]"
                    )
                    console.print(
                        f"[red]Exception processing {subject_id}: {exc}[/red]"
                    )

                # Advance progress
                progress.advance(task)

    # Display final summary
    console.print(
        Panel.fit(
            f"[bold green]✓ Processing Complete![/bold green]\n"
            f"[green]Successfully processed: {len(completed_subjects)}[/green]\n"
            f"[red]Failed: {len(failed_subjects)}[/red]",
            style="green",
        )
    )

    # Display failed subjects if any
    if failed_subjects:
        console.print("\n[bold red]Failed Subjects:[/bold red]")
        for subj_id, error in failed_subjects:
            console.print(f"  [red]• {subj_id}: {error}[/red]")


#####################################################################################################
def create_fsaverage_links(
    fssubj_dir: str, fsavg_dir: str = None, refsubj_name: str = None
):
    """
    Create symbolic links to the fsaverage reference subject folder.

    Creates symbolic links from a custom FreeSurfer subjects directory to the
    standard fsaverage template, enabling FreeSurfer tools to locate reference data.

    Parameters
    ----------
    fssubj_dir : str
        Target FreeSurfer subjects directory where the link will be created.

    fsavg_dir : str, optional
        Source fsaverage directory path. If None, uses FREESURFER_HOME/subjects/fsaverage.
        Default is None.

    refsubj_name : str, optional
        Reference subject name. If None, uses 'fsaverage'. Default is None.

    Returns
    -------
    link_folder : str
        Path to the created symbolic link.

    Raises
    ------
    ValueError
        If the FreeSurfer subjects directory or fsaverage directory doesn't exist.

    Examples
    --------
    >>> # Create standard fsaverage link
    >>> link_path = create_fsaverage_links('/data/freesurfer_subjects')
    >>>
    >>> # Create link with custom reference
    >>> link_path = create_fsaverage_links(
    ...     '/data/freesurfer_subjects',
    ...     refsubj_name='fsaverage6'
    ... )
    """

    # Verify if the FreeSurfer directory exists
    if not os.path.isdir(fssubj_dir):
        raise ValueError("The selected FreeSurfer directory does not exist")

    # Creating and verifying the freesurfer directory for the reference name
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

    if not os.path.exists(link_folder):  # Changed from os.path.isdir
        try:
            if sys.platform.startswith("win"):
                # Windows implementation
                subprocess.run(
                    ["mklink", "/D", link_folder, fsavg_dir], check=True, shell=True
                )
            else:
                # Unix/Linux implementation
                subprocess.run(
                    ["ln", "-s", fsavg_dir, link_folder],  # Fixed: specify exact target
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Failed to create symbolic link: {e}")

    return link_folder


####################################################################################################
def remove_fsaverage_links(linkavg_folder: str):
    """
    Remove symbolic links to the fsaverage folder.

    Safely removes symbolic links to fsaverage directories that don't point
    to the original FreeSurfer installation location.

    Parameters
    ----------
    linkavg_folder : str
        Path to the fsaverage link folder to potentially remove.

    Notes
    -----
    Only removes links that don't point to the original FREESURFER_HOME
    location to prevent accidental deletion of the actual fsaverage data.

    Examples
    --------
    >>> # Remove custom fsaverage link
    >>> remove_fsaverage_links('/data/freesurfer_subjects/fsaverage')
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


####################################################################################################
def region_to_vertexwise(
    reg_values: np.ndarray, labels: np.ndarray, reg_ctable: np.ndarray
) -> np.ndarray:
    """
    Map regional values to vertex-wise values using parcellation labels.

    Assigns regional measurements to individual vertices based on their
    parcellation labels, creating vertex-wise data from region-wise data.

    Parameters
    ----------
    reg_values : np.ndarray
        Array of values for each region. Can be 1D or 2D array.

    labels : np.ndarray
        Array of parcellation labels for each vertex.

    reg_ctable : np.ndarray
        Color table with shape (N, 5) where column 4 contains region labels.

    Returns
    -------
    vertex_values : np.ndarray
        Array with values mapped to each vertex. Shape: (num_vertices, num_measures).

    Raises
    ------
    ValueError
        If number of regions in reg_values doesn't match reg_ctable dimensions
        or if reg_values has more than 2 dimensions.

    Examples
    --------
    >>> # Map thickness values to vertices
    >>> regional_thickness = np.array([2.5, 3.1, 2.8])
    >>> vertex_thickness = region_to_vertexwise(
    ...     regional_thickness, vertex_labels, color_table
    ... )
    """

    # check that the number of rows of reg_values matches the number of regions in reg_ctable
    if reg_values.shape[0] != reg_ctable.shape[0]:
        raise ValueError(
            "The number of rows in reg_values must match the number of regions in reg_ctable"
        )
    # Ensure reg_values is a 2D array
    if reg_values.ndim == 1:
        reg_values = reg_values.reshape(-1, 1)
    elif reg_values.ndim > 2:
        raise ValueError("reg_values must be a 1D or 2D array")

    n_cols = reg_values.shape[1]

    vertex_values = np.zeros(
        (len(labels), n_cols), dtype=np.float32
    )  # Default value is 0

    for i, region_info in enumerate(reg_ctable):
        # Find vertices with this label
        indices = np.where(labels == region_info[4])[0]

        # Assign the region color (RGB from first 3 columns)
        if len(indices) > 0:
            vertex_values[indices, :] = reg_values[i, :]

    return vertex_values


#####################################################################################################
def create_vertex_colors(labels: np.ndarray, reg_ctable: np.ndarray) -> np.ndarray:
    """
    Create per-vertex RGBA colors based on parcellation labels.

    Assigns colors to vertices based on their parcellation region using
    the color table information.

    Parameters
    ----------
    labels : np.ndarray
        Array of parcellation labels for each vertex.

    reg_ctable : np.ndarray
        Color table with shape (N, 5) where first 3 columns are RGB values
        and column 4 contains region labels.

    Returns
    -------
    vertex_colors : np.ndarray
        Array of RGB colors for each vertex with shape (num_vertices, 3).
        Default color is gray (240, 240, 240) for unlabeled vertices.

    Examples
    --------
    >>> # Create vertex colors for visualization
    >>> colors = create_vertex_colors(vertex_labels, color_table)
    >>> print(f"Colors shape: {colors.shape}")  # (num_vertices, 3)
    """

    # Automatically detect the range of the colors in reg_ctable
    if reg_ctable.shape[1] != 5:
        raise ValueError(
            "The color table must have 5 columns: R, G, B, A, and packed RGB value"
        )

    return cltmisc.get_colors_from_colortable(labels, reg_ctable)


#####################################################################################################
def colors2colortable(colors: Union[list, np.ndarray]):
    """
    Convert color list to FreeSurfer color table format.

    Transforms hexadecimal colors or RGB arrays into FreeSurfer's standard
    color table format with packed RGB values.

    Parameters
    ----------
    colors : list or np.ndarray
        List of hexadecimal color strings (e.g., ['#FF0000', '#00FF00'])
        or numpy array of RGB values.

    Returns
    -------
    colortable : np.ndarray
        FreeSurfer color table with shape (N, 5) containing RGB values,
        alpha channel, and packed RGB values.

    Raises
    ------
    ValueError
        If colors is not a list or numpy array.

    Examples
    --------
    >>> # Convert hex colors to color table
    >>> hex_colors = ["#FF0000", "#00FF00", "#0000FF"]
    >>> ctab = colors2colortable(hex_colors)
    >>> print(f"Color table shape: {ctab.shape}")
    """

    colortable = cltmisc.colors_to_table(colors)

    # Ensure the color table has 5 columns: R, G, B, A, packed RGB and they are integer type
    colortable = colortable.astype(np.uint32)

    return colortable


#####################################################################################################
def resolve_colortable_duplicates(color_table):
    """
    Make all RGB colors in FreeSurfer color table unique.

    Identifies duplicate RGB colors and modifies them by incrementing color
    channels until uniqueness is achieved. Updates packed RGB values accordingly.

    Parameters
    ----------
    color_table : np.ndarray
        FreeSurfer color table with shape (n, 5) where columns 0-2 are RGB,
        column 3 is alpha, and column 4 is packed RGB value.

    Returns
    -------
    modified_table : np.ndarray
        Updated color table with unique colors.

    modification_log : dict
        Dictionary containing details of changes made including original
        duplicates count and list of modifications.

    packed_values_mapping : dict
        Dictionary with 'old_values' and 'new_values' arrays for updating
        label maps accordingly.

    Examples
    --------
    >>> # Resolve duplicate colors
    >>> unique_table, log, mapping = resolve_colortable_duplicates(color_table)
    >>> print(f"Modified {len(log['modifications_made'])} colors")
    """

    # Create a copy to avoid modifying the original
    table = color_table.copy()

    # Track modifications
    modification_log = {
        "original_duplicates": 0,
        "modifications_made": [],
        "final_unique_colors": 0,
    }

    # Track old and new packed values for map correction
    old_packed_values = []
    new_packed_values = []

    # Find duplicates by grouping indices by RGB values
    rgb_to_indices = defaultdict(list)
    for i, row in enumerate(table):
        rgb_key = tuple(row[:3].astype(int))  # Ensure integers (R, G, B)
        rgb_to_indices[rgb_key].append(i)

    # Count original duplicates
    duplicate_groups = {
        rgb: indices for rgb, indices in rgb_to_indices.items() if len(indices) > 1
    }
    modification_log["original_duplicates"] = sum(
        len(indices) - 1 for indices in duplicate_groups.values()
    )

    # Process each duplicate group
    for rgb_key, indices in duplicate_groups.items():
        # Keep the first occurrence unchanged, modify the rest
        for table_idx in indices[1:]:
            original_rgb = table[table_idx][:3].copy().astype(int)

            # Store old packed value
            old_packed_value = int(table[table_idx][4])
            old_packed_values.append(old_packed_value)

            # Find a unique color by trying modifications
            new_rgb = find_unique_color(
                table[table_idx][:3].astype(int), rgb_to_indices
            )

            # Update the table
            table[table_idx][:3] = new_rgb

            # Recalculate packed RGB value: R + (G << 8) + (B << 16)
            packed_rgb = (
                int(new_rgb[0]) + (int(new_rgb[1]) << 8) + (int(new_rgb[2]) << 16)
            )
            table[table_idx][4] = packed_rgb

            # Store new packed value
            new_packed_values.append(packed_rgb)

            # Update our tracking dictionary
            new_rgb_key = tuple(new_rgb)
            rgb_to_indices[new_rgb_key] = [table_idx]
            rgb_to_indices[rgb_key].remove(table_idx)

            # Determine which channel was modified and by how much
            diff = new_rgb - original_rgb
            modified_channel_idx = np.nonzero(diff)[0][0]  # First non-zero difference
            channel_names = ["R", "G", "B"]

            # Log the modification
            modification_log["modifications_made"].append(
                {
                    "index": table_idx,
                    "original_rgb": original_rgb.tolist(),
                    "new_rgb": new_rgb.tolist(),
                    "channel_modified": channel_names[modified_channel_idx],
                    "increment_applied": int(diff[modified_channel_idx]),
                    "new_packed_value": packed_rgb,
                }
            )

    # Verify uniqueness
    final_rgb_values = [tuple(row[:3].astype(int)) for row in table]
    unique_colors = len(set(final_rgb_values))
    modification_log["final_unique_colors"] = unique_colors

    # Verify no negative values
    if np.any(table[:, :3] < 0):
        raise ValueError("Negative RGB values detected after processing!")

    # Verify no values exceed 255
    if np.any(table[:, :3] > 255):
        raise ValueError("RGB values exceeding 255 detected after processing!")

    # Create mapping dictionary for correcting maps
    packed_values_mapping = {
        "old_values": np.array(old_packed_values),
        "new_values": np.array(new_packed_values),
    }

    return table, modification_log, packed_values_mapping


######################################################################################################
def find_unique_color(rgb, rgb_to_indices):
    """
    Find unique RGB color by systematically modifying channels.

    Helper function that searches for an unused RGB color by incrementing
    or decrementing color channels in order: Blue, Green, Red.

    Parameters
    ----------
    rgb : np.ndarray
        Original RGB values [R, G, B].

    rgb_to_indices : dict
        Dictionary mapping RGB tuples to indices for collision detection.

    Returns
    -------
    np.ndarray
        New unique RGB values.

    Raises
    ------
    RuntimeError
        If no unique color can be found (extremely rare).
    """
    rgb = rgb.astype(int)

    # Try modifying channels in order: B, G, R (to minimize visual impact)
    for channel_idx in [2, 1, 0]:  # B, G, R

        # First, try positive increments (safer as they avoid negative values)
        for increment in range(1, 256):  # Try +1, +2, +3, ... up to +255
            new_rgb = rgb.copy()

            # Check if we can add this increment without exceeding 255
            if new_rgb[channel_idx] + increment <= 255:
                new_rgb[channel_idx] += increment
                new_rgb_key = tuple(new_rgb)

                # Check if this color is unique (not in the dictionary)
                if new_rgb_key not in rgb_to_indices:
                    return new_rgb

        # If positive increments don't work, try negative increments
        for decrement in range(1, 256):  # Try -1, -2, -3, ... up to -255
            new_rgb = rgb.copy()

            # Check if we can subtract this decrement without going below 0
            if new_rgb[channel_idx] - decrement >= 0:
                new_rgb[channel_idx] -= decrement
                new_rgb_key = tuple(new_rgb)

                # Check if this color is unique (not in the dictionary)
                if new_rgb_key not in rgb_to_indices:
                    return new_rgb

    # If we get here, we couldn't find a unique color by modifying single channels
    # This is extremely unlikely but let's try modifying two channels
    for channel1 in [0, 1, 2]:
        for channel2 in [0, 1, 2]:
            if channel1 != channel2:
                for inc1 in range(1, 10):  # Limit to small increments for two channels
                    for inc2 in range(1, 10):
                        new_rgb = rgb.copy()

                        if (
                            new_rgb[channel1] + inc1 <= 255
                            and new_rgb[channel2] + inc2 <= 255
                        ):
                            new_rgb[channel1] += inc1
                            new_rgb[channel2] += inc2
                            new_rgb_key = tuple(new_rgb)

                            if new_rgb_key not in rgb_to_indices:
                                return new_rgb

    # Last resort: this should never happen with proper color tables
    raise RuntimeError(
        f"Could not find unique color for RGB {rgb}. Color space might be saturated."
    )


#######################################################################################################
def verify_packed_rgb_values(color_table):
    """
    Verify that packed RGB values are correctly calculated.

    Checks if the packed RGB values (column 4) match the calculated values
    from RGB channels using the formula: R + (G << 8) + (B << 16).

    Parameters
    ----------
    color_table : np.ndarray
        FreeSurfer color table to verify.

    Returns
    -------
    all_correct : bool
        True if all packed values are correct.

    incorrect_indices : list
        List of dictionaries with details of incorrect entries.

    Examples
    --------
    >>> # Verify color table integrity
    >>> is_correct, errors = verify_packed_rgb_values(color_table)
    >>> if not is_correct:
    ...     print(f"Found {len(errors)} incorrect packed values")
    """

    all_correct = True
    incorrect_indices = []

    for i, row in enumerate(color_table):
        r, g, b = int(row[0]), int(row[1]), int(row[2])
        expected_packed = r + (g << 8) + (b << 16)
        actual_packed = int(row[4])

        if expected_packed != actual_packed:
            all_correct = False
            incorrect_indices.append(
                {
                    "index": i,
                    "rgb": [r, g, b],
                    "expected_packed": expected_packed,
                    "actual_packed": actual_packed,
                }
            )

    return all_correct, incorrect_indices


########################################################################################################
def detect_hemi(file_name: str):
    """
    Detect hemisphere from filename using common naming conventions.

    Identifies left ('lh') or right ('rh') hemisphere from FreeSurfer and
    BIDS-style filenames using various naming patterns.

    Parameters
    ----------
    file_name : str
        Filename to analyze for hemisphere information.

    Returns
    -------
    hemi : str or None
        Hemisphere identifier ('lh' or 'rh') or None if not detected.

    Notes
    -----
    Recognizes patterns like 'lh.', 'rh.', 'hemi-L', 'hemi-left', etc.
    Issues warning if hemisphere cannot be determined.

    Examples
    --------
    >>> # Detect from FreeSurfer filename
    >>> hemi = detect_hemi('lh.aparc.annot')
    >>> print(hemi)  # 'lh'
    >>>
    >>> # Detect from BIDS filename
    >>> hemi = detect_hemi('sub-01_hemi-L_pial.surf.gii')
    >>> print(hemi)  # 'lh'
    """

    # Detecting the hemisphere
    surf_name = os.path.basename(file_name)
    file_name = surf_name.lower()

    # Find in the string annot_name if it is lh. or rh.
    if "lh." in surf_name:
        hemi = "lh"
    elif "rh." in surf_name:
        hemi = "rh"
    elif "hemi-" in surf_name:
        tmp_hemi = surf_name.split("-")[1].split("_")[0]
        tmp_ent = cltbids.str2entity(file_name)
        if "hemi" in tmp_ent.keys():
            tmp_hemi = tmp_ent["hemi"]

        if tmp_hemi in ["lh", "l", "left", "lefthemisphere"]:
            hemi = "lh"
        elif tmp_hemi in ["rh", "r", "right", "righthemisphere"]:
            hemi = "rh"
        else:
            hemi = None
            warnings.warn(
                "The hemisphere could not be extracted from the annot filename. Please provide it as an argument"
            )
    else:
        hemi = None
        warnings.warn(
            "The hemisphere could not be extracted from the annot filename. Please provide it as an argument"
        )

    return hemi


############################################################################################################
def load_lobes_json(lobes_json: str = None):
    """
    Load JSON file containing anatomical lobe definitions.

    Loads configuration file that defines how brain regions are grouped
    into anatomical lobes for coarser-grained analyses.

    Parameters
    ----------
    lobes_json : str, optional
        Path to custom JSON file. If None, uses default configuration
        file included with the package. Default is None.

    Returns
    -------
    pipe_dict : dict
        Dictionary containing lobe definitions and grouping schemes.

    Raises
    ------
    ValueError
        If the specified JSON file doesn't exist.

    Examples
    --------
    >>> # Load default lobe definitions
    >>> lobes_config = load_lobes_json()
    >>> print(lobes_config.keys())
    >>>
    >>> # Load custom definitions
    >>> custom_config = load_lobes_json('/path/to/custom_lobes.json')
    """

    # Get the absolute of this file
    if lobes_json is None:
        cwd = os.path.dirname(os.path.abspath(__file__))
        lobes_json = os.path.join(cwd, "config", "lobes.json")
    else:
        if not os.path.isfile(lobes_json):
            raise ValueError(
                "Please, provide a valid JSON file containing the lobes definition dictionary."
            )

    with open(lobes_json, encoding="utf-8") as f:
        pipe_dict = json.load(f)

    return pipe_dict


############################################################################################################
def get_version(cont_tech: str = "local", cont_image: str = None):
    """
    Get FreeSurfer version number from installation or container.

    Queries FreeSurfer installation to determine version number, supporting
    both local installations and containerized environments.

    Parameters
    ----------
    cont_tech : str, optional
        Container technology: 'local', 'docker', 'singularity'. Default is 'local'.

    cont_image : str, optional
        Container image specification when using containerization. Default is None.

    Returns
    -------
    vers_cad : str
        FreeSurfer version number (e.g., '7.2.0').

    Examples
    --------
    >>> # Get local FreeSurfer version
    >>> version = get_version()
    >>> print(f"FreeSurfer version: {version}")
    >>>
    >>> # Get version from Docker container
    >>> version = get_version(
    ...     cont_tech='docker',
    ...     cont_image='freesurfer/freesurfer:7.2.0'
    ... )
    """

    # Running the version command
    cmd_bashargs = ["recon-all", "-version"]
    cmd_cont = cltmisc.generate_container_command(
        cmd_bashargs, cont_tech, cont_image
    )  # Generating container command
    out_cmd = subprocess.run(cmd_cont, stdout=subprocess.PIPE, universal_newlines=True)

    for st_ver in out_cmd.stdout.split("-"):
        if "." in st_ver:
            vers_cad = st_ver
            break

    # Delete all the non numeric characters from the string except the "."
    vers_cad = "".join(filter(lambda x: x.isdigit() or x == ".", vers_cad))

    return vers_cad
