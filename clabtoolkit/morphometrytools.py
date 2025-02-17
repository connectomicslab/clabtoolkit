import os
import shutil
from typing import Union, Dict, List
import copy
from pyvista import _vtk, PolyData
from numpy import split, ndarray
import json

import pandas as pd
import nibabel as nib
import numpy as np

import clabtoolkit.misctools as cltmisc
import clabtoolkit.freesurfertools as cltfree
import clabtoolkit.surfacetools as cltsurf
import clabtoolkit.parcellationtools as cltparc

####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############           Methods dedicated to compute metrics from surfaces               ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################


def compute_reg_val_fromannot(
    metric_file: Union[str, np.ndarray],
    parc_file: Union[str, cltfree.AnnotParcellation],
    hemi: str,  # Hemisphere id. It could be lh or rh
    metric: str = "unknown",
    stats_list: Union[str, list] = ["mean", "median", "std", "min", "max"],
    format: str = "region",
    include_unknown: bool = False,
) -> pd.DataFrame:
    """
    This method computes the regional values from a surface metric file and an annotation file.

    Parameters
    ----------
    metric_file : str
        Path to the surface map file. It represents the values of the metric on the vertices of the surface.

    parc_file : str
        Path to the annotation file. It represents the regions of the surface.

    hemi : str
        Hemisphere id. It could be lh or rh.

    metric : str
        Name of the metric. It is used to create the column names of the output DataFrame.

    stats_list : Union[str, list], optional
        List of statistics to compute. The default is ["mean", "median", "std", "min", "max"].


    format : str, optional
        Format of the output. It could be "region" or "metric". The default is "region".
        With the "region" format, the output is a DataFrame with the regional values where each column
        represent the value of column metric for each specific region. With the "metric" format, the output
        is a DataFrame with the regional values where each column represent the value of a specific metric
        for each region.

    include_unknown : bool, optional
        If True, the unknown regions are included in the output. The default is False.
        This includes on the table the regions with the following names: medialwall, unknown, corpuscallosum.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with the regional values.
    metric_vect : np.ndarray


    Examples
    --------
    >>> import clabtoolkit.morphometrytools as clmorphtools
    >>> import os
    >>> import pandas as pd
    >>> metric_file = os.path.join('..', 'data', 'lh.thickness')
    >>> parc_file = os.path.join('..', 'data', 'lh.aparc.annot')
    >>> df = clmorphtools.compute_reg_val_fromannot(metric_file, parc_file, 'lh')
    >>> print(df.head())

    """

    # Detecting if the stats_list is a string
    if isinstance(stats_list, str):
        stats_list = [stats_list]

    # Detect if the format is not region or metric
    if format not in ["region", "metric"]:
        raise ValueError("The format should be region or metric.")

    # Detecting if the needed parcellation file as a string or an object
    if isinstance(parc_file, str):
        # Checking if the file exists if the file is a string. If exists, read the file and create the object
        # Otherwise, raise an error
        if not os.path.exists(parc_file):
            raise FileNotFoundError("The annotation file does not exist.")
        else:
            # Reading the annot file
            sparc_data = cltfree.AnnotParcellation(
                parc_file=parc_file,
            )

    elif isinstance(parc_file, cltfree.AnnotParcellation):
        # If the file is an object, copy the object
        sparc_data = copy.deepcopy(parc_file)

    # Detecting if the needed metric file as a string or an object. If the file is a string, check if the file exists
    # Otherwise, raise an error
    if isinstance(metric_file, str):
        if not os.path.exists(metric_file):
            raise FileNotFoundError("The metric file does not exist.")
        else:
            # Reading the vertex-wise metric file
            metric_vect = nib.freesurfer.io.read_morph_data(metric_file)

    elif isinstance(metric_file, np.ndarray):
        metric_vect = metric_file

    # Converting to lower case
    stats_list = list(map(lambda x: x.lower(), stats_list))  # Converting to lower case

    if not include_unknown:
        tmp_names = sparc_data.regnames
        unk_indexes = cltmisc.get_indexes_by_substring(
            tmp_names, ["medialwall", "unknown", "corpuscallosum"]
        ).astype(int)

        if len(unk_indexes) > 0:
            # get the values of the unknown regions
            unk_codes = sparc_data.regtable[unk_indexes, 4]
            unk_vert = np.isin(sparc_data.codes, unk_codes)

            sparc_data.codes[unk_vert] = 0
            sparc_data.regnames = np.delete(sparc_data.regnames, unk_indexes).tolist()
            sparc_data.regtable = np.delete(sparc_data.regtable, unk_indexes, axis=0)

    # Setting the vertex values to 0 if the values are not in the table
    # Set to 0 the values of sparc_data.codes that are not in the regtable
    unique_codes = np.unique(sparc_data.codes)

    # Get the indexes of the unique codes that are not in the regtable
    not_in_table = np.setdiff1d(unique_codes, sparc_data.regtable[:, 4])

    # Set to 0 the values of the codes that are not in the regtable
    sparc_data.codes[np.isin(sparc_data.codes, not_in_table)] = 0

    sts = np.unique(sparc_data.codes)
    # Remove the 0
    sts = sts[sts != 0]
    nreg = len(sts)

    dict_of_cols = {}

    # Values for each region in the annot
    df = pd.DataFrame()

    # Compute the whole hemisphere
    temp = stats_from_vector(
        metric_vect[np.isin(sparc_data.codes, sparc_data.regtable[:, 4])], stats_list
    )
    dict_of_cols["ctx-" + hemi + "-hemisphere"] = temp

    for regname in sparc_data.regnames:

        # Get the index of the region in the color table
        index = cltmisc.get_indexes_by_substring(
            sparc_data.regnames, regname, matchww=True
        )

        if len(index):
            temp = stats_from_vector(
                metric_vect[sparc_data.codes == sparc_data.regtable[index, 4]],
                stats_list,
            )
            dict_of_cols[regname] = temp

        else:
            dict_of_cols[regname] = np.zeros_like(stats_list).tolist()

    df = pd.DataFrame.from_dict(dict_of_cols)

    colnames = df.columns.tolist()
    colnames = cltmisc.correct_names(colnames, prefix="ctx-" + hemi + "-")
    df.columns = colnames

    if format == "region":
        df.index = stats_list

        # Converting the row names to a new column called statistics
        df = df.reset_index()
        df = df.rename(columns={"index": "statistics"})

        # Convert the index to a column with name "metric"

    else:
        df = df.T
        df.columns = stats_list

        # Converting the row names to a new column called statistics
        df = df.reset_index()
        df = df.rename(columns={"index": "region"})

    nrows = df.shape[0]

    # Inserting the units
    units = get_units(metric)
    df.insert(0, "metric", [metric] * nrows)
    df.insert(1, "units", units * nrows)

    # Insert a column at the begining of the dataframe

    return df, metric_vect


def compute_reg_area_fromsurf(
    surf_file: Union[str, cltsurf.Surface],
    parc_file: Union[str, cltfree.AnnotParcellation],
    hemi: str,  # Hemisphere id. It could be lh or rh
    format: str = "region",
    include_unknown: bool = False,
) -> pd.DataFrame:
    """
    This method computes the surface area for each region in the annotation file.

    Parameters
    ----------
    surf_file : str
        Path to the surface file.

    parc_file : str
        Path to the annotation file. It represents the regions of the surface.

    hemi : str
        Hemisphere id. It could be lh or rh.

    format : str, optional
        Format of the output. It could be "region" or "metric". The default is "region".
        With the "region" format, the output is a DataFrame with the regional values where each column
        represent the value of column metric for each specific region. With the "metric" format, the output
        is a DataFrame with the regional values where each column represent the value of a specific metric
        for each region.

    include_unknown : bool, optional
        If True, the unknown regions are included in the output. The default is False.
        This includes on the table the regions with the following names: medialwall, unknown, corpuscallosum.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with the regional values.
    metric_vect : np.ndarray


    Examples
    --------
    >>> import clabtoolkit.morphometrytools as clmorphtools
    >>> import os
    >>> import pandas as pd
    >>> surf_file = '/opt/freesurfer/subjects/fsaverage/surf/lh.white'
    >>> annot_file = '/opt/freesurfer/subjects/fsaverage/label/lh.aparc.annot'
    >>> df = clmorphtools.compute_reg_area_fromsurf(surf_file, annot_file, 'lh')
    >>> print(df.head())
    """

    # Detect if the format is not region or metric
    if format not in ["region", "metric"]:
        raise ValueError("The format should be region or metric.")

    # Detecting if the needed parcellation file as a string or an object
    if isinstance(parc_file, str):
        # Checking if the file exists if the file is a string. If exists, read the file and create the object
        # Otherwise, raise an error
        if not os.path.exists(parc_file):
            raise FileNotFoundError("The annotation file does not exist.")
        else:
            # Reading the annot file
            sparc_data = cltfree.AnnotParcellation(
                parc_file=parc_file,
            )

        if isinstance(surf_file, str):
            # Checking if the file exists if the file is a string. If exists, read the file and create the object
            # Otherwise, raise an error
            if not os.path.exists(surf_file):
                raise FileNotFoundError("The surface file does not exist.")
            else:
                # Reading the surface file
                surf = cltsurf.Surface(surface_file=surf_file)

        elif isinstance(surf_file, cltsurf.Surface):
            # If the file is an object, copy the object
            surf = copy.deepcopy(surf_file)

    coords = surf.mesh.points
    cells = surf.mesh.GetPolys()
    c = _vtk.vtk_to_numpy(cells.GetConnectivityArray())
    o = _vtk.vtk_to_numpy(cells.GetOffsetsArray())
    faces = split(c, o[1:-1])
    faces = np.squeeze(faces)
    if not include_unknown:
        tmp_names = sparc_data.regnames
        unk_indexes = cltmisc.get_indexes_by_substring(
            tmp_names, ["medialwall", "unknown", "corpuscallosum"]
        ).astype(int)

        if len(unk_indexes) > 0:
            # get the values of the unknown regions
            unk_codes = sparc_data.regtable[unk_indexes, 4]
            unk_vert = np.isin(sparc_data.codes, unk_codes)

            sparc_data.codes[unk_vert] = 0
            sparc_data.regnames = np.delete(sparc_data.regnames, unk_indexes).tolist()
            sparc_data.regtable = np.delete(sparc_data.regtable, unk_indexes, axis=0)

    # Setting the vertex values to 0 if the values are not in the table
    # Set to 0 the values of sparc_data.codes that are not in the regtable
    unique_codes = np.unique(sparc_data.codes)

    # Get the indexes of the unique codes that are not in the regtable
    not_in_table = np.setdiff1d(unique_codes, sparc_data.regtable[:, 4])

    # Set to 0 the values of the codes that are not in the regtable
    sparc_data.codes[np.isin(sparc_data.codes, not_in_table)] = 0

    dict_of_cols = {}

    # # Compute the whole hemisphere
    # temp = area_from_mesh(metric_vect[np.isin(sparc_data.codes, sparc_data.regtable[:, 4])], stats_list)
    # dict_of_cols['ctx-' + hemi + '-hemisphere'] = temp

    for regname in sparc_data.regnames:

        # Get the index of the region in the color table
        index = cltmisc.get_indexes_by_substring(
            sparc_data.regnames, regname, matchww=True
        )
        ind = np.where(sparc_data.codes == sparc_data.regtable[index, 4])

        temp = np.isin(faces, ind).astype(int)
        nps = np.sum(temp, axis=1)
        reg_faces_3v = np.squeeze(
            faces[np.where(nps == 3), :]
        )  # All the vertices belong to the region
        reg_faces_2v = np.squeeze(
            faces[np.where(nps == 2), :]
        )  # Two vertices belong to the region
        reg_faces_1v = np.squeeze(
            faces[np.where(nps == 1), :]
        )  # One vertex belong to the region

        if len(index):
            # metric_vect[sparc_data.codes == sparc_data.regtable[index, 4]]
            temp_3v, _ = area_from_mesh(coords, reg_faces_3v)
            temp_2v, _ = area_from_mesh(coords, reg_faces_2v)
            temp_1v, _ = area_from_mesh(coords, reg_faces_1v)

            dict_of_cols[regname] = [temp_3v + temp_2v + temp_1v]

        else:
            dict_of_cols[regname] = [0]

    df = pd.DataFrame.from_dict(dict_of_cols)

    # Insert the column at the begining of the dataframe
    df.insert(0, "ctx-" + hemi + "-hemisphere", df.sum(axis=1))

    colnames = df.columns.tolist()
    colnames = cltmisc.correct_names(colnames, prefix="ctx-" + hemi + "-")
    df.columns = colnames

    if format == "region":
        df.index = ["global"]
        df = df.reset_index()
        df = df.rename(columns={"index": "statistics"})

        # Convert the index to a column with name "metric"

    else:
        df = df.T
        df.columns = ["global"]

        # Converting the row names to a new column called statistics
        df = df.reset_index()
        df = df.rename(columns={"index": "region"})

    nrows = df.shape[0]

    # Inserting the units
    units = get_units("area")
    df.insert(0, "metric", ["area"] * nrows)
    df.insert(1, "units", units * nrows)

    return df


def compute_euler_fromsurf(
    surf_file: Union[str, cltsurf.Surface],
    hemi: str,  # Hemisphere id. It could be lh or rh
    format: str = "region",
) -> pd.DataFrame:
    """
    This method computes the Euler characteristic of a surface.

    Parameters
    ----------
    surf_file : str
        Path to the surface file.

    hemi : str
        Hemisphere id. It could be lh or rh.

    format : str, optional
        Format of the output. It could be "region" or "metric". The default is "region".
        With the "region" format, the output is a DataFrame with the regional values where each column
        represent the value of column metric for each specific region. With the "metric" format, the output
        is a DataFrame with the regional values where each column represent the value of a specific metric
        for each region.


    Returns
    -------
    df : pd.DataFrame
        DataFrame with the euler value.


    Examples
    --------
    >>> import clabtoolkit.morphometrytools as clmorphtools
    >>> import os
    >>> import pandas as pd
    >>> surf_file = '/opt/freesurfer/subjects/fsaverage/surf/lh.white'
    >>> annot_file = '/opt/freesurfer/subjects/fsaverage/label/lh.aparc.annot'
    >>> df = clmorphtools.compute_euler_fromsurf(surf_file, 'lh')
    >>> print(df.head())
    """

    # Detect if the format is not region or metric
    if format not in ["region", "metric"]:
        raise ValueError("The format should be region or metric.")

    if isinstance(surf_file, str):
        # Checking if the file exists if the file is a string. If exists, read the file and create the object
        # Otherwise, raise an error
        if not os.path.exists(surf_file):
            raise FileNotFoundError("The surface file does not exist.")
        else:
            # Reading the surface file
            surf = cltsurf.Surface(surface_file=surf_file)

    elif isinstance(surf_file, cltsurf.Surface):
        # If the file is an object, copy the object
        surf = copy.deepcopy(surf_file)

    coords = surf.mesh.points
    cells = surf.mesh.GetPolys()
    c = _vtk.vtk_to_numpy(cells.GetConnectivityArray())
    o = _vtk.vtk_to_numpy(cells.GetOffsetsArray())
    faces = split(c, o[1:-1])
    faces = np.squeeze(faces)

    euler = euler_from_mesh(coords, faces)

    dict_of_cols = {}
    dict_of_cols["ctx-" + hemi + "-hemisphere"] = [euler]
    df = pd.DataFrame.from_dict(dict_of_cols)

    colnames = df.columns.tolist()
    colnames = cltmisc.correct_names(colnames, prefix="ctx-" + hemi + "-")
    df.columns = colnames

    if format == "region":
        df.index = ["global"]
        df = df.reset_index()
        df = df.rename(columns={"index": "statistics"})

        # Convert the index to a column with name "metric"

    else:
        df = df.T
        df.columns = ["global"]

        # Converting the row names to a new column called statistics
        df = df.reset_index()
        df = df.rename(columns={"index": "region"})

    nrows = df.shape[0]

    # Inserting the units
    units = get_units("euler")
    df.insert(0, "metric", ["euler"] * nrows)
    df.insert(1, "units", units * nrows)

    return df


def area_from_mesh(coords, faces):
    """
    This method computes the area of a mesh given the coordinates of the vertices and the faces of the mesh.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of the vertices of the mesh. The shape of the array should be (n,3) where n is the number of vertices.

    faces : np.ndarray
        Faces of the mesh. The shape of the array should be (m,3) where m is the number of faces.

    Returns
    -------
    face_area : float
        Total area of the mesh.

    tri_area : np.ndarray
        Area of each triangle in the mesh.

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    >>> faces = np.array([[0,1,2],[1,2,3]])
    >>> area_from_mesh(coords, faces)
    (1.0, array([0.5, 0.5]))

    """

    # Computing the distances between the vertices of the faces
    d12 = np.power(
        np.sum(np.power(coords[faces[:, 0], :] - coords[faces[:, 1], :], 2), axis=1),
        0.5,
    )
    d13 = np.power(
        np.sum(np.power(coords[faces[:, 0], :] - coords[faces[:, 2], :], 2), axis=1),
        0.5,
    )
    d23 = np.power(
        np.sum(np.power(coords[faces[:, 1], :] - coords[faces[:, 2], :], 2), axis=1),
        0.5,
    )

    # Computing the perimeter of the triangles
    per = (d12 + d23 + d13) / 2

    # Computing the area of the triangles
    tri_area = np.power(per * (per - d12) * (per - d13) * (per - d23), 0.5) / 100
    face_area = np.sum(tri_area)  # cm2

    return face_area, tri_area


def euler_from_mesh(coords, faces):
    """
    This method computes the Euler characteristic of a mesh given the coordinates of the vertices and the faces of the mesh.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of the vertices of the mesh. The shape of the array should be (n,3) where n is the number of vertices.

    faces : np.ndarray
        Faces of the mesh. The shape of the array should be (m,3) where m is the number of faces.

    Returns
    -------
    euler : float
        Euler characteristic of the mesh.


    Examples
    --------
    >>> import numpy as np
    >>> coords = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    >>> faces = np.array([[0,1,2],[1,2,3]])
    >>> euler_from_mesh(coords, faces)


    """
    # Step 1: Count vertices
    V = np.shape(coords)[0]

    # Step 2: Count faces
    F = np.shape(faces)[0]

    # Step 3: Count unique edges
    # Create an array of all edges from faces
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])

    # Sort each edge pair (to treat (v1, v2) the same as (v2, v1)) and remove duplicates
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    # Count edges
    E = len(edges)

    # Calculate Euler's characteristic
    euler = V - E + F

    return euler


def stats_from_vector(metric_vect, stats_list):
    """
    This method computes the statistics from a vector.

    Parameters
    ----------
    metric_vect : np.ndarray
        Vector with the values of the metric.

    stats_list : list
        List of statistics to compute.

    Returns
    -------

    out_vals : list
        List with the computed statistics.

    """

    stats_list = list(map(lambda x: x.lower(), stats_list))  # Converting to lower case

    out_vals = []
    for v in stats_list:
        if v == "mean":
            val = np.mean(metric_vect)

        if v == "median":
            val = np.median(metric_vect)

        if v == "std":
            val = np.std(metric_vect)

        if v == "min":
            val = np.min(metric_vect)

        if v == "max":
            val = np.max(metric_vect)

        out_vals.append(val)
    return out_vals


def get_units(metrics: Union[str, list], metrics_json: str = None) -> list:
    """
    This method returns the units of a specific metric.

    Parameters
    ----------
    metrics : str or list
        Name of the metrics. It could be a string or a list of strings.

    metrics_json : str, optional
        Path to the json file with the information of the metrics. The default is None.
        If None, the method uses the default json file.

    Returns
    -------
    units : list
        Units of the supplied metrics.

    Examples
    --------
    >>> import clabtoolkit.morphometrytools as clmorphtools
    >>> clmorphtools.units_from_metric('thickness')
    ['mm']
    """

    if isinstance(metrics, str):
        metrics = [metrics]

    if metrics_json is None:
        metrics_json = os.path.join(
            os.path.dirname(__file__), "config", "metrics_units.json"
        )
    else:
        if not os.path.isfile(metrics_json):
            raise ValueError(
                "Please, provide a valid JSON file containing the units dictionary."
            )

    with open(metrics_json) as f:
        metric_dict = json.load(f)

    # get dictionary keys
    metric_keys = metric_dict.keys()
    # lower all the metric_keys
    metric_keys = list(map(lambda x: x.lower(), metric_keys))

    # Search for the metric in the dictionary
    units = []

    for metric in metrics:
        if metric.lower() in metric_keys:
            units.append(metric_dict[metric.lower()])
        else:
            units.append("unknown")

    return units


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############          Methods dedicated to compute metrics from parcellations           ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################


def compute_reg_val_fromparcellation(
    metric_file: Union[str, np.ndarray],
    parc_file: Union[str, cltparc.Parcellation],
    metric: str = "unknown",
    stats_list: Union[str, list] = ["mean", "median", "std", "min", "max"],
    format: str = "region",
) -> pd.DataFrame:
    """
    This method computes the regional values of a scalar map inside each region of a specified parcellation.

    Parameters
    ----------
    metric_file : str
        Path to the volumetric scalar map file. It represents the voxel-wise values of the metric.

    parc_file : str
        Path to the parcellation file.

    metric : str
        Name of the metric. It is used to create the column names of the output DataFrame.

    stats_list : Union[str, list], optional
        List of statistics to compute. The default is ["mean", "median", "std", "min", "max"].


    format : str, optional
        Format of the output. It could be "region" or "metric". The default is "region".
        With the "region" format, the output is a DataFrame with the regional values where each column
        represent the value of column metric for each specific region. With the "metric" format, the output
        is a DataFrame with the regional values where each column represent the value of a specific metric
        for each region.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with the regional values.
    metric_vect : np.ndarray


    Examples
    --------
    >>> import clabtoolkit.morphometrytools as clmorphtools
    >>> import os
    >>> import pandas as pd
    >>> metric_file = os.path.join('..', 'data', 'metric.nii.gz')
    >>> parc_file = os.path.join('..', 'data', 'parcellation.nii.gz')
    >>> df = clmorphtools.compute_reg_val_fromparcellation(metric_file, parc_file)
    >>> print(df.head())

    """

    # Detecting if the stats_list is a string
    if isinstance(stats_list, str):
        stats_list = [stats_list]

    # Detect if the format is not region or metric
    if format not in ["region", "metric"]:
        raise ValueError("The format should be region or metric.")

    # Detecting if the needed parcellation file as a string or an object
    if isinstance(parc_file, str):
        # Checking if the file exists if the file is a string. If exists, read the file and create the object
        # Otherwise, raise an error
        if not os.path.exists(parc_file):
            raise FileNotFoundError("The annotation file does not exist.")
        else:
            # Reading the annot file
            vparc_data = cltparc.Parcellation(
                parc_file=parc_file,
            )

    elif isinstance(parc_file, cltfree.AnnotParcellation):
        # If the file is an object, copy the object
        vparc_data = copy.deepcopy(parc_file)

    # Detecting if the needed metric file as a string or an object. If the file is a string, check if the file exists
    # Otherwise, raise an error
    if isinstance(metric_file, str):
        if not os.path.exists(metric_file):
            raise FileNotFoundError("The metric file does not exist.")
        else:
            # Reading the vertex-wise metric file
            metric_vol = nib.load(metric_file).get_fdata()

    elif isinstance(metric_file, np.ndarray):
        metric_vol = metric_file

    # Converting to lower case
    stats_list = list(map(lambda x: x.lower(), stats_list))  # Converting to lower case

    if not include_unknown:
        tmp_names = sparc_data.regnames
        unk_indexes = cltmisc.get_indexes_by_substring(
            tmp_names, ["medialwall", "unknown", "corpuscallosum"]
        ).astype(int)

        if len(unk_indexes) > 0:
            # get the values of the unknown regions
            unk_codes = sparc_data.regtable[unk_indexes, 4]
            unk_vert = np.isin(sparc_data.codes, unk_codes)

            sparc_data.codes[unk_vert] = 0
            sparc_data.regnames = np.delete(sparc_data.regnames, unk_indexes).tolist()
            sparc_data.regtable = np.delete(sparc_data.regtable, unk_indexes, axis=0)

    # Setting the vertex values to 0 if the values are not in the table
    # Set to 0 the values of sparc_data.codes that are not in the regtable
    unique_codes = np.unique(sparc_data.codes)

    # Get the indexes of the unique codes that are not in the regtable
    not_in_table = np.setdiff1d(unique_codes, sparc_data.regtable[:, 4])

    # Set to 0 the values of the codes that are not in the regtable
    sparc_data.codes[np.isin(sparc_data.codes, not_in_table)] = 0

    sts = np.unique(sparc_data.codes)
    # Remove the 0
    sts = sts[sts != 0]
    nreg = len(sts)

    dict_of_cols = {}

    # Values for each region in the annot
    df = pd.DataFrame()

    # Compute the whole hemisphere
    temp = stats_from_vector(
        metric_vect[np.isin(sparc_data.codes, sparc_data.regtable[:, 4])], stats_list
    )
    dict_of_cols["ctx-" + hemi + "-hemisphere"] = temp

    for regname in sparc_data.regnames:

        # Get the index of the region in the color table
        index = cltmisc.get_indexes_by_substring(
            sparc_data.regnames, regname, matchww=True
        )

        if len(index):
            temp = stats_from_vector(
                metric_vect[sparc_data.codes == sparc_data.regtable[index, 4]],
                stats_list,
            )
            dict_of_cols[regname] = temp

        else:
            dict_of_cols[regname] = np.zeros_like(stats_list).tolist()

    df = pd.DataFrame.from_dict(dict_of_cols)

    colnames = df.columns.tolist()
    colnames = cltmisc.correct_names(colnames, prefix="ctx-" + hemi + "-")
    df.columns = colnames

    if format == "region":
        df.index = stats_list

        # Converting the row names to a new column called statistics
        df = df.reset_index()
        df = df.rename(columns={"index": "statistics"})

        # Convert the index to a column with name "metric"

    else:
        df = df.T
        df.columns = stats_list

        # Converting the row names to a new column called statistics
        df = df.reset_index()
        df = df.rename(columns={"index": "region"})

    nrows = df.shape[0]

    # Inserting the units
    units = get_units(metric)
    df.insert(0, "metric", [metric] * nrows)
    df.insert(1, "units", units * nrows)

    # Insert a column at the begining of the dataframe

    return df, metric_vect
