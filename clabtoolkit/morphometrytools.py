import os
from typing import Union, Tuple, Optional, Dict, List
import copy
from pyvista import _vtk, PolyData
from numpy import split, ndarray
import json
import warnings
import tempfile

import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

# Importing local modules
from . import misctools as cltmisc
from . import surfacetools as cltsurf
from . import parcellationtools as cltparc
from . import bidstools as cltbids
from . import freesurfertools as cltfree


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############      Section 1: Methods dedicated to compute metrics from surfaces         ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def compute_reg_val_fromannot(
    metric_file: Union[str, np.ndarray],
    parc_file: Union[str, cltfree.AnnotParcellation],
    hemi: str,
    output_table: str = None,
    metric: str = "unknown",
    units: str = None,
    stats_list: Union[str, list] = ["value", "median", "std", "min", "max"],
    table_type: str = "metric",
    include_unknown: bool = False,
    include_global: bool = True,
    add_bids_entities: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[str]]:
    """
    Compute regional statistics from a surface metric file and an annotation file.

    This function extracts regional values by combining vertex-wise surface metrics with
    anatomical parcellation data. It supports various statistical measures and output formats.

    Parameters
    ----------
    metric_file : str or np.ndarray
        Path to the surface map file or array containing metric values for each vertex.

    parc_file : str or cltfree.AnnotParcellation
        Path to the annotation file or AnnotParcellation object defining regions.

    hemi : str
        Hemisphere identifier ('lh' or 'rh').

    output_table : str, optional
        Path to save the resulting table. If None, the table is not saved.

    metric : str, default="unknown"
        Name of the metric being analyzed. Used for naming columns in the output DataFrame.

    units : str, optional
        Units of the metric. If None, units are determined from the metric name.

    stats_list : str or list, default=["value", "median", "std", "min", "max"]
        Statistics to compute for each region. Note: "value" is equivalent to the mean.

    table_type : str, default="metric"
        Output format specification:
        - "metric": Each column represents a specific statistic for each region
        - "region": Each column represents a region, with rows for different statistics

    include_unknown : bool, default=False
        Whether to include non-anatomical regions (medialwall, unknown, corpuscallosum).

    include_global : bool, default=True
        Whether to include hemisphere-wide statistics in the output.

    add_bids_entities : bool, default=True
        Whether to include BIDS entities as columns in the resulting DataFrame.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the computed regional statistics.

    metric_vect : np.ndarray
        Array of metric values.

    output_path : str or None
        Path where the table was saved, or None if no table was saved.

    Examples
    --------
    Basic usage with default parameters:

    >>> import os
    >>> import clabtoolkit.morphometrytools as morpho
    >>> hemi = 'lh'
    >>> metric_name = 'thickness'
    >>> fs_dir = os.environ.get('FREESURFER_HOME')
    >>> metric_file = os.path.join(fs_dir, 'subjects', 'bert', 'surf', f'{hemi}.{metric_name}')
    >>> parc_file = os.path.join(fs_dir, 'subjects', 'bert', 'label', f'{hemi}.aparc.annot')
    >>> df_region, metric_values, _ = morpho.compute_reg_val_fromannot(
    ...     metric_file, parc_file, hemi, metric=metric_name, include_global=False
    ... )

    Using region format for output:

    >>> df_metric, _, _ = morpho.compute_reg_val_fromannot(
    ...     metric_file, parc_file, hemi, metric=metric_name,
    ...     include_global=False, table_type="region", add_bids_entities=True
    ... )

    Including hemisphere-wide statistics and saving to file:

    >>> output_path = '/path/to/output/regional_stats.csv'
    >>> df_global, _, saved_path = morpho.compute_reg_val_fromannot(
    ...     metric_file, parc_file, hemi, output_table=output_path,
    ...     metric=metric_name, include_global=True
    ... )
    >>> print(df_global.head())
    """

    # Input validation
    if isinstance(stats_list, str):
        stats_list = [stats_list]

    stats_list = [stat.lower() for stat in stats_list]

    if table_type not in ["region", "metric"]:
        raise ValueError(
            f"Invalid table_type: '{table_type}'. Expected 'region' or 'metric'."
        )

    # Process parcellation file
    if isinstance(parc_file, str):
        if not os.path.exists(parc_file):
            raise FileNotFoundError(f"Annotation file not found: {parc_file}")

        sparc_data = cltfree.AnnotParcellation()
        sparc_data.load_from_file(parc_file=parc_file)

    elif isinstance(parc_file, cltfree.AnnotParcellation):
        sparc_data = copy.deepcopy(parc_file)
    else:
        raise TypeError(
            f"parc_file must be a string or AnnotParcellation object, got {type(parc_file)}"
        )

    # Process metric file
    filename = ""
    if isinstance(metric_file, str):
        if not os.path.exists(metric_file):
            raise FileNotFoundError(f"Metric file not found: {metric_file}")

        metric_vect = nib.freesurfer.io.read_morph_data(metric_file)
        filename = metric_file
    elif isinstance(metric_file, np.ndarray):
        metric_vect = metric_file
    else:
        raise TypeError(
            f"metric_file must be a string or numpy array, got {type(metric_file)}"
        )

    # Filter unknown regions if needed
    if not include_unknown:
        tmp_names = sparc_data.regnames
        unk_indexes = cltmisc.get_indexes_by_substring(
            tmp_names, ["medialwall", "unknown", "corpuscallosum"]
        ).astype(int)

        if len(unk_indexes) > 0:
            unk_codes = sparc_data.regtable[unk_indexes, 4]
            unk_vert = np.isin(sparc_data.codes, unk_codes)

            sparc_data.codes[unk_vert] = 0
            sparc_data.regnames = np.delete(sparc_data.regnames, unk_indexes).tolist()
            sparc_data.regtable = np.delete(sparc_data.regtable, unk_indexes, axis=0)

    # Clean up codes that don't exist in the region table
    unique_codes = np.unique(sparc_data.codes)
    not_in_table = np.setdiff1d(unique_codes, sparc_data.regtable[:, 4])
    sparc_data.codes[np.isin(sparc_data.codes, not_in_table)] = 0

    # Get unique valid region codes
    sts = np.unique(sparc_data.codes)
    sts = sts[sts != 0]

    # Prepare data structures for results
    dict_of_cols = {}

    # Compute global hemisphere statistics if requested
    if include_global:
        valid_vertices = np.isin(sparc_data.codes, sparc_data.regtable[:, 4])
        global_stats = stats_from_vector(metric_vect[valid_vertices], stats_list)
        dict_of_cols[f"ctx-{hemi}-hemisphere"] = global_stats

    # Compute statistics for each region
    for regname in sparc_data.regnames:
        index = cltmisc.get_indexes_by_substring(
            sparc_data.regnames, regname, match_entire_word=True
        )

        if len(index):
            region_mask = sparc_data.codes == sparc_data.regtable[index, 4]
            region_stats = stats_from_vector(metric_vect[region_mask], stats_list)
            dict_of_cols[regname] = region_stats
        else:
            dict_of_cols[regname] = [0] * len(stats_list)

    # Create DataFrame
    df = pd.DataFrame.from_dict(dict_of_cols)

    # Add column prefixes
    colnames = df.columns.tolist()
    colnames = cltmisc.correct_names(colnames, prefix=f"ctx-{hemi}-")
    df.columns = colnames

    # Format table according to specified type
    if table_type == "region":
        # Create region-oriented table
        df.index = [stat_name.title() for stat_name in stats_list]
        df = df.reset_index()
        df = df.rename(columns={"index": "Statistics"})
    else:
        # Create metric-oriented table
        df = df.T
        df.columns = [stat_name.title() for stat_name in stats_list]
        df = df.reset_index()
        df = df.rename(columns={"index": "Region"})

        # Split region names into components
        reg_names = df["Region"].str.split("-", expand=True)
        df.insert(0, "Supraregion", reg_names[0])
        df.insert(1, "Hemisphere", reg_names[1])

    # Add metadata columns
    nrows = df.shape[0]

    if units is None:
        units = get_units(metric)[0]

    df.insert(0, "Source", ["vertices"] * nrows)
    df.insert(1, "Metric", [metric] * nrows)
    df.insert(2, "Units", [units] * nrows)
    df.insert(3, "MetricFile", [filename] * nrows)

    # Add BIDS entities if requested
    if add_bids_entities and isinstance(metric_file, str):
        ent_list = cltbids.entities4table()
        df_add = cltbids.entities_to_table(
            filepath=metric_file, entities_to_extract=ent_list
        )
        df = cltmisc.expand_and_concatenate(df_add, df)

    # Save table if requested
    if output_table is not None:
        output_dir = os.path.dirname(output_table)
        if output_dir and not os.path.exists(output_dir):
            raise FileNotFoundError(
                f"Directory does not exist: {output_dir}. Please create the directory before saving."
            )

        df.to_csv(output_table, sep="\t", index=False)

    return df, metric_vect, output_table


####################################################################################################
def compute_reg_area_fromsurf(
    surf_file: Union[str, cltsurf.Surface],
    parc_file: Union[str, cltfree.AnnotParcellation],
    hemi: str,
    table_type: str = "metric",
    surf_type: str = "",
    include_unknown: bool = False,
    include_global: bool = True,
    add_bids_entities: bool = True,
    output_table: str = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Compute surface area for each region defined in an annotation file.

    This function calculates the area for anatomical regions by combining
    surface mesh data with parcellation information. It supports different
    output formats and can include global hemisphere measurements.

    Parameters
    ----------
    surf_file : str or cltsurf.Surface
        Path to the surface file or Surface object containing mesh data.
    parc_file : str or cltfree.AnnotParcellation
        Path to the annotation file or AnnotParcellation object defining regions.
    hemi : str
        Hemisphere identifier ('lh' or 'rh').
    table_type : str, default="metric"
        Output format specification:
        - "metric": Each row represents a region with area value in a column
        - "region": Each column represents a region with area values in rows
    surf_type : str, default=""
        Description of the surface type (e.g., "white", "pial"). Used for metadata.
    include_unknown : bool, default=False
        Whether to include non-anatomical regions (medialwall, unknown, corpuscallosum).
    include_global : bool, default=True
        Whether to include hemisphere-wide area calculations in the output.
    add_bids_entities : bool, default=True
        Whether to include BIDS entities as columns in the resulting DataFrame.
    output_table : str, optional
        Path to save the resulting table. If None, the table is not saved.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the computed regional area values.
    output_path : str or None
        Path where the table was saved, or None if no table was saved.

    Examples
    --------
    Basic usage with default parameters:

    >>> import os
    >>> import clabtoolkit.morphometrytools as morpho
    >>> fs_dir = os.environ.get('FREESURFER_HOME')
    >>> surf_file = os.path.join(fs_dir, 'subjects', 'fsaverage', 'surf', 'lh.white')
    >>> parc_file = os.path.join(fs_dir, 'subjects', 'fsaverage', 'label', 'lh.aparc.annot')
    >>> df_area, _ = morpho.compute_reg_area_fromsurf(surf_file, parc_file, 'lh', surf_type="white")
    >>> print(df_area.head())

    Using region format for output:

    >>> df_region, _ = morpho.compute_reg_area_fromsurf(
    ...     surf_file, parc_file, 'lh',
    ...     table_type="region", surf_type="white", include_global=False
    ... )
    >>> print(df_region.head())

    Using Surface and AnnotParcellation objects:

    >>> import clabtoolkit.surfacetools as cltsurf
    >>> import clabtoolkit.freesurfertools as cltfree
    >>> surf = cltsurf.Surface(surface_file=surf_file)
    >>> annot = cltfree.AnnotParcellation(parc_file=parc_file)
    >>> df_obj, _ = morpho.compute_reg_area_fromsurf(surf, annot, 'lh', surf_type="white")
    >>> print(df_obj.head())

    Saving results to a file:

    >>> output_path = '/path/to/area_stats.tsv'
    >>> df_out, saved_path = morpho.compute_reg_area_fromsurf(
    ...     surf_file, parc_file, 'lh', output_table=output_path, surf_type="white"
    ... )
    >>> print(f"Table saved to: {saved_path}")
    """
    # Input validation
    if table_type not in ["region", "metric"]:
        raise ValueError(
            f"Invalid table_type: '{table_type}'. Expected 'region' or 'metric'."
        )

    # Process parcellation file
    if isinstance(parc_file, str):
        if not os.path.exists(parc_file):
            raise FileNotFoundError(f"Annotation file not found: {parc_file}")

        sparc_data = cltfree.AnnotParcellation()
        sparc_data.load_from_file(parc_file=parc_file)

    elif isinstance(parc_file, cltfree.AnnotParcellation):
        sparc_data = copy.deepcopy(parc_file)
    else:
        raise TypeError(
            f"parc_file must be a string or AnnotParcellation object, got {type(parc_file)}"
        )

    # Process surface file
    filename = ""
    if isinstance(surf_file, str):
        if not os.path.exists(surf_file):
            raise FileNotFoundError(f"Surface file not found: {surf_file}")

        surf = cltsurf.Surface(surface_file=surf_file)
        filename = surf_file
    elif isinstance(surf_file, cltsurf.Surface):
        surf = copy.deepcopy(surf_file)
    else:
        raise TypeError(
            f"surf_file must be a string or Surface object, got {type(surf_file)}"
        )

    # Extract mesh data
    coords = surf.mesh.points
    cells = surf.mesh.GetPolys()
    c = _vtk.vtk_to_numpy(cells.GetConnectivityArray())
    o = _vtk.vtk_to_numpy(cells.GetOffsetsArray())
    faces = split(c, o[1:-1])
    faces = np.squeeze(faces)

    # Filter unknown regions if needed
    if not include_unknown:
        tmp_names = sparc_data.regnames
        unk_indexes = cltmisc.get_indexes_by_substring(
            tmp_names, ["medialwall", "unknown", "corpuscallosum"]
        ).astype(int)

        if len(unk_indexes) > 0:
            unk_codes = sparc_data.regtable[unk_indexes, 4]
            unk_vert = np.isin(sparc_data.codes, unk_codes)

            sparc_data.codes[unk_vert] = 0
            sparc_data.regnames = np.delete(sparc_data.regnames, unk_indexes).tolist()
            sparc_data.regtable = np.delete(sparc_data.regtable, unk_indexes, axis=0)

    # Clean up codes that don't exist in the region table
    unique_codes = np.unique(sparc_data.codes)
    not_in_table = np.setdiff1d(unique_codes, sparc_data.regtable[:, 4])
    sparc_data.codes[np.isin(sparc_data.codes, not_in_table)] = 0

    # Calculate area for each region
    dict_of_cols = {}

    for regname in sparc_data.regnames:
        # Get the index of the region in the color table
        index = cltmisc.get_indexes_by_substring(
            sparc_data.regnames, regname, match_entire_word=True
        )

        if len(index):
            # Find vertices belonging to this region
            ind = np.where(sparc_data.codes == sparc_data.regtable[index, 4])

            # Identify faces with different numbers of vertices in this region
            temp = np.isin(faces, ind).astype(int)
            nps = np.sum(temp, axis=1)

            # Group faces by how many vertices belong to the region
            reg_faces_3v = np.squeeze(
                faces[np.where(nps == 3), :]
            )  # All vertices in region
            reg_faces_2v = np.squeeze(
                faces[np.where(nps == 2), :]
            )  # Two vertices in region
            reg_faces_1v = np.squeeze(
                faces[np.where(nps == 1), :]
            )  # One vertex in region

            # Calculate area for each group
            temp_3v, _ = area_from_mesh(coords, reg_faces_3v)
            temp_2v, _ = area_from_mesh(coords, reg_faces_2v)
            temp_1v, _ = area_from_mesh(coords, reg_faces_1v)

            # Sum areas
            dict_of_cols[regname] = [temp_3v + temp_2v + temp_1v]
        else:
            dict_of_cols[regname] = [0]

    # Create DataFrame
    df = pd.DataFrame.from_dict(dict_of_cols)

    # Add global area if requested
    if include_global:
        df.insert(0, f"ctx-{hemi}-hemisphere", df.sum(axis=1))

    # Add column prefixes
    colnames = df.columns.tolist()
    colnames = cltmisc.correct_names(colnames, prefix=f"ctx-{hemi}-")
    df.columns = colnames

    # Format table according to specified type
    if table_type == "region":
        # Create region-oriented table
        df.index = ["Value"]
        df = df.reset_index()
        df = df.rename(columns={"index": "Statistics"})
    else:
        # Create metric-oriented table
        df = df.T
        df.columns = ["Value"]
        df = df.reset_index()
        df = df.rename(columns={"index": "Region"})

        # Split region names into components
        reg_names = df["Region"].str.split("-", expand=True)
        df.insert(0, "Supraregion", reg_names[0])
        df.insert(1, "Hemisphere", reg_names[1])

    # Add metadata columns
    nrows = df.shape[0]
    units = get_units("area")[0]

    df.insert(0, "Source", [surf_type] * nrows)
    df.insert(1, "Metric", ["area"] * nrows)
    df.insert(2, "Units", [units] * nrows)
    df.insert(3, "MetricFile", [filename] * nrows)

    # Add BIDS entities if requested
    if add_bids_entities and isinstance(parc_file, str):
        ent_list = cltbids.entities4table()
        df_add = cltbids.entities_to_table(
            filepath=parc_file, entities_to_extract=ent_list
        )
        df = cltmisc.expand_and_concatenate(df_add, df)

    # Save table if requested
    if output_table is not None:
        output_dir = os.path.dirname(output_table)
        if output_dir and not os.path.exists(output_dir):
            raise FileNotFoundError(
                f"Directory does not exist: {output_dir}. Please create the directory before saving."
            )

        df.to_csv(output_table, sep="\t", index=False)

    return df, output_table


####################################################################################################
def compute_euler_fromsurf(
    surf_file: Union[str, cltsurf.Surface],
    hemi: str,
    output_table: str = None,
    table_type: str = "metric",
    surf_type: str = "",
    add_bids_entities: bool = True,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Compute the Euler characteristic of a surface mesh.

    This function calculates the Euler characteristic (χ = V - E + F) of a surface mesh,
    which is a topological invariant that provides information about the surface's topology.

    Parameters
    ----------
    surf_file : str or cltsurf.Surface
        Path to the surface file or Surface object containing the mesh.
    hemi : str
        Hemisphere identifier ('lh' or 'rh').
    output_table : str, optional
        Path to save the resulting table. If None, the table is not saved.
    table_type : str, default="metric"
        Output format specification:
        - "metric": Each column represents a specific metric for each region
        - "region": Each column represents a region, with rows for different metrics
    surf_type : str, default=""
        Type of surface (e.g., "white", "pial") for metadata. If empty, determined from filename.
    add_bids_entities : bool, default=True
        Whether to include BIDS entities as columns in the resulting DataFrame.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the computed Euler characteristic.
    output_path : str or None
        Path where the table was saved, or None if no table was saved.

    Examples
    --------
    Basic usage with default parameters:

    >>> import os
    >>> import clabtoolkit.morphometrytools as morpho
    >>> fs_dir = os.environ.get('FREESURFER_HOME')
    >>> hemi = 'lh'
    >>> surf_file = os.path.join(fs_dir, 'subjects', 'bert', 'surf', f'{hemi}.white')
    >>> df, _ = morpho.compute_euler_fromsurf(surf_file, hemi)
    >>> print(df.head())

    Using region format for output:

    >>> df_region, _ = morpho.compute_euler_fromsurf(
    ...     surf_file, hemi, table_type="region", add_bids_entities=True
    ... )
    >>> print(df_region.head())

    Saving results to a file:

    >>> output_path = '/path/to/output/euler_stats.csv'
    >>> df_saved, saved_path = morpho.compute_euler_fromsurf(
    ...     surf_file, hemi, output_table=output_path
    ... )
    >>> print(f"Table saved to: {saved_path}")

    Notes
    -----
    The Euler characteristic (χ) is calculated as χ = V - E + F, where:
    - V is the number of vertices
    - E is the number of edges
    - F is the number of faces

    For a closed, orientable surface without boundaries, the Euler characteristic
    is related to the genus (g) by the formula: χ = 2 - 2g.
    """
    # Input validation
    if table_type not in ["region", "metric"]:
        raise ValueError(
            f"Invalid table_type: '{table_type}'. Expected 'region' or 'metric'."
        )

    # Process surface file
    filename = ""
    if isinstance(surf_file, str):
        if not os.path.exists(surf_file):
            raise FileNotFoundError(f"Surface file not found: {surf_file}")

        surf = cltsurf.Surface(surface_file=surf_file)
        filename = surf_file

        # Extract surface type from filename if not provided
        if not surf_type and os.path.basename(surf_file).split(".")[-1] not in [
            "gii",
            "vtk",
        ]:
            surf_type = os.path.basename(surf_file).split(".")[-1]
    elif isinstance(surf_file, cltsurf.Surface):
        surf = copy.deepcopy(surf_file)
    else:
        raise TypeError(
            f"surf_file must be a string or Surface object, got {type(surf_file)}"
        )

    # Extract mesh components
    coords = surf.mesh.points
    cells = surf.mesh.GetPolys()
    c = _vtk.vtk_to_numpy(cells.GetConnectivityArray())
    o = _vtk.vtk_to_numpy(cells.GetOffsetsArray())
    faces = split(c, o[1:-1])
    faces = np.squeeze(faces)

    # Compute Euler characteristic
    euler = euler_from_mesh(coords, faces)

    # Create dictionary for DataFrame
    dict_of_cols = {}
    dict_of_cols[f"ctx-{hemi}-hemisphere"] = [euler]

    # Create DataFrame
    df = pd.DataFrame.from_dict(dict_of_cols)

    # Add column prefixes
    colnames = df.columns.tolist()
    colnames = cltmisc.correct_names(colnames, prefix=f"ctx-{hemi}-")
    df.columns = colnames

    # Format table according to specified type
    if table_type == "region":
        # Create region-oriented table
        df.index = ["Value"]
        df = df.reset_index()
        df = df.rename(columns={"index": "Statistics"})
    else:
        # Create metric-oriented table
        df = df.T
        df.columns = ["Value"]
        df = df.reset_index()
        df = df.rename(columns={"index": "Region"})

        # Split region names into components
        reg_names = df["Region"].str.split("-", expand=True)
        df.insert(0, "Supraregion", reg_names[0])
        df.insert(1, "Hemisphere", reg_names[1])

    # Add metadata columns
    nrows = df.shape[0]
    units = (
        get_units("euler")[0]
        if isinstance(get_units("euler"), list)
        else get_units("euler")
    )

    df.insert(0, "Source", [surf_type] * nrows)
    df.insert(1, "Metric", ["euler"] * nrows)
    df.insert(2, "Units", [units] * nrows)
    df.insert(3, "MetricFile", [filename] * nrows)

    # Add BIDS entities if requested
    if add_bids_entities and isinstance(surf_file, str):
        ent_list = cltbids.entities4table()
        df_add = cltbids.entities_to_table(
            filepath=surf_file, entities_to_extract=ent_list
        )
        df = cltmisc.expand_and_concatenate(df_add, df)

    # Save table if requested
    output_path = None
    if output_table is not None:
        output_dir = os.path.dirname(output_table)
        if output_dir and not os.path.exists(output_dir):
            raise FileNotFoundError(
                f"Directory does not exist: {output_dir}. Please create the directory before saving."
            )

        df.to_csv(output_table, sep="\t", index=False)
        output_path = output_table

    return df, output_path


####################################################################################################
def area_from_mesh(coords: np.ndarray, faces: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute the total area and per-triangle areas of a mesh surface.

    This function calculates the area of each triangle in a mesh using Heron's formula
    and returns both the total surface area and individual triangle areas.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of the vertices of the mesh.
        Shape must be (n, 3) where n is the number of vertices.
        Each row contains the [x, y, z] coordinates of a vertex.

    faces : np.ndarray
        Triangular faces of the mesh defined by vertex indices.
        Shape must be (m, 3) where m is the number of faces.
        Each row contains three indices referring to vertices in the coords array.

    Returns
    -------
    face_area : float
        Total surface area of the mesh in square centimeters (cm²).

    tri_area : np.ndarray
        Array of areas for each triangle in the mesh in square centimeters (cm²).
        Shape is (m,) where m is the number of faces.

    Notes
    -----
    The function uses Heron's formula to calculate the area of each triangle:
        Area = √(s(s-a)(s-b)(s-c))
    where s is the semi-perimeter: s = (a + b + c)/2, and a, b, c are the side lengths.

    The resulting areas are converted to square centimeters (cm²) by dividing by 100
    (assuming the input coordinates are in millimeters).

    Examples
    --------
    Calculate area of a simple mesh with two triangles:

    >>> import numpy as np
    >>> coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    >>> faces = np.array([[0, 1, 2], [1, 3, 2]])
    >>> total_area, triangle_areas = area_from_mesh(coords, faces)
    >>> print(f"Total area: {total_area:.4f} cm²")
    Total area: 1.0000 cm²
    >>> print(f"Triangle areas: {triangle_areas}")
    Triangle areas: [0.5 0.5]

    Calculate area of a pyramid:

    >>> coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]])
    >>> faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 2, 1], [0, 3, 2]])
    >>> total_area, _ = area_from_mesh(coords, faces)
    >>> print(f"Total area: {total_area:.4f} cm²")
    Total area: ...
    """
    # Input validation
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (n, 3), got {coords.shape}")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (m, 3), got {faces.shape}")

    if np.any(faces >= coords.shape[0]) or np.any(faces < 0):
        raise ValueError("faces contains invalid vertex indices")

    # Extract vertex coordinates for each face
    v1 = coords[faces[:, 0]]
    v2 = coords[faces[:, 1]]
    v3 = coords[faces[:, 2]]

    # Compute edge lengths using Euclidean distance
    d12 = np.sqrt(np.sum((v1 - v2) ** 2, axis=1))
    d23 = np.sqrt(np.sum((v2 - v3) ** 2, axis=1))
    d31 = np.sqrt(np.sum((v3 - v1) ** 2, axis=1))

    # Compute semi-perimeter for each triangle
    s = (d12 + d23 + d31) / 2

    # Compute area of each triangle using Heron's formula
    # Division by 100 converts from mm² to cm²
    tri_area = np.sqrt(np.maximum(0, s * (s - d12) * (s - d23) * (s - d31))) / 100

    # Compute total mesh area
    face_area = np.sum(tri_area)

    return face_area, tri_area


####################################################################################################
def euler_from_mesh(coords: np.ndarray, faces: np.ndarray) -> int:
    """
    Compute the Euler characteristic of a mesh surface.

    The Euler characteristic (χ) is a topological invariant that describes the shape or
    structure of a topological space regardless of how it is bent or deformed. For a mesh,
    it is calculated as χ = V - E + F, where V is the number of vertices, E is the number
    of edges, and F is the number of faces.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of the vertices of the mesh.
        Shape must be (n, 3) where n is the number of vertices.
        Each row contains the [x, y, z] coordinates of a vertex.

    faces : np.ndarray
        Triangular faces of the mesh defined by vertex indices.
        Shape must be (m, 3) where m is the number of faces.
        Each row contains three indices referring to vertices in the coords array.

    Returns
    -------
    euler : int
        Euler characteristic of the mesh.
        For a closed manifold surface of genus g, χ = 2 - 2g.
        - Sphere: χ = 2 (genus 0)
        - Torus: χ = 0 (genus 1)
        - Double torus: χ = -2 (genus 2)

    Notes
    -----
    The Euler characteristic provides information about the topology of a mesh:
    - For closed, orientable surfaces: χ = 2 - 2g, where g is the genus (number of "holes")
    - For surfaces with boundaries (like cortical surfaces): χ = 2 - 2g - b, where b is the
    number of boundary components

    A change in the Euler characteristic can indicate topological defects in a surface.

    Examples
    --------
    Calculate Euler characteristic of a tetrahedron (a closed surface):

    >>> import numpy as np
    >>> coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    >>> euler = euler_from_mesh(coords, faces)
    >>> print(f"Euler characteristic: {euler}")
    Euler characteristic: 2

    Calculate Euler characteristic of a simple two-triangle surface:

    >>> coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    >>> faces = np.array([[0, 1, 2], [1, 2, 3]])
    >>> euler = euler_from_mesh(coords, faces)
    >>> print(f"Euler characteristic: {euler}")
    Euler characteristic: 1
    """
    # Input validation
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (n, 3), got {coords.shape}")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (m, 3), got {faces.shape}")

    if np.any(faces >= coords.shape[0]) or np.any(faces < 0):
        raise ValueError("faces contains invalid vertex indices")

    # Step 1: Count vertices
    V = coords.shape[0]

    # Step 2: Count faces
    F = faces.shape[0]

    # Step 3: Count unique edges
    # Create an array of all edges from faces
    edges = np.vstack(
        [
            faces[:, [0, 1]],  # First edge of each face
            faces[:, [1, 2]],  # Second edge of each face
            faces[:, [2, 0]],  # Third edge of each face
        ]
    )

    # Sort each edge to ensure (v1,v2) and (v2,v1) are treated as the same edge
    edges = np.sort(edges, axis=1)

    # Remove duplicate edges using unique
    unique_edges = np.unique(edges, axis=0)

    # Count edges
    E = len(unique_edges)

    # Calculate Euler characteristic
    euler = V - E + F

    return euler


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############     Section 2: Methods dedicated to compute metrics from parcellations     ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def compute_reg_val_fromparcellation(
    metric_file: Union[str, np.ndarray],
    parc_file: Union[str, cltparc.Parcellation, np.ndarray],
    output_table: str = None,
    metric: str = "unknown",
    units: str = None,
    stats_list: Union[str, list] = ["value", "median", "std", "min", "max"],
    table_type: str = "metric",
    exclude_by_code: Union[list, np.ndarray] = None,
    exclude_by_name: Union[list, str] = None,
    add_bids_entities: bool = True,
    region_prefix: str = "supra-side",
    interp_method: str = "linear",
) -> Tuple[pd.DataFrame, np.ndarray, Optional[str]]:
    """
    Compute regional statistics from a volumetric metric map and a parcellation.

    This function extracts regional values by combining voxel-wise volumetric metrics with
    parcellation data defining anatomical regions. It supports various statistical measures
    and output formats to facilitate regional analysis of volumetric neuroimaging data.

    If the metric and parcellation files have different resolutions, the metric data will be
    automatically resampled to match the parcellation's resolution.

    Parameters
    ----------
    metric_file : str or np.ndarray
        Path to the volumetric metric file or array containing metric values for each voxel.
        If array, it should have the same dimensions as the parcellation data.

    parc_file : str, cltparc.Parcellation, or np.ndarray
        Path to the parcellation file, Parcellation object, or numpy array defining regions.
        Each unique integer value in the array represents a different anatomical region.

    output_table : str, optional
        Path to save the resulting table. If None, the table is not saved.

    metric : str, default="unknown"
        Name of the metric being analyzed. Used for naming columns in the output DataFrame
        and determining appropriate units.

    units : str, optional
        Units of the metric being analyzed. If None, units are determined based on the metric.
        Supported units include "intensity", "thickness", "area", "euler", "volume", etc.
        If not specified, the function will attempt to infer units from the metric name.

    stats_list : str or list, default=["value", "median", "std", "min", "max"]
        Statistics to compute for each region. Note: "value" is equivalent to the mean.
        Supported statistics: "value", "median", "std", "min", "max", "count", "sum".

    table_type : str, default="metric"
        Output format specification:
        - "metric": Each column represents a specific statistic for each region (regions as rows)
        - "region": Each column represents a region, with rows for different statistics

    exclude_by_code : list or np.ndarray, optional
        Region codes to exclude from the analysis. If None, no regions are excluded by code.
        Useful for excluding regions like ventricles or non-brain tissue.

    exclude_by_name : list or str, optional
        Region names to exclude from the analysis. If None, no regions are excluded by name.
        Example: ["Ventricles", "White-Matter"] to focus only on gray matter regions.

    add_bids_entities : bool, default=True
        Whether to include BIDS entities as columns in the resulting DataFrame.
        This extracts subject, session, and other metadata from the filename.

    region_prefix : str, default="region-unknown-"
        Prefix to use for region names when they cannot be determined from the parcellation object.
        The prefix will be combined with the region index number.

    interp_method : str, default="linear"
        Interpolation method to use when resampling the metric data to match parcellation resolution.
        Options include: "linear", "nearest", "cubic". Use "nearest" for categorical data.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the computed regional statistics.

    metric_data : np.ndarray
        Array of metric values used in the calculation.

    output_path : str or None
        Path where the table was saved, or None if no table was saved.

    Examples
    --------
    Basic usage with default parameters:

    >>> import os
    >>> import clabtoolkit.morphometrytools as morpho
    >>> # Define paths to sample data
    >>> metric_file = os.path.join('data', 'sub-01', 'anat', 'sub-01_T1w_intensity.nii.gz')
    >>> parc_file = os.path.join('data', 'sub-01', 'anat', 'sub-01_T1w_parcellation.nii.gz')
    >>> # Compute regional statistics
    >>> df, metric_values, _ = morpho.compute_reg_val_fromparcellation(
    ...     metric_file, parc_file, metric='intensity'
    ... )
    >>> # Display the first few rows of results
    >>> print(f"Number of regions: {df.shape[0]}")
    >>> print(df[['Region', 'Value', 'Median', 'Std']].head())

    Using region format for output (regions as columns, statistics as rows):

    >>> df_region, _, _ = morpho.compute_reg_val_fromparcellation(
    ...     metric_file, parc_file, metric='intensity',
    ...     table_type="region", add_bids_entities=True
    ... )
    >>> # View statistics across regions
    >>> print(df_region[['Statistics', 'brain-brain-wholebrain']].head())

    Working with images that have different resolutions:

    >>> # High-resolution functional metric with lower-resolution anatomical parcellation
    >>> metric_file = os.path.join('data', 'sub-01', 'func', 'sub-01_task-rest_bold.nii.gz')
    >>> parc_file = os.path.join('data', 'sub-01', 'anat', 'sub-01_T1w_parcellation.nii.gz')
    >>> # The function will automatically resample the metric to the parcellation's space
    >>> df, _, _ = morpho.compute_reg_val_fromparcellation(
    ...     metric_file, parc_file, metric='bold',
    ...     interpolation='linear'  # Use linear interpolation for continuous data
    ... )
    >>> print(df.head())

    Computing only specific statistics for each region:

    >>> df_custom_stats, _, _ = morpho.compute_reg_val_fromparcellation(
    ...     metric_file, parc_file, metric='FA',
    ...     stats_list=['median', 'std'], table_type="metric"
    ... )
    >>> # View only median and standard deviation
    >>> print(df_custom_stats[['Region', 'Median', 'Std']].head())

    Excluding specific regions from analysis:

    >>> # Exclude regions by name
    >>> exclude_names = ["brain-left-ventricle", "brain-right-ventricle"]
    >>> df_filtered, _, _ = morpho.compute_reg_val_fromparcellation(
    ...     metric_file, parc_file, metric='thickness',
    ...     exclude_by_name=exclude_names
    ... )
    >>> # Check that ventricles are not in results
    >>> ventricle_count = sum(1 for r in df_filtered['Region'] if 'ventricle' in r.lower())
    >>> print(f"Ventricle regions in results: {ventricle_count}")

    Saving the results to a file:

    >>> output_path = os.path.join('results', 'sub-01_regional_intensity.tsv')
    >>> df_saved, _, saved_path = morpho.compute_reg_val_fromparcellation(
    ...     metric_file, parc_file, output_table=output_path,
    ...     metric='intensity'
    ... )
    >>> print(f"Table saved to: {saved_path}")
    >>> # You can load this table later with pandas
    >>> import pandas as pd
    >>> df_loaded = pd.read_csv(saved_path, sep='\t')

    Working with in-memory data instead of files:

    >>> import numpy as np
    >>> import nibabel as nib
    >>> # Load data into memory first
    >>> metric_obj = nib.load(metric_file)
    >>> metric_data = metric_obj.get_fdata()
    >>> parc_obj = nib.load(parc_file)
    >>> parc_data = parc_obj.get_fdata()
    >>> # Process the in-memory arrays
    >>> df_memory, _, _ = morpho.compute_reg_val_fromparcellation(
    ...     metric_data, parc_data, metric='intensity',
    ...     add_bids_entities=False  # No BIDS entities for in-memory data
    ... )
    >>> print(df_memory.head())

    Notes
    -----
    This function is designed for volumetric data, extracting statistics from voxel-wise
    metrics within each region defined by a parcellation. For surface-based metrics,
    consider using `compute_reg_val_fromannot` instead.

    The function handles both file paths and in-memory arrays, making it versatile for
    different workflows. When working with arrays directly, ensure the metric and
    parcellation arrays have the same dimensions.

    When metric and parcellation images have different resolutions, the metric data is
    automatically resampled to match the parcellation's resolution using the specified
    interpolation method. For continuous metrics (like intensity), linear or cubic
    interpolation is recommended. For categorical data, use 'nearest' interpolation.

    When working with BIDS-formatted data, setting `add_bids_entities=True` will extract
    subject, session, and other metadata from the filename to include in the output table.

    See Also
    --------
    compute_reg_val_fromannot : Similar function for surface-based metrics and annotations
    """

    # Input validation
    if isinstance(stats_list, str):
        stats_list = [stats_list]

    stats_list = [stat.lower() for stat in stats_list]

    if table_type not in ["region", "metric"]:
        raise ValueError(
            f"Invalid table_type: '{table_type}'. Expected 'region' or 'metric'."
        )

    if interp_method not in ["linear", "nearest", "cubic"]:
        raise ValueError(
            f"Invalid interpolation: '{interp_method}'. Expected 'linear', 'nearest', or 'cubic'."
        )

    # Process parcellation file
    parc_img = None
    if isinstance(parc_file, str):
        if not os.path.exists(parc_file):
            raise FileNotFoundError(f"Parcellation file not found: {parc_file}")

        parc_img = nib.load(parc_file)
        vparc_data = cltparc.Parcellation(parc_file=parc_file)
    elif isinstance(parc_file, cltparc.Parcellation):
        vparc_data = copy.deepcopy(parc_file)
        if hasattr(vparc_data, "img"):
            parc_img = vparc_data.img
    elif isinstance(parc_file, np.ndarray):
        vparc_data = cltparc.Parcellation(parc_file=parc_file)
    else:
        raise TypeError(
            f"parc_file must be a string, Parcellation object, or numpy array, got {type(parc_file)}"
        )

    # Process metric file
    metric_img = None
    filename = ""
    if isinstance(metric_file, str):
        if not os.path.exists(metric_file):
            raise FileNotFoundError(f"Metric file not found: {metric_file}")

        metric_img = nib.load(metric_file)
        metric_vol = metric_img.get_fdata()
        filename = metric_file
    elif isinstance(metric_file, np.ndarray):
        metric_vol = metric_file
    else:
        raise TypeError(
            f"metric_file must be a string or numpy array, got {type(metric_file)}"
        )

    # Handle resolution mismatch when we have both images
    temp_file = None
    if metric_img is not None and parc_img is not None:
        # Check if dimensions or affines don't match
        metric_shape = metric_img.shape
        parc_shape = parc_img.shape

        if (metric_shape != parc_shape) or not np.allclose(
            metric_img.affine, parc_img.affine
        ):
            warnings.warn(
                f"Metric image ({metric_shape}) and parcellation image ({parc_shape}) have different "
                f"dimensions or orientations. Resampling metric to match parcellation."
            )

            # Resample metric to match parcellation space
            resampled_metric_img = resample_from_to(
                metric_img,
                parc_img,
                order={"linear": 1, "nearest": 0, "cubic": 3}[interp_method],
            )

            # Save to temporary file if needed for other operations
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
                temp_file = f.name

            nib.save(resampled_metric_img, temp_file)

            # Update metric volume with resampled data
            metric_vol = resampled_metric_img.get_fdata()

    # Check that dimensions match with parcellation data
    if metric_vol.shape != vparc_data.data.shape:
        # If we already resampled and still don't match, there's a problem
        if temp_file:
            os.unlink(temp_file)  # Clean up temp file
            raise ValueError(
                f"Resampled metric data shape {metric_vol.shape} still does not match "
                f"parcellation data shape {vparc_data.data.shape}. Please check your inputs."
            )
        else:
            raise ValueError(
                f"Metric data shape {metric_vol.shape} does not match parcellation data shape "
                f"{vparc_data.data.shape}. Use file inputs instead of arrays for automatic resampling."
            )

    # Apply exclusions if specified
    if exclude_by_code is not None:
        vparc_data.remove_by_code(codes2remove=exclude_by_code)

    if exclude_by_name is not None:
        vparc_data.remove_by_name(names2remove=exclude_by_name)

    # Prepare data structures for results
    dict_of_cols = {}

    # Compute global brain statistics (non-zero parcellation values)
    brain_mask = vparc_data.data != 0
    if np.any(brain_mask):  # Check if there are any non-zero values
        global_stats = stats_from_vector(metric_vol[brain_mask], stats_list)
        dict_of_cols["brain-brain-wholebrain"] = global_stats
    else:
        # Handle empty/invalid parcellation
        dict_of_cols["brain-brain-wholebrain"] = [0] * len(stats_list)

    # Compute statistics for each region
    # Use unique region indices from the data itself
    unique_indices = np.unique(vparc_data.data)
    unique_indices = unique_indices[unique_indices != 0]  # Exclude background

    for index in unique_indices:
        # Get region name from the parcellation object if available
        if hasattr(vparc_data, "name") and hasattr(vparc_data, "index"):
            idx_pos = np.where(np.array(vparc_data.index) == index)[0]
            if len(idx_pos) > 0:
                regname = vparc_data.name[idx_pos[0]]
            else:
                regname = cltmisc.create_names_from_indices(index, prefix=region_prefix)
        else:
            regname = cltmisc.create_names_from_indices(index, prefix=region_prefix)

        region_mask = vparc_data.data == index

        if np.any(region_mask):
            region_values = metric_vol[region_mask]
            if len(region_values) > 0:  # Check if there are any values
                region_stats = stats_from_vector(region_values, stats_list)
                dict_of_cols[regname] = region_stats
            else:
                dict_of_cols[regname] = [0] * len(stats_list)
        else:
            dict_of_cols[regname] = [0] * len(stats_list)

    # Check if we found any regions
    if len(dict_of_cols) == 0:
        if temp_file:
            os.unlink(temp_file)  # Clean up temp file
        raise ValueError("No valid regions found in the parcellation data")

    # Create DataFrame
    df = pd.DataFrame.from_dict(dict_of_cols)

    # Format table according to specified type
    if table_type == "region":
        # Create region-oriented table
        df.index = [stat_name.title() for stat_name in stats_list]
        df = df.reset_index()
        df = df.rename(columns={"index": "Statistics"})
    else:
        # Create metric-oriented table
        df = df.T
        df.columns = [stat_name.title() for stat_name in stats_list]
        df = df.reset_index()
        df = df.rename(columns={"index": "Region"})

        # Split region names into components
        reg_names = df["Region"].str.split("-", expand=True)

        # Safely handle region names that might not have 3 components
        if reg_names.shape[1] >= 3:
            df.insert(0, "Supraregion", reg_names[0])
            df.insert(1, "Hemisphere", reg_names[1])
        elif reg_names.shape[1] == 2:
            df.insert(0, "Supraregion", reg_names[0])
            df.insert(1, "Hemisphere", "unknown")
        else:
            df.insert(0, "Supraregion", "unknown")
            df.insert(1, "Hemisphere", "unknown")

    # Add metadata columns
    nrows = df.shape[0]
    if units is None:
        units = get_units(metric)

    if isinstance(units, list) and len(units) > 0:
        units = units[0]
    elif units is None or (isinstance(units, list) and len(units) == 0):
        units = "unknown"

    df.insert(0, "Source", ["volume"] * nrows)
    df.insert(1, "Metric", [metric] * nrows)
    df.insert(2, "Units", [units] * nrows)
    df.insert(3, "MetricFile", [filename] * nrows)

    # Add BIDS entities if requested
    if add_bids_entities and isinstance(metric_file, str):
        try:
            ent_list = cltbids.entities4table()
            df_add = cltbids.entities_to_table(
                filepath=metric_file, entities_to_extract=ent_list
            )
            df = cltmisc.expand_and_concatenate(df_add, df)
        except Exception as e:
            warnings.warn(f"Could not add BIDS entities: {str(e)}")

    # Save table if requested
    output_path = None
    if output_table is not None:
        output_dir = os.path.dirname(output_table)
        if output_dir and not os.path.exists(output_dir):
            raise FileNotFoundError(
                f"Directory does not exist: {output_dir}. Please create the directory before saving."
            )

        df.to_csv(output_table, sep="\t", index=False)
        output_path = output_table

    # Clean up temporary file if it exists
    if temp_file and os.path.exists(temp_file):
        os.unlink(temp_file)

    return df, metric_vol, output_path


####################################################################################################
def compute_reg_volume_fromparcellation(
    parc_file: Union[str, cltparc.Parcellation, np.ndarray],
    output_table: str = None,
    table_type: str = "metric",
    exclude_by_code: Union[list, np.ndarray] = None,
    exclude_by_name: Union[list, str] = None,
    include_by_code: Union[list, np.ndarray] = None,
    include_by_name: Union[list, str] = None,
    add_bids_entities: bool = True,
    region_prefix: str = "supra-side",
    include_global: bool = True,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Compute volume for all regions in a parcellation.

    This function calculates the volume of each region defined in a parcellation by counting
    the number of voxels in each region and multiplying by the voxel volume. It supports
    various output formats and can exclude specific regions from the analysis.

    Parameters
    ----------
    parc_file : str, cltparc.Parcellation, or np.ndarray
        Path to the parcellation file, Parcellation object, or numpy array defining regions.
        Each unique integer value in the array represents a different anatomical region.
    output_table : str, optional
        Path to save the resulting table. If None, the table is not saved.
    table_type : str, default="metric"
        Output format specification:
        - "metric": Each column represents a specific statistic for each region (regions as rows)
        - "region": Each column represents a region, with rows for different statistics
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
    add_bids_entities : bool, default=True
        Whether to include BIDS entities as columns in the resulting DataFrame.
        This extracts subject, session, and other metadata from the filename.
    region_prefix : str, default="supra-side"
        Prefix to use for region names when they cannot be determined from the parcellation object.
        The prefix will be combined with the region index number.
    include_global : bool, default=True
        Whether to include a the total volume in the output table.
        If True, adds a row for the total volume calculated from the parcellation.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the computed regional volumes.
    output_path : str or None
        Path where the table was saved, or None if no table was saved.

    Examples
    --------
    Basic usage with default parameters:

    >>> import os
    >>> import clabtoolkit.morphometrytools as morpho
    >>> # Define path to sample data
    >>> parc_file = os.path.join('data', 'sub-01', 'anat', 'sub-01_T1w_parcellation.nii.gz')
    >>> # Compute regional volumes
    >>> df, _ = morpho.compute_reg_volume_fromparcellation(parc_file)
    >>> # Display the first few rows of results
    >>> print(f"Number of regions: {df.shape[0]}")
    >>> print(df[['Region', 'Value']].head())

    Using region format for output (regions as columns):

    >>> df_region, _ = morpho.compute_reg_volume_fromparcellation(
    ...     parc_file, table_type="region", add_bids_entities=True
    ... )
    >>> # View volumes across regions
    >>> print(df_region.head())

    Excluding specific regions from analysis:

    >>> # Exclude regions by name
    >>> exclude_names = ["brain-left-ventricle", "brain-right-ventricle"]
    >>> df_filtered, _ = morpho.compute_reg_volume_fromparcellation(
    ...     parc_file, exclude_by_name=exclude_names
    ... )
    >>> # Check that ventricles are not in results
    >>> ventricle_count = sum(1 for r in df_filtered['Region'] if 'ventricle' in r.lower())
    >>> print(f"Ventricle regions in results: {ventricle_count}")

    Saving the results to a file:

    >>> output_path = os.path.join('results', 'sub-01_regional_volumes.tsv')
    >>> df_saved, saved_path = morpho.compute_reg_volume_fromparcellation(
    ...     parc_file, output_table=output_path
    ... )
    >>> print(f"Table saved to: {saved_path}")
    >>> # You can load this table later with pandas
    >>> import pandas as pd
    >>> df_loaded = pd.read_csv(saved_path, sep='\t')

    Working with in-memory data instead of files:

    >>> import numpy as np
    >>> import nibabel as nib
    >>> # Load data into memory first
    >>> parc_obj = nib.load(parc_file)
    >>> parc_data = parc_obj.get_fdata()
    >>> # Create a custom affine matrix (example: 1mm isotropic voxels)
    >>> affine = np.eye(4)
    >>> # Process the in-memory array
    >>> df_memory, _ = morpho.compute_reg_volume_fromparcellation(
    ...     parc_data, add_bids_entities=False
    ... )
    >>> print(df_memory.head())

    Notes
    -----
    This function calculates volumes in milliliters (ml) by default. The voxel volume is
    calculated from the affine transformation matrix of the parcellation image. For arrays
    without an affine matrix, an identity matrix is assumed (1mm isotropic voxels).

    The regional volumes are calculated by counting the number of voxels in each region
    and multiplying by the voxel volume in cubic millimeters, then dividing by 1000 to
    convert to milliliters.

    When working with BIDS-formatted data, setting `add_bids_entities=True` will extract
    subject, session, and other metadata from the filename to include in the output table.

    See Also
    --------
    compute_reg_val_fromparcellation : Calculate statistics for metric values within parcellation regions
    """

    # Input validation
    if table_type not in ["region", "metric"]:
        raise ValueError(
            f"Invalid table_type: '{table_type}'. Expected 'region' or 'metric'."
        )

    # Process parcellation file
    filename = ""
    if isinstance(parc_file, str):
        if not os.path.exists(parc_file):
            raise FileNotFoundError(f"Parcellation file not found: {parc_file}")

        vparc_data = cltparc.Parcellation(parc_file=parc_file)
        affine = vparc_data.affine
        filename = parc_file
    elif isinstance(parc_file, cltparc.Parcellation):
        vparc_data = copy.deepcopy(parc_file)
        affine = vparc_data.affine
        filename = vparc_data.parc_file
    elif isinstance(parc_file, np.ndarray):
        vparc_data = cltparc.Parcellation(parc_file=parc_file)
        affine = vparc_data.affine
        filename = "3darray"
    else:
        raise TypeError(
            f"parc_file must be a string, Parcellation object, or numpy array, got {type(parc_file)}"
        )

    # Apply exclusions if specified
    if exclude_by_code is not None:
        vparc_data.remove_by_code(codes2remove=exclude_by_code)

    if exclude_by_name is not None:
        vparc_data.remove_by_name(names2remove=exclude_by_name)

    # Apply inclusion if specified
    if include_by_code is not None:
        vparc_data.keep_by_code(codes2keep=include_by_code)

    if include_by_name is not None:
        vparc_data.keep_by_name(names2look=include_by_name)

    # Computing the voxel volume (in cubic mm)
    vox_size = np.linalg.norm(affine[:3, :3], axis=1)
    vox_vol = np.prod(vox_size)

    # Prepare data structures for results
    dict_of_cols = {}

    # Compute global volume for the entire brain (convert to ml by dividing by 1000)
    if include_global:
        brain_mask = vparc_data.data != 0
        if np.any(brain_mask):  # Check if there are any non-zero values
            global_volume_ml = np.sum(brain_mask) * vox_vol / 1000
            dict_of_cols["brain-brain-wholebrain"] = [global_volume_ml]
        else:
            # Handle empty/invalid parcellation
            dict_of_cols["brain-brain-wholebrain"] = [0]

    # Compute volume for each region
    # Use unique region indices from the data itself
    unique_indices = np.unique(vparc_data.data)
    unique_indices = unique_indices[unique_indices != 0]  # Exclude background

    for index in unique_indices:
        # Get region name from the parcellation object if available
        if hasattr(vparc_data, "name") and hasattr(vparc_data, "index"):
            idx_pos = np.where(np.array(vparc_data.index) == index)[0]
            if len(idx_pos) > 0:
                regname = vparc_data.name[idx_pos[0]]
            else:
                regname = cltmisc.create_names_from_indices(index, prefix=region_prefix)
        else:
            regname = cltmisc.create_names_from_indices(index, prefix=region_prefix)

        region_mask = vparc_data.data == index
        region_volume_ml = np.sum(region_mask) * vox_vol / 1000

        if region_volume_ml > 0:
            dict_of_cols[regname] = [region_volume_ml]
        else:
            dict_of_cols[regname] = [0]

    # Check if we found any regions
    if len(dict_of_cols) == 0:
        raise ValueError("No valid regions found in the parcellation data")

    # Create DataFrame
    df = pd.DataFrame.from_dict(dict_of_cols)

    # Format table according to specified type
    if table_type == "region":
        # Create region-oriented table
        df.index = ["Value"]
        df = df.reset_index()
        df = df.rename(columns={"index": "Statistics"})
    else:
        # Create metric-oriented table
        df = df.T
        df.columns = ["Value"]
        df = df.reset_index()
        df = df.rename(columns={"index": "Region"})

        # Split region names into components
        reg_names = df["Region"].str.split("-", expand=True)

        # Safely handle region names that might not have 3 components
        if reg_names.shape[1] >= 3:
            df.insert(0, "Supraregion", reg_names[0])
            df.insert(1, "Hemisphere", reg_names[1])
        elif reg_names.shape[1] == 2:
            df.insert(0, "Supraregion", reg_names[0])
            df.insert(1, "Hemisphere", "unknown")
        else:
            df.insert(0, "Supraregion", "unknown")
            df.insert(1, "Hemisphere", "unknown")

    # Add metadata columns
    nrows = df.shape[0]
    units = get_units("volume")
    if isinstance(units, list) and len(units) > 0:
        units = units[0]
    elif units is None or (isinstance(units, list) and len(units) == 0):
        units = "ml"

    df.insert(0, "Source", ["parcellation"] * nrows)
    df.insert(1, "Metric", ["volume"] * nrows)
    df.insert(2, "Units", [units] * nrows)
    df.insert(3, "MetricFile", [filename] * nrows)

    # Add BIDS entities if requested
    if add_bids_entities and isinstance(parc_file, str):
        try:
            ent_list = cltbids.entities4table()
            df_add = cltbids.entities_to_table(
                filepath=parc_file, entities_to_extract=ent_list
            )
            df = cltmisc.expand_and_concatenate(df_add, df)
        except Exception as e:
            warnings.warn(f"Could not add BIDS entities: {str(e)}")

    # Save table if requested
    output_path = None
    if output_table is not None:
        output_dir = os.path.dirname(output_table)
        if output_dir and not os.path.exists(output_dir):
            raise FileNotFoundError(
                f"Directory does not exist: {output_dir}. Please create the directory before saving."
            )

        df.to_csv(output_table, sep="\t", index=False)
        output_path = output_table

    return df, output_path


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############  Section 3: Methods dedicated to parse stats file from freesurfer results  ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def parse_freesurfer_global_fromaseg(
    stat_file: str,
    output_table: str = None,
    table_type: str = "metric",
    add_bids_entities: bool = True,
    include_missing: bool = True,
    config_json: str = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Parse global volume measurements from a FreeSurfer aseg.stats file.

    This function extracts key global volumetric measurements from FreeSurfer's aseg.stats file,
    including intracranial volume, brain volume, gray/white matter volumes, and ventricle
    volumes. It converts values to milliliters and organizes them into a structured DataFrame.

    Parameters
    ----------
    stat_file : str
        Path to the aseg.stats file generated by FreeSurfer.
    output_table : str, optional
        Path to save the resulting table. If None, the table is not saved.
    table_type : str, default="metric"
        Output format specification:
        - "metric": Each column represents a specific statistic for each region (regions as rows)
        - "region": Each column represents a region, with rows for different statistics
    add_bids_entities : bool, default=True
        Whether to include BIDS entities as columns in the resulting DataFrame.
        This extracts subject, session, and other metadata from the filename.
    include_missing : bool, default=True
        Whether to include missing values as zeros in the output. If False, missing values
        will be excluded from the DataFrame.
    config_json : str, optional
        Path to a JSON configuration file defining volume measurements to extract.
        If None, a default configuration will be used.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the extracted global volume measurements.
    output_path : str or None
        Path where the table was saved, or None if no table was saved.
    """

    # Verify if the file exists
    if not os.path.isfile(stat_file):
        raise FileNotFoundError(f"Stats file not found: {stat_file}")

    # Input validation
    if table_type not in ["region", "metric"]:
        raise ValueError(
            f"Invalid table_type: '{table_type}'. Expected 'region' or 'metric'."
        )

    # Load volume measurements from config file or use defaults
    if config_json and os.path.isfile(config_json):
        try:
            with open(config_json, "r") as f:
                config_data = json.load(f)
                # Check if the config has a "global" key
                if "global" in config_data:
                    volume_measurements = config_data["global"]
                else:
                    volume_measurements = config_data
        except Exception as e:
            warnings.warn(f"Error loading config file {config_json}: {e}")
            volume_measurements = get_stats_dictionary("global")
    else:
        volume_measurements = get_stats_dictionary("global")

    # Dictionary to store the extracted values
    extracted_values = {}

    # Read the stats file
    try:
        with open(stat_file, "r") as file:
            file_content = file.readlines()

            # Create dictionaries to store parsed data
            global_measures = {}  # For "# Measure" lines - global stats
            segmented_data = {}  # For tabular data - segmentation stats

            # First, parse all global measures (lines starting with "# Measure")
            for line in file_content:
                if line.startswith("# Measure"):
                    try:
                        parts = line.split(", ")
                        if len(parts) >= 4:
                            # Get description (which is parts[2]) and value (parts[3])
                            measure_description = parts[2].strip()
                            measure_value = float(parts[3].strip())
                            global_measures[measure_description] = measure_value

                            # Also store by the short name (parts[1]) for alternative lookup
                            if len(parts) >= 2:
                                short_name = parts[1].strip()
                                global_measures[short_name] = measure_value
                    except Exception as e:
                        warnings.warn(
                            f"Error parsing global measure line: {line.strip()}. Error: {e}"
                        )

            # Next, parse segmentation table (lines starting with a number)
            # First identify where the table starts
            table_start = False
            for i, line in enumerate(file_content):
                if line.startswith("# ColHeaders"):
                    table_start = True
                    continue

                if table_start and line.strip() and line.strip()[0].isdigit():
                    try:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Structure name is in position 4, volume is in position 3
                            seg_name = parts[4]
                            seg_volume = float(parts[3])
                            segmented_data[seg_name] = seg_volume
                    except Exception as e:
                        warnings.warn(
                            f"Error parsing table line: {line.strip()}. Error: {e}"
                        )

            # If we didn't find table data with the precise method, try a more general approach
            if not segmented_data:
                for line in file_content:
                    if line.strip() and line.strip()[0].isdigit():
                        try:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                seg_name = parts[4]
                                seg_volume = float(parts[3])
                                segmented_data[seg_name] = seg_volume
                        except Exception as e:
                            pass  # Silently skip lines that don't match expected format

            # Now extract the values based on the configuration
            for region_key, region_info in volume_measurements.items():
                value_found = False

                # Get configuration details
                key = region_info["key"]
                divisor = region_info.get("divisor", 1000)

                # STEP 1: Try to find in global measures
                if key in global_measures:
                    value = global_measures[key] / divisor
                    extracted_values[region_key] = [value]
                    value_found = True
                    continue

                # STEP 2: Try to find in segmented data
                if key in segmented_data:
                    value = segmented_data[key] / divisor
                    extracted_values[region_key] = [value]
                    value_found = True
                    continue

                # STEP 3: Try any alternate keys from config
                alt_keys = region_info.get("alternate_keys", [])
                for alt_key in alt_keys:
                    if alt_key in global_measures:
                        value = global_measures[alt_key] / divisor
                        extracted_values[region_key] = [value]
                        value_found = True
                        break
                    if alt_key in segmented_data:
                        value = segmented_data[alt_key] / divisor
                        extracted_values[region_key] = [value]
                        value_found = True
                        break

                if value_found:
                    continue

                # STEP 4: Fallback to line search (legacy method)
                if "index" in region_info:
                    for line in file_content:
                        if key in line:
                            try:
                                index = region_info["index"]
                                parts = line.split()

                                if index < 0:
                                    index = len(parts) + index

                                if 0 <= index < len(parts):
                                    value_str = parts[index].split(",")[0]
                                    value = float(value_str) / divisor
                                    extracted_values[region_key] = [value]
                                    value_found = True
                                    break
                            except (IndexError, ValueError) as e:
                                warnings.warn(
                                    f"Error parsing value for {region_key} using index: {e}"
                                )

                # If value not found and we're including missing values
                if not value_found and include_missing:
                    extracted_values[region_key] = [0.0]
                    warnings.warn(
                        f"Value for {region_key} (key: {key}) not found in {stat_file}"
                    )

    except Exception as e:
        raise RuntimeError(f"Error reading stats file {stat_file}: {e}")

    # Check if we found any values
    if not extracted_values:
        raise ValueError(f"No volume measurements found in {stat_file}")

    # Create a dataframe with the values
    df = pd.DataFrame.from_dict(extracted_values)

    # Format table according to specified type
    if table_type == "region":
        # Create region-oriented table
        df.index = ["Value"]
        df = df.reset_index()
        df = df.rename(columns={"index": "Statistics"})
    else:
        # Create metric-oriented table
        df = df.T
        df.columns = ["Value"]

        # Converting the row names to a new column called statistics
        df = df.reset_index()
        df = df.rename(columns={"index": "Region"})

        # Split region names into components
        reg_names = df["Region"].str.split("-", expand=True)

        # Safely handle region names that might not have 3 components
        if reg_names.shape[1] >= 3:
            df.insert(0, "Supraregion", reg_names[0])
            df.insert(1, "Hemisphere", reg_names[1])
        elif reg_names.shape[1] == 2:
            df.insert(0, "Supraregion", reg_names[0])
            df.insert(1, "Hemisphere", "unknown")
        else:
            df.insert(0, "Supraregion", "unknown")
            df.insert(1, "Hemisphere", "unknown")

    # Add metadata columns
    nrows = df.shape[0]
    units = get_units("volume")
    if isinstance(units, list) and len(units) > 0:
        units = units[0]
    elif units is None or (isinstance(units, list) and len(units) == 0):
        units = "ml"

    df.insert(0, "Source", ["statsfile"] * nrows)
    df.insert(1, "Metric", ["volume"] * nrows)
    df.insert(2, "Units", [units] * nrows)
    df.insert(3, "MetricFile", [stat_file] * nrows)

    # Add BIDS entities if requested
    if add_bids_entities:
        try:
            ent_list = cltbids.entities4table()
            df_add = cltbids.entities_to_table(
                filepath=stat_file, entities_to_extract=ent_list
            )
            df = cltmisc.expand_and_concatenate(df_add, df)
        except Exception as e:
            warnings.warn(f"Could not add BIDS entities: {str(e)}")

    # Save table if requested
    output_path = None
    if output_table is not None:
        output_dir = os.path.dirname(output_table)
        if output_dir and not os.path.exists(output_dir):
            raise FileNotFoundError(
                f"Directory does not exist: {output_dir}. Please create the directory before saving."
            )

        df.to_csv(output_table, sep="\t", index=False)
        output_path = output_table

    return df, output_path


####################################################################################################
def parse_freesurfer_stats_fromaseg(
    stat_file: str,
    output_table: str = None,
    table_type: str = "metric",
    add_bids_entities: bool = True,
    include_missing: bool = True,
    config_json: str = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Parse regional volume measurements from a FreeSurfer aseg.stats file.

    This function extracts volume measurements for specific brain regions from the tabular
    data section of FreeSurfer's aseg.stats file. It converts values to milliliters
    and organizes them into a structured DataFrame.

    Parameters
    ----------
    stat_file : str
        Path to the aseg.stats file generated by FreeSurfer.
    output_table : str, optional
        Path to save the resulting table. If None, the table is not saved.
    table_type : str, default="metric"
        Output format specification:
        - "metric": Each column represents a specific statistic for each region (regions as rows)
        - "region": Each column represents a region, with rows for different statistics
    add_bids_entities : bool, default=True
        Whether to include BIDS entities as columns in the resulting DataFrame.
        This extracts subject, session, and other metadata from the filename.
    include_missing : bool, default=True
        Whether to include missing values as zeros in the output. If False, missing values
        will be excluded from the DataFrame.
    config_json : str, optional
        Path to a JSON configuration file defining region measurements to extract.
        If None, a default configuration will be used.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the extracted region volume measurements.
    output_path : str or None
        Path where the table was saved, or None if no table was saved.

    Examples
    --------
    Basic usage with default parameters:

    >>> import os
    >>> import clabtoolkit.morphometrytools as morpho
    >>> # Define path to FreeSurfer stats file
    >>> stats_file = os.path.join('freesurfer', 'sub-01', 'stats', 'aseg.stats')
    >>> # Parse the stats file
    >>> df, _ = morpho.parse_freesurfer_stats_fromaseg(stats_file)
    >>> # Display the first few rows of results
    >>> print(df[['Region', 'Value']].head())

    Using region format (regions as columns, statistics as rows):

    >>> df_region, _ = morpho.parse_freesurfer_stats_fromaseg(
    ...     stats_file, table_type="region"
    ... )
    >>> # View volumes across regions
    >>> print(df_region.head())

    Saving the results to a file:

    >>> output_path = os.path.join('results', 'sub-01_fs-regional-volumes.tsv')
    >>> df_saved, saved_path = morpho.parse_freesurfer_stats_fromaseg(
    ...     stats_file, output_table=output_path
    ... )
    >>> print(f"Table saved to: {saved_path}")

    Notes
    -----
    This function extracts regional volume measurements from the table section of the aseg.stats file.
    The aseg.stats file contains a table with measurements for different brain regions, with
    each row representing a different segmented region.

    All volumes are converted to milliliters (ml) by dividing the FreeSurfer values
    (typically in mm³) by 1000.

    The aseg.stats file is typically found in the `[subject]/stats/` directory of a
    FreeSurfer output directory.

    See Also
    --------
    parse_freesurfer_global_fromaseg : Extract global volume measurements from aseg.stats
    """

    # Verify if the file exists
    if not os.path.isfile(stat_file):
        raise FileNotFoundError(f"Stats file not found: {stat_file}")

    # Input validation
    if table_type not in ["region", "metric"]:
        raise ValueError(
            f"Invalid table_type: '{table_type}'. Expected 'region' or 'metric'."
        )

    # Load region measurements from config file or use defaults
    if config_json and os.path.isfile(config_json):
        try:
            with open(config_json, "r") as f:
                config_data = json.load(f)
                # Check if the config has an "aseg" key
                if "aseg" in config_data:
                    region_measurements = config_data["aseg"]
                else:
                    region_measurements = config_data
        except Exception as e:
            warnings.warn(f"Error loading config file {config_json}: {e}")
            region_measurements = get_stats_dictionary("aseg")
    else:
        region_measurements = get_stats_dictionary("aseg")

    # Dictionary to store the extracted values
    extracted_values = {}

    # Read the stats file
    try:
        with open(stat_file, "r") as file:
            file_content = file.readlines()

            # Parse the tabular data from the aseg.stats file
            segmented_data = {}
            region_data = {}

            # First determine the column positions by looking for TableCol definitions
            column_indices = {}
            volume_index = 3  # Default volume index if not specified

            for line in file_content:
                if line.startswith("# TableCol"):
                    try:
                        parts = line.split()
                        if len(parts) >= 4 and "ColHeader" in line:
                            col_num = int(parts[2]) - 1  # Convert to 0-based index
                            col_name = parts[-1]
                            column_indices[col_name] = col_num
                            if col_name == "Volume_mm3":
                                volume_index = col_num
                    except (ValueError, IndexError) as e:
                        warnings.warn(
                            f"Error parsing TableCol line: {line.strip()}. Error: {e}"
                        )

            # Parse the table rows (lines starting with a number)
            for line in file_content:
                if line.strip() and line.strip()[0].isdigit():
                    try:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Structure name is typically in position 4
                            seg_name = parts[4]
                            # Use detected volume index or default to 3
                            vol_idx = (
                                volume_index if 0 <= volume_index < len(parts) else 3
                            )
                            seg_volume = float(parts[vol_idx])
                            segmented_data[seg_name] = seg_volume

                            # Also store additional information about the region
                            region_data[seg_name] = {
                                "SegId": int(parts[1]),
                                "NVoxels": int(parts[2]),
                                "Volume": seg_volume,
                            }
                    except Exception as e:
                        warnings.warn(
                            f"Error parsing table line: {line.strip()}. Error: {e}"
                        )

            # Extract values based on the configuration
            for region_key, region_info in region_measurements.items():
                value_found = False

                # Get configuration details
                key = region_info["key"]
                divisor = region_info.get("divisor", 1000)

                # Try to find in segmented data
                if key in segmented_data:
                    value = segmented_data[key] / divisor
                    extracted_values[region_key] = [value]
                    value_found = True
                    continue

                # Try any alternate keys from config
                alt_keys = region_info.get("alternate_keys", [])
                for alt_key in alt_keys:
                    if alt_key in segmented_data:
                        value = segmented_data[alt_key] / divisor
                        extracted_values[region_key] = [value]
                        value_found = True
                        break

                if value_found:
                    continue

                # Fallback to SegId lookup if provided
                seg_id = region_info.get("seg_id")
                if seg_id is not None:
                    for name, data in region_data.items():
                        if data["SegId"] == seg_id:
                            value = data["Volume"] / divisor
                            extracted_values[region_key] = [value]
                            value_found = True
                            break

                # If value not found and we're including missing values
                if not value_found and include_missing:
                    extracted_values[region_key] = [0.0]
                    warnings.warn(
                        f"Value for {region_key} (key: {key}) not found in {stat_file}"
                    )

    except Exception as e:
        raise RuntimeError(f"Error reading stats file {stat_file}: {e}")

    # Check if we found any values
    if not extracted_values:
        raise ValueError(f"No region measurements found in {stat_file}")

    # Create a dataframe with the values
    df = pd.DataFrame.from_dict(extracted_values)

    # Format table according to specified type
    if table_type == "region":
        # Create region-oriented table
        df.index = ["Value"]
        df = df.reset_index()
        df = df.rename(columns={"index": "Statistics"})
    else:
        # Create metric-oriented table
        df = df.T
        df.columns = ["Value"]

        # Converting the row names to a new column called statistics
        df = df.reset_index()
        df = df.rename(columns={"index": "Region"})

        # Split region names into components
        reg_names = df["Region"].str.split("-", expand=True)

        # Safely handle region names that might not have 3 components
        if reg_names.shape[1] >= 3:
            df.insert(0, "Supraregion", reg_names[0])
            df.insert(1, "Hemisphere", reg_names[1])
        elif reg_names.shape[1] == 2:
            df.insert(0, "Supraregion", reg_names[0])
            df.insert(1, "Hemisphere", "unknown")
        else:
            df.insert(0, "Supraregion", "unknown")
            df.insert(1, "Hemisphere", "unknown")

    # Add metadata columns
    nrows = df.shape[0]
    units = get_units("volume")
    if isinstance(units, list) and len(units) > 0:
        units = units[0]
    elif units is None or (isinstance(units, list) and len(units) == 0):
        units = "ml"

    df.insert(0, "Source", ["statsfile"] * nrows)
    df.insert(1, "Metric", ["volume"] * nrows)
    df.insert(2, "Units", [units] * nrows)
    df.insert(3, "MetricFile", [stat_file] * nrows)

    # Add BIDS entities if requested
    if add_bids_entities:
        try:
            ent_list = cltbids.entities4table()
            df_add = cltbids.entities_to_table(
                filepath=stat_file, entities_to_extract=ent_list
            )
            df = cltmisc.expand_and_concatenate(df_add, df)
        except Exception as e:
            warnings.warn(f"Could not add BIDS entities: {str(e)}")

    # Save table if requested
    output_path = None
    if output_table is not None:
        output_dir = os.path.dirname(output_table)
        if output_dir and not os.path.exists(output_dir):
            raise FileNotFoundError(
                f"Directory does not exist: {output_dir}. Please create the directory before saving."
            )

        df.to_csv(output_table, sep="\t", index=False)
        output_path = output_table

    return df, output_path


####################################################################################################
def parse_freesurfer_cortex_stats(
    stats_file: str,
    output_table: str = None,
    table_type: str = "metric",
    add_bids_entities: bool = True,
    hemi: str = None,
    config_json: str = None,
    include_metrics: list = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Parse cortical parcellation statistics from a FreeSurfer aparc.stats file.

    This function extracts regional measurements from FreeSurfer's aparc.stats files,
    including surface area, gray matter volume, cortical thickness, and curvature
    for each cortical region. It organizes the data into a structured DataFrame.

    Surface area is automatically converted from mm² to cm² (divided by 100)
    and volume is automatically converted from mm³ to cm³ (divided by 1000).

    Parameters
    ----------
    stats_file : str
        Path to the aparc.stats file generated by FreeSurfer (lh.aparc.stats or rh.aparc.stats).
    output_table : str, optional
        Path to save the resulting table. If None, the table is not saved.
    table_type : str, default="metric"
        Output format specification:
        - "metric": Each row represents a region, with columns for different metrics
        - "region": Each row represents a metric, with columns for different regions
    add_bids_entities : bool, default=True
        Whether to include BIDS entities as columns in the resulting DataFrame.
        This extracts subject, session, and other metadata from the filename.
    hemi : str, optional
        Hemisphere identifier ('lh' for left or 'rh' for right). If None, it will be
        automatically detected from the filename or the file content.
    config_json : str, optional
        Path to a JSON configuration file defining cortical metrics to extract.
        If None, a default configuration will be used.
    include_metrics : list, optional
        List of metrics to extract from the stats file. If provided, only these metrics
        will be extracted from the configuration. If None, all metrics in the configuration
        will be extracted.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the extracted cortical measurements.
    output_path : str or None
        Path where the table was saved, or None if no table was saved.

    Examples
    --------
    Basic usage with default parameters:

    >>> import os
    >>> import clabtoolkit.morphometrytools as morpho
    >>> # Define path to FreeSurfer stats file
    >>> stats_file = os.path.join('freesurfer', 'sub-01', 'stats', 'lh.aparc.stats')
    >>> # Parse the stats file
    >>> df, _ = morpho.parse_freesurfer_cortex_stats(stats_file)
    >>> # Display the results for thickness
    >>> thickness_df = df[df['Metric'] == 'thickness']
    >>> print(thickness_df[['Region', 'Value', 'Std']].head())

    Using a custom configuration file:

    >>> config_file = os.path.join('config', 'stats_mapping.json')
    >>> df, _ = morpho.parse_freesurfer_cortex_stats(stats_file, config_json=config_file)

    Extract only specific metrics:

    >>> df_area_vol, _ = morpho.parse_freesurfer_cortex_stats(
    ...     stats_file, include_metrics=["area", "volume"]
    ... )
    >>> print(df_area_vol['Metric'].unique())

    Using region format (metrics as rows, regions as columns):

    >>> df_region, _ = morpho.parse_freesurfer_cortex_stats(
    ...     stats_file, table_type="region"
    ... )
    >>> # View thickness across regions
    >>> thickness_row = df_region[df_region['Statistics'] == 'thickness']
    >>> print(thickness_row.iloc[:, :5])  # Print first 5 columns for thickness

    Saving the results to a file:

    >>> output_path = os.path.join('results', 'sub-01_lh_cortical-metrics.tsv')
    >>> df_saved, saved_path = morpho.parse_freesurfer_cortex_stats(
    ...     stats_file, output_table=output_path
    ... )
    >>> print(f"Table saved to: {saved_path}")

    Notes
    -----
    This function extracts metrics from aparc.stats files based on the configuration.
    By default, these include:
    - Surface area (SurfArea column) in cm² (converted from mm² by dividing by 100)
    - Gray matter volume (GrayVol column) in cm³ (converted from mm³ by dividing by 1000)
    - Cortical thickness (ThickAvg column) in mm (unchanged)
    - Thickness standard deviation (ThickStd column) in mm (unchanged)
    - Mean curvature (MeanCurv column) in mm⁻¹ (unchanged)

    The function automatically detects the hemisphere from the filename or file content
    if not specified.

    Cortical parcellation stats files (lh.aparc.stats and rh.aparc.stats) are typically
    found in the `[subject]/stats/` directory of a FreeSurfer output directory.

    See Also
    --------
    parse_freesurfer_global_fromaseg : Parse global volumes from aseg.stats file
    parse_freesurfer_stats_fromaseg : Parse regional volumes from aseg.stats file
    """

    # Verify if the file exists
    if not os.path.isfile(stats_file):
        raise FileNotFoundError(f"Stats file not found: {stats_file}")

    # Input validation
    if table_type not in ["region", "metric"]:
        raise ValueError(
            f"Invalid table_type: '{table_type}'. Expected 'region' or 'metric'."
        )

    # Detect hemisphere from filename or file content if not provided
    if hemi is None:
        # Try to detect from filename
        basename = os.path.basename(stats_file)
        if "lh." in basename:
            hemi = "lh"
        elif "rh." in basename:
            hemi = "rh"
        else:
            # Try to detect from file content
            with open(stats_file, "r") as f:
                content = f.read()
                if "# hemi lh" in content:
                    hemi = "lh"
                elif "# hemi rh" in content:
                    hemi = "rh"
                else:
                    warnings.warn(
                        f"Could not determine hemisphere from file: {stats_file}. Using 'lh' as default."
                    )
                    hemi = "lh"

    # Load metrics configuration from file or use defaults
    if config_json and os.path.isfile(config_json):
        try:
            with open(config_json, "r") as f:
                config_data = json.load(f)
                # Check if the config has a "cortex" key
                if "cortex" in config_data:
                    metric_mapping = config_data["cortex"]
                else:
                    metric_mapping = config_data

                # Update unit information in the configuration
                if (
                    "area" in metric_mapping
                    and metric_mapping["area"].get("unit") == "mm²"
                ):
                    metric_mapping["area"]["unit"] = "cm²"

                if (
                    "volume" in metric_mapping
                    and metric_mapping["volume"].get("unit") == "mm³"
                ):
                    metric_mapping["volume"]["unit"] = "cm³"

        except Exception as e:
            warnings.warn(f"Error loading config file {config_json}: {e}")
            metric_mapping = get_stats_dictionary("cortex")
    else:
        metric_mapping = get_stats_dictionary("cortex")

    # Filter metrics if include_metrics is provided
    if include_metrics:
        include_metrics = [m.lower() for m in include_metrics]
        metric_mapping = {
            k: v for k, v in metric_mapping.items() if k.lower() in include_metrics
        }

        # Validate that requested metrics exist
        if not metric_mapping:
            raise ValueError(
                f"None of the requested metrics {include_metrics} found in configuration."
            )

    # Validate metrics after filtering
    valid_metrics = list(metric_mapping.keys())
    if not valid_metrics:
        raise ValueError("No valid metrics found in configuration.")

    # Read the stats file
    try:
        with open(stats_file, "r") as file:
            lines = file.readlines()

        # Find the data section by looking for column headers
        col_headers_line = None
        column_headers = []
        for i, line in enumerate(lines):
            if "# ColHeaders" in line:
                col_headers_line = i
                column_headers = line.replace("# ColHeaders", "").strip().split()
                break

        # If column headers not found, try to find the end of the comment section
        if not column_headers:
            for i, line in enumerate(lines):
                if not line.startswith("#") and line.strip():
                    # This might be the start of the data section
                    if i > 0 and "# TableCol" in lines[i - 1]:
                        # The previous line was a table column definition, this is likely the data
                        parts = line.strip().split()
                        if len(parts) >= 7:  # Ensure enough columns
                            # Assume a fixed structure based on the example file
                            column_headers = [
                                "StructName",
                                "NumVert",
                                "SurfArea",
                                "GrayVol",
                                "ThickAvg",
                                "ThickStd",
                                "MeanCurv",
                                "GausCurv",
                                "FoldInd",
                                "CurvInd",
                            ]
                            col_headers_line = i - 1
                            print(
                                f"Inferred column headers at line {i}: {column_headers}"
                            )
                            break

        if not column_headers:
            # Final fallback: manually parse the TableCol definitions
            table_cols = {}
            for line in lines:
                if "# TableCol" in line and "ColHeader" in line:
                    try:
                        parts = line.split()
                        col_num = int(parts[2])
                        col_name = parts[-1]
                        table_cols[col_num] = col_name
                    except (ValueError, IndexError):
                        continue

            if table_cols:
                # Sort by column number
                column_headers = [table_cols[i] for i in sorted(table_cols.keys())]
                print(
                    f"Extracted column headers from TableCol definitions: {column_headers}"
                )

        if not column_headers:
            raise ValueError(f"Could not find column headers in {stats_file}")

        # Create column index mapping
        column_indices = {name: idx for idx, name in enumerate(column_headers)}

        # Determine where to start reading data
        data_lines = []
        if col_headers_line is not None:
            # Start from the line after the column headers
            data_start = col_headers_line + 1
        else:
            # Try to find the first non-comment line
            data_start = 0
            for i, line in enumerate(lines):
                if not line.startswith("#") and line.strip():
                    data_start = i
                    break

        # Extract data lines (non-comment, non-empty lines)
        for i in range(data_start, len(lines)):
            line = lines[i]
            if not line.startswith("#") and line.strip():
                data_lines.append(line)

        # Now parse the data lines to extract region metrics
        regions_data = []

        for line in data_lines:
            parts = line.strip().split()
            if len(parts) < len(column_headers):
                print(
                    f"Warning: Line has fewer parts ({len(parts)}) than headers ({len(column_headers)}): {line[:50]}..."
                )
                continue

            # Extract region name (first field in most aparc.stats files)
            region_name = parts[0]

            # Process each metric
            for metric_name, metric_info in metric_mapping.items():
                column = metric_info.get("column")

                # Get the column index
                col_idx = None
                if column in column_indices:
                    col_idx = column_indices[column]
                elif "index" in metric_info:
                    # Fallback to index if provided
                    col_idx = int(metric_info["index"])
                    if col_idx >= len(parts):
                        print(
                            f"Warning: Index {col_idx} out of range for line with {len(parts)} parts"
                        )
                        continue
                else:
                    print(
                        f"Warning: Could not find column {column} for metric {metric_name}"
                    )
                    continue

                # Get metric value
                try:
                    value = float(parts[col_idx])

                    # Apply unit conversions
                    # Convert area from mm² to cm²
                    if metric_name.lower() == "area" or column == "SurfArea":
                        value = value / 100.0  # mm² to cm²
                        # Update the unit in metric_info to reflect conversion
                        if "unit" in metric_info and metric_info["unit"] == "mm²":
                            metric_info["unit"] = "cm²"
                        elif "unit" not in metric_info:
                            metric_info["unit"] = "cm²"

                    # Convert volume from mm³ to cm³
                    elif metric_name.lower() == "volume" or column == "GrayVol":
                        value = value / 1000.0  # mm³ to cm³
                        # Update the unit in metric_info to reflect conversion
                        if "unit" in metric_info and metric_info["unit"] == "mm³":
                            metric_info["unit"] = "cm³"
                        elif "unit" not in metric_info:
                            metric_info["unit"] = "cm³"

                    # Get standard deviation if applicable
                    std_value = None
                    std_column = metric_info.get("std_index")

                    if std_column is not None and std_column not in (
                        None,
                        "null",
                        "None",
                    ):
                        if isinstance(std_column, int) or (
                            isinstance(std_column, str) and std_column.isdigit()
                        ):
                            std_idx = int(std_column)
                            if 0 <= std_idx < len(parts):
                                std_value = float(parts[std_idx])
                        elif (
                            "ThickStd" in column_indices
                            and metric_name.lower() == "thickness"
                        ):
                            std_idx = column_indices["ThickStd"]
                            std_value = float(parts[std_idx])

                    # Add to results
                    region_data = {
                        "Region": f"ctx-{hemi}-{region_name}",
                        "Metric": metric_name,
                        "Value": value,
                        "Source": metric_info.get("source", "statsfile"),
                        "Units": metric_info.get("unit", ""),
                    }

                    if std_value is not None:
                        region_data["Std"] = std_value

                    regions_data.append(region_data)
                except (IndexError, ValueError) as e:
                    print(f"Error parsing {metric_name} for region {region_name}: {e}")

        # Check if we found any data
        if not regions_data:
            print(f"Warning: No data parsed from {len(data_lines)} data lines")
            # Print a sample of the first data line for debugging
            if data_lines:
                print(f"Sample data line: {data_lines[0]}")
            raise ValueError(f"No cortical parcellation data found in {stats_file}")

        # Create DataFrame
        df = pd.DataFrame(regions_data)

        # Split region names into components
        df["Hemisphere"] = hemi
        df["Supraregion"] = "ctx"

        # Add metadata column for the source file
        df["MetricFile"] = stats_file

        # Reorder columns
        column_order = [
            "Source",
            "Metric",
            "Units",
            "MetricFile",
            "Supraregion",
            "Hemisphere",
            "Region",
            "Value",
        ]

        # Add Std if it exists
        if "Std" in df.columns:
            column_order.append("Std")

        # Filter columns to those that exist
        column_order = [col for col in column_order if col in df.columns]
        df = df[column_order]

        # Format table according to specified type
        if table_type == "region":
            # Create region-oriented table (pivot)
            value_cols = ["Value"]
            if "Std" in df.columns:
                value_cols.append("Std")

            pivot_df = pd.pivot_table(
                df,
                values=value_cols,
                index=[
                    "Source",
                    "Metric",
                    "Units",
                    "MetricFile",
                    "Supraregion",
                    "Hemisphere",
                ],
                columns="Region",
            )

            # Flatten the multi-index columns
            if isinstance(pivot_df.columns, pd.MultiIndex):
                pivot_df.columns = [
                    f"{col[0]}_{col[1]}" if col[0] != "" else col[1]
                    for col in pivot_df.columns
                ]

            # Reset index and extract metric as Statistics
            pivot_df = pivot_df.reset_index()
            pivot_df = pivot_df.rename(columns={"Metric": "Statistics"})

            # Final DataFrame
            df = pivot_df

        # Add BIDS entities if requested
        if add_bids_entities:
            try:
                ent_list = cltbids.entities4table()
                df_add = cltbids.entities_to_table(
                    filepath=stats_file, entities_to_extract=ent_list
                )
                df = cltmisc.expand_and_concatenate(df_add, df)
            except Exception as e:
                warnings.warn(f"Could not add BIDS entities: {str(e)}")

        # Save table if requested
        output_path = None
        if output_table is not None:
            output_dir = os.path.dirname(output_table)
            if output_dir and not os.path.exists(output_dir):
                raise FileNotFoundError(
                    f"Directory does not exist: {output_dir}. Please create the directory before saving."
                )

            df.to_csv(output_table, sep="\t", index=False)
            output_path = output_table

        return df, output_path

    except Exception as e:
        raise RuntimeError(f"Error parsing stats file {stats_file}: {e}")


####################################################################################################
def get_stats_dictionary(region_level: str = "global"):
    """
    Return the default global volume measurements configuration for FreeSurfer aseg.stats files.

    Returns
    -------
    dict
        Dictionary containing configuration for extracting global volume measurements.
    """

    # Get the absolute of this file
    cwd = os.path.dirname(os.path.abspath(__file__))
    mapping_stats_json = os.path.join(cwd, "config", "stats_mapping.json")

    with open(mapping_stats_json, encoding="utf-8") as f:
        mapp_dict = json.load(f)

    return mapp_dict[region_level]


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############   Section 5: Methods dedicated to extract metrics from connectivity        ############
############            matrices based on graph theory                                  ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def network_metrics_to_table(cmat, lut_name, cmat_met):

    if os.path.isfile(lut_name):

        # Reading the colorlut
        st_codes, st_names, st_colors = cltparc.Parcellation.read_luttable(lut_name)
    else:
        # Reading the tsv
        st_codes, st_names, st_colors = cltparc.Parcellation.read_luttable(tsv_name)

    net_metrics = [
        "degree",
        "strength",
        "clustering_coeff",
        "betw_centrality",
        "loc_efficiency",
        "glob_efficiency",
        "transitivity",
        "density_coeff",
    ]
    cmat_bin = cmat > 0
    deg_coeff = bct.degree.degrees_und(cmat_bin)
    str_coeff = bct.degree.strengths_und(cmat)
    clu_coeff = bct.clustering_coef_bu(cmat_bin)
    btw_cent = bct.betweenness_bin(cmat_bin)
    sho_path = bct.distance_bin(cmat_bin)
    loc_eff = bct.efficiency_bin(cmat_bin, local=True)

    trans_coeff_g = bct.transitivity_bu(cmat_bin)
    glob_eff_g = bct.efficiency_bin(cmat_bin)
    den_coeff_g = bct.density_und(cmat_bin)

    dict_of_cols = {}
    dict_of_cols["metric"] = ["conn_matrix_" + cmat_met] * len(net_metrics)
    dict_of_cols["statistics"] = net_metrics
    dict_of_cols["units"] = ["au"] * len(net_metrics)
    dict_of_cols["total_brain"] = [""] * (len(net_metrics) - 3) + [
        glob_eff_g,
        trans_coeff_g,
        den_coeff_g[0],
    ]

    if os.path.isfile(lut_name) or os.path.isfile(tsv_name):

        # Values for each region in the annot
        nreg = len(st_codes)
        # outvals = []
        # outnames = []
        for i in range(1, nreg + 1):
            if i < len(cmat):
                outvals = [
                    deg_coeff[i - 1],
                    str_coeff[i - 1],
                    clu_coeff[i - 1],
                    btw_cent[i - 1],
                    loc_eff[i - 1],
                ] + [""] * 3
            else:
                outvals = [""] * (len(net_metrics))  #

            dict_of_cols[st_names[i - 1]] = outvals

    df = pd.DataFrame.from_dict(dict_of_cols)
    return df


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############                        Section 6: Auxiliary methods                        ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def stats_from_vector(metric_vect, stats_list):
    """
    Computes specified statistics from a numeric vector.

    This function calculates various statistical measures from a numpy array
    based on the requested statistics in stats_list.

    Parameters
    ----------
    metric_vect : np.ndarray
        Vector with the values of the metric.
    stats_list : list
        List of statistics to compute. Supported values are:
        'mean', 'value' (same as 'mean'), 'median', 'std', 'min', 'max'.

    Returns
    -------
    list
        List with the computed statistics in the same order as requested.
        Values are returned as Python floats, not numpy.float64.

    Raises
    ------
    ValueError
        If an unsupported statistic is requested.
    TypeError
        If inputs are not of expected types.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> stats_from_vector(data, ['mean', 'median', 'std'])
    [3.0, 3.0, 1.4142135623730951]

    >>> stats_from_vector(data, ['min', 'max'])
    [1.0, 5.0]

    >>> # Case-insensitive statistic names
    >>> stats_from_vector(data, ['MEAN', 'Mean', 'mean'])
    [3.0, 3.0, 3.0]

    >>> # 'value' is an alias for 'mean'
    >>> stats_from_vector(data, ['mean', 'value'])
    [3.0, 3.0]

    >>> # Empty arrays return NaN for all statistics
    >>> stats_from_vector(np.array([]), ['mean', 'median'])
    [nan, nan]

    >>> # Error on unsupported statistic
    >>> stats_from_vector(data, ['mean', 'mode'])
    Traceback (most recent call last):
        ...
    ValueError: Unsupported statistics: mode
    """
    if not isinstance(stats_list, (list, tuple)):
        raise TypeError("stats_list must be a list or tuple")

    if len(metric_vect) == 0:
        return [float("nan")] * len(stats_list)

    # Map of statistic names to their computation functions
    stats_map = {
        "mean": np.mean,
        "value": np.mean,
        "median": np.median,
        "std": np.std,
        "min": np.min,
        "max": np.max,
    }

    # Convert all stats to lowercase for case-insensitive matching
    lowercase_stats = [s.lower() for s in stats_list]

    # Check for unsupported statistics
    unsupported = [s for s in lowercase_stats if s not in stats_map]
    if unsupported:
        raise ValueError(f"Unsupported statistics: {', '.join(unsupported)}")

    # Compute all requested statistics and convert to native Python float
    return [float(stats_map[stat](metric_vect)) for stat in lowercase_stats]


####################################################################################################
def get_units(
    metrics: Union[str, List[str]], metrics_json: Optional[Union[str, Dict]] = None
) -> List[str]:
    """
    Get the units associated with specified metrics.

    Retrieves the corresponding units for one or more metrics from either a provided
    JSON file, dictionary, or the default configuration.

    Parameters
    ----------
    metrics : str or list of str
        Name(s) of the metrics. Can be a single metric as a string or multiple metrics as a list.

    metrics_json : str or dict, optional
        Either:
        - Path to a JSON file containing the metrics units dictionary
        - Dictionary directly containing the metrics units mapping
        - None (default), which uses the package's built-in configuration

    Returns
    -------
    list of str
        Units corresponding to each requested metric. Returns "unknown" for any metric
        not found in the dictionary.

    Raises
    ------
    ValueError
        If the provided JSON file path is invalid or the metrics_json structure is incorrect.

    Examples
    --------
    >>> import clabtoolkit.morphometrytools as clmorphtools
    >>> # Get unit for a single metric
    >>> clmorphtools.get_units('thickness')
    ['mm']
    >>>
    >>> # Get units for multiple metrics
    >>> clmorphtools.get_units(['thickness', 'area', 'volume'])
    ['mm', 'cm²', 'cm³']
    >>>
    >>> # Using a custom metrics dictionary
    >>> custom_dict = {"metrics_units": {"custom_metric": "kg"}}
    >>> clmorphtools.get_units('custom_metric', metrics_json=custom_dict)
    ['kg']
    >>>
    >>> # Handling unknown metrics
    >>> clmorphtools.get_units(['thickness', 'unknown_metric'])
    ['mm', 'unknown']
    """
    # Convert single metric to list for uniform processing
    if isinstance(metrics, str):
        metrics = [metrics]

    # Get metrics dictionary from appropriate source
    if metrics_json is None:
        # Use default configuration
        config_path = os.path.join(os.path.dirname(__file__), "config", "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            metric_dict = config_data.get("metrics_units", {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading default configuration: {str(e)}")
    elif isinstance(metrics_json, str):
        # Load from provided JSON file path
        if not os.path.isfile(metrics_json):
            raise ValueError(f"Invalid JSON file path: {metrics_json}")
        try:
            with open(metrics_json, "r") as f:
                config_data = json.load(f)
            metric_dict = config_data.get("metrics_units", {})
            if not metric_dict:
                raise ValueError(
                    "Missing 'metrics_units' key in the provided JSON file"
                )
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {metrics_json}")
    elif isinstance(metrics_json, dict):
        # Use provided dictionary
        metric_dict = metrics_json.get("metrics_units", metrics_json)
    else:
        raise ValueError("metrics_json must be a file path, dictionary, or None")

    # Create a case-insensitive lookup dictionary (only once)
    lookup_dict = {k.lower(): v for k, v in metric_dict.items()}

    # Lookup units for each metric
    return [lookup_dict.get(metric.lower(), "unknown") for metric in metrics]


####################################################################################################
