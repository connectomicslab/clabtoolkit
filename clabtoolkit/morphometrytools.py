import os
import shutil
from typing import Union, Tuple, Optional, Dict, List
import copy
from pyvista import _vtk, PolyData
from numpy import split, ndarray
import json

import pandas as pd
import nibabel as nib
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
        raise ValueError(f"Invalid table_type: '{table_type}'. Expected 'region' or 'metric'.")
    
    # Process parcellation file
    if isinstance(parc_file, str):
        if not os.path.exists(parc_file):
            raise FileNotFoundError(f"Annotation file not found: {parc_file}")
        
        sparc_data = cltfree.AnnotParcellation(parc_file=parc_file)
    elif isinstance(parc_file, cltfree.AnnotParcellation):
        sparc_data = copy.deepcopy(parc_file)
    else:
        raise TypeError(f"parc_file must be a string or AnnotParcellation object, got {type(parc_file)}")
    
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
        raise TypeError(f"metric_file must be a string or numpy array, got {type(metric_file)}")
    
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
            sparc_data.regnames, regname, match_entire_world=True
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
        df.insert(1, "Side", reg_names[1])
    
    # Add metadata columns
    nrows = df.shape[0]
    units = get_units(metric)[0]
    
    df.insert(0, "Source", ["vertices"] * nrows)
    df.insert(1, "Metric", [metric] * nrows)
    df.insert(2, "Units", [units] * nrows)
    df.insert(3, "MetricFile", [filename] * nrows)
    
    # Add BIDS entities if requested
    if add_bids_entities and isinstance(metric_file, str):
        ent_list = entities4morphotable()
        df_add = df2add(in_file=metric_file, ent2add=ent_list)
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
        raise ValueError(f"Invalid table_type: '{table_type}'. Expected 'region' or 'metric'.")
    
    # Process parcellation file
    if isinstance(parc_file, str):
        if not os.path.exists(parc_file):
            raise FileNotFoundError(f"Annotation file not found: {parc_file}")
        
        sparc_data = cltfree.AnnotParcellation(parc_file=parc_file)
    elif isinstance(parc_file, cltfree.AnnotParcellation):
        sparc_data = copy.deepcopy(parc_file)
    else:
        raise TypeError(f"parc_file must be a string or AnnotParcellation object, got {type(parc_file)}")
    
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
        raise TypeError(f"surf_file must be a string or Surface object, got {type(surf_file)}")
    
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
            sparc_data.regnames, regname, match_entire_world=True
        )
        
        if len(index):
            # Find vertices belonging to this region
            ind = np.where(sparc_data.codes == sparc_data.regtable[index, 4])
            
            # Identify faces with different numbers of vertices in this region
            temp = np.isin(faces, ind).astype(int)
            nps = np.sum(temp, axis=1)
            
            # Group faces by how many vertices belong to the region
            reg_faces_3v = np.squeeze(faces[np.where(nps == 3), :])  # All vertices in region
            reg_faces_2v = np.squeeze(faces[np.where(nps == 2), :])  # Two vertices in region
            reg_faces_1v = np.squeeze(faces[np.where(nps == 1), :])  # One vertex in region
            
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
        df.insert(1, "Side", reg_names[1])
    
    # Add metadata columns
    nrows = df.shape[0]
    units = get_units("area")[0]
    
    df.insert(0, "Source", [surf_type] * nrows)
    df.insert(1, "Metric", ["area"] * nrows)
    df.insert(2, "Units", [units] * nrows)
    df.insert(3, "MetricFile", [filename] * nrows)
    
    # Add BIDS entities if requested
    if add_bids_entities and isinstance(parc_file, str):
        ent_list = entities4morphotable()
        df_add = df2add(in_file=parc_file, ent2add=ent_list)
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
        raise ValueError(f"Invalid table_type: '{table_type}'. Expected 'region' or 'metric'.")

    # Process surface file
    filename = ""
    if isinstance(surf_file, str):
        if not os.path.exists(surf_file):
            raise FileNotFoundError(f"Surface file not found: {surf_file}")
        
        surf = cltsurf.Surface(surface_file=surf_file)
        filename = surf_file
        
        # Extract surface type from filename if not provided
        if not surf_type and os.path.basename(surf_file).split('.')[-1] not in ['gii', 'vtk']:
            surf_type = os.path.basename(surf_file).split('.')[-1]
    elif isinstance(surf_file, cltsurf.Surface):
        surf = copy.deepcopy(surf_file)
    else:
        raise TypeError(f"surf_file must be a string or Surface object, got {type(surf_file)}")

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
        df.insert(1, "Side", reg_names[1])
    
    # Add metadata columns
    nrows = df.shape[0]
    units = get_units("euler")[0] if isinstance(get_units("euler"), list) else get_units("euler")
    
    df.insert(0, "Source", [surf_type] * nrows)
    df.insert(1, "Metric", ["euler"] * nrows)
    df.insert(2, "Units", [units] * nrows)
    df.insert(3, "MetricFile", [filename] * nrows)
    
    # Add BIDS entities if requested
    if add_bids_entities and isinstance(surf_file, str):
        ent_list = entities4morphotable()
        df_add = df2add(in_file=surf_file, ent2add=ent_list)
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
    d12 = np.sqrt(np.sum((v1 - v2)**2, axis=1))
    d23 = np.sqrt(np.sum((v2 - v3)**2, axis=1))
    d31 = np.sqrt(np.sum((v3 - v1)**2, axis=1))
    
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
    edges = np.vstack([
        faces[:, [0, 1]],  # First edge of each face
        faces[:, [1, 2]],  # Second edge of each face
        faces[:, [2, 0]]   # Third edge of each face
    ])

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
############          Methods dedicated to compute metrics from parcellations           ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def compute_reg_val_fromparcellation(
    metric_file: Union[str, np.ndarray],
    parc_file: Union[str, cltparc.Parcellation, np.ndarray],
    output_table: str = None,
    metric: str = "unknown",
    stats_list: Union[str, list] = ["value", "median", "std", "min", "max"],
    table_type: str = "metric",
    exclude_by_code: Union[list, np.ndarray] = None,
    exclude_by_name: Union[list, str] = None,
    add_bids_entities: bool = True,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Compute regional statistics from a volumetric metric map and a parcellation.
    
    This function extracts regional values by combining voxel-wise volumetric metrics with 
    parcellation data defining anatomical regions. It supports various statistical measures 
    and output formats to facilitate regional analysis of volumetric neuroimaging data.
    
    Parameters
    ----------
    metric_file : str or np.ndarray
        Path to the volumetric metric file or array containing metric values for each voxel.
    parc_file : str, cltparc.Parcellation, or np.ndarray
        Path to the parcellation file, Parcellation object, or numpy array defining regions.
    output_table : str, optional
        Path to save the resulting table. If None, the table is not saved.
    metric : str, default="unknown"
        Name of the metric being analyzed. Used for naming columns in the output DataFrame.
    stats_list : str or list, default=["value", "median", "std", "min", "max"]
        Statistics to compute for each region. Note: "value" is equivalent to the mean.
    table_type : str, default="metric"
        Output format specification:
        - "metric": Each column represents a specific statistic for each region
        - "region": Each column represents a region, with rows for different statistics
    exclude_by_code : list or np.ndarray, optional
        Region codes to exclude from the analysis. If None, no regions are excluded by code.
    exclude_by_name : list or str, optional
        Region names to exclude from the analysis. If None, no regions are excluded by name.
    add_bids_entities : bool, default=True
        Whether to include BIDS entities as columns in the resulting DataFrame.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the computed regional statistics.
    output_path : str or None
        Path where the table was saved, or None if no table was saved.
        
    Examples
    --------
    Basic usage with default parameters:
    
    >>> import os
    >>> import clabtoolkit.morphometrytools as morpho
    >>> metric_file = os.path.join('data', 'sub-01_T1w_intensity.nii.gz')
    >>> parc_file = os.path.join('data', 'sub-01_T1w_parcellation.nii.gz')
    >>> df, _ = morpho.compute_reg_val_fromparcellation(metric_file, parc_file, metric='intensity')
    >>> print(df.head())
    
    Using region format for output:
    
    >>> df_region, _ = morpho.compute_reg_val_fromparcellation(
    ...     metric_file, parc_file, metric='intensity', 
    ...     table_type="region", add_bids_entities=True
    ... )
    >>> print(df_region.head())
    
    Excluding specific regions:
    
    >>> exclude_names = ["Cerebellum", "Ventricles"]
    >>> df_filtered, _ = morpho.compute_reg_val_fromparcellation(
    ...     metric_file, parc_file, metric='intensity',
    ...     exclude_by_name=exclude_names
    ... )
    >>> print(df_filtered.head())
    
    Saving the results to a file:
    
    >>> output_path = os.path.join('results', 'regional_intensity.tsv')
    >>> df_saved, saved_path = morpho.compute_reg_val_fromparcellation(
    ...     metric_file, parc_file, output_table=output_path,
    ...     metric='intensity'
    ... )
    >>> print(f"Table saved to: {saved_path}")
    
    Notes
    -----
    This function is designed for volumetric data, extracting statistics from voxel-wise 
    metrics within each region defined by a parcellation. For surface-based metrics, 
    consider using `compute_reg_val_fromannot` instead.
    """
    # Input validation
    if isinstance(stats_list, str):
        stats_list = [stats_list]
    
    stats_list = [stat.lower() for stat in stats_list]
    
    if table_type not in ["region", "metric"]:
        raise ValueError(f"Invalid table_type: '{table_type}'. Expected 'region' or 'metric'.")
    
    # Process parcellation file
    if isinstance(parc_file, str):
        if not os.path.exists(parc_file):
            raise FileNotFoundError(f"Parcellation file not found: {parc_file}")
        
        vparc_data = cltparc.Parcellation(parc_file=parc_file)
    elif isinstance(parc_file, cltparc.Parcellation):
        vparc_data = copy.deepcopy(parc_file)
    elif isinstance(parc_file, np.ndarray):
        vparc_data = cltparc.Parcellation(parc_file=parc_file)
    else:
        raise TypeError(f"parc_file must be a string, Parcellation object, or numpy array, got {type(parc_file)}")
    
    # Detecting if the needed metric file as a string or an object. If the file is a string, check if the file exists
    # Otherwise, raise an error
    if isinstance(metric_file, str):
        if not os.path.exists(metric_file):
            raise FileNotFoundError(f"Metric file not found: {metric_file}")
        
        metric_vol = nib.load(metric_file).get_fdata()

    elif isinstance(metric_file, np.ndarray):
        metric_vol = metric_file
    else:
        raise TypeError(f"metric_file must be a string or numpy array, got {type(metric_file)}")
    
    # Converting to lower case
    stats_list = list(map(lambda x: x.lower(), stats_list))  # Converting to lower case

    if exclude_by_code is not None:
        vparc_data.remove_by_code(codes2remove=exclude_by_code)
    
    if exclude_by_name is not None:
        vparc_data.remove_by_name(names2remove=exclude_by_name)
    
    # Prepare data structures for results
    dict_of_cols = {}
    
    # Compute global brain statistics
    brain_mask = vparc_data.data != 0
    global_stats = stats_from_vector(metric_vol[brain_mask], stats_list)
    dict_of_cols["brain-brain-wholebrain"] = global_stats
    
    # Compute statistics for each region
    # Use unique region indices from the data itself
    unique_indices = np.unique(vparc_data.data)
    unique_indices = unique_indices[unique_indices != 0]  # Exclude background
    
    for i, index in enumerate(unique_indices):
        # Get region name from the parcellation object if available
        if hasattr(vparc_data, 'name') and hasattr(vparc_data, 'index'):
            idx_pos = np.where(np.array(vparc_data.index) == index)[0]
            if len(idx_pos) > 0:
                regname = vparc_data.name[idx_pos[0]]
            else:
                regname = f"region-unknown-{index}"
        else:
            regname = f"region-unknown-{index}"
            
        region_mask = vparc_data.data == index
        
        if np.any(region_mask):
            region_values = metric_vol[region_mask]
            region_stats = stats_from_vector(region_values, stats_list)
            dict_of_cols[regname] = region_stats
        else:
            dict_of_cols[regname] = [0] * len(stats_list)
    
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
        df.insert(0, "Supraregion", reg_names[0])
        df.insert(1, "Side", reg_names[1])
    
    # Add metadata columns
    nrows = df.shape[0]
    units = get_units(metric)
    if isinstance(units, list):
        units = units[0]
    
    df.insert(0, "Source", ["volume"] * nrows)
    df.insert(1, "Metric", [metric] * nrows)
    df.insert(2, "Units", [units] * nrows)
    df.insert(3, "MetricFile", [filename] * nrows)
    
    # Add BIDS entities if requested
    if add_bids_entities and isinstance(metric_file, str):
        ent_list = entities4morphotable()
        df_add = df2add(in_file=metric_file, ent2add=ent_list)
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
def compute_reg_volume_fromparcellation(
    parc_file: Union[str, cltparc.Parcellation, np.ndarray],
    format: str = "metric",
    exclude_by_code: Union[list, np.ndarray] = None,
    exclude_by_name: Union[list, str] = None,
    add_bids_entities: bool = False,
) -> pd.DataFrame:
    """
    This method computes the volume for all the regions of a specified parcellation.

    Parameters
    ----------

    parc_file : str
        Path to the parcellation file.

    format : str, optional
        Format of the output. It could be "region" or "metric". The default is "metric".
        With the "region" format, the output is a DataFrame with the regional values where each column
        represent the value of column metric for each specific region. With the "metric" format, the output
        is a DataFrame with the regional values where each column represent the value of a specific metric
        for each region.

    exclude_by_code : Union[list, np.ndarray], optional
        List of codes to exclude from the analysis. The default is None. If None, no codes are excluded.
        Please see the method "build_indexes" in the miscellaneous module inside clabtoolkit to see how to build the list of codes.

    exclude_by_name : Union[list, str], optional
        List of names to exclude from the analysis. The default is None. If None, no names are excluded.

    add_bids_entities: bool, optional
        Boolean variable to include the BIDs entities as columns in the resulting dataframe. The default is True.


    Returns
    -------
    df : pd.DataFrame
        DataFrame with the regional values.
    metric_vect : np.ndarray


    Examples
    --------
    >>> import clabtoolkit.morphometrytools as cltmorphtools
    >>> import os
    >>> import pandas as pd
    >>> metric_file = os.path.join('..', 'data', 'metric.nii.gz')
    >>> parc_file = os.path.join('..', 'data', 'parcellation.nii.gz')
    >>> df = cltmorphtools.compute_reg_volume_fromparcellation(parc_file)
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
            raise FileNotFoundError("The parcellation file does not exist.")
        else:
            # Reading the annot file
            vparc_data = cltparc.Parcellation(
                parc_file=parc_file,
            )
            affine = vparc_data.affine
    elif isinstance(parc_file, cltfree.AnnotParcellation):
        # If the file is an object, copy the object
        vparc_data = copy.deepcopy(parc_file)
        affine = vparc_data.affine

    elif isinstance(parc_file, np.ndarray):
        vparc_data = cltparc.Parcellation(parc_file=parc_file)
        affine = vparc_data.affine

    # Excluding regions
    if exclude_by_code is not None:
        vparc_data.remove_by_code(codes2remove=exclude_by_code)

    if exclude_by_name is not None:
        vparc_data.remove_by_name(names2remove=exclude_by_name)

    # Computing the voxel volume
    vox_size = np.linalg.norm(affine[:3, :3], axis=1)
    vox_vol = np.prod(vox_size)

    dict_of_cols = {}

    # Values for each region in the annot
    df = pd.DataFrame()

    # Computing global metric
    temp = len(vparc_data.data[vparc_data.data != 0]) * vox_vol
    dict_of_cols["brain-brain-wholebrain"] = temp / 1000

    for i, index in enumerate(vparc_data.index):
        regname = vparc_data.name[i]
        ind = np.where(vparc_data.data == index)
        temp = len(ind[0]) * vox_vol
        if temp > 0:
            dict_of_cols[regname] = [temp / 1000]

        else:
            dict_of_cols[regname] = [0]

    df = pd.DataFrame.from_dict(dict_of_cols)

    if format == "region":
        df.index = ["summary"]

        # Converting the row names to a new column called statistics
        df = df.reset_index()
        df = df.rename(columns={"index": "statistics"})

        # Convert the index to a column with name "metric"

    else:
        df = df.T
        df.columns = ["summary"]

        # Converting the row names to a new column called statistics
        df = df.reset_index()
        df = df.rename(columns={"index": "region"})

        # Get the column called "region" and split it into three columns "supraregion", "side" and "region"
        reg_names = df["region"].str.split("-", expand=True)

        # Insert the new columns before the column "region"
        df.insert(0, "supraregion", reg_names[0])
        df.insert(1, "side", reg_names[1])

    nrows = df.shape[0]

    # Inserting the units
    units = get_units("volume")
    df.insert(0, "metric", ["volume"] * nrows)
    df.insert(1, "units", units * nrows)

    # Adding the entities related to BIDs
    if add_bids_entities:
        ent_list = entities4morphotable()
        df_add = df2add(in_file=parc_file, ent_list=ent_list)

        # Expand a first dataframe and concatenate with the second dataframe
        df = cltmisc.expand_and_concatenate(df_add, df)

    return df


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############       Methods dedicated to parse stats file from freesurfer results        ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################


def parse_freesurfer_statsfile(
    stat_file: str, format: str = "metric", 
    add_bids_entities: bool = True
) -> pd.DataFrame:
    """
    This function reads the estimated volume values from freesurfer aseg.stats file

    Parameters

    ----------
    stat_file : str
        Path to the aseg.stats file generated by freesurfer

    format : str, optional
        Format of the output. It could be "region" or "metric". The default is "metric".
        With the "region" format, the output is a DataFrame with the regional values where each column
        represent the value of column metric for each specific region. With the "metric" format, the output
        is a DataFrame with the regional values where each column represent the value of a specific metric
        for each region.

    add_bids_entities: bool, optional
        Boolean variable to include the BIDs entities as columns in the resulting dataframe. The default is True.

    Returns

    -------
    df : pandas.DataFrame
        A dataframe with the values of the estimated volume

    """

    # Verify if the file exists
    if os.path.isfile(stat_file):

        # Read the each file and extract the eTIV, total gray matter volume and cerebral white matter volume
        with open(stat_file, "r") as file:
            for line in file:
                if (
                    "EstimatedTotalIntraCranialVol" in line
                ):  # Selecting the line with the eTIV
                    eTIV = line.split()[-2].split(",")[
                        0
                    ]  # Splitting the line and selecting the eTIV value
                    # Convert the value to float and divide by 1000 to convert to cm3
                    eTIV = float(eTIV) / 1000

                if "Brain Segmentation Volume," in line:
                    brain_seg = line.split()[-2].split(",")[0]
                    brain_seg = float(brain_seg) / 1000

                if "TotalGrayVol" in line:
                    total_grey = line.split()[-2].split(",")[0]
                    total_grey = float(total_grey) / 1000

                if "CerebralWhiteMatterVol" in line:
                    white_matter = line.split()[-2].split(",")[0]
                    white_matter = float(white_matter) / 1000

                if "Left-Lateral-Ventricle" in line:
                    left_lat_vent = line.split()[3]
                    left_lat_vent = float(left_lat_vent) / 1000

                if "Left-Inf-Lat-Vent" in line:
                    left_inf_lat_vent = line.split()[3]
                    left_inf_lat_vent = float(left_inf_lat_vent) / 1000

                if "Right-Lateral-Ventricle" in line:
                    right_lat_vent = line.split()[3]
                    right_lat_vent = float(right_lat_vent) / 1000

                if "Right-Inf-Lat-Vent" in line:
                    right_inf_lat_vent = line.split()[3]
                    right_inf_lat_vent = float(right_inf_lat_vent) / 1000

        # Create a dictionary with the values
        dict_of_cols = {}
        dict_of_cols["brain-brain-intracraneal"] = [eTIV]
        dict_of_cols["brain-brain-wholebrain"] = [brain_seg]
        dict_of_cols["gm-brain-graymatter"] = [total_grey]
        dict_of_cols["wm-brain-whitematter"] = [white_matter]
        dict_of_cols["vent-lh-lateral"] = [left_lat_vent]
        dict_of_cols["vent-lh-inferior"] = [left_inf_lat_vent]
        dict_of_cols["vent-rh-lateral"] = [right_lat_vent]
        dict_of_cols["vent-rh-inferior"] = [right_inf_lat_vent]

    # Create a dataframe with the values
    df = pd.DataFrame.from_dict(dict_of_cols)

    if format == "region":
        df.index = ["summary"]
        df = df.reset_index()
        df = df.rename(columns={"index": "statistics"})

        # Convert the index to a column with name "metric"

    else:
        df = df.T
        df.columns = ["summary"]

        # Converting the row names to a new column called statistics
        df = df.reset_index()
        df = df.rename(columns={"index": "region"})

        # Get the column called "region" and split it into three columns "supraregion", "side" and "region"
        reg_names = df["region"].str.split("-", expand=True)

        # Insert the new columns before the column "region"
        df.insert(0, "supraregion", reg_names[0])
        df.insert(1, "side", reg_names[1])

    nrows = df.shape[0]

    # Inserting the units
    units = get_units("volume")
    df.insert(0, "metric", ["volume"] * nrows)
    df.insert(1, "units", units * nrows)

    # Adding the entities related to BIDs
    if add_bids_entities:
        ent_list = entities4morphotable()
        df_add = df2add(in_file=stat_file, ent_list=ent_list)

        # Expand a first dataframe and concatenate with the second dataframe
        df = cltmisc.expand_and_concatenate(df_add, df)

    return df


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############                        Auxiliary methods                                   ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################


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
        if v == "mean" or v == "summary":
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


def entities4morphotable(entities_json: str = None) -> list:
    """
    This method returns the BIDs entities that will be included in the morphometric table.

    Parameters
    ----------
    entities_json : str, optional
        Path to the json file with the information of the metrics. The default is None.
        If None, the method uses the default config json file.

    Returns
    -------
    entities : list
        List of valid entities.

    Examples
    --------
    >>> import clabtoolkit.morphometrytools as clmorphtools
    >>> clmorphtools.entities4morphotable()
    ["sub",
    "ses",
    "acq",
    "dir",
    "run",
    "ce",
    "rec",
    "space",
    "res",
    "model",
    "desc",
    "atlas",
    "scale",
    "seg",
    "grow"]
    """
    config_json = os.path.join(os.path.dirname(__file__), "config", "config.json")

    if entities_json is None:
        with open(config_json) as f:
            config_json = json.load(f)
        entities = config_json["bids_entities"]
    elif isinstance(entities_json, str):
        if not os.path.isfile(entities_json):
            raise ValueError(
                "Please, provide a valid JSON file containing the entities dictionary."
            )
        else:
            with open(entities_json) as f:
                entities = json.load(f)

    return entities


def get_units(metrics: Union[str, list], metrics_json: Union[str, dict] = None) -> list:
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
    >>> clmorphtools.get_units('thickness')
    ['mm']
    """

    if isinstance(metrics, str):
        metrics = [metrics]

    if metrics_json is None:
        config_json = os.path.join(os.path.dirname(__file__), "config", "config.json")
        with open(config_json) as f:
            config_json = json.load(f)
        metric_dict = config_json["metrics_units"]
    else:
        if isinstance(metrics_json, str):
            if not os.path.isfile(metrics_json):
                raise ValueError(
                    "Please, provide a valid JSON file containing the units dictionary."
                )
            else:
                with open(metrics_json) as f:
                    metric_dict = json.load(f)
        elif isinstance(metrics_json, dict):
            metric_dict = metrics_json

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


def df2add(in_file: str, ent_list: Union[str, list] == None):
    """
    Method to create a dataframe that could be added in front of the metrics dataframe.

    Parameters:
    in_file : str
        Filename path to extract the entities from.

    """

    file_path = os.path.dirname(in_file)
    file_name = os.path.basename(in_file)
    ent_dict = cltbids.str2entity(file_name)

    df2add = pd.DataFrame()

    if cltbids.is_bids_filename(file_name):
        if ent_list is not None:
            if isinstance(ent_list, str):
                ent_list = [ent_list]
            for entity in reversed(ent_list):

                if entity in ent_dict.keys():
                    value = ent_dict[entity]
                else:
                    value = ""

                if entity == "sub":
                    df2add.insert(0, "participant_id", value)
                elif entity == "ses":
                    df2add.insert(0, "session_id", value)
                elif entity == "atlas":
                    if "chimera" in value:
                        df2add.insert(0, "atlas_id", "chimera")
                        # Remove the word chimera from tmp string
                        tmp = value.replace("chimera", "")
                        df2add.insert(1, "chimera_id", tmp)
                    else:
                        df2add.insert(0, "atlas_id", value)
                        df2add.insert(1, "chimera_id", "")

                elif entity == "desc":
                    df2add.insert(0, "desc_id", value)
                    if "grow" in value:
                        tmp = value.replace("grow", "")
                        df2add.insert(1, "grow", tmp)
                else:
                    df2add.insert(0, entity + "_id", value)
        else:
            # Adding the participant_id as the full name of the file without the extension
            if "extension" in ent_dict.keys():
                ent_dict["suffix"] = ""

            df2add.insert(0, "participant_id", cltbids.entity2str(ent_dict))

    return df2add
