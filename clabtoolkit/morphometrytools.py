import copy
import json
import os
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.funcs import squeeze_image
from nibabel.processing import resample_from_to

# Importing local modules
from . import bidstools as cltbids
from . import colorstools as cltcol
from . import freesurfertools as cltfree
from . import misctools as cltmisc
from . import parcellationtools as cltparc
from . import surfacetools as cltsurf
from . import connectivitytools as cltconn

# Regions of an annotation that are not anatomical cortical regions
_UNKNOWN_SUBSTRINGS = ["medialwall", "unknown", "corpuscallosum"]

# Statistics supported by stats_from_vector
_SUPPORTED_STATS = ("mean", "value", "median", "std", "min", "max", "count", "sum")

# Length units expressed in mm, used to convert areas and volumes
_LENGTH_IN_MM = {
    "um": 1e-3,
    "μm": 1e-3,
    "mm": 1.0,
    "cm": 10.0,
    "dm": 100.0,
    "m": 1000.0,
}


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
    metric_file: str | Path | np.ndarray,
    parc_file: str | Path | cltfree.AnnotParcellation,
    hemi: str,
    output_table: str | Path = None,
    nonzeros_only: bool = False,
    metric: str = "unknown",
    units: str = None,
    stats_list: str | list = None,
    table_type: str = "metric",
    include_unknown: bool = False,
    include_global: bool = True,
    add_bids_entities: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, str | None]:
    """
    Compute regional statistics from a surface metric map and an annotation file.

    Parameters
    ----------
    metric_file : str, Path or np.ndarray
        Path to a FreeSurfer surface map (e.g. lh.thickness) or array with one
        value per vertex.

    parc_file : str, Path or cltfree.AnnotParcellation
        Path to the annotation file or AnnotParcellation object.

    hemi : str
        Hemisphere identifier ('lh' or 'rh').

    output_table : str or Path, optional
        Path to save the table as TSV. If None, the table is not saved.

    nonzeros_only : bool, default=False
        Compute the statistics using only the non-zero values of each region.

    metric : str, default="unknown"
        Name of the metric. If it is "area" or "volume", the map is assumed to be in
        mm² or mm³ (FreeSurfer convention) and is converted to the units defined
        in config.json.

    units : str, optional
        Units of the metric. If None, they are taken from config.json. Ignored for
        "area" and "volume", whose units always come from config.json.

    stats_list : str or list, default=["value", "median", "std", "min", "max"]
        Statistics to compute. "value" is the mean.

    table_type : {"metric", "region"}, default="metric"
        "metric": one row per region. "region": one column per region.

    include_unknown : bool, default=False
        Include non-anatomical regions (medialwall, unknown, corpuscallosum).

    include_global : bool, default=True
        Include hemisphere-wide statistics.

    add_bids_entities : bool, default=True
        Add BIDS entities extracted from the metric file name.

    Returns
    -------
    df : pd.DataFrame
        Regional statistics.

    metric_vect : np.ndarray
        Vertex-wise values used in the computation (converted to the config units
        for "area" and "volume").

    output_path : str or None
        Path of the saved table, or None.

    Raises
    ------
    ValueError
        If the number of values does not match the number of vertices of the annotation.

    Examples
    --------
    >>> df, values, _ = compute_reg_val_fromannot(
    ...     'lh.thickness', 'lh.aparc.annot', 'lh', metric='thickness'
    ... )
    """
    stats_list = _normalize_stats_list(stats_list)
    _validate_table_type(table_type)
    output_table = _check_output_table(output_table)

    annot = _clean_annot(_load_annot(parc_file), include_unknown)

    # Metric values
    filename = ""
    if isinstance(metric_file, (str, Path)):
        filename = str(metric_file)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Metric file not found: {filename}")
        metric_vect = nib.freesurfer.io.read_morph_data(filename)
    elif isinstance(metric_file, np.ndarray):
        metric_vect = metric_file.ravel()
    else:
        raise TypeError(
            f"metric_file must be a string, Path or numpy array, got {type(metric_file)}"
        )

    _check_vertex_count(annot, metric_vect.shape[0], "metric map")

    # Areas and volumes are converted from mm² / mm³ to the config units
    if metric.lower() in ("area", "volume"):
        metric_vect, config_units = convert_from_mm(
            np.asarray(metric_vect, dtype=np.float64), metric
        )
        if units is not None and units != config_units:
            warnings.warn(
                f"units='{units}' ignored: '{metric}' is reported in '{config_units}' "
                "as defined in config.json.",
                stacklevel=2,
            )
        units = config_units
    elif units is None:
        units = get_units(metric)[0]

    region_codes = annot.regtable[:, 4]
    _check_unique_names(annot.regnames)

    dict_of_cols = {}
    if include_global:
        valid_vertices = np.isin(annot.codes, region_codes)
        dict_of_cols[f"ctx-{hemi}-hemisphere"] = stats_from_vector(
            metric_vect[valid_vertices], stats_list, nonzeros_only=nonzeros_only
        )

    for regname, code in zip(annot.regnames, region_codes, strict=True):
        dict_of_cols[regname] = stats_from_vector(
            metric_vect[annot.codes == code], stats_list, nonzeros_only=nonzeros_only
        )

    dict_of_cols = _prefix_region_names(dict_of_cols, prefix=f"ctx-{hemi}-")

    df = _format_table(dict_of_cols, [s.title() for s in stats_list], table_type)
    df = _add_metadata(df, "vertices", metric, units, filename)
    df, output_path = _finalize_table(df, filename, add_bids_entities, output_table)

    return df, metric_vect, output_path


####################################################################################################
def compute_reg_area_fromsurf(
    surf_file: str | Path | cltsurf.Surface,
    parc_file: str | Path | cltfree.AnnotParcellation,
    hemi: str,
    table_type: str = "metric",
    surf_type: str = "",
    include_unknown: bool = False,
    include_global: bool = True,
    add_bids_entities: bool = True,
    output_table: str | Path = None,
) -> tuple[pd.DataFrame, str | None]:
    """
    Compute the surface area of each region defined in an annotation file.

    Every vertex receives one third of the area of each triangle it belongs to, and
    the area of a region is the sum of the areas of its vertices. Triangles on region
    boundaries are therefore split between regions instead of being counted several
    times, and the regional areas add up to the hemisphere area. Areas are expressed
    in the units defined for "area" in config.json.

    Parameters
    ----------
    surf_file : str, Path or cltsurf.Surface
        Surface file (coordinates in mm) or Surface object.

    parc_file : str, Path or cltfree.AnnotParcellation
        Annotation file or AnnotParcellation object.

    hemi : str
        Hemisphere identifier ('lh' or 'rh').

    table_type : {"metric", "region"}, default="metric"
        "metric": one row per region. "region": one column per region.

    surf_type : str, default=""
        Surface type (e.g. "white", "pial"), stored in the Source column.

    include_unknown : bool, default=False
        Include non-anatomical regions (medialwall, unknown, corpuscallosum).

    include_global : bool, default=True
        Include the hemisphere area (sum of the regional areas).

    add_bids_entities : bool, default=True
        Add BIDS entities extracted from the annotation file name.

    output_table : str or Path, optional
        Path to save the table as TSV. If None, the table is not saved.

    Returns
    -------
    df : pd.DataFrame
        Regional areas.

    output_path : str or None
        Path of the saved table, or None.

    Examples
    --------
    >>> df, _ = compute_reg_area_fromsurf('lh.white', 'lh.aparc.annot', 'lh', surf_type='white')
    """
    _validate_table_type(table_type)
    output_table = _check_output_table(output_table)

    annot = _clean_annot(_load_annot(parc_file), include_unknown)
    surf, filename = _load_surface(surf_file)

    coords = np.asarray(surf.mesh.points, dtype=np.float64)
    faces = np.asarray(surf.get_faces(), dtype=np.int64)
    _check_vertex_count(annot, coords.shape[0], "surface")

    # Triangle areas are already in the config units
    _, tri_area = area_from_mesh(coords, faces)
    units = get_units("area")[0]

    vertex_area = np.bincount(
        faces.ravel(),
        weights=np.repeat(tri_area / 3.0, 3),
        minlength=coords.shape[0],
    )

    _check_unique_names(annot.regnames)
    regions = {
        regname: [float(vertex_area[annot.codes == code].sum())]
        for regname, code in zip(annot.regnames, annot.regtable[:, 4], strict=True)
    }

    dict_of_cols = {}
    if include_global:
        dict_of_cols[f"ctx-{hemi}-hemisphere"] = [
            float(sum(v[0] for v in regions.values()))
        ]
    dict_of_cols.update(regions)
    dict_of_cols = _prefix_region_names(dict_of_cols, prefix=f"ctx-{hemi}-")

    df = _format_table(dict_of_cols, ["Value"], table_type)
    df = _add_metadata(df, surf_type, "area", units, filename)
    bids_file = str(parc_file) if isinstance(parc_file, (str, Path)) else ""
    return _finalize_table(df, bids_file, add_bids_entities, output_table)


####################################################################################################
def compute_reg_nvertices_fromsurf(
    surf_file: str | Path | cltsurf.Surface,
    parc_file: str | Path | cltfree.AnnotParcellation,
    hemi: str,
    table_type: str = "metric",
    surf_type: str = "",
    include_unknown: bool = False,
    include_global: bool = True,
    add_bids_entities: bool = True,
    output_table: str | Path = None,
) -> tuple[pd.DataFrame, str | None]:
    """
    Compute the number of vertices of each region defined in an annotation file.

    Parameters
    ----------
    surf_file : str, Path or cltsurf.Surface
        Surface file or Surface object. Used to check that the annotation matches
        the surface.

    parc_file : str, Path or cltfree.AnnotParcellation
        Annotation file or AnnotParcellation object.

    hemi : str
        Hemisphere identifier ('lh' or 'rh').

    table_type : {"metric", "region"}, default="metric"
        "metric": one row per region. "region": one column per region.

    surf_type : str, default=""
        Surface type (e.g. "white", "pial"), stored in the Source column.

    include_unknown : bool, default=False
        Include non-anatomical regions (medialwall, unknown, corpuscallosum).

    include_global : bool, default=True
        Include the number of vertices of the hemisphere (sum over regions).

    add_bids_entities : bool, default=True
        Add BIDS entities extracted from the annotation file name.

    output_table : str or Path, optional
        Path to save the table as TSV. If None, the table is not saved.

    Returns
    -------
    df : pd.DataFrame
        Number of vertices per region.

    output_path : str or None
        Path of the saved table, or None.

    Examples
    --------
    >>> df, _ = compute_reg_nvertices_fromsurf('lh.white', 'lh.aparc.annot', 'lh')
    """
    _validate_table_type(table_type)
    output_table = _check_output_table(output_table)

    annot = _clean_annot(_load_annot(parc_file), include_unknown)
    surf, filename = _load_surface(surf_file)
    _check_vertex_count(annot, np.asarray(surf.mesh.points).shape[0], "surface")

    _check_unique_names(annot.regnames)
    regions = {
        regname: [int(np.count_nonzero(annot.codes == code))]
        for regname, code in zip(annot.regnames, annot.regtable[:, 4], strict=True)
    }

    dict_of_cols = {}
    if include_global:
        dict_of_cols[f"ctx-{hemi}-hemisphere"] = [sum(v[0] for v in regions.values())]
    dict_of_cols.update(regions)
    dict_of_cols = _prefix_region_names(dict_of_cols, prefix=f"ctx-{hemi}-")

    df = _format_table(dict_of_cols, ["Value"], table_type)
    df = _add_metadata(df, surf_type, "nvertices", get_units("nvertices")[0], filename)
    bids_file = str(parc_file) if isinstance(parc_file, (str, Path)) else ""
    return _finalize_table(df, bids_file, add_bids_entities, output_table)


####################################################################################################
def compute_euler_fromsurf(
    surf_file: str | Path | cltsurf.Surface,
    hemi: str,
    output_table: str | Path = None,
    table_type: str = "metric",
    surf_type: str = "",
    add_bids_entities: bool = True,
) -> tuple[pd.DataFrame, str | None]:
    """
    Compute the Euler characteristic (χ = V - E + F) of a surface mesh.

    Parameters
    ----------
    surf_file : str, Path or cltsurf.Surface
        Surface file or Surface object.

    hemi : str
        Hemisphere identifier ('lh' or 'rh').

    output_table : str or Path, optional
        Path to save the table as TSV. If None, the table is not saved.

    table_type : {"metric", "region"}, default="metric"
        "metric": one row per region. "region": one column per region.

    surf_type : str, default=""
        Surface type. If empty, it is taken from the file extension (e.g. lh.white).

    add_bids_entities : bool, default=True
        Add BIDS entities extracted from the surface file name.

    Returns
    -------
    df : pd.DataFrame
        Euler characteristic of the hemisphere.

    output_path : str or None
        Path of the saved table, or None.

    Notes
    -----
    For a closed orientable surface of genus g, χ = 2 - 2g.

    Examples
    --------
    >>> df, _ = compute_euler_fromsurf('lh.white', 'lh')
    """
    _validate_table_type(table_type)
    output_table = _check_output_table(output_table)

    surf, filename = _load_surface(surf_file)
    if filename and not surf_type:
        extension = os.path.basename(filename).split(".")[-1]
        if extension not in ("gii", "vtk"):
            surf_type = extension

    euler = euler_from_mesh(
        np.asarray(surf.mesh.points, dtype=np.float64),
        np.asarray(surf.get_faces(), dtype=np.int64),
    )

    dict_of_cols = _prefix_region_names(
        {f"ctx-{hemi}-hemisphere": [euler]}, prefix=f"ctx-{hemi}-"
    )

    df = _format_table(dict_of_cols, ["Value"], table_type)
    df = _add_metadata(df, surf_type, "euler", get_units("euler")[0], filename)
    return _finalize_table(df, filename, add_bids_entities, output_table)


####################################################################################################
def area_from_mesh(coords: np.ndarray, faces: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute the total area and the per-triangle areas of a triangular mesh.

    The coordinates are assumed to be in mm. The areas are returned in the units
    defined for "area" in config.json.

    Parameters
    ----------
    coords : np.ndarray
        Vertex coordinates in mm, shape (n, 3).

    faces : np.ndarray
        Triangles as vertex indices, shape (m, 3).

    Returns
    -------
    total_area : float
        Total mesh area, in the config units.

    tri_area : np.ndarray
        Area of every triangle, shape (m,), in the config units.

    Notes
    -----
    The area of a triangle is half the norm of the cross product of two of its
    edges. Unlike Heron's formula, this is numerically stable for the thin
    triangles that are common on cortical meshes.

    Examples
    --------
    >>> coords = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
    >>> faces = np.array([[0, 1, 2], [1, 3, 2]])
    >>> total_area, _ = area_from_mesh(coords, faces)   # 100 mm² = 1 cm²
    """
    coords = np.asarray(coords, dtype=np.float64)
    faces = np.asarray(faces)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (n, 3), got {coords.shape}")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (m, 3), got {faces.shape}")

    if np.any(faces >= coords.shape[0]) or np.any(faces < 0):
        raise ValueError("faces contains invalid vertex indices")

    v1 = coords[faces[:, 0]]
    v2 = coords[faces[:, 1]]
    v3 = coords[faces[:, 2]]

    tri_area_mm2 = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)
    tri_area, _ = convert_from_mm(tri_area_mm2, "area")

    return float(np.sum(tri_area)), tri_area


####################################################################################################
def euler_from_mesh(coords: np.ndarray, faces: np.ndarray) -> int:
    """
    Compute the Euler characteristic of a triangular mesh.

    Parameters
    ----------
    coords : np.ndarray
        Vertex coordinates, shape (n, 3).

    faces : np.ndarray
        Triangles as vertex indices, shape (m, 3).

    Returns
    -------
    int
        Euler characteristic χ = V - E + F.

    Examples
    --------
    >>> coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    >>> euler_from_mesh(coords, faces)
    2
    """
    coords = np.asarray(coords)
    faces = np.asarray(faces)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (n, 3), got {coords.shape}")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (m, 3), got {faces.shape}")

    if np.any(faces >= coords.shape[0]) or np.any(faces < 0):
        raise ValueError("faces contains invalid vertex indices")

    n_vertices = coords.shape[0]
    n_faces = faces.shape[0]

    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    n_edges = len(np.unique(np.sort(edges, axis=1), axis=0))

    return int(n_vertices - n_edges + n_faces)


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
    metric_file: str | Path | np.ndarray,
    parc_file: str | Path | cltparc.Parcellation | np.ndarray,
    output_table: str | Path = None,
    metric: str = "unknown",
    units: str = None,
    stats_list: str | list = None,
    nonzeros_only: bool = False,
    table_type: str = "metric",
    exclude_by_code: list | np.ndarray = None,
    exclude_by_name: list | str = None,
    include_by_code: list | np.ndarray = None,
    include_by_name: list | str = None,
    include_global: bool = True,
    add_bids_entities: bool = True,
    region_prefix: str = "supra-side",
    interp_method: str = "linear",
) -> tuple[pd.DataFrame, np.ndarray, str | None]:
    """
    Compute regional statistics from a volumetric metric map and a parcellation.

    When the metric map is a file and the parcellation has a real affine (file or
    Parcellation object), the map is resampled to the parcellation grid whenever
    their shapes or affines differ.

    Parameters
    ----------
    metric_file : str, Path or np.ndarray
        3D metric image or array. Arrays must already be on the parcellation grid.

    parc_file : str, Path, cltparc.Parcellation or np.ndarray
        Parcellation file, Parcellation object or label array.

    output_table : str or Path, optional
        Path to save the table as TSV. If None, the table is not saved.

    metric : str, default="unknown"
        Name of the metric, stored in the Metric column.

    units : str, optional
        Units of the metric. If None, they are taken from config.json.

    stats_list : str or list, default=["value", "median", "std", "min", "max"]
        Statistics to compute: "value" (mean), "mean", "median", "std", "min",
        "max", "count", "sum".

    nonzeros_only : bool, default=False
        Compute the statistics using only the non-zero values of each region.

    table_type : {"metric", "region"}, default="metric"
        "metric": one row per region. "region": one column per region.

    exclude_by_code, exclude_by_name : optional
        Regions to remove before computing the statistics.

    include_by_code, include_by_name : optional
        Regions to keep, applied after the exclusions.

    include_global : bool, default=True
        Include the statistics over all labeled voxels ("brain-brain-wholebrain").

    add_bids_entities : bool, default=True
        Add BIDS entities extracted from the metric file name.

    region_prefix : str, default="supra-side"
        Prefix of the names generated for labels missing from the color table.

    interp_method : {"linear", "nearest", "cubic"}, default="linear"
        Interpolation used when resampling. Use "nearest" for categorical maps.

    Returns
    -------
    df : pd.DataFrame
        Regional statistics.

    metric_data : np.ndarray
        Metric volume used in the computation (after resampling, if any).

    output_path : str or None
        Path of the saved table, or None.

    Raises
    ------
    ValueError
        If the metric map is not 3D, its shape does not match the parcellation, no
        region is left after filtering, or two regions share the same name.

    Examples
    --------
    >>> df, _, _ = compute_reg_val_fromparcellation('FA.nii.gz', 'parc.nii.gz', metric='fa')
    """
    stats_list = _normalize_stats_list(stats_list)
    _validate_table_type(table_type)
    output_table = _check_output_table(output_table)

    orders = {"nearest": 0, "linear": 1, "cubic": 3}
    if interp_method not in orders:
        raise ValueError(
            f"Invalid interp_method: '{interp_method}'. Expected 'linear', 'nearest' or 'cubic'."
        )

    vparc_data, _ = _load_parcellation(parc_file)
    parc_has_affine = not isinstance(parc_file, np.ndarray)
    target_shape = vparc_data.data.shape

    # Metric volume
    filename = ""
    if isinstance(metric_file, (str, Path)):
        filename = str(metric_file)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Metric file not found: {filename}")

        metric_img = squeeze_image(nib.load(filename))
        if metric_img.ndim != 3:
            raise ValueError(
                f"The metric image must be 3D, got shape {metric_img.shape}."
            )

        if parc_has_affine and (
            metric_img.shape != target_shape
            or not np.allclose(metric_img.affine, vparc_data.affine, atol=1e-4)
        ):
            warnings.warn(
                f"Metric image {metric_img.shape} and parcellation {target_shape} differ "
                "in shape or orientation. Resampling the metric to the parcellation grid.",
                stacklevel=2,
            )
            metric_img = resample_from_to(
                metric_img,
                (target_shape, vparc_data.affine),
                order=orders[interp_method],
            )

        metric_vol = metric_img.get_fdata()

    elif isinstance(metric_file, np.ndarray):
        metric_vol = np.asarray(metric_file, dtype=np.float64)
        if metric_vol.ndim > 3 and all(s == 1 for s in metric_vol.shape[3:]):
            metric_vol = metric_vol.reshape(metric_vol.shape[:3])
    else:
        raise TypeError(
            f"metric_file must be a string, Path or numpy array, got {type(metric_file)}"
        )

    if metric_vol.shape != target_shape:
        raise ValueError(
            f"Metric data shape {metric_vol.shape} does not match parcellation shape "
            f"{target_shape}. Use file inputs for automatic resampling."
        )

    _apply_region_filters(
        vparc_data, exclude_by_code, exclude_by_name, include_by_code, include_by_name
    )

    # Group the metric values of all labeled voxels by label, in a single pass
    flat_labels = vparc_data.data.ravel()
    labeled = flat_labels != 0
    labels_nz = flat_labels[labeled]
    values_nz = metric_vol.ravel()[labeled]

    if labels_nz.size == 0:
        raise ValueError("No valid regions found in the parcellation data")

    order = np.argsort(labels_nz, kind="stable")
    labels, starts = np.unique(labels_nz[order], return_index=True)
    region_values = np.split(values_nz[order], starts[1:])

    names = _region_names(vparc_data, labels, region_prefix)
    _check_unique_names(names)

    dict_of_cols = {}
    if include_global:
        dict_of_cols["brain-brain-wholebrain"] = stats_from_vector(
            values_nz, stats_list, nonzeros_only=nonzeros_only
        )
    for name, vals in zip(names, region_values, strict=True):
        dict_of_cols[name] = stats_from_vector(
            vals, stats_list, nonzeros_only=nonzeros_only
        )

    if units is None:
        units = get_units(metric)[0]

    df = _format_table(dict_of_cols, [s.title() for s in stats_list], table_type)
    df = _add_metadata(df, "volume", metric, units, filename)
    df, output_path = _finalize_table(df, filename, add_bids_entities, output_table)

    return df, metric_vol, output_path


####################################################################################################
def compute_reg_volume_fromparcellation(
    parc_file: str | Path | cltparc.Parcellation | np.ndarray,
    output_table: str | Path = None,
    table_type: str = "metric",
    exclude_by_code: list | np.ndarray = None,
    exclude_by_name: list | str = None,
    include_by_code: list | np.ndarray = None,
    include_by_name: list | str = None,
    add_bids_entities: bool = True,
    region_prefix: str = "supra-side",
    include_global: bool = True,
) -> tuple[pd.DataFrame, str | None]:
    """
    Compute the volume of every region of a parcellation.

    The volume of a region is its number of voxels times the voxel volume (the
    absolute determinant of the affine), expressed in the units defined for
    "volume" in config.json.

    Parameters
    ----------
    parc_file : str, Path, cltparc.Parcellation or np.ndarray
        Parcellation file, Parcellation object or label array. Arrays are assumed
        to have 1 mm isotropic voxels.

    output_table : str or Path, optional
        Path to save the table as TSV. If None, the table is not saved.

    table_type : {"metric", "region"}, default="metric"
        "metric": one row per region. "region": one column per region.

    exclude_by_code, exclude_by_name : optional
        Regions to remove before computing the volumes.

    include_by_code, include_by_name : optional
        Regions to keep, applied after the exclusions.

    add_bids_entities : bool, default=True
        Add BIDS entities extracted from the parcellation file name.

    region_prefix : str, default="supra-side"
        Prefix of the names generated for labels missing from the color table.

    include_global : bool, default=True
        Include the total labeled volume ("brain-brain-wholebrain").

    Returns
    -------
    df : pd.DataFrame
        Regional volumes.

    output_path : str or None
        Path of the saved table, or None.

    Raises
    ------
    ValueError
        If no region is left after filtering or two regions share the same name.

    Examples
    --------
    >>> df, _ = compute_reg_volume_fromparcellation('parc.nii.gz')
    """
    _validate_table_type(table_type)
    output_table = _check_output_table(output_table)

    vparc_data, filename = _load_parcellation(parc_file)
    bids_file = filename
    if not filename:
        filename = "3darray"

    _apply_region_filters(
        vparc_data, exclude_by_code, exclude_by_name, include_by_code, include_by_name
    )

    vox_vol_mm3 = abs(np.linalg.det(np.asarray(vparc_data.affine)[:3, :3]))

    labels, counts = np.unique(vparc_data.data, return_counts=True)
    keep = labels != 0
    labels, counts = labels[keep], counts[keep]

    if labels.size == 0:
        raise ValueError("No valid regions found in the parcellation data")

    volumes, units = convert_from_mm(counts.astype(np.float64) * vox_vol_mm3, "volume")

    names = _region_names(vparc_data, labels, region_prefix)
    _check_unique_names(names)

    dict_of_cols = {}
    if include_global:
        dict_of_cols["brain-brain-wholebrain"] = [float(volumes.sum())]
    for name, volume in zip(names, volumes, strict=True):
        dict_of_cols[name] = [float(volume)]

    df = _format_table(dict_of_cols, ["Value"], table_type)
    df = _add_metadata(df, "parcellation", "volume", units, filename)
    return _finalize_table(df, bids_file, add_bids_entities, output_table)


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
    stat_file: str | Path,
    output_table: str | Path = None,
    table_type: str = "metric",
    add_bids_entities: bool = True,
    include_missing: bool = True,
    config_json: str | Path = None,
) -> tuple[pd.DataFrame, str | None]:
    """
    Parse global measurements from a FreeSurfer aseg.stats file.

    The unit of every "# Measure" line is read from the file. Volumes (mm³) and
    areas (mm²) are converted to the units defined in config.json. Unitless
    measures (e.g. BrainSegVol-to-eTIV, SurfaceHoles) are kept unchanged, with
    Metric "measure" and Units "au".

    Parameters
    ----------
    stat_file : str or Path
        Path to the aseg.stats file.

    output_table : str or Path, optional
        Path to save the table as TSV. If None, the table is not saved.

    table_type : {"metric", "region"}, default="metric"
        "metric": one row per measure, with its own Metric and Units.
        "region": one column per measure. If the measures have different units,
        the Metric and Units columns are set to "mixed".

    add_bids_entities : bool, default=True
        Add BIDS entities extracted from the stats file name.

    include_missing : bool, default=True
        Report measures that are not found as 0. A warning is issued either way.

    config_json : str or Path, optional
        JSON file defining the measures to extract. If None, the "global" entry of
        stats_mapping.json is used. Any "divisor" key is ignored: units come
        from config.json.

    Returns
    -------
    df : pd.DataFrame
        Global measurements.

    output_path : str or None
        Path of the saved table, or None.

    Examples
    --------
    >>> df, _ = parse_freesurfer_global_fromaseg('aseg.stats')
    """
    stat_file = _check_stats_file(stat_file)
    _validate_table_type(table_type)
    output_table = _check_output_table(output_table)

    measurements = _load_stats_config(config_json, "global")
    lines, measures, table = _read_aseg_stats(stat_file)

    values, metrics, units = {}, {}, {}

    for region_key, region_info in measurements.items():
        key = region_info["key"]
        candidates = [key] + list(region_info.get("alternate_keys", []))

        found = None
        for candidate in candidates:
            if candidate in measures:
                found = measures[candidate]
                break
            if candidate in table:
                found = (table[candidate]["Volume_mm3"], "mm^3")
                break

        if found is None and "index" in region_info:
            found = _legacy_line_lookup(lines, key, region_info["index"])

        if found is None:
            warnings.warn(
                f"Value for {region_key} (key: {key}) not found in {stat_file}",
                stacklevel=2,
            )
            if not include_missing:
                continue
            found = (0.0, "mm^3")

        value, metric_name, unit = _fs_measure_to_config(*found)
        values[region_key] = [value]
        metrics[region_key] = metric_name
        units[region_key] = unit

    if not values:
        raise ValueError(f"No volume measurements found in {stat_file}")

    keys = list(values)
    metric_col = [metrics[k] for k in keys]
    unit_col = [units[k] for k in keys]
    if table_type == "region":
        metric_col = _collapse_values(metric_col)
        unit_col = _collapse_values(unit_col)

    df = _format_table(values, ["Value"], table_type)
    df = _add_metadata(df, "statsfile", metric_col, unit_col, stat_file)
    return _finalize_table(df, stat_file, add_bids_entities, output_table)


####################################################################################################
def parse_freesurfer_stats_fromaseg(
    stat_file: str | Path,
    output_table: str | Path = None,
    table_type: str = "metric",
    add_bids_entities: bool = True,
    include_missing: bool = True,
    config_json: str | Path = None,
) -> tuple[pd.DataFrame, str | None]:
    """
    Parse regional volumes from the table of a FreeSurfer aseg.stats file.

    The table volumes (Volume_mm3) are converted to the units defined for "volume"
    in config.json.

    Parameters
    ----------
    stat_file : str or Path
        Path to the aseg.stats file.

    output_table : str or Path, optional
        Path to save the table as TSV. If None, the table is not saved.

    table_type : {"metric", "region"}, default="metric"
        "metric": one row per region. "region": one column per region.

    add_bids_entities : bool, default=True
        Add BIDS entities extracted from the stats file name.

    include_missing : bool, default=True
        Report regions that are not found as 0. A warning is issued either way.

    config_json : str or Path, optional
        JSON file defining the regions to extract. If None, the "aseg" entry of
        stats_mapping.json is used. Any "divisor" key is ignored: units come
        from config.json.

    Returns
    -------
    df : pd.DataFrame
        Regional volumes.

    output_path : str or None
        Path of the saved table, or None.

    Examples
    --------
    >>> df, _ = parse_freesurfer_stats_fromaseg('aseg.stats')
    """
    stat_file = _check_stats_file(stat_file)
    _validate_table_type(table_type)
    output_table = _check_output_table(output_table)

    region_measurements = _load_stats_config(config_json, "aseg")
    _, _, table = _read_aseg_stats(stat_file)
    by_segid = {row["SegId"]: row for row in table.values()}

    volumes_mm3 = {}
    for region_key, region_info in region_measurements.items():
        key = region_info["key"]
        candidates = [key] + list(region_info.get("alternate_keys", []))

        found = next((table[c]["Volume_mm3"] for c in candidates if c in table), None)

        seg_id = region_info.get("seg_id")
        if found is None and seg_id is not None and int(seg_id) in by_segid:
            found = by_segid[int(seg_id)]["Volume_mm3"]

        if found is None:
            warnings.warn(
                f"Value for {region_key} (key: {key}) not found in {stat_file}",
                stacklevel=2,
            )
            if not include_missing:
                continue
            found = 0.0

        volumes_mm3[region_key] = found

    if not volumes_mm3:
        raise ValueError(f"No region measurements found in {stat_file}")

    volumes, units = convert_from_mm(
        np.fromiter(volumes_mm3.values(), dtype=np.float64), "volume"
    )
    dict_of_cols = {k: [float(v)] for k, v in zip(volumes_mm3, volumes, strict=True)}

    df = _format_table(dict_of_cols, ["Value"], table_type)
    df = _add_metadata(df, "statsfile", "volume", units, stat_file)
    return _finalize_table(df, stat_file, add_bids_entities, output_table)


####################################################################################################
def parse_freesurfer_cortex_stats(
    stats_file: str | Path,
    output_table: str | Path = None,
    table_type: str = "metric",
    add_bids_entities: bool = True,
    hemi: str = None,
    config_json: str | Path = None,
    include_metrics: list = None,
) -> tuple[pd.DataFrame, str | None]:
    """
    Parse cortical parcellation statistics from a FreeSurfer aparc.stats file.

    The SurfArea (mm²) and GrayVol (mm³) columns are converted to the units defined
    for "area" and "volume" in config.json. The units of the other metrics are
    taken from the metric configuration ("unit" key) or, if absent, from config.json.

    Parameters
    ----------
    stats_file : str or Path
        Path to an lh/rh aparc.stats file.

    output_table : str or Path, optional
        Path to save the table as TSV. If None, the table is not saved.

    table_type : {"metric", "region"}, default="metric"
        "metric": one row per region and metric. "region": one row per metric,
        one column per region.

    add_bids_entities : bool, default=True
        Add BIDS entities extracted from the stats file name.

    hemi : str, optional
        'lh' or 'rh'. If None, detected from the file name or content.

    config_json : str or Path, optional
        JSON file defining the metrics to extract. If None, the "cortex" entry of
        stats_mapping.json is used.

    include_metrics : list, optional
        Metrics of the configuration to extract. If None, all are extracted.

    Returns
    -------
    df : pd.DataFrame
        Cortical measurements.

    output_path : str or None
        Path of the saved table, or None.

    Raises
    ------
    ValueError
        If no column headers or no data can be parsed, or none of the requested
        metrics is in the configuration.

    Examples
    --------
    >>> df, _ = parse_freesurfer_cortex_stats('lh.aparc.stats', include_metrics=['area', 'thickness'])
    """
    stats_file = _check_stats_file(stats_file)
    _validate_table_type(table_type)
    output_table = _check_output_table(output_table)

    with open(stats_file, encoding="utf-8") as f:
        lines = f.readlines()

    if hemi is None:
        hemi = _detect_hemisphere(stats_file, lines)

    metric_mapping = _load_stats_config(config_json, "cortex")

    if include_metrics:
        requested = [m.lower() for m in include_metrics]
        metric_mapping = {
            k: v for k, v in metric_mapping.items() if k.lower() in requested
        }
        if not metric_mapping:
            raise ValueError(
                f"None of the requested metrics {include_metrics} found in configuration."
            )

    if not metric_mapping:
        raise ValueError("No valid metrics found in configuration.")

    column_headers = _parse_aparc_headers(lines)
    column_indices = {name: idx for idx, name in enumerate(column_headers)}
    name_idx = column_indices.get("StructName", 0)

    data_rows = [
        line.split()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]

    rows = []
    missing_columns = set()

    for parts in data_rows:
        if len(parts) < len(column_headers):
            warnings.warn(
                f"Skipping line with {len(parts)} fields (expected {len(column_headers)}): "
                f"{' '.join(parts)[:50]}...",
                stacklevel=2,
            )
            continue

        region_name = parts[name_idx]

        for metric_name, metric_info in metric_mapping.items():
            column = metric_info.get("column")

            if column in column_indices:
                col_idx = column_indices[column]
            elif "index" in metric_info:
                col_idx = int(metric_info["index"])
            else:
                if metric_name not in missing_columns:
                    warnings.warn(
                        f"Column {column} for metric {metric_name} not found in {stats_file}",
                        stacklevel=2,
                    )
                    missing_columns.add(metric_name)
                continue

            if col_idx >= len(parts):
                continue

            try:
                value = float(parts[col_idx])
            except ValueError:
                warnings.warn(
                    f"Could not parse {metric_name} for region {region_name}",
                    stacklevel=2,
                )
                continue

            actual_column = (
                column_headers[col_idx] if col_idx < len(column_headers) else column
            )

            if actual_column == "SurfArea":
                value, unit = convert_from_mm(value, "area")
            elif actual_column == "GrayVol":
                value, unit = convert_from_mm(value, "volume")
            else:
                unit = metric_info.get("unit") or get_units(metric_name)[0]

            region_row = {
                "Region": f"ctx-{hemi}-{region_name}",
                "Metric": metric_name,
                "Value": value,
                "Source": metric_info.get("source", "statsfile"),
                "Units": unit,
            }

            std_value = _parse_std_value(
                parts, metric_name, metric_info, column_indices
            )
            if std_value is not None:
                region_row["Std"] = std_value

            rows.append(region_row)

    if not rows:
        raise ValueError(f"No cortical parcellation data found in {stats_file}")

    df = pd.DataFrame(rows)
    df["Hemisphere"] = hemi
    df["Supraregion"] = "ctx"
    df["MetricFile"] = stats_file

    column_order = [
        "Source",
        "Metric",
        "Units",
        "MetricFile",
        "Supraregion",
        "Hemisphere",
        "Region",
        "Value",
        "Std",
    ]
    df = df[[c for c in column_order if c in df.columns]]

    if table_type == "region":
        value_cols = ["Value"] + (["Std"] if "Std" in df.columns else [])

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

        if isinstance(pivot_df.columns, pd.MultiIndex):
            pivot_df.columns = [
                f"{col[0]}_{col[1]}" if col[0] != "" else col[1]
                for col in pivot_df.columns
            ]

        df = pivot_df.reset_index().rename(columns={"Metric": "Statistics"})

    return _finalize_table(df, stats_file, add_bids_entities, output_table)


####################################################################################################
def get_stats_dictionary(region_level: str = "global") -> dict:
    """
    Return the default measurement configuration for FreeSurfer stats files.

    Parameters
    ----------
    region_level : {"global", "aseg", "cortex"}, default="global"
        Entry of stats_mapping.json to return.

    Returns
    -------
    dict
        Configuration of the measurements to extract.

    Raises
    ------
    KeyError
        If region_level is not an entry of stats_mapping.json.
    """
    mapping_stats_json = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config", "stats_mapping.json"
    )

    with open(mapping_stats_json, encoding="utf-8") as f:
        mapp_dict = json.load(f)

    if region_level not in mapp_dict:
        raise KeyError(
            f"'{region_level}' not found in stats_mapping.json. "
            f"Available entries: {list(mapp_dict)}"
        )

    return mapp_dict[region_level]


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############   Section 4: Methods dedicated to extract metrics from connectivity        ############
############            matrices based on graph theory                                  ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def network_metrics_to_table(
    conn_mat: np.ndarray | cltconn.Connectome,
    lut_file: str | Path | dict = None,
    cmat_met: str | list[str] = None,
    metrics: str | list[str] = None,
    weighting: str = "auto",
    output_table: str | Path = None,
    table_type: str = "metric",
    include_global: bool = True,
    add_bids_entities: bool = True,
    source_file: str | Path = None,
    region_prefix: str = "supra-side",
    seed: int = None,
) -> tuple[pd.DataFrame, str | None]:
    """
    Compute graph theory metrics from a connectivity matrix.

    Nodal metrics are computed for every region. Global metrics are reported for the
    "brain-brain-wholebrain" region only. The name of each metric is stored in the
    Metric column, and the connectivity matrix type in the Source column. Uses the
    Brain Connectivity Toolbox (bctpy, imported as ``bct``).

    The graph is classified as binary (all non-zero off-diagonal entries share one
    value) or weighted. When ``metrics`` is None, the default metrics for that
    weighting are computed. The defaults are read from the "network_metrics" entry
    of the package config.json, which has a "binary" and a "weighted" list:

        "network_metrics": {
            "binary":   ["glob_efficiency", ..., "degree", ...],
            "weighted": ["glob_efficiency_wei", ..., "strength", ...]
        }

    Edit these lists to change the defaults; their order is the order of the
    metrics in the output table.

    Binary metrics use the binarized matrix (weights > 0). Weighted metrics ("_wei")
    use the weights normalized to [0, 1] (clustering, efficiency, transitivity) or
    converted to lengths as 1/weight (betweenness, path length, eccentricity, radius,
    diameter).

    Available metrics
    -----------------
    Nodal:
        degree, strength, clustering_coeff, clustering_coeff_wei, betw_centrality,
        betw_centrality_wei, loc_efficiency, loc_efficiency_wei,
        eigenvector_centrality, pagerank_centrality, subgraph_centrality,
        kcoreness_centrality, eccentricity, eccentricity_wei, participation_coeff,
        within_module_zscore

    Global:
        glob_efficiency, glob_efficiency_wei, transitivity, transitivity_wei,
        density_coeff, char_path_length, char_path_length_wei, radius, radius_wei,
        diameter, diameter_wei, assortativity, assortativity_wei, modularity

    Parameters
    ----------
    conn_mat : np.ndarray or cltconn.Connectome
        Square, undirected connectivity matrix or Connectome object. Self-connections
        (diagonal) are removed before computing the metrics.

    lut_file : str, Path or dict, optional
        Lookup table (file or dict with 'index' and 'name') naming the regions, in
        the order of the matrix rows. If given, it overrides the region names
        stored in a Connectome object. If None, the Connectome names are used
        when available; otherwise names are generated from region_prefix.

    cmat_met : str or list of str, optional
        Label of the connectivity matrix type, stored in the Source column as
        "conn_matrix_<cmat_met>". A list is joined with "-". If None, the
        weighting of the graph is used ("binary" or "weighted").

    metrics : str or list of str, optional
        Metrics to compute (case-insensitive), in the order given. If None, the
        default metrics for the graph weighting are computed (config.json,
        "network_metrics"). An explicit list is always honored; a warning is issued
        if it contains metrics that are not in the defaults for that weighting.

    weighting : {"auto", "binary", "weighted"}, default="auto"
        Weighting used to select the default metrics. "auto" detects it from the
        matrix.

    output_table : str or Path, optional
        Path to save the table as TSV. If None, the table is not saved.

    table_type : {"metric", "region"}, default="metric"
        "metric": one row per region and metric. "region": one row per metric,
        one column per region.

    include_global : bool, default=True
        Compute the global metrics. If False, global metrics are skipped even if
        they are listed in metrics.

    add_bids_entities : bool, default=True
        Add BIDS entities extracted from source_file.

    source_file : str or Path, optional
        File the connectivity matrix comes from. Stored in the MetricFile column and
        used to extract the BIDS entities.

    region_prefix : str, default="supra-side"
        Prefix of the names generated when no region names are available.

    seed : int, optional
        Random seed of the Louvain community detection used by participation_coeff,
        within_module_zscore and modularity. Set it for reproducible results.

    Returns
    -------
    df : pd.DataFrame
        Network metrics.

    output_path : str or None
        Path of the saved table, or None.

    Raises
    ------
    ValueError
        If the matrix is not square, weighting is invalid, an unknown metric is
        requested or listed in config.json, no metric is left to compute, fewer
        region names than nodes are available, or two regions share the same name.

    Examples
    --------
    >>> conn_mat = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    >>> df, _ = network_metrics_to_table(conn_mat)               # weighted metrics
    >>> df, _ = network_metrics_to_table(conn_mat > 0)           # binary metrics
    >>> df, _ = network_metrics_to_table(conn_mat, metrics=["degree", "strength"])
    """
    try:
        import bct
    except ImportError as err:
        raise ImportError(
            "bctpy is required for network_metrics_to_table. Install it with `pip install bctpy`."
        ) from err

    _validate_table_type(table_type)
    output_table = _check_output_table(output_table)

    if weighting not in ("auto", "binary", "weighted"):
        raise ValueError(
            f"Invalid weighting: '{weighting}'. Expected 'auto', 'binary' or 'weighted'."
        )

    conn_names = None
    if isinstance(conn_mat, cltconn.Connectome):
        if conn_mat.region_names is not None and len(conn_mat.region_names) > 0:
            conn_names = [str(n) for n in conn_mat.region_names]
        conn_mat = conn_mat.matrix

    # Copy, so the caller's matrix is never modified
    conn_mat = np.array(conn_mat, dtype=np.float64, copy=True)
    if conn_mat.ndim != 2 or conn_mat.shape[0] != conn_mat.shape[1]:
        raise ValueError(
            f"conn_mat must be a square matrix, got shape {conn_mat.shape}"
        )
    if not np.allclose(conn_mat, conn_mat.T, equal_nan=True):
        warnings.warn(
            "conn_mat is not symmetric; it is treated as undirected.", stacklevel=2
        )
    if np.any(np.diag(conn_mat) != 0):
        warnings.warn(
            "conn_mat has self-connections; the diagonal is set to 0.", stacklevel=2
        )
        np.fill_diagonal(conn_mat, 0)

    if weighting == "auto":
        weighting = _detect_weighting(conn_mat)

    n_nodes = conn_mat.shape[0]

    # Region names: the lookup table takes precedence over the Connectome names
    if lut_file is not None:
        if isinstance(lut_file, (str, Path)):
            if not os.path.exists(lut_file):
                raise FileNotFoundError(f"Lookup table not found: {lut_file}")
            col_dict = cltcol.ColorTableLoader.load_colortable(lut_file)
        elif isinstance(lut_file, dict):
            col_dict = copy.deepcopy(lut_file)
        else:
            raise TypeError(
                f"lut_file must be a string, Path or dictionary, got {type(lut_file)}"
            )
        st_names = [str(n) for n in col_dict["name"]]
        names_source = "lookup table"
    elif conn_names is not None:
        st_names = conn_names
        names_source = "Connectome object"
    else:
        st_names = None

    if st_names is not None:
        if len(st_names) < n_nodes:
            raise ValueError(
                f"The {names_source} names {len(st_names)} regions but the matrix has "
                f"{n_nodes} nodes."
            )
        if len(st_names) > n_nodes:
            warnings.warn(
                f"The {names_source} names {len(st_names)} regions but the matrix has "
                f"{n_nodes} nodes. Only the first {n_nodes} names are used.",
                stacklevel=2,
            )
            st_names = st_names[:n_nodes]
    else:
        st_names = [
            str(n)
            for n in cltmisc.create_names_from_indices(
                list(range(1, n_nodes + 1)), prefix=region_prefix
            )
        ]

    _check_unique_names(st_names)

    if cmat_met is None:
        cmat_met = weighting
    elif isinstance(cmat_met, (list, tuple)):
        cmat_met = "-".join(str(m) for m in cmat_met)
    source = f"conn_matrix_{cmat_met}"
    source_file = str(source_file) if source_file is not None else ""

    # Intermediate matrices, computed only when a requested metric needs them
    cache = {}

    def _get(key, func):
        if key not in cache:
            cache[key] = func()
        return cache[key]

    def w_bin():
        return _get("bin", lambda: (conn_mat > 0).astype(np.float64))

    def w_norm():
        return _get("norm", lambda: bct.weight_conversion(conn_mat, "normalize"))

    def w_len():
        return _get("len", lambda: bct.weight_conversion(conn_mat, "lengths"))

    def charpath_bin():
        # (char. path length, efficiency, eccentricity, radius, diameter);
        # disconnected node pairs are excluded
        return _get(
            "cp_bin",
            lambda: bct.charpath(
                bct.distance_bin(w_bin()),
                include_diagonal=False,
                include_infinite=False,
            ),
        )

    def charpath_wei():
        return _get(
            "cp_wei",
            lambda: bct.charpath(
                bct.distance_wei(w_len())[0],
                include_diagonal=False,
                include_infinite=False,
            ),
        )

    def communities():
        # (community assignment, modularity Q)
        return _get("ci", lambda: bct.community_louvain(conn_mat, seed=seed))

    nodal_funcs = {
        "degree": lambda: bct.degrees_und(w_bin()),
        "strength": lambda: bct.strengths_und(conn_mat),
        "clustering_coeff": lambda: bct.clustering_coef_bu(w_bin()),
        "clustering_coeff_wei": lambda: bct.clustering_coef_wu(w_norm()),
        "betw_centrality": lambda: bct.betweenness_bin(w_bin()),
        "betw_centrality_wei": lambda: bct.betweenness_wei(w_len()),
        "loc_efficiency": lambda: bct.efficiency_bin(w_bin(), local=True),
        "loc_efficiency_wei": lambda: bct.efficiency_wei(w_norm(), local=True),
        "eigenvector_centrality": lambda: bct.eigenvector_centrality_und(conn_mat),
        "pagerank_centrality": lambda: bct.pagerank_centrality(conn_mat, d=0.85),
        "subgraph_centrality": lambda: bct.subgraph_centrality(w_bin()),
        "kcoreness_centrality": lambda: bct.kcoreness_centrality_bu(w_bin())[0],
        "eccentricity": lambda: charpath_bin()[2],
        "eccentricity_wei": lambda: charpath_wei()[2],
        "participation_coeff": lambda: bct.participation_coef(
            conn_mat, communities()[0]
        ),
        "within_module_zscore": lambda: bct.module_degree_zscore(
            conn_mat, communities()[0], flag=0
        ),
    }

    global_funcs = {
        "glob_efficiency": lambda: bct.efficiency_bin(w_bin()),
        "glob_efficiency_wei": lambda: bct.efficiency_wei(w_norm()),
        "transitivity": lambda: bct.transitivity_bu(w_bin()),
        "transitivity_wei": lambda: bct.transitivity_wu(w_norm()),
        "density_coeff": lambda: bct.density_und(w_bin())[0],
        "char_path_length": lambda: charpath_bin()[0],
        "char_path_length_wei": lambda: charpath_wei()[0],
        "radius": lambda: charpath_bin()[3],
        "radius_wei": lambda: charpath_wei()[3],
        "diameter": lambda: charpath_bin()[4],
        "diameter_wei": lambda: charpath_wei()[4],
        "assortativity": lambda: bct.assortativity_bin(w_bin(), flag=0),
        "assortativity_wei": lambda: bct.assortativity_wei(conn_mat, flag=0),
        "modularity": lambda: communities()[1],
    }

    # Metric selection: the defaults for each weighting come from config.json
    available = list(global_funcs) + list(nodal_funcs)
    default_metrics = _default_network_metrics()

    unknown_cfg = sorted(
        {m for names in default_metrics.values() for m in names if m not in available}
    )
    if unknown_cfg:
        raise ValueError(
            f"Unknown metrics in config.json 'network_metrics': {', '.join(unknown_cfg)}. "
            f"Available: {', '.join(available)}"
        )
    weighting_defaults = default_metrics[weighting]

    if metrics is None:
        selected = list(weighting_defaults)
    else:
        if isinstance(metrics, str):
            metrics = [metrics]
        if not isinstance(metrics, (list, tuple)):
            raise TypeError(
                f"metrics must be a string, list or None, got {type(metrics)}"
            )
        selected = list(dict.fromkeys(str(m).strip().lower() for m in metrics))
        unknown = [m for m in selected if m not in available]
        if unknown:
            raise ValueError(
                f"Unknown metrics: {', '.join(unknown)}. "
                f"Available: {', '.join(available)}"
            )
        mismatched = [m for m in selected if m not in weighting_defaults]
        if mismatched:
            warnings.warn(
                f"The graph is {weighting}, but {mismatched} are not among the "
                f"default {weighting} metrics (config.json 'network_metrics'). "
                "They are computed anyway.",
                stacklevel=2,
            )

    if not include_global:
        skipped = [m for m in selected if m in global_funcs]
        if skipped and metrics is not None:
            warnings.warn(
                f"include_global=False: skipping global metrics {skipped}.",
                stacklevel=2,
            )
        selected = [m for m in selected if m not in global_funcs]

    if not selected:
        raise ValueError("No metrics left to compute.")

    # One block per metric, concatenated into a single table
    tables = []
    for metric_name in selected:
        if metric_name in global_funcs:
            dict_of_cols = {
                "brain-brain-wholebrain": [float(global_funcs[metric_name]())]
            }
        else:
            metric_values = np.asarray(
                nodal_funcs[metric_name](), dtype=np.float64
            ).ravel()
            dict_of_cols = {
                name: [float(metric_values[i])] for i, name in enumerate(st_names)
            }

        df_metric = _format_table(dict_of_cols, ["Value"], table_type)
        tables.append(_add_metadata(df_metric, source, metric_name, "au", source_file))

    df = pd.concat(tables, ignore_index=True)
    return _finalize_table(df, source_file, add_bids_entities, output_table)


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############                        Section 5: Auxiliary methods                        ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def stats_from_vector(
    metric_vect,
    stats_list: list | tuple = None,
    nonzeros_only: bool = True,
) -> list:
    """
    Compute statistics from a numeric vector.

    Parameters
    ----------
    metric_vect : array-like
        Values of the metric, coerced to a flat float64 array.

    stats_list : list or tuple, optional
        Statistics to compute (case-insensitive): "mean", "value" (alias of mean),
        "median", "std", "min", "max", "count", "sum".
        Default is ["value", "median", "std", "min", "max"].

    nonzeros_only : bool, default=True
        Compute the statistics only on the non-zero elements.

    Returns
    -------
    list of float
        Statistics in the requested order. For an empty input (or an all-zero
        input with nonzeros_only=True), "count" and "sum" are 0; the other
        statistics are NaN for an empty input and 0 for an all-zero input.

    Raises
    ------
    TypeError
        If stats_list is not a list or tuple.
    ValueError
        If an unsupported statistic is requested.

    Examples
    --------
    >>> stats_from_vector(np.array([1, 2, 3, 4, 5]), ['mean', 'median', 'count'])
    [3.0, 3.0, 5.0]
    """
    if stats_list is None:
        stats_list = ["value", "median", "std", "min", "max"]
    if not isinstance(stats_list, (list, tuple)):
        raise TypeError("stats_list must be a list or tuple")

    lowercase_stats = [s.lower() for s in stats_list]
    unsupported = [s for s in lowercase_stats if s not in _SUPPORTED_STATS]
    if unsupported:
        raise ValueError(f"Unsupported statistics: {', '.join(unsupported)}")

    metric_vect = np.asarray(metric_vect, dtype=np.float64).ravel()

    if metric_vect.size == 0:
        return [0.0 if s in ("count", "sum") else float("nan") for s in lowercase_stats]

    if nonzeros_only:
        metric_vect = metric_vect[metric_vect != 0]
        if metric_vect.size == 0:
            return [0.0] * len(lowercase_stats)

    stats_map = {
        "mean": np.mean,
        "value": np.mean,
        "median": np.median,
        "std": np.std,
        "min": np.min,
        "max": np.max,
        "count": np.size,
        "sum": np.sum,
    }

    return [float(stats_map[stat](metric_vect)) for stat in lowercase_stats]


####################################################################################################
def get_units(
    metrics: str | list[str], metrics_json: str | Path | dict | None = None
) -> list[str]:
    """
    Get the units of one or more metrics.

    Parameters
    ----------
    metrics : str or list of str
        Metric name(s), case-insensitive.

    metrics_json : str, Path or dict, optional
        JSON file or dictionary with a "metrics_units" mapping. If None, the
        package's config/config.json is used.

    Returns
    -------
    list of str
        Units of each metric, "unknown" for metrics not in the mapping.

    Raises
    ------
    ValueError
        If the JSON file is invalid or lacks the "metrics_units" key.

    Examples
    --------
    >>> get_units(['thickness', 'area', 'volume'])
    ['mm', 'cm2', 'cm3']
    >>> get_units('custom_metric', metrics_json={"metrics_units": {"custom_metric": "kg"}})
    ['kg']
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    if metrics_json is None:
        lookup_dict = _default_units_lookup()

    elif isinstance(metrics_json, (str, Path)):
        if not os.path.isfile(metrics_json):
            raise ValueError(f"Invalid JSON file path: {metrics_json}")
        try:
            with open(metrics_json, encoding="utf-8") as f:
                config_data = json.load(f)
        except json.JSONDecodeError as err:
            raise ValueError(f"Invalid JSON format in file: {metrics_json}") from err

        metric_dict = config_data.get("metrics_units", {})
        if not metric_dict:
            raise ValueError("Missing 'metrics_units' key in the provided JSON file")
        lookup_dict = {k.lower(): v for k, v in metric_dict.items()}

    elif isinstance(metrics_json, dict):
        metric_dict = metrics_json.get("metrics_units", metrics_json)
        lookup_dict = {k.lower(): v for k, v in metric_dict.items()}

    else:
        raise ValueError("metrics_json must be a file path, dictionary, or None")

    return [lookup_dict.get(metric.lower(), "unknown") for metric in metrics]


####################################################################################################
def convert_from_mm(
    values: float | np.ndarray,
    metric: str,
    metrics_json: str | Path | dict | None = None,
) -> tuple[float | np.ndarray, str]:
    """
    Convert areas in mm² or volumes in mm³ to the units defined in config.json.

    Parameters
    ----------
    values : float or array-like
        Values in mm² (metric="area") or mm³ (metric="volume").

    metric : {"area", "volume"}
        Quantity to convert.

    metrics_json : str, Path or dict, optional
        Units configuration. If None, the package's config.json is used.

    Returns
    -------
    converted : float or np.ndarray
        Values in the configured units.

    unit : str
        Configured unit (e.g. "cm2", "cm3").

    Raises
    ------
    ValueError
        If metric is not "area" or "volume", or the configured unit is missing or
        not a valid unit for that quantity.

    Examples
    --------
    >>> convert_from_mm(np.array([100.0, 250.0]), "area")      # with "area": "cm2"
    (array([1. , 2.5]), 'cm2')
    >>> convert_from_mm(1500.0, "volume")                     # with "volume": "cm3"
    (1.5, 'cm3')
    """
    power = {"area": 2, "volume": 3}.get(metric.lower())
    if power is None:
        raise ValueError(
            f"convert_from_mm only handles 'area' and 'volume', got '{metric}'."
        )

    unit = get_units(metric, metrics_json=metrics_json)[0]
    if unit == "unknown":
        raise ValueError(f"No unit defined for '{metric}' in the units configuration.")

    factor = _mm_power_per_unit(unit, power)

    if np.isscalar(values):
        return float(values) / factor, unit
    return np.asarray(values, dtype=np.float64) / factor, unit


####################################################################################################
# Private helpers
####################################################################################################
@lru_cache(maxsize=1)
def _default_units_lookup() -> dict:
    """Load and cache the case-insensitive metrics_units mapping of config.json."""
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.json")
    try:
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error loading default configuration: {str(e)}") from e

    return {k.lower(): v for k, v in config_data.get("metrics_units", {}).items()}


@lru_cache(maxsize=1)
def _default_network_metrics() -> dict[str, tuple[str, ...]]:
    """
    Read the default graph metrics for binary and weighted graphs.

    The lists come from the "network_metrics" entry of the package config.json,
    which has one key per weighting ("binary" and "weighted"). The order of each
    list is the order of the metrics in the output table.

    Returns
    -------
    dict
        {"binary": (...), "weighted": (...)}, with lower-case metric names.

    Raises
    ------
    FileNotFoundError
        If config.json is not found.
    ValueError
        If config.json is not valid JSON, or "network_metrics" is missing or
        malformed.
    """
    config_file = os.path.join(os.path.dirname(__file__), "config", "config.json")
    try:
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Configuration file not found: {config_file}") from err
    except json.JSONDecodeError as err:
        raise ValueError(f"Invalid JSON in {config_file}: {err}") from err

    network_metrics = config.get("network_metrics")
    if not isinstance(network_metrics, dict):
        raise ValueError(f"'network_metrics' dictionary not found in {config_file}")

    defaults = {}
    for weighting in ("binary", "weighted"):
        names = network_metrics.get(weighting)
        if not isinstance(names, list) or not names:
            raise ValueError(
                f"'network_metrics' in {config_file} needs a non-empty "
                f"'{weighting}' list of metric names."
            )
        # Tuples, so the cached value cannot be modified by the caller
        defaults[weighting] = tuple(
            dict.fromkeys(str(m).strip().lower() for m in names)
        )
    return defaults


def _detect_weighting(conn_mat: np.ndarray) -> str:
    """'binary' if all non-zero off-diagonal entries share one value, else 'weighted'."""
    off_diag = conn_mat[~np.eye(conn_mat.shape[0], dtype=bool)]
    nonzero = off_diag[off_diag != 0]
    if nonzero.size == 0 or np.all(nonzero == nonzero[0]):
        return "binary"
    return "weighted"


def _mm_power_per_unit(unit: str, power: int) -> float:
    """Number of mm^power in one `unit` (e.g. 'cm2' -> 100, 'cm3' or 'ml' -> 1000)."""
    u = unit.strip().lower().replace("²", "2").replace("³", "3").replace("^", "")

    if power == 3 and u in ("ml", "cc"):
        u = "cm3"
    elif power == 3 and u == "l":
        u = "dm3"

    length, exponent = u[:-1], u[-1:]
    if exponent != str(power) or length not in _LENGTH_IN_MM:
        raise ValueError(
            f"Unsupported unit '{unit}' for a quantity of dimension {power}."
        )

    return _LENGTH_IN_MM[length] ** power


def _fs_measure_to_config(value: float, file_unit: str) -> tuple[float, str, str]:
    """
    Convert a FreeSurfer measure to the config units.

    Returns (value, metric, unit): mm³ -> volume, mm² -> area, anything else is
    returned unchanged as a unitless "measure".
    """
    u = (
        (file_unit or "")
        .strip()
        .lower()
        .replace("^", "")
        .replace("³", "3")
        .replace("²", "2")
    )

    if u == "mm3":
        converted, unit = convert_from_mm(value, "volume")
        return converted, "volume", unit
    if u == "mm2":
        converted, unit = convert_from_mm(value, "area")
        return converted, "area", unit
    if u in ("", "unitless"):
        return float(value), "measure", "au"
    return float(value), "measure", file_unit


def _normalize_stats_list(stats_list) -> list[str]:
    if stats_list is None:
        stats_list = ["value", "median", "std", "min", "max"]
    if isinstance(stats_list, str):
        stats_list = [stats_list]

    stats_list = [s.lower() for s in stats_list]
    unsupported = [s for s in stats_list if s not in _SUPPORTED_STATS]
    if unsupported:
        raise ValueError(
            f"Unsupported statistics: {', '.join(unsupported)}. "
            f"Supported: {', '.join(_SUPPORTED_STATS)}"
        )
    return stats_list


def _validate_table_type(table_type: str) -> None:
    if table_type not in ("region", "metric"):
        raise ValueError(
            f"Invalid table_type: '{table_type}'. Expected 'region' or 'metric'."
        )


def _check_output_table(output_table) -> str | None:
    """Validate the output path before any computation, so a bad path fails early."""
    if output_table is None:
        return None
    if not isinstance(output_table, (str, Path)):
        raise TypeError(
            f"output_table must be a string or Path, got {type(output_table)}"
        )

    output_table = str(output_table)
    output_dir = os.path.dirname(output_table)
    if output_dir and not os.path.isdir(output_dir):
        raise FileNotFoundError(
            f"Directory does not exist: {output_dir}. Please create the directory before saving."
        )
    return output_table


def _check_stats_file(stats_file) -> str:
    stats_file = str(stats_file)
    if not os.path.isfile(stats_file):
        raise FileNotFoundError(f"Stats file not found: {stats_file}")
    return stats_file


def _check_unique_names(names) -> None:
    duplicates = sorted(n for n, c in Counter(names).items() if c > 1)
    if duplicates:
        raise ValueError(
            f"Several regions share the same name: {duplicates}. "
            "Their values would overwrite each other in the table."
        )


def _check_vertex_count(annot, n_vertices: int, what: str) -> None:
    n_codes = np.asarray(annot.codes).shape[0]
    if n_codes != n_vertices:
        raise ValueError(
            f"The annotation has {n_codes} vertices but the {what} has {n_vertices}."
        )


def _load_annot(parc_file) -> "cltfree.AnnotParcellation":
    if isinstance(parc_file, (str, Path)):
        parc_file = str(parc_file)
        if not os.path.exists(parc_file):
            raise FileNotFoundError(f"Annotation file not found: {parc_file}")
        annot = cltfree.AnnotParcellation()
        annot.load_from_file(parc_file=parc_file)
        return annot

    if isinstance(parc_file, cltfree.AnnotParcellation):
        return copy.deepcopy(parc_file)

    raise TypeError(
        f"parc_file must be a string, Path or AnnotParcellation object, got {type(parc_file)}"
    )


def _clean_annot(annot, include_unknown: bool):
    """Remove non-anatomical regions (optionally) and codes missing from the table."""
    if not include_unknown:
        unk_indexes = cltmisc.get_indexes_by_substring(
            annot.regnames, _UNKNOWN_SUBSTRINGS
        )

        if len(unk_indexes) > 0:
            unk_codes = annot.regtable[unk_indexes, 4]
            annot.codes[np.isin(annot.codes, unk_codes)] = 0
            annot.regnames = np.delete(annot.regnames, unk_indexes).tolist()
            annot.regtable = np.delete(annot.regtable, unk_indexes, axis=0)

    not_in_table = np.setdiff1d(np.unique(annot.codes), annot.regtable[:, 4])
    annot.codes[np.isin(annot.codes, not_in_table)] = 0
    return annot


def _load_surface(surf_file):
    """Return (Surface, filename). The surface is not modified, so it is not copied."""
    if isinstance(surf_file, (str, Path)):
        surf_file = str(surf_file)
        if not os.path.exists(surf_file):
            raise FileNotFoundError(f"Surface file not found: {surf_file}")
        return cltsurf.Surface(surface_file=surf_file), surf_file

    if isinstance(surf_file, cltsurf.Surface):
        return surf_file, ""

    raise TypeError(
        f"surf_file must be a string, Path or Surface object, got {type(surf_file)}"
    )


def _load_parcellation(parc_file):
    """Return (Parcellation copy, filename), filename being '' when there is no real file."""
    if isinstance(parc_file, (str, Path)):
        parc_file = str(parc_file)
        if not os.path.exists(parc_file):
            raise FileNotFoundError(f"Parcellation file not found: {parc_file}")
        return cltparc.Parcellation(parc_file=parc_file), parc_file

    if isinstance(parc_file, cltparc.Parcellation):
        vparc_data = copy.deepcopy(parc_file)
        source = getattr(vparc_data, "parc_file", "")
        filename = source if isinstance(source, str) and os.path.isfile(source) else ""
        return vparc_data, filename

    if isinstance(parc_file, np.ndarray):
        return cltparc.Parcellation(parc_file=parc_file), ""

    raise TypeError(
        f"parc_file must be a string, Path, Parcellation object or numpy array, got {type(parc_file)}"
    )


def _apply_region_filters(
    vparc_data, exclude_by_code, exclude_by_name, include_by_code, include_by_name
) -> None:
    if exclude_by_code is not None:
        vparc_data.remove_by_code(codes2remove=exclude_by_code)
    if exclude_by_name is not None:
        vparc_data.remove_by_name(names2remove=exclude_by_name)
    if include_by_code is not None:
        vparc_data.keep_by_code(codes2keep=include_by_code)
    if include_by_name is not None:
        vparc_data.keep_by_name(names2keep=include_by_name)


def _region_names(vparc_data, labels, region_prefix: str) -> list[str]:
    """Names of the labels, generated from region_prefix for labels not in the table."""
    lut = {}
    if (
        getattr(vparc_data, "index", None) is not None
        and getattr(vparc_data, "name", None) is not None
    ):
        lut = {
            int(c): str(n)
            for c, n in zip(vparc_data.index, vparc_data.name, strict=False)
        }

    names = []
    for label in labels:
        name = lut.get(int(label))
        if name is None:
            name = str(
                cltmisc.create_names_from_indices([int(label)], prefix=region_prefix)[0]
            )
        names.append(name)
    return names


def _prefix_region_names(dict_of_cols: dict, prefix: str) -> dict:
    names = cltmisc.correct_names(list(dict_of_cols.keys()), prefix=prefix)
    return dict(zip(names, dict_of_cols.values(), strict=True))


def _split_region_names(df: pd.DataFrame) -> None:
    """Insert Supraregion and Hemisphere columns parsed from 'supra-hemi-name' region names."""
    supraregions, hemispheres = [], []
    for name in df["Region"].astype(str):
        parts = name.split("-")
        if len(parts) >= 3:
            supraregions.append(parts[0])
            hemispheres.append(parts[1])
        elif len(parts) == 2:
            supraregions.append(parts[0])
            hemispheres.append("unknown")
        else:
            supraregions.append("unknown")
            hemispheres.append("unknown")

    df.insert(0, "Supraregion", supraregions)
    df.insert(1, "Hemisphere", hemispheres)


def _format_table(
    dict_of_cols: dict, row_labels: list[str], table_type: str
) -> pd.DataFrame:
    """Build a 'metric' (one row per region) or 'region' (one column per region) table."""
    df = pd.DataFrame.from_dict(dict_of_cols)

    if table_type == "region":
        df.index = row_labels
        return df.reset_index().rename(columns={"index": "Statistics"})

    df = df.T
    df.columns = row_labels
    df = df.reset_index().rename(columns={"index": "Region"})
    _split_region_names(df)
    return df


def _add_metadata(df: pd.DataFrame, source, metric, units, filename) -> pd.DataFrame:
    """Prepend Source, Metric, Units and MetricFile. Each can be a scalar or a per-row list."""
    n_rows = df.shape[0]

    def _column(value):
        if isinstance(value, (list, tuple)):
            if len(value) != n_rows:
                raise ValueError(f"Expected {n_rows} values, got {len(value)}.")
            return list(value)
        return [value] * n_rows

    df.insert(0, "Source", _column(source))
    df.insert(1, "Metric", _column(metric))
    df.insert(2, "Units", _column(units))
    df.insert(3, "MetricFile", _column(filename))
    return df


def _collapse_values(values: list) -> str:
    return values[0] if len(set(values)) == 1 else "mixed"


def _finalize_table(df: pd.DataFrame, bids_file, add_bids_entities: bool, output_table):
    """Add BIDS entities, drop empty columns and save the table."""
    if add_bids_entities and bids_file:
        try:
            ent_list = cltbids.entities4table()
            df_add = cltbids.entities_to_table(
                filepath=bids_file, entities_to_extract=ent_list
            )
            df = cltmisc.expand_and_concatenate(df_add, df)
        except Exception as e:
            warnings.warn(f"Could not add BIDS entities: {str(e)}", stacklevel=3)

    df = cltmisc.drop_empty_columns(df)

    output_path = None
    if output_table is not None:
        df.to_csv(output_table, sep="\t", index=False)
        output_path = output_table

    return df, output_path


def _load_stats_config(config_json, level: str) -> dict:
    """Load the measurement configuration of a stats parser, falling back to the defaults."""
    if config_json is not None:
        config_json = str(config_json)
        if not os.path.isfile(config_json):
            warnings.warn(
                f"Config file not found: {config_json}. Using the default configuration.",
                stacklevel=3,
            )
        else:
            try:
                with open(config_json, encoding="utf-8") as f:
                    config_data = json.load(f)
                return config_data.get(level, config_data)
            except (OSError, json.JSONDecodeError) as e:
                warnings.warn(
                    f"Error loading config file {config_json}: {e}. Using the default configuration.",
                    stacklevel=3,
                )

    return get_stats_dictionary(level)


def _read_aseg_stats(stat_file: str):
    """
    Parse an aseg.stats file.

    Returns
    -------
    lines : list of str
        Raw lines of the file.
    measures : dict
        {short name or description: (value, unit)} from the "# Measure" lines.
    table : dict
        {StructName: {"SegId", "NVoxels", "Volume_mm3"}} from the segmentation table.
    """
    with open(stat_file, encoding="utf-8") as f:
        lines = f.readlines()

    measures = {}
    headers = None
    table_cols = {}
    data_rows = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("# Measure"):
            # "# Measure <struct>, <short name>, <description>, <value>, <unit>"
            parts = [p.strip() for p in stripped[len("# Measure") :].split(",")]
            if len(parts) < 4:
                continue
            try:
                value = float(parts[3])
            except ValueError:
                warnings.warn(f"Error parsing measure line: {stripped}", stacklevel=3)
                continue
            unit = parts[4] if len(parts) > 4 else ""
            measures[parts[1]] = (value, unit)
            measures[parts[2]] = (value, unit)
            # The structure name is ambiguous across lines, so it never overrides
            measures.setdefault(parts[0], (value, unit))

        elif stripped.startswith("# ColHeaders"):
            headers = stripped[len("# ColHeaders") :].split()

        elif stripped.startswith("# TableCol") and "ColHeader" in stripped:
            parts = stripped.split()
            try:
                table_cols[int(parts[2]) - 1] = parts[-1]
            except (ValueError, IndexError):
                pass

        elif not stripped.startswith("#"):
            data_rows.append(stripped.split())

    if headers is None and table_cols:
        headers = [table_cols[i] for i in sorted(table_cols)]

    col = {name: i for i, name in enumerate(headers or [])}
    i_seg = col.get("SegId", 1)
    i_nvox = col.get("NVoxels", 2)
    i_vol = col.get("Volume_mm3", 3)
    i_name = col.get("StructName", 4)
    max_idx = max(i_seg, i_nvox, i_vol, i_name)

    table = {}
    for parts in data_rows:
        if len(parts) <= max_idx:
            continue
        try:
            table[parts[i_name]] = {
                "SegId": int(parts[i_seg]),
                "NVoxels": int(float(parts[i_nvox])),
                "Volume_mm3": float(parts[i_vol]),
            }
        except ValueError:
            warnings.warn(f"Error parsing table line: {' '.join(parts)}", stacklevel=3)

    return lines, measures, table


def _legacy_line_lookup(lines, key: str, index: int):
    """Fallback lookup of a value by position in the first line containing `key`."""
    for line in lines:
        if key not in line:
            continue
        parts = line.split()
        idx = index + len(parts) if index < 0 else index
        if not 0 <= idx < len(parts):
            continue
        try:
            value = float(parts[idx].split(",")[0])
        except ValueError:
            continue
        unit = (
            line.strip().split(",")[-1].strip()
            if line.startswith("# Measure")
            else "mm^3"
        )
        return value, unit
    return None


def _detect_hemisphere(stats_file: str, lines) -> str:
    basename = os.path.basename(stats_file)
    if "lh." in basename:
        return "lh"
    if "rh." in basename:
        return "rh"

    for line in lines:
        if line.startswith("# hemi"):
            parts = line.split()
            if len(parts) >= 3 and parts[2] in ("lh", "rh"):
                return parts[2]

    warnings.warn(
        f"Could not determine hemisphere from file: {stats_file}. Using 'lh' as default.",
        stacklevel=3,
    )
    return "lh"


def _parse_aparc_headers(lines) -> list[str]:
    """Column headers of an aparc.stats table: ColHeaders, then TableCol, then the standard layout."""
    for line in lines:
        if line.startswith("# ColHeaders"):
            return line[len("# ColHeaders") :].split()

    table_cols = {}
    for line in lines:
        if line.startswith("# TableCol") and "ColHeader" in line:
            parts = line.split()
            try:
                table_cols[int(parts[2])] = parts[-1]
            except (ValueError, IndexError):
                continue
    if table_cols:
        return [table_cols[i] for i in sorted(table_cols)]

    if any(line.strip() and not line.startswith("#") for line in lines):
        warnings.warn(
            "No column headers found; assuming the standard aparc.stats layout.",
            stacklevel=3,
        )
        return [
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

    raise ValueError("Could not find column headers in the stats file")


def _parse_std_value(parts, metric_name: str, metric_info: dict, column_indices: dict):
    """Standard deviation of a metric, given by column index or column name in 'std_index'."""
    std_ref = metric_info.get("std_index")
    if std_ref in (None, "null", "None", ""):
        return None

    if isinstance(std_ref, (int, np.integer)) or (
        isinstance(std_ref, str) and std_ref.isdigit()
    ):
        std_idx = int(std_ref)
    else:
        std_idx = column_indices.get(str(std_ref))
        if std_idx is None and metric_name.lower() == "thickness":
            std_idx = column_indices.get("ThickStd")

    if std_idx is None or not 0 <= std_idx < len(parts):
        return None

    try:
        return float(parts[std_idx])
    except ValueError:
        return None
