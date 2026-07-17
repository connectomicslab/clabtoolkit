morphometrytools module
=======================

.. automodule:: clabtoolkit.morphometrytools
   :members:
   :undoc-members:
   :show-inheritance:

The morphometrytools module provides specialized tools for surface-based and volume-based morphometric analysis, enabling extraction and analysis of cortical measurements across brain regions.

Key Features
------------
- Regional value extraction from surface annotations
- Surface area, vertex count and Euler characteristic computation from meshes
- Volume-based morphometry from parcellations
- FreeSurfer statistics file parsing and processing
- Statistical summary generation and unit management
- BIDS entity propagation into the output tables

Main Functions
--------------

Surface-based Analysis
~~~~~~~~~~~~~~~~~~~~~~
- ``compute_reg_val_fromannot()``: Extract regional statistics from surface scalar data
- ``compute_reg_area_fromsurf()``: Compute regional surface area from a mesh
- ``compute_reg_nvertices_fromsurf()``: Compute the vertex count per region
- ``compute_euler_fromsurf()``: Compute the Euler characteristic of a surface
- ``area_from_mesh()``: Compute the area of a mesh
- ``euler_from_mesh()``: Compute the Euler characteristic of a mesh

Volume-based Analysis
~~~~~~~~~~~~~~~~~~~~~
- ``compute_reg_val_fromparcellation()``: Extract regional statistics from a parcellation
- ``compute_reg_volume_fromparcellation()``: Compute regional volumes from a parcellation

FreeSurfer Analysis
~~~~~~~~~~~~~~~~~~~
- ``parse_freesurfer_global_fromaseg()``: Parse FreeSurfer global statistics from aseg
- ``parse_freesurfer_stats_fromaseg()``: Parse FreeSurfer statistics from aseg
- ``parse_freesurfer_cortex_stats()``: Parse FreeSurfer cortex statistics
- ``get_stats_dictionary()``: Get the statistics dictionary from FreeSurfer data

Utility Functions
~~~~~~~~~~~~~~~~~
- ``network_metrics_to_table()``: Compute network metrics from a connectivity matrix
- ``stats_from_vector()``: Compute summary statistics from a vector
- ``get_units()``: Get the measurement units for a list of metrics

Common Usage Examples
---------------------

Basic regional morphometry extraction::

    from clabtoolkit.morphometrytools import compute_reg_val_fromannot

    # Extract cortical thickness by region
    thickness_stats, metric_vect, output_table = compute_reg_val_fromannot(
        metric_file="/path/to/lh.thickness",
        parc_file="/path/to/lh.aparc.a2009s.annot",
        hemi="lh",
        metric="thickness",
        stats_list=["value", "median", "std", "min", "max"]
    )

    print(thickness_stats.head())

Multi-hemisphere analysis::

    import pandas as pd

    # Process both hemispheres
    all_stats = {}
    for hemi in ["lh", "rh"]:
        stats, _, _ = compute_reg_val_fromannot(
            metric_file=f"/path/to/{hemi}.thickness",
            parc_file=f"/path/to/{hemi}.aparc.annot",
            hemi=hemi,
            metric="thickness"
        )
        all_stats[hemi] = stats

    # Combine hemisphere data
    combined_stats = pd.concat(all_stats, axis=0)

Surface geometry measures::

    from clabtoolkit.morphometrytools import (
        compute_reg_area_fromsurf,
        compute_euler_fromsurf,
    )

    # Regional surface area
    area_df, _ = compute_reg_area_fromsurf(
        surf_file="/path/to/lh.pial",
        parc_file="/path/to/lh.aparc.annot",
        hemi="lh",
        surf_type="pial"
    )

    # Euler characteristic, a common QC measure
    euler_df, _ = compute_euler_fromsurf(
        surf_file="/path/to/lh.pial",
        hemi="lh",
        surf_type="pial"
    )

Volume-based morphometry::

    from clabtoolkit.morphometrytools import compute_reg_volume_fromparcellation

    # Regional volumes straight from a parcellation
    volume_df, out_path = compute_reg_volume_fromparcellation(
        parc_file="/path/to/aparc+aseg.nii.gz",
        output_table="/path/to/volumes.tsv",
        include_global=True
    )

FreeSurfer statistics parsing::

    from clabtoolkit.morphometrytools import (
        parse_freesurfer_cortex_stats,
        parse_freesurfer_global_fromaseg,
        get_units,
    )

    # Parse cortical statistics
    cortex_stats, _ = parse_freesurfer_cortex_stats(
        stats_file="/path/to/freesurfer/stats/lh.aparc.stats",
        hemi="lh"
    )

    # Global statistics from aseg
    global_stats, _ = parse_freesurfer_global_fromaseg(
        "/path/to/freesurfer/stats/aseg.stats"
    )

    # Units for a list of metrics
    units = get_units(metrics=["thickness", "area", "volume"])
    print(units)

Summary statistics from a vector::

    from clabtoolkit.morphometrytools import stats_from_vector

    # Compute summary statistics for an arbitrary metric vector
    stats = stats_from_vector(
        metric_vect=values,
        stats_list=["value", "median", "std", "min", "max"],
        nonzeros_only=True
    )
