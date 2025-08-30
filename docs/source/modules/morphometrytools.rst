morphometrytools module
=======================

.. automodule:: clabtoolkit.morphometrytools
   :members:
   :undoc-members:
   :show-inheritance:

The morphometrytools module provides specialized tools for surface-based morphometric analysis, enabling extraction and analysis of cortical measurements across brain regions.

Key Features
------------
- Regional value extraction from surface annotations  
- Multi-hemisphere morphometric analysis
- Surface area and Euler characteristic computation from meshes
- FreeSurfer statistics file parsing and processing
- Volume-based morphometry from parcellations
- Statistical summary generation and unit management

Main Functions
--------------

Regional Analysis
~~~~~~~~~~~~~~~~~
- ``compute_reg_val_fromannot()``: Extract regional statistics from surface scalar data
- ``process_hemisphere_morphometry()``: Process morphometric data for single hemisphere
- ``generate_morphometry_table()``: Create comprehensive morphometry tables
- ``validate_morphometric_data()``: Quality control for morphometric measurements

FreeSurfer Analysis
~~~~~~~~~~~~~~~~~~~
- ``parse_freesurfer_global_fromaseg()``: Parse FreeSurfer global statistics from aseg
- ``parse_freesurfer_stats_fromaseg()``: Parse FreeSurfer statistics from aseg
- ``parse_freesurfer_cortex_stats()``: Parse FreeSurfer cortex statistics
- ``get_stats_dictionary()``: Get statistics dictionary from FreeSurfer data

Utility Functions
~~~~~~~~~~~~~~~~~
- ``network_metrics_to_table()``: Convert network metrics to table format
- ``stats_from_vector()``: Compute statistics from vector data
- ``get_units()``: Get measurement units for morphometric data

Common Usage Examples
---------------------

Basic regional morphometry extraction::

    from clabtoolkit.morphometrytools import compute_reg_val_fromannot
    
    # Extract cortical thickness by region
    thickness_stats = compute_reg_val_fromannot(
        scalar_file="lh.thickness.mgh",
        annot_file="lh.aparc.a2009s.annot", 
        lut_file="aparc.a2009s.lut"
    )
    
    print(thickness_stats.head())

Multi-hemisphere analysis::

    # Process both hemispheres
    hemispheres = ['lh', 'rh']
    all_stats = {}
    
    for hemi in hemispheres:
        stats = compute_reg_val_fromannot(
            scalar_file=f"{hemi}.thickness.mgh",
            annot_file=f"{hemi}.aparc.annot",
            lut_file="aparc.lut"
        )
        all_stats[hemi] = stats
    
    # Combine hemisphere data
    combined_stats = pd.concat(all_stats, axis=0)

FreeSurfer statistics parsing::

    # Parse cortical statistics from FreeSurfer
    cortex_stats = parse_freesurfer_cortex_stats(
        stats_file="/path/to/freesurfer/stats/lh.aparc.stats"
    )
    
    # Get global statistics from aseg
    global_stats = parse_freesurfer_global_fromaseg(
        aseg_file="/path/to/freesurfer/stats/aseg.stats"
    )
    
    # Convert to table format with proper units
    stats_table = network_metrics_to_table(
        stats_dict=cortex_stats,
        subject_id="sub-001"
    )
    
    # Get measurement units
    units = get_units(measure_type="thickness")
    print(f"Thickness measured in: {units}")