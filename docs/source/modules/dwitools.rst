dwitools module
===============

.. automodule:: clabtoolkit.dwitools
   :members:
   :undoc-members:
   :show-inheritance:

The dwitools module provides specialized tools for diffusion-weighted imaging (DWI) analysis, tractography processing, and white matter analysis.

Key Features
------------
- DWI volume manipulation and quality control
- Tractography file processing (.trk, .tck formats)
- B-value and gradient direction management
- Bundle analysis and clustering
- DTI and advanced diffusion modeling support
- Integration with DIPY and MRtrix workflows

Main Functions
--------------

Volume Management
~~~~~~~~~~~~~~~~~
- ``delete_dwi_volumes()``: Remove specific DWI volumes based on indices or b-values
- ``get_b0s()``: Extract b=0 volumes from DWI data

Tractography Processing
~~~~~~~~~~~~~~~~~~~~~~~
- ``tck2trk()``: Convert TCK format to TRK format
- ``trk2tck()``: Convert TRK format to TCK format
- ``concatenate_tractograms()``: Concatenate multiple tractogram files
- ``resample_streamlines()``: Resample streamlines in tractograms
- ``resample_tractogram()``: Resample entire tractogram
- ``compute_tractogram_centroids()``: Compute centroids of tractogram streamlines
- ``create_trackvis_colored_trk()``: Create colored TRK files for TrackVis
- ``extract_cluster_by_id()``: Extract specific cluster from tractogram
- ``explore_trk()``: Explore TRK file properties
- ``interpolate_on_tractogram()``: Interpolate data on tractogram

Main Classes
------------

TRKExplorer
~~~~~~~~~~~
Class for exploring and analyzing TRK tractogram files.

Common Usage Examples
---------------------

DWI volume manipulation::

    from clabtoolkit.dwitools import delete_dwi_volumes
    
    # Remove specific volumes from DWI dataset
    delete_dwi_volumes(
        dwi_file="dwi.nii.gz",
        bval_file="dwi.bval",
        bvec_file="dwi.bvec",
        volumes_to_delete=[0, 5, 10],  # Remove specific volumes
        output_prefix="cleaned_dwi"
    )

Working with b-values::

    # Extract b=0 volumes
    b0_indices = get_b0s(
        bval_file="dwi.bval",
        tolerance=50
    )

Tractography format conversion::

    # Convert between tractography formats
    tck2trk(
        input_tck="tractography.tck",
        output_trk="tractography.trk",
        reference="dwi.nii.gz"
    )
    
    # Explore tractography file
    explorer = TRKExplorer("tractography.trk")
    summary = explorer.explore()
    print(f"Number of streamlines: {summary['n_streamlines']}")
    
    # Concatenate multiple tractograms
    concatenate_tractograms(
        input_files=["tract1.trk", "tract2.trk"],
        output_file="combined.trk"
    )