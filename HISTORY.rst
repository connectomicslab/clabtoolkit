=======
History
=======

0.4.5 (2026-08-28)
------------------

* Added create_5tt to parcellationtools to generate MRtrix3 5TT images
* Added a method to merge the cortical gray matter with the white matter adjacent to the cortex, and used the merged image to build the 5TT image
* Added the apply_affine method to the Surface, PointCloud and Tractogram classes, with an inplace option
* Added connected components detection for binary images
* Added the option to interpolate on Parcellation objects
* Added a method to compare dictionaries
* Added the process_subject function to bidstools
* Added the brain_pos_col and cb_pos functions to build_visualization_layout
* Added the ventral DC to the list of subcortical regions
* Added examples for create_5tt, merge_ctx_wm, apply_cras and apply_affine
* Refactored the visualizationtools module and fixed several plotting errors
* Renamed the force parameter to overwrite in all the functions and methods writing files
* Renamed the transform method of the PointCloud class to apply_affine
* Fixed the apply_cras method of the Surface class
* Fixed a bug in the subject selection of bidstools
* Fixed the line spacing, the append handling and the header lines of export_to_lutctab in colorstools
* Fixed the anchors, the links and the FreeSurfer paths of the notebooks
* Silenced the tight_layout warning of create_carpet_plot
* Raised the minimum Python version to 3.11 and adopted the built-in generics and the union operator in the type hints

0.4.4 (2026-07-17)
------------------

* Added get/set accessor methods for Parcellation attributes (data, affine, index, names and colors) with validation
* Added support for organizing DICOM folders without requiring individual per-series folders
* Improved the method to compress DICOM sessions
* Fixed save_tractogram writing the wrong format when overwriting an existing file
* Fixed colortable loading in tracttools for point-only maps and for maps absent from both data_per_point and data_per_streamline
* Fixed tractogram merging in tracttools
* Fixed out_image=None crash and added pathlib.Path support in delete_dwi_volumes
* Fixed docstrings and type hints in tracttools
* Renamed the streamline point count variable to nb_points for consistency
* Removed surf as a variable name for tractograms
* Removed the old connectome and old_dwitool modules (use connectivitytools and dwitools instead)
* Reorganized the package imports
* Documented the tracttools, pointstools and colorstools modules for the first time
* Corrected the module documentation to match the actual API

0.4.3 (2026-06-10)
------------------

* Added compute_fc_matrix method to Parcellation class
* Added get_info method to Connectome and Parcellation classes
* Added create_carpet_plot function in visualizationtools
* Added compute_scalar_maps_from_tensor in diffusiontools
* Added RegionTimeSeries object and integration across the package
* Added method to generate connectomes
* Added connected components computation from edge arrays
* Added region lookup methods in AnnotParcellation
* Added comprehensive get_info() for AnnotParcellation
* Added method to binarize and dilate 3D arrays by millimeter distance
* Added support for pathlib.Path objects as inputs
* Added option to make TSV file compatible with templateflow
* Added new colors to the bcolors class
* Added bundle id map and updated names
* Added notebook with examples for OHBM 2026
* Added new ecosystem figure and logo
* Refactored stats_from_vector for edge cases with nonzeros_only=True
* Refactored AnnotParcellation gii2annot/annot2gii to use nibabel instead of FreeSurfer mris_convert
* Extended merge_to_4d method to deal with DWI data
* Fixed strict entity and suffix validation in is_bids_filename
* Fixed pyvista notebook crash from threaded display
* Fixed color table names bug
* Fixed dimensions field bugs ("dims" vs "dim")
* Multiple bug fixes, dependency updates, and documentation improvements

0.4.1 (2026-02-19)
------------------

* Major refactoring of Parcellation class with improved attribute handling
* Enhanced color table loading and export with multiple format support
* Added region adjacency computation to Parcellation
* Improved BIDS file entity extraction with parallel processing
* Added DiffusionScheme class for gradient visualization
* Enhanced Connectome class initialization and metrics computation
* Added derivatives inventory functionality
* Improved surface tools with PyVista fallback support
* Added usage examples notebooks
* Multiple bug fixes in mask handling, color processing, and data type conversion
* Improved documentation and code organization

0.3.4 (2025-01-09)
------------------

* Enhanced documentation and ReadTheDocs configuration
* Moved region growing method to imagetools module  
* Improved Sphinx autodoc configuration for better API documentation
* Multiple improvements to surface visualization and plotting functionality
* Added new morphometry and connectivity analysis tools
* Enhanced BIDS dataset handling and entity management
* Improved surface mesh operations and color mapping
* Added network analysis tools and visualization capabilities

0.3.1 (2025-05-22)
------------------

* Fourth release on PyPI. 
