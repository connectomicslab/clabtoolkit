=======
History
=======

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
