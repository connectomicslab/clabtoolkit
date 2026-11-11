=======
History
=======

Unreleased
----------

* ``generate_image_selection_webpage`` (``qcqatools``) has a new ``images`` input to build the page from chosen images instead of searching a folder: one image, a list of images, or a text file with one path per line (empty lines and lines starting with ``#`` are ignored). The images can be in different folders and in any format a browser shows (PNG, JPG, GIF, WebP, SVG, BMP); paths that do not exist are reported and left out. ``root_directory`` is now optional when ``images`` is given (the folder common to the images is used). The images are now linked with paths relative to the folder of the page, so they are also shown when ``output_html`` is outside the root directory, and special characters in the paths are encoded. Each image of the page can now be rated as pass, doubtful or fail, with a comment (``rating=True``, the default): the ratings are kept in the browser, the keyboard can be used (``j``/``k`` for the next and previous image, ``p``/``d``/``f`` to rate the image in view) and a button downloads them as ``image_ratings.tsv``, a table with one row per image
* New ``is_fs_output`` function in ``freesurfertools``: tells if a folder is the output folder of a FreeSurfer subject (it has the ``mri``, ``surf``, ``label`` and ``scripts`` folders and ``mri/orig/001.mgz``), which the templates such as ``fsaverage`` are not. The quality control reports are only generated for these folders, and ``generate_cohort_qc_webpage`` selects the subjects with ``fs_ids`` (one identifier, a list or a text file); the folders that are not subjects get no report and are listed at the end of the index with the status ``not-freesurfer`` (with a warning when they were requested). The index has a box for each status and each rating to show any combination of them, a box in each row to leave it out, and a button to download the identifiers of the rows shown and ticked as a text file that ``fs_ids`` accepts
* New ``generate_cohort_qc_webpage`` function in ``freesurfertools``: creates ``<subjs_dir>/qcqa/index.html``, a table with one row per subject of the FreeSurfer subjects directory (processing status, holes and defects of the surfaces before the topology correction, eTIV, mean cortical thickness computed from the thickness maps, rating and a link to the individual report) that can be sorted and filtered, together with ``metadata/cohort_qc.tsv`` and ``metadata/cohort_qc.json``. With ``generate_reports=True`` it also generates the individual reports that are missing. When at least ``min_subjects`` (20) subjects have a morphometry table in their report, the index flags the regional measures (thickness and white surface area of the lobes and of the Desikan-Killiany regions, and volumes of the aseg) that are farther than ``outlier_threshold`` (4) robust standard deviations from the median of the cohort, and saves them in ``metadata/cohort_outliers.tsv``. ``max_workers`` generates several individual reports at the same time, each one in its own process. The ratings of all the subjects are exported to ``metadata/cohort_ratings.tsv`` (one row for the subject and one for each rated or commented figure), and ``subjects.js`` lists the reports so the individual reports can link to the previous and next subject
* New ``FreeSurferSubject.generate_qc_webpage`` method: creates the folder ``qcqa`` (``images``, ``metadata`` and ``index.html``) inside the FreeSurfer folder of the subject with a visual quality control report. The page starts with the metadata of the subject (processing status, FreeSurfer version, last line of ``recon-all.log``, mask volumes) and shows mosaics of the ``T1.mgz`` alone and overlaid with the binarized ``brainmask.mgz`` and ``wm.mgz`` and with the ``aparc+aseg``, ``aparc.a2009s+aseg`` and ``aparc.DKTatlas+aseg`` parcellations, colored with ``$FREESURFER_HOME/FreeSurferColorLUT.txt``, and with the lines where the white and pial surfaces intersect the slices. Four more figures, rendered with ``BrainPlotter.plot_hemispheres``, show the cortical thickness over the pial surfaces, the curvature over the white surfaces and the ``aparc`` and ``aparc.a2009s`` annotations over the inflated surfaces. The names and units of the maps are the ones of ``config/config.json``. With ``stats_table=True`` (the default) the table of ``create_subject_stats_table`` is saved as ``metadata/<subj_id>_stats_table.csv`` and shown in a Morphometry section: thickness and surface areas of the hemispheres and lobes, and a table with all the regional measures that can be filtered and sorted. Clicking a figure opens the full-size image in a new tab, a button prints the report or saves it as PDF with a layout for the page (also in the index of the cohort), and the keyboard can be used to review: ``j``/``k`` move between the figures, ``p``/``d``/``f`` rate the figure in view (the subject with Shift) and the arrow keys go to the previous and next subject. The page also has a table of global measures (volumes of ``aseg.stats``, surface area, mean cortical thickness computed from the thickness maps, and the holes and defects of the surfaces before the topology correction) and a rating widget (pass, doubtful or fail per figure and for the subject, with comments, downloaded as a JSON file). The helpers of the report are in the new ``freesurfertools_utils`` module. ``n_slices`` sets the number of slices along X, Y and Z and ``surf_views`` the views of the hemispheres
* ``config/config.json`` now has a ``metric`` field instead of ``metrics_units``: a dictionary with one entry per metric, such as ``"thickness": {"fullname": "Cortical thickness", "units": "mm"}``. ``get_units`` reads it (a plain ``{name: units}`` mapping or the old ``metrics_units`` key are still accepted in a custom configuration), and the new ``get_fullnames`` returns the full names
* Added the DOI of version 0.6.3 to ``CITATION.cff``
* The continuous integration now also tests Python 3.12 on Windows and macOS, so every combination of the three operating systems and Python 3.11 to 3.13 is tested
* Removed the unused ``.travis.yml`` (the CI runs on GitHub Actions) and the temporary ``environment_tmp.yaml``

0.6.3 (2026-11-04)
------------------

* Removed from ``examples-representative-workflows.ipynb`` a leftover call that saved the filtered tractogram to a fixed folder of the author's computer, which does not exist on other machines. The notebook already saves the TRK and TCK files to a temporary folder
* The header of the Talairach transform of the sample data no longer contains the name of the user who created it
* ``CITATION.cff`` lists the DOI of the concept as an identifier, and ``version`` and ``date-released`` identify the release. The DOI of each version is added to the file after its release, because Zenodo creates it when the release is published

0.6.2 (2026-11-04)
------------------

* A version tag is now published to PyPI only if the test suite passes: the publishing workflow has a ``tests`` job (``ruff`` and ``pytest``) that the PyPI job waits for
* ``pyvista`` is now installed with its ``jupyter`` extra (``pyvista[jupyter]``) in the dependencies, the requirement files and the environment files, so the notebook rendering works without a separate installation
* The sample data are now fetched with the DOI of the version of the Zenodo record (10.5281/zenodo.23702132) instead of the DOI of the concept (10.5281/zenodo.23702131), so the downloads keep matching their SHA-256 checksums if a new version of the record is published
* The unit of ``charm_dperp`` in ``config/config.json`` is now ``mm2/s``, as for the other diffusion metrics
* Removed ``config/viz_views_test.json``, which was not used
* Added the notebook ``examples-regression-checks.ipynb``, which reproduces the problems reported during the review and shows that they are fixed, and updated the introduction of ``clabtoolkit_examples.ipynb``

0.6.1 (2026-10-29)
------------------

* Added the ``save_path``, ``dpi`` and ``overwrite`` options to ``Connectome.plot_matrix``, to save the figure instead of showing it
* Added the ``groups`` option to ``Connectome.plot_matrix``, which orders the regions by group (given by region names or codes), separates the groups with lines and labels them
* Renamed the ``PointCloud`` class to ``PointsCloud``, ``merge_pointclouds`` to ``merge_pointsclouds`` and its ``pointclouds`` parameter to ``pclouds``. The old names are not kept as aliases
* Added ``PointsCloud.to_surface``, which converts the points to a surface with one triangular sphere per point and transfers the maps: maps with a colortable become a parcellation (optionally returned as ``AnnotParcellation`` objects) and the rest vertex-wise maps
* Added ``list_sample_data``, ``fetch_sample_data`` and ``get_sample_data_dir`` to ``misctools``. The sample data (the Colin27 average brain in the tallin space) is published in a Zenodo record (DOI 10.5281/zenodo.23702131): the surfaces, cortical maps and tractograms are downloaded the first time they are needed with ``pooch``, checked against their SHA-256 checksums and cached in the user cache folder (set ``CLABTOOLKIT_DATA_DIR`` to use another one), while the annotations, the parcellation and the transform are shipped with the package. ``pooch`` is now a dependency
* Reorganized the sample data of the package: ``data/parcellationtools`` is now ``data/parcellations``, the subject is ``sub-colin27``, the file names follow the BIDS entity order (``hemi`` before ``space``) and the annotations are ``.label.gii`` files. The surfaces, cortical maps and tractograms were removed from the package
* Added ``Surface.load_from_file`` support for GIFTI, VTK, PLY, OBJ and STL surfaces, ``Surface.save_surface`` GIFTI export and GIFTI shape maps in ``Surface.load_scalar_maps``
* ``compute_reg_val_fromannot`` now reads GIFTI maps
* Added ``Tractogram.remove_maps``, which removes one or several maps and their colortables
* ``AnnotParcellation.export_to_tsv`` gives the index 0 to the ``unknown`` region and numbers the other regions from 1
* The usage notebooks, the quickstart and the README now use the sample data retrieved from Zenodo instead of simulated data or FreeSurfer outputs, and the new notebook ``examples-representative-workflows.ipynb`` and the docs cards of the workflows and ``clabtoolkit_examples.ipynb`` were added
* The table of contents of the usage notebooks links to the headings, so it works in the VS Code outline, and typos were corrected

0.6.0 (2026-10-25)
------------------

* Added shape, indexing, iteration and the NumPy array protocol to ``RegionTimeSeries``
* Added the ``reload_config`` method to ``BrainPlotter``
* ``process_v_limits`` now accepts ``None`` and automatic limits
* ``compute_fc_matrix`` now accepts a selection of a single region
* ``BrainPlotter.plot`` keeps the maps in the order requested
* Colorbars take their font sizes from the figure configuration
* ``interpolate_on_tractogram`` keeps the points on the last voxel inside the interpolation grid
* ``load_scalar_maps`` only falls back to the morphometry reader for non-table files
* ``get_subjects`` validates the root directory and lists only directories
* Imported the colors in ``misctools_utils`` from ``colorstools``
* Documented every public function of ``misctools``, added module docstrings and an overview above the API listing in each module page
* Reorganized the documentation: task index, card grids with module icons in the API Reference and Examples pages, and a "What can you do with CLABTOOLKIT?" section in the README and the docs
* Usage examples no longer use fixed paths, and the examples of ``filter_by_mask`` and the simulated branches were added or fixed
* Extended the test suite (``BrainPlotter``, ``plot_hemispheres``, ``plot_scene``, ``create_carpet_plot``, ``RegionTimeSeries``, ``dicomtools`` and the ``misctools`` helpers), using fake plotters so the tests do not need a display
* Updated ``CONTRIBUTING.rst`` (coding standards, documentation build, maintainer-only releases) and included the css and svg files of the docs in ``MANIFEST.in``

0.5.3 (2026-10-22)
------------------

* Pointed the documentation badge to app.readthedocs.org, so it renders on PyPI instead of showing its alt text
* Included .zenodo.json in the source distribution (fixes the check-manifest step of the CI)

0.5.2 (2026-10-22)
------------------

* Updated the README (absolute image URLs so they render on PyPI, new badges and ecosystem figure)
* Added .zenodo.json so the Zenodo record has a description and complete metadata
* Added the Zenodo DOI of the toolkit to CITATION.cff and the README
* Replaced the LICENSE file by the unmodified Apache-2.0 text, so GitHub recognizes the license, and moved the copyright statement to a NOTICE file

0.5.1 (2026-10-21)
------------------

* Added CITATION.cff, with the citation of the toolkit paper (arXiv:2607.02638), so GitHub offers "Cite this repository"
* Included CITATION.cff in the source distribution
* Updated the release steps of CONTRIBUTING.rst: version and date of CITATION.cff, and the GitHub release that archives the version in Zenodo

0.5.0 (2026-10-21)
------------------

* Fixed the FA, VF, RA and GA maps of maps_from_tensor_eigenvalues in dwitools (FA used a factor of 0.5 instead of 3/2, VF and RA used the radial instead of the mean diffusivity, and GA was not the geodesic anisotropy). Maps computed with previous versions will change
* Fixed the gradient signs recovered by DiffusionScheme.from_bmatrix_array (they were always positive)
* Fixed delete_dwi_volumes and get_b0s reading only 5 b-values from bval files stored as a column, and added a check of the number of b-values against the number of volumes
* Fixed get_b0s crashing without an output name or a bval file and saving an empty image when no B0 was found; it now always returns (image, indices)
* Fixed delete_dwi_volumes returning a single path instead of a tuple when no volume was removed, and restored its docstring
* Added DiffusionScheme.simulate_dwi_acq_scheme to simulate shelled and cartesian (DSI) acquisition schemes
* Added save_path, non_blocking and window_size to DiffusionScheme.plot to save the scheme as an image, a vector graphic or an interactive HTML file
* DiffusionScheme.plot draws the directions as sphere meshes so they keep their 3D shading in notebooks and HTML files, uses the right range in the colorbar and places the DSI shells at their q-space radius
* Added window_size to BrainPlotter.plot, plot_hemispheres and plot_scene. Notebook figures now use PyVista's default window size (1024 x 768) instead of the monitor size, which overflowed the cell output
* Fixed the Gaussian smoothing of extract_centroid_from_volume and extract_mesh_from_volume in imagetools (the boolean dtype shrank the region and the mesh ignored the smoothing)
* extract_centroid_from_volume returns the voxel count of the input region, which fixes the region volumes reported by Parcellation.compute_centroids
* Fixed MorphologicalOperations.closing removing foreground voxels on the image border
* Fixed crop_image_from_mask excluding the last voxel of the mask on each axis
* Fixed create_spams with a dictionary lookup table and the name of the colored SPAM image
* spams2maxprob_from_volume and region_growing no longer modify their input arrays
* Fixed the session names of dicomtools losing the leading zero of the study time (a study at 09:15:30 gave 'ses-2024031291530' instead of 'ses-20240312091530'). Session folders of studies acquired before 10:00 will be named differently
* Fixed copy_dicom_file (and org_conv_dicoms) deleting the existing copy without copying the file again when overwrite=True
* The ses_id parameter of org_conv_dicoms and copy_dicom_file now sets the session label ('ses-<ses_id>'); it was ignored
* Fixed create_session_series_names failing without SeriesNumber, StudyDate or StudyTime, and copy_dicom_file failing on files that are not DICOMs
* org_conv_dicoms returns instead of exiting Python when no subjects are found, matches the subject IDs exactly (with or without 'sub-'), accepts acq_date as MM/DD/YYYY or YYYY-MM-DD, and matches the demographics rows by exact participant ID
* Added the example notebooks for the imagetools, dwitools and dicomtools modules
* Renamed get_ids2process to get_fullids2process in pipelinetools, with a source parameter ('folders' or 'files') to build the Full IDs from the sub-* folders or from the image files
* The suffix column of get_bids_database_table is now named Type, and the tables of pipelinetools use the original names of the BIDS entities (Participant, Session, Run, ...) followed by Type, Extension, Full_path and Count. Code reading these columns by their old names must be updated
* entities_to_table and process_file accept pathlib.Path objects. entities_to_table raises ValueError instead of TypeError when filepath is None
* Added the param entity (Parameter) and the dwimap suffix to the BIDS configuration, for the maps derived from DWI
* update_figure_config validates all its parameters before applying any, so a call with an invalid value no longer leaves the figure configuration partially updated
* Corrected the docstrings of the BrainPlotter themes and figure configuration: default auto_save values, default figure settings and the list of themes
* Added the example notebook for the pipelinetools module
* Added test suites for the pipelinetools, dicomtools, imagetools, segmentationtools and visualizationtools modules
* Updated CONTRIBUTING.rst (development setup, CI checks, tests, release steps and issue links), tox.ini (Python 3.11-3.13 and the lint checks) and the version and Python requirements of DOCUMENTATION.md
* Fixed the formatting of the pipelinetools docstrings that produced warnings in the documentation build

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
