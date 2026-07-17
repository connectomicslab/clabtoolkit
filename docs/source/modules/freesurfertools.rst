freesurfertools module
======================

.. automodule:: clabtoolkit.freesurfertools
   :members:
   :undoc-members:
   :show-inheritance:

The freesurfertools module provides comprehensive integration with the FreeSurfer neuroimaging suite, enabling processing of cortical surfaces, annotation files, and morphometric data.

Key Features
------------
- Process FreeSurfer annotation files (.annot, .gcs, .gii formats)
- Correct and fill vertexwise parcellations
- Whole-subject management and stats table generation
- Surface-based and volume-based morphometry computation
- Container technology integration (Docker/Singularity)
- Multi-format conversion capabilities
- Parse FreeSurfer LTA transform files and extract CRAS coordinates

Main Classes
------------

AnnotParcellation
~~~~~~~~~~~~~~~~~
The primary class for working with FreeSurfer annotation files.

Loading and Saving:
- ``load_from_file()``: Load an annotation file with an optional reference surface
- ``create_from_data()``: Build an annotation from codes, a colour table and names
- ``save_annotation()``: Save the annotation
- ``is_loaded()``: Check whether annotation data is loaded

Region Information:
- ``get_info()``: Summary information about the annotation
- ``get_info_from_name()``: Look up a region by name
- ``get_info_from_id()``: Look up a region by ID
- ``get_regions_info()``: Retrieve information for a list of regions

Parcellation Processing:
- ``correct_vertexwise_parcellation()``: Fix unlabeled vertices in a parcellation
- ``fill_parcellation()``: Fill a parcellation using a label and surface file
- ``group_into_lobes()``: Group regions into anatomical lobes
- ``map_values()``: Map regional values onto the annotation

Conversion and Export:
- ``annot2gcs()``: Convert an annotation to GCS format
- ``gcs2annot()``: Convert a GCS file to annotation format
- ``annot2gii()`` / ``gii2annot()``: Convert between annotation and GIfTI
- ``export_to_tsv()``: Export the annotation to TSV

FreeSurferSubject
~~~~~~~~~~~~~~~~~
Manages a full FreeSurfer subject directory and its derived data.

Key Methods:
- ``get_proc_status()``: Check the processing status of the subject
- ``launch_freesurfer()``: Run FreeSurfer processing
- ``set_freesurfer_directory()``: Point the object at a FreeSurfer directory
- ``get_cras()``: Extract the CRAS coordinates for the subject
- ``get_surface()`` / ``get_annotation()`` / ``get_vertexwise_map()``: Retrieve subject data
- ``create_stats_table()``: Build a stats table for the subject
- ``volume_morpho()``: Compute volume-based morphometry
- ``surface_hemi_morpho()``: Compute surface morphometry for one hemisphere
- ``annot2ind()`` / ``gcs2ind()``: Map atlas annotations to individual space
- ``surf2vol()``: Convert a surface atlas to a volumetric parcellation
- ``conform2native()``: Convert conformed space back to native space

Main Functions
--------------

Table Generation
~~~~~~~~~~~~~~~~
- ``create_individual_freesurfer_table()``: Build a morphometry table for one subject
- ``create_freesurfer_table()``: Build tables for many subjects in parallel
- ``process_subject()``: Process a single subject

Transforms and Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``parse_freesurfer_lta()``: Parse a FreeSurfer LTA transform file
- ``get_cras_coordinates()``: Extract CRAS coordinates from an LTA file

Colours and Labels
~~~~~~~~~~~~~~~~~~
- ``region_to_vertexwise()``: Expand regional values to vertexwise values
- ``create_vertex_colors()``: Build per-vertex colours from labels
- ``colors2colortable()``: Convert colours into a FreeSurfer colour table
- ``resolve_colortable_duplicates()``: Resolve duplicate colours in a colour table

Utilities
~~~~~~~~~
- ``create_fsaverage_links()`` / ``remove_fsaverage_links()``: Manage fsaverage links
- ``detect_hemi()``: Detect the hemisphere from a filename
- ``load_lobes_json()``: Load a lobe grouping definition
- ``get_version()``: Report the FreeSurfer version

Common Usage Examples
---------------------

Working with annotation files::

    from clabtoolkit.freesurfertools import AnnotParcellation

    # Load an annotation file
    annot = AnnotParcellation("/path/to/lh.aparc.a2009s.annot")

    # Summary information
    annot.get_info(verbose=True)

    # Correct the parcellation by filling unlabeled vertices
    annot.correct_vertexwise_parcellation(
        surf="/path/to/lh.pial",
        cortex_file="/path/to/lh.cortex.label"
    )

    # Region information
    region = annot.get_info_from_name("G_precentral")
    regions_df = annot.get_regions_info(return_df=True)

    # Convert to GCS format
    annot.annot2gcs(gcs_file="/path/to/lh.aparc.a2009s.gcs")

    # Group the regions into lobes
    annot.group_into_lobes(grouping="desikan", out_annot="/path/to/lh.lobes.annot")

Working with a subject::

    from clabtoolkit.freesurfertools import FreeSurferSubject

    # Point at a subject inside a FreeSurfer subjects directory
    subj = FreeSurferSubject("sub-01", subjs_dir="/path/to/freesurfer/subjects")

    # Check the processing status
    subj.get_proc_status()

    # Retrieve subject data
    surf = subj.get_surface(hemi="lh", surf_type="pial")
    annot = subj.get_annotation(hemi="lh", annot_type="aparc")

    # CRAS coordinates for this subject
    cras = subj.get_cras()

    # Build a stats table
    stats = subj.create_stats_table(
        lobes_grouping="desikan",
        output_file="/path/to/sub-01_stats.tsv"
    )

Morphometry tables::

    from clabtoolkit.freesurfertools import (
        create_individual_freesurfer_table,
        create_freesurfer_table,
    )

    # One subject
    table = create_individual_freesurfer_table(
        subj_id="sub-01",
        subjs_dir="/path/to/freesurfer/subjects",
        out_tab_file="/path/to/sub-01_morphometry.tsv"
    )

    # Many subjects, in parallel
    create_freesurfer_table(
        out_folder="/path/to/output",
        fs_subject_dir="/path/to/freesurfer/subjects",
        max_workers=4
    )

Transform and coordinate processing::

    from clabtoolkit.freesurfertools import parse_freesurfer_lta, get_cras_coordinates

    # Parse an LTA transform file
    lta_data = parse_freesurfer_lta("/path/to/transforms/talairach.lta")

    # Extract CRAS coordinates directly from an LTA file
    cras_coordinates = get_cras_coordinates(
        "/path/to/transforms/talairach.lta",
        source=True
    )
