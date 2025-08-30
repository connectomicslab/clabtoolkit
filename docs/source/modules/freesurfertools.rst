freesurfertools module
=====================

.. automodule:: clabtoolkit.freesurfertools
   :members:
   :undoc-members:
   :show-inheritance:

The freesurfertools module provides comprehensive integration with the FreeSurfer neuroimaging suite, enabling processing of cortical surfaces, annotation files, and morphometric data.

Key Features
------------
- Process FreeSurfer annotation files (.annot, .gcs formats)
- Correct and validate parcellations
- Parse FreeSurfer statistics files
- Surface-based morphometry computation
- Container technology integration
- Multi-format conversion capabilities

Main Classes
------------

AnnotParcellation
~~~~~~~~~~~~~~~~~
The primary class for working with FreeSurfer annotation files.

Key Methods:
- ``load_from_file()``: Load annotation files with optional reference surfaces
- ``correct_parcellation()``: Fix unlabeled vertices in parcellations  
- ``save_as_gcs()``: Convert annotation to GCS format
- ``get_region_info()``: Extract parcellation region information

Common Usage Examples
---------------------

Working with annotation files::

    from clabtoolkit.freesurfertools import AnnotParcellation
    
    # Load annotation file
    annot = AnnotParcellation("lh.aparc.a2009s.annot")
    
    # Correct parcellation by filling unlabeled vertices
    annot.correct_parcellation()
    
    # Convert to GCS format
    annot.save_as_gcs("lh.aparc.a2009s.gcs")
    
    # Get region information
    region_info = annot.get_region_info()

Surface morphometry processing::

    # Compute morphometry table for FreeSurfer subjects directory
    from clabtoolkit import freesurfertools as fs
    
    # Process all subjects in FreeSurfer subjects directory
    morphometry_table = fs.compute_morphometry_table(
        subjects_dir="/path/to/freesurfer/subjects",
        parcellation="aparc.a2009s"
    )