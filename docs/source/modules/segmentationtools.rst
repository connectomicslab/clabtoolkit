segmentationtools module
=======================

.. automodule:: clabtoolkit.segmentationtools
   :members:
   :undoc-members:
   :show-inheritance:

The segmentationtools module provides atlas-based and automated image segmentation capabilities, with particular focus on brain parcellation using template registration.

Key Features
------------
- Atlas-based parcellation using ANTs registration
- Template-based segmentation workflows
- Multi-atlas segmentation support
- Registration parameter optimization
- Quality control for segmentation results
- Integration with multiple neuroimaging atlases

Main Functions
--------------

Atlas-based Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~
- ``abased_parcellation()``: Perform atlas-based parcellation using ANTs
- ``tissue_seg_table()``: Create tissue segmentation table from FreeSurfer data

Common Usage Examples
---------------------

Basic atlas-based segmentation::

    from clabtoolkit.segmentationtools import abased_parcellation
    
    # Perform atlas-based parcellation
    abased_parcellation(
        moving_image="sub-001_T1w.nii.gz",
        atlas_image="MNI152_T1_1mm.nii.gz", 
        atlas_labels="MNI152_parcellation.nii.gz",
        output_prefix="sub-001_space-MNI152",
        registration_type="SyN"
    )

Tissue segmentation analysis::

    # Create tissue segmentation summary table
    tissue_stats = tissue_seg_table(
        aseg_file="/path/to/aseg.mgz",
        subject_id="sub-001",
        session_id="ses-01"
    )
    
    print(tissue_stats[['region', 'volume_mm3']].head())