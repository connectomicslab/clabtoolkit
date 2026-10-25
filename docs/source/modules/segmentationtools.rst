segmentationtools module
========================

The segmentationtools module provides atlas-based parcellation with ANTs and the standard tissue segmentation table.

Key Features
------------
- Atlas-based parcellation: the T1-weighted image is registered to a template and the atlas is brought back to the subject space
- Atlases given as probability maps (SPAM) or as discrete labels (maxprob)
- ANTs can run locally or inside a Singularity or Docker container
- Standard lookup table of the tissue classes (CSF, GM and WM)

API reference
-------------

.. automodule:: clabtoolkit.segmentationtools
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------
- ``abased_parcellation(t1, t1_temp, atlas, out_parc, xfm_output, atlas_type="spam", interp="Linear", cont_tech="local", cont_image=None, overwrite=False)``: parcellation in the subject space
- ``tissue_seg_table(tsv_filename=None, overwrite=True)``: ``pandas.DataFrame`` with the columns ``index``, ``name``, ``abbreviation`` and ``color``

Common Usage Examples
---------------------

Atlas-based parcellation (requires ANTs, or a container with it)::

    from clabtoolkit.segmentationtools import abased_parcellation

    parc_file = abased_parcellation(
        t1="sub-001_T1w.nii.gz",
        t1_temp="MNI152_T1_1mm.nii.gz",
        atlas="atlas_probabilities.nii.gz",
        out_parc="sub-001_space-T1w_dseg.nii.gz",
        xfm_output="sub-001_from-MNI152_to-T1w",
        atlas_type="spam",
    )

    # Run ANTs inside a container
    # abased_parcellation(..., cont_tech="singularity", cont_image="/path/to/ants.sif")

Tissue segmentation table::

    from clabtoolkit.segmentationtools import tissue_seg_table

    seg_df = tissue_seg_table()                 # table only
    seg_df = tissue_seg_table("tissues.tsv")    # table and TSV file
    print(seg_df)
    #    index                  name abbreviation    color
    # 0      1  cerebro_spinal_fluid          CSF  #ac0000
    # 1      2           gray_matter           GM  #00994c
    # 2      3          white_matter           WM  #0066cc
