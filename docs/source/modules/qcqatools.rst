qcqatools module
================

The qcqatools module supports the visual quality control of neuroimaging data. It renders sagittal, coronal and axial slices of images as PNG files, for one image or a whole dataset, and builds an HTML page where the images can be reviewed and selected.

Key Features
------------
- Selection of slices inside the area of the image that has signal
- Composite PNG with three views, consistent intensity scaling and an optional colorbar
- Batch processing of folders with filters, parallel jobs and progress bars
- Interactive HTML page with the images grouped by folder, descriptions from JSON files, checkboxes and a downloadable list of the selected files
- Works with NIfTI, MGZ and MINC images

API reference
-------------

.. automodule:: clabtoolkit.qcqatools
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------
- ``get_valid_slices(data, ignore_value=None, slice_positions=None)``: slice indices (sagittal, coronal, axial) at relative positions of the non-background extent
- ``generate_slices(nifti_path, output_path=None, ...)``: PNG with the slices of one image
- ``recursively_generate_slices(input_folder, output_folder=None, and_filter=None, or_filter=None, n_jobs=1, ...)``: PNGs for all the images found in a folder
- ``generate_image_selection_webpage(root_directory, output_html=None, ...)``: HTML page with all the PNG files of a folder
- ``create_png_webpage_from_generated_slices(root_directory, output_html=None, ...)``: shortcut for the PNGs created by the functions above

Common Usage Examples
---------------------

Slices of one image::

    import nibabel as nib
    from clabtoolkit.qcqatools import get_valid_slices, generate_slices

    # Slice indices at 30%, 50% and 70% of the extent of the brain
    data = nib.load("/path/to/T1w.nii.gz").get_fdata()
    sagittal, coronal, axial = get_valid_slices(data, slice_positions=[0.3, 0.5, 0.7])

    # PNG with the three views (saved next to the image if output_path is None)
    generate_slices("/path/to/T1w.nii.gz", output_path="/path/to/T1w.png",
                    slice_positions=[0.3, 0.5, 0.7], dpi=150)

Slices of a whole dataset::

    from clabtoolkit.qcqatools import recursively_generate_slices

    png_files = recursively_generate_slices(
        "/path/to/bids/dataset",
        output_folder="/path/to/qc_output",
        or_filter="T1w",   # only images whose name contains "T1w"
        n_jobs=4,
    )

Review page::

    from clabtoolkit.qcqatools import generate_image_selection_webpage

    generate_image_selection_webpage(
        "/path/to/qc_output",
        output_html="/path/to/qc_report.html",
        title="Quality Control Report",
        overwrite=True,
    )

    # Or, for the PNG files generated next to the images
    # (writes png_selection.html in the root folder)
    # create_png_webpage_from_generated_slices("/path/to/bids/dataset")
