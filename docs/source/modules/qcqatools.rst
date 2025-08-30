qcqatools module
================

.. automodule:: clabtoolkit.qcqatools
   :members:
   :undoc-members:
   :show-inheritance:

The qcqatools module provides comprehensive quality control and quality assessment tools for neuroimaging data, enabling automated validation and visual inspection workflows.

Key Features
------------
- Automated quality control metrics computation
- Image artifact detection and reporting
- Optimal slice selection for visual inspection
- Multi-modal data validation
- Quality report generation
- Visual assessment tools

Main Functions
--------------

Quality Assessment
~~~~~~~~~~~~~~~~~~
- ``get_valid_slices()``: Identify optimal slice positions for visualization
- ``generate_slices()``: Generate image slices for quality control
- ``recursively_generate_slices()``: Recursively generate slices for multiple files
- ``generate_image_selection_webpage()``: Generate webpage for image selection
- ``create_png_webpage_from_generated_slices()``: Create PNG webpage from generated slices

Common Usage Examples
---------------------

Automated slice selection::

    from clabtoolkit.qcqatools import get_valid_slices
    
    # Find optimal slices for quality control visualization
    optimal_slices = get_valid_slices(
        image_file="/path/to/T1w.nii.gz",
        n_slices=5,
        plane='axial'
    )
    
    print(f"Recommended slices: {optimal_slices}")

Generate quality control slices::

    # Generate slices for quality control
    generate_slices(
        image_file="/path/to/T1w.nii.gz",
        output_dir="/path/to/qc_slices",
        n_slices=5,
        plane='axial'
    )

Create QC webpage::

    # Generate interactive QC webpage
    generate_image_selection_webpage(
        image_dir="/path/to/qc_slices",
        output_html="/path/to/qc_report.html",
        title="Quality Control Report"
    )
    
    # Process multiple files recursively
    recursively_generate_slices(
        input_dir="/path/to/bids/dataset",
        output_dir="/path/to/qc_output",
        file_pattern="*T1w.nii.gz"
    )