dicomtools module
=================

.. automodule:: clabtoolkit.dicomtools
   :members:
   :undoc-members:
   :show-inheritance:

The dicomtools module provides DICOM file organization and BIDS conversion capabilities with multi-threaded processing for efficient handling of large datasets.

Key Features
------------
- Multi-threaded DICOM file organization
- BIDS conversion workflows from DICOM to NIfTI
- Demographics data integration
- Session and acquisition management
- Batch processing of DICOM archives
- Quality control for DICOM conversion

Main Functions
--------------

DICOM Organization
~~~~~~~~~~~~~~~~~~
- ``org_conv_dicoms()``: Organize and convert DICOM files
- ``copy_dicom_file()``: Copy DICOM file
- ``create_session_series_names()``: Create session series names
- ``uncompress_dicom_session()``: Uncompress DICOM session
- ``compress_dicom_session()``: Compress DICOM session
- ``progress_indicator()``: Progress indicator for DICOM operations

Common Usage Examples
---------------------

Basic DICOM organization with multi-threaded DICOM processing::

    from clabtoolkit.dicomtools import org_conv_dicoms
    
    # Organize DICOM files into structured format
    org_conv_dicoms(
        in_dic_dir="/path/to/raw/dicoms",
        out_dic_dir="/path/to/organized/dicoms",
        nthreads=4,
        nosub=True
    )

Compressing the DICOM sessions::

    # Organize DICOMs and compress the sessions
    org_conv_dicoms(
        in_dic_dir="/path/to/raw/dicoms",
        out_dic_dir="/path/to/organized/dicoms",
        nthreads=4,
        boolcomp=True
    )

Session management::
    # Compress processed DICOM session
    compress_dicom_session(
        dic_dir="/path/to/bids_dataset",
    )