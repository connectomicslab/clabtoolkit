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

Basic DICOM organization::

    from clabtoolkit.dicomtools import org_conv_dicoms
    
    # Organize DICOM files into structured format
    org_conv_dicoms(
        dicom_dir="/path/to/raw/dicoms",
        output_dir="/path/to/organized/dicoms",
        n_threads=4,
        include_demographics=True
    )

Multi-threaded DICOM processing::

    # Organize DICOMs with demographics integration
    org_conv_dicoms(
        dicom_dir="/path/to/raw/dicoms",
        output_dir="/path/to/organized",
        n_threads=4,
        uncompress_files=True
    )

Batch processing with demographics::

    # Process multiple DICOM archives with demographics integration
    archives = [
        "/path/to/archive1.zip",
        "/path/to/archive2.tar.gz", 
        "/path/to/archive3/"
    ]
    
    for archive in archives:
        org_conv_dicoms(
            dicom_dir=archive,
            output_dir="/path/to/organized",
            demographics_file="/path/to/demographics.csv",
            validate_series=True,
            n_threads=6
        )

Session management::

    # Create session and series names
    session_names = create_session_series_names(
        dicom_dir="/path/to/dicoms",
        naming_convention="date_time"
    )
    
    # Compress processed DICOM session
    compress_dicom_session(
        session_dir="/path/to/session",
        output_archive="compressed_session.tar.gz"
    )