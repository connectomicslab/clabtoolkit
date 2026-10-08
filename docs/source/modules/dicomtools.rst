dicomtools module
=================

.. automodule:: clabtoolkit.dicomtools
   :members:
   :undoc-members:
   :show-inheritance:

The dicomtools module organizes raw DICOM files into a
``<subject>/<session>/<series>`` folder hierarchy, compresses and uncompresses the
session folders, and reads metadata from DICOM files. It does not convert DICOMs
to NIfTI; use a converter such as dcm2niix on the organized folders.

Key Features
------------
- Multi-threaded organization of DICOM files into subject, session and series folders
- Subjects taken from the input folders (``sub-*``) or from the ``PatientID`` tag
- Session and series names built from the DICOM headers
- Visit IDs added to the session names from a demographics table
- Compression of each session into a ``tar.gz`` archive, and extraction
- DICOM metadata extraction

Session and series names
------------------------
- Session: ``ses-`` followed by StudyDate and StudyTime as ``YYYYMMDDHHMMSS``
  (e.g. ``ses-20240312091530``). Missing values are replaced by zeros.
- Series: the zero-padded SeriesNumber followed by the SeriesDescription (or
  SequenceName, ProtocolName), without special characters and with ``-`` as
  separator (e.g. ``0001-T1w-MPRAGE``).
- With a demographics table, the visit ID of the row with the closest acquisition
  date is appended to the session name without a separator, so the label stays
  BIDS-valid (e.g. ``ses-20240312091530V1``).
- With ``ses_id``, every session of a subject is named ``ses-<ses_id>``.

Main Functions
--------------

DICOM Organization
~~~~~~~~~~~~~~~~~~
- ``org_conv_dicoms()``: Organize the DICOMs of each ``sub-*`` folder. Supports a
  demographics table (``participant_id``, ``session_id`` and ``acq_date`` as
  ``MM/DD/YYYY`` or ``YYYY-MM-DD``), a subject selection (``ids_file``: a text file
  or a comma-separated list, with or without ``sub-``), a fixed session label
  (``ses_id``), compression (``boolcomp``) and several threads (``nthreads``).
- ``org_dicom_folder()``: Organize a folder with the DICOMs of any number of
  subjects, naming each subject after its ``PatientID``. Non-DICOM files are
  reported and skipped.
- ``organize_dicom_files()``: Calls ``org_conv_dicoms()``, or ``org_dicom_folder()``
  when ``no_sub_folder=True``.
- ``copy_dicom_file()``: Copy one DICOM file into its session and series folder.
  Returns the destination folder, or None for files that are not DICOMs.
- ``create_session_series_names()``: Build the session and series names of a
  DICOM dataset.

Sessions
~~~~~~~~
- ``compress_dicom_session()``: Compress each session folder into a ``tar.gz`` archive
- ``uncompress_dicom_session()``: Extract the session archives

Auxiliary
~~~~~~~~~
- ``get_dicom_info()``: Extract all the tags of a DICOM file, or only some of them
- ``progress_indicator()``: Progress bar callback used by the threaded organization

Common Usage Examples
---------------------

Organizing the DICOMs of each subject folder::

    from clabtoolkit.dicomtools import org_conv_dicoms

    # /path/to/raw/dicoms/sub-01, sub-02, ... -> <subject>/<session>/<series>
    org_conv_dicoms(
        in_dic_dir="/path/to/raw/dicoms",
        out_dic_dir="/path/to/organized/dicoms",
        nthreads=4,
    )

    # Add the visit IDs from a demographics table and process two subjects only
    org_conv_dicoms(
        in_dic_dir="/path/to/raw/dicoms",
        out_dic_dir="/path/to/organized/dicoms",
        demog_file="/path/to/demographics.csv",
        ids_file="01,02",
    )

    # Organize the DICOMs and compress the sessions
    org_conv_dicoms(
        in_dic_dir="/path/to/raw/dicoms",
        out_dic_dir="/path/to/organized/dicoms",
        boolcomp=True,
    )

Organizing a folder with the DICOMs of several subjects::

    from clabtoolkit.dicomtools import org_dicom_folder

    # The subject folders are named after the PatientID tag
    org_dicom_folder("/path/to/scanner/export", "/path/to/organized/dicoms", nthreads=8)

Compressing and uncompressing sessions::

    from clabtoolkit.dicomtools import compress_dicom_session, uncompress_dicom_session

    failed = compress_dicom_session("/path/to/organized/dicoms")
    failed = uncompress_dicom_session("/path/to/organized/dicoms", boolrmtar=True)

Reading DICOM metadata::

    from clabtoolkit.dicomtools import get_dicom_info

    info = get_dicom_info("/path/to/file.dcm", tags=["PatientID", "StudyDate", "SeriesDescription"])
