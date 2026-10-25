pipelinetools module
====================

The pipelinetools module helps to manage batch processing of BIDS datasets. It resolves which subjects to process, scans the derivatives folders of the pipelines, and reports which subjects have incomplete or mismatched outputs. The reports can be kept as tables, JSON files or a SQLite database.

Key Features
------------
- Build the list of subjects from a list, a comma-separated string, a text file or a folder
- Table with the number of files that each subject has in every pipeline of a derivatives folder
- File inventories of one or all the pipelines, built in parallel with progress bars
- Detection of subjects with missing or extra files, saved as JSON or SQLite
- Queries on the processing-status database

API reference
-------------

.. automodule:: clabtoolkit.pipelinetools
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

Subjects
~~~~~~~~
- ``get_fullids2process(ids=None, in_dir=None)``: list of subject IDs from a list, a string, a text file or the ``sub-*`` folders of ``in_dir``

Status tables and inventories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``create_processing_status_table(deriv_dir, subj_ids, output_table=None, n_jobs=-1)``: returns the status table and the path where it was saved
- ``scan_derivatives``, ``build_inventory``, ``build_derivatives_inventory``: inventory of the files of one or all the pipelines

Processing status details
~~~~~~~~~~~~~~~~~~~~~~~~~
- ``get_processing_status_details_json(proc_status_df, subj_ids, deriv_dir, ...)``: details of the subjects with incomplete or mismatched files, as a dictionary and optionally a JSON file
- ``get_processing_status_details_sqlite3(proc_status_df, subj_ids, deriv_dir, ..., db_path=...)``: same, also stored in a SQLite database
- ``query_processing_status_db(db_path, query_type=...)``: query the database; ``query_type`` can be ``"subjects_with_mismatches"``, ``"pipelines_with_mismatches"``, ``"missing_files_count"`` or ``"extra_files_count"``
- ``export_db_to_json(db_path, out_json)``: export the database to JSON

Common Usage Examples
---------------------

Selecting the subjects::

    import clabtoolkit.pipelinetools as cltpipe

    # From a derivatives folder (only the folders starting with "sub-")
    ids = cltpipe.get_fullids2process(None, in_dir="/path/to/derivatives/freesurfer")

    # From a string or a text file
    ids = cltpipe.get_fullids2process("sub-01, sub-02")
    ids = cltpipe.get_fullids2process("/path/to/subjects.txt")

Processing status of a derivatives folder::

    deriv_dir = "/path/to/derivatives"

    # One row per subject, one column per pipeline with the number of files
    status_df, saved_path = cltpipe.create_processing_status_table(
        deriv_dir, ids, output_table="/path/to/status.csv", n_jobs=4
    )

    # Inventory of the T1w/T2w/... files of all the pipelines
    inventory = cltpipe.build_derivatives_inventory(
        deriv_dir, output_csv="/path/to/inventory.csv", n_workers=8
    )

Finding incomplete subjects::

    # Subjects with a different number of files than expected
    details, _ = cltpipe.get_processing_status_details_sqlite3(
        status_df, ids, deriv_dir,
        out_json="/path/to/details.json", db_path="/path/to/status.db",
    )

    # Query the database
    mismatches = cltpipe.query_processing_status_db(
        "/path/to/status.db", query_type="subjects_with_mismatches"
    )

    # Export it again to JSON
    cltpipe.export_db_to_json("/path/to/status.db", "/path/to/status_export.json")
