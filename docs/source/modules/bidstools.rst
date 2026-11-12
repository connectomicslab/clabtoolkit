bidstools module
================

The bidstools module provides support for Brain Imaging Data Structure (BIDS) datasets. It lets you work with BIDS naming conventions, edit entities in single filenames or across a whole dataset, copy and inspect datasets, remove subjects, sessions or acquisitions from them, and generate database tables from BIDS structures.

Key Features
------------
- Convert between BIDS filename strings and entity dictionaries
- Delete, insert and replace entity keys and values, in one filename or recursively in a whole dataset
- Check that a filename follows the BIDS convention
- List subjects, derivatives folders and the entities used in a dataset
- Copy a subset of a BIDS dataset (with its derivatives) to a new location
- Remove subjects, sessions or acquisitions from a dataset and its derivatives
- Build a database table of all the neuroimaging files of a dataset
- Draw the folder tree of a dataset and run a basic structure validation

API reference
-------------

.. automodule:: clabtoolkit.bidstools
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

Entities in a filename
~~~~~~~~~~~~~~~~~~~~~~
- ``str2entity`` / ``entity2str``: convert between a filename and an entity dictionary
- ``delete_entity``, ``insert_entity``, ``replace_entity_value``, ``replace_entity_key``: edit entities of a filename or dictionary
- ``is_bids_filename``: validate the structure of a filename
- ``entities_to_table``, ``entities4table``: extract entities into a ``pandas.DataFrame``

Entities in a dataset
~~~~~~~~~~~~~~~~~~~~~
- ``recursively_delete_entity``, ``recursively_insert_entity``, ``recursively_replace_entity_value``, ``recursively_replace_entity_key``: rename all the files and folders of a dataset
- ``get_all_entities``: set of all the entities found in a dataset

Dataset management
~~~~~~~~~~~~~~~~~~
- ``get_subjects``, ``get_derivatives_folders``, ``get_individual_files_and_folders``: find subjects, derivatives, and files or folders matching a string
- ``copy_bids_folder``: copy selected subjects and derivatives to a new location
- ``remove_bids_folder``: remove subjects, sessions or acquisitions from the derivatives and, optionally, from the raw data
- ``get_bids_database_table``: table with all the files of a dataset
- ``generate_bids_tree``, ``generate_bids_tree_with_stats``: tree view of the dataset
- ``validate_bids_structure``: basic validation that returns warnings
- ``load_bids_json``: load the BIDS configuration file

Common Usage Examples
---------------------

Basic entity manipulation::

    import clabtoolkit.bidstools as bids

    # Parse a BIDS filename
    entities = bids.str2entity("sub-01_ses-M00_T1w.nii.gz")
    # {'sub': '01', 'ses': 'M00', 'suffix': 'T1w', 'extension': 'nii.gz'}

    # Convert it back to a filename
    filename = bids.entity2str(entities)

    # Modify entities (they take a dictionary, not a key and a value)
    bids.replace_entity_value(filename, {"ses": "M12"})  # sub-01_ses-M12_T1w.nii.gz
    bids.delete_entity(filename, "ses")                  # sub-01_T1w.nii.gz
    bids.insert_entity(filename, {"run": "01"}, prev_entity="ses")  # sub-01_ses-M00_run-01_T1w.nii.gz
    bids.replace_entity_key(filename, {"ses": "session"})           # sub-01_session-M00_T1w.nii.gz

    bids.is_bids_filename(filename)  # True

Dataset management::

    # Get all subjects in a BIDS dataset
    subjects = bids.get_subjects("/path/to/bids/dataset")

    # Generate a table with all the files of the dataset
    database = bids.get_bids_database_table("/path/to/bids/dataset", n_jobs=4)

    # Copy a subset of the dataset, including a derivatives folder
    bids.copy_bids_folder(
        bids_dir="/path/to/source",
        out_dir="/path/to/target",
        subjects_to_copy=["sub-01", "sub-02"],
        deriv_dir="/path/to/source/derivatives",
        include_derivatives="freesurfer",
    )

    # Show the folder tree (up to 3 levels) and check the structure
    bids.generate_bids_tree("/path/to/bids/dataset", max_depth=3)
    warnings = bids.validate_bids_structure("/path/to/bids/dataset")

Removing subjects, sessions or acquisitions::

    # An identifier is a subject ("sub-01"), a session ("sub-01_ses-M00") or a longer
    # combination of entities ("sub-01_ses-M00_acq-3T_run-01"). The files and folders
    # removed are the ones that have all its entities, in their own name or in the
    # sub-* and ses-* folders that contain them.

    # See what would be removed, without removing anything
    paths = bids.remove_bids_folder(
        bids_dir="/path/to/bids/dataset",
        full_ids=["sub-01_ses-M00_acq-3T_run-01", "sub-02"],
        deriv_dir="/path/to/bids/dataset/derivatives",
        deriv_folders="freesurfer",
        simulate=True,
    )

    # Remove the identifiers of a text file (one per line) from all the derivatives.
    # The raw data are kept unless only_derivatives=False
    bids.remove_bids_folder(
        bids_dir="/path/to/bids/dataset",
        full_ids="/path/to/ids_to_remove.txt",
        deriv_dir="/path/to/bids/dataset/derivatives",
        deriv_folders="all",
        n_jobs=8,
    )

.. warning::

   ``remove_bids_folder`` deletes files and folders and it cannot be undone. Run it first
   with ``simulate=True`` and check the list of paths that it returns.

Both layouts of the derivatives are handled: ``<pipeline>/sub-01/ses-M00/...`` and the folders
named with the identifier directly inside the pipeline, as FreeSurfer does
(``freesurfer/sub-01_ses-M00_acq-3T_run-01``). The folders left empty are removed, as well as the
folders of the subject or session that were already empty, and symbolic links are not followed.
