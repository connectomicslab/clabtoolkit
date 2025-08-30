bidstools module
================

.. automodule:: clabtoolkit.bidstools
   :members:
   :undoc-members:
   :show-inheritance:

The bidstools module provides comprehensive support for Brain Imaging Data Structure (BIDS) datasets. This module enables you to work with BIDS naming conventions, manipulate entities, organize datasets, and generate database tables from BIDS structures.

Key Features
------------
- Convert between BIDS filename strings and entity dictionaries
- Manipulate BIDS entities (add, remove, replace)
- Extract subject lists and dataset summaries
- Copy and organize BIDS folders with filtering
- Generate comprehensive database tables from BIDS datasets
- Validate BIDS compliance

Common Usage Examples
---------------------

Basic entity manipulation::

    import clabtoolkit.bidstools as bids
    
    # Parse BIDS filename
    entities = bids.str2entity("sub-01_ses-M00_T1w.nii.gz")
    # Returns: {'sub': '01', 'ses': 'M00', 'suffix': 'T1w', 'extension': 'nii.gz'}
    
    # Convert back to filename
    filename = bids.entity2str(entities)
    
    # Modify entities
    new_filename = bids.replace_entity_value(filename, 'ses', 'M12')

Dataset management::

    # Get all subjects in a BIDS dataset
    subjects = bids.get_subjects("/path/to/bids/dataset")
    
    # Generate dataset overview table
    database = bids.get_bids_database_table("/path/to/bids/dataset")
    
    # Copy subset of BIDS dataset
    bids.copy_bids_folder(
        source_dir="/path/to/source",
        target_dir="/path/to/target",
        subjects=['sub-01', 'sub-02']
    )