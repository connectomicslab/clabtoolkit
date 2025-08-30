pipelinetools module
===================

.. automodule:: clabtoolkit.pipelinetools
   :members:
   :undoc-members:
   :show-inheritance:

The pipelinetools module provides workflow orchestration and batch processing capabilities for large-scale neuroimaging analysis pipelines.

Key Features
------------
- Subject ID management for batch processing
- Parallel processing utilities with progress tracking
- Pipeline workflow orchestration
- Error handling and recovery mechanisms
- Integration with BIDS datasets
- Resource management and optimization

Main Functions
--------------

Batch Processing
~~~~~~~~~~~~~~~~
- ``get_ids2process()``: Generate subject IDs for batch processing workflows
- ``create_processing_status_table()``: Create table to track processing status
- ``get_processing_status_details_json()``: Get processing status details in JSON format
- ``get_processing_status_details_sqlite3()``: Get processing status details from SQLite database
- ``query_processing_status_db()``: Query processing status database
- ``export_db_to_json()``: Export database contents to JSON format

Common Usage Examples
---------------------

Batch subject processing::

    from clabtoolkit.pipelinetools import get_ids2process
    from concurrent.futures import ThreadPoolExecutor
    
    # Get subjects to process from BIDS dataset
    subjects_to_process = get_ids2process(
        bids_dir="/path/to/bids/dataset",
        exclude_processed="/path/to/derivatives",
        pattern="sub-*"
    )
    
    print(f"Processing {len(subjects_to_process)} subjects")

Parallel processing pipeline::

    def process_single_subject(subject_id):
        """Process a single subject through the analysis pipeline"""
        # Your processing logic here
        return f"Processed {subject_id}"
    
    # Execute parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_single_subject, subjects_to_process))

Processing status tracking::

    # Create processing status tracking table
    status_table = create_processing_status_table(
        subjects_list=subjects_to_process,
        processing_steps=["preproc", "morphometry", "qc"]
    )
    
    # Query processing status from database
    status_results = query_processing_status_db(
        database_path="/path/to/processing_status.db",
        query="SELECT * FROM status WHERE step='morphometry'"
    )

Data export and management::

    # Export processing status to JSON
    export_db_to_json(
        database_path="/path/to/processing_status.db",
        output_json="/path/to/status_export.json"
    )
    
    # Get detailed processing status
    details = get_processing_status_details_json(
        json_file="/path/to/status.json",
        subject_filter=["sub-001", "sub-002"]
    )