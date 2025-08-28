import os
import shutil
import pandas as pd
import time
import queue
import threading
import numpy as np

from typing import Union, Dict, List, Optional

import re
import json
from glob import glob

from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    TextColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)
from rich.console import Console
from rich.panel import Panel

# Importing the clabtoolkit modules
from . import misctools as cltmisc
from . import bidstools as cltbids


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############                     Section 1: Utility methods                             ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def get_ids2process(
    ids: Union[str, List[str], None] = None, in_dir: str = None
) -> List[str]:
    """
    Get list of subject IDs to process from various input sources.

    Parameters
    ----------
    ids : str, list of str, or None, optional
        Subject IDs specification. Can be:
        - None: discover all subjects in `in_dir` (default)
        - list: list of subject ID strings
        - str: comma-separated IDs, single ID, or path to text file

    in_dir : str, optional
        Directory path to scan for subjects when `ids` is None.
        Only used when `ids` is None.

    Returns
    -------
    list of str
        List of subject ID strings, with empty entries filtered out.

    Raises
    ------
    ValueError
        If `ids` is not None/list/str, or if `in_dir` is invalid when `ids` is None.
    FileNotFoundError
        If specified file path in `ids` does not exist.
    IOError
        If file cannot be read due to permissions or other IO issues.

    Examples
    --------
    >>> # Discover subjects from directory
    >>> get_ids2process(ids=None, in_dir='/data/subjects')
    ['sub-001', 'sub-002', 'sub-003']

    >>> # From list
    >>> get_ids2process(['sub-001', 'sub-002'])
    ['sub-001', 'sub-002']

    >>> # From comma-separated string
    >>> get_ids2process('sub-001, sub-002, sub-003')
    ['sub-001', 'sub-002', 'sub-003']

    >>> # Single subject ID
    >>> get_ids2process('sub-001')
    ['sub-001']

    >>> # From text file
    >>> get_ids2process('/path/to/subjects.txt')
    ['sub-001', 'sub-002', 'sub-003']

    Notes
    -----
    When scanning directories (ids=None), only directories starting with 'sub-'
    are considered valid subject directories.

    Text files should contain one subject ID per line. Empty lines and
    whitespace are automatically filtered out.
    """
    # Handle None case - discover from directory
    if ids is None:
        if not in_dir or not os.path.isdir(in_dir):
            raise ValueError(f"Valid in_dir required when ids is None. Got: {in_dir}")
        return [d for d in os.listdir(in_dir) if d.startswith("sub-")]

    # Handle list case
    if isinstance(ids, list):
        return [str(id_).strip() for id_ in ids if str(id_).strip()]

    # Handle string case
    if isinstance(ids, str):
        ids = ids.strip()
        if not ids:
            return []

        # File path
        if os.path.isfile(ids):
            with open(ids, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]

        # Comma-separated or single ID
        return [id_.strip() for id_ in ids.split(",") if id_.strip()]

    raise ValueError(f"ids must be None, list, or string, got {type(ids)}")


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############  Section 2: Methods for assessing the processing status of the pipelines   ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def create_processing_status_table(
    deriv_dir: str,
    subj_ids: Union[list, str],
    output_table: str = None,
    n_jobs: int = -1,
):
    """
    This method creates a table with the processing status of the subjects in the BIDs derivatives directory.
    Uses parallel processing for improved performance with rich progress visualization.

    Parameters
    ----------
    deriv_dir : str
        Path to the derivatives directory.

    subj_ids : list or str
        List of subject IDs or a text file containing the subject IDs.

    output_table : str, optional
        Path to save the resulting table. If None, the table is not saved.

    n_jobs : int, optional
        Number of parallel jobs to run. Default is -1 which uses all available cores.


    Returns
    -------
    pd.DataFrame
        DataFrame containing the processing status of the subjects.

    str
        Path to the saved table if output_table is provided, otherwise None.

    Raises
    ------
    FileNotFoundError
        If the derivatives directory or the subject IDs file does not exist.
    ValueError
        If no derivatives folders are found or if the subject IDs list is empty.

    TypeError
        If subj_ids is not a list or a string path to a file.

    Examples
    --------
    >>> deriv_dir = "/path/to/derivatives"
    >>> subj_ids = ["sub-01", "sub-02"]
    >>> output_table = "/path/to/output_table.csv"
    >>> df, saved_path = create_processing_status_table(deriv_dir, subj_ids, output_table)
    >>> print(df)
    """

    from joblib import Parallel, delayed
    from . import morphometrytools as cltmorpho

    # Initialize rich console
    console = Console()

    # Check if the derivatives directory exists
    deriv_dir = cltmisc.remove_trailing_separators(deriv_dir)

    if not os.path.isdir(deriv_dir):
        raise FileNotFoundError(
            f"The derivatives directory {deriv_dir} does not exist."
        )

    # Process subject IDs
    if isinstance(subj_ids, str):
        if not os.path.isfile(subj_ids):
            raise FileNotFoundError(f"The file {subj_ids} does not exist.")
        else:
            with open(subj_ids, "r") as f:
                subj_list = f.read().splitlines()
    elif isinstance(subj_ids, list):
        if len(subj_ids) == 0:
            raise ValueError("The list of subject IDs is empty.")
        else:
            subj_list = subj_ids
    else:
        raise TypeError("subj_ids must be a list or a string path to a file")

    # Number of Ids
    n_subj = len(subj_list)

    # Find all the derivatives folders
    pipe_dirs = cltbids.get_derivatives_folders(deriv_dir)

    if len(pipe_dirs) == 0:
        raise ValueError(
            "No derivatives folders were found in the specified directory."
        )

    # Create a message queue to communicate across threads
    progress_queue = queue.Queue()

    # Function to process a single subject
    def process_subject(full_id):
        try:
            # Parse the subject ID
            id_dict = cltbids.str2entity(full_id)
            subject = id_dict["sub"]

            # Get entity information for this subject
            ent_list = cltbids.entities4table(selected_entities=full_id)
            df_add = cltbids.entities_to_table(
                filepath=full_id, entities_to_extract=ent_list
            )

            # Create a new DataFrame for this subject's processing status
            proc_table = pd.DataFrame(
                columns=pipe_dirs, index=[0]
            )  # Single row for this subject

            # Remove suffix and extension from entities
            clean_id_dict = id_dict.copy()
            if "suffix" in clean_id_dict:
                del clean_id_dict["suffix"]
            if "extension" in clean_id_dict:
                del clean_id_dict["extension"]

            # Create list of entity key-value pairs
            subj_ent = [f"{k}-{v}" for k, v in clean_id_dict.items()]

            # Process each derivatives directory
            for tmp_pipe_deriv in pipe_dirs:
                # Find subject's directory in this pipeline
                ind_der_dir = glob(
                    os.path.join(
                        deriv_dir, tmp_pipe_deriv, "sub-" + clean_id_dict["sub"] + "*"
                    )
                )

                # Filter if multiple directories found
                if len(ind_der_dir) > 1:
                    ind_der_dir = cltmisc.filter_by_substring(
                        ind_der_dir,
                        or_filter=[clean_id_dict["sub"]],
                        and_filter=subj_ent,
                    )

                # Set count to 0 if no directory found
                if len(ind_der_dir) == 0:
                    proc_table.at[0, tmp_pipe_deriv] = 0
                    continue

                # Count files for this subject in this pipeline
                all_pip_files = cltmisc.get_all_files(ind_der_dir[0])
                subj_pipe_files = cltmisc.filter_by_substring(
                    all_pip_files, or_filter=clean_id_dict["sub"], and_filter=subj_ent
                )
                n_files = len(subj_pipe_files)

                # Store the count
                proc_table.at[0, tmp_pipe_deriv] = n_files

            # Combine the entity info with processing counts
            subj_proc_table = cltmisc.expand_and_concatenate(df_add, proc_table)

            # Signal completion through the queue
            progress_queue.put((True, full_id))

            return subj_proc_table
        except Exception as e:
            # Signal error through the queue
            progress_queue.put((False, f"{full_id}: {str(e)}"))
            raise e

    # Use Rich for progress tracking
    all_results = []
    stop_monitor = threading.Event()

    # Start a separate thread for the progress bar
    def progress_monitor(progress_queue, total_subjects, progress_task, stop_event):
        completed = 0
        errors = 0

        while (completed + errors < total_subjects) and not stop_event.is_set():
            try:
                success, message = progress_queue.get(timeout=0.1)

                if success:
                    completed += 1
                else:
                    errors += 1
                    console.print(f"[bold red]Error: {message}[/bold red]")

                # Update progress bar
                progress.update(
                    progress_task,
                    completed=completed,
                    description=f"[yellow]Processing subjects - {completed}/{total_subjects} completed",
                )

                # Important: mark the task as done in the queue
                progress_queue.task_done()

            except queue.Empty:
                # No updates, just continue waiting
                pass

        # Ensure the progress bar shows 100% completion
        if not stop_event.is_set():
            progress.update(progress_task, completed=total_subjects)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=50),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeRemainingColumn(),
        MofNCompleteColumn(),
        console=console,
        refresh_per_second=10,  # Increase refresh rate
    ) as progress:
        # Add main task to track progress
        main_task = progress.add_task("[yellow]Processing subjects", total=n_subj)

        # Start progress monitor thread
        monitor_thread = threading.Thread(
            target=progress_monitor,
            args=(progress_queue, n_subj, main_task, stop_monitor),
            daemon=True,
        )
        monitor_thread.start()

        try:
            # Process subjects in parallel with joblib
            results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
                delayed(process_subject)(subject) for subject in subj_list
            )

            # Allow some time for the queue to process final updates
            time.sleep(0.5)

            # Directly set progress to 100% after all processing is done
            progress.update(main_task, completed=n_subj, refresh=True)

            # Wait for the progress queue to be empty
            progress_queue.join()

        finally:
            # Signal the monitor thread to stop
            stop_monitor.set()

            # Make absolutely sure progress shows completion
            progress.update(main_task, completed=n_subj, refresh=True)

    # Combine all results
    proc_status_df = pd.concat(results, ignore_index=True)

    # Save table if requested
    if output_table is not None:
        output_dir = os.path.dirname(output_table)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        proc_status_df.to_csv(output_table, sep="\t", index=False)

    return proc_status_df, output_table


####################################################################################################
def get_processing_status_details_json(
    proc_status_df: Union[str, dict],
    subj_ids: Union[List[str], str],
    deriv_dir: str,
    pipe_dirs: Union[List[str], str] = None,
    out_json: str = None,
    only_ids: bool = False,
):
    """
    This function creates a dictionary with the details of the processing status of the subjects in the BIDs derivatives directory.
    It provides the IDs of the subjects with incomplete or mismatched number of files.

    Parameters
    ----------
    proc_status_df : str or dict
        Path to the processing status DataFrame or a DataFrame itself. This DataFrame can be
        obtained with the function "create_processing_status_table".

    subj_ids : list or str
        List of subject IDs or a text file containing the subject IDs.

    deriv_dir : str
        Path to the derivatives directory.

    pipe_dirs : list or str, optional
        List of processing pipelines to check. If None, all pipelines will be checked.

    out_json : str, optional
        Path to save the output JSON file. If None, the JSON file will not be saved.

    only_ids : bool, optional
        If True, only the IDs of the subjects with mismatches will be returned, without the file details.

    Returns
    -------
    dict
        Dictionary containing the details of the processing status of the subjects.
    str
        Path to the saved JSON file if out_json is provided, otherwise None.
    """

    from . import morphometrytools as cltmorpho
    import os
    import numpy as np

    if isinstance(proc_status_df, str):
        if not os.path.isfile(proc_status_df):
            raise FileNotFoundError(f"The file {proc_status_df} does not exist.")
        else:
            proc_status_df = cltmisc.smart_read_table(proc_status_df)
    elif not isinstance(proc_status_df, pd.DataFrame):
        raise TypeError("proc_status_df must be a DataFrame or a string path to a file")

    # Process subject IDs
    if isinstance(subj_ids, str):
        if not os.path.isfile(subj_ids):
            raise FileNotFoundError(f"The file {subj_ids} does not exist.")
        else:
            with open(subj_ids, "r") as f:
                subj_list = f.read().splitlines()
    elif isinstance(subj_ids, list):
        if len(subj_ids) == 0:
            raise ValueError("The list of subject IDs is empty.")
        else:
            subj_list = subj_ids
    else:
        raise TypeError("subj_ids must be a list or a string path to a file")

    # Check if the derivatives directory exists
    deriv_dir = cltmisc.remove_trailing_separators(deriv_dir)

    if not os.path.isdir(deriv_dir):
        raise FileNotFoundError(
            f"The derivatives directory {deriv_dir} does not exist."
        )

    # Find all the derivatives folders
    all_pipe_dirs = cltbids.get_derivatives_folders(deriv_dir)

    if len(all_pipe_dirs) == 0:
        raise ValueError(
            "No derivatives folders were found in the specified directory."
        )

    if pipe_dirs is not None:
        if isinstance(pipe_dirs, str):
            pipe_dirs = [pipe_dirs]

        pipe_dirs = cltmisc.filter_by_substring(all_pipe_dirs, or_filter=pipe_dirs)
    else:
        pipe_dirs = all_pipe_dirs

    # All entities
    ent_list = cltbids.entities4table()

    # Get all the columns names
    col_names = proc_status_df.columns.tolist()

    # Get all the columns that are not in the pipe_dirs
    subj_columns = list(set(col_names) - set(pipe_dirs))

    subj_ids_df = proc_status_df[subj_columns]

    # Create a consistent structure for the output dictionary
    missmatch_summary = {}

    # Process each pipeline
    for i in pipe_dirs:
        proc_status_df[i] = proc_status_df[i].astype(int)
        pipe_dir_fold = os.path.join(deriv_dir, i)

        # Initialize consistent structure
        missmatch_pipe = {"ref_fullid": "", "missmatch_fullid": {}}

        # Get the mode for the column to determine the reference value
        mode_value = proc_status_df[i].mode()[0]

        # Find rows that match the mode (will be used as reference)
        agreement_rows = proc_status_df[proc_status_df[i] == mode_value].index

        # Get reference subject details (using the first row that matches the mode)
        ref_ids = subj_ids_df.loc[agreement_rows].iloc[0, :]

        # Create identifiers for the reference subject
        cad2look_ref = [
            f"{key}-{ref_ids[value]}"
            for key, value in ent_list.items()
            if value in subj_columns
        ]

        # Get files for the reference subject
        ref_files = cltbids.get_individual_files_and_folders(
            pipe_dir_fold,
            cad2look_ref,
        )

        # Find the full ID of the reference subject
        try:
            ref_full_id = cltmisc.filter_by_substring(
                subj_list, or_filter=cad2look_ref[0], and_filter=cad2look_ref
            )[0]
        except IndexError:
            # Handle case where reference ID is not found
            ref_full_id = "unknown_reference"

        missmatch_pipe["ref_fullid"] = ref_full_id

        # Find rows that don't match the mode (disagreement rows)
        disagreement_rows = proc_status_df[proc_status_df[i] != mode_value].index

        # Only process mismatches if reference files exist and there are disagreements
        if ref_files and len(disagreement_rows) > 0:
            # Process reference files to remove path prefixes for comparison
            cad2look_ref.append(pipe_dir_fold)
            tmp_ref_files = cltmisc.remove_substrings(ref_files, cad2look_ref)

            # Get the ids of the subjects with disagreement
            subtable_ids = subj_ids_df.loc[disagreement_rows]

            # Loop through all subjects with disagreement
            for j in range(len(disagreement_rows)):
                # Get the subject ID
                sub_row = subtable_ids.iloc[j, :]

                # Create identifiers for this subject
                cad2look_ind = [
                    f"{key}-{sub_row[value]}"
                    for key, value in ent_list.items()
                    if value in subj_columns
                ]

                # Get files for this subject
                indiv_files = cltbids.get_individual_files_and_folders(
                    pipe_dir_fold,
                    cad2look_ind,
                )

                try:
                    # Find the full ID of this subject
                    indiv_full_id = cltmisc.filter_by_substring(
                        subj_list,
                        or_filter=cad2look_ind[0],
                        and_filter=cad2look_ind,
                    )[0]
                except IndexError:
                    # Handle case where subject ID is not found
                    indiv_full_id = f"unknown_subject_{j}"

                # Initialize results for this subject
                missmatch_subject = {"missing_files": [], "extra_files": []}

                if indiv_files:
                    # Process individual files to remove path prefixes for comparison
                    cad2look_ind.append(pipe_dir_fold)
                    tmp_indiv_files = cltmisc.remove_substrings(
                        indiv_files, cad2look_ind
                    )

                    # Find missing files (in reference but not in this subject)
                    tmp_miss = list(set(tmp_ref_files) - set(tmp_indiv_files))
                    if tmp_miss:
                        miss_indices = cltmisc.get_indexes_by_substring(
                            tmp_ref_files, tmp_miss
                        )
                        selected_files_ref = [ref_files[i] for i in miss_indices]
                        missmatch_subject["missing_files"] = cltmisc.replace_substrings(
                            selected_files_ref, cad2look_ref, cad2look_ind
                        )

                    # Find extra files (in this subject but not in reference)
                    tmp_extra = list(set(tmp_indiv_files) - set(tmp_ref_files))
                    if tmp_extra:
                        extra_indices = cltmisc.get_indexes_by_substring(
                            tmp_indiv_files, tmp_extra
                        )
                        selected_files_indiv = [indiv_files[i] for i in extra_indices]
                        missmatch_subject["extra_files"] = cltmisc.replace_substrings(
                            selected_files_indiv, cad2look_ind, cad2look_ref
                        )
                else:
                    # If no files found for this subject, all reference files are missing
                    missmatch_subject["missing_files"] = cltmisc.replace_substrings(
                        ref_files, cad2look_ref, cad2look_ind
                    )

                # Add this subject's details to the results
                missmatch_pipe["missmatch_fullid"][indiv_full_id] = missmatch_subject

        # Add this pipeline's results to the summary
        missmatch_summary[i] = missmatch_pipe

    # If only_ids is True, simplify the output to just include IDs
    if only_ids:
        for i in missmatch_summary.keys():
            missmatch_summary[i]["missmatch_fullid"] = list(
                missmatch_summary[i]["missmatch_fullid"].keys()
            )

    # Save results to JSON if requested
    if out_json is not None:
        json_path = os.path.dirname(out_json)
        if not os.path.isdir(json_path):
            # Raise an error if the directory does not exist
            raise FileNotFoundError(f"The directory {json_path} does not exist.")

        cltmisc.save_dictionary_to_json(missmatch_summary, out_json)

    return missmatch_summary, out_json


####################################################################################################
def get_processing_status_details_sqlite3(
    proc_status_df: Union[str, dict],
    subj_ids: Union[List[str], str],
    deriv_dir: str,
    pipe_dirs: Union[List[str], str] = None,
    out_json: str = None,
    db_path: str = None,
    only_ids: bool = False,
):
    """
    This function creates a dictionary with the details of the processing status of the subjects in the BIDs derivatives directory.
    It provides the IDs of the subjects with incomplete or mismatched number of files.

    Parameters
    ----------
    proc_status_df : str or dict
        Path to the processing status DataFrame or a DataFrame itself. This DataFrame can be
        obtained with the function "create_processing_status_table".

    subj_ids : list or str
        List of subject IDs or a text file containing the subject IDs.

    deriv_dir : str
        Path to the derivatives directory.

    pipe_dirs : list or str, optional
        List of processing pipelines to check. If None, all pipelines will be checked.

    out_json : str, optional
        Path to save the output JSON file. If None, the JSON file will not be saved.

    db_path : str, optional
        Path to save the SQLite database file. If None, the database will not be created.

    only_ids : bool, optional
        If True, only the IDs of the subjects with mismatches will be returned, without the file details.

    Returns
    -------
    dict
        Dictionary containing the details of the processing status of the subjects.

    str
        Path to the saved JSON file if out_json is provided, otherwise None.
    """

    from . import morphometrytools as cltmorpho
    import sqlite3

    if isinstance(proc_status_df, str):
        if not os.path.isfile(proc_status_df):
            raise FileNotFoundError(f"The file {proc_status_df} does not exist.")
        else:
            proc_status_df = cltmisc.smart_read_table(proc_status_df)
    elif not isinstance(proc_status_df, pd.DataFrame):
        raise TypeError("proc_status_df must be a DataFrame or a string path to a file")

    # Process subject IDs
    if isinstance(subj_ids, str):
        if not os.path.isfile(subj_ids):
            raise FileNotFoundError(f"The file {subj_ids} does not exist.")
        else:
            with open(subj_ids, "r") as f:
                subj_list = f.read().splitlines()
    elif isinstance(subj_ids, list):
        if len(subj_ids) == 0:
            raise ValueError("The list of subject IDs is empty.")
        else:
            subj_list = subj_ids
    else:
        raise TypeError("subj_ids must be a list or a string path to a file")

    # Check if the derivatives directory exists
    deriv_dir = cltmisc.remove_trailing_separators(deriv_dir)

    if not os.path.isdir(deriv_dir):
        raise FileNotFoundError(
            f"The derivatives directory {deriv_dir} does not exist."
        )

    # Find all the derivatives folders
    all_pipe_dirs = cltbids.get_derivatives_folders(deriv_dir)

    if len(all_pipe_dirs) == 0:
        raise ValueError(
            "No derivatives folders were found in the specified directory."
        )

    if pipe_dirs is not None:
        if isinstance(pipe_dirs, str):
            pipe_dirs = [pipe_dirs]

        pipe_dirs = cltmisc.filter_by_substring(all_pipe_dirs, or_filter=pipe_dirs)
    else:
        pipe_dirs = all_pipe_dirs

    # All entities
    ent_list = cltbids.entities4table()

    # Get all the columns names
    col_names = proc_status_df.columns.tolist()

    # Get all the columns that are not in the pipe_dirs
    subj_columns = list(set(col_names) - set(pipe_dirs))

    subj_ids_df = proc_status_df[subj_columns]

    # Create a consistent structure for the output dictionary
    missmatch_summary = {}

    # Initialize SQLite database if db_path is provided
    if db_path:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS pipelines (
            pipeline_id TEXT PRIMARY KEY,
            ref_fullid TEXT
        )"""
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS mismatches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pipeline_id TEXT,
            subject_id TEXT,
            FOREIGN KEY (pipeline_id) REFERENCES pipelines(pipeline_id),
            UNIQUE (pipeline_id, subject_id)
        )"""
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS file_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mismatch_id INTEGER,
            file_path TEXT,
            status TEXT,
            FOREIGN KEY (mismatch_id) REFERENCES mismatches(id)
        )"""
        )

        # Clear existing data if needed
        cursor.execute("DELETE FROM file_details")
        cursor.execute("DELETE FROM mismatches")
        cursor.execute("DELETE FROM pipelines")

    # Process each pipeline
    for i in pipe_dirs:
        proc_status_df[i] = proc_status_df[i].astype(int)
        pipe_dir_fold = os.path.join(deriv_dir, i)

        # Initialize consistent structure
        missmatch_pipe = {"ref_fullid": "", "missmatch_fullid": {}}

        # Get the mode for the column to determine the reference value
        mode_value = proc_status_df[i].mode()[0]

        # Find rows that match the mode (will be used as reference)
        agreement_rows = proc_status_df[proc_status_df[i] == mode_value].index

        # Get reference subject details (using the first row that matches the mode)
        ref_ids = subj_ids_df.loc[agreement_rows].iloc[0, :]

        # Create identifiers for the reference subject
        cad2look_ref = [
            f"{key}-{ref_ids[value]}"
            for key, value in ent_list.items()
            if value in subj_columns
        ]

        # Get files for the reference subject
        ref_files = cltbids.get_individual_files_and_folders(
            pipe_dir_fold,
            cad2look_ref,
        )

        # Find the full ID of the reference subject
        try:
            ref_full_id = cltmisc.filter_by_substring(
                subj_list, or_filter=cad2look_ref[0], and_filter=cad2look_ref
            )[0]
        except IndexError:
            # Handle case where reference ID is not found
            ref_full_id = "unknown_reference"

        missmatch_pipe["ref_fullid"] = ref_full_id

        # If using SQLite, insert pipeline info
        if db_path:
            cursor.execute(
                "INSERT OR REPLACE INTO pipelines VALUES (?, ?)", (i, ref_full_id)
            )

        # Find rows that don't match the mode (disagreement rows)
        disagreement_rows = proc_status_df[proc_status_df[i] != mode_value].index

        # Only process mismatches if reference files exist and there are disagreements
        if ref_files and len(disagreement_rows) > 0:
            # Process reference files to remove path prefixes for comparison
            cad2look_ref.append(pipe_dir_fold)
            tmp_ref_files = cltmisc.remove_substrings(ref_files, cad2look_ref)

            # Get the ids of the subjects with disagreement
            subtable_ids = subj_ids_df.loc[disagreement_rows]

            # Loop through all subjects with disagreement
            for j in range(len(disagreement_rows)):
                # Get the subject ID
                sub_row = subtable_ids.iloc[j, :]

                # Create identifiers for this subject
                cad2look_ind = [
                    f"{key}-{sub_row[value]}"
                    for key, value in ent_list.items()
                    if value in subj_columns
                ]

                # Get files for this subject
                indiv_files = cltbids.get_individual_files_and_folders(
                    pipe_dir_fold,
                    cad2look_ind,
                )

                try:
                    # Find the full ID of this subject
                    indiv_full_id = cltmisc.filter_by_substring(
                        subj_list,
                        or_filter=cad2look_ind[0],
                        and_filter=cad2look_ind,
                    )[0]
                except IndexError:
                    # Handle case where subject ID is not found
                    indiv_full_id = f"unknown_subject_{j}"

                # Initialize results for this subject
                missmatch_subject = {"missing_files": [], "extra_files": []}

                # Insert subject into mismatches table if using SQLite
                if db_path:
                    cursor.execute(
                        "INSERT OR REPLACE INTO mismatches (pipeline_id, subject_id) VALUES (?, ?)",
                        (i, indiv_full_id),
                    )
                    mismatch_id = cursor.lastrowid

                if indiv_files:
                    # Process individual files to remove path prefixes for comparison
                    cad2look_ind.append(pipe_dir_fold)
                    tmp_indiv_files = cltmisc.remove_substrings(
                        indiv_files, cad2look_ind
                    )

                    # Find missing files (in reference but not in this subject)
                    tmp_miss = list(set(tmp_ref_files) - set(tmp_indiv_files))
                    if tmp_miss:
                        miss_indices = cltmisc.get_indexes_by_substring(
                            tmp_ref_files, tmp_miss
                        )
                        selected_files_ref = [ref_files[i] for i in miss_indices]
                        missing_files = cltmisc.replace_substrings(
                            selected_files_ref, cad2look_ref, cad2look_ind
                        )
                        missmatch_subject["missing_files"] = missing_files

                        # Insert missing files into database if using SQLite
                        if db_path:
                            for file_path in missing_files:
                                cursor.execute(
                                    "INSERT INTO file_details (mismatch_id, file_path, status) VALUES (?, ?, ?)",
                                    (mismatch_id, file_path, "missing"),
                                )

                    # Find extra files (in this subject but not in reference)
                    tmp_extra = list(set(tmp_indiv_files) - set(tmp_ref_files))
                    if tmp_extra:
                        extra_indices = cltmisc.get_indexes_by_substring(
                            tmp_indiv_files, tmp_extra
                        )
                        selected_files_indiv = [indiv_files[i] for i in extra_indices]
                        extra_files = cltmisc.replace_substrings(
                            selected_files_indiv, cad2look_ind, cad2look_ref
                        )
                        missmatch_subject["extra_files"] = extra_files

                        # Insert extra files into database if using SQLite
                        if db_path:
                            for file_path in extra_files:
                                cursor.execute(
                                    "INSERT INTO file_details (mismatch_id, file_path, status) VALUES (?, ?, ?)",
                                    (mismatch_id, file_path, "extra"),
                                )
                else:
                    # If no files found for this subject, all reference files are missing
                    missing_files = cltmisc.replace_substrings(
                        ref_files, cad2look_ref, cad2look_ind
                    )
                    missmatch_subject["missing_files"] = missing_files

                    # Insert missing files into database if using SQLite
                    if db_path:
                        for file_path in missing_files:
                            cursor.execute(
                                "INSERT INTO file_details (mismatch_id, file_path, status) VALUES (?, ?, ?)",
                                (mismatch_id, file_path, "missing"),
                            )

                # Add this subject's details to the results
                missmatch_pipe["missmatch_fullid"][indiv_full_id] = missmatch_subject

        # Add this pipeline's results to the summary
        missmatch_summary[i] = missmatch_pipe

    # If only_ids is True, simplify the output to just include IDs
    if only_ids:
        for i in missmatch_summary.keys():
            missmatch_summary[i]["missmatch_fullid"] = list(
                missmatch_summary[i]["missmatch_fullid"].keys()
            )

    # Commit changes and close database connection if using SQLite
    if db_path:
        conn.commit()
        conn.close()

    # Save results to JSON if requested
    if out_json is not None:
        json_path = os.path.dirname(out_json)
        if not os.path.isdir(json_path):
            # Raise an error if the directory does not exist
            raise FileNotFoundError(f"The directory {json_path} does not exist.")

        cltmisc.save_dictionary_to_json(missmatch_summary, out_json)

    return missmatch_summary, out_json


####################################################################################################
def query_processing_status_db(
    db_path, query_type="subjects_with_mismatches", pipeline=None
):
    """
    Query the processing status database to extract useful information.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.

    query_type : str, optional
        Type of query to run. Options:
        - "subjects_with_mismatches": Get all subjects with mismatches
        - "pipelines_with_mismatches": Get all pipelines with mismatches and count
        - "missing_files_count": Get number of missing files per subject
        - "extra_files_count": Get number of extra files per subject

    pipeline : str, optional
        Name of the pipeline to filter by. Used only with certain query types.

    Returns
    -------
    pd.DataFrame
        Result of the query as a DataFrame.
    """
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(db_path)

    if query_type == "subjects_with_mismatches":
        if pipeline:
            query = """
            SELECT subject_id, pipeline_id
            FROM mismatches
            WHERE pipeline_id = ?
            ORDER BY subject_id
            """
            df = pd.read_sql_query(query, conn, params=(pipeline,))
        else:
            query = """
            SELECT subject_id, GROUP_CONCAT(pipeline_id) as pipelines
            FROM mismatches
            GROUP BY subject_id
            ORDER BY subject_id
            """
            df = pd.read_sql_query(query, conn)

    elif query_type == "pipelines_with_mismatches":
        query = """
        SELECT pipeline_id, COUNT(DISTINCT subject_id) as subject_count
        FROM mismatches
        GROUP BY pipeline_id
        ORDER BY subject_count DESC
        """
        df = pd.read_sql_query(query, conn)

    elif query_type == "missing_files_count":
        if pipeline:
            query = """
            SELECT m.subject_id, COUNT(*) as missing_count 
            FROM mismatches m
            JOIN file_details f ON m.id = f.mismatch_id
            WHERE f.status = 'missing' AND m.pipeline_id = ?
            GROUP BY m.subject_id
            ORDER BY missing_count DESC
            """
            df = pd.read_sql_query(query, conn, params=(pipeline,))
        else:
            query = """
            SELECT m.subject_id, m.pipeline_id, COUNT(*) as missing_count 
            FROM mismatches m
            JOIN file_details f ON m.id = f.mismatch_id
            WHERE f.status = 'missing'
            GROUP BY m.subject_id, m.pipeline_id
            ORDER BY missing_count DESC
            """
            df = pd.read_sql_query(query, conn)

    elif query_type == "extra_files_count":
        if pipeline:
            query = """
            SELECT m.subject_id, COUNT(*) as extra_count 
            FROM mismatches m
            JOIN file_details f ON m.id = f.mismatch_id
            WHERE f.status = 'extra' AND m.pipeline_id = ?
            GROUP BY m.subject_id
            ORDER BY extra_count DESC
            """
            df = pd.read_sql_query(query, conn, params=(pipeline,))
        else:
            query = """
            SELECT m.subject_id, m.pipeline_id, COUNT(*) as extra_count 
            FROM mismatches m
            JOIN file_details f ON m.id = f.mismatch_id
            WHERE f.status = 'extra'
            GROUP BY m.subject_id, m.pipeline_id
            ORDER BY extra_count DESC
            """
            df = pd.read_sql_query(query, conn)

    else:
        raise ValueError(f"Unknown query type: {query_type}")

    conn.close()
    return df


####################################################################################################
def export_db_to_json(db_path, out_json):
    """
    Export the processing status database to a JSON file in the same format
    as returned by get_processing_status_details.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.

    out_json : str
        Path to save the output JSON file.

    Returns
    -------
    dict
        Dictionary containing the details of the processing status of the subjects.
    """
    import sqlite3
    import json
    import os

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all pipelines
    cursor.execute("SELECT pipeline_id, ref_fullid FROM pipelines")
    pipelines = cursor.fetchall()

    # Create the output dictionary
    output_dict = {}

    for pipe_id, ref_fullid in pipelines:
        # Initialize pipeline entry
        pipe_entry = {"ref_fullid": ref_fullid, "missmatch_fullid": {}}

        # Get all mismatches for this pipeline
        cursor.execute(
            """
        SELECT id, subject_id 
        FROM mismatches 
        WHERE pipeline_id = ?
        """,
            (pipe_id,),
        )
        mismatches = cursor.fetchall()

        for mismatch_id, subject_id in mismatches:
            # Get missing files
            cursor.execute(
                """
            SELECT file_path 
            FROM file_details 
            WHERE mismatch_id = ? AND status = 'missing'
            """,
                (mismatch_id,),
            )
            missing_files = [row[0] for row in cursor.fetchall()]

            # Get extra files
            cursor.execute(
                """
            SELECT file_path 
            FROM file_details 
            WHERE mismatch_id = ? AND status = 'extra'
            """,
                (mismatch_id,),
            )
            extra_files = [row[0] for row in cursor.fetchall()]

            # Add to dictionary
            pipe_entry["missmatch_fullid"][subject_id] = {
                "missing_files": missing_files,
                "extra_files": extra_files,
            }

        # Add pipeline to output
        output_dict[pipe_id] = pipe_entry

    conn.close()

    # Save to JSON
    with open(out_json, "w") as f:
        json.dump(output_dict, f, indent=2)

    return output_dict
