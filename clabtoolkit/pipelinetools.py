# Standard library imports
import json
import os
import queue
import re
import shutil
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

# Local imports
# - Importing the submodules of clabtoolkit that are used in this file. This is done to avoid circular imports and to keep the code organized.
from . import bidstools as cltbids
from . import freesurfertools as cltfree
from . import misctools as cltmisc

# Suppress specific deprecation warnings from ipykernel and ipywidgets that can occur in certain environments
warnings.filterwarnings(
    "ignore",
    message="Kernel._parent_header is deprecated",
    category=DeprecationWarning,
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ipywidgets")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ipykernel")


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


######################################################################################################
def process_file(filepath: str):
    """
    Parse BIDS entities from a single file path.

    Parameters
    ----------
    filepath : str
        Full path to the file to be processed.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with extracted entities if parsing is successful, otherwise None.

    """
    try:
        table = cltbids.entities_to_table(filepath, include_suffix=True)
        if table is None or table.empty:
            return None, filepath
        return table, None
    except Exception:
        return None, filepath


######################################################################################################
def process_freesurfer_subject(args):
    """
    Process a single FreeSurfer subject.

    Parameters
    ----------
    args : tuple
        Tuple containing (fs_id, pipe_dir) where:
        - fs_id: FreeSurfer subject ID (e.g., 'sub-001')
        - pipe_dir: Path to the pipeline derivatives directory to scan for this subject (e.g., '/path/to/derivatives/fsl-firstparc')

    Returns
    -------
    pd.DataFrame or None
        DataFrame with file type counts for the subject if successful, otherwise None.



    """
    fs_id, pipe_dir = args
    try:
        fs_subj = cltfree.FreeSurferSubject(fs_id, pipe_dir)
        f_types = fs_subj.fs_files_count.keys()
        f_counts = [fs_subj.fs_files_count[ft] for ft in f_types]
        table = cltbids.entities_to_table(fs_id)
        tmp_df = pd.DataFrame({"Type": f_types, "Count": f_counts})
        tmp_df = cltmisc.expand_and_concatenate(table, tmp_df)
        return tmp_df, None
    except Exception:
        return None, fs_id


########################################################################################################
def scan_derivatives(
    pipe_dir: str,
    subj_ids: Union[str, list] = None,
    extensions: list = [
        ".nii.gz",
        ".nii",
        ".mgz",
        ".stats",
        ".annot",
        ".gii",
        ".gii.gz",
    ],
) -> list:
    """
    Recursively collect all matching files under the derivatives folder.

    Parameters
    ----------
    deriv_dir : str
        Path to the derivatives directory to scan.

    extensions : list, optional
        Tuple of file extensions to include in the scan. Default is [".nii.gz", ".nii", ".mgz", ".stats", ".annot", ".gii", ".gii.gz"].

    Returns
    list
        Sorted list of file paths that match the specified extensions and start with "sub-".

    Notes
    -----
    - Only files that start with "sub-" and end with one of the specified extensions are included.


    """
    # Detect all the directories in case no subject IDs are provided, just to check if there are any subject folders in the derivatives
    if subj_ids is None:
        subj_ids = get_ids2process(None, in_dir=pipe_dir)

    # Clean list with the the subject IDs to look for in the file paths
    if isinstance(subj_ids, str):
        subj_ids = [subj_ids]

    files = []
    for subj_id in subj_ids:
        ent_dict = cltbids.str2entity(
            subj_id
        )  # Just to check if the IDs are valid BIDS entities

        # Check if there is any folder in the derivatives that starts with sub-ent_dict["sub"]
        matching_folders = glob(os.path.join(pipe_dir, f"sub-{ent_dict['sub']}*"))
        if matching_folders:
            for folder in matching_folders:
                sub_files = cltmisc.get_all_files(
                    folder, or_filter=[subj_id], recursive=True
                )
                files.extend(sub_files)

    files = [f for f in files if any(f.endswith(ext) for ext in extensions)]

    return sorted(files)


###############################################################################################
def _run_parallel(
    items: list,
    submit_fn,
    progress: Progress,
    task_id,
    n_workers: int,
) -> tuple[list, list]:
    """
    Submit items to a ThreadPoolExecutor and advance the progress bar in the
    main thread via as_completed, which is guaranteed to yield every future
    exactly once — eliminating the callback timing issues that caused the bar
    to stall before 100%.

    Parameters
    ----------
    items     : list       Items to process.
    submit_fn : callable   Worker function; must return (result, fail_info).
    progress  : Progress   Active Rich Progress instance.
    task_id   :            Task ID returned by progress.add_task().
    n_workers : int        Thread-pool size.

    Returns
    -------
    results : list  Non-None return values from submit_fn.
    failed  : list  Items whose submit_fn returned None or raised.
    """
    results, failed = [], []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(submit_fn, item): item for item in items}

        for f in as_completed(futures):
            progress.advance(
                task_id
            )  # main thread; always fires exactly once per future
            try:
                result, fail_info = f.result()
                if result is not None:
                    results.append(result)
                else:
                    failed.append(fail_info)
            except Exception as e:
                print(f"[ERROR] {futures[f]}: {e}")
                failed.append(futures[f])

    return results, failed


###################################################################################################
def build_inventory(
    deriv_dir: str,
    pipe_id: str,
    pipe_index: int,
    pipe_total: int,
    progress: Progress,
    subj_ids: Union[str, list] = None,
    extensions: list = [
        ".nii.gz",
        ".nii",
        ".mgz",
        ".stats",
        ".annot",
        ".gii",
        ".gii.gz",
    ],
    output_csv: Union[str, Path] = None,
    n_workers: int = 8,
) -> pd.DataFrame:
    """
    Build a file inventory for a single pipeline derivative folder.

    Parameters
    ----------
    deriv_dir  : str
        Root derivatives directory.

    pipe_id    : str
        Name of the pipeline sub-folder inside deriv_dir.

    pipe_index : int
        1-based index of this pipeline (used in the progress bar label).

    pipe_total : int
        Total number of pipelines being processed (used in the progress bar label).

    progress   : Progress
        Active Rich Progress instance to attach progress bars to.

    extensions : list, optional
        File extensions to include in the scan. Defaults to [".nii.gz", ".nii", ".mgz", ".stats", ".annot", ".gii", ".gii.gz"].

    output_csv : str or Path, optional
        If provided, save the inventory DataFrame to this CSV path.

    n_workers  : int, optional
        Number of parallel worker threads.

    Returns
    -------
    pd.DataFrame
        Inventory table for this pipeline.
    """
    pipe_dir = os.path.join(deriv_dir, pipe_id)
    bar_label = f"Pipeline: {pipe_id} [{pipe_index}/{pipe_total}]"

    if not os.path.isdir(pipe_dir):
        progress.print(f"[bold red]Error:[/bold red] '{pipe_dir}' does not exist.")
        return pd.DataFrame()

    # ── Decide upfront which processing path to take ──────────────────────
    # Ensures exactly one progress task is created per pipeline.
    if subj_ids is None:
        subj_ids = get_ids2process(
            None, in_dir=pipe_dir
        )  # Just to check if subject folders exist

    files = scan_derivatives(pipe_dir, subj_ids, extensions)

    if files:
        # ── BIDS path ─────────────────────────────────────────────────────
        task_id = progress.add_task(bar_label, total=len(files))
        results, _ = _run_parallel(
            items=files,
            submit_fn=process_file,
            progress=progress,
            task_id=task_id,
            n_workers=n_workers,
        )

        if results:
            df = pd.concat(results, ignore_index=True)
            df.drop_duplicates(inplace=True)
            df["Count"] = 1

            priority_cols = [
                "count",
                "sub",
                "ses",
                "run",
                "acq",
                "space",
                "model",
                "desc",
                "res",
                "suffix",
                "extension",
                "full_path",
            ]
            ordered = [c for c in priority_cols if c in df.columns]
            remaining = [c for c in df.columns if c not in ordered]
            df = df[ordered + remaining]
            df.sort_values(
                by=[c for c in ["sub", "ses", "run"] if c in df.columns], inplace=True
            )
        else:
            progress.print(
                f"  [yellow]No parseable BIDS files found in '{pipe_id}'.[/yellow]"
            )
            df = pd.DataFrame()

    else:
        # ── FreeSurfer fallback ───────────────────────────────────────────
        progress.print(
            f"  [yellow]No matching files found — falling back to FreeSurfer subject scan for '{pipe_id}'…[/yellow]"
        )
        fs_ids = get_ids2process(None, in_dir=pipe_dir)
        task_id = progress.add_task(bar_label, total=len(fs_ids))

        fs_results, _ = _run_parallel(
            items=[(fs_id, pipe_dir) for fs_id in fs_ids],
            submit_fn=process_freesurfer_subject,
            progress=progress,
            task_id=task_id,
            n_workers=n_workers,
        )

        df = pd.concat(fs_results, ignore_index=True)
        df.drop_duplicates(inplace=True)
        df.sort_values(
            by=[c for c in ["sub", "ses", "Type"] if c in df.columns], inplace=True
        )

        # Remove the columns that are completely empty except the column name
        df = df.apply(
            lambda col: col.map(
                lambda x: pd.NA if isinstance(x, str) and x.strip() == "" else x
            )
        )
        df = df.dropna(axis=1, how="all")

    # ── Save ──────────────────────────────────────────────────────────────
    if output_csv is not None:
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            progress.print(
                f"  [yellow]Warning: output directory '{output_dir}' does not exist — skipping save.[/yellow]"
            )
        else:
            save_path = (
                str(output_csv)
                if str(output_csv).endswith(".csv")
                else f"{output_csv}.csv"
            )
            df.to_csv(save_path, index=False)
            progress.print(
                f"  Saved [bold]{len(df)}[/bold] rows → [green]{save_path}[/green]"
            )
    else:
        progress.print(f"  Returning [bold]{len(df)}[/bold] rows (no save).")

    return df


####################################################################################################
def build_derivatives_inventory(
    deriv_dir: str,
    pipe_dirs: Union[list, None] = None,
    subj_ids: Union[str, list] = None,
    extensions: list = [
        ".nii.gz",
        ".nii",
        ".mgz",
        ".stats",
        ".annot",
        ".gii",
        ".gii.gz",
    ],
    output_csv: Union[str, Path] = None,
    n_workers: int = 8,
) -> pd.DataFrame:
    """
    Build a combined file inventory across all pipeline derivative folders.

    Parameters
    ----------
    deriv_dir  : str
        Root derivatives directory containing one sub-folder per pipeline.

    pipe_dirs  : list of str, optional
        Pipeline sub-folder names to process. If None, all sub-folders
        discovered by cltbids.get_derivatives_folders() are used.

    extensions : list, optional
        File extensions to include in the scan.

        Defaults to (".nii.gz", ".nii").
    output_csv : str or Path, optional
        If provided, save the combined inventory DataFrame to this CSV path.

    n_workers  : int, optional
        Number of parallel worker threads per pipeline. Defaults to 8.

    Returns
    -------
    pd.DataFrame
        Combined inventory table with an extra leading "Pipeline" column.

    Raises
    ------
    ValueError
        If no pipeline folders are found or provided.
    """
    if pipe_dirs is None:
        pipe_dirs = cltbids.get_derivatives_folders(deriv_dir)
    if not pipe_dirs:
        raise ValueError(
            "No derivatives folders were found in the specified directory."
        )

    n_pipes = len(pipe_dirs)
    combined_summary = pd.DataFrame()

    with Progress() as progress:
        for idx, tmp_pipe_deriv in enumerate(pipe_dirs, start=1):
            tmp_summary = build_inventory(
                deriv_dir=deriv_dir,
                pipe_id=tmp_pipe_deriv,
                subj_ids=subj_ids,
                pipe_index=idx,
                pipe_total=n_pipes,
                progress=progress,
                extensions=extensions,
                n_workers=n_workers,
            )
            tmp_summary.insert(0, "Pipeline", tmp_pipe_deriv)

            combined_summary = (
                tmp_summary
                if idx == 1
                else pd.concat([combined_summary, tmp_summary], ignore_index=True)
            )

    # ── Save combined result ──────────────────────────────────────────────
    if output_csv is not None:
        save_path = (
            str(output_csv) if str(output_csv).endswith(".csv") else f"{output_csv}.csv"
        )
        combined_summary.to_csv(save_path, index=False)
        print(f"\n✓ Combined inventory ({len(combined_summary)} rows) → {save_path}")
    else:
        print(
            f"\n✓ Combined inventory ({len(combined_summary)} rows) returned (no save)."
        )

    return combined_summary


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
