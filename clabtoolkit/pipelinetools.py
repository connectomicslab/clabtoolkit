"""The pipelinetools module helps to manage batch processing of BIDS datasets.

It resolves which subjects to process, scans the derivatives folders of the pipelines,
and reports which subjects have incomplete or mismatched outputs. The reports can be
kept as tables, JSON files or a SQLite database.
"""

# Standard library imports
import json
import os
import queue
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path

# Third-party imports
import pandas as pd
from rich.console import Console
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
def get_fullids2process(
    ids: str | list[str] | None = None,
    in_dir: str = None,
    or_filter: str | list[str] = "T1w",
    and_filter: str | list[str] = None,
    source: str = "folders",  # Can be "folders" or "files"
) -> list[str]:
    """
    Get list of subject Full IDs to process. These IDs are collected from the BIDs dataset.
    They are equal to the name of the files without the suffixes and extensions.
    Example: if a file is named 'sub-001_ses-01_T1w.nii', the corresponding Full
    ID will be 'sub-001_ses-01'.

    Parameters
    ----------
    ids : str, list of str, or None, optional
        Subject Full IDs specification. Can be:

        - None: discover all subjects in `in_dir` (default)
        - list: list of subject Full ID strings
        - str: comma-separated IDs, single Full ID, or path to text file

    in_dir : str, optional
        Directory path to scan for subjects when `ids` is None.
        Only used when `ids` is None.

    or_filter : str or list of str, optional
        File extensions or patterns to filter for when scanning subject directories.
        Default is 'T1w'. A list of strings can be provided to specify multiple patterns.
        Only used when `ids` is None and `source` is 'files'.

    and_filter : str or list of str, optional
        Additional file extensions or patterns to filter for when scanning subject directories.
        Only used when `ids` is None and `source` is 'files'.

    source : str, optional
        How to build the IDs when `ids` is None:

        - 'folders' (default): one ID per ``sub-*`` folder (e.g. 'sub-001')
        - 'files': one Full ID per image file, i.e. its entity string without
          suffix and extension (e.g. 'sub-001_ses-01_acq-mprage')

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
    >>> get_fullids2process(ids=None, in_dir='/data/subjects')
    ['sub-001', 'sub-002', 'sub-003']

    >>> # From list
    >>> get_fullids2process(['sub-001', 'sub-002'])
    ['sub-001', 'sub-002']

    >>> # From comma-separated string
    >>> get_fullids2process('sub-001, sub-002, sub-003')
    ['sub-001', 'sub-002', 'sub-003']

    >>> # Single subject ID
    >>> get_fullids2process('sub-001')
    ['sub-001']

    >>> # From text file
    >>> get_fullids2process('/path/to/subjects.txt')
    ['sub-001', 'sub-002', 'sub-003']

    Notes
    -----
    When scanning directories (ids=None), only directories starting with 'sub-'
    are considered valid subject directories.

    Text files should contain one subject ID per line. Empty lines and
    whitespace are automatically filtered out.
    """

    if source not in ["folders", "files"]:
        raise ValueError(f"source must be 'folders' or 'files', got {source}")

    # Handle None case - discover from directory
    if ids is None:
        if not in_dir or not os.path.isdir(in_dir):
            raise ValueError(f"Valid in_dir required when ids is None. Got: {in_dir}")

        all_ids = []
        for d in os.listdir(in_dir):
            if d.startswith("sub-"):
                if source == "files":
                    tmp_files = cltmisc.get_all_files(
                        os.path.join(in_dir, d),
                        or_filter=or_filter,
                        and_filter=and_filter,
                    )
                    if tmp_files:
                        tmp_files = cltmisc.filter_by_substring(
                            tmp_files, or_filter=[".nii", ".nii.gz", ".mgz", ".mnc"]
                        )
                        if tmp_files:

                            for t in tmp_files:
                                # Get the filename
                                filename = os.path.basename(t)
                                bids_dict = cltbids.str2entity(filename)

                                # Remove the extension from the bids_dict
                                if "extension" in bids_dict.keys():
                                    del bids_dict["extension"]

                                if "suffix" in bids_dict.keys():
                                    del bids_dict["suffix"]

                                all_ids.append(cltbids.entity2str(bids_dict))
                else:
                    all_ids.append(d)
        all_ids = list(set(all_ids))  # Remove duplicates
        all_ids.sort()  # Sort the IDs for consistency
        return all_ids

    # Handle list case
    if isinstance(ids, list):
        all_ids = [str(id_).strip() for id_ in ids if str(id_).strip()]
        all_ids.sort()
        return all_ids

    # Handle string case
    if isinstance(ids, str):
        ids = ids.strip()
        if not ids:
            return []

        # File path
        if os.path.isfile(ids):
            with open(ids, encoding="utf-8") as f:
                all_ids = [line.strip() for line in f if line.strip()]
                all_ids.sort()
                return all_ids

        # Comma-separated or single ID
        all_ids = [id_.strip() for id_ in ids.split(",") if id_.strip()]
        all_ids.sort()
        return all_ids

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
    full_ids: list | str,
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

    full_ids : list or str
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
        If full_ids is not a list or a string path to a file.

    Examples
    --------
    >>> deriv_dir = "/path/to/derivatives"
    >>> full_ids = ["sub-01", "sub-02"]
    >>> output_table = "/path/to/output_table.csv"
    >>> df, saved_path = create_processing_status_table(deriv_dir, full_ids, output_table)
    >>> print(df)
    """

    from joblib import Parallel, delayed

    # Initialize rich console
    console = Console()

    # Check if the derivatives directory exists
    deriv_dir = cltmisc.remove_trailing_separators(deriv_dir)

    if not os.path.isdir(deriv_dir):
        raise FileNotFoundError(
            f"The derivatives directory {deriv_dir} does not exist."
        )

    # Process subject IDs (empty lines in a text file are ignored)
    subj_list = _load_full_ids(full_ids)

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
            id_dict["sub"]

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
def process_file(filepath: str | Path):
    """
    Parse BIDS entities from a single file path.

    Parameters
    ----------
    filepath : str or Path
        Full path to the file to be processed.

    Returns
    -------
    tuple
        (table, None) if parsing is successful, where `table` is a DataFrame with the
        extracted entities; otherwise (None, filepath). Files whose name does not
        start with 'sub-' are not BIDS files and are reported as failed.

    """

    if filepath is None:
        raise ValueError("filepath must not be None")

    if not isinstance(filepath, (str, Path)):
        raise TypeError("filepath must be a string or Path object")

    if isinstance(filepath, Path):
        filepath = str(filepath)

    if not os.path.basename(filepath).startswith("sub-"):
        return None, filepath

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
    tuple
        (table, None) if successful, where `table` is a DataFrame with the file counts
        by type (mri, vparc, surf, map, sparc, stats); otherwise (None, fs_id).
        A subject whose folder does not exist is reported as failed (and no folder
        is created).
    """
    fs_id, pipe_dir = args

    # FreeSurferSubject creates missing folders, so check before instantiating it
    if not os.path.isdir(os.path.join(str(pipe_dir), str(fs_id))):
        return None, fs_id

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
    full_ids: str | list = None,
    extensions: list = None,
) -> list:
    """
    Recursively collect all matching files under the derivatives folder.

    Parameters
    ----------
    pipe_dir : str
        Path to the pipeline derivatives directory to scan.

    full_ids : str or list, optional
        Full ID(s) to look for. If None, all ``sub-*`` folders in `pipe_dir` are used.

    extensions : list, optional
        File extensions to include in the scan. Default is [".nii.gz", ".nii", ".mgz", ".stats", ".annot", ".gii", ".gii.gz"].

    Returns
    -------
    list
        Sorted list of file paths that match the specified extensions and start with "sub-".

    Notes
    -----
    - Only files that start with "sub-" and end with one of the specified extensions are included.


    """
    # Detect all the directories in case no subject IDs are provided, just to check if there are any subject folders in the derivatives
    if extensions is None:
        extensions = [".nii.gz", ".nii", ".mgz", ".stats", ".annot", ".gii", ".gii.gz"]
    if full_ids is None:
        full_ids = get_fullids2process(None, in_dir=pipe_dir)

    # Clean list with the the subject IDs to look for in the file paths
    if isinstance(full_ids, str):
        full_ids = [full_ids]

    files = []
    for full_id in full_ids:
        ent_dict = cltbids.str2entity(
            full_id
        )  # Just to check if the IDs are valid BIDS entities

        # Check if there is any folder in the derivatives that starts with sub-ent_dict["sub"]
        matching_folders = glob(os.path.join(pipe_dir, f"sub-{ent_dict['sub']}*"))
        if matching_folders:
            for folder in matching_folders:
                sub_files = cltmisc.get_all_files(
                    folder, or_filter=[full_id], recursive=True
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
    full_ids: str | list = None,
    extensions: list = None,
    output_csv: str | Path = None,
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

    full_ids   : str or list, optional
        Full ID(s) to include. If None, all ``sub-*`` folders in the pipeline are used.

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
    if extensions is None:
        extensions = [".nii.gz", ".nii", ".mgz", ".stats", ".annot", ".gii", ".gii.gz"]
    pipe_dir = os.path.join(deriv_dir, pipe_id)
    bar_label = f"Pipeline: {pipe_id} [{pipe_index}/{pipe_total}]"

    if not os.path.isdir(pipe_dir):
        progress.print(f"[bold red]Error:[/bold red] '{pipe_dir}' does not exist.")
        return pd.DataFrame()

    # ── Decide upfront which processing path to take ──────────────────────
    # Ensures exactly one progress task is created per pipeline.
    if full_ids is None:
        full_ids = get_fullids2process(
            None, in_dir=pipe_dir
        )  # Just to check if subject folders exist

    files = scan_derivatives(pipe_dir, full_ids, extensions)

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

            ent_dict = cltbids.entities4table()
            # Get all the keys from the entity dictionary
            all_values = list(ent_dict.values())

            priority_names = all_values + ["Type", "Extension", "Full_path", "Count"]

            ordered = [c for c in priority_names if c in df.columns]
            remaining = [c for c in df.columns if c not in ordered]
            df = df[ordered + remaining]
            sort_cols = [
                c for c in ["Participant", "Session", "Run"] if c in df.columns
            ]
            if sort_cols:
                df = df.sort_values(by=sort_cols)
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
        fs_ids = get_fullids2process(None, in_dir=pipe_dir)
        task_id = progress.add_task(bar_label, total=len(fs_ids))

        fs_results, _ = _run_parallel(
            items=[(fs_id, pipe_dir) for fs_id in fs_ids],
            submit_fn=process_freesurfer_subject,
            progress=progress,
            task_id=task_id,
            n_workers=n_workers,
        )

        if fs_results:
            df = pd.concat(fs_results, ignore_index=True)
            df.drop_duplicates(inplace=True)
            sort_cols = [c for c in ["sub", "ses", "Type"] if c in df.columns]
            if sort_cols:
                df = df.sort_values(by=sort_cols)

            # Remove the columns that are completely empty except the column name
            df = df.apply(
                lambda col: col.map(
                    lambda x: pd.NA if isinstance(x, str) and x.strip() == "" else x
                )
            )
            df = df.dropna(axis=1, how="all")
        else:
            progress.print(
                f"  [yellow]No FreeSurfer subjects could be read in '{pipe_id}'.[/yellow]"
            )
            df = pd.DataFrame()

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
    pipe_dirs: list | None = None,
    full_ids: str | list = None,
    extensions: list = None,
    output_table: str | Path = None,
    overwrite: bool = False,
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

    full_ids   : str or list, optional
        Full ID(s) to include. If None, all ``sub-*`` folders of each pipeline are used.

    extensions : list, optional
        File extensions to include in the scan.
        Defaults to [".nii.gz", ".nii", ".mgz", ".stats", ".annot", ".gii", ".gii.gz"].

    output_table : str or Path, optional
        If provided, save the combined inventory DataFrame to this path (tab-separated).

    overwrite    : bool, optional
        If True, overwrite the existing output table if it exists. Defaults to False.

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
    if extensions is None:
        extensions = [".nii.gz", ".nii", ".mgz", ".stats", ".annot", ".gii", ".gii.gz"]
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
                full_ids=full_ids,
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
    if output_table is not None:

        # Check if the output_table has a valid path
        output_dir = os.path.dirname(str(output_table))
        if output_dir and not os.path.exists(output_dir):
            # Show an error message and raise an exception
            raise ValueError(
                f"The directory for the output table does not exist: {output_dir}"
            )

        if os.path.exists(output_table) and not overwrite:
            raise ValueError(
                f"The output table already exists and overwrite is set to False: {output_table}"
            )
        combined_summary.to_csv(str(output_table), index=False, sep="\t")
        print(
            f"\n✓ Combined inventory ({len(combined_summary)} rows) → {str(output_table)}"
        )
    else:
        print(
            f"\n✓ Combined inventory ({len(combined_summary)} rows) returned (no save)."
        )

    return combined_summary


####################################################################################################
def _load_status_table(proc_status_df: str | pd.DataFrame) -> pd.DataFrame:
    """
    Load the processing status table, keeping entity values as strings.

    Reading entity columns as strings matters: values such as '01' would otherwise
    be converted to the integer 1, and the rebuilt identifiers ('sub-1') would no
    longer match the files on disk ('sub-01').

    Parameters
    ----------
    proc_status_df : str or pd.DataFrame
        Path to the table (.tsv, .csv or other delimited text) or the DataFrame itself.

    Returns
    -------
    pd.DataFrame
        A copy of the processing status table.
    """
    if isinstance(proc_status_df, (str, Path)):
        proc_status_df = str(proc_status_df)
        if not os.path.isfile(proc_status_df):
            raise FileNotFoundError(f"The file {proc_status_df} does not exist.")

        if proc_status_df.endswith(".tsv"):
            return pd.read_csv(proc_status_df, sep="\t", dtype=str)
        if proc_status_df.endswith(".csv"):
            return pd.read_csv(proc_status_df, sep=",", dtype=str)
        return pd.read_csv(proc_status_df, sep=None, engine="python", dtype=str)

    if isinstance(proc_status_df, pd.DataFrame):
        return proc_status_df.copy()

    raise TypeError("proc_status_df must be a DataFrame or a string path to a file")


####################################################################################################
def _load_full_ids(full_ids: list[str] | str) -> list[str]:
    """
    Return the list of Full IDs from a list or from a text file (one ID per line).
    Empty lines and surrounding whitespace are removed.
    """
    if isinstance(full_ids, str):
        if not os.path.isfile(full_ids):
            raise FileNotFoundError(f"The file {full_ids} does not exist.")
        with open(full_ids, encoding="utf-8") as f:
            subj_list = [line.strip() for line in f if line.strip()]
    elif isinstance(full_ids, list):
        subj_list = [str(x).strip() for x in full_ids if str(x).strip()]
    else:
        raise TypeError("full_ids must be a list or a string path to a file")

    if len(subj_list) == 0:
        raise ValueError("The list of subject IDs is empty.")

    return subj_list


####################################################################################################
def _resolve_pipe_dirs(
    deriv_dir: str, pipe_dirs: list[str] | str | None
) -> tuple[str, list[str], list[str]]:
    """
    Validate the derivatives directory and resolve the pipelines to check.

    Returns
    -------
    deriv_dir : str
        Derivatives directory without trailing separators.
    all_pipe_dirs : list of str
        All pipeline folders found in the derivatives directory.
    pipe_dirs : list of str
        Pipelines selected for checking.
    """
    deriv_dir = cltmisc.remove_trailing_separators(deriv_dir)

    if not os.path.isdir(deriv_dir):
        raise FileNotFoundError(
            f"The derivatives directory {deriv_dir} does not exist."
        )

    all_pipe_dirs = cltbids.get_derivatives_folders(deriv_dir)

    if len(all_pipe_dirs) == 0:
        raise ValueError(
            "No derivatives folders were found in the specified directory."
        )

    if pipe_dirs is None:
        return deriv_dir, all_pipe_dirs, list(all_pipe_dirs)

    if isinstance(pipe_dirs, str):
        pipe_dirs = [pipe_dirs]

    # Exact names first; fall back to substring matching for anything not found exactly
    selected = [p for p in all_pipe_dirs if p in pipe_dirs]
    not_exact = [p for p in pipe_dirs if p not in all_pipe_dirs]
    if not_exact:
        for p in cltmisc.filter_by_substring(all_pipe_dirs, or_filter=not_exact):
            if p not in selected:
                selected.append(p)

    if len(selected) == 0:
        raise ValueError(
            f"None of the requested pipelines {pipe_dirs} were found in {deriv_dir}."
        )

    return deriv_dir, all_pipe_dirs, selected


####################################################################################################
def _row_entities(row: pd.Series, ent_list: dict, ent_cols: list[str]) -> dict:
    """
    Build an ordered {entity_key: 'key-value'} dictionary from one table row.
    Entities that are empty or missing (NaN) for this row are skipped, so that
    no 'acq-nan'-like strings are generated.
    """
    ents = {}
    for key, col in ent_list.items():
        if col not in ent_cols:
            continue
        val = row[col]
        if pd.isna(val) or str(val).strip() == "":
            continue
        ents[key] = f"{key}-{str(val).strip()}"
    return ents


####################################################################################################
def _find_full_id(subj_list: list[str], ent_strs: list[str], fallback: str) -> str:
    """
    Find the Full ID whose entities match `ent_strs`.

    Matching is done on whole entity tokens (split by '_'), so 'sub-01' never
    matches 'sub-010'. An exact match (same set of entities) is preferred; if
    none exists, the first Full ID containing all the entities is returned.
    """
    target = set(ent_strs)
    if not target:
        return fallback

    partial = None
    for full_id in subj_list:
        tokens = set(full_id.split("_"))
        if tokens == target:
            return full_id
        if partial is None and target.issubset(tokens):
            partial = full_id

    return partial if partial is not None else fallback


####################################################################################################
def _compute_processing_mismatches(
    proc_status_df: pd.DataFrame,
    subj_list: list[str],
    deriv_dir: str,
    pipe_dirs: list[str],
    all_pipe_dirs: list[str],
) -> dict:
    """
    Core comparison shared by get_processing_status_details_json and
    get_processing_status_details_sqlite3.

    For every pipeline, the most frequent file count (mode) defines the expected
    state. The first subject with that count is used as reference, and every
    subject with a different count is compared file by file against it.

    Returns
    -------
    dict
        {pipeline: {"ref_fullid": str,
                    "missmatch_fullid": {full_id: {"missing_files": [...],
                                                   "extra_files": [...]}}}}
    """
    ent_list = cltbids.entities4table()

    # Entity columns = every column that is not a pipeline column
    # (all pipelines are excluded, not only the selected ones)
    ent_cols = [c for c in proc_status_df.columns if c not in all_pipe_dirs]

    missing_cols = [p for p in pipe_dirs if p not in proc_status_df.columns]
    if missing_cols:
        raise ValueError(
            f"The processing status table has no column for the pipeline(s): {missing_cols}. "
            "Re-create it with create_processing_status_table."
        )

    missmatch_summary = {}

    for pipe in pipe_dirs:
        counts = pd.to_numeric(proc_status_df[pipe], errors="coerce").fillna(0)
        counts = counts.astype(int)
        pipe_dir_fold = os.path.join(deriv_dir, pipe)

        missmatch_pipe = {"ref_fullid": "", "missmatch_fullid": {}}

        # The most frequent number of files is considered the expected state
        mode_value = counts.mode()[0]
        agreement_rows = counts[counts == mode_value].index
        disagreement_rows = counts[counts != mode_value].index

        # Reference subject: first row that matches the mode
        ref_ents = _row_entities(
            proc_status_df.loc[agreement_rows[0]], ent_list, ent_cols
        )
        ref_strs = list(ref_ents.values())

        ref_files = cltbids.get_individual_files_and_folders(pipe_dir_fold, ref_strs)
        missmatch_pipe["ref_fullid"] = _find_full_id(
            subj_list, ref_strs, "unknown_reference"
        )

        if ref_files and len(disagreement_rows) > 0:
            # File names without the entity strings and pipeline path, for comparison
            ref_names = cltmisc.remove_substrings(ref_files, ref_strs + [pipe_dir_fold])

            for j, row_idx in enumerate(disagreement_rows):
                ind_ents = _row_entities(
                    proc_status_df.loc[row_idx], ent_list, ent_cols
                )
                ind_strs = list(ind_ents.values())

                indiv_full_id = _find_full_id(
                    subj_list, ind_strs, f"unknown_subject_{j}"
                )

                # Mapping between reference and individual entities, matched by key
                ref2ind = {
                    ref_ents[k]: ind_ents[k]
                    for k in ref_ents
                    if k in ind_ents and ref_ents[k] != ind_ents[k]
                }

                indiv_files = cltbids.get_individual_files_and_folders(
                    pipe_dir_fold, ind_strs
                )

                missmatch_subject = {"missing_files": [], "extra_files": []}

                if indiv_files:
                    ind_names = cltmisc.remove_substrings(
                        indiv_files, ind_strs + [pipe_dir_fold]
                    )

                    # Missing: present for the reference, absent for this subject
                    miss_set = set(ref_names) - set(ind_names)
                    if miss_set:
                        sel = [
                            ref_files[k]
                            for k, name in enumerate(ref_names)
                            if name in miss_set
                        ]
                        missmatch_subject["missing_files"] = (
                            cltmisc.replace_substrings(sel, ref2ind) if ref2ind else sel
                        )

                    # Extra: present for this subject, absent for the reference.
                    # These files exist on disk, so their real paths are reported.
                    extra_set = set(ind_names) - set(ref_names)
                    if extra_set:
                        missmatch_subject["extra_files"] = [
                            indiv_files[k]
                            for k, name in enumerate(ind_names)
                            if name in extra_set
                        ]
                else:
                    # No files at all for this subject: every reference file is missing
                    missmatch_subject["missing_files"] = (
                        cltmisc.replace_substrings(ref_files, ref2ind)
                        if ref2ind
                        else list(ref_files)
                    )

                missmatch_pipe["missmatch_fullid"][indiv_full_id] = missmatch_subject

        missmatch_summary[pipe] = missmatch_pipe

    return missmatch_summary


####################################################################################################
def _finalize_summary(missmatch_summary: dict, only_ids: bool, out_json: str | None):
    """
    Optionally reduce the summary to IDs only and save it as JSON.
    """
    if only_ids:
        missmatch_summary = {
            pipe: {
                "ref_fullid": info["ref_fullid"],
                "missmatch_fullid": list(info["missmatch_fullid"].keys()),
            }
            for pipe, info in missmatch_summary.items()
        }

    if out_json is not None:
        json_dir = os.path.dirname(str(out_json))
        if json_dir and not os.path.isdir(json_dir):
            raise FileNotFoundError(f"The directory {json_dir} does not exist.")

        cltmisc.save_dictionary_to_json(missmatch_summary, str(out_json))

    return missmatch_summary


####################################################################################################
def get_processing_status_details_json(
    proc_status_df: str | pd.DataFrame,
    full_ids: list[str] | str,
    deriv_dir: str,
    pipe_dirs: list[str] | str = None,
    out_json: str = None,
    only_ids: bool = False,
):
    """
    This function creates a dictionary with the details of the processing status of the subjects in the BIDs derivatives directory.
    It provides the IDs of the subjects with incomplete or mismatched number of files.

    Parameters
    ----------
    proc_status_df : str or pd.DataFrame
        Path to the processing status table or a DataFrame itself. This table can be
        obtained with the function "create_processing_status_table".

    full_ids : list or str
        List of Full IDs or a text file containing the Full IDs (one per line).

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

    proc_status_df = _load_status_table(proc_status_df)
    subj_list = _load_full_ids(full_ids)
    deriv_dir, all_pipe_dirs, pipe_dirs = _resolve_pipe_dirs(deriv_dir, pipe_dirs)

    missmatch_summary = _compute_processing_mismatches(
        proc_status_df, subj_list, deriv_dir, pipe_dirs, all_pipe_dirs
    )

    missmatch_summary = _finalize_summary(missmatch_summary, only_ids, out_json)

    return missmatch_summary, out_json


####################################################################################################
def _write_mismatches_to_db(missmatch_summary: dict, db_path: str):
    """
    Write a full (not only_ids) mismatch summary into an SQLite database.
    The tables are dropped and re-created on every call, so an existing database
    with an older schema is always replaced.
    """
    import sqlite3

    db_dir = os.path.dirname(str(db_path))
    if db_dir and not os.path.isdir(db_dir):
        raise FileNotFoundError(f"The directory {db_dir} does not exist.")

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()

        # Drop old tables so schema changes are always applied (children first)
        cursor.execute("DROP TABLE IF EXISTS file_details")
        cursor.execute("DROP TABLE IF EXISTS mismatches")
        cursor.execute("DROP TABLE IF EXISTS pipelines")

        cursor.execute(
            """
        CREATE TABLE pipelines (
            pipeline_id TEXT PRIMARY KEY,
            ref_fullid TEXT
        )"""
        )

        cursor.execute(
            """
        CREATE TABLE mismatches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pipeline_id TEXT,
            subject_fullid TEXT,
            FOREIGN KEY (pipeline_id) REFERENCES pipelines(pipeline_id),
            UNIQUE (pipeline_id, subject_fullid)
        )"""
        )

        cursor.execute(
            """
        CREATE TABLE file_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mismatch_id INTEGER,
            file_path TEXT,
            status TEXT,
            FOREIGN KEY (mismatch_id) REFERENCES mismatches(id)
        )"""
        )

        for pipe, info in missmatch_summary.items():
            cursor.execute(
                "INSERT INTO pipelines (pipeline_id, ref_fullid) VALUES (?, ?)",
                (pipe, info["ref_fullid"]),
            )

            for full_id, details in info["missmatch_fullid"].items():
                cursor.execute(
                    "INSERT INTO mismatches (pipeline_id, subject_fullid) VALUES (?, ?)",
                    (pipe, full_id),
                )
                mismatch_id = cursor.lastrowid

                rows = [
                    (mismatch_id, fp, "missing") for fp in details["missing_files"]
                ] + [(mismatch_id, fp, "extra") for fp in details["extra_files"]]

                if rows:
                    cursor.executemany(
                        "INSERT INTO file_details (mismatch_id, file_path, status) VALUES (?, ?, ?)",
                        rows,
                    )

        conn.commit()
    finally:
        conn.close()


####################################################################################################
def get_processing_status_details_sqlite3(
    proc_status_df: str | pd.DataFrame,
    full_ids: list[str] | str,
    deriv_dir: str,
    pipe_dirs: list[str] | str = None,
    out_json: str = None,
    db_path: str = None,
    only_ids: bool = False,
):
    """
    This function creates a dictionary with the details of the processing status of the subjects in the BIDs derivatives directory.
    It provides the IDs of the subjects with incomplete or mismatched number of files, and optionally
    stores the results in an SQLite database.

    Parameters
    ----------
    proc_status_df : str or pd.DataFrame
        Path to the processing status table or a DataFrame itself. This table can be
        obtained with the function "create_processing_status_table".

    full_ids : list or str
        List of Full IDs or a text file containing the Full IDs (one per line).

    deriv_dir : str
        Path to the derivatives directory.

    pipe_dirs : list or str, optional
        List of processing pipelines to check. If None, all pipelines will be checked.

    out_json : str, optional
        Path to save the output JSON file. If None, the JSON file will not be saved.

    db_path : str, optional
        Path to save the SQLite database file. If None, the database will not be created.
        If the database already exists, its tables are dropped and re-created.

    only_ids : bool, optional
        If True, only the IDs of the subjects with mismatches will be returned, without the file details.
        The database always stores the full file details.

    Returns
    -------
    dict
        Dictionary containing the details of the processing status of the subjects.

    str
        Path to the saved JSON file if out_json is provided, otherwise None.
    """

    proc_status_df = _load_status_table(proc_status_df)
    subj_list = _load_full_ids(full_ids)
    deriv_dir, all_pipe_dirs, pipe_dirs = _resolve_pipe_dirs(deriv_dir, pipe_dirs)

    missmatch_summary = _compute_processing_mismatches(
        proc_status_df, subj_list, deriv_dir, pipe_dirs, all_pipe_dirs
    )

    # The database always receives the full details
    if db_path:
        _write_mismatches_to_db(missmatch_summary, db_path)

    missmatch_summary = _finalize_summary(missmatch_summary, only_ids, out_json)

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

    valid_queries = [
        "subjects_with_mismatches",
        "pipelines_with_mismatches",
        "missing_files_count",
        "extra_files_count",
    ]
    if query_type not in valid_queries:
        raise ValueError(
            f"Unknown query type: {query_type}. Valid options: {valid_queries}"
        )

    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"The database {db_path} does not exist.")

    conn = sqlite3.connect(db_path)

    try:
        df = _run_status_query(conn, query_type, pipeline)
    finally:
        conn.close()

    return df


def _run_status_query(conn, query_type: str, pipeline: str | None) -> pd.DataFrame:
    """Run one of the predefined queries of query_processing_status_db."""

    if query_type == "subjects_with_mismatches":
        if pipeline:
            query = """
            SELECT subject_fullid, pipeline_id
            FROM mismatches
            WHERE pipeline_id = ?
            ORDER BY subject_fullid
            """
            df = pd.read_sql_query(query, conn, params=(pipeline,))
        else:
            query = """
            SELECT subject_fullid, GROUP_CONCAT(pipeline_id) as pipelines
            FROM mismatches
            GROUP BY subject_fullid
            ORDER BY subject_fullid
            """
            df = pd.read_sql_query(query, conn)

    elif query_type == "pipelines_with_mismatches":
        query = """
        SELECT pipeline_id, COUNT(DISTINCT subject_fullid) as subject_count
        FROM mismatches
        GROUP BY pipeline_id
        ORDER BY subject_count DESC
        """
        df = pd.read_sql_query(query, conn)

    elif query_type == "missing_files_count":
        if pipeline:
            query = """
            SELECT m.subject_fullid, COUNT(*) as missing_count
            FROM mismatches m
            JOIN file_details f ON m.id = f.mismatch_id
            WHERE f.status = 'missing' AND m.pipeline_id = ?
            GROUP BY m.subject_fullid
            ORDER BY missing_count DESC
            """
            df = pd.read_sql_query(query, conn, params=(pipeline,))
        else:
            query = """
            SELECT m.subject_fullid, m.pipeline_id, COUNT(*) as missing_count
            FROM mismatches m
            JOIN file_details f ON m.id = f.mismatch_id
            WHERE f.status = 'missing'
            GROUP BY m.subject_fullid, m.pipeline_id
            ORDER BY missing_count DESC
            """
            df = pd.read_sql_query(query, conn)

    elif query_type == "extra_files_count":
        if pipeline:
            query = """
            SELECT m.subject_fullid, COUNT(*) as extra_count
            FROM mismatches m
            JOIN file_details f ON m.id = f.mismatch_id
            WHERE f.status = 'extra' AND m.pipeline_id = ?
            GROUP BY m.subject_fullid
            ORDER BY extra_count DESC
            """
            df = pd.read_sql_query(query, conn, params=(pipeline,))
        else:
            query = """
            SELECT m.subject_fullid, m.pipeline_id, COUNT(*) as extra_count
            FROM mismatches m
            JOIN file_details f ON m.id = f.mismatch_id
            WHERE f.status = 'extra'
            GROUP BY m.subject_fullid, m.pipeline_id
            ORDER BY extra_count DESC
            """
            df = pd.read_sql_query(query, conn)

    else:
        raise ValueError(f"Unknown query type: {query_type}")

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

    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"The database {db_path} does not exist.")

    conn = sqlite3.connect(db_path)
    try:
        output_dict = _db_to_dict(conn)
    finally:
        conn.close()

    # Save to JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)

    return output_dict


def _db_to_dict(conn) -> dict:
    """Rebuild the mismatch summary dictionary from an open SQLite connection."""
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
        SELECT id, subject_fullid
        FROM mismatches
        WHERE pipeline_id = ?
        """,
            (pipe_id,),
        )
        mismatches = cursor.fetchall()

        for mismatch_id, subject_fullid in mismatches:
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
            pipe_entry["missmatch_fullid"][subject_fullid] = {
                "missing_files": missing_files,
                "extra_files": extra_files,
            }

        # Add pipeline to output
        output_dict[pipe_id] = pipe_entry

    return output_dict
