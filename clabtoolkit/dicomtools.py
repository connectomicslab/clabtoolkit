import os
from glob import glob
import tarfile
import shutil
import pandas as pd
import sys
import pydicom
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rich.progress import Progress
from threading import Lock
import time

from typing import List, Optional, Union


# Importing local modules
from . import misctools as cltmisc


# simple progress indicator callback function
def progress_indicator(future):
    """
    A simple progress indicator for the concurrent futures
    :param future: future object

    """
    global lock, n_dics, n_comp, pb, pb1, subj_id, dicom_files
    # obtain the lock
    with lock:
        # update the counter
        n_comp += 1
        # report progress
        # print(f'{tasks_completed}/{n_subj} completed, {n_subj-tasks_completed} remain.')
        # pb.update(task_id=pb1, description= f'[red]Completed {n_comp}/{n_subj}', completed=n_subj)
        pb.update(
            task_id=pb1,
            description=f"[red]{subj_id}: Finished ({n_comp}/{n_dics})",
            completed=n_comp,
        )


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############      Section 1: Methods dedicated to organice and copy DICOM files         ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def org_conv_dicoms(
    in_dic_dir: str,
    out_dic_dir: str,
    demog_file: str = None,
    ids_file: str = None,
    ses_id: str = None,
    nosub: bool = False,
    booldic: bool = True,
    boolcomp: bool = False,
    force: bool = False,
    nthreads: int = 0,
):
    """
    This method organizes the DICOM files in sessions and series. It could also use the demographics file to define the session ID.

    Parameters
    ----------
    in_dic_dir : str
        Directory containing the subjects. It assumes all individual folders inside the directory as individual subjects.
        The subjects directory should start with 'sub-' otherwise the subjects will not be considered unless the "nosub"
        variable is set to True.

    out_dic_dir : str
        Output directory where the organized DICOM files will be saved. A new folder called 'Dicom' will be created inside this directory.

    demog_file : str, optional
        Demographics file containing the information about the subjects. The file should contain the following mandatory columns:
        'participant_id', 'session_id', 'acq_date'. Other columns such as 'birth_date', 'sex', 'group_id' or 'scanner_id' could be added.

    ids_file : str, optional
        Text file containing the list of subject IDs to be considered. The file should contain the subject IDs in a single column.

    ses_id : str, optional
        Session ID to be added to the session name. If not provided, the session ID will be the date of the study or the session ID
        extracted from the demographics table.

    nosub : bool, optional, default=False
        Boolean variable to consider the subjects that do not start with 'sub-'.

    booldic : bool, optional, default=True
        Boolean variable to organize the DICOM files. If False it will leave the folders as they are.

    boolcomp : bool, optional, default=False
        Boolean variable to compress the sessions containing the organized DICOM files. If True it will compress the sessions.

    force : bool, optional, default=False
        Boolean variable to force the copy of the DICOM file if the file already exists.

    nthreds : int, optional, default=0
        Number of threads to be used in the process. Default is 0 that means automatic selection of the number of cores.

    Returns
    -------
    None
        This method performs file organization operations and does not return a value.

    Raises
    ------
    FileNotFoundError
        If the input directory does not exist.

    ValueError
        If the demographics file is provided but does not contain the mandatory columns.

    PermissionError
        If there are insufficient permissions to write to the output directory.

    Examples
    --------
    >>> # Basic usage with input and output directories
    >>> organize_dicom_files('/path/to/input/dicoms', '/path/to/output')

    >>> # Using demographics file and custom session ID
    >>> organize_dicom_files(
    ...     in_dic_dir='/path/to/input/dicoms',
    ...     out_dic_dir='/path/to/output',
    ...     demog_file='/path/to/demographics.csv',
    ...     ses_id='session01'
    ... )

    >>> # Process only specific subjects with compression
    >>> organize_dicom_files(
    ...     in_dic_dir='/path/to/input/dicoms',
    ...     out_dic_dir='/path/to/output',
    ...     ids_file='/path/to/subject_ids.txt',
    ...     boolcomp=True,
    ...     nthreds=4
    ... )
    """

    # Declaring global variables
    global pb, pb1, n_dics, n_comp, lock, subj_id, dicom_files

    # Detecting the number of cores to be used
    ncores = os.cpu_count()
    if nthreads == 0:
        nthreads = ncores
        if nthreads > 4:
            nthreads = nthreads - 4
        else:
            nthreads = 1

    # Listing the subject ids inside the dicom folder
    my_list = os.listdir(in_dic_dir)
    subj_ids = []
    for it in my_list:
        if nosub == False:
            if "sub-" in it:
                subj_ids.append(it)
        else:
            subj_ids.append(it)

    subj_ids.sort()

    # If subj_ids is empty do not continue
    if not subj_ids:
        print("No subjects found in the input directory")
        sys.exit()

    if ids_file != None:
        if os.path.isfile(ids_file):
            subj_ids = cltmisc.select_ids_from_file(subj_ids, ids_file)

        else:
            s_ids = ids_file.split(",")

            if nosub == False:
                temp_ids = [s.strip("sub-") for s in subj_ids]
                s_ids = cltmisc.list_intercept(s_ids, temp_ids)

            if not s_ids:
                s_ids = subj_ids
            else:
                s_ids = ["sub-" + s for s in s_ids]
            subj_ids = s_ids

    # Reading demographics
    demobool = False  # Boolean variable to use the demographics table for the session id definition
    if demog_file != None:
        if os.path.isfile(demog_file):
            demobool = True  # Boolean variable to use the demographics table for the session id definition
            demoDB = pd.read_csv(demog_file)

    all_ser_dirs = []
    cont_subj = 0
    n_subj = len(subj_ids)
    failed_ids = []
    # Creating the progress bars
    with Progress() as pb:
        pb2 = pb.add_task("[green]Subjects...", total=n_subj)

        for cont_subj, subj_id in enumerate(subj_ids):  # Loop along the IDs

            # create a lock for the counter
            lock = Lock()

            n_comp = 0
            failed = []

            pb.update(
                task_id=pb2,
                description=f"[green]Subject: {subj_id} ({cont_subj+1}/{n_subj})",
                completed=cont_subj + 1,
            )

            subj_dir = os.path.join(in_dic_dir, subj_id)
            if os.path.isdir(subj_dir):
                # Default value for these variables for each subject
                gendVar = "Unknown"
                groupVar = "Unknown"
                AgeatScan = "Unknown"
                subTB = None
                date_times = []

                if demobool:
                    # Sub-table containing only the selected ID
                    subTB = demoDB[
                        demoDB["participant_id"].str.contains(subj_id.split("-")[-1])
                    ]

                    # Date times of all the series acquired for the current subject
                    nrows = np.shape(subTB)[0]
                    for nr in np.arange(0, nrows):
                        temp = subTB.iloc[nr]["acq_date"]
                        tempVar = temp.split("/")
                        date_time = datetime(
                            day=int(tempVar[1]),
                            month=int(tempVar[0]),
                            year=int(tempVar[2]),
                        )
                        date_times.append(date_time)
                try:
                    if booldic:
                        dicom_files = cltmisc.get_all_files(subj_dir)
                        ses_idprev = []
                        ser_idprev = []

                        n_dics = len(dicom_files)
                        if nthreads == 1:

                            pb1 = pb.add_task(
                                f"[red]Copying DICOMs: Subject {subj_id} ({cont_subj + 1}/{n_subj}) ",
                                total=n_dics,
                            )
                            for cont_dic, dfiles in enumerate(dicom_files):
                                ser_dir = copy_dicom_file(
                                    dfiles,
                                    subj_id,
                                    out_dic_dir,
                                    ses_id,
                                    date_times,
                                    demobool,
                                    subTB,
                                    force,
                                )
                                all_ser_dirs.append(ser_dir)
                                pb.update(
                                    task_id=pb1,
                                    description=f"[red]Copying DICOMs: Subject {subj_id} ({cont_dic+1}/{n_dics})",
                                    completed=cont_dic + 1,
                                )

                        else:

                            # create a progress bar for the subjects
                            pb1 = pb.add_task(
                                f"[red]Copying DICOMs: Subject {subj_id} ({cont_subj + 1}/{n_subj}) ",
                                total=n_dics,
                            )

                            # Adjusting the number of threads to the number of subjects
                            if n_dics < nthreads:
                                nthreads = n_dics

                            # start the thread pool
                            with ThreadPoolExecutor(nthreads) as executor:
                                # send in the tasks
                                # futures = [executor.submit(build_parcellation, t1s[i],
                                # bids_dir, deriv_dir, parccode, growwm) for i in range(n_subj)]

                                futures = [
                                    executor.submit(
                                        copy_dicom_file,
                                        dicom_files[i],
                                        subj_id,
                                        out_dic_dir,
                                        ses_id,
                                        date_times,
                                        demobool,
                                        subTB,
                                        force,
                                    )
                                    for i in range(n_dics)
                                ]
                                # futures = [executor.submit(test, i) for i in range(n_dics)]

                                # register the progress indicator callback
                                for future in futures:
                                    future.add_done_callback(progress_indicator)
                                # wait for all tasks to complete

                    else:

                        for ses_id in os.listdir(subj_dir):  # Loop along the session
                            ses_dir = os.path.join(subj_dir, ses_id)
                            if not ses_id[-2].isalpha():
                                if (
                                    demobool
                                ):  # Adding the Visit ID to the last part o the session ID only in the DICOM Folder
                                    tempVar = ses_id.split("-")[-1]
                                    sdate_time = datetime.strptime(
                                        tempVar, "%Y%m%d%H%M%S"
                                    )
                                    timediff = np.array(date_times) - np.array(
                                        sdate_time
                                    )
                                    clostd = np.argmin(abs(timediff))
                                    visitVar = subTB.iloc[clostd]["session_id"]
                                    newses_id = ses_id + visitVar
                                    newses_dir = os.path.join(subj_dir, newses_id)
                                    os.rename(ses_dir, newses_dir)
                                    ses_dir = newses_dir

                            if os.path.isdir(ses_dir):
                                for ser_id in os.listdir(
                                    ses_dir
                                ):  # Loop along the series
                                    serDir = os.path.join(ses_dir, ser_id)

                                    if os.path.isdir(serDir):
                                        all_ser_dirs.append(serDir)
                except:
                    failed_ids.append(subj_id)
                    print("Error at subject: " + subj_id)
            else:
                print("Subject: " + subj_id + " does not exist.")

        #     pb.update(task_id=t2, completed=cont_subj+1)
        # pb.update(task_id=t2, completed=n_subj)

    all_ser_dirs = list(set(all_ser_dirs))
    all_ser_dirs.sort()

    if boolcomp:
        compress_dicom_session(out_dic_dir)


####################################################################################################
def copy_dicom_file(
    dic_file: str,
    subj_id: str,
    out_dic_dir: str,
    ses_id: str = None,
    date_times: list = None,
    demogbool: bool = False,
    demog_tab: pd.DataFrame = None,
    force: bool = False,
):
    """
    Function to copy the DICOM files to the output directory.

    Parameters
    -----------
    dic_file: str
        Path to the DICOM file.

    subj_id: str
        Subject ID.

    out_dic_dir: str
        Output directory where the DICOM files will be saved.

    ses_id: str
        Session ID to be added to the session name. If not provided, the session ID will be the date of the study or the session ID
        extracted from the demographics table.

    date_times: list
        List containing the date and time of all the studies for that subject ID.

    demogbool: bool
        Boolean variable to use the demographics table for the session id definition.

    demog_tab: pd.DataFrame
        Demographics table containing the information about the subjects.

    force: bool
        Boolean variable to force the copy of the DICOM file.

    Returns
    --------
    dest_dic_dir: str
        Destination directory where the DICOM file was copied.


    """

    try:
        dataset = pydicom.dcmread(dic_file)
        dic_path = os.path.dirname(dic_file)
        dic_name = os.path.basename(dic_file)

        # Extracting the study date from DICOM file
        attributes = dataset.dir("")

        if attributes:
            sdate = dataset.data_element("StudyDate").value
            stime = dataset.data_element("StudyTime").value
            year = int(sdate[:4])
            month = int(sdate[4:6])
            day = int(sdate[6:8])

            # Date format
            sdate_time = datetime(day=day, month=month, year=year)

            # Creating default current Session ID
            ses_id, ser_id = create_session_series_names(dataset)

            if not ses_id == None:
                ses_id = "ses-" + ses_id

            if "000000" in ses_id and ser_id in ser_idprev:
                ses_id = ses_idprev

            # visitId = dfiles.split('/')[8].split('-')[1]
            # ses_id = 'ses-'+ visitId

            ses_idprev = ses_id
            ser_idprev = ser_id

            # Changing the session Id in case we have access to the demographics file
            if demogbool:
                timediff = np.array(date_times) - np.array(sdate_time)
                clostd = np.argmin(abs(timediff))
                visitVar = demog_tab.iloc[clostd]["session_id"]
                ses_id = ses_id + visitVar

            dest_dic_dir = os.path.join(out_dic_dir, subj_id, ses_id, ser_id)

            # Create the destination path
            if not os.path.isdir(dest_dic_dir):
                path = Path(dest_dic_dir)
                path.mkdir(parents=True, exist_ok=True)
            #                     print(newPath)
            dest_dic = os.path.join(dest_dic_dir, dic_name)
            if force:
                if os.path.isfile(dest_dic):
                    os.remove(dest_dic)
                else:
                    copyfile(dic_file, dest_dic)
            else:
                if not os.path.isfile(dest_dic):
                    copyfile(dic_file, dest_dic)

    except pydicom.errors.InvalidDicomError:
        print("Error at file at path :  " + dic_file)
    pass

    return dest_dic_dir


####################################################################################################
def create_session_series_names(dataset):
    """
    Function to create names from a DICOM object.

    Parameters
    ----------
    dataset: pydicom.dataset.FileDataset
        DICOM dataset object.

    Returns
    -------
    ses_id: str
        Session ID.

    ser_id: str
        Series ID.

    """
    # % This function creates the session and the series name for a dicom object

    # Extracting the study date from DICOM file
    attributes = dataset.dir("")
    sdate = dataset.data_element("StudyDate").value
    stime = dataset.data_element("StudyTime").value

    ########### ========== Creating current Session ID
    if sdate and stime:
        ses_id = str(sdate) + str(int(np.floor(float(stime))))
    elif sdate and not stime:
        ses_id = str(sdate) + "000000"
    elif stime and not sdate:
        ses_id = "00000000" + str(stime)

    ########### ========== Creating current Series ID
    if any("SeriesDescription" in s for s in attributes):
        ser_id = dataset.data_element("SeriesDescription").value
    elif any("SeriesDescription" in s for s in attributes) == False and any(
        "SequenceName" in s for s in attributes
    ):
        ser_id = dataset.data_element("SequenceName").value
    elif (
        any("SeriesDescription" in s for s in attributes) == False
        and any("SequenceName" in s for s in attributes) == False
        and any("ProtocolName" in s for s in attributes)
    ):
        ser_id = dataset.data_element("ProtocolName").value
    elif (
        any("SeriesDescription" in s for s in attributes) == False
        and any("SequenceName" in s for s in attributes) == False
        and any("ProtocolName" in s for s in attributes) == False
        and any("ScanningSequence" in s for s in attributes)
        and any("SequenceVariant" in s for s in attributes)
    ):
        ser_id = (
            dataset.data_element("ScanningSequence").value
            + "_"
            + dataset.data_element("SequenceVariant").value
        )
    else:
        ser_id = "NoSerName"

    # Removing and substituting unwanted characters
    ser_id = ser_id.replace(" ", "_")
    ser_id = ser_id.replace("/", "_")

    # This function removes some characters from a string
    ser2rem = [
        "*",
        "+",
        "(",
        ")",
        "=",
        ",",
        ">",
        "<",
        ";",
        ":",
        '"',
        "'",
        "?",
        "!",
        "@",
        "#",
        "$",
        "%",
        "^",
        "&",
        "*",
    ]
    for cad in ser2rem:
        ser_id = ser_id.replace(cad, "")

    # Removing the dupplicated _ characters and replacing the remaining by -
    ser_id = cltmisc.rem_duplicate_char(ser_id, "_")
    ser_id = ser_id.replace("_", "-")

    if any("SeriesNumber" in s for s in attributes):
        serNumb = dataset.data_element("SeriesNumber").value

    # Adding the series number
    sNumb = f"{int(serNumb):04d}"
    ser_id = sNumb + "-" + ser_id

    return ses_id, ser_id


####################################################################################################
def uncompress_dicom_session(
    dic_dir: str,
    boolrmtar: bool = False,
    subj_ids: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    """
    Uncompress session folders containing the DICOM files for all the series.

    Parameters
    ----------
    dic_dir : str
        Directory containing the subjects. It assumes an organization in:
        <subj_id>/<session_id>/<series_id>

    boolrmtar : bool, optional, default=False
        Boolean variable to remove the tar files after uncompressing the session.

    subj_ids : str, list of str, or None, optional
        Subject IDs to be considered. Can be:
        - None: consider all subjects in the directory (default)
        - str: path to text file containing subject IDs (one per line)
        - list of str: explicit list of subject IDs

    Returns
    -------
    list of str
        List of tar files that failed to be uncompressed. Empty list if all successful.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.

    ValueError
        If subj_ids is not None, str, or list, or if subject IDs file cannot be read.

    tarfile.TarError
        If there are issues with reading or extracting tar files.

    PermissionError
        If there are insufficient permissions to extract files or remove tar archives.

    OSError
        If there are filesystem-related errors during extraction.

    Examples
    --------
    >>> # Basic usage - uncompress all sessions in directory
    >>> failed = uncompress_dicom_session('/path/to/dicom/directory')
    >>> if not failed:
    ...     print("All sessions uncompressed successfully")

    >>> # Uncompress sessions and remove tar files after extraction
    >>> failed = uncompress_dicom_session('/path/to/dicom/directory', boolrmtar=True)

    >>> # Uncompress sessions for specific subjects only
    >>> failed = uncompress_dicom_session(
    ...     dic_dir='/path/to/dicom/directory',
    ...     subj_ids=['sub-001', 'sub-002', 'sub-003']
    ... )

    >>> # Use subject IDs from file
    >>> failed = uncompress_dicom_session(
    ...     dic_dir='/path/to/dicom/directory',
    ...     subj_ids='/path/to/subject_ids.txt',
    ...     boolrmtar=True
    ... )
    """

    # Validate input directory
    dic_path = Path(dic_dir)
    if not dic_path.exists():
        raise FileNotFoundError(f"Directory {dic_dir} does not exist")
    if not dic_path.is_dir():
        raise ValueError(f"{dic_dir} is not a directory")

    # Process subject IDs
    if subj_ids is None:
        # Get all subjects with 'sub-' prefix
        subj_ids = [
            item.name
            for item in dic_path.iterdir()
            if item.is_dir() and item.name.startswith("sub-")
        ]
        subj_ids.sort()
    elif isinstance(subj_ids, str):
        # Read subject IDs from file
        try:
            with open(subj_ids, "r", encoding="utf-8") as file:
                subj_ids = [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Subject IDs file {subj_ids} not found")
        except Exception as e:
            raise ValueError(f"Error reading subject IDs file: {e}")
    elif isinstance(subj_ids, list):
        # Validate list elements
        if not all(isinstance(subj_id, str) for subj_id in subj_ids):
            raise ValueError("All subject IDs must be strings")
    else:
        raise ValueError("subj_ids must be None, str (file path), or list of str")

    if not subj_ids:
        print("No subjects found to process")
        return []

    n_subj = len(subj_ids)
    failed_sessions = []

    with Progress() as pb:
        task = pb.add_task("[green]Uncompressing sessions...", total=n_subj)

        for i, subj_id in enumerate(subj_ids):
            subj_dir = dic_path / subj_id

            pb.update(
                task_id=task,
                description=f"[green]Processing {subj_id} ({i+1}/{n_subj})",
                completed=i,
            )

            # Skip if subject directory doesn't exist
            if not subj_dir.exists():
                print(f"Warning: Subject directory {subj_dir} not found, skipping...")
                continue

            # Find all tar.gz files in subject directory
            tar_files = list(subj_dir.glob("*.tar.gz"))

            for tar_file in tar_files:
                try:
                    # Use Python's tarfile module for better error handling
                    with tarfile.open(tar_file, "r:gz") as tar:
                        # Extract to subject directory
                        tar.extractall(path=subj_dir)

                    # Remove tar file if requested
                    if boolrmtar:
                        tar_file.unlink()

                except tarfile.TarError as e:
                    print(f"Error extracting {tar_file}: {e}")
                    failed_sessions.append(str(tar_file))
                except PermissionError as e:
                    print(f"Permission error with {tar_file}: {e}")
                    failed_sessions.append(str(tar_file))
                except Exception as e:
                    print(f"Unexpected error with {tar_file}: {e}")
                    failed_sessions.append(str(tar_file))

        pb.update(
            task_id=task,
            description=f"[green]Completed uncompression",
            completed=n_subj,
        )

    # Report results
    if failed_sessions:
        print("\nTHE PROCESS FAILED TO UNCOMPRESS THE FOLLOWING TAR FILES:")
        for failed_file in failed_sessions:
            print(f"  - {failed_file}")
    else:
        print("\nAll sessions uncompressed successfully!")

    print(f"\nProcessed {n_subj} subjects with {len(failed_sessions)} failures.")
    return failed_sessions


####################################################################################################
def compress_dicom_session(
    dic_dir: str,
    subj_ids: Optional[Union[str, List[str]]] = None,
    remove_original: bool = True,
) -> List[str]:
    """
    Compress session folders containing DICOM files into tar.gz archives.

    Parameters
    ----------
    dic_dir : str
        Directory containing the subjects. It assumes an organization in:
        <subj_id>/<session_id>/<series_id>

    subj_ids : str, list of str, or None, optional
        Subject IDs to be considered. Can be:
        - None: consider all subjects in the directory (default)
        - str: path to text file containing subject IDs (one per line)
        - list of str: explicit list of subject IDs

    remove_original : bool, optional, default=True
        Whether to remove the original session directories after successful compression.

    Returns
    -------
    list of str
        List of session directories that failed to be compressed. Empty list if all successful.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.

    ValueError
        If subj_ids is not None, str, or list, or if subject IDs file cannot be read.

    tarfile.TarError
        If there are issues with creating or writing tar files.

    PermissionError
        If there are insufficient permissions to compress files or remove directories.

    OSError
        If there are filesystem-related errors during compression.

    Examples
    --------
    >>> # Basic usage - compress all sessions in directory
    >>> failed = compress_dicom_session('/path/to/dicom/directory')
    >>> if not failed:
    ...     print("All sessions compressed successfully")

    >>> # Compress sessions but keep original directories
    >>> failed = compress_dicom_session(
    ...     dic_dir='/path/to/dicom/directory',
    ...     remove_original=False
    ... )

    >>> # Compress sessions for specific subjects only
    >>> failed = compress_dicom_session(
    ...     dic_dir='/path/to/dicom/directory',
    ...     subj_ids=['sub-001', 'sub-002', 'sub-003']
    ... )

    >>> # Use subject IDs from file
    >>> failed = compress_dicom_session(
    ...     dic_dir='/path/to/dicom/directory',
    ...     subj_ids='/path/to/subject_ids.txt'
    ... )
    """
    # Validate input directory
    dic_path = Path(dic_dir)
    if not dic_path.exists():
        raise FileNotFoundError(f"Directory {dic_dir} does not exist")
    if not dic_path.is_dir():
        raise ValueError(f"{dic_dir} is not a directory")

    # Process subject IDs
    if subj_ids is None:
        # Get all subjects with 'sub-' prefix
        subj_ids = [
            item.name
            for item in dic_path.iterdir()
            if item.is_dir() and item.name.startswith("sub-")
        ]
        subj_ids.sort()
    elif isinstance(subj_ids, str):
        # Read subject IDs from file
        try:
            with open(subj_ids, "r", encoding="utf-8") as file:
                subj_ids = [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Subject IDs file {subj_ids} not found")
        except Exception as e:
            raise ValueError(f"Error reading subject IDs file: {e}")
    elif isinstance(subj_ids, list):
        # Validate list elements
        if not all(isinstance(subj_id, str) for subj_id in subj_ids):
            raise ValueError("All subject IDs must be strings")
    else:
        raise ValueError("subj_ids must be None, str (file path), or list of str")

    if not subj_ids:
        print("No subjects found to process")
        return []

    n_subj = len(subj_ids)
    failed_sessions = []
    total_sessions = 0
    compressed_sessions = 0

    with Progress() as pb:
        task = pb.add_task("[green]Compressing sessions...", total=n_subj)

        for i, subj_id in enumerate(subj_ids):
            subj_dir = dic_path / subj_id

            pb.update(
                task_id=task,
                description=f"[green]Processing {subj_id} ({i+1}/{n_subj})",
                completed=i,
            )

            # Skip if subject directory doesn't exist
            if not subj_dir.exists():
                print(f"Warning: Subject directory {subj_dir} not found, skipping...")
                continue

            # Find all session directories (starting with 'ses-')
            session_dirs = [
                item
                for item in subj_dir.iterdir()
                if item.is_dir() and item.name.startswith("ses-")
            ]

            total_sessions += len(session_dirs)

            for ses_dir in session_dirs:
                tar_file_path = ses_dir.with_suffix(".tar.gz")

                # Skip if tar file already exists
                if tar_file_path.exists():
                    print(f"Warning: {tar_file_path} already exists, skipping...")
                    continue

                try:
                    # Create tar.gz archive using Python's tarfile module
                    with tarfile.open(tar_file_path, "w:gz") as tar:
                        # Add the session directory to the archive
                        # Use arcname to preserve the directory structure
                        tar.add(ses_dir, arcname=ses_dir.name)

                    # Remove original directory if requested and compression succeeded
                    if remove_original:
                        shutil.rmtree(ses_dir)

                    compressed_sessions += 1

                except tarfile.TarError as e:
                    print(f"Error compressing {ses_dir}: {e}")
                    failed_sessions.append(str(ses_dir))
                    # Clean up partially created tar file
                    if tar_file_path.exists():
                        try:
                            tar_file_path.unlink()
                        except Exception:
                            pass
                except PermissionError as e:
                    print(f"Permission error with {ses_dir}: {e}")
                    failed_sessions.append(str(ses_dir))
                except Exception as e:
                    print(f"Unexpected error with {ses_dir}: {e}")
                    failed_sessions.append(str(ses_dir))

        pb.update(
            task_id=task, description=f"[green]Completed compression", completed=n_subj
        )

    # Report results
    if failed_sessions:
        print("\nTHE PROCESS FAILED TO COMPRESS THE FOLLOWING SESSIONS:")
        for failed_session in failed_sessions:
            print(f"  - {failed_session}")
    else:
        print("\nAll sessions compressed successfully!")

    print(
        f"\nProcessed {n_subj} subjects, {compressed_sessions}/{total_sessions} sessions compressed successfully."
    )
    return failed_sessions
