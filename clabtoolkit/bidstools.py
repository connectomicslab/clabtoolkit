import os
import shutil
import pandas as pd
import clabtoolkit.misctools as cltmisc
from typing import Union, Dict, List
import re
import json
from glob import glob

from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    TextColumn,
    MofNCompleteColumn
)

####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############           Methods dedicated to work with BIDs naming conventions           ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def str2entity(string: str) -> dict:
    """
    Converts a formatted string into a dictionary.

    Parameters
    ----------
    string : str
        String to convert, with the format `key1-value1_key2-value2...suffix.extension`.

    Returns
    -------
    dict
        Dictionary containing the entities extracted from the string.

    Examples
    --------
    >>> str2entity("sub-01_ses-M00_acq-3T_dir-AP_run-01_T1w.nii.gz")
    Returns: {'sub': '01', 'ses': 'M00', 'acq': '3T', 'dir': 'AP', 'run': '01', 'suffix': 'T1w', 'extension': 'nii.gz'}

    """
    ent_dict = {}
    suffix, extension = "", ""

    # Split the string into entities based on underscores.
    ent_list = string.split("_")

    # Detect suffix and extension
    for ent in ent_list[:]:
        if "-" not in ent:
            # If entity does not contain a '-', it's a suffix or extension.
            if "." in ent:
                # Split suffix and extension parts
                suffix, extension = ent.split(".", 1)
            else:
                suffix = ent
            ent_list.remove(ent)

    # Process the remaining entities
    for ent in ent_list:
        key, value = ent.split("-", 1)  # Split each entity on the first "-"
        ent_dict[key] = value

    # Add suffix and extension to the dictionary if they were found
    if suffix:
        ent_dict["suffix"] = suffix
    if extension:
        ent_dict["extension"] = extension

    return ent_dict


####################################################################################################
def entity2str(entity: dict) -> str:
    """
    Converts an entity dictionary to a string representation.

    Parameters
    ----------
    entity : dict
        Dictionary containing the entities.

    Returns
    -------
    str
        String containing the entities in the format `key1-value1_key2-value2...suffix.extension`.

    Examples
    --------
    >>> entity2str({'sub': '01', 'ses': 'M00', 'acq': '3T', 'dir': 'AP', 'run': '01', 'suffix': 'T1w', 'extension': 'nii.gz'})
    Returns: "sub-01_ses-M00_acq-3T_dir-AP_run-01_T1w.nii.gz"

    """
    # Make a copy of the entity dictionary to avoid mutating the original.
    entity = entity.copy()

    # Extract optional 'suffix' and 'extension' fields if present.
    suffix = entity.pop("suffix", "")
    extension = entity.pop("extension", "")

    # Construct the main part of the string by joining key-value pairs with '_'
    ent_string = "_".join(f"{key}-{str(value)}" for key, value in entity.items())

    # Append suffix if it exists
    if suffix:
        ent_string += "_" + suffix
    else:
        ent_string = ent_string.rstrip("_")  # Remove trailing underscore if no suffix

    # Append extension if it exists
    if extension:
        ent_string += f".{extension}"

    return ent_string


####################################################################################################
def delete_entity(
    entity: Union[dict, str], key2rem: Union[List[str], str]
) -> Union[dict, str]:
    """
    Removes specified keys from an entity dictionary or string representation.

    Parameters
    ----------
    entity : dict or str
        Dictionary or string containing the entities.
    key2rem : list or str
        Key(s) to remove from the entity dictionary or string.

    Returns
    -------
    Union[dict, str]
        The updated entity as a dictionary or string (matching the input type).

    Examples
    --------
    >>> delete_entity("sub-01_ses-M00_acq-3T_dir-AP_run-01_T1w.nii.gz", "acq")
    Returns: "sub-01_ses-M00_dir-AP_run-01_T1w.nii.gz"

    """
    # Determine if `entity` is a string and convert if necessary.
    is_string = isinstance(entity, str)
    if is_string:
        entity_out = str2entity(entity)
    elif isinstance(entity, dict):
        entity_out = entity.copy()
    else:
        raise ValueError("The entity must be a dictionary or a string.")

    # Ensure `key2rem` is a list for uniform processing.
    if isinstance(key2rem, str):
        key2rem = [key2rem]

    # Remove specified keys from the entity dictionary.
    for key in key2rem:
        entity_out.pop(key, None)  # `pop` with default `None` avoids KeyErrors.

    # Convert back to string format if original input was a string.
    if is_string:
        return entity2str(entity_out)

    return entity_out


####################################################################################################
def replace_entity_value(
    entity: Union[dict, str], ent2replace: dict, verbose: bool = False
) -> Union[dict, str]:
    """
    Replaces values in an entity dictionary or string representation.

    Parameters
    ----------
    entity : dict or str
        Dictionary or string containing the entities.

    ent2replace : dict
        Dictionary of entities to replace with new values.

    verbose : bool, optional
        If True, prints warnings for non-existent or empty values.

    Returns
    -------
    Union[dict, str]
        Updated entity as a dictionary or string (matching the input type).

    Examples
    --------
    >>> replace_entity_value("sub-01_ses-M00_acq-3T_dir-AP_run-01_T1w.nii.gz", {"acq": "7T"})
    Returns: "sub-01_ses-M00_acq-7T_dir-AP_run-01_T1w.nii.gz"

    """
    # Determine if `entity` is a string and convert if necessary.
    is_string = isinstance(entity, str)
    if is_string:
        entity_out = str2entity(entity)
    elif isinstance(entity, dict):
        entity_out = entity.copy()
    else:
        raise ValueError("The entity must be a dictionary or a string.")

    # Remove any empty keys or values from `ent2replace`.
    ent2replace = {k: v for k, v in ent2replace.items() if v}

    # Replace values in `entity_out` based on `ent2replace`.
    for key, new_value in ent2replace.items():
        if key in entity_out:
            if new_value:
                entity_out[key] = new_value
            elif verbose:
                print(f"Warning: Replacement value for '{key}' is empty.")
        elif verbose:
            print(f"Warning: Entity '{key}' not found in entity dictionary.")

    # Convert back to string format if original input was a string.
    if is_string:
        return entity2str(entity_out)

    return entity_out


####################################################################################################
def replace_entity_key(
    entity: Union[dict, str], keys2replace: Dict[str, str], verbose: bool = False
) -> Union[dict, str]:
    """
    Replaces specified keys in an entity dictionary or string representation.

    Parameters
    ----------
    entity : dict or str
        Dictionary containing the entities or a string that follows the BIDS naming specifications.

    keys2replace : dict
        Dictionary mapping old keys to new keys.

    verbose : bool, optional
        If True, prints warnings for keys in `keys2replace` that are not found in `entity`.

    Returns
    -------
    Union[dict, str]
        Updated entity as a dictionary or string (matching the input type).

    Examples
    --------
    >>> replace_entity_key("sub-01_ses-M00_acq-3T_dir-AP_run-01_T1w.nii.gz", {"acq": "TESTrep1", "dir": "TESTrep2"})
    Returns: "sub-01_ses-M00_TESTrep1-3T_TESTrep2-AP_run-01_T1w.nii.gz"

    """
    # Convert `entity` to a dictionary if it's a string
    is_string = isinstance(entity, str)
    if is_string:
        entity = str2entity(entity)
    elif not isinstance(entity, dict):
        raise ValueError("The entity must be a dictionary or a string.")

    # Validate that `keys2replace` is a dictionary
    if not isinstance(keys2replace, dict):
        raise ValueError("The keys2replace parameter must be a dictionary.")

    # Filter out any empty keys or values from `keys2replace`
    keys2replace = {k: v for k, v in keys2replace.items() if k and v}

    # Replace key names in the entity
    entity_out = {}
    for key, value in entity.items():
        # Use the new key if it exists in `keys2replace`, otherwise keep the original key
        new_key = keys2replace.get(key, key)
        entity_out[new_key] = value

        # Verbose output if the key to replace does not exist in the entity
        if verbose and key in keys2replace and key not in entity:
            print(f"Warning: Key '{key}' not found in the original dictionary.")

    # Convert back to string format if the original input was a string
    if is_string:
        return entity2str(entity_out)

    return entity_out


####################################################################################################
def insert_entity(
    entity: Union[dict, str], entity2add: Dict[str, str], prev_entity: str = None
) -> Union[dict, str]:
    """
    Adds entities to an existing entity dictionary or string representation.

    Parameters
    ----------
    entity : dict or str
        Dictionary containing the entities or a string that follows the BIDS naming specifications.

    entity2add : dict
        Dictionary containing the entities to add.

    prev_entity : str, optional
        Key in `entity` after which to insert the new entities.

    Returns
    -------
    Union[dict, str]
        Updated entity with the new entities added (matching the input type).

    Examples
    --------
    >>> insert_entity("sub-01_ses-M00_acq-3T_dir-AP_run-01_T1w.nii.gz", {"task": "rest"})
    Returns: "sub-01_ses-M00_acq-3T_dir-AP_run-01_task-rest_T1w.nii.gz"

    >>> insert_entity("sub-01_ses-M00_acq-3T_dir-AP_run-01_T1w.nii.gz", {"task": "rest"}, prev_entity="ses")
    Returns: "sub-01_ses-M00_task-rest_acq-3T_dir-AP_run-01_T1w.nii.gz"

    """

    # Determine if `entity` is a string and convert if necessary
    is_string = isinstance(entity, str)
    if is_string:
        entity = str2entity(entity)
    elif not isinstance(entity, dict):
        raise ValueError("The entity must be a dictionary or a string.")

    # Clean `entity2add` by removing any empty keys or values
    entity2add = {k: v for k, v in entity2add.items() if k and v}

    # Validate `prev_entity` if provided
    if prev_entity is not None and prev_entity not in entity:
        raise ValueError(
            f"Reference entity '{prev_entity}' is not in the entity dictionary."
        )

    # Temporarily remove `suffix` and `extension` if they exist
    suffix = entity.pop("suffix", None)
    extension = entity.pop("extension", None)

    # Build `ent_out` by adding items from `entity`, and insert `entity2add` after `prev_entity` if specified
    ent_out = {}
    for key, value in entity.items():
        ent_out[key] = value
        if key == prev_entity:
            ent_out.update(
                entity2add
            )  # Insert new entities immediately after `prev_entity`

    # If no `prev_entity` is specified or if `prev_entity` is "suffix", append `entity2add` at the end
    if prev_entity is None or prev_entity == "suffix":
        ent_out.update(entity2add)

    # Restore `suffix` and `extension` if they were removed
    if suffix:
        ent_out["suffix"] = suffix
    if extension:
        ent_out["extension"] = extension

    # Convert back to string format if the original input was a string
    if is_string:
        return entity2str(ent_out)

    return ent_out


####################################################################################################
def recursively_replace_entity_value(
    root_dir: str, dict2old: Union[dict, str], dict2new: Union[dict, str]
):
    """
    This method replaces the values of certain entities in all the files and folders of a BIDs dataset.

    Parameters:
    ----------
    root_dir: str
        Root directory of the BIDs dataset

    dict2old: dict or str
        Dictionary containing the entities to replace and their old values

    dict2new: dict or str
        Dictionary containing the entities to replace and their new values


    """

    # Detect if the BIDs directory exists
    if not os.path.isdir(root_dir):
        raise ValueError("The BIDs directory does not exist.")

    # Convert the strings to dictionaries
    if isinstance(dict2old, str):
        dict2old = str2entity(dict2old)
    if isinstance(dict2new, str):
        dict2new = str2entity(dict2new)

    # Leave in the dictionaries only the keys that are common
    dict2old = {k: dict2old[k] for k in dict2old if k in dict2new}
    dict2new = {k: dict2new[k] for k in dict2new if k in dict2old}

    # Order the dictionaries alphabetically by key
    dict2old = dict(sorted(dict2old.items()))
    dict2new = dict(sorted(dict2new.items()))

    # Creating the list of strings
    dict2old_list = [f"{key}-{value}" for key, value in dict2old.items()]
    dict2new_list = [f"{key}-{value}" for key, value in dict2new.items()]

    # Find all the files and folders that contain a certain string any of the key values in the dictionary dict2old

    # Walk through the directory from bottom to top (reverse)
    for root, dirs, files in os.walk(root_dir, topdown=False):
        # Rename files
        for file_name in files:

            for i, subst_x in enumerate(dict2old_list):
                subst_y = dict2new_list[i]
                if subst_x in file_name:
                    old_path = os.path.join(root, file_name)
                    new_name = file_name.replace(subst_x, subst_y)
                    new_path = os.path.join(root, new_name)
                    os.rename(old_path, new_path)
                    file_name = new_name

                # the file is the tsv open the file and replace the string
                if file_name.endswith("sessions.tsv"):
                    tsv_file = os.path.join(root, file_name)
                    # Read line by line and replace the string
                    # Load the TSV file
                    df = pd.read_csv(tsv_file, sep="\t")

                    # Replace subst_x with subst_y in all string columns
                    df = df.applymap(
                        lambda x: (
                            x.replace(subst_x, subst_y) if isinstance(x, str) else x
                        )
                    )

                    # Save the modified DataFrame as a TSV file
                    df.to_csv(tsv_file, sep="\t", index=False)

        # Rename directories
        for dir_name in dirs:
            if subst_x in dir_name:
                old_path = os.path.join(root, dir_name)
                new_name = dir_name.replace(subst_x, subst_y)
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                
                
####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############           Methods dedicated to work with BIDs file organization            ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################

def get_subjects(bids_dir: str) -> list:
    """
    Get a list of all subjects in the BIDs directory.
    
    Parameters
    ----------
    bids_dir : str
        Path to the BIDs directory.
        
    Returns
    -------
    list
        List of subject IDs.
        
    Usage example:
    >>> bids_dir = "/path/to/bids"
    >>> print(get_subjects(bids_dir))
    ["sub-01", "sub-02", ...]
    
    """
    subjects = []
    
    for root, dirs, files in os.walk(bids_dir):
        for dir_name in dirs:
            if dir_name.startswith("sub-"):
                subjects.append(dir_name)
    
    return subjects

####################################################################################################
def copy_bids_folder(
    bids_dir: str,
    out_dir: str,
    subjects_to_copy: Union[list, str] = None,
    folders_to_copy: Union[list, str] = "all",
    deriv_dir: str = None,
    include_derivatives: Union[str, list] = None,
):
    """
    This function copies the BIDs folder and its derivatives for given subjects to a new location.
    
    Parameters
    ----------
    bids_dir : str
        Path to the BIDs directory.
    out_dir : str
        Path to the output directory where the copied BIDs folder will be saved.
    subjects_to_copy : list or str, optional
        List of subject IDs to copy. If None, all subjects will be copied.
    folders_to_copy : list or str, optional
        List of BIDs folders to copy. If "all", all folders will be copied. Default is "all".
    deriv_dir : str, optional
        Path to the derivatives directory. If None, it will be set to "derivatives" in the BIDs directory.
    include_derivatives : str or list, optional
        List of derivatives to include. If "all", all derivatives will be included. Default is None.
        If None, no derivatives will be copied.
        If "chimera", only the chimera derivatives will be copied.
        If "all", all derivatives will be copied.
        If a list, only the derivatives in the list will be copied.
        If a string, only the derivatives with the name in the string will be copied.
        
    Returns
    -------
    None
        Copies the specified folders and subjects to the output directory.
        
    Usage example:
    >>> bids_dir = "/path/to/bids"
    >>> out_dir = "/path/to/output"
    >>> copy_bids_folder(bids_dir, out_dir, subjects_to_copy=["sub-01"], folders_to_copy=["anat"])
    >>> copy_bids_folder(bids_dir, out_dir, subjects_to_copy=["sub-01"], include_derivatives=["chimera", "freesurfer"])
    >>> copy_bids_folder(bids_dir, out_dir, subjects_to_copy=["sub-01"], deriv_dir="/path/to/derivatives")
        
    """

    bids_dir = cltmisc.remove_trailing_separators(bids_dir)
    out_dir = cltmisc.remove_trailing_separators(out_dir)
    
    if not os.path.isdir(bids_dir):
        raise FileNotFoundError(f"The BIDs directory {bids_dir} does not exist.")
    
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"The output directory {out_dir} does not exist.")
    
    if deriv_dir is not None:
        deriv_dir = cltmisc.remove_trailing_separators(deriv_dir)
        if not os.path.isdir(deriv_dir):
            raise FileNotFoundError(f"The derivatives directory {deriv_dir} does not exist.")
    
    # Selecting the subjects that will be copied
    if isinstance(subjects_to_copy, str):
        subjects_to_copy = [subjects_to_copy]
        
    if subjects_to_copy is None:
        subjects_to_copy = get_subjects(bids_dir)
    
    # Selecting the BIDs folders that will be copied
    if isinstance(folders_to_copy, str):
        folders_to_copy = [folders_to_copy]
        
    if "all" in folders_to_copy:
        folders_to_copy = ["all"]
    
    # Number of subjects to copy
    n_subj = len(subjects_to_copy)
    
    # Selecting the derivatives folder
    if include_derivatives is not None:
        copy_derivatives = True
        
        if deriv_dir is None:
            deriv_dir = os.path.join(bids_dir, "derivatives")

        if not os.path.isdir(deriv_dir):
            # Lunch a warning message if the derivatives folder does not exist
            print("WARNING: The derivatives folder does not exist.")
            print("WARNING: The derivatives folder will not be copied.")
            copy_derivatives = False
        else:
            # Check if the derivatives folder is empty
            if len(os.listdir(deriv_dir)) == 0:
                print("WARNING: The derivatives folder is empty.")
                print("WARNING: The derivatives folder will not be copied.")
                copy_derivatives = False

        # Selecting all the derivatives folders
        directories = os.listdir(deriv_dir)
        der_pipe_folders = []
        for directory in directories:
            pipe_dir = os.path.join(deriv_dir, directory)
            if not directory.startswith(".") and os.path.isdir(pipe_dir):
                der_pipe_folders.append(pipe_dir)
        
        if isinstance(include_derivatives, str):
            include_derivatives = [include_derivatives]
        
        if "all" not in include_derivatives:
            include_derivatives = [os.path.join(deriv_dir, i) for i in include_derivatives]
            der_pipe_folders = cltmisc.list_intercept(der_pipe_folders, include_derivatives)
            
        if len(der_pipe_folders) == 0:
            print("WARNING: No derivatives folders were found with the specified names.")
            copy_derivatives = False
            
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        MofNCompleteColumn(),
    )
    with progress:
        task = progress.add_task(
            "[cyan]Copying BIDs folder and derivatives...", total=None
        )
        progress.update(task, total=n_subj)
        progress.start_task(task)
        progress.update(task, completed=0)
            
        # Loop around all the subjects
        for i, full_id in enumerate(subjects_to_copy):  # Loop along the IDs

            # Extract the subject id
            subj_entity = str2entity(full_id)
            
            # Remove the extension from the subj_entity
            if "extension" in subj_entity.keys():
                del subj_entity["extension"]
            
            if "suffix" in subj_entity.keys():
                suffix = subj_entity["suffix"]
                del subj_entity["suffix"]
                
            else:
                suffix = None
                
            # Generate a list with the format [key_value]
            file_filter = [f"{k}-{v}" for k, v in subj_entity.items()]
            fold_filter = [f"{k}-{v}" for k, v in subj_entity.items()]
            
            if suffix is not None:
                file_filter.append(f"_{suffix}")
            
            if subj_entity is None:
                print(f"WARNING: The subject ID {full_id} is not valid.")
                continue
            
            subj_id = subj_entity["sub"]
            
            subj_dir = os.path.join(bids_dir, f'sub-{subj_id}')
            all_subj_subfold = cltmisc.detect_leaf_directories(subj_dir)
            
            # Upgrade the progress bar and include the subject id on the text 
            progress.update(task, description=f"[red]Copying subject: {full_id}", completed=i+1)
            
            # Detect the session id if it exists
            ses_id = None
            if "ses" in subj_entity.keys():
                ses_id = subj_entity["ses"]
                full_ses_id = f'ses-{ses_id}'
                
                all_subj_subfold = cltmisc.filter_by_substring(input_list=all_subj_subfold, or_filter=full_ses_id)
            
            #######  Copying the BIDs folder
            if "all" not in folders_to_copy:
                tmp_list = [ os.path.basename(i) for i in all_subj_subfold]
                indexes = cltmisc.get_indexes_by_substring(tmp_list, folders_to_copy)
                all_subj_subfold = [all_subj_subfold[i] for i in indexes]
                
            raw_files_to_copy = []
            # Loop along all the folders to copy
            for fc in all_subj_subfold:
                all_subj_raw_files = cltmisc.detect_recursive_files(fc)
                
                raw_files_to_copy = raw_files_to_copy + all_subj_raw_files
            
            # Filtering the files to copy according to the full
            raw_files_to_copy =  cltmisc.filter_by_substring(input_list=raw_files_to_copy, or_filter = f'sub-{subj_id}', and_filter=file_filter)

            for file in raw_files_to_copy:
                if os.path.isfile(file):
                    # Copying the file
                    try:
                        dest_dir = file.replace(bids_dir, out_dir)
                        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
                        shutil.copy2(file, dest_dir)
                    except:
                        print(f"WARNING: The file {file} could not be copied.")
                        continue
            
            # Copying the Derivatives folder
            if copy_derivatives:
                # Loop along all the derivatives folders
                for pipe_dir in der_pipe_folders:
                    # Check if the derivatives folder exists
                    if os.path.isdir(pipe_dir):
                        # Copying the derivatives folder
                        subj_deriv = glob(os.path.join(pipe_dir, f'sub-{subj_id}*'))
                        
                        if len(subj_deriv) > 0:
                            
                            # Detecting the files and folders to copy.
                            # We are doing this separation because there are derivatives that contain
                            # only the folders with the name of the subject and not the files (i.e. freesurfer)
                            
                            # Loop along all the folders to copy
                            deriv_files_to_copy = []
                            deriv_folds_to_copy = []
                            for deriv in subj_deriv:
                                
                                # Detect the files and folders to copy
                                tmp_fold_to_copy =  cltmisc.filter_by_substring(input_list=deriv, or_filter = f'sub-{subj_id}', and_filter=fold_filter)

                                if len(tmp_fold_to_copy) > 0:
                                    # Detect the files and folders to copy
                                    for tmp in tmp_fold_to_copy:
                                        if os.path.isdir(tmp):
                                            deriv_folds_to_copy.append(tmp)
                                        else:
                                            deriv_files_to_copy.append(tmp)
                                else:
                                    all_subj_deriv_files = cltmisc.detect_recursive_files(deriv)
                                    
                                    # Filtering the files to copy according to the full
                                    all_subj_deriv_files =  cltmisc.filter_by_substring(input_list=all_subj_deriv_files, or_filter = f'sub-{subj_id}', and_filter=fold_filter)

                                    deriv_files_to_copy = deriv_files_to_copy + all_subj_deriv_files

                            # Copying the folders
                            if len(deriv_folds_to_copy) > 0:
                                for fold in deriv_folds_to_copy:
                                    try:
                                        dest_dir = fold.replace(deriv_dir, os.path.join(out_dir, "derivatives"))
                                        os.makedirs(dest_dir, exist_ok=True)
                                        shutil.copytree(fold, dest_dir, dirs_exist_ok=True)
                                    except:
                                        print(f"WARNING: The folder or file {fold} could not be copied.")
                                        continue
                            
                            # Copying the files
                            if len(deriv_files_to_copy) > 0:
                                for file in deriv_files_to_copy:
                                    try:
                                        dest_dir = file.replace(deriv_dir, os.path.join(out_dir, "derivatives"))
                                        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
                                        shutil.copy2(file, dest_dir)
                                    except:
                                        print(f"WARNING: The file {file} could not be copied.")
                                        continue

        # Update the progress bar to 100%
        progress.update(task, completed=n_subj)


####################################################################################################


# def is_bids_filename(filename: str, json_file: str = None) -> bool:
#     """
#     Check if a given filename follows the Brain Imaging Data Structure (BIDS) format.
#     Automatically detects whether it's a derivative or raw filename.

#     Args:
#         filename (str): The filename to check.
#         json_file (str | dict, optional): Path to a JSON file containing BIDS patterns or a dictionary.

#     Returns:
#         bool: True if the filename is in BIDS format, False otherwise.
#     """

#     # Load JSON config
#     if json_file is None:
#         config_path = os.path.join(os.path.dirname(__file__), "config", "config.json")
#         with open(config_path) as f:
#             config_data = json.load(f)

#         # Extract patterns
#         valid_suffixes = (
#             config_data["bids_entities"]["raw_suffix"]
#             + config_data["bids_entities"]["derivatives_suffix"]
#         )
#     elif isinstance(json_file, str):
#         if not os.path.isfile(json_file):
#             raise ValueError(
#                 "Please provide a valid JSON file containing the entities dictionary."
#             )
#         with open(json_file) as f:
#             config_data = json.load(f)
#             valid_suffixes = config_data["bids_suffixes"]
#     elif isinstance(json_file, dict):
#         config_data = json_file
#         valid_suffixes = config_data["bids_suffixes"]
#     else:
#         raise TypeError("json_file must be a file path or a dictionary.")

#     # Regular expression to match BIDS filenames
#     # Remove extension

#     base_name = os.path.basename(filename)

#     ent_dict = entity2str(base_name)
#     ent_dict["extension"] = ""
#     ent_dict["suffix"] = ""

#     base_name = entity2str(ent_dict)

#     parts = base_name.split("_")

#     if len(parts) < 2:
#         return False  # At least one key-value pair and a suffix are needed

#     key_value_pattern = re.compile(r"^[a-zA-Z0-9]+-[a-zA-Z0-9]+$")

#     # Check key-value pairs (all except the last part)
#     for part in parts:
#         if not key_value_pattern.match(part):
#             return False  # Invalid key-value pair

#     return False  # Invalid if the last part is not a known suffix


def is_bids_filename(filename: str) -> bool:
    """
    Validates a BIDS filename structure, handling extensions and entity order.

    Args:
        filename (str): The filename to validate.

    Returns:
        bool: True if valid BIDS filename, False otherwise.
    """
    # Remove extension if present
    base_filename = filename.split(".")[0]

    parts = base_filename.split("_")
    if not parts:
        return False

    entity_pattern = re.compile(r"^[a-zA-Z0-9]+-[a-zA-Z0-9]+$")

    # Check that at least one entity-label pair is present
    has_entity_label = False
    for part in parts:
        if "-" in part:
            has_entity_label = True
            if not entity_pattern.match(part):
                return False
    if not has_entity_label:
        return False

    return True
