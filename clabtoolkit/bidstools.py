import os
import shutil
import pandas as pd
import time
import queue
import threading
import numpy as np

from typing import Union, Dict, Set, Tuple, List, Optional
from pathlib import Path

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


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############     Section 1: Methods dedicated to work with BIDs naming conventions      ############
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
    entity: Union[dict, str], ent2rem: Union[List[str], str, dict]
) -> Union[dict, str]:
    """
    Removes specified keys from an entity dictionary or string representation.

    Parameters
    ----------
    entity : dict or str
        Dictionary or string containing the entities.

    ent2rem : List[str], str or dict
        Entities to be removed from the entity dictionary or string. If ent2rem is a dictionary,
        only the combination key-value will be removed from the filenames.

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
    rem_is_dict = False  # Boolean variable in case you want to delete an entity with a certain value and the key2remis a dictionary
    if is_string:
        entity_out = str2entity(entity)
    elif isinstance(entity, dict):
        entity_out = entity.copy()
    else:
        raise ValueError("The entity must be a dictionary or a string.")

    # Ensure `key2rem` is a list for uniform processing.
    if isinstance(ent2rem, str):
        key2rem = [ent2rem]

    elif isinstance(ent2rem, list):
        key2rem = list(set(ent2rem))  # Convert to a set for unique keys

    elif isinstance(ent2rem, dict):
        rem_is_dict = True
        key2rem = list(ent2rem.keys())

    else:
        raise ValueError(
            "The ent2rem parameter must be a string, list of strings, or dictionary."
        )

    # Remove specified keys from the entity dictionary.
    for key in key2rem:
        if rem_is_dict:
            # If `key2rem` is a dictionary, check if the key exists and has the specified value.
            if key in entity_out and entity_out[key] in ent2rem[key]:
                entity_out.pop(key, None)
        else:
            entity_out.pop(key, None)  # `pop` with default `None` avoids KeyErrors.

    # Convert back to string format if original input was a string.
    if is_string:
        return entity2str(entity_out)

    return entity_out


####################################################################################################
def replace_entity_value(
    entity: Union[dict, str], ent2replace: Union[dict, str], verbose: bool = False
) -> Union[dict, str]:
    """
    Replaces values in an entity dictionary or string representation.

    Parameters
    ----------
    entity : dict or str
        Dictionary or string containing the entities.

    ent2replace : dict or str
        Dictionary or string containing entities to replace with new values.

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

    # Adding the possibility to enter a string value. It will convert it to a dictionary
    if isinstance(ent2replace, str):
        ent2replace = str2entity(ent2replace)

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
        IMPORTANT: If the `entity2add` contains keys that already exist in the `entity`, they will not be added.

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

    # Check if there are some entities already on the name and removing those items from
    # the dictionary
    if len(cltmisc.list_intercept(list(entity2add.keys()), list(entity.keys()))) > 0:
        # Removes the items from `entity2add` that already exist in `entity`
        entity2add = {k: v for k, v in entity2add.items() if k not in entity}

    # If `entity2add` is empty after filtering, return the original entity
    if not entity2add:
        if is_string:
            return entity2str(entity)
        return entity

    else:
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

    Parameters
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

    replacements = dict(zip(dict2old_list, dict2new_list))

    #
    all_files = cltmisc.get_all_files(
        root_dir, or_filter=dict2old_list[0], and_filter=dict2old_list
    )

    if not all_files:
        print(
            "No files found that match the specified entities. Please check the input parameters."
        )
        return

    else:
        all_dirs = []
        for file in all_files:
            file_path = os.path.dirname(file)
            file_name = os.path.basename(file)
            all_dirs.append(file_path)

            for i, subst_x in enumerate(dict2old_list):
                subst_y = dict2new_list[i]
                if subst_x in file_name:
                    old_path = os.path.join(file_path, file_name)
                    new_name = file_name.replace(subst_x, subst_y)
                    new_path = os.path.join(file_path, new_name)
                    os.rename(old_path, new_path)
                    file_name = (
                        new_name  # Update old_path to the new path after renaming
                    )

        all_dirs = set(all_dirs)  # Remove duplicates from the directory list

        # Renaming the directories
        cltmisc.rename_folders(all_dirs, replacements)


####################################################################################################
def recursively_replace_entity_key(root_dir: str, replacements: dict):
    """
    This method replaces the keys of certain entities in all the files and folders of a BIDs dataset.

    Parameters
    ----------
    root_dir: str
        Root directory of the BIDs dataset

    replacements: dict
        Dictionary containing the entities to replace and their new keys.
        Example: {'acq': 'desc', 'run': 'runny'}

    Returns
    -------
    None
        The method will rename the files and folders in the BIDs dataset. All the files or folders containing the old
        entities' names on their names will be renamed and the old entities will be replaced with the new entities.

    """
    # Detect if the BIDs directory exists
    if not os.path.isdir(root_dir):
        raise ValueError("The BIDs directory does not exist.")

    old_keys = list(replacements.keys())
    new_keys = list(replacements.values())

    all_files = cltmisc.get_all_files(
        root_dir, or_filter=old_keys[0], and_filter=old_keys
    )

    if not all_files:
        print(
            "No files found that match the specified entities. Please check the input parameters."
        )
        return

    else:
        all_dirs = []
        for file in all_files:
            file_path = os.path.dirname(file)
            file_name = os.path.basename(file)
            all_dirs.append(file_path)

            for i, subst_x in enumerate(old_keys):
                subst_y = new_keys[i]
                if subst_x in file_name:
                    old_path = os.path.join(file_path, file_name)
                    new_name = file_name.replace(subst_x, subst_y)
                    new_path = os.path.join(file_path, new_name)
                    os.rename(old_path, new_path)
                    file_name = (
                        new_name  # Update old_path to the new path after renaming
                    )

        all_dirs = set(all_dirs)  # Remove duplicates from the directory list

        # Renaming the directories
        cltmisc.rename_folders(all_dirs, replacements)


def recursively_delete_entity(root_dir: str, key2rem: Union[List[str], str, dict]):
    """
    This method deletes entities in all the files and folders of a BIDs dataset.

    Parameters
    ----------
    root_dir: str
        Root directory of the BIDs dataset

    key2rem: list or str
        Key(s) of the entities that will be removed from the files and folders.

    Returns
    -------
    None
        The method will rename the files and folders in the BIDs dataset, removing from file names and folder names the entities containing the specified keys.


    """

    # Detect if the BIDs directory exists
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        raise ValueError("The BIDs directory does not exist.")

    # Convert the strings to lists
    if isinstance(key2rem, str):
        key2rem = [key2rem]

    if isinstance(key2rem, dict):
        tmp_keys = list(key2rem.keys())
        all_files = cltmisc.get_all_files(
            root_dir, or_filter=tmp_keys[0], and_filter=tmp_keys
        )
    else:

        all_files = cltmisc.get_all_files(
            root_dir, or_filter=key2rem[0], and_filter=key2rem
        )

    if not all_files:
        print(
            "No files found that match the specified entities. Please check the input parameters."
        )
        return

    else:
        all_dirs = []
        for file in all_files:
            file_path = os.path.dirname(file)
            file_name = os.path.basename(file)
            all_dirs.append(file_path)

            new_entity = delete_entity(file_name, key2rem)

            old_path = os.path.join(file_path, file_name)
            new_path = os.path.join(file_path, new_entity)
            os.rename(old_path, new_path)

        all_dirs = set(all_dirs)  # Remove duplicates from the directory list

        # Renaming the directories
        key2rem_dict = {key: "" for key in key2rem}  # Create a dict with empty values
        cltmisc.rename_folders(all_dirs, key2rem_dict)


def recursively_insert_entity(
    root_dir: str, entity2add: Dict[str, str], prev_entity: str = None
) -> None:
    """
    This method inserts entities in all the files and folders of a BIDs dataset.

    Parameters
    ----------
    root_dir: str
        Root directory of the BIDs dataset

    entity2add: dict
        Dictionary containing the entities to add.
        Example: {'task': 'rest', 'run': '01'}

    prev_entity: str, optional
        Key in `entity` after which to insert the new entities. Otherwise it will be added at the end of the file name, just before the suffix.

    Returns
    -------
    None
        The method will rename the files and folders in the BIDs dataset. All the files or folders containing the old
        entities' names on their names will be renamed and the old entities will be replaced with the new entities.

    """

    # Detect if the BIDs directory exists
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        raise ValueError("The BIDs directory does not exist.")

    # Convert the strings to dictionaries
    if isinstance(entity2add, str):
        entity2add = str2entity(entity2add)

    # Order the dictionaries alphabetically by key
    entity2add = dict(sorted(entity2add.items()))

    # Creating the list of strings
    entity2add_list = [f"{key}-{value}" for key, value in entity2add.items()]

    if prev_entity is not None:
        all_files = cltmisc.get_all_files(root_dir, or_filter=prev_entity)
    else:
        all_files = cltmisc.get_all_files(root_dir, or_filter="sub-")

    if not all_files:
        print(
            "No files found that match the specified entities. Please check the input parameters."
        )
        return

    else:
        all_dirs = []
        for file in all_files:
            file_path = os.path.dirname(file)
            file_name = os.path.basename(file)
            all_dirs.append(file_path)

            new_entity = insert_entity(file_name, entity2add, prev_entity=prev_entity)

            old_path = os.path.join(file_path, file_name)
            new_path = os.path.join(file_path, new_entity)
            os.rename(old_path, new_path)


####################################################################################################
def get_all_entities(root_dir: str) -> Tuple[Dict[str, Set[str]], List[str]]:
    """
    Returns a set of all unique entities found in the BIDS dataset.

    Parameters
    ----------
    root_dir : str
        Root directory of the BIDS dataset.

    Returns
    -------
    all_entities: Set[str]
        A set of unique entity names found in the dataset.

    all_suffixes: List[str]
        A list of unique suffixes found in the dataset.

    Raises
    ------
    ValueError
        If the specified root directory does not exist.
    FileNotFoundError
        If the default configuration file is not found.
    ValueError
        If the default configuration JSON does not have the expected structure.

    Examples
    --------
    >>> get_all_entities('/path/to/bids/dataset')
    {'sub', 'ses', 'task', 'run', ...}
    >>> get_all_entities('/path/to/bids/dataset')
    {'sub', 'ses', 'task', 'run', ...}, ['T1w', 'bold', ...]

    """

    # Check if the root directory exists
    if not os.path.isdir(root_dir):
        raise ValueError(f"The specified root directory does not exist: {root_dir}")

    # Load the config bids.json file
    cwd = os.path.dirname(os.path.abspath(__file__))

    try:
        config_data = load_bids_json()

        # Merge raw and derivatives entities
        if (
            "bids_entities" in config_data
            and "raw_entities" in config_data["bids_entities"]
            and "derivatives_entities" in config_data["bids_entities"]
        ):
            entities = {
                **config_data["bids_entities"]["raw_entities"],
                **config_data["bids_entities"]["derivatives_entities"],
            }
        else:
            raise ValueError(
                "Default config JSON does not have the expected structure."
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Default configuration file not found at: {os.path.join(cwd, 'config', 'bids.json')}"
        )

    # Get all the leaf directories in the BIDS dataset
    bids_folders = cltmisc.get_leaf_directories(root_dir)

    # Get all the files in the folder
    # Leave only the files that have a correct BIDs suffix
    suffixes = config_data["bids_entities"]["raw_suffix"]
    extensions = config_data["bids_entities"]["extensions"]

    # Multiply the suffixes and extensions to get all the possible combinations
    suffixes = [f"_{s}{e}" for s in suffixes for e in extensions]

    bids_files = cltmisc.get_all_files(root_dir, or_filter=suffixes)

    # Leave only the directories that are BIDs folders
    bids_folders = cltmisc.filter_by_substring(
        bids_folders, config_data["bids_entities"]["bids_folders"]
    )

    # Leave only the files that are BIDs files
    bids_files = cltmisc.filter_by_substring(bids_files, bids_folders)

    # Initialize a set to store all unique entities
    all_entities = []
    all_suffixes = []
    for file in bids_files:

        filename = os.path.basename(file)
        ent_dict = str2entity(filename)

        # Remove suffix and extension from the entity dictionary
        if "suffix" in ent_dict:
            suffix = ent_dict["suffix"]
            all_suffixes.append(suffix)
            del ent_dict["suffix"]

        if "extension" in ent_dict:
            del ent_dict["extension"]

        if "run" in ent_dict:
            del ent_dict["run"]

        file_ent_keys = list(ent_dict.keys())

        all_entities.extend(file_ent_keys)

    # Unique entities
    all_entities = sorted(set(all_entities))
    all_suffixes = sorted(set(all_suffixes))

    selected_ent_dict = entities4table(selected_entities=all_entities)

    return selected_ent_dict, all_suffixes


####################################################################################################
def entities4table(
    entities_json: str = None, selected_entities: Union[str, Dict, List] = None
) -> Dict:
    """
    Returns the BIDS entities that will be included in the morphometric table.

    This function loads BIDS entities from a JSON configuration file and filters
    them based on optional selected entities.

    Parameters
    ----------
    entities_json : str, optional
        Path to the JSON file with entity definitions.
        If None, the method uses the default config JSON file.
    selected_entities : Union[str, Dict, List], optional
        Entities to select from the loaded entities. Can be:
        - A string with comma-separated entity names
        - A dictionary with entity names as keys
        - A list of entity names
        If None, all entities are included.

    Returns
    -------
    Dict
        Dictionary of entity names and their values.

    Raises
    ------
    ValueError
        If the provided JSON file path is invalid or the JSON format is incorrect.
    FileNotFoundError
        If the specified JSON file does not exist.

    Examples
    --------
    >>> # Using default config file (returns all entities)
    >>> entities4table()
    {'sub': {'...'}, 'ses': {'...'}, ... 'scale': {'...'}}

    >>> # Using a custom JSON file
    >>> entities4table('path/to/custom/entities.json')
    {'sub': {'...'}, 'ses': {'...'}, ... 'scale': {'...'}}

    >>> # Selecting specific entities
    >>> entities4table(selected_entities='sub,ses,run')
    {'sub': {'...'}, 'ses': {'...'}, 'run': {'...'}}

    >>> # Using a dictionary to select entities
    >>> entities4table(selected_entities={'sub': None, 'ses': None})
    {'sub': {'...'}, 'ses': {'...'}}

    >>> # Using a list to select entities
    >>> entities4table(selected_entities=['sub', 'ses'])
    {'sub': {'...'}, 'ses': {'...'}}
    """
    import os
    import json
    from typing import Dict, Union, List

    # Load entities from JSON
    if entities_json is None:
        # Define path to default config JSON
        default_config_path = os.path.join(
            os.path.dirname(__file__), "config", "bids.json"
        )
        try:
            config_data = load_bids_json(default_config_path)

            # Merge raw and derivatives entities
            if (
                "bids_entities" in config_data
                and "raw_entities" in config_data["bids_entities"]
                and "derivatives_entities" in config_data["bids_entities"]
            ):
                ent_out_dict = {
                    **config_data["bids_entities"]["raw_entities"],
                    **config_data["bids_entities"]["derivatives_entities"],
                }
            else:
                raise ValueError(
                    "Default config JSON does not have the expected structure."
                )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Default configuration file not found at: {default_config_path}"
            )

    elif isinstance(entities_json, str):
        # Load from provided JSON file path
        if not os.path.isfile(entities_json):
            raise FileNotFoundError(f"JSON file not found: {entities_json}")

        try:

            ent_out_dict = cltmisc.extract_string_values(entities_json)
        except json.JSONDecodeError:
            raise ValueError(f"Error parsing the JSON file: {entities_json}")
    else:
        raise TypeError("entities_json must be None or a string path to a JSON file.")

    # Filter entities based on selected_entities
    if selected_entities is not None:
        selected_entity_keys = []

        # Handle string input (convert to list of keys)
        if isinstance(selected_entities, str):
            try:
                # Assume it's a comma-separated string
                if "," in selected_entities:
                    selected_entity_keys = [
                        e.strip() for e in selected_entities.split(",")
                    ]
                elif is_bids_filename(selected_entities):
                    selected_entities = str2entity(selected_entities)
                    selected_entity_keys = list(selected_entities.keys())

            except (ImportError, AttributeError):
                raise ValueError(
                    "Cannot parse selected_entities string. Provide a comma-separated list or a BIDs-like string (e.g. sub-XXX_ses-SSS_run-01 )."
                )

        # Handle dictionary input
        elif isinstance(selected_entities, dict):
            selected_entity_keys = list(selected_entities.keys())

        # Handle list input
        elif isinstance(selected_entities, list):
            selected_entity_keys = selected_entities

        # Filter the output dictionary to include only selected entities
        ent_out_dict = {
            k: v for k, v in ent_out_dict.items() if k in selected_entity_keys
        }

    return ent_out_dict


####################################################################################################
def entities_to_table(
    filepath: str,
    entities_to_extract: Optional[Union[str, List[str], Dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Creates a DataFrame with BIDS entities extracted from a filename.

    This function parses BIDS-compliant filenames to extract specified entities
    (such as subject, session, task) and organizes them into a DataFrame with
    appropriate column names. It supports special handling for certain entities
    like atlas and description fields.

    Parameters
    ----------
    filepath : str
        Full path to the BIDS file from which to extract entities.
    entities_to_extract : str, list, dict, or None, default=None
        Specifies which entities to extract from the filename:
        - If str: A single entity name to extract
        - If list: Multiple entity names to extract
        - If dict: Keys are entity names, values are custom column names
        - If None: Returns a single column with the full filename

    Returns
    -------
    pd.DataFrame
        DataFrame containing the extracted entities as columns.
        If the file is not BIDS-compliant, returns an empty DataFrame.

    Examples
    --------
    >>> # Extract subject and session from a BIDS file
    >>> df = entities_to_table(
    ...     '/data/sub-01/ses-pre/sub-01_ses-pre_task-rest_bold.nii.gz',
    ...     ['sub', 'ses']
    ... )
    >>> print(df)
        Participant Session
    0          01     pre

    >>> # Extract entities with custom column names
    >>> df = entities_to_table(
    ...     '/data/sub-01/anat/sub-01_T1w.nii.gz',
    ...     {'sub': 'SubjectID'}
    ... )
    >>> print(df)
        SubjectID
    0        01

    >>> # Extract atlas with special handling
    >>> df = entities_to_table(
    ...     '/data/sub-01/atlas-chimera123_desc-parcellation.nii.gz',
    ...     ['atlas', 'desc']
    ... )
    >>> print(df)
        Atlas ChimeraCode Description
    0   chimera        123 parcellation

    """
    # Type checking
    if filepath is None or not isinstance(filepath, str):
        raise TypeError("filepath must be a string")

    # Reading the mapping dictionary
    cwd = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(cwd, "config", "bids.json")
    try:
        with open(default_config_path, "r") as f:
            config_data = json.load(f)

        # Merge raw and derivatives entities
        if (
            "bids_entities" in config_data
            and "raw_entities" in config_data["bids_entities"]
            and "derivatives_entities" in config_data["bids_entities"]
        ):
            ent_out_dict = {
                **config_data["bids_entities"]["raw_entities"],
                **config_data["bids_entities"]["derivatives_entities"],
            }
        else:
            raise ValueError(
                "Default config JSON does not have the expected structure."
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Default configuration file not found at: {default_config_path}"
        )
    except json.JSONDecodeError:
        raise ValueError(
            f"Error parsing the default configuration file: {default_config_path}"
        )

    # Extract file directory and name
    file_directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    # Create an empty DataFrame
    result_df = pd.DataFrame()

    # Only process if the file follows BIDS naming convention
    if is_bids_filename(filename):

        # Parse entities from the filename
        entities_dict = str2entity(filename)

        # Remove the 'extension' key if it exists
        if "extension" in entities_dict:
            entities_dict.pop("extension")
        # Remove the 'suffix' key if it exists
        if "suffix" in entities_dict:
            entities_dict.pop("suffix")

        if entities_to_extract is not None:
            # Convert string input to list
            if isinstance(entities_to_extract, str):
                entities_to_extract = [entities_to_extract]

            # Convert list to dictionary for unified processing
            if isinstance(entities_to_extract, list):
                # Create dict with entity names as both keys and values
                entities_to_extract = {entity: "" for entity in entities_to_extract}
        else:
            entities_to_extract = {entity: "" for entity in entities_dict.keys()}

        # Initialize DataFrame with one empty row
        if result_df.empty:
            result_df = pd.DataFrame([{}])

        # Process entities in reverse order to maintain column order
        entity_keys = list(entities_to_extract.keys())
        for entity in reversed(entity_keys):
            # Get entity value from filename or empty string if not found
            value = entities_dict.get(entity, "")

            if entity in ent_out_dict.keys():
                # If entity is in the mapping dictionary, use its value
                var_name = ent_out_dict[entity]
                if entity == "atlas":
                    # Special handling for atlas
                    if "chimera" in value:
                        result_df.insert(0, "ChimeraCode", value.replace("chimera", ""))
                        result_df.insert(0, "Atlas", "chimera")

                    else:
                        result_df.insert(0, "ChimeraCode", "")
                        result_df.insert(0, "Atlas", value)

                elif entity == "desc":
                    # Special handling for description
                    result_df.insert(0, "Description", value)
                    if "grow" in value:
                        result_df.insert(0, "GrowIntoWM", value.replace("grow", ""))

                else:
                    result_df.insert(0, var_name, value)
            else:

                result_df.insert(0, entity.capitalize(), value)

    else:
        # If no entities specified, use full filename as Participant
        if result_df.empty:
            result_df = pd.DataFrame([{}])

        if "Participant" not in result_df.columns:
            # If 'Participant' column doesn't exist, create it
            # Remove the extension from the filename in case it exists
            temp = os.path.splitext(filename)[0]
            result_df.insert(0, "Participant", temp)

    # Declaring all the columns as string type
    for col in result_df.columns:
        result_df[col] = result_df[col].astype(str)

    return result_df


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############     Section 2: Methods dedicated to work with BIDs file organization       ############
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
            raise FileNotFoundError(
                f"The derivatives directory {deriv_dir} does not exist."
            )

    # Selecting the subjects that will be copied
    if isinstance(subjects_to_copy, str):
        subjects_to_copy = [subjects_to_copy]

    if subjects_to_copy is None:
        subjects_to_copy = get_subjects(bids_dir)
    else:
        # Check if the cad "sub-" is in the subjects_to_copy of not add it
        subjects_to_copy = [
            sub if sub.startswith("sub-") else f"sub-{sub}" for sub in subjects_to_copy
        ]

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
            include_derivatives = [
                os.path.join(deriv_dir, i) for i in include_derivatives
            ]
            der_pipe_folders = cltmisc.list_intercept(
                der_pipe_folders, include_derivatives
            )

        if len(der_pipe_folders) == 0:
            print(
                "WARNING: No derivatives folders were found with the specified names."
            )
            copy_derivatives = False
    else:
        copy_derivatives = False
        der_pipe_folders = []

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

            subj_dir = os.path.join(bids_dir, f"sub-{subj_id}")
            all_subj_subfold = cltmisc.get_leaf_directories(subj_dir)

            # Upgrade the progress bar and include the subject id on the text
            progress.update(
                task, description=f"[red]Copying subject: {full_id}", completed=i + 1
            )

            # Detect the session id if it exists
            ses_id = None
            if "ses" in subj_entity.keys():
                ses_id = subj_entity["ses"]
                full_ses_id = f"ses-{ses_id}"

                all_subj_subfold = cltmisc.filter_by_substring(
                    input_list=all_subj_subfold, or_filter=full_ses_id
                )

            #######  Copying the BIDs folder
            if "all" not in folders_to_copy:
                tmp_list = [os.path.basename(i) for i in all_subj_subfold]
                indexes = cltmisc.get_indexes_by_substring(tmp_list, folders_to_copy)
                all_subj_subfold = [all_subj_subfold[i] for i in indexes]

            raw_files_to_copy = []
            # Loop along all the folders to copy
            for fc in all_subj_subfold:
                all_subj_raw_files = cltmisc.get_all_files(fc)

                raw_files_to_copy = raw_files_to_copy + all_subj_raw_files

            # Filtering the files to copy according to the full
            raw_files_to_copy = cltmisc.filter_by_substring(
                input_list=raw_files_to_copy,
                or_filter=f"sub-{subj_id}",
                and_filter=file_filter,
            )

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
                        subj_deriv = glob(os.path.join(pipe_dir, f"sub-{subj_id}*"))

                        if len(subj_deriv) > 0:

                            # Detecting the files and folders to copy.
                            # We are doing this separation because there are derivatives that contain
                            # only the folders with the name of the subject and not the files (i.e. freesurfer)

                            # Loop along all the folders to copy
                            deriv_files_to_copy = []
                            deriv_folds_to_copy = []
                            for deriv in subj_deriv:

                                # Detect the files and folders to copy
                                tmp_fold_to_copy = cltmisc.filter_by_substring(
                                    input_list=deriv,
                                    or_filter=f"sub-{subj_id}",
                                    and_filter=fold_filter,
                                )

                                if len(tmp_fold_to_copy) > 0:
                                    # Detect the files and folders to copy
                                    for tmp in tmp_fold_to_copy:
                                        if os.path.isdir(tmp):
                                            deriv_folds_to_copy.append(tmp)
                                        else:
                                            deriv_files_to_copy.append(tmp)
                                else:
                                    all_subj_deriv_files = cltmisc.get_all_files(deriv)

                                    # Filtering the files to copy according to the full
                                    all_subj_deriv_files = cltmisc.filter_by_substring(
                                        input_list=all_subj_deriv_files,
                                        or_filter=f"sub-{subj_id}",
                                        and_filter=fold_filter,
                                    )

                                    deriv_files_to_copy = (
                                        deriv_files_to_copy + all_subj_deriv_files
                                    )

                            # Copying the folders
                            if len(deriv_folds_to_copy) > 0:
                                for fold in deriv_folds_to_copy:
                                    try:
                                        dest_dir = fold.replace(
                                            deriv_dir,
                                            os.path.join(out_dir, "derivatives"),
                                        )
                                        os.makedirs(dest_dir, exist_ok=True)
                                        shutil.copytree(
                                            fold, dest_dir, dirs_exist_ok=True
                                        )
                                    except:
                                        print(
                                            f"WARNING: The folder or file {fold} could not be copied."
                                        )
                                        continue

                            # Copying the files
                            if len(deriv_files_to_copy) > 0:
                                for file in deriv_files_to_copy:
                                    try:
                                        dest_dir = file.replace(
                                            deriv_dir,
                                            os.path.join(out_dir, "derivatives"),
                                        )
                                        os.makedirs(
                                            os.path.dirname(dest_dir), exist_ok=True
                                        )
                                        shutil.copy2(file, dest_dir)
                                    except:
                                        print(
                                            f"WARNING: The file {file} could not be copied."
                                        )
                                        continue

        # Update the progress bar to 100%
        progress.update(task, completed=n_subj)


####################################################################################################
def get_bids_database_table(
    root_dir: str, output_table: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate a comprehensive summary table of all neuroimaging files in a BIDS dataset.

    This function scans a BIDS dataset directory structure and creates a detailed table
    containing all BIDS entities (subject, session, acquisition, etc.) and file counts.
    The output table provides an overview of the dataset composition, making it easy to
    identify data availability, missing files, and dataset structure.

    Parameters
    ----------
    root_dir : str
        Path to the BIDS dataset root directory. This should be the top-level directory
        containing subject folders (sub-*) and optionally a dataset_description.json file.

    output_table : str, optional
        Path where the resulting CSV table should be saved. If None, the table is not
        saved to disk but still returned as a DataFrame. Default is None.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for each detected BIDS entity (Subject, Session,
        Acquisition, etc.), plus 'suffix' (image type like T1w, FLAIR) and 'N'
        (number of files for each unique combination). Each row represents a unique
        combination of BIDS entities and their file count.

    Raises
    ------
    FileNotFoundError
        If the specified root_dir does not exist.
    NotADirectoryError
        If root_dir exists but is not a directory.
    ValueError
        If no subjects are found in the BIDS dataset (no sub-* folders).

    Examples
    --------
    Basic usage - analyze dataset and return summary table:

    >>> import pandas as pd
    >>> bids_table = get_bids_table('/path/to/bids/dataset')
    >>> print(f"Dataset contains {len(bids_table)} unique file combinations")
    >>> print(f"Total files: {bids_table['N'].sum()}")

    Save summary table to CSV file:

    >>> bids_table = get_bids_table(
    ...     root_dir='/data/my_study',
    ...     output_table='/data/my_study/bids_summary.csv'
    ... )

    Analyze specific aspects of the dataset:

    >>> # Count files by image type
    >>> suffix_counts = bids_table.groupby('suffix')['N'].sum()
    >>> print("Files by image type:")
    >>> print(suffix_counts)

    >>> # Check data availability per subject
    >>> subject_counts = bids_table.groupby('Subject')['N'].sum()
    >>> print("Files per subject:")
    >>> print(subject_counts)

    >>> # Find subjects with specific image types
    >>> t1w_subjects = bids_table[bids_table['suffix'] == 'T1w']['Subject'].unique()
    >>> print(f"Subjects with T1w images: {len(t1w_subjects)}")

    Example output table structure:

    >>> print(bids_table.head())
        Subject Session Acquisition  suffix  N
    0    sub-01     ses-01        acq-mprage    T1w  1
    1    sub-01     ses-01        acq-space    T2w  1
    2    sub-01     ses-01           None   FLAIR  1
    3    sub-02     ses-01        acq-mprage    T1w  1
    4    sub-02     ses-02        acq-mprage    T1w  1

    Notes
    -----
    - Only processes .nii.gz files (NIfTI compressed format)
    - Automatically detects all BIDS entities present in the dataset
    - Groups identical combinations and sums file counts
    - Results are sorted by Subject, Session, and suffix for readability
    - Progress is displayed using Rich progress bar during processing
    - Column names are converted to human-readable format (e.g., 'sub' -> 'Subject')

    See Also
    --------
    clabtoolkit.bidstools.get_subjects : Get list of subjects in BIDS dataset
    clabtoolkit.bidstools.get_all_entities : Extract all BIDS entities from dataset
    clabtoolkit.bidstools.str2entity : Parse BIDS filename to extract entities
    """

    # Check if the root directory exists
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"BIDS root directory not found: {root_dir}")

    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"Provided path is not a directory: {root_dir}")

    # Initialize console
    console = Console()

    # Get all the subjects in the BIDs dataset
    subj_ids = get_subjects(root_dir)

    if not subj_ids:
        raise ValueError(f"No subjects found in the BIDS dataset at {root_dir}")

    # Get all entities in the BIDs dataset
    all_dataset_entities, all_dataset_suffixes = get_all_entities(root_dir)

    console.print(f"[green]Detected entities: {list(all_dataset_entities.keys())}")

    n_subj = len(subj_ids)

    # Create column names
    column_names = list(all_dataset_entities.keys())
    column_names.extend(["suffix", "N"])

    # List to store all rows
    all_rows = []

    # Create Progress with explicit configuration
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        MofNCompleteColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,
        auto_refresh=True,
        refresh_per_second=10,
    )

    # Use the progress bar
    with progress:
        # Add the task
        task_id = progress.add_task(
            description="Starting BIDS processing...", total=n_subj
        )

        # Process each subject
        for i, subj_id in enumerate(subj_ids):
            # Update task description at the start of processing
            progress.update(
                task_id, description=f"Processing subject {subj_id}...", completed=i
            )

            subj_dir = os.path.join(root_dir, subj_id)
            all_files = cltmisc.get_all_files(
                subj_dir, recursive=True, or_filter=[".nii.gz"]
            )

            files_processed = 0
            for file_path in all_files:
                file_name = os.path.basename(file_path)

                # Parse the BIDS filename to get entities
                ent_dict = str2entity(file_name)

                # Create a row dictionary with all possible entity columns initialized to None
                row_dict = {col: None for col in column_names}

                # Fill in the entities that are present in this file
                for entity, value in ent_dict.items():
                    if entity in row_dict:
                        row_dict[entity] = value

                # Extract suffix from filename
                row_dict["suffix"] = ent_dict["suffix"]

                # Set N to 1 for individual files (we'll aggregate later)
                row_dict["N"] = 1

                all_rows.append(row_dict)
                files_processed += 1

            # Update progress AFTER processing each subject (this is the key fix!)
            progress.update(
                task_id,
                description=f"Completed {subj_id} ({files_processed} files)",
                completed=i + 1,  # i+1 because we just finished this subject
            )

            # Force refresh to ensure display updates
            progress.refresh()

            # Small delay to see the progress update
            time.sleep(0.1)

        # Final update
        progress.update(task_id, description="Processing complete!", completed=n_subj)

        # Brief pause to show completion
        time.sleep(0.5)

    # Create DataFrame from all rows
    df_table = pd.DataFrame(all_rows)

    if df_table.empty:
        return df_table

    # Group by all columns except 'N' and sum the counts
    groupby_cols = [col for col in df_table.columns if col != "N"]
    df_grouped = (
        df_table.groupby(groupby_cols, dropna=False).agg({"N": "sum"}).reset_index()
    )

    # Sort the DataFrame for better readability
    if not df_grouped.empty:
        sort_cols = []
        if "sub" in df_grouped.columns:
            sort_cols.append("sub")
        if "ses" in df_grouped.columns:
            sort_cols.append("ses")
        if "suffix" in df_grouped.columns:
            sort_cols.append("suffix")

        if sort_cols:
            df_grouped = df_grouped.sort_values(sort_cols).reset_index(drop=True)

    all_entities = entities4table()

    old_colnames = cltmisc.list_intercept(
        list(df_grouped.columns), list(all_entities.keys())
    )
    new_colnames = list(entities4table(selected_entities=old_colnames).values())

    # Rename columns to match BIDS entities
    df_grouped.rename(columns=dict(zip(old_colnames, new_colnames)), inplace=True)

    # Save the table if output path is provided
    if output_table:
        df_grouped.to_csv(output_table, index=False)

    return df_grouped


####################################################################################################
def get_derivatives_folders(
    deriv_dir: str,
) -> list:
    """
    Get a list of all derivatives folders in the specified directory.

    Parameters
    ----------
    deriv_dir : str
        Path to the derivatives directory.

    Returns
    -------
    list
        List of derivatives folder names.

    Raises
    ------
    ValueError
        If the derivatives directory does not exist.

    TypeError
        If the derivatives directory is not a string.

    Usage example:
    >>> deriv_dir = "/path/to/derivatives"
    >>> print(get_derivatives_folders(deriv_dir))

    """

    # Check if the derivatives directory exists
    if not os.path.isdir(deriv_dir):
        raise ValueError("The derivatives directory does not exist.")

    # Get all directories in the derivatives directory
    directories = os.listdir(deriv_dir)

    # Filter out hidden directories and keep only valid directories
    der_pipe_folders = []
    for directory in directories:
        pipe_dir = os.path.join(deriv_dir, directory)
        if not directory.startswith(".") and os.path.isdir(pipe_dir):
            der_pipe_folders.append(directory)

    # Remove the derivatives folders that do not include folders starting with "sub-"
    der_pipe_folders = [
        i
        for i in der_pipe_folders
        if any(j.startswith("sub-") for j in os.listdir(os.path.join(deriv_dir, i)))
    ]

    return der_pipe_folders


####################################################################################################
def is_bids_filename(filename: str) -> bool:
    """
    Validates a BIDS filename structure, handling extensions and entity order.

    Args:
        filename (str): The filename to validate.

    Returns
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


####################################################################################################
def get_individual_files_and_folders(
    input_folder: str, cad4query: Union[str, list, dict]
):
    """
    This function detects all the files or folders inside a folder and its subfolders containing the strings supplied by the variable cad4query.

    Parameters
    ----------
    input_folder : str
        Path to the input folder.

    cad4query : str, list, or dict
        String or list of strings to filter the files and folders. If a dictionary is provided, it should contain key-value pairs where the key is the string before '-' and the value is the string after '-'.

    Returns
    -------
    list
        List of files or folders that match the query.

    Raises
    ------
    ValueError
        If the input folder does not exist.

    TypeError
        If the input folder is not a string.

    Examples
    --------
    >>> input_folder = "/path/to/input/folder"
    >>> cad4query = "sub-01"
    >>> files = get_individual_files_and_folders(input_folder, cad4query)


    """

    # Checking if the folder is not a string or if the folder does not exist
    if not isinstance(input_folder, str):
        raise TypeError("The input folder must be a string.")

    if not os.path.isdir(input_folder):
        raise ValueError(f"The input folder {input_folder} does not exist.")

    # Create a dictionary from cad4query where the key is the string before - and the value is the string after -
    if isinstance(cad4query, str):
        cad4query = [cad4query]

    if isinstance(cad4query, list):
        clean_id_dict = dict([i.split("-") for i in cad4query])

    elif isinstance(cad4query, dict):
        clean_id_dict = cad4query.copy()

    # Detecting the all the files for the reference subject
    ind_der_dir = glob(os.path.join(input_folder, "sub-" + clean_id_dict["sub"] + "*"))

    # Filter if multiple directories found
    if len(ind_der_dir) > 1:
        ind_der_dir = cltmisc.filter_by_substring(
            ind_der_dir,
            or_filter=[clean_id_dict["sub"]],
            and_filter=cad4query,
        )

    # Set count to 0 if no directory found
    if len(ind_der_dir) > 0:
        # Count files for this subject in this pipeline
        all_files = cltmisc.get_all_files(ind_der_dir[0])
        all_files = cltmisc.filter_by_substring(
            all_files, or_filter=clean_id_dict["sub"], and_filter=cad4query
        )

    else:
        all_files = []

    return all_files


####################################################################################################
def generate_bids_tree(
    bids_root: str,
    max_depth: Optional[int] = None,
    show_hidden: bool = False,
    exclude_patterns: Optional[Set[str]] = None,
    save_to_file: Optional[str] = None,
) -> str:
    """
    Generate an MS-DOS tree-style visualization of a BIDS folder structure.

    Parameters
    ----------
    bids_root : str
        Path to the BIDS root directory.

    max_depth : int, optional
        Maximum depth to traverse. If None (default), traverses entire directory
        structure without depth limitation.

    show_hidden : bool, optional
        Whether to show hidden files and folders (starting with '.').
        Default is False.

    exclude_patterns : set of str, optional
        Set of file/folder name patterns to exclude from the tree. If None,
        defaults to {'.git', '__pycache__', '.DS_Store', 'Thumbs.db'}.

    save_to_file : str, optional
        Path to save the tree output as a text file. If None, only returns
        the string without saving.

    Returns
    -------
    str
        MS-DOS tree representation of the BIDS structure with proper tree
        symbols (├──, └──, │) and directory indicators (/).

    Raises
    ------
    FileNotFoundError
        If the specified bids_root path does not exist.

    NotADirectoryError
        If the specified bids_root path is not a directory.

    PermissionError
        If there are insufficient permissions to read certain directories.
        Individual permission errors are handled gracefully and noted in output.

    OSError
        If there are file system related errors during tree generation or
        file saving operations.

    Notes
    -----
    - Directories are displayed with a trailing '/' to distinguish from files
    - Items are sorted with directories first, then files, both alphabetically
    - Hidden files/folders (starting with '.') are excluded by default
    - Permission errors for individual subdirectories are handled gracefully
    - The tree uses standard MS-DOS tree symbols for proper visualization
    - When max_depth is None, the entire directory structure is traversed

    Examples
    --------
    Basic usage with unlimited depth:

    >>> tree = generate_bids_tree('/path/to/bids/dataset')
    >>> print(tree)
    my-bids-dataset/
    ├── dataset_description.json
    ├── participants.tsv
    ├── sub-01/
    │   ├── anat/
    │   │   └── sub-01_T1w.nii.gz
    │   └── func/
    │       ├── sub-01_task-rest_bold.nii.gz
    │       └── sub-01_task-rest_events.tsv
    └── derivatives/
        └── preprocessing/
            └── sub-01/

    Limited depth with file saving:

    >>> tree = generate_bids_tree('/path/to/bids/dataset',
    ...                          max_depth=2,
    ...                          save_to_file='bids_tree.txt')
    >>> print("Tree saved to bids_tree.txt")

    Include hidden files and custom exclusions:

    >>> tree = generate_bids_tree('/path/to/bids/dataset',
    ...                          show_hidden=True,
    ...                          exclude_patterns={'temp', 'backup'})
    """

    if exclude_patterns is None:
        exclude_patterns = {".git", "__pycache__", ".DS_Store", "Thumbs.db"}

    bids_path = Path(bids_root)

    if not bids_path.exists():
        raise FileNotFoundError(f"The specified path does not exist: {bids_root}")

    if not bids_path.is_dir():
        raise NotADirectoryError(f"The specified path is not a directory: {bids_root}")

    tree_lines = [f"{bids_path.name}/"]

    def _build_tree(current_path: Path, prefix: str = "", depth: int = 0) -> None:
        """Recursively build the tree structure."""

        if max_depth is not None and depth >= max_depth:
            return

        try:
            # Get all items in current directory
            items = []
            for item in current_path.iterdir():
                # Skip hidden files if not requested
                if not show_hidden and item.name.startswith("."):
                    continue
                # Skip excluded patterns
                if item.name in exclude_patterns:
                    continue
                items.append(item)

            # Sort items: directories first, then files, both alphabetically
            items.sort(key=lambda x: (x.is_file(), x.name.lower()))

            for i, item in enumerate(items):
                is_last = i == len(items) - 1

                # Choose the appropriate tree symbols
                if is_last:
                    current_prefix = "└── "
                    next_prefix = prefix + "    "
                else:
                    current_prefix = "├── "
                    next_prefix = prefix + "│   "

                # Add item to tree
                if item.is_dir():
                    tree_lines.append(f"{prefix}{current_prefix}{item.name}/")
                    # Recursively process subdirectory
                    _build_tree(item, next_prefix, depth + 1)
                else:
                    tree_lines.append(f"{prefix}{current_prefix}{item.name}")

        except PermissionError:
            tree_lines.append(f"{prefix}├── [Permission Denied]")
        except Exception as e:
            tree_lines.append(f"{prefix}├── [Error: {str(e)}]")

    # Build the tree starting from the root
    _build_tree(bids_path)

    # Join all tree lines
    tree_output = "\n".join(tree_lines)

    # Save to file if requested
    if save_to_file:
        try:
            with open(save_to_file, "w", encoding="utf-8") as f:
                f.write(tree_output)
        except OSError as e:
            raise OSError(f"Failed to save tree to file '{save_to_file}': {str(e)}")

    return tree_output


####################################################################################################
def generate_bids_tree_with_stats(bids_root: str, **kwargs) -> str:
    """
    Generate a BIDS tree with additional statistics.

    Parameters
    ----------
    bids_root : str
        Path to the BIDS root directory.

    **kwargs
        Additional keyword arguments passed to generate_bids_tree().
        See generate_bids_tree() documentation for available parameters.

    Returns
    -------
    str
        Tree representation with file and folder count statistics appended.

    Raises
    ------
    FileNotFoundError
        If the specified bids_root path does not exist.

    NotADirectoryError
        If the specified bids_root path is not a directory.

    PermissionError
        If there are insufficient permissions to read directories.

    OSError
        If there are file system related errors.

    Notes
    -----
    Statistics are calculated by recursively counting all files and directories
    in the BIDS structure, regardless of the max_depth parameter used for
    tree visualization.

    Examples
    --------
    >>> tree_with_stats = generate_bids_tree_with_stats('/path/to/bids/dataset')
    >>> print(tree_with_stats)
    my-bids-dataset/
    ├── dataset_description.json
    └── sub-01/
        └── anat/
            └── sub-01_T1w.nii.gz

    Statistics:
    ├── Directories: 2
    └── Files: 2
    """

    tree = generate_bids_tree(bids_root, **kwargs)

    # Count files and directories
    bids_path = Path(bids_root)
    file_count = 0
    dir_count = 0

    try:
        for item in bids_path.rglob("*"):
            if item.is_file():
                file_count += 1
            elif item.is_dir():
                dir_count += 1
    except Exception:
        pass

    stats = f"\n\nStatistics:\n├── Directories: {dir_count}\n└── Files: {file_count}"

    return tree + stats


####################################################################################################
def validate_bids_structure(bids_root: str) -> List[str]:
    """
    Basic validation of BIDS structure and return warnings.

    Parameters
    ----------
    bids_root : str
        Path to the BIDS root directory.

    Returns
    -------
    list of str
        List of validation warnings and notes about the BIDS structure.
        Empty list indicates no issues found.

    Raises
    ------
    FileNotFoundError
        If the specified bids_root path does not exist.
    NotADirectoryError
        If the specified bids_root path is not a directory.

    Notes
    -----
    This function performs basic BIDS validation including:
    - Checking for required files (dataset_description.json)
    - Verifying presence of subject directories (sub-*)
    - Noting presence of derivatives directory

    For comprehensive BIDS validation, consider using the official
    BIDS validator tool.

    Examples
    --------
    >>> warnings = validate_bids_structure('/path/to/bids/dataset')
    >>> if warnings:
    ...     for warning in warnings:
    ...         print(f"⚠️ {warning}")
    >>> else:
    ...     print("✅ Basic BIDS structure looks good!")
    """

    warnings = []
    bids_path = Path(bids_root)

    if not bids_path.exists():
        raise FileNotFoundError(f"The specified path does not exist: {bids_root}")

    if not bids_path.is_dir():
        raise NotADirectoryError(f"The specified path is not a directory: {bids_root}")

    # Check for required files
    required_files = ["dataset_description.json"]
    for req_file in required_files:
        if not (bids_path / req_file).exists():
            warnings.append(f"Missing required file: {req_file}")

    # Check for common BIDS directories
    common_dirs = ["sub-*"]  # Using glob pattern
    subject_dirs = list(bids_path.glob("sub-*"))
    if not subject_dirs:
        warnings.append("No subject directories found (sub-*)")

    # Check for derivatives directory
    derivatives_dir = bids_path / "derivatives"
    if derivatives_dir.exists():
        warnings.append("Derivatives directory found")

    return warnings


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############           Section 3: Useful methods for BIDs file organization             ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################


# Loading the JSON file containing the BIDs configuration file. It includes the entities and suffixes
# of the BIDs dataset.
####################################################################################################
def load_bids_json(bids_json: str = None):
    """
    Load the JSON file containing the BIDs configuration file.

    Parameters
    ----------
    bids_json : str
        JSON file containing the BIDs configuration.

    Returns
    --------
    config_dict : dict
        Dictionary containing the default .

    """

    # Get the absolute of this file
    if bids_json is None:
        cwd = os.path.dirname(os.path.abspath(__file__))
        bids_json = os.path.join(cwd, "config", "bids.json")
    else:
        if not os.path.isfile(bids_json):
            raise ValueError(
                "Please, provide a valid JSON file containing the BIDs configuration dictionary."
            )
    try:
        with open(bids_json) as f:
            config_dict = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Default configuration file not found at: {bids_json}")
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing the default configuration file: {bids_json}")

    return config_dict
