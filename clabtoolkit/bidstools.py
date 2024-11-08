import os
import shutil
import pandas as pd
import clabtoolkit.misctools as cltmisc
from typing import Union, Dict, List


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
        ent_string += '_' + suffix
    else:
        ent_string = ent_string.rstrip("_")  # Remove trailing underscore if no suffix
    
    # Append extension if it exists
    if extension:
        ent_string += f".{extension}"

    return ent_string

####################################################################################################
def delete_entity(entity: Union[dict, str], 
                    key2rem: Union[List[str], str]) -> Union[dict, str]:
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
def replace_entity_value(entity: Union[dict, str], 
                        ent2replace: dict, 
                        verbose: bool = False) -> Union[dict, str]:
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
def replace_entity_key(entity: Union[dict, str], 
                        keys2replace: Dict[str, str], 
                        verbose: bool = False) -> Union[dict, str]:
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
def insert_entity(entity: Union[dict, str], 
                    entity2add: Dict[str, str], 
                    prev_entity: str = None) -> Union[dict, str]:
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
        raise ValueError(f"Reference entity '{prev_entity}' is not in the entity dictionary.")

    # Temporarily remove `suffix` and `extension` if they exist
    suffix = entity.pop("suffix", None)
    extension = entity.pop("extension", None)

    # Build `ent_out` by adding items from `entity`, and insert `entity2add` after `prev_entity` if specified
    ent_out = {}
    for key, value in entity.items():
        ent_out[key] = value
        if key == prev_entity:
            ent_out.update(entity2add)  # Insert new entities immediately after `prev_entity`

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
def recursively_replace_entity_value(root_dir:str, 
                            dict2old: Union[dict, str],
                            dict2new: Union[dict, str]):
    
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
                if file_name.endswith('sessions.tsv'):
                    tsv_file = os.path.join(root,file_name)
                    # Read line by line and replace the string
                    # Load the TSV file
                    df = pd.read_csv(tsv_file, sep='\t')

                    # Replace subst_x with subst_y in all string columns
                    df = df.applymap(lambda x: x.replace(subst_x, subst_y) if isinstance(x, str) else x)

                    # Save the modified DataFrame as a TSV file
                    df.to_csv(tsv_file, sep='\t', index=False)

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

# This function copies the BIDs folder and its derivatives for e given subjects to a new location
def copy_bids_folder(
    bids_dir: str,
    out_dir: str,
    fold2copy: list = ["anat"],
    subjs2copy: str = None,
    deriv_dir: str = None,
    include_derivatives: bool = False,
):
    """
    Copy full bids folders
    @params:
        bids_dir     - Required  : BIDs dataset directory:
        out_dir      - Required  : Output directory:
        fold2copy    - Optional  : List of folders to copy: default = ['anat']
        subjs2copy   - Optional  : List of subjects to copy:
        deriv_dir    - Optional  : Derivatives directory: default = None
        include_derivatives - Optional  : Include derivatives folder: default = False
    """

    # Listing the subject ids inside the dicom folder
    if subjs2copy is None:
        my_list = os.listdir(bids_dir)
        subj_ids = []
        for it in my_list:
            if "sub-" in it:
                subj_ids.append(it)
        subj_ids.sort()
    else:
        subj_ids = subjs2copy

    # Selecting the derivatives folder
    if include_derivatives:
        if deriv_dir is None:
            deriv_dir = os.path.join(bids_dir, "derivatives")

        if not os.path.isdir(deriv_dir):
            # Lunch a warning message if the derivatives folder does not exist
            print("WARNING: The derivatives folder does not exist.")
            print("WARNING: The derivatives folder will not be copied.")
            include_derivatives = False

        # Selecting all the derivatives folders
        der_pipe_folders = []
        directories = os.listdir(deriv_dir)
        der_pipe_folders = []
        for directory in directories:
            pipe_dir = os.path.join(deriv_dir, directory)
            if not directory.startswith(".") and os.path.isdir(pipe_dir):
                der_pipe_folders.append(pipe_dir)

    # Failed sessions and derivatives
    fail_sess = []
    fail_deriv = []

    # Loop around all the subjects
    nsubj = len(subj_ids)
    for i, subj_id in enumerate(subj_ids):  # Loop along the IDs
        subj_dir = os.path.join(bids_dir, subj_id)
        out_subj_dir = os.path.join(out_dir, subj_id)

        cltmisc.printprogressbar(
            i + 1,
            nsubj,
            "Processing subject "
            + subj_id
            + ": "
            + "("
            + str(i + 1)
            + "/"
            + str(nsubj)
            + ")",
        )

        # Loop along all the sessions inside the subject directory
        for ses_id in os.listdir(subj_dir):  # Loop along the session
            ses_dir = os.path.join(subj_dir, ses_id)
            out_ses_dir = os.path.join(out_subj_dir, ses_id)

            # print('Copying SubjectId: ' + subjId + ' ======>  Session: ' +  sesId)

            if fold2copy[0] == "all":
                directories = os.listdir(ses_dir)
                fold2copy = []
                for directory in directories:
                    if not directory.startswith(".") and os.path.isdir(
                        os.path.join(ses_dir, directory)
                    ):
                        print(directory)
                        fold2copy.append(directory)

            for fc in fold2copy:
                # Copying the anat folder
                if os.path.isdir(ses_dir):
                    fold_to_copy = os.path.join(ses_dir, fc)

                    try:
                        # Creating destination directory using make directory
                        dest_dir = os.path.join(out_ses_dir, fc)
                        os.makedirs(dest_dir, exist_ok=True)

                        shutil.copytree(fold_to_copy, dest_dir, dirs_exist_ok=True)

                    except:
                        fail_sess.append(fold_to_copy)

            if include_derivatives:
                # Copying the derivatives folder

                for pipe_dir in der_pipe_folders:
                    if os.path.isdir(pipe_dir):

                        out_pipe_dir = os.path.join(
                            out_dir, "derivatives", os.path.basename(pipe_dir)
                        )

                        pipe_indiv_subj_in = os.path.join(pipe_dir, subj_id, ses_id)
                        pipe_indiv_subj_out = os.path.join(
                            out_pipe_dir, subj_id, ses_id
                        )

                        if os.path.isdir(pipe_indiv_subj_in):
                            try:
                                # Creating destination directory using make directory
                                os.makedirs(pipe_indiv_subj_out, exist_ok=True)

                                # Copying the folder
                                shutil.copytree(
                                    pipe_indiv_subj_in,
                                    pipe_indiv_subj_out,
                                    dirs_exist_ok=True,
                                )

                            except:
                                fail_deriv.append(pipe_indiv_subj_in)

    # Print the failed sessions and derivatives
    print(" ")
    if fail_sess:
        print("THE PROCESS FAILED COPYING THE FOLLOWING SESSIONS:")
        for i in fail_sess:
            print(i)
    print(" ")

    if fail_deriv:
        print("THE PROCESS FAILED COPYING THE FOLLOWING DERIVATIVES:")
        for i in fail_deriv:
            print(i)
    print(" ")

    print("End of copying the files.")
