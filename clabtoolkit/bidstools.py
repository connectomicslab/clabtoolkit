import os
import shutil
import clabtoolkit.misctools as cltmisc
from typing import Union


def _delete_from_entity(entity:dict, key2rem:Union[list, str]):
    """
    This function removes some keys from a dictionary.

    Parameters:
    ----------
    entity: dict
        Dictionary containing the entities
    key2rem: list or str
        List of keys to remove
        
    Returns:
    -------
    entity_out: dict
        Dictionary containing the entities without the keys to remove
    
    """
    
    if isinstance(key2rem, str):
        key2rem = [key2rem]
    
    entity_out = entity.copy()
    for key in key2rem:

        if key in entity_out:
            entity_out.pop(key, None)
    
    return entity_out


def _delete_from_name(str_in:str, key2rem:Union[list, str]):
    """
    This function removes an entity and its value from a string that follows the BIDs naming specifications.  
    
    Parameters:
    ----------
    str_in: str
        String of characters 
    
    key2rem: list or str
        List of keys to remove
        
    Returns:
    -------
    str_out: str
        Ouput string without the entities that were removed  
    """

    if isinstance(key2rem, list):
        key2rem = [key2rem]

    ent_in = _str2entity(str_in)
    ent_out = _delete_from_entity(ent_in, key2rem =key2rem)
    str_out = _entity2str(ent_out)


    return str_out


def _replace_entity_value(str_in:str, 
                            ent_name:Union[list, str], 
                            ent_value:Union[list, str],
                            verbose: bool = False):
    """
    This function replace an entity value in a string that follows the BIDs naming specifications.  
    
    Parameters:
    ----------
    str_in: str
        String of characters 
    
    ent_name: str or list
        Name of the entities that will be replaced.
        The length of names should coincide with the length of values.
    
    ent_value: str or list
        New values for the entities that will be replaced. 
        The length of values should coincide with the length of names.
            
    Returns:
    -------
    ent_out: dict
        Dictionary containing the entities with the new entities added
        
    """

    ent = _str2entity(str_in)

    if isinstance(ent_name, str):
        ent_name = [ent_name]

    if isinstance(ent_value, str):
        ent_value = [ent_value]

    if len(ent_name) !=len(ent_value):
        raise ValueError("The length of names and values should coincide.")

    for i, key in enumerate(ent_name):

        if key in ent:
            ent[ent_name[i]] = ent_value[i]
        else:
            if verbose:
                print(f"The entity {ent_name[i]} is not in the string")        

    # Converting to string 
    str_out = _entity2str(ent)

    return str_out


def _str2entity(string:str):
    """
    This function converts a string to a dictionary.
    
    Parameters:
    ----------
    string: str
        String to convert
        
    Returns:
    -------
    ent_dict: dict
        Dictionary containing the entities extracted from the string
    """

    ent_list = string.split("_")
    # Detecting the suffix and the extension
    # Detect which entity does not contain a "-"
    for ent in ent_list:
        if not "-" in ent:

            if "." in ent:
                temp = ent.split(".")
                if temp[0]:
                    suffix = temp[0]
                    extension = '.'.join(temp[1:])
                else:
                    extension = ent
            else:
                suffix = ent
            
            ent_list.remove(ent)


    ent_dict = {}
    for ent in ent_list:
        ent_dict[ent.split("-")[0]] = ent.split("-")[1]
    
    if "suffix" in locals():
        ent_dict["suffix"] = suffix
    if "extension" in locals():
        ent_dict["extension"] = extension

    return ent_dict


def _insert_entity(entity:dict, 
                    entity2add:dict, 
                    prev_entity:str = None):
    """
    This function adds entities to an existing entity dictionary.
    
    Parameters:
    ----------
    entity: dict
        Dictionary containing the entities
    
    entity2add: dict
        Dictionary containing the entities that will be added
    
    prev_entity: str
        Previous entity. This value will serve as reference to add the new entities.
        
    Returns:
    -------
    ent_out: dict
        Dictionary containing the entities with the new entities added
        
    """
    if prev_entity is not None:
        ent_list = entity.keys()
        if prev_entity not in ent_list:
            raise ValueError("The entity to add is not in the entity dictionary")
        
    if "suffix" in entity:
        suffix = entity["suffix"]
        entity.pop("suffix", None)
    if "extension" in entity:
        extension = entity["extension"]
        entity.pop("extension", None)

    
    ent_string = ""
    ent_out = {}
    for key, value in entity.items():
        
        if key == prev_entity:
            for key2add, value2add in entity2add.items():
                ent_out[key2add] = value2add
        
        ent_out[key] = value
    
    if prev_entity is None or prev_entity == "suffix":
        for key2add, value2add in entity2add.items():
            ent_out[key2add] = value2add
    
    if "suffix" in locals():
        ent_out["suffix"]  = suffix
        
    if "extension" in locals():
        ent_out["extension"]  = extension
    

    return ent_out


def _entity2str(entity:dict):
    """
    This function converts an entity dictionary to a string.
    
    Parameters:
    ----------
    entity: dict
        Dictionary containing the entities
        
    Returns:
    -------
    ent_string: str
        String containing the entities
        
    """
    entity = entity.copy()
    
    if "suffix" in entity:
        suffix = entity["suffix"]
        entity.pop("suffix", None)
    if "extension" in entity:
        extension = entity["extension"]
        entity.pop("extension", None)

    ent_string = ""
    for key, value in entity.items():
        ent_string = ent_string + key + "-" + value + "_"
    
    if "suffix" in locals():
        ent_string = ent_string + suffix
    else:
        ent_string = ent_string[0:-1]
    
    if "extension" in locals():
        ent_string = ent_string + "." + extension

    return ent_string



# This function copies the BIDs folder and its derivatives for e given subjects to a new location
def _copy_bids_folder(
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

        cltmisc._printprogressbar(
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
