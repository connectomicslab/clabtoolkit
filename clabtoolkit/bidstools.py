import os
import shutil
import clabtoolkit.misctools as cltmisc
from typing import Union


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############           Methods dedicated to work with BIDs naming conventions           ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################

def _entity2str(entity:dict) -> str:
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


####################################################################################################
def _str2entity(string:str) -> dict:
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

####################################################################################################
def _delete_entity(entity: Union[dict, str], 
                    key2rem:Union[list, str]) -> Union[dict,  str]:
    """
    This function removes some entities from a dictionary or string following the BIDs naming convention. 

    Parameters:
    ----------
    entity: dict or str
        Dictionary containing the entities
    key2rem: list or str
        List of keys to remove
        
    Returns:
    -------
    entity_out: dict
        Dictionary containing the entities without the keys to remove
    
    """
    
    is_string = False
    if isinstance(entity, str):
        entity_out = _str2entity(entity)
        is_string = True
        
    elif isinstance(entity, dict):
        entity_out = entity.copy()
    
    else:
        raise ValueError("The entity must be a dictionary or a string")
    
    if isinstance(key2rem, str):
        key2rem = [key2rem]
    

    for key in key2rem:

        if key in entity_out:
            entity_out.pop(key, None)
    
    if is_string:
        entity_out = _entity2str(entity_out)
    
    return entity_out

####################################################################################################
def _replace_entity_value(entity:Union[dict, str], 
                            ent2replace:dict, 
                            verbose:bool = False) -> Union[dict, str]:
    """
    This function replace an entity value from an entity dictionary
    
    Parameters:
    ----------
    entity: dict
        Dictionary containing the entities
    
    ent2replace: dict
        Dictionary containing the entities to replace and their new values
            
    Returns:
    -------
    entity_out: dict or str
        Dictionary containing the entities with the new values or a string if the input was a string
        
    """

    is_string = False
    if isinstance(entity, str):
        entity_out = _str2entity(entity)
        is_string = True
    
    elif isinstance(entity, dict):
        entity_out = entity.copy()
    
    else:
        raise ValueError("The entity must be a dictionary or a string")
    
    ent2replace = cltmisc._remove_empty_keys_or_values(ent2replace) 
    
    # Replace values from dictionary
    ent_list = list(entity_out.keys())
        
    ent_name = list(ent2replace.keys())

    for key in ent_name:
        if key in ent_list:

            if len(ent2replace[key]) !=0:
                entity_out[key] = ent2replace[key]
            else:
                if verbose
                    print("Warning: The value to replace is empty for the entity: ", key)
        else:
            if verbose:
                print(f"The entity {key} is not in the entities dictionary") 
            
    # Convert the dictionary to a string if the input was a string
    if is_string:
        entity_out = _entity2str(entity_out)
    
    return entity_out

####################################################################################################
def _replace_entity_key(entity:Union[dict, str], 
                            keys2replace:dict,
                            verbose:bool = False) -> Union[dict, str]:
    """
    This function replace the entity keys names from an entity dictionary
    
    Parameters:
    ----------
    entity: dict or str
        Dictionary containing the entities or a string that follows the BIDs naming specifications
    
    keys2replace: dict
        Dictionary containing the key to be renamed and the new key
            
    Returns:
    -------
    entity_out: dict or str
        Dictionary containing the entities with the new keys or a string if the input was a string
    
    """

    is_string = False
    if isinstance(entity, str):
        entity = _str2entity(entity)
        is_string = True
        
    elif isinstance(entity, dict):
        pass
        
    else:
        raise ValueError("The entity must be a dictionary or a string")
    
    if not isinstance(keys2replace, dict):
        raise ValueError("The keys2replace must be a dictionary")
    
    keys2replace = cltmisc._remove_empty_keys_or_values(keys2replace) 
    
    # Replace key names from dictionary
    for old_key in keys2replace:
        if old_key not in entity:
            print(f"Warning: Key '{old_key}' not found in the original dictionary.")
    
    entity_out = {keys2replace.get(k, k): v for k, v in entity.items()}
    
    # Convert the dictionary to a string if the input was a string
    if is_string:
        entity_out = _entity2str(entity_out)
    
    return entity_out


####################################################################################################
def _insert_entity(entity:Union[dict, str], 
                    entity2add:dict, 
                    prev_entity:str = None) -> Union[dict, str]:
    """
    This function adds entities to an existing entity dictionary.
    
    Parameters:
    ----------
    entity: dict or str
        Dictionary containing the entities or a string that follows the BIDs naming specifications
    
    entity2add: dict
        Dictionary containing the entities that will be added
    
    prev_entity: str
        Previous entity. This value will serve as reference to add the new entities.
        
    Returns:
    -------
    ent_out: dict or str
        Dictionary containing the entities with the new entities added or a string if the input was a string
        
    """
    
    is_string = False
    if isinstance(entity, str):
        entity = _str2entity(entity)
        is_string = True
        
    elif isinstance(entity, dict):
        pass
        
    else:
        raise ValueError("The entity must be a dictionary or a string")
    
    entity2add = cltmisc._remove_empty_keys_or_values(entity2add)
    
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
    
    # Converting to string if the input was a string
    if is_string:
        ent_out = _entity2str(ent_out)

    return ent_out

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
