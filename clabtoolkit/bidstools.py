import os
import shutil
import clabtoolkit as clt


# This function copies the BIDs folder and its derivatives for e given subjects to a new location
def _copy_bids_folder(
    bids_dir: str, out_dir: str, fold2copy: list = ["anat"], subjs2copy: str = None, deriv_dir: str = None,
include_derivatives: bool = False):
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

        clt._printprogressbar(
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
                    if not directory.startswith(".") and os.path.isdir(os.path.join(ses_dir, directory)):
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

                        out_pipe_dir = os.path.join(out_dir, "derivatives", os.path.basename(pipe_dir))

                        pipe_indiv_subj_in = os.path.join(pipe_dir, subj_id, ses_id)
                        pipe_indiv_subj_out = os.path.join(out_pipe_dir, subj_id, ses_id)

                        if os.path.isdir(pipe_indiv_subj_in):
                            try:
                                # Creating destination directory using make directory
                                os.makedirs(pipe_indiv_subj_out, exist_ok=True)

                                # Copying the folder
                                shutil.copytree(pipe_indiv_subj_in, pipe_indiv_subj_out, dirs_exist_ok=True)

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
