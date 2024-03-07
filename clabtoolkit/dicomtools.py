import clabtoolkit.misctools as cltmisc
import os
from glob import glob
import subprocess

def sum(a, b):
    return a + b


def _uncompress_dicom_session(dic_dir: str, subj_ids = None):
    """
    Uncompress session folders
    @params:
        dic_dir     - Required  : Directory containing the subjects. It assumes an organization in:
        <subj_id>/<session_id>/<series_id>(Str)
    """

    if subj_ids is None:
        # Listing the subject ids inside the dicom folder
        my_list = os.listdir(dic_dir)
        subj_ids = []
        for it in my_list:
            if "sub-" in it:
                subj_ids.append(it)
        subj_ids.sort()
    else:
        if isinstance(subj_ids, str):
            # Read  the text file and save the lines in a list
            with open(subj_ids, "r") as file:
                subj_ids = file.readlines()
                subj_ids = [x.strip() for x in subj_ids]
        elif not isinstance(subj_ids, list):
            raise ValueError("The subj_ids parameter must be a list or a string")
        

    # Failed sessions
    fail_sess = []

    # Loop around all the subjects
    nsubj = len(subj_ids)
    for i, subj_id in enumerate(subj_ids):  # Loop along the IDs
        subj_dir = os.path.join(dic_dir, subj_id)

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
        for ses_tar in glob(
            subj_dir + os.path.sep + "*.tar.gz"
        ):  # Loop along the session
            #         print('SubjectId: ' + subjId + ' ======>  Session: ' +  sesId)
            # Compress only if it is a folder
            if os.path.isfile(ses_tar):
                try:
                    # Compressing the folder
                    subprocess.run(
                        ["tar", "xzf", ses_tar, "-C", subj_dir],
                        stdout=subprocess.PIPE,
                        universal_newlines=True,
                    )

                    # Removing the uncompressed dicom folder
                    # subprocess.run(
                    #     ['rm', '-r', ses_tar], stdout=subprocess.PIPE, universal_newlines=True)

                except:
                    fail_sess.append(ses_tar)
    if fail_sess:
        print("THE PROCESS FAILED TO UNCOMPRESS THE FOLLOWING TAR FILES:")
        for i in fail_sess:
            print(i)
    print(" ")
    print("End of the uncompression process.")


def _compress_dicom_session(dic_dir: str, subj_ids = None):
    """
    Compress session folders
    @params:
        dic_dir     - Required  : Directory containing the subjects. It assumes an organization in:
        <subj_id>/<session_id>/<series_id>(Str)
    """

    if subj_ids is None:
        # Listing the subject ids inside the dicom folder
        my_list = os.listdir(dic_dir)
        subj_ids = []
        for it in my_list:
            if "sub-" in it:
                subj_ids.append(it)
        subj_ids.sort()
    else:
        if isinstance(subj_ids, str):
            # Read  the text file and save the lines in a list
            with open(subj_ids, "r") as file:
                subj_ids = file.readlines()
                subj_ids = [x.strip() for x in subj_ids]
        elif not isinstance(subj_ids, list):
            raise ValueError("The subj_ids parameter must be a list or a string")

    # Failed sessions
    fail_sess = []

    # Loop around all the subjects
    nsubj = len(subj_ids)
    for i, subj_id in enumerate(subj_ids):  # Loop along the IDs
        subj_dir = os.path.join(dic_dir, subj_id)

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
            #         print('SubjectId: ' + subjId + ' ======>  Session: ' +  sesId)
            # Compress only if it is a folder
            if os.path.isdir(ses_dir):
                tar_filename = ses_dir + ".tar.gz"
                try:
                    # Compressing the folder
                    subprocess.run(
                        ["tar", "-C", subj_dir, "-czvf", tar_filename, ses_id],
                        stdout=subprocess.PIPE,
                        universal_newlines=True,
                    )

                    # Removing the uncompressed dicom folder
                    subprocess.run(
                        ["rm", "-r", ses_dir],
                        stdout=subprocess.PIPE,
                        universal_newlines=True,
                    )

                except:
                    fail_sess.append(ses_dir)
    if fail_sess:
        print("THE PROCESS FAILED TO COMPRESS THE FOLLOWING SESSIONS:")
        for i in fail_sess:
            print(i)
    print(" ")
    print("End of the compression process.")
