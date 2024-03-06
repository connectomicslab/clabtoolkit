import argparse
import os
import numpy as np
import nibabel as nib
from datetime import datetime
from glob import glob
from pathlib import Path
from shutil import copyfile
import subprocess
from skimage import measure
import numpy as np
import pandas as pd
import subprocess


def _annot2gcs(
    annot_folder: str,
    gcs_folder: str,
    freesurfer_dir: str = "/opt/freesurfer/subjects/",
):
    """
    Convert FreeSurfer annot files to gcs files
    @params:
        annot_folder     - Required  : Folder containing the annot files:
        gcs_folder       - Required  : Output folder:
        freesurfer_dir   - Optional  : FreeSurfer directory:
    """

    # Create te gcs folder if it does not exist
    if not os.path.exists(gcs_folder):
        os.makedirs(gcs_folder)

    os.environ["SUBJECTS_DIR"] = freesurfer_dir

    any_annots = glob(os.path.join(annot_folder, "*.annot"))
    out_gcs = []
    for annot_file in any_annots:
        print(annot_file)
        annot_name = os.path.basename(annot_file)
        annot_name = annot_name.replace(".annot", "")
        hemi = annot_name[0:2]

        print(annot_name)
        # Read the annot file using nibabel
        sdata = nib.freesurfer.io.read_annot(annot_file)

        # Read the colors from lh_annot
        codes = sdata[0]
        colors = sdata[1][1:, 0:3]
        stnames = sdata[2][1:]

        # Create the lookup table for the right hemisphere
        luttable = [
            "{:<4} {:<40} {:>3} {:>3} {:>3} {:>3}".format(
                0, "ctx-unknown", 250, 250, 250, 0
            )
        ]
        for roi_pos, roi_name in enumerate(stnames):
            temp_name = "ctx-{}".format(roi_name.decode("utf-8"))
            luttable.append(
                "{:<4} {:<40} {:>3} {:>3} {:>3} {:>3}".format(
                    roi_pos + 1,
                    temp_name,
                    colors[roi_pos, 0],
                    colors[roi_pos, 1],
                    colors[roi_pos, 2],
                    0,
                )
            )

        # Save the lookup table for the left hemisphere
        ctab_file = os.path.join(gcs_folder, annot_name + ".ctab")
        with open(ctab_file, "w") as colorLUT_f:
            colorLUT_f.write("\n".join(luttable))

        # Create the gcs file
        gcs_file = os.path.join(gcs_folder, annot_name + ".gcs")

        cmd_cont = [
            "mris_ca_train",
            "-n",
            "2",
            "-t",
            ctab_file,
            hemi,
            "sphere.reg",
            annot_file,
            "fsaverage",
            gcs_file,
        ]
        subprocess.run(
            cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
        )  # Running container command

        out_gcs.append(gcs_file)

    return out_gcs


# This function removes the B0s volumes located at the end of the diffusion 4D volume.
def _remove_empty_dwi_Volume(dwifile):
    """
    Remove the B0s volumes located at the end of the diffusion 4D volume.
    @params:
        dwifile     - Required  : Diffusion 4D volume:
    """
    
    # Creating the name for the json file
    pth = os.path.dirname(dwifile)
    fname = os.path.basename(dwifile)
    if fname.endswith(".nii.gz"):
        flname = fname[0:-7]
    elif fname.endswith(".nii"):
        flname = fname[0:-4]

    # Creating filenames
    bvecfile = os.path.join(pth, flname + ".bvec")
    bvalfile = os.path.join(pth, flname + ".bval")
    jsonfile = os.path.join(pth, flname + ".json")

    # Loading bvalues
    if os.path.exists(bvalfile):
        bvals = np.loadtxt(bvalfile, dtype=float, max_rows=5).astype(int)

        tempBools = list(bvals < 10)
        if tempBools[-1]:
            if os.path.exists(bvecfile):
                bvecs = np.loadtxt(bvecfile, dtype=float)

            # Reading the image
            mapI = nib.load(dwifile)
            diffData = mapI.get_fdata()
            affine = mapI.affine

            # Detecting the clusters of B0s
            lb_bvals = measure.label(bvals, 2)

            lab2rem = lb_bvals[-1]
            vols2rem = np.where(lb_bvals == lab2rem)[0]
            vols2keep = np.where(lb_bvals != lab2rem)[0]

            # Removing the volumes
            array_data = np.delete(diffData, vols2rem, 3)

            # Temporal image and diffusion scheme
            tmp_dwi_file = os.path.join(pth, flname + ".nii.gz")
            array_img = nib.Nifti1Image(array_data, affine)
            nib.save(array_img, tmp_dwi_file)

            select_bvecs = bvecs[:, vols2keep]
            select_bvals = bvals[vols2keep]
            select_bvals.transpose()

            # Saving new bvecs and new bvals
            tmp_bvecs_file = os.path.join(pth, flname + ".bvec")
            np.savetxt(tmp_bvecs_file, select_bvecs, fmt="%f")

            tmp_bvals_file = os.path.join(pth, flname + ".bval")
            np.savetxt(tmp_bvals_file, select_bvals, newline=" ", fmt="%d")

    return dwifile, bvecfile, bvalfile, jsonfile


# Print iterations progress
def _printprogressbar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printend="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printend    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledlength = int(length * iteration // total)
    bar = fill * filledlength + "-" * (length - filledlength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printend)
    # Print New Line on Complete
    if iteration == total:
        print()


def _uncompress_dicom_session(dic_dir: str):
    """
    Uncompress session folders
    @params:
        dic_dir     - Required  : Directory containing the subjects. It assumes an organization in:
        <subj_id>/<session_id>/<series_id>(Str)
    """

    # Listing the subject ids inside the dicom folder
    my_list = os.listdir(dic_dir)
    subj_ids = []
    for it in my_list:
        if "sub-" in it:
            subj_ids.append(it)
    subj_ids.sort()

    # Failed sessions
    fail_sess = []

    # Loop around all the subjects
    nsubj = len(subj_ids)
    for i, subj_id in enumerate(subj_ids):  # Loop along the IDs
        subj_dir = os.path.join(dic_dir, subj_id)

        _printprogressbar(
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


def _uncompress_dicom_session_tosolve(dic_dir: str):
    """
    Uncompress session folders
    @params:
        dic_dir     - Required  : Directory containing the subjects. It assumes an organization in:
        <subj_id>/<session_id>/<series_id>(Str)
    """

    # Listing the subject ids inside the dicom folder
    my_list = os.listdir(dic_dir)
    subj_ids = []
    for it in my_list:
        if "sub-" in it:
            subj_ids.append(it)
    subj_ids.sort()

    # Failed sessions
    fail_sess = []

    # Loop around all the subjects
    nsubj = len(subj_ids)
    for i, subj_id in enumerate(subj_ids):  # Loop along the IDs
        subj_dir = os.path.join(dic_dir, subj_id)

        _printprogressbar(
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

                    ses_id = os.path.basename(ses_tar)[:-7]
                    outDir = os.path.join(subj_dir, ses_id)
                    os.makedirs(outDir)
                    t = os.path.join(
                        subj_dir,
                        "media",
                        "yaleman",
                        "Database",
                        "IMAGING-PROJECTS",
                        "ENDOCRINOLOGY_PROJECT",
                        "Dicom",
                        subj_id,
                        ses_id,
                    )
                    cmd = "mv " + t + os.path.sep + "* " + outDir
                    os.system(cmd)
                    os.system("rm -r " + subj_dir + os.path.sep + "media")

                    # Removing the uncompressed dicom folder
                    subprocess.run(
                        ["rm", "-r", ses_tar],
                        stdout=subprocess.PIPE,
                        universal_newlines=True,
                    )

                except:
                    fail_sess.append(ses_tar)
    if fail_sess:
        print("THE PROCESS FAILED TO UNCOMPRESS THE FOLLOWING TAR FILES:")
        for i in fail_sess:
            print(i)
    print(" ")
    print("End of the uncompression process.")


def _compress_dicom_session(dic_dir: str):
    """
    Compress session folders
    @params:
        dic_dir     - Required  : Directory containing the subjects. It assumes an organization in:
        <subj_id>/<session_id>/<series_id>(Str)
    """

    # Listing the subject ids inside the dicom folder
    my_list = os.listdir(dic_dir)
    subj_ids = []
    for it in my_list:
        if "sub-" in it:
            subj_ids.append(it)
    subj_ids.sort()

    # Failed sessions
    fail_sess = []

    # Loop around all the subjects
    nsubj = len(subj_ids)
    for i, subj_id in enumerate(subj_ids):  # Loop along the IDs
        subj_dir = os.path.join(dic_dir, subj_id)

        _printprogressbar(
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

        _printprogressbar(
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




def _detect_dwi_nvols(bids_dir: str, out_dir: str):
    """
    Copy full anat  folders
    @params:
        bids_dir     - Required  : BIDs dataset directory:
        out_dir      - Required  : Output directory:
    """

    # Listing the subject ids inside the dicom folder
    my_list = os.listdir(bids_dir)
    subj_ids = []
    for it in my_list:
        if "sub-" in it:
            subj_ids.append(it)
    subj_ids.sort()

    # Failed sessions
    fail_sess = []

    # Loop around all the subjects
    nsubj = len(subj_ids)
    for i, subj_id in enumerate(subj_ids):  # Loop along the IDs
        subj_dir = os.path.join(bids_dir, subj_id)
        out_subj_dir = os.path.join(out_dir, subj_id)

        _printprogressbar(
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

            #         print('SubjectId: ' + subjId + ' ======>  Session: ' +  sesId)
            # Copying the anat folder
            if os.path.isdir(ses_dir):
                anat_fold = os.path.join(ses_dir, "dwi")

                try:
                    # Creating destination directory
                    subprocess.run(
                        ["mkdir", "-p", out_ses_dir],
                        stdout=subprocess.PIPE,
                        universal_newlines=True,
                    )

                    # Copying the folder
                    subprocess.run(
                        ["cp", "-r", anat_fold, out_ses_dir],
                        stdout=subprocess.PIPE,
                        universal_newlines=True,
                    )

                except:
                    fail_sess.append(anat_fold)
    if fail_sess:
        print("THE PROCESS FAILED TO COMPRESS THE FOLLOWING SESSIONS:")
        for i in fail_sess:
            print(i)
    print(" ")
    print("End of the compression process.")


def read_fscolorlut(lutFile: str):
    """
    Reading freesurfer lut file
    @params:
        lutFile     - Required  : FreeSurfer color lut:
    """
    # Readind a color LUT file
    fid = open(lutFile)
    LUT = fid.readlines()
    fid.close()

    # Make dictionary of labels
    LUT = [row.split() for row in LUT]
    st_names = []
    st_codes = []
    cont = 0
    for row in LUT:
        if (
            len(row) > 1 and row[0][0] != "#" and row[0][0] != "\\\\"
        ):  # Get rid of the comments
            st_codes.append(int(row[0]))
            st_names.append(row[1])
            if cont == 0:
                st_colors = np.array([[int(row[2]), int(row[3]), int(row[4])]])
            else:
                ctemp = np.array([[int(row[2]), int(row[3]), int(row[4])]])
                st_colors = np.append(st_colors, ctemp, axis=0)
            cont = cont + 1

    return st_codes, st_names, st_colors


def _convertluts_freesurfer2fsl(freelut: str, fsllut: str):
    """
    Convert FreeSurfer lut file to FSL lut file
    @params:
        freelut     - Required  : FreeSurfer color lut:
        fsllut      - Required  : FSL color lut:
    """

    # Reading FreeSurfer color lut
    st_codes_lut, st_names_lut, st_colors_lut = read_fscolorlut(freelut)

    lut_lines = []
    for roi_pos, st_code in enumerate(st_codes_lut):
        st_name = st_names_lut[roi_pos]
        lut_lines.append(
            "{:<4} {:>3.5f} {:>3.5f} {:>3.5f} {:<40} ".format(
                st_code,
                st_colors_lut[roi_pos, 0] / 255,
                st_colors_lut[roi_pos, 1] / 255,
                st_colors_lut[roi_pos, 2] / 255,
                st_name,
            )
        )

    with open(fsllut, "w") as colorLUT_f:
        colorLUT_f.write("\n".join(lut_lines))


# Load csv table and save the column participant_id as a list called subj_ids
# csv_file = '/media/HPCdata/Jagruti_DTI_DSI/participants_DSI_DTI.csv'
# import pandas as pd
# df = pd.read_csv(csv_file)
# subj_ids = df['participant_id'].tolist()

# # Add 'sub-' to the subject ids
# subj_ids = ['sub-' + i for i in subj_ids]


# _copy_bids_folder('/media/HPCdata/Mindfulness/', '/media/yaleman/HagmannHDD/Test/',["anat", "dwi"], ["sub-S001"], include_derivatives=True, deriv_dir='/media/HPCdata/Mindfulness/derivatives')
# _uncompress_dicom_session('/media/yaleman/Database/IMAGING-PROJECTS/Dicom')

# _compress_dicom_session('/media/COSAS/Yasser/Work2Do/ReconVertDatabase/Dicom')

# atdir = '/media/COSAS/Yasser/Work2Do/ReconVertDatabase/derivatives/chimera-atlases/sub-CHUVA001/ses-V2/anat'
# freelut = os.path.join(atdir, 'sub-CHUVA001_ses-V2_run-1_space-orig_atlas-chimeraBFIIHIFIF_desc-grow0mm_dseg.lut')
# fsllut = '/home/yaleman/BFIIHIFIF.lut'
# _convertluts_freesurfer2fsl(freelut, fsllut)
#
# freelut = os.path.join(atdir, 'sub-CHUVA001_ses-V2_run-1_space-orig_atlas-chimeraHFIIIIFIF_desc-7p1grow0mm_dseg.lut')
# fsllut = '/home/yaleman/HFIIIIFIF.lut'
# _convertluts_freesurfer2fsl(freelut, fsllut)
#
# freelut = os.path.join(atdir, 'sub-CHUVA001_ses-V2_run-1_space-orig_atlas-chimeraLFMIIIFIF_desc-scale1grow0mm_dseg.lut')
# fsllut = '/home/yaleman/LFMIIIFIF.lut'
# _convertluts_freesurfer2fsl(freelut, fsllut)

# dwifile = '/media/yaleman/Database/LENNARDS/BIDsDataset/sub-LEN0199/ses-20210804175649/dwi/sub-LEN0199_ses-20210804175649_run-1_acq-dtiNdir30_dwi.nii.gz'
# _remove_empty_dwi_Volume(dwifile)

# freelut = '/media/COSAS/Yasser/Work2Do/ReconVertDatabase/derivatives/chimera-atlases/sub-CHUVA001/ses-V2/anat/sub-CHUVA001_ses-V2_run-1_space-orig_atlas-chimeraLFMIIIFIF_desc-scale5growwm_dseg.lut'
# fsllut = '/home/yaleman/test.fsllut'

# _uncompress_dicom_session_tosolve('/media/yaleman/Database/IMAGING-PROJECTS/test/Dicom')
