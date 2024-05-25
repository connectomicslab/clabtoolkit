import clabtoolkit.misctools as cltmisc
import os
from glob import glob
import subprocess
import pandas as pd
import sys
import pydicom
from datetime import datetime
from pathlib import Path
from shutil import copyfile
import concurrent.futures
import numpy as np
from rich.progress import Progress


def _org_conv_dicoms(in_dic_dir: str, 
                    out_dic_dir: str, 
                    demog_file: str=None, 
                    ids_file:str=None,
                    ses_id:str=None,
                    nosub:bool=False,
                    booldic:bool=True,
                    boolforce:bool=False,
                    boolcomp:bool=False):
    """
    This method organizes the DICOM files in sessions and series. It could also uses the demographics file to define the session ID.
    
    @params:
        in_dic_dir     - Required  : Directory containing the subjects. It assumes all individual folders inside the directory as individual subjects.
                                    The subjects directory should start with 'sub-' otherwise the subjects will not be considered unless the "nosub" 
                                    variable is set to True.
        out_dic_dir    - Required  : Output directory where the organized DICOM files will be saved. A new folder called 'Dicom' will be created inside this directory.
        demog_file     - Optional  : Demographics file containing the information about the subjects. The file should contain the following mandatory columns:
                                    'participant_id', ' session_id', 'acq_date'. Other columns such as 'birth_date', 'sex', 'group_id' or 'scanner_id' could be added.
        ids_file       - Optional  : Text file containing the list of subject IDs to be considered. The file should contain the subject IDs in a single column.
        ses_id         - Optional  : Session ID to be added to the session name. If not provided, the session ID will be the date of the study or the session ID 
                                    extracted from the demographics table.
        nosub          - Optional  : Boolean variable to consider the subjects that do not start with 'sub-'. Default is False.
        booldic        - Optional  : Boolean variable to organice the DICOM files. Default is True. If False it will leave the folders as they are.
        boolcomp       - Optional  : Boolean variable to compress the sessions containing the organized DICOM files. Default is False. If True it will compress the sessions.
        
    """
    nthreads = os.cpu_count()
    
    # Output directories
    dicomDir = os.path.join(out_dic_dir, 'Dicom')

    my_list = os.listdir(in_dic_dir)
    subj_ids = []
    for it in my_list:
        if nosub == False:
            if 'sub-' in it:
                subj_ids.append(it)
        else:
            subj_ids.append(it)

    subj_ids.sort()

    # If subj_ids is empty do not continue
    if not subj_ids:
        print('No subjects found in the input directory')
        sys.exit()
    
    if ids_file != None:
        if os.path.isfile(ids_file):
            subj_ids = cltmisc._select_ids_from_file(subj_ids, ids_file)
            
        else:
            s_ids = ids_file.split(',')

            if nosub == False:
                temp_ids = [s.strip('sub-') for s in subj_ids]
                s_ids = cltmisc._list_intercept(s_ids, temp_ids)

            if not s_ids:
                s_ids = subj_ids
            else:
                s_ids = ["sub-"+ s for s in s_ids]
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
    
    # Creating the progress bars
    with Progress() as pb:
        t2 = pb.add_task('[green]Subjects...', total=n_subj)
        
        
        for cont_subj, subj_id in enumerate(subj_ids):  # Loop along the IDs

            # cont_subj += 1
            # cltmisc._printprogressbar(cont_subj, n_subj,
            #                 'Processing subject ' + subj_id + ': ' + '(' + str(cont_subj) + '/' + str(n_subj) + ')')

            subj_dir = os.path.join(in_dic_dir, subj_id)

            if os.path.isdir(subj_dir):
                # Default value for these variables for each subject
                gendVar = 'Unknown'
                groupVar = 'Unknown'
                AgeatScan = 'Unknown'
                subTB = None
                date_times = []
                
                if demobool:
                    # Sub-table containing only the selected ID
                    subTB = demoDB[demoDB['participant_id'].str.contains(subj_id.split('-')[-1])]

                    # Date times of all the series acquired for the current subject
                    nrows = np.shape(subTB)[0]
                    for nr in np.arange(0, nrows):
                        temp = subTB.iloc[nr]['acq_date']
                        tempVar = temp.split('/')
                        date_time = datetime(day=int(tempVar[1]), month=int(tempVar[0]), year=int(tempVar[2]))
                        date_times.append(date_time)

                if booldic:
                    dicom_files = cltmisc._detect_recursive_files(subj_dir)
                    ses_idprev = []
                    ser_idprev = []
                    
                    
                    ndics = len(dicom_files)
                    # if nthreads > 4:
                    #     nthreads = nthreads - 4
                    # with concurrent.futures.ProcessPoolExecutor(nthreads) as executor:
                    # #     results = [executor.submit(do_something, sec) for sec in secs]
                    #     results = list(executor.map(_copy_dicom_file, dicom_files,
                    #     [subj_id] * ndwis, [out_dic_dir] * ndwis, [ses_id] * ndwis, [date_times] * ndwis, [demobool] * ndwis, [subTB] * ndwis, [boolforce] * ndwis))

                    t1 = pb.add_task(f'[red]Copying DICOMs: Subject {cont_subj + 1}/{n_subj} ', total=ndics)
                    
                    for cont_dic, dfiles in enumerate(dicom_files):
                        ser_dir = _copy_dicom_file(dfiles, subj_id, out_dic_dir, ses_id, date_times, demobool, subTB, boolforce)
                        all_ser_dirs.append(ser_dir)
                        pb.update(task_id=t1, completed=cont_dic+1)
                    pb.update(task_id=t1, completed=ndics)
                    
                else:

                    for ses_id in os.listdir(subj_dir):  # Loop along the session
                        ses_dir = os.path.join(subj_dir, ses_id)
                        if not ses_id[-2].isalpha():
                            if demobool:  # Adding the Visit ID to the last part o the session ID only in the DICOM Folder
                                tempVar = ses_id.split('-')[-1]
                                sdate_time = datetime.strptime(tempVar, '%Y%m%d%H%M%S')
                                timediff = np.array(date_times) - np.array(sdate_time)
                                clostd = np.argmin(abs(timediff))
                                visitVar = subTB.iloc[clostd]['session_id']
                                newses_id = ses_id + visitVar
                                newses_dir = os.path.join(subj_dir, newses_id)
                                os.rename(ses_dir, newses_dir)
                                ses_dir = newses_dir

                        if os.path.isdir(ses_dir):
                            for ser_id in os.listdir(ses_dir):  # Loop along the series
                                serDir = os.path.join(ses_dir, ser_id)

                                if os.path.isdir(serDir):
                                    all_ser_dirs.append(serDir)
            else:
                print('Subject: ' + subj_id + ' does not exist.')
                
            pb.update(task_id=t2, completed=cont_subj+1)
        pb.update(task_id=t2, completed=n_subj) 
        
    all_ser_dirs = list(set(all_ser_dirs))
    all_ser_dirs.sort()


def _copy_dicom_file(dic_file: str, 
                    subj_id: str,
                    out_dic_dir: str,
                    ses_id: str = None,
                    date_times: list = None,
                    demogbool: bool = False,
                    demog_tab: pd.DataFrame = None,
                    force: bool = False):
    """
    Function to copy the DICOM files to the output directory.
    
    Parameters:
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
        
    Returns:
    --------
    dest_dic_dir: str
        Destination directory where the DICOM file was copied.
        
        
    """
    
    
    try:
        dataset = pydicom.read_file(dic_file)
        dic_path = os.path.dirname(dic_file)
        dic_name = os.path.basename(dic_file)

        # Extracting the study date from DICOM file
        attributes = dataset.dir("")

        if attributes:
            sdate = dataset.data_element('StudyDate').value
            stime = dataset.data_element('StudyTime').value
            year = int(sdate[:4])
            month = int(sdate[4:6])
            day = int(sdate[6:8])

            # Date format
            sdate_time = datetime(day=day, month=month, year=year)

            # Creating default current Session ID
            ses_id, ser_id = _create_session_series_names(dataset)

            if not ses_id == None:
                ses_id = 'ses-' + ses_id

            if '000000' in ses_id and ser_id in ser_idprev:
                ses_id = ses_idprev

            #visitId = dfiles.split('/')[8].split('-')[1]
            #ses_id = 'ses-'+ visitId
            
            ses_idprev = ses_id
            ser_idprev = ser_id

            # Changing the session Id in case we have access to the demographics file
            if demogbool:
                timediff = np.array(date_times) - np.array(sdate_time)
                clostd = np.argmin(abs(timediff))
                visitVar = demog_tab.iloc[clostd]['session_id']
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


# Extract Series Id and Sessions Id from a pydicom object
def _create_session_series_names(dataset):
    # % This function creates the session and the series name for a dicom object

    # Extracting the study date from DICOM file
    attributes = dataset.dir("")
    sdate = dataset.data_element('StudyDate').value
    stime = dataset.data_element('StudyTime').value

    ########### ========== Creating current Session ID
    if sdate and stime:
        ses_id = str(sdate) + str(int(np.floor(float(stime))))
    elif sdate and not stime:
        ses_id = str(sdate) + '000000'
    elif stime and not sdate:
        ses_id = '00000000' + str(stime)

    ########### ========== Creating current Series ID
    if any("SeriesDescription" in s for s in attributes):
        ser_id = dataset.data_element('SeriesDescription').value
    elif any("SeriesDescription" in s for s in attributes) == False and any("SequenceName" in s for s in attributes):
        ser_id = dataset.data_element('SequenceName').value
    elif any("SeriesDescription" in s for s in attributes) == False and any(
            "SequenceName" in s for s in attributes) == False and any("ProtocolName" in s for s in attributes):
        ser_id = dataset.data_element('ProtocolName').value
    elif any("SeriesDescription" in s for s in attributes) == False and any(
            "SequenceName" in s for s in attributes) == False and any(
            "ProtocolName" in s for s in attributes) == False and any(
            "ScanningSequence" in s for s in attributes) and any("SequenceVariant" in s for s in attributes):
        ser_id = dataset.data_element('ScanningSequence').value + '_' + dataset.data_element('SequenceVariant').value
    else:
        ser_id = 'NoSerName'

    # Removing and substituting unwanted characters
    ser_id = ser_id.replace(" ", "_")
    ser_id = ser_id.replace("/", "_")

    # This function removes some characters from a string
    ser2rem = ['*', '+', '(', ')', '=', ',', '>', '<', ';', ':', '"', "'", '?', '!', '@', '#', '$', '%', '^', '&', '*']
    for cad in ser2rem:
        ser_id = ser_id.replace(cad, '')

    # Removing the dupplicated _ characters and replacing the remaining by -
    ser_id = cltmisc._rem_dupplicate_char(ser_id, '_')
    ser_id = ser_id.replace("_", "-")

    if any("SeriesNumber" in s for s in attributes):
        serNumb = dataset.data_element('SeriesNumber').value

    # Adding the series number
    sNumb = f'{int(serNumb):04d}'
    ser_id = sNumb + '-' + ser_id

    return ses_id, ser_id

def _uncompress_dicom_session(dic_dir: str, subj_ids=None):
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


def _compress_dicom_session(dic_dir: str, subj_ids=None):
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
