import os
import numpy as np
import nibabel as nib
from glob import glob
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

