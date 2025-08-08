# pylint: disable=import-error
import os
import sys
import subprocess
from pathlib import Path
from glob import glob
from typing import Union
import numpy as np
import nibabel as nib

# Importing local modules
from . import misctools as cltmisc
from . import bidstools as cltbids
from . import imagetools as cltimg
from . import parcellationtools as cltparc


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############           Section 1: Methods dedicated to image segmentation               ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def abased_parcellation(
    t1: str,
    t1_temp: str,
    atlas: str,
    out_parc: str,
    xfm_output: str,
    atlas_type: str = "spam",
    interp: str = "Linear",
    cont_tech: str = "local",
    cont_image: str = None,
    force: bool = False,
):
    """
    Perform atlas-based parcellation using ANTs registration and transformation.

    Registers individual T1-weighted image to template space, then applies inverse
    transformation to bring atlas into subject space, creating subject-specific
    parcellation based on template atlas.

    Parameters
    ----------
    t1 : str
        Path to input T1-weighted image.

    t1_temp : str
        Path to T1-weighted template image for registration target.

    atlas : str
        Path to atlas image (either SPAM probabilities or discrete labels).

    out_parc : str
        Path for output parcellation file.

    xfm_output : str
        Base path/name for transformation files. Extensions and descriptors
        will be automatically added.

    atlas_type : str, optional
        Atlas format: 'spam' for probability maps or 'maxprob' for discrete labels.
        Default is 'spam'.

    interp : str, optional
        Interpolation method: 'Linear', 'NearestNeighbor', 'BSpline'. Default is 'Linear'.

    cont_tech : str, optional
        Container technology: 'local', 'singularity', 'docker'. Default is 'local'.

    cont_image : str, optional
        Container image specification. Default is None.

    force : bool, optional
        Whether to overwrite existing files. Default is False.

    Returns
    -------
    str
        Path to the generated parcellation file.

    Notes
    -----
    The method performs the following steps:
    1. ANTs SyN registration between subject T1 and template
    2. Generates affine and non-linear transformation files
    3. Applies inverse transformation to atlas
    4. For SPAM atlases, preserves probability values
    5. For maxprob atlases, uses nearest neighbor interpolation

    Generated transformation files follow BIDS naming conventions with
    descriptors: affine, warp, iwarp.

    Examples
    --------
    >>> # Basic SPAM atlas parcellation
    >>> parc_file = abased_parcellation(
    ...     t1='subject_T1w.nii.gz',
    ...     t1_temp='MNI152_T1_1mm.nii.gz',
    ...     atlas='AAL_SPAM.nii.gz',
    ...     out_parc='subject_AAL.nii.gz',
    ...     xfm_output='transforms/subject_to_MNI'
    ... )
    >>>
    >>> # Discrete atlas with nearest neighbor interpolation
    >>> parc_file = abased_parcellation(
    ...     t1='T1w.nii.gz',
    ...     t1_temp='template.nii.gz',
    ...     atlas='discrete_atlas.nii.gz',
    ...     out_parc='parcellation.nii.gz',
    ...     xfm_output='xfm/transform',
    ...     atlas_type='maxprob'
    ... )
    """

    ######## -- Registration to the template space  ------------ #
    # Creating spatial transformation folder
    stransf_dir = Path(os.path.dirname(xfm_output))
    stransf_name = os.path.basename(xfm_output)
    out_parc_dir = Path(os.path.dirname(out_parc))

    # If the directory does not exist create the directory and if it fails because it does not have write access send an error
    try:
        stransf_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(
            "The directory to store the spatial transformations does not have write access."
        )
        sys.exit()

    try:
        out_parc_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print("The directory to store the parcellation does not have write access.")
        sys.exit()

    # Spatial transformation files (Temporal).
    tmp_xfm_basename = os.path.join(stransf_dir, "temp")
    temp_xfm_affine = tmp_xfm_basename + "_0GenericAffine.mat"
    temp_xfm_nl = tmp_xfm_basename + "_1Warp.nii.gz"
    temp_xfm_invnl = tmp_xfm_basename + "_1InverseWarp.nii.gz"
    temp_xfm_invnlw = tmp_xfm_basename + "_InverseWarped.nii.gz"
    temp_xfm_nlw = tmp_xfm_basename + "_Warped.nii.gz"

    if stransf_name.endswith(".nii.gz"):
        stransf_name = stransf_name[:-7]
    elif stransf_name.endswith(".nii") or stransf_name.endswith(".mat"):
        stransf_name = stransf_name[:-4]

    if stransf_name.endswith("_xfm"):
        stransf_name = stransf_name[:-4]

    if "_desc-" in stransf_name:
        affine_name = cltbids.replace_entity_value(stransf_name, {"desc": "affine"})
        nl_name = cltbids.replace_entity_value(stransf_name, {"desc": "warp"})
        invnl_name = cltbids.replace_entity_value(stransf_name, {"desc": "iwarp"})
    else:
        affine_name = stransf_name + "_desc-affine"
        nl_name = stransf_name + "_desc-warp"
        invnl_name = stransf_name + "_desc-iwarp"

    # Affine transformation filename
    xfm_affine = os.path.join(stransf_dir, affine_name + "_xfm.mat")

    # Non-linear transformation filename
    xfm_nl = os.path.join(stransf_dir, nl_name + "_xfm.nii.gz")

    # Filename for the inverse of the Non-linear transformation
    xfm_invnl = os.path.join(stransf_dir, invnl_name + "_xfm.nii.gz")

    if not os.path.isfile(xfm_invnl) or force:
        # Registration to MNI template

        cmd_bashargs = [
            "antsRegistrationSyN.sh",
            "-d",
            "3",
            "-f",
            t1_temp,
            "-m",
            t1,
            "-t",
            "s",
            "-o",
            tmp_xfm_basename + "_",
        ]

        cmd_cont = cltmisc.generate_container_command(
            cmd_bashargs, cont_tech, cont_image
        )  # Generating container command
        subprocess.run(
            cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
        )  # Running container command

        # Changing the names
        cmd_bashargs = ["mv", temp_xfm_affine, xfm_affine]
        cmd_cont = cltmisc.generate_container_command(
            cmd_bashargs, cont_tech, cont_image
        )  # Generating container command
        subprocess.run(
            cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
        )  # Running container command

        cmd_bashargs = ["mv", temp_xfm_nl, xfm_nl]
        cmd_cont = cltmisc.generate_container_command(
            cmd_bashargs, cont_tech, cont_image
        )  # Generating container command
        subprocess.run(
            cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
        )  # Running container command

        cmd_bashargs = ["mv", temp_xfm_invnl, xfm_invnl]
        cmd_cont = cltmisc.generate_container_command(
            cmd_bashargs, cont_tech, cont_image
        )  # Generating container command
        subprocess.run(
            cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
        )  # Running container command

        # Deleting the warped images
        cmd_bashargs = ["rm", tmp_xfm_basename + "*Warped.nii.gz"]
        warped_imgs = glob(tmp_xfm_basename + "*Warped.nii.gz")
        if len(warped_imgs) > 0:
            for w_img in warped_imgs:
                os.remove(w_img)

    # Applying spatial transform to the atlas
    if not os.path.isfile(out_parc):

        if atlas_type == "spam":
            # Applying spatial transform
            cmd_bashargs = [
                "antsApplyTransforms",
                "-d",
                "3",
                "-e",
                "3",
                "-i",
                atlas,
                "-o",
                out_parc,
                "-r",
                t1,
                "-t",
                xfm_invnl,
                "-t",
                "[" + xfm_affine + ",1]",
                "-n",
                interp,
            ]

            cmd_cont = cltmisc.generate_container_command(
                cmd_bashargs, cont_tech, cont_image
            )  # Generating container command
            subprocess.run(
                cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
            )  # Running container command

        elif atlas_type == "maxprob":
            # Applying spatial transform
            cmd_bashargs = [
                "antsApplyTransforms",
                "-d",
                "3",
                "-i",
                atlas,
                "-o",
                out_parc,
                "-r",
                t1,
                "-t",
                xfm_invnl,
                "-t",
                "[" + xfm_affine + ",1]",
                "-n",
                "NearestNeighbor",
            ]

            cmd_cont = cltmisc.generate_container_command(
                cmd_bashargs, cont_tech, cont_image
            )  # Generating container command
            subprocess.run(
                cmd_cont, stdout=subprocess.PIPE, universal_newlines=True
            )  # Running container command

            tmp_parc = cltparc.Parcellation(parc_file=out_parc)
            tmp_parc.save_parcellation(
                out_file=out_parc,
                affine=tmp_parc.affine,
                save_lut=False,
                save_tsv=False,
            )

        # Removing the Warped images
        if os.path.isfile(temp_xfm_invnlw):
            os.remove(temp_xfm_invnlw)

        if os.path.isfile(temp_xfm_nlw):
            os.remove(temp_xfm_nlw)

    return out_parc


######################################################################################################
@staticmethod
def tissue_seg_table(tsv_filename):
    """
    Create standard tissue segmentation lookup table.

    Parameters
    ----------
    tsv_filename : str
        Output TSV file path.

    Returns
    -------
    pd.DataFrame
        DataFrame with tissue segmentation information (CSF, GM, WM).

    Examples
    --------
    >>> seg_df = Parcellation.tissue_seg_table('tissues.tsv')
    >>> print(seg_df)
    """

    # Table for tissue segmentation
    # 1. Default values for tissues segmentation table
    seg_rgbcol = np.array([[172, 0, 0], [0, 153, 76], [0, 102, 204]])
    seg_codes = np.array([1, 2, 3])
    seg_names = ["cerebro_spinal_fluid", "gray_matter", "white_matter"]
    seg_acron = ["CSF", "GM", "WM"]

    # 2. Converting colors to hexidecimal string
    seg_hexcol = []
    nrows, ncols = seg_rgbcol.shape
    for i in np.arange(0, nrows):
        seg_hexcol.append(
            cltmisc.rgb2hex(seg_rgbcol[i, 0], seg_rgbcol[i, 1], seg_rgbcol[i, 2])
        )

    seg_df = pd.DataFrame(
        {
            "index": seg_codes,
            "name": seg_names,
            "abbreviation": seg_acron,
            "color": seg_hexcol,
        }
    )
    # Save the tsv table
    with open(tsv_filename, "w+") as tsv_file:
        tsv_file.write(seg_df.to_csv(sep="\t", index=False))

    return seg_df
