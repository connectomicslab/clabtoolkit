import os
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from scipy.interpolate import RegularGridInterpolator

import nibabel as nib
from nibabel.streamlines import Field, ArraySequence
from nibabel.streamlines.trk import TrkFile
from nibabel.orientations import aff2axcodes

from skimage import measure
from typing import Union, Dict, List
from dipy.segment.clustering import QuickBundlesX, QuickBundles
from dipy.tracking.streamline import set_number_of_points
from dipy.io.streamline import save_trk
from dipy.io.stateful_tractogram import StatefulTractogram, Space


# add progress bar using rich progress bar
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn


# Importing the internal modules
from . import misctools as cltmisc


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############                Section 1: Methods to work with DWI images                  ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def delete_dwi_volumes(
    in_image: str,
    bvec_file: str = None,
    bval_file: str = None,
    out_image: str = None,
    bvals_to_delete: Union[int, List[Union[int, tuple, list, str, np.ndarray]]] = None,
    vols_to_delete: Union[int, List[Union[int, tuple, list, str, np.ndarray]]] = None,
) -> str:
    """
    Remove specific volumes from DWI image. If no volumes are specified, the function will remove the last B0s of the DWI image.

    Parameters
    ----------
    in_image : str
        Path to the diffusion weighted image file.

    bvec_file : str, optional
        Path to the bvec file. If None, it will assume the bvec file is in the same directory as the DWI file with the same name but with the .bvec extension.

    bval_file : str, optional
        Path to the bval file. If None, it will assume the bval file is in the same directory as the DWI file with the same name but with the .bval extension.

    out_image : str, optional
        Path to the output file. If None, it will assume the output file is in the same directory as the DWI file with the same name but with the .nii.gz extension.
        The original file will be overwritten if the output file is not specified.

    bvals_to_delete : int, list, optional
        List of bvals to delete. If None, it will assume the bvals to delete are the last B0s of the DWI image.
        Some conditions could be used to delete the volumes.
            For example:
                1. If you want to delete all the volumes with bval = 0, you can use:
                bvals_to_delete = [0]

                2. If you want to delete all the volumes with b-values higher than 1000, you can use:
                bvals_to_delete = [bvals > 1000]  or  bvals_to_delete = [bvals >= 1000] if you want to include the 1000 bvals.

                3. If you want to delete all the volumes with b-values between 1000 and 3000 you can use:
                bvals_to_delete = [1000 < bvals < 3000] or bvals_to_delete = [1000 <= bvals < 3000] if you want to include the 1000 but not the 3000 bvals.

            For more complex conditions, you can see the function get_indices_by_condition. Included in the clabtoolkit.misctools module.

    vols_to_delete : int, list, optional
        Indices of the volumes to delete. If None, it will assume the volumes to delete are the last B0s of the DWI image.
        Some conditions could be used to delete the volumes.
            For example:
                1. If you want to delete the first 3 volumes, you can use:
                    vols_to_delete = [0, 1, 2]

                2. If you want to delete the volumes from 0 to 10, you can use:
                    vols_to_delete = [0:10] or vols_to_delete = [0-10]

                3. If you want to delete the volumes from 0 to 10 and 20 to 30, you can use:
                    vols_to_delete = [0:10, 20:30] or vols_to_delete = [0-10, 20-30]

                4. If you want to delete the volumes from 0 to 10 and the volumes 40 and 60, you can use:
                    vols_to_delete = [0:10, 40, 60] or vols_to_delete = [0-10, 40, 60] or vols_to_delete = ['0-10, 40, 60'], etc

                For more complex conditions, you can see the function build_indices. Included in the clabtoolkit.misctools module.

        If both bvals_to_delete and vols_to_delete are specified, the function will remove the volumes with the bvals specified
        and the volumes specified in the vols_to_delete list.
        The function will unify all the indices in a single list and remove the volumes from the DWI image.

    Returns
    -------
    out_image : str
        Path to the diffusion weighted image file.

    out_bvecs_file : str
        Path to the bvec file. If None, it will assume the bvec file is in the same directory as the DWI file with the same name but with the .bvec extension.

    out_bvals_file : str
        Path to the bval file. If None, it will assume the bval file is in the same directory as the DWI file with the same name but with the .bval extension.

    vols2rem : list
        List of volumes removed.

    Notes
    -----
    IMPORTANT: The function will overwrite the original DWI file if the output file is not specified.
    IMPORTANT: The function will overwrite the original bvec and bval files if the output file is not specified.
    IMPORTANT: The function will remove the last B0s of the DWI image if no volumes are specified.

    Examples
    -----------

    >>> delete_volumes('dwi.nii.gz') # will remove the last B0s. The original file will be overwritten.

    >>> delete_volumes('dwi.nii.gz', out_image='dwi_clean.nii.gz') # will remove the last B0s and save the output in dwi_clean.nii.gz

    >>> delete_volumes('dwi.nii.gz', vols_to_delete=[0, 1, 2]) # will remove the first 3 volumes

    >>> delete_volumes('dwi.nii.gz', bvec_file='dwi.bvec', bval_file='dwi.bval') # will remove the last B0s and it will assume the bvec and bval files are in the same directory as the DWI file.

    >>> delete_volumes('dwi.nii.gz', bvec_file='dwi.bvec', bval_file='dwi.bval', bvals_to_delete= [3000, "bvals >=5000"], out_image='dwi_clean.nii.gz') # will remove the volumes with bvals equal to 3000 and equal or higher than 5000.
        The output will be saved in in dwi_clean.nii.gz
        IMPORTANT: the b-values file dwi.bval should be in the same directory as the DWI file.

    """

    # Creating the name for the json file
    if os.path.isfile(in_image):
        pth = os.path.dirname(in_image)
        fname = os.path.basename(in_image)
    else:
        raise FileNotFoundError(f"File {in_image} not found.")

    if fname.endswith(".nii.gz"):
        flname = fname[0:-7]
    elif fname.endswith(".nii"):
        flname = fname[0:-4]

    # Checking if the file exists. If it is None assume it is in the same directory with the same name as the DWI file but with the .bvec extensions.
    if bvec_file is None:
        bvec_file = os.path.join(pth, flname + ".bvec")

    # Checking if the file exists. If it is None assume it is in the same directory with the same name as the DWI file but with the .bval extensions.
    if bval_file is None:
        bval_file = os.path.join(pth, flname + ".bval")

    # Checking the ouput basename
    if out_image is not None:
        fl_out_name = os.path.basename(out_image)

        if fl_out_name.endswith(".nii.gz"):
            fl_out_name = fl_out_name[0:-7]
        elif fl_out_name.endswith(".nii"):
            fl_out_name = fl_out_name[0:-4]

        fl_out_path = os.path.dirname(out_image)

        if not os.path.isdir(fl_out_path):
            raise FileNotFoundError(f"Output path {fl_out_path} does not exist.")
    else:
        fl_out_name = fname
        fl_out_path = pth

    # Checking the volumes to delete
    if vols_to_delete is not None:
        if not isinstance(vols_to_delete, list):
            vols_to_delete = [vols_to_delete]

        vols_to_delete = cltmisc.build_indices(vols_to_delete, nonzeros=False)

    # Checking the bvals to delete. This variable will overwrite the vols_to_delete variable if it is not None.
    if bvals_to_delete is not None:
        if not isinstance(bvals_to_delete, list):
            bvals_to_delete = [bvals_to_delete]

        # Loading bvalues
        if os.path.exists(bval_file):
            bvals = np.loadtxt(bval_file, dtype=float, max_rows=5).astype(int)

        tmp_bvals = cltmisc.build_values_with_conditions(
            bvals_to_delete, bvals=bvals, nonzeros=False
        )
        tmp_bvals_to_delete = np.where(np.isin(bvals, tmp_bvals))[0]

        if vols_to_delete is not None:
            vols_to_delete += tmp_bvals_to_delete.tolist()

            # Remove duplicates
            vols_to_delete = list(set(vols_to_delete))

        else:
            vols_to_delete = tmp_bvals_to_delete

    if vols_to_delete is not None:
        # check if vols_to_delete is not empty
        if len(vols_to_delete) == 0:
            print(f"No volumes to delete. The volumes to delete are empty.")
            return in_image

    # Loading the DWI image
    mapI = nib.load(in_image)

    # getting the dimensions of the image
    dim = mapI.shape
    # Only remove the volumes is the image is 4D

    if len(dim) == 4:
        # Getting the number of volumes
        nvols = dim[3]

        if vols_to_delete is not None:

            if len(vols_to_delete) == nvols:
                # If the number of volumes to delete is equal to the number of volumes, send a warning and return the original file
                print(
                    f"Number of volumes to delete is equal to the number of volumes. No volumes will be deleted."
                )

                return in_image

            # Check if the volumes to delete are in the range of the number of volumes
            if np.max(vols_to_delete) >= nvols:
                # Detect which values of the list vols_to_delete are out of range

                # Convert the list to a numpy array
                vols_to_delete = np.array(vols_to_delete)

                # Check if the values are out of range
                out_of_range = np.where(vols_to_delete >= nvols)[0]
                # Raise an error with the out of range values
                raise ValueError(
                    f"Volumes out of the range:  {vols_to_delete[out_of_range]} . The values should be between 0 and {nvols-1}."
                )

            # Check if the volumes to delete are in the range of the number of volumes
            if np.min(vols_to_delete) < 0:
                raise ValueError(
                    f"Volumes to delete {vols_to_delete} are out of range. The values shoudl be between 0 and {nvols-1}."
                )

            vols2rem = np.where(np.isin(np.arange(nvols), vols_to_delete))[0]
            vols2keep = np.where(
                np.isin(np.arange(nvols), vols_to_delete, invert=True)
            )[0]
        else:

            # Loading bvalues
            if os.path.exists(bval_file):
                bvals = np.loadtxt(bval_file, dtype=float, max_rows=5).astype(int)

                mask = bvals < 10
                lb_bvals = measure.label(mask, 2)

                if np.max(lb_bvals) > 1 and lb_bvals[-1] != 0:

                    # Removing the last cluster of B0s
                    lab2rem = lb_bvals[-1]
                    vols2rem = np.where(lb_bvals == lab2rem)[0]
                    vols2keep = np.where(lb_bvals != lab2rem)[0]

                else:
                    # Exit the function if there are no B0s to remove at the end of the volume. Leave a message.
                    print("No B0s to remove at the end of the volume.")

                    return in_image
            else:
                raise FileNotFoundError(
                    f"File {bval_file} not found. It is mandatory if the volumes to remove are not specified (vols_to_delete)."
                )

        diffData = mapI.get_fdata()
        affine = mapI.affine

        # Removing the volumes
        array_data = np.delete(diffData, vols2rem, 3)

        # Temporal image and diffusion scheme
        array_img = nib.Nifti1Image(array_data, affine)
        nib.save(array_img, out_image)

        # Saving new bvecs and new bvals
        if os.path.isfile(bvec_file):
            bvecs = np.loadtxt(bvec_file, dtype=float)
            if bvecs.shape[0] == 3:
                select_bvecs = bvecs[:, vols2keep]
            else:
                select_bvecs = bvecs[vols2keep, :]

            select_bvecs.transpose()
            if out_image.endswith("nii.gz"):
                out_bvecs_file = out_image.replace(".nii.gz", ".bvec")
            elif out_image.endswith("nii"):
                out_bvecs_file = out_image.replace(".nii", ".bvec")

            np.savetxt(out_bvecs_file, select_bvecs, fmt="%f")
        else:
            out_bvecs_file = None

        # Saving new bvals
        if os.path.isfile(bval_file):
            bvals = np.loadtxt(bval_file, dtype=float, max_rows=5).astype(int)
            select_bvals = bvals[vols2keep]
            select_bvals.transpose()

            if out_image.endswith("nii.gz"):
                out_bvals_file = out_image.replace(".nii.gz", ".bval")
            elif out_image.endswith("nii"):
                out_bvals_file = out_image.replace(".nii", ".bval")
            np.savetxt(out_bvals_file, select_bvals, newline=" ", fmt="%d")
        else:
            out_bvals_file = None

    else:
        vols2rem = None
        raise Warning(f"Image {in_image} is not a 4D image. No volumes to remove.")

    return out_image, out_bvecs_file, out_bvals_file, vols2rem


####################################################################################################
def get_b0s(
    dwi_img: str, b0s_img: str, bval_file: str = None, bval_thresh: int = 0
) -> str:
    """
    Extract B0 volumes from a DWI image and save them as a separate NIfTI file.

    Parameters
    ----------
    dwi_img : str
        Path to the input DWI image file.

    b0s_img : str
        Path to the output B0 image file.

    bval_file : str, optional
        Path to the bval file. If None, it will assume the bval file is in the same directory as the DWI file with the same name but with the .bval extension.
        The bval file is used to identify the B0 volumes in the DWI image.

    bval_thresh : int, optional
        Threshold for identifying B0 volumes. Default is 0. Volumes with b-values below this threshold will be considered B0 volumes.

    Returns
    -------
    b0s_img : str
        Path to the output B0 image file.

    b0_vols : List[int]
        List of indices of the B0 volumes extracted from the DWI image.

    Raises
    ------
    FileNotFoundError
        If the input DWI image file or the bval file does not exist.

    ValueError
        If the output path for the B0 image file does not exist.

    Examples
    -----------

    >>> dwi_img = 'path/to/dwi_image.nii.gz'
    >>> b0s_img = 'path/to/b0_image.nii.gz'
    >>> bval_file = 'path/to/bvals.bval'
    >>> b0s_img, b0_vols = get_b0s(dwi_img, b0s_img, bval_file)
    >>> print(f"B0 image saved at: {b0s_img}")
    >>> print(f"B0 volumes indices: {b0_vols}")

    >>> b0s_img, b0_vols = get_b0s(dwi_img, b0s_img, bval_file, bval_thresh=10)
    >>> print(f"B0 image saved at: {b0s_img}")
    >>> print(f"B0 volumes indices: {b0_vols}")
    >>> All the volumes with b-values below 10 will be considered B0 volumes.

    >>> b0s_img, b0_vols = get_b0s(dwi_img, b0s_img)
    >>> print(f"B0 image saved at: {b0s_img}")
    >>> print(f"B0 volumes indices: {b0_vols}")
    >>> The bval file will be assumed to be in the same directory as the DWI file with the same name but with the .bval extension.

    """

    # Creating the name for the json file
    if os.path.isfile(dwi_img):
        pth = os.path.dirname(dwi_img)
        fname = os.path.basename(dwi_img)
    else:
        raise FileNotFoundError(f"File {dwi_img} not found.")

    if fname.endswith(".nii.gz"):
        flname = fname[0:-7]
    elif fname.endswith(".nii"):
        flname = fname[0:-4]

    # Checking if the file exists. If it is None assume it is in the same directory with the same name as the DWI file but with the .bval extensions.
    if bval_file is None:
        bval_file = os.path.join(pth, flname + ".bval")

    # Checking the ouput basename
    if b0s_img is not None:
        fl_out_name = os.path.basename(b0s_img)

        if fl_out_name.endswith(".nii.gz"):
            fl_out_name = fl_out_name[0:-7]
        elif fl_out_name.endswith(".nii"):
            fl_out_name = fl_out_name[0:-4]

        fl_out_path = os.path.dirname(b0s_img)

        if not os.path.isdir(fl_out_path):
            raise FileNotFoundError(f"Output path {fl_out_path} does not exist.")
    else:
        fl_out_name = fname
        fl_out_path = pth

    # Loading bvalues
    if os.path.exists(bval_file):
        bvals = np.loadtxt(bval_file, dtype=float, max_rows=5).astype(int)

        # Generate search cad
        cad = ["bvals > " + str(bval_thresh)]

        # Get the indices of the volumes that will be removed
        vols2rem = cltmisc.build_indices_with_conditions(
            cad, bvals=bvals, nonzeros=False
        )

        b0_vols = np.setdiff1d(np.arange(bvals.shape[0]), vols2rem)

        if len(vols2rem) == 0:
            print(f"No B0s to remove. The volumes to delete are empty.")
            return dwi_img
        else:

            mapI = nib.load(dwi_img)
            diffData = mapI.get_fdata()
            affine = mapI.affine

            # Removing the volumes
            array_data = np.delete(diffData, vols2rem, 3)

            # Temporal image and diffusion scheme
            array_img = nib.Nifti1Image(array_data, affine)
            nib.save(array_img, b0s_img)

    return b0s_img, b0_vols


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############                 Section 2: Methods to work with streamlines                           ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def tck2trk(
    in_tract: str, ref_img: str, out_tract: str = None, force: bool = False
) -> str:
    """
    Convert a TCK file to a TRK file using a reference image for the header.

    Parameters
    ----------
    in_tract : str
        Path to the input TCK file.

    ref_img : str
        Path to the reference NIfTI image for creating the TRK header.

    out_tract : str, optional
        Path for the output TRK file. Defaults to replacing the .tck extension with .trk.

    force : bool, optional
        If True, overwrite the output file if it exists. Defaults to False.

    Returns
    -------
    str
        Path to the output TRK file.

    Raises
    ------
    ValueError
        If the input file format is not TCK.

    FileExistsError
        If the output file exists and force is False.

    FileNotFoundError
        If the reference image does not exist.

    Examples
    -----------
    >>> tck2trk('input.tck', 'reference.nii.gz')  # Saves as 'input.trk'
    >>> tck2trk('input.tck', 'reference.nii.gz', 'output.trk')  # Saves as 'output.trk'
    >>> tck2trk('input.tck', 'reference.nii.gz', force=True)  # Overwrites 'input.trk' if it exists
    >>> tck2trk('input.tck', 'reference.nii.gz', out_tract='output.trk', force=True)  # Overwrites 'output.trk' if it exists

    """
    # Validate input file format
    if nib.streamlines.detect_format(in_tract) is not nib.streamlines.TckFile:
        raise ValueError(f"Invalid input file format: {in_tract} is not a TCK file.")

    # Define output filename
    if out_tract is None:
        out_tract = in_tract.replace(".tck", ".trk")

    # Handle overwrite scenario
    if not os.path.exists(out_tract) or force:
        # Load reference image
        ref_nifti = nib.load(ref_img)

        # Construct TRK header
        header = {
            Field.VOXEL_TO_RASMM: ref_nifti.affine.copy(),
            Field.VOXEL_SIZES: ref_nifti.header.get_zooms()[:3],
            Field.DIMENSIONS: ref_nifti.shape[:3],
            Field.VOXEL_ORDER: "".join(aff2axcodes(ref_nifti.affine)),
        }

        # Load and save tractogram
        tck = nib.streamlines.load(in_tract)
        nib.streamlines.save(tck.tractogram, out_tract, header=header)

    return out_tract


####################################################################################################
def trk2tck(in_tract: str, out_tract: str = None, force: bool = False) -> str:
    """
    Convert a TRK file to a TCK file.

    Parameters
    ----------
    in_tract : str
        Input TRK file.

    out_tract : str, optional
        Output TCK file. If None, the output file will have the same name as the input with the extension changed to TCK.

    force : bool, optional
        If True, overwrite the output file if it exists.

    Returns
    -------
    out_tract : str
        Output TCK file.

    Examples
    ---------
    >>> trk2tck('input.trk')  # Saves as 'input.tck'
    >>> trk2tck('input.trk', 'output.tck')  # Saves as 'output.tck'
    >>> trk2tck('input.trk', force=True)  # Overwrites 'input.tck' if it exists
    >>> trk2tck('input.trk', 'output.tck', force=True)  # Overwrites 'output.tck' if it exists

    """

    # Ensure the input is a TRK file
    if nib.streamlines.detect_format(in_tract) is not nib.streamlines.TrkFile:
        raise ValueError(f"Input file '{in_tract}' is not a valid TRK file.")

    # Set output filename
    if out_tract is None:
        out_tract = in_tract.replace(".trk", ".tck")

    # Check if output file exists
    if os.path.isfile(out_tract) and not force:
        raise FileExistsError(
            f"File '{out_tract}' already exists. Use 'force=True' to overwrite."
        )

    # Load the TRK file
    trk = nib.streamlines.load(in_tract)

    # Save as a TCK file
    nib.streamlines.save(trk.tractogram, out_tract)

    return out_tract


####################################################################################################
def concatenate_tractograms(
    in_tracts: list,
    concat_tract: str = None,
    show_progress: bool = False,
    skip_missing: bool = False,
):
    """
    Concatenate multiple tractograms with flexible error handling.

    Parameters
    ----------
    in_tracts : list of str
        List of file paths to the tractograms to concatenate.
    concat_tract : str, optional
        File path for the output concatenated tractogram.
    show_progress : bool, optional
        Whether to show a progress bar during processing.
    skip_missing : bool, optional
        If True, skip missing files instead of raising an error.

    Returns
    -------
    result : nibabel.streamlines.Tractogram or str
        The concatenated tractogram or output file path.
    Raises
    ------
    ValueError
        If the input tractograms are not a list or if less than two tractograms are provided.
    FileNotFoundError
        If any of the specified tractogram files are missing and `skip_missing` is False.
    FileExistsError
        If the output file already exists and `concat_tract` is specified without `skip_missing

    Examples
    --------
    >>> in_tracts = ['tract1.trk', 'tract2.trk', 'tract3.trk']
    >>> concat_tract = 'concatenated.trk'
    >>> result = concatenate_tractograms(in_tracts, concat_tract, show_progress=True, skip_missing=True)
    >>> print(f"Concatenated tractogram saved at: {result}")
    """
    # Input validation
    if not isinstance(in_tracts, list):
        raise ValueError("trks must be a list of file paths.")

    if len(in_tracts) < 2:
        raise ValueError("At least two tractograms are required.")

    if concat_tract is not None:
        output_dir = os.path.dirname(concat_tract)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            warnings.warn(f"Created output directory: {output_dir}")

    # Filter existing files
    existing_files = []
    missing_files = []

    for trk in in_tracts:
        if os.path.exists(trk):
            existing_files.append(trk)
        else:
            missing_files.append(trk)

    # Handle missing files
    if missing_files:
        if skip_missing:
            if len(existing_files) < 2:
                raise ValueError(
                    f"After skipping missing files, less than 2 files remain. Missing: {missing_files}"
                )
            warnings.warn(f"Skipping missing files: {missing_files}")
        else:
            raise FileNotFoundError(f"Missing files: {missing_files}")

    # Process files
    trkall = None
    files_to_process = existing_files
    cont = 0
    if show_progress:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            SpinnerColumn(),
        )
        progress.start()
        task = progress.add_task(
            "Concatenating tractograms...", total=len(files_to_process)
        )

    try:
        for i, trk_file in enumerate(files_to_process):
            if show_progress:
                progress.update(
                    task,
                    description=f"Processing {os.path.basename(trk_file)} ({i+1}/{len(files_to_process)})",
                )

            trk = nib.streamlines.load(trk_file, lazy_load=False)

            if cont == 0:
                trkall = trk
            else:
                trkall.tractogram.streamlines.extend(trk.tractogram.streamlines)
            cont += 1

        else:
            if show_progress:
                progress.update(task, advance=1)

    finally:
        if show_progress:
            progress.stop()

    # Save or return
    if concat_tract is not None:
        nib.streamlines.save(trkall, concat_tract)
        return concat_tract

    return trkall


####################################################################################################
def resample_streamlines(
    in_streamlines: nib.streamlines.array_sequence.ArraySequence, nb_points: int = 51
) -> nib.streamlines.array_sequence.ArraySequence:
    """
    Resample streamlines to a specified number of points.

    Parameters
    ----------
    in_streamlines : nib.streamlines.array_sequence.ArraySequence
        Input streamlines to be resampled.

    nb_points : int, optional
        Number of points to resample each streamline to. Default is 51.

    Returns
    -------
    resampled_streamlines : nib.streamlines.array_sequence.ArraySequence
        Resampled streamlines.

    Raises
    ------
    ValueError
        If the input streamlines are not in the expected format.

    ValueError
        If the input streamlines are empty.

    ValueError
        If nb_points is not a positive integer.

    Examples
    --------
    >>> in_streamlines = nib.streamlines.load('input.trk').streamlines
    >>> nb_points = 100
    >>> resampled_streamlines = resample_streamlines(in_streamlines, nb_points)
    """
    # Check if input is an ArraySequence
    if not isinstance(in_streamlines, nib.streamlines.array_sequence.ArraySequence):
        raise ValueError(
            "Input streamlines must be in the format of nibabel ArraySequence."
        )

    # Check if the input streamlines are empty
    if len(in_streamlines) == 0:
        raise ValueError(
            "Input streamlines are empty. Please provide valid streamlines."
        )

    # Check if nb_points is a positive integer
    if not isinstance(nb_points, int) or nb_points <= 0:
        raise ValueError("Number of points (nb_points) must be a positive integer.")

    # Check if individual streamlines are valid numpy arrays
    for i, streamline in enumerate(in_streamlines):
        if not isinstance(streamline, np.ndarray):
            raise ValueError(f"Streamline {i} is not a valid numpy array.")

        if streamline.ndim != 2 or streamline.shape[1] != 3:
            raise ValueError(
                f"Streamline {i} must be a 2D array with shape (n_points, 3)."
            )

    # Resample each streamline to the specified number of points
    resampled_streamlines = set_number_of_points(in_streamlines, nb_points)

    return resampled_streamlines


####################################################################################################
def resample_tractogram(
    in_tract: str,
    out_tract: str = None,
    nb_points: int = 51,
    force: bool = False,
) -> str:
    """
    Resample a tractogram to a specified number of points.

    Parameters
    ----------
    in_tract : str
        Path to the input tractogram file (TRK or TCK format).
    out_tract : str, optional
        Path for the output resampled tractogram file. If None, it will replace the input file extension with .trk or .tck.
    nb_points : int, optional
        Number of points to resample each streamline to. Default is 51.
    force : bool, optional
        If True, overwrite the output file if it exists. Default is False.

    Returns
    -------
    str
        Path to the output resampled tractogram file.

    Raises
    ------
    ValueError
        If the input file format is not supported.
    FileExistsError
        If the output file exists and force is False.

    Examples
    -----------
    >>> resample_tractogram('input.trk', nb_points=100)  # Saves as 'input_resampled.trk'
    >>> resample_tractogram('input.tck', out_tract='output.tck', nb_points=100)  # Saves as 'output.tck'
    >>> resample_tractogram('input.trk', force=True)  # Overwrites 'input_resampled.trk' if it exists

    """

    # Validate input file format
    if nib.streamlines.detect_format(in_tract) not in [
        nib.streamlines.TrkFile,
        nib.streamlines.TckFile,
    ]:
        raise ValueError(f"Invalid input file format: {in_tract}. Must be TRK or TCK.")

    # Define output filename
    if out_tract is not None and os.path.exists(out_tract) and not force:
        raise FileExistsError(
            f"Output file '{out_tract}' already exists. Use 'force=True' to overwrite."
        )

    # Load the tractogram
    trk = nib.streamlines.load(in_tract)

    # Resample each streamline to the specified number of points
    resampled_streamlines = resample_streamlines(trk.streamlines, nb_points)

    # Create a new tractogram with the resampled streamlines
    resampled_tractogram = nib.streamlines.Tractogram(
        resampled_streamlines, affine_to_rasmm=trk.tractogram.affine_to_rasmm
    )
    # Create a header (copy from original)
    header = trk.header.copy()

    # Update count in header
    header["nb_streamlines"] = len(resampled_streamlines)

    if out_tract is None:
        return resampled_streamlines
    else:
        # Check if the output directory exists, if not raise an error
        out_dir = os.path.dirname(out_tract)
        if out_dir and not os.path.exists(out_dir):
            raise FileNotFoundError(
                f"Output directory '{out_dir}' does not exist. Please create it before saving the resampled tractogram."
            )
        # If the output directory exists, create it if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)

        # Save the resampled tractogram
        nib.streamlines.save(resampled_tractogram, out_tract, header=header)

        return out_tract


####################################################################################################
def compute_tractogram_centroids(
    in_tract: str,
    centroid_tract: str,
    clustered_tract: str = None,
    nb_points=51,
    method="qb",
    thresholds=[10],
    save_scalars_for_trackvis=True,
):
    """
    Extract bundle centroids from tractogram and save them as .trk files.

    Parameters
    ----------
    in_tract : str
        Path to input tractogram file (.trk)

    centroid_tract : str
        Path to output file for centroids (.trk)

    clustered_tract : str
        Path to output file for clustered streamlines (.trk)

    nb_points : int, optional
        Number of points to resample the streamlines to. Default is 51.

    method : str, optional
        Clustering method to use. Can be 'qbx' or 'qb'. Default is 'qb'.

    thresholds : list of int, optional
        List of thresholds to use for clustering (only for qbx). Default is [10].
        If using 'qb', only the first threshold will be used.

    save_scalars_for_trackvis : bool, optional
        If True, saves scalar data in TrackVis-compatible format. Default is True.

    Returns
    -------
    dict
        Dictionary containing clustering information (number of clusters, etc.)

    Raises
    ------
    FileNotFoundError
        If the input tracking file does not exist or output directories do not exist.

    ValueError
        If the input tracking file is not in the expected format or if nb_points is not a positive integer.

    ValueError
        If the clustering method is not recognized or if thresholds are not provided correctly.

    ValueError
        If the input streamlines are empty or not in the expected format.

    Examples
    -----------
    >>> compute_tractogram_centroids('input.trk', 'centroids.trk', 'clustered.trk', nb_points=100, method='qb', thresholds=[10])
    >>> compute_tractogram_centroids('input.tck', 'centroids.trk', 'clustered.trk', nb_points=100, method='qbx', thresholds=[5,
    10, 15], save_scalars_for_trackvis=True)
    >>> compute_tractogram_centroids('input.tck', 'centroids.trk', 'clustered.trk', nb_points=100, method='qb', thresholds=[10], save
    _scalars_for_trackvis=False)
    >>> compute_tractogram_centroids('input.trk', 'centroids.trk', 'clustered.trk', nb_points=100, method='qbx', thresholds=[5, 10, 15], save_scalars_for_trackvis=True)
    >>> compute_tractogram_centroids('input.tck', 'centroids.trk', 'clustered.trk', nb_points=100, method='qb', thresholds=[10], save_scalars_for_trackvis=True)
    >>> compute_tractogram_centroids('input.tck', 'centroids.trk', 'clustered.trk', nb_points=100, method='qbx', thresholds=[5, 10, 15], save_scalars_for_trackvis=False)
    >>> compute_tractogram_centroids('input.trk', 'centroids.trk', 'clustered.trk', nb_points=100, method='qb', thresholds=[10], save_scalars_for_trackvis=False)
    >>> compute_tractogram_centroids('input.tck', 'centroids.trk', 'clustered.trk', nb_points=100, method='qbx', thresholds=[5, 10, 15], save_scalars_for_trackvis=True)
    >>> compute_tractogram_centroids('input.tck', 'centroids.trk', 'clustered.trk', nb_points=100, method='qb', thresholds=[10], save_scalars_for_trackvis=True)

    """

    # Check if the input tracking file exists
    if not os.path.isfile(in_tract):
        raise FileNotFoundError(f"Input tracking file {in_tract} does not exist.")

    # Check if output directories exist
    # If the output directories do not exist, create them
    centroid_dir = os.path.dirname(centroid_tract)
    if not os.path.exists(centroid_dir):
        # Raise an error if the directory does not exist
        raise FileNotFoundError(f"Output directory {centroid_dir} does not exist.")

    if clustered_tract is not None:
        clustered_dir = os.path.dirname(clustered_tract)
        if not os.path.exists(clustered_dir):
            # Raise an error if the directory does not exist
            raise FileNotFoundError(f"Output directory {clustered_dir} does not exist.")

    # Load the tractogram
    tractogram = nib.streamlines.load(in_tract)
    original_streamlines = tractogram.streamlines

    # === PREPROCESS: RESAMPLE ===
    if nb_points is not None:
        # Check if nb_points is a positive integer
        if not isinstance(nb_points, int) or nb_points <= 0:
            raise ValueError("Number of points (nb_points) must be a positive integer.")
        # Resample the streamlines to the specified number of points
        streamlines = resample_streamlines(original_streamlines, nb_points)
    else:
        # If nb_points is None, use the original streamlines
        streamlines = original_streamlines

    # === CLUSTERING ===
    if method == "qbx":
        if not isinstance(thresholds, list) or len(thresholds) == 0:
            raise ValueError("Thresholds must be a non-empty list for QuickBundlesX")
        qbx = QuickBundlesX(thresholds)
        clusters = qbx.cluster(streamlines)

    elif method == "qb":
        # Use the first threshold if provided, otherwise default to 10
        threshold = thresholds[0] if thresholds else 10
        qb = QuickBundles(threshold=threshold)
        clusters = qb.cluster(streamlines)

    else:
        raise ValueError(f"Unknown clustering method: {method}. Use 'qb' or 'qbx'.")

    # Collect centroids and clustered streamlines with metadata
    centroids_list = []
    clustered_streamlines_list = []

    # Metadata for centroids
    centroid_ids = []
    cluster_sizes = []

    # Metadata for clustered streamlines
    cluster_labels = []
    original_indices = []
    distances_to_centroid = []

    # Process each cluster
    for i, cluster in enumerate(clusters):
        # Get the centroid of the cluster
        cluster_centroid = cluster.centroid
        centroids_list.append(cluster_centroid)

        # Metadata for centroids
        centroid_ids.append(i)
        cluster_sizes.append(len(cluster.indices))

        # Get the streamlines for this cluster
        cluster_streamlines = streamlines[cluster.indices]
        clustered_streamlines_list.extend(cluster_streamlines)

        # Metadata for clustered streamlines
        cluster_labels.extend([i] * len(cluster.indices))
        original_indices.extend(cluster.indices)

        # Calculate distances to centroid for each streamline in cluster
        for j, idx in enumerate(cluster.indices):
            # Simple distance metric: you can customize this
            # For now, we'll use the cluster's internal distance if available
            try:
                if hasattr(cluster, "distances") and cluster.distances is not None:
                    distance = (
                        cluster.distances[j] if j < len(cluster.distances) else 0.0
                    )
                else:
                    # Fallback: set distance to 0 (could implement custom distance calculation here)
                    distance = 0.0
            except:
                distance = 0.0
            distances_to_centroid.append(distance)

    # Convert to ArraySequence for nibabel
    centroids_array = ArraySequence(centroids_list)
    clustered_streamlines_array = ArraySequence(clustered_streamlines_list)

    # Create tractograms with metadata for TrackVis compatibility
    # For centroids tractogram
    centroids_tractogram = nib.streamlines.Tractogram(
        streamlines=centroids_array,
        affine_to_rasmm=tractogram.tractogram.affine_to_rasmm,
    )

    # For clustered streamlines tractogram
    clustered_tractogram = nib.streamlines.Tractogram(
        streamlines=clustered_streamlines_array,
        affine_to_rasmm=tractogram.tractogram.affine_to_rasmm,
    )

    if save_scalars_for_trackvis:
        # TrackVis-compatible scalar format
        # Store cluster IDs as scalar data per streamline
        # Each streamline gets its cluster ID as a scalar value

        # For centroids: store centroid ID and cluster size
        centroid_scalar_data = []
        for i in range(len(centroids_list)):
            # Create scalar data: [centroid_id, cluster_size]
            scalars = np.array([[centroid_ids[i], cluster_sizes[i]]], dtype=np.float32)
            centroid_scalar_data.append(scalars)

        # For clustered streamlines: store cluster_id as scalar
        clustered_scalar_data = []
        for i in range(len(clustered_streamlines_list)):
            # Create scalar data: [cluster_id, distance_to_centroid]
            scalars = np.array(
                [[cluster_labels[i], distances_to_centroid[i]]], dtype=np.float32
            )
            clustered_scalar_data.append(scalars)

        # Note: TrackVis expects scalars in a specific format
        # We'll store this in data_per_streamline for nibabel compatibility
        # The scalar data will be written to the TRK file in the proper format

    # Always store metadata in nibabel format for programmatic access
    centroids_tractogram.data_per_streamline = {
        "centroid_id": np.array(centroid_ids, dtype=np.int32),
        "cluster_size": np.array(cluster_sizes, dtype=np.int32),
    }
    clustered_tractogram.data_per_streamline = {
        "cluster_id": np.array(cluster_labels, dtype=np.int32),
        "original_index": np.array(original_indices, dtype=np.int32),
        "distance_to_centroid": np.array(distances_to_centroid, dtype=np.float32),
    }

    # Create TrkFile objects
    centroids_trk = nib.streamlines.TrkFile(
        tractogram=centroids_tractogram, header=tractogram.header.copy()
    )

    clustered_trk = nib.streamlines.TrkFile(
        tractogram=clustered_tractogram, header=tractogram.header.copy()
    )

    # Update headers with new streamline counts
    centroids_trk.header["nb_streamlines"] = len(centroids_list)

    clustered_trk.header["nb_streamlines"] = len(clustered_streamlines_list)

    # Save the files
    nib.streamlines.save(centroids_trk, centroid_tract)

    if clustered_tract is not None:
        nib.streamlines.save(clustered_trk, clustered_tract)


########################################################################################################
def create_trackvis_colored_trk(
    clustered_trk_path: str, output_path: str, color_by="cluster_id"
):
    """
    Create a TrackVis-compatible TRK file with scalar coloring.

    Parameters
    ----------
    clustered_trk_path : str
        Path to the clustered streamlines TRK file

    output_path : str
        Path for the output TRK file optimized for TrackVis coloring

    color_by : str, optional
        Which metadata field to use for coloring. Options: 'cluster_id',
        'distance_to_centroid', 'original_index'. Default is 'cluster_id'.

    Returns
    -------
    str
        Path to the created file

    Raises
    ------
    ValueError
        If the input file does not contain the specified color_by metadata,
        or if the input file does not contain any metadata.
    FileNotFoundError
        If the input clustered_trk_path does not exist.

    Examples
    -----------
    >>> create_trackvis_colored_trk('clustered.trk', 'colored_output.trk', color_by='cluster_id')
    """

    # Check if the input file exists
    if not os.path.isfile(clustered_trk_path):
        raise FileNotFoundError(f"Input file {clustered_trk_path} does not exist.")

    # Check if the output directory exists, if not raise an error
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        raise FileNotFoundError(
            f"Output directory {output_dir} does not exist. Please create it before saving the colored TRK file."
        )

    # Load the tractogram
    tractogram = nib.streamlines.load(clustered_trk_path)

    if not hasattr(tractogram.tractogram, "data_per_streamline"):
        raise ValueError("No metadata found in the input file.")

    metadata = tractogram.tractogram.data_per_streamline

    if color_by not in metadata:
        available_keys = list(metadata.keys())
        raise ValueError(f"'{color_by}' not found. Available keys: {available_keys}")

    # Get the scalar values for coloring
    scalar_values = metadata[color_by]

    # Create a new tractogram with scalar data in TrackVis format
    streamlines_with_scalars = []

    for i, streamline in enumerate(tractogram.streamlines):
        # For TrackVis, we need to create a streamline with scalar data
        # The scalar value is repeated for each point in the streamline
        scalar_value = float(scalar_values[i])

        # Create scalar array for this streamline (one scalar per point)
        n_points = len(streamline)
        scalars = np.full((n_points, 1), scalar_value, dtype=np.float32)

        # Store as tuple (streamline_points, scalars)
        streamlines_with_scalars.append((streamline, scalars))

    # Create header for the new file
    new_header = tractogram.header.copy()
    new_header["n_scalars"] = 1  # One scalar per point
    new_header["scalar_name"] = [color_by.encode("utf-8")]
    new_header["n_properties"] = 0

    # Use DIPY's save_trk function which handles TrackVis format properly
    from dipy.io.streamline import save_trk
    from dipy.io.stateful_tractogram import StatefulTractogram
    from dipy.io.utils import create_tractogram_header

    # Extract just the streamlines and scalars
    streamlines_only = [s[0] for s in streamlines_with_scalars]
    scalars_only = [s[1] for s in streamlines_with_scalars]

    # Create StatefulTractogram with scalars
    sft = StatefulTractogram(
        streamlines_only,
        tractogram.header,  # Use original header for spatial info
        Space.RASMM,
    )

    # Add scalar data
    sft.data_per_point = {color_by: scalars_only}

    # Save with DIPY
    save_trk(sft, output_path)

    print(f"TrackVis-compatible file saved: {output_path}")
    print(f"Colored by: {color_by}")
    print(f"Scalar range: {np.min(scalar_values):.2f} - {np.max(scalar_values):.2f}")
    print("\nTo view in TrackVis:")
    print("1. Open the .trk file in TrackVis")
    print("2. In the Property panel, find 'Color Code'")
    print("3. Change from 'Directional' to 'Scalar'")
    print(f"4. The streamlines will be colored by {color_by}")

    return output_path


#####################################################################################################
def extract_cluster_by_id(
    clustered_trk_path: str, cluster_ids: Union[List[int], int], output_path=None
):
    """
    Extract all streamlines belonging to specific clusters.

    Parameters
    ----------
    clustered_trk_path : str
        Path to the clustered streamlines TRK file

    cluster_ids : int or list of int
        Cluster ID or list of cluster IDs to extract.
        If a single integer is provided, extracts that cluster.
        If a list is provided, extracts all specified clusters.

    output_path : str, optional
        Path to save the extracted cluster. If None, returns the data.

    Returns
    -------
    dict or None
        If output_path is None, returns dictionary with streamlines and metadata
    """

    # Check if the input file exists
    if not os.path.isfile(clustered_trk_path):
        raise FileNotFoundError(f"Input file {clustered_trk_path} does not exist.")

    # Check if output directory exists, if not raise an error
    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            raise FileNotFoundError(
                f"Output directory {output_dir} does not exist. Please create it before saving the extracted cluster."
            )

    # Load the tractogram
    tractogram = nib.streamlines.load(clustered_trk_path)

    if (
        not hasattr(tractogram.tractogram, "data_per_streamline")
        or "cluster_id" not in tractogram.tractogram.data_per_streamline
    ):
        raise ValueError("No cluster_id metadata found in the file.")

    streamline_cluster_ids = tractogram.tractogram.data_per_streamline["cluster_id"]

    # Find indices of streamlines belonging to the specified cluster
    if isinstance(cluster_ids, (int, np.integer)):
        # Handle single integer
        cluster_indices = [
            i for i, cid in enumerate(streamline_cluster_ids) if cid == cluster_ids
        ]
    else:
        # Handle list of integers
        cluster_indices = [
            i for i, cid in enumerate(streamline_cluster_ids) if cid in cluster_ids
        ]

    if len(cluster_indices) == 0:
        raise ValueError(f"No streamlines found for cluster(s): {cluster_ids}")

    # Extract streamlines
    cluster_streamlines = [tractogram.streamlines[i] for i in cluster_indices]

    # Extract corresponding metadata
    cluster_metadata = {}
    for key, values in tractogram.tractogram.data_per_streamline.items():
        cluster_values = [values[i] for i in cluster_indices]
        # Convert to numpy array with appropriate dtype
        if key in ["centroid_id", "cluster_id", "original_index", "cluster_size"]:
            cluster_metadata[key] = np.array(cluster_values, dtype=np.int32)
        else:  # for distance_to_centroid and other float values
            cluster_metadata[key] = np.array(cluster_values, dtype=np.float32)

    print(
        f"Extracted {len(cluster_streamlines)} streamlines from cluster(s): {cluster_ids}"
    )

    if output_path:
        # Create new tractogram
        new_tractogram = nib.streamlines.Tractogram(
            streamlines=ArraySequence(cluster_streamlines),
            affine_to_rasmm=tractogram.tractogram.affine_to_rasmm,
        )
        new_tractogram.data_per_streamline = cluster_metadata

        # Create and save TrkFile
        new_trk = nib.streamlines.TrkFile(
            tractogram=new_tractogram, header=tractogram.header.copy()
        )
        new_trk.header["nb_streamlines"] = len(cluster_streamlines)

        nib.streamlines.save(new_trk, output_path)

        return None
    else:
        return {
            "streamlines": cluster_streamlines,
            "metadata": cluster_metadata,
            "cluster_ids": (
                cluster_ids if isinstance(cluster_ids, list) else [cluster_ids]
            ),
            "n_streamlines": len(cluster_streamlines),
        }


#####################################################################################################
class TRKExplorer:
    """
    A class to explore and summarize TRK (TrackVis) format tractogram files using nibabel.
    """

    def __init__(self, filepath: str):
        """
        Initialize the TRK explorer with a file path.

        Parameters
        ----------
        filepath : str
            Path to the TRK file to explore.

        Raises
        ------
        FileNotFoundError
            If the specified TRK file does not exist.

        Examples
        -----------
        >>> explorer = TRKExplorer('path/to/your/file.trk')
        >>> summary = explorer.explore(max_streamline_samples=10)
        >>> print(summary)
        >>> # To get detailed information about streamlines and properties
        >>> explorer._analyze_streamlines(max_sample=5)
        >>> explorer._analyze_data_properties()
        >>> print(explorer.streamlines_info)
        >>> print(explorer.data_properties)
        """
        self.filepath = Path(filepath)
        self.trk_file = None
        self.header = {}
        self.streamlines_info = {}
        self.data_properties = {}

        if not self.filepath.exists():
            raise FileNotFoundError(f"TRK file not found: {filepath}")

        self._load_trk_file()

    def _load_trk_file(self):
        """Load the TRK file using nibabel."""
        self.trk_file = nib.streamlines.load(str(self.filepath))
        self.header = self.trk_file.header

    def _analyze_streamlines(self, max_sample: int = 5):
        """
        Analyze streamlines using nibabel.

        Parameters
        ----------
        max_sample : int, optional
            Maximum number of streamlines to sample for detailed info.

        Raises
        ------
        ValueError
            If the TRK file does not contain any streamlines or if the number of streamlines
            exceeds the specified maximum sample size.

        Notes
        -----
        This method collects detailed information about the streamlines in the TRK file,
        including their lengths, sizes, and data types. It also computes statistics such as
        the total number of streamlines, minimum and maximum lengths, and average length.
        The results are stored in the `streamlines_info` attribute of the class.

        """
        streamlines = self.trk_file.streamlines
        n_streamlines = len(streamlines)

        streamlines_info = {
            "total_count": n_streamlines,
            "samples": [],
            "statistics": {
                "lengths": [],
                "total_points": 0,
                "min_length": float("inf"),
                "max_length": 0,
                "avg_length": 0,
            },
        }

        if n_streamlines == 0:
            self.streamlines_info = streamlines_info
            return

        lengths = []
        total_points = 0
        sample_count = 0

        # Analyze streamlines
        for i, streamline in enumerate(streamlines):
            n_points = len(streamline)
            lengths.append(n_points)
            total_points += n_points

            # Store sample information
            if sample_count < max_sample:
                streamline_size_kb = (n_points * 3 * 4) / 1024  # xyz coords in float32
                streamlines_info["samples"].append(
                    {
                        "index": i,
                        "n_points": n_points,
                        "size_kb": streamline_size_kb,
                        "data_type": "float32",
                    }
                )
                sample_count += 1

        # Calculate statistics
        streamlines_info["statistics"] = {
            "lengths": lengths,
            "total_points": total_points,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        }

        self.streamlines_info = streamlines_info

    def _analyze_data_properties(self):
        """Analyze data properties using nibabel."""
        properties = {}

        # Get scalar names from header
        scalar_names = []
        if "scalar_name" in self.header:
            scalar_names = [
                name.decode("utf-8") if isinstance(name, bytes) else name
                for name in self.header["scalar_name"]
                if name and name.strip()
            ]

        # Get property names from header
        property_names = []
        if "property_name" in self.header:
            property_names = [
                name.decode("utf-8") if isinstance(name, bytes) else name
                for name in self.header["property_name"]
                if name and name.strip()
            ]

        # Check for per-point data (scalars)
        n_scalars = self.header.get("nb_scalars_per_point", 0)
        if n_scalars > 0:
            for i in range(n_scalars):
                if i < len(scalar_names) and scalar_names[i]:
                    name = scalar_names[i]
                else:
                    name = f"scalar_{i}"

                properties[name] = {
                    "type": "per_point",
                    "data_type": "float32",
                    "index": i,
                }

        # Check for per-streamline data (properties)
        n_properties = self.header.get("nb_properties_per_streamline", 0)
        if n_properties > 0:
            for i in range(n_properties):
                if i < len(property_names) and property_names[i]:
                    name = property_names[i]
                else:
                    name = f"property_{i}"

                properties[name] = {
                    "type": "per_streamline",
                    "data_type": "float32",
                    "index": i,
                }

        self.data_properties = properties

    ####################################################################################################
    def explore(self, max_streamline_samples: int = 5) -> str:
        """
        Generate a comprehensive summary of the TRK file.

        Parameters
        ----------
            max_streamline_samples (int):
                Maximum number of streamlines to sample for detailed info

        Returns
            str: Formatted summary string

        Raises
            FileNotFoundError: If the TRK file does not exist.
            ValueError: If the TRK file is not in the expected format.
        Examples
        ----------
        >>> explorer = TRKExplorer('path/to/your/file.trk')
        >>> summary = explorer.explore(max_streamline_samples=10)
        >>> print(summary)
        >>> # To get detailed information about streamlines and properties
        >>> explorer._analyze_streamlines(max_sample=5)
        >>> explorer._analyze_data_properties()
        >>> print(explorer.streamlines_info)
        """
        self._analyze_streamlines(max_streamline_samples)
        self._analyze_data_properties()

        # Get file size
        file_size_mb = self.filepath.stat().st_size / (1024 * 1024)

        # Build summary
        summary_lines = []

        # File header
        summary_lines.append(
            f"📁 {self.filepath.name} (TrackVis format, {file_size_mb:.1f} MB)"
        )

        # Header section - FIXED: Convert numpy types to Python native types
        summary_lines.append("├── 📋 Header")
        dimensions = [
            int(x) for x in self.header["dimensions"]
        ]  # Convert numpy types to int
        voxel_sizes = [
            round(float(x), 2) for x in self.header["voxel_sizes"]
        ]  # Convert to float first

        summary_lines.append(f"│   ├── 🔢 dimensions {dimensions}")
        summary_lines.append(f"│   ├── 🔢 voxel_sizes {voxel_sizes}")
        summary_lines.append(
            f"│   ├── 📝 n_streamlines = {self.streamlines_info['total_count']:,}"
        )
        summary_lines.append(
            f"│   ├── 📊 n_scalars = {self.header.get('nb_scalars_per_point', 0)}"
        )
        summary_lines.append(
            f"│   ├── 🏷️ n_properties = {self.header.get('nb_properties_per_streamline', 0)}"
        )
        summary_lines.append(
            f"│   └── 📌 version = {self.header.get('version', 'unknown')}"
        )

        # Streamlines section
        if self.streamlines_info["total_count"] > 0:
            stats = self.streamlines_info["statistics"]
            summary_lines.append(
                f"├── 🧵 Streamlines ({self.streamlines_info['total_count']:,} total)"
            )
            summary_lines.append(
                f"│   ├── 📏 length range: {stats['min_length']}-{stats['max_length']} points"
            )
            summary_lines.append(
                f"│   ├── 📊 average length: {stats['avg_length']:.1f} points"
            )
            summary_lines.append(f"│   ├── 🔢 total points: {stats['total_points']:,}")

            # Sample streamlines
            for i, sample in enumerate(self.streamlines_info["samples"]):
                prefix = (
                    "│   ├──"
                    if i < len(self.streamlines_info["samples"]) - 1
                    else "│   └──"
                )
                summary_lines.append(
                    f"{prefix} 📊 streamline_{sample['index']} "
                    f"[{sample['n_points']} × 3] {sample['data_type']} "
                    f"({sample['size_kb']:.1f} KB)"
                )

            if (
                len(self.streamlines_info["samples"])
                < self.streamlines_info["total_count"]
            ):
                remaining = self.streamlines_info["total_count"] - len(
                    self.streamlines_info["samples"]
                )
                summary_lines.append(f"│       📝 ... {remaining:,} more streamlines")
        else:
            summary_lines.append("├── 🧵 Streamlines (0 total)")

        # Data properties section
        if self.data_properties:
            summary_lines.append("└── 🏷️ Data Properties")
            prop_items = list(self.data_properties.items())
            for i, (name, info) in enumerate(prop_items):
                prefix = "    ├──" if i < len(prop_items) - 1 else "    └──"
                summary_lines.append(
                    f"{prefix} 📊 {name} ({info['type']}, {info['data_type']})"
                )
        else:
            summary_lines.append("└── 🏷️ Data Properties (none)")

        return "\n".join(summary_lines)

    ####################################################################################################
    def get_header_info(self) -> Dict[str, Any]:
        """Return header information as a dictionary."""
        return dict(self.header)

    ####################################################################################################
    def get_streamlines_summary(self) -> Dict[str, Any]:
        """Return streamlines summary information."""
        return self.streamlines_info.copy()

    ####################################################################################################
    def get_data_properties(self) -> Dict[str, Any]:
        """Return data properties information."""
        return self.data_properties.copy()

    ####################################################################################################
    def get_streamlines(self):
        """Return the actual streamlines data."""
        return self.trk_file.streamlines


#########################################################################################################
def explore_trk(filepath: str, max_streamline_samples: int = 5) -> str:
    """
    Quick function to explore a TRK file and return a summary.

    Parameters
    ----------
    filepath : str
        Path to the TRK file to explore.

    max_streamline_samples : int, optional
        Maximum number of streamlines to sample for detailed info.
        Default is 5.

    Returns
    -------
    str
        Formatted summary string of the TRK file.

    Raises
    ------
    FileNotFoundError
        If the specified TRK file does not exist.

    ValueError
        If the TRK file is not in the expected format or does not contain streamlines.

    Examples
    --------
    >>> summary = explore_trk('path/to/your/file.trk', max_streamline_samples=10)
    >>> print(summary)
    >>> # To get detailed information about streamlines and properties
    >>> explorer = TRKExplorer('path/to/your/file.trk')
    >>> explorer._analyze_streamlines(max_sample=5)
    >>> explorer._analyze_data_properties()
    >>> print(explorer.streamlines_info)
    >>> print(explorer.data_properties)
    >>> streamlines = explorer.get_streamlines()
    >>> header_info = explorer.get_header_info()
    >>> print(header_info)
    >>> print(streamlines)

    """

    # Check if the file exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"TRK file not found: {filepath}")

    explorer = TRKExplorer(filepath)

    return print(explorer.explore(max_streamline_samples))


##########################################################################################################
def interpolate_on_tractogram(
    in_tract: str,
    scal_map: str,
    out_tract: str = None,
    interp_method: str = "linear",
    storage_mode: str = "data_per_point",
    map_name: str = "fa",
    reduction: str = "mean",
    preserve_both_storage_modes: bool = False,
):
    """
    Interpolate scalar values (e.g., FA) from a NIfTI image onto a tractogram.

    This function loads a tractogram and a scalar map, then interpolates the scalar
    values at each streamline point or aggregates them per streamline. The resulting
    tractogram is saved with the interpolated values attached as metadata.

    Parameters
    ----------
    in_tract : str or Path
        Path to input .trk tractogram file. Must exist and be readable.

    scal_map : str or Path
        Path to scalar image (e.g., FA map in NIfTI format). Must exist and be readable.

    out_tract : str or Path
        Path to save the new tractogram with interpolated values. The parent directory
        must exist and be writable.

    interp_method : {'linear', 'nearest'}, default='linear'
        Interpolation method used for RegularGridInterpolator.
        - 'linear': Trilinear interpolation
        - 'nearest': Nearest neighbor interpolation

    storage_mode : {'data_per_point', 'data_per_streamline'}, default='data_per_point'
        Storage format for the interpolated values:
        - 'data_per_point': Store scalar value for each streamline point
        - 'data_per_streamline': Store aggregated scalar value per streamline

    map_name : str, default='fa'
        Name used for the scalar map in the output tractogram metadata.
        This will be the key in data_per_point or data_per_streamline.

    reduction : {'mean', 'median', 'min', 'max'}, default='mean'
        Aggregation method used when storage_mode='data_per_streamline'.
        Applied to all scalar values along each streamline to produce a single value.

    preserve_both_storage_modes : bool, default=False
        If True, preserve existing data in both data_per_point and data_per_streamline.
        **Warning**: This may cause visualization conflicts in some tools (FSLeyes, etc.)
        that have trouble rendering tractograms with both storage modes present.
        Use only when you specifically need both storage modes for different applications.

    Returns
    -------
    new_tractogram : nibabel.streamlines.Tractogram
        The tractogram object with interpolated scalar values attached.

    scalar_values_per_streamline : list of numpy.ndarray
        List containing the interpolated scalar values for each streamline.
        Each array has shape (n_points,) where n_points is the number of points
        in the corresponding streamline. Contains NaN for points outside the scalar map.

    Raises
    ------
    FileNotFoundError
        If input tractogram file or scalar map file does not exist.

    NotADirectoryError
        If the parent directory of the output path does not exist.

    PermissionError
        If the output directory is not writable.

    ValueError
        If interp_method is not 'linear' or 'nearest', if storage_mode is not
        'data_per_point' or 'data_per_streamline', or if reduction method is
        not one of 'mean', 'median', 'min', 'max'.

    IOError
        If there are issues reading the input files or writing the output file.

    Notes
    -----
    - Points outside the scalar map boundaries will have NaN values
    - Empty streamlines are handled gracefully with empty arrays
    - The function preserves the original tractogram's affine transformation
    - When using 'data_per_streamline' mode, the map name will be suffixed
    with the reduction method (e.g., 'fa_mean')
    - **Important**: By default, the function only populates the requested
    storage_mode and clears the other to prevent visualization conflicts.
    Many tools (FSLeyes, etc.) have trouble rendering tractograms with both
    data_per_point and data_per_streamline present simultaneously.
    - Set preserve_both_storage_modes=True only if you specifically need both
    storage modes for different applications, but be aware of potential
    visualization issues.

    Examples
    --------
    Basic usage with FA map:

    >>> new_tract, values = interpolate_on_tractogram(
    ...     'input.trk', 'fa_map.nii.gz', 'output_with_fa.trk',
    ...     map_name='fractional_anisotropy'
    ... )

    Using median aggregation per streamline:

    >>> new_tract, values = interpolate_on_tractogram(
    ...     'input.trk', 'md_map.nii.gz', 'output_with_md.trk',
    ...     storage_mode='data_per_streamline',
    ...     reduction='median',
    ...     map_name='mean_diffusivity'
    ... )

    Preserving both storage modes (use with caution):

    >>> new_tract, values = interpolate_on_tractogram(
    ...     'input.trk', 'fa_map.nii.gz', 'output_with_fa.trk',
    ...     preserve_both_storage_modes=True
    ... )
    # Warning: May cause visualization issues in FSLeyes and other tools
    """

    # --- Input validation ---
    in_tract = Path(in_tract)
    scal_map = Path(scal_map)

    # Check if input files exist
    if not in_tract.exists():
        raise FileNotFoundError(f"Input tractogram file not found: {in_tract}")

    if not scal_map.exists():
        raise FileNotFoundError(f"Scalar map file not found: {scal_map}")

    # Check if output directory exists
    if out_tract is not None:
        out_tract = Path(out_tract)
        out_dir = out_tract.parent
        if not out_dir.exists():
            raise NotADirectoryError(f"Output directory does not exist: {out_dir}")

        # Check if output directory is writable
        if not os.access(out_dir, os.W_OK):
            raise PermissionError(f"Output directory is not writable: {out_dir}")

    # Validate parameters
    valid_interp_methods = ["linear", "nearest"]
    if interp_method not in valid_interp_methods:
        raise ValueError(
            f"Invalid interpolation method '{interp_method}'. "
            f"Choose from {valid_interp_methods}"
        )

    valid_storage_modes = ["data_per_point", "data_per_streamline"]
    if storage_mode not in valid_storage_modes:
        raise ValueError(
            f"Invalid storage mode '{storage_mode}'. "
            f"Choose from {valid_storage_modes}"
        )

    valid_reductions = ["mean", "median", "min", "max"]
    if reduction not in valid_reductions:
        raise ValueError(
            f"Invalid reduction method '{reduction}'. "
            f"Choose from {valid_reductions}"
        )

    # --- Load tractogram ---
    try:
        trk_file = nib.streamlines.load(str(in_tract))
    except Exception as e:
        raise IOError(f"Failed to load tractogram file '{in_tract}': {e}")

    tractogram = trk_file.tractogram
    streamlines = tractogram.streamlines

    if tractogram.affine_to_rasmm is not None:
        original_affine = tractogram.affine_to_rasmm
    else:
        original_affine = trk_file.affine

    # --- Load scalar image ---
    try:
        scalar_img = nib.load(str(scal_map))
    except Exception as e:
        raise IOError(f"Failed to load scalar map '{scal_map}': {e}")

    scalar_data = scalar_img.get_fdata()
    inv_affine = np.linalg.inv(scalar_img.affine)

    # Creating interpolation function
    x = np.arange(scalar_data.shape[0])
    y = np.arange(scalar_data.shape[1])
    z = np.arange(scalar_data.shape[2])
    my_interpolating_scalmap = RegularGridInterpolator(
        (x, y, z), scalar_data, method=interp_method
    )

    # --- Interpolate scalar values per streamline point ---
    scalar_values_per_streamline = []
    for sl in streamlines:
        if len(sl) == 0:
            scalar_values_per_streamline.append(np.array([]))
            continue

        ones = np.ones((len(sl), 1))
        coords_hom = np.hstack([sl, ones])
        voxel_coords = (inv_affine @ coords_hom.T).T[:, :3].T

        mask = (
            (voxel_coords[0] >= 0)
            & (voxel_coords[0] < scalar_data.shape[0])
            & (voxel_coords[1] >= 0)
            & (voxel_coords[1] < scalar_data.shape[1])
            & (voxel_coords[2] >= 0)
            & (voxel_coords[2] < scalar_data.shape[2])
        )

        values = np.full(voxel_coords.shape[1], np.nan)
        if np.any(mask):
            values[mask] = my_interpolating_scalmap(voxel_coords[:, mask].T)

        scalar_values_per_streamline.append(values)

    # --- Store scalar values ---
    # Handle storage mode conflicts - clear the other mode unless user explicitly wants both
    if preserve_both_storage_modes:
        # Preserve both storage modes (may cause visualization conflicts)
        data_per_point = (
            tractogram.data_per_point.copy() if tractogram.data_per_point else {}
        )
        data_per_streamline = (
            tractogram.data_per_streamline.copy()
            if tractogram.data_per_streamline
            else {}
        )
        print(
            "⚠️  Warning: Preserving both storage modes may cause visualization conflicts in some tools"
        )
    else:
        # Only use the requested storage mode to avoid visualization conflicts
        if storage_mode == "data_per_point":
            data_per_point = (
                tractogram.data_per_point.copy() if tractogram.data_per_point else {}
            )
            data_per_streamline = {}  # Clear to avoid conflicts
        else:  # storage_mode == 'data_per_streamline'
            data_per_point = {}  # Clear to avoid conflicts
            data_per_streamline = (
                tractogram.data_per_streamline.copy()
                if tractogram.data_per_streamline
                else {}
            )

    if storage_mode == "data_per_point":
        formatted = [
            val.reshape(-1, 1) if len(val) > 0 else np.empty((0, 1))
            for val in scalar_values_per_streamline
        ]
        data_per_point[map_name] = formatted

    elif storage_mode == "data_per_streamline":
        reducer = {
            "mean": np.nanmean,
            "median": np.nanmedian,
            "min": np.nanmin,
            "max": np.nanmax,
        }[reduction]

        values = [
            reducer(v) if len(v) > 0 and not np.all(np.isnan(v)) else np.nan
            for v in scalar_values_per_streamline
        ]
        data_per_streamline[f"{map_name}_{reduction}"] = np.array(values).reshape(-1, 1)

    # --- Save new tractogram ---
    new_tractogram = nib.streamlines.Tractogram(
        streamlines=streamlines,
        data_per_point=data_per_point,
        data_per_streamline=data_per_streamline,
        affine_to_rasmm=original_affine,
    )

    header = trk_file.header.copy()
    trk_with_header = TrkFile(new_tractogram, header=header)

    if out_tract is not None:
        try:
            nib.streamlines.save(trk_with_header, str(out_tract))
        except Exception as e:
            raise IOError(f"Failed to save output tractogram '{out_tract}': {e}")

    return new_tractogram, scalar_values_per_streamline
