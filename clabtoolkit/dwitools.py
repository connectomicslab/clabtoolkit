import os
import numpy as np
import warnings

import nibabel as nib
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes
from skimage import measure
from typing import Union, Dict, List

# add progress bar using rich progress bar
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn


# Importing the internal modules
from . import misctools as cltmisc

####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############                 Methods to work with DWI images                            ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################


def delete_volumes(
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

    Notes:
    -----
    IMPORTANT: The function will overwrite the original DWI file if the output file is not specified.
    IMPORTANT: The function will overwrite the original bvec and bval files if the output file is not specified.
    IMPORTANT: The function will remove the last B0s of the DWI image if no volumes are specified.


    How to use:
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

    How to use:
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
############                 Methods to work with streamlines                           ############
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

    How to use:
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

    How to use:
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

# Alternative version with more flexible error handling
def concatenate_tractograms(
    trks: list,
    concat_trk: str = None,
    show_progress: bool = False,
    skip_missing: bool = False,
):
    """
    Concatenate multiple tractograms with flexible error handling.

    Parameters
    ----------
    trks : list of str
        List of file paths to the tractograms to concatenate.
    concat_trk : str, optional
        File path for the output concatenated tractogram.
    show_progress : bool, optional
        Whether to show a progress bar during processing.
    skip_missing : bool, optional
        If True, skip missing files instead of raising an error.

    Returns
    -------
    result : nibabel.streamlines.Tractogram or str
        The concatenated tractogram or output file path.
    """
    # Input validation
    if not isinstance(trks, list):
        raise ValueError("trks must be a list of file paths.")

    if len(trks) < 2:
        raise ValueError("At least two tractograms are required.")

    # Filter existing files
    existing_files = []
    missing_files = []

    for trk in trks:
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
                    print(f"File {trk_file} does not exist. Skipping.")

        else:
            if os.path.exists(trk_file):

            if show_progress:
                progress.update(task, advance=1)

    finally:
        if show_progress:
            progress.stop()

    # Save or return
    if concat_trk is not None:
        output_dir = os.path.dirname(concat_trk)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            warnings.warn(f"Created output directory: {output_dir}")

        nib.streamlines.save(trkall, concat_trk)
        return concat_trk

    return trkall
