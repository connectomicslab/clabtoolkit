import os
import numpy as np
import nibabel as nib
from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes
from skimage import measure
from typing import Union, Dict, List

# add progress bar using rich progress bar
from rich.progress import Progress
from rich.progress import BarColumn, TextColumn, SpinnerColumn

import inspect
import sys

import misctools as cltmisc


# This function removes the B0s volumes located at the end of the diffusion 4D volume.
def remove_dwi_volumes(
    dwifile: str,
    bvecfile: str = None,
    bvalfile: str = None,
    jsonfile: str = None,
    out_basename: str = None,
    vols_to_delete: Union[int, List[int]] = None,
) -> str:
    """
    Remove specific volumes from DWI image. If no volumes are specified, the function will remove the last B0s of the DWI image.

    Parameters
    ----------
    dwifile : str
        Path to the diffusion weighted image file.

    Returns
    -------
    dwifile : str
        Path to the diffusion weighted image file.

    bvecfile : str
        Path to the bvec file.

    bvalfile : str
        Path to the bval file.

    jsonfile : str
        Path to the json file.

    Notes:
    -----
    The function assumes that the B0 volumes are the last volumes in the 4D volume.
    The function will create a new bvec and bval file with the same name as the DWI file, but with the .bvec and .bval extensions.

    Usage:
    ------
    >>> remove_dwi_volumes('dwi.nii.gz', 'dwi.bvec', 'dwi.bval', 'dwi.json') # will remove the last B0s
    >>> remove_dwi_volumes('dwi.nii.gz', 'dwi.bvec', 'dwi.bval', 'dwi.json', 0) # will remove the first volume
    >>> remove_dwi_volumes('dwi.nii.gz', 'dwi.bvec', 'dwi.bval', 'dwi.json', [0, 1, 2]) # will remove the first 3 volumes
    >>> remove_dwi_volumes('dwi.nii.gz') # will remove the last B0s and it will assume the bvec and bval files are in the same directory as the DWI file.

    """

    # Creating the name for the json file

    if os.path.isfile(dwifile):
        pth = os.path.dirname(dwifile)
        fname = os.path.basename(dwifile)
    else:
        raise FileNotFoundError(f"File {dwifile} not found.")

    if fname.endswith(".nii.gz"):
        flname = fname[0:-7]
    elif fname.endswith(".nii"):
        flname = fname[0:-4]

    # Checking if the file exists. If it is None assume it is in the same directory with the same name as the DWI file but with the .bvec extensions.
    if bvecfile is None:
        bvecfile = os.path.join(pth, flname + ".bvec")
    else:
        if not os.path.isfile(bvecfile):
            raise FileNotFoundError(f"File {bvecfile} not found.")

    # Checking if the file exists. If it is None assume it is in the same directory with the same name as the DWI file but with the .bval extensions.
    if bvalfile is None:
        bvalfile = os.path.join(pth, flname + ".bval")
    else:
        if not os.path.isfile(bvalfile):
            raise FileNotFoundError(f"File {bvalfile} not found.")

    # Checking the ouput basename
    if out_basename is not None:
        fl_out_name = os.path.basename(out_basename)
        fl_out_path = os.path.dirname(out_basename)

        if not os.path.isdir(fl_out_path):
            raise FileNotFoundError(f"Output path {fl_out_path} does not exist.")
    else:
        fl_out_name = fname
        fl_out_path = pth

    # Loading the DWI image
    mapI = nib.load(dwifile)

    # getting the dimensions of the image
    dim = mapI.shape

    # Only remove the volumes is the image is 4D
    if len(dim) == 4:
        # Getting the number of volumes
        nvols = dim[3]

        if vols_to_delete is not None:
            if isinstance(vols_to_delete, int):
                vols_to_delete = [vols_to_delete]

            vols_to_delete = cltmisc.build_indexes(vols_to_delete)

            vols2rem = np.where(np.isin(np.arange(nvols), vols_to_delete))[0]
            vols2keep = np.where(
                np.isin(np.arange(nvols), vols_to_delete, invert=True)
            )[0]
        else:

            # Loading bvalues
            if os.path.exists(bvalfile):
                bvals = np.loadtxt(bvalfile, dtype=float, max_rows=5).astype(int)

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

                    return dwifile
            else:
                raise FileNotFoundError(
                    f"File {bvalfile} not found. It is mandatory if the volumes to remove are not specified (vols_to_delete)."
                )

        diffData = mapI.get_fdata()
        affine = mapI.affine

        # Removing the volumes
        array_data = np.delete(diffData, vols2rem, 3)

        # Temporal image and diffusion scheme
        out_dwi_file = os.path.join(fl_out_path, fl_out_name + ".nii.gz")
        array_img = nib.Nifti1Image(array_data, affine)
        nib.save(array_img, out_dwi_file)

        # Saving new bvecs and new bvals
        if os.path.isfile(bvecfile):
            bvecs = np.loadtxt(bvecfile, dtype=float)
            if bvecs.shape[0] == 3:
                select_bvecs = bvecs[:, vols2keep]
            else:
                select_bvecs = bvecs[vols2keep, :]

            select_bvecs.transpose()
            out_bvecs_file = os.path.join(fl_out_path, fl_out_name + ".bvec")
            np.savetxt(out_bvecs_file, select_bvecs, fmt="%f")

        # Saving new bvals
        if os.path.isfile(bvalfile):
            select_bvals = bvals[vols2keep]
            select_bvals.transpose()

            tmp_bvals_file = os.path.join(fl_out_path, fl_out_name + ".bval")
            np.savetxt(tmp_bvals_file, select_bvals, newline=" ", fmt="%d")

    else:
        raise Warning(f"Image {dwifile} is not a 4D image. No volumes to remove.")

    return out_dwi_file, bvecfile, tmp_bvals_file


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

    Examples
    --------
    >>> trk2tck('input.trk', 'output.tck')  # Saves as 'output.tck'
    >>> trk2tck('input.trk')  # Saves as 'input.tck'
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


def concatenate_tractograms(
    trks: list, concat_trk: str = None, show_progress: bool = False
):
    """
    Concatenate multiple tractograms into a single tractogram.

    Parameters
    ----------
    trks : list of str
        List of file paths to the tractograms to concatenate. It can be trk files or tck files.
    concat_trk : str
        File path for the output concatenated tractogram.

    Returns
    -------
    trkall : nibabel.streamlines.Tractogram or str
        The concatenated tractogram will be returned as a nibabel streamlines object if this variable is None.
        If concat_trk is provided, the concatenated tractogram will be saved to this file path.
        If the output directory does not exist, the concatenated tractogram will be returned as a nibabel streamlines object.

    """

    if not isinstance(trks, list):
        raise ValueError(
            "trks must be a list of file paths to the tractograms to concatenate."
        )

    if len(trks) < 2:
        raise ValueError("At least two tractograms are required to concatenate.")

    save_bool = False
    cont = 0
    for trk_file in trks:
        if show_progress:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                SpinnerColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Processing...", total=len(trks))
                progress.update(task, description=f"Processing {trk_file}")
                progress.update(task, advance=1)

                if os.path.exists(trk_file):
                    save_bool = True
                    trk = nib.streamlines.load(trk_file, False)

                    if cont == 0:
                        trkall = trk
                    else:
                        trkall.tractogram.streamlines.extend(trk.tractogram.streamlines)
                    cont += 1
                else:
                    print(f"File {trk_file} does not exist. Skipping.")

        else:
            if os.path.exists(trk_file):

                save_bool = True
                trk = nib.streamlines.load(trk_file, False)

                if cont == 0:
                    trkall = trk
                else:
                    trkall.tractogram.streamlines.extend(trk.tractogram.streamlines)
                cont += 1
            else:
                print(f"File {trk_file} does not exist. Skipping.")

    # # Save the final trk file
    if save_bool:
        if concat_trk is not None:
            output_track = concat_trk

            # Verify the output directory exists
            output_dir = os.path.dirname(concat_trk)
            if not os.path.exists(output_dir):
                # Print an error message
                error_message = f"Output directory does not exist: {output_dir}"
                print(error_message)
                output_track = trkall

            nib.streamlines.save(trkall, concat_trk)
        else:
            output_track = trkall

    return output_track
