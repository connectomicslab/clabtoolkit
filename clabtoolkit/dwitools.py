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


# This function removes the B0s volumes located at the end of the diffusion 4D volume.
def remove_empty_dwi_Volume(dwifile: str):
    """
    Remove the B0s volumes located at the end of the diffusion 4D volume.

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

def concatenate_tractograms(trks: list, concat_trk: str = None, show_progress: bool = False):
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
        raise ValueError("trks must be a list of file paths to the tractograms to concatenate.")
    
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