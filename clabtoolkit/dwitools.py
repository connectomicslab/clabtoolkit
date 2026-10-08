import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from skimage import measure

from . import colorstools as cltcol

# Importing the internal modules
from . import misctools as cltmisc
from . import plottools as cltplot

####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############                Section 1: Methods to work with DWI images                  ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################


def _load_bvals(bval_file: str) -> np.ndarray:
    """Load a bval file stored either as a single row or as a single column."""
    return np.round(np.loadtxt(bval_file, dtype=float, ndmin=1).ravel()).astype(int)


####################################################################################################
def delete_dwi_volumes(
    in_image: str | Path,
    bvec_file: str | Path = None,
    bval_file: str | Path = None,
    out_image: str | Path = None,
    bvals_to_delete: int | list[int | tuple | list | str | np.ndarray] = None,
    vols_to_delete: int | list[int | tuple | list | str | np.ndarray] = None,
) -> tuple:
    """
    Remove specific volumes from DWI image. If no volumes are specified, the function will remove the last B0s of the DWI image.

    Parameters
    ----------
    in_image : str or Path
        Path to the diffusion weighted image file.

    bvec_file : str or Path, optional
        Path to the bvec file. If None, it will assume the bvec file is in the same directory as the DWI file with the same name but with the .bvec extension.

    bval_file : str or Path, optional
        Path to the bval file. If None, it will assume the bval file is in the same directory as the DWI file with the same name but with the .bval extension.
        The b-values can be stored as a single row or as a single column.

    out_image : str or Path, optional
        Path to the output file. If None, it will assume the output file is in the same directory as the DWI file with the same name but with the .nii.gz extension.
        The original file will be overwritten if the output file is not specified.

    bvals_to_delete : int, list, optional
        List of bvals to delete. If None, it will assume the bvals to delete are the last B0s of the DWI image.
        Some conditions could be used to delete the volumes.
            For example:
                1. If you want to delete all the volumes with bval = 0, you can use:
                bvals_to_delete = [0]

                2. If you want to delete all the volumes with b-values higher than 1000, you can use:
                bvals_to_delete = ["bvals > 1000"]  or  bvals_to_delete = ["bvals >= 1000"] if you want to include the 1000 bvals.

                3. If you want to delete all the volumes with b-values between 1000 and 3000 you can use:
                bvals_to_delete = ["1000 < bvals < 3000"] or bvals_to_delete = ["1000 <= bvals < 3000"] if you want to include the 1000 but not the 3000 bvals.

            For more complex conditions, you can see the function get_indices_by_condition. Included in the clabtoolkit.misctools module.

    vols_to_delete : int, list, optional
        Indices of the volumes to delete. If None, it will assume the volumes to delete are the last B0s of the DWI image.
        Some conditions could be used to delete the volumes.
            For example:
                1. If you want to delete the first 3 volumes, you can use:
                    vols_to_delete = [0, 1, 2]

                2. If you want to delete the volumes from 0 to 10, you can use:
                    vols_to_delete = ["0:10"] or vols_to_delete = ["0-10"]

                3. If you want to delete the volumes from 0 to 10 and 20 to 30, you can use:
                    vols_to_delete = ["0:10", "20:30"] or vols_to_delete = ["0-10", "20-30"]

                4. If you want to delete the volumes from 0 to 10 and the volumes 40 and 60, you can use:
                    vols_to_delete = ["0:10", 40, 60] or vols_to_delete = ["0-10, 40, 60"], etc

                For more complex conditions, you can see the function build_indices. Included in the clabtoolkit.misctools module.

        If both bvals_to_delete and vols_to_delete are specified, the function will remove the volumes with the bvals specified
        and the volumes specified in the vols_to_delete list.
        The function will unify all the indices in a single list and remove the volumes from the DWI image.

    Returns
    -------
    out_image : str
        Path to the diffusion weighted image file.

    out_bvecs_file : str or None
        Path to the bvec file, or None if no bvec file was found.

    out_bvals_file : str or None
        Path to the bval file, or None if no bval file was found.

    vols2rem : np.ndarray
        Indices of the volumes removed. When nothing is removed, the input paths are
        returned unchanged together with an empty array.

    Raises
    ------
    FileNotFoundError
        If the DWI image, the output directory, or a bval file required to select
        the volumes does not exist.

    ValueError
        If the image is not 4D, the number of b-values does not match the number of
        volumes, or the volumes to delete are out of range.

    Notes
    -----
    IMPORTANT: The function will overwrite the original DWI file if the output file is not specified.
    IMPORTANT: The function will overwrite the original bvec and bval files if the output file is not specified.
    IMPORTANT: The function will remove the last B0s of the DWI image if no volumes are specified.

    Examples
    -----------

    >>> delete_dwi_volumes('dwi.nii.gz') # will remove the last B0s. The original file will be overwritten.

    >>> delete_dwi_volumes('dwi.nii.gz', out_image='dwi_clean.nii.gz') # will remove the last B0s and save the output in dwi_clean.nii.gz

    >>> delete_dwi_volumes('dwi.nii.gz', vols_to_delete=[0, 1, 2]) # will remove the first 3 volumes

    >>> delete_dwi_volumes('dwi.nii.gz', bvec_file='dwi.bvec', bval_file='dwi.bval') # will remove the last B0s

    >>> delete_dwi_volumes('dwi.nii.gz', bvec_file='dwi.bvec', bval_file='dwi.bval', bvals_to_delete= [3000, "bvals >=5000"], out_image='dwi_clean.nii.gz') # will remove the volumes with bvals equal to 3000 and equal or higher than 5000.
        The output will be saved in in dwi_clean.nii.gz

    """

    # Normalize all path-like arguments to plain strings up front
    in_image = str(in_image)
    if bvec_file is not None:
        bvec_file = str(bvec_file)
    if bval_file is not None:
        bval_file = str(bval_file)
    if out_image is not None:
        out_image = str(out_image)

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
    else:
        raise ValueError(
            f"File {in_image} does not have a recognized NIfTI extension (.nii or .nii.gz)."
        )

    # Checking if the file exists. If it is None assume it is in the same directory with the same name as the DWI file but with the .bvec extensions.
    if bvec_file is None:
        bvec_file = os.path.join(pth, flname + ".bvec")

    # Checking if the file exists. If it is None assume it is in the same directory with the same name as the DWI file but with the .bval extensions.
    if bval_file is None:
        bval_file = os.path.join(pth, flname + ".bval")

    # Outputs returned when no volume is removed
    no_change = (
        in_image,
        bvec_file if os.path.isfile(bvec_file) else None,
        bval_file if os.path.isfile(bval_file) else None,
        np.array([], dtype=int),
    )

    # Checking the output basename
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
        fl_out_name = flname
        fl_out_path = pth

    # Rebuild out_image so it always has the .nii.gz extension
    out_image = os.path.join(fl_out_path, fl_out_name + ".nii.gz")

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
            bvals = _load_bvals(bval_file)
        else:
            raise FileNotFoundError(
                f"File {bval_file} not found. It is mandatory if bvals_to_delete is specified."
            )

        tmp_bvals = cltmisc.build_values_with_conditions(
            bvals_to_delete, bvals=bvals, nonzeros=False
        )
        tmp_bvals_to_delete = np.where(np.isin(bvals, tmp_bvals))[0]

        if vols_to_delete is not None:
            vols_to_delete += tmp_bvals_to_delete.tolist()
            vols_to_delete = list(set(vols_to_delete))
        else:
            vols_to_delete = tmp_bvals_to_delete.tolist()

    if vols_to_delete is not None:
        if len(vols_to_delete) == 0:
            print("No volumes to delete. The volumes to delete are empty.")
            return no_change

    # Loading the DWI image
    mapI = nib.load(in_image)
    dim = mapI.shape

    if len(dim) != 4:
        raise ValueError(f"Image {in_image} is not a 4D image. No volumes to remove.")

    nvols = dim[3]

    # The b-values must describe every volume of the image
    if os.path.isfile(bval_file):
        bvals = _load_bvals(bval_file)
        if len(bvals) != nvols:
            raise ValueError(
                f"The bval file {bval_file} has {len(bvals)} values but the image has {nvols} volumes."
            )

    if vols_to_delete is not None:
        if len(vols_to_delete) == nvols:
            print(
                "Number of volumes to delete is equal to the number of volumes. No volumes will be deleted."
            )
            return no_change

        if np.max(vols_to_delete) >= nvols:
            vols_to_delete = np.array(vols_to_delete)
            out_of_range = np.where(vols_to_delete >= nvols)[0]
            raise ValueError(
                f"Volumes out of the range:  {vols_to_delete[out_of_range]} . The values should be between 0 and {nvols-1}."
            )

        if np.min(vols_to_delete) < 0:
            raise ValueError(
                f"Volumes to delete {vols_to_delete} are out of range. The values should be between 0 and {nvols-1}."
            )

        vols2rem = np.where(np.isin(np.arange(nvols), vols_to_delete))[0]
        vols2keep = np.where(np.isin(np.arange(nvols), vols_to_delete, invert=True))[0]
    else:
        if os.path.exists(bval_file):
            mask = bvals < 10
            lb_bvals = measure.label(mask, 2)

            if np.max(lb_bvals) > 1 and lb_bvals[-1] != 0:
                lab2rem = lb_bvals[-1]
                vols2rem = np.where(lb_bvals == lab2rem)[0]
                vols2keep = np.where(lb_bvals != lab2rem)[0]
            else:
                print("No B0s to remove at the end of the volume.")
                return no_change
        else:
            raise FileNotFoundError(
                f"File {bval_file} not found. It is mandatory if the volumes to remove are not specified (vols_to_delete)."
            )

    diffData = mapI.get_fdata()
    affine = mapI.affine

    array_data = np.delete(diffData, vols2rem, 3)
    array_img = nib.Nifti1Image(array_data, affine)
    nib.save(array_img, out_image)

    if os.path.isfile(bvec_file):
        bvecs = np.loadtxt(bvec_file, dtype=float)
        if bvecs.shape[0] == 3:
            select_bvecs = bvecs[:, vols2keep]
        else:
            select_bvecs = bvecs[vols2keep, :]

        out_bvecs_file = out_image.replace(".nii.gz", ".bvec")
        np.savetxt(out_bvecs_file, select_bvecs, fmt="%f")
    else:
        out_bvecs_file = None

    if os.path.isfile(bval_file):
        select_bvals = bvals[vols2keep]

        out_bvals_file = out_image.replace(".nii.gz", ".bval")
        np.savetxt(out_bvals_file, select_bvals, newline=" ", fmt="%d")
    else:
        out_bvals_file = None

    return out_image, out_bvecs_file, out_bvals_file, vols2rem


####################################################################################################
def get_b0s(
    dwi_img: str | Path,
    b0s_img: str | Path = None,
    bval_file: str | Path = None,
    bval_thresh: int = 0,
) -> tuple:
    """
    Extract B0 volumes from a DWI image and save them as a separate NIfTI file.

    Parameters
    ----------
    dwi_img : str or Path
        Path to the input DWI image file.

    b0s_img : str or Path, optional
        Path to the output B0 image file. If None, the B0s are saved next to the DWI
        image with the suffix ``_b0s`` (e.g. ``dwi.nii.gz`` -> ``dwi_b0s.nii.gz``).

    bval_file : str or Path, optional
        Path to the bval file. If None, it will assume the bval file is in the same directory as the DWI file with the same name but with the .bval extension.
        The bval file is used to identify the B0 volumes in the DWI image. The b-values can be stored as a single row or as a single column.

    bval_thresh : int, optional
        Threshold for identifying B0 volumes. Default is 0. Volumes with b-values lower than or equal to this threshold will be considered B0 volumes.

    Returns
    -------
    b0s_img : str
        Path to the output B0 image file.

    b0_vols : np.ndarray
        Indices of the B0 volumes extracted from the DWI image.

    Raises
    ------
    FileNotFoundError
        If the input DWI image file, the bval file or the output directory does not exist.

    ValueError
        If the number of b-values does not match the number of volumes, or no volume
        has a b-value lower than or equal to bval_thresh.

    Examples
    -----------

    >>> dwi_img = 'path/to/dwi_image.nii.gz'
    >>> b0s_img = 'path/to/b0_image.nii.gz'
    >>> bval_file = 'path/to/bvals.bval'
    >>> b0s_img, b0_vols = get_b0s(dwi_img, b0s_img, bval_file)
    >>> print(f"B0 image saved at: {b0s_img}")
    >>> print(f"B0 volumes indices: {b0_vols}")

    >>> b0s_img, b0_vols = get_b0s(dwi_img, b0s_img, bval_file, bval_thresh=10)
    >>> # All the volumes with b-values lower than or equal to 10 will be considered B0 volumes.

    >>> b0s_img, b0_vols = get_b0s(dwi_img)
    >>> # The bval file is assumed to be next to the DWI file, and the B0s are saved as dwi_image_b0s.nii.gz

    """

    dwi_img = str(dwi_img)
    if b0s_img is not None:
        b0s_img = str(b0s_img)
    if bval_file is not None:
        bval_file = str(bval_file)

    if os.path.isfile(dwi_img):
        pth = os.path.dirname(dwi_img)
        fname = os.path.basename(dwi_img)
    else:
        raise FileNotFoundError(f"File {dwi_img} not found.")

    if fname.endswith(".nii.gz"):
        flname = fname[0:-7]
    elif fname.endswith(".nii"):
        flname = fname[0:-4]
    else:
        raise ValueError(
            f"File {dwi_img} does not have a recognized NIfTI extension (.nii or .nii.gz)."
        )

    # Checking if the file exists. If it is None assume it is in the same directory with the same name as the DWI file but with the .bval extensions.
    if bval_file is None:
        bval_file = os.path.join(pth, flname + ".bval")

    if not os.path.isfile(bval_file):
        raise FileNotFoundError(f"File {bval_file} not found.")

    # Checking the output name
    if b0s_img is None:
        b0s_img = os.path.join(pth, flname + "_b0s.nii.gz")
    else:
        fl_out_path = os.path.dirname(b0s_img)
        if fl_out_path and not os.path.isdir(fl_out_path):
            raise FileNotFoundError(f"Output path {fl_out_path} does not exist.")

    # Loading bvalues
    bvals = _load_bvals(bval_file)
    b0_vols = np.where(bvals <= bval_thresh)[0]

    if len(b0_vols) == 0:
        raise ValueError(
            f"No B0 volumes found: no b-value in {bval_file} is lower than or equal to {bval_thresh}. "
            "Increase bval_thresh if the B0s were acquired with a small non-zero b-value."
        )

    mapI = nib.load(dwi_img)
    if mapI.ndim != 4 or mapI.shape[3] != len(bvals):
        raise ValueError(
            f"The bval file {bval_file} has {len(bvals)} values but the image has shape {mapI.shape}."
        )

    array_data = mapI.get_fdata()[..., b0_vols]
    nib.save(nib.Nifti1Image(array_data, mapI.affine), b0s_img)

    return b0s_img, b0_vols


############################################################################################################
def maps_from_tensor_eigenvalues(
    eigvals: str | Path | list | tuple,
    out_basename: str | Path,
    dtmaps: list = None,
    overwrite: bool = False,
) -> dict:
    """
    Compute scalar maps derived from diffusion tensor eigenvalues.

    Eigenvalues can be supplied either as a single 4D NIfTI image (volumes
    ordered as λ1, λ2, λ3 along the 4th axis) or as a list/tuple of three
    separate 3D NIfTI files ``[l1_path, l2_path, l3_path]``.

    Division-by-zero voxels are handled safely: whenever the denominator is
    zero the result at that voxel is set to 0.

    Parameters
    ----------
    eigvals : str or list/tuple of str
        Path to a 4D eigenvalue NIfTI image **or** a list/tuple of three paths
        to the individual eigenvalue volumes ``[λ1, λ2, λ3]``.

    out_basename : str
        Full path prefix for the output files. The map tag and ``.nii.gz``
        extension are appended automatically (e.g. ``/path/sub-01_desc-DTI``
        → ``/path/sub-01_desc-DTI_FA.nii.gz``).

    dtmaps : list of str, optional
        Scalar maps to compute. Use ``['all']`` (default) to compute every
        supported map. Supported tags (case-insensitive):

        ========  ==============================================
        Tag       Description
        ========  ==============================================
        ``AD``    Axial Diffusivity (λ1)
        ``RD``    Radial Diffusivity ((λ2 + λ3) / 2)
        ``MD``    Mean Diffusivity ((λ1 + λ2 + λ3) / 3)
        ``FA``    Fractional Anisotropy
                  sqrt(3/2) * sqrt(Σ(λi - MD)²) / sqrt(Σλi²)
        ``CL``    Linear Anisotropy Coefficient ((λ1 - λ2) / Σλi)
        ``CP``    Planar Anisotropy Coefficient (2(λ2 - λ3) / Σλi)
        ``CS``    Spherical Anisotropy Coefficient (3λ3 / Σλi)
        ``VF``    Volume Fraction (1 - λ1λ2λ3 / MD³)
        ``GA``    Geodesic Anisotropy
                  sqrt(Σ(log λi - mean(log λ))²), 0 if any λi <= 0
        ``RA``    Relative Anisotropy
                  sqrt(Σ(λi - MD)²) / (sqrt(3) * MD)
        ========  ==============================================

    overwrite : bool, optional
        If ``True``, recompute and overwrite existing output files.
        Default is ``False``.

    Returns
    -------
    dict
        Dictionary mapping each requested tag to the path of the saved
        NIfTI file, or to an empty string if the file could not be created.

    Raises
    ------
    ValueError
        If ``eigvals`` is not a str, list, or tuple; or if a list/tuple does
        not contain exactly three elements.
    FileNotFoundError
        If any of the supplied eigenvalue paths do not exist.

    Examples
    --------
    >>> # 4D eigenvalue image
    >>> maps = maps_from_tensor_eigenvalues(
    ...     "sub-01_eigvals.nii.gz",
    ...     "out/sub-01",
    ...     dtmaps=["FA", "MD"],
    ... )

    >>> # Three separate eigenvalue files
    >>> maps = maps_from_tensor_eigenvalues(
    ...     ["sub-01_l1.nii.gz", "sub-01_l2.nii.gz", "sub-01_l3.nii.gz"],
    ...     "out/sub-01",
    ...     dtmaps=["all"],
    ... )
    """

    # ------------------------------------------------------------------ #
    # Input validation and eigenvalue loading
    # ------------------------------------------------------------------ #
    # Normalize Path objects to strings
    if dtmaps is None:
        dtmaps = ["all"]
    if isinstance(out_basename, Path):
        out_basename = str(out_basename)

    if isinstance(eigvals, Path):
        eigvals = str(eigvals)
    elif isinstance(eigvals, (list, tuple)):
        eigvals = [str(p) if isinstance(p, Path) else p for p in eigvals]

    if isinstance(eigvals, str):
        if not os.path.isfile(eigvals):
            raise FileNotFoundError(f"Eigenvalue file not found: {eigvals}")
        ref_img = nib.load(eigvals)
        data4d = ref_img.get_fdata()
        if data4d.ndim != 4 or data4d.shape[3] < 3:
            raise ValueError(
                "4D eigenvalue image must have at least 3 volumes along the 4th axis."
            )
        affine = ref_img.affine
        l1_data = data4d[..., 0]
        l2_data = data4d[..., 1]
        l3_data = data4d[..., 2]

    elif isinstance(eigvals, (list, tuple)):
        if len(eigvals) != 3:
            raise ValueError(
                "When supplying separate eigenvalue files, exactly 3 paths are required "
                f"(got {len(eigvals)})."
            )
        missing = [p for p in eigvals if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"Eigenvalue file(s) not found: {missing}")

        ref_img = nib.load(eigvals[0])
        affine = ref_img.affine
        l1_data = ref_img.get_fdata()
        l2_data = nib.load(eigvals[1]).get_fdata()
        l3_data = nib.load(eigvals[2]).get_fdata()

    else:
        raise ValueError(
            "'eigvals' must be a path string or a list/tuple of three path strings."
        )

    # ------------------------------------------------------------------ #
    # Output directory
    # ------------------------------------------------------------------ #
    out_path = os.path.dirname(out_basename)
    if out_path and not os.path.isdir(out_path):
        # If the output directory does not exist, raise an error
        raise FileNotFoundError(f"Output directory does not exist: {out_path}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    dtmaps = [x.lower() for x in dtmaps]
    compute_all = dtmaps[0] == "all"

    def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        """Element-wise division; returns 0 where denominator is zero."""
        return np.divide(
            num, den, out=np.zeros_like(num, dtype=np.float64), where=den != 0
        )

    def _safe_log(arr: np.ndarray) -> np.ndarray:
        """Element-wise natural log; returns 0 where arr <= 0."""
        return np.where(arr > 0, np.log(np.maximum(arr, np.finfo(float).tiny)), 0.0)

    def _save_map(data: np.ndarray, tag: str) -> str:
        """NaN-fill → save → return path (or '' on failure)."""
        fpath = f"{out_basename}_{tag}.nii.gz"
        data = np.where(np.isnan(data), 0.0, data)
        nib.save(nib.Nifti1Image(data, affine), fpath)
        return fpath if os.path.isfile(fpath) else ""

    # Pre-compute quantities shared across multiple maps
    suma = l1_data + l2_data + l3_data  # used by CL, CP, CS
    RD_data = (l2_data + l3_data) / 2  # used by RD
    MD_data = suma / 3  # used by MD, FA, VF, RA

    scalar_maps: dict = {}

    # ------------------------------------------------------------------ #
    # AD — Axial Diffusivity
    # ------------------------------------------------------------------ #
    if "ad" in dtmaps or compute_all:
        fpath = f"{out_basename}_AD.nii.gz"
        if not os.path.isfile(fpath) or overwrite:
            scalar_maps["AD"] = _save_map(l1_data.copy(), "AD")
        else:
            scalar_maps["AD"] = fpath

    # ------------------------------------------------------------------ #
    # RD — Radial Diffusivity
    # ------------------------------------------------------------------ #
    if "rd" in dtmaps or compute_all:
        fpath = f"{out_basename}_RD.nii.gz"
        if not os.path.isfile(fpath) or overwrite:
            scalar_maps["RD"] = _save_map(RD_data, "RD")
        else:
            scalar_maps["RD"] = fpath

    # ------------------------------------------------------------------ #
    # MD — Mean Diffusivity
    # ------------------------------------------------------------------ #
    if "md" in dtmaps or compute_all:
        fpath = f"{out_basename}_MD.nii.gz"
        if not os.path.isfile(fpath) or overwrite:
            scalar_maps["MD"] = _save_map(MD_data, "MD")
        else:
            scalar_maps["MD"] = fpath

    # ------------------------------------------------------------------ #
    # FA — Fractional Anisotropy
    # ------------------------------------------------------------------ #
    if "fa" in dtmaps or compute_all:
        fpath = f"{out_basename}_FA.nii.gz"
        if not os.path.isfile(fpath) or overwrite:
            num = (
                (l1_data - MD_data) ** 2
                + (l2_data - MD_data) ** 2
                + (l3_data - MD_data) ** 2
            )
            den = l1_data**2 + l2_data**2 + l3_data**2
            FA = np.sqrt(1.5 * _safe_div(num, den))
            scalar_maps["FA"] = _save_map(FA, "FA")
        else:
            scalar_maps["FA"] = fpath

    # ------------------------------------------------------------------ #
    # CL — Linear Anisotropy Coefficient
    # ------------------------------------------------------------------ #
    if "cl" in dtmaps or compute_all:
        fpath = f"{out_basename}_CL.nii.gz"
        if not os.path.isfile(fpath) or overwrite:
            CL = _safe_div(l1_data - l2_data, suma)
            scalar_maps["CL"] = _save_map(CL, "CL")
        else:
            scalar_maps["CL"] = fpath

    # ------------------------------------------------------------------ #
    # CP — Planar Anisotropy Coefficient
    # ------------------------------------------------------------------ #
    if "cp" in dtmaps or compute_all:
        fpath = f"{out_basename}_CP.nii.gz"
        if not os.path.isfile(fpath) or overwrite:
            CP = _safe_div(2 * (l2_data - l3_data), suma)
            scalar_maps["CP"] = _save_map(CP, "CP")
        else:
            scalar_maps["CP"] = fpath

    # ------------------------------------------------------------------ #
    # CS — Spherical Anisotropy Coefficient
    # ------------------------------------------------------------------ #
    if "cs" in dtmaps or compute_all:
        fpath = f"{out_basename}_CS.nii.gz"
        if not os.path.isfile(fpath) or overwrite:
            CS = _safe_div(3 * l3_data, suma)
            scalar_maps["CS"] = _save_map(CS, "CS")
        else:
            scalar_maps["CS"] = fpath

    # ------------------------------------------------------------------ #
    # VF — Volume Fraction
    # ------------------------------------------------------------------ #
    if "vf" in dtmaps or compute_all:
        fpath = f"{out_basename}_VF.nii.gz"
        if not os.path.isfile(fpath) or overwrite:
            product = l1_data * l2_data * l3_data
            den = MD_data**3
            # Background voxels (MD = 0) are isotropic by convention: VF = 0
            VF = np.where(den != 0, 1 - _safe_div(product, den), 0.0)
            scalar_maps["VF"] = _save_map(VF, "VF")
        else:
            scalar_maps["VF"] = fpath

    # ------------------------------------------------------------------ #
    # GA — Geodesic Anisotropy
    # ------------------------------------------------------------------ #
    if "ga" in dtmaps or compute_all:
        fpath = f"{out_basename}_GA.nii.gz"
        if not os.path.isfile(fpath) or overwrite:
            # Geodesic anisotropy (Batchelor et al., 2005). It is only defined
            # for positive-definite tensors, so it is 0 where any eigenvalue <= 0
            log_l1 = _safe_log(l1_data)
            log_l2 = _safe_log(l2_data)
            log_l3 = _safe_log(l3_data)
            mean_log = (log_l1 + log_l2 + log_l3) / 3
            GA = np.sqrt(
                (log_l1 - mean_log) ** 2
                + (log_l2 - mean_log) ** 2
                + (log_l3 - mean_log) ** 2
            )
            positive = (l1_data > 0) & (l2_data > 0) & (l3_data > 0)
            GA = np.where(positive, GA, 0.0)
            scalar_maps["GA"] = _save_map(GA, "GA")
        else:
            scalar_maps["GA"] = fpath

    # ------------------------------------------------------------------ #
    # RA — Relative Anisotropy
    # ------------------------------------------------------------------ #
    if "ra" in dtmaps or compute_all:
        fpath = f"{out_basename}_RA.nii.gz"
        if not os.path.isfile(fpath) or overwrite:
            num = np.sqrt(
                (l1_data - MD_data) ** 2
                + (l2_data - MD_data) ** 2
                + (l3_data - MD_data) ** 2
            )
            RA = _safe_div(num, np.sqrt(3) * MD_data)
            scalar_maps["RA"] = _save_map(RA, "RA")
        else:
            scalar_maps["RA"] = fpath

    return scalar_maps


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############                 Section 2: Class to work with Diffusion Schemes            ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
class DiffusionScheme:
    def __init__(self):
        self.gradients = None  # (N, 3)
        self.bvals = None  # (N,)
        self.scheme_type = None

    # -------------------------
    # Loaders
    # -------------------------
    @classmethod
    def from_bvec_bval_files(cls, bvec_file, bval_file):
        bvecs = np.loadtxt(bvec_file)
        bvals = np.loadtxt(bval_file)
        return cls.from_bvec_bval_arrays(bvecs, bvals)

    @classmethod
    def from_bvec_bval_arrays(cls, bvecs, bvals):
        obj = cls()

        bvecs = np.asarray(bvecs)
        bvals = np.asarray(bvals)

        if bvecs.shape[0] == 3:
            bvecs = bvecs.T

        obj.gradients = bvecs
        obj.bvals = bvals.flatten()  # Ensure 1D array
        obj._detect_scheme()
        return obj

    @classmethod
    def from_bmatrix_file(cls, bmat_file):
        bmat = np.loadtxt(bmat_file)
        return cls.from_bmatrix_array(bmat)

    @classmethod
    def from_bmatrix_array(cls, bmat):
        """
        bmat shape: (N, 6) with:
        [Bxx, Byy, Bzz, Bxy, Bxz, Byz]
        """
        obj = cls()
        bmat = np.asarray(bmat)

        Bxx, Byy, Bzz, Bxy, Bxz, Byz = bmat.T
        obj.bvals = Bxx + Byy + Bzz

        # The diagonal gives the magnitude of each gradient component (b * g_i^2)
        gradients = np.vstack(
            [
                np.sqrt(np.maximum(Bxx, 0)),
                np.sqrt(np.maximum(Byy, 0)),
                np.sqrt(np.maximum(Bzz, 0)),
            ]
        ).T

        # The signs come from the off-diagonal terms (b * g_i * g_j). A gradient and
        # its opposite give the same b-matrix, so the largest component is taken as
        # positive and the sign of each other component is the sign of its product
        # with that one.
        offdiag = np.array(
            [
                [np.zeros_like(Bxy), Bxy, Bxz],
                [Bxy, np.zeros_like(Bxy), Byz],
                [Bxz, Byz, np.zeros_like(Bxy)],
            ]
        )  # (3, 3, N)
        ref = np.argmax(gradients, axis=1)
        rows = np.arange(len(ref))
        signs = np.sign(offdiag[ref, :, rows])  # (N, 3)
        signs[signs == 0] = 1
        gradients = gradients * signs

        # Normalize
        norms = np.linalg.norm(gradients, axis=1)
        norms[norms == 0] = 1
        obj.gradients = gradients / norms[:, None]

        obj._detect_scheme()
        return obj

    # -------------------------
    # Simulation
    # -------------------------
    @classmethod
    def simulate_dwi_acq_scheme(
        cls,
        scheme_type: str = "shelled",
        shells: dict = None,
        n_b0s: int = 6,
        bmax: float = 4000,
        radius: int = 4,
        n_iter: int = 200,
    ) -> "DiffusionScheme":
        """
        Simulate a diffusion acquisition scheme (shelled or cartesian/DSI).

        Parameters
        ----------
        scheme_type : str, optional
            ``"shelled"`` (single- or multi-shell HARDI) or ``"cartesian"`` (DSI q-space
            grid). ``"shell"`` and ``"dsi"`` are accepted as aliases. Default is ``"shelled"``.

        shells : dict, optional
            Only for shelled schemes. Mapping ``{b-value: number of directions}``.
            Default is ``{1000: 30, 2000: 60}``.

        n_b0s : int, optional
            Number of B0 volumes placed at the beginning of the scheme. Used by both
            scheme types. Default is 6.

        bmax : float, optional
            Only for cartesian schemes. b-value at the outermost grid radius. Default is 4000.

        radius : int, optional
            Only for cartesian schemes. Radius of the q-space sphere in grid units.
            Every grid point with ``0 < |q| <= radius`` in the upper half-space
            (``qz >= 0``) is sampled. Default is 4.

        n_iter : int, optional
            Only for shelled schemes. Iterations of the electrostatic repulsion used to
            spread the directions of each shell. Default is 200.

        Returns
        -------
        DiffusionScheme
            Object with ``gradients`` (N, 3), ``bvals`` (N,) and ``scheme_type`` set.

        Examples
        --------
        >>> scheme = DiffusionScheme.simulate_dwi_acq_scheme(
        ...     "shelled", shells={1000: 32, 2000: 64, 3000: 96}, n_b0s=8
        ... )
        >>> scheme = DiffusionScheme.simulate_dwi_acq_scheme(
        ...     "cartesian", bmax=6000, radius=5, n_b0s=1
        ... )
        >>> scheme.plot()
        """

        stype = str(scheme_type).lower()
        if stype in ("shelled", "shell", "hardi", "multishell"):
            stype = "shelled"
        elif stype in ("cartesian", "dsi", "grid"):
            stype = "cartesian"
        else:
            raise ValueError(
                f"Unknown scheme_type '{scheme_type}'. Use 'shelled' or 'cartesian'."
            )

        if n_b0s < 0:
            raise ValueError("n_b0s must be greater than or equal to 0.")

        if stype == "shelled":
            if shells is None:
                shells = {1000: 30, 2000: 60}
            bvecs, bvals = cls._shelled_scheme(shells, n_b0s=n_b0s, n_iter=n_iter)
        else:
            bvecs, bvals = cls._dsi_scheme(bmax=bmax, radius=radius, n_b0s=n_b0s)

        obj = cls.from_bvec_bval_arrays(bvecs, bvals)
        obj.scheme_type = stype  # The type is known, no need to rely on detection
        return obj

    @staticmethod
    def _sphere_dirs(n: int, n_iter: int = 200) -> np.ndarray:
        """
        Return ``n`` unit vectors (n, 3) evenly spread over the half sphere (z >= 0).

        Directions start from a Fibonacci spiral and are refined with an antipodally
        symmetric electrostatic repulsion (each direction also repels the opposite
        of the others), as gradient directions and their opposites are equivalent.
        """
        if n <= 0:
            return np.zeros((0, 3))

        # Fibonacci spiral initialization on the upper hemisphere
        i = np.arange(n) + 0.5
        z = 1 - i / n
        r = np.sqrt(1 - z**2)
        phi = np.pi * (1 + np.sqrt(5)) * i
        dirs = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

        if n > 1:
            # Step size proportional to the mean spacing between 2n points on the sphere
            step = 0.2 * np.sqrt(4 * np.pi / (2 * n))
            for it in range(n_iter):
                diff_m = dirs[:, None, :] - dirs[None, :, :]
                diff_p = dirs[:, None, :] + dirs[None, :, :]
                dm = np.linalg.norm(diff_m, axis=2)
                dp = np.linalg.norm(diff_p, axis=2)
                np.fill_diagonal(dm, np.inf)
                np.fill_diagonal(dp, np.inf)

                force = (diff_m / dm[..., None] ** 3).sum(axis=1) + (
                    diff_p / dp[..., None] ** 3
                ).sum(axis=1)

                # Keep only the tangential component
                force -= np.sum(force * dirs, axis=1)[:, None] * dirs
                fmax = np.linalg.norm(force, axis=1).max()
                if fmax == 0:
                    break

                dirs = dirs + step * (1 - it / n_iter) * force / fmax
                dirs /= np.linalg.norm(dirs, axis=1)[:, None]

        # Bring every direction to the upper hemisphere
        dirs[dirs[:, 2] < 0] *= -1
        return dirs

    @staticmethod
    def _shelled_scheme(shells: dict, n_b0s: int = 6, n_iter: int = 200) -> tuple:
        """bvecs (3 x N) and bvals (N,) for a shelled acquisition: {b-value: number of directions}."""
        if not isinstance(shells, dict) or len(shells) == 0:
            raise ValueError("shells must be a non-empty dict {b-value: n_directions}.")

        g = [np.zeros((n_b0s, 3))] + [
            DiffusionScheme._sphere_dirs(int(n), n_iter=n_iter) for n in shells.values()
        ]
        b = [np.zeros(n_b0s)] + [
            np.full(int(n), float(bval)) for bval, n in shells.items()
        ]
        return np.vstack(g).T, np.concatenate(b)

    @staticmethod
    def _dsi_scheme(bmax: float = 4000, radius: int = 4, n_b0s: int = 1) -> tuple:
        """bvecs (3 x N) and bvals (N,) for a cartesian (DSI) q-space grid inside a sphere of the given radius."""
        if radius < 1:
            raise ValueError("radius must be an integer greater than or equal to 1.")

        rng = range(-radius, radius + 1)
        grid = np.array(
            [
                (i, j, k)
                for i in rng
                for j in rng
                for k in range(0, radius + 1)
                if 0 < i * i + j * j + k * k <= radius**2
            ],
            dtype=float,
        )
        q = np.linalg.norm(grid, axis=1)

        g = np.vstack([np.zeros((n_b0s, 3)), grid / q[:, None]])
        b = np.concatenate([np.zeros(n_b0s), bmax * (q / radius) ** 2])
        return g.T, b

    # -------------------------
    # Scheme detection
    # -------------------------
    def _detect_scheme(self):
        """
        Detect if the acquisition scheme is shelled (HARDI/multi-shell) or cartesian (DSI).

        Detection criteria:
        - Shelled: Few discrete b-value shells (typically 2-6) with many samples per shell
        - Cartesian (DSI): Many b-value shells (typically >8) with samples distributed across wide range
        """
        b0_thresh = 50  # Threshold to identify b0 images

        # Get non-b0 values
        non_b0_bvals = self.bvals[self.bvals > b0_thresh]

        if len(non_b0_bvals) == 0:
            self.scheme_type = "b0_only"
            return

        # Round to nearest 100 to account for small variations
        rounded_bvals = np.round(non_b0_bvals, -2)
        unique_shells = np.unique(rounded_bvals)
        n_shells = len(unique_shells)

        # Calculate distribution metrics
        mean_bval = non_b0_bvals.mean()
        std_bval = non_b0_bvals.std()
        cv = std_bval / mean_bval  # Coefficient of variation

        # Count samples per shell
        samples_per_shell = []
        for shell in unique_shells:
            n_samples = np.sum(np.abs(rounded_bvals - shell) < 50)
            samples_per_shell.append(n_samples)

        mean_samples_per_shell = np.mean(samples_per_shell)

        # Decision criteria
        # Shelled data typically has:
        # - Few shells (2-6)
        # - Many samples per shell (>10)
        # - Lower coefficient of variation (<0.4)
        #
        # DSI data typically has:
        # - Many shells (>8)
        # - Few samples per shell (<15)
        # - Higher coefficient of variation (>0.35)

        if n_shells <= 6 and mean_samples_per_shell > 10:
            self.scheme_type = "shelled"
        elif n_shells > 8 and cv > 0.35:
            self.scheme_type = "cartesian"
        else:
            # Borderline case - use number of shells as primary criterion
            if n_shells <= 6:
                self.scheme_type = "shelled"
            else:
                self.scheme_type = "cartesian"

    # -------------------------
    # Visualization
    # -------------------------
    def plot(
        self,
        show=True,
        use_notebook: bool = False,
        radius: float = 10.0,
        colormap: str = "jet",
        toroid_radius: float = None,
        toroid_alpha: float = 0.3,
        b0_thresh: float = 10.0,
        show_colorbar: bool = True,
        show_axes: bool = True,
        show_opposite_dirs: bool = True,
    ):

        g = self.gradients
        b = self.bvals

        if self.scheme_type is None:
            self._detect_scheme()

        # Apply appropriate coordinate transformation
        if self.scheme_type == "shelled":
            # HARDI: scale by b-value
            coords = g * b[:, None]
        else:
            # DSI: Coords = max(bvals)*grads.*sqrt(bvals/max(bvals))
            # This simplifies to: grads * sqrt(bvals * max(bvals))
            b_max = b.max()
            coords = g * np.sqrt(b[:, None] * b_max)

        # FIGURE CONFIGURATION
        figure_conf = {
            "background_color": "black",
            "title_font_color": "white",
            "colorbar_font_color": "white",
            "title_font_type": "arial",
            "title_font_size": 10,
            "title_shadow": True,
            "mesh_ambient": 0.2,
            "mesh_diffuse": 0.5,
            "mesh_specular": 0.5,
            "mesh_specular_power": 15,
            "mesh_smooth_shading": True,
        }

        rgba_data = cltcol.values2colors(
            b, cmap=colormap, vmin=b.min(), vmax=b.max(), output_format="rgb"
        )

        # Optionally add opposite directions (mirror across origin)
        if show_opposite_dirs:
            coords = np.vstack([coords, -coords])
            rgba_data = np.vstack([rgba_data, rgba_data])
            b = np.concatenate([b, b])

        # Detecting the screen size for the plotter
        screen_size = cltplot.get_current_monitor_size()

        # Create PyVista plotter with appropriate rendering mode
        plotter_kwargs = {
            "notebook": use_notebook,
            "window_size": [screen_size[0], screen_size[1]],
        }

        pv_plotter = pv.Plotter(**plotter_kwargs)
        pv_plotter.set_background(figure_conf["background_color"])

        # Add gradient points as spheres
        pv_plotter.add_points(
            coords,
            render_points_as_spheres=True,
            point_size=radius,
            scalars=rgba_data,
            rgb=True,
            ambient=figure_conf["mesh_ambient"],
            diffuse=figure_conf["mesh_diffuse"],
            specular=figure_conf["mesh_specular"],
            specular_power=figure_conf["mesh_specular_power"],
            smooth_shading=figure_conf["mesh_smooth_shading"],
            show_scalar_bar=False,
        )

        # Add center sphere for b0
        pv_plotter.add_points(
            np.array([[0.0, 0.0, 0.0]]),
            render_points_as_spheres=True,
            point_size=radius,
            color="white",
            ambient=figure_conf["mesh_ambient"],
            diffuse=figure_conf["mesh_diffuse"],
            specular=figure_conf["mesh_specular"],
            specular_power=figure_conf["mesh_specular_power"],
            smooth_shading=figure_conf["mesh_smooth_shading"],
        )

        # Add toroidal shells at each unique b-value
        # Use original b-values (not duplicated) for toroids
        original_bvals = self.bvals
        unique_bvals = np.unique(original_bvals)
        unique_bvals = unique_bvals[unique_bvals > b0_thresh]  # Exclude b0

        # Get colors for each unique b-value
        unique_colors = cltcol.values2colors(
            unique_bvals,
            cmap=colormap,
            vmin=original_bvals.min(),
            vmax=original_bvals.max(),
            output_format="rgb",
        )

        # Auto-calculate toroid tube radius if not provided
        if toroid_radius is None:
            b_max = original_bvals.max()
            toroid_radius = b_max * 0.005 if b_max > 0 else 0.5

        for bval, color in zip(unique_bvals, unique_colors, strict=False):
            torus = pv.ParametricTorus(
                ringradius=bval, crosssectionradius=toroid_radius
            )
            pv_plotter.add_mesh(
                torus,
                color=color,
                opacity=toroid_alpha,
                ambient=figure_conf["mesh_ambient"],
                diffuse=figure_conf["mesh_diffuse"],
                specular=figure_conf["mesh_specular"],
                specular_power=figure_conf["mesh_specular_power"],
                smooth_shading=figure_conf["mesh_smooth_shading"],
            )

        # Add coordinate axes
        if show_axes:
            b_max = original_bvals.max()
            axis_length = b_max * 1.1

            # X-axis
            pv_plotter.add_lines(
                np.array([[-axis_length, 0, 0], [axis_length, 0, 0]]),
                color="white",
                width=2,
            )
            # Y-axis
            pv_plotter.add_lines(
                np.array([[0, -axis_length, 0], [0, axis_length, 0]]),
                color="white",
                width=2,
            )
            # Z-axis
            pv_plotter.add_lines(
                np.array([[0, 0, -axis_length], [0, 0, axis_length]]),
                color="white",
                width=2,
            )

        # Add colorbar - vertical on the right
        if show_colorbar:
            # Create a dummy mesh for the colorbar
            dummy_mesh = pv.PolyData(coords)
            dummy_mesh["bvalues"] = b

            pv_plotter.add_mesh(
                dummy_mesh,
                scalars="bvalues",
                cmap=colormap,
                show_edges=False,
                opacity=0,  # Make it invisible
                scalar_bar_args={
                    "title": "b-value (s/mm²)",
                    "title_font_size": 20,
                    "label_font_size": 16,
                    "color": "white",
                    "position_x": 0.90,  # Far right
                    "position_y": 0.25,  # Vertically centered
                    "width": 0.08,  # Narrow width for vertical bar
                    "height": 0.5,  # Tall for vertical orientation
                    "vertical": True,  # Explicitly vertical
                    "n_labels": 5,  # Number of labels
                    "fmt": "%.0f",  # Format as integers
                },
            )

        # Count b0 and DWI images (use original counts)
        n_b0s = np.sum(original_bvals <= b0_thresh)
        n_dwi = len(original_bvals) - n_b0s

        # Add title
        pv_plotter.add_text(
            f"q-Space Plot: {n_b0s} B0 Images and {n_dwi} Diffusion Images\nScheme: {self.scheme_type}",
            position="upper_edge",
            font_size=14,
            color="white",
            font="arial",
        )

        # Set camera and lighting
        pv_plotter.add_light(pv.Light(position=(1, 1, 1)))
        pv_plotter.view_isometric()

        if show:
            pv_plotter.show()

        return pv_plotter
