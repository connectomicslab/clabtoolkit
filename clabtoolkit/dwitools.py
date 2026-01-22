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
from . import plottools as cltplot
from . import colorstools as cltcol

import pyvista as pv


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

        gradients = np.vstack(
            [
                np.sqrt(np.maximum(Bxx, 0)),
                np.sqrt(np.maximum(Byy, 0)),
                np.sqrt(np.maximum(Bzz, 0)),
            ]
        ).T

        # Normalize
        norms = np.linalg.norm(gradients, axis=1)
        norms[norms == 0] = 1
        obj.gradients = gradients / norms[:, None]

        obj._detect_scheme()
        return obj

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
            np.array([[0, 0, 0]]),
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

        for bval, color in zip(unique_bvals, unique_colors):
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
