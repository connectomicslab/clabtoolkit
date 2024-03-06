import os
import numpy as np
import nibabel as nib
from skimage import measure


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
