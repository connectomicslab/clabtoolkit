import os
import numpy as np
import nibabel as nib
from skimage import measure


# This function removes the B0s volumes located at the end of the diffusion 4D volume.
def remove_empty_dwi_Volume(dwifile: str):
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

def prepare_parcellation_for_tracking(parc_file: Union[str, np.uint] = None, 
                                    out_file: Union[str, np.uint] = None):
    """
    Prepare the parcellation for fibre tracking. It will add the parcellated wm voxels to its
    corresponding gm label. It also puts to zero the voxels that are not in the gm.

    Parameters
    ----------
    parc_file : str or np.uint
        The path to the parcellation file or the parcellation data. The parcellation data 
        should be a numpy array.
        
    out_file : str or np.uint
        The path to save the parcellation file or the parcellation data. The parcellation data 
        should be a numpy array.

    Returns
    -------
    out_file : str or np.uint
        The path to the parcellation file or numpy array with the parcellation data.
        
    Examples
    --------
    >>> out_file = prepare_parcellation_for_tracking(parc_file='/home/yaleman/parc.nii.gz')
        
    
    """
    
    # Verify if the input is a numpy array or a file
    if isinstance(parc_file, np.ndarray):
        iparc = cltparc.Parcellation(data=parc_file)  

    elif isinstance(parc_file, str):
        if not os.path.exists(parc_file):
            raise ValueError(f"File {parc_file} does not exist.")
        else:
            iparc = cltparc.Parcellation(parc_file=parc_file)
            
    else:
        raise ValueError(f"Invalid input for parc_file: {parc_file}")

    # Unique of non-zero values
    sts_vals = np.unique(iparc.data)

    # sts_vals as integers
    sts_vals = sts_vals.astype(int)

    # get the values of sts_vals that are bigger or equaal to 5000 and create a list with them
    indexes = [x for x in sts_vals if x >= 5000]

    iparc.remove_by_code(codes2remove=indexes)

    # Get the labeled wm values
    ind = np.argwhere(iparc.data >=3000)
    
    # Add the wm voxels to the gm label
    iparc.data[ind[:,0],ind[:,1],ind[:,2]] = iparc.data[ind[:,0],ind[:,1],ind[:,2]] - 3000
    
    # Adjust the values
    iparc.adjust_values()

    # Save the parcellation
    if isinstance(out_file, str):
        iparc.save_parcellation(out_file=out_file, save_lut=True)
        
    elif isinstance(out_file, np.ndarray):
        return iparc.data
        
    
    return out_file
