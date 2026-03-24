import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union


def load_bvecs(path: Union[str, Path]) -> np.ndarray:
    """
    Load a .bvec file and return array of shape (3, N).

    Parameters
    ----------
    path : Union[str, Path]
        Path to the .bvec file.

    Returns
    -------
    np.ndarray
        Array of shape (3, N) containing the gradient directions.
    """

    # Loading the file
    if isinstance(path, str):
        path = Path(path)

    bvecs = np.loadtxt(path)

    # Validating the shape
    if bvecs.shape[0] != 3:
        raise ValueError(
            f"Expected bvec file with 3 rows (3, N), got shape {bvecs.shape} in {path}"
        )
    return bvecs


########################################################################################
def load_bvals(path: Union[str, Path]) -> np.ndarray:
    """
    Load a .bval file and return a 1D array of shape (N,).

    Parameters
    ----------
    path : Union[str, Path]
        Path to the .bval file.

    Returns
    -------
    np.ndarray
        1D array of shape (N,) containing the b-values.

    """

    # Load the file
    if isinstance(path, str):
        path = Path(path)

    bvals = np.loadtxt(path)

    if bvals.ndim != 1:
        raise ValueError(
            f"Expected bval file with 1 row, got shape {bvals.shape} in {path}"
        )
    return bvals


########################################################################################
def save_bvecs(bvecs: np.ndarray, path: Union[str, Path]) -> None:
    """
    Method to save bvecs to a file.

    Parameters
    ----------
    bvecs : np.ndarray
        Array of shape (3, N) containing the gradient directions.

    path : Union[str, Path]
        Path to the output .bvec file.
    """

    # Check that bvecs is a 3x N array if the number of rows is not 3 transpose it
    if bvecs.shape[0] != 3:
        bvecs = bvecs.T

    if isinstance(path, str):
        path = Path(path)

    np.savetxt(path, bvecs, fmt="%.6f")


########################################################################################
def save_bvals(bvals: np.ndarray, path: Union[str, Path]) -> None:
    """
    Method to save bvals to a file.

    Parameters
    ----------
    bvals : np.ndarray
        Array of shape (N,) containing the b-values.

    path : Union[str, Path]
        Path to the output .bval file.

    """

    if isinstance(path, str):
        path = Path(path)

    np.savetxt(path, bvals[np.newaxis, :], fmt="%g")
