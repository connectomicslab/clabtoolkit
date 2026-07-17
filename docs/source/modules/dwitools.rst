dwitools module
===============

.. automodule:: clabtoolkit.dwitools
   :members:
   :undoc-members:
   :show-inheritance:

The dwitools module provides tools for diffusion-weighted imaging (DWI) data: volume management, b-value and gradient direction handling, and tensor-derived map generation.

.. note::
   Tractogram handling lives in :doc:`tracttools`, not here. Streamline loading,
   clustering, format conversion (``trk2tck`` / ``tck2trk``) and visualization are
   all provided by ``clabtoolkit.tracttools``.

Key Features
------------
- DWI volume manipulation and removal by index or b-value
- B0 volume extraction
- Acquisition scheme handling from bvec/bval or b-matrix sources
- Gradient direction visualization
- Tensor eigenvalue to scalar map conversion (FA, MD, and related maps)

Main Classes
------------

DiffusionScheme
~~~~~~~~~~~~~~~
Represents a diffusion acquisition scheme, built through class-method constructors rather than direct instantiation.

Key Methods:
- ``from_bvec_bval_files()``: Build a scheme from bvec and bval files
- ``from_bvec_bval_arrays()``: Build a scheme from bvec and bval arrays
- ``from_bmatrix_file()``: Build a scheme from a b-matrix file
- ``from_bmatrix_array()``: Build a scheme from a b-matrix array
- ``plot()``: Visualize the gradient directions on a sphere

Main Functions
--------------

Volume Management
~~~~~~~~~~~~~~~~~
- ``delete_dwi_volumes()``: Remove DWI volumes by volume index or by b-value
- ``get_b0s()``: Extract b=0 volumes from a DWI dataset

Tensor Maps
~~~~~~~~~~~
- ``maps_from_tensor_eigenvalues()``: Derive scalar maps from tensor eigenvalues

Common Usage Examples
---------------------

DWI volume manipulation::

    from clabtoolkit.dwitools import delete_dwi_volumes

    # Remove specific volumes, keeping the bvec/bval files in sync
    delete_dwi_volumes(
        in_image="dwi.nii.gz",
        bvec_file="dwi.bvec",
        bval_file="dwi.bval",
        vols_to_delete=[0, 5, 10],
        out_image="cleaned_dwi.nii.gz"
    )

    # Or remove every volume acquired at a given b-value
    delete_dwi_volumes(
        in_image="dwi.nii.gz",
        bvec_file="dwi.bvec",
        bval_file="dwi.bval",
        bvals_to_delete=[3000],
        out_image="cleaned_dwi.nii.gz"
    )

Working with b-values::

    from clabtoolkit.dwitools import get_b0s

    # Extract the b=0 volumes into their own image
    b0s_img, b0_vols = get_b0s(
        dwi_img="dwi.nii.gz",
        b0s_img="dwi_b0s.nii.gz",
        bval_file="dwi.bval",
        bval_thresh=50
    )
    print(f"Found {len(b0_vols)} b0 volumes")

Inspecting an acquisition scheme::

    from clabtoolkit.dwitools import DiffusionScheme

    # Build the scheme from the gradient files
    scheme = DiffusionScheme.from_bvec_bval_files(
        bvec_file="dwi.bvec",
        bval_file="dwi.bval"
    )

    # Visualize the gradient directions
    scheme.plot(show=True)

    # A scheme can also be built from a b-matrix
    scheme = DiffusionScheme.from_bmatrix_file("dwi.bmat")

Tensor-derived maps::

    from clabtoolkit.dwitools import maps_from_tensor_eigenvalues

    # Generate scalar maps from tensor eigenvalues
    maps = maps_from_tensor_eigenvalues(
        eigvals="dti_eigenvalues.nii.gz",
        out_basename="/path/to/output/sub-01_dti",
        dtmaps=["all"],
        overwrite=True
    )
    print(maps)  # dict mapping each map tag to its saved path
