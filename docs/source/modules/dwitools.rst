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
- q-space visualization of the acquisition scheme, displayed in a window or a
  notebook, or saved as an image, a vector graphic or an interactive HTML file
- Tensor eigenvalue to scalar map conversion (FA, MD, and related maps)
- b-values stored as a single row (FSL) or as a single column

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
  (``[Bxx, Byy, Bzz, Bxy, Bxz, Byz]`` per row). The gradient signs are recovered
  from the off-diagonal terms; a direction and its opposite give the same
  b-matrix, so each gradient is returned with its largest component positive.
- ``simulate_dwi_acq_scheme()``: Simulate a shelled (``{b-value: directions}``)
  or a cartesian/DSI (q-space grid) acquisition scheme
- ``plot()``: Visualize the gradient directions in q-space

The scheme type (``"shelled"``, ``"cartesian"`` or ``"b0_only"``) is detected
automatically and stored in ``scheme_type``.

``plot()`` options:

- ``save_path``: save the figure instead of displaying it. The format is chosen
  from the extension: ``.html``/``.htm`` exports an interactive HTML file,
  ``.svg``, ``.pdf``, ``.eps``, ``.ps`` or ``.tex`` a vector graphic, and any
  other extension (e.g. ``.png``) a screenshot. Returns ``None`` after saving.
- ``use_notebook``: display the figure inside a Jupyter notebook.
- ``window_size``: figure size in pixels. By default notebook figures use
  PyVista's default window size (1024 x 768) so they fit in the cell output,
  while windows and saved figures use the monitor size.
- ``non_blocking``: open the window in a separate thread.
- ``show=False``: return the PyVista plotter without displaying it.
- ``radius``, ``colormap``, ``toroid_radius``, ``toroid_alpha``,
  ``show_colorbar``, ``show_axes``, ``show_opposite_dirs``: appearance.

The gradient directions are drawn as sphere meshes, so they keep their 3D
shading in the window, in notebooks and in the exported HTML.

Main Functions
--------------

Volume Management
~~~~~~~~~~~~~~~~~
- ``delete_dwi_volumes()``: Remove DWI volumes by volume index or by b-value.
  Without volumes or b-values, the trailing B0s are removed. Always returns
  ``(out_image, out_bvec, out_bval, removed_volumes)``; when nothing is removed
  the input paths are returned with an empty array.
- ``get_b0s()``: Extract the volumes with a b-value lower than or equal to
  ``bval_thresh`` (default 0). Returns ``(b0s_img, b0_volumes)``. Without an
  output name, the B0s are saved next to the DWI image as ``<name>_b0s.nii.gz``.
  A ``ValueError`` is raised when no B0 volume is found.

Both functions check that the number of b-values matches the number of volumes.

Tensor Maps
~~~~~~~~~~~
- ``maps_from_tensor_eigenvalues()``: Derive scalar maps from tensor eigenvalues,
  given as a 4D image (λ1, λ2, λ3 along the 4th axis) or as three 3D images.
  Voxels where a map is undefined (e.g. background) are set to 0.

  ========  ===============================  ==========================================
  Tag       Map                              Definition
  ========  ===============================  ==========================================
  ``AD``    Axial diffusivity                λ1
  ``RD``    Radial diffusivity               (λ2 + λ3) / 2
  ``MD``    Mean diffusivity                 (λ1 + λ2 + λ3) / 3
  ``FA``    Fractional anisotropy            √(3/2) · √Σ(λi − MD)² / √Σλi²
  ``CL``    Linear anisotropy (Westin)       (λ1 − λ2) / Σλi
  ``CP``    Planar anisotropy (Westin)       2 (λ2 − λ3) / Σλi
  ``CS``    Spherical anisotropy (Westin)    3 λ3 / Σλi
  ``VF``    Volume fraction                  1 − λ1 λ2 λ3 / MD³
  ``GA``    Geodesic anisotropy              √Σ(log λi − mean(log λ))², 0 if any λi ≤ 0
  ``RA``    Relative anisotropy              √Σ(λi − MD)² / (√3 · MD)
  ========  ===============================  ==========================================

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

    # Without an output name the B0s are saved as dwi_b0s.nii.gz next to dwi.nii.gz
    b0s_img, b0_vols = get_b0s("dwi.nii.gz")

Inspecting an acquisition scheme::

    from clabtoolkit.dwitools import DiffusionScheme

    # Build the scheme from the gradient files
    scheme = DiffusionScheme.from_bvec_bval_files(
        bvec_file="dwi.bvec",
        bval_file="dwi.bval"
    )

    print(scheme.scheme_type)  # "shelled", "cartesian" or "b0_only"

    # Visualize the gradient directions in an interactive window
    scheme.plot()

    # Inside a Jupyter notebook (the figure fits the cell output)
    scheme.plot(use_notebook=True)
    scheme.plot(use_notebook=True, window_size=(800, 600))

    # Save the figure instead of displaying it
    scheme.plot(save_path="scheme.png")                     # Screenshot
    scheme.plot(save_path="scheme.html")                    # Interactive HTML
    scheme.plot(save_path="scheme.pdf", show_axes=False)    # Vector graphic

    # A scheme can also be built from a b-matrix
    scheme = DiffusionScheme.from_bmatrix_file("dwi.bmat")

    # Or simulated, e.g. to design or illustrate an acquisition
    hardi = DiffusionScheme.simulate_dwi_acq_scheme("shelled", shells={1000: 30, 2000: 60})
    dsi = DiffusionScheme.simulate_dwi_acq_scheme("cartesian", bmax=4000, radius=4)
    dsi.plot(save_path="dsi_scheme.html")

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
