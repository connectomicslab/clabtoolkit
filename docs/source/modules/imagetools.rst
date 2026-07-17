imagetools module
=================

.. automodule:: clabtoolkit.imagetools
   :members:
   :undoc-members:
   :show-inheritance:

The imagetools module provides advanced neuroimaging operations, morphological processing, and image manipulation capabilities specifically designed for brain imaging data.

Key Features
------------
- 2D/3D morphological operations on binary arrays
- Structuring element generation, including millimetre-aware dilation
- Affine utilities: voxel size, volume, centre and coordinate conversion
- Image cropping, merging and resampling
- Probabilistic map (SPAM) creation and maximum-probability conversion
- Mesh and centroid extraction from volumes
- 3D/4D scalar data interpolation at specified coordinates
- Region growing and image simulation

Main Classes
------------

MorphologicalOperations
~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive class for binary array morphological processing.

Key Methods:
- ``create_structuring_element()``: Build a structuring element of a given shape and size
- ``erode()``: Erode binary structures
- ``dilate()``: Dilate binary structures
- ``dilate_mm()``: Dilate by a distance in millimetres, using the affine
- ``opening()``: Morphological opening (erosion followed by dilation)
- ``closing()``: Morphological closing (dilation followed by erosion)
- ``fill_holes()``: Fill holes in binary structures
- ``remove_small_objects()``: Filter small connected components
- ``gradient()``: Morphological gradient
- ``tophat()`` / ``blackhat()``: Top-hat and black-hat transforms

Main Functions
--------------

Morphology
~~~~~~~~~~
- ``quick_morphology()``: Apply a named morphological operation in one call
- ``dilate_mm()``: Dilate an array by a distance in millimetres
- ``region_growing()``: Grow a region within a mask

Affine Utilities
~~~~~~~~~~~~~~~~
- ``get_voxel_size()`` / ``get_voxel_volume()``: Voxel geometry from an affine
- ``get_center()`` / ``get_rotation_matrix()``: Affine decomposition
- ``vox2mm()`` / ``mm2vox()``: Convert between voxel and world coordinates
- ``get_vox_neighbors()``: Neighbouring voxel coordinates

Image Manipulation
~~~~~~~~~~~~~~~~~~
- ``crop_image_from_mask()``: Crop an image using a mask
- ``cropped_to_native()``: Map a cropped image back to native space
- ``apply_multi_transf()``: Apply multiple transformations to an image
- ``merge_to_4d()``: Merge several images into a 4D volume
- ``delete_volumes_from_4D_images()`` / ``delete_volumes_from_4D_array()``: Drop volumes from 4D data

Probabilistic Maps
~~~~~~~~~~~~~~~~~~
- ``create_spams()`` / ``create_spams_from_volume()``: Build probabilistic maps
- ``spams2maxprob()`` / ``spams2maxprob_from_volume()``: Convert SPAMs to a maximum-probability parcellation

Mesh and Statistics
~~~~~~~~~~~~~~~~~~~
- ``extract_mesh_from_volume()``: Extract a mesh from a volume
- ``extract_centroid_from_volume()``: Extract a centroid from a volume
- ``compute_statistics_at_nonzero_voxels()``: Compute statistics at non-zero voxels
- ``interpolate()``: Interpolate 3D/4D scalar data at voxel coordinates

Simulation
~~~~~~~~~~
- ``simulate_image()`` / ``simulate_array()``: Simulate image data

Common Usage Examples
---------------------

Binary image processing::

    from clabtoolkit.imagetools import MorphologicalOperations
    import nibabel as nib
    import numpy as np

    # Load a binary image
    img = nib.load("/path/to/binary_mask.nii.gz")
    binary_data = img.get_fdata().astype(bool)

    # Initialize morphological operations
    morph = MorphologicalOperations()

    # Build a structuring element and close the image
    # Supported shapes: 'cube'/'square', 'ball'/'disk', 'cross'
    structure = morph.create_structuring_element(shape="ball", size=3)
    closed_image = morph.closing(binary_data, structure=structure)

    # Fill holes
    filled_image = morph.fill_holes(closed_image)

Chaining operations::

    # Open, then drop small connected components
    kernel = morph.create_structuring_element(shape="cube", size=3)
    processed = morph.opening(binary_data, structure=kernel)
    processed = morph.remove_small_objects(processed, min_size=100)

    # Save the result
    output_img = nib.Nifti1Image(processed.astype(np.uint8), img.affine, img.header)
    nib.save(output_img, "/path/to/processed_mask.nii.gz")

    # A single operation can also be applied directly
    from clabtoolkit.imagetools import quick_morphology
    eroded = quick_morphology(binary_data, operation="erode")

Millimetre-aware dilation::

    # Dilate by a physical distance rather than a voxel count
    dilated = morph.dilate_mm(
        array=binary_data,
        affine=img.affine,
        shape="ball",
        dilation_mm=2.5
    )

Affine utilities::

    from clabtoolkit.imagetools import get_voxel_size, get_voxel_volume, vox2mm

    voxel_size = get_voxel_size(img.affine)
    voxel_volume = get_voxel_volume(img.affine)

    # Convert voxel coordinates to world coordinates
    mm_coords = vox2mm(np.array([[10, 20, 15]]), img.affine)

3D/4D data interpolation::

    from clabtoolkit.imagetools import interpolate

    # Load a 3D scalar volume
    scalar_img = nib.load("/path/to/scalar_data.nii.gz")
    scalar_data = scalar_img.get_fdata()

    # Voxel coordinates to sample (Nx3, fractional coordinates allowed)
    vertices_vox = np.array([
        [10.5, 20.3, 15.7],
        [25.2, 30.1, 45.9],
        [12.0, 18.5, 22.3]
    ])

    # Interpolate values at the specified coordinates
    values = interpolate(
        scalar_data=scalar_data,
        vertices_vox=vertices_vox,
        interp_method="linear"  # 'linear', 'nearest' or 'slinear'
    )

    # For 4D data, one value is returned per timepoint
    functional_data = nib.load("/path/to/4d_functional.nii.gz").get_fdata()
    time_series_values = interpolate(
        scalar_data=functional_data,
        vertices_vox=vertices_vox,
        interp_method="linear"
    )  # shape: (n_vertices, n_timepoints)
