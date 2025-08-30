imagetools module
=================

.. automodule:: clabtoolkit.imagetools
   :members:
   :undoc-members:
   :show-inheritance:

The imagetools module provides advanced neuroimaging operations, morphological processing, and image manipulation capabilities specifically designed for brain imaging data.

Key Features
------------
- 2D/3D morphological operations on binary images
- Volume filtering and hole filling algorithms  
- Image resampling and geometric transformations
- Quality control and preprocessing utilities
- Multi-modal image processing support
- Structuring element generation for morphological operations

Main Classes
------------

MorphologicalOperations
~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive class for binary image morphological processing.

Key Methods:
- ``erosion()``: Erode binary structures
- ``dilation()``: Dilate binary structures  
- ``opening()``: Morphological opening (erosion + dilation)
- ``closing()``: Morphological closing (dilation + erosion)
- ``fill_holes()``: Fill holes in binary structures
- ``remove_small_objects()``: Filter small connected components

Common Usage Examples
---------------------

Binary image processing::

    from clabtoolkit.imagetools import MorphologicalOperations
    import nibabel as nib
    
    # Load binary image
    img = nib.load("/path/to/binary_mask.nii.gz")
    binary_data = img.get_fdata().astype(bool)
    
    # Initialize morphological operations
    morph = MorphologicalOperations()
    
    # Perform morphological closing
    structuring_element = morph.create_spherical_kernel(radius=2)
    closed_image = morph.closing(binary_data, structuring_element)
    
    # Fill holes in the image
    filled_image = morph.fill_holes(closed_image)

Advanced morphological processing::

    # Create custom structuring element
    kernel = morph.create_cubic_kernel(size=3)
    
    # Chain operations
    processed = morph.opening(binary_data, kernel)
    processed = morph.remove_small_objects(processed, min_size=100)
    
    # Save result
    output_img = nib.Nifti1Image(processed.astype(np.uint8), img.affine, img.header)
    nib.save(output_img, "/path/to/processed_mask.nii.gz")

Image quality control::

    # Validate image properties
    if morph.validate_binary_image(binary_data):
        print("Image is valid binary format")
    
    # Get image statistics
    stats = morph.get_image_statistics(binary_data)
    print(f"Volume: {stats['volume']} voxels")