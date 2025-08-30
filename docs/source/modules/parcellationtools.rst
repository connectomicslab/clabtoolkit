parcellationtools module
=======================

.. automodule:: clabtoolkit.parcellationtools
   :members:
   :undoc-members:
   :show-inheritance:

The parcellationtools module provides comprehensive brain parcellation handling, regional analysis, and atlas-based processing capabilities.

Key Features
------------
- Load parcellations with associated lookup tables
- Regional filtering and grouping operations
- Volume calculations and statistical analysis
- Multi-format support (NIfTI, TSV, LUT)
- Parcellation validation and correction
- BIDS-compliant output generation

Main Classes
------------

Parcellation
~~~~~~~~~~~~
The core class for managing brain parcellations and their associated metadata.

Key Methods:
- ``load_from_file()``: Load parcellation with lookup table
- ``filter_regions()``: Filter parcellation by region criteria
- ``group_regions()``: Group regions into larger anatomical units
- ``compute_regional_volumes()``: Calculate volumes for each region
- ``save_to_file()``: Export parcellation in various formats
- ``validate()``: Check parcellation integrity

Common Usage Examples
---------------------

Basic parcellation loading and analysis::

    from clabtoolkit.parcellationtools import Parcellation
    
    # Load parcellation with lookup table
    parc = Parcellation()
    parc.load_from_file(
        "/path/to/parcellation.nii.gz",
        "/path/to/lookup_table.lut"
    )
    
    # Get basic information
    print(f"Number of regions: {len(parc.regions)}")
    print(f"Volume dimensions: {parc.shape}")

Regional analysis::

    # Filter specific anatomical regions
    parc.filter_regions(['cortex', 'cerebellum'])
    
    # Group regions by lobes
    parc.group_regions(grouping_file="/path/to/lobe_mapping.json")
    
    # Compute regional volumes
    volumes = parc.compute_regional_volumes()
    volumes_df = pd.DataFrame(volumes)

Export and conversion::

    # Save filtered parcellation
    parc.save_to_file("/path/to/output_parcellation.nii.gz")
    
    # Export lookup table
    parc.export_lut("/path/to/output_lut.txt")