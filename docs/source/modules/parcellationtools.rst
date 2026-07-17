parcellationtools module
========================

.. automodule:: clabtoolkit.parcellationtools
   :members:
   :undoc-members:
   :show-inheritance:

The parcellationtools module provides comprehensive brain parcellation handling, regional analysis, and atlas-based processing capabilities.

Key Features
------------
- Load parcellations with associated colour lookup tables
- Regional filtering by region name or region code
- Grouping of regions into larger anatomical units
- Volume and morphometry table computation
- Region adjacency, centroids and regionwise time series
- Multi-format support (NIfTI, TSV, LUT)
- Parcellation harmonization, rearranging and value replacement

Main Classes
------------

Parcellation
~~~~~~~~~~~~
The core class for managing brain parcellations and their associated metadata. A parcellation is normally constructed directly from a file, optionally with a colour table.

Accessors:
- ``get_data()`` / ``set_data()``: Parcellation voxel data
- ``get_affine()`` / ``set_affine()``: Affine transformation
- ``get_index()`` / ``set_index()``: Region codes
- ``get_names()`` / ``set_names()``: Region names
- ``get_colors()`` / ``set_colors()``: Region colours
- ``get_space_id()`` / ``set_space_id()``: Space identifier
- ``get_parcellation_id()``: Parcellation identifier
- ``get_info()``: Summary information

Region Selection:
- ``keep_by_name()``: Keep only the named regions
- ``keep_by_code()``: Keep only the regions with the given codes
- ``remove_by_name()``: Remove the named regions
- ``remove_by_code()``: Remove the regions with the given codes
- ``group_by_names()``: Group regions into larger units by name
- ``group_by_codes()``: Group regions into larger units by code
- ``apply_mask()`` / ``mask_image()``: Restrict the parcellation with a mask

Analysis:
- ``compute_volume_table()``: Compute a table of regional volumes
- ``compute_morphometry_table()``: Compute a regional morphometry table
- ``compute_centroids()``: Compute the centroid of each region
- ``compute_region_adjacency()``: Compute the region adjacency matrix
- ``get_regionwise_timeseries()``: Extract regionwise time series from 4D data
- ``compute_fc_matrix()``: Compute a functional connectivity matrix
- ``surface_extraction()``: Extract surfaces from the parcellation
- ``prepare_for_tracking()``: Prepare the parcellation for tractography

Colour Tables and I/O:
- ``load_colortable()``: Load and attach a colour lookup table
- ``export_colortable()``: Export the colour table
- ``save_parcellation()``: Save the parcellation to file
- ``export_summary_to_hdf5()``: Export a summary to HDF5

Maintenance:
- ``adjust_values()``: Adjust the parcellation values
- ``replace_values()``: Replace specific values
- ``sort_index()``: Sort the region index
- ``rearrange()``: Rearrange the region codes
- ``harmonize()``: Harmonize the parcellation against a reference
- ``add_parcellation()``: Merge another parcellation into this one
- ``parc_range()``: Report the range of parcellation values
- ``print_properties()``: Print the parcellation properties

RegionTimeSeries
~~~~~~~~~~~~~~~~
Holds regionwise time series extracted from a parcellation.

Key Methods:
- ``compute_fc_matrix()``: Compute a functional connectivity matrix
- ``get_info()``: Summary information
- ``show_content()``: Display the stored content

Common Usage Examples
---------------------

Basic parcellation loading and analysis::

    from clabtoolkit.parcellationtools import Parcellation

    # Load a parcellation, optionally with its lookup table
    parc = Parcellation(
        parc_file="/path/to/parcellation.nii.gz",
        color_table="/path/to/lookup_table.lut"
    )

    # A colour table can also be attached afterwards
    parc.load_colortable("/path/to/lookup_table.lut")

    # Inspect the parcellation
    parc.print_properties()
    print(f"Number of regions: {len(parc.get_index())}")
    print(f"Volume dimensions: {parc.get_data().shape}")

Regional selection and grouping::

    # Keep or remove regions by name or by code
    parc.keep_by_name(["ctx-lh-precentral", "ctx-lh-postcentral"])
    parc.remove_by_code([0, 5], rearrange=True)

    # Group regions into larger anatomical units
    parc.group_by_names(
        group_dict={
            "frontal": ["ctx-lh-precentral", "ctx-lh-superiorfrontal"],
            "parietal": ["ctx-lh-postcentral", "ctx-lh-superiorparietal"],
        },
        keep_ungrouped=False
    )

    # The same grouping can be expressed with region codes
    parc.group_by_codes(group_dict={"frontal": [1024, 1028]})

    # Restrict the parcellation with a mask
    parc.apply_mask("/path/to/mask.nii.gz")

Regional analysis::

    # Compute regional volumes
    volume_table = parc.compute_volume_table()

    # Compute a full morphometry table
    morpho_table = parc.compute_morphometry_table()

    # Region geometry and adjacency
    centroids = parc.compute_centroids()
    adjacency = parc.compute_region_adjacency()

Functional connectivity::

    # Extract regionwise time series from a 4D image
    ts = parc.get_regionwise_timeseries("/path/to/bold.nii.gz")
    ts.get_info()

    # Compute the functional connectivity matrix
    fc_matrix = ts.compute_fc_matrix()

Export and conversion::

    # Save the parcellation
    parc.save_parcellation("/path/to/output_parcellation.nii.gz")

    # Export the colour table
    parc.export_colortable("/path/to/output_lut.lut")

    # Export a summary to HDF5
    parc.export_summary_to_hdf5("/path/to/summary.h5")
