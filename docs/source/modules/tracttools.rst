tracttools module
=================

.. automodule:: clabtoolkit.tracttools
   :members:
   :undoc-members:
   :show-inheritance:

The tracttools module provides tractogram loading, manipulation, clustering and visualization, with support for the TRK and TCK formats and per-point or per-streamline scalar maps.

Key Features
------------
- Load tractograms from file or from nibabel Tractogram objects
- Per-point and per-streamline scalar maps, with conversion between the two
- Streamline resampling, smoothing and reduction
- Bundle clustering with QuickBundles/QuickBundlesX and centroid extraction
- Scalar interpolation from NIfTI volumes onto streamlines
- Colortable support and 3D rendering via PyVista
- TRK/TCK format conversion

Main Classes
------------

Tractogram
~~~~~~~~~~
The core class for managing tractograms, their streamlines and associated scalar maps.

Key Methods:
- ``load_tractogram_from_file()``: Load a tractogram from a .trk or .tck file
- ``load_tractogram_from_object()``: Load from an existing nibabel Tractogram object
- ``load_colortable()``: Attach a colortable to a named map
- ``resample_streamlines()``: Resample all streamlines to a fixed number of points
- ``smooth_streamlines()``: Smooth streamlines with a Gaussian filter
- ``reduce_streamlines()``: Randomly reduce the streamline count by a percentage
- ``filter()``: Filter streamlines by length
- ``compute_streamline_lengths()``: Compute the length of every streamline
- ``compute_centroids()``: Cluster streamlines and extract bundle centroids
- ``label_streamlines_by_clusters()``: Label each streamline with its cluster ID
- ``get_cluster_streamlines()``: Retrieve the streamlines of one cluster
- ``centroids_to_tractogram()``: Convert computed centroids into a new Tractogram
- ``interpolate_on_tractogram()``: Interpolate a NIfTI scalar map onto the streamlines
- ``streamline_to_points()``: Expand a per-streamline map to a per-point map
- ``points_to_streamline()``: Reduce a per-point map to a per-streamline map
- ``add_tractogram()``: Merge one or more Tractogram objects into this one
- ``list_maps()``: List the available scalar maps
- ``get_maps_info()``: Retrieve information about the stored scalar maps
- ``get_tractogram_info()``: Retrieve basic tractogram information
- ``explore_tractogram()``: Print a comprehensive summary of the tractogram
- ``save_tractogram()``: Save the tractogram to .trk or .tck
- ``plot()``: Render the tractogram with an optional scalar overlay

Main Functions
--------------

Format Conversion
~~~~~~~~~~~~~~~~~
- ``trk2tck()``: Convert a .trk tractogram to .tck format
- ``tck2trk()``: Convert a .tck tractogram to .trk format using a reference image

Streamline Utilities
~~~~~~~~~~~~~~~~~~~~
- ``resample_streamlines()``: Resample a raw streamline collection to a fixed number of points
- ``interpolate_streamline_values()``: Interpolate values between streamline coordinates
- ``merge_tractograms()``: Merge multiple tractograms into a single tractogram

Common Usage Examples
---------------------

Basic loading and inspection::

    from clabtoolkit.tracttools import Tractogram

    # Load a tractogram from file
    tract = Tractogram("/path/to/tractogram.trk")

    # Print a full summary of streamlines, maps and geometry
    tract.explore_tractogram()

    # Basic information and available scalar maps
    info = tract.get_tractogram_info()
    tract.list_maps()

    # Streamline lengths
    lengths = tract.compute_streamline_lengths()

Resampling and filtering::

    # Resample every streamline to a fixed number of points
    tract.resample_streamlines(nb_points=100)

    # Smooth streamline geometry
    tract.smooth_streamlines(iterations=5, sigma=1.0)

    # Keep only the longer streamlines
    tract.filter(condition="length > 50")

    # Randomly keep half of the streamlines
    tract.reduce_streamlines(percentage=50)

Interpolating scalar maps onto streamlines::

    # Interpolate an FA map onto each point of every streamline
    tract.interpolate_on_tractogram(
        "/path/to/fa_map.nii.gz",
        map_name="fa",
        storage_mode="data_per_point",
        interp_method="linear"
    )

    # Store a single mean value per streamline instead
    tract.interpolate_on_tractogram(
        "/path/to/md_map.nii.gz",
        map_name="md",
        storage_mode="data_per_streamline",
        reduction="mean"
    )

    # Convert between per-point and per-streamline representations
    tract.points_to_streamline(map_name="fa", streamline_map_name="fa_mean", metric="mean")

Bundle clustering and centroids::

    # Cluster streamlines with QuickBundles and extract centroids
    centroids, indices = tract.compute_centroids(method="qb", thresholds=[10])

    # Multi-threshold QuickBundlesX
    centroids, indices = tract.compute_centroids(method="qbx", thresholds=[30, 20, 10])

    # Label streamlines by cluster and pull one bundle out
    tract.label_streamlines_by_clusters()
    bundle = tract.get_cluster_streamlines(cluster_id=0)

    # Turn the centroids into a tractogram of their own
    centroid_tract = tract.centroids_to_tractogram()

Visualization and saving::

    # Attach a colortable and render with an overlay
    tract.load_colortable("/path/to/lookup_table.lut", map_name="fa")
    tract.plot(
        overlay_name="fa",
        cmap="hot",
        views=["lateral"],
        plot_style="tube",
        show_colorbar=True,
        colorbar_title="FA"
    )

    # Save in either supported format
    tract.save_tractogram("/path/to/output.trk", file_type="trk", overwrite=True)

Merging and format conversion::

    from clabtoolkit.tracttools import merge_tractograms, trk2tck, tck2trk

    # Merge several tractograms, tagging each with a tract_id map
    merged = merge_tractograms(
        [tract_a, tract_b],
        color_table="/path/to/bundles.lut",
        map_name="tract_id"
    )

    # Or merge in place
    tract_a.add_tractogram(tract_b)

    # Convert between tractography formats
    trk2tck("/path/to/tractogram.trk", "/path/to/tractogram.tck", overwrite=True)
    tck2trk("/path/to/tractogram.tck", "/path/to/reference.nii.gz", "/path/to/out.trk")
