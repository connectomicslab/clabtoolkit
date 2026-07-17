pointstools module
==================

.. automodule:: clabtoolkit.pointstools
   :members:
   :undoc-members:
   :show-inheritance:

The pointstools module provides point cloud representation and manipulation, with per-point scalar data, spatial filtering, affine transformation and 3D visualization.

Key Features
------------
- Point cloud construction from coordinate arrays or files
- Per-point scalar data with named maps
- Affine transformation and spatial (bounding box) filtering
- Colortable support for categorical point data
- Conversion to pandas DataFrames and multi-format saving
- Merging of multiple point clouds
- 3D rendering via PyVista

Main Classes
------------

PointCloud
~~~~~~~~~~
The core class for representing and manipulating point clouds.

Key Methods:
- ``load()``: Load a point cloud from a file (class method)
- ``save()``: Save the point cloud to a file
- ``copy()``: Create a deep copy of the point cloud
- ``add_point_data()``: Attach named scalar data to each point
- ``transform()``: Apply an affine transformation to the coordinates
- ``filter_by_bounds()``: Filter points by spatial bounds
- ``filter()``: Filter points by coordinate values or point data attributes
- ``get_bounds()``: Compute the bounding box
- ``get_centroid()``: Compute the geometric center
- ``append()``: Append another PointCloud to this one
- ``load_colortable()``: Attach a colortable to a named map
- ``get_pointwise_colors()``: Compute per-point colors for an overlay
- ``list_maps()``: List the available scalar maps
- ``to_dataframe()``: Convert the point cloud to a pandas DataFrame
- ``explore_pointcloud()``: Print a comprehensive summary
- ``plot()``: Render the point cloud with an optional overlay

Main Functions
--------------
- ``merge_pointclouds()``: Merge multiple point clouds into one
- ``smooth_curve_coordinates()``: Smooth a 3D curve with Gaussian-weighted neighborhood averaging

Common Usage Examples
---------------------

Basic construction and inspection::

    import numpy as np
    from clabtoolkit.pointstools import PointCloud

    # Build a point cloud from an Nx3 coordinate array
    coords = np.random.rand(100, 3) * 50
    pc = PointCloud(points=coords, name="seeds")

    # Print a full summary
    pc.explore_pointcloud()

    # Geometry
    bounds = pc.get_bounds()
    centroid = pc.get_centroid()

Attaching and using scalar data::

    # Attach a named scalar map to the points
    pc.add_point_data(np.random.rand(100), name="fa")

    # List available maps
    pc.list_maps()

    # Filter by a point data attribute or by coordinates
    high_fa = pc.filter("fa > 0.5", inplace=False)

Spatial operations::

    # Restrict to a bounding box
    pc.filter_by_bounds(x_range=(0, 25), y_range=(0, 25), z_range=None)

    # Apply an affine transformation
    affine = np.eye(4)
    pc.transform(affine, inplace=True)

Colortables and visualization::

    # Attach a colortable to a categorical map and render
    pc.load_colortable("/path/to/lookup_table.lut", map_name="region")
    pc.plot(overlay_name="fa", cmap="viridis")

Export and merging::

    from clabtoolkit.pointstools import merge_pointclouds

    # Convert to a DataFrame for downstream analysis
    df = pc.to_dataframe(include_data=True)

    # Save to file
    pc.save("/path/to/points.tsv")

    # Reload later
    pc_reloaded = PointCloud.load("/path/to/points.tsv")

    # Merge several point clouds
    merged = merge_pointclouds([pc_a, pc_b])

    # Or append in place
    pc_a.append(pc_b, inplace=True)
