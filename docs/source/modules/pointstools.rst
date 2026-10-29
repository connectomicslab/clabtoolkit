pointstools module
==================

The pointstools module provides point cloud representation and manipulation, with per-point scalar data, spatial filtering, affine transformation, conversion to surfaces and 3D visualization.

Key Features
------------
- Point cloud construction from coordinate arrays or files
- Per-point scalar data with named maps
- Affine transformation and spatial (bounding box) filtering
- Colortable support for categorical point data
- Conversion to pandas DataFrames and multi-format saving
- Merging of multiple point clouds
- Conversion to a surface of spheres, transferring the maps as parcellations or vertex-wise maps
- 3D rendering via PyVista

API reference
-------------

.. automodule:: clabtoolkit.pointstools
   :members:
   :undoc-members:
   :show-inheritance:

Main Classes
------------

PointsCloud
~~~~~~~~~~~
The core class for representing and manipulating point clouds.

Key Methods:
- ``load()``: Load a point cloud from a file (class method)
- ``save()``: Save the point cloud to a file
- ``copy()``: Create a deep copy of the point cloud
- ``add_point_data()``: Attach named scalar data to each point
- ``apply_affine()``: Apply an affine transformation to the coordinates
- ``filter_by_bounds()``: Filter points by spatial bounds
- ``filter()``: Filter points by coordinate values or point data attributes
- ``get_bounds()``: Compute the bounding box
- ``get_centroid()``: Compute the geometric center
- ``append()``: Append another PointsCloud to this one
- ``load_colortable()``: Attach a colortable to a named map
- ``get_pointwise_colors()``: Compute per-point colors for an overlay
- ``list_maps()``: List the available scalar maps
- ``to_dataframe()``: Convert the point cloud to a pandas DataFrame
- ``to_surface()``: Convert the point cloud to a surface with one sphere per point, transferring its maps
- ``get_info()``: Print a comprehensive summary
- ``plot()``: Render the point cloud with an optional overlay

Main Functions
--------------
- ``merge_pointsclouds()``: Merge multiple point clouds into one
- ``smooth_curve_coordinates()``: Smooth a 3D curve with Gaussian-weighted neighborhood averaging

Common Usage Examples
---------------------

Basic construction and inspection::

    import numpy as np
    from clabtoolkit.pointstools import PointsCloud

    # Build a point cloud from an Nx3 coordinate array
    coords = np.random.rand(100, 3) * 50
    pc = PointsCloud(points=coords, name="seeds")

    # Print a full summary
    pc.get_info()

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
    pc.apply_affine(affine, inplace=True)

Colortables and visualization::

    # Attach a colortable to a categorical map and render
    pc.load_colortable("/path/to/lookup_table.lut", map_name="region")
    pc.plot(maps="fa", cmap="viridis")

Conversion to a surface::

    # One triangular sphere (162 vertices by default) centered on each point.
    # The radius can be a single value or one value per point.
    surf = pc.to_surface(radius=2.0)

    # Maps with a colortable (e.g. the regions) are stored as a parcellation, so all
    # the vertices of a sphere get the color of its point. Maps without a colortable
    # (e.g. "fa") are stored as vertex-wise maps.
    surf.plot(maps="default", cmap=None)

    # Get the parcellation of the maps with a colortable as AnnotParcellation objects
    surf, annots = pc.to_surface(maps="default", return_annotations=True)
    annots["default"].save_annotation("/path/to/lh.points.annot")

Export and merging::

    from clabtoolkit.pointstools import merge_pointsclouds

    # Convert to a DataFrame for downstream analysis
    df = pc.to_dataframe(include_data=True)

    # Save to file
    pc.save("/path/to/points.npy", format="npy")

    # Reload later
    pc_reloaded = PointsCloud.load("/path/to/points.npy", format="npy")

    # Merge several point clouds
    merged = merge_pointsclouds([pc_a, pc_b])

    # Or append in place
    pc_a.append(pc_b, inplace=True)
