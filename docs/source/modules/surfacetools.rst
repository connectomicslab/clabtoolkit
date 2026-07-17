surfacetools module
===================

.. automodule:: clabtoolkit.surfacetools
   :members:
   :undoc-members:
   :show-inheritance:

The surfacetools module provides advanced brain surface mesh processing and visualization capabilities using PyVista for 3D rendering and FreeSurfer surface format support.

Key Features
------------
- Load FreeSurfer surface files (.pial, .white, .inflated, .sphere)
- Scalar map overlays and named overlay management
- Annotation (parcellation) integration and display
- Mesh geometry queries: vertices, faces, edges and normals
- Volume-to-surface projection of 3D/4D data
- Interactive 3D plotting with PyVista
- Export to FreeSurfer, OBJ and PyVista formats

Main Classes
------------

Surface
~~~~~~~
The primary class for surface mesh processing and visualization.

Loading:
- ``load_from_file()``: Load a FreeSurfer surface file
- ``load_from_arrays()``: Build a surface from vertex and face arrays
- ``load_from_mesh()``: Build a surface from an existing mesh
- ``is_loaded()``: Check whether surface data is loaded

Geometry:
- ``get_vertices()`` / ``get_faces()``: Access the mesh arrays
- ``get_edges()`` / ``get_boundary_edges()`` / ``get_manifold_edges()``: Edge queries
- ``compute_normals()`` / ``get_normals()``: Surface normals
- ``separate_mesh_components()``: Split the mesh into connected components
- ``add_surface()``: Merge another surface into this one

Overlays and Annotations:
- ``load_scalar_maps()``: Load and attach scalar data to the surface
- ``load_annotation()``: Load a parcellation annotation overlay
- ``list_overlays()``: List the attached overlays
- ``set_active_overlay()`` / ``remove_overlay()``: Manage the active overlay
- ``get_overlay_info()``: Inspect an overlay
- ``get_region_vertices()``: Get the vertices of a named region
- ``get_vertexwise_colors()`` / ``prepare_colors()``: Compute per-vertex colours
- ``map_volume_to_surface()``: Project 3D/4D volumetric data onto the mesh

Visualization and Export:
- ``plot()``: Interactive 3D visualization
- ``save_surface()``: Save the surface
- ``export_to_freesurfer()`` / ``export_to_obj()`` / ``export_to_pyvista()``: Format-specific export
- ``export_annotation()``: Export an annotation

Main Functions
--------------
- ``merge_surfaces()``: Merge multiple surfaces into one
- ``create_surface_colortable()``: Build a colour table for surface regions

Common Usage Examples
---------------------

Basic surface visualization::

    from clabtoolkit.surfacetools import Surface

    # Load a surface
    surface = Surface("/path/to/lh.pial")

    # Simple surface plot
    surface.plot()

    # Load scalar data and plot it as an overlay
    surface.load_scalar_maps("/path/to/lh.thickness", maps_names="Thickness")
    surface.plot(overlay_name="Thickness", cmap="viridis")

Working with annotations::

    # Load a surface with a parcellation annotation
    surface = Surface("/path/to/lh.pial")
    surface.load_annotation("/path/to/lh.aparc.annot", parc_name="aparc")

    # Plot the parcellation
    surface.plot(overlay_name="aparc")

    # Inspect the available overlays and pull one region out
    surface.list_overlays()
    vertices = surface.get_region_vertices(parc_name="aparc", region_name="precentral")

Multi-view visualization::

    # Create multiple views of the same surface
    surface.plot(
        overlay_name="Thickness",
        views=["lateral", "medial", "dorsal"],
        views_orientation="grid",
        cmap="jet",
        save_path="/path/to/output.png"
    )

Mesh geometry::

    # Access the underlying mesh arrays
    vertices = surface.get_vertices()
    faces = surface.get_faces()

    # Surface normals and edge queries
    surface.compute_normals()
    normals = surface.get_normals()
    boundary = surface.get_boundary_edges()

Volume-to-surface mapping::

    # Project a 3D structural image onto the surface
    surface = Surface("/path/to/lh.pial")
    surface_values = surface.map_volume_to_surface(
        image="/path/to/structural.nii.gz",
        method="nilearn",
        interp_method="linear"
    )

    # Project 4D functional data, storing it as a named overlay
    functional_values = surface.map_volume_to_surface(
        image="/path/to/functional_4d.nii.gz",
        method="custom",
        interp_method="linear",
        overlay_name="activation"
    )

Export::

    # Save in FreeSurfer format, carrying an annotation along
    surface.save_surface(
        "/path/to/output/lh.pial",
        format="freesurfer",
        save_annotation="aparc"
    )

    # Or export to other formats
    surface.export_to_obj("/path/to/output/lh.obj", overwrite=True)
    surface.export_annotation("/path/to/output/lh.aparc.annot", parc_name="aparc")
