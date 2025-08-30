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
- Scalar data overlay and visualization
- Parcellation integration and display
- Interactive 3D plotting with PyVista
- Surface-based analysis tools
- Publication-quality visualization

Main Classes
------------

Surface
~~~~~~~
The primary class for surface mesh processing and visualization.

Key Methods:
- ``load_from_file()``: Load FreeSurfer surface files
- ``load_scalar_data()``: Load and attach scalar data to surface
- ``load_parcellation()``: Load parcellation overlays
- ``plot()``: Interactive 3D visualization with customization options
- ``get_mesh_info()``: Extract surface geometry information

Common Usage Examples
---------------------

Basic surface visualization::

    from clabtoolkit.surfacetools import Surface
    
    # Load surface
    surface = Surface("/path/to/lh.pial")
    
    # Simple surface plot
    surface.plot()
    
    # Load and visualize scalar data
    surface.load_scalar_data("/path/to/lh.thickness.mgh")
    surface.plot(scalar_map=True, colormap='viridis')

Advanced visualization with parcellation::

    # Load surface with parcellation overlay
    surface = Surface("/path/to/lh.pial")
    surface.load_parcellation("/path/to/lh.aparc.annot")
    
    # Plot with parcellation boundaries
    surface.plot(
        show_parcellation=True,
        background_color='black',
        lighting=True
    )

Multi-view visualization::

    # Create multiple views of the same surface
    surface.plot(
        views=['lateral', 'medial', 'dorsal'],
        scalar_map=True,
        save_figure="/path/to/output.png"
    )