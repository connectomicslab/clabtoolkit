visualizationtools module
========================

.. automodule:: clabtoolkit.visualizationtools
   :members:
   :undoc-members:
   :show-inheritance:

The visualizationtools module provides publication-quality brain surface visualization with flexible layout configurations and advanced rendering capabilities.

Key Features
------------
- Multi-view brain surface visualization
- JSON-based view configuration system
- Publication-ready figure generation
- Flexible colormap and annotation support
- PyVista-powered 3D rendering
- Customizable layouts and camera positions

Main Classes
------------

SurfacePlotter
~~~~~~~~~~~~~~
Advanced brain surface visualization with multi-view layouts.

Key Methods:
- ``plot_surface()``: Render single surface with scalar overlays
- ``plot_multi_view()``: Create multi-view layouts
- ``configure_views()``: Set up custom view configurations
- ``save_figure()``: Export high-resolution figures
- ``add_colorbar()``: Add publication-quality colorbars

Common Usage Examples
---------------------

Basic surface visualization::

    from clabtoolkit.visualizationtools import SurfacePlotter
    
    # Initialize plotter with default configuration
    plotter = SurfacePlotter()
    
    # Plot surface with scalar overlay
    plotter.plot_surface(
        surface_file="lh.pial",
        scalar_file="lh.thickness.mgh",
        colormap="viridis",
        background_color="white"
    )

Multi-view layouts::

    # Create custom view configuration
    view_config = {
        "views": [
            {"name": "lateral", "position": [1, 0, 0]},
            {"name": "medial", "position": [-1, 0, 0]},
            {"name": "dorsal", "position": [0, 0, 1]}
        ],
        "layout": {"rows": 1, "cols": 3}
    }
    
    # Plot with multiple views
    plotter = SurfacePlotter(config=view_config)
    plotter.plot_multi_view(
        surface_files=["lh.pial", "rh.pial"],
        scalar_files=["lh.thickness.mgh", "rh.thickness.mgh"]
    )

Publication-ready figures::

    # Generate high-quality publication figure
    plotter.plot_surface(
        surface_file="lh.inflated",
        scalar_file="lh.curvature.mgh", 
        save_figure="/path/to/publication_figure.png",
        dpi=300,
        figure_size=(12, 8),
        show_colorbar=True,
        colorbar_title="Curvature (mm⁻¹)"
    )