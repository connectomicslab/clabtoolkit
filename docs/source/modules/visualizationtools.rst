visualizationtools module
=========================

.. automodule:: clabtoolkit.visualizationtools
   :members:
   :undoc-members:
   :show-inheritance:

The visualizationtools module provides publication-quality brain surface visualization with flexible layout configurations and advanced rendering capabilities.

Key Features
------------
- Multi-view brain surface visualization
- JSON-based view and layout configuration
- Named themes with persistent figure configuration
- Publication-ready figure generation
- Flexible colormap and colorbar support
- PyVista-powered 3D rendering
- Carpet plots for time series data

Main Classes
------------

BrainPlotter
~~~~~~~~~~~~
A comprehensive brain surface visualization tool using PyVista. Configuration is supplied through a JSON config file rather than an inline dictionary.

Plotting:
- ``plot()``: Render one or more objects with the requested views
- ``plot_hemispheres()``: Render left and right hemispheres side by side
- ``plot_scene()``: Render a configured scene of multiple objects

Views and Layouts:
- ``list_available_view_names()``: List the view names that can be requested
- ``list_available_layouts()``: List the available layouts
- ``get_layout_details()``: Inspect the layout used for a set of views

Configuration and Themes:
- ``get_figure_config()``: Retrieve the current figure configuration
- ``update_figure_config()``: Update the figure configuration
- ``list_figure_config_options()``: List the configurable options
- ``reset_figure_config()``: Restore the default figure configuration
- ``list_available_themes()``: List the available themes
- ``apply_theme()``: Apply a named theme
- ``preview_theme()``: Preview a theme before applying it
- ``save_config()``: Persist the current configuration

Main Functions
--------------
- ``create_carpet_plot()``: Create a carpet plot from time series data

Common Usage Examples
---------------------

Basic surface visualization::

    import clabtoolkit.surfacetools as cltsurf
    from clabtoolkit.visualizationtools import BrainPlotter

    # Load a surface and attach a scalar overlay
    surf_lh = cltsurf.Surface("/path/to/lh.pial")
    surf_lh.load_scalar_maps("/path/to/lh.thickness", maps_names="Thickness")

    # Render it through the plotter
    plotter = BrainPlotter()
    plotter.plot(
        objs2plot=surf_lh,
        hemi_id="lh",
        views="lateral",
        map_names=["Thickness"],
        colormaps="viridis",
        colorbar=True
    )

.. note::
   For a single surface, ``Surface.plot()`` in :doc:`surfacetools` is the shorter
   route. ``BrainPlotter`` is intended for multi-object and multi-view scenes.

Multi-view layouts::

    # Discover what views and layouts are available
    plotter.list_available_view_names()
    plotter.list_available_layouts()

    # Render several views at once
    plotter.plot(
        objs2plot=surf_lh,
        views=["lateral", "medial", "dorsal"],
        views_orientation="horizontal",
        map_names=["Thickness"]
    )

Both hemispheres together::

    surf_rh = cltsurf.Surface("/path/to/rh.pial")
    surf_rh.load_scalar_maps("/path/to/rh.thickness", maps_names="Thickness")

    plotter.plot_hemispheres(
        obj_rh=surf_rh,
        obj_lh=surf_lh,
        map_name="Thickness",
        views="lateral",
        colormap="viridis",
        colorbar=True,
        colorbar_title="Thickness (mm)"
    )

Themes and configuration::

    # A custom configuration is supplied as a JSON file
    plotter = BrainPlotter(config_file="/path/to/view_config.json")

    # Inspect and apply themes
    plotter.list_available_themes()
    plotter.preview_theme("dark")
    plotter.apply_theme("dark", auto_save=True)

    # Inspect and change the figure configuration
    plotter.list_figure_config_options()
    plotter.update_figure_config(auto_save=True)
    plotter.reset_figure_config()

Publication-ready figures::

    # Render a scene and write it straight to disk
    plotter.plot_scene(
        scene_objects=[surf_lh, surf_rh],
        views=["lateral", "medial"],
        colorbar=True,
        colorbar_position="right",
        save_path="/path/to/publication_figure.png"
    )

Carpet plots::

    from clabtoolkit.visualizationtools import create_carpet_plot

    # Visualize regionwise time series as a carpet plot
    create_carpet_plot(data=timeseries_array, structure_names=region_names)
