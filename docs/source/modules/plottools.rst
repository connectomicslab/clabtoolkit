plottools module
================

The plottools module contains the layout, screen and font helpers used by the plotting classes of clabtoolkit (for example ``BrainPlotter``). They can also be used on their own to size and arrange Matplotlib or PyVista figures.

Key Features
------------
- Screen and monitor size detection, and DPI estimation
- Subplot grid computation that matches the screen proportions
- Ready-to-use figures with proportional subplots
- Font sizes scaled to the plot size, the monitor and the colorbar layout
- Random points inside a sphere, used to simulate region coordinates

API reference
-------------

.. automodule:: clabtoolkit.plottools
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
--------------

Screen
~~~~~~
- ``get_screen_size()``, ``get_current_monitor_size()``: size in pixels
- ``estimate_monitor_dpi(screen_width, screen_height)``: DPI estimated from the resolution

Layout
~~~~~~
- ``calculate_optimal_subplots_grid(num_views)``: grid dimensions and cell positions
- ``calculate_subplot_layout(n_plots, screen_width=None, screen_height=None, target_aspect_ratio=None)``: rows, columns and the aspect ratio used
- ``create_proportional_subplots(n_plots, figsize_base=4, **layout_kwargs)``: figure, axes and layout information

Fonts
~~~~~
- ``calculate_font_sizes(plot_width, plot_height, ...)``: font sizes (title, axis labels, ticks, colorbar) for Matplotlib
- ``get_pyvista_fonts(plot_width, plot_height, ...)``: the same sizes as integers for PyVista

Utilities
~~~~~~~~~
- ``generate_spherical_coords(n_regions, rng, radius=50.0)``: random points inside a sphere

Common Usage Examples
---------------------

Subplot layout::

    from clabtoolkit.plottools import calculate_optimal_subplots_grid, calculate_subplot_layout

    # Grid for 8 views: [rows, cols] and the (row, col) position of every view
    (n_rows, n_cols), positions = calculate_optimal_subplots_grid(8)  # [2, 4]

    # Layout that follows the proportions of a 1920x1080 screen
    n_rows, n_cols, aspect = calculate_subplot_layout(8, screen_width=1920, screen_height=1080)
    # 2, 4, 1.78

Figures with proportional subplots::

    from clabtoolkit.plottools import create_proportional_subplots

    fig, axes, layout = create_proportional_subplots(8, figsize_base=3)

    for i, ax in enumerate(axes.flat):
        if i < 8:
            ax.plot([1, 2, 3], [1, 4, 2])
        else:
            ax.set_visible(False)  # hide the unused cells

Screen-aware sizes and fonts::

    from clabtoolkit.plottools import get_screen_size, calculate_font_sizes, get_pyvista_fonts

    screen_width, screen_height = get_screen_size()

    fonts = calculate_font_sizes(800, 600, screen_width=1920, screen_height=1080,
                                 auto_detect_monitor=False)
    print(fonts["title"], fonts["tick_labels"])  # 28.0 20.0

    # Integer sizes for PyVista
    pv_fonts = get_pyvista_fonts(800, 600, screen_width=1920, screen_height=1080,
                                 auto_detect_monitor=False)
