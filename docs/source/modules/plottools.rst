plottools module
================

.. automodule:: clabtoolkit.plottools
   :members:
   :undoc-members:
   :show-inheritance:

The plottools module provides low-level plotting infrastructure and layout calculation utilities that support visualization across the clabtoolkit package.

Key Features
------------
- Dynamic subplot grid calculation
- Screen size detection and multi-monitor support
- Optimal layout computation for complex visualizations
- Monitor resolution detection
- Flexible plotting utilities

Main Functions
--------------

Layout Calculation
~~~~~~~~~~~~~~~~~~
- ``calculate_optimal_subplots_grid()``: Compute optimal subplot arrangements
- ``calculate_subplot_layout()``: Calculate subplot layout parameters
- ``create_proportional_subplots()``: Create proportional subplot grids
- ``get_screen_size()``: Detect screen dimensions for optimal figure sizing
- ``get_current_monitor_size()``: Get current monitor dimensions

Common Usage Examples
---------------------

Dynamic subplot layout::

    from clabtoolkit.plottools import calculate_optimal_subplots_grid
    import matplotlib.pyplot as plt
    
    # Calculate optimal grid for 8 subplots
    rows, cols = calculate_optimal_subplots_grid(8)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    
    # Use calculated layout
    for i, ax in enumerate(axes.flat):
        if i < 8:  # Only plot actual data
            ax.plot([1, 2, 3], [1, 4, 2])
        else:  # Hide unused subplots
            ax.set_visible(False)

Screen-aware plotting::

    # Get screen dimensions for optimal figure sizing
    screen_width, screen_height = get_screen_size()
    
    # Calculate figure size as proportion of screen
    fig_width = screen_width * 0.8 / 100  # 80% of screen width in inches
    fig_height = screen_height * 0.6 / 100  # 60% of screen height in inches
    
    plt.figure(figsize=(fig_width, fig_height))

Proportional subplot creation::

    # Create proportional subplots based on data
    proportions = [2, 1, 3]  # Relative sizes
    fig, axes = create_proportional_subplots(
        proportions=proportions,
        orientation='horizontal',
        figsize=(12, 6)
    )
    
    # Get current monitor size for optimal display
    monitor_width, monitor_height = get_current_monitor_size()
    print(f"Monitor: {monitor_width}x{monitor_height}")
    
    # Calculate layout for multiple plots
    layout = calculate_subplot_layout(
        n_plots=8,
        aspect_ratio=1.5
    )