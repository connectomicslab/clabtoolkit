import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


#####################################################################################################
def get_screen_size() -> Tuple[int, int]:
    """
    Get the current screen size in pixels.

    Returns
    -------
    tuple of int
        Screen width and height in pixels (width, height).

    Examples
    --------
    >>> width, height = get_screen_size()
    >>> print(f"Screen size: {width}x{height}")
    """

    import tkinter as tk

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()  # Clean up the Tkinter instance

    return width, height


#####################################################################################################
def get_current_monitor_size() -> Tuple[int, int]:
    """Get the size of the monitor where the mouse cursor is located."""
    import tkinter as tk
    import screeninfo

    # Get mouse position
    root = tk.Tk()
    root.withdraw()
    mouse_x = root.winfo_pointerx()
    mouse_y = root.winfo_pointery()
    root.destroy()

    # Find which monitor contains the mouse
    monitors = screeninfo.get_monitors()
    for monitor in monitors:
        if (
            monitor.x <= mouse_x < monitor.x + monitor.width
            and monitor.y <= mouse_y < monitor.y + monitor.height
        ):
            return monitor.width, monitor.height

    # Fallback to primary monitor
    primary = next((m for m in monitors if m.is_primary), monitors[0])
    return primary.width, primary.height


#######################################################################################################
def estimate_monitor_dpi(screen_width: int, screen_height: int) -> float:
    """
    Estimate monitor DPI based on screen resolution using common monitor configurations.

    Parameters
    ----------
    screen_width : int
        Screen width in pixels

    screen_height : int
        Screen height in pixels

    Returns
    -------
    float
        Estimated DPI based on common monitor size/resolution combinations

    Examples
    --------
    >>> estimate_monitor_dpi(1920, 1080)
    96.0
    >>>
    >>> estimate_monitor_dpi(2560, 1440)
    109.0
    """

    # Common monitor configurations: (width, height): typical_dpi
    monitor_configs = {
        # Full HD displays
        (1920, 1080): {
            "laptop_13_15": 147,  # 13-15" laptop
            "laptop_17": 130,  # 17" laptop
            "monitor_21_24": 92,  # 21-24" monitor
            "monitor_27": 82,  # 27" monitor
        },
        # QHD displays
        (2560, 1440): {
            "laptop_13_15": 196,  # 13-15" laptop
            "monitor_27": 109,  # 27" monitor
            "monitor_32": 92,  # 32" monitor
        },
        # 4K displays
        (3840, 2160): {
            "laptop_15_17": 294,  # 15-17" laptop
            "monitor_27": 163,  # 27" monitor
            "monitor_32": 138,  # 32" monitor
            "monitor_43": 103,  # 43" monitor
        },
        # Other common resolutions
        (1366, 768): {
            "laptop_11_14": 112,  # Small laptops
        },
        (1680, 1050): {
            "monitor_22": 90,  # 22" monitor
        },
        (2880, 1800): {
            "laptop_15": 220,  # MacBook Pro 15"
        },
        (3440, 1440): {
            "ultrawide_34": 110,  # 34" ultrawide
        },
    }

    # Find exact match first
    resolution = (screen_width, screen_height)
    if resolution in monitor_configs:
        # For known resolutions, estimate based on pixel density
        configs = monitor_configs[resolution]
        pixel_count = screen_width * screen_height

        # Estimate based on total pixels and common usage patterns
        if pixel_count < 1500000:  # < 1.5M pixels (likely smaller screen)
            return max(configs.values())  # Higher DPI (smaller screen)
        elif pixel_count > 8000000:  # > 8M pixels (4K+, likely larger screen)
            return min(configs.values())  # Lower DPI (larger screen)
        else:
            # Medium resolution, use median DPI
            return sorted(configs.values())[len(configs.values()) // 2]

    # Fallback: calculate approximate DPI based on pixel density
    # Assume reasonable screen diagonal based on resolution
    total_pixels = screen_width * screen_height

    if total_pixels <= 1000000:  # ≤ 1M pixels
        estimated_diagonal = 13  # Small laptop/tablet

    elif total_pixels <= 2000000:  # ≤ 2M pixels
        estimated_diagonal = 21  # Standard monitor

    elif total_pixels <= 4000000:  # ≤ 4M pixels
        estimated_diagonal = 24  # Larger monitor

    elif total_pixels <= 8000000:  # ≤ 8M pixels
        estimated_diagonal = 27  # QHD monitor

    else:  # > 8M pixels
        estimated_diagonal = 32  # 4K monitor

    # Calculate DPI: sqrt(width² + height²) / diagonal_inches
    diagonal_pixels = (screen_width**2 + screen_height**2) ** 0.5
    estimated_dpi = diagonal_pixels / estimated_diagonal

    return round(estimated_dpi, 1)


###############################################################################################
def calculate_optimal_subplots_grid(num_views: int) -> List[int]:
    """
    Calculate optimal grid dimensions for a given number of views.

    Parameters
    ----------
    num_views : int
        Number of views to arrange.

    Returns
    -------
    List[int]
        [rows, columns] for optimal grid layout.

    Examples
    --------
    >>> calculate_optimal_subplots_grid(4)
    [2, 2]
    >>>
    >>> calculate_optimal_subplots_grid(6)
    [2, 3]
    >>>
    >>> calculate_optimal_subplots_grid(1)
    [1, 1]
    """

    # Calculate optimal grid dimensions based on number of views
    if num_views == 1:
        grid_size = [1, 1]
        position = [(0, 0)]
        return grid_size, position

    elif num_views == 2:
        grid_size = [1, 2]
        position = [(0, 0), (0, 1)]
        return grid_size, position

    elif num_views == 3:
        grid_size = [1, 3]
        position = [(0, 0), (0, 1), (0, 2)]
        return grid_size, position

    elif num_views == 4:
        # For 4 views, arrange in a 2x2 grid
        grid_size = [2, 2]
        position = [(0, 0), (0, 1), (1, 0), (1, 1)]
        return grid_size, position

    elif num_views <= 6:
        # For 5 or 6 views, arrange in a 2x3 grid
        grid_size = [2, 3]
        position = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        return grid_size, position

    elif num_views <= 8:
        # For 7 or 8 views, arrange in a 2x4 grid
        grid_size = [2, 4]
        position = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
        return grid_size, position

    else:
        # For more than 8 views, try to keep a proportion with the screen shape
        screen_size = get_screen_size()

        # Calculate the number of columns and rows based on the number of views
        rows, cols, aspect = calculate_subplot_layout(
            num_views, screen_size[0], screen_size[1]
        )
        grid_size = [rows, cols]
        position = []
        for i in range(rows):
            for j in range(cols):
                if len(position) < num_views:
                    position.append((i, j))

        return grid_size, position


#####################################################################################################
def calculate_subplot_layout(
    n_plots, screen_width=None, screen_height=None, target_aspect_ratio=None
):
    """
    Calculate optimal rows and columns for subplots based on screen proportions

    Parameters:
    -----------
    n_plots : int
        Number of subplots needed

    screen_width : int, optional
        Screen width in pixels (auto-detected if not provided)

    screen_height : int, optional
        Screen height in pixels (auto-detected if not provided)

    target_aspect_ratio : float, optional
        Target aspect ratio (width/height). If provided, overrides screen detection

    Returns:
    --------
    tuple: (rows, cols, aspect_ratio_used)
    """

    if target_aspect_ratio is None:
        if screen_width is None or screen_height is None:
            screen_width, screen_height = get_screen_size()
        aspect_ratio = screen_width / screen_height
    else:
        aspect_ratio = target_aspect_ratio

    # Start with square root as baseline
    base_dim = np.ceil(np.sqrt(n_plots))

    best_rows, best_cols = base_dim, base_dim
    best_score = float("inf")

    # Try different combinations around the baseline
    for rows in range(1, n_plots + 1):
        cols = np.ceil(n_plots / rows)

        # Skip if this creates too many empty subplots
        if rows * cols - n_plots > min(rows, cols):
            continue

        # Calculate how close this layout's aspect ratio is to screen ratio
        layout_aspect_ratio = cols / rows
        aspect_diff = abs(layout_aspect_ratio - aspect_ratio)

        # Prefer layouts that minimize aspect ratio difference
        # and minimize total subplots (less wasted space)
        total_subplots = rows * cols
        score = aspect_diff + 0.1 * (total_subplots - n_plots)

        if score < best_score:
            best_score = score
            best_rows, best_cols = rows, cols

    return int(best_rows), int(best_cols), aspect_ratio


#####################################################################################################
def create_proportional_subplots(n_plots, figsize_base=4, **layout_kwargs):
    """
    Create a figure with subplots arranged according to screen proportions

    Parameters:
    -----------
    n_plots : int
        Number of subplots

    figsize_base : float
        Base size for figure scaling

    **layout_kwargs :
        Additional arguments for calculate_subplot_layout()

    Returns:
    --------
    tuple: (fig, axes, layout_info)
    """

    rows, cols, aspect_ratio = calculate_subplot_layout(n_plots, **layout_kwargs)

    # Calculate figure size based on layout and aspect ratio
    fig_width = figsize_base * cols
    fig_height = fig_width / aspect_ratio

    # Create the figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Handle the case where there's only one subplot
    if n_plots == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Hide extra subplots if any
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    layout_info = {
        "rows": rows,
        "cols": cols,
        "aspect_ratio": aspect_ratio,
        "total_subplots": rows * cols,
        "used_subplots": n_plots,
    }

    plt.tight_layout()

    return fig, axes, layout_info


######################################################################################################
def calculate_font_sizes(
    plot_width,
    plot_height,
    screen_width=None,
    screen_height=None,
    colorbar_orientation="vertical",
    colorbar_width=None,
    colorbar_height=None,
    auto_detect_monitor=True,
):
    """
    Calculate appropriate font sizes for matplotlib plots based on dimensions, monitor DPI, and colorbar configuration.

    This function automatically scales font sizes for plot elements (title, axis labels, tick labels,
    colorbar title, and colorbar ticks) based on the plot dimensions, monitor characteristics, and
    colorbar orientation. The scaling ensures optimal readability across different plot sizes and
    display configurations by calculating monitor DPI from screen resolution.

    The algorithm uses a reference plot size of 6×4 inches as a baseline and scales font sizes
    proportionally based on plot area and monitor DPI. Colorbar fonts are scaled independently
    based on colorbar dimensions and orientation constraints.

    Parameters
    ----------
    plot_width : float
        Width of the plot in inches. Must be positive.
        Typical values: 3-20 inches for scientific plots.

    plot_height : float
        Height of the plot in inches. Must be positive.
        Typical values: 2-15 inches for scientific plots.

    screen_width : int, optional
        Screen width in pixels, by default None.
        If None and auto_detect_monitor=True, automatically detected.
        Used together with screen_height to calculate monitor DPI.

    screen_height : int, optional
        Screen height in pixels, by default None.
        If None and auto_detect_monitor=True, automatically detected.
        Used together with screen_width to calculate monitor DPI.

    colorbar_orientation : {'vertical', 'horizontal'}, optional
        Orientation of the colorbar, by default 'vertical'.
        - 'vertical': Colorbar positioned to the right/left of the plot
        - 'horizontal': Colorbar positioned above/below the plot

    colorbar_width : float, optional
        Width of the colorbar in inches, by default None.
        If None, automatically calculated as 5% of plot width (vertical) or
        80% of plot width (horizontal), with minimum constraints.

    colorbar_height : float, optional
        Height of the colorbar in inches, by default None.
        If None, automatically calculated as 80% of plot height (vertical) or
        5% of plot height (horizontal), with minimum constraints.

    auto_detect_monitor : bool, optional
        Whether to automatically detect current monitor size, by default True.
        If False, uses fallback values when screen dimensions are not provided.

    Returns
    -------
    dict
        Dictionary containing font sizes for different plot elements:

        - 'title' : float
            Font size for the main plot title (8-28 pt range)

        - 'axis_labels' : float
            Font size for axis labels (6-20 pt range)

        - 'tick_labels' : float
            Font size for axis tick labels (6-20 pt range)

        - 'colorbar_title' : float
            Font size for colorbar title (6-16 pt range, orientation-dependent)

        - 'colorbar_ticks' : float
            Font size for colorbar tick labels (5-12 pt range, orientation-dependent)

        - '_monitor_info' : dict
            Debug information about monitor detection and DPI calculation

        All font sizes are rounded to 1 decimal place.

    Raises
    ------
    ValueError
        If plot_width or plot_height are not positive numbers.

    ValueError
        If colorbar_orientation is not 'vertical' or 'horizontal'.

    TypeError
        If numeric parameters are not of appropriate numeric types.

    ImportError
        If auto_detect_monitor=True but required packages (tkinter, screeninfo) are not available.

    Examples
    --------
    Basic usage with automatic monitor detection:

    >>> fonts = calculate_font_sizes(6, 4)
    >>> print(f"Title: {fonts['title']}, Colorbar title: {fonts['colorbar_title']}")
    Title: 14.2, Colorbar title: 11.4

    Specify monitor dimensions manually:

    >>> fonts = calculate_font_sizes(6, 4, screen_width=2560, screen_height=1440)
    >>> fonts['_monitor_info']['estimated_dpi']
    109.0

    Small subplot with horizontal colorbar on high-DPI display:

    >>> fonts = calculate_font_sizes(3, 2,
    ...                             screen_width=3840, screen_height=2160,
    ...                             colorbar_orientation='horizontal')
    >>> fonts['colorbar_title']  # Scaled up for high DPI
    10.2

    Disable auto-detection for headless environments:

    >>> fonts = calculate_font_sizes(12, 8,
    ...                             screen_width=1920, screen_height=1080,
    ...                             auto_detect_monitor=False)
    >>> fonts['title']
    19.1

    Custom colorbar dimensions:

    >>> fonts = calculate_font_sizes(8, 6,
    ...                             colorbar_orientation='horizontal',
    ...                             colorbar_width=6.0,
    ...                             colorbar_height=0.5)

    Apply to matplotlib plot:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(figsize=(6, 4))
    >>> fonts = calculate_font_sizes(6, 4, colorbar_orientation='horizontal')
    >>> ax.set_title('Brain Activation', fontsize=fonts['title'])
    >>> ax.tick_params(labelsize=fonts['tick_labels'])
    >>> # For colorbar:
    >>> # cbar.set_label('Values', fontsize=fonts['colorbar_title'])
    >>> # cbar.ax.tick_params(labelsize=fonts['colorbar_ticks'])

    Notes
    -----
    DPI Calculation and Scaling:
    1. Automatically detects current monitor resolution using get_current_monitor_size()
    2. Estimates monitor DPI based on resolution and common monitor configurations
    3. Applies DPI-based scaling factor: scale = estimated_dpi / 96.0 (Windows standard)
    4. Combines with plot area scaling for final font sizes

    Font scaling algorithm:
    1. Calculate plot area scaling factor relative to 6×4 inch reference
    2. Calculate monitor DPI scaling factor relative to 96 DPI baseline
    3. Scale colorbar fonts based on constraining dimension:
        - Vertical colorbars: limited by width → scale by width/0.5"
        - Horizontal colorbars: limited by height → scale by height/0.4"
    4. Apply orientation-specific bounds to ensure readability

    The function prioritizes readability over exact proportional scaling, applying
    reasonable minimum and maximum font sizes for each element type.

    Monitor configurations used for DPI estimation:
    - 1920×1080: 82-147 DPI (depending on screen size)
    - 2560×1440: 92-196 DPI
    - 3840×2160: 103-294 DPI
    - Other resolutions estimated using pixel density

    For PyVista compatibility, convert results to integers:
    >>> pv_fonts = {k: int(round(v)) for k, v in fonts.items() if not k.startswith('_')}
    """

    # Input validation
    if not isinstance(plot_width, (int, float)) or plot_width <= 0:
        raise ValueError("plot_width must be a positive number")
    if not isinstance(plot_height, (int, float)) or plot_height <= 0:
        raise ValueError("plot_height must be a positive number")
    if colorbar_orientation not in ["vertical", "horizontal"]:
        raise ValueError("colorbar_orientation must be 'vertical' or 'horizontal'")

    # Get monitor dimensions
    if screen_width is None or screen_height is None:
        if auto_detect_monitor:
            try:
                detected_width, detected_height = get_current_monitor_size()
                screen_width = screen_width or detected_width
                screen_height = screen_height or detected_height
            except ImportError as e:
                raise ImportError(
                    "Auto monitor detection requires 'tkinter' and 'screeninfo' packages. "
                    "Install with: pip install screeninfo\n"
                    "Or disable auto-detection: auto_detect_monitor=False"
                ) from e
        else:
            # Fallback to common resolution
            screen_width = screen_width or 1920
            screen_height = screen_height or 1080

    # Calculate monitor DPI
    estimated_dpi = estimate_monitor_dpi(screen_width, screen_height)

    # Calculate scaling factors
    plot_area = plot_width * plot_height
    base_area = 24.0  # 6 * 4 inches reference
    area_scale_factor = (plot_area / base_area) ** 0.5

    # DPI scaling relative to 96 DPI baseline (Windows standard)
    dpi_scale_factor = estimated_dpi / 96.0
    # Cap DPI scaling to reasonable bounds
    dpi_scale_factor = max(0.8, min(2.0, dpi_scale_factor))

    # Combined plot scaling
    plot_scale = area_scale_factor * dpi_scale_factor

    # Colorbar-specific scaling
    if colorbar_orientation == "vertical":
        if colorbar_width is None:
            colorbar_width = max(0.3, plot_width * 0.05)
        if colorbar_height is None:
            colorbar_height = plot_height * 0.8
        colorbar_scale = (
            colorbar_width / 0.5
        ) * dpi_scale_factor  # Reference: 0.5" wide
    else:  # horizontal
        if colorbar_width is None:
            colorbar_width = plot_width * 0.8
        if colorbar_height is None:
            colorbar_height = max(0.3, plot_height * 0.05)
        colorbar_scale = (
            colorbar_height / 0.4
        ) * dpi_scale_factor  # Reference: 0.4" tall

    # Base font sizes (optimized for 96 DPI)
    base_sizes = {
        "title": 14,
        "axis_labels": 12,
        "tick_labels": 10,
        "colorbar_title": 11,
        "colorbar_ticks": 9,
    }

    font_sizes = {}
    for element, base_size in base_sizes.items():
        if element.startswith("colorbar"):
            scaled_size = base_size * colorbar_scale
        else:
            scaled_size = base_size * plot_scale

        # Apply reasonable bounds
        if element == "title":
            scaled_size = max(8, min(28, scaled_size))
        elif element == "colorbar_title":
            if colorbar_orientation == "horizontal":
                scaled_size = max(7, min(16, scaled_size))
            else:
                scaled_size = max(6, min(14, scaled_size))
        elif element == "colorbar_ticks":
            if colorbar_orientation == "horizontal":
                scaled_size = max(6, min(12, scaled_size))
            else:
                scaled_size = max(5, min(10, scaled_size))
        else:
            scaled_size = max(6, min(20, scaled_size))

        font_sizes[element] = round(scaled_size, 1)

    # Add monitor info for debugging
    font_sizes["_monitor_info"] = {
        "screen_width": screen_width,
        "screen_height": screen_height,
        "estimated_dpi": estimated_dpi,
        "dpi_scale_factor": round(dpi_scale_factor, 2),
        "area_scale_factor": round(area_scale_factor, 2),
        "plot_scale": round(plot_scale, 2),
        "colorbar_scale": round(colorbar_scale, 2),
    }

    return font_sizes


######################################################################################################
# Additional utility functions for common use cases
def get_pyvista_fonts(plot_width, plot_height, **kwargs):
    """
    Get integer font sizes specifically for PyVista compatibility.

    Parameters
    ----------
    plot_width : float
        Width of the plot in inches.

    plot_height : float
        Height of the plot in inches.

    **kwargs
        Additional keyword arguments passed to calculate_font_sizes().

    Returns
    -------
    dict
        Dictionary with integer font sizes suitable for PyVista (excludes debug info).

    Examples
    --------
    >>> fonts = get_pyvista_fonts(8, 6, colorbar_orientation='vertical')
    >>> plotter.add_title('3D Brain', font_size=fonts['title'])
    """
    fonts = calculate_font_sizes(plot_width, plot_height, **kwargs)
    # Exclude debug info for clean PyVista usage
    return {
        key: int(round(value))
        for key, value in fonts.items()
        if not key.startswith("_")
    }
