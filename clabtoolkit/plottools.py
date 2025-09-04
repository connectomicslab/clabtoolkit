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
            'laptop_13_15': 147,    # 13-15" laptop
            'laptop_17': 130,       # 17" laptop  
            'monitor_21_24': 92,    # 21-24" monitor
            'monitor_27': 82,       # 27" monitor
        },
        # QHD displays
        (2560, 1440): {
            'laptop_13_15': 196,    # 13-15" laptop
            'monitor_27': 109,      # 27" monitor
            'monitor_32': 92,       # 32" monitor
        },
        # 4K displays  
        (3840, 2160): {
            'laptop_15_17': 294,    # 15-17" laptop
            'monitor_27': 163,      # 27" monitor
            'monitor_32': 138,      # 32" monitor
            'monitor_43': 103,      # 43" monitor
        },
        # Other common resolutions
        (1366, 768): {
            'laptop_11_14': 112,    # Small laptops
        },
        (1680, 1050): {
            'monitor_22': 90,       # 22" monitor
        },
        (2880, 1800): {
            'laptop_15': 220,       # MacBook Pro 15"
        },
        (3440, 1440): {
            'ultrawide_34': 110,    # 34" ultrawide
        }
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
            return sorted(configs.values())[len(configs.values())//2]
    
    # Fallback: calculate approximate DPI based on pixel density
    # Assume reasonable screen diagonal based on resolution
    total_pixels = screen_width * screen_height
    
    if total_pixels <= 1000000:    # ≤ 1M pixels
        estimated_diagonal = 13      # Small laptop/tablet

    elif total_pixels <= 2000000:  # ≤ 2M pixels  
        estimated_diagonal = 21      # Standard monitor

    elif total_pixels <= 4000000:  # ≤ 4M pixels
        estimated_diagonal = 24      # Larger monitor
        
    elif total_pixels <= 8000000:  # ≤ 8M pixels
        estimated_diagonal = 27      # QHD monitor

    else:                          # > 8M pixels
        estimated_diagonal = 32      # 4K monitor
    
    # Calculate DPI: sqrt(width² + height²) / diagonal_inches
    diagonal_pixels = (screen_width**2 + screen_height**2)**0.5
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
