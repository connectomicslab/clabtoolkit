"""
Module for visualization utilities in the clabtoolkit package.
"""

import os
import json
import numpy as np
from typing import Union, List, Optional, Tuple, Dict, Any, TYPE_CHECKING
import pyvista as pv
import threading
from pathlib import Path

from nibabel.streamlines import ArraySequence

# Importing local modules
from . import misctools as cltmisc
from . import plottools as cltplot

# Use TYPE_CHECKING to avoid circular imports
from . import surfacetools as cltsurf
from . import tracttools as clttract


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############   Module dedicated to prepare objects for the VisualizationTools module    ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def load_configs(config_file: Union[str, Path]) -> None:
    """
    Load figure and view configurations from JSON file.

    Parameters
    ----------
    config_file : str
        Path to the JSON configuration file.

    Raises
    ------
    FileNotFoundError
        If the configuration file doesn't exist.

    json.JSONDecodeError
        If the configuration file contains invalid JSON.

    KeyError
        If required configuration keys 'figure_conf' or 'views_conf' are missing.

    Examples
    --------
    >>> plotter = SurfacePlotter("configs.json")
    >>> plotter._load_configs()  # Reloads configurations from file
    """

    if isinstance(config_file, Path):
        config_file = str(config_file)

    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found")

    # Load configurations from JSON file
    try:
        with open(config_file, "r") as f:
            configs = json.load(f)

        # Validate structure and load configurations
        if "figure_conf" not in configs:
            raise KeyError("Missing 'figure_conf' key in configuration file")
        if "views_conf" not in configs:
            raise KeyError("Missing 'views_conf' key in configuration file")

        return configs

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_file}' not found")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in configuration file: {e}")


##################################################################################################
def get_views_to_plot(
    plotobj, views: Union[str, List[str]], hemi_id: Union[str, List[str]] = "lh"
) -> List[str]:
    """
    Get the list of views to plot based on user input and hemisphere.
    This method normalizes the input views, validates them against the available
    views configuration, and filters them based on the specified hemisphere.

    Parameters
    ----------
    plotobj : SurfacePlotter or TractogramPlotter
        Instance of the plotting class containing views_conf and layouts_conf attributes.

    views : Union[str, List[str]]
        The view names to plot. Can be a single string or a list of strings.
        If a single string is provided, it will be converted to a list.

    hemi_id : Union[str, List[str]]
        The hemisphere identifiers to consider. Can be a single string or a list of strings.
        Common identifiers are "lh" for left hemisphere and "rh" for right hemisphere.

    Returns
    -------
    List[str]
        A list of valid view names to plot, filtered by the specified hemisphere.

    Raises
    ------
    ValueError
        If the provided views are not a string or a list of strings.
        If no valid views are found after filtering.

    Notes
    -----
    This method is designed to work with the available views defined in the
    `views_conf` attribute of the class. It ensures that the views are compatible
    with the hemisphere specified and returns a list of valid view names.
    If the input views are not valid or do not match any available views,
    """

    # Normalize input views to a list
    if isinstance(views, str):
        views = [views]
    elif not isinstance(views, list):
        raise ValueError(
            "Views must be a string or a list of strings representing view names"
        )

    # Validate views
    valid_views = plotobj._get_valid_views(views)

    # Get number of views
    if len(valid_views) == 1:
        if valid_views[0] in plotobj._list_multiviews_layouts():
            view_ids = plotobj.layouts_conf[valid_views[0]]["views"]
            if "lh" in hemi_id and "rh" not in hemi_id:
                # LH only, remove the view_ids that contain rh- on the name
                view_ids = [v for v in view_ids if "rh-" not in v]
            elif "rh" in hemi_id and "lh" not in hemi_id:
                # RH only, remove the view_ids that contain lh- on the name
                view_ids = [v for v in view_ids if "lh-" not in v]
                # Flip the view_ids and the last will be the first
                view_ids = view_ids[::-1]

        elif valid_views[0] in plotobj._list_single_views():
            # Single view layout, take all the possible views
            view_ids = list(plotobj.views_conf.keys())
            # Selecting the views based on the supplied names
            view_ids = cltmisc.filter_by_substring(view_ids, valid_views)
            # Filter views based on hemisphere
            if "lh" in hemi_id and "rh" not in hemi_id:
                view_ids = [v for v in view_ids if "rh-" not in v]
            elif "rh" in hemi_id and "lh" not in hemi_id:
                view_ids = [v for v in view_ids if "lh-" not in v]
    else:
        # Multiple view names provided
        view_ids = list(plotobj.views_conf.keys())
        # Selecting the views based on the supplied names
        view_ids = cltmisc.filter_by_substring(view_ids, valid_views)
        # Filter views based on hemisphere
        if "lh" in hemi_id and "rh" not in hemi_id:
            view_ids = [v for v in view_ids if "rh-" not in v]
        elif "rh" in hemi_id and "lh" not in hemi_id:
            view_ids = [v for v in view_ids if "lh-" not in v]

    return view_ids


#################################################################################################
def colorbar_needed(maps_names, plotsobj) -> bool:
    """Check if colorbar is actually needed based on surface colortables."""
    if not plotsobj:
        return True

    # Check if any map is not already on the surface
    for map_name in maps_names:
        if map_name not in plotsobj[0].colortables:
            return True
    return False


#################################################################################################
def finalize_plot(
    plotter: pv.Plotter,
    save_mode: bool,
    save_path: Optional[str],
    use_threading: bool = False,
) -> None:
    """
    Handle final rendering - either save or display the plot.

    Parameters
    ----------
    plotter : pv.Plotter
        PyVista plotter instance ready for final rendering.

    save_mode : bool
        If True, save the plot; if False, display it.

    save_path : str, optional
        File path for saving (required if save_mode is True).

    use_threading : bool, default False
        If True, display plot in separate thread (non-blocking mode).
        Only applies when save_mode is False.
    """
    if save_mode and save_path:

        if save_path.lower().endswith((".html", ".htm")):
            # Save as HTML
            try:

                plotter.export_html(save_path)
                print(f"Figure saved to: {save_path}")

            except Exception as e:
                print(f"Error saving HTML: {e}")
            finally:
                plotter.close()

        elif save_path.lower().endswith((".svg", ".pdf", ".eps", ".ps", ".tex")):
            # Save as vector graphic
            try:
                plotter.save_graphic(save_path)
                print(f"Figure saved to: {save_path}")
            except Exception as e:
                print(f"Error saving vector graphic: {e}")
            finally:
                plotter.close()

        else:
            # Save mode - render and save without displaying
            plotter.render()
            try:
                plotter.screenshot(save_path)
                print(f"Figure saved to: {save_path}")
            except Exception as e:
                print(f"Error saving screenshot: {e}")
                # Try alternative approach
                try:
                    img = plotter.screenshot(save_path, return_img=True)
                    if img is not None:
                        print(f"Figure saved to: {save_path} (alternative method)")
                except Exception as e2:
                    print(f"Alternative screenshot method also failed: {e2}")
            finally:
                plotter.close()
    else:
        # Display mode
        if use_threading:
            # Non-blocking mode - show in separate thread
            create_threaded_plot(plotter)
        else:
            # Blocking mode - show normally
            plotter.show()


##################################################################################################
def link_brain_subplot_cameras(pv_plotter, brain_positions):
    """
    Link cameras for brain subplots that share the same view index.

    Args:
        pv_plotter: PyVista plotter object
        brain_positions: Dict with keys (m_idx, s_idx, v_idx) and values (row, col)
    """
    # Group positions by view index using defaultdict for cleaner code
    from collections import defaultdict

    grouped_by_v_idx = defaultdict(list)
    for (_, _, v_idx), (row, col) in brain_positions.items():
        grouped_by_v_idx[v_idx].append((row, col))

    # Convert back to regular dict if needed
    grouped_by_v_idx = dict(grouped_by_v_idx)

    n_rows, n_cols = pv_plotter.shape
    successful_links = 0

    # Link views for each group
    for v_idx, positions in grouped_by_v_idx.items():
        if len(positions) <= 1:
            continue  # Need at least 2 positions to link

        # Calculate and validate subplot indices
        valid_indices = []
        invalid_positions = []

        for row, col in positions:
            # Validate position bounds
            if not (0 <= row < n_rows and 0 <= col < n_cols):
                invalid_positions.append((row, col, "out of bounds"))
                continue

            subplot_idx = row * n_cols + col

            # Validate renderer exists
            if subplot_idx >= len(pv_plotter.renderers):
                invalid_positions.append(
                    (
                        row,
                        col,
                        f"index {subplot_idx} >= {len(pv_plotter.renderers)}",
                    )
                )
                continue

            # Validate renderer is not None
            if pv_plotter.renderers[subplot_idx] is None:
                invalid_positions.append(
                    (row, col, f"renderer at index {subplot_idx} is None")
                )
                continue

            valid_indices.append(subplot_idx)

        # Report any invalid positions
        if invalid_positions:
            print(
                f"Warning: Skipped {len(invalid_positions)} invalid positions for view {v_idx}:"
            )
            for row, col, reason in invalid_positions:
                print(f"  Position ({row}, {col}): {reason}")

        # Link views if we have enough valid indices
        if len(valid_indices) > 1:
            try:
                pv_plotter.link_views(valid_indices)
                successful_links += 1
                print(
                    f"✓ Linked {len(valid_indices)} views for v_idx {v_idx}: indices {valid_indices}"
                )
            except Exception as e:
                print(f"✗ Failed to link views for v_idx {v_idx}: {e}")
        else:
            print(
                f"⚠ Not enough valid renderers for v_idx {v_idx} ({len(valid_indices)}/2+ needed)"
            )

    print(
        f"\nSummary: Successfully linked {successful_links}/{len(grouped_by_v_idx)} view groups"
    )
    return successful_links


################################################################################################
def prepare_obj_for_plotting(
    obj2plot: Union[cltsurf.Surface, clttract.Tractogram],
    map_name: str,
    colormap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
    range_color: List[int, int, int, int] = [128, 128, 128, 255],
) -> cltsurf.Surface:
    """
    Prepare Surface or Tractogram object for plotting with color mapping.

    Parameters
    ----------
    obj2plot : Union[cltsurf.Surface, clttract.Tractogram]
        The object to prepare for plotting. Can be a Surface or Tractogram instance.

    map_name : str
        Name of the data array to use for color mapping.
    colormap : str
        Matplotlib colormap name to use for color mapping.

    vmin : float, optional
        Minimum value for color scaling. If None, computed from data.

    vmax : float, optional
        Maximum value for color scaling. If None, computed from data.

    range_min : float, optional
        Minimum value for value range masking. Values below this will be displayed in gray.

    range_max : float, optional
        Maximum value for value range masking. Values above this will be displayed in gray.

    range_color : List[int, int, int, int], optional
        RGBA color to use for values outside the specified range. Default is gray [128, 128, 128, 255].

    Returns
    -------
    cltsurf.Surface or clttract.Tractogram
        The prepared object with color mapping applied.


    """
    if isinstance(obj2plot, clttract.Tractogram):
        # Check if map_name exists in data_per_streamline or data_per_point
        if (
            map_name not in obj2plot.data_per_streamline
            and map_name not in obj2plot.data_per_point
        ):
            raise ValueError(
                f"Data array '{map_name}' not found in streamline or point data"
            )
        elif map_name in obj2plot.data_per_streamline and not obj2plot.data_per_point:
            # We have to convert it to data_per_point
            obj2plot.streamline_to_points(map_name)

        point_values = obj2plot.data_per_point[map_name]

        # Concatenate the list of arrays into a single array
        point_values = np.concatenate(point_values)

        if vmin is None:
            vmin = np.min(point_values)

        if vmax is None:
            vmax = np.max(point_values)

        point_values = np.nan_to_num(
            point_values,
            nan=0.0,
        )  # Handle NaNs and infinities

        obj2plot.data_per_point["rgba"] = obj2plot.get_pointwise_colors(
            map_name, colormap, vmin, vmax
        )

        if range_min is not None or range_max is not None:
            rgba_colors = obj2plot.data_per_point["rgba"]

            # Create mask for out-of-range values
            mask = np.zeros(len(point_values), dtype=bool)
            if range_min is not None:
                mask |= data_values < range_min

            if range_max is not None:
                mask |= data_values > range_max

            # Set out-of-range values to a specified color
            if rgba_colors.shape[1] == 4:  # RGBA
                rgba_colors[mask] = range_color

            elif rgba_colors.shape[1] == 3:  # RGB
                rgba_colors[mask] = range_color[:3]

            obj2plot.data_per_point["rgba"] = rgba_colors

    elif isinstance(obj2plot, cltsurf.Surface):
        if vmin is None:
            vmin = np.min(obj2plot.mesh.point_data[map_name])

        if vmax is None:
            vmax = np.max(obj2plot.mesh.point_data[map_name])

        try:
            vertex_values = obj2plot.mesh.point_data[map_name]
            vertex_values = np.nan_to_num(
                vertex_values,
                nan=0.0,
            )  # Handle NaNs and infinities
            obj2plot.mesh.point_data[map_name] = vertex_values

        except KeyError:
            raise ValueError(f"Data array '{map_name}' not found in surface point_data")

        # Apply colors to mesh data
        obj2plot.mesh.point_data["rgba"] = obj2plot.get_vertexwise_colors(
            map_name, colormap, vmin, vmax
        )

        # Apply gray color to values outside the specified range
        if range_min is not None or range_max is not None:
            data_values = obj2plot.mesh.point_data[map_name]
            rgba_colors = obj2plot.mesh.point_data["rgba"]

            # Create mask for out-of-range values
            mask = np.zeros(len(data_values), dtype=bool)
            if range_min is not None:
                mask |= data_values < range_min
            if range_max is not None:
                mask |= data_values > range_max

            # Set out-of-range values to a specified color
            if rgba_colors.shape[1] == 4:  # RGBA
                rgba_colors[mask] = range_color

            elif rgba_colors.shape[1] == 3:  # RGB
                rgba_colors[mask] = range_color[:3]

            obj2plot.mesh.point_data["rgba"] = rgba_colors

    else:
        raise TypeError("obj2plot must be a Surface or Tractogram instance")

    return obj2plot


################################################################################################
def process_v_limits(
    v_limits: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]],
    n_maps: int,
) -> List[Tuple[Optional[float], Optional[float]]]:
    """
    Process and validate the v_limits parameter.

    Parameters
    ----------
    v_limits : tuple or List[tuple], optional
        The v_limits parameter from the main method.

    n_maps : int
        Number of maps to be plotted.

    Returns
    -------
    List[Tuple[Optional[float], Optional[float]]]
        List of (vmin, vmax) tuples, one for each map.

    Raises
    ------
    TypeError
        If v_limits format is invalid.

    ValueError
        If v_limits list length doesn't match number of maps.
    """

    # Validate v_limits input
    if v_limits is None or (isinstance(v_limits, tuple) and len(v_limits) == 2):
        # Single tuple or None - use for all maps
        v_limits = [v_limits] * n_maps if v_limits else [(None, None)]

    elif isinstance(v_limits, list) and all(
        isinstance(limits, tuple) and len(limits) == 2 for limits in v_limits
    ):
        # List of tuples - validate length and content
        if len(v_limits) != n_maps:
            raise ValueError(
                f"v_limits list length ({len(v_limits)}) must match number of maps ({n_maps})"
            )
    else:
        raise TypeError(
            "v_limits must be None, a tuple (vmin, vmax), or a list of tuples [(vmin1, vmax1), ...]"
        )

    if v_limits is None:
        # Auto-compute limits for each map
        return [(None, None)] * n_maps

    elif isinstance(v_limits, tuple) and len(v_limits) == 2:
        # Single tuple - use for all maps
        vmin, vmax = v_limits
        if not (isinstance(vmin, (int, float)) and isinstance(vmax, (int, float))):
            raise TypeError("v_limits tuple must contain numeric values")
        if vmin >= vmax:
            raise ValueError(f"vmin ({vmin}) must be less than vmax ({vmax})")

        print(f"Using same limits for all {n_maps} maps: vmin={vmin}, vmax={vmax}")
        return [(vmin, vmax)] * n_maps

    elif isinstance(v_limits, list):
        # List of tuples - validate length and content
        if len(v_limits) != n_maps:
            raise ValueError(
                f"v_limits list length ({len(v_limits)}) must match number of maps ({n_maps})"
            )

        processed_limits = []
        for i, limits in enumerate(v_limits):
            if not (isinstance(limits, tuple) and len(limits) == 2):
                raise TypeError(f"v_limits[{i}] must be a tuple of length 2")

            vmin, vmax = limits
            if not (isinstance(vmin, (int, float)) and isinstance(vmax, (int, float))):
                raise TypeError(f"v_limits[{i}] must contain numeric values")
            if vmin >= vmax:
                raise ValueError(
                    f"v_limits[{i}]: vmin ({vmin}) must be less than vmax ({vmax})"
                )

            processed_limits.append((vmin, vmax))

        print(f"Using individual limits for {n_maps} maps:")
        for i, (vmin, vmax) in enumerate(processed_limits):
            print(f"  Map {i}: vmin={vmin}, vmax={vmax}")

        return processed_limits

    else:
        raise TypeError(
            "v_limits must be None, a tuple (vmin, vmax), or a list of tuples [(vmin1, vmax1), ...]"
        )


###############################################################################################
def add_colorbar(
    plotobj,
    plotter: pv.Plotter,
    colorbar_subplot: Tuple[int, int],
    vmin: Any,
    vmax: Any,
    map_name: str,
    colormap: str,
    colorbar_title: str,
    colorbar_position: str,
) -> None:
    """
    Add a properly positioned colorbar to the plot.

    Parameters
    ----------
    plotter : pv.Plotter
        PyVista plotter instance.

    config : Dict[str, Any]
        View configuration containing shape information.

    data_values : np.ndarray
        Data values from the merged surface for color mapping.

    map_name : str
        Name of the data array to use for colorbar.

    colormap : str
        Matplotlib colormap name.

    colorbar_title : str
        Title text for the colorbar.

    colorbar_position : str
        Position of colorbar: "top", "bottom", "left", "right".

    Raises
    ------
    KeyError
        If map_name is not found in surf_merged point_data.

    ValueError
        If colorbar_position is invalid or data array is empty.

    Examples
    --------
    >>> self._add_colorbar(
    ...     plotter, config, surf_merged, "thickness",
    ...     "viridis", "Cortical Thickness", "bottom"
    ... )
    # Adds horizontal colorbar at bottom of plot
    """

    if isinstance(map_name, list):
        map_name = map_name[0]

    plotter.subplot(*colorbar_subplot)
    # Set background color for colorbar subplot
    plotter.set_background(plotobj.figure_conf["background_color"])

    # Create colorbar mesh with proper data range
    n_points = 256
    colorbar_mesh = pv.Line((0, 0, 0), (1, 0, 0), resolution=n_points - 1)
    scalar_values = np.linspace(vmin, vmax, n_points)
    colorbar_mesh[map_name] = scalar_values

    # Determine font sizes based on colorbar orientation and subplot size
    # Get the current renderer
    current_renderer = plotter.renderer

    # Get viewport bounds (normalized coordinates 0-1)
    viewport = current_renderer.GetViewport()
    # viewport returns (xmin, ymin, xmax, ymax)

    # Convert to actual pixel dimensions
    window_size = plotter.window_size
    subplot_width = (viewport[2] - viewport[0]) * window_size[0]
    subplot_height = (viewport[3] - viewport[1]) * window_size[1]
    font_sizes = cltplot.calculate_font_sizes(
        subplot_width, subplot_height, colorbar_orientation=colorbar_position
    )

    # Add invisible mesh for colorbar reference
    dummy_actor = plotter.add_mesh(
        colorbar_mesh,
        scalars=map_name,
        cmap=colormap,
        clim=[vmin, vmax],
        show_scalar_bar=False,
    )
    dummy_actor.visibility = False

    # Create scalar bar manually using VTK
    import vtk

    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(dummy_actor.mapper.lookup_table)

    # Set outline
    if not plotobj.figure_conf["colorbar_outline"]:
        scalar_bar.DrawFrameOff()

    # scalar_bar.SetPosition(0.1, 0.1)
    # scalar_bar.SetPosition2(0.9, 0.9)
    # Position colorbar appropriately
    if colorbar_position == "horizontal":
        # Horizontal colorbar
        scalar_bar.SetPosition(0.05, 0.05)  # 5% from left, 5% from bottom
        scalar_bar.SetPosition2(0.9, 0.7)  # 90% width, 70% height
        scalar_bar.SetOrientationToHorizontal()

    else:
        # More conventional vertical version with same positioning philosophy:
        scalar_bar.SetPosition(0.05, 0.05)  # 5% from left, 5% from bottom
        scalar_bar.SetPosition2(0.7, 0.9)  # 12% width, 90% height
        scalar_bar.SetOrientationToVertical()

    colorbar_title = colorbar_title.capitalize()
    scalar_bar.SetTitle(colorbar_title)

    scalar_bar.SetMaximumNumberOfColors(256)
    scalar_bar.SetNumberOfLabels(plotobj.figure_conf["colorbar_n_labels"])

    # Get text properties for title and labels
    title_prop = scalar_bar.GetTitleTextProperty()
    label_prop = scalar_bar.GetLabelTextProperty()

    # Set colors
    title_color = pv.Color(plotobj.figure_conf["colorbar_font_color"]).float_rgb
    title_prop.SetColor(*title_color)
    label_prop.SetColor(*title_color)

    # Set font properties - key fix for consistent sizing
    if plotobj.figure_conf["colorbar_font_type"].lower() == "arial":
        title_prop.SetFontFamilyToArial()
        label_prop.SetFontFamilyToArial()

    elif plotobj.figure_conf["colorbar_font_type"].lower() == "courier":
        title_prop.SetFontFamilyToCourier()
        label_prop.SetFontFamilyToCourier()

    else:
        title_prop.SetFontFamilyToTimes()  # Ensure consistent font family
        label_prop.SetFontFamilyToTimes()

    base_title_size = font_sizes["colorbar_title"]
    base_label_size = font_sizes["colorbar_ticks"]

    # Apply font sizes with explicit scaling
    title_prop.SetFontSize(base_title_size)
    label_prop.SetFontSize(base_label_size)

    # Enable/disable bold for better consistency
    title_prop.BoldOff()
    title_prop.ItalicOff()
    label_prop.BoldOff()

    # Set text properties for better rendering consistency
    title_prop.SetJustificationToCentered()
    title_prop.SetVerticalJustificationToCentered()
    label_prop.SetJustificationToCentered()
    label_prop.SetVerticalJustificationToCentered()

    # Additional properties for consistent rendering
    scalar_bar.SetLabelFormat("%.2f")  # Consistent number formatting
    # scalar_bar.SetMaximumWidthInPixels(1000)  # Prevent excessive scaling
    # scalar_bar.SetMaximumHeightInPixels(1000)

    # Set text margin for better spacing
    scalar_bar.SetTextPad(4)
    scalar_bar.SetVerticalTitleSeparation(10)

    # Add the scalar bar to the plotter
    plotter.add_actor(scalar_bar)


###############################################################################################
def create_threaded_plot(plotter: pv.Plotter) -> None:
    """
    Create and show plot in a separate thread for non-blocking visualization.

    Parameters
    ----------
    plotter : pv.Plotter
        PyVista plotter instance ready for display.
    """

    def show_plot():
        """Internal function to run in separate thread."""
        try:
            plotter.show()
        except Exception as e:
            print(f"Error displaying plot in thread: {e}")
        finally:
            # Clean up if needed
            pass

    # Create and start the thread
    plot_thread = threading.Thread(target=show_plot)
    plot_thread.daemon = True  # Thread will close when main program closes
    plot_thread.start()

    print("Plot opened in separate window. Terminal remains interactive.")
    print("Note: Plot window may take a moment to appear.")


###############################################################################################
def determine_render_mode(
    save_path: Optional[str], notebook: bool, non_blocking: bool = False
) -> Tuple[bool, bool, bool, bool]:
    """
    Determine rendering parameters based on save path and environment.

    Parameters
    ----------
    save_path : str, optional
        File path for saving the figure, or None for display.

    notebook : bool
        Whether running in Jupyter notebook environment.

    non_blocking : bool, default False
        Whether to run the visualization in a separate thread (non-blocking mode).

    Returns
    -------
    Tuple[bool, bool, bool, bool]
        (save_mode, use_off_screen, use_notebook, use_threading).
        - save_mode: True if saving, False if displaying.
        - use_off_screen: True if off-screen rendering is needed.
        - use_notebook: True if running in notebook environment.
        - use_threading: True if using threading for non-blocking display.

    Notes
    -----
    - If save_path is provided, save_mode is True and off-screen rendering is used.
    - If save_path is None, display mode is used with notebook and non_blocking settings.
    - If the save directory doesn't exist, falls back to display mode with a warning.
    """
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir == "":
            save_dir = "."
        if os.path.exists(save_dir):
            # Save mode - use off_screen rendering, no threading needed for saving
            return True, True, False, False
        else:
            # Directory doesn't exist, fall back to display mode
            print(
                f"Warning: Directory '{save_dir}' does not exist. "
                f"Displaying plot instead of saving."
            )
            return False, False, notebook, non_blocking
    else:
        # Display mode
        return False, False, notebook, non_blocking


###############################################################################################
def list_available_view_names(plotobj) -> List[str]:
    """
    List available view names for dynamic view selection.

    Returns
    -------
    List[str]
        Available view names that can be used in views parameter:
        ['Lateral', 'Medial', 'Dorsal', 'Ventral', 'Rostral', 'Caudal'].

    Examples
    --------
    >>> plotter = SurfacePlotter()
    >>> view_names = visutils.list_available_view_names(plotter)
    >>> print(f"Available views: {view_names}")
    """

    view_names = list(plotobj._view_name_mapping.keys())
    view_names_capitalized = [name.capitalize() for name in view_names]

    print("🧠 Available View Names for Dynamic Selection:")
    print("=" * 50)
    for i, (name, titles) in enumerate(plotobj._view_name_mapping.items(), 1):
        print(f"{i:2d}. {name.capitalize():8s} → {', '.join(titles)}")

    print("\n💡 Usage Examples:")
    print(
        "   views=['Lateral', 'Medial']           # Shows both hemispheres lateral and medial"
    )
    print("   views=['Dorsal', 'Ventral']           # Shows top and bottom views")
    print("   views=['Lateral', 'Medial', 'Dorsal'] # Custom 3-view layout")
    print("   views=['Rostral', 'Caudal']           # Shows front and back views")
    print("=" * 50)

    return view_names_capitalized


###############################################################################################
def list_available_layouts(plotobj) -> Dict[str, Dict[str, Any]]:
    """
    Display available visualization layouts and their configurations.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary containing detailed layout information for each configuration.
        Keys are configuration names, values contain shape, window_size,
        num_views, and views information.

    Examples
    --------
    >>> plotter = SurfacePlotter("configs.json")
    >>> layouts = visutils.list_available_layouts(plotter)
    >>> print(f"Available layouts: {list(layouts.keys())}")
    >>>
    >>> # Access specific layout info
    >>> layout_info = layouts['8_views']
    >>> print(f"Shape: {layout_info['shape']}")
    >>> print(f"Views: {layout_info['num_views']}")
    """

    layout_info = {}

    print("Available Brain Visualization Layouts:")
    print("=" * 50)

    for views, config in plotobj.layouts_conf.items():
        shape = config["shape"]
        ly_views = config["views"]
        num_views = len(ly_views)

        print(f"\n📊 {ly_views}")
        print(f"   Shape: {shape[0]}x{shape[1]} ({num_views} views)")

        # Create an auxiliary array with subplot positions e.g()
        positions = list(np.ndindex(*shape))

        print("   Subplot Positions:")
        for i, pos in enumerate(positions):
            if i < num_views:
                print(f"     {i+1:2d}. Position {pos} → {ly_views[i]}")
            else:
                print(f"     {i+1:2d}. Position {pos} → (empty)")

        # Store in return dictionary
        layout_info[views] = {
            "shape": shape,
            "num_views": num_views,
            "views": positions,
        }

    print("\n" + "=" * 50)
    print("\n🎯 Dynamic View Selection:")
    print("   You can also use a list of view names for custom layouts:")
    print("   Available view names: Lateral, Medial, Dorsal, Ventral, Rostral, Caudal")
    print("   Example: views=['Lateral', 'Medial', 'Dorsal']")
    print("=" * 50)

    return layout_info


###############################################################################################
def get_layout_details(plotobj, views: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific layout configuration.

    Parameters
    ----------
    views : str
        Name of the configuration to examine.

    Returns
    -------
    Dict[str, Any] or None
        Detailed configuration information if found, None if configuration
        doesn't exist. Contains shape, window_size, and views information.

    Examples
    --------

    >>> plotter = SurfacePlotter("configs.json")
    >>> details = visutils.get_layout_details("8_views")
    >>> if details:
    ...     print(f"Grid shape: {details['shape']}")
    ...     print(f"Views: {len(details['views'])}")
    >>>
    >>> # Handle non-existent configuration
    >>> details = plotter.get_layout_details("invalid_config")
    """

    if views not in plotobj.layouts_conf:
        print(f"❌ Configuration '{views}' not found!")
        print(f"Available configs: {list(self.layouts_conf.keys())}")
        return None

    config = plotobj.layouts_conf[views]
    shape = config["shape"]

    print(f"🧠 Layout Details: {views}")
    print("=" * 40)
    print(f"Grid Shape: {shape[0]} rows × {shape[1]} columns")
    print(f"Total Views: {len(config['views'])}")
    print("\nView Details:")

    positions = list(np.ndindex(*shape))

    for i, view in enumerate(config["views"]):
        pos = positions[i - 1]
        if "merg" in view:
            # Substitute the word 'merge' with lh
            view = view.replace("merg", "lh")

        tmp_view = plotobj.views_conf.get(view, {})
        tmp_title = tmp_view["title"].capitalize()

        if "lh-" in view:
            tmp_title = "Left hemisphere: " + tmp_view["title"].capitalize()
        elif "rh-" in view:
            tmp_title = "Right hemisphere: " + tmp_view["title"].capitalize()

        print(f"  {i:2d}. Position ({pos[0]},{pos[1]}): {tmp_title}")
        print(
            f"      Camera: az={tmp_view['azimuth']}°, el={tmp_view['elevation']}°, zoom={tmp_view['zoom']}"
        )

    return config


###############################################################################################
def reload_config(plotobj) -> None:
    """
    Reload the configuration file to pick up any changes.

    Useful when modifying configuration files during development.

    Raises
    ------
    FileNotFoundError
        If the configuration file no longer exists.

    json.JSONDecodeError
        If the configuration file contains invalid JSON.

    KeyError
        If required configuration keys 'figure_conf' or 'views_conf' are missing.

    Examples
    --------
    >>> plotter = SurfacePlotter("configs.json")
    >>> # ... modify configs.json externally ...
    >>> plotter.reload_config()  # Pick up the changes
    """

    print(f"Reloading configuration from: {plotobj.config_file}")

    # Create attributes
    plotobj.figure_conf = configs["figure_conf"]
    plotobj.views_conf = configs["views_conf"]
    plotobj.layouts_conf = configs["layouts_conf"]
    plotobj.themes_conf = configs["themes_conf"]


###############################################################################################
def get_figure_config(plotobj) -> Dict[str, Any]:
    """
    Get the current figure configuration settings.

    Parameters
    ----------
    plotobj : SurfacePlotter or TractogramPlotter
        Instance of the plotting class containing figure configuration.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all figure styling settings including
        background color, font settings, mesh properties, and colorbar options.

    Examples
    --------
    >>> import cltvis.utils as visutils
    >>> plotter = SurfacePlotter("configs.json")
    >>> fig_config = visutils.get_figure_config(plotter)
    >>> print(fig_config)
    """

    print("🎨 Current Figure Configuration:")
    print("=" * 40)
    print("Background & Colors:")
    print(f"  Background Color: {plotobj.figure_conf['background_color']}")
    print(f"  Title Color: {plotobj.figure_conf['title_font_color']}")
    print(f"  Colorbar Color: {plotobj.figure_conf['colorbar_font_color']}")

    print("\nTitle Settings:")
    print(f"  Font Type: {plotobj.figure_conf['title_font_type']}")
    print(f"  Font Size: {plotobj.figure_conf['title_font_size']}")
    print(f"  Shadow: {plotobj.figure_conf['title_shadow']}")

    print("\nColorbar Settings:")
    print(f"  Font Type: {plotobj.figure_conf['colorbar_font_type']}")
    print(f"  Font Size: {plotobj.figure_conf['colorbar_font_size']}")
    print(f"  Title Font Size: {plotobj.figure_conf['colorbar_title_font_size']}")
    print(f"  Outline: {plotobj.figure_conf['colorbar_outline']}")
    print(f"  Number of Labels: {plotobj.figure_conf['colorbar_n_labels']}")

    print("\nMesh Properties:")
    print(f"  Ambient: {plotobj.figure_conf['mesh_ambient']}")
    print(f"  Diffuse: {plotobj.figure_conf['mesh_diffuse']}")
    print(f"  Specular: {plotobj.figure_conf['mesh_specular']}")
    print(f"  Specular Power: {plotobj.figure_conf['mesh_specular_power']}")
    print(f"  Smooth Shading: {plotobj.figure_conf['mesh_smooth_shading']}")

    print("=" * 40)
    return plotobj.figure_conf.copy()


###############################################################################################
def list_all_views_and_layouts(plotobj) -> List[str]:
    """
    List available layout configurations from the loaded JSON file.

    Parameters
    ----------
    plotobj : SurfacePlotter or TractogramPlotter
        Instance of the plotting class containing view configurations.

    Returns
    -------
    List[str]
        List of configuration names available for plotting.

    Examples
    --------
    >>> plotter = SurfacePlotter("configs.json")
    >>> layouts = list_all_views_and_layouts()
    >>> print(layouts)
    ['8_views', '8_views_8x1', '8_views_1x8', '6_views', '6_views_6x1', '6_views_1x6', '4_views', '4_views_4x1', '4_views_1x4', '2_views', 'lateral', 'medial', 'dorsal', 'ventral', 'rostral', 'caudal']
    """

    all_views_and_layouts = list_multiviews_layouts(plotobj) + list_single_views(
        plotobj
    )

    return all_views_and_layouts


###############################################################################################
def list_multiviews_layouts(plotobj) -> List[str]:
    """
    List available multi-view configurations from the loaded JSON file.

    Returns
    -------
    List[str]
        List of multi-view configuration names available for plotting.

    Examples
    --------
    >>> plotter = SurfacePlotter("configs.json")
    >>> multiviews = plotter._list_multiviews_layouts()
    >>> print(multiviews)
    ['8_views', '6_views', '4_views', '8_views_8x1', '6_views_6x1', '4_views_4x1', '8_views_1x8', '6_views_1x6', '4_views_1x4', '2_views']
    """

    return [name for name in plotobj.layouts_conf.keys()]


###############################################################################################
def list_single_views(plotobj) -> List[str]:
    """
    List available single view names.

    """

    all_single_views = plotobj.views_conf.keys()

    # Remove the hemisphere information from the view names
    single_views = []
    for i, view in enumerate(all_single_views):
        # Remove the hemisphere information from the view names
        if view.startswith("lh-"):
            view = view.replace("lh-", "")

            single_views.append(view)

    return single_views


################################################################################################
def get_valid_views(plotobj, views: Union[str, List[str]]) -> List[str]:
    """
    Get valid view names from the provided views parameter.

    Parameters
    ----------
    views : str or List[str]
        Either a single view name or a list of view names.

    Returns
    -------
    List[str]
        List of valid view names.

    Raises
    ------
    ValueError
        If no valid views are found.

    Examples
    --------
    >>> plotter = SurfacePlotter("configs.json")
    >>> valid_views = plotter._get_valid_views("8_views")
    >>> print(valid_views)
    ['lateral', 'medial', 'dorsal', 'ventral', 'rostral', 'caudal']
    """
    # Configure views
    if isinstance(views, str):
        views = [views]  # Convert single string to list for consistency

    # Lowrcase views for consistency
    views = [v.lower() for v in views]

    # Get the multiviews layouts
    multiviews_layouts = list_multiviews_layouts(plotobj)

    # Get the single views
    single_views = list_single_views(plotobj)

    # Check if all the views are valid
    valid_views = cltmisc.list_intercept(views, multiviews_layouts + single_views)

    if len(valid_views) == 0:
        raise ValueError(
            f"No valid views found in '{views}'. "
            f"Available options for multi-views layouts: {list_multiviews_layouts(plotobj)}"
            f" and for single views: {list_single_views(plotobj)}"
        )

    multiv_cont = 0
    for v_view in valid_views:
        # Check it there are many multiple views. They are the one different from
        # ["lateral", "medial", "dorsal", "ventral", "rostral", "caudal"]
        if v_view not in single_views:
            multiv_cont += 1

    if multiv_cont > 1:
        # If there are multiple multi-view layouts, we cannot proceed
        raise ValueError(
            f"Different multi-views layout cannot be supplied together. "
            "If you want to use a multi-views layout, please use only one multi-views layout "
            "from the list: "
            f"{list_multiviews_layouts(plotobj)}. "
            f"Received: {valid_views}"
        )
    elif multiv_cont == 1 and len(valid_views) > 1:
        # If there is only one multi-view layout, we can proceed
        print(
            f"Warning: Using a multi-views layout '{valid_views}' together with other views. "
            "The multi-views layout will be used as the main layout, "
            "and the other views will be ignored."
        )
        valid_views = cltmisc.list_intercept(valid_views, multiviews_layouts)

    elif multiv_cont == 0 and len(valid_views) > 0:

        # If there are no multi-view layouts, we can proceed with single views
        valid_views = cltmisc.list_intercept(valid_views, single_views)

    return valid_views


###############################################################################################
def update_figure_config(plotobj, auto_save: bool = True, **kwargs) -> None:
    """
    Update figure configuration parameters with validation and automatic saving.

    This method allows you to easily customize the visual appearance of your
    brain plots by updating styling parameters like colors, fonts, and mesh properties.

    Parameters
    ----------
    plotobj : SurfacePlotter or TractogramPlotter
        Instance of the plotting class containing the figure configuration.

    auto_save : bool, default True
        Whether to automatically save changes to the JSON configuration file.

    **kwargs : dict
        Figure configuration parameters to update. Valid parameters include:

        **Background & Colors:**
        - background_color : str (e.g., "black", "white", "#1e1e1e")
        - title_font_color : str (e.g., "white", "black", "#ffffff")
        - colorbar_font_color : str (e.g., "white", "black", "#ffffff")

        **Title Settings:**
        - title_font_type : str (e.g., "arial", "times", "courier")
        - title_font_size : int (6-30, default: 10)
        - title_shadow : bool (True/False)

        **Colorbar Settings:**
        - colorbar_font_type : str (e.g., "arial", "times", "courier")
        - colorbar_font_size : int (6-20, default: 10)
        - colorbar_title_font_size : int (8-25, default: 15)
        - colorbar_outline : bool (True/False)
        - colorbar_n_labels : int (3-15, default: 11)

        **Mesh Properties:**
        - mesh_ambient : float (0.0-1.0, default: 0.2)
        - mesh_diffuse : float (0.0-1.0, default: 0.5)
        - mesh_specular : float (0.0-1.0, default: 0.5)
        - mesh_specular_power : int (1-100, default: 50)
        - mesh_smooth_shading : bool (True/False)

    Raises
    ------
    ValueError
        If invalid parameter names or values are provided.

    TypeError
        If parameter values are of incorrect type.

    Examples
    --------
    >>> plotter = SurfacePlotter("configs.json")
    >>>
    >>> # Change background to white with black text
    >>> plotter.update_figure_config(
    ...     background_color="white",
    ...     title_font_color="black",
    ...     colorbar_font_color="black"
    ... )
    >>>
    >>> # Increase font sizes
    >>> plotter.update_figure_config(
    ...     title_font_size=14,
    ...     colorbar_font_size=12,
    ...     colorbar_title_font_size=18
    ... )
    >>>
    >>> # Adjust mesh lighting for better visibility
    >>> plotter.update_figure_config(
    ...     mesh_ambient=0.3,
    ...     mesh_diffuse=0.7,
    ...     mesh_specular=0.2
    ... )
    """

    # Define valid parameters with their types and ranges
    valid_params = {
        # Background & Colors
        "background_color": {"type": str, "example": '"black", "white", "#1e1e1e"'},
        "title_font_color": {"type": str, "example": '"white", "black", "#ffffff"'},
        "colorbar_font_color": {
            "type": str,
            "example": '"white", "black", "#ffffff"',
        },
        # Title Settings
        "title_font_type": {"type": str, "example": '"arial", "times", "courier"'},
        "title_font_size": {"type": int, "range": (6, 30), "default": 10},
        "title_shadow": {"type": bool, "example": "True, False"},
        # Colorbar Settings
        "colorbar_font_type": {
            "type": str,
            "example": '"arial", "times", "courier"',
        },
        "colorbar_font_size": {"type": int, "range": (6, 20), "default": 10},
        "colorbar_title_font_size": {"type": int, "range": (8, 25), "default": 15},
        "colorbar_outline": {"type": bool, "example": "True, False"},
        "colorbar_n_labels": {"type": int, "range": (3, 15), "default": 11},
        # Mesh Properties
        "mesh_ambient": {"type": float, "range": (0.0, 1.0), "default": 0.2},
        "mesh_diffuse": {"type": float, "range": (0.0, 1.0), "default": 0.5},
        "mesh_specular": {"type": float, "range": (0.0, 1.0), "default": 0.5},
        "mesh_specular_power": {"type": int, "range": (1, 100), "default": 50},
        "mesh_smooth_shading": {"type": bool, "example": "True, False"},
    }

    if not kwargs:
        print("No parameters provided to update.")
        print("Use plotter.list_figure_config_options() to see available parameters.")
        return

    # Validate and update parameters
    updated_params = []
    for param, value in kwargs.items():
        if param not in valid_params:
            available_params = list(valid_params.keys())
            raise ValueError(
                f"Invalid parameter '{param}'. "
                f"Available parameters: {available_params}"
            )

        # Type validation
        expected_type = valid_params[param]["type"]
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Parameter '{param}' must be of type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

        # Range validation for numeric types
        if "range" in valid_params[param]:
            min_val, max_val = valid_params[param]["range"]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Parameter '{param}' must be between {min_val} and {max_val}, "
                    f"got {value}"
                )

        # Update the configuration
        old_value = plotobj.figure_conf.get(param, "Not set")
        plotobj.figure_conf[param] = value
        updated_params.append(f"  {param}: {old_value} → {value}")

    # Display update summary
    print("✅ Figure configuration updated:")
    print("\n".join(updated_params))

    # Auto-save if requested
    if auto_save:
        plotobj.save_config()
        print(f"💾 Changes saved to: {plotobj.config_file}")


def apply_theme(plotobj, theme_name: str, auto_save: bool = True) -> None:
    """
    Apply predefined visual themes to quickly customize plot appearance.

    Parameters
    ----------
    plotobj : SurfacePlotter or TractogramPlotter
        Instance of the plotting class containing the figure configuration.

    theme_name : str
        Name of the theme to apply. Available themes:
        - "dark" : Dark background with white text (default)
        - "light" : Light background with dark text
        - "high_contrast" : Maximum contrast for presentations
        - "minimal" : Clean, minimal styling
        - "publication" : Optimized for academic publications
        - "colorful" : Vibrant colors for engaging visuals

    auto_save : bool, default True
        Whether to automatically save theme to configuration file.

    Raises
    ------
    ValueError
        If theme_name is not recognized.

    Examples
    --------
    >>> plotter = SurfacePlotter("configs.json")
    >>>
    >>> # Apply light theme for presentations
    >>> plotter.apply_theme("light")
    >>>
    >>> # Use high contrast for better visibility
    >>> plotter.apply_theme("high_contrast")
    >>>
    >>> # Publication-ready styling
    >>> plotter.apply_theme("publication")
    """

    themes = plotobj.themes_conf

    if theme_name not in themes:
        available_themes = list(themes.keys())
        raise ValueError(
            f"Theme '{theme_name}' not recognized. "
            f"Available themes: {available_themes}"
        )

    theme = themes[theme_name].copy()
    description = theme.pop("description")

    # Apply theme parameters (excluding description)
    print(f"🎨 Applying '{theme_name}' theme: {description}")

    updated_params = []
    for param, value in theme.items():
        old_value = plotobj.figure_conf.get(param, "Not set")
        plotobj.figure_conf[param] = value
        updated_params.append(f"  {param}: {old_value} → {value}")

    print("Updated parameters:")
    print("\n".join(updated_params))

    if auto_save:
        save_config(plotobj)
        print(f"💾 Theme saved to: {plotobj.config_file}")


################################################################################################
def list_available_themes(plotobj) -> None:
    """
    Display all available themes with descriptions and previews.
    Shows theme names, descriptions, and usage examples to help users
    choose the right theme for their visualization needs.

    Parameters
    ----------
    plotobj : SurfacePlotter or TractogramPlotter
        Instance of the plotting class containing the figure configuration.


    Examples
    --------
    >>> plotter = SurfacePlotter("configs.json")
    >>> plotter.list_available_themes()
    """

    themes = {
        "dark": "Dark background with white text (default)",
        "light": "Light background with dark text",
        "high_contrast": "Maximum contrast for presentations",
        "minimal": "Clean, minimal styling",
        "publication": "Optimized for academic publications",
        "colorful": "Vibrant colors for engaging visuals",
    }

    print("🎨 Available Themes:")
    print("=" * 50)
    for i, (theme_name, description) in enumerate(themes.items(), 1):
        print(f"{i:2d}. {theme_name:12s} - {description}")

    print("\n💡 Usage:")
    print("   plotter.apply_theme('light')     # Apply light theme")
    print("   plotter.apply_theme('publication', auto_save=False)  # Don't save")
    print("=" * 50)


################################################################################################
def list_figure_config_options(plotobj) -> None:
    """
    Display all available figure configuration parameters with descriptions.

    Shows parameter names, types, valid ranges, and examples to help users
    understand what can be customized.

    Examples
    --------
    >>> plotter = SurfacePlotter("configs.json")
    >>> plotter.list_figure_config_options()
    """

    print("🎛️  Available Figure Configuration Parameters:")
    print("=" * 60)

    categories = {
        "Background & Colors": [
            (
                "background_color",
                "str",
                "Background color",
                '"black", "white", "#1e1e1e"',
            ),
            (
                "title_font_color",
                "str",
                "Title text color",
                '"white", "black", "#ffffff"',
            ),
            (
                "colorbar_font_color",
                "str",
                "Colorbar text color",
                '"white", "black", "#ffffff"',
            ),
        ],
        "Title Settings": [
            (
                "title_font_type",
                "str",
                "Title font family",
                '"arial", "times", "courier"',
            ),
            ("title_font_size", "int", "Title font size (6-30)", "10, 12, 14"),
            ("title_shadow", "bool", "Enable title shadow", "True, False"),
        ],
        "Colorbar Settings": [
            (
                "colorbar_font_type",
                "str",
                "Colorbar font family",
                '"arial", "times", "courier"',
            ),
            (
                "colorbar_font_size",
                "int",
                "Colorbar font size (6-20)",
                "10, 12, 14",
            ),
            (
                "colorbar_title_font_size",
                "int",
                "Colorbar title size (8-25)",
                "15, 18, 20",
            ),
            ("colorbar_outline", "bool", "Show colorbar outline", "True, False"),
            (
                "colorbar_n_labels",
                "int",
                "Number of colorbar labels (3-15)",
                "11, 7, 5",
            ),
        ],
        "Mesh Properties": [
            (
                "mesh_ambient",
                "float",
                "Ambient lighting (0.0-1.0)",
                "0.2, 0.3, 0.4",
            ),
            (
                "mesh_diffuse",
                "float",
                "Diffuse lighting (0.0-1.0)",
                "0.5, 0.6, 0.7",
            ),
            (
                "mesh_specular",
                "float",
                "Specular reflection (0.0-1.0)",
                "0.5, 0.3, 0.7",
            ),
            ("mesh_specular_power", "int", "Specular power (1-100)", "50, 30, 80"),
            ("mesh_smooth_shading", "bool", "Enable smooth shading", "True, False"),
        ],
    }

    for category, params in categories.items():
        print(f"\n📁 {category}:")
        print("-" * 40)
        for param, param_type, description, examples in params:
            current_value = plotobj.figure_conf.get(param, "Not set")
            print(f"  {param:25s} ({param_type:5s}) - {description}")
            print(f"  {'':25s} Current: {current_value}, Examples: {examples}")
            print()

    print("💡 Usage Examples:")
    print("   plotter.update_figure_config(background_color='white')")
    print("   plotter.update_figure_config(title_font_size=14, mesh_ambient=0.3)")
    print("   plotter.update_figure_config(auto_save=False, **params)")
    print("=" * 60)


###############################################################################################
def reset_figure_config(plotobj, auto_save: bool = True) -> None:
    """
    Reset figure configuration to default values.

    Parameters
    ----------
    plotobj : SurfacePlotter or TractogramPlotter
        Instance of the plotting class containing the figure configuration.

    auto_save : bool, default True
        Whether to automatically save reset configuration to file.

    Examples
    --------
    >>> plotter = SurfacePlotter("configs.json")
    >>> plotter.reset_figure_config()  # Reset to defaults
    """

    default_config = {
        "background_color": "black",
        "title_font_type": "arial",
        "title_font_size": 10,
        "title_font_color": "white",
        "title_shadow": True,
        "colorbar_font_type": "arial",
        "colorbar_font_size": 10,
        "colorbar_title_font_size": 15,
        "colorbar_font_color": "white",
        "colorbar_outline": False,
        "colorbar_n_labels": 11,
        "mesh_ambient": 0.2,
        "mesh_diffuse": 0.5,
        "mesh_specular": 0.5,
        "mesh_specular_power": 50,
        "mesh_smooth_shading": True,
    }

    print("🔄 Resetting figure configuration to defaults...")

    # Show what's changing
    changes = []
    for param, default_value in default_config.items():
        old_value = plotobj.figure_conf.get(param, "Not set")
        if old_value != default_value:
            changes.append(f"  {param}: {old_value} → {default_value}")

    if changes:
        print("Changes:")
        print("\n".join(changes))
    else:
        print("Configuration already at default values.")

    # Apply defaults
    plotobj.figure_conf.update(default_config)

    if auto_save:
        plotobj.save_config()
        print(f"💾 Default configuration saved to: {self.config_file}")


###############################################################################################
def save_config(plotobj) -> None:
    """
    Save current configuration (both figure_conf and views_conf) to JSON file.

    Raises
    ------
    IOError
        If unable to write to configuration file.

    Examples
    --------
    >>> plotter = SurfacePlotter("configs.json")
    >>> plotter.update_figure_config(background_color="white", auto_save=False)
    >>> plotter.save_config()  # Manually save changes
    """

    try:
        # Combine both configurations
        complete_config = {
            "figure_conf": plotobj.figure_conf,
            "views_conf": plotobj.views_conf,
        }

        # Write to file with proper formatting
        with open(plotobj.config_file, "w") as f:
            json.dump(complete_config, f, indent=4, sort_keys=False)

        print(f"✅ Configuration saved successfully to: {plotobj.config_file}")

    except Exception as e:
        raise IOError(f"Failed to save configuration: {e}")


################################################################################################
def preview_theme(plotobj, theme_name: str) -> None:
    """
    Preview a theme's parameters without applying them.

    Parameters
    ----------
    plotobj : SurfacePlotter or TractogramPlotter
        Instance of the plotting class containing the figure configuration.

    theme_name : str
        Name of the theme to preview.

    Examples
    --------
    >>> plotter = SurfacePlotter("configs.json")
    >>> plotter.preview_theme("light")  # See what light theme would change
    """

    themes = {
        "dark": {
            "background_color": "black",
            "title_font_color": "white",
            "colorbar_font_color": "white",
            "title_shadow": True,
            "colorbar_outline": False,
            "mesh_ambient": 0.2,
            "description": "Dark background with white text (default)",
        },
        "light": {
            "background_color": "white",
            "title_font_color": "black",
            "colorbar_font_color": "black",
            "title_shadow": False,
            "colorbar_outline": True,
            "mesh_ambient": 0.3,
            "description": "Light background with dark text",
        },
        # ... (other themes would be included here)
    }

    if theme_name not in themes:
        available_themes = list(themes.keys())
        raise ValueError(
            f"Theme '{theme_name}' not found. Available: {available_themes}"
        )

    theme = themes[theme_name].copy()
    description = theme.pop("description")

    print(f"👀 Preview of '{theme_name}' theme: {description}")
    print("=" * 50)
    print("Would change:")

    for param, new_value in theme.items():
        current_value = self.figure_conf.get(param, "Not set")
        if current_value != new_value:
            print(f"  {param:25s}: {current_value} → {new_value}")

    print("\n💡 To apply: plotter.apply_theme('{}')".format(theme_name))
    print("=" * 50)


################################################################################################
def get_shared_limits(
    objs2plot: Union[List[cltsurf.Surface], List[clttract.Tractogram]],
    map_name: str,
    vmin: float,
    vmax: float,
) -> Tuple[float, float]:
    """
    Get shared vmin and vmax from surfaces if not provided.

    Parameters
    ----------
    objs2plot : list
        List of Surface or Tractogram objects to plot.

    map_name : str
        Name of the data array to use for color mapping.

    vmin : float or None
        Minimum value for color mapping. If None, will be computed.

    vmax : float or None
        Maximum value for color mapping. If None, will be computed.

    Returns
    -------
    Tuple[float, float]
        (vmin, vmax) values for color mapping.

    Raises
    ------
    ValueError
        If no data found for the specified map_name.

    KeyError
        If map_name is not found in any of the objects.

    TypeError
        If objs2plot contains unsupported object types.

    Examples
    --------
    >>> vmin, vmax = _get_shared_limits([surf1, surf2], "thickness", None, None)
    >>> print(f"Shared limits: vmin={vmin}, vmax={vmax}")


    """

    if not isinstance(objs2plot, list):
        objs2plot = [objs2plot]

    # Concatenate data from all the surfaces/tractograms
    for i, obj2plot in enumerate(objs2plot):
        if isinstance(obj2plot, cltsurf.Surface):
            if map_name in obj2plot.mesh.point_data:
                if i == 0:
                    data = obj2plot.mesh.point_data[map_name]
                else:
                    data = np.concatenate((data, obj2plot.mesh.point_data[map_name]))

        elif isinstance(obj2plot, clttract.Tractogram):
            if map_name in (obj2plot.data_per_point.keys()):
                if i == 0:
                    data = obj2plot.data_per_point[map_name]
                else:
                    data = np.concatenate((data, obj2plot.data_per_point[map_name]))

            elif map_name in (obj2plot.data_per_streamline.keys()):
                if i == 0:
                    data = obj2plot.data_per_streamline[map_name]
                else:
                    data = np.concatenate(
                        (data, obj2plot.data_per_streamline[map_name])
                    )

        else:
            raise TypeError(
                f"Unsupported object type: {type(obj2plot)}. "
                "Expected cltsurf.Surface or clttract.Tractogram."
            )

    # Ensure data was found
    if "data" not in locals():
        raise ValueError(
            f"No data found for map_name '{map_name}' in the provided objects."
        )

    # Compute vmin and vmax if not provided
    if vmin is None:
        vmin = np.min(data)

    if vmax is None:
        vmax = np.max(data)

    return vmin, vmax


################################################################################################
def get_map_limits(
    objs2plot: Union[List[cltsurf.Surface], List[clttract.Tractogram]],
    map_name: str,
    colormap_style: str,
    v_limits: Tuple[Optional[float], Optional[float]],
) -> List[Tuple[float, float, str]]:
    """
    Get real vmin and vmax from surfaces if not provided.

    Parameters
    ----------
    objs2plot : list
        List of Surface or Tractogram objects to plot, or a single Surface.

    map_name : str
        Name of the data array to use for color mapping.

    colormap_style : str
        "individual" for separate limits per object, "shared" for same limits.

    v_limits : tuple
        (vmin, vmax) values for color mapping. Use None to compute automatically.

    Returns
    -------
    list of tuples
        List of (vmin, vmax, map_name) tuples for each object.

    Raises
    """
    vmin, vmax = v_limits
    real_limits = []

    if not isinstance(objs2plot, list):
        objs2plot = [objs2plot]

    if isinstance(map_name, list):
        map_name = map_name[0]

    if colormap_style == "individual":
        for obj2plot in objs2plot:
            if isinstance(obj2plot, cltsurf.Surface):
                data = obj2plot.mesh.point_data[map_name]

            elif isinstance(obj2plot, clttract.Tractogram):
                if map_name in (obj2plot.data_per_point.keys()):
                    data = obj2plot.data_per_point[map_name]
                elif map_name in (obj2plot.data_per_streamline.keys()):
                    data = obj2plot.data_per_streamline[map_name]
                else:
                    raise KeyError(
                        f"Map name '{map_name}' not found in Tractogram data."
                    )
            else:
                raise TypeError(
                    f"Unsupported object type: {type(obj2plot)}. "
                    "Expected cltsurf.Surface or clttract.Tractogram."
                )

            # Ensure data was found
            if "data" not in locals():
                raise ValueError(
                    f"No data found for map_name '{map_name}' in the provided objects."
                )
            # Concatenate if data is a list of arrays
            if isinstance(data, list):
                data = np.concatenate(data)

            if isinstance(data, ArraySequence):
                data = np.concatenate(data)

            # Compute vmin and vmax if not provided
            if vmin is None:
                real_vmin = np.min(data)
            else:
                real_vmin = vmin

            if vmax is None:
                real_vmax = np.max(data)
            else:
                real_vmax = vmax

            real_limits.append((real_vmin, real_vmax, map_name))
        return real_limits
    else:  # shared
        vmin, vmax = get_shared_limits(objs2plot, map_name, vmin, vmax)
        return [(vmin, vmax, map_name)] * len(objs2plot)
