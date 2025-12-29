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
import copy

from nibabel.streamlines import ArraySequence

# Importing local modules
from . import misctools as cltmisc
from . import plottools as cltplot

# Use TYPE_CHECKING to avoid circular imports
from . import surfacetools as cltsurf
from . import tracttools as clttract
from . import pointstools as cltpts
from . import colorstools as cltcolor


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
    >>> plotter = BrainPlotter("configs.json")
    >>> plotter._load_configs()  # Reloads configurations from file
    """

    configs = cltmisc.load_json(config_file)

    # Validate structure and load configurations
    if "figure_conf" not in configs:
        raise KeyError("Missing 'figure_conf' key in configuration file")

    if "views_conf" not in configs:
        raise KeyError("Missing 'views_conf' key in configuration file")

    return configs


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
    plotobj : BrainPlotter
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

    # def flatten_objects(obj_list):
    #     """Generator that yields all individual objects from nested lists"""
    #     for item in obj_list:
    #         if isinstance(item, list):
    #             yield from flatten_objects(item)
    #         else:
    #             yield item

    # Check if any map is not in any object's colortables
    for obj in flatten_objects(plotsobj):
        if hasattr(obj, "colortables"):
            for map_name in maps_names:
                if map_name not in obj.colortables:
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


#################################################################################################
def prepare_list_obj_for_plotting(
    obj2plot: Union[List[cltsurf.Surface], List[clttract.Tractogram]],
    map_name: str,
    colormap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
    range_color: Tuple = (128, 128, 128, 255),
) -> Union[cltsurf.Surface, clttract.Tractogram]:
    """
    Prepare a list of Surface or Tractogram objects for plotting with color mapping.

    Parameters
    ----------
    obj2plot : Union[List[cltsurf.Surface], List[clttract.Tractogram]]
        The list of objects to prepare for plotting. Can be a list of Surface or Tractogram instances.

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
    Union[List[cltsurf.Surface], List[clttract.Tractogram]]
        The prepared list of objects with color mapping applied.


    """
    if not isinstance(obj2plot, list):
        obj2plot = [obj2plot]

    prepared_objects = []
    for obj in obj2plot:
        if not isinstance(
            obj, (cltsurf.Surface, clttract.Tractogram, cltpts.PointCloud)
        ):
            raise TypeError(
                "All objects in obj2plot must be Surface or Tractogram instances"
            )

        prepared_obj = prepare_obj_for_plotting(
            obj,
            map_name,
            colormap,
            vmin,
            vmax,
            range_min,
            range_max,
            range_color,
        )
        prepared_objects.append(prepared_obj)

    return prepared_objects


################################################################################################
def prepare_obj_for_plotting(
    obj2plot: Union[cltsurf.Surface, clttract.Tractogram],
    map_name: str,
    colormap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
    range_color: Tuple = (128, 128, 128, 255),
) -> Union[cltsurf.Surface, clttract.Tractogram]:
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
        elif (
            map_name in obj2plot.data_per_streamline
            and map_name not in obj2plot.data_per_point
        ):
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
            map_name, colormap, vmin, vmax, range_min, range_max, range_color
        )

    elif isinstance(obj2plot, cltsurf.Surface):
        if vmin is None:
            if range_min is not None:
                vmin = range_min
            else:
                vmin = np.min(obj2plot.mesh.point_data[map_name])

        if vmax is None:
            if range_max is not None:
                vmax = range_max
            else:
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
            map_name, colormap, vmin, vmax, range_min, range_max, range_color
        )

        # # Apply gray color to values outside the specified range
        # if range_min is not None or range_max is not None:
        #     data_values = obj2plot.mesh.point_data[map_name]
        #     rgba_colors = obj2plot.mesh.point_data["rgba"]

        #     # Create mask for out-of-range values
        #     mask = np.zeros(len(data_values), dtype=bool)
        #     if range_min is not None:
        #         mask |= data_values < range_min
        #     if range_max is not None:
        #         mask |= data_values > range_max

        #     # Set out-of-range values to a specified color
        #     if rgba_colors.shape[1] == 4:  # RGBA
        #         rgba_colors[mask] = range_color

        #     elif rgba_colors.shape[1] == 3:  # RGB
        #         rgba_colors[mask] = range_color[:3]

        #     obj2plot.mesh.point_data["rgba"] = rgba_colors

    elif isinstance(obj2plot, cltpts.PointCloud):
        if vmin is None:
            if range_min is not None:
                vmin = range_min
            else:
                vmin = np.min(obj2plot.point_data[map_name])

        if vmax is None:
            if range_max is not None:
                vmax = range_max
            else:
                vmax = np.max(obj2plot.point_data[map_name])

        try:
            point_values = obj2plot.point_data[map_name]
            point_values = np.nan_to_num(
                point_values,
                nan=0.0,
            )  # Handle NaNs and infinities
            obj2plot.point_data[map_name] = point_values

        except KeyError:
            raise ValueError(f"Data array '{map_name}' not found in point cloud data")

        # Apply colors to point data
        obj2plot.point_data["rgba"] = obj2plot.get_pointwise_colors(
            map_name, colormap, vmin, vmax, range_min, range_max, range_color
        )

        # Apply gray color to values outside the specified range
        if range_min is not None or range_max is not None:
            data_values = obj2plot.point_data[map_name]
            rgba_colors = obj2plot.point_data["rgba"]

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

            obj2plot.point_data["rgba"] = rgba_colors

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
    >>> plotter = BrainPlotter()
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
    >>> plotter = BrainPlotter("configs.json")
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

    >>> plotter = BrainPlotter("configs.json")
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
    >>> plotter = BrainPlotter("configs.json")
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
    plotobj : BrainPlotter
        Instance of the plotting class containing figure configuration.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all figure styling settings including
        background color, font settings, mesh properties, and colorbar options.

    Examples
    --------
    >>> import cltvis.utils as visutils
    >>> plotter = BrainPlotter("configs.json")
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
    plotobj : BrainPlotter
        Instance of the plotting class containing view configurations.

    Returns
    -------
    List[str]
        List of configuration names available for plotting.

    Examples
    --------
    >>> plotter = BrainPlotter("configs.json")
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
    >>> plotter = BrainPlotter("configs.json")
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
    >>> plotter = BrainPlotter("configs.json")
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
    plotobj : BrainPlotter
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
    >>> plotter = BrainPlotter("configs.json")
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


def apply_theme(plotobj, theme_name: str, auto_save: bool = False) -> None:
    """
    Apply predefined visual themes to quickly customize plot appearance.

    Parameters
    ----------
    plotobj : BrainPlotter
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
    >>> plotter = BrainPlotter("configs.json")
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
    plotobj : BrainPlotter
        Instance of the plotting class containing the figure configuration.


    Examples
    --------
    >>> plotter = BrainPlotter("configs.json")
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
    >>> plotter = BrainPlotter("configs.json")
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
    plotobj : BrainPlotter
        Instance of the plotting class containing the figure configuration.

    auto_save : bool, default True
        Whether to automatically save reset configuration to file.

    Examples
    --------
    >>> plotter = BrainPlotter("configs.json")
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
    >>> plotter = BrainPlotter("configs.json")
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
    plotobj : BrainPlotter
        Instance of the plotting class containing the figure configuration.

    theme_name : str
        Name of the theme to preview.

    Examples
    --------
    >>> plotter = BrainPlotter("configs.json")
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
def get_shared_limits(objs2plot, map_name, vmin, vmax):
    """Get shared vmin and vmax across all objects."""

    all_objects = flatten_objects(objs2plot)
    all_min_values = []
    all_max_values = []

    for obj in all_objects:
        if isinstance(obj, cltsurf.Surface):
            data = obj.mesh.point_data[map_name]
        elif isinstance(obj, clttract.Tractogram):
            if map_name in obj.data_per_point.keys():
                data = obj.data_per_point[map_name]
            elif map_name in obj.data_per_streamline.keys():
                data = obj.data_per_streamline[map_name]
            else:
                raise KeyError(f"Map name '{map_name}' not found in Tractogram data.")

        # Handle list or ArraySequence data
        if isinstance(data, (list, ArraySequence)):
            data = np.concatenate(data)

        all_min_values.append(np.min(data))
        all_max_values.append(np.max(data))

    # Compute shared limits
    shared_vmin = np.min(all_min_values) if vmin is None else vmin
    shared_vmax = np.max(all_max_values) if vmax is None else vmax

    return shared_vmin, shared_vmax


################################################################################################
def get_plot_config_dimensions(limits: dict) -> tuple:
    """
    Get the number of objects, surfaces, and views from the limits dictionary.

    Parameters
    ----------
    limits : dict
        Dictionary where keys are (n_obj, n_surf, n_view) tuples.

    Returns
    -------
    tuple
        Tuple containing (n_obj, n_surf, n_view).
    Raises
    ------
    ValueError
        If limits dictionary is empty.

    ValueError
        If keys in limits dictionary are not tuples of length 3.

    """
    if not limits:
        raise ValueError("Limits dictionary is empty.")

    for key in limits.keys():
        if not (isinstance(key, tuple) and len(key) == 3):
            raise ValueError("Keys in limits dictionary must be tuples of length 3.")

    #
    tuples_list = list(limits.keys())
    result = tuple(max(values) for values in zip(*tuples_list))

    n_map = result[0] + 1
    n_surf = result[1] + 1
    n_view = result[2] + 1

    return (n_map, n_surf, n_view)


#################################################################################################
def get_map_characteristics(objs2plot, maps_dict: dict):
    """
    Compute vmin and vmax for each map based on the specified style.

    Parameters
    ----------
    objs2plot : list
        List of Surface or Tractogram objects to plot, or nested lists of objects.

    maps_dict : dict
        Dictionary where keys are map names and values.

    Returns
    -------
    limits_dict : dict
        Dictionary with vmin and vmax for each map and overall shared limits.

    charact_dict : dict
        Dictionary with colormap and colorbar title for each map and overall shared characteristics.

    Raises
    ------
    ValueError
        If an invalid style is provided.


    """

    maps_names = list(maps_dict)

    n_maps = len(maps_names)
    limits_dict = {}
    charact_dict = {}
    n_objs = len(objs2plot)

    cat_bool = True
    cont_global = 0
    for map_idx in range(n_maps):

        map_name = maps_names[map_idx]
        v_min = maps_dict[map_name]["vmin"]
        v_max = maps_dict[map_name]["vmax"]
        colormap = maps_dict[map_name]["colormap"]
        colorbar_title = maps_dict[map_name]["colorbar_title"]

        v_limits = (v_min, v_max)
        obj_limits = get_map_limits(objs2plot, map_name, v_limits)
        map_limits = {}
        map_limits["individual"] = obj_limits

        map_charact = {}
        map_charact["colormap"] = colormap
        map_charact["colorbar_title"] = colorbar_title

        if colormap == "colortable":
            map_limits["shared_by_map"] = (None, None, map_name)

        else:
            cat_bool = False
            cont = 0
            for surf_idx in range(n_objs):
                # Initialize overall shared limits

                if cont == 0:
                    # Initialize shared limits per map
                    shared_map_vmin = obj_limits[surf_idx][0]
                    shared_map_vmax = obj_limits[surf_idx][1]
                else:
                    # Update shared limits per map
                    shared_map_vmin = min(shared_map_vmin, obj_limits[surf_idx][0])
                    shared_map_vmax = max(shared_map_vmax, obj_limits[surf_idx][1])

                cont += 1

            if cont_global == 0:
                # Initialize overall shared limits
                shared_vmin = shared_map_vmin
                shared_vmax = shared_map_vmax

                shared_colormap = colormap
                shared_colorbar_title = colorbar_title

            else:

                # Update overall shared limits
                shared_vmin = min(shared_vmin, shared_map_vmin)
                shared_vmax = max(shared_vmax, shared_map_vmax)

                shared_colorbar_title = shared_colorbar_title + " + " + colorbar_title

            cont_global += 1

            map_limits["shared_by_map"] = (shared_map_vmin, shared_map_vmax, map_name)

        limits_dict[map_name] = map_limits
        charact_dict[map_name] = map_charact

    if cat_bool:
        limits_dict["shared"] = (None, None, map_name)
        charact_dict["shared"] = {
            "colormap": "colortable",
            "colorbar_title": map_name,
        }
    else:
        limits_dict["shared"] = (shared_vmin, shared_vmax, map_name)
        charact_dict["shared"] = {
            "colormap": shared_colormap,
            "colorbar_title": shared_colorbar_title,
        }

    return limits_dict, charact_dict


################################################################################################
def get_map_limits(
    objs2plot: Union[
        List[cltsurf.Surface], List[clttract.Tractogram], List[cltpts.PointCloud]
    ],
    map_name: str,
    v_limits: Tuple[Optional[float], Optional[float]],
) -> List[Tuple[float, float, str]]:
    """
    Get real vmin and vmax from surfaces if not provided.

    Parameters
    ----------
    objs2plot : list
        List of Surface or Tractogram objects to plot, or nested lists of objects.

    map_name : str
        Name of the data array to use for color mapping.

    v_limits : tuple
        (vmin, vmax) values for color mapping. Use None to compute automatically.

    Returns
    -------
    list of tuples
        List of (vmin, vmax, map_name) tuples for each object.

    Raises
    ------
    KeyError
        If map_name is not found in any object's data.


    """
    vmin, vmax = v_limits

    # Handle single object input
    if not isinstance(objs2plot, list):
        objs2plot = [objs2plot]

    # Handle list of map names
    if isinstance(map_name, list):
        map_name = map_name[0]

    real_limits = []

    # Process each element in objs2plot
    for obj in objs2plot:
        # If this element is a list, process all objects in it together
        if isinstance(obj, list):
            all_data = []
            for obj_sing in obj:
                data = get_data_from_object(obj_sing, map_name)
                # Concatenate if data is a list of arrays
                if isinstance(data, list):
                    data = np.concatenate(data)
                if isinstance(data, ArraySequence):
                    data = np.concatenate(data)
                all_data.append(data.flatten())

            # Concatenate all data from this sublist
            all_data = np.concatenate(all_data)

        else:
            # Single object - compute its min/max
            all_data = get_data_from_object(obj, map_name)
            # Concatenate if data is a list of arrays
            if isinstance(all_data, list):
                all_data = np.concatenate(all_data)
            if isinstance(all_data, ArraySequence):  # Fixed: was 'data'
                all_data = np.concatenate(all_data)

        # Compute vmin and vmax
        local_vmin = np.min(all_data) if vmin is None else vmin  # Fixed: was 'data'
        local_vmax = np.max(all_data) if vmax is None else vmax  # Fixed: was 'data'

        real_limits.append((local_vmin, local_vmax, map_name))

    return real_limits


################################################################################################
def flatten_objects(obj_list):
    """
    Recursively flatten nested lists and return all individual objects

    Parameters
    ----------
    obj_list : list
        List of objects or nested lists containing Tractogram/Surface objects

    Returns
    -------
    flattened : list
        Flattened list of individual objects

    """
    flattened = []
    for item in obj_list:
        if isinstance(item, list):
            flattened.extend(flatten_objects(item))
        else:
            flattened.append(item)
    return flattened


################################################################################################
def get_data_from_object(obj, map_name):
    """
    Extract data array from a single object

    Parameters
    ----------
    obj : cltsurf.Surface or clttract.Tractogram
        Object to extract data from.

    map_name : str
        Name of the data array to extract.

    Returns
    -------
    np.ndarray
        Data array corresponding to the map_name.

    Raises
    ------
    KeyError
        If map_name is not found in the object's data.


    """
    if isinstance(obj, cltsurf.Surface):
        return obj.mesh.point_data[map_name]

    elif isinstance(obj, cltpts.PointCloud):
        return obj.point_data[map_name]

    elif isinstance(obj, clttract.Tractogram):
        if map_name in obj.data_per_point.keys():
            return obj.data_per_point[map_name]
        elif map_name in obj.data_per_streamline.keys():
            return obj.data_per_streamline[map_name]
        else:
            raise KeyError(f"Map name '{map_name}' not found in Tractogram data.")
    else:
        raise TypeError(
            f"Unsupported object type: {type(obj)}. "
            "Expected cltsurf.Surface, cltpts.PointCloud, clttract.Tractogram."
        )


##################################################################################################
def create_default_object_config(obj2plot):
    """
    Create default configuration for each object to plot.

    Parameters:
    -----------
    obj2plot : list
        List of objects or nested lists containing Tractogram/Surface objects

    Returns:
    --------
    obj_characts_list : list
        List of dictionaries with default characteristics for each object.
    """
    all_objects = flatten_objects(obj2plot)
    maps_config_default = create_default_map_config(all_objects)

    obj_characts_list = []

    for sing_obj in all_objects:
        obj_characts = {}

        # Get map names and data based on object type
        if isinstance(sing_obj, clttract.Tractogram):
            map_list_dict = sing_obj.list_maps()
            st_maps = map_list_dict.get("maps_per_streamline", [])
            pt_maps = map_list_dict.get("maps_per_point", [])
            all_maps = st_maps + pt_maps

            def get_tract_data(map_name):
                """Get data for a tractogram map."""
                # Prioritize point data if available in both
                if map_name in pt_maps:
                    return np.concatenate(sing_obj.data_per_point[map_name])
                else:
                    return sing_obj.data_per_streamline[map_name]

            map_data_pairs = [(name, get_tract_data(name)) for name in all_maps]

        elif isinstance(sing_obj, cltsurf.Surface):
            all_maps = list(sing_obj.mesh.point_data.keys())
            map_data_pairs = [
                (name, sing_obj.mesh.point_data[name]) for name in all_maps
            ]

        elif isinstance(sing_obj, cltpts.PointCloud):
            all_maps = list(sing_obj.point_data.keys())
            map_data_pairs = [(name, sing_obj.point_data[name]) for name in all_maps]

        else:
            obj_characts_list.append(obj_characts)
            continue

        # Process all maps for this object
        for map_name, data in map_data_pairs:
            has_colortable = map_name in sing_obj.colortables

            if has_colortable:
                obj_characts[map_name] = {
                    "colormap": "colortable",
                    "v_limits": (None, None),
                    "v_range": (None, None),
                    "range_color": (128, 128, 128, 255),
                    "opacity": 1.0,
                    "colorbar": False,
                    "colorbar_title": map_name,
                }
            else:
                min_val = np.min(data)
                max_val = np.max(data)

                obj_characts[map_name] = {
                    "colormap": maps_config_default[map_name]["colormap"],
                    "v_limits": (min_val, max_val),
                    "v_range": (min_val, max_val),
                    "range_color": (128, 128, 128, 255),
                    "opacity": 1.0,
                    "colorbar": True,
                    "colorbar_title": map_name,
                }

        obj_characts_list.append(obj_characts)

    return obj_characts_list


#################################################################################################
def create_default_map_config(obj2plot):
    """
    Get all unique map names from a list of objects.

    Parameters:
    -----------
    obj2plot : list
        List of objects or nested lists containing Tractogram/Surface objects

    Returns:
    --------
    map_characts_dict : dict
        Dictionary of map characteristics including colormap, limits, ranges, object indices, etc.
    """
    all_objects = flatten_objects(obj2plot)
    map_characts_dict = {}

    def init_map_config(map_name):
        """Initialize map configuration if it doesn't exist."""
        if map_name not in map_characts_dict:
            map_characts_dict[map_name] = {
                "objects": [],
                "colormap": "viridis",
                "v_limits": [None, None],
                "v_range": [None, None],
                "range_color": (128, 128, 128, 255),
                "opacity": 1.0,
                "colorbar": True,
                "colorbar_title": map_name,
            }

    def update_range(map_name, min_val, max_val):
        """Update the min/max range for a map."""
        config = map_characts_dict[map_name]

        # Update min
        if config["v_range"][0] is None:
            config["v_range"][0] = min_val
        else:
            config["v_range"][0] = min(config["v_range"][0], min_val)

        # Update max
        if config["v_range"][1] is None:
            config["v_range"][1] = max_val
        else:
            config["v_range"][1] = max(config["v_range"][1], max_val)

        # Sync v_limits with v_range
        config["v_limits"] = config["v_range"].copy()

    def process_map(map_name, data, has_colortable, obj_idx):
        """Process a single map and update its configuration."""
        init_map_config(map_name)

        # Add object index if not already present
        if obj_idx not in map_characts_dict[map_name]["objects"]:
            map_characts_dict[map_name]["objects"].append(obj_idx)

        if has_colortable:
            map_characts_dict[map_name]["colormap"] = "colortable"
        else:
            min_val = data.min()
            max_val = data.max()
            update_range(map_name, min_val, max_val)

    # Process all objects
    for obj_idx, sing_obj in enumerate(all_objects):
        if isinstance(sing_obj, clttract.Tractogram):
            map_list_dict = sing_obj.list_maps()

            # Process streamline maps
            st_maps = map_list_dict.get("maps_per_streamline")
            if st_maps:
                for map_name in st_maps:
                    has_colortable = map_name in sing_obj.colortables
                    data = sing_obj.data_per_streamline[map_name]
                    process_map(map_name, data, has_colortable, obj_idx)

            # Process point maps
            pt_maps = map_list_dict.get("maps_per_point")
            if pt_maps:
                for map_name in pt_maps:
                    has_colortable = map_name in sing_obj.colortables
                    data = np.concatenate(sing_obj.data_per_point[map_name])
                    process_map(map_name, data, has_colortable, obj_idx)

        elif isinstance(sing_obj, cltsurf.Surface):
            for map_name in sing_obj.mesh.point_data.keys():
                has_colortable = map_name in sing_obj.colortables
                data = sing_obj.mesh.point_data[map_name]
                process_map(map_name, data, has_colortable, obj_idx)

        elif isinstance(sing_obj, cltpts.PointCloud):
            for map_name in sing_obj.point_data.keys():
                has_colortable = map_name in sing_obj.colortables
                data = sing_obj.point_data[map_name]
                process_map(map_name, data, has_colortable, obj_idx)

    # Classify maps and assign colormaps
    div_maps = []
    seq_maps = []

    for map_name, config in map_characts_dict.items():
        if config["colormap"] != "colortable":
            min_val = config["v_limits"][0]
            max_val = config["v_limits"][1]

            if round(min_val, 3) < 0 and round(max_val, 3) > 0:
                div_maps.append(map_name)
            else:
                seq_maps.append(map_name)

    # Assign colormaps
    seq_cmaps = cltcolor.get_colormaps_names(len(seq_maps), cmap_type="sequential")
    div_cmaps = cltcolor.get_colormaps_names(len(div_maps), cmap_type="diverging")

    for i, map_name in enumerate(seq_maps):
        map_characts_dict[map_name]["colormap"] = seq_cmaps[i]

    for i, map_name in enumerate(div_maps):
        map_characts_dict[map_name]["colormap"] = div_cmaps[i]

    return map_characts_dict


#####################################################################################################
def create_final_object_config(obj2plot, maps_config: dict):
    """
    Create final configuration for each object to plot based on maps configuration.

    Parameters:
    -----------
    obj2plot : list
        List of objects or nested lists containing Tractogram/Surface objects

    maps_config : dict
        Dictionary specifying maps configuration either by map names or by object indices.
        - If keys are strings, they represent map names with associated settings.
        - If keys are integers, they represent object indices with associated settings.
        It is not mandatory to specify all maps or all objects; unspecified entries will use default settings.

    Returns:
    --------
    fin_obj_config : list
        List of dictionaries with final characteristics for each object.
    """
    import warnings

    def _compute_limits_and_range(user_config, default_v_limits, default_v_range):
        """
        Helper to compute v_limits and v_range based on user config and defaults.

        Returns:
        --------
        tuple: (v_limits_min, v_limits_max, v_range_min, v_range_max)
        """
        user_v_range = user_config.get("v_range")
        user_v_limits = user_config.get("v_limits")

        if user_v_range is not None:
            # v_range specified
            v_range_min = (
                user_v_range[0] if user_v_range[0] is not None else default_v_range[0]
            )
            v_range_max = (
                user_v_range[1] if user_v_range[1] is not None else default_v_range[1]
            )

            # Determine v_limits based on v_range
            if user_v_limits is not None:
                v_limits_min = (
                    user_v_limits[0] if user_v_limits[0] is not None else v_range_min
                )
                v_limits_max = (
                    user_v_limits[1] if user_v_limits[1] is not None else v_range_max
                )
            else:
                v_limits_min = v_range_min
                v_limits_max = v_range_max
        else:
            # v_range not specified, sync with v_limits
            if user_v_limits is not None:
                v_limits_min = (
                    user_v_limits[0]
                    if user_v_limits[0] is not None
                    else default_v_limits[0]
                )
                v_limits_max = (
                    user_v_limits[1]
                    if user_v_limits[1] is not None
                    else default_v_limits[1]
                )
            else:
                v_limits_min = default_v_limits[0]
                v_limits_max = default_v_limits[1]

            v_range_min = v_limits_min
            v_range_max = v_limits_max

        return v_limits_min, v_limits_max, v_range_min, v_range_max

    # Flatten objects and get default configuration
    all_objects = flatten_objects(obj2plot)
    default_objects_config = create_default_object_config(all_objects)
    n_objects = len(default_objects_config)

    # Return default configuration if maps_config is empty
    if not maps_config:
        return [
            {"map_name": "default", **obj_conf["default"]}
            for obj_conf in default_objects_config
        ]

    # Detect configuration type
    first_key = next(iter(maps_config.keys()))
    config_type = "maps" if isinstance(first_key, str) else "objects"

    # Initialize final configuration with defaults
    fin_obj_config = [
        {"map_name": "default", **obj_conf["default"]}
        for obj_conf in default_objects_config
    ]

    # Handle map-based configuration.
    if config_type == "maps":
        for map_name, map_config in maps_config.items():
            obj_indices = map_config.get("objects", [])

            # Validate and filter object indices
            valid_indices = [idx for idx in obj_indices if 0 <= idx < n_objects]
            invalid_indices = [idx for idx in obj_indices if idx not in valid_indices]

            if invalid_indices:
                warnings.warn(
                    f"Map '{map_name}': Ignoring invalid object indices {invalid_indices}. "
                    f"Valid range is 0-{n_objects-1}."
                )

            if not valid_indices:
                warnings.warn(
                    f"Map '{map_name}': No valid object indices specified. Skipping."
                )
                continue

            # Compute global min/max across valid objects
            limits = []
            ranges = []

            for idx in valid_indices:
                # Use 'default' if map_name not available for this object
                obj_map_name = (
                    map_name if map_name in default_objects_config[idx] else "default"
                )
                obj_config = default_objects_config[idx][obj_map_name]

                if obj_config["colormap"] == "colortable":
                    warnings.warn(
                        f"Map '{map_name}': Object {idx} uses 'colortable' colormap. "
                        "Skipping from global limits calculation."
                    )
                else:
                    limits.append(obj_config["v_limits"])
                    ranges.append(obj_config["v_range"])

            # Calculate global values from collected limits
            if limits:
                global_min = min(lim[0] for lim in limits)
                global_max = max(lim[1] for lim in limits)
            else:
                global_min, global_max = None, None

            if ranges:
                global_range_min = min(rng[0] for rng in ranges)
                global_range_max = max(rng[1] for rng in ranges)
            else:
                global_range_min, global_range_max = None, None

            # Get default settings from first valid object
            first_idx = valid_indices[0]
            obj_map_name = (
                map_name if map_name in default_objects_config[first_idx] else "default"
            )
            default_settings = default_objects_config[first_idx][obj_map_name]

            # Compute final limits and range using helper function
            v_limits_min, v_limits_max, v_range_min, v_range_max = (
                _compute_limits_and_range(
                    map_config,
                    (global_min, global_max),
                    (global_range_min, global_range_max),
                )
            )

            # Apply configuration to all valid objects
            final_config = {
                "map_name": map_name,
                "colorbar": map_config.get("colorbar", default_settings["colorbar"]),
                "colorbar_title": map_config.get(
                    "colorbar_title", default_settings["colorbar_title"]
                ),
                "colormap": map_config.get("colormap", default_settings["colormap"]),
                "v_limits": (v_limits_min, v_limits_max),
                "v_range": (v_range_min, v_range_max),
                "range_color": map_config.get(
                    "range_color", default_settings["range_color"]
                ),
                "opacity": map_config.get("opacity", default_settings["opacity"]),
            }

            for idx in valid_indices:
                fin_obj_config[idx] = final_config.copy()

    # Handle object-based configuration.
    elif config_type == "objects":
        for idx, obj_config in maps_config.items():
            # Validate object index
            if not (0 <= idx < n_objects):
                warnings.warn(
                    f"Invalid object index {idx}. Valid range is 0-{n_objects-1}. Skipping."
                )
                continue

            map_name = obj_config.get("map_name")
            if not map_name:
                warnings.warn(f"Object {idx}: 'map_name' not specified. Using default.")
                continue

            # Check if map exists for this object
            if map_name not in default_objects_config[idx]:
                warnings.warn(
                    f"Object {idx}: Map '{map_name}' not available. Using 'default' map."
                )
                map_name = "default"

            default_config = default_objects_config[idx][map_name]

            # Compute limits and range using helper function
            v_limits_min, v_limits_max, v_range_min, v_range_max = (
                _compute_limits_and_range(
                    obj_config, default_config["v_limits"], default_config["v_range"]
                )
            )

            # Build configuration with fallbacks to defaults
            fin_obj_config[idx] = {
                "map_name": map_name,
                "colorbar": obj_config.get("colorbar", default_config["colorbar"]),
                "colorbar_title": obj_config.get(
                    "colorbar_title", default_config["colorbar_title"]
                ),
                "colormap": obj_config.get("colormap", default_config["colormap"]),
                "v_limits": (v_limits_min, v_limits_max),
                "v_range": (v_range_min, v_range_max),
                "range_color": obj_config.get(
                    "range_color", default_config["range_color"]
                ),
                "opacity": obj_config.get("opacity", default_config["opacity"]),
            }

    return fin_obj_config


################################################################################################
def find_common_map_names(obj2plot, map_names):
    """
    Find map names that are present in all objects.

    Parameters:
    -----------
    obj2plot : list
        List of objects or nested lists containing Tractogram/Surface objects

    map_names : list
        List of map names to check

    Returns:
    --------
    no_ctab_maps : list
        List of map names that do not have associated colortables in any object

    fin_map_names : list
        List of map names that are present in all objects
    """

    # Get all individual objects
    all_objects = flatten_objects(obj2plot)
    n_objects = len(all_objects)

    fin_map_names = []
    no_ctab_maps = []

    for i, map_name in enumerate(map_names):
        cont_map = 0

        # Check if the map_name is available in all objects
        for sing_obj in all_objects:
            available_maps = []

            if isinstance(sing_obj, clttract.Tractogram):
                map_list_dict = sing_obj.list_maps()

                st_maps = map_list_dict["maps_per_streamline"]
                pt_maps = map_list_dict["maps_per_point"]

                if map_name in st_maps and map_name not in pt_maps:
                    sing_obj.streamline_to_points(
                        map_name=map_name,
                        point_map_name=map_name,
                    )

                if st_maps is not None:
                    available_maps.extend(st_maps)

                if pt_maps is not None:
                    available_maps.extend(pt_maps)

            elif isinstance(sing_obj, cltsurf.Surface):
                available_maps = list(sing_obj.mesh.point_data.keys())

            elif isinstance(sing_obj, cltpts.PointCloud):
                available_maps = list(sing_obj.point_data.keys())

            # Check if colortable is missing
            if map_name not in sing_obj.colortables.keys():
                no_ctab_maps.append(map_name)

            if map_name in available_maps:
                cont_map = cont_map + 1

        # Check if map_name is in all objects
        if cont_map == n_objects:
            fin_map_names.append(map_name)

        # Make unique list of maps without colortable
        no_ctab_maps = list(set(no_ctab_maps))

        # Intersect the no_ctab_maps with fin_map_names to remove them
        no_ctab_maps = cltmisc.list_intercept(fin_map_names, no_ctab_maps)

    return no_ctab_maps, fin_map_names


################################################################################################
def prepare_map_plotting_params(
    map_names: List[str],
    colormaps: Union[str, List[str]],
    v_limits: Union[
        Tuple[Optional[float], Optional[float]],
        List[Tuple[Optional[float], Optional[float]]],
    ],
    v_range: Union[
        Tuple[Optional[float], Optional[float]],
        List[Tuple[Optional[float], Optional[float]]],
    ],
    range_color: tuple = (128, 128, 128, 255),
    colorbar_titles: Optional[Union[str, List[str]]] = None,
) -> dict:
    """
    Prepare and validate parameters for plotting maps on surfaces or tractograms.

    Parameters
    ----------
    map_names : List[str]
        List of map names to be plotted.

    colormaps : str or List[str]
        Colormap(s) to use for each map. If a single string is provided, it will be used for all maps.

    v_limits : tuple or List[tuple]
        Value limits for color mapping. Can be a single tuple (vmin, vmax) or a list of tuples for each map.

    v_range : tuple or List[tuple]
        Value ranges for color mapping. Can be a single tuple (min, max) or a list of tuples for each map.

    range_color : tuple, default (128, 128, 128, 255)
        RGBA color to use for values outside the specified range.

    colorbar_titles : str or List[str], optional
        Titles for the colorbars. If a single string is provided, it will be used for all maps.
        If None, map names will be used as titles.

    Returns
    -------
    dict
        Dictionary containing plotting parameters for each map.

    Raises
    ------
    ValueError
        If input parameters are invalid or inconsistent.

    Examples
    --------
    >>> map_names = ["map1", "map2"]
    >>> colormaps = ["viridis", "plasma"]
    >>> v_limits = [(0, 1), (10, 100)]
    >>> v_range = [(0, 1), (10, 100)]
    >>> colorbar_titles = ["Map 1", "Map 2"]
    >>> plot_params = prepare_map_plotting_params(
    ...     map_names, colormaps, v_limits, v_range, range_color=(255, 0, 0, 255), colorbar_titles=colorbar_titles
    ... )
    >>> print(plot_params)
    {
        'map1': {
            'colormap': 'viridis',
            'vmin': 0,
            'vmax': 1,
            'range_min': 0,
            'range_max': 1,
            'range_color': (255, 0, 0, 255),
            'colorbar_title': 'Map 1'
        },
        'map2': {
            'colormap': 'plasma',
            'vmin': 10,
            'vmax': 100,
            'range_min': 10,
            'range_max': 100,
            'range_color': (255, 0, 0, 255),
            'colorbar_title': 'Map 2'
        }
    }

    """

    n_maps = len(map_names)

    # Process and validate v_limits parameter
    if isinstance(v_limits, tuple):
        if len(v_limits) != 2:
            v_limits = (None, None)
        v_limits = [v_limits] * n_maps

    elif isinstance(v_limits, list):
        if len(v_limits) != n_maps:
            if len(v_limits[0]) != 2:
                v_limits = [(None, None)] * n_maps
            else:
                v_limits = [v_limits[0]] * n_maps

    # Process and validate v_range parameter
    if isinstance(v_range, tuple):
        if len(v_range) != 2:
            v_range = (None, None)
        v_range = [v_range] * n_maps

    elif isinstance(v_range, list):
        if len(v_range) != n_maps:
            if len(v_range[0]) != 2:
                v_range = [(None, None)] * n_maps
            else:
                v_range = [v_range[0]] * n_maps

    # Validate v_range elements
    for vr in v_range:
        if not (isinstance(vr, tuple) and len(vr) == 2):
            raise ValueError("Each element in v_range must be a tuple of (min, max).")
        # Check that the min is less than max if both are not None
        if vr[0] is not None and vr[1] is not None:
            if vr[0] >= vr[1]:
                raise ValueError("In v_range, min value must be less than max value.")

    # Validate v_limits elements
    for vl in v_limits:
        if not (isinstance(vl, tuple) and len(vl) == 2):
            raise ValueError("Each element in v_limits must be a tuple of (min, max).")
        # Check that the min is less than max if both are not None
        if vl[0] is not None and vl[1] is not None:
            if vl[0] >= vl[1]:
                raise ValueError("In v_limits, min value must be less than max value.")

    # Merge v_limits with v_range where needed
    for i, vl in enumerate(v_limits):
        vr_tmp = v_range[i]
        new_lower = vr_tmp[0] if vl[0] is None and vr_tmp[0] is not None else vl[0]
        new_upper = vr_tmp[1] if vl[1] is None and vr_tmp[1] is not None else vl[1]
        v_limits[i] = (new_lower, new_upper)

    if isinstance(colormaps, str):
        colormaps = [colormaps]

    if len(colormaps) >= n_maps:
        colormaps = colormaps[:n_maps]

    else:
        # If not enough colormaps are provided, repeat the first one
        colormaps = [colormaps[0]] * n_maps

    if colorbar_titles is not None:
        if isinstance(colorbar_titles, str):
            colorbar_titles = [colorbar_titles]

        if len(colorbar_titles) != n_maps:
            # If not enough titles are provided, repeat the first one
            colorbar_titles = [colorbar_titles[0]] * n_maps

    else:
        colorbar_titles = map_names

    map_plot_config = {}
    for i, map_name in enumerate(map_names):
        map_plot_config[map_name] = {
            "colorbar": True,
            "colormap": colormaps[i],
            "vmin": v_limits[i][0],
            "vmax": v_limits[i][1],
            "range_min": v_range[i][0],
            "range_max": v_range[i][1],
            "range_color": range_color,
            "colorbar_title": colorbar_titles[i],
        }

    return map_plot_config
