"""
Brain Surface Visualization Tools

This module provides comprehensive tools for visualizing brain surfaces with various
anatomical views and data overlays. It supports FreeSurfer surface formats and
provides flexible layout options for publication-quality figures.

Classes:
    SurfacePlotter: Main class for creating multi-view brain surface layouts
"""

import os
import json
import math
import copy
import numpy as np
import nibabel as nib
from typing import Union, List, Optional, Tuple, Dict, Any, TYPE_CHECKING
from nilearn import plotting
import pyvista as pv
import threading


# Importing external modules
import matplotlib.pyplot as plt

# Importing local modules
from . import freesurfertools as cltfree
from . import misctools as cltmisc
from . import plottools as cltplot

# Use TYPE_CHECKING to avoid circular imports
from . import surfacetools as cltsurf


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############            Section 1: Class dedicated to plot Surface objects              ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
class SurfacePlotter:
    """
    A comprehensive brain surface visualization tool using PyVista.

    This class provides flexible brain plotting capabilities with multiple view configurations,
    customizable colormaps, and optional colorbar support for neuroimaging data visualization.

    Attributes
    ----------
    config_file : str
        Path to the JSON configuration file containing layout definitions.

    figure_conf : dict
        Loaded figure configuration with styling settings.

    views_conf : dict
        Loaded views configuration with layout definitions.

    Examples
    --------
    >>> plotter = SurfacePlotter("brain_plot_configs.json")
    >>> plotter.plot_hemispheres(surf_lh, surf_rh, map_name="thickness",
    ...                          views="8_views", colorbar=True)
    >>>
    >>> # Dynamic view selection
    >>> plotter.plot_hemispheres(surf_lh, surf_rh, views=["lateral", "medial", "dorsal"])
    """

    ###############################################################################################
    def __init__(self, config_file: str = None):
        """
        Initialize the SurfacePlotter with configuration file.

        Parameters
        ----------
        config_file : str, optional
            Path to JSON file containing figure and view configurations.
            If None, uses default "viz_views.json" from config directory.

        Raises
        ------
        FileNotFoundError
            If the configuration file doesn't exist.

        json.JSONDecodeError
            If the configuration file contains invalid JSON.

        KeyError
            If required keys 'figure_conf' or 'views_conf' are missing.

        Examples
        --------
        >>> plotter = SurfacePlotter()  # Use default config
        >>>
        >>> plotter = SurfacePlotter("custom_views.json")  # Use custom config
        """
        # Initialize configuration attributes

        # Get the absolute of this file
        if config_file is None:
            cwd = os.path.dirname(os.path.abspath(__file__))
            # Default to the standard configuration file
            config_file = os.path.join(cwd, "config", "viz_views.json")
        else:
            # Use the provided configuration file path
            config_file = os.path.abspath(config_file)

        # Check if the configuration file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found")

        # Load configurations from the JSON file
        self.config_file = config_file
        self._load_configs()

        # Define mapping from simple view names to configuration titles
        self._view_name_mapping = {
            "lateral": ["LH: Lateral view", "RH: Lateral view"],
            "medial": ["LH: Medial view", "RH: Medial view"],
            "dorsal": ["Dorsal view"],
            "ventral": ["Ventral view"],
            "rostral": ["Rostral view"],
            "caudal": ["Caudal view"],
        }

    #########################################################################################################
    def _load_configs(self) -> None:
        """
        Load figure and view configurations from JSON file.

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

        # Load configurations from JSON file
        try:
            with open(self.config_file, "r") as f:
                configs = json.load(f)

            # Validate structure and load configurations
            if "figure_conf" not in configs:
                raise KeyError("Missing 'figure_conf' key in configuration file")
            if "views_conf" not in configs:
                raise KeyError("Missing 'views_conf' key in configuration file")

            self.figure_conf = configs["figure_conf"]
            self.views_conf = configs["views_conf"]
            self.layouts_conf = configs["layouts_conf"]
            self.themes_conf = configs["themes_conf"]

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file '{self.config_file}' not found"
            )
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in configuration file: {e}")

    def _get_views_to_plot(
        self, views: Union[str, List[str]], hemi_id: Union[str, List[str]] = "lh"
    ) -> List[str]:
        """
        Get the list of views to plot based on user input and hemisphere.
        This method normalizes the input views, validates them against the available
        views configuration, and filters them based on the specified hemisphere.
        Parameters
        ----------
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
        valid_views = self._get_valid_views(views)

        # Get number of views
        if len(valid_views) == 1:
            if valid_views[0] in self._list_multiviews_layouts():
                view_ids = self.layouts_conf[valid_views[0]]["views"]
                if "lh" in hemi_id and "rh" not in hemi_id:
                    # LH only, remove the view_ids that contain rh- on the name
                    view_ids = [v for v in view_ids if "rh-" not in v]
                elif "rh" in hemi_id and "lh" not in hemi_id:
                    # RH only, remove the view_ids that contain lh- on the name
                    view_ids = [v for v in view_ids if "lh-" not in v]
                    # Flip the view_ids and the last will be the first
                    view_ids = view_ids[::-1]

            elif valid_views[0] in self._list_single_views():
                # Single view layout, take all the possible views
                view_ids = list(self.views_conf.keys())
                # Selecting the views based on the supplied names
                view_ids = cltmisc.filter_by_substring(view_ids, valid_views)
                # Filter views based on hemisphere
                if "lh" in hemi_id and "rh" not in hemi_id:
                    view_ids = [v for v in view_ids if "rh-" not in v]
                elif "rh" in hemi_id and "lh" not in hemi_id:
                    view_ids = [v for v in view_ids if "lh-" not in v]
        else:
            # Multiple view names provided
            view_ids = list(self.views_conf.keys())
            # Selecting the views based on the supplied names
            view_ids = cltmisc.filter_by_substring(view_ids, valid_views)
            # Filter views based on hemisphere
            if "lh" in hemi_id and "rh" not in hemi_id:
                view_ids = [v for v in view_ids if "rh-" not in v]
            elif "rh" in hemi_id and "lh" not in hemi_id:
                view_ids = [v for v in view_ids if "lh-" not in v]

        return view_ids

    def _build_plotting_config(
        self,
        views: list,
        hemi_id: str = ["lh", "rh"],
        orientation: str = "horizontal",
        maps_names: Union[str, List[str]] = ["surface"],
        colormaps: Union[str, List[str]] = "viridis",
        v_limits: Union[Tuple[float, float], List[Tuple[float, float]]] = (None, None),
        surfaces: Union[Any, List[Any]] = None,  # cltsurf.Surface
        colorbar: bool = False,
        colorbar_titles: Union[str, List[str]] = None,
        colormap_style: str = "individual",
        colorbar_position: str = "right",
    ):
        """
        Build the plotting configuration based on user inputs.

        Returns
        -------
        Tuple[List[int], List[float], List[float], List[Tuple], Dict, List[Dict]]
            (shape, row_weights, col_weights, groups, brain_positions, colorbar_positions)
        """

        # Constants
        colorbar_size = self.figure_conf["colorbar_size"]

        # Normalize inputs
        maps_names = self._normalize_to_list(maps_names)
        colormaps = self._normalize_to_list(colormaps)
        v_limits = self._normalize_to_list(v_limits)
        colorbar_titles = (
            self._normalize_to_list(colorbar_titles) if colorbar_titles else None
        )
        surfaces = self._normalize_to_list(surfaces) if surfaces else []

        n_maps = len(maps_names)
        n_surfaces = len(surfaces)

        # Force single view when both maps and surfaces > 1
        if n_maps > 1 and n_surfaces > 1:
            print(
                "🔧 FORCING single view (dorsal) because both n_maps > 1 and n_surfaces > 1"
            )
            views = ["dorsal"]

        # Get view configuration
        view_ids = self._get_views_to_plot(views, hemi_id=hemi_id)
        n_views = len(view_ids)

        if n_maps > 1 and n_surfaces > 1:
            view_ids = ["merg-dorsal"]
            n_views = 1

        print(
            f"Number of views: {n_views}, Number of maps: {n_maps}, Number of surfaces: {n_surfaces}"
        )

        # Check if colorbar is needed
        colorbar = colorbar and self._colorbar_needed(maps_names, surfaces)

        # Build configuration based on dimensions
        config, colorbar_list = self._build_layout_config(
            view_ids,
            maps_names,
            surfaces,
            v_limits,
            colormaps,
            orientation,
            colorbar,
            colormap_style,
            colorbar_position,
            colorbar_titles,
        )

        return (
            view_ids,
            config,
            colorbar_list,
        )

    def _normalize_to_list(self, item):
        """Convert single items to lists for consistent handling."""
        if not isinstance(item, list):
            return [item]
        return item

    def _colorbar_needed(self, maps_names, surfaces):
        """Check if colorbar is actually needed based on surface colortables."""
        if not surfaces:
            return True

        # Check if any map is not already on the surface
        for map_name in maps_names:
            if map_name not in surfaces[0].colortables:
                return True
        return False

    def _build_layout_config(
        self,
        valid_views,
        maps_names,
        surfaces,
        v_limits,
        colormaps,
        orientation,
        colorbar,
        colormap_style,
        colorbar_position,
        colorbar_titles,
    ):
        """Build the basic layout configuration."""

        n_views = len(valid_views)
        n_maps = len(maps_names)
        n_surfaces = len(surfaces)
        colorbar_size = self.figure_conf["colorbar_size"]

        if n_views == 1 and n_maps == 1 and n_surfaces == 1:  # Works fine
            # Check if maps_names[0] is present in the surface
            if colormap_style not in ["individual", "shared"]:
                colormap_style = "individual"

            if maps_names[0] in list(surfaces[0].colortables.keys()):
                colorbar = False

            return self._single_element_layout(
                surfaces,
                maps_names,
                v_limits,
                colormaps,
                colorbar_titles,
                colorbar,
                colorbar_position,
                colorbar_size,
            )

        elif n_views == 1 and n_maps == 1 and n_surfaces > 1:  # Works fine
            if colormap_style not in ["individual", "shared"]:
                colormap_style = "individual"

            # Check if maps_names[0] is present in ALL surfaces
            if all(
                maps_names[0] in surfaces[i].colortables.keys()
                for i in range(n_surfaces)
            ):
                colorbar = False

            return self._single_map_multi_surface_layout(
                surfaces,
                maps_names,
                v_limits,
                colormaps,
                colorbar_titles,
                orientation,
                colorbar,
                colormap_style,
                colorbar_position,
                colorbar_size,
            )

        elif n_views == 1 and n_maps > 1 and n_surfaces == 1:  # Works fine
            if colormap_style not in ["individual", "shared"]:
                colormap_style = "individual"

            return self._multi_map_single_surface_layout(
                surfaces,
                maps_names,
                v_limits,
                colormaps,
                colorbar_titles,
                orientation,
                colorbar,
                colormap_style,
                colorbar_position,
                colorbar_size,
            )

        elif n_views == 1 and n_maps > 1 and n_surfaces > 1:  # Works fine
            colorbar_data = any(
                map_name not in surface.colortables
                for map_name in maps_names
                for surface in surfaces
            )
            if colorbar_data == False:
                colorbar = False

            return self._multi_map_multi_surface_layout(
                surfaces,
                maps_names,
                v_limits,
                colormaps,
                colorbar_titles,
                orientation,
                colorbar,
                colormap_style,
                colorbar_position,
                colorbar_size,
            )

        elif n_views > 1 and n_maps == 1 and n_surfaces == 1:  # Works fine
            if all(
                maps_names[0] in surfaces[i].colortables.keys()
                for i in range(n_surfaces)
            ):
                colorbar = False
            return self._multi_view_single_element_layout(
                surfaces[0],
                valid_views,
                maps_names[0],
                v_limits[0],
                colormaps[0],
                colorbar_titles[0],
                orientation,
                colorbar,
                colorbar_position,
                colorbar_size,
            )

        elif n_views > 1 and n_maps == 1 and n_surfaces > 1:  # Works fine
            if all(
                maps_names[0] in surfaces[i].colortables.keys()
                for i in range(n_surfaces)
            ):
                colorbar = False
            return self._multi_view_multi_surface_layout(
                surfaces,
                valid_views,
                maps_names,
                v_limits,
                colormaps,
                colorbar_titles,
                orientation,
                colorbar,
                colorbar_position,
                colormap_style,
                colorbar_size,
            )

        elif n_views > 1 and n_maps > 1 and n_surfaces == 1:  # Works fine
            colorbar_data = any(
                map_name not in surfaces[0].colortables for map_name in maps_names
            )
            if colorbar_data == False:
                colorbar = False

            return self._multi_view_multi_map_layout(
                surfaces,
                valid_views,
                maps_names,
                v_limits,
                colormaps,
                colorbar_titles,
                orientation,
                colorbar,
                colormap_style,
                colorbar_position,
                colorbar_size,
            )

        else:
            # Default fallback for any remaining cases
            return {
                "shape": [1, 1],
                "row_weights": [1],
                "col_weights": [1],
                "groups": [],
                "brain_positions": {(0, 0, 0): (0, 0)},
            }

    def _single_element_layout(
        self,
        surfaces,
        maps_names,
        v_limits,
        colormaps,
        colorbar_titles,
        colorbar,
        colorbar_position,
        colorbar_size,
    ):
        """Handle single view, single map, single surface case."""
        brain_positions = {(0, 0, 0): (0, 0)}
        colormap_limits = {}

        limits_list = _get_map_limits(
            surfaces=surfaces,
            map_name=maps_names[0],
            colormap_style="individual",
            v_limits=v_limits[0],
        )
        colormap_limits[(0, 0, 0)] = limits_list[0]

        colorbar_list = []

        if maps_names[0] in surfaces[0].colortables:
            colorbar = False

        if not colorbar:
            shape = [1, 1]
            row_weights = [1]
            col_weights = [1]

        else:

            cb_dict = {}
            if colorbar_position == "right":
                shape = [1, 2]
                row_weights = [1]
                col_weights = [1, colorbar_size]
                cb_dict["position"] = (0, 1)
                cb_dict["orientation"] = "vertical"

            elif colorbar_position == "bottom":
                shape = [2, 1]
                row_weights = [1, colorbar_size]
                col_weights = [1]
                cb_dict["position"] = (1, 0)
                cb_dict["orientation"] = "horizontal"

            cb_dict["colormap"] = colormaps[0]
            cb_dict["map_name"] = maps_names[0]

            cb_dict["vmin"] = limits_list[0][0]
            cb_dict["vmax"] = limits_list[0][1]

            if colorbar_titles:
                cb_dict["title"] = colorbar_titles[0]
            else:
                cb_dict["title"] = maps_names[0]

            colorbar_list.append(cb_dict)

        layout_config = {
            "shape": shape,
            "row_weights": row_weights,
            "col_weights": col_weights,
            "groups": [],
            "brain_positions": brain_positions,
            "colormap_limits": colormap_limits,
        }

        return layout_config, colorbar_list

    def _multi_map_single_surface_layout(
        self,
        surfaces,
        maps_names,
        v_limits,
        colormaps,
        colorbar_titles,
        orientation,
        colorbar,
        colormap_style,
        colorbar_position,
        colorbar_size,
    ):
        """Handle multiple maps, single surface case."""
        brain_positions = {}

        if orientation == "horizontal":
            return self._horizontal_multi_map_layout(
                surfaces,
                maps_names,
                v_limits,
                colormaps,
                colorbar_titles,
                colorbar,
                colormap_style,
                colorbar_position,
                colorbar_size,
            )
        elif orientation == "vertical":
            return self._vertical_multi_map_layout(
                surfaces,
                maps_names,
                v_limits,
                colormaps,
                colorbar_titles,
                colorbar,
                colormap_style,
                colorbar_position,
                colorbar_size,
            )
        else:  # grid
            return self._grid_multi_map_layout(
                surfaces,
                maps_names,
                v_limits,
                colormaps,
                colorbar_titles,
                colorbar,
                colormap_style,
                colorbar_position,
                colorbar_size,
            )

    def _horizontal_multi_map_layout(
        self,
        surfaces,
        maps_names,
        v_limits,
        colormaps,
        colorbar_titles,
        colorbar,
        colormap_style,
        colorbar_position,
        colorbar_size,
    ):
        """Handle horizontal layout for multiple maps."""

        n_maps = len(maps_names)

        brain_positions = {}
        colormap_limits = {}

        for map_idx in range(n_maps):
            brain_positions[(map_idx, 0, 0)] = (0, map_idx)
            map_limits = _get_map_limits(
                surfaces=surfaces,
                map_name=maps_names[map_idx],
                colormap_style="individual",
                v_limits=v_limits[map_idx],
            )[0]
            colormap_limits[(map_idx, 0, 0)] = map_limits

        colorbar_list = []
        if not colorbar:
            shape = [1, n_maps]
            row_weights = [1]
            col_weights = [1] * n_maps
            groups = []

        else:
            if colormap_style == "individual":
                if colorbar_position == "right":
                    shape = [1, n_maps * 2]
                    row_weights = [1]
                    col_weights = [1, colorbar_size] * n_maps
                    groups = []

                    for map_idx in range(n_maps):
                        brain_positions[(map_idx, 0, 0)] = (0, map_idx * 2)

                        cb_dict = {}
                        cb_dict["position"] = (0, map_idx * 2 + 1)
                        cb_dict["orientation"] = "vertical"
                        cb_dict["colormap"] = (
                            colormaps[map_idx]
                            if map_idx < len(colormaps)
                            else colormaps[-1]
                        )
                        cb_dict["map_name"] = maps_names[map_idx]

                        limits_list = _get_map_limits(
                            surfaces=surfaces,
                            map_name=maps_names[map_idx],
                            colormap_style="individual",
                            v_limits=v_limits[map_idx],
                        )
                        cb_dict["vmin"] = limits_list[0][0]
                        cb_dict["vmax"] = limits_list[0][1]

                        if colorbar_titles:
                            cb_dict["title"] = (
                                colorbar_titles[map_idx]
                                if map_idx < len(colorbar_titles)
                                else colorbar_titles[-1]
                            )
                        else:
                            cb_dict["title"] = maps_names[map_idx]

                        colorbar_list.append(cb_dict)

                else:  # bottom
                    shape = [2, n_maps]
                    row_weights = [1, colorbar_size]
                    col_weights = [1] * n_maps
                    groups = []

                    for map_idx in range(n_maps):
                        brain_positions[(map_idx, 0, 0)] = (0, map_idx)

                        cb_dict = {}
                        cb_dict["position"] = (1, map_idx)
                        cb_dict["orientation"] = "horizontal"
                        cb_dict["colormap"] = (
                            colormaps[map_idx]
                            if map_idx < len(colormaps)
                            else colormaps[-1]
                        )
                        cb_dict["map_name"] = maps_names[map_idx]

                        limits_list = _get_map_limits(
                            surfaces=surfaces,
                            map_name=maps_names[map_idx],
                            colormap_style="individual",
                            v_limits=v_limits[map_idx],
                        )
                        cb_dict["vmin"] = limits_list[0][0]
                        cb_dict["vmax"] = limits_list[0][1]

                        if colorbar_titles:
                            cb_dict["title"] = (
                                colorbar_titles[map_idx]
                                if map_idx < len(colorbar_titles)
                                else colorbar_titles[-1]
                            )
                        else:
                            cb_dict["title"] = maps_names[map_idx]

                        colorbar_list.append(cb_dict)
            else:  # shared colorbar
                cb_dict = {}
                cb_dict["colormap"] = colormaps[0]
                cb_dict["map_name"] = " + ".join(maps_names)
                for map_idx in range(n_maps):
                    map_limits = _get_map_limits(
                        surfaces=surfaces,
                        map_name=maps_names[map_idx],
                        colormap_style="shared",
                        v_limits=v_limits[map_idx],
                    )[0]
                    if map_idx == 0:
                        limits_list = map_limits
                    else:
                        limits_list = (
                            min(limits_list[0], map_limits[0]),
                            max(limits_list[1], map_limits[1]),
                        )

                cb_dict["vmin"] = limits_list[0]
                cb_dict["vmax"] = limits_list[1]

                if colorbar_titles:
                    cb_dict["title"] = colorbar_titles[0]
                else:
                    cb_dict["title"] = " + ".join(maps_names)

                if colorbar_position == "right":
                    shape = [1, n_maps + 1]
                    row_weights = [1]
                    col_weights = [1] * n_maps + [colorbar_size]
                    groups = []

                    cb_dict["position"] = (0, n_maps)
                    cb_dict["orientation"] = "vertical"

                else:  # bottom
                    shape = [2, n_maps]
                    row_weights = [1, colorbar_size]
                    col_weights = [1] * n_maps

                    groups = [(1, slice(0, n_maps))]  # Colorbar in last row
                    cb_dict["position"] = (1, 0)
                    cb_dict["orientation"] = "horizontal"

                    for map_idx in range(n_maps):
                        brain_positions[(map_idx, 0, 0)] = (0, map_idx)

                for map_idx in range(n_maps):
                    colormap_limits[(map_idx, 0, 0)] = (
                        cb_dict["vmin"],
                        cb_dict["vmax"],
                        maps_names[0],
                    )

                colorbar_list.append(cb_dict)

        layout_config = {
            "shape": shape,
            "row_weights": row_weights,
            "col_weights": col_weights,
            "groups": groups,
            "brain_positions": brain_positions,
            "colormap_limits": colormap_limits,
        }
        return layout_config, colorbar_list

    def _vertical_multi_map_layout(
        self,
        surfaces,
        maps_names,
        v_limits,
        colormaps,
        colorbar_titles,
        colorbar,
        colormap_style,
        colorbar_position,
        colorbar_size,
    ):
        """Handle vertical layout for multiple maps."""
        n_maps = len(maps_names)

        brain_positions = {}
        colormap_limits = {}

        for map_idx in range(n_maps):
            brain_positions[(map_idx, 0, 0)] = (map_idx, 0)
            map_limits = _get_map_limits(
                surfaces=surfaces,
                map_name=maps_names[map_idx],
                colormap_style="individual",
                v_limits=v_limits[map_idx],
            )[0]
            colormap_limits[(map_idx, 0, 0)] = map_limits

        colorbar_list = []
        if not colorbar:
            shape = [n_maps, 1]
            row_weights = [1] * n_maps
            col_weights = [1]
            groups = []

        else:
            if colormap_style == "individual":
                if colorbar_position == "right":
                    shape = [n_maps, 2]
                    row_weights = [1] * n_maps
                    col_weights = [1, colorbar_size]
                    groups = []

                    for map_idx in range(n_maps):
                        brain_positions[(map_idx, 0, 0)] = (map_idx, 0)

                        cb_dict = {}
                        cb_dict["position"] = (map_idx, 1)
                        cb_dict["orientation"] = "vertical"
                        cb_dict["colormap"] = (
                            colormaps[map_idx]
                            if map_idx < len(colormaps)
                            else colormaps[-1]
                        )
                        cb_dict["map_name"] = maps_names[map_idx]

                        limits_list = _get_map_limits(
                            surfaces=surfaces,
                            map_name=maps_names[map_idx],
                            colormap_style="individual",
                            v_limits=v_limits[map_idx],
                        )
                        cb_dict["vmin"] = limits_list[0][0]
                        cb_dict["vmax"] = limits_list[0][1]

                        if colorbar_titles:
                            cb_dict["title"] = (
                                colorbar_titles[map_idx]
                                if map_idx < len(colorbar_titles)
                                else colorbar_titles[-1]
                            )
                        else:
                            cb_dict["title"] = maps_names[map_idx]

                        colorbar_list.append(cb_dict)

                else:  # bottom
                    shape = [n_maps * 2, 1]
                    row_weights = [1, colorbar_size] * n_maps
                    col_weights = [1]
                    groups = []

                    for map_idx in range(n_maps):
                        brain_positions[(map_idx, 0, 0)] = (map_idx * 2, 0)

                        cb_dict = {}
                        cb_dict["position"] = (map_idx * 2 + 1, 0)
                        cb_dict["orientation"] = "horizontal"
                        cb_dict["colormap"] = (
                            colormaps[map_idx]
                            if map_idx < len(colormaps)
                            else colormaps[-1]
                        )
                        cb_dict["map_name"] = maps_names[map_idx]

                        limits_list = _get_map_limits(
                            surfaces=surfaces,
                            map_name=maps_names[map_idx],
                            colormap_style="individual",
                            v_limits=v_limits[map_idx],
                        )
                        cb_dict["vmin"] = limits_list[0][0]
                        cb_dict["vmax"] = limits_list[0][1]

                        if colorbar_titles:
                            cb_dict["title"] = (
                                colorbar_titles[map_idx]
                                if map_idx < len(colorbar_titles)
                                else colorbar_titles[-1]
                            )
                        else:
                            cb_dict["title"] = maps_names[map_idx]

                        colorbar_list.append(cb_dict)
            else:  # shared colorbar
                cb_dict = {}
                cb_dict["colormap"] = colormaps[0]
                cb_dict["map_name"] = " + ".join(maps_names)
                for map_idx in range(n_maps):
                    map_limits = _get_map_limits(
                        surfaces=surfaces,
                        map_name=maps_names[map_idx],
                        colormap_style="shared",
                        v_limits=v_limits[map_idx],
                    )[0]
                    if map_idx == 0:
                        limits_list = map_limits
                    else:
                        limits_list = (
                            min(limits_list[0], map_limits[0]),
                            max(limits_list[1], map_limits[1]),
                        )
                cb_dict["vmin"] = limits_list[0]
                cb_dict["vmax"] = limits_list[1]
                if colorbar_titles:
                    cb_dict["title"] = colorbar_titles[0]
                else:
                    cb_dict["title"] = " + ".join(maps_names)
                if colorbar_position == "right":
                    shape = [n_maps, 2]
                    row_weights = [1] * n_maps
                    col_weights = [1, colorbar_size]
                    groups = [(slice(0, n_maps), 1)]

                    cb_dict["position"] = (0, 1)
                    cb_dict["orientation"] = "vertical"
                else:  # bottom
                    shape = [n_maps + 1, 1]
                    row_weights = [1] * n_maps + [colorbar_size]
                    col_weights = [1]
                    groups = [(n_maps, 0)]
                    cb_dict["position"] = (n_maps, 0)
                    cb_dict["orientation"] = "horizontal"
                    for map_idx in range(n_maps):
                        brain_positions[(map_idx, 0, 0)] = (map_idx, 0)
                for map_idx in range(n_maps):
                    colormap_limits[(map_idx, 0, 0)] = (
                        cb_dict["vmin"],
                        cb_dict["vmax"],
                        maps_names[0],
                    )
                colorbar_list.append(cb_dict)
        layout_config = {
            "shape": shape,
            "row_weights": row_weights,
            "col_weights": col_weights,
            "groups": groups,
            "brain_positions": brain_positions,
            "colormap_limits": colormap_limits,
        }
        return layout_config, colorbar_list

    def _grid_multi_map_layout(
        self,
        surfaces,
        maps_names,
        v_limits,
        colormaps,
        colorbar_titles,
        colorbar,
        colormap_style,
        colorbar_position,
        colorbar_size,
    ):
        """Handle grid layout for multiple maps."""

        n_maps = len(maps_names)
        optimal_grid, positions = cltplot.calculate_optimal_subplots_grid(n_maps)
        brain_positions = {}
        colormap_limits = {}

        for map_idx in range(n_maps):
            pos = positions[map_idx]
            brain_positions[(map_idx, 0, 0)] = pos
            map_limits = _get_map_limits(
                surfaces=surfaces,
                map_name=maps_names[map_idx],
                colormap_style="individual",
                v_limits=v_limits[map_idx],
            )[0]
            colormap_limits[(map_idx, 0, 0)] = map_limits

        colorbar_list = []
        if not colorbar:
            shape = list(optimal_grid)
            row_weights = [1] * optimal_grid[0]
            col_weights = [1] * optimal_grid[1]
            groups = []
        else:
            if colormap_style == "individual":
                if colorbar_position == "right":
                    shape = [optimal_grid[0], optimal_grid[1] * 2]
                    row_weights = [1] * optimal_grid[0]
                    col_weights = [1, colorbar_size] * optimal_grid[1]
                    groups = []

                    for map_idx in range(n_maps):
                        pos = positions[map_idx]
                        brain_positions[(map_idx, 0, 0)] = (pos[0], pos[1] * 2)

                        cb_dict = {}
                        cb_dict["position"] = (pos[0], pos[1] * 2 + 1)
                        cb_dict["orientation"] = "vertical"
                        cb_dict["colormap"] = (
                            colormaps[map_idx]
                            if map_idx < len(colormaps)
                            else colormaps[-1]
                        )
                        cb_dict["map_name"] = maps_names[map_idx]

                        limits_list = _get_map_limits(
                            surfaces=surfaces,
                            map_name=maps_names[map_idx],
                            colormap_style="individual",
                            v_limits=v_limits[map_idx],
                        )
                        cb_dict["vmin"] = limits_list[0][0]
                        cb_dict["vmax"] = limits_list[0][1]

                        if colorbar_titles:
                            cb_dict["title"] = (
                                colorbar_titles[map_idx]
                                if map_idx < len(colorbar_titles)
                                else colorbar_titles[-1]
                            )
                        else:
                            cb_dict["title"] = maps_names[map_idx]

                        colorbar_list.append(cb_dict)

                else:  # bottom
                    shape = [optimal_grid[0] * 2, optimal_grid[1]]
                    row_weights = [1, colorbar_size] * optimal_grid[0]
                    col_weights = [1] * optimal_grid[1]
                    groups = []

                    for map_idx in range(n_maps):
                        pos = positions[map_idx]
                        brain_positions[(map_idx, 0, 0)] = (pos[0] * 2, pos[1])

                        cb_dict = {}
                        cb_dict["position"] = (pos[0] * 2 + 1, pos[1])
                        cb_dict["orientation"] = "horizontal"
                        cb_dict["colormap"] = (
                            colormaps[map_idx]
                            if map_idx < len(colormaps)
                            else colormaps[-1]
                        )
                        cb_dict["map_name"] = maps_names[map_idx]

                        limits_list = _get_map_limits(
                            surfaces=surfaces,
                            map_name=maps_names[map_idx],
                            colormap_style="individual",
                            v_limits=v_limits[map_idx],
                        )
                        cb_dict["vmin"] = limits_list[0][0]
                        cb_dict["vmax"] = limits_list[0][1]

                        if colorbar_titles:
                            cb_dict["title"] = (
                                colorbar_titles[map_idx]
                                if map_idx < len(colorbar_titles)
                                else colorbar_titles[-1]
                            )
                        else:
                            cb_dict["title"] = maps_names[map_idx]

                        colorbar_list.append(cb_dict)
            else:  # shared colorbar
                cb_dict = {}
                cb_dict["colormap"] = colormaps[0]
                cb_dict["map_name"] = " + ".join(maps_names)
                for map_idx in range(n_maps):
                    map_limits = _get_map_limits(
                        surfaces=surfaces,
                        map_name=maps_names[map_idx],
                        colormap_style="shared",
                        v_limits=v_limits[map_idx],
                    )[0]
                    if map_idx == 0:
                        limits_list = map_limits
                    else:
                        limits_list = (
                            min(limits_list[0], map_limits[0]),
                            max(limits_list[1], map_limits[1]),
                        )
                cb_dict["vmin"] = limits_list[0]
                cb_dict["vmax"] = limits_list[1]
                if colorbar_titles:
                    cb_dict["title"] = colorbar_titles[0]
                else:
                    cb_dict["title"] = " + ".join(maps_names)
                if colorbar_position == "right":
                    shape = [optimal_grid[0], optimal_grid[1] + 1]
                    row_weights = [1] * optimal_grid[0]
                    col_weights = [1] * optimal_grid[1] + [colorbar_size]
                    groups = [(slice(0, optimal_grid[0]), optimal_grid[1])]

                    cb_dict["position"] = (0, optimal_grid[1])
                    cb_dict["orientation"] = "vertical"
                else:  # bottom
                    shape = [optimal_grid[0] + 1, optimal_grid[1]]
                    row_weights = [1] * optimal_grid[0] + [colorbar_size]
                    col_weights = [1] * optimal_grid[1]
                    groups = [(optimal_grid[0], slice(0, optimal_grid[1]))]
                    cb_dict["position"] = (optimal_grid[0], 0)
                    cb_dict["orientation"] = "horizontal"
                for map_idx in range(n_maps):
                    colormap_limits[(map_idx, 0, 0)] = (
                        cb_dict["vmin"],
                        cb_dict["vmax"],
                        maps_names[0],
                    )
                for map_idx in range(n_maps):
                    pos = positions[map_idx]
                    brain_positions[(map_idx, 0, 0)] = pos
                colorbar_list.append(cb_dict)
        layout_config = {
            "shape": shape,
            "row_weights": row_weights,
            "col_weights": col_weights,
            "groups": groups,
            "brain_positions": brain_positions,
            "colormap_limits": colormap_limits,
        }
        return layout_config, colorbar_list

    ################################################################################
    # Multiple surfaces and multiple maps cases
    ################################################################################
    def _multi_map_multi_surface_layout(
        self,
        surfaces,
        maps_names,
        v_limits,
        colormaps,
        colorbar_titles,
        orientation,
        colorbar,
        colormap_style,
        colorbar_position,
        colorbar_size,
    ):
        """Handle multiple maps and multiple surfaces case."""

        n_maps = len(maps_names)
        n_surfaces = len(surfaces)
        brain_positions = {}
        colormap_limits = {}
        colorbar_list = []
        if orientation == "horizontal":

            if not colorbar:
                shape = [n_surfaces, n_maps]
                row_weights = [1] * n_surfaces
                col_weights = [1] * n_maps
                groups = []

                # Maps in columns, surfaces in rows
                for map_idx in range(n_maps):
                    for surf_idx in range(n_surfaces):
                        brain_positions[(map_idx, surf_idx, 0)] = (surf_idx, map_idx)
                        map_limits = _get_map_limits(
                            surfaces=surfaces[surf_idx],
                            map_name=maps_names[map_idx],
                            colormap_style="individual",
                            v_limits=v_limits[map_idx],
                        )[0]
                        colormap_limits[(map_idx, surf_idx, 0)] = map_limits
            else:

                # Force colorbar to bottom for this case
                if colormap_style == "individual":
                    if colorbar_position == "right":
                        shape = [n_surfaces, n_maps * 2]
                        row_weights = [1] * n_surfaces
                        col_weights = [1, colorbar_size] * n_maps
                        groups = []

                        for map_idx in range(n_maps):
                            for surf_idx in range(n_surfaces):
                                brain_positions[(map_idx, surf_idx, 0)] = (
                                    surf_idx,
                                    map_idx * 2,
                                )

                                cb_dict = {}
                                cb_dict["position"] = (surf_idx, map_idx * 2 + 1)
                                cb_dict["orientation"] = "vertical"
                                cb_dict["colormap"] = (
                                    colormaps[map_idx]
                                    if map_idx < len(colormaps)
                                    else colormaps[-1]
                                )
                                cb_dict["map_name"] = maps_names[map_idx]
                                if colorbar_titles:
                                    cb_dict["title"] = (
                                        colorbar_titles[map_idx]
                                        if map_idx < len(colorbar_titles)
                                        else colorbar_titles[-1]
                                    )
                                else:
                                    cb_dict["title"] = maps_names[map_idx]
                                limits_list = _get_map_limits(
                                    surfaces=surfaces[surf_idx],
                                    map_name=maps_names[map_idx],
                                    colormap_style="individual",
                                    v_limits=v_limits[map_idx],
                                )
                                cb_dict["vmin"] = limits_list[0][0]
                                cb_dict["vmax"] = limits_list[0][1]
                                colormap_limits[(map_idx, surf_idx, 0)] = limits_list[0]

                                if (
                                    maps_names[map_idx]
                                    not in surfaces[surf_idx].colortables
                                ):
                                    colorbar_list.append(cb_dict)

                    elif colorbar_position == "bottom":
                        shape = [n_surfaces * 2, n_maps]
                        row_weights = [1, colorbar_size] * n_surfaces
                        col_weights = [1] * n_maps
                        groups = []

                        for map_idx in range(n_maps):
                            for surf_idx in range(n_surfaces):
                                brain_positions[(map_idx, surf_idx, 0)] = (
                                    surf_idx * 2,
                                    map_idx,
                                )
                                cb_dict = {}
                                cb_dict["position"] = (surf_idx * 2 + 1, map_idx)
                                cb_dict["orientation"] = "horizontal"
                                cb_dict["colormap"] = (
                                    colormaps[map_idx]
                                    if map_idx < len(colormaps)
                                    else colormaps[-1]
                                )
                                cb_dict["map_name"] = maps_names[map_idx]
                                if colorbar_titles:
                                    cb_dict["title"] = (
                                        colorbar_titles[map_idx]
                                        if map_idx < len(colorbar_titles)
                                        else colorbar_titles[-1]
                                    )
                                else:
                                    cb_dict["title"] = maps_names[map_idx]
                                limits_list = _get_map_limits(
                                    surfaces=surfaces[surf_idx],
                                    map_name=maps_names[map_idx],
                                    colormap_style="individual",
                                    v_limits=v_limits[map_idx],
                                )
                                cb_dict["vmin"] = limits_list[0][0]
                                cb_dict["vmax"] = limits_list[0][1]
                                colormap_limits[(map_idx, surf_idx, 0)] = limits_list[0]

                                if (
                                    maps_names[map_idx]
                                    not in surfaces[surf_idx].colortables
                                ):
                                    colorbar_list.append(cb_dict)

                else:

                    # Map-wise limits
                    maps_limits = []
                    for map_idx in range(n_maps):
                        if maps_names[map_idx] not in surfaces[0].colortables:
                            map_limits = _get_map_limits(
                                surfaces=surfaces,
                                map_name=maps_names[map_idx],
                                colormap_style="shared",
                                v_limits=v_limits[map_idx],
                            )[0]
                            maps_limits.append(map_limits)

                    if colormap_style == "shared":
                        ######### Global colorbar #########
                        # Compute the global limits
                        global_limits = (
                            min(l[0] for l in maps_limits),
                            max(l[1] for l in maps_limits),
                        )
                        cb_dict = {}
                        cb_dict["colormap"] = colormaps[0]
                        cb_dict["map_name"] = " + ".join(maps_names)
                        cb_dict["vmin"] = global_limits[0]
                        cb_dict["vmax"] = global_limits[1]
                        if colorbar_titles:
                            cb_dict["title"] = colorbar_titles[0]
                        else:
                            cb_dict["title"] = " + ".join(maps_names)

                        for map_idx in range(n_maps):
                            for surf_idx in range(n_surfaces):
                                brain_positions[(map_idx, surf_idx, 0)] = (
                                    surf_idx,
                                    map_idx,
                                )
                                colormap_limits[(map_idx, surf_idx, 0)] = (
                                    global_limits + (maps_names[0],)
                                )

                        if colorbar_position == "right":
                            shape = [n_surfaces, n_maps + 1]
                            row_weights = [1] * n_surfaces
                            col_weights = [1] * n_maps + [colorbar_size]
                            groups = [(slice(0, n_surfaces), n_maps)]

                            cb_dict["position"] = (0, n_maps)
                            cb_dict["orientation"] = "vertical"

                        elif colorbar_position == "bottom":
                            shape = [n_surfaces + 1, n_maps]
                            row_weights = [1] * n_surfaces + [colorbar_size]
                            col_weights = [1] * n_maps
                            groups = [(n_surfaces, slice(0, n_maps))]

                            cb_dict["position"] = (n_surfaces, 0)
                            cb_dict["orientation"] = "horizontal"

                        colorbar_list.append(cb_dict)

                    elif colormap_style == "shared_by_map":
                        ######### One colorbar per map #########
                        for map_idx in range(n_maps):
                            if not maps_names[map_idx] in surfaces[0].colortables:
                                map_limits = maps_limits[map_idx]

                                for surf_idx in range(n_surfaces):
                                    brain_positions[(map_idx, surf_idx, 0)] = (
                                        surf_idx,
                                        map_idx,
                                    )
                                    colormap_limits[(map_idx, surf_idx, 0)] = (
                                        maps_limits[map_idx]
                                    )
                            else:
                                for surf_idx in range(n_surfaces):
                                    brain_positions[(map_idx, surf_idx, 0)] = (
                                        surf_idx,
                                        map_idx,
                                    )
                                    colormap_limits[(map_idx, surf_idx, 0)] = (
                                        None,
                                        None,
                                        maps_names[map_idx],
                                    )

                        shape = [n_surfaces + 1, n_maps]
                        row_weights = [1] * n_surfaces + [colorbar_size]
                        col_weights = [1] * n_maps
                        groups = []

                        for map_idx in range(n_maps):
                            cb_dict = {}
                            cb_dict["position"] = (n_surfaces, map_idx)
                            cb_dict["orientation"] = "horizontal"
                            cb_dict["colormap"] = (
                                colormaps[map_idx]
                                if map_idx < len(colormaps)
                                else colormaps[-1]
                            )
                            cb_dict["map_name"] = maps_names[map_idx]
                            if colorbar_titles:
                                cb_dict["title"] = (
                                    colorbar_titles[map_idx]
                                    if map_idx < len(colorbar_titles)
                                    else colorbar_titles[-1]
                                )
                            else:
                                cb_dict["title"] = maps_names[map_idx]

                            if maps_names[map_idx] not in surfaces[0].colortables:
                                cb_dict["vmin"] = maps_limits[map_idx][0]
                                cb_dict["vmax"] = maps_limits[map_idx][1]
                                colorbar_list.append(cb_dict)

        else:  # vertical

            if not colorbar:
                # Maps in rows, surfaces in columns
                shape = [n_maps, n_surfaces]
                row_weights = [1] * n_maps
                col_weights = [1] * n_surfaces
                groups = []

                for map_idx in range(n_maps):
                    for surf_idx in range(n_surfaces):
                        brain_positions[(map_idx, surf_idx, 0)] = (map_idx, surf_idx)

            # Force colorbar to right for this case
            if colormap_style == "individual":
                if colorbar_position == "right":
                    shape = [n_maps, n_surfaces * 2]
                    row_weights = [1] * n_maps
                    col_weights = [1, colorbar_size] * n_surfaces
                    groups = []

                    for map_idx in range(n_maps):
                        for surf_idx in range(n_surfaces):
                            brain_positions[(map_idx, surf_idx, 0)] = (
                                map_idx,
                                surf_idx * 2,
                            )
                            cb_dict = {}
                            cb_dict["position"] = (map_idx, surf_idx * 2 + 1)
                            cb_dict["orientation"] = "vertical"
                            cb_dict["colormap"] = (
                                colormaps[map_idx]
                                if map_idx < len(colormaps)
                                else colormaps[-1]
                            )
                            cb_dict["map_name"] = maps_names[map_idx]
                            if colorbar_titles:
                                cb_dict["title"] = (
                                    colorbar_titles[map_idx]
                                    if map_idx < len(colorbar_titles)
                                    else colorbar_titles[-1]
                                )
                            else:
                                cb_dict["title"] = maps_names[map_idx]
                            limits_list = _get_map_limits(
                                surfaces=surfaces[surf_idx],
                                map_name=maps_names[map_idx],
                                colormap_style="individual",
                                v_limits=v_limits[map_idx],
                            )
                            cb_dict["vmin"] = limits_list[0][0]
                            cb_dict["vmax"] = limits_list[0][1]
                            colormap_limits[(map_idx, surf_idx, 0)] = limits_list[0]
                            if (
                                maps_names[map_idx]
                                not in surfaces[surf_idx].colortables
                            ):
                                colorbar_list.append(cb_dict)

                elif colorbar_position == "bottom":
                    shape = [n_maps * 2, n_surfaces]
                    row_weights = [1, colorbar_size] * n_maps
                    col_weights = [1] * n_surfaces
                    groups = []

                    for map_idx in range(n_maps):
                        for surf_idx in range(n_surfaces):
                            brain_positions[(map_idx, surf_idx, 0)] = (
                                map_idx * 2,
                                surf_idx,
                            )
                            cb_dict = {}
                            cb_dict["position"] = (map_idx * 2 + 1, surf_idx)
                            cb_dict["orientation"] = "horizontal"
                            cb_dict["colormap"] = (
                                colormaps[map_idx]
                                if map_idx < len(colormaps)
                                else colormaps[-1]
                            )
                            cb_dict["map_name"] = maps_names[map_idx]
                            if colorbar_titles:
                                cb_dict["title"] = (
                                    colorbar_titles[map_idx]
                                    if map_idx < len(colorbar_titles)
                                    else colorbar_titles[-1]
                                )
                            else:
                                cb_dict["title"] = maps_names[map_idx]
                            limits_list = _get_map_limits(
                                surfaces=surfaces[surf_idx],
                                map_name=maps_names[map_idx],
                                colormap_style="individual",
                                v_limits=v_limits[map_idx],
                            )
                            cb_dict["vmin"] = limits_list[0][0]
                            cb_dict["vmax"] = limits_list[0][1]
                            colormap_limits[(map_idx, surf_idx, 0)] = limits_list[0]
                            if (
                                maps_names[map_idx]
                                not in surfaces[surf_idx].colortables
                            ):
                                colorbar_list.append(cb_dict)

            else:
                # Map-wise limits
                maps_limits = []
                for map_idx in range(n_maps):
                    if any(
                        maps_names[map_idx] not in surface.colortables
                        for surface in surfaces
                    ):
                        map_limits = _get_map_limits(
                            surfaces=surfaces,
                            map_name=maps_names[map_idx],
                            colormap_style="shared",
                            v_limits=v_limits[map_idx],
                        )[0]
                        maps_limits.append(map_limits)

                if colormap_style == "shared":
                    ######### Global colorbar #########
                    # Compute the global limits
                    global_limits = (
                        min(l[0] for l in maps_limits),
                        max(l[1] for l in maps_limits),
                    )
                    cb_dict = {}
                    cb_dict["colormap"] = colormaps[0]
                    cb_dict["map_name"] = " + ".join(maps_names)
                    cb_dict["vmin"] = global_limits[0]
                    cb_dict["vmax"] = global_limits[1]
                    if colorbar_titles:
                        cb_dict["title"] = colorbar_titles[0]
                    else:
                        cb_dict["title"] = " + ".join(maps_names)

                    for map_idx in range(n_maps):
                        for surf_idx in range(n_surfaces):
                            brain_positions[(map_idx, surf_idx, 0)] = (
                                map_idx,
                                surf_idx,
                            )
                            colormap_limits[(map_idx, surf_idx, 0)] = global_limits + (
                                maps_names[0],
                            )

                    if colorbar_position == "right":
                        shape = [n_maps, n_surfaces + 1]
                        row_weights = [1] * n_maps
                        col_weights = [1] * n_surfaces + [colorbar_size]
                        groups = [(slice(0, n_maps), n_surfaces)]

                        cb_dict["position"] = (0, n_surfaces)
                        cb_dict["orientation"] = "vertical"

                    elif colorbar_position == "bottom":
                        shape = [n_maps + 1, n_surfaces]
                        row_weights = [1] * n_maps + [colorbar_size]
                        col_weights = [1] * n_surfaces
                        groups = [(n_maps, slice(0, n_surfaces))]

                        cb_dict["position"] = (n_maps, 0)
                        cb_dict["orientation"] = "horizontal"

                    if any(
                        maps_names[map_idx] not in surface.colortables
                        for map_idx in range(n_maps)
                        for surface in surfaces
                    ):
                        colorbar_list.append(cb_dict)

                elif colormap_style == "shared_by_map":
                    ######### One colorbar per map #########
                    for map_idx in range(n_maps):
                        if maps_names[map_idx] not in surfaces[0].colortables:
                            map_limits = maps_limits[map_idx]
                            for surf_idx in range(n_surfaces):
                                brain_positions[(map_idx, surf_idx, 0)] = (
                                    map_idx,
                                    surf_idx,
                                )
                                colormap_limits[(map_idx, surf_idx, 0)] = maps_limits[
                                    map_idx
                                ]
                        else:
                            for surf_idx in range(n_surfaces):
                                brain_positions[(map_idx, surf_idx, 0)] = (
                                    map_idx,
                                    surf_idx,
                                )
                                colormap_limits[(map_idx, surf_idx, 0)] = (
                                    None,
                                    None,
                                    maps_names[map_idx],
                                )

                    shape = [n_maps, n_surfaces + 1]
                    row_weights = [1] * n_maps
                    col_weights = [1] * n_surfaces + [colorbar_size]
                    groups = []

                    for map_idx in range(n_maps):
                        cb_dict = {}
                        cb_dict["position"] = (map_idx, n_surfaces)
                        cb_dict["orientation"] = "vertical"
                        cb_dict["colormap"] = (
                            colormaps[map_idx]
                            if map_idx < len(colormaps)
                            else colormaps[-1]
                        )
                        cb_dict["map_name"] = maps_names[map_idx]
                        if colorbar_titles:
                            cb_dict["title"] = (
                                colorbar_titles[map_idx]
                                if map_idx < len(colorbar_titles)
                                else colorbar_titles[-1]
                            )
                        else:
                            cb_dict["title"] = maps_names[map_idx]
                        if maps_names[map_idx] not in surfaces[0].colortables:
                            cb_dict["vmin"] = maps_limits[map_idx][0]
                            cb_dict["vmax"] = maps_limits[map_idx][1]
                            colorbar_list.append(cb_dict)

        layout_config = {
            "shape": shape,
            "row_weights": row_weights,
            "col_weights": col_weights,
            "groups": groups,
            "brain_positions": brain_positions,
            "colormap_limits": colormap_limits,
        }
        return layout_config, colorbar_list

    def _multi_view_single_element_layout(
        self,
        surface,
        view_ids,
        map_name,
        v_limits,
        colormap,
        colorbar_title,
        orientation,
        colorbar,
        colorbar_position,
        colorbar_size,
    ):
        """Handle multiple views, single map, single surface case."""

        if map_name in surface.colortables:
            colorbar = False

        n_views = len(view_ids)
        brain_positions = {}
        colormap_limits = {}
        colorbar_list = []

        map_limits = _get_map_limits(
            surfaces=surface,
            map_name=map_name,
            colormap_style="individual",
            v_limits=v_limits,
        )[0]
        if orientation == "horizontal":
            for view_idx in range(n_views):
                brain_positions[(0, 0, view_idx)] = (0, view_idx)
                colormap_limits[(0, 0, view_idx)] = map_limits

            if not colorbar:
                shape = [1, n_views]
                row_weights = [1]
                col_weights = [1] * n_views
                groups = []

            else:
                shape = [1, n_views + 1]
                row_weights = [1]
                col_weights = [1] * n_views + [colorbar_size]
                groups = []

                cb_dict = {}
                cb_dict["colormap"] = colormap
                cb_dict["map_name"] = map_name
                cb_dict["vmin"] = map_limits[0]
                cb_dict["vmax"] = map_limits[1]
                if colorbar_title:
                    cb_dict["title"] = colorbar_title
                else:
                    cb_dict["title"] = map_name

                if colorbar_position == "right":
                    cb_dict["position"] = (0, n_views)
                    cb_dict["orientation"] = "vertical"
                else:  # bottom
                    shape = [2, n_views]
                    row_weights = [1, colorbar_size]
                    col_weights = [1] * n_views
                    groups = [(1, slice(0, n_views))]
                    cb_dict["position"] = (1, 0)
                    cb_dict["orientation"] = "horizontal"

                colorbar_list.append(cb_dict)

        elif orientation == "vertical":
            for view_idx in range(n_views):
                brain_positions[(0, 0, view_idx)] = (view_idx, 0)
                colormap_limits[(0, 0, view_idx)] = map_limits

            if not colorbar:
                shape = [n_views, 1]
                row_weights = [1] * n_views
                col_weights = [1]
                groups = []

            else:
                shape = [n_views, 2]
                row_weights = [1] * n_views
                col_weights = [1, colorbar_size]
                groups = []

                cb_dict = {}
                cb_dict["colormap"] = colormap
                cb_dict["map_name"] = map_name
                cb_dict["vmin"] = map_limits[0]
                cb_dict["vmax"] = map_limits[1]
                if colorbar_title:
                    cb_dict["title"] = colorbar_title
                else:
                    cb_dict["title"] = map_name

                if colorbar_position == "right":
                    cb_dict["position"] = (0, 1)
                    cb_dict["orientation"] = "vertical"
                    groups = [(slice(0, n_views), 1)]
                else:
                    shape = [n_views + 1, 1]
                    row_weights = [1] * n_views + [colorbar_size]
                    col_weights = [1]
                    cb_dict["position"] = (n_views, 0)
                    cb_dict["orientation"] = "horizontal"
                colorbar_list.append(cb_dict)
        else:
            optimal_grid, positions = cltplot.calculate_optimal_subplots_grid(n_views)
            for view_idx in range(n_views):
                pos = positions[view_idx]
                brain_positions[(0, 0, view_idx)] = pos
                colormap_limits[(0, 0, view_idx)] = map_limits
            if not colorbar:
                shape = list(optimal_grid)
                row_weights = [1] * optimal_grid[0]
                col_weights = [1] * optimal_grid[1]
                groups = []
            else:
                shape = [optimal_grid[0], optimal_grid[1] + 1]
                row_weights = [1] * optimal_grid[0]
                col_weights = [1] * optimal_grid[1] + [colorbar_size]
                groups = [(slice(0, optimal_grid[0]), optimal_grid[1])]

                cb_dict = {}
                cb_dict["colormap"] = colormap
                cb_dict["map_name"] = map_name
                cb_dict["vmin"] = map_limits[0]
                cb_dict["vmax"] = map_limits[1]
                if colorbar_title:
                    cb_dict["title"] = colorbar_title
                else:
                    cb_dict["title"] = map_name

                if colorbar_position == "right":
                    cb_dict["position"] = (0, optimal_grid[1])
                    cb_dict["orientation"] = "vertical"
                    shape = [optimal_grid[0], optimal_grid[1] + 1]
                    row_weights = [1] * optimal_grid[0]
                    col_weights = [1] * optimal_grid[1] + [colorbar_size]
                    groups = [(slice(0, optimal_grid[0]), optimal_grid[1])]

                else:  # bottom
                    shape = [optimal_grid[0] + 1, optimal_grid[1]]
                    row_weights = [1] * optimal_grid[0] + [colorbar_size]
                    col_weights = [1] * optimal_grid[1]
                    groups = [(optimal_grid[0], slice(0, optimal_grid[1]))]
                    cb_dict["position"] = (optimal_grid[0], 0)
                    cb_dict["orientation"] = "horizontal"
                colorbar_list.append(cb_dict)

        layout_config = {
            "shape": shape,
            "row_weights": row_weights,
            "col_weights": col_weights,
            "groups": groups,
            "brain_positions": brain_positions,
            "colormap_limits": colormap_limits,
        }
        return layout_config, colorbar_list

    def _multi_view_multi_surface_layout(
        self,
        surfaces,
        valid_views,
        maps_names,
        v_limits,
        colormaps,
        colorbar_titles,
        orientation,
        colorbar,
        colorbar_position,
        colormap_style,
        colorbar_size,
    ):
        """Handle multiple views and multiple surfaces case."""
        n_surfaces = len(surfaces)
        n_views = len(valid_views)
        brain_positions = {}
        colormap_limits = {}
        colorbar_list = []
        if orientation == "horizontal":
            if not colorbar:
                shape = [n_surfaces, n_views]
                row_weights = [1] * n_surfaces
                col_weights = [1] * n_views
                groups = []

                # Views in columns, surfaces in rows
                for view_idx in range(n_views):
                    for surf_idx in range(n_surfaces):
                        brain_positions[(0, surf_idx, view_idx)] = (surf_idx, view_idx)
                        map_limits = _get_map_limits(
                            surfaces=surfaces[surf_idx],
                            map_name=maps_names[0],
                            colormap_style="individual",
                            v_limits=v_limits[0],
                        )[0]
                        colormap_limits[(0, surf_idx, view_idx)] = map_limits
            else:

                if colormap_style == "individual":
                    shape = [n_surfaces, n_views + 1]
                    row_weights = [1] * n_surfaces
                    col_weights = [1] * n_views + [colorbar_size]
                    groups = []
                    for surf_idx in range(n_surfaces):
                        cb_dict = {}
                        cb_dict["position"] = (surf_idx, n_views)
                        cb_dict["orientation"] = "vertical"
                        cb_dict["colormap"] = (
                            colormaps[0] if 0 < len(colormaps) else "viridis"
                        )
                        cb_dict["map_name"] = maps_names[0]
                        if colorbar_titles:
                            cb_dict["title"] = colorbar_titles[0]
                        else:
                            cb_dict["title"] = maps_names[0]
                        limits_list = _get_map_limits(
                            surfaces=surfaces[surf_idx],
                            map_name=maps_names[0],
                            colormap_style="individual",
                            v_limits=v_limits[0],
                        )
                        cb_dict["vmin"] = limits_list[0][0]
                        cb_dict["vmax"] = limits_list[0][1]

                        for view_idx in range(n_views):
                            brain_positions[(0, surf_idx, view_idx)] = (
                                surf_idx,
                                view_idx,
                            )
                            colormap_limits[(0, surf_idx, view_idx)] = limits_list[0]
                        colorbar_list.append(cb_dict)
                else:
                    # View-wise limits
                    views_limits = []
                    for view_idx in range(n_views):
                        view_limits = _get_map_limits(
                            surfaces=surfaces,
                            map_name=maps_names[0],
                            colormap_style="shared",
                            v_limits=v_limits[0],
                        )[0]
                        views_limits.append(view_limits)

                    ######### Global colorbar #########
                    # Compute the global limits
                    global_limits = (
                        min(l[0] for l in views_limits),
                        max(l[1] for l in views_limits),
                    )
                    cb_dict = {}
                    cb_dict["colormap"] = (
                        colormaps[0] if 0 < len(colormaps) else "viridis"
                    )
                    cb_dict["map_name"] = maps_names[0]
                    cb_dict["vmin"] = global_limits[0]
                    cb_dict["vmax"] = global_limits[1]
                    if colorbar_titles:
                        cb_dict["title"] = colorbar_titles[0]
                    else:
                        cb_dict["title"] = maps_names[0]

                    for view_idx in range(n_views):
                        for surf_idx in range(n_surfaces):
                            brain_positions[(0, surf_idx, view_idx)] = (
                                surf_idx,
                                view_idx,
                            )
                            colormap_limits[(0, surf_idx, view_idx)] = global_limits + (
                                maps_names[0],
                            )

                    if colorbar_position == "right":
                        shape = [n_surfaces, n_views + 1]
                        row_weights = [1] * n_surfaces
                        col_weights = [1] * n_views + [colorbar_size]
                        groups = [(slice(0, n_surfaces), n_views)]

                        cb_dict["position"] = (0, n_views)
                        cb_dict["orientation"] = "vertical"

                    elif colorbar_position == "bottom":
                        shape = [n_surfaces + 1, n_views]
                        row_weights = [1] * n_surfaces + [colorbar_size]
                        col_weights = [1] * n_views
                        groups = [(n_surfaces, slice(0, n_views))]

                        cb_dict["position"] = (n_surfaces, 0)
                        cb_dict["orientation"] = "horizontal"
                    colorbar_list.append(cb_dict)

        else:  # vertical
            if not colorbar:
                # Views in rows, surfaces in columns
                shape = [n_views, n_surfaces]
                row_weights = [1] * n_views
                col_weights = [1] * n_surfaces
                groups = []

                for view_idx in range(n_views):
                    for surf_idx in range(n_surfaces):
                        brain_positions[(0, surf_idx, view_idx)] = (view_idx, surf_idx)
                        map_limits = _get_map_limits(
                            surfaces=surfaces[surf_idx],
                            map_name=maps_names[0],
                            colormap_style="individual",
                            v_limits=v_limits[0],
                        )[0]
                        colormap_limits[(0, surf_idx, view_idx)] = map_limits
            else:

                if colormap_style == "individual":
                    shape = [n_views + 1, n_surfaces]
                    row_weights = [1] * n_views + [colorbar_size]
                    col_weights = [1] * n_surfaces
                    groups = []
                    for surf_idx in range(n_surfaces):
                        cb_dict = {}
                        cb_dict["position"] = (n_views, surf_idx)
                        cb_dict["orientation"] = "horizontal"
                        cb_dict["colormap"] = (
                            colormaps[0] if 0 < len(colormaps) else "viridis"
                        )
                        cb_dict["map_name"] = maps_names[0]
                        if colorbar_titles:
                            cb_dict["title"] = colorbar_titles[0]
                        else:
                            cb_dict["title"] = maps_names[0]
                        limits_list = _get_map_limits(
                            surfaces=surfaces[surf_idx],
                            map_name=maps_names[0],
                            colormap_style="individual",
                            v_limits=v_limits[0],
                        )
                        cb_dict["vmin"] = limits_list[0][0]
                        cb_dict["vmax"] = limits_list[0][1]

                        for view_idx in range(n_views):
                            brain_positions[(0, surf_idx, view_idx)] = (
                                view_idx,
                                surf_idx,
                            )
                            colormap_limits[(0, surf_idx, view_idx)] = limits_list[0]

                        colorbar_list.append(cb_dict)
                else:
                    # View-wise limits
                    views_limits = []
                    for view_idx in range(n_views):
                        view_limits = _get_map_limits(
                            surfaces=surfaces,
                            map_name=maps_names[0],
                            colormap_style="shared",
                            v_limits=v_limits[0],
                        )[0]
                        views_limits.append(view_limits)

                    ######### Global colorbar #########
                    # Compute the global limits
                    global_limits = (
                        min(l[0] for l in views_limits),
                        max(l[1] for l in views_limits),
                    )
                    cb_dict = {}
                    cb_dict["colormap"] = (
                        colormaps[0] if 0 < len(colormaps) else "viridis"
                    )
                    cb_dict["map_name"] = maps_names[0]
                    cb_dict["vmin"] = global_limits[0]
                    cb_dict["vmax"] = global_limits[1]
                    if colorbar_titles:
                        cb_dict["title"] = colorbar_titles[0]
                    else:
                        cb_dict["title"] = maps_names[0]

                    for view_idx in range(n_views):
                        for surf_idx in range(n_surfaces):
                            brain_positions[(0, surf_idx, view_idx)] = (
                                view_idx,
                                surf_idx,
                            )
                            colormap_limits[(0, surf_idx, view_idx)] = global_limits + (
                                maps_names[0],
                            )

                    if colorbar_position == "right":
                        shape = [n_views, n_surfaces + 1]
                        row_weights = [1] * n_views
                        col_weights = [1] * n_surfaces + [colorbar_size]
                        groups = [(slice(0, n_views), n_surfaces)]

                        cb_dict["position"] = (0, n_surfaces)
                        cb_dict["orientation"] = "vertical"

                    elif colorbar_position == "bottom":
                        shape = [n_views + 1, n_surfaces]
                        row_weights = [1] * n_views + [colorbar_size]
                        col_weights = [1] * n_surfaces
                        groups = [(n_views, slice(0, n_surfaces))]

                        cb_dict["position"] = (n_views, 0)
                        cb_dict["orientation"] = "horizontal"
                    colorbar_list.append(cb_dict)

        layout_config = {
            "shape": shape,
            "row_weights": row_weights,
            "col_weights": col_weights,
            "groups": groups,
            "brain_positions": brain_positions,
            "colormap_limits": colormap_limits,
        }
        return layout_config, colorbar_list

    def _multi_view_multi_map_layout(
        self,
        surfaces,
        valid_views,
        maps_names,
        v_limits,
        colormaps,
        colorbar_titles,
        orientation,
        colorbar,
        colormap_style,
        colorbar_position,
        colorbar_size,
    ):
        """Handle multiple views and multiple maps case."""

        n_views = len(valid_views)
        n_maps = len(maps_names)

        brain_positions = {}
        colormap_limits = {}
        colorbar_list = []
        if orientation == "horizontal":
            if not colorbar:
                shape = [n_maps, n_views]
                row_weights = [1] * n_maps
                col_weights = [1] * n_views
                groups = []

                # Views in columns, maps in rows
                for view_idx in range(n_views):
                    for map_idx in range(n_maps):
                        brain_positions[(map_idx, 0, view_idx)] = (map_idx, view_idx)
                        map_limits = _get_map_limits(
                            surfaces=surfaces[0],
                            map_name=maps_names[map_idx],
                            colormap_style="individual",
                            v_limits=v_limits[map_idx],
                        )[0]
                        colormap_limits[(map_idx, 0, view_idx)] = map_limits
            else:
                if colormap_style == "individual":
                    shape = [n_maps, n_views + 1]
                    row_weights = [1] * n_maps
                    col_weights = [1] * n_views + [colorbar_size]
                    groups = []
                    for map_idx in range(n_maps):
                        cb_dict = {}
                        cb_dict["position"] = (map_idx, n_views)
                        cb_dict["orientation"] = "vertical"
                        cb_dict["colormap"] = (
                            colormaps[map_idx]
                            if map_idx < len(colormaps)
                            else colormaps[-1]
                        )
                        cb_dict["map_name"] = maps_names[map_idx]
                        if colorbar_titles:
                            cb_dict["title"] = (
                                colorbar_titles[map_idx]
                                if map_idx < len(colorbar_titles)
                                else colorbar_titles[-1]
                            )
                        else:
                            cb_dict["title"] = maps_names[map_idx]
                        limits_list = _get_map_limits(
                            surfaces=surfaces[0],
                            map_name=maps_names[map_idx],
                            colormap_style="individual",
                            v_limits=v_limits[map_idx],
                        )
                        cb_dict["vmin"] = limits_list[0][0]
                        cb_dict["vmax"] = limits_list[0][1]

                        for view_idx in range(n_views):
                            brain_positions[(map_idx, 0, view_idx)] = (
                                map_idx,
                                view_idx,
                            )
                            colormap_limits[(map_idx, 0, view_idx)] = limits_list[0]
                        if maps_names[map_idx] not in surfaces[0].colortables:
                            colorbar_list.append(cb_dict)

                else:
                    # Get the global limits
                    maps_limits = []
                    for map_idx in range(n_maps):
                        if maps_names[map_idx] not in surfaces[0].colortables:
                            map_limits = _get_map_limits(
                                surfaces=surfaces,
                                map_name=maps_names[map_idx],
                                colormap_style="shared",
                                v_limits=v_limits[map_idx],
                            )[0]
                            maps_limits.append(map_limits)

                    ######### Global colorbar #########
                    # Compute the global limits
                    global_limits = (
                        min(l[0] for l in maps_limits),
                        max(l[1] for l in maps_limits),
                    )
                    cb_dict = {}
                    cb_dict["colormap"] = (
                        colormaps[0] if 0 < len(colormaps) else "viridis"
                    )
                    cb_dict["map_name"] = " + ".join(maps_names)
                    cb_dict["vmin"] = global_limits[0]
                    cb_dict["vmax"] = global_limits[1]
                    if colorbar_titles:
                        cb_dict["title"] = colorbar_titles[0]
                    else:
                        cb_dict["title"] = " + ".join(maps_names)

                    for map_idx in range(n_maps):
                        for view_idx in range(n_views):
                            brain_positions[(map_idx, 0, view_idx)] = (
                                map_idx,
                                view_idx,
                            )
                            colormap_limits[(map_idx, 0, view_idx)] = global_limits + (
                                maps_names[0],
                            )

                    if colorbar_position == "right":
                        shape = [n_maps, n_views + 1]
                        row_weights = [1] * n_maps
                        col_weights = [1] * n_views + [colorbar_size]
                        groups = [(slice(0, n_maps), n_views)]

                        cb_dict["position"] = (0, n_views)
                        cb_dict["orientation"] = "vertical"

                    elif colorbar_position == "bottom":
                        shape = [n_maps + 1, n_views]
                        row_weights = [1] * n_maps + [colorbar_size]
                        col_weights = [1] * n_views
                        groups = [(n_maps, slice(0, n_views))]

                        cb_dict["position"] = (n_maps, 0)
                        cb_dict["orientation"] = "horizontal"
                    colorbar_list.append(cb_dict)
        else:  # vertical
            if not colorbar:
                # Views in rows, maps in columns
                shape = [n_views, n_maps]
                row_weights = [1] * n_views
                col_weights = [1] * n_maps
                groups = []

                for view_idx in range(n_views):
                    for map_idx in range(n_maps):
                        brain_positions[(map_idx, 0, view_idx)] = (view_idx, map_idx)
                        map_limits = _get_map_limits(
                            surfaces=surfaces[0],
                            map_name=maps_names[map_idx],
                            colormap_style="individual",
                            v_limits=v_limits[map_idx],
                        )[0]
                        colormap_limits[(map_idx, 0, view_idx)] = map_limits
            else:
                if colormap_style == "individual":
                    shape = [n_views + 1, n_maps]
                    row_weights = [1] * n_views + [colorbar_size]
                    col_weights = [1] * n_maps
                    groups = []

                    for map_idx in range(n_maps):
                        cb_dict = {}
                        cb_dict["position"] = (n_views, map_idx)
                        cb_dict["orientation"] = "horizontal"
                        cb_dict["colormap"] = (
                            colormaps[map_idx]
                            if map_idx < len(colormaps)
                            else colormaps[-1]
                        )
                        cb_dict["map_name"] = maps_names[map_idx]
                        if colorbar_titles:
                            cb_dict["title"] = (
                                colorbar_titles[map_idx]
                                if map_idx < len(colorbar_titles)
                                else colorbar_titles[-1]
                            )
                        else:
                            cb_dict["title"] = maps_names[map_idx]
                        limits_list = _get_map_limits(
                            surfaces=surfaces[0],
                            map_name=maps_names[map_idx],
                            colormap_style="individual",
                            v_limits=v_limits[map_idx],
                        )
                        cb_dict["vmin"] = limits_list[0][0]
                        cb_dict["vmax"] = limits_list[0][1]

                        for view_idx in range(n_views):
                            brain_positions[(map_idx, 0, view_idx)] = (
                                view_idx,
                                map_idx,
                            )
                            colormap_limits[(map_idx, 0, view_idx)] = limits_list[0]

                        if maps_names[map_idx] not in surfaces[0].colortables:
                            colorbar_list.append(cb_dict)
                else:
                    # Get the global limits
                    maps_limits = []
                    for map_idx in range(n_maps):
                        if maps_names[map_idx] not in surfaces[0].colortables:
                            map_limits = _get_map_limits(
                                surfaces=surfaces,
                                map_name=maps_names[map_idx],
                                colormap_style="shared",
                                v_limits=v_limits[map_idx],
                            )[0]
                            maps_limits.append(map_limits)

                    ######### Global colorbar #########
                    # Compute the global limits
                    global_limits = (
                        min(l[0] for l in maps_limits),
                        max(l[1] for l in maps_limits),
                    )
                    cb_dict = {}
                    cb_dict["colormap"] = (
                        colormaps[0] if 0 < len(colormaps) else "viridis"
                    )
                    cb_dict["map_name"] = " + ".join(maps_names)
                    cb_dict["vmin"] = global_limits[0]
                    cb_dict["vmax"] = global_limits[1]
                    if colorbar_titles:
                        cb_dict["title"] = colorbar_titles[0]
                    else:
                        cb_dict["title"] = " + ".join(maps_names)
                    for map_idx in range(n_maps):
                        for view_idx in range(n_views):
                            brain_positions[(map_idx, 0, view_idx)] = (
                                view_idx,
                                map_idx,
                            )
                            colormap_limits[(map_idx, 0, view_idx)] = global_limits + (
                                maps_names[0],
                            )
                    if colorbar_position == "right":
                        shape = [n_views, n_maps + 1]
                        row_weights = [1] * n_views
                        col_weights = [1] * n_maps + [colorbar_size]
                        groups = [(slice(0, n_views), n_maps)]

                        cb_dict["position"] = (0, n_maps)
                        cb_dict["orientation"] = "vertical"

                    elif colorbar_position == "bottom":
                        shape = [n_views + 1, n_maps]
                        row_weights = [1] * n_views + [colorbar_size]
                        col_weights = [1] * n_maps
                        groups = [(n_views, slice(0, n_maps))]

                        cb_dict["position"] = (n_views, 0)
                        cb_dict["orientation"] = "horizontal"
                    colorbar_list.append(cb_dict)
        layout_config = {
            "shape": shape,
            "row_weights": row_weights,
            "col_weights": col_weights,
            "groups": groups,
            "brain_positions": brain_positions,
            "colormap_limits": colormap_limits,
        }
        return layout_config, colorbar_list

    def _single_map_multi_surface_layout(
        self,
        surfaces,
        maps_names,
        v_limits,
        colormaps,
        colorbar_titles,
        orientation,
        colorbar,
        colormap_style,
        colorbar_position,
        colorbar_size,
    ):
        """Handle single map, multiple surfaces case."""
        brain_positions = {}
        n_surfaces = len(surfaces)

        # Getting the limits for each surface and storing them in a list
        limits_list = _get_map_limits(
            surfaces=surfaces,
            map_name=maps_names[0],
            colormap_style=colormap_style,
            v_limits=v_limits[0],
        )

        if orientation == "horizontal":
            return self._horizontal_multi_surface_layout(
                surfaces,
                maps_names,
                limits_list,
                colormaps,
                colorbar_titles,
                colorbar,
                colormap_style,
                colorbar_position,
                colorbar_size,
            )
        elif orientation == "vertical":
            return self._vertical_multi_surface_layout(
                surfaces,
                maps_names,
                limits_list,
                colormaps,
                colorbar_titles,
                colorbar,
                colormap_style,
                colorbar_position,
                colorbar_size,
            )
        else:  # grid
            return self._grid_multi_surface_layout(
                surfaces,
                maps_names,
                limits_list,
                colormaps,
                colorbar_titles,
                colorbar,
                colormap_style,
                colorbar_position,
                colorbar_size,
            )

    def _horizontal_multi_surface_layout(
        self,
        surfaces,
        maps_names,
        limits_list,
        colormaps,
        colorbar_titles,
        colorbar,
        colormap_style,
        colorbar_position,
        colorbar_size,
    ):
        """Handle horizontal layout for multiple surfaces."""
        brain_positions = {}
        colormap_limits = {}

        n_surfaces = len(surfaces)
        for surf_idx in range(n_surfaces):
            brain_positions[(0, surf_idx, 0)] = (0, surf_idx)
            colormap_limits[(0, surf_idx, 0)] = limits_list[surf_idx]

        colorbar_list = []
        if not colorbar:
            shape = [1, n_surfaces]
            row_weights = [1]
            col_weights = [1] * n_surfaces
            groups = []

        else:
            if colormap_style == "individual":
                for surf_idx in range(n_surfaces):
                    if maps_names[0] in surfaces[surf_idx].colortables:
                        indiv_colorbar = False
                    else:
                        indiv_colorbar = True

                    if indiv_colorbar:
                        groups = []
                        cb_dict = {}

                        cb_dict["vmin"] = limits_list[surf_idx][0]
                        cb_dict["vmax"] = limits_list[surf_idx][1]

                        if colorbar_titles:
                            if colorbar_titles[0]:
                                cb_dict["title"] = colorbar_titles[0]
                            else:
                                cb_dict["title"] = maps_names[0]
                        else:
                            cb_dict["title"] = maps_names[0]

                        cb_dict["colormap"] = colormaps[0]
                        cb_dict["map_name"] = maps_names[0]

                        if colorbar_position == "right":
                            shape = [1, n_surfaces * 2]
                            row_weights = [1]
                            col_weights = [1, colorbar_size] * n_surfaces
                            brain_positions[(0, surf_idx, 0)] = (0, surf_idx * 2)
                            colormap_limits[(0, surf_idx, 0)] = limits_list[surf_idx]

                            cb_dict["position"] = (0, surf_idx * 2 + 1)
                            cb_dict["orientation"] = "vertical"

                        else:  # bottom
                            shape = [2, n_surfaces]
                            row_weights = [1, colorbar_size]
                            col_weights = [1] * n_surfaces
                            brain_positions[(0, surf_idx, 0)] = (0, surf_idx)
                            colormap_limits[(0, surf_idx, 0)] = limits_list[surf_idx]

                            cb_dict["position"] = (1, surf_idx)
                            cb_dict["orientation"] = "horizontal"
                    else:
                        cb_dict = False

                    colorbar_list.append(cb_dict)

            else:  # shared colorbar
                cb_dict = {}
                if colorbar_titles:
                    cb_dict["title"] = colorbar_titles[0]
                else:
                    cb_dict["title"] = maps_names[0]

                cb_dict["colormap"] = colormaps[0]
                cb_dict["map_name"] = maps_names[0]

                cb_dict["vmin"] = limits_list[0][0]
                cb_dict["vmax"] = limits_list[0][1]

                if colorbar_position == "right":
                    shape = [1, n_surfaces + 1]
                    row_weights = [1]
                    col_weights = [1] * n_surfaces + [colorbar_size]
                    groups = []
                    cb_dict["position"] = (0, n_surfaces)  # Colorbar in last column
                    cb_dict["orientation"] = "vertical"

                    for surf_idx in range(n_surfaces):
                        brain_positions[(0, surf_idx, 0)] = (0, surf_idx)
                        colormap_limits[(0, surf_idx, 0)] = limits_list[surf_idx]

                else:  # bottom
                    shape = [2, n_surfaces]
                    row_weights = [1, colorbar_size]
                    col_weights = [1] * n_surfaces
                    groups = [(1, slice(0, n_surfaces))]  # Colorbar in last row
                    cb_dict["position"] = (1, 0)  # Color
                    cb_dict["orientation"] = "horizontal"

                    for surf_idx in range(n_surfaces):
                        brain_positions[(0, surf_idx, 0)] = (0, surf_idx)
                        colormap_limits[(0, surf_idx, 0)] = limits_list[surf_idx]

                colorbar_list.append(cb_dict)

        layout_config = {
            "shape": shape,
            "row_weights": row_weights,
            "col_weights": col_weights,
            "groups": groups,
            "brain_positions": brain_positions,
            "colormap_limits": colormap_limits,
        }

        return layout_config, colorbar_list

    def _vertical_multi_surface_layout(
        self,
        surfaces,
        maps_names,
        limits_list,
        colormaps,
        colorbar_titles,
        colorbar,
        colormap_style,
        colorbar_position,
        colorbar_size,
    ):
        """Handle vertical layout for multiple surfaces."""
        brain_positions = {}
        colormap_limits = {}

        n_surfaces = len(surfaces)

        for surf_idx in range(n_surfaces):
            brain_positions[(0, surf_idx, 0)] = (surf_idx, 0)
            colormap_limits[(0, surf_idx, 0)] = limits_list[surf_idx]

        colorbar_list = []
        if not colorbar:
            shape = [n_surfaces, 1]
            row_weights = [1] * n_surfaces
            col_weights = [1]
            groups = []

        else:

            if colormap_style == "individual":
                for surf_idx in range(n_surfaces):
                    cb_dict = {}

                    cb_dict["vmin"] = limits_list[surf_idx][0]
                    cb_dict["vmax"] = limits_list[surf_idx][1]

                    if colorbar_titles:
                        cb_dict["title"] = colorbar_titles[0]
                    else:
                        cb_dict["title"] = maps_names[0]

                    cb_dict["colormap"] = colormaps[0]
                    cb_dict["map_name"] = maps_names[0]

                    if colorbar_position == "right":
                        shape = [n_surfaces, 2]
                        row_weights = [1] * n_surfaces
                        col_weights = [1, colorbar_size]
                        groups = []

                        cb_dict["position"] = (surf_idx, 1)
                        cb_dict["orientation"] = "vertical"
                        brain_positions[(0, surf_idx, 0)] = (surf_idx, 0)
                        colormap_limits[(0, surf_idx, 0)] = limits_list[surf_idx]

                    elif colorbar_position == "bottom":
                        shape = [n_surfaces * 2, 1]
                        row_weights = [1, colorbar_size] * n_surfaces
                        col_weights = [1]
                        groups = []

                        cb_dict["position"] = (surf_idx * 2 + 1, 0)
                        cb_dict["orientation"] = "horizontal"
                        brain_positions[(0, surf_idx, 0)] = (surf_idx * 2, 0)
                        colormap_limits[(0, surf_idx, 0)] = limits_list[surf_idx]

                    colorbar_list.append(cb_dict)

            else:  # shared colorbar
                cb_dict = {}
                if colorbar_titles:
                    cb_dict["title"] = colorbar_titles[0]
                else:
                    cb_dict["title"] = maps_names[0]

                cb_dict["colormap"] = colormaps[0]
                cb_dict["map_name"] = maps_names[0]

                cb_dict["vmin"] = limits_list[0][0]
                cb_dict["vmax"] = limits_list[0][1]

                if colorbar_position == "right":
                    shape = [n_surfaces, 2]
                    row_weights = [1] * n_surfaces
                    col_weights = [1, colorbar_size]
                    groups = [(slice(0, n_surfaces), 1)]
                    cb_dict["position"] = (0, 1)
                    cb_dict["orientation"] = "vertical"

                elif colorbar_position == "bottom":
                    shape = [n_surfaces + 1, 1]
                    row_weights = [1] * n_surfaces + [colorbar_size]
                    col_weights = [1]
                    groups = [(n_surfaces, slice(0, 1))]
                    cb_dict["position"] = (n_surfaces, 0)
                    cb_dict["orientation"] = "horizontal"

                colorbar_list.append(cb_dict)

        layout_config = {
            "shape": shape,
            "row_weights": row_weights,
            "col_weights": col_weights,
            "groups": groups,
            "brain_positions": brain_positions,
            "colormap_limits": colormap_limits,
        }
        return layout_config, colorbar_list

    def _grid_multi_surface_layout(
        self,
        surfaces,
        maps_names,
        limits_list,
        colormaps,
        colorbar_titles,
        colorbar,
        colormap_style,
        colorbar_position,
        colorbar_size,
    ):
        """Handle grid layout for multiple surfaces."""

        n_surfaces = len(surfaces)
        optimal_grid, positions = cltplot.calculate_optimal_subplots_grid(n_surfaces)
        brain_positions = {}
        colormap_limits = {}

        for surf_idx in range(n_surfaces):
            brain_positions[(0, surf_idx, 0)] = positions[surf_idx]
            colormap_limits[(0, surf_idx, 0)] = limits_list[surf_idx]

        colorbar_list = []
        if not colorbar:
            shape = optimal_grid
            row_weights = [1] * optimal_grid[0]
            col_weights = [1] * optimal_grid[1]
            groups = []

        else:
            if colormap_style == "individual":
                for surf_idx in range(n_surfaces):
                    cb_dict = {}

                    cb_dict["vmin"] = limits_list[surf_idx][0]
                    cb_dict["vmax"] = limits_list[surf_idx][1]

                    if colorbar_titles:
                        cb_dict["title"] = colorbar_titles[0]
                    else:
                        cb_dict["title"] = maps_names[0]

                    cb_dict["colormap"] = colormaps[0]
                    cb_dict["map_name"] = maps_names[0]

                    pos = positions[surf_idx]
                    colormap_limits[(0, surf_idx, 0)] = limits_list[surf_idx]
                    if colorbar_position == "right":
                        shape = [optimal_grid[0], optimal_grid[1] * 2]
                        row_weights = [1] * optimal_grid[0]
                        col_weights = [1, colorbar_size] * optimal_grid[1]
                        groups = []

                        brain_positions[(0, surf_idx, 0)] = (pos[0], pos[1] * 2)

                        cb_dict["position"] = (pos[0], pos[1] * 2 + 1)
                        cb_dict["orientation"] = "vertical"

                    else:
                        shape = [optimal_grid[0] * 2, optimal_grid[1]]
                        row_weights = [1, colorbar_size] * optimal_grid[0]
                        col_weights = [1] * optimal_grid[1]
                        groups = []

                        brain_positions[(0, surf_idx, 0)] = (pos[0] * 2, pos[1])

                        cb_dict["position"] = (pos[0] * 2 + 1, pos[1])
                        cb_dict["orientation"] = "horizontal"

                    colorbar_list.append(cb_dict)
            else:  # shared colorbar
                cb_dict = {}
                if colorbar_titles:
                    cb_dict["title"] = colorbar_titles[0]
                else:
                    cb_dict["title"] = maps_names[0]

                cb_dict["colormap"] = colormaps[0]
                cb_dict["map_name"] = maps_names[0]
                cb_dict["vmin"] = limits_list[0][0]
                cb_dict["vmax"] = limits_list[0][1]

                if colorbar_position == "right":
                    shape = [optimal_grid[0], optimal_grid[1] + 1]
                    row_weights = [1] * optimal_grid[0]
                    col_weights = [1] * optimal_grid[1] + [colorbar_size]
                    groups = [(slice(0, optimal_grid[0]), optimal_grid[1])]
                    cb_dict["position"] = (0, optimal_grid[1])
                    cb_dict["orientation"] = "vertical"

                else:  # bottom
                    shape = [optimal_grid[0] + 1, optimal_grid[1]]
                    row_weights = [1] * optimal_grid[0] + [colorbar_size]
                    col_weights = [1] * optimal_grid[1]
                    groups = [(optimal_grid[0], slice(0, optimal_grid[1]))]
                    cb_dict["position"] = (optimal_grid[0], 0)
                    cb_dict["orientation"] = "horizontal"

                colorbar_list.append(cb_dict)

        layout_config = {
            "shape": shape,
            "row_weights": row_weights,
            "col_weights": col_weights,
            "groups": groups,
            "brain_positions": brain_positions,
            "colormap_limits": colormap_limits,
        }
        return layout_config, colorbar_list

    def _hemispheres_multi_map_layout(
        self,
        surf_lh,
        surf_rh,
        surf_merg,
        valid_views,
        maps_names,
        v_limits,
        colormaps,
        colorbar_titles,
        orientation,
        colorbar,
        colormap_style,
        colorbar_position,
    ):
        """Handle multiple views and multiple maps case."""

        colorbar_size = self.figure_conf["colorbar_size"]

        n_views = len(valid_views)
        n_maps = len(maps_names)

        brain_positions = {}
        colormap_limits = {}
        colorbar_list = []

        if len(maps_names) == 1:
            map_name = maps_names[0]
            vmin, vmax = v_limits[0]
            colormap = colormaps[0]

            colorbar_data = any(
                map_name not in surface.colortables
                for map_name in maps_names
                for surface in [surf_lh, surf_rh, surf_merg]
            )
            if colorbar_data == False:
                colorbar = False

            map_limits = _get_map_limits(
                surfaces=[surf_lh, surf_rh, surf_merg],
                map_name=map_name,
                colormap_style="individual",
                v_limits=(vmin, vmax),
            )

            optimal_grid, positions = cltplot.calculate_optimal_subplots_grid(n_views)
            brain_positions = {}
            colormap_limits = {}

            for view_idx in range(n_views):
                pos = positions[view_idx]
                brain_positions[(0, 0, view_idx)] = pos
                if valid_views[view_idx].startswith("lh"):
                    colormap_limits[(0, 0, view_idx)] = map_limits[0]

                elif valid_views[view_idx].startswith("rh"):
                    colormap_limits[(0, 0, view_idx)] = map_limits[1]

                elif valid_views[view_idx].startswith("merg"):
                    colormap_limits[(0, 0, view_idx)] = map_limits[2]

            colorbar_list = []
            if not colorbar:
                shape = list(optimal_grid)
                row_weights = [1] * optimal_grid[0]
                col_weights = [1] * optimal_grid[1]
                groups = []

            else:
                if colormap_style == "individual":
                    if colorbar_position == "right":
                        shape = [optimal_grid[0], optimal_grid[1] * 2]
                        row_weights = [1] * optimal_grid[0]
                        col_weights = [1, colorbar_size] * optimal_grid[1]

                    elif colorbar_position == "bottom":
                        shape = [optimal_grid[0] * 2, optimal_grid[1]]
                        row_weights = [1, colorbar_size] * optimal_grid[0]
                        col_weights = [1] * optimal_grid[1]
                    groups = []

                    for view_idx in range(n_views):
                        pos = positions[view_idx]

                        cb_dict = {}
                        if colorbar_position == "right":
                            brain_positions[(0, 0, view_idx)] = (pos[0], pos[1] * 2)
                            cb_dict["position"] = (pos[0], pos[1] * 2 + 1)
                            cb_dict["orientation"] = "vertical"

                        elif colorbar_position == "bottom":
                            brain_positions[(0, 0, view_idx)] = (pos[0] * 2, pos[1])
                            cb_dict["position"] = (pos[0] * 2 + 1, pos[1])
                            cb_dict["orientation"] = "horizontal"

                        cb_dict["colormap"] = colormap
                        cb_dict["map_name"] = map_name

                        if valid_views[view_idx].startswith("lh"):
                            cb_dict["vmin"] = map_limits[0][0]
                            cb_dict["vmax"] = map_limits[0][1]

                        elif valid_views[view_idx].startswith("rh"):
                            cb_dict["vmin"] = map_limits[1][0]
                            cb_dict["vmax"] = map_limits[1][1]

                        elif valid_views[view_idx].startswith("merg"):
                            cb_dict["vmin"] = map_limits[2][0]
                            cb_dict["vmax"] = map_limits[2][1]

                        if colorbar_titles:
                            cb_dict["title"] = colorbar_titles[0]

                        else:
                            cb_dict["title"] = map_name

                        colorbar_list.append(cb_dict)

                else:  # shared colorbar
                    cb_dict = {}
                    cb_dict["colormap"] = colormap
                    cb_dict["map_name"] = map_name

                    # Compute the global limits
                    global_limits = (
                        min(l[0] for l in map_limits),
                        max(l[1] for l in map_limits),
                    )

                    cb_dict["vmin"] = global_limits[0]
                    cb_dict["vmax"] = global_limits[1]

                    if colorbar_titles:
                        cb_dict["title"] = colorbar_titles[0]

                    else:
                        cb_dict["title"] = map_name

                    if colorbar_position == "right":
                        shape = [optimal_grid[0], optimal_grid[1] + 1]
                        row_weights = [1] * optimal_grid[0]
                        col_weights = [1] * optimal_grid[1] + [colorbar_size]
                        groups = [(slice(0, optimal_grid[0]), optimal_grid[1])]
                        cb_dict["position"] = (0, optimal_grid[1])
                        cb_dict["orientation"] = "vertical"

                    else:  # bottom
                        shape = [optimal_grid[0] + 1, optimal_grid[1]]
                        row_weights = [1] * optimal_grid[0] + [colorbar_size]
                        col_weights = [1] * optimal_grid[1]
                        groups = [(optimal_grid[0], slice(0, optimal_grid[1]))]
                        cb_dict["position"] = (optimal_grid[0], 0)
                        cb_dict["orientation"] = "horizontal"

                    for view_idx in range(n_views):
                        pos = positions[view_idx]
                        brain_positions[(0, 0, view_idx)] = pos
                        if valid_views[view_idx].startswith("lh"):
                            colormap_limits[(0, 0, view_idx)] = map_limits[0]

                        elif valid_views[view_idx].startswith("rh"):
                            colormap_limits[(0, 0, view_idx)] = map_limits[1]

                        elif valid_views[view_idx].startswith("merg"):
                            colormap_limits[(0, 0, view_idx)] = map_limits[2]

                    # Append colorbar dictionary to the list
                    colorbar_list.append(cb_dict)

        else:
            if orientation == "horizontal":
                if not colorbar:
                    shape = [n_maps, n_views]
                    row_weights = [1] * n_maps
                    col_weights = [1] * n_views
                    groups = []

                    # Views in columns, maps in rows
                    for view_idx in range(n_views):
                        if valid_views[view_idx].startswith("lh"):
                            tmp_surface = copy.deepcopy(surf_lh)

                        elif valid_views[view_idx].startswith("rh"):
                            tmp_surface = copy.deepcopy(surf_rh)

                        elif valid_views[view_idx].startswith("merg"):
                            tmp_surface = copy.deepcopy(surf_merg)

                        for map_idx in range(n_maps):
                            brain_positions[(map_idx, 0, view_idx)] = (
                                map_idx,
                                view_idx,
                            )
                            map_limits = _get_map_limits(
                                surfaces=tmp_surface,
                                map_name=maps_names[map_idx],
                                colormap_style="individual",
                                v_limits=v_limits[map_idx],
                            )[0]
                            colormap_limits[(map_idx, 0, view_idx)] = map_limits
                else:
                    if colormap_style == "individual":
                        if colorbar_position == "right":
                            shape = [n_maps, n_views * 2]
                            row_weights = [1] * n_maps
                            col_weights = [1, colorbar_size] * n_views
                        elif colorbar_position == "bottom":
                            shape = [n_maps * 2, n_views]
                            row_weights = [1, colorbar_size] * n_maps
                            col_weights = [1] * n_views
                        groups = []

                        for map_idx in range(n_maps):
                            for view_idx in range(n_views):
                                cb_dict = {}
                                if colorbar_position == "right":
                                    cb_dict["position"] = (map_idx, view_idx * 2 + 1)
                                    cb_dict["orientation"] = "vertical"
                                    brain_positions[(map_idx, 0, view_idx)] = (
                                        map_idx,
                                        view_idx * 2,
                                    )

                                elif colorbar_position == "bottom":
                                    cb_dict["position"] = (map_idx * 2 + 1, view_idx)
                                    cb_dict["orientation"] = "horizontal"
                                    brain_positions[(map_idx, 0, view_idx)] = (
                                        map_idx * 2,
                                        view_idx,
                                    )

                                cb_dict["colormap"] = (
                                    colormaps[map_idx]
                                    if map_idx < len(colormaps)
                                    else colormaps[-1]
                                )
                                cb_dict["map_name"] = maps_names[map_idx]
                                if colorbar_titles:
                                    cb_dict["title"] = (
                                        colorbar_titles[map_idx]
                                        if map_idx < len(colorbar_titles)
                                        else colorbar_titles[-1]
                                    )
                                else:
                                    cb_dict["title"] = maps_names[map_idx]

                                if valid_views[view_idx].startswith("lh"):
                                    tmp_surface = copy.deepcopy(surf_lh)

                                elif valid_views[view_idx].startswith("rh"):
                                    tmp_surface = copy.deepcopy(surf_rh)

                                elif valid_views[view_idx].startswith("merg"):
                                    tmp_surface = copy.deepcopy(surf_merg)

                                limits_list = _get_map_limits(
                                    surfaces=tmp_surface,
                                    map_name=maps_names[map_idx],
                                    colormap_style="individual",
                                    v_limits=v_limits[map_idx],
                                )
                                cb_dict["vmin"] = limits_list[0][0]
                                cb_dict["vmax"] = limits_list[0][1]

                                colormap_limits[(map_idx, 0, view_idx)] = limits_list[0]

                                if maps_names[map_idx] not in tmp_surface.colortables:
                                    colorbar_list.append(cb_dict)

                    else:
                        # Get the global limits
                        maps_limits = []
                        for map_idx in range(n_maps):
                            if maps_names[map_idx] not in surf_merg.colortables:
                                map_limits = _get_map_limits(
                                    surfaces=surf_merg,
                                    map_name=maps_names[map_idx],
                                    colormap_style="shared",
                                    v_limits=v_limits[map_idx],
                                )[0]
                                maps_limits.append(map_limits)

                        ######### Global colorbar #########
                        # Compute the global limits
                        global_limits = (
                            min(l[0] for l in maps_limits),
                            max(l[1] for l in maps_limits),
                        )
                        cb_dict = {}
                        cb_dict["colormap"] = (
                            colormaps[0] if 0 < len(colormaps) else "viridis"
                        )
                        cb_dict["map_name"] = " + ".join(maps_names)
                        cb_dict["vmin"] = global_limits[0]
                        cb_dict["vmax"] = global_limits[1]
                        if colorbar_titles:
                            cb_dict["title"] = colorbar_titles[0]
                        else:
                            cb_dict["title"] = " + ".join(maps_names)
                        for map_idx in range(n_maps):
                            for view_idx in range(n_views):
                                brain_positions[(map_idx, 0, view_idx)] = (
                                    map_idx,
                                    view_idx,
                                )
                                colormap_limits[(map_idx, 0, view_idx)] = (
                                    global_limits + (maps_names[0],)
                                )

                        if colorbar_position == "right":
                            shape = [n_maps, n_views + 1]
                            row_weights = [1] * n_maps
                            col_weights = [1] * n_views + [colorbar_size]
                            groups = [(slice(0, n_maps), n_views)]

                            cb_dict["position"] = (0, n_views)
                            cb_dict["orientation"] = "vertical"

                        elif colorbar_position == "bottom":
                            shape = [n_maps + 1, n_views]
                            row_weights = [1] * n_maps + [colorbar_size]
                            col_weights = [1] * n_views
                            groups = [(n_maps, slice(0, n_views))]

                            cb_dict["position"] = (n_maps, 0)
                            cb_dict["orientation"] = "horizontal"
                        colorbar_list.append(cb_dict)

            elif orientation == "vertical":  # vertical
                if not colorbar:
                    # Views in rows, maps in columns
                    shape = [n_views, n_maps]
                    row_weights = [1] * n_views
                    col_weights = [1] * n_maps
                    groups = []

                    for view_idx in range(n_views):
                        for map_idx in range(n_maps):
                            brain_positions[(map_idx, 0, view_idx)] = (
                                view_idx,
                                map_idx,
                            )

                            if valid_views[view_idx].startswith("lh"):
                                tmp_surface = copy.deepcopy(surf_lh)

                            elif valid_views[view_idx].startswith("rh"):
                                tmp_surface = copy.deepcopy(surf_rh)

                            elif valid_views[view_idx].startswith("merg"):
                                tmp_surface = copy.deepcopy(surf_merg)

                            for map_idx in range(n_maps):
                                brain_positions[(map_idx, 0, view_idx)] = (
                                    map_idx,
                                    view_idx,
                                )
                                map_limits = _get_map_limits(
                                    surfaces=tmp_surface,
                                    map_name=maps_names[map_idx],
                                    colormap_style="individual",
                                    v_limits=v_limits[map_idx],
                                )[0]
                            colormap_limits[(map_idx, 0, view_idx)] = map_limits
                else:
                    if colormap_style == "individual":
                        if colorbar_position == "right":
                            shape = [n_views, n_maps * 2]
                            row_weights = [1] * n_views
                            col_weights = [1, colorbar_size] * n_maps
                        elif colorbar_position == "bottom":
                            shape = [n_views * 2, n_maps]
                            row_weights = [1, colorbar_size] * n_views
                            col_weights = [1] * n_maps
                        groups = []
                        for view_idx in range(n_views):
                            for map_idx in range(n_maps):
                                cb_dict = {}
                                if colorbar_position == "right":
                                    cb_dict["position"] = (view_idx, map_idx * 2 + 1)
                                    cb_dict["orientation"] = "vertical"
                                    brain_positions[(map_idx, 0, view_idx)] = (
                                        view_idx,
                                        map_idx * 2,
                                    )

                                elif colorbar_position == "bottom":
                                    cb_dict["position"] = (view_idx * 2 + 1, map_idx)
                                    cb_dict["orientation"] = "horizontal"
                                    brain_positions[(map_idx, 0, view_idx)] = (
                                        view_idx * 2,
                                        map_idx,
                                    )

                                cb_dict["colormap"] = (
                                    colormaps[map_idx]
                                    if map_idx < len(colormaps)
                                    else colormaps[-1]
                                )
                                cb_dict["map_name"] = maps_names[map_idx]
                                if colorbar_titles:
                                    cb_dict["title"] = (
                                        colorbar_titles[map_idx]
                                        if map_idx < len(colorbar_titles)
                                        else colorbar_titles[-1]
                                    )
                                else:
                                    cb_dict["title"] = maps_names[map_idx]

                                if valid_views[view_idx].startswith("lh"):
                                    tmp_surface = copy.deepcopy(surf_lh)

                                elif valid_views[view_idx].startswith("rh"):
                                    tmp_surface = copy.deepcopy(surf_rh)

                                elif valid_views[view_idx].startswith("merg"):
                                    tmp_surface = copy.deepcopy(surf_merg)

                                limits_list = _get_map_limits(
                                    surfaces=tmp_surface,
                                    map_name=maps_names[map_idx],
                                    colormap_style="individual",
                                    v_limits=v_limits[map_idx],
                                )
                                cb_dict["vmin"] = limits_list[0][0]
                                cb_dict["vmax"] = limits_list[0][1]
                                colormap_limits[(map_idx, 0, view_idx)] = limits_list[0]
                                if maps_names[map_idx] not in tmp_surface.colortables:
                                    colorbar_list.append(cb_dict)
                    else:
                        # Get the global limits
                        maps_limits = []
                        for map_idx in range(n_maps):
                            if maps_names[map_idx] not in surf_merg.colortables:
                                map_limits = _get_map_limits(
                                    surfaces=surf_merg,
                                    map_name=maps_names[map_idx],
                                    colormap_style="shared",
                                    v_limits=v_limits[map_idx],
                                )[0]
                                maps_limits.append(map_limits)

                        ######### Global colorbar #########
                        # Compute the global limits
                        global_limits = (
                            min(l[0] for l in maps_limits),
                            max(l[1] for l in maps_limits),
                        )
                        cb_dict = {}
                        cb_dict["colormap"] = (
                            colormaps[0] if 0 < len(colormaps) else "viridis"
                        )
                        cb_dict["map_name"] = " + ".join(maps_names)
                        cb_dict["vmin"] = global_limits[0]
                        cb_dict["vmax"] = global_limits[1]
                        if colorbar_titles:
                            cb_dict["title"] = colorbar_titles[0]
                        else:
                            cb_dict["title"] = " + ".join(maps_names)
                        for view_idx in range(n_views):
                            for map_idx in range(n_maps):
                                brain_positions[(map_idx, 0, view_idx)] = (
                                    view_idx,
                                    map_idx,
                                )
                                colormap_limits[(map_idx, 0, view_idx)] = (
                                    global_limits + (maps_names[0],)
                                )

                        if colorbar_position == "right":
                            shape = [n_views, n_maps + 1]
                            row_weights = [1] * n_views
                            col_weights = [1] * n_maps + [colorbar_size]
                            groups = [(slice(0, n_views), n_maps)]

                            cb_dict["position"] = (0, n_maps)
                            cb_dict["orientation"] = "vertical"

                        elif colorbar_position == "bottom":
                            shape = [n_views + 1, n_maps]
                            row_weights = [1] * n_views + [colorbar_size]
                            col_weights = [1] * n_maps
                            groups = [(n_views, slice(0, n_maps))]

                            cb_dict["position"] = (n_views, 0)
                            cb_dict["orientation"] = "horizontal"
                        colorbar_list.append(cb_dict)

        layout_config = {
            "shape": shape,
            "row_weights": row_weights,
            "col_weights": col_weights,
            "groups": groups,
            "brain_positions": brain_positions,
            "colormap_limits": colormap_limits,
        }
        return layout_config, colorbar_list

    def _create_colorbar_configs(
        self,
        maps_names,
        colormaps,
        v_limits,
        colorbar_titles,
        surfaces,
        config,
        colormap_style,
        colorbar_position,
    ):
        """Create colorbar configurations based on layout."""
        colorbar_configs = []
        brain_positions = config["brain_positions"]
        shape = config["shape"]

        # Determine the number of dimensions
        n_maps = len(maps_names)
        n_surfaces = len(surfaces) if surfaces else 1

        # Get unique combinations that need colorbars
        unique_maps = set()
        unique_map_surface_pairs = set()

        for map_idx, surf_idx, view_idx in brain_positions.keys():
            unique_maps.add(map_idx)
            unique_map_surface_pairs.add((map_idx, surf_idx))

        if colormap_style == "individual":
            colorbar_configs = self._create_individual_colorbar_configs(
                maps_names,
                colormaps,
                v_limits,
                colorbar_titles,
                surfaces,
                brain_positions,
                colorbar_position,
                shape,
                unique_map_surface_pairs,
            )
        else:  # shared
            colorbar_configs = self._create_shared_colorbar_configs(
                maps_names,
                colormaps,
                v_limits,
                colorbar_titles,
                surfaces,
                brain_positions,
                colorbar_position,
                shape,
                unique_maps,
            )

        return colorbar_configs

    def _finalize_plot(
        self,
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
                self._create_threaded_plot(plotter)
            else:
                # Blocking mode - show normally
                plotter.show()

    def plot_hemispheres(
        self,
        surf_rh: cltsurf.Surface,
        surf_lh: cltsurf.Surface,
        maps_names: Union[str, List[str]] = ["surface"],
        views: Union[str, List[str]] = "dorsal",
        views_orientation: str = "horizontal",
        v_limits: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = (
            None,
            None,
        ),
        use_opacity: bool = True,
        colormaps: Union[str, List[str]] = "BrBG",
        colorbar: bool = True,
        colorbar_titles: Union[str, List[str]] = None,
        colormap_style: str = "individual",
        colorbar_position: str = "right",
        notebook: bool = False,
        non_blocking: bool = False,
        save_path: Optional[str] = None,
    ):
        """
        Plot brain hemispheres with multiple views and multiple maps.

        Parameters
        ----------
        surf_rh : cltsurf.Surface
            Right hemisphere surface with associated data.

        surf_lh : cltsurf.Surface
            Left hemisphere surface with associated data.

        maps_names : str or list of str, default ["surface"]
            Name(s) of the data maps to visualize. Must be present in both surfaces.

        views : str or list of str, default "dorsal"
            View(s) to display. Options include 'dorsal', 'ventral', 'lateral', 'medial', 'anterior', 'posterior'.
            Can be a single view or a list of views. It can also include different multiple views specified as layouts:
            >>> plotter = SurfacePlotter("configs.json")
            >>> layouts = plotter.list_available_layouts()

        views_orientation : str, default "horizontal"
            Orientation of views when multiple views are provided. Options are 'horizontal' or 'vertical'.

        v_limits : tuple or list of tuples, optional
            Value limits for color mapping. If a single tuple is provided, it applies to all maps
            (e.g., (vmin, vmax)). If a list is provided, it should match the number of maps.
            If None, limits are determined from the data.

        colormaps : str or list of str, default "BrBG"
            Colormap(s) to use for visualization. If a single colormap is provided, it applies to all maps.
            If a list is provided, it should match the number of maps.

        colorbar : bool, default True
            Whether to display colorbars for the maps.

        colorbar_titles : str or list of str, optional
            Title(s) for the colorbars. If a single title is provided, it applies to all maps.
            If a list is provided, it should match the number of maps. If None, map names are used.

        colormap_style : str, default "individual"
            Style of colormap application. Options are 'individual' (each map has its own colormap)
            or 'shared' (all maps share the same colormap).

        colorbar_position : str, default "right"
            Position of the colorbars. Options are 'right' or 'bottom'.

        notebook : bool, default False
            Whether to render the plot in a Jupyter notebook environment.
            If True, uses notebook-compatible rendering.

        non_blocking : bool, default False
            If True, displays the plot in a non-blocking manner using threading.
            Only applicable when `notebook` is False and `save_path` is None.

        save_path : str, optional
            File path to save the rendered figure. If provided, the figure is saved to this path
            instead of being displayed.

        Returns
        -------
        None
            The function does not return any value. It either displays the plot or saves it to a
            file, depending on the parameters provided.

        Raises
        ------
        ValueError
            If no valid maps are found in the provided surfaces, or if multiple maps are provided
            but the function is not set up to handle them.
        ValueError
            If the provided views are not valid or if the orientation is incorrect.
        ValueError
            If the colormap style or colorbar position is invalid.

        Examples
        --------
        >>> plotter = SurfacePlotter("configs.json")
        >>> plotter.plot_hemispheres(surf_rh, surf_lh, maps_names="thickness", views=["dorsal", "lateral"], colormaps="viridis", colorbar_titles="Cortical Thickness", save_path="hemispheres.png")

        """

        # Creating the merge surface
        surf_merg = cltsurf.merge_surfaces_list([surf_lh, surf_rh])

        # Filter to only available maps
        if isinstance(maps_names, str):
            maps_names = [maps_names]
        n_maps = len(maps_names)

        if n_maps == 0:
            raise ValueError("No maps names provided.")

        if n_maps > 1:
            raise ValueError("Multiple maps are not supported in this function.")
        # Check if the maps are available in all surfaces

        fin_map_names = []
        cont_map = 0
        # Check if the map_name is available in any of the surfaces
        for surf in [surf_lh, surf_rh, surf_merg]:
            available_maps = list(surf.mesh.point_data.keys())
            if maps_names[0] in available_maps:
                cont_map = cont_map + 1

        # If the map is present in all surfaces, add it to the final list
        if cont_map == 3:
            fin_map_names.append(maps_names[0])

        # Available overlays
        maps_names = fin_map_names
        n_maps = len(maps_names)

        if n_maps == 0:
            raise ValueError(
                "No valid maps found in the provided surfaces. The maps_names must be present in all surfaces."
            )

        # Process and validate v_limits parameter
        if isinstance(v_limits, Tuple):
            if len(v_limits) != 2:
                v_limits = (None, None)
            v_limits = [v_limits] * n_maps

        elif isinstance(v_limits, List[Tuple[float, float]]):
            if len(v_limits) != n_maps:
                v_limits = [(None, None)] * n_maps

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
            colorbar_titles = maps_names

        # Get view configuration
        view_ids = self._get_views_to_plot(views, ["lh", "rh"])

        # Determine rendering mode based on save_path, environment, and threading preference
        save_mode, use_off_screen, use_notebook, use_threading = (
            self._determine_render_mode(save_path, notebook, non_blocking)
        )

        # Detecting the screen size for the plotter
        screen_size = cltplot.get_current_monitor_size()

        config_dict, colorbar_dict_list = self._hemispheres_multi_map_layout(
            surf_lh,
            surf_rh,
            surf_merg,
            view_ids,
            maps_names,
            v_limits,
            colormaps,
            colorbar_titles=colorbar_titles,
            orientation=views_orientation,
            colorbar=colorbar,
            colormap_style=colormap_style,
            colorbar_position=colorbar_position,
        )

        # Determine rendering mode based on save_path, environment, and threading preference
        save_mode, use_off_screen, use_notebook, use_threading = (
            self._determine_render_mode(save_path, notebook, non_blocking)
        )

        # Detecting the screen size for the plotter
        screen_size = cltplot.get_current_monitor_size()

        # Create PyVista plotter with appropriate rendering mode
        plotter_kwargs = {
            "notebook": use_notebook,
            "window_size": [screen_size[0], screen_size[1]],
            "off_screen": use_off_screen,
            "shape": config_dict["shape"],
            "row_weights": config_dict["row_weights"],
            "col_weights": config_dict["col_weights"],
            "border": True,
        }

        groups = config_dict["groups"]
        if groups:
            plotter_kwargs["groups"] = groups

        pv_plotter = pv.Plotter(**plotter_kwargs)

        brain_positions = config_dict["brain_positions"]
        map_limits = config_dict["colormap_limits"]
        for (map_idx, surf_idx, view_idx), (row, col) in brain_positions.items():
            pv_plotter.subplot(row, col)
            # Set background color from figure configuration
            pv_plotter.set_background(self.figure_conf["background_color"])

            tmp_view_name = view_ids[view_idx]

            # Split the view name if it contains '_'
            if "-" in tmp_view_name:
                tmp_view_name = tmp_view_name.split("-")[1]

                # Capitalize the first letter
                tmp_view_name = tmp_view_name.capitalize()

                # Detecting if the view is left or right
                if "lh" in view_ids[view_idx]:
                    subplot_title = "Left hemisphere: " + tmp_view_name + " view"
                elif "rh" in view_ids[view_idx]:
                    subplot_title = "Right hemisphere: " + tmp_view_name + " view"
                elif "merg" in view_ids[view_idx]:
                    subplot_title = tmp_view_name + " view"

            pv_plotter.add_text(
                subplot_title,
                font_size=self.figure_conf["title_font_size"],
                position="upper_edge",
                color=self.figure_conf["title_font_color"],
                shadow=self.figure_conf["title_shadow"],
                font=self.figure_conf["title_font_type"],
            )

            # Geting the vmin and vmax for the current map
            vmin, vmax, map_name = map_limits[map_idx, surf_idx, view_idx]

            # Select the colormap for the current map
            idx = [i for i, name in enumerate(maps_names) if name == map_name]
            colormap = colormaps[idx[0]] if idx else colormaps[0]

            # Add the brain surface mesh
            if "lh" in view_ids[view_idx]:
                surf = copy.deepcopy(surf_lh)

            elif "rh" in view_ids[view_idx]:
                surf = copy.deepcopy(surf_rh)

            elif "merg" in view_ids[view_idx]:
                surf = copy.deepcopy(surf_merg)

            surf = self._prepare_surface(
                surf, maps_names[map_idx], colormap, vmin=vmin, vmax=vmax
            )

            if not use_opacity:
                # delete the alpha channel if exists
                if "rgba" in surf.mesh.point_data:
                    surf.mesh.point_data["rgba"] = surf.mesh.point_data["rgba"][:, :3]

            pv_plotter.add_mesh(
                copy.deepcopy(surf.mesh),
                scalars="rgba",
                rgb=True,
                ambient=self.figure_conf["mesh_ambient"],
                diffuse=self.figure_conf["mesh_diffuse"],
                specular=self.figure_conf["mesh_specular"],
                specular_power=self.figure_conf["mesh_specular_power"],
                smooth_shading=self.figure_conf["mesh_smooth_shading"],
                show_scalar_bar=False,
            )

            # Set the camera view
            tmp_view = view_ids[view_idx]
            if tmp_view.startswith("merg"):
                tmp_view = tmp_view.replace("merg", "lh")

            camera_params = self.views_conf[tmp_view]
            pv_plotter.camera_position = camera_params["view"]
            pv_plotter.camera.azimuth = camera_params["azimuth"]
            pv_plotter.camera.elevation = camera_params["elevation"]
            pv_plotter.camera.zoom(camera_params["zoom"])

        # And place colorbars at their positions
        if len(colorbar_dict_list):

            for colorbar_dict in colorbar_dict_list:
                if colorbar_dict is not False:
                    row, col = colorbar_dict["position"]
                    orientation = colorbar_dict["orientation"]
                    colorbar_id = colorbar_dict["map_name"]
                    colormap = colorbar_dict["colormap"]
                    colorbar_title = colorbar_dict["title"]
                    vmin = colorbar_dict["vmin"]
                    vmax = colorbar_dict["vmax"]
                    pv_plotter.subplot(row, col)

                    self._add_colorbar(
                        plotter=pv_plotter,
                        colorbar_subplot=(row, col),
                        vmin=vmin,
                        vmax=vmax,
                        map_name=colorbar_id,
                        colormap=colormap,
                        colorbar_title=colorbar_title,
                        colorbar_position=orientation,
                    )

        # successful_links = self._link_brain_subplot_cameras(pv_plotter, brain_positions)

        # Handle final rendering - either save, display blocking, or display non-blocking
        self._finalize_plot(pv_plotter, save_mode, save_path, use_threading)

    def _link_brain_subplot_cameras(self, pv_plotter, brain_positions):
        """
        Link cameras for brain subplots that share the same view index.

        Args:
            pv_plotter: PyVista plotter object
            brain_positions: Dict with keys (m_idx, s_idx, v_idx) and values (row, col)
        """
        # Group positions by view index using defaultdict for cleaner code
        from collections import defaultdict

        grouped_by_v_idx = defaultdict(list)
        for (m_idx, s_idx, v_idx), (row, col) in brain_positions.items():
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

    def plot_surfaces(
        self,
        surfaces: Union[
            cltsurf.Surface, List[cltsurf.Surface], List[List[cltsurf.Surface]]
        ],
        hemi_id: List[str] = ["lh"],
        views: Union[str, List[str]] = "dorsal",
        views_orientation: str = "horizontal",
        notebook: bool = False,
        map_names: Union[str, List[str]] = ["surface"],
        v_limits: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = (
            None,
            None,
        ),
        use_opacity: bool = True,
        colormaps: Union[str, List[str]] = "BrBG",
        save_path: Optional[str] = None,
        non_blocking: bool = True,
        colorbar: bool = True,
        colormap_style: str = "individual",
        colorbar_titles: Union[str, List[str]] = None,
        colorbar_position: str = "right",
    ) -> None:
        """
        Plot brain surfaces with optional threading and screenshot support.

        Parameters
        ----------
        surfaces : Union[cltsurf.Surface, List[cltsurf.Surface], List[List[cltsurf.Surface]]]
            Brain surface(s) to plot.

        hemi_id : List[str], default ["lh"]
            Hemisphere identifiers.

        views : Union[str, List[str]], default "dorsal"
            View angles for the surfaces.

        views_orientation : str, default "horizontal"
            Orientation of the views layout.

        notebook : bool, default False
            Whether running in Jupyter notebook environment.

        map_names : Union[str, List[str]], default ["surface"]
            Names of the surface maps to plot.

        v_limits : Optional[Union[Tuple[float, float], List[Tuple[float, float]]]], default (None, None)
            Value limits for colormapping.

        colormaps : Union[str, List[str]], default "BrBG"
            Colormaps to use for each map.

        use_opacity : bool, default True
            Whether to use opacity in the surface rendering. This is important when saving to HTML format to
            ensure proper visualization. If False, surfaces will be fully opaque.

        save_path : Optional[str], default None
            File path for saving the figure. If None, plot is displayed.


        non_blocking : bool, default False
            If True, display the plot in a separate thread, allowing the terminal
            or notebook to remain interactive. Only applies when save_path is None.

        colorbar : bool, default True
            Whether to show colorbars.

        colormap_style : str, default "individual"
            Style of colormap application.

        colorbar_titles : Union[str, List[str]], optional
            Titles for the colorbars.

        colorbar_position : str, default "right"
            Position of the colorbars.

        """

        # Preparing the surfaces to be plotted
        if isinstance(surfaces, cltsurf.Surface):
            surf2plot = [copy.deepcopy(surfaces)]

        elif isinstance(surfaces, list):
            # If all the elements are of type cltsurf.Surface
            surf2plot = []
            for surf in surfaces:
                if isinstance(surf, cltsurf.Surface):
                    surf2plot.append(copy.deepcopy(surf))

                elif isinstance(surf, list) and all(
                    isinstance(s, cltsurf.Surface) for s in surf
                ):
                    surf2plot.append(cltsurf.merge_surfaces_list(surf))

                else:
                    raise TypeError(
                        "All elements must be of type cltsurf.Surface or a list of such."
                    )

        # Number of surfaces
        n_surfaces = len(surf2plot)

        # Filter to only available maps
        if isinstance(map_names, str):
            map_names = [map_names]
        n_maps = len(map_names)

        fin_map_names = []
        for i, map_name in enumerate(map_names):
            cont_map = 0
            # Check if the map_name is available in any of the surfaces
            for surf in surf2plot:
                available_maps = list(surf.mesh.point_data.keys())
                if map_name in available_maps:
                    cont_map = cont_map + 1

            #
            if cont_map == n_surfaces:
                fin_map_names.append(map_name)

        # Available overlays
        map_names = fin_map_names
        n_maps = len(map_names)

        if n_maps == 0:
            raise ValueError(
                "No valid maps found in the provided surfaces. The map_names must be present in all surfaces."
            )

        # Process and validate v_limits parameter
        if isinstance(v_limits, Tuple):
            if len(v_limits) != 2:
                v_limits = (None, None)
            v_limits = [v_limits] * n_maps

        elif isinstance(v_limits, List[Tuple[float, float]]):
            if len(v_limits) != n_maps:
                v_limits = [(None, None)] * n_maps

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

        # Check if the is colortable at any of the surfaces for any of the maps

        (
            view_ids,
            config_dict,
            colorbar_dict_list,
        ) = self._build_plotting_config(
            views=views,
            maps_names=map_names,
            surfaces=surf2plot,
            colormaps=colormaps,
            v_limits=v_limits,
            orientation=views_orientation,
            hemi_id=hemi_id,
            colorbar=colorbar,
            colorbar_titles=colorbar_titles,
            colormap_style=colormap_style,
            colorbar_position=colorbar_position,
        )

        # Determine rendering mode based on save_path, environment, and threading preference
        save_mode, use_off_screen, use_notebook, use_threading = (
            self._determine_render_mode(save_path, notebook, non_blocking)
        )

        # Detecting the screen size for the plotter
        screen_size = cltplot.get_current_monitor_size()

        # Create PyVista plotter with appropriate rendering mode
        plotter_kwargs = {
            "notebook": use_notebook,
            "window_size": [screen_size[0], screen_size[1]],
            "off_screen": use_off_screen,
            "shape": config_dict["shape"],
            "row_weights": config_dict["row_weights"],
            "col_weights": config_dict["col_weights"],
            "border": True,
        }

        groups = config_dict["groups"]
        if groups:
            plotter_kwargs["groups"] = groups

        pv_plotter = pv.Plotter(**plotter_kwargs)
        # Now you can place brain surfaces at specific positions
        pv_plotter.set_background(self.figure_conf["background_color"])

        brain_positions = config_dict["brain_positions"]

        # Computing the plot indexes
        subplot_indices = []
        n_subplots = len(pv_plotter.renderers)
        n_rows = config_dict["shape"][0]
        n_cols = config_dict["shape"][1]

        subplot_indices = []

        for (map_idx, surf_idx, view_idx), position in brain_positions.items():
            # Handle case where position might be a list/tuple of coordinates
            if isinstance(position, (list, tuple)) and len(position) >= 2:
                row, col = position[0], position[1]
            else:
                row, col = position

            # Ensure row and col are integers
            if isinstance(row, (list, tuple)):
                row = row[0] if row else 0

            if isinstance(col, (list, tuple)):
                col = col[0] if col else 0

            subplot_indices.append(int(row) * n_cols + int(col))

        # If there is any element of subplot_indices that is bigger than n_subplots do something else
        if any(sp_index > n_subplots for sp_index in subplot_indices):
            # Remove all the elements that are bigger than n_subplots

            # Take a vector from 0 to 6*4 and reshape it to a matrix of 6 rows and 4 columns and print it
            tmp = np.arange(0, n_rows * n_cols).reshape(n_rows, n_cols)
            # Now remove the last column and print the matrix
            tmp = tmp[:, :-1]

            # Now, if the matrix has n_rows bigger than 3, remove , from rows 3 to n_rows -1
            if tmp.shape[0] > 3:
                for cont, r in enumerate(range(1, tmp.shape[0])):
                    tmp[r, :] = tmp[r, :] - cont

            subplot_indices = tmp.T.flatten().tolist()

        map_limits = config_dict["colormap_limits"]
        for (map_idx, surf_idx, view_idx), (row, col) in brain_positions.items():
            pv_plotter.subplot(row, col)
            # Set background color from figure configuration
            pv_plotter.set_background(self.figure_conf["background_color"])
            tmp_view_name = view_ids[view_idx]

            # Split the view name if it contains '_'
            if "-" in tmp_view_name:
                tmp_view_name = tmp_view_name.split("-")[1]

                # Capitalize the first letter
                tmp_view_name = tmp_view_name.capitalize()

            pv_plotter.add_text(
                f"{map_names[map_idx]}, Surface: {surf_idx}, View: {tmp_view_name}",
                font_size=self.figure_conf["title_font_size"],
                position="upper_edge",
                color=self.figure_conf["title_font_color"],
                shadow=self.figure_conf["title_shadow"],
                font=self.figure_conf["title_font_type"],
            )

            # Geting the vmin and vmax for the current map
            vmin, vmax, map_name = map_limits[map_idx, surf_idx, view_idx]

            # Select the colormap for the current map
            idx = [i for i, name in enumerate(map_names) if name == map_name]
            colormap = colormaps[idx[0]] if idx else colormaps[0]

            # Add the brain surface mesh
            surf = surf2plot[surf_idx]
            surf = self._prepare_surface(
                surf, map_names[map_idx], colormap, vmin=vmin, vmax=vmax
            )
            if not use_opacity:
                # delete the alpha channel if exists
                if "rgba" in surf.mesh.point_data:
                    surf.mesh.point_data["rgba"] = surf.mesh.point_data["rgba"][:, :3]

            pv_plotter.add_mesh(
                copy.deepcopy(surf.mesh),
                scalars="rgba",
                rgb=True,
                ambient=self.figure_conf["mesh_ambient"],
                diffuse=self.figure_conf["mesh_diffuse"],
                specular=self.figure_conf["mesh_specular"],
                specular_power=self.figure_conf["mesh_specular_power"],
                smooth_shading=self.figure_conf["mesh_smooth_shading"],
                show_scalar_bar=False,
            )

            # Set the camera view
            tmp_view = view_ids[view_idx]

            # Replace merg from the view id if needed
            if "merg" in tmp_view:
                tmp_view = tmp_view.replace("merg", "lh")

            camera_params = self.views_conf[tmp_view]
            pv_plotter.camera_position = camera_params["view"]
            pv_plotter.camera.azimuth = camera_params["azimuth"]
            pv_plotter.camera.elevation = camera_params["elevation"]
            pv_plotter.camera.zoom(camera_params["zoom"])

        # And place colorbars at their positions
        if len(colorbar_dict_list):

            for colorbar_dict in colorbar_dict_list:
                if colorbar_dict is not False:
                    row, col = colorbar_dict["position"]
                    orientation = colorbar_dict["orientation"]
                    colorbar_id = colorbar_dict["map_name"]
                    colormap = colorbar_dict["colormap"]
                    colorbar_title = colorbar_dict["title"]
                    vmin = colorbar_dict["vmin"]
                    vmax = colorbar_dict["vmax"]
                    pv_plotter.subplot(row, col)

                    self._add_colorbar(
                        plotter=pv_plotter,
                        colorbar_subplot=(row, col),
                        vmin=vmin,
                        vmax=vmax,
                        map_name=colorbar_id,
                        colormap=colormap,
                        colorbar_title=colorbar_title,
                        colorbar_position=orientation,
                    )

        # Linking the cameras from the subplots with the same view
        unique_v_indices = set(key[2] for key in brain_positions.keys())
        grouped_by_v_idx = {}

        for v_idx in unique_v_indices:
            grouped_by_v_idx[v_idx] = []
            for i, ((m_idx, s_idx, v_idx), (row, col)) in enumerate(
                brain_positions.items()
            ):
                if v_idx in grouped_by_v_idx:  # Safety check
                    grouped_by_v_idx[v_idx].append(subplot_indices[i])

        # After all subplots are created and populated, link the views
        for v_idx, positions in grouped_by_v_idx.items():
            if len(positions) > 1:
                # Link all views in this group
                pv_plotter.link_views(grouped_by_v_idx[v_idx])

        # Handle final rendering - either save, display blocking, or display non-blocking
        self._finalize_plot(pv_plotter, save_mode, save_path, use_threading)

    #################################################################################################
    def _create_threaded_plot(self, plotter: pv.Plotter) -> None:
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

    ################################################################################################
    def _prepare_surface(
        self,
        surface: cltsurf.Surface,
        map_name: str,
        colormap: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> cltsurf.Surface:
        """ """

        if vmin is None:
            vmin = np.min(surface.mesh.point_data[map_name])

        if vmax is None:
            vmax = np.max(surface.mesh.point_data[map_name])

        try:
            vertex_values = surface.mesh.point_data[map_name]
            vertex_values = np.nan_to_num(
                vertex_values,
                nan=0.0,
            )  # Handle NaNs and infinities
            surface.mesh.point_data[map_name] = vertex_values

        except KeyError:
            raise ValueError(f"Data array '{map_name}' not found in surface point_data")

        # Apply colors to mesh data
        surface.mesh.point_data["rgba"] = surface.get_vertexwise_colors(
            map_name, colormap, vmin, vmax
        )

        return surface

    ################################################################################################
    def _process_v_limits(
        self,
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
                if not (
                    isinstance(vmin, (int, float)) and isinstance(vmax, (int, float))
                ):
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
    def _add_colorbar(
        self,
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
        plotter.set_background(self.figure_conf["background_color"])

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
        if not self.figure_conf["colorbar_outline"]:
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
        scalar_bar.SetNumberOfLabels(self.figure_conf["colorbar_n_labels"])

        # Get text properties for title and labels
        title_prop = scalar_bar.GetTitleTextProperty()
        label_prop = scalar_bar.GetLabelTextProperty()

        # Set colors
        title_color = pv.Color(self.figure_conf["colorbar_font_color"]).float_rgb
        title_prop.SetColor(*title_color)
        label_prop.SetColor(*title_color)

        # Set font properties - key fix for consistent sizing
        if self.figure_conf["colorbar_font_type"].lower() == "arial":
            title_prop.SetFontFamilyToArial()
            label_prop.SetFontFamilyToArial()

        elif self.figure_conf["colorbar_font_type"].lower() == "courier":
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
    def _determine_render_mode(
        self, save_path: Optional[str], notebook: bool, non_blocking: bool = False
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
    def list_available_view_names(self) -> List[str]:
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
        >>> view_names = plotter.list_available_view_names()
        >>> print(f"Available views: {view_names}")
        """

        view_names = list(self._view_name_mapping.keys())
        view_names_capitalized = [name.capitalize() for name in view_names]

        print("🧠 Available View Names for Dynamic Selection:")
        print("=" * 50)
        for i, (name, titles) in enumerate(self._view_name_mapping.items(), 1):
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
    def list_available_layouts(self) -> Dict[str, Dict[str, Any]]:
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
        >>> layouts = plotter.list_available_layouts()
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

        for views, config in self.views_conf.items():
            shape = config["shape"]
            window_size = config["window_size"]
            num_views = len(config["views"])

            print(f"\n📊 {views}")
            print(f"   Shape: {shape[0]}x{shape[1]} ({num_views} views)")
            print(f"   Window: {window_size[0]}x{window_size[1]}")

            # Create layout visualization grid
            layout_grid = {}
            for view in config["views"]:
                pos = tuple(view["subplot"])
                layout_grid[pos] = {
                    "title": view["title"],
                    "mesh": view["mesh"],
                    "view_type": view["view"],
                }

            # Display ASCII grid representation
            print("   Layout:")
            for row in range(shape[0]):
                row_str = "   "
                for col in range(shape[1]):
                    if (row, col) in layout_grid:
                        view_info = layout_grid[(row, col)]
                        title = view_info["title"][:12]  # Truncate long titles
                        row_str += f"[{title:>12}]"
                    else:
                        row_str += f"[{'empty':>12}]"
                print(row_str)

            # Store in return dictionary
            layout_info[views] = {
                "shape": shape,
                "window_size": window_size,
                "num_views": num_views,
                "views": layout_grid,
            }

        print("\n" + "=" * 50)
        print("\n🎯 Dynamic View Selection:")
        print("   You can also use a list of view names for custom layouts:")
        print(
            "   Available view names: Lateral, Medial, Dorsal, Ventral, Rostral, Caudal"
        )
        print("   Example: views=['Lateral', 'Medial', 'Dorsal']")
        print("=" * 50)

        return layout_info

    ###############################################################################################
    def get_layout_details(self, views: str) -> Optional[Dict[str, Any]]:
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
        >>> details = plotter.get_layout_details("8_views")
        >>> if details:
        ...     print(f"Grid shape: {details['shape']}")
        ...     print(f"Views: {len(details['views'])}")
        >>>
        >>> # Handle non-existent configuration
        >>> details = plotter.get_layout_details("invalid_config")
        """

        if views not in self.views_conf:
            print(f"❌ Configuration '{views}' not found!")
            print(f"Available configs: {list(self.views_conf.keys())}")
            return None

        config = self.views_conf[views]
        shape = config["shape"]

        print(f"🧠 Layout Details: {views}")
        print("=" * 40)
        print(f"Grid Shape: {shape[0]} rows × {shape[1]} columns")
        print(f"Window Size: {config['window_size'][0]} × {config['window_size'][1]}")
        print(f"Total Views: {len(config['views'])}")
        print("\nView Details:")

        for i, view in enumerate(config["views"], 1):
            pos = view["subplot"]
            print(f"  {i:2d}. Position ({pos[0]},{pos[1]}): {view['title']}")
            print(f"      Mesh: {view['mesh']}, View: {view['view']}")
            print(
                f"      Camera: az={view['azimuth']}°, el={view['elevation']}°, zoom={view['zoom']}"
            )

        return config

    ###############################################################################################
    def reload_config(self) -> None:
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

        print(f"Reloading configuration from: {self.config_file}")
        self._load_configs()
        print(
            f"Successfully loaded figure config and {len(self.views_conf)} view configurations"
        )

    ###############################################################################################
    def get_figure_config(self) -> Dict[str, Any]:
        """
        Get the current figure configuration settings.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all figure styling settings including
            background color, font settings, mesh properties, and colorbar options.

        Examples
        --------
        >>> plotter = SurfacePlotter("configs.json")
        >>> fig_config = plotter.get_figure_config()
        >>> print(f"Background color: {fig_config['background_color']}")
        >>> print(f"Title font: {fig_config['title_font_type']}")
        """

        print("🎨 Current Figure Configuration:")
        print("=" * 40)
        print("Background & Colors:")
        print(f"  Background Color: {self.figure_conf['background_color']}")
        print(f"  Title Color: {self.figure_conf['title_font_color']}")
        print(f"  Colorbar Color: {self.figure_conf['colorbar_font_color']}")

        print("\nTitle Settings:")
        print(f"  Font Type: {self.figure_conf['title_font_type']}")
        print(f"  Font Size: {self.figure_conf['title_font_size']}")
        print(f"  Shadow: {self.figure_conf['title_shadow']}")

        print("\nColorbar Settings:")
        print(f"  Font Type: {self.figure_conf['colorbar_font_type']}")
        print(f"  Font Size: {self.figure_conf['colorbar_font_size']}")
        print(f"  Title Font Size: {self.figure_conf['colorbar_title_font_size']}")
        print(f"  Outline: {self.figure_conf['colorbar_outline']}")
        print(f"  Number of Labels: {self.figure_conf['colorbar_n_labels']}")

        print("\nMesh Properties:")
        print(f"  Ambient: {self.figure_conf['mesh_ambient']}")
        print(f"  Diffuse: {self.figure_conf['mesh_diffuse']}")
        print(f"  Specular: {self.figure_conf['mesh_specular']}")
        print(f"  Specular Power: {self.figure_conf['mesh_specular_power']}")
        print(f"  Smooth Shading: {self.figure_conf['mesh_smooth_shading']}")

        print("=" * 40)
        return self.figure_conf.copy()

    ###############################################################################################
    def _list_all_views_and_layouts(self) -> List[str]:
        """
        List available layout configurations from the loaded JSON file.

        Returns
        -------
        List[str]
            List of configuration names available for plotting.

        Examples
        --------
        >>> plotter = SurfacePlotter("configs.json")
        >>> layouts = plotter._list_all_views_and_layouts()
        >>> print(layouts)
        ['8_views', '8_views_8x1', '8_views_1x8', '6_views', '6_views_6x1', '6_views_1x6', '4_views', '4_views_4x1', '4_views_1x4', '2_views', 'lateral', 'medial', 'dorsal', 'ventral', 'rostral', 'caudal']
        """

        all_views_and_layouts = (
            self._list_multiviews_layouts() + self._list_single_views()
        )

        return all_views_and_layouts

    ###############################################################################################
    def _list_multiviews_layouts(self) -> List[str]:
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

        return [name for name in self.layouts_conf.keys()]

    ###############################################################################################
    def _list_single_views(self) -> List[str]:
        """
        List available single view names.

        """

        all_single_views = self.views_conf.keys()

        # Remove the hemisphere information from the view names
        single_views = []
        for i, view in enumerate(all_single_views):
            # Remove the hemisphere information from the view names
            if view.startswith("lh-"):
                view = view.replace("lh-", "")

                single_views.append(view)

        return single_views

    def _get_valid_views(self, views: Union[str, List[str]]) -> List[str]:
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
        multiviews_layouts = self._list_multiviews_layouts()

        # Get the single views
        single_views = self._list_single_views()

        # Check if all the views are valid
        valid_views = cltmisc.list_intercept(views, multiviews_layouts + single_views)

        if len(valid_views) == 0:
            raise ValueError(
                f"No valid views found in '{views}'. "
                f"Available options for multi-views layouts: {self._list_multiviews_layouts()}"
                f" and for single views: {self._list_single_views()}"
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
                f"{self._list_multiviews_layouts()}. "
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
    def _get_config_for_views(
        self, valid_views: list, views_orientation: str = "horizontal", n_maps: int = 1
    ):
        """
        Get the configuration for the specified views.

        Parameters
        ----------
        valid_views : list
            List of valid view names to filter the configuration.

        n_maps : int, default 1
            Number of maps to determine the configuration. I

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary containing shape, window_size, and views.

        Raises
        ------
        KeyError
            If the specified views configuration is not found.

        Examples
        --------
        >>> plotter = SurfacePlotter("configs.json")
        >>> config = plotter._get_config_for_views("8_views")
        >>> print(config["shape"])  # Output: (2, 4)
        """

        # Get the list of available multiviews layouts
        multiviews_layouts = self._list_multiviews_layouts()

        # Get the list of available single views
        multi_views = cltmisc.list_intercept(valid_views, multiviews_layouts)

        if len(multi_views) > 0:

            if n_maps > 1:
                if views_orientation == "horizontal":
                    # Use the first multi-views layout with 1x on the name
                    config_name = [v for v in multi_views if "1x" in v]

                    if len(config_name) == 0:
                        # If no 1x layout found, use the first multi-views layout
                        config_name = (
                            "6_views_1x6" if "6_views" in multi_views else "8_views_1x8"
                        )
                    else:
                        config_name = config_name[0]

                else:
                    # Use the first multi-views layout with x1 on the name
                    config_name = [v for v in multi_views if "x1" in v]

                    if len(config_name) == 0:
                        # If no x1 layout found, use the first multi-views layout
                        config_name = (
                            "6_views_6x1" if "6_views" in multi_views else "8_views_8x1"
                        )
                    else:
                        config_name = config_name[0]

            else:
                # Use the first multi-views layout without 1x or x1 on the name
                config_name = multi_views[0]

            config = self.views_conf[config_name]

        else:
            if views_orientation == "horizontal":
                reference_view = "8_views_1x8"
            else:
                reference_view = "8_views_8x1"

            # Create a dynamic configuration based on the valid views
            valid_views = cltmisc.list_intercept(valid_views, self._list_single_views())
            config = self._create_dynamic_config_for_hemisphere(
                valid_views, reference_view
            )

        return config

    ###############################################################################################
    def update_figure_config(self, auto_save: bool = True, **kwargs) -> None:
        """
        Update figure configuration parameters with validation and automatic saving.

        This method allows you to easily customize the visual appearance of your
        brain plots by updating styling parameters like colors, fonts, and mesh properties.

        Parameters
        ----------
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
            print(
                "Use plotter.list_figure_config_options() to see available parameters."
            )
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
            old_value = self.figure_conf.get(param, "Not set")
            self.figure_conf[param] = value
            updated_params.append(f"  {param}: {old_value} → {value}")

        # Display update summary
        print("✅ Figure configuration updated:")
        print("\n".join(updated_params))

        # Auto-save if requested
        if auto_save:
            self.save_config()
            print(f"💾 Changes saved to: {self.config_file}")

    def apply_theme(self, theme_name: str, auto_save: bool = True) -> None:
        """
        Apply predefined visual themes to quickly customize plot appearance.

        Parameters
        ----------
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

        themes = self.themes_conf

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
            old_value = self.figure_conf.get(param, "Not set")
            self.figure_conf[param] = value
            updated_params.append(f"  {param}: {old_value} → {value}")

        print("Updated parameters:")
        print("\n".join(updated_params))

        if auto_save:
            self.save_config()
            print(f"💾 Theme saved to: {self.config_file}")

    def list_available_themes(self) -> None:
        """
        Display all available themes with descriptions and previews.

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

    def list_figure_config_options(self) -> None:
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
                current_value = self.figure_conf.get(param, "Not set")
                print(f"  {param:25s} ({param_type:5s}) - {description}")
                print(f"  {'':25s} Current: {current_value}, Examples: {examples}")
                print()

        print("💡 Usage Examples:")
        print("   plotter.update_figure_config(background_color='white')")
        print("   plotter.update_figure_config(title_font_size=14, mesh_ambient=0.3)")
        print("   plotter.update_figure_config(auto_save=False, **params)")
        print("=" * 60)

    def reset_figure_config(self, auto_save: bool = True) -> None:
        """
        Reset figure configuration to default values.

        Parameters
        ----------
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
            old_value = self.figure_conf.get(param, "Not set")
            if old_value != default_value:
                changes.append(f"  {param}: {old_value} → {default_value}")

        if changes:
            print("Changes:")
            print("\n".join(changes))
        else:
            print("Configuration already at default values.")

        # Apply defaults
        self.figure_conf.update(default_config)

        if auto_save:
            self.save_config()
            print(f"💾 Default configuration saved to: {self.config_file}")

    def save_config(self) -> None:
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
                "figure_conf": self.figure_conf,
                "views_conf": self.views_conf,
            }

            # Write to file with proper formatting
            with open(self.config_file, "w") as f:
                json.dump(complete_config, f, indent=4, sort_keys=False)

            print(f"✅ Configuration saved successfully to: {self.config_file}")

        except Exception as e:
            raise IOError(f"Failed to save configuration: {e}")

    def preview_theme(self, theme_name: str) -> None:
        """
        Preview a theme's parameters without applying them.

        Parameters
        ----------
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


################################# Helper Functions ################################
def _get_shared_limits(surfaces, map_name, vmin, vmax):
    """Get shared vmin and vmax from surfaces if not provided."""

    # Concatenate data from all surfaces
    for i, surf in enumerate(surfaces):
        if map_name in surf.mesh.point_data:
            if i == 0:
                data = surf.mesh.point_data[map_name]
            else:
                data = np.concatenate((data, surf.mesh.point_data[map_name]))

    if vmin is None:
        vmin = np.min(data)

    if vmax is None:
        vmax = np.max(data)

    return vmin, vmax


def _get_map_limits(surfaces, map_name, colormap_style, v_limits):
    """Get real vmin and vmax from surfaces if not provided."""
    vmin, vmax = v_limits
    real_limits = []

    if not isinstance(surfaces, list):
        surfaces = [surfaces]

    if isinstance(map_name, list):
        map_name = map_name[0]

    if colormap_style == "individual":
        for surf in surfaces:
            data = surf.mesh.point_data[map_name]
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
        vmin, vmax = _get_shared_limits(surfaces, map_name, vmin, vmax)
        return [(vmin, vmax, map_name)] * len(surfaces)
