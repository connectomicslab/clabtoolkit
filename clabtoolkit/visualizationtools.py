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
from . import visualization_utils as visutils


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
        configs = visutils.load_configs(self.config_file)

        # Create attributes
        self.figure_conf = configs["figure_conf"]
        self.views_conf = configs["views_conf"]
        self.layouts_conf = configs["layouts_conf"]
        self.themes_conf = configs["themes_conf"]

        # Define mapping from simple view names to configuration titles
        self._view_name_mapping = {
            "lateral": ["LH: Lateral view", "RH: Lateral view"],
            "medial": ["LH: Medial view", "RH: Medial view"],
            "dorsal": ["Dorsal view"],
            "ventral": ["Ventral view"],
            "rostral": ["Rostral view"],
            "caudal": ["Caudal view"],
        }

    ###############################################################################################
    def _build_plotting_config(
        self,
        views: list,
        hemi_id: str = ["lh", "rh"],
        orientation: str = "horizontal",
        maps_names: Union[str, List[str]] = ["default"],
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
        maps_names = cltmisc.to_list(maps_names)
        colormaps = cltmisc.to_list(colormaps)
        v_limits = cltmisc.to_list(v_limits)
        colorbar_titles = cltmisc.to_list(colorbar_titles) if colorbar_titles else None
        surfaces = cltmisc.to_list(surfaces) if surfaces else []

        n_maps = len(maps_names)
        n_surfaces = len(surfaces)

        # Force single view when both maps and surfaces > 1
        if n_maps > 1 and n_surfaces > 1:
            print(
                "🔧 FORCING single view (dorsal) because both n_maps > 1 and n_surfaces > 1"
            )
            views = ["dorsal"]

        # Get view configuration
        view_ids = visutils.get_views_to_plot(self, views, hemi_id=hemi_id)
        n_views = len(view_ids)

        if n_maps > 1 and n_surfaces > 1:
            view_ids = ["merg-dorsal"]
            n_views = 1

        print(
            f"Number of views: {n_views}, Number of maps: {n_maps}, Number of surfaces: {n_surfaces}"
        )

        # Check if colorbar is needed
        colorbar = colorbar and visutils.colorbar_needed(maps_names, surfaces)

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

    ################################################################################################
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

        limits_list = visutils.get_map_limits(
            objs2plot=surfaces,
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
            map_limits = visutils.get_map_limits(
                objs2plot=surfaces,
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

                        limits_list = visutils.get_map_limits(
                            objs2plot=surfaces,
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

                        limits_list = visutils.get_map_limits(
                            objs2plot=surfaces,
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
                    map_limits = visutils.get_map_limits(
                        objs2plot=surfaces,
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
            map_limits = visutils.get_map_limits(
                objs2plot=surfaces,
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

                        limits_list = visutils.get_map_limits(
                            objs2plot=surfaces,
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

                        limits_list = visutils.get_map_limits(
                            objs2plot=surfaces,
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
                    map_limits = visutils.get_map_limits(
                        objs2plot=surfaces,
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
            map_limits = visutils.get_map_limits(
                objs2plot=surfaces,
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

                        limits_list = visutils.get_map_limits(
                            objs2plot=surfaces,
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

                        limits_list = visutils.get_map_limits(
                            objs2plot=surfaces,
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
                    map_limits = visutils.get_map_limits(
                        objs2plot=surfaces,
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
                        map_limits = visutils.get_map_limits(
                            objs2plot=surfaces[surf_idx],
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
                                limits_list = visutils.get_map_limits(
                                    objs2plot=surfaces[surf_idx],
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
                                limits_list = visutils.get_map_limits(
                                    objs2plot=surfaces[surf_idx],
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
                            map_limits = visutils.get_map_limits(
                                objs2plot=surfaces,
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
                            limits_list = visutils.get_map_limits(
                                objs2plot=surfaces[surf_idx],
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
                            limits_list = visutils.get_map_limits(
                                objs2plot=surfaces[surf_idx],
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
                        map_limits = visutils.get_map_limits(
                            objs2plot=surfaces,
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

        map_limits = visutils.get_map_limits(
            objs2plot=surface,
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
                        map_limits = visutils.get_map_limits(
                            objs2plot=surfaces[surf_idx],
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
                        limits_list = visutils.get_map_limits(
                            objs2plot=surfaces[surf_idx],
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
                        view_limits = visutils.get_map_limits(
                            objs2plot=surfaces,
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
                        map_limits = visutils.get_map_limits(
                            objs2plot=surfaces[surf_idx],
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
                        limits_list = visutils.get_map_limits(
                            objs2plot=surfaces[surf_idx],
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
                        view_limits = visutils.get_map_limits(
                            objs2plot=surfaces,
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
                        map_limits = visutils.get_map_limits(
                            objs2plot=surfaces[0],
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
                        limits_list = visutils.get_map_limits(
                            objs2plot=surfaces[0],
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
                            map_limits = visutils.get_map_limits(
                                objs2plot=surfaces,
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
                        map_limits = visutils.get_map_limits(
                            objs2plot=surfaces[0],
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
                        limits_list = visutils.get_map_limits(
                            objs2plot=surfaces[0],
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
                            map_limits = visutils.get_map_limits(
                                objs2plot=surfaces,
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
        limits_list = visutils.get_map_limits(
            objs2plot=surfaces,
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

            map_limits = visutils.get_map_limits(
                objs2plot=[surf_lh, surf_rh, surf_merg],
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
                            map_limits = visutils.get_map_limits(
                                objs2plot=tmp_surface,
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

                                limits_list = visutils.get_map_limits(
                                    objs2plot=tmp_surface,
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
                                map_limits = visutils.get_map_limits(
                                    objs2plot=surf_merg,
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
                                map_limits = visutils.get_map_limits(
                                    objs2plot=tmp_surface,
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

                                limits_list = visutils.get_map_limits(
                                    objs2plot=tmp_surface,
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
                                map_limits = visutils.get_map_limits(
                                    objs2plot=surf_merg,
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

    ###############################################################################
    # Public methods
    def plot_hemispheres(
        self,
        surf_rh: cltsurf.Surface,
        surf_lh: cltsurf.Surface,
        maps_names: Union[str, List[str]] = ["default"],
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

        maps_names : str or list of str, default ["default"]
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
        surf_merg = cltsurf.merge_surfaces([surf_lh, surf_rh])

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
        view_ids = visutils.get_views_to_plot(self, views, ["lh", "rh"])

        # Determine rendering mode based on save_path, environment, and threading preference
        save_mode, use_off_screen, use_notebook, use_threading = (
            visutils.determine_render_mode(save_path, notebook, non_blocking)
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
            visutils.determine_render_mode(save_path, notebook, non_blocking)
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

            surf = visutils.prepare_obj_for_plotting(
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

                    visutils.add_colorbar(
                        self,
                        plotter=pv_plotter,
                        colorbar_subplot=(row, col),
                        vmin=vmin,
                        vmax=vmax,
                        map_name=colorbar_id,
                        colormap=colormap,
                        colorbar_title=colorbar_title,
                        colorbar_position=orientation,
                    )

        # successful_links = visutils.link_brain_subplot_cameras(pv_plotter, brain_positions)

        # Handle final rendering - either save, display blocking, or display non-blocking
        visutils.finalize_plot(pv_plotter, save_mode, save_path, use_threading)

    ###############################################################################
    def plot_surfaces(
        self,
        surfaces: Union[
            cltsurf.Surface, List[cltsurf.Surface], List[List[cltsurf.Surface]]
        ],
        hemi_id: Union[str, List[str]] = "both",
        views: Union[str, List[str]] = "dorsal",
        views_orientation: str = "horizontal",
        notebook: bool = False,
        map_names: Union[str, List[str]] = ["default"],
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

        map_names : Union[str, List[str]], default ["default"]
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

        # Validate and process hemi_id parameter
        if isinstance(hemi_id, str):
            hemi_id = [hemi_id]

        # the hemi_id must be one of the following
        valid_hemi_ids = ["lh", "rh", "both"]

        # Leave in hemi_id only valid values
        hemi_id = [h for h in hemi_id if h in valid_hemi_ids]

        if "both" in hemi_id and len(hemi_id) > 1:
            hemi_id = ["lh", "rh"]

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
                    surf2plot.append(cltsurf.merge_surfaces(surf))

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
            visutils.determine_render_mode(save_path, notebook, non_blocking)
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
            surf = visutils.prepare_obj_for_plotting(
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

                    visutils.add_colorbar(
                        self,
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
        visutils.finalize_plot(pv_plotter, save_mode, save_path, use_threading)

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

        return visutils.list_available_view_names(self)

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

        return visutils.list_available_layouts(self)

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

        return visutils.get_layout_details(self, views)

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

        return visutils.get_figure_config(self)

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

        all_views_and_layouts = visutils.list_multiviews_layouts(
            self
        ) + visutils.list_single_views(self)

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

        return visutils.list_multiviews_layouts(self)

    ###############################################################################################
    def _list_single_views(self) -> List[str]:
        """
        List available single view names.

        """

        return visutils.list_single_views(self)

    def _create_threaded_plot(self, plotter: pv.Plotter) -> None:
        """
        Create and show plot in a separate thread for non-blocking visualization.

        Parameters
        ----------
        plotter : pv.Plotter
            PyVista plotter instance ready for display.
        """

        visutils.create_threaded_plot(plotter)

        print("Plot opened in separate window. Terminal remains interactive.")
        print("Note: Plot window may take a moment to appear.")

    ###############################################################################################
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

        return visutils.get_valid_views(self, views)

    ###############################################################################################
    def update_figure_config(self, auto_save: bool = False, **kwargs) -> None:
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

        visutils.update_figure_config(self, auto_save, **kwargs)

    ###############################################################################################
    def apply_theme(self, theme_name: str, auto_save: bool = False) -> None:
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

        visutils.apply_theme(self, theme_name, auto_save)

    ###############################################################################################
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

        visutils.list_figure_config_options(self)

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

        visutils.reset_figure_config(self, auto_save)

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

        visutils.save_config(self)

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

        visutils.preview_theme(self, theme_name)
