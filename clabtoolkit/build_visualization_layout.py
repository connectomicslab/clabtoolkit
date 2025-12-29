"""
Module for building visualization layout configurations for brain surface plots.

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


################################################################################################
def build_layout_config(
    plotobj,
    valid_views,
    objs2plot,
    maps_dict,
    colorbar,
    orientation,
    colorbar_style,
    colorbar_position,
):
    """Build the basic layout configuration."""

    maps_names = list(maps_dict.keys())
    n_views = len(valid_views)
    n_maps = len(maps_names)
    n_objects = len(objs2plot)
    colorbar_size = plotobj.figure_conf["colorbar_size"]

    limits_dict, charac_dict = visutils.get_map_characteristics(objs2plot, maps_dict)
    ##### Determine colormap limits based on colorbar style #####
    colormap_limits = {}
    for view_idx in range(n_views):
        for map_idx in range(n_maps):
            map_name = maps_names[map_idx]

            for surf_idx in range(n_objects):

                if colorbar_style == "individual":
                    colormap_limits[(map_idx, surf_idx, view_idx)] = [
                        limits_dict[map_name]["individual"][surf_idx]
                    ]

                elif colorbar_style == "shared_by_map":
                    colormap_limits[(map_idx, surf_idx, view_idx)] = [
                        limits_dict[map_name]["shared_by_map"]
                    ]

                else:
                    colormap_limits[(map_idx, surf_idx, view_idx)] = [
                        limits_dict["shared"]
                    ]

    if n_views == 1 and n_maps == 1 and n_objects == 1:  # Works fine!

        if maps_dict[maps_names[0]]["colormap"] == "colortable":
            colorbar = False

        return single_element_layout(
            maps_dict,
            colormap_limits,
            charac_dict,
            colorbar,
            colorbar_position,
            colorbar_size,
        )

    elif n_views == 1 and n_maps == 1 and n_objects > 1:  # Works fine !

        return single_map_multi_surface_layout(
            maps_dict,
            colormap_limits,
            charac_dict,
            colorbar,
            orientation,
            colorbar_style,
            colorbar_position,
            colorbar_size,
        )

    elif n_views == 1 and n_maps > 1 and n_objects == 1:  # Works fine !

        return multi_map_single_surface_layout(
            maps_dict,
            colormap_limits,
            charac_dict,
            colorbar,
            orientation,
            colorbar_style,
            colorbar_position,
            colorbar_size,
        )

    elif n_views == 1 and n_maps > 1 and n_objects > 1:  # Works fine !

        return multi_map_multi_surface_layout(
            maps_dict,
            colormap_limits,
            charac_dict,
            colorbar,
            orientation,
            colorbar_style,
            colorbar_position,
            colorbar_size,
        )

    elif n_views > 1 and n_maps == 1 and n_objects == 1:  # Works fine !

        return multi_view_single_element_layout(
            maps_dict,
            colormap_limits,
            charac_dict,
            valid_views,
            colorbar,
            orientation,
            colorbar_position,
            colorbar_size,
        )

    elif n_views > 1 and n_maps == 1 and n_objects > 1:  # Works fine !

        return multi_view_multi_surface_layout(
            maps_dict,
            colormap_limits,
            charac_dict,
            valid_views,
            colorbar,
            orientation,
            colorbar_position,
            colorbar_style,
            colorbar_size,
        )

    elif n_views > 1 and n_maps > 1 and n_objects == 1:  # Works fine !

        return multi_view_multi_map_layout(
            maps_dict,
            colormap_limits,
            charac_dict,
            valid_views,
            colorbar,
            orientation,
            colorbar_position,
            colorbar_style,
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


###############################################################################################
def single_element_layout(
    maps_dict: Dict,
    colormap_limits: Dict,
    charac_dict: Dict,
    colorbar: bool,
    colorbar_position: str,
    colorbar_size: float,
):
    """Handle single view, single map, single surface case."""
    brain_positions = {(0, 0, 0): (0, 0)}

    map_names = list(maps_dict.keys())[0]
    colormap = charac_dict[map_names]["colormap"]
    colorbar_title = charac_dict[map_names]["colorbar_title"]

    colorbar_list = []

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

        cb_dict["colormap"] = colormap
        cb_dict["map_name"] = map_names
        cb_dict["vmin"] = colormap_limits[(0, 0, 0)][0][0]
        cb_dict["vmax"] = colormap_limits[(0, 0, 0)][0][1]
        cb_dict["title"] = colorbar_title

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


###############################################################################################
def multi_map_single_surface_layout(
    maps_dict,
    colormap_limits,
    charac_dict,
    colorbar,
    orientation,
    colorbar_style,
    colorbar_position,
    colorbar_size,
):
    """Handle multiple maps, single surface case."""

    if orientation == "horizontal":
        return horizontal_multi_map_layout(
            maps_dict,
            colormap_limits,
            charac_dict,
            colorbar,
            colorbar_style,
            colorbar_position,
            colorbar_size,
        )
    elif orientation == "vertical":
        return vertical_multi_map_layout(
            maps_dict,
            colormap_limits,
            charac_dict,
            colorbar,
            colorbar_style,
            colorbar_position,
            colorbar_size,
        )
    else:  # grid
        return grid_multi_map_layout(
            maps_dict,
            colormap_limits,
            charac_dict,
            colorbar,
            colorbar_style,
            colorbar_position,
            colorbar_size,
        )


###############################################################################################
def horizontal_multi_map_layout(
    maps_dict,
    colormap_limits,
    charac_dict,
    colorbar,
    colorbar_style,
    colorbar_position,
    colorbar_size,
):
    """Handle horizontal layout for multiple maps."""

    brain_positions = {}
    dims = visutils.get_plot_config_dimensions(colormap_limits)
    n_objs2plot = dims[1]

    maps_names = list(maps_dict.keys())
    n_maps = len(maps_names)

    ##### Determine colormap limits based on colorbar style #####
    colorbar_list = []
    if not colorbar:
        shape = [1, n_maps]
        row_weights = [1]
        col_weights = [1] * n_maps
        groups = []

        for map_idx in range(n_maps):
            brain_positions[(map_idx, 0, 0)] = (0, map_idx)

    else:

        if colorbar_style == "individual":
            # Set position-specific parameters
            if colorbar_position == "right":
                shape = [1, n_maps * 2]
                row_weights = [1]
                col_weights = [1, colorbar_size] * n_maps
                brain_pos_col = lambda idx: idx * 2
                cb_pos = lambda idx: (0, idx * 2 + 1)
                orientation = "vertical"

            else:  # bottom
                shape = [2, n_maps]
                row_weights = [1, colorbar_size]
                col_weights = [1] * n_maps
                brain_pos_col = lambda idx: idx
                cb_pos = lambda idx: (1, idx)
                orientation = "horizontal"

            groups = []

            # Single loop for both cases
            for map_idx in range(n_maps):
                brain_positions[(map_idx, 0, 0)] = (0, brain_pos_col(map_idx))

                # Extract map properties
                colorbar_title = charac_dict[maps_names[map_idx]]["colorbar_title"]
                colormap = charac_dict[maps_names[map_idx]]["colormap"]
                # Build colorbar dictionary
                cb_dict = {
                    "position": cb_pos(map_idx),
                    "orientation": orientation,
                    "colormap": colormap,
                    "map_name": maps_names[map_idx],
                    "vmin": colormap_limits[(map_idx, 0, 0)][0][0],
                    "vmax": colormap_limits[(map_idx, 0, 0)][0][1],
                    "title": colorbar_title,
                }

                colorbar_list.append(cb_dict)

        else:  # shared colorbar

            colorbar_title = charac_dict["shared"]["colorbar_title"]
            colormap = charac_dict["shared"]["colormap"]

            # Set position-specific parameters first
            if colorbar_position == "right":
                shape = [1, n_maps + 1]
                row_weights = [1]
                col_weights = [1] * n_maps + [colorbar_size]
                groups = []
                cb_position = (0, n_maps)
                cb_orientation = "vertical"
            else:  # bottom
                shape = [2, n_maps]
                row_weights = [1, colorbar_size]
                col_weights = [1] * n_maps
                groups = [
                    (1, slice(0, n_maps))
                ]  # Colorbar spans all columns in bottom row
                cb_position = (1, 0)
                cb_orientation = "horizontal"

            # Set brain positions
            for map_idx in range(n_maps):
                brain_positions[(map_idx, 0, 0)] = (0, map_idx)

            # Build shared colorbar dictionary
            cb_dict = {
                "position": cb_position,
                "orientation": cb_orientation,
                "colormap": colormap,
                "map_name": " + ".join(maps_names),
                "vmin": colormap_limits[(0, 0, 0)][0][0],
                "vmax": colormap_limits[(0, 0, 0)][0][1],
                "title": colorbar_title,
            }

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


###############################################################################################
def vertical_multi_map_layout(
    maps_dict,
    colormap_limits,
    charac_dict,
    colorbar,
    colorbar_style,
    colorbar_position,
    colorbar_size,
):
    """Handle horizontal layout for multiple maps."""

    brain_positions = {}
    maps_names = list(maps_dict.keys())
    n_maps = len(maps_names)

    colorbar_list = []
    if not colorbar:
        shape = [n_maps, 1]
        row_weights = [1] * n_maps
        col_weights = [1]
        groups = []

        for map_idx in range(n_maps):
            brain_positions[(map_idx, 0, 0)] = (map_idx, 0)

    else:
        if colorbar_style == "individual":
            # Set position-specific parameters
            if colorbar_position == "right":
                shape = [n_maps, 2]
                row_weights = [1] * n_maps
                col_weights = [1, colorbar_size]
                brain_pos_col = lambda idx: idx
                cb_pos = lambda idx: (idx, 1)
                orientation = "vertical"

            else:  # bottom
                shape = [n_maps * 2, 1]
                row_weights = [1, colorbar_size] * n_maps
                col_weights = [1]
                brain_pos_col = lambda idx: idx * 2
                cb_pos = lambda idx: (idx * 2 + 1, 0)
                orientation = "horizontal"

            groups = []

            # Single loop for both cases
            for map_idx in range(n_maps):
                brain_positions[(map_idx, 0, 0)] = (brain_pos_col(map_idx), 0)

                # Extract map properties
                colorbar_title = charac_dict[maps_names[map_idx]]["colorbar_title"]
                colormap = charac_dict[maps_names[map_idx]]["colormap"]
                # Build colorbar dictionary

                # Build colorbar dictionary
                cb_dict = {
                    "position": cb_pos(map_idx),
                    "orientation": orientation,
                    "colormap": colormap,
                    "map_name": maps_names[map_idx],
                    "vmin": colormap_limits[(map_idx, 0, 0)][0][0],
                    "vmax": colormap_limits[(map_idx, 0, 0)][0][1],
                    "title": colorbar_title,
                }

                colorbar_list.append(cb_dict)

        else:  # shared colorbar
            # Set position-specific parameters first
            colorbar_title = charac_dict["shared"]["colorbar_title"]
            colormap = charac_dict["shared"]["colormap"]
            if colorbar_position == "right":
                shape = [n_maps, 2]
                row_weights = [1] * n_maps
                col_weights = [1, colorbar_size]
                groups = [(slice(0, n_maps), 1)]

                cb_position = (0, 1)
                cb_orientation = "vertical"
            else:  # bottom
                shape = [n_maps + 1, 1]
                row_weights = [1] * n_maps + [colorbar_size]
                col_weights = [1]
                groups = [(n_maps, 0)]

                cb_position = (n_maps, 0)
                cb_orientation = "horizontal"

            # Set brain positions
            for map_idx in range(n_maps):
                brain_positions[(map_idx, 0, 0)] = (map_idx, 0)

            # Build shared colorbar dictionary
            cb_dict = {
                "position": cb_position,
                "orientation": cb_orientation,
                "colormap": colormap,  # Use first map's colormap
                "map_name": " + ".join(maps_names),
                "vmin": colormap_limits[(0, 0, 0)][0][0],
                "vmax": colormap_limits[(0, 0, 0)][0][1],
                "title": colorbar_title,
            }

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


###############################################################################################
def grid_multi_map_layout(
    maps_dict,
    colormap_limits,
    charac_dict,
    colorbar,
    colorbar_style,
    colorbar_position,
    colorbar_size,
):
    """Handle grid layout for multiple maps."""

    #
    brain_positions = {}
    maps_names = list(maps_dict.keys())
    n_maps = len(maps_names)

    # Determine optimal grid
    optimal_grid, positions = cltplot.calculate_optimal_subplots_grid(n_maps)

    colorbar_list = []
    if not colorbar:
        shape = list(optimal_grid)
        row_weights = [1] * optimal_grid[0]
        col_weights = [1] * optimal_grid[1]
        groups = []

        for map_idx in range(n_maps):
            pos = positions[map_idx]
            brain_positions[(map_idx, 0, 0)] = (pos[0], pos[1])

    else:
        if colorbar_style == "individual":
            if colorbar_position == "right":
                shape = [optimal_grid[0], optimal_grid[1] * 2]
                row_weights = [1] * optimal_grid[0]
                col_weights = [1, colorbar_size] * optimal_grid[1]
                groups = []

                for map_idx in range(n_maps):
                    pos = positions[map_idx]
                    brain_positions[(map_idx, 0, 0)] = (pos[0], pos[1] * 2)

                    # Extract map properties
                    map_data = maps_dict[maps_names[map_idx]]
                    colormap = charac_dict[maps_names[map_idx]]["colormap"]
                    colorbar_title = charac_dict[maps_names[map_idx]]["colorbar_title"]

                    cb_dict = {}
                    cb_dict["position"] = (pos[0], pos[1] * 2 + 1)
                    cb_dict["orientation"] = "vertical"
                    cb_dict["colormap"] = colormap
                    cb_dict["map_name"] = maps_names[map_idx]
                    cb_dict["vmin"] = colormap_limits[(map_idx, 0, 0)][0][0]
                    cb_dict["vmax"] = colormap_limits[(map_idx, 0, 0)][0][1]
                    cb_dict["title"] = colorbar_title

                    colorbar_list.append(cb_dict)

            else:  # bottom
                shape = [optimal_grid[0] * 2, optimal_grid[1]]
                row_weights = [1, colorbar_size] * optimal_grid[0]
                col_weights = [1] * optimal_grid[1]
                groups = []

                for map_idx in range(n_maps):
                    pos = positions[map_idx]
                    brain_positions[(map_idx, 0, 0)] = (pos[0] * 2, pos[1])

                    # Extract map properties
                    map_data = maps_dict[maps_names[map_idx]]
                    colormap = charac_dict[maps_names[map_idx]]["individual"][
                        "colormap"
                    ]
                    colorbar_title = charac_dict[maps_names[map_idx]]["individual"][
                        "colorbar_title"
                    ]

                    cb_dict = {}
                    cb_dict["position"] = (pos[0] * 2 + 1, pos[1])
                    cb_dict["orientation"] = "horizontal"
                    cb_dict["colormap"] = colormap
                    cb_dict["map_name"] = maps_names[map_idx]
                    cb_dict["vmin"] = colormap_limits[(map_idx, 0, 0)][0][0]
                    cb_dict["vmax"] = colormap_limits[(map_idx, 0, 0)][0][1]
                    cb_dict["title"] = colorbar_title

                    colorbar_list.append(cb_dict)

        else:  # shared colorbar
            cb_dict = {}

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

                # Set brain positions
            for map_idx in range(n_maps):
                pos = positions[map_idx]
                brain_positions[(map_idx, 0, 0)] = pos

            # Extract map properties
            colormap = charac_dict["shared"]["colormap"]
            colorbar_title = charac_dict["shared"]["colorbar_title"]

            # Store shared limits for all maps
            cb_dict["colormap"] = colormap
            cb_dict["map_name"] = " + ".join(maps_names)
            cb_dict["vmin"] = colormap_limits[(0, 0, 0)][0][0]
            cb_dict["vmax"] = colormap_limits[(0, 0, 0)][0][1]
            cb_dict["title"] = colorbar_title

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
# Multiple objs2plot and multiple maps cases
################################################################################
def multi_map_multi_surface_layout(
    maps_dict,
    colormap_limits,
    charac_dict,
    colorbar,
    orientation,
    colorbar_style,
    colorbar_position,
    colorbar_size,
):
    """Handle multiple maps and multiple objs2plot case."""

    maps_names = list(maps_dict.keys())

    brain_positions = {}
    colorbar_list = []

    dims = visutils.get_plot_config_dimensions(colormap_limits)
    n_objs = dims[1]
    n_maps = dims[0]

    ################################################################################

    if orientation == "horizontal":  # The orientation of the maps

        if not colorbar:
            shape = [n_objs, n_maps]
            row_weights = [1] * n_objs
            col_weights = [1] * n_maps
            groups = []
            for map_idx in range(n_maps):
                for surf_idx in range(n_objs):
                    brain_positions[(map_idx, surf_idx, 0)] = (surf_idx, map_idx)

        else:

            # Force colorbar to bottom for this case
            if colorbar_style == "individual":
                if colorbar_position == "right":
                    shape = [n_objs, n_maps * 2]
                    row_weights = [1] * n_objs
                    col_weights = [1, colorbar_size] * n_maps
                    groups = []

                    for map_idx in range(n_maps):
                        map_data = maps_dict[maps_names[map_idx]]
                        colormap = map_data["colormap"]
                        map_name = maps_names[map_idx]
                        colorbar_title = map_data["colorbar_title"]

                        for surf_idx in range(n_objs):
                            brain_positions[(map_idx, surf_idx, 0)] = (
                                surf_idx,
                                map_idx * 2,
                            )

                            cb_dict = {}
                            cb_dict["position"] = (surf_idx, map_idx * 2 + 1)
                            cb_dict["orientation"] = "vertical"
                            cb_dict["colormap"] = colormap
                            cb_dict["map_name"] = map_name
                            cb_dict["title"] = colorbar_title

                            cb_dict["vmin"] = colormap_limits[(map_idx, surf_idx, 0)][
                                0
                            ][0]
                            cb_dict["vmax"] = colormap_limits[(map_idx, surf_idx, 0)][
                                0
                            ][1]

                            colorbar_list.append(cb_dict)

                elif colorbar_position == "bottom":
                    shape = [n_objs * 2, n_maps]
                    row_weights = [1, colorbar_size] * n_objs
                    col_weights = [1] * n_maps
                    groups = []

                    for map_idx in range(n_maps):
                        map_data = maps_dict[maps_names[map_idx]]
                        colormap = map_data["colormap"]
                        map_name = maps_names[map_idx]
                        colorbar_title = map_data["colorbar_title"]

                        for surf_idx in range(n_objs):
                            brain_positions[(map_idx, surf_idx, 0)] = (
                                surf_idx * 2,
                                map_idx,
                            )
                            cb_dict = {}
                            cb_dict["position"] = (surf_idx * 2 + 1, map_idx)
                            cb_dict["orientation"] = "horizontal"
                            cb_dict["colormap"] = colormap
                            cb_dict["map_name"] = map_name
                            cb_dict["title"] = colorbar_title

                            cb_dict["vmin"] = colormap_limits[(map_idx, surf_idx, 0)][
                                0
                            ][0]
                            cb_dict["vmax"] = colormap_limits[(map_idx, surf_idx, 0)][
                                0
                            ][1]

                            colorbar_list.append(cb_dict)

            else:
                for map_idx in range(n_maps):
                    for surf_idx in range(n_objs):
                        brain_positions[(map_idx, surf_idx, 0)] = (
                            surf_idx,
                            map_idx,
                        )
                if colorbar_style == "shared":

                    colormap = charac_dict["shared"]["colormap"]
                    colorbar_title = charac_dict["shared"]["colorbar_title"]

                    cb_dict = {}
                    cb_dict["colormap"] = colormap
                    cb_dict["title"] = colorbar_title
                    cb_dict["map_name"] = " + ".join(maps_names)
                    cb_dict["vmin"] = colormap_limits[(0, 0, 0)][0][0]
                    cb_dict["vmax"] = colormap_limits[(0, 0, 0)][0][1]

                    if colorbar_position == "right":
                        shape = [n_objs, n_maps + 1]
                        row_weights = [1] * n_objs
                        col_weights = [1] * n_maps + [colorbar_size]
                        groups = [(slice(0, n_objs), n_maps)]

                        cb_dict["position"] = (0, n_maps)
                        cb_dict["orientation"] = "vertical"

                    elif colorbar_position == "bottom":
                        shape = [n_objs + 1, n_maps]
                        row_weights = [1] * n_objs + [colorbar_size]
                        col_weights = [1] * n_maps
                        groups = [(n_objs, slice(0, n_maps))]

                        cb_dict["position"] = (n_objs, 0)
                        cb_dict["orientation"] = "horizontal"

                    colorbar_list.append(cb_dict)

                elif colorbar_style == "shared_by_map":
                    ######### One colorbar per map #########

                    shape = [n_objs + 1, n_maps]
                    row_weights = [1] * n_objs + [colorbar_size]
                    col_weights = [1] * n_maps
                    groups = []

                    for map_idx in range(n_maps):

                        colormap = charac_dict[maps_names[map_idx]]["colormap"]
                        colorbar_title = charac_dict[maps_names[map_idx]][
                            "colorbar_title"
                        ]

                        cb_dict = {}
                        cb_dict["colormap"] = colormap
                        cb_dict["title"] = colorbar_title
                        cb_dict["map_name"] = maps_names[map_idx]
                        cb_dict["vmin"] = colormap_limits[(map_idx, 0, 0)][0][0]
                        cb_dict["vmax"] = colormap_limits[(map_idx, 0, 0)][0][1]

                        cb_dict["position"] = (n_objs, map_idx)
                        cb_dict["orientation"] = "horizontal"

                        colorbar_list.append(cb_dict)

    else:  # vertical

        if not colorbar:
            # Maps in rows, objs2plot in columns
            shape = [n_maps, n_objs]
            row_weights = [1] * n_maps
            col_weights = [1] * n_objs
            groups = []

            for map_idx in range(n_maps):
                for surf_idx in range(n_objs):
                    brain_positions[(map_idx, surf_idx, 0)] = (map_idx, surf_idx)
        else:
            # Force colorbar to right for this case
            if colorbar_style == "individual":
                if colorbar_position == "right":
                    shape = [n_maps, n_objs * 2]
                    row_weights = [1] * n_maps
                    col_weights = [1, colorbar_size] * n_objs
                    groups = []

                    for map_idx in range(n_maps):
                        map_data = maps_dict[maps_names[map_idx]]
                        colormap = map_data["colormap"]
                        map_name = maps_names[map_idx]
                        colorbar_title = map_data["colorbar_title"]

                        for surf_idx in range(n_objs):
                            brain_positions[(map_idx, surf_idx, 0)] = (
                                map_idx,
                                surf_idx * 2,
                            )

                            cb_dict = {}
                            cb_dict["position"] = (map_idx, surf_idx * 2 + 1)
                            cb_dict["orientation"] = "vertical"
                            cb_dict["colormap"] = colormap
                            cb_dict["map_name"] = map_name
                            cb_dict["title"] = colorbar_title

                            cb_dict["vmin"] = colormap_limits[(map_idx, surf_idx, 0)][
                                0
                            ][0]
                            cb_dict["vmax"] = colormap_limits[(map_idx, surf_idx, 0)][
                                0
                            ][1]

                            colorbar_list.append(cb_dict)

                elif colorbar_position == "bottom":
                    shape = [n_maps * 2, n_objs]
                    row_weights = [1, colorbar_size] * n_maps
                    col_weights = [1] * n_objs
                    groups = []

                    for map_idx in range(n_maps):
                        map_data = maps_dict[maps_names[map_idx]]
                        colormap = map_data["colormap"]
                        map_name = maps_names[map_idx]
                        colorbar_title = map_data["colorbar_title"]
                        for surf_idx in range(n_objs):
                            brain_positions[(map_idx, surf_idx, 0)] = (
                                map_idx * 2,
                                surf_idx,
                            )
                            cb_dict = {}
                            cb_dict["position"] = (map_idx * 2 + 1, surf_idx)
                            cb_dict["orientation"] = "horizontal"
                            cb_dict["colormap"] = colormap
                            cb_dict["map_name"] = map_name
                            cb_dict["title"] = colorbar_title
                            cb_dict["vmin"] = colormap_limits[(map_idx, surf_idx, 0)][
                                0
                            ][0]
                            cb_dict["vmax"] = colormap_limits[(map_idx, surf_idx, 0)][
                                0
                            ][1]
                            colorbar_list.append(cb_dict)

            else:
                for map_idx in range(n_maps):
                    for surf_idx in range(n_objs):
                        brain_positions[(map_idx, surf_idx, 0)] = (
                            map_idx,
                            surf_idx,
                        )

                if colorbar_style == "shared":
                    colormap = charac_dict[maps_names[map_idx]]["colormap"]
                    colorbar_title = charac_dict[maps_names[map_idx]]["colorbar_title"]

                    cb_dict = {}
                    cb_dict["colormap"] = colormap
                    cb_dict["title"] = colorbar_title
                    cb_dict["map_name"] = " + ".join(maps_names)
                    cb_dict["vmin"] = colormap_limits[(0, 0, 0)][0][0]
                    cb_dict["vmax"] = colormap_limits[(0, 0, 0)][0][1]

                    if colorbar_position == "right":
                        shape = [n_maps, n_objs + 1]
                        row_weights = [1] * n_maps
                        col_weights = [1] * n_objs + [colorbar_size]
                        groups = [(slice(0, n_maps), n_objs)]

                        cb_dict["position"] = (0, n_objs)
                        cb_dict["orientation"] = "vertical"

                    elif colorbar_position == "bottom":
                        shape = [n_maps + 1, n_objs]
                        row_weights = [1] * n_maps + [colorbar_size]
                        col_weights = [1] * n_objs
                        groups = [(n_maps, slice(0, n_objs))]

                        cb_dict["position"] = (n_maps, 0)
                        cb_dict["orientation"] = "horizontal"

                    colorbar_list.append(cb_dict)

                elif colorbar_style == "shared_by_map":
                    shape = [n_maps, n_objs + 1]
                    row_weights = [1] * n_maps
                    col_weights = [1] * n_objs + [colorbar_size]
                    groups = []

                    for map_idx in range(n_maps):

                        map_data = maps_dict[maps_names[map_idx]]
                        colormap = map_data["colormap"]
                        colorbar_title = map_data["colorbar_title"]

                        cb_dict = {}
                        cb_dict["colormap"] = colormap
                        cb_dict["title"] = colorbar_title
                        cb_dict["map_name"] = maps_names[map_idx]
                        cb_dict["vmin"] = colormap_limits[(map_idx, 0, 0)][0][0]
                        cb_dict["vmax"] = colormap_limits[(map_idx, 0, 0)][0][1]

                        cb_dict["position"] = (map_idx, n_objs)
                        cb_dict["orientation"] = "vertical"

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


###############################################################################################
def multi_view_single_element_layout(
    maps_dict,
    colormap_limits,
    charac_dict,
    valid_views,
    colorbar,
    orientation,
    colorbar_position,
    colorbar_size,
):
    """Handle multiple views, single map, single surface case."""

    n_views = len(valid_views)
    brain_positions = {}
    colorbar_list = []

    brain_positions = {}
    map_name = list(maps_dict.keys())[0]

    colormap = charac_dict["shared"]["colormap"]
    colorbar_title = charac_dict["shared"]["colorbar_title"]
    groups = []
    if orientation == "horizontal":
        for view_idx in range(n_views):
            brain_positions[(0, 0, view_idx)] = (0, view_idx)

        if not colorbar:
            shape = [1, n_views]
            row_weights = [1]
            col_weights = [1] * n_views

        else:

            cb_dict = {}
            cb_dict["colormap"] = colormap
            cb_dict["map_name"] = map_name
            cb_dict["vmin"] = colormap_limits[(0, 0, 0)][0][0]
            cb_dict["vmax"] = colormap_limits[(0, 0, 0)][0][1]
            cb_dict["title"] = colorbar_title

            if colorbar_position == "right":
                shape = [1, n_views + 1]
                row_weights = [1]
                col_weights = [1] * n_views + [colorbar_size]
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

        layout_config = {
            "shape": shape,
            "row_weights": row_weights,
            "col_weights": col_weights,
            "groups": groups,
            "brain_positions": brain_positions,
            "colormap_limits": colormap_limits,
        }
        return layout_config, colorbar_list

    elif orientation == "vertical":
        for view_idx in range(n_views):
            brain_positions[(0, 0, view_idx)] = (view_idx, 0)

        if not colorbar:
            shape = [n_views, 1]
            row_weights = [1] * n_views
            col_weights = [1]

        else:
            shape = [n_views, 2]
            row_weights = [1] * n_views
            col_weights = [1, colorbar_size]

            cb_dict = {}
            cb_dict["colormap"] = colormap
            cb_dict["map_name"] = map_name
            cb_dict["vmin"] = colormap_limits[(0, 0, 0)][0][0]
            cb_dict["vmax"] = colormap_limits[(0, 0, 0)][0][1]
            cb_dict["title"] = colorbar_title

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

        layout_config = {
            "shape": shape,
            "row_weights": row_weights,
            "col_weights": col_weights,
            "groups": groups,
            "brain_positions": brain_positions,
            "colormap_limits": colormap_limits,
        }
        return layout_config, colorbar_list

    else:

        return grid_multi_views_layout(
            maps_dict,
            colormap_limits,
            charac_dict,
            valid_views,
            colorbar,
            colorbar_position,
            colorbar_size,
        )


###############################################################################################
def multi_view_multi_surface_layout(
    maps_dict,
    colormap_limits,
    charac_dict,
    valid_views,
    colorbar,
    orientation,
    colorbar_position,
    colorbar_style,
    colorbar_size,
):
    """Handle multiple views and multiple objs2plot case."""
    map_name = list(maps_dict.keys())[0]
    colormap = charac_dict["shared"]["colormap"]
    colorbar_title = charac_dict["shared"]["colorbar_title"]

    dims = visutils.get_plot_config_dimensions(colormap_limits)
    n_objs = dims[1]
    n_views = len(valid_views)

    brain_positions = {}

    colorbar_list = []
    if orientation == "horizontal":
        for surf_idx in range(n_objs):
            for view_idx in range(n_views):
                brain_positions[(0, surf_idx, view_idx)] = (surf_idx, view_idx)

        if not colorbar:
            shape = [n_objs, n_views]
            row_weights = [1] * n_objs
            col_weights = [1] * n_views
            groups = []

        else:
            if colorbar_style == "individual":

                shape = [n_objs, n_views + 1]
                row_weights = [1] * n_objs
                col_weights = [1] * n_views + [colorbar_size]
                groups = []
                for surf_idx in range(n_objs):
                    cb_dict = {}
                    cb_dict["position"] = (surf_idx, n_views)
                    cb_dict["orientation"] = "vertical"
                    cb_dict["colormap"] = colormap
                    cb_dict["map_name"] = map_name
                    cb_dict["title"] = colorbar_title
                    cb_dict["vmin"] = colormap_limits[(0, surf_idx, 0)][0][0]
                    cb_dict["vmax"] = colormap_limits[(0, surf_idx, 0)][0][1]

                    colorbar_list.append(cb_dict)
            else:
                cb_dict = {}
                cb_dict["colormap"] = colormap
                cb_dict["map_name"] = map_name
                cb_dict["vmin"] = colormap_limits[(0, 0, 0)][0][0]
                cb_dict["vmax"] = colormap_limits[(0, 0, 0)][0][1]
                cb_dict["title"] = colorbar_title

                if colorbar_position == "right":
                    shape = [n_objs, n_views + 1]
                    row_weights = [1] * n_objs
                    col_weights = [1] * n_views + [colorbar_size]
                    groups = [(slice(0, n_objs), n_views)]

                    cb_dict["position"] = (0, n_views)
                    cb_dict["orientation"] = "vertical"

                elif colorbar_position == "bottom":
                    shape = [n_objs + 1, n_views]
                    row_weights = [1] * n_objs + [colorbar_size]
                    col_weights = [1] * n_views
                    groups = [(n_objs, slice(0, n_views))]

                    cb_dict["position"] = (n_objs, 0)
                    cb_dict["orientation"] = "horizontal"

                colorbar_list.append(cb_dict)

    else:  # vertical

        for surf_idx in range(n_objs):
            for view_idx in range(n_views):
                brain_positions[(0, surf_idx, view_idx)] = (view_idx, surf_idx)

        if not colorbar:
            # Views in rows, objs2plot in columns
            shape = [n_views, n_objs]
            row_weights = [1] * n_views
            col_weights = [1] * n_objs
            groups = []

        else:

            if colorbar_style == "individual":
                shape = [n_views + 1, n_objs]
                row_weights = [1] * n_views + [colorbar_size]
                col_weights = [1] * n_objs
                groups = []
                for surf_idx in range(n_objs):
                    cb_dict = {}
                    cb_dict["position"] = (n_views, surf_idx)
                    cb_dict["orientation"] = "horizontal"
                    cb_dict["colormap"] = colormap
                    cb_dict["map_name"] = map_name
                    cb_dict["title"] = colorbar_title

                    cb_dict["vmin"] = colormap_limits[(0, surf_idx, 0)][0][0]
                    cb_dict["vmax"] = colormap_limits[(0, surf_idx, 0)][0][1]

                    colorbar_list.append(cb_dict)

            else:

                cb_dict = {}
                cb_dict["colormap"] = colormap
                cb_dict["map_name"] = map_name
                cb_dict["vmin"] = colormap_limits[(0, 0, 0)][0][0]
                cb_dict["vmax"] = colormap_limits[(0, 0, 0)][0][1]
                cb_dict["title"] = colorbar_title

                if colorbar_position == "right":
                    shape = [n_views, n_objs + 1]
                    row_weights = [1] * n_views
                    col_weights = [1] * n_objs + [colorbar_size]
                    groups = [(slice(0, n_views), n_objs)]

                    cb_dict["position"] = (0, n_objs)
                    cb_dict["orientation"] = "vertical"

                elif colorbar_position == "bottom":
                    shape = [n_views + 1, n_objs]
                    row_weights = [1] * n_views + [colorbar_size]
                    col_weights = [1] * n_objs
                    groups = [(n_views, slice(0, n_objs))]

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


###############################################################################################
def multi_view_multi_map_layout(
    maps_dict,
    colormap_limits,
    charac_dict,
    valid_views,
    colorbar,
    orientation,
    colorbar_position,
    colorbar_style,
    colorbar_size,
):
    """Handle multiple views and multiple maps case."""

    n_views = len(valid_views)
    maps_names = list(maps_dict.keys())
    n_maps = len(maps_names)

    brain_positions = {}
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

        else:
            if colorbar_style == "individual":
                shape = [n_maps, n_views + 1]
                row_weights = [1] * n_maps
                col_weights = [1] * n_views + [colorbar_size]
                groups = []
                for map_idx in range(n_maps):
                    cb_dict = {}
                    cb_dict["position"] = (map_idx, n_views)
                    cb_dict["orientation"] = "vertical"

                    colormap = charac_dict[maps_names[map_idx]]["colormap"]
                    colorbar_title = charac_dict[maps_names[map_idx]]["colorbar_title"]

                    cb_dict["colormap"] = colormap
                    cb_dict["map_name"] = maps_names[map_idx]
                    cb_dict["title"] = colorbar_title
                    cb_dict["vmin"] = colormap_limits[(map_idx, 0, 0)][0][0]
                    cb_dict["vmax"] = colormap_limits[(map_idx, 0, 0)][0][1]

                    for view_idx in range(n_views):
                        brain_positions[(map_idx, 0, view_idx)] = (
                            map_idx,
                            view_idx,
                        )

                    colorbar_list.append(cb_dict)
            else:

                colorbar_title = charac_dict["shared"]["colorbar_title"]
                colormap = charac_dict["shared"]["colormap"]

                cb_dict = {}
                cb_dict["colormap"] = colormap
                cb_dict["map_name"] = " + ".join(maps_names)
                cb_dict["vmin"] = colormap_limits[(0, 0, 0)][0][0]
                cb_dict["vmax"] = colormap_limits[(0, 0, 0)][0][1]
                cb_dict["title"] = colorbar_title

                for map_idx in range(n_maps):
                    for view_idx in range(n_views):
                        brain_positions[(map_idx, 0, view_idx)] = (
                            map_idx,
                            view_idx,
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

        else:
            if colorbar_style == "individual":
                shape = [n_views + 1, n_maps]
                row_weights = [1] * n_views + [colorbar_size]
                col_weights = [1] * n_maps
                groups = []

                for map_idx in range(n_maps):
                    cb_dict = {}
                    cb_dict["position"] = (n_views, map_idx)
                    cb_dict["orientation"] = "horizontal"

                    colormap = charac_dict[maps_names[map_idx]]["colormap"]
                    colorbar_title = charac_dict[maps_names[map_idx]]["colorbar_title"]
                    cb_dict["colormap"] = colormap
                    cb_dict["map_name"] = maps_names[map_idx]
                    cb_dict["title"] = colorbar_title
                    cb_dict["vmin"] = colormap_limits[(map_idx, 0, 0)][0][0]
                    cb_dict["vmax"] = colormap_limits[(map_idx, 0, 0)][0][1]

                    for view_idx in range(n_views):
                        brain_positions[(map_idx, 0, view_idx)] = (
                            view_idx,
                            map_idx,
                        )

                    colorbar_list.append(cb_dict)

            else:

                colorbar_title = charac_dict["shared"]["colorbar_title"]
                colormap = charac_dict["shared"]["colormap"]

                cb_dict = {}
                cb_dict["colormap"] = colormap
                cb_dict["map_name"] = " + ".join(maps_names)
                cb_dict["vmin"] = colormap_limits[(0, 0, 0)][0][0]
                cb_dict["vmax"] = colormap_limits[(0, 0, 0)][0][1]
                cb_dict["title"] = colorbar_title

                for map_idx in range(n_maps):
                    for view_idx in range(n_views):
                        brain_positions[(map_idx, 0, view_idx)] = (view_idx, map_idx)

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


###############################################################################################
def single_map_multi_surface_layout(
    maps_dict,
    colormap_limits,
    charac_dict,
    colorbar,
    orientation,
    colorbar_style,
    colorbar_position,
    colorbar_size,
):
    """Handle single map, multiple objs2plot case."""

    if orientation == "horizontal":
        return horizontal_multi_surface_layout(
            maps_dict,
            colormap_limits,
            charac_dict,
            colorbar,
            colorbar_style,
            colorbar_position,
            colorbar_size,
        )
    elif orientation == "vertical":
        return vertical_multi_surface_layout(
            maps_dict,
            colormap_limits,
            charac_dict,
            colorbar,
            colorbar_style,
            colorbar_position,
            colorbar_size,
        )
    else:  # grid
        return grid_multi_surface_layout(
            maps_dict,
            colormap_limits,
            charac_dict,
            colorbar,
            colorbar_style,
            colorbar_position,
            colorbar_size,
        )


###############################################################################################
def horizontal_multi_surface_layout(
    maps_dict,
    colormap_limits,
    charac_dict,
    colorbar,
    colorbar_style,
    colorbar_position,
    colorbar_size,
):
    """Handle horizontal layout for multiple objs2plot."""
    brain_positions = {}
    dims = visutils.get_plot_config_dimensions(colormap_limits)
    n_objs2plot = dims[1]

    map_name = list(maps_dict.keys())[0]

    colormap = maps_dict[map_name]["colormap"]
    colorbar_title = maps_dict[map_name]["colorbar_title"]

    colorbar_list = []
    if not colorbar:
        shape = [1, n_objs2plot]
        row_weights = [1]
        col_weights = [1] * n_objs2plot
        groups = []

        for surf_idx in range(n_objs2plot):
            brain_positions[(0, surf_idx, 0)] = (0, surf_idx)

    else:
        if colorbar_style == "individual":
            groups = []
            # Define layout parameters outside the loop
            if colorbar_position == "right":
                shape = [1, n_objs2plot * 2]
                row_weights = [1]
                col_weights = [1, colorbar_size] * n_objs2plot
                cb_orientation = "vertical"

            else:  # bottom
                shape = [2, n_objs2plot]
                row_weights = [1, colorbar_size]
                col_weights = [1] * n_objs2plot
                cb_orientation = "horizontal"

            for surf_idx in range(n_objs2plot):
                # Calculate positions based on layout
                if colorbar_position == "right":
                    brain_positions[(0, surf_idx, 0)] = (0, surf_idx * 2)
                    cb_position = (0, surf_idx * 2 + 1)
                else:  # bottom
                    brain_positions[(0, surf_idx, 0)] = (0, surf_idx)
                    cb_position = (1, surf_idx)

                # Build colorbar dict in one go
                colorbar_list.append(
                    {
                        "vmin": colormap_limits[(0, surf_idx, 0)][0][0],
                        "vmax": colormap_limits[(0, surf_idx, 0)][0][1],
                        "colormap": colormap,
                        "title": colorbar_title,
                        "map_name": map_name,
                        "position": cb_position,
                        "orientation": cb_orientation,
                    }
                )

        else:  # shared colorbar
            # Configure layout based on colorbar position
            if colorbar_position == "right":
                shape = [1, n_objs2plot + 1]
                row_weights = [1]
                col_weights = [1] * n_objs2plot + [colorbar_size]
                groups = []
                cb_position = (0, n_objs2plot)
                cb_orientation = "vertical"
            else:  # bottom
                shape = [2, n_objs2plot]
                row_weights = [1, colorbar_size]
                col_weights = [1] * n_objs2plot
                groups = [(1, slice(0, n_objs2plot))]
                cb_position = (1, 0)
                cb_orientation = "horizontal"

            # Set brain positions
            for surf_idx in range(n_objs2plot):
                brain_positions[(0, surf_idx, 0)] = (0, surf_idx)

            # Build colorbar dict
            colorbar_list.append(
                {
                    "title": charac_dict["shared"]["colorbar_title"],
                    "colormap": colormap,
                    "map_name": map_name,
                    "vmin": colormap_limits[(0, 0, 0)][0][0],
                    "vmax": colormap_limits[(0, 0, 0)][0][1],
                    "position": cb_position,
                    "orientation": cb_orientation,
                }
            )

    layout_config = {
        "shape": shape,
        "row_weights": row_weights,
        "col_weights": col_weights,
        "groups": groups,
        "brain_positions": brain_positions,
        "colormap_limits": colormap_limits,
    }

    return layout_config, colorbar_list


###############################################################################################
def vertical_multi_surface_layout(
    maps_dict,
    colormap_limits,
    charac_dict,
    colorbar,
    colorbar_style,
    colorbar_position,
    colorbar_size,
):
    """Handle vertical layout for multiple objs2plot."""
    brain_positions = {}
    dims = visutils.get_plot_config_dimensions(colormap_limits)
    n_objs2plot = dims[1]

    map_name = list(maps_dict.keys())[0]
    colormap = maps_dict[map_name]["colormap"]
    colorbar_title = maps_dict[map_name]["colorbar_title"]

    colorbar_list = []

    if not colorbar:
        shape = [n_objs2plot, 1]
        row_weights = [1] * n_objs2plot
        col_weights = [1]
        groups = []

        for surf_idx in range(n_objs2plot):
            brain_positions[(0, surf_idx, 0)] = (surf_idx, 0)

    else:
        if colorbar_style == "individual":
            groups = []

            # Define layout parameters outside the loop
            if colorbar_position == "right":
                shape = [n_objs2plot, 2]
                row_weights = [1] * n_objs2plot
                col_weights = [1, colorbar_size]
                cb_orientation = "vertical"
            else:  # bottom
                shape = [n_objs2plot * 2, 1]
                row_weights = [1, colorbar_size] * n_objs2plot
                col_weights = [1]
                cb_orientation = "horizontal"

            for surf_idx in range(n_objs2plot):
                # Calculate positions based on layout
                if colorbar_position == "right":
                    brain_positions[(0, surf_idx, 0)] = (surf_idx, 0)
                    cb_position = (surf_idx, 1)
                else:  # bottom
                    brain_positions[(0, surf_idx, 0)] = (surf_idx * 2, 0)
                    cb_position = (surf_idx * 2 + 1, 0)

                # Build colorbar dict in one go
                colorbar_list.append(
                    {
                        "vmin": colormap_limits[(0, surf_idx, 0)][0][0],
                        "vmax": colormap_limits[(0, surf_idx, 0)][0][1],
                        "colormap": colormap,
                        "title": colorbar_title,
                        "map_name": map_name,
                        "position": cb_position,
                        "orientation": cb_orientation,
                    }
                )

        else:  # shared colorbar
            # Configure layout based on colorbar position
            if colorbar_position == "right":
                shape = [n_objs2plot, 2]
                row_weights = [1] * n_objs2plot
                col_weights = [1, colorbar_size]
                groups = [(slice(0, n_objs2plot), 1)]
                cb_position = (0, 1)
                cb_orientation = "vertical"
            else:  # bottom
                shape = [n_objs2plot + 1, 1]
                row_weights = [1] * n_objs2plot + [colorbar_size]
                col_weights = [1]
                groups = [(n_objs2plot, slice(0, 1))]
                cb_position = (n_objs2plot, 0)
                cb_orientation = "horizontal"

            # Set brain positions
            for surf_idx in range(n_objs2plot):
                brain_positions[(0, surf_idx, 0)] = (surf_idx, 0)

            # Build colorbar dict
            colorbar_list.append(
                {
                    "title": charac_dict["shared"]["colorbar_title"],
                    "colormap": colormap,
                    "map_name": map_name,
                    "vmin": colormap_limits[(0, 0, 0)][0][0],
                    "vmax": colormap_limits[(0, 0, 0)][0][1],
                    "position": cb_position,
                    "orientation": cb_orientation,
                }
            )

    layout_config = {
        "shape": shape,
        "row_weights": row_weights,
        "col_weights": col_weights,
        "groups": groups,
        "brain_positions": brain_positions,
        "colormap_limits": colormap_limits,
    }

    return layout_config, colorbar_list


###############################################################################################
def grid_multi_surface_layout(
    maps_dict,
    colormap_limits,
    charac_dict,
    colorbar,
    colorbar_style,
    colorbar_position,
    colorbar_size,
):
    """Handle grid layout for multiple objs2plot."""
    brain_positions = {}
    dims = visutils.get_plot_config_dimensions(colormap_limits)
    n_objs2plot = dims[1]

    optimal_grid, positions = cltplot.calculate_optimal_subplots_grid(n_objs2plot)

    map_name = list(maps_dict.keys())[0]
    colormap = maps_dict[map_name]["colormap"]
    colorbar_title = maps_dict[map_name]["colorbar_title"]

    colorbar_list = []

    if not colorbar:
        shape = optimal_grid
        row_weights = [1] * optimal_grid[0]
        col_weights = [1] * optimal_grid[1]
        groups = []

        for surf_idx in range(n_objs2plot):
            brain_positions[(0, surf_idx, 0)] = positions[surf_idx]

    else:
        if colorbar_style == "individual":
            groups = []

            # Define layout parameters outside the loop
            if colorbar_position == "right":
                shape = [optimal_grid[0], optimal_grid[1] * 2]
                row_weights = [1] * optimal_grid[0]
                col_weights = [1, colorbar_size] * optimal_grid[1]
                cb_orientation = "vertical"
            else:  # bottom
                shape = [optimal_grid[0] * 2, optimal_grid[1]]
                row_weights = [1, colorbar_size] * optimal_grid[0]
                col_weights = [1] * optimal_grid[1]
                cb_orientation = "horizontal"

            for surf_idx in range(n_objs2plot):
                pos = positions[surf_idx]

                # Calculate positions based on layout
                if colorbar_position == "right":
                    brain_positions[(0, surf_idx, 0)] = (pos[0], pos[1] * 2)
                    cb_position = (pos[0], pos[1] * 2 + 1)
                else:  # bottom
                    brain_positions[(0, surf_idx, 0)] = (pos[0] * 2, pos[1])
                    cb_position = (pos[0] * 2 + 1, pos[1])

                # Build colorbar dict in one go
                colorbar_list.append(
                    {
                        "vmin": colormap_limits[(0, surf_idx, 0)][0][0],
                        "vmax": colormap_limits[(0, surf_idx, 0)][0][1],
                        "colormap": colormap,
                        "title": colorbar_title,
                        "map_name": map_name,
                        "position": cb_position,
                        "orientation": cb_orientation,
                    }
                )

        else:  # shared colorbar
            # Configure layout based on colorbar position
            if colorbar_position == "right":
                shape = [optimal_grid[0], optimal_grid[1] + 1]
                row_weights = [1] * optimal_grid[0]
                col_weights = [1] * optimal_grid[1] + [colorbar_size]
                groups = [(slice(0, optimal_grid[0]), optimal_grid[1])]
                cb_position = (0, optimal_grid[1])
                cb_orientation = "vertical"
            else:  # bottom
                shape = [optimal_grid[0] + 1, optimal_grid[1]]
                row_weights = [1] * optimal_grid[0] + [colorbar_size]
                col_weights = [1] * optimal_grid[1]
                groups = [(optimal_grid[0], slice(0, optimal_grid[1]))]
                cb_position = (optimal_grid[0], 0)
                cb_orientation = "horizontal"

            # Set brain positions
            for surf_idx in range(n_objs2plot):
                brain_positions[(0, surf_idx, 0)] = positions[surf_idx]

            # Build colorbar dict
            colorbar_list.append(
                {
                    "title": charac_dict["shared"]["colorbar_title"],
                    "colormap": colormap,
                    "map_name": map_name,
                    "vmin": colormap_limits[(0, 0, 0)][0][0],
                    "vmax": colormap_limits[(0, 0, 0)][0][1],
                    "position": cb_position,
                    "orientation": cb_orientation,
                }
            )

    layout_config = {
        "shape": shape,
        "row_weights": row_weights,
        "col_weights": col_weights,
        "groups": groups,
        "brain_positions": brain_positions,
        "colormap_limits": colormap_limits,
    }

    return layout_config, colorbar_list


###############################################################################################
def grid_multi_views_layout(
    maps_dict,
    colormap_limits,
    charac_dict,
    valid_views,
    colorbar,
    colorbar_position,
    colorbar_size,
):
    """Handle multiple views, single map, single surface case."""

    n_views = len(valid_views)
    brain_positions = {}
    colorbar_list = []

    brain_positions = {}
    map_name = list(maps_dict.keys())[0]

    colormap = charac_dict[map_name]["colormap"]
    colorbar_title = charac_dict[map_name]["colorbar_title"]
    groups = []

    """Handle grid layout for multiple objs2plot."""
    optimal_grid, positions = cltplot.calculate_optimal_subplots_grid(n_views)
    for view_idx in range(n_views):
        pos = positions[view_idx]
        brain_positions[(0, 0, view_idx)] = pos

    if not colorbar:
        shape = list(optimal_grid)
        row_weights = [1] * optimal_grid[0]
        col_weights = [1] * optimal_grid[1]
    else:
        shape = [optimal_grid[0], optimal_grid[1] + 1]
        row_weights = [1] * optimal_grid[0]
        col_weights = [1] * optimal_grid[1] + [colorbar_size]
        groups = [(slice(0, optimal_grid[0]), optimal_grid[1])]

        cb_dict = {}
        cb_dict["colormap"] = colormap
        cb_dict["map_name"] = map_name
        cb_dict["vmin"] = colormap_limits[(0, 0, 0)][0][0]
        cb_dict["vmax"] = colormap_limits[(0, 0, 0)][0][1]
        cb_dict["title"] = colorbar_title

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


###############################################################################################
def scene_layout(
    valid_views,
    colorbar,
    colorbar_position,
    colorbar_size,
):
    """
    Handle scene layout for single map, single object,
    multiple views case.

    Parameters
    ----------
    valid_views : list
        List of valid views to plot.

    colorbar : bool
        Whether to include a colorbar in the layout.

    colorbar_position : str
        Position of the colorbar ('right' or 'bottom').

    colorbar_size : int
        Size of the colorbar.

    Returns
    -------
    layout_config : dict
        Configuration dictionary for the layout.


    """

    n_views = len(valid_views)
    brain_positions = {}

    groups = []

    """Handle grid layout for multiple objs2plot."""
    optimal_grid, positions = cltplot.calculate_optimal_subplots_grid(n_views)
    for view_idx in range(n_views):
        pos = positions[view_idx]
        brain_positions[(0, 0, view_idx)] = pos

    if not colorbar:
        shape = list(optimal_grid)
        row_weights = [1] * optimal_grid[0]
        col_weights = [1] * optimal_grid[1]
    else:
        shape = [optimal_grid[0], optimal_grid[1] + 1]
        row_weights = [1] * optimal_grid[0]
        col_weights = [1] * optimal_grid[1] + [colorbar_size]
        groups = [(slice(0, optimal_grid[0]), optimal_grid[1])]

        if colorbar_position == "right":
            shape = [optimal_grid[0], optimal_grid[1] + 1]
            row_weights = [1] * optimal_grid[0]
            col_weights = [1] * optimal_grid[1] + [colorbar_size]
            groups = [(slice(0, optimal_grid[0]), optimal_grid[1])]

        else:  # bottom
            shape = [optimal_grid[0] + 1, optimal_grid[1]]
            row_weights = [1] * optimal_grid[0] + [colorbar_size]
            col_weights = [1] * optimal_grid[1]
            groups = [(optimal_grid[0], slice(0, optimal_grid[1]))]

    layout_config = {
        "shape": shape,
        "row_weights": row_weights,
        "col_weights": col_weights,
        "groups": groups,
        "brain_positions": brain_positions,
    }

    return layout_config
