"""
Visualization tools for brain surface plotting and layout configuration.

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
    maps_names,
    objs2plot,
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
    n_surfaces = len(objs2plot)
    colorbar_size = plotobj.figure_conf["colorbar_size"]

    if n_views == 1 and n_maps == 1 and n_surfaces == 1:  # Works fine
        # Check if maps_names[0] is present in the surface
        if colormap_style not in ["individual", "shared"]:
            colormap_style = "individual"

        if maps_names[0] in list(objs2plot[0].colortables.keys()):
            colorbar = False

        return single_element_layout(
            objs2plot,
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

        # Check if maps_names[0] is present in ALL objs2plot
        if all(
            maps_names[0] in objs2plot[i].colortables.keys() for i in range(n_surfaces)
        ):
            colorbar = False

        return single_map_multi_surface_layout(
            objs2plot,
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

        return multi_map_single_surface_layout(
            objs2plot,
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
            for surface in objs2plot
        )
        if colorbar_data == False:
            colorbar = False

        return multi_map_multi_surface_layout(
            objs2plot,
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
            maps_names[0] in objs2plot[i].colortables.keys() for i in range(n_surfaces)
        ):
            colorbar = False
        return multi_view_single_element_layout(
            objs2plot[0],
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
            maps_names[0] in objs2plot[i].colortables.keys() for i in range(n_surfaces)
        ):
            colorbar = False
        return multi_view_multi_surface_layout(
            objs2plot,
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
            map_name not in objs2plot[0].colortables for map_name in maps_names
        )
        if colorbar_data == False:
            colorbar = False

        return multi_view_multi_map_layout(
            objs2plot,
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


def single_element_layout(
    objs2plot,
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
        objs2plot=objs2plot,
        map_name=maps_names[0],
        colormap_style="individual",
        v_limits=v_limits[0],
    )
    colormap_limits[(0, 0, 0)] = limits_list[0]

    colorbar_list = []

    if maps_names[0] in objs2plot[0].colortables:
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


def multi_map_single_surface_layout(
    objs2plot,
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
        return horizontal_multi_map_layout(
            objs2plot,
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
        return vertical_multi_map_layout(
            objs2plot,
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
        return grid_multi_map_layout(
            objs2plot,
            maps_names,
            v_limits,
            colormaps,
            colorbar_titles,
            colorbar,
            colormap_style,
            colorbar_position,
            colorbar_size,
        )


def horizontal_multi_map_layout(
    objs2plot,
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
            objs2plot=objs2plot,
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
                        objs2plot=objs2plot,
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
                        objs2plot=objs2plot,
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
                    objs2plot=objs2plot,
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


def vertical_multi_map_layout(
    objs2plot,
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
            objs2plot=objs2plot,
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
                        objs2plot=objs2plot,
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
                        objs2plot=objs2plot,
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
                    objs2plot=objs2plot,
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


def grid_multi_map_layout(
    objs2plot,
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
            objs2plot=objs2plot,
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
                        objs2plot=objs2plot,
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
                        objs2plot=objs2plot,
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
                    objs2plot=objs2plot,
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
# Multiple objs2plot and multiple maps cases
################################################################################
def multi_map_multi_surface_layout(
    objs2plot,
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
    """Handle multiple maps and multiple objs2plot case."""

    n_maps = len(maps_names)
    n_surfaces = len(objs2plot)
    brain_positions = {}
    colormap_limits = {}
    colorbar_list = []
    if orientation == "horizontal":

        if not colorbar:
            shape = [n_surfaces, n_maps]
            row_weights = [1] * n_surfaces
            col_weights = [1] * n_maps
            groups = []

            # Maps in columns, objs2plot in rows
            for map_idx in range(n_maps):
                for surf_idx in range(n_surfaces):
                    brain_positions[(map_idx, surf_idx, 0)] = (surf_idx, map_idx)
                    map_limits = visutils.get_map_limits(
                        objs2plot=objs2plot[surf_idx],
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
                                objs2plot=objs2plot[surf_idx],
                                map_name=maps_names[map_idx],
                                colormap_style="individual",
                                v_limits=v_limits[map_idx],
                            )
                            cb_dict["vmin"] = limits_list[0][0]
                            cb_dict["vmax"] = limits_list[0][1]
                            colormap_limits[(map_idx, surf_idx, 0)] = limits_list[0]

                            if (
                                maps_names[map_idx]
                                not in objs2plot[surf_idx].colortables
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
                                objs2plot=objs2plot[surf_idx],
                                map_name=maps_names[map_idx],
                                colormap_style="individual",
                                v_limits=v_limits[map_idx],
                            )
                            cb_dict["vmin"] = limits_list[0][0]
                            cb_dict["vmax"] = limits_list[0][1]
                            colormap_limits[(map_idx, surf_idx, 0)] = limits_list[0]

                            if (
                                maps_names[map_idx]
                                not in objs2plot[surf_idx].colortables
                            ):
                                colorbar_list.append(cb_dict)

            else:

                # Map-wise limits
                maps_limits = []
                for map_idx in range(n_maps):
                    if maps_names[map_idx] not in objs2plot[0].colortables:
                        map_limits = visutils.get_map_limits(
                            objs2plot=objs2plot,
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
                            colormap_limits[(map_idx, surf_idx, 0)] = global_limits + (
                                maps_names[0],
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
                        if not maps_names[map_idx] in objs2plot[0].colortables:
                            map_limits = maps_limits[map_idx]

                            for surf_idx in range(n_surfaces):
                                brain_positions[(map_idx, surf_idx, 0)] = (
                                    surf_idx,
                                    map_idx,
                                )
                                colormap_limits[(map_idx, surf_idx, 0)] = maps_limits[
                                    map_idx
                                ]
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

                        if maps_names[map_idx] not in objs2plot[0].colortables:
                            cb_dict["vmin"] = maps_limits[map_idx][0]
                            cb_dict["vmax"] = maps_limits[map_idx][1]
                            colorbar_list.append(cb_dict)

    else:  # vertical

        if not colorbar:
            # Maps in rows, objs2plot in columns
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
                            objs2plot=objs2plot[surf_idx],
                            map_name=maps_names[map_idx],
                            colormap_style="individual",
                            v_limits=v_limits[map_idx],
                        )
                        cb_dict["vmin"] = limits_list[0][0]
                        cb_dict["vmax"] = limits_list[0][1]
                        colormap_limits[(map_idx, surf_idx, 0)] = limits_list[0]
                        if maps_names[map_idx] not in objs2plot[surf_idx].colortables:
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
                            objs2plot=objs2plot[surf_idx],
                            map_name=maps_names[map_idx],
                            colormap_style="individual",
                            v_limits=v_limits[map_idx],
                        )
                        cb_dict["vmin"] = limits_list[0][0]
                        cb_dict["vmax"] = limits_list[0][1]
                        colormap_limits[(map_idx, surf_idx, 0)] = limits_list[0]
                        if maps_names[map_idx] not in objs2plot[surf_idx].colortables:
                            colorbar_list.append(cb_dict)

        else:
            # Map-wise limits
            maps_limits = []
            for map_idx in range(n_maps):
                if any(
                    maps_names[map_idx] not in surface.colortables
                    for surface in objs2plot
                ):
                    map_limits = visutils.get_map_limits(
                        objs2plot=objs2plot,
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
                    for surface in objs2plot
                ):
                    colorbar_list.append(cb_dict)

            elif colormap_style == "shared_by_map":
                ######### One colorbar per map #########
                for map_idx in range(n_maps):
                    if maps_names[map_idx] not in objs2plot[0].colortables:
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
                    if maps_names[map_idx] not in objs2plot[0].colortables:
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


def multi_view_single_element_layout(
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


def multi_view_multi_surface_layout(
    objs2plot,
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
    """Handle multiple views and multiple objs2plot case."""
    n_surfaces = len(objs2plot)
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

            # Views in columns, objs2plot in rows
            for view_idx in range(n_views):
                for surf_idx in range(n_surfaces):
                    brain_positions[(0, surf_idx, view_idx)] = (surf_idx, view_idx)
                    map_limits = visutils.get_map_limits(
                        objs2plot=objs2plot[surf_idx],
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
                        objs2plot=objs2plot[surf_idx],
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
                        objs2plot=objs2plot,
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
                cb_dict["colormap"] = colormaps[0] if 0 < len(colormaps) else "viridis"
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
            # Views in rows, objs2plot in columns
            shape = [n_views, n_surfaces]
            row_weights = [1] * n_views
            col_weights = [1] * n_surfaces
            groups = []

            for view_idx in range(n_views):
                for surf_idx in range(n_surfaces):
                    brain_positions[(0, surf_idx, view_idx)] = (view_idx, surf_idx)
                    map_limits = visutils.get_map_limits(
                        objs2plot=objs2plot[surf_idx],
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
                        objs2plot=objs2plot[surf_idx],
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
                        objs2plot=objs2plot,
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
                cb_dict["colormap"] = colormaps[0] if 0 < len(colormaps) else "viridis"
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


def multi_view_multi_map_layout(
    objs2plot,
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
                        objs2plot=objs2plot[0],
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
                        objs2plot=objs2plot[0],
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
                    if maps_names[map_idx] not in objs2plot[0].colortables:
                        colorbar_list.append(cb_dict)

            else:
                # Get the global limits
                maps_limits = []
                for map_idx in range(n_maps):
                    if maps_names[map_idx] not in objs2plot[0].colortables:
                        map_limits = visutils.get_map_limits(
                            objs2plot=objs2plot,
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
                cb_dict["colormap"] = colormaps[0] if 0 < len(colormaps) else "viridis"
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
                        objs2plot=objs2plot[0],
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
                        objs2plot=objs2plot[0],
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

                    if maps_names[map_idx] not in objs2plot[0].colortables:
                        colorbar_list.append(cb_dict)
            else:
                # Get the global limits
                maps_limits = []
                for map_idx in range(n_maps):
                    if maps_names[map_idx] not in objs2plot[0].colortables:
                        map_limits = visutils.get_map_limits(
                            objs2plot=objs2plot,
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
                cb_dict["colormap"] = colormaps[0] if 0 < len(colormaps) else "viridis"
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


def single_map_multi_surface_layout(
    objs2plot,
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
    """Handle single map, multiple objs2plot case."""
    brain_positions = {}
    n_surfaces = len(objs2plot)

    # Getting the limits for each surface and storing them in a list
    limits_list = visutils.get_map_limits(
        objs2plot=objs2plot,
        map_name=maps_names[0],
        colormap_style=colormap_style,
        v_limits=v_limits[0],
    )

    if orientation == "horizontal":
        return horizontal_multi_surface_layout(
            objs2plot,
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
        return vertical_multi_surface_layout(
            objs2plot,
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
        return grid_multi_surface_layout(
            objs2plot,
            maps_names,
            limits_list,
            colormaps,
            colorbar_titles,
            colorbar,
            colormap_style,
            colorbar_position,
            colorbar_size,
        )


def horizontal_multi_surface_layout(
    objs2plot,
    maps_names,
    limits_list,
    colormaps,
    colorbar_titles,
    colorbar,
    colormap_style,
    colorbar_position,
    colorbar_size,
):
    """Handle horizontal layout for multiple objs2plot."""
    brain_positions = {}
    colormap_limits = {}

    n_surfaces = len(objs2plot)
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
                if maps_names[0] in objs2plot[surf_idx].colortables:
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


def vertical_multi_surface_layout(
    objs2plot,
    maps_names,
    limits_list,
    colormaps,
    colorbar_titles,
    colorbar,
    colormap_style,
    colorbar_position,
    colorbar_size,
):
    """Handle vertical layout for multiple objs2plot."""
    brain_positions = {}
    colormap_limits = {}

    n_surfaces = len(objs2plot)

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


def grid_multi_surface_layout(
    objs2plot,
    maps_names,
    limits_list,
    colormaps,
    colorbar_titles,
    colorbar,
    colormap_style,
    colorbar_position,
    colorbar_size,
):
    """Handle grid layout for multiple objs2plot."""

    n_surfaces = len(objs2plot)
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
