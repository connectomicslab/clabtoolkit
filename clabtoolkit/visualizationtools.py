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
import numpy as np
import nibabel as nib
from typing import Union, List, Optional, Tuple, Dict, Any, TYPE_CHECKING
from nilearn import plotting
import pyvista as pv

# Importing local modules
from . import freesurfertools as cltfree
from . import misctools as cltmisc

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
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

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file '{self.config_file}' not found"
            )
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in configuration file: {e}")

    ##########################################################################################################
    def _create_dynamic_config(
        self, selected_views: List[str], reference_view: str = "8_views"
    ) -> Dict[str, Any]:
        """
        Create a dynamic configuration based on selected view names.

        Parameters
        ----------
        selected_views : List[str]
            List of view names to include.

        reference_view : str, default "8_views"
            Base configuration to use as reference.

        Returns
        -------
        Dict[str, Any]
            Dynamic configuration with filtered views and optimal layout.

        Raises
        ------
        ValueError
            If no valid views are found or invalid view names provided.

        Examples
        --------
        >>> config = plotter._create_dynamic_config(["lateral", "medial"])
        >>> print(config["shape"])  # [1, 2]
        >>>
        >>> print(len(config["views"]))  # 4 (LH/RH lateral + LH/RH medial)
        """

        # Load base 8_views configuration
        if "8_views" not in self.views_conf:
            raise ValueError("Base '8_views' configuration not found in config file")

        base_config = self.views_conf[reference_view]
        base_views = base_config["views"]

        # Normalize input view names (case-insensitive)
        selected_views_lower = [view.lower() for view in selected_views]

        # Find matching views from base configuration
        filtered_views = []
        for view_name in selected_views_lower:
            if view_name not in self._view_name_mapping:
                available_views = list(self._view_name_mapping.keys())
                raise ValueError(
                    f"Invalid view name '{view_name}'. "
                    f"Available options: {available_views}"
                )

            # Find matching view configurations
            target_titles = self._view_name_mapping[view_name]
            for base_view in base_views:
                if base_view["title"] in target_titles:
                    filtered_views.append(base_view.copy())

        if not filtered_views:
            raise ValueError("No valid views found for the provided selection")

        # Calculate optimal grid layout
        num_views = len(filtered_views)
        optimal_shape = self._calculate_optimal_grid(num_views)

        # Reassign subplot positions in optimal grid
        for i, view in enumerate(filtered_views):
            row = i // optimal_shape[1]
            col = i % optimal_shape[1]
            view["subplot"] = [row, col]

        # Calculate window size based on shape and aspect ratio
        base_window_size = base_config["window_size"]
        aspect_ratio = base_window_size[0] / base_window_size[1]

        # Maintain reasonable window sizing
        if optimal_shape[1] > optimal_shape[0]:  # Wider than tall
            new_width = min(1800, 450 * optimal_shape[1])
            new_height = int(
                new_width / aspect_ratio * optimal_shape[0] / optimal_shape[1]
            )
        else:  # Taller than wide
            new_height = min(1200, 400 * optimal_shape[0])
            new_width = int(
                new_height * aspect_ratio * optimal_shape[1] / optimal_shape[0]
            )

        # Create dynamic configuration
        dynamic_config = {
            "shape": optimal_shape,
            "window_size": [new_width, new_height],
            "views": filtered_views,
        }

        return dynamic_config

    ################################################################################################
    def _create_single_hemisphere_config(
        self, selected_views: List[str], hemi: str
    ) -> Dict[str, Any]:
        """
        Create a dynamic configuration for single hemisphere based on selected view names.

        Parameters
        ----------
        selected_views : List[str]
            List of view names to include.

        hemi : str
            Hemisphere specification: "lh" or "rh".

        Returns
        -------
        Dict[str, Any]
            Dynamic configuration with filtered views and optimal layout.

        Raises
        ------
        ValueError
            If no valid views are found or invalid view names provided.

        Examples
        --------
        >>> config = plotter._create_single_hemisphere_config(["lateral", "medial"], "lh")
        >>> print(config["shape"])  # [1, 2]
        >>>
        >>> print(len(config["views"]))  # 2 (LH lateral + LH medial)
        """

        # Load base 8_views configuration
        if "8_views" not in self.views_conf:
            raise ValueError("Base '8_views' configuration not found in config file")

        base_config = self.views_conf["8_views"]
        base_views = base_config["views"]

        # Normalize input view names (case-insensitive)
        selected_views_lower = [view.lower() for view in selected_views]

        # Find matching views from base configuration for specified hemisphere
        filtered_views = []
        for view_name in selected_views_lower:
            if view_name not in self._view_name_mapping:
                available_views = list(self._view_name_mapping.keys())
                raise ValueError(
                    f"Invalid view name '{view_name}'. "
                    f"Available options: {available_views}"
                )

            # Find matching view configurations for the specified hemisphere
            target_titles = self._view_name_mapping[view_name]
            for base_view in base_views:
                # Filter views based on hemisphere
                title = base_view["title"]
                if view_name in ["lateral", "medial"]:
                    # Hemisphere-specific views
                    hemi_prefix = f"{hemi.upper()}:"
                    if title in target_titles and title.startswith(hemi_prefix):
                        view_copy = base_view.copy()
                        view_copy["mesh"] = hemi  # Set mesh to hemisphere
                        filtered_views.append(view_copy)
                else:
                    # Global views (dorsal, ventral, rostral, caudal)
                    if title in target_titles:
                        view_copy = base_view.copy()
                        view_copy["mesh"] = hemi  # Set mesh to hemisphere
                        filtered_views.append(view_copy)

        if not filtered_views:
            raise ValueError("No valid views found for the provided selection")

        # Calculate optimal grid layout
        num_views = len(filtered_views)
        optimal_shape = self._calculate_optimal_grid(num_views)

        # Reassign subplot positions in optimal grid
        for i, view in enumerate(filtered_views):
            row = i // optimal_shape[1]
            col = i % optimal_shape[1]
            view["subplot"] = [row, col]

        # Keep the same window size as the base configuration
        base_window_size = base_config["window_size"]

        # Create dynamic configuration
        dynamic_config = {
            "shape": optimal_shape,
            "window_size": base_window_size,
            "views": filtered_views,
        }

        return dynamic_config

    ###############################################################################################
    def _filter_config_for_hemisphere(
        self, config: Dict[str, Any], hemi: str
    ) -> Dict[str, Any]:
        """
        Filter a predefined configuration to show only views relevant to a single hemisphere.

        Parameters
        ----------
        config : Dict[str, Any]
            Original configuration with all views.

        hemi : str
            Hemisphere specification: "lh" or "rh".

        Returns
        -------
        Dict[str, Any]
            Filtered configuration for single hemisphere.

        Raises
        ------
        ValueError
            If no valid views are found for the hemisphere.

        Examples
        --------
        >>> filtered = plotter._filter_config_for_hemisphere(config_8_views, "lh")
        >>> print(len(filtered["views"]))  # Reduced number of views for LH only
        """

        # Filter views for the specified hemisphere
        filtered_views = []

        for view in config["views"]:
            title = view["title"]
            view_copy = view.copy()

            # Check if view is hemisphere-specific or global
            if "LH:" in title or "RH:" in title:
                # Hemisphere-specific view
                hemi_prefix = f"{hemi.upper()}:"
                if title.startswith(hemi_prefix):
                    view_copy["mesh"] = hemi
                    filtered_views.append(view_copy)
            else:
                # Global view (dorsal, ventral, rostral, caudal)
                view_copy["mesh"] = hemi
                filtered_views.append(view_copy)

        if not filtered_views:
            raise ValueError(f"No valid views found for hemisphere '{hemi}'")

        # Recalculate optimal layout for filtered views
        num_views = len(filtered_views)

        # If the views are not row or column aligned, we need to adjust the shape
        if config["shape"][0] == 1 or config["shape"][1] == 1:
            # Ensure at least 2 views per row/column for better visualization

            if config["shape"][0] == 1:
                optimal_shape = (1, num_views)
            else:
                optimal_shape = (num_views, 1)

            if config["shape"][0] == 1:
                for i, view in enumerate(filtered_views):
                    view["subplot"] = [0, i]
            elif config["shape"][0] == 1:
                for i, view in enumerate(filtered_views):
                    view["subplot"] = [i, 0]
        else:

            optimal_shape = self._calculate_optimal_grid(num_views)
            # Reassign subplot positions
            for i, view in enumerate(filtered_views):
                row = i // optimal_shape[1]
                col = i % optimal_shape[1]
                view["subplot"] = [row, col]

        # Keep the same window size as the original configuration
        base_window_size = config["window_size"]

        # Create filtered configuration
        filtered_config = {
            "shape": optimal_shape,
            "window_size": base_window_size,
            "views": filtered_views,
        }

        return filtered_config

    ###############################################################################################
    def _calculate_optimal_grid(self, num_views: int) -> List[int]:
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
        >>> plotter._calculate_optimal_grid(4)
        [2, 2]
        >>>
        >>> plotter._calculate_optimal_grid(6)
        [2, 3]
        >>>
        >>> plotter._calculate_optimal_grid(1)
        [1, 1]
        """

        # Calculate optimal grid dimensions based on number of views
        if num_views == 1:
            return [1, 1]
        elif num_views == 2:
            return [1, 2]
        elif num_views == 3:
            return [1, 3]
        elif num_views == 4:
            return [2, 2]
        elif num_views <= 6:
            return [2, 3]
        elif num_views <= 8:
            return [2, 4]
        else:
            # For more than 8 views, try to keep roughly square
            cols = math.ceil(math.sqrt(num_views))
            rows = math.ceil(num_views / cols)
            return [rows, cols]

    ###############################################################################################
    def _calculate_single_view_multi_map_layout(
        self, n_maps: int, colorbar: bool, colorbar_style: str, colorbar_position: str
    ) -> Tuple[
        List[int],
        List[float],
        List[float],
        List[Tuple],
        Dict[Tuple[int, int], Tuple[int, int]],
        Dict[str, Tuple[int, int]],
    ]:
        """
        Calculate grid layout for single view with multiple maps.

        Creates a compact grid layout where maps are distributed across rows
        and columns rather than being arranged in a single row or column.

        Parameters
        ----------
        n_maps : int
            Number of maps to display.

        colorbar : bool
            Whether to show colorbars.

        colorbar_style : str
            Colorbar style: "individual" or "shared".

        colorbar_position : str
            Colorbar position: "right" or "bottom".

        Returns
        -------
        Tuple
            Grid layout information including shape, weights, and positions.
        """

        brain_positions = {}
        colorbar_positions = {}
        groups = []

        # Calculate optimal grid shape for maps
        map_grid_shape = self._calculate_optimal_grid(n_maps)
        map_rows, map_cols = map_grid_shape

        if not colorbar:
            # Simple grid layout without colorbars
            grid_shape = map_grid_shape
            row_weights = [1] * map_rows
            col_weights = [1] * map_cols

            # Position maps in grid
            for map_idx in range(n_maps):
                row = map_idx // map_cols
                col = map_idx % map_cols
                brain_positions[(map_idx, 0)] = (row, col)  # view_idx is always 0

        else:
            if colorbar_style == "individual":
                if colorbar_position == "right":
                    # Each map gets a colorbar to its right: [map_rows, map_cols * 2]
                    grid_shape = [map_rows, map_cols * 2]
                    row_weights = [1] * map_rows
                    col_weights = []
                    for _ in range(map_cols):
                        col_weights.extend([1, 0.2])  # map column, colorbar column

                    # Position maps and colorbars
                    for map_idx in range(n_maps):
                        row = map_idx // map_cols
                        col = map_idx % map_cols
                        # Map position
                        brain_positions[(map_idx, 0)] = (row, col * 2)
                        # Colorbar position (next to map)
                        colorbar_positions[f"individual_{map_idx}"] = (row, col * 2 + 1)

                else:  # bottom
                    # Each map gets a colorbar below it: [map_rows * 2, map_cols]
                    grid_shape = [map_rows * 2, map_cols]
                    row_weights = []
                    for _ in range(map_rows):
                        row_weights.extend([1, 0.2])  # map row, colorbar row
                    col_weights = [1] * map_cols

                    # Position maps and colorbars
                    for map_idx in range(n_maps):
                        row = map_idx // map_cols
                        col = map_idx % map_cols
                        # Map position
                        brain_positions[(map_idx, 0)] = (row * 2, col)
                        # Colorbar position (below map)
                        colorbar_positions[f"individual_{map_idx}"] = (row * 2 + 1, col)

            else:  # shared colorbar
                if colorbar_position == "right":
                    # Add one extra column for shared colorbar
                    grid_shape = [map_rows, map_cols + 1]
                    row_weights = [1] * map_rows
                    col_weights = [1] * map_cols + [0.2]
                    groups = [(slice(None), map_cols)]  # Span all rows in last column

                    # Position maps
                    for map_idx in range(n_maps):
                        row = map_idx // map_cols
                        col = map_idx % map_cols
                        brain_positions[(map_idx, 0)] = (row, col)

                    # Shared colorbar in last column
                    colorbar_positions["shared"] = (0, map_cols)

                else:  # bottom
                    # Add one extra row for shared colorbar
                    grid_shape = [map_rows + 1, map_cols]
                    row_weights = [1] * map_rows + [0.2]
                    col_weights = [1] * map_cols
                    groups = [(map_rows, slice(None))]  # Span all columns in last row

                    # Position maps
                    for map_idx in range(n_maps):
                        row = map_idx // map_cols
                        col = map_idx % map_cols
                        brain_positions[(map_idx, 0)] = (row, col)

                    # Shared colorbar in last row
                    colorbar_positions["shared"] = (map_rows, 0)

        return (
            grid_shape,
            row_weights,
            col_weights,
            groups,
            brain_positions,
            colorbar_positions,
        )

    ###############################################################################################
    def _calculate_multi_map_layout(
        self,
        n_maps: int,
        n_views: int,
        views_orientation: str,
        colorbar: bool,
        colorbar_style: str,
        colorbar_position: str,
    ) -> Tuple[
        List[int],
        List[float],
        List[float],
        List[Tuple],
        Dict[Tuple[int, int], Tuple[int, int]],
        Dict[str, Tuple[int, int]],
    ]:
        """
        Calculate the complete grid layout for multi-map visualization with colorbars.

        Parameters
        ----------
        n_maps : int
            Number of maps to display.

        n_views : int
            Number of views per map.

        views_orientation : str
            Layout orientation: "horizontal" or "vertical".

        colorbar : bool
            Whether to show colorbars.

        colorbar_style : str
            Colorbar style: "individual" or "shared".

        colorbar_position : str
            Colorbar position: "right" or "bottom".

        Returns
        -------
        Tuple
            Grid layout information including shape, weights, groups, and positions.
        """

        brain_positions = {}
        colorbar_positions = {}
        groups = []

        # Auto-correct invalid combinations for individual colorbars
        original_position = colorbar_position
        if colorbar and colorbar_style == "individual":
            if views_orientation == "horizontal" and colorbar_position != "right":
                print(
                    f"Warning: Individual colorbars with horizontal orientation only support 'right' position. Changing from '{colorbar_position}' to 'right'."
                )
                colorbar_position = "right"
            elif views_orientation == "vertical" and colorbar_position != "bottom":
                print(
                    f"Warning: Individual colorbars with vertical orientation only support 'bottom' position. Changing from '{colorbar_position}' to 'bottom'."
                )
                colorbar_position = "bottom"

        # MANDATORY: Auto-correct problematic combinations for multi-view individual colorbars
        if colorbar and colorbar_style == "individual" and n_views > 1:
            if views_orientation == "horizontal" and original_position == "bottom":
                print(
                    f"🔧 FORCING: Horizontal orientation with {n_views} views requires RIGHT position. Changing from 'bottom' to 'right'."
                )
                colorbar_position = "right"
            elif views_orientation == "vertical" and original_position == "right":
                print(
                    f"🔧 FORCING: Vertical orientation with {n_views} views requires BOTTOM position. Changing from 'right' to 'bottom'."
                )
                colorbar_position = "bottom"

        if not colorbar:
            # Simple case: no colorbars
            if views_orientation == "horizontal":
                # Maps as rows, views as columns
                grid_shape = [n_maps, n_views]
                row_weights = [1] * n_maps
                col_weights = [1] * n_views

                # Fill brain positions
                for map_idx in range(n_maps):
                    for view_idx in range(n_views):
                        brain_positions[(map_idx, view_idx)] = (map_idx, view_idx)
            else:
                # Views as rows, maps as columns
                grid_shape = [n_views, n_maps]
                row_weights = [1] * n_views
                col_weights = [1] * n_maps

                # Fill brain positions
                for map_idx in range(n_maps):
                    for view_idx in range(n_views):
                        brain_positions[(map_idx, view_idx)] = (view_idx, map_idx)

        else:
            # With colorbars
            if views_orientation == "horizontal":
                # Maps as rows, views as columns
                if colorbar_style == "individual":
                    # Individual colorbars: only RIGHT position makes sense
                    # Add 1 extra column for individual colorbars
                    grid_shape = [n_maps, n_views + 1]
                    row_weights = [1] * n_maps
                    col_weights = [1] * n_views + [
                        0.2
                    ]  # Give colorbars a bit more space

                    # Brain positions (unchanged)
                    for map_idx in range(n_maps):
                        for view_idx in range(n_views):
                            brain_positions[(map_idx, view_idx)] = (map_idx, view_idx)
                        # Each map gets its colorbar in the extra column at its row
                        colorbar_positions[f"individual_{map_idx}"] = (map_idx, n_views)

                else:  # shared
                    if colorbar_position == "right":
                        # Add 1 extra column for shared colorbar
                        grid_shape = [n_maps, n_views + 1]
                        row_weights = [1] * n_maps
                        col_weights = [1] * n_views + [0.2]
                        groups = [
                            (slice(None), n_views)
                        ]  # Span all rows in last column

                        # Brain positions
                        for map_idx in range(n_maps):
                            for view_idx in range(n_views):
                                brain_positions[(map_idx, view_idx)] = (
                                    map_idx,
                                    view_idx,
                                )
                        # Shared colorbar spans the entire extra column
                        colorbar_positions["shared"] = (0, n_views)

                    else:  # bottom
                        # Add 1 extra row for shared colorbar
                        grid_shape = [n_maps + 1, n_views]
                        row_weights = [1] * n_maps + [0.2]
                        col_weights = [1] * n_views
                        groups = [(n_maps, slice(None))]  # Span all columns in last row

                        # Brain positions
                        for map_idx in range(n_maps):
                            for view_idx in range(n_views):
                                brain_positions[(map_idx, view_idx)] = (
                                    map_idx,
                                    view_idx,
                                )
                        # Shared colorbar spans the entire extra row
                        colorbar_positions["shared"] = (n_maps, 0)

            else:  # vertical orientation
                # Views as rows, maps as columns
                if colorbar_style == "individual":
                    # Individual colorbars: only BOTTOM position makes sense
                    # Add 1 extra row for individual colorbars
                    grid_shape = [n_views + 1, n_maps]
                    row_weights = [1] * n_views + [
                        0.2
                    ]  # Give colorbars a bit more space
                    col_weights = [1] * n_maps

                    # Brain positions
                    for map_idx in range(n_maps):
                        for view_idx in range(n_views):
                            brain_positions[(map_idx, view_idx)] = (view_idx, map_idx)
                        # Each map gets its colorbar in the extra row at its column
                        colorbar_positions[f"individual_{map_idx}"] = (n_views, map_idx)

                else:  # shared
                    if colorbar_position == "bottom":
                        # Add 1 extra row for shared colorbar
                        grid_shape = [n_views + 1, n_maps]
                        row_weights = [1] * n_views + [0.2]
                        col_weights = [1] * n_maps
                        groups = [
                            (n_views, slice(None))
                        ]  # Span all columns in last row

                        # Brain positions
                        for map_idx in range(n_maps):
                            for view_idx in range(n_views):
                                brain_positions[(map_idx, view_idx)] = (
                                    view_idx,
                                    map_idx,
                                )
                        # Shared colorbar spans the entire extra row
                        colorbar_positions["shared"] = (n_views, 0)

                    else:  # right
                        # Add 1 extra column for shared colorbar
                        grid_shape = [n_views, n_maps + 1]
                        row_weights = [1] * n_views
                        col_weights = [1] * n_maps + [0.2]
                        groups = [(slice(None), n_maps)]  # Span all rows in last column

                        # Brain positions
                        for map_idx in range(n_maps):
                            for view_idx in range(n_views):
                                brain_positions[(map_idx, view_idx)] = (
                                    view_idx,
                                    map_idx,
                                )
                        # Shared colorbar spans the entire extra column
                        colorbar_positions["shared"] = (0, n_maps)

        return (
            grid_shape,
            row_weights,
            col_weights,
            groups,
            brain_positions,
            colorbar_positions,
        )

    ###############################################################################################
    def _create_views(
        self,
        plotter: pv.Plotter,
        config: Dict[str, Any],
        surfaces: Dict[str, Any],
        view_offset: Tuple[int, int],
    ) -> Any:
        """
        Create all configured brain views in the plotter.

        Parameters
        ----------
        plotter : pv.Plotter
            PyVista plotter instance.

        config : Dict[str, Any]
            View configuration with views list.

        surfaces : Dict[str, Any]
            Dictionary containing surface objects.

        view_offset : Tuple[int, int]
            Offset for subplot positioning due to colorbar.

        Returns
        -------
        Any
            Actor reference for colorbar creation.

        Raises
        ------
        KeyError
            If required surface mesh is not found in surfaces dictionary.

        ValueError
            If view configuration contains invalid parameters.

        Examples
        --------
        >>> surfaces = {"lh": surf_lh, "rh": surf_rh, "merged": surf_merged}
        >>> actor = self._create_views(plotter, config, surfaces, (0, 0))
        >>> print(type(actor))  # PyVista actor object
        """

        actor_for_colorbar = None

        for view_config in config["views"]:
            # Apply offset for colorbar space
            subplot_pos = (
                view_config["subplot"][0] + view_offset[0],
                view_config["subplot"][1] + view_offset[1],
            )
            plotter.subplot(*subplot_pos)

            # Set background color from figure configuration
            plotter.set_background(self.figure_conf["background_color"])

            # Add view title using figure configuration
            plotter.add_text(
                view_config["title"],
                font_size=self.figure_conf["title_font_size"],
                position="upper_edge",
                color=self.figure_conf["title_font_color"],
                shadow=self.figure_conf["title_shadow"],
                font=self.figure_conf["title_font_type"],
            )

            # Add brain mesh using figure configuration
            surface = surfaces[view_config["mesh"]]
            actor = plotter.add_mesh(
                surface.mesh,
                scalars="RGB",
                rgb=True,
                ambient=self.figure_conf["mesh_ambient"],
                diffuse=self.figure_conf["mesh_diffuse"],
                specular=self.figure_conf["mesh_specular"],
                specular_power=self.figure_conf["mesh_specular_power"],
                smooth_shading=self.figure_conf["mesh_smooth_shading"],
                show_scalar_bar=False,
            )

            # Store first actor for colorbar reference
            if actor_for_colorbar is None:
                actor_for_colorbar = actor

            # Configure camera view
            getattr(plotter, f"view_{view_config['view']}")()
            plotter.camera.azimuth = view_config["azimuth"]
            plotter.camera.elevation = view_config["elevation"]
            plotter.camera.zoom(view_config["zoom"])

        return actor_for_colorbar

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
        config: Dict[str, Any],
        surf_merged: Any,
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

        surf_merged : Surface
            Merged surface object containing data for colorbar range.

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

        # Determine colorbar subplot position
        if colorbar_position == "top":
            colorbar_subplot = (0, 0)
        elif colorbar_position == "bottom":
            colorbar_subplot = (config["shape"][0], 0)
        elif colorbar_position == "left":
            colorbar_subplot = (0, 0)
        else:  # right
            colorbar_subplot = (0, config["shape"][1])

        plotter.subplot(*colorbar_subplot)

        # Set background color for colorbar subplot
        plotter.set_background(self.figure_conf["background_color"])

        # Get data range from merged surface
        data_values = surf_merged.mesh.point_data[map_name]

        # Create colorbar mesh with proper data range
        n_points = 256
        colorbar_mesh = pv.Line((0, 0, 0), (1, 0, 0), resolution=n_points - 1)
        scalar_values = np.linspace(np.min(data_values), np.max(data_values), n_points)
        colorbar_mesh[map_name] = scalar_values

        # Add invisible mesh for colorbar reference
        dummy_actor = plotter.add_mesh(
            colorbar_mesh,
            scalars=map_name,
            cmap=colormap,
            clim=[np.min(data_values), np.max(data_values)],
            show_scalar_bar=False,
        )
        dummy_actor.visibility = False

        # Configure and add scalar bar using figure configuration
        scalar_bar_kwargs = {
            "color": self.figure_conf["colorbar_font_color"],
            "title": colorbar_title,
            "outline": self.figure_conf["colorbar_outline"],
            "title_font_size": self.figure_conf["colorbar_title_font_size"],
            "label_font_size": self.figure_conf["colorbar_font_size"],
            "n_labels": self.figure_conf["colorbar_n_labels"],
        }

        scalar_bar = plotter.add_scalar_bar(**scalar_bar_kwargs)
        scalar_bar.SetLookupTable(dummy_actor.mapper.lookup_table)
        scalar_bar.SetMaximumNumberOfColors(256)

        # Position colorbar appropriately
        if colorbar_position in ["top", "bottom"]:
            # Horizontal colorbar
            scalar_bar.SetPosition(0.05, 0.2)
            scalar_bar.SetPosition2(0.9, 0.6)
            scalar_bar.SetOrientationToHorizontal()
        else:
            # Vertical colorbar
            scalar_bar.SetPosition(0.2, 0.05)
            scalar_bar.SetPosition2(0.6, 0.9)
            scalar_bar.SetOrientationToVertical()

    ###########################################################################################################
    def _add_shared_colorbar(
        self,
        plotter: pv.Plotter,
        combined_data: np.ndarray,
        colormap: str,
        title: str,
        position: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        """
        Add a shared colorbar for multiple maps.

        Parameters
        ----------
        plotter : pv.Plotter
            PyVista plotter instance.

        combined_data : np.ndarray
            Combined data from all maps for range calculation.

        colormap : str
            Colormap to use.

        title : str
            Colorbar title.

        position : str
            Colorbar position: "right" or "bottom".

        vmin : float, optional
            Minimum value for colormap scaling.

        vmax : float, optional
            Maximum value for colormap scaling.
        """

        # Calculate shared data range
        if vmin is None:
            vmin = np.min(combined_data)

        if vmax is None:
            vmax = np.max(combined_data)

        # Create colorbar mesh
        n_points = 256
        if position == "right":
            colorbar_mesh = pv.Line((0, 0, 0), (0, 1, 0), resolution=n_points - 1)
        else:  # bottom
            colorbar_mesh = pv.Line((0, 0, 0), (1, 0, 0), resolution=n_points - 1)

        scalar_values = np.linspace(vmin, vmax, n_points)
        colorbar_mesh["shared_data"] = scalar_values

        # Add invisible mesh for colorbar reference
        dummy_actor = plotter.add_mesh(
            colorbar_mesh,
            scalars="shared_data",
            cmap=colormap,
            clim=[vmin, vmax],
            show_scalar_bar=False,
        )
        dummy_actor.visibility = False

        # Configure and add scalar bar
        scalar_bar_kwargs = {
            "color": self.figure_conf["colorbar_font_color"],
            "title": title,
            "outline": self.figure_conf["colorbar_outline"],
            "title_font_size": self.figure_conf["colorbar_title_font_size"],
            "label_font_size": self.figure_conf["colorbar_font_size"],
            "n_labels": self.figure_conf["colorbar_n_labels"],
        }

        scalar_bar = plotter.add_scalar_bar(**scalar_bar_kwargs)
        scalar_bar.SetLookupTable(dummy_actor.mapper.lookup_table)
        scalar_bar.SetMaximumNumberOfColors(256)

        # Position colorbar appropriately
        if position == "right":
            scalar_bar.SetPosition(0.1, 0.1)
            scalar_bar.SetPosition2(0.8, 0.8)
            scalar_bar.SetOrientationToVertical()
        else:  # bottom
            scalar_bar.SetPosition(0.05, 0.2)
            scalar_bar.SetPosition2(0.9, 0.6)
            scalar_bar.SetOrientationToHorizontal()

    ###############################################################################################
    def _add_concatenated_colorbar(
        self,
        plotter: pv.Plotter,
        map_data_list: List[Dict],
        colorbar_position: str,
    ) -> None:
        """
        Add a concatenated colorbar showing multiple colormaps separated by spaces.

        Parameters
        ----------
        plotter : pv.Plotter
            PyVista plotter instance with active subplot.

        map_data_list : List[Dict]
            List of dictionaries containing data, colormap, title, vmin, vmax
            for each map.

        colorbar_position : str
            Position of colorbar: "right" or "bottom".
        """

        n_maps = len(map_data_list)
        background_color = self.figure_conf["background_color"]

        # Calculate segment parameters
        if colorbar_position == "bottom":
            # Horizontal concatenated colorbar
            total_length = 1.0
            separator_width = 0.05  # 5% of total length for separators
            available_width = total_length - (n_maps - 1) * separator_width
            segment_width = available_width / n_maps

            print(
                f"📏 Horizontal concatenated colorbar: {n_maps} segments of {segment_width:.3f} width each"
            )

            # Create each colorbar segment
            current_x = 0.0
            for i, map_data in enumerate(map_data_list):
                # Create colorbar mesh for this segment
                n_points = 128  # Fewer points per segment for performance
                start_point = (current_x, 0, 0)
                end_point = (current_x + segment_width, 0, 0)

                segment_mesh = pv.Line(start_point, end_point, resolution=n_points - 1)
                scalar_values = np.linspace(
                    map_data["vmin"], map_data["vmax"], n_points
                )
                segment_mesh[f"segment_{i}"] = scalar_values

                # Add the segment mesh (invisible, just for scalar bar generation)
                segment_actor = plotter.add_mesh(
                    segment_mesh,
                    scalars=f"segment_{i}",
                    cmap=map_data["colormap"],
                    clim=[map_data["vmin"], map_data["vmax"]],
                    show_scalar_bar=True,
                    scalar_bar_args={
                        "color": self.figure_conf["colorbar_font_color"],
                        "title": map_data["title"],
                        "outline": self.figure_conf["colorbar_outline"],
                        "title_font_size": max(
                            8, self.figure_conf["colorbar_title_font_size"] - 2
                        ),  # Smaller font for multiple titles
                        "label_font_size": max(
                            6, self.figure_conf["colorbar_font_size"] - 2
                        ),
                        "n_labels": max(
                            3, self.figure_conf["colorbar_n_labels"] - 2
                        ),  # Fewer labels per segment
                        "position_x": 0.05
                        + current_x * 0.9,  # Position based on segment
                        "position_y": 0.15,
                        "width": segment_width * 0.8,  # Slightly smaller than segment
                        "height": 0.4,
                        "vertical": False,
                    },
                )

                # Hide the actual mesh, keep only the colorbar
                segment_actor.visibility = False

                # Add separator (background-colored space) except after last segment
                if i < n_maps - 1:
                    current_x += segment_width + separator_width
                else:
                    current_x += segment_width

        else:  # right position - vertical concatenated colorbar
            # Vertical concatenated colorbar
            total_length = 1.0
            separator_height = 0.05  # 5% of total length for separators
            available_height = total_length - (n_maps - 1) * separator_height
            segment_height = available_height / n_maps

            print(
                f"📏 Vertical concatenated colorbar: {n_maps} segments of {segment_height:.3f} height each"
            )

            # Create each colorbar segment (from top to bottom)
            current_y = total_length  # Start from top
            for i, map_data in enumerate(map_data_list):
                # Calculate segment position (from top)
                segment_start_y = current_y - segment_height

                # Create colorbar mesh for this segment
                n_points = 128
                start_point = (0, segment_start_y, 0)
                end_point = (0, current_y, 0)

                segment_mesh = pv.Line(start_point, end_point, resolution=n_points - 1)
                scalar_values = np.linspace(
                    map_data["vmin"], map_data["vmax"], n_points
                )
                segment_mesh[f"segment_{i}"] = scalar_values

                # Add the segment mesh (invisible, just for scalar bar generation)
                segment_actor = plotter.add_mesh(
                    segment_mesh,
                    scalars=f"segment_{i}",
                    cmap=map_data["colormap"],
                    clim=[map_data["vmin"], map_data["vmax"]],
                    show_scalar_bar=True,
                    scalar_bar_args={
                        "color": self.figure_conf["colorbar_font_color"],
                        "title": map_data["title"],
                        "outline": self.figure_conf["colorbar_outline"],
                        "title_font_size": max(
                            8, self.figure_conf["colorbar_title_font_size"] - 2
                        ),
                        "label_font_size": max(
                            6, self.figure_conf["colorbar_font_size"] - 2
                        ),
                        "n_labels": max(3, self.figure_conf["colorbar_n_labels"] - 2),
                        "position_x": 0.15,
                        "position_y": 0.05
                        + segment_start_y * 0.8,  # Position based on segment
                        "width": 0.7,
                        "height": segment_height * 0.8,  # Slightly smaller than segment
                        "vertical": True,
                    },
                )

                # Hide the actual mesh, keep only the colorbar
                segment_actor.visibility = False

                # Move to next segment position
                if i < n_maps - 1:
                    current_y = segment_start_y - separator_height

        # Set appropriate camera for the colorbar subplot
        if colorbar_position == "bottom":
            plotter.camera.position = (0.5, 0, 2)
            plotter.camera.focal_point = (0.5, 0, 0)
            plotter.camera.up = (0, 1, 0)
            plotter.camera.zoom(0.8)
        else:  # right
            plotter.camera.position = (2, 0.5, 0)
            plotter.camera.focal_point = (0, 0.5, 0)
            plotter.camera.up = (0, 1, 0)
            plotter.camera.zoom(1.2)

        # Remove axes and grid for cleaner colorbar display
        plotter.hide_axes()

        print(
            f"✅ Created concatenated {colorbar_position.upper()} colorbar with {n_maps} segments"
        )

    ###############################################################################################
    def _add_individual_colorbar(
        self,
        plotter: pv.Plotter,
        surf_merged: Any,
        map_name: str,
        colormap: str,
        colorbar_title: str,
        colorbar_position: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        n_views: int = 1,
        views_orientation: str = "horizontal",
    ) -> None:
        """
        Add an individual colorbar to the current subplot for multi-map visualization.

        Parameters
        ----------
        plotter : pv.Plotter
            PyVista plotter instance with active subplot.

        surf_merged : Surface
            Merged surface object containing data for colorbar range.

        map_name : str
            Name of the data array to use for colorbar.

        colormap : str
            Matplotlib colormap name.

        colorbar_title : str
            Title text for the colorbar.

        colorbar_position : str
            Position of colorbar (already corrected).

        vmin : float, optional
            Minimum value for colormap scaling.

        vmax : float, optional
            Maximum value for colormap scaling.

        n_views : int, default 1
            Number of views (for debugging).

        views_orientation : str, default "horizontal"
            Views orientation (for debugging).
        """

        # Get data range from merged surface for this specific map
        try:
            data_values = surf_merged.mesh.point_data[map_name]
        except KeyError:
            print(f"Warning: Map '{map_name}' not found in surface data")
            return

        # Calculate symmetric range around zero (like in the main plotting method)
        if vmin is None:
            data_min = np.min(data_values)
        else:
            data_min = vmin

        if vmax is None:
            data_max = np.max(data_values)
        else:
            data_max = vmax

        # SIMPLE POSITION-BASED ORIENTATION (position already corrected)
        n_points = 256

        if colorbar_position == "bottom":
            # BOTTOM position → HORIZONTAL orientation
            colorbar_mesh = pv.Line(
                (0, 0, 0), (1, 0, 0), resolution=n_points - 1
            )  # Horizontal line
            orientation = "horizontal"
            is_vertical = False
            position_x = 0.05
            position_y = 0.15
            width = 0.9
            height = 0.4
            print(
                f"📍 BOTTOM → HORIZONTAL colorbar for {map_name} (n_views={n_views}, orientation={views_orientation})"
            )

        else:  # right position (default)
            # RIGHT position → VERTICAL orientation
            colorbar_mesh = pv.Line(
                (0, 0, 0), (0, 1, 0), resolution=n_points - 1
            )  # Vertical line
            orientation = "vertical"
            is_vertical = True
            position_x = 0.15
            position_y = 0.05
            width = 0.7
            height = 0.9
            print(
                f"📍 RIGHT → VERTICAL colorbar for {map_name} (n_views={n_views}, orientation={views_orientation})"
            )

        # Add scalar data to the mesh
        scalar_values = np.linspace(data_min, data_max, n_points)
        colorbar_mesh[f"{map_name}_colorbar"] = scalar_values

        # Add the colorbar mesh (invisible, just for scalar bar generation)
        colorbar_actor = plotter.add_mesh(
            colorbar_mesh,
            scalars=f"{map_name}_colorbar",
            cmap=colormap,
            clim=[data_min, data_max],
            show_scalar_bar=True,  # This will create the scalar bar
            scalar_bar_args={
                "color": self.figure_conf["colorbar_font_color"],
                "title": colorbar_title,
                "outline": self.figure_conf["colorbar_outline"],
                "title_font_size": self.figure_conf["colorbar_title_font_size"],
                "label_font_size": self.figure_conf["colorbar_font_size"],
                "n_labels": self.figure_conf["colorbar_n_labels"],
                "position_x": position_x,
                "position_y": position_y,
                "width": width,
                "height": height,
                "vertical": is_vertical,  # EXPLICIT enforcement of orientation
            },
        )

        # Hide the actual mesh, keep only the colorbar
        colorbar_actor.visibility = False

        # Set appropriate camera for each orientation
        if orientation == "vertical":
            plotter.camera.position = (2, 0, 0)
            plotter.camera.focal_point = (0, 0, 0)
            plotter.camera.up = (0, 1, 0)
            plotter.camera.zoom(1.2)
        else:
            plotter.camera.position = (0, 0, 2)
            plotter.camera.focal_point = (0, 0, 0)
            plotter.camera.up = (0, 1, 0)
            plotter.camera.zoom(0.8)

        # Remove axes and grid for cleaner colorbar display
        plotter.hide_axes()

        print(
            f"✅ Created {orientation.upper()} colorbar for {map_name} at {colorbar_position.upper()} position"
        )

    ###############################################################################################
    def _process_vertex_colors(
        self,
        surface: Any,
        vertex_values: np.ndarray,
        map_name: str,
        colormap: str,
        vmin: float,
        vmax: float,
    ) -> np.ndarray:
        """
        Process vertex values into RGB colors using colortables or colormaps.

        Parameters
        ----------
        surface : Surface
            Surface object containing colortables.

        vertex_values : np.ndarray
            Array of values to be colored.

        map_name : str
            Name of the data map.

        colormap : str
            Matplotlib colormap name.

        vmin : float
            Minimum value for color range.

        vmax : float
            Maximum value for color range.

        Returns
        -------
        np.ndarray
            RGB color array for vertices with shape (n_vertices, 3).

        Raises
        ------
        KeyError
            If map_name is not found in surface data or colortables.

        ValueError
            If vertex_values array is invalid or empty.

        Examples
        --------
        >>> colors = plotter._process_vertex_colors(
        ...     surf_lh, thickness_values, "thickness", "viridis", 0.0, 5.0
        ... )
        >>> print(colors.shape)  # (n_vertices, 3) for RGB values
        """

        overlay_dict = surface.list_overlays()

        if overlay_dict[map_name] == "color":
            vertex_colors = surface.mesh.point_data[map_name]
        else:
            dict_ctables = surface.colortables

            if map_name in dict_ctables.keys():
                # Use predefined colortable for parcellations
                vertex_colors = cltfree.create_vertex_colors(
                    vertex_values, surface.colortables[map_name]["color_table"]
                )
            else:
                # Use matplotlib colormap for continuous data
                vertex_colors = cltmisc.values2colors(
                    vertex_values,
                    cmap=colormap,
                    output_format="rgb",
                    vmin=vmin,
                    vmax=vmax,
                )

        return vertex_colors

    ###############################################################################################
    def _determine_render_mode(
        self, save_path: Optional[str], notebook: bool
    ) -> Tuple[bool, bool, bool]:
        """
        Determine rendering parameters based on save path and environment.

        Parameters
        ----------
        save_path : str, optional
            File path for saving the figure, or None for display.

        notebook : bool
            Whether running in Jupyter notebook environment.

        Returns
        -------
        Tuple[bool, bool, bool]
            (save_mode, use_off_screen, use_notebook).

        Examples
        --------
        >>> save_mode, off_screen, notebook = plotter._determine_render_mode(
        ...     "output.png", False
        ... )
        >>> print(save_mode)  # True if directory exists, False otherwise
        """

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir == "":
                save_dir = "."

            if os.path.exists(save_dir):
                # Save mode - use off_screen rendering
                return True, True, False
            else:
                # Directory doesn't exist, fall back to display mode
                print(
                    f"Warning: Directory '{save_dir}' does not exist. "
                    f"Displaying plot instead of saving."
                )
                return False, False, notebook
        else:
            # Display mode
            return False, False, notebook

    ###############################################################################################
    def _setup_plotter(
        self,
        config: Dict[str, Any],
        colorbar: bool,
        colorbar_position: str,
        use_notebook: bool,
        use_off_screen: bool,
    ) -> Tuple[pv.Plotter, Tuple[int, int]]:
        """
        Setup PyVista plotter with appropriate grid layout for colorbar.

        Parameters
        ----------
        config : Dict[str, Any]
            View configuration containing shape and window_size.

        colorbar : bool
            Whether to reserve space for colorbar.

        colorbar_position : str
            Position of colorbar: "top", "bottom", "left", "right".

        use_notebook : bool
            Whether to use notebook-optimized rendering.

        use_off_screen : bool
            Whether to use off-screen rendering.

        Returns
        -------
        Tuple[pv.Plotter, Tuple[int, int]]
            (plotter instance, view_offset).

        Raises
        ------
        ValueError
            If colorbar_position is not one of the valid options.

        Examples
        --------
        >>> plotter, offset = self._setup_plotter(
        ...     config, True, "bottom", False, False
        ... )
        >>> print(offset)  # (0, 0) for bottom colorbar
        """
        # Validate colorbar position
        valid_positions = ["top", "bottom", "left", "right"]
        if colorbar_position not in valid_positions:
            colorbar_position = "bottom"

        if colorbar:
            original_shape = config["shape"]

            if isinstance(config["window_size"], str):
                screen_width, screen_height = get_screen_size()
                config["window_size"] = [screen_width, screen_height]

            if colorbar_position in ["top", "bottom"]:
                # Add row for horizontal colorbar
                if colorbar_position == "top":
                    new_shape = (original_shape[0] + 1, original_shape[1])
                    row_weights = [0.15] + [1] * original_shape[0]
                    col_weights = [1] * original_shape[1]
                    groups = [(0, slice(None))]
                    view_offset = (1, 0)
                else:  # bottom
                    new_shape = (original_shape[0] + 1, original_shape[1])
                    row_weights = [1] * original_shape[0] + [0.15]
                    col_weights = [1] * original_shape[1]
                    groups = [(original_shape[0], slice(None))]
                    view_offset = (0, 0)
            else:  # left or right
                # Add column for vertical colorbar
                if colorbar_position == "left":
                    new_shape = (original_shape[0], original_shape[1] + 1)
                    row_weights = [1] * original_shape[0]
                    col_weights = [0.15] + [1] * original_shape[1]
                    groups = [(slice(None), 0)]
                    view_offset = (0, 1)
                else:  # right
                    new_shape = (original_shape[0], original_shape[1] + 1)
                    row_weights = [1] * original_shape[0]
                    col_weights = [1] * original_shape[1] + [0.15]
                    groups = [(slice(None), original_shape[1])]
                    view_offset = (0, 0)

            # Create plotter with colorbar space
            plotter = pv.Plotter(
                notebook=use_notebook,
                off_screen=use_off_screen,
                window_size=config["window_size"],
                shape=new_shape,
                row_weights=row_weights,
                col_weights=col_weights,
                groups=groups,
                border=False,
            )
        else:
            # Create standard plotter
            plotter = pv.Plotter(
                notebook=use_notebook,
                off_screen=use_off_screen,
                window_size=config["window_size"],
                shape=config["shape"],
                border=False,
            )
            view_offset = (0, 0)

        return plotter, view_offset

    ###############################################################################################
    def _finalize_plot(
        self, plotter: pv.Plotter, save_mode: bool, save_path: Optional[str]
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

        Raises
        ------
        Exception
            If screenshot saving fails (with fallback attempts).

        IOError
            If save path is invalid or write permissions are insufficient.

        Examples
        --------
        >>> self._finalize_plot(plotter, True, "brain_plot.png")
        # Saves plot to file and closes plotter
        >>>
        >>> self._finalize_plot(plotter, False, None)
        # Displays plot in interactive window
        """

        if save_mode and save_path:
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
            # Display mode - show the plot
            plotter.show()

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
    def plot_surface(
        self,
        surface: Any,
        hemi: str = "lh",
        views: Union[str, List[str]] = "8_views",
        map_name: str = "surface",
        colormap: str = "BrBG",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        notebook: bool = False,
        colorbar: bool = True,
        colorbar_title: str = "Value",
        colorbar_position: str = "bottom",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create brain surface plots for a single hemisphere.

        Generates multi-view brain surface visualizations for one hemisphere
        with customizable layouts, coloring schemes, and optional colorbar display.

        Parameters
        ----------
        surface : Surface
            Surface object with mesh data and point_data arrays for the
            specified hemisphere.

        hemi : str, default "lh"
            Hemisphere specification: "lh" (left) or "rh" (right).

        views : str or List[str], default "8_views"
            Either configuration name from JSON file (e.g., "8_views", "6_views")
            or list of view names (e.g., ["lateral", "medial", "dorsal"]).

        map_name : str, default "surface"
            Name of data array in point_data to use for surface coloring.

        colormap : str, default "BrBG"
            Matplotlib colormap name for continuous data.

        vmin : float, optional
            Minimum value for colormap scaling. Auto-computed if None.

        vmax : float, optional
            Maximum value for colormap scaling. Auto-computed if None.

        notebook : bool, default False
            Whether to optimize for Jupyter notebook display.

        colorbar : bool, default True
            Whether to display colorbar. Automatically disabled for parcellations.

        colorbar_title : str, default "Value"
            Title text for the colorbar.

        colorbar_position : str, default "bottom"
            Colorbar position: "right", "left", "top", or "bottom".

        save_path : str, optional
            File path to save figure. If None, displays interactively.

        Raises
        ------
        ValueError
            If invalid hemisphere specified, invalid view names provided, or
            required data arrays are missing.

        KeyError
            If views configuration not found.

        Examples
        --------
        >>> # Basic usage with left hemisphere
        >>> plotter = SurfacePlotter("configs.json")
        >>> plotter.plot_surface(surf_lh, hemi="lh", views="8_views")
        >>>
        >>> # Right hemisphere with dynamic view selection
        >>> plotter.plot_surface(surf_rh, hemi="rh", views=["lateral", "medial", "dorsal"])
        >>>
        >>> # Advanced usage with custom settings
        >>> plotter.plot_surface(
        ...     surf_lh,
        ...     hemi="lh",
        ...     views=["lateral", "medial"],
        ...     map_name="cortical_thickness",
        ...     colorbar=True,
        ...     colorbar_title="Thickness (mm)",
        ...     colormap="viridis",
        ...     save_path="left_hemisphere_thickness.png"
        ... )
        """

        # Validate hemisphere parameter
        if hemi not in ["lh", "rh"]:
            raise ValueError(
                f"Invalid hemisphere '{hemi}'. Must be 'lh' (left) or 'rh' (right)"
            )

        # Handle dynamic view selection vs predefined configurations
        if isinstance(views, list):
            # Dynamic view selection - filter views relevant to single hemisphere
            config = self._create_single_hemisphere_config(views, hemi)
            print(
                f"Created dynamic {hemi.upper()} hemisphere configuration with {len(config['views'])} views: {views}"
            )
        else:
            # Predefined configuration - filter for single hemisphere
            if views not in self.views_conf:
                if views in [
                    "lateral",
                    "medial",
                    "dorsal",
                    "ventral",
                    "rostral",
                    "caudal",
                ]:
                    # Special case for global views
                    config = self._create_single_hemisphere_config([views], hemi)
                else:
                    # Raise error for invalid configuration name
                    available_configs = list(self.views_conf.keys())
                    raise KeyError(
                        f"Configuration '{views}' not found. "
                        f"Available options: {available_configs}"
                    )
            else:
                config = self._filter_config_for_hemisphere(
                    self.views_conf[views], hemi
                )
                config_name = f"{views}_{hemi}"

        # Set colorbar to False if the map_name is on the colortable
        if map_name in surface.colortables:
            colorbar = False

        if vmin is None:
            vmin = np.min(surface.mesh.point_data[map_name])

        if vmax is None:
            vmax = np.max(surface.mesh.point_data[map_name])

        try:
            vertex_values = surface.mesh.point_data[map_name]
        except KeyError:
            raise ValueError(f"Data array '{map_name}' not found in surface point_data")

        # Process vertex colors
        vertex_colors = self._process_vertex_colors(
            surface, vertex_values, map_name, colormap, vmin, vmax
        )

        # Apply colors to mesh data
        surface.mesh.point_data["RGB"] = vertex_colors

        # Determine rendering mode (save vs display)
        save_mode, use_off_screen, use_notebook = self._determine_render_mode(
            save_path, notebook
        )

        # Setup plotter with optional colorbar space
        plotter, view_offset = self._setup_plotter(
            config, colorbar, colorbar_position, use_notebook, use_off_screen
        )

        # Create surface mapping for single hemisphere
        surfaces = {
            hemi: surface,
            "merged": surface,  # For single hemisphere, merged is same as the surface
        }

        # Render each configured view
        actor_for_colorbar = self._create_views(plotter, config, surfaces, view_offset)

        # Add colorbar if requested
        if colorbar:
            self._add_colorbar(
                plotter,
                config,
                surface,
                map_name,
                colormap,
                colorbar_title,
                colorbar_position,
            )

        # Execute final rendering/display
        self._finalize_plot(plotter, save_mode, save_path)

    ###############################################################################################
    def plot_surface_multiple_maps(
        self,
        surface: Any,
        hemi: str = "lh",
        views: Union[str, List[str]] = "8_views",
        map_name: str = "surface",
        colormap: str = "BrBG",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        notebook: bool = False,
        colorbar: bool = True,
        colorbar_title: str = "Value",
        colorbar_position: str = "bottom",
        save_path: Optional[str] = None,
    ) -> None:

        pass

    ###############################################################################################
    def plot_hemispheres(
        self,
        surf_lh: Any,
        surf_rh: Any,
        views: Union[str, List[str]] = "8_views",
        notebook: bool = False,
        map_name: str = "surface",
        colormap: str = "BrBG",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        save_path: Optional[str] = None,
        colorbar: bool = True,
        colorbar_title: str = "Value",
        colorbar_position: str = "bottom",
    ) -> None:
        """
        Create brain surface plots with flexible multi-view configurations.

        Generates multi-view brain surface visualizations with customizable
        layouts, coloring schemes, and optional colorbar display for both
        hemispheres.

        Parameters
        ----------
        surf_lh : Surface
            Left hemisphere surface object with mesh data and point_data arrays.

        surf_rh : Surface
            Right hemisphere surface object with mesh data and point_data arrays.

        views : str or List[str], default "8_views"
            Either configuration name from JSON file (e.g., "8_views", "6_views")
            or list of view names (e.g., ["lateral", "medial", "dorsal"]).

        notebook : bool, default False
            Whether to optimize for Jupyter notebook display.

        map_name : str, default "surface"
            Name of data array in point_data to use for surface coloring.

        colormap : str, default "BrBG"
            Matplotlib colormap name for continuous data.

        vmin : float, optional
            Minimum value for colormap scaling. Auto-computed if None.

        vmax : float, optional
            Maximum value for colormap scaling. Auto-computed if None.

        save_path : str, optional
            File path to save figure. If None, displays interactively.

        colorbar : bool, default True
            Whether to display colorbar. Automatically disabled for parcellations.

        colorbar_title : str, default "Value"
            Title text for the colorbar.

        colorbar_position : str, default "bottom"
            Colorbar position: "right", "left", "top", or "bottom".

        Raises
        ------
        KeyError
            If views configuration not found or data array missing.

        ValueError
            If invalid view names provided or data arrays not found.

        Examples
        --------
        >>> # Basic usage with predefined configuration
        >>> plotter = SurfacePlotter("configs.json")
        >>> plotter.plot_hemispheres(surf_lh, surf_rh, views="8_views")
        >>>
        >>> # Dynamic view selection with list of view names
        >>> plotter.plot_hemispheres(surf_lh, surf_rh, views=["lateral", "medial", "dorsal"])
        >>>
        >>> # Advanced usage with custom settings
        >>> plotter.plot_hemispheres(
        ...     surf_lh, surf_rh,
        ...     views=["lateral", "medial"],
        ...     map_name="cortical_thickness",
        ...     colorbar=True,
        ...     colorbar_title="Thickness (mm)",
        ...     colormap="viridis",
        ...     save_path="brain_thickness.png"
        ... )
        """

        # Handle dynamic view selection vs predefined configurations
        if isinstance(views, list):
            # Dynamic view selection
            config = self._create_dynamic_config(views)
            config_name = f"dynamic_{len(views)}_views"
            print(
                f"Created dynamic configuration with {len(config['views'])} views: {views}"
            )
        else:
            # Predefined configuration
            if views not in self.views_conf:
                available_configs = list(self.views_conf.keys())
                raise KeyError(
                    f"Configuration '{views}' not found. "
                    f"Available options: {available_configs}"
                )
            config = self.views_conf[views]
            config_name = views

        # Set colorbar to False if the map_name is on the colortable
        if map_name in surf_lh.colortables or map_name in surf_rh.colortables:
            colorbar = False

        # Extract vertex values for coloring
        try:
            vertex_values_lh = surf_lh.mesh.point_data[map_name]
            vertex_values_rh = surf_rh.mesh.point_data[map_name]
            vertex_values_lh = np.nan_to_num(vertex_values_lh, nan=0.0)
            vertex_values_rh = np.nan_to_num(vertex_values_rh, nan=0.0)
        except KeyError:
            raise ValueError(f"Data array '{map_name}' not found in surface point_data")

        if vmin is None:
            vmin = np.min(np.concatenate((vertex_values_lh, vertex_values_rh)))

        if vmax is None:
            vmax = np.max(np.concatenate((vertex_values_lh, vertex_values_rh)))

        # Process left hemisphere colors
        vertex_colors_lh = self._process_vertex_colors(
            surf_lh, vertex_values_lh, map_name, colormap, vmin, vmax
        )

        # Process right hemisphere colors
        vertex_colors_rh = self._process_vertex_colors(
            surf_rh, vertex_values_rh, map_name, colormap, vmin, vmax
        )

        # Apply colors to mesh data
        surf_lh.mesh.point_data["RGB"] = vertex_colors_lh
        surf_rh.mesh.point_data["RGB"] = vertex_colors_rh

        # Create merged surface for colorbar data range
        surf_merged = surf_lh.merge_surfaces([surf_rh])

        # Determine rendering mode (save vs display)
        save_mode, use_off_screen, use_notebook = self._determine_render_mode(
            save_path, notebook
        )

        # Setup plotter with optional colorbar space
        plotter, view_offset = self._setup_plotter(
            config, colorbar, colorbar_position, use_notebook, use_off_screen
        )

        # Create surface mapping for easy access
        surfaces = {"lh": surf_lh, "rh": surf_rh, "merged": surf_merged}

        # Render each configured view
        actor_for_colorbar = self._create_views(plotter, config, surfaces, view_offset)

        # Add colorbar if requested
        if colorbar:
            self._add_colorbar(
                plotter,
                config,
                surf_merged,
                map_name,
                colormap,
                colorbar_title,
                colorbar_position,
            )

        # Execute final rendering/display
        self._finalize_plot(plotter, save_mode, save_path)

    ###############################################################################################
    def plot_hemispheres_multiple_maps(
        self,
        surf_lh: Any,
        surf_rh: Any,
        views: Union[str, List[str]] = "dorsal",
        views_orientation: str = "horizontal",
        notebook: bool = False,
        map_names: List[str] = ["surface"],
        v_limits: Optional[
            Union[Tuple[float, float], List[Tuple[float, float]]]
        ] = None,
        colormaps: Union[str, List[str]] = "BrBG",
        save_path: Optional[str] = None,
        colorbar: bool = True,
        colorbar_style: str = "individual",
        colorbar_title: Union[str, List[str]] = "Value",
        colorbar_position: str = "right",
    ) -> None:
        """
        Create brain surface plots with multiple overlay maps.

        Generates multi-view visualizations with multiple overlay maps arranged
        in a grid layout. Supports individual or shared colorbars with automatic
        layout optimization.

        Parameters
        ----------
        surf_lh : Surface
            Left hemisphere surface object with mesh data and point_data arrays.

        surf_rh : Surface
            Right hemisphere surface object with mesh data and point_data arrays.

        views : str or List[str], default "dorsal"
            Either configuration name from JSON file or list of view names to
            dynamically select.

        views_orientation : str, default "horizontal"
            Layout orientation: "horizontal" (maps as rows) or "vertical"
            (maps as columns). Ignored when using grid layout.

        notebook : bool, default False
            Whether to optimize for Jupyter notebook display.

        map_names : List[str], default ["surface"]
            List of data arrays to use for surface coloring.

        v_limits : tuple or List[tuple], optional
            Colormap limits for the maps. Can be single tuple for all maps or
            list of tuples for individual maps.

        colormaps : str or List[str], default "BrBG"
            Colormap(s) to use. Either single colormap for all maps or list
            of colormaps.

        save_path : str, optional
            File path to save the figure.

        colorbar : bool, default True
            Whether to display colorbars.

        colorbar_style : str, default "individual"
            Colorbar style: "individual" (each map gets its own colorbar) or
            "shared" (single colorbar for all maps).

        colorbar_title : str or List[str], default "Value"
            Title text for the colorbar(s). If string, same title used for all.
            If list, each element is the title for the corresponding map.

        colorbar_position : str, default "right"
            Colorbar placement: "right" or "bottom". Invalid combinations
            will be auto-corrected.

        Raises
        ------
        TypeError
            If map_names is not a string or list of strings, or v_limits format
            is invalid.

        ValueError
            If no valid maps found in both hemispheres, or v_limits list length
            doesn't match maps.

        Examples
        --------
        >>> # Concatenated shared colorbar with different colormaps
        >>> plotter.plot_hemispheres_multiple_maps(
        ...     surf_lh, surf_rh,
        ...     map_names=["thickness", "curvature", "area"],
        ...     colormaps=["viridis", "coolwarm", "plasma"],
        ...     colorbar_style="shared",
        ...     colorbar_title=["Thickness (mm)", "Curvature", "Area (mm²)"]
        ... )
        """

        # Validate inputs
        if views_orientation not in ["horizontal", "vertical"]:
            views_orientation = "horizontal"

        if colorbar_position not in ["right", "bottom"]:
            colorbar_position = "right"

        if colorbar_style not in ["individual", "shared"]:
            colorbar_style = "individual"

        # Handle map_names input
        if isinstance(map_names, str):
            map_names = [map_names]
        elif not isinstance(map_names, list):
            raise TypeError("map_names must be a string or a list of strings")

        # Get available maps on both hemispheres
        lh_map_names = list(surf_lh.mesh.point_data.keys())
        rh_map_names = list(surf_rh.mesh.point_data.keys())
        available_maps = cltmisc.list_intercept(lh_map_names, rh_map_names)

        # Filter to only available maps
        map_names = [m for m in map_names if m in available_maps]
        n_maps = len(map_names)

        if n_maps == 0:
            raise ValueError("No valid maps found in both hemispheres")

        # Process and validate v_limits parameter
        map_limits = self._process_v_limits(v_limits, n_maps)

        # Handle colormaps
        if isinstance(colormaps, str):
            colormaps = [colormaps] * n_maps
        elif len(colormaps) != n_maps:
            # Extend or truncate colormaps to match number of maps
            colormaps = (colormaps * ((n_maps // len(colormaps)) + 1))[:n_maps]

        # Check if we have different colormaps for concatenated colorbar
        has_different_colormaps = len(set(colormaps)) > 1

        # Handle colorbar titles
        if isinstance(colorbar_title, str):
            colorbar_titles = [colorbar_title] * n_maps
        elif isinstance(colorbar_title, list):
            if len(colorbar_title) != n_maps:
                if len(colorbar_title) == 0:
                    colorbar_titles = ["Value"] * n_maps
                else:
                    # Use first title for all if lengths don't match
                    colorbar_titles = [colorbar_title[0]] * n_maps
                    print(
                        f"Warning: colorbar_title list length ({len(colorbar_title)}) doesn't match number of maps ({n_maps}). Using '{colorbar_title[0]}' for all colorbars."
                    )
            else:
                colorbar_titles = colorbar_title.copy()
        else:
            colorbar_titles = ["Value"] * n_maps

        # Configure views
        if isinstance(views, str):
            if views.lower() in [
                "lateral",
                "medial",
                "dorsal",
                "ventral",
                "rostral",
                "caudal",
            ]:
                # Single view name - create dynamic config
                config = self._create_dynamic_config([views])
            elif views.lower() in self.views_conf:
                # Predefined configuration
                config = self.views_conf[views]
            else:
                # Default to lateral if invalid
                config = self._create_dynamic_config(["dorsal"])
        elif isinstance(views, list):
            # Dynamic view selection
            config = self._create_dynamic_config(views)
        else:
            raise TypeError("views must be a string or a list of view names")

        n_views = len(config["views"])

        if (
            n_views > 1
            and views_orientation == "vertical"
            and colorbar_style == "individual"
        ):
            colorbar_position = "bottom"
        elif (
            n_views > 1
            and views_orientation == "horizontal"
            and colorbar_style == "individual"
        ):
            colorbar_position = "right"

        # Check for special case: single view with multiple maps
        use_grid_layout = n_views == 1 and n_maps > 1

        if use_grid_layout:
            # Calculate grid layout and setup plotter parameters for single-view multi-map
            (
                grid_shape,
                row_weights,
                col_weights,
                groups,
                brain_positions,
                colorbar_positions,
            ) = self._calculate_single_view_multi_map_layout(
                n_maps, colorbar, colorbar_style, colorbar_position
            )
        else:
            # Use original multi-map layout for multiple views
            (
                grid_shape,
                row_weights,
                col_weights,
                groups,
                brain_positions,
                colorbar_positions,
            ) = self._calculate_multi_map_layout(
                n_maps,
                n_views,
                views_orientation,
                colorbar,
                colorbar_style,
                colorbar_position,
            )

        # Always use fullscreen - get actual screen size
        window_size = list(get_screen_size())

        # Determine rendering mode
        save_mode, use_off_screen, use_notebook = self._determine_render_mode(
            save_path, notebook
        )

        # Create plotter with proper parameters
        plotter_kwargs = {
            "notebook": use_notebook,
            "off_screen": use_off_screen,
            "window_size": window_size,
            "shape": grid_shape,
            "row_weights": row_weights,
            "col_weights": col_weights,
            "border": False,
        }

        if groups:
            plotter_kwargs["groups"] = groups

        plotter = pv.Plotter(**plotter_kwargs)

        # Store surfaces and data for colorbar
        all_data_values = []
        map_data_for_colorbar = (
            []
        )  # Store individual map data for concatenated colorbar

        # Process and plot each map
        for map_idx, map_name in enumerate(map_names):
            colormap = colormaps[map_idx]
            individual_colorbar_title = colorbar_titles[map_idx]
            vmin, vmax = map_limits[map_idx]  # Get the limits for this specific map

            # Extract vertex values for this map
            try:
                vertex_values_lh = surf_lh.mesh.point_data[map_name]
                vertex_values_lh = np.nan_to_num(vertex_values_lh, nan=0.0)
                vertex_values_rh = surf_rh.mesh.point_data[map_name]
                vertex_values_rh = np.nan_to_num(vertex_values_rh, nan=0.0)

            except KeyError:
                print(f"Warning: Map '{map_name}' not found, skipping...")
                continue

            # Check if using colortables
            use_colortable = (
                map_name in surf_lh.colortables or map_name in surf_rh.colortables
            )
            show_colorbar = colorbar and not use_colortable

            # Calculate color range for this map if not provided
            if not use_colortable:
                if vmin is None or vmax is None:
                    combined_values = np.concatenate(
                        (vertex_values_lh, vertex_values_rh)
                    )
                    if vmin is None:
                        vmin = np.min(combined_values)
                    if vmax is None:
                        vmax = np.max(combined_values)

                    print(
                        f"Auto-computed limits for '{map_name}': vmin={vmin:.3f}, vmax={vmax:.3f}"
                    )
                else:
                    print(
                        f"Using provided limits for '{map_name}': vmin={vmin:.3f}, vmax={vmax:.3f}"
                    )

                all_data_values.extend([vertex_values_lh, vertex_values_rh])

                # Store data for concatenated colorbar
                map_data_for_colorbar.append(
                    {
                        "data": np.concatenate((vertex_values_lh, vertex_values_rh)),
                        "colormap": colormap,
                        "title": individual_colorbar_title,
                        "vmin": vmin,
                        "vmax": vmax,
                        "map_name": map_name,
                    }
                )
            else:
                vmin = vmax = None

            # Process vertex colors using the same method as plot_hemispheres
            vertex_colors_lh = self._process_vertex_colors(
                surf_lh, vertex_values_lh, map_name, colormap, vmin, vmax
            )
            vertex_colors_rh = self._process_vertex_colors(
                surf_rh, vertex_values_rh, map_name, colormap, vmin, vmax
            )

            # Create temporary surfaces with unique color arrays to avoid interference
            import copy

            temp_surf_lh = copy.deepcopy(surf_lh)
            temp_surf_rh = copy.deepcopy(surf_rh)

            temp_surf_lh.mesh.point_data["RGB"] = vertex_colors_lh
            temp_surf_rh.mesh.point_data["RGB"] = vertex_colors_rh

            # Create merged surface for colorbar
            temp_surf_merged = temp_surf_lh.merge_surfaces([temp_surf_rh])

            # Create surface mapping for this map
            surfaces = {
                "lh": temp_surf_lh,
                "rh": temp_surf_rh,
                "merged": temp_surf_merged,
            }

            # Plot each view for this map
            if use_grid_layout:
                # Single view grid layout: only plot once per map
                view_config = config["views"][0]  # Use the single view
                brain_key = (map_idx, 0)  # view_idx is always 0
                if brain_key in brain_positions:
                    subplot_pos = brain_positions[brain_key]

                    plotter.subplot(*subplot_pos)
                    plotter.set_background(self.figure_conf["background_color"])

                    # Add title
                    title = f"{map_name}"  # Cleaner title for grid layout
                    plotter.add_text(
                        title,
                        font_size=self.figure_conf["title_font_size"],
                        position="upper_edge",
                        color=self.figure_conf["title_font_color"],
                        shadow=self.figure_conf["title_shadow"],
                        font=self.figure_conf["title_font_type"],
                    )

                    # Add brain mesh
                    surface = surfaces[view_config["mesh"]]
                    plotter.add_mesh(
                        surface.mesh,
                        scalars="RGB",
                        rgb=True,
                        ambient=self.figure_conf["mesh_ambient"],
                        diffuse=self.figure_conf["mesh_diffuse"],
                        specular=self.figure_conf["mesh_specular"],
                        specular_power=self.figure_conf["mesh_specular_power"],
                        smooth_shading=self.figure_conf["mesh_smooth_shading"],
                        show_scalar_bar=False,
                    )

                    # Configure camera view
                    getattr(plotter, f"view_{view_config['view']}")()
                    plotter.camera.azimuth = view_config["azimuth"]
                    plotter.camera.elevation = view_config["elevation"]
                    plotter.camera.zoom(view_config["zoom"])
            else:
                # Original multi-view layout: plot all views for this map
                for view_idx, view_config in enumerate(config["views"]):
                    # Get brain subplot position from pre-calculated positions
                    brain_key = (map_idx, view_idx)
                    if brain_key in brain_positions:
                        subplot_pos = brain_positions[brain_key]

                        plotter.subplot(*subplot_pos)
                        plotter.set_background(self.figure_conf["background_color"])

                        # Add title
                        title = f"{map_name}: {view_config['title']}"
                        plotter.add_text(
                            title,
                            font_size=self.figure_conf["title_font_size"],
                            position="upper_edge",
                            color=self.figure_conf["title_font_color"],
                            shadow=self.figure_conf["title_shadow"],
                            font=self.figure_conf["title_font_type"],
                        )

                        # Add brain mesh
                        surface = surfaces[view_config["mesh"]]
                        plotter.add_mesh(
                            surface.mesh,
                            scalars="RGB",
                            rgb=True,
                            ambient=self.figure_conf["mesh_ambient"],
                            diffuse=self.figure_conf["mesh_diffuse"],
                            specular=self.figure_conf["mesh_specular"],
                            specular_power=self.figure_conf["mesh_specular_power"],
                            smooth_shading=self.figure_conf["mesh_smooth_shading"],
                            show_scalar_bar=False,
                        )

                        # Configure camera view
                        getattr(plotter, f"view_{view_config['view']}")()
                        plotter.camera.azimuth = view_config["azimuth"]
                        plotter.camera.elevation = view_config["elevation"]
                        plotter.camera.zoom(view_config["zoom"])

            # Add individual colorbar for this map
            if show_colorbar and colorbar_style == "individual":
                colorbar_key = f"individual_{map_idx}"
                if colorbar_key in colorbar_positions:
                    colorbar_pos = colorbar_positions[colorbar_key]

                    plotter.subplot(*colorbar_pos)
                    plotter.set_background(self.figure_conf["background_color"])

                    # Add individual colorbar for this specific map
                    self._add_individual_colorbar(
                        plotter,
                        temp_surf_merged,
                        map_name,
                        colormap,
                        individual_colorbar_title,  # Use the title directly from the list
                        colorbar_position,
                        vmin,
                        vmax,
                        n_views,  # Pass number of views for multi-view enforcement
                        views_orientation,  # Pass orientation for multi-view enforcement
                    )

        # Add shared colorbar if requested
        if colorbar and colorbar_style == "shared" and map_data_for_colorbar:
            colorbar_key = "shared"
            if colorbar_key in colorbar_positions:
                colorbar_pos = colorbar_positions[colorbar_key]

                plotter.subplot(*colorbar_pos)
                plotter.set_background(self.figure_conf["background_color"])

                if has_different_colormaps:
                    # Create concatenated colorbar with different colormaps
                    print(
                        f"🎨 Creating concatenated colorbar with {len(set(colormaps))} different colormaps"
                    )
                    self._add_concatenated_colorbar(
                        plotter, map_data_for_colorbar, colorbar_position
                    )
                else:
                    # Use traditional shared colorbar with single colormap
                    combined_data = np.concatenate(all_data_values)
                    shared_title = colorbar_titles[0] if colorbar_titles else "Value"

                    # For shared colorbar, use the first map's limits or compute from all data
                    shared_vmin, shared_vmax = map_limits[0]
                    if shared_vmin is None or shared_vmax is None:
                        if shared_vmin is None:
                            shared_vmin = np.min(combined_data)
                        if shared_vmax is None:
                            shared_vmax = np.max(combined_data)

                    # Create traditional shared colorbar
                    self._add_shared_colorbar(
                        plotter,
                        combined_data,
                        colormaps[0],
                        shared_title,
                        colorbar_position,
                        shared_vmin,
                        shared_vmax,
                    )
        if n_views == 1 and n_maps > 1:
            plotter.link_views()

        # Execute final rendering/display
        self._finalize_plot(plotter, save_mode, save_path)

    ################################################################################################
    def plot_surfaces_list(
        self,
        surfaces: List[Any],
        views: Union[str, List[str]] = "8_views",
        notebook: bool = False,
        map_names: str = "surface",
        colormap: str = "BrBG",
        v_limits: Optional[
            Union[Tuple[float, float], List[Tuple[float, float]]]
        ] = None,
        save_path: Optional[str] = None,
        colorbar: bool = True,
        colorbar_title: str = "Value",
        colorbar_position: str = "bottom",
    ) -> None:
        """
        Create a grid of brain surface plots from a list of surfaces.

        Parameters
        ----------
        surfaces : List[Surface]
            List of surface objects with mesh data and point_data arrays.

        views : str or List[str], default "8_views"
            Either configuration name from JSON file or list of view names.

        notebook : bool, default False
            Whether to optimize for Jupyter notebook display.

        map_name : str, default "surface"
            Name of data array in point_data to use for surface coloring.

        colormap : str, default "BrBG"
            Matplotlib colormap name for continuous data.

        vmin : float, optional
            Minimum value for colormap scaling. Auto-computed if None.

        vmax : float, optional
            Maximum value for colormap scaling. Auto-computed if None.

        save_path : str, optional
            File path to save figure. If None, displays interactively.

        colorbar : bool, default True
            Whether to display colorbar. Automatically disabled for parcellations.

        colorbar_title : str, default "Value"
            Title text for the colorbar.

        colorbar_position : str, default "bottom"
            Colorbar position: "right", "left", "top", or "bottom".

        Raises
        ------
        KeyError
            If views configuration not found or data array missing.

        ValueError
            If invalid view names provided or data arrays not found.

        Examples
        --------
        >>> # Basic usage with predefined configuration
        >>> plotter = SurfacePlotter("configs.json")
        >>> plotter.plot_surfaces_list(surfaces)

        >>> # Dynamic view selection with list of view names
        >>> plotter.plot_surfaces_list(surfaces, views=["lateral", "medial", "dorsal"])

        >>>
        """
        pass


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
    return width, height
