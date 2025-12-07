"""
Brain Surface Visualization Tools

This module provides comprehensive tools for visualizing brain surfaces with various
anatomical views and data overlays. It supports FreeSurfer surface formats and
provides flexible layout options for publication-quality figures.

Classes:
    SurfacePlotter: Main class for creating multi-view brain surface layouts
"""

import os
import copy
import numpy as np
from typing import Union, List, Optional, Tuple, Dict, Any
import pyvista as pv

# Importing local modules
from . import misctools as cltmisc
from . import plottools as cltplot

# Use TYPE_CHECKING to avoid circular imports
from . import surfacetools as cltsurf
from . import tracttools as clttract
from . import visualization_utils as visutils
from . import build_visualization_layout as vislayout


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############            Section 1: Class dedicated to plot Surface objects              ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
class BrainPlotter:
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

        # Load configurations from the tractogram file
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
        config, colorbar_list = vislayout.build_layout_config(
            self,
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

    ###############################################################################
    def plot(
        self,
        objs2plot: Union[cltsurf.Surface, clttract.Tractogram, List],
        hemi_id: Union[str, List[str]] = "both",
        views: Union[str, List[str]] = "dorsal",
        views_orientation: str = "horizontal",
        notebook: bool = False,
        map_names: Union[str, List[str]] = ["default"],
        v_limits: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = (
            None,
            None,
        ),
        v_range: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = (
            None,
            None,
        ),
        range_color: Tuple = (128, 128, 128, 255),
        use_opacity: bool = True,
        tract_plot_style: str = "tube",
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
        objs2plot : Union[cltsurf.Surface, clttract.Tractogram, List[cltsurf.Surface], List[List[cltsurf.Surface]], List[clttract.Tractogram], List[List[clttract.Tractogram]]]
            Single or list of brain surface objects (cltsurf.Surface) or tractogram objects (clttract.Tractogram).
            Each element can also be a list of such objects to be merged and plotted together.

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

        v_range : Optional[Union[Tuple[float, float], List[Tuple[float, float]]]], default (None, None)
            Value range for colormap application. Values outside this range will be displayed in range_color.

        range_color : Tuple, default (128, 128, 128, 255)
            RGBA color to use for values outside the specified v_range.

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
        if isinstance(objs2plot, cltsurf.Surface):
            obj2plot = [copy.deepcopy(objs2plot)]

        elif isinstance(objs2plot, clttract.Tractogram):
            obj2plot = [copy.deepcopy(objs2plot)]

        elif isinstance(objs2plot, list):
            # If all the elements are of type cltsurf.Surface

            obj2plot = []
            for sing_obj in objs2plot:
                if isinstance(sing_obj, (cltsurf.Surface, clttract.Tractogram)):
                    obj2plot.append(copy.deepcopy(sing_obj))

                elif isinstance(sing_obj, list) and all(
                    hasattr(s, "mesh") for s in sing_obj
                ):
                    obj2plot.append(cltsurf.merge_surfaces(sing_obj))

                elif isinstance(sing_obj, list) and all(
                    hasattr(s, "tracts") for s in sing_obj
                ):
                    obj2plot.append(clttract.merge_tractograms(sing_obj))

                else:
                    raise TypeError(
                        "All elements must be of type Surface, Tractogram or a list of such."
                    )

        # Number of objects to plot
        n_objects = len(obj2plot)

        # Filter to only available maps
        if isinstance(map_names, str):
            map_names = [map_names]
        n_maps = len(map_names)

        fin_map_names = []
        for i, map_name in enumerate(map_names):
            cont_map = 0
            # Check if the map_name is available in any of the surfaces or tractograms
            for sing_obj in obj2plot:
                if isinstance(sing_obj, clttract.Tractogram):
                    map_list_dict = sing_obj.list_maps()

                    st_maps = map_list_dict["maps_per_streamline"]
                    pt_maps = map_list_dict["maps_per_point"]

                    if map_name in st_maps and map_name not in pt_maps:
                        sing_obj.streamline_to_points(
                            map_name=map_name,
                            point_map_name=map_name,
                        )

                    available_maps = []
                    if st_maps is not None:
                        available_maps = available_maps + st_maps

                    if pt_maps is not None:
                        available_maps = available_maps + pt_maps

                elif isinstance(sing_obj, cltsurf.Surface):
                    available_maps = list(sing_obj.mesh.point_data.keys())

                if map_name in available_maps:
                    cont_map = cont_map + 1

            #
            if cont_map == n_objects:
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

        if isinstance(v_range, Tuple):
            if len(v_range) != 2:
                v_range = (None, None)
            v_range = [v_range] * n_maps

        elif isinstance(v_range, List[Tuple[float, float]]):
            if len(v_range) != n_maps:
                v_range = [(None, None)] * n_maps

        for vr in v_range:
            if not (isinstance(vr, Tuple) and len(vr) == 2):
                raise ValueError(
                    "Each element in v_range must be a tuple of (min, max)."
                )
            # Check that the min is less than max if both are not None
            if vr[0] is not None and vr[1] is not None:
                if vr[0] >= vr[1]:
                    raise ValueError(
                        "In v_range, min value must be less than max value."
                    )

        for vl in v_limits:
            if not (isinstance(vl, Tuple) and len(vl) == 2):
                raise ValueError(
                    "Each element in v_limits must be a tuple of (min, max)."
                )

            # Check that the min is less than max if both are not None
            if vl[0] is not None and vl[1] is not None:
                if vl[0] >= vl[1]:
                    raise ValueError(
                        "In v_limits, min value must be less than max value."
                    )

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

        # Check if the is colortable at any of the surfaces for any of the maps

        (
            view_ids,
            config_dict,
            colorbar_dict_list,
        ) = self._build_plotting_config(
            views=views,
            maps_names=map_names,
            surfaces=obj2plot,
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
            "border": self.figure_conf.get("subplot_border", True),
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

        for (map_idx, obj_idx, view_idx), position in brain_positions.items():
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
        for (map_idx, obj_idx, view_idx), (row, col) in brain_positions.items():
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
                f"{map_names[map_idx]}, Surface: {obj_idx}, View: {tmp_view_name}",
                font_size=self.figure_conf["title_font_size"],
                position="upper_edge",
                color=self.figure_conf["title_font_color"],
                shadow=self.figure_conf["title_shadow"],
                font=self.figure_conf["title_font_type"],
            )

            # Geting the vmin and vmax for the current map
            vmin, vmax, map_name = map_limits[map_idx, obj_idx, view_idx]

            # Select the colormap for the current map
            idx = [i for i, name in enumerate(map_names) if name == map_name]
            colormap = colormaps[idx[0]] if idx else colormaps[0]

            # Add the brain surface mesh
            tmp_obj = obj2plot[obj_idx]
            tmp_obj = visutils.prepare_obj_for_plotting(
                tmp_obj,
                map_names[map_idx],
                colormap,
                vmin=vmin,
                vmax=vmax,
                range_min=v_range[map_idx][0],
                range_max=v_range[map_idx][1],
                range_color=range_color,
            )
            if isinstance(tmp_obj, clttract.Tractogram):

                tracts = tmp_obj.tracts
                rgba_data = tmp_obj.data_per_point["rgba"]

                # 1. Concatenate all points and colors
                all_points = np.vstack(tracts)
                all_rgba = np.vstack(rgba_data)

                # 2. Build the lines connectivity array
                #    Format: [n1, idx0, idx1, ..., n2, idx0, idx1, ...]
                lines = []
                offset = 0
                for tract in tracts:
                    n = len(tract)
                    lines.append(n)
                    lines.extend(range(offset, offset + n))
                    offset += n
                lines = np.array(lines, dtype=np.int_)

                # 3. Create single PolyData with all curves
                if tract_plot_style == "tube":
                    # Create a PolyData object for tube representation
                    # Create a PolyData object for tube representation
                    poly = pv.PolyData()
                    poly.points = all_points
                    poly.lines = lines

                    # Attach your RGBA scalars
                    poly.point_data["rgba"] = all_rgba  # <-- important

                    # Add tube filter (tube cannot take scalars directly)
                    tube_radius = 0.1
                    tube_sides = 10
                    tube_poly = poly.tube(
                        radius=tube_radius,
                        n_sides=tube_sides,
                    )

                    # Add the mesh with tube representation
                    pv_plotter.add_mesh(
                        tube_poly,
                        scalars="rgba",  # use the same name
                        rgb=True,
                        ambient=self.figure_conf["mesh_ambient"],
                        diffuse=self.figure_conf["mesh_diffuse"],
                        specular=self.figure_conf["mesh_specular"],
                        specular_power=self.figure_conf["mesh_specular_power"],
                        smooth_shading=self.figure_conf["mesh_smooth_shading"],
                        show_scalar_bar=False,
                    )
                else:
                    poly = pv.PolyData()
                    poly.points = all_points
                    poly.lines = lines
                    poly.point_data["rgba"] = all_rgba

                    # 4. Single add_mesh call
                    pv_plotter.add_mesh(
                        poly,
                        scalars="rgba",
                        line_width=2,
                        rgb=True,
                        ambient=self.figure_conf["mesh_ambient"],
                        diffuse=self.figure_conf["mesh_diffuse"],
                        specular=self.figure_conf["mesh_specular"],
                        specular_power=self.figure_conf["mesh_specular_power"],
                        smooth_shading=self.figure_conf["mesh_smooth_shading"],
                        show_scalar_bar=False,
                    )

            elif isinstance(tmp_obj, cltsurf.Surface):
                if not use_opacity:
                    # delete the alpha channel if exists
                    if "rgba" in tmp_obj.mesh.point_data:
                        tmp_obj.mesh.point_data["rgba"] = tmp_obj.mesh.point_data[
                            "rgba"
                        ][:, :3]

                pv_plotter.add_mesh(
                    copy.deepcopy(tmp_obj.mesh),
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

    ###########################################################################
    def plot_surface_hemispheres(
        self,
        surf_rh: cltsurf.Surface,
        surf_lh: cltsurf.Surface,
        map_name: str = "default",
        views: Union[str, List[str]] = "dorsal",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
        range_color: Tuple = (128, 128, 128, 255),
        use_opacity: bool = True,
        colormap: str = "viridis",
        colorbar: bool = True,
        colorbar_title: str = None,
        colorbar_position: str = "right",
        notebook: bool = False,
        non_blocking: bool = False,
        save_path: Optional[str] = None,
    ):
        """
        Plot brain hemispheres with multiple views.

        Parameters
        ----------
        surf_rh : cltsurf.Surface
            Right hemisphere surface with associated data.

        surf_lh : cltsurf.Surface
            Left hemisphere surface with associated data.

        map_name : str "default"
            Name of the data maps to visualize. Must be present in both surfaces.

        views : str or list of str, default "dorsal"
            View(s) to display. Options include 'dorsal', 'ventral', 'lateral', 'medial', 'anterior', 'posterior'.
            Can be a single view or a list of views. It can also include different multiple views specified as layouts:
            >>> plotter = SurfacePlotter("configs.json")
            >>> layouts = plotter.list_available_layouts()

        vmin : float, optional
            Minimum value for colormap scaling. If None, uses the minimum value from the data.

        vmax : float, optional
            Maximum value for colormap scaling. If None, uses the maximum value from the data.

        range_min : float, optional
            Minimum value for the value range. Values below this will be colored with `range_color`.

        range_max : float, optional
            Maximum value for the value range. Values above this will be colored with `range_color`.

        range_color : Tuple, default (128, 128, 128, 255)
            RGBA color to use for values outside the specified v_range.

        colormap : str or list of str, default "BrBG"
            Colormap to use for visualization.

        colorbar : bool, default True
            Whether to display colorbars for the maps.

        colorbar_title : str, optional
            Title for the colorbar. If None, map name are used.

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
        >>> plotter.plot_surface_hemispheres(surf_lh, surf_rh, map_name="thickness",
        ...                                 views=["lateral", "medial", "dorsal"],
        ...                                 colorbar=True)
        """

        # Check that the map_name is a str. If is a list select the first element
        if isinstance(map_name, list):
            map_name = map_name[0]

        if isinstance(colormap, list):
            colormap = colormap[0]

        if isinstance(colorbar_title, list):
            colorbar_title = colorbar_title[0]

        if vmin is None and range_min is not None:
            vmin = range_min

        if vmax is None and range_max is not None:
            vmax = range_max

        # Creating the merge surface
        surf_merg = cltsurf.merge_surfaces([surf_lh, surf_rh])
        colorbar_size = self.figure_conf["colorbar_size"]

        # Check if the map_name is available in any of the surfaces
        for surf in [surf_lh, surf_rh, surf_merg]:
            available_maps = list(surf.mesh.point_data.keys())
            if map_name not in available_maps:
                raise ValueError(
                    f"Map '{map_name}' not found in one of the provided surfaces. Available maps: {available_maps}"
                )

        if colorbar_title is None:
            colorbar_title = map_name

        # Get view configuration
        view_ids = visutils.get_views_to_plot(self, views, ["lh", "rh"])

        # Determine rendering mode based on save_path, environment, and threading preference
        save_mode, use_off_screen, use_notebook, use_threading = (
            visutils.determine_render_mode(save_path, notebook, non_blocking)
        )

        # Detecting the screen size for the plotter
        screen_size = cltplot.get_current_monitor_size()

        config_dict, colorbar_dict_list = vislayout.hemispheres_layout(
            surf_lh,
            surf_rh,
            surf_merg,
            view_ids,
            map_name,
            vmin,
            vmax,
            colormap,
            colorbar_title=colorbar_title,
            colorbar=colorbar,
            colorbar_position=colorbar_position,
            colorbar_size=colorbar_size,
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
            "border": self.figure_conf.get("subplot_border", True),
        }

        groups = config_dict["groups"]
        if groups:
            plotter_kwargs["groups"] = groups

        pv_plotter = pv.Plotter(**plotter_kwargs)

        brain_positions = config_dict["brain_positions"]
        map_limits = config_dict["colormap_limits"]
        for (map_idx, obj_idx, view_idx), (row, col) in brain_positions.items():
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
            vmin, vmax, map_name = map_limits[0, 0, view_idx]

            # Add the brain surface mesh
            if "lh" in view_ids[view_idx]:
                surf = copy.deepcopy(surf_lh)

            elif "rh" in view_ids[view_idx]:
                surf = copy.deepcopy(surf_rh)

            elif "merg" in view_ids[view_idx]:
                surf = copy.deepcopy(surf_merg)

            surf = visutils.prepare_obj_for_plotting(
                surf,
                map_name,
                colormap,
                vmin=vmin,
                vmax=vmax,
                range_min=range_min,
                range_max=range_max,
                range_color=range_color,
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

    ###############################################################################################
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

    ###############################################################################################
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
