"""
Brain Surface Visualization Tools

This module provides comprehensive tools for visualizing brain surfaces with various
anatomical views and data overlays. It supports FreeSurfer surface formats and
provides flexible layout options for publication-quality figures.

Classes:
    DefineLayout: Main class for creating multi-view brain surface layouts
"""

import os
import numpy as np
import nibabel as nib
from typing import Union, List, Optional, Tuple, Dict, Any, TYPE_CHECKING
from nilearn import plotting
import pyvista as pv

# Importing local modules
from . import freesurfertools as cltfree

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from . import surfacetools as cltsurf


class DefineLayout:
    """
    A comprehensive class for creating multi-view layouts of brain surface visualizations.

    This class handles the creation of publication-quality brain surface figures with
    multiple anatomical views, colorbars, titles, and synchronized camera controls.
    It supports both single and multiple hemisphere visualizations with flexible
    layout options.

    Attributes
    ----------
    meshes : List[cltsurf.Surface]
        List of Surface objects to visualize
    views : List[str]
        List of anatomical views to display
    shape : Tuple[int, int]
        Shape of the subplot grid (rows, cols)
    layout : List[Dict]
        Layout configuration for each subplot
    colorbar : Optional[Dict]
        Colorbar configuration
    title : Optional[Dict]
        Title configuration
    lh_viewdict : Dict
        Left hemisphere camera positions
    rh_viewdict : Dict
        Right hemisphere camera positions
    """

    def __init__(
        self,
        meshes: Union[str, "cltsurf.Surface", List[Union[str, "cltsurf.Surface"]]],
        both: bool = False,
        views: Union[str, List[str]] = "all",
        showtitle: bool = False,
        showcolorbar: bool = False,
        colorbar_title: Optional[str] = None,
        ambient: float = 0.2,
        diffuse: float = 0.5,
        specular: float = 0.5,
        specular_power: int = 50,
        opacity: float = 1.0,
        style: str = "surface",
        smooth_shading: bool = True,
    ) -> None:
        """
        Initialize the DefineLayout class for brain surface visualization.

        Parameters
        ----------
        meshes : str, Surface, or list of such
            Surface file paths or Surface objects to visualize. Can be:
            - Single surface file path (str)
            - Single Surface object
            - List of surface file paths or Surface objects
        both : bool, default False
            If True, assumes bilateral visualization (not currently implemented)
        views : str or list of str, default "all"
            Anatomical views to display. Options:
            - "all": All available views
            - Single view: "lateral", "medial", "dorsal", "ventral", "rostral", "caudal"
            - List of views: ["lateral", "medial"]
        showtitle : bool, default False
            Whether to show a title row in the layout
        showcolorbar : bool, default False
            Whether to show a colorbar for scalar data
        colorbar_title : str, optional
            Title for the colorbar. If None and colorbar is shown, uses "Values"
        ambient : float, default 0.2
            Ambient lighting coefficient (0.0-1.0)
        diffuse : float, default 0.5
            Diffuse lighting coefficient (0.0-1.0)
        specular : float, default 0.5
            Specular lighting coefficient (0.0-1.0)
        specular_power : int, default 50
            Specular power for surface shininess
        opacity : float, default 1.0
            Surface opacity (0.0-1.0)
        style : str, default "surface"
            Surface rendering style ("surface", "wireframe", "points")
        smooth_shading : bool, default True
            Whether to use smooth shading

        Raises
        ------
        ValueError
            If invalid views are specified
        TypeError
            If meshes parameter is of incorrect type

        Examples
        --------
        >>> # Single surface, single view
        >>> layout = DefineLayout("lh.pial", views="lateral")
        >>> layout.plot()

        >>> # Multiple views with colorbar
        >>> surf = Surface("lh.pial")
        >>> surf.load_map("thickness.mgh")
        >>> layout = DefineLayout([surf], views=["lateral", "medial"],
        ...                      showcolorbar=True, colorbar_title="Thickness (mm)")
        >>> layout.plot()

        >>> # All views for publication
        >>> layout = DefineLayout("lh.pial", views="all", showtitle=True)
        >>> layout.plot()
        """
        # Handle different input types for meshes
        # Import here to avoid circular import
        from . import surfacetools as cltsurf

        if isinstance(meshes, str):
            meshes = [cltsurf.Surface(surface_file=meshes)]

        if isinstance(meshes, cltsurf.Surface):
            meshes = [meshes]
        elif isinstance(meshes, list):
            processed_meshes = []
            for mesh in meshes:
                if isinstance(mesh, str):
                    processed_meshes.append(cltsurf.Surface(surface_file=mesh))
                elif isinstance(mesh, cltsurf.Surface):
                    processed_meshes.append(mesh)
                else:
                    raise TypeError(f"Invalid mesh type: {type(mesh)}")
            meshes = processed_meshes
        else:
            raise TypeError(
                f"meshes must be str, Surface, or list of such, got {type(meshes)}"
            )

        self.meshes = meshes
        self.showtitle = showtitle
        self.showcolorbar = showcolorbar
        self.colorbar_title = colorbar_title or "Values"

        # Store rendering parameters (exclude intensity as it's not used by PyVista add_mesh)
        self._render_params = {
            "ambient": ambient,
            "diffuse": diffuse,
            "specular": specular,
            "specular_power": specular_power,
            "opacity": opacity,
            "style": style,
            "smooth_shading": smooth_shading,
        }

        # Define camera positions for each hemisphere
        self.lh_viewdict = {
            "lateral": {"position": "yz", "azimuth": 180, "elevation": 0},
            "dorsal": {"position": "xy", "azimuth": 0, "elevation": 0},
            "rostral": {"position": "xz", "azimuth": 180, "elevation": 0},
            "caudal": {"position": "xz", "azimuth": 0, "elevation": 0},
            "ventral": {"position": "xy", "azimuth": 180, "elevation": 0},
            "medial": {"position": "yz", "azimuth": 0, "elevation": 0},
        }

        self.rh_viewdict = {
            "lateral": {"position": "yz", "azimuth": 0, "elevation": 0},
            "dorsal": {"position": "xy", "azimuth": 0, "elevation": 0},
            "rostral": {"position": "xz", "azimuth": 180, "elevation": 0},
            "caudal": {"position": "xz", "azimuth": 0, "elevation": 0},
            "ventral": {"position": "xy", "azimuth": 180, "elevation": 0},
            "medial": {"position": "yz", "azimuth": 180, "elevation": 0},
        }

        # Process and validate views
        self.views = self._process_views(views)
        if not self.views:
            raise ValueError("No valid views specified")

        # Build layout configuration
        self._build_layout()

    def _process_views(self, views: Union[str, List[str]]) -> List[str]:
        """
        Process and validate the requested views.

        Parameters
        ----------
        views : str or list of str
            Views to process

        Returns
        -------
        list of str
            Validated and ordered list of views

        Raises
        ------
        ValueError
            If invalid views are specified
        """
        available_views = list(self.lh_viewdict.keys())

        if isinstance(views, str):
            if views == "all":
                processed_views = available_views.copy()
            elif views in available_views:
                processed_views = [views]
            else:
                raise ValueError(
                    f"Invalid view '{views}'. Available: {available_views}"
                )

        elif isinstance(views, list):
            if "all" in views:
                processed_views = available_views.copy()
            else:
                processed_views = []
                invalid_views = []

                for view in views:
                    if view in available_views:
                        processed_views.append(view)
                    else:
                        invalid_views.append(view)

                if invalid_views:
                    print(
                        f"Warning: Invalid views {invalid_views} removed. Available: {available_views}"
                    )
        else:
            raise TypeError(f"views must be str or list of str, got {type(views)}")

        # Maintain consistent ordering
        ordered_views = [v for v in available_views if v in processed_views]

        print(f"Views to plot: {ordered_views}")
        return ordered_views

    def _build_layout(self) -> None:
        """
        Build the subplot layout configuration.

        This method calculates the grid shape, weights, groups, and subplot
        positions based on the number of views, surfaces, and display options.
        """
        n_views = len(self.views)

        # Process surfaces and create mesh links
        surfs2plot = []
        mesh_links = []
        for s, mesh in enumerate(self.meshes):
            if isinstance(mesh, list):
                for submesh in mesh:
                    surfs2plot.append(submesh)
                    mesh_links.append(s)
            else:
                mesh_links.append(s)
                surfs2plot.append(mesh)

        # Determine number of mesh rows
        if "lateral" not in self.views and "medial" not in self.views:
            n_meshes = max(mesh_links) + 1 if mesh_links else 1
        else:
            n_meshes = len(surfs2plot)

        # Calculate layout parameters
        self._calculate_layout_params(n_views, n_meshes)

        # Create subplot layout
        self._create_subplot_layout(n_views, n_meshes, surfs2plot, mesh_links)

    def _calculate_layout_params(self, n_views: int, n_meshes: int) -> None:
        """Calculate grid shape, weights, and groups for the layout."""
        if len(self.views) == 1:
            # Single view layout
            if self.showcolorbar and self.showtitle:
                self.row_offset = 1
                self.shape = (3, n_meshes)
                self.row_weights = [0.2, 1, 0.3]
                self.col_weights = [1] * n_meshes
                self.groups = [(0, slice(None)), (2, slice(None))]
            elif self.showcolorbar and not self.showtitle:
                self.row_offset = 0
                self.shape = (2, n_meshes)
                self.row_weights = [1, 0.3]
                self.col_weights = [1] * n_meshes
                self.groups = [(1, slice(None))]
            elif self.showtitle and not self.showcolorbar:
                self.row_offset = 1
                self.shape = (2, n_meshes)
                self.row_weights = [0.2, 1]
                self.col_weights = [1] * n_meshes
                self.groups = [(0, slice(None))]
            else:
                self.row_offset = 0
                self.shape = (1, n_meshes)
                self.row_weights = [1]
                self.col_weights = [1] * n_meshes
                self.groups = None
        else:
            # Multiple view layout
            if self.showcolorbar and self.showtitle:
                self.row_offset = 1
                self.col_offset = 0
                self.shape = (n_meshes + 2, n_views)
                self.row_weights = [0.2] + [1] * n_meshes + [0.3]
                self.col_weights = [1] * n_views
                self.groups = [(0, slice(None)), (n_meshes + 1, slice(None))]
            elif self.showcolorbar and not self.showtitle:
                self.row_offset = 0
                self.col_offset = 0
                self.shape = (n_meshes + 1, n_views)
                self.row_weights = [1] * n_meshes + [0.3]
                self.col_weights = [1] * n_views
                self.groups = [(n_meshes, slice(None))]
            elif self.showtitle and not self.showcolorbar:
                self.row_offset = 1
                self.col_offset = 0
                self.shape = (n_meshes + 1, n_views)
                self.row_weights = [0.2] + [1] * n_meshes
                self.col_weights = [1] * n_views
                self.groups = [(0, slice(None))]
            else:
                self.row_offset = 0
                self.col_offset = 0
                self.shape = (n_meshes, n_views)
                self.row_weights = [1] * n_meshes
                self.col_weights = [1] * n_views
                self.groups = None

    def _create_subplot_layout(
        self, n_views: int, n_meshes: int, surfs2plot: List, mesh_links: List[int]
    ) -> None:
        """Create the detailed subplot layout configuration."""
        # Import here to avoid circular import
        from . import surfacetools as cltsurf

        layout = []

        if len(self.views) == 1:
            # Single view layout
            view = self.views[0]
            for col in range(n_meshes):
                mesh2plot = self._get_mesh_for_subplot(
                    view, col, surfs2plot, mesh_links
                )
                camdict = self._get_camera_dict(mesh2plot)

                subp_dic = {
                    "subp": {"row": self.row_offset, "col": col},
                    "surf2plot": mesh2plot,
                    "text": {
                        "label": view.capitalize(),
                        "position": "upper_edge",
                        "font_size": 12,
                        "color": "black",
                        "font": "arial",
                        "shadow": False,
                    },
                    "camera": {
                        "position": camdict[view]["position"],
                        "azimuth": camdict[view]["azimuth"],
                        "elevation": camdict[view]["elevation"],
                    },
                }
                layout.append(subp_dic)

        else:
            # Multiple view layout
            for row in range(n_meshes):
                for col in range(n_views):
                    view = self.views[col]
                    mesh2plot = self._get_mesh_for_subplot(
                        view, row, surfs2plot, mesh_links
                    )
                    camdict = self._get_camera_dict(mesh2plot)

                    subp_dic = {
                        "subp": {
                            "row": row + self.row_offset,
                            "col": col + getattr(self, "col_offset", 0),
                        },
                        "surf2plot": mesh2plot,
                        "text": {
                            "label": view.capitalize(),
                            "position": "upper_edge",
                            "font_size": 12,
                            "color": "black",
                            "font": "arial",
                            "shadow": False,
                        },
                        "camera": {
                            "position": camdict[view]["position"],
                            "azimuth": camdict[view]["azimuth"],
                            "elevation": camdict[view]["elevation"],
                        },
                    }
                    layout.append(subp_dic)

        self.layout = layout

        # Set colorbar and title positions
        if self.showcolorbar:
            if len(self.views) == 1:
                colorbar_row = self.row_offset + 1
            else:
                colorbar_row = n_meshes + self.row_offset
            # Center the colorbar
            colorbar_col = (n_views - 1) // 2 if n_views > 1 else 0
            self.colorbar = {"colorbar": {"row": colorbar_row, "col": colorbar_col}}
        else:
            self.colorbar = None

        # Don't create title subplot for now - use PyVista's built-in title
        self.title = None

    def _get_mesh_for_subplot(
        self, view: str, index: int, surfs2plot: List, mesh_links: List[int]
    ) -> Union["cltsurf.Surface", List["cltsurf.Surface"]]:
        """Get the appropriate mesh(es) for a specific subplot."""
        # For lateral and medial views, we always use individual surfaces
        return surfs2plot[index] if index < len(surfs2plot) else surfs2plot[0]

    def _get_camera_dict(
        self, mesh2plot: Union["cltsurf.Surface", List["cltsurf.Surface"]]
    ) -> Dict:
        """Determine the appropriate camera dictionary based on hemisphere."""
        # Import here to avoid circular import
        from . import surfacetools as cltsurf

        if isinstance(mesh2plot, list):
            if mesh2plot and isinstance(mesh2plot[0], cltsurf.Surface):
                hemi = mesh2plot[0].hemi
            else:
                return self.lh_viewdict
        elif isinstance(mesh2plot, cltsurf.Surface):
            hemi = mesh2plot.hemi
        else:
            return self.lh_viewdict

        if hemi and hemi.startswith("rh"):
            return self.rh_viewdict
        else:
            return self.lh_viewdict

    def plot(
        self,
        link_views: bool = False,
        window_size: Tuple[int, int] = (1400, 900),
        background_color: str = "white",
        title: Optional[str] = None,
        show_borders: bool = False,  # Add this parameter
        **kwargs,
    ) -> pv.Plotter:
        """
        Plot the brain surface layout with all specified views.

        Parameters
        ----------
        link_views : bool, default False
            Whether to link camera movements between views for synchronized
            rotation and zooming
        window_size : tuple of int, default (1400, 900)
            Window size as (width, height) in pixels
        background_color : str, default "white"
            Background color for the visualization
        title : str, optional
            Main title for the entire figure
        **kwargs
            Additional rendering parameters that override defaults

        Returns
        -------
        pv.Plotter
            The PyVista plotter object for further customization

        Raises
        ------
        RuntimeError
            If no surfaces are available to plot

        Examples
        --------
        >>> layout = DefineLayout("lh.pial", views=["lateral", "medial"])
        >>> plotter = layout.plot(link_views=True, title="Brain Surface")
        >>> # plotter.show()  # Called automatically

        >>> # Custom rendering parameters
        >>> layout.plot(ambient=0.3, opacity=0.8, style="wireframe")
        """

        # Update rendering parameters with any kwargs
        render_params = self._render_params.copy()
        render_params.update(kwargs)

        # Create plotter
        pl = pv.Plotter(
            shape=self.shape,
            row_weights=self.row_weights,
            col_weights=self.col_weights,
            groups=self.groups,
            notebook=False,
            window_size=window_size,
            border=show_borders,  # Use the parameter
        )

        # Set background
        pl.background_color = background_color

        # Track colors and values for colorbar
        all_colors = []
        all_values = []
        subplot_refs = []  # For view linking

        # Plot each subplot
        for subp in self.layout:
            pl.subplot(subp["subp"]["row"], subp["subp"]["col"])

            # Add view label
            pl.add_text(
                subp["text"]["label"],
                position=subp["text"]["position"],
                font_size=subp["text"]["font_size"],
                color=subp["text"]["color"],
                font=subp["text"]["font"],
                shadow=subp["text"]["shadow"],
            )

            # Import here to avoid circular import
            from . import surfacetools as cltsurf

            # Plot surfaces
            if isinstance(subp["surf2plot"], list):
                for surf in subp["surf2plot"]:
                    if isinstance(surf, cltsurf.Surface):
                        self._add_surface_to_plot(
                            pl, surf, render_params, all_colors, all_values
                        )
            else:
                if isinstance(subp["surf2plot"], cltsurf.Surface):
                    self._add_surface_to_plot(
                        pl, subp["surf2plot"], render_params, all_colors, all_values
                    )

            # Only remove scalar bar if it exists
            try:
                pl.remove_scalar_bar()
            except (StopIteration, KeyError, AttributeError):
                pass

            # Set camera position with debugging
            view_name = subp["text"]["label"]
            cam_pos = subp["camera"]["position"]
            azimuth = subp["camera"]["azimuth"]
            elevation = subp["camera"]["elevation"]

            print(
                f"Setting view '{view_name}': position={cam_pos}, azimuth={azimuth}°, elevation={elevation}°"
            )

            # Reset camera first
            pl.reset_camera()

            # Set base orientation
            if cam_pos == "yz":
                pl.view_yz()
            elif cam_pos == "xy":
                pl.view_xy()
            elif cam_pos == "xz":
                pl.view_xz()

            # Apply specific rotations for different views
            if view_name.lower() == "lateral":
                if hasattr(subp["surf2plot"], "hemi") and subp[
                    "surf2plot"
                ].hemi.startswith("lh"):
                    pl.camera.azimuth = 180  # Left lateral
                else:
                    pl.camera.azimuth = 0  # Right lateral
            elif view_name.lower() == "medial":
                if hasattr(subp["surf2plot"], "hemi") and subp[
                    "surf2plot"
                ].hemi.startswith("lh"):
                    pl.camera.azimuth = 0  # Left medial
                else:
                    pl.camera.azimuth = 180  # Right medial
            else:
                # Apply the specified rotations
                pl.camera.azimuth = azimuth
                pl.camera.elevation = elevation

            # Store subplot reference for linking
            if link_views:
                subplot_refs.append(pl.camera)

        # Link views if requested
        if link_views and len(subplot_refs) > 1:
            self._link_camera_views(pl)

        # Add colorbar if requested and data is available
        if self.colorbar is not None and all_colors and all_values:
            self._add_colorbar(pl, all_colors[0], all_values)

        # Add main title using PyVista's window title approach
        if title:
            # Set window title
            pl.title = title

        pl.show()
        return pl

    def _add_surface_to_plot(
        self,
        pl: pv.Plotter,
        surf: "cltsurf.Surface",
        render_params: Dict,
        all_colors: List,
        all_values: List,
    ) -> None:
        """Add a single surface to the current subplot."""
        if "vertex_colors" in surf.mesh.point_data:
            # Surface with scalar data
            if np.shape(surf.mesh.point_data["vertex_colors"])[1] == 3:
                # RGB colors
                pl.add_mesh(
                    surf.mesh, scalars="vertex_colors", rgb=True, **render_params
                )
                all_colors.append(surf.mesh.point_data["vertex_colors"])
            # all_values.extend(surf.mesh["values"])
        else:
            # Plain surface without scalar data
            pl.add_mesh(surf.mesh, **render_params)

    def _link_camera_views(self, pl: pv.Plotter) -> None:
        """Link camera movements between all subplots."""
        try:
            pl.link_views()
            print("Views linked for synchronized navigation")
        except AttributeError:
            print("Warning: View linking not available in this PyVista version")

    def _add_colorbar(
        self, pl: pv.Plotter, color_scheme: Any, values: List[float]
    ) -> None:
        """Add a centered colorbar with proper title."""
        colorbar_row = self.colorbar["colorbar"]["row"]
        colorbar_col = self.colorbar["colorbar"]["col"]

        print(f"Adding colorbar to subplot ({colorbar_row}, {colorbar_col})")
        pl.subplot(colorbar_row, colorbar_col)

        # Calculate value range
        scalar_range = (np.min(values), np.max(values))

        # Create invisible mesh for colorbar
        dummy_mesh = pv.Sphere(radius=0.001)  # Very small sphere
        dummy_mesh["values"] = np.linspace(
            scalar_range[0], scalar_range[1], dummy_mesh.n_points
        )

        # Add invisible mesh to generate colorbar
        pl.add_mesh(
            dummy_mesh,
            scalars="values",
            cmap=color_scheme,
            show_edges=False,
            opacity=0.0,
            scalar_bar_args={
                "title": self.colorbar_title,
                "title_font_size": 14,
                "label_font_size": 11,
                "shadow": True,
                "n_labels": 5,
                "italic": False,
                "fmt": "%.2f",
                "position_x": 0.1,
                "position_y": 0.1,
                "width": 0.8,
                "height": 0.8,
            },
        )

        # Hide axes for colorbar subplot
        pl.hide_axes()
        print(f"Colorbar added to subplot ({colorbar_row}, {colorbar_col})")

    def print_available_views(self, hemisphere: str = "both") -> None:
        """
        Print available views with colorized output showing camera orientations.

        Parameters
        ----------
        hemisphere : str, default "both"
            Which hemisphere views to show. Options:
            - "both": Show both hemispheres (default)
            - "lh" or "left": Show only left hemisphere
            - "rh" or "right": Show only right hemisphere

        Examples
        --------
        >>> layout = DefineLayout("lh.pial")
        >>> layout.print_available_views()
        >>> layout.print_available_views("rh")
        """
        # ANSI color codes
        colors = {
            "header": "\033[95m",  # Magenta
            "view_name": "\033[94m",  # Blue
            "position": "\033[92m",  # Green
            "azimuth": "\033[93m",  # Yellow
            "elevation": "\033[91m",  # Red
            "reset": "\033[0m",  # Reset
            "bold": "\033[1m",  # Bold
        }

        print(
            f"\n{colors['header']}{colors['bold']}Available Brain Surface Views{colors['reset']}"
        )
        print("=" * 50)

        if hemisphere.lower() in ["lh", "left", "both"]:
            print(
                f"\n{colors['header']}{colors['bold']}Left Hemisphere (LH) Views:{colors['reset']}"
            )
            print("-" * 30)
            for view_name, params in self.lh_viewdict.items():
                print(
                    f"{colors['view_name']}{colors['bold']}{view_name.upper():>8}{colors['reset']}: "
                    f"{colors['position']}pos={params['position']:<2}{colors['reset']} | "
                    f"{colors['azimuth']}azimuth={params['azimuth']:>3}°{colors['reset']} | "
                    f"{colors['elevation']}elevation={params['elevation']:>2}°{colors['reset']}"
                )

        if hemisphere.lower() in ["rh", "right", "both"]:
            print(
                f"\n{colors['header']}{colors['bold']}Right Hemisphere (RH) Views:{colors['reset']}"
            )
            print("-" * 30)
            for view_name, params in self.rh_viewdict.items():
                print(
                    f"{colors['view_name']}{colors['bold']}{view_name.upper():>8}{colors['reset']}: "
                    f"{colors['position']}pos={params['position']:<2}{colors['reset']} | "
                    f"{colors['azimuth']}azimuth={params['azimuth']:>3}°{colors['reset']} | "
                    f"{colors['elevation']}elevation={params['elevation']:>2}°{colors['reset']}"
                )

        print(f"\n{colors['bold']}Usage Examples:{colors['reset']}")
        print(
            f"  Single view:     {colors['view_name']}views='lateral'{colors['reset']}"
        )
        print(
            f"  Multiple views:  {colors['view_name']}views=['lateral', 'medial']{colors['reset']}"
        )
        print(f"  All views:       {colors['view_name']}views='all'{colors['reset']}")

        print(f"\n{colors['bold']}Camera Parameters:{colors['reset']}")
        print(
            f"  {colors['position']}position{colors['reset']}: Camera coordinate system (xy=top, xz=front, yz=side)"
        )
        print(
            f"  {colors['azimuth']}azimuth{colors['reset']}:  Horizontal rotation angle (degrees)"
        )
        print(
            f"  {colors['elevation']}elevation{colors['reset']}: Vertical rotation angle (degrees)"
        )
        print()

    def get_layout_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current layout configuration.

        Returns
        -------
        dict
            Dictionary containing layout information including:
            - shape: Grid shape (rows, cols)
            - views: List of views being displayed
            - n_surfaces: Number of surfaces
            - has_colorbar: Whether colorbar is shown
            - has_title: Whether title is shown

        Examples
        --------
        >>> layout = DefineLayout("lh.pial", views=["lateral", "medial"])
        >>> info = layout.get_layout_info()
        >>> print(f"Grid shape: {info['shape']}")
        """
        return {
            "shape": self.shape,
            "views": self.views,
            "n_surfaces": len(self.meshes),
            "has_colorbar": self.showcolorbar,
            "has_title": self.showtitle,
            "colorbar_title": self.colorbar_title if self.showcolorbar else None,
            "row_weights": self.row_weights,
            "col_weights": self.col_weights,
            "n_subplots": len(self.layout),
        }
