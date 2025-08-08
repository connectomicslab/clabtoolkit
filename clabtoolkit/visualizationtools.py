"""
Brain Surface Visualization Tools

This module provides comprehensive tools for visualizing brain surfaces with various
anatomical views and data overlays. It supports FreeSurfer surface formats and
provides flexible layout options for publication-quality figures.

Classes:
    DefineLayout: Main class for creating multi-view brain surface layouts
"""

import os
import json
import math
import numpy as np
import nibabel as nib
from typing import Union, List, Optional, Tuple, Dict, Any, TYPE_CHECKING
from nilearn import plotting
import pyvista as pv
import tkinter as tk
import tkinter as tk

# Importing local modules
from . import freesurfertools as cltfree
from . import misctools as cltmisc


# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from . import surfacetools as cltsurf


def get_screen_size() -> Tuple[int, int]:
    """
    Get the current screen size in pixels.

    Returns
    -------
    tuple of int
        Screen width and height in pixels

    Examples
    --------
    >>> width, height = get_screen_size()
    >>> print(f"Screen size: {width}x{height}")
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    return width, height


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

class SurfacePlotter:
    """
    A comprehensive brain surface visualization tool using PyVista.
    
    This class provides flexible brain plotting capabilities with multiple view configurations,
    customizable colormaps, and optional colorbar support for neuroimaging data visualization.
    
    Attributes:
        config_file (str): Path to the JSON configuration file containing layout definitions
        figure_conf (dict): Loaded figure configuration with styling settings
        views_conf (dict): Loaded views configuration with layout definitions
        
    Examples:
        >>> plotter = SurfacePlotter("brain_plot_configs.json")
        >>> plotter.create_plot(surf_lh, surf_rh, map_name="thickness", 
        ...                     views="8_views", colorbar=True)
        >>> # New dynamic view selection
        >>> plotter.create_plot(surf_lh, surf_rh, views=["Lateral", "Medial", "Dorsal"])
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize the SurfacePlotter with configuration file.
        
        Parameters:
            config_file (str, optional): Path to JSON file containing figure and view configurations.
                                    Defaults to "brain_plot_configs.json"
                            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the configuration file is not valid JSON
            KeyError: If required configuration keys 'figure_conf' or 'views_conf' are missing
        
        
        Examples:
            >>> plotter = SurfacePlotter()  # Use default config file
            >>> plotter = SurfacePlotter("custom_views.json")  # Use custom config
        """
        # Load configuration file

        # Get the absolute of this file
        cwd = os.path.dirname(os.path.abspath(__file__))

        if config_file is None:
            # Default to the standard configuration file
            config_file = os.path.join(cwd, "config", "viz_views.json")
        else:
            # Use the provided configuration file path
            config_file = os.path.abspath(config_file)
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found")

        self.config_file = config_file
        self._load_configs()
        
        # Define mapping from simple view names to configuration titles
        self._view_name_mapping = {
            "lateral": ["LH: Lateral view", "RH: Lateral view"],
            "medial": ["LH: Medial view", "RH: Medial view"],
            "dorsal": ["Dorsal view"],
            "ventral": ["Ventral view"],
            "rostral": ["Rostral view"],
            "caudal": ["Caudal view"]
        }
    
    def _load_configs(self) -> None:
        """
        Load figure and view configurations from JSON file.

        Parameters:
            None

        Returns:
            None: Configurations are loaded into instance attributes

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the configuration file contains invalid JSON
            KeyError: If required configuration keys 'figure_conf' or 'views_conf' are missing

        Examples:
            >>> plotter = SurfacePlotter("configs.json")
            >>> plotter._load_configs()  # Reloads configurations from file
        """   
            
        try:
            with open(self.config_file, 'r') as f:
                configs = json.load(f)
            
            # Validate structure and load configurations
            if "figure_conf" not in configs:
                raise KeyError("Missing 'figure_conf' key in configuration file")
            if "views_conf" not in configs:
                raise KeyError("Missing 'views_conf' key in configuration file")
                
            self.figure_conf = configs["figure_conf"]
            self.views_conf = configs["views_conf"]
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in configuration file: {e}")
    
    def _create_dynamic_config(self, selected_views: List[str]) -> Dict[str, Any]:
        """
        Create a dynamic configuration based on selected view names.

        Parameters:
            selected_views (List[str]): List of view names to include

        Returns:
            Dict[str, Any]: Dynamic configuration with filtered views and optimal layout

        Raises:
            ValueError: If no valid views are found or invalid view names provided

        Examples:
            >>> config = plotter._create_dynamic_config(["lateral", "medial"])
            >>> print(config["shape"])  # [1, 2]
            >>> print(len(config["views"]))  # 4 (LH/RH lateral + LH/RH medial)
        """
        # Load base 8_views configuration
        if "8_views" not in self.views_conf:
            raise ValueError("Base '8_views' configuration not found in config file")
        
        base_config = self.views_conf["8_views"]
        base_views = base_config["views"]
        
        # Normalize input view names (case-insensitive)
        selected_views_lower = [view.lower() for view in selected_views]
        
        # Find matching views from base configuration
        filtered_views = []
        for view_name in selected_views_lower:
            if view_name not in self._view_name_mapping:
                available_views = list(self._view_name_mapping.keys())
                raise ValueError(f"Invalid view name '{view_name}'. "
                                f"Available options: {available_views}")
            
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
            new_height = int(new_width / aspect_ratio * optimal_shape[0] / optimal_shape[1])
        else:  # Taller than wide
            new_height = min(1200, 400 * optimal_shape[0])
            new_width = int(new_height * aspect_ratio * optimal_shape[1] / optimal_shape[0])
        
        # Create dynamic configuration
        dynamic_config = {
            "shape": optimal_shape,
            "window_size": [new_width, new_height],
            "views": filtered_views
        }
        
        return dynamic_config
    
    def _calculate_optimal_grid(self, num_views: int) -> List[int]:
        """
        Calculate optimal grid dimensions for a given number of views.

        Parameters:
            num_views (int): Number of views to arrange

        Returns:
            List[int]: [rows, columns] for optimal grid layout

        Raises:
            None: This method does not raise exceptions

        Examples:
            >>> plotter._calculate_optimal_grid(4)
            [2, 2]
            >>> plotter._calculate_optimal_grid(6)
            [2, 3]
            >>> plotter._calculate_optimal_grid(1)
            [1, 1]
        """
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
    
    def plot_hemispheres(self, 
                surf_lh: Any, 
                surf_rh: Any,
                views: Union[str, List[str]] = "8_views",
                notebook: bool = False,
                map_name: str = "surface",
                colormap: str = "BrBG",
                save_path: Optional[str] = None,
                colorbar: bool = True,
                colorbar_title: str = "Value",
                colorbar_position: str = "bottom") -> None:
        """
        Create brain surface plots with flexible configurations.
        
        This method generates multi-view brain surface visualizations with customizable
        layouts, coloring schemes, and optional colorbar display.
        
        Parameters:
            surf_lh (Any): Left hemisphere surface object with mesh data and point_data arrays
            surf_rh (Any): Right hemisphere surface object with mesh data and point_data arrays
            views (Union[str, List[str]], optional): Either a configuration name from JSON file 
                                                    (e.g., "8_views", "6_views") OR a list of view names 
                                                    to dynamically select (e.g., ["Lateral", "Medial", "Dorsal"]).
                                                    Available view names: "Lateral", "Medial", "Dorsal", 
                                                    "Ventral", "Rostral", "Caudal". Defaults to "8_views"
            notebook (bool, optional): If True, optimize for Jupyter notebook display.
                                        If False, create independent window. Defaults to False
            map_name (str, optional): Name of the data array in point_data to use for surface 
                                    coloring. Defaults to "surface"
            colormap (str, optional): Matplotlib colormap name (e.g., "bwr", "viridis", "hot", 
                                    "BuRd"). Defaults to "BuRd"
            save_path (str, optional): File path to save the figure (e.g., "brain_plot.png").
                                    If provided and directory exists, saves without displaying.
                                    If None, displays the plot. Defaults to None
            colorbar (bool, optional): If True, adds a common colorbar for all views. 
                                    Automatically set to False if map_name uses colortables.
                                    Defaults to True
            colorbar_title (str, optional): Title text for the colorbar. Defaults to "Value"
            colorbar_position (str, optional): Colorbar placement: "right", "left", "top", 
                                            or "bottom". Defaults to "bottom"
            
        Returns:
            None: The method either displays the plot or saves it to file
            
        Raises:
            KeyError: If views string is not found in the configuration file
            ValueError: If invalid view names are provided in views list or required data arrays are missing
            Exception: If screenshot saving fails (with fallback attempts)
            
        Examples:
            >>> # Basic usage with predefined configuration
            >>> plotter = SurfacePlotter("configs.json")
            >>> plotter.create_plot(surf_lh, surf_rh, views="8_views")
            
            >>> # Dynamic view selection with list of view names
            >>> plotter.create_plot(surf_lh, surf_rh, views=["Lateral", "Medial", "Dorsal"])
            
            >>> # Advanced usage with custom settings
            >>> plotter.create_plot(
            ...     surf_lh, surf_rh, 
            ...     views=["Lateral", "Medial"],
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
            print(f"Created dynamic configuration with {len(config['views'])} views: {views}")
        else:
            # Predefined configuration
            if views not in self.views_conf:
                available_configs = list(self.views_conf.keys())
                raise KeyError(f"Configuration '{views}' not found. "
                            f"Available options: {available_configs}")
            config = self.views_conf[views]
            config_name = views
        
        # Set colorbar to False if the map_name is on the colortable
        if map_name in surf_lh.colortables or map_name in surf_rh.colortables:
            colorbar = False
        
        # Extract vertex values for coloring
        try:
            vertex_values_lh = surf_lh.mesh.point_data[map_name]
            vertex_values_rh = surf_rh.mesh.point_data[map_name]
        except KeyError:
            raise ValueError(f"Data array '{map_name}' not found in surface point_data")

        # Calculate symmetric color range based on absolute maximum
        abs_val = np.max(np.abs(np.concatenate((vertex_values_lh, vertex_values_rh))))
        vmin = -abs_val
        vmax = abs_val

        # Process left hemisphere colors
        vertex_colors_lh = self._process_vertex_colors(
            surf_lh, vertex_values_lh, map_name, colormap, vmin, vmax
        )
        
        # Process right hemisphere colors  
        vertex_colors_rh = self._process_vertex_colors(
            surf_rh, vertex_values_rh, map_name, colormap, vmin, vmax
        )
        
        # Apply colors to mesh data
        surf_lh.mesh.point_data["vertex_colors"] = vertex_colors_lh
        surf_rh.mesh.point_data["vertex_colors"] = vertex_colors_rh
        
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
        surfaces = {
            "lh": surf_lh,
            "rh": surf_rh, 
            "merged": surf_merged
        }
        
        # Render each configured view
        actor_for_colorbar = self._create_views(
            plotter, config, surfaces, view_offset
        )
        
        # Add colorbar if requested
        if colorbar:
            self._add_colorbar(
                plotter, config, surf_merged, map_name, colormap, 
                colorbar_title, colorbar_position
            )
        
        # Execute final rendering/display
        self._finalize_plot(plotter, save_mode, save_path)

    def _process_vertex_colors(self, surface: Any, vertex_values: np.ndarray, 
                            map_name: str, colormap: str, 
                            vmin: float, vmax: float) -> np.ndarray:
        """
        Process vertex values into RGB colors using either colortables or colormaps.

        Parameters:
            surface (Any): Surface object containing colortables
            vertex_values (np.ndarray): Array of values to be colored
            map_name (str): Name of the data map
            colormap (str): Matplotlib colormap name
            vmin (float): Minimum value for color range
            vmax (float): Maximum value for color range

        Returns:
            np.ndarray: RGB color array for vertices

        Raises:
            KeyError: If map_name is not found in surface data or colortables
            ValueError: If vertex_values array is invalid or empty

        Examples:
            >>> colors = plotter._process_vertex_colors(
            ...     surf_lh, thickness_values, "thickness", "viridis", 0.0, 5.0
            ... )
            >>> print(colors.shape)  # (n_vertices, 3) for RGB values
        """
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

    def _determine_render_mode(self, save_path: Optional[str], 
                            notebook: bool) -> Tuple[bool, bool, bool]:
        """
        Determine rendering parameters based on save path and environment.

        Parameters:
            save_path (Optional[str]): File path for saving the figure, or None for display
            notebook (bool): Whether running in Jupyter notebook environment

        Returns:
            Tuple[bool, bool, bool]: (save_mode, use_off_screen, use_notebook)

        Raises:
            None: This method does not raise exceptions but prints warnings for invalid paths

        Examples:
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
                print(f"Warning: Directory '{save_dir}' does not exist. "
                    f"Displaying plot instead of saving.")
                return False, False, notebook
        else:
            # Display mode
            return False, False, notebook

    def _setup_plotter(self, config: Dict[str, Any], colorbar: bool, 
                    colorbar_position: str, use_notebook: bool, 
                    use_off_screen: bool) -> Tuple[pv.Plotter, Tuple[int, int]]:
        """
        Setup PyVista plotter with appropriate grid layout for colorbar.

        Parameters:
            config (Dict[str, Any]): View configuration containing shape and window_size
            colorbar (bool): Whether to reserve space for colorbar
            colorbar_position (str): Position of colorbar ("top", "bottom", "left", "right")
            use_notebook (bool): Whether to use notebook-optimized rendering
            use_off_screen (bool): Whether to use off-screen rendering

        Returns:
            Tuple[pv.Plotter, Tuple[int, int]]: (plotter instance, view_offset)

        Raises:
            ValueError: If colorbar_position is not one of the valid options
            Exception: If PyVista plotter creation fails

        Examples:
            >>> plotter, offset = self._setup_plotter(
            ...     config, True, "bottom", False, False
            ... )
            >>> print(offset)  # (0, 0) for bottom colorbar
        """
        if colorbar:
            original_shape = config["shape"]
            
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
                border=False
            )
        else:
            # Create standard plotter
            plotter = pv.Plotter(
                notebook=use_notebook,
                off_screen=use_off_screen,
                window_size=config["window_size"],
                shape=config["shape"],
                border=False
            )
            view_offset = (0, 0)
        
        return plotter, view_offset

    def _create_views(self, plotter: pv.Plotter, config: Dict[str, Any], 
                     surfaces: Dict[str, Any], view_offset: Tuple[int, int]) -> Any:
        """
        Create all configured brain views in the plotter.

        Parameters:
            plotter (pv.Plotter): PyVista plotter instance
            config (Dict[str, Any]): View configuration with views list
            surfaces (Dict[str, Any]): Dictionary containing surface objects
            view_offset (Tuple[int, int]): Offset for subplot positioning due to colorbar

        Returns:
            Any: Actor reference for colorbar creation

        Raises:
            KeyError: If required surface mesh is not found in surfaces dictionary
            ValueError: If view configuration contains invalid parameters

        Examples:
            >>> surfaces = {"lh": surf_lh, "rh": surf_rh, "merged": surf_merged}
            >>> actor = self._create_views(plotter, config, surfaces, (0, 0))
            >>> print(type(actor))  # PyVista actor object
        """
        actor_for_colorbar = None
        
        for view_config in config["views"]:
            # Apply offset for colorbar space
            subplot_pos = (view_config["subplot"][0] + view_offset[0], 
                          view_config["subplot"][1] + view_offset[1])
            plotter.subplot(*subplot_pos)
            
            # Set background color from figure configuration
            plotter.set_background(self.figure_conf["background_color"])
            
            # Add view title using figure configuration
            plotter.add_text(
                view_config["title"], 
                font_size=self.figure_conf["title_font_size"], 
                position='upper_edge', 
                color=self.figure_conf["title_font_color"], 
                shadow=self.figure_conf["title_shadow"],
                font=self.figure_conf["title_font_type"]
            )
            
            # Add brain mesh using figure configuration
            surface = surfaces[view_config["mesh"]]
            actor = plotter.add_mesh(
                surface.mesh,
                scalars="vertex_colors",
                rgb=True,
                ambient=self.figure_conf["mesh_ambient"],
                diffuse=self.figure_conf["mesh_diffuse"],
                specular=self.figure_conf["mesh_specular"],
                specular_power=self.figure_conf["mesh_specular_power"],
                smooth_shading=self.figure_conf["mesh_smooth_shading"],
                show_scalar_bar=False
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

    def _add_colorbar(self, plotter: pv.Plotter, config: Dict[str, Any], 
                    surf_merged: Any, map_name: str, colormap: str,
                    colorbar_title: str, colorbar_position: str) -> None:
        """
        Add a properly positioned colorbar to the plot.

        Parameters:
            plotter (pv.Plotter): PyVista plotter instance
            config (Dict[str, Any]): View configuration containing shape information
            surf_merged (Any): Merged surface object containing data for colorbar range
            map_name (str): Name of the data array to use for colorbar
            colormap (str): Matplotlib colormap name
            colorbar_title (str): Title text for the colorbar
            colorbar_position (str): Position of colorbar ("top", "bottom", "left", "right")

        Returns:
            None: Colorbar is added to the plotter in place

        Raises:
            KeyError: If map_name is not found in surf_merged point_data
            ValueError: If colorbar_position is invalid or data array is empty

        Examples:
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
        colorbar_mesh = pv.Line((0, 0, 0), (1, 0, 0), resolution=n_points-1)
        scalar_values = np.linspace(np.min(data_values), np.max(data_values), n_points)
        colorbar_mesh[map_name] = scalar_values
        
        # Add invisible mesh for colorbar reference
        dummy_actor = plotter.add_mesh(
            colorbar_mesh, 
            scalars=map_name,
            cmap=colormap,
            clim=[np.min(data_values), np.max(data_values)],
            show_scalar_bar=False
        )
        dummy_actor.visibility = False
        
        # Configure and add scalar bar using figure configuration
        scalar_bar_kwargs = {
            'color': self.figure_conf["colorbar_font_color"],
            'title': colorbar_title,
            'outline': self.figure_conf["colorbar_outline"],
            'title_font_size': self.figure_conf["colorbar_title_font_size"],
            'label_font_size': self.figure_conf["colorbar_font_size"],
            'n_labels': self.figure_conf["colorbar_n_labels"],
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

    def _finalize_plot(self, plotter: pv.Plotter, save_mode: bool, 
                    save_path: Optional[str]) -> None:
        """
        Handle final rendering - either save or display the plot.

        Parameters:
            plotter (pv.Plotter): PyVista plotter instance ready for final rendering
            save_mode (bool): If True, save the plot; if False, display it
            save_path (Optional[str]): File path for saving (required if save_mode is True)

        Returns:
            None: Plot is either saved to file or displayed

        Raises:
            Exception: If screenshot saving fails (with fallback attempts)
            IOError: If save path is invalid or write permissions are insufficient

        Examples:
            >>> self._finalize_plot(plotter, True, "brain_plot.png")
            # Saves plot to file and closes plotter
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

    def plot_surface(self, 
            surface: Any,
            hemi: str = "lh",
            views: Union[str, List[str]] = "8_views",
            map_name: str = "surface",
            colormap: str = "BrBG",
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            overwrite_colors: bool = False,
            notebook: bool = False,
            colorbar: bool = True,
            colorbar_title: str = "Value",
            colorbar_position: str = "bottom",
            save_path: Optional[str] = None) -> None:
        """
        Create brain surface plots for a single hemisphere with flexible configurations.
        
        This method generates multi-view brain surface visualizations for a single hemisphere
        with customizable layouts, coloring schemes, and optional colorbar display.
        
        Parameters:
            surface (Any): Surface object with mesh data and point_data arrays for the specified hemisphere
            hemi (str, optional): Hemisphere specification. Either "lh" for left hemisphere or "rh" for right hemisphere.
                                Defaults to "lh"
            views (Union[str, List[str]], optional): Either a configuration name from JSON file 
                                                (e.g., "8_views", "6_views") OR a list of view names 
                                                to dynamically select (e.g., ["Lateral", "Medial", "Dorsal"]).
                                                Available view names: "Lateral", "Medial", "Dorsal", 
                                                "Ventral", "Rostral", "Caudal". Defaults to "8_views"
            notebook (bool, optional): If True, optimize for Jupyter notebook display.
                                    If False, create independent window. Defaults to False
            map_name (str, optional): Name of the data array in point_data to use for surface 
                                    coloring. Defaults to "surface"
            colormap (str, optional): Matplotlib colormap name (e.g., "bwr", "viridis", "hot", 
                                    "BuRd"). Defaults to "BrBG"
            save_path (str, optional): File path to save the figure (e.g., "brain_plot.png").
                                    If provided and directory exists, saves without displaying.
                                    If None, displays the plot. Defaults to None
            colorbar (bool, optional): If True, adds a common colorbar for all views. 
                                    Automatically set to False if map_name uses colortables.
                                    Defaults to True
            colorbar_title (str, optional): Title text for the colorbar. Defaults to "Value"
            colorbar_position (str, optional): Colorbar placement: "right", "left", "top", 
                                            or "bottom". Defaults to "bottom"
            
        Returns:
            None: The method either displays the plot or saves it to file
            
        Raises:
            KeyError: If views string is not found in the configuration file
            ValueError: If invalid view names are provided in views list, invalid hemisphere specified,
                    or required data arrays are missing
            Exception: If screenshot saving fails (with fallback attempts)
            
        Examples:
            >>> # Basic usage with left hemisphere
            >>> plotter = SurfacePlotter("configs.json")
            >>> plotter.plot_surface(surf_lh, hemi="lh", views="8_views")
            
            >>> # Right hemisphere with dynamic view selection
            >>> plotter.plot_surface(surf_rh, hemi="rh", views=["Lateral", "Medial", "Dorsal"])
            
            >>> # Advanced usage with custom settings
            >>> plotter.plot_surface(
            ...     surf_lh, 
            ...     hemi="lh",
            ...     views=["Lateral", "Medial"],
            ...     map_name="cortical_thickness",
            ...     colorbar=True,
            ...     colorbar_title="Thickness (mm)",
            ...     colormap="viridis",
            ...     save_path="left_hemisphere_thickness.png"
            ... )
        """
        
        # Validate hemisphere parameter
        if hemi not in ["lh", "rh"]:
            raise ValueError(f"Invalid hemisphere '{hemi}'. Must be 'lh' (left) or 'rh' (right)")
        
        # Handle dynamic view selection vs predefined configurations
        if isinstance(views, list):
            # Dynamic view selection - filter views relevant to single hemisphere
            config = self._create_single_hemisphere_config(views, hemi)
            config_name = f"dynamic_{len(views)}_views_{hemi}"
            print(f"Created dynamic {hemi.upper()} hemisphere configuration with {len(config['views'])} views: {views}")
        else:
            # Predefined configuration - filter for single hemisphere
            if views not in self.views_conf:
                available_configs = list(self.views_conf.keys())
                raise KeyError(f"Configuration '{views}' not found. "
                            f"Available options: {available_configs}")
            config = self._filter_config_for_hemisphere(self.views_conf[views], hemi)
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
        surface.mesh.point_data["vertex_colors"] = vertex_colors
        
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
            "merged": surface  # For single hemisphere, merged is same as the surface
        }
        
        # Render each configured view
        actor_for_colorbar = self._create_views(
            plotter, config, surfaces, view_offset
        )
        
        # Add colorbar if requested
        if colorbar:
            self._add_colorbar(
                plotter, config, surface, map_name, colormap, 
                colorbar_title, colorbar_position
            )
        
        # Execute final rendering/display
        self._finalize_plot(plotter, save_mode, save_path)


    def _create_single_hemisphere_config(self, selected_views: List[str], hemi: str) -> Dict[str, Any]:
        """
        Create a dynamic configuration for single hemisphere based on selected view names.
        
        Parameters:
            selected_views (List[str]): List of view names to include
            hemi (str): Hemisphere specification ("lh" or "rh")
            
        Returns:
            Dict[str, Any]: Dynamic configuration with filtered views and optimal layout
            
        Raises:
            ValueError: If no valid views are found or invalid view names provided
            
        Examples:
            >>> config = plotter._create_single_hemisphere_config(["lateral", "medial"], "lh")
            >>> print(config["shape"])  # [1, 2]
            >>> print(len(config["views"]))  # 2 (LH lateral + LH medial)
            >>> print(config["window_size"])  # Same window size as base config
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
                raise ValueError(f"Invalid view name '{view_name}'. "
                            f"Available options: {available_views}")
            
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
            "views": filtered_views
        }
        
        return dynamic_config

    def _filter_config_for_hemisphere(self, config: Dict[str, Any], hemi: str) -> Dict[str, Any]:
        """
        Filter a predefined configuration to show only views relevant to a single hemisphere.
        
        Parameters:
            config (Dict[str, Any]): Original configuration with all views
            hemi (str): Hemisphere specification ("lh" or "rh")
            
        Returns:
            Dict[str, Any]: Filtered configuration for single hemisphere
            
        Raises:
            ValueError: If no valid views are found for the hemisphere
            
        Examples:
            >>> filtered = plotter._filter_config_for_hemisphere(config_8_views, "lh")
            >>> print(len(filtered["views"]))  # Reduced number of views for LH only
            >>> print(filtered["window_size"])  # Same window size as original config
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
            "views": filtered_views
        }
        
        return filtered_config

    def list_available_layouts(self) -> Dict[str, Dict[str, Any]]:
        """
        Display available visualization layouts and their configurations.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary containing detailed layout information 
                                    for each configuration. Keys are configuration names,
                                    values contain shape, window_size, num_views, and 
                                    views information.
            
        Examples:
            >>> plotter = SurfacePlotter("configs.json")
            >>> layouts = plotter.list_available_layouts()
            >>> print(f"Available layouts: {list(layouts.keys())}")
            >>> # Output: Available layouts: ['8_views', '6_views', '4_views_2x2', ...]
            
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
                    "view_type": view["view"]
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
                "views": layout_grid
            }
        
        print("\n" + "=" * 50)
        print("\n🎯 Dynamic View Selection:")
        print("   You can also use a list of view names for custom layouts:")
        print("   Available view names: Lateral, Medial, Dorsal, Ventral, Rostral, Caudal")
        print("   Example: views=['Lateral', 'Medial', 'Dorsal']")
        print("=" * 50)
        
        return layout_info

    def list_available_view_names(self) -> List[str]:
        """
        List available view names for dynamic view selection.
        
        Returns:
            List[str]: Available view names that can be used in views parameter
            
        Examples:
            >>> plotter = SurfacePlotter()
            >>> view_names = plotter.list_available_view_names()
            >>> print(f"Available views: {view_names}")
            >>> # Output: Available views: ['Lateral', 'Medial', 'Dorsal', 'Ventral', 'Rostral', 'Caudal']
        """
        view_names = list(self._view_name_mapping.keys())
        view_names_capitalized = [name.capitalize() for name in view_names]
        
        print("🧠 Available View Names for Dynamic Selection:")
        print("=" * 50)
        for i, (name, titles) in enumerate(self._view_name_mapping.items(), 1):
            print(f"{i:2d}. {name.capitalize():8s} → {', '.join(titles)}")
        
        print("\n💡 Usage Examples:")
        print("   views=['Lateral', 'Medial']           # Shows both hemispheres lateral and medial")
        print("   views=['Dorsal', 'Ventral']           # Shows top and bottom views") 
        print("   views=['Lateral', 'Medial', 'Dorsal'] # Custom 3-view layout")
        print("   views=['Rostral', 'Caudal']           # Shows front and back views")
        print("=" * 50)
        
        return view_names_capitalized

    def get_layout_details(self, views: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific layout configuration.
        
        Parameters:
            views (str): Name of the configuration to examine
            
        Returns:
            Optional[Dict[str, Any]]: Detailed configuration information if found,
                                    None if configuration doesn't exist. Contains
                                    shape, window_size, and views information.
            
        Examples:
            >>> plotter = SurfacePlotter("configs.json")
            >>> details = plotter.get_layout_details("8_views")
            >>> if details:
            ...     print(f"Grid shape: {details['shape']}")
            ...     print(f"Views: {len(details['views'])}")
            >>> # Output: Grid shape: [2, 4]
            >>> #         Views: 8
            
            >>> # Handle non-existent configuration
            >>> details = plotter.get_layout_details("invalid_config")
            >>> # Output: ❌ Configuration 'invalid_config' not found!
            >>> #         Available configs: ['8_views', '6_views', ...]
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
            print(f"      Camera: az={view['azimuth']}°, el={view['elevation']}°, zoom={view['zoom']}")
        
        return config

    def reload_config(self) -> None:
        """
        Reload the configuration file to pick up any changes.
        
        Useful when modifying configuration files during development.
        
        Returns:
            None: Configuration is reloaded in place
        
        Raises:
            FileNotFoundError: If the configuration file no longer exists
            json.JSONDecodeError: If the configuration file contains invalid JSON
            KeyError: If required configuration keys 'figure_conf' or 'views_conf' are missing
            
        Examples:
            >>> plotter = SurfacePlotter("configs.json")
            >>> # ... modify configs.json externally ...
            >>> plotter.reload_config()  # Pick up the changes
            >>> # Output: Reloading configuration from: configs.json
            >>> #         Successfully loaded figure config and 8 view configurations
        """
        print(f"Reloading configuration from: {self.config_file}")
        self._load_configs()
        print(f"Successfully loaded figure config and {len(self.views_conf)} view configurations")

    def get_figure_config(self) -> Dict[str, Any]:
        """
        Get the current figure configuration settings.
        
        Returns:
            Dict[str, Any]: Dictionary containing all figure styling settings including
                          background color, font settings, mesh properties, and colorbar options.
            
        Examples:
            >>> plotter = SurfacePlotter("configs.json")
            >>> fig_config = plotter.get_figure_config()
            >>> print(f"Background color: {fig_config['background_color']}")
            >>> print(f"Title font: {fig_config['title_font_type']}")
            >>> # Output: Background color: white
            >>> #         Title font: arial
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
        
    