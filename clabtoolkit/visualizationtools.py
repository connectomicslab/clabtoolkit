import os
import numpy as np
import nibabel as nib
from typing import Union
from nilearn import plotting

from typing import Union, List
import pyvista as pv

# Importing local modules
from . import freesurfertools as cltfree
from . import surfacetools as cltsurf


class DefineLayout:

    def __init__(
        self,
        meshes,
        both: bool = False,
        views: Union[str, List[str]] = "all",
        showtitle: bool = False,
        showcolorbar: bool = False,
        ambient: float = 0.2,
        diffuse: float = 0.5,
        specular: float = 0.5,
        specular_power: int = 50,
        opacity: float = 1,
        style: str = "surface",
        smooth_shading: bool = True,
        intensity: float = 0.2,
    ):

        self.meshes = meshes
        self.showtitle = showtitle
        self.showcolorbar = showcolorbar

        # Defining the camera positions for each view taking into account the hemisphere
        lh_viewdict = {
            "lateral": {"position": "yz", "azimuth": 180, "elevation": 0},
            "dorsal": {"position": "xy", "azimuth": 0, "elevation": 0},
            "rostral": {"position": "xz", "azimuth": 180, "elevation": 0},
            "caudal": {"position": "xz", "azimuth": 0, "elevation": 0},
            "ventral": {"position": "xy", "azimuth": 180, "elevation": 0},
            "medial": {"position": "yz", "azimuth": 0, "elevation": 0},
        }

        rh_viewdict = {
            "lateral": {"position": "yz", "azimuth": 0, "elevation": 0},
            "dorsal": {"position": "xy", "azimuth": 0, "elevation": 0},
            "rostral": {"position": "xz", "azimuth": 180, "elevation": 0},
            "caudal": {"position": "xz", "azimuth": 0, "elevation": 0},
            "ventral": {"position": "xy", "azimuth": 180, "elevation": 0},
            "medial": {"position": "yz", "azimuth": 180, "elevation": 0},
        }

        # Defining the views
        if isinstance(views, str):
            if views == "all":
                views = lh_viewdict.keys()
            elif views in lh_viewdict.keys():
                views = [views]
            else:
                print(
                    "Invalid view. Please use one of the following: all, lateral, dorsal, rostral, caudal, ventral, medial"
                )
                return

        elif isinstance(views, list):

            # If all is in views, I will replace it with all the views
            if "all" in views:
                views = lh_viewdict.keys()

            views2del = []
            for view in views:
                if view not in lh_viewdict.keys():

                    views2del.append(view)

            if views2del:
                for view in views2del:
                    views.remove(view)
                print(
                    f"Invalid views: {views2del} not found. Removed from the list of views"
                )

        if not views:
            print("No valid valid views found")
            return

        # Ordering the views to maintain the same order in the layout
        temp_views = []
        for v in list(lh_viewdict.keys()):
            if v in views:
                temp_views.append(v)
        views = temp_views

        print(views)

        self.views = views
        self.lh_viewdict = lh_viewdict
        self.rh_viewdict = rh_viewdict

        # Defining the layout
        n_views = len(views)

        # Detecting the number of boxplots in the rows
        # Detecting the surfaces to plot and the links between surfaces.
        surfs2plot = []
        mesh_links = []
        for s, mesh in enumerate(meshes):
            if isinstance(mesh, list):
                for submesh in mesh:
                    surfs2plot.append(submesh)
                    mesh_links.append(s)
            else:
                mesh_links.append(s)
                surfs2plot.append(mesh)

        # Number of rows in the layout dedicated to plot the surfaces

        if "lateral" not in views and "medial" not in views:
            n_meshes = max(mesh_links) + 1
        else:
            n_meshes = len(surfs2plot)

        if len(views) == 1:
            if showcolorbar and showtitle:
                row_offset = 1
                shape = (3, n_meshes)
                row_weights = [0.2, 1, 0.3]
                col_weights = [1] * n_meshes
                groups = [
                    (
                        0,
                        np.s_[:],
                    )
                ] + [
                    (
                        3,
                        np.s_[:],
                    )
                ]

            elif showcolorbar and not showtitle:
                row_offset = 0
                shape = (2, n_meshes)
                row_weights = [1, 0.3]
                col_weights = [1] * n_meshes
                groups = [
                    (
                        2,
                        np.s_[:],
                    )
                ]

            elif showtitle and not showcolorbar:
                row_offset = 1
                shape = (2, n_meshes)
                row_weights = [0.2, 1]
                col_weights = [1] * n_meshes
                groups = [
                    (
                        0,
                        np.s_[:],
                    )
                ]

            else:
                row_offset = 0
                shape = (1, n_meshes)
                row_weights = [1]
                col_weights = [1] * n_meshes
                groups = None

            self.shape = shape
            self.row_weights = row_weights
            self.col_weights = col_weights
            self.groups = groups

            # Creating the layout
            layout = []
            view = views[0]
            for col in range(n_meshes):

                if view != "lateral" and view != "medial":
                    indexes = [
                        i for i, e in enumerate(mesh_links) if e == mesh_links[col]
                    ]
                    temp_surf2plot = [surfs2plot[i] for i in indexes]

                    mesh2plot = temp_surf2plot
                else:
                    mesh2plot = surfs2plot[col]

                if isinstance(mesh2plot, list):
                    if isinstance(mesh2plot[0], cltsurf.Surface):
                        temp_label = mesh2plot[0].hemi

                        if temp_label.startswith("lh"):
                            camdict = lh_viewdict
                        elif temp_label.startswith("rh"):
                            camdict = rh_viewdict
                        else:
                            camdict = lh_viewdict
                    else:
                        camdict = lh_viewdict
                elif isinstance(mesh2plot, cltsurf.Surface):
                    temp_label = mesh2plot.hemi

                    if temp_label.startswith("lh"):
                        camdict = lh_viewdict
                    elif temp_label.startswith("rh"):
                        camdict = rh_viewdict
                    else:
                        camdict = lh_viewdict
                else:
                    camdict = lh_viewdict

                subp_dic = {
                    "subp": {"row": row_offset, "col": col},
                    "surf2plot": mesh2plot,
                    "text": {
                        "label": view.capitalize(),
                        "position": "upper_edge",
                        "font_size": 7,
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

                if showcolorbar:
                    self.colorbar = {"colorbar": {"row": row_offset, "col": 0}}
                else:
                    self.colorbar = None

                if showtitle:
                    self.title = {"title": {"row": 0, "col": 0}}
                else:
                    self.title = None

        else:
            if showcolorbar and showtitle:
                row_offset = 1
                col_offset = 0
                shape = (n_meshes + 2, n_views)
                row_weights = [0.2] + [1] * n_meshes + [0.3]
                col_weights = [1] * n_views
                groups = [
                    (
                        0,
                        np.s_[:],
                    )
                ] + [
                    (
                        n_meshes + 1,
                        np.s_[:],
                    )
                ]

            elif showcolorbar and not showtitle:
                row_offset = 0
                col_offset = 0
                shape = (n_meshes + 1, n_views)
                row_weights = [1] * n_meshes + [0.3]
                col_weights = [1] * n_views
                groups = [
                    (
                        n_meshes,
                        np.s_[:],
                    )
                ]

            elif showtitle and not showcolorbar:
                row_offset = 1
                col_offset = 0
                shape = (n_meshes + 1, n_views)
                row_weights = [0.2] + [1] * n_meshes
                col_weights = [1] * n_views
                groups = [
                    (
                        0,
                        np.s_[:],
                    )
                ]

            else:
                row_offset = 0
                col_offset = 0
                shape = (n_meshes, n_views)
                row_weights = [1] * n_meshes
                col_weights = [1] * n_views
                groups = None

            self.shape = shape
            self.row_weights = row_weights
            self.col_weights = col_weights
            self.groups = groups

            # Creating the layout
            layout = []
            for row in range(n_meshes):
                for col in range(n_views):

                    view = views[col]
                    if view != "lateral" and view != "medial":
                        indexes = [
                            i for i, e in enumerate(mesh_links) if e == mesh_links[row]
                        ]
                        temp_surf2plot = [surfs2plot[i] for i in indexes]

                        mesh2plot = temp_surf2plot
                    else:
                        mesh2plot = surfs2plot[row]

                    if isinstance(mesh2plot, list):
                        if isinstance(mesh2plot[0], cltsurf.Surface):
                            temp_label = mesh2plot[0].hemi

                            if temp_label.startswith("lh"):
                                camdict = lh_viewdict
                            elif temp_label.startswith("rh"):
                                camdict = rh_viewdict
                            else:
                                camdict = lh_viewdict
                        else:
                            camdict = lh_viewdict
                    elif isinstance(mesh2plot, cltsurf.Surface):
                        temp_label = mesh2plot.hemi

                        if temp_label.startswith("lh"):
                            camdict = lh_viewdict
                        elif temp_label.startswith("rh"):
                            camdict = rh_viewdict
                        else:
                            camdict = lh_viewdict
                    else:
                        camdict = lh_viewdict

                    subp_dic = {
                        "subp": {"row": row + row_offset, "col": col + col_offset},
                        "surf2plot": mesh2plot,
                        "text": {
                            "label": view.capitalize(),
                            "position": "upper_edge",
                            "font_size": 7,
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

                    if showcolorbar:
                        self.colorbar = {"colorbar": {"row": n_meshes, "col": 0}}
                    else:
                        self.colorbar = None

                    if showtitle:
                        self.title = {"title": {"row": 0, "col": 0}}
                    else:
                        self.title = None

        self.layout = layout

    def plot(
        self,
        ambient: float = 0.2,
        diffuse: float = 0.5,
        specular: float = 0.5,
        specular_power: int = 50,
        opacity: float = 1,
        style: str = "surface",
        smooth_shading: bool = True,
        intensity: float = 0.2,
    ):
        #
        pl = pv.Plotter(
            shape=self.shape,
            row_weights=self.row_weights,
            col_weights=self.col_weights,
            groups=self.groups,
            notebook=0,
            window_size=(1200, 800),
        )

        all_colors = []
        all_values = []
        for subp in self.layout:
            pl.subplot(subp["subp"]["row"], subp["subp"]["col"])
            pl.add_text(
                subp["text"]["label"],
                position=subp["text"]["position"],
                font_size=subp["text"]["font_size"],
                color=subp["text"]["color"],
                font=subp["text"]["font"],
                shadow=subp["text"]["shadow"],
            )
            if isinstance(subp["surf2plot"], list):
                for surf in subp["surf2plot"]:

                    if isinstance(surf, cltsurf.Surface):
                        pl.add_mesh(
                            surf.mesh,
                            scalars="values",
                            cmap=surf.colors,
                            ambient=ambient,
                            diffuse=diffuse,
                            specular=specular,
                            opacity=opacity,
                            specular_power=specular_power,
                            style=style,
                            smooth_shading=smooth_shading,
                        )
                        all_colors.append(surf.colors)
                        all_values.append(surf.mesh["values"])

            else:
                if isinstance(subp["surf2plot"], cltsurf.Surface):
                    pl.add_mesh(
                        subp["surf2plot"].mesh,
                        scalars="values",
                        cmap=subp["surf2plot"].colors,
                        ambient=ambient,
                        diffuse=diffuse,
                        specular=specular,
                        opacity=opacity,
                        specular_power=specular_power,
                        style=style,
                        smooth_shading=smooth_shading,
                    )
                    all_colors.append(subp["surf2plot"].colors)
                    all_values.append(subp["surf2plot"].mesh["values"])

            pl.remove_scalar_bar()
            pl.camera_position = subp["camera"]["position"]
            pl.camera.azimuth = subp["camera"]["azimuth"]
            pl.camera.elevation = subp["camera"]["elevation"]

        if self.colorbar is not None:
            # Selecting the first subplot to add the colorbar
            pl.subplot(self.colorbar["colorbar"]["row"], 0)

            lut = pv.LookupTable(cmap=all_colors)
            lut.scalar_range = (np.min(all_values), np.min(all_values))
            lut.below_range_color = pv.Color("grey", opacity=0.5)
            lut.above_range_color = "r"

        pl.show()
