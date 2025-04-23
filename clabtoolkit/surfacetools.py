import os
import numpy as np
import nibabel as nib

from typing import Union, List
import pyvista as pv

# Importing local modules
from . import freesurfertools as cltfree
from . import visualizationtools as cltvis

class Surface:

    def __init__(self, surface_file: str, hemi: str = None):

        self.surf = surface_file
        self.mesh = self.load_surf()

        if hemi is not None:
            self.hemi = hemi
        else:
            self.hemi = cltfree.detect_hemi(self.surf)

        # Trying to guess the hemisphere from BIDs organization
        surf_name = os.path.basename(self.surf)

        hemi = cltfree.detect_hemi(surf_name)

        if hemi is None:
            self.hemi = "lh"

    def load_surf(self):

        vertices, faces = nib.freesurfer.read_geometry(self.surf)

        # add a column with 3's to the faces array. The column is needed for pyvista
        faces = np.c_[np.full(len(faces), 3), faces]

        mesh = pv.PolyData(vertices, faces)

        mesh.colors = "white"

        return mesh

    def load_map(self, map_file: str, cmap: str = "viridis"):

        if not os.path.isfile(map_file):
            print("Map file not found")
            return

        # Read the map file
        map_data = nib.freesurfer.read_morph_data(map_file)

        self.mesh["values"] = map_data
        lut = pv.LookupTable(cmap, n_values=len(np.unique(self.mesh["values"])))
        lut.scalar_range = (np.min(self.mesh["values"]), np.max(self.mesh["values"]))
        self.colors = lut

    def load_annot(self, annot_file: str):

        if not os.path.isfile(annot_file):
            print("Annotation file not found")
            return

        # Read the annotation file
        lab, reg_ctable, reg_names = nib.freesurfer.read_annot(
            annot_file, orig_ids=True
        )

        # Convert the region names to utf-8
        reg_names = [name.decode("utf-8") for name in reg_names]

        # Labels for all the regions
        sts = np.unique(lab)

        # Relabelling to assing the correct colors
        lab_ord = np.full(len(lab), 0)

        # Create an an empty numpy array to store the colors
        reg_ctable_ord = np.ones((len(sts), 4)) * 255

        # Names of the regions
        names_ord = []

        # Loop along all the regions
        for i, st in enumerate(sts):
            ind = np.where(lab == st)[
                0
            ]  # Find the indices of the vertices with label st

            lab_ord[ind] = i

            # Find the index of the region in the region table
            ind = np.where(reg_ctable[:, 4] == st)[0]

            if len(ind) > 0:
                reg_ctable_ord[i, :4] = reg_ctable[ind, :4]
                names_ord.append(reg_names[ind[0]])
            else:

                names_ord.append("unknown")

        ctable = reg_ctable_ord[:, :3]
        # Add a 4th column with 255's
        ctable = np.c_[ctable, np.full(len(ctable), 255)]

        # Create a matrix of ones with dimensions len(sts) x 3
        # # Generate the pyvista mesh objects using polydata

        # Create a lookup table
        lut = pv.LookupTable()
        lut.values = ctable
        self.mesh["values"] = lab_ord
        lut.scalar_range = (np.min(self.mesh["values"]), np.max(self.mesh["values"]))

        self.colors = lut
        self.names = reg_names

    def show(
        self,
        cmap=None,
        view: Union[str, List[str]] = "lateral",
        ambient: float = 0.2,
        diffuse: float = 0.5,
        specular: float = 0.5,
        specular_power: int = 50,
        opacity: float = 1,
        style: str = "surface",
        smooth_shading: bool = True,
        intensity: float = 0.2,
    ):

        if cmap is not None:
            lut = pv.LookupTable(cmap, n_values=len(np.unique(self.mesh["values"])))
            lut.scalar_range = (
                np.min(self.mesh["values"]),
                np.max(self.mesh["values"]),
            )
            self.mesh.colors = lut

        if isinstance(view, str):
            view = [view]
            if not isinstance(view, List[str]):
                raise ValueError("view must be a string or a list of strings")

        ly2plot = cltvis.DefineLayout(
            [self.mesh], both=True, views=view, showtitle=False, showcolorbar=True
        )

        ly2plot.plot()

    #     light1 = pv.Light(
    #         position=(-0.9037, -0.4282, 0),
    #         focal_point=(0, 0, 0),
    #         color=[1.0, 1.0, 0.9843, 1.0],  # Color temp. 5400 K
    #         intensity=intensity,
    #     )

    #     light2 = pv.Light(
    #         position=(0.7877, -0.6160, 0),
    #         focal_point=(0, 0, 0),
    #         color=[1.0, 1.0, 0.9843, 1.0],  # Color temp. 5400 K
    #         intensity=intensity,
    #     )
    #     light3 = pv.Light(
    #         position=(0.1978, 0.9802, 0),
    #         focal_point=(0, 0, 0),
    #         color=[1.0, 1.0, 0.9843, 1.0],  # Color temp. 5400 K
    #         intensity=intensity,
    #     )
    #     light4 = pv.Light(
    #         position=(-0.9650, -0.2624, 0),
    #         focal_point=(0, 0, 0),
    #         color=[1.0, 1.0, 0.9843, 1.0],  # Color temp. 5400 K
    #         intensity=intensity,
    #     )
    #     light5 = pv.Light(
    #         position=(-0.4481, 0, 0.8940),
    #         focal_point=(0, 0, 0),
    #         color=[1.0, 1.0, 0.9843, 1.0],  # Color temp. 5400 K
    #         intensity=intensity,
    #     )
    #     light6 = pv.Light(
    #         position=(0.9844, 0, -0.1760),
    #         focal_point=(0, 0, 0),
    #         color=[1.0, 1.0, 0.9843, 1.0],  # Color temp. 5400 K
    #         intensity=intensity,
    #     )

    #     pl = pv.Plotter(notebook=0, window_size=(800, 800))
    #     pl.add_light(light1)
    #     pl.add_light(light2)
    #     pl.add_light(light3)
    #     pl.add_light(light4)
    #     pl.add_light(light5)
    #     pl.add_light(light6)

    #     if cmap is None:
    #         if hasattr(self, "values"):
    #             if hasattr(self, "colors"):
    #                 pl.add_mesh(
    #                     self.mesh,
    #                     scalars="values",
    #                     cmap=self.colors,
    #                     ambient=ambient,
    #                     diffuse=diffuse,
    #                     specular=specular,
    #                     opacity=opacity,
    #                     specular_power=specular_power,
    #                     style=style,
    #                     smooth_shading=smooth_shading,
    #                 )
    #             else:
    #                 pl.add_mesh(
    #                     self.mesh,
    #                     scalars="values",
    #                     ambient=ambient,
    #                     diffuse=diffuse,
    #                     specular=specular,
    #                     opacity=opacity,
    #                     specular_power=specular_power,
    #                     style=style,
    #                     smooth_shading=smooth_shading,
    #                 )
    #         else:
    #             if hasattr(self, "colors"):
    #                 pl.add_mesh(
    #                     self.mesh,
    #                     cmap=self.colors,
    #                     ambient=ambient,
    #                     diffuse=diffuse,
    #                     specular=specular,
    #                     opacity=opacity,
    #                     specular_power=specular_power,
    #                     style=style,
    #                     smooth_shading=smooth_shading,
    #                 )

    #             else:
    #                 pl.add_mesh(
    #                     self.mesh,
    #                     ambient=ambient,
    #                     diffuse=diffuse,
    #                     specular=specular,
    #                     opacity=opacity,
    #                     specular_power=specular_power,
    #                     style=style,
    #                     smooth_shading=smooth_shading,
    #                 )
    #     else:

    #         lut = pv.LookupTable(cmap, n_values=len(np.unique(self.mesh["values"])))
    #         lut.scalar_range = (
    #             np.min(self.mesh["values"]),
    #             np.max(self.mesh["values"]),
    #         )
    #         self.colors = lut

    #         pl.add_mesh(
    #             self.mesh,
    #             scalars="values",
    #             cmap=self.colors,
    #             ambient=ambient,
    #             diffuse=diffuse,
    #             specular=specular,
    #             opacity=opacity,
    #             specular_power=specular_power,
    #             style=style,
    #             smooth_shading=smooth_shading,
    #         )

    #     pl.link_views()
    #     pl.view_xy()  # link all the views
    #     pl.show()

    # def color_mesh(self):
    #     if self.colors == "aparc":
    #         mesh = self.mesh
    #         mesh["values"] = self.annot
    #         mesh = mesh.point_data_to_cell_data()
    #         mesh.set_active_scalars("values")
    #         mesh = mesh.warp_by_scalar()
    #         mesh = mesh.smooth()
    #         mesh = mesh.decimate(0.95)
    #         return mesh
    #     else:
    #         mesh = self.mesh
    #         mesh["values"] = 0
    #         return mesh

    # def print_attributes(self):
    #     # Print all attributes and their values
    #     for key, value in vars(self).items():
    #         print(f"{key}:")
    #         self._print_value(value, level=1)

    # def _print_value(self, value, level):
    #     """Helper method to print values hierarchically."""
    #     if isinstance(value, dict):
    #         for sub_key, sub_value in value.items():
    #             print("  " * level + f"{sub_key}:")
    #             self._print_value(sub_value, level + 1)
    #     else:
    #         print("  " * level + str(value))


def _vert_neib(faces, max_neib: int = 100):
    """
    Returns a list of neighboring vertices for each vertex in a mesh
    Parameters
    ----------
    faces : numpy array of shape (n_faces, M). M is the number of vertices

    Returns
    -------
    vert_neib : numpy array of shape (n_vertices, max_neib)
        Each row contains the indices of the neighboring vertices
        for the corresponding vertex
    """
    n_vert = np.max(faces) + 1

    # Create an empty numpy array of n_vert x 100
    vert_neib = np.zeros((n_vert, max_neib), dtype=int)
    for i in range(n_vert):
        # Find all faces that contain vertex i
        faces_with_i = np.where(faces == i)[0]

        # Find all vertices that are connected to vertex i
        temp = np.unique(faces[faces_with_i, :])

        # Remove vertex i from the list
        temp = temp[temp != i]
        n_neib = len(temp)

        # add the vertex index and the number of neighbors in front of temp array
        temp = np.hstack((i, n_neib, temp))

        # add temp array to vert_neib array
        vert_neib[i, : len(temp)] = temp

    # Remove the colums that are all zeros
    vert_neib = vert_neib[:, ~np.all(vert_neib == 0, axis=0)]

    return vert_neib


def annot2pyvista(annot_file):
    """
    Reads a FreeSurfer annotation file and returns the vertex labels and colors
    Parameters
    ----------
    annot_file : str
        Path to the annotation file

    Returns
    -------
    lab_ord : numpy array of shape (n_vertices,)
        Vertex labels
    reg_ctable_ord : numpy array of shape (n_regions, 4)
        Colors for each region
    names_ord : list of strings
        Names of the regions in the annotation file

    """

    if not os.path.isfile(annot_file):
        print("Annotation file not found")
        return

    # Read the annotation file
    lab, reg_ctable, reg_names = nib.freesurfer.read_annot(annot_file, orig_ids=True)

    # Convert the region names to utf-8
    reg_names = [name.decode("utf-8") for name in reg_names]

    # Labels for all the regions
    sts = np.unique(lab)

    # Relabelling to assing the correct colors
    lab_ord = np.full(len(lab), 0)

    # Create an an empty numpy array to store the colors
    reg_ctable_ord = np.ones((len(sts), 4)) * 255

    # Names of the regions
    names_ord = []

    # Loop along all the regions
    for i, st in enumerate(sts):
        ind = np.where(lab == st)[0]  # Find the indices of the vertices with label st

        lab_ord[ind] = i

        # Find the index of the region in the region table
        ind = np.where(reg_ctable[:, 4] == st)[0]

        if len(ind) > 0:
            reg_ctable_ord[i, :4] = reg_ctable[ind, :4]
            names_ord.append(reg_names[ind[0]])
        else:

            names_ord.append("unknown")

    return lab_ord, reg_ctable_ord, names_ord
