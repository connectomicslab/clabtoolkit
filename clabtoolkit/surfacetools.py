import os
import numpy as np
import nibabel as nib

from typing import Union, List
import pyvista as pv

# Importing local modules
from . import freesurfertools as cltfree
from . import visualizationtools as cltvis


class Surface:
    """
    A class for loading and visualizing brain surfaces from FreeSurfer format files.

    This class provides comprehensive functionality for loading brain surface meshes,
    overlaying scalar data (maps), loading anatomical annotations, and creating
    publication-quality visualizations with multiple anatomical views.

    Attributes
    ----------
    surf : str
        Path to the surface file
    mesh : pv.PolyData
        PyVista mesh object containing the surface geometry
    hemi : str
        Hemisphere identifier ("lh" for left, "rh" for right)
    colors : pv.LookupTable, optional
        Color lookup table for scalar data visualization
    names : list of str, optional
        Region names from annotation files

    Examples
    --------
    >>> # Load a basic surface
    >>> surf = Surface("lh.pial")
    >>> surf.show()

    >>> # Load surface with scalar data
    >>> surf = Surface("lh.pial", hemi="lh")
    >>> surf.load_map("thickness.mgh")
    >>> surf.show(view=["lateral", "medial"], colorbar_title="Thickness (mm)")

    >>> # Load surface with anatomical parcellation
    >>> surf.load_annot("lh.aparc.annot")
    >>> surf.show(view="all", title="Anatomical Parcellation")
    """

    def __init__(self, surface_file: str, hemi: str = None):
        """
        Initialize a Surface object from a FreeSurfer surface file.

        Parameters
        ----------
        surface_file : str
            Path to the FreeSurfer surface file (e.g., "lh.pial", "rh.white")
        hemi : str, optional
            Hemisphere identifier ("lh" or "rh"). If None, attempts to auto-detect
            from the filename.

        Raises
        ------
        FileNotFoundError
            If the surface file does not exist
        ValueError
            If the surface file cannot be loaded

        Examples
        --------
        >>> surf = Surface("/path/to/lh.pial")
        >>> surf = Surface("rh.white", hemi="rh")
        """
        self.surf = surface_file
        self.mesh = self.load_surf()

        if hemi is not None:
            self.hemi = hemi
        else:
            self.hemi = cltfree.detect_hemi(self.surf)

        # Trying to guess the hemisphere from BIDS organization
        surf_name = os.path.basename(self.surf)
        detected_hemi = cltfree.detect_hemi(surf_name)

        if detected_hemi is None:
            self.hemi = "lh"  # Default to left hemisphere

    def load_surf(self):
        """
        Load surface geometry from FreeSurfer format file.

        Returns
        -------
        pv.PolyData
            PyVista mesh object containing vertices and faces

        Raises
        ------
        FileNotFoundError
            If surface file cannot be found
        ValueError
            If surface file format is invalid

        Notes
        -----
        This method reads FreeSurfer surface files using nibabel and converts
        them to PyVista PolyData format for visualization. The faces array is
        modified to include a leading column of 3's as required by PyVista.
        """
        vertices, faces = nib.freesurfer.read_geometry(self.surf)

        # add a column with 3's to the faces array. The column is needed for pyvista
        faces = np.c_[np.full(len(faces), 3), faces]

        mesh = pv.PolyData(vertices, faces)
        mesh.colors = "white"

        return mesh

    def load_map(self, map_file: str, cmap: str = "viridis"):
        """
        Load scalar data (map) onto the surface for visualization.

        Parameters
        ----------
        map_file : str
            Path to the scalar data file. Supported formats include FreeSurfer
            morphometry files (.mgh, .mgz), curvature files (.curv), and others.
        cmap : str, default "viridis"
            Colormap name for visualizing the scalar data. Any matplotlib
            colormap name is supported.

        Raises
        ------
        FileNotFoundError
            If the map file does not exist
        ValueError
            If the map file format is not supported or data dimensions don't match

        Examples
        --------
        >>> surf = Surface("lh.pial")
        >>> surf.load_map("lh.thickness")  # Load cortical thickness
        >>> surf.load_map("lh.curv", cmap="RdBu")  # Load curvature with custom colormap

        Notes
        -----
        The scalar data is stored in the mesh's "values" array and can be
        visualized using the show() method. The colormap is stored as a
        PyVista LookupTable for consistent visualization.
        """
        if not os.path.isfile(map_file):
            raise FileNotFoundError(f"Map file not found: {map_file}")

        # Read the map file
        map_data = nib.freesurfer.read_morph_data(map_file)

        self.mesh["values"] = map_data
        lut = pv.LookupTable(cmap, n_values=len(np.unique(self.mesh["values"])))
        lut.scalar_range = (np.min(self.mesh["values"]), np.max(self.mesh["values"]))
        self.colors = lut

    def load_annot(self, annot_file: str):
        """
        Load anatomical annotation data onto the surface.

        Parameters
        ----------
        annot_file : str
            Path to the FreeSurfer annotation file (e.g., "lh.aparc.annot")

        Raises
        ------
        FileNotFoundError
            If the annotation file does not exist
        ValueError
            If the annotation file format is invalid or incompatible

        Examples
        --------
        >>> surf = Surface("lh.pial")
        >>> surf.load_annot("lh.aparc.annot")  # Desikan-Killiany atlas
        >>> surf.load_annot("lh.aparc.a2009s.annot")  # Destrieux atlas

        Notes
        -----
        Annotation files contain anatomical parcellations with region labels
        and colors. The data is processed to create consistent vertex labeling
        and a color lookup table. Region names are stored in the 'names' attribute.
        """
        if not os.path.isfile(annot_file):
            raise FileNotFoundError(f"Annotation file not found: {annot_file}")

        # Read the annotation file
        lab, reg_ctable, reg_names = nib.freesurfer.read_annot(
            annot_file, orig_ids=True
        )

        # Convert the region names to utf-8
        reg_names = [name.decode("utf-8") for name in reg_names]

        # Labels for all the regions
        sts = np.unique(lab)

        # Relabelling to assign the correct colors
        lab_ord = np.full(len(lab), 0)

        # Create an empty numpy array to store the colors
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

        # Create a lookup table
        lut = pv.LookupTable()
        lut.values = ctable
        self.mesh["values"] = lab_ord
        lut.scalar_range = (np.min(self.mesh["values"]), np.max(self.mesh["values"]))

        self.colors = lut
        self.names = names_ord

    def show(
        self,
        cmap: str = None,
        view: Union[str, List[str]] = "lateral",
        link_views: bool = False,
        colorbar_title: str = None,
        title: str = None,
        window_size: tuple = (1400, 900),
        background_color: str = "white",
        ambient: float = 0.2,
        diffuse: float = 0.5,
        specular: float = 0.5,
        specular_power: int = 50,
        opacity: float = 1.0,
        style: str = "surface",
        smooth_shading: bool = True,
        intensity: float = 0.2,
    ):
        """
        Display the brain surface with specified views and rendering options.

        Parameters
        ----------
        cmap : str, optional
            Colormap name for scalar data visualization. Only used if surface has values.
        view : str or list of str, default "lateral"
            Anatomical view(s) to display. Options: "lateral", "medial", "dorsal",
            "ventral", "rostral", "caudal", "all", or list of views.
        link_views : bool, default False
            Whether to synchronize camera movements between multiple views.
        colorbar_title : str, optional
            Title for the colorbar. Auto-detected if not provided.
        title : str, optional
            Main title for the visualization.
        window_size : tuple, default (1400, 900)
            Window size as (width, height) in pixels.
        background_color : str, default "white"
            Background color for the visualization.
        ambient : float, default 0.2
            Ambient lighting coefficient (0.0-1.0).
        diffuse : float, default 0.5
            Diffuse lighting coefficient (0.0-1.0).
        specular : float, default 0.5
            Specular lighting coefficient (0.0-1.0).
        specular_power : int, default 50
            Specular power for surface shininess.
        opacity : float, default 1.0
            Surface opacity (0.0-1.0).
        style : str, default "surface"
            Surface rendering style ("surface", "wireframe", "points").
        smooth_shading : bool, default True
            Whether to use smooth shading.
        intensity : float, default 0.2
            Light intensity (0.0-1.0).

        Returns
        -------
        pv.Plotter
            The PyVista plotter object for further customization.

        Examples
        --------
        >>> surf = Surface("lh.pial")
        >>> surf.show(view="lateral")

        >>> surf.load_map("thickness.mgh")
        >>> surf.show(view=["lateral", "medial"], colorbar_title="Thickness (mm)",
        ...          link_views=True)

        >>> surf.show(view="all", title="Brain Surface - All Views")
        """
        # Import here to avoid circular import
        from . import visualizationtools as cltvis

        # Apply custom colormap if provided and surface has values
        if cmap is not None and "values" in self.mesh.array_names:
            lut = pv.LookupTable(cmap, n_values=len(np.unique(self.mesh["values"])))
            lut.scalar_range = (
                np.min(self.mesh["values"]),
                np.max(self.mesh["values"]),
            )
            self.colors = lut

        # Validate view parameter
        if isinstance(view, str):
            view = [view]
        elif not isinstance(view, list):
            raise ValueError("view must be a string or a list of strings")

        # Determine if we should show colorbar
        show_colorbar = "values" in self.mesh.array_names and hasattr(self, "colors")

        # Auto-detect colorbar title if not provided
        if show_colorbar and colorbar_title is None:
            colorbar_title = "Values"  # Default title

        # Create layout (excluding intensity as it's not used by PyVista add_mesh)
        layout = cltvis.DefineLayout(
            meshes=[self],
            views=view,
            showtitle=title is not None,  # Enable title row if title is provided
            showcolorbar=show_colorbar,
            colorbar_title=colorbar_title,
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            specular_power=specular_power,
            opacity=opacity,
            style=style,
            smooth_shading=smooth_shading,
        )

        # Plot and return plotter object
        return layout.plot(
            link_views=link_views,
            window_size=window_size,
            background_color=background_color,
            title=title,
        )

    def print_available_views(self, hemisphere: str = None):
        """
        Print available views for brain surface visualization.

        Parameters
        ----------
        hemisphere : str, optional
            Which hemisphere views to show. If None, uses the surface's hemisphere.
            Options: 'lh', 'rh', 'both'
        """
        if hemisphere is None:
            # Use the surface's hemisphere, but show both if unclear
            if hasattr(self, "hemi") and self.hemi:
                hemisphere = self.hemi
            else:
                hemisphere = "both"

        # Import here to avoid circular import
        from . import visualizationtools as cltvis

        # Create a temporary layout object to access the print method
        temp_layout = cltvis.DefineLayout([self], views=["lateral"])
        temp_layout.print_available_views(hemisphere)


def vert_neib(faces, max_neib: int = 100):
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

    # Remove the columns that are all zeros
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

    # Relabelling to assign the correct colors
    lab_ord = np.full(len(lab), 0)

    # Create an empty numpy array to store the colors
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
