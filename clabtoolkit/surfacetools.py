import os
import numpy as np
import nibabel as nib
from typing import Union, List, Dict, Optional, Tuple
from pathlib import Path
import pyvista as pv
import pandas as pd
import copy

# Importing local modules
from . import freesurfertools as cltfree
from . import misctools as cltmisc


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############                Section 1: Class and methods work with meshes               ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
class Surface:
    """
    Comprehensive class for loading and visualizing brain surface data.

    Provides interface for working with brain surface geometries including loading
    from files or arrays, managing scalar maps and parcellations, and creating
    visualizations using PyVista. Supports FreeSurfer and other surface formats.

    Attributes
    ----------
    surf : str or None
        Path to surface file if loaded from file.

    mesh : pv.PolyData
        PyVista mesh object containing surface geometry and data.

    hemi : str
        Hemisphere designation ('lh', 'rh', or 'unknown').

    colortables : dict
        Dictionary storing color table information for parcellations.

    Examples
    --------
    >>> # Load from FreeSurfer surface file
    >>> surface = Surface('lh.pial')
    >>>
    >>> # Create from vertex/face arrays
    >>> surface = Surface(vertices=verts, faces=faces, hemi='lh')
    >>>
    >>> # Load scalar data and parcellations
    >>> surface.load_scalar_map('thickness.mgh', 'thickness')
    >>> surface.load_annotation('lh.aparc.annot', 'aparc')
    """

    ##############################################################################################
    def __init__(
        self,
        surface_file: Union[str, Path] = None,
        vertices: np.ndarray = None,
        faces: np.ndarray = None,
        color: Union[str, np.ndarray] = "#f0f0f0",
        alpha: float = 1.0,
        hemi: str = None,
    ) -> None:
        """
        Initialize Surface object from file, arrays, or create empty instance.

        Parameters
        ----------
        surface_file : str or Path, optional
            Path to surface file (FreeSurfer .pial, .white, .inflated). Default is None.

        vertices : np.ndarray, optional
            Vertex coordinates array with shape (n_vertices, 3). Default is None.

        faces : np.ndarray, optional
            Face connectivity array with shape (n_faces, 3). Default is None.

        color : str or np.ndarray, optional
            Color for the surface mesh. Can be a hex color string (e.g., '#f0f0f0')
            or an RGB array in [0, 1] or [0, 255] range. Default is '#f0f0f0'.

        alpha : float, optional
            Alpha transparency value in [0, 1] for the surface color. Default is 1.0 (opaque).

        hemi : str, optional
            Hemisphere designation ('lh' or 'rh'). Auto-detected from filename
            if None. Default is None.

        Raises
        ------
        ValueError
            If both surface_file and vertices/faces are provided, or if only
            one of vertices/faces is provided.
        FileNotFoundError
            If surface file doesn't exist.

        Examples
        --------
        >>> # Load from file with auto-detection
        >>> surface = Surface('lh.pial')
        >>>
        >>> # Create from arrays
        >>> vertices = np.random.rand(100, 3)
        >>> faces = np.array([[0, 1, 2], [1, 2, 3]])
        >>> surface = Surface(vertices=vertices, faces=faces, hemi='lh')
        >>>
        >>> # Create empty instance
        >>> surface = Surface()
        """

        # Initialize attributes to None (empty instance)
        self.surf = None
        self.mesh = None
        self.hemi = None
        self.colortables: Dict[str, Dict] = {}

        # Create the defalt colortable for the surface

        # Set the colortable for the surface
        # Validate alpha value
        if isinstance(alpha, int):
            alpha = float(alpha)

        # If the alpha is not in the range [0, 1], raise an error
        if not (0 <= alpha <= 1):
            raise ValueError(f"Alpha value must be in the range [0, 1], got {alpha}")

        # Handle color input
        color = cltmisc.harmonize_colors(color, output_format="rgb") / 255

        tmp_ctable = cltmisc.colors_to_table(colors=color, alpha_values=alpha)
        tmp_ctable[:, :3] = tmp_ctable[:, :3] / 255  # Ensure colors are between 0 and 1

        # Store parcellation information in organized structure
        self.colortables["surface"] = {
            "struct_names": ["surface"],
            "color_table": tmp_ctable,
            "lookup_table": None,  # Will be populated by _create_parcellation_colortable if needed
        }

        # Validate input parameters
        if surface_file is not None and (vertices is not None or faces is not None):
            raise ValueError("Cannot specify both surface_file and vertices/faces")

        if vertices is not None and faces is None:
            raise ValueError("If vertices are provided, faces must also be provided")

        if faces is not None and vertices is None:
            raise ValueError("If faces are provided, vertices must also be provided")

        # Load data if provided
        if surface_file is not None:
            if isinstance(surface_file, Path):
                surface_file = str(surface_file)

            if isinstance(surface_file, str):
                if not os.path.isfile(surface_file):
                    raise FileNotFoundError(f"Surface file not found: {surface_file}")

                self.surf = surface_file
                self.load_from_file(surface_file, color, alpha, hemi)

            elif isinstance(surface_file, pv.PolyData):
                self.load_from_mesh(surface_file, color, alpha, hemi)

        elif vertices is not None and faces is not None:
            self.load_from_arrays(vertices, faces, color, alpha, hemi=hemi)

    ################################################################################################
    def load_from_file(
        self,
        surface_file: Union[str, Path],
        color: Union[str, np.ndarray] = "#f0f0f0",
        alpha: np.float32 = 1.0,
        hemi: str = None,
    ) -> None:
        """
        Load surface geometry from FreeSurfer or compatible surface file.

        Parameters
        ----------
        surface_file : str or Path
            Path to surface file (e.g., FreeSurfer .pial, .white, .inflated).

        color : str or np.ndarray, optional
            Color for the surface mesh. Can be a hex color string (e.g., '#f0f0f0')
            or an RGB array in [0, 1] or [0, 255] range. Default is '#f0f0f0'.

        alpha : float, optional
            Alpha transparency value in [0, 1] for the surface color. Default is 1.0 (opaque).

        hemi : str, optional
            Hemisphere designation ('lh' or 'rh'). Auto-detected from filename
            if None. Default is None.

        Raises
        ------
        FileNotFoundError
            If surface file cannot be found.
        ValueError
            If surface file format is unsupported or corrupted.

        Notes
        -----
        - Automatically detects hemisphere from filename if not provided.
        - Converts color string to RGB array if provided as hex.
        - Adds alpha channel to color for RGBA representation.
        - Creates default parcellation data for visualization.
        - Uses nibabel to read FreeSurfer geometry files.
        - Handles both left ('lh') and right ('rh') hemisphere surfaces.
        - If color is not provided, defaults to light gray ('#f0f0f0').

        Examples
        --------
        >>> surface = Surface()
        >>> surface.load_from_file('lh.pial')
        >>> print(f"Loaded {surface.mesh.n_points} vertices")
        >>>
        >>> # Explicit hemisphere specification
        >>> surface.load_from_file('brain_surface.surf', hemi='rh')
        """

        # Check if the surface file exists
        if isinstance(surface_file, Path):
            surface_file = str(surface_file)

        if not os.path.isfile(surface_file):
            raise FileNotFoundError(f"Surface file not found: {surface_file}")

        # Store the surface file path
        self.surf = surface_file

        # Validate alpha value
        if isinstance(alpha, int):
            alpha = float(alpha)

        # If the alpha is not in the range [0, 1], raise an error
        if not (0 <= alpha <= 1):
            raise ValueError(f"Alpha value must be in the range [0, 1], got {alpha}")

        # Handle color input
        color = cltmisc.harmonize_colors(color, output_format="rgb") / 255

        # Load the surface geometry
        try:
            vertices, faces = nib.freesurfer.read_geometry(self.surf)

            # Add column with 3's to faces array for PyVista
            faces = np.c_[np.full(len(faces), 3), faces]

            mesh = pv.PolyData(vertices, faces)

            # Add default surface colors if not present

            # Adding the mesh
            self.mesh = mesh

        except Exception as e:
            raise ValueError(f"Failed to load surface file '{self.surf}': {e}")

        # Hemisphere detection from filename
        if hemi is not None:
            self.hemi = hemi
        else:
            self.hemi = cltfree.detect_hemi(self.surf)

        # Fallback hemisphere detection from BIDS organization
        surf_name = os.path.basename(self.surf)
        detected_hemi = cltfree.detect_hemi(surf_name)

        if detected_hemi is None:
            self.hemi = "lh"  # Default to left hemisphere

        # Create default parcellation data
        self._create_default_parcellation(
            color=color,
            alpha=alpha,
        )

    ##############################################################################################
    def load_from_arrays(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: Union[str, np.ndarray] = "#f0f0f0",
        alpha: float = 1.0,
        hemi: str = None,
        surface_file: str = None,
    ) -> None:
        """
        Load surface geometry from vertex and face arrays.

        Parameters
        ----------
        vertices : np.ndarray
            Vertex coordinates with shape (n_vertices, 3).

        faces : np.ndarray
            Face connectivity with shape (n_faces, 3).

        color : str or np.ndarray, optional
            Color for the surface mesh. Can be a hex color string (e.g., '#f0f0f0')
            or an RGB array in [0, 1] or [0, 255] range. Default is '#f0f0f0'.

        alpha : float, optional
            Alpha transparency value in [0, 1] for the surface color. Default is 1.0 (opaque).

        hemi : str, optional
            Hemisphere designation ('lh' or 'rh'). Defaults to 'lh'.

        surface_file : str, optional
            Associated surface file path for metadata. Default is None.

        Raises
        ------
        ValueError
            If vertices or faces arrays have incorrect shapes.

        Examples
        --------
        >>> # Basic triangle mesh
        >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        >>> faces = np.array([[0, 1, 2]])
        >>> surface = Surface()
        >>> surface.load_from_arrays(vertices, faces, hemi='lh')
        """

        # Validate alpha value
        if isinstance(alpha, int):
            alpha = float(alpha)

        # If the alpha is not in the range [0, 1], raise an error
        if not (0 <= alpha <= 1):
            raise ValueError(f"Alpha value must be in the range [0, 1], got {alpha}")

        # Handle color input
        color = cltmisc.harmonize_colors(color, output_format="rgb") / 255

        self.surf = surface_file
        self.mesh = self.create_mesh_from_arrays(vertices, faces)
        self.hemi = hemi if hemi is not None else "lh"  # Default to left hemisphere

        # Create default parcellation data
        self._create_default_parcellation(
            color=color,
            alpha=alpha,
        )

    ##############################################################################################
    def load_from_mesh(
        self,
        mesh: pv.PolyData,
        color: Union[str, np.ndarray] = "#f0f0f0",
        alpha: float = 1.0,
        hemi: str = None,
    ) -> None:
        """
        Load surface geometry from existing PyVista mesh object.

        Parameters
        ----------
        mesh : pv.PolyData
            PyVista mesh object containing surface geometry.

        color : str or np.ndarray, optional
            Color for the surface mesh. Can be a hex color string (e.g., '#f0f0f0')
            or an RGB array in [0, 1] or [0, 255] range. Default is '#f0f0f0'.

        alpha : float, optional
            Alpha transparency value in [0, 1] for the surface color. Default is 1.0 (opaque).

        hemi : str, optional
            Hemisphere designation ('lh' or 'rh'). Defaults to 'lh'.

        Notes
        -----
        Creates a deep copy of the input mesh to avoid modifying the original.
        Adds default surface colors if not present in the mesh.

        Examples
        --------
        >>> # From existing PyVista mesh
        >>> existing_mesh = pv.PolyData(vertices, faces)
        >>> surface = Surface()
        >>> surface.load_from_mesh(existing_mesh, hemi='rh')
        >>>
        >>> # From procedural mesh
        >>> sphere = pv.Sphere(radius=50)
        >>> surface.load_from_mesh(sphere, hemi='lh')
        """

        self.surf = None
        self.mesh = copy.deepcopy(mesh)  # Make a copy to avoid modifying the original
        self.hemi = hemi if hemi is not None else "lh"  # Default to left hemisphere

        # Validate alpha value
        if isinstance(alpha, int):
            alpha = float(alpha)

        # If the alpha is not in the range [0, 1], raise an error
        if not (0 <= alpha <= 1):
            raise ValueError(f"Alpha value must be in the range [0, 1], got {alpha}")

        # Handle color input
        color = cltmisc.harmonize_colors(color, output_format="rgb") / 255

        # Ensure mesh has default surface colors if not present
        self._create_default_parcellation(color=color, alpha=alpha)

    ##############################################################################################
    def is_loaded(self) -> bool:
        """
        Check whether surface data has been loaded.

        Returns
        -------
        bool
            True if surface data is loaded, False otherwise.

        Examples
        --------
        >>> surface = Surface()
        >>> print(surface.is_loaded())  # False
        >>> surface.load_from_file('lh.pial')
        >>> print(surface.is_loaded())  # True
        """
        return self.mesh is not None

    ##############################################################################################
    def _create_default_parcellation(
        self, color: Union[str, np.ndarray] = "#f0f0f0", alpha: np.ndarray = 1.0
    ) -> None:
        """
        Create default parcellation data for surface visualization.

        Internal method that sets up basic parcellation with uniform surface
        colors for initial visualization before loading specific annotations.

        Parameters
        ----------

        color : str or np.ndarray
            Color for the surface mesh. Can be a hex color string (e.g., '#f0f0f0')
            or an RGB array in [0, 1] or [0, 255] range.

        alpha : float
            Alpha transparency value in [0, 1] for the surface color. Default is 1.0 (opaque).

        Notes
        -----
        Creates a single-region parcellation with default gray color values
        assigned to all vertices.
        """

        tmp_ctable = cltmisc.colors_to_table(colors=color, alpha_values=alpha)
        tmp_ctable[:, :3] = tmp_ctable[:, :3] / 255  # Ensure colors are between 0 and 1

        self._store_parcellation_data(
            np.ones((self.mesh.n_points,), dtype=np.uint32) * int(tmp_ctable[0, 4]),
            tmp_ctable,
            ["surface"],
            "surface",
        )

    ##############################################################################################
    def create_mesh_from_arrays(
        self, vertices: np.ndarray, faces: np.ndarray
    ) -> pv.PolyData:
        """
        Create PyVista mesh object from vertex and face arrays.

        Parameters
        ----------
        vertices : np.ndarray
            Vertex coordinates with shape (n_vertices, 3).

        faces : np.ndarray
            Face connectivity with shape (n_faces, 3).

        Returns
        -------
        pv.PolyData
            PyVista mesh object with vertices, faces, and default surface colors.

        Raises
        ------
        ValueError
            If arrays have incorrect shapes or face indices are invalid.

        Notes
        -----
        Validates input arrays and creates properly formatted PyVista mesh
        with default surface colors. Adds normals to point data if provided.

        Examples
        --------
        >>> surface = Surface()
        >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        >>> faces = np.array([[0, 1, 2]])
        >>> mesh = surface.create_mesh_from_arrays(vertices, faces)
        >>> print(f"Created mesh with {mesh.n_points} vertices")
        """

        # Validate array shapes
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("Vertices array must have shape (n_vertices, 3)")

        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Faces array must have shape (n_faces, 3)")

        # Check that face indices are valid
        if np.any(faces >= len(vertices)) or np.any(faces < 0):
            raise ValueError(
                "Face indices must be valid indices into the vertices array"
            )

        mesh = self._create_pyvista_mesh(vertices, faces)

        return mesh

    ##############################################################################################
    def _create_pyvista_mesh(
        self, vertices: np.ndarray, faces: np.ndarray
    ) -> pv.PolyData:
        """
        Internal method to create PyVista mesh from vertices and faces.

        Parameters
        ----------
        vertices : np.ndarray
            Vertex coordinates with shape (n_vertices, 3).

        faces : np.ndarray
            Face connectivity with shape (n_faces, 3).

        Returns
        -------
        pv.PolyData
            PyVista mesh object with default surface colors.

        Notes
        -----
        Handles PyVista-specific formatting requirements including adding
        the face size prefix and setting up default point data.
        """

        # Add column with 3's to faces array for PyVista
        faces_pv = np.c_[np.full(len(faces), 3), faces]

        mesh = pv.PolyData(vertices, faces_pv)
        vertices_colors = (
            np.ones((len(vertices), 3), dtype=np.uint8) * 240
        )  # Default colors
        mesh.point_data["surface"] = np.c_[
            vertices_colors, np.ones(len(vertices), dtype=np.uint8) * 255
        ]
        return mesh

    ##############################################################################################
    def get_vertices(self) -> np.ndarray:
        """
        Get vertex coordinates from the surface mesh.

        Returns
        -------
        np.ndarray
            Array of vertex coordinates with shape (n_vertices, 3).

        Raises
        ------
        RuntimeError
            If no surface data has been loaded.

        Examples
        --------
        >>> surface = Surface('lh.pial')
        >>> vertices = surface.get_vertices()
        >>> print(f"Surface has {len(vertices)} vertices")
        >>> print(f"First vertex: {vertices[0]}")
        """

        if not self.is_loaded():
            raise RuntimeError("No surface data loaded. Load data first.")
        return self.mesh.points

    ##############################################################################################
    def get_faces(self) -> np.ndarray:
        """
        Get face connectivity from the surface mesh.

        Returns
        -------
        np.ndarray
            Array of face indices with shape (n_faces, 3). Each row contains
            three vertex indices forming a triangular face.

        Raises
        ------
        RuntimeError
            If no surface data has been loaded.

        Notes
        -----
        Extracts face connectivity from PyVista's internal format which stores
        faces as [n_vertices, vertex_id1, vertex_id2, ...]. This method returns
        only the vertex indices in standard format.

        Examples
        --------
        >>> surface = Surface('lh.pial')
        >>> faces = surface.get_faces()
        >>> print(f"Surface has {len(faces)} triangular faces")
        >>> print(f"First face connects vertices: {faces[0]}")
        """

        if not self.is_loaded():
            raise RuntimeError("No surface data loaded. Load data first.")

        # PyVista stores faces as [n_vertices, vertex_id1, vertex_id2, ...]
        # We need to extract just the vertices indices
        faces_raw = self.mesh.faces
        n_faces = self.mesh.n_cells
        faces = faces_raw.reshape(n_faces, 4)[
            :, 1:4
        ]  # Skip the first column (n_vertices)
        return faces

    ###############################################################################################
    def get_edges(self, return_counts: bool = False) -> np.ndarray:
        """
        Extract unique edges from a triangular mesh using vectorized operations.

        This function efficiently extracts all unique edges from a triangular mesh
        represented as a faces array. Each triangle contributes three edges, and
        the function automatically removes duplicates that occur when triangles
        share edges.

        Parameters
        ----------
        faces : np.ndarray of shape (n_faces, 3)
            Array where each row represents a triangular face defined by three
            vertex indices. Vertex indices should be non-negative integers.

        return_counts : bool, optional
            If True, also return the count of how many faces each edge belongs to.
            This is useful for identifying boundary edges (count=1) vs interior
            edges (count=2). Default is False.

        Returns
        -------
        edges : np.ndarray of shape (n_edges, 2)
            Array of unique edges where each row contains two vertex indices
            [v1, v2] with v1 <= v2. Edges are sorted lexicographically.

        counts : np.ndarray of shape (n_edges,), optional
            Number of faces that contain each edge. Only returned if
            return_counts=True. Boundary edges have count=1, interior edges
            have count=2.

        Raises
        ------
        ValueError
            If faces array does not have exactly 3 columns (not triangular).
            If faces array is empty.
            If faces array contains negative indices.

        Examples
        --------
        >>> # Simple triangular mesh with 2 triangles sharing an edge
        >>> faces = np.array([[0, 1, 2], [1, 3, 2]])
        >>> edges = get_edges(faces)
        >>> print(edges)
        [[0 1]
        [0 2]
        [1 2]
        [1 3]
        [2 3]]

        >>> # Get edge counts to identify boundary vs interior edges
        >>> edges, counts = get_edges(faces, return_counts=True)
        >>> boundary_edges = edges[counts == 1]
        >>> interior_edges = edges[counts == 2]
        >>> print("Boundary edges:", boundary_edges)
        >>> print("Interior edges:", interior_edges)
        Boundary edges: [[0 1]
                        [0 2]
                        [1 3]
                        [2 3]]
        Interior edges: [[1 2]]

        >>> # Cube mesh (8 vertices, 12 triangular faces)
        >>> cube_faces = np.array([
        ...     [0, 1, 2], [0, 2, 3],  # Bottom face
        ...     [4, 5, 6], [4, 6, 7],  # Top face
        ...     [0, 1, 5], [0, 5, 4],  # Front face
        ...     [2, 3, 7], [2, 7, 6],  # Back face
        ...     [0, 3, 7], [0, 7, 4],  # Left face
        ...     [1, 2, 6], [1, 6, 5]   # Right face
        ... ])
        >>> edges = get_edges(cube_faces)
        >>> print(f"Cube has {len(edges)} unique edges")
        Cube has 18 unique edges

        Notes
        -----
        This function uses vectorized NumPy operations for high performance on
        large meshes. The algorithm:

        1. Extracts all three edges from each triangle simultaneously
        2. Sorts vertex pairs to canonical form (smaller index first)
        3. Uses numpy.unique to efficiently remove duplicates

        Time complexity: O(n log n) where n is the number of faces
        Space complexity: O(n) for intermediate arrays

        For non-triangular meshes, use the general `extract_edges_from_faces`
        function instead.

        The canonical edge representation ensures that edge (i, j) and edge (j, i)
        are treated as the same edge, with the final representation always having
        the smaller vertex index first.

        See Also
        --------
        extract_edges_from_faces : General version for arbitrary polygon meshes
        numpy.unique : Used internally for deduplication
        """

        # Getting the faces array from the mesh
        faces = self.mesh.faces[
            :, 1:4
        ]  # Extract only the vertex indices, skip the first column

        # Input validation
        if faces.size == 0:
            raise ValueError("Faces array cannot be empty")

        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(
                f"Faces array must have shape (n_faces, 3), got {faces.shape}"
            )

        if np.any(faces < 0):
            raise ValueError("Faces array cannot contain negative vertex indices")

        # Extract all edges from all triangles using vectorized operations
        # Each triangle contributes 3 edges: (v0,v1), (v1,v2), (v2,v0)
        all_edges = np.concatenate(
            [
                faces[:, [0, 1]],  # Edge from vertex 0 to vertex 1
                faces[:, [1, 2]],  # Edge from vertex 1 to vertex 2
                faces[:, [2, 0]],  # Edge from vertex 2 to vertex 0
            ],
            axis=0,
        )

        # Sort each edge to canonical form (smaller vertex index first)
        # This ensures (i,j) and (j,i) are treated as the same edge
        canonical_edges = np.sort(all_edges, axis=1)

        # Remove duplicate edges and optionally count occurrences
        if return_counts:
            unique_edges, counts = np.unique(
                canonical_edges, axis=0, return_counts=True
            )
            return unique_edges, counts
        else:
            unique_edges = np.unique(canonical_edges, axis=0)
            return unique_edges

    ###############################################################################################
    def get_boundary_edges(self) -> np.ndarray:
        """
        Extract only the boundary edges from a triangular mesh.

        Boundary edges are those that belong to only one triangle, indicating
        the mesh boundary or holes in the mesh.

        Parameters
        ----------
        faces : np.ndarray of shape (n_faces, 3)
            Triangular mesh faces array.

        Returns
        -------
        boundary_edges : np.ndarray of shape (n_boundary_edges, 2)
            Array of boundary edges where each edge belongs to only one face.

        Examples
        --------
        >>> # Mesh with a hole (incomplete sphere)
        >>> faces = np.array([[0, 1, 2], [1, 3, 2], [3, 4, 2]])
        >>> boundary = get_boundary_edges(faces)
        >>> print("Boundary edges:", boundary)
        """

        # Getting the faces array from the mesh
        faces = self.mesh.faces[
            :, 1:4
        ]  # Extract only the vertex indices, skip the first column

        edges, counts = self.get_edges(faces, return_counts=True)
        return edges[counts == 1]

    ###############################################################################################
    def get_manifold_edges(self) -> np.ndarray:
        """
        Extract only the manifold (interior) edges from a triangular mesh.

        Manifold edges are those shared by exactly two triangles, indicating
        proper mesh topology without boundaries or non-manifold geometry.

        Parameters
        ----------
        faces : np.ndarray of shape (n_faces, 3)
            Triangular mesh faces array.

        Returns
        -------
        manifold_edges : np.ndarray of shape (n_manifold_edges, 2)
            Array of manifold edges where each edge belongs to exactly two faces.

        Examples
        --------
        >>> # Closed mesh (tetrahedron)
        >>> faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]])
        >>> manifold = get_manifold_edges(faces)
        >>> print("Manifold edges:", manifold)
        """
        # Getting the faces array from the mesh
        faces = self.mesh.faces[:, 1:4]  # Extract only the vertex indices

        edges, counts = self.get_edges(faces, return_counts=True)
        return edges[counts == 2]

    ##############################################################################################
    def compute_normals(self) -> None:
        """
        Compute and store vertex normals for the surface mesh.

        Calculates unit normal vectors for each vertex and stores them in the
        mesh point data under the key "Normals". Normals are automatically
        normalized to unit length.

        Raises
        ------
        RuntimeError
            If no surface data has been loaded.
        RuntimeError
            If computed normals have zero length and cannot be normalized.

        Notes
        -----
        Uses PyVista's built-in normal computation which averages face normals
        at each vertex. The resulting normals are forced to be unit vectors.
        Overwrites any existing normals in the mesh.

        Examples
        --------
        >>> surface = Surface('lh.pial')
        >>> surface.compute_normals()
        >>> normals = surface.get_normals()
        >>> print(f"Computed {len(normals)} unit normal vectors")
        >>>
        >>> # Check that normals are unit vectors
        >>> norms = np.linalg.norm(normals, axis=1)
        >>> print(f"Normal lengths range: {norms.min():.3f} - {norms.max():.3f}")
        """

        if not self.is_loaded():
            raise RuntimeError("No surface data loaded. Load data first.")

        self.mesh.compute_normals(inplace=True)  # Compute normals and store in mesh

        # Force the normals to be unit vectors
        if "Normals" in self.mesh.point_data:
            normals = self.mesh.point_data["Normals"]
            norms = np.linalg.norm(normals, axis=1)
            if np.any(norms > 0):
                self.mesh.point_data["Normals"] = normals / norms[:, np.newaxis]
            else:
                raise RuntimeError(
                    "Computed normals have zero length. Cannot normalize."
                )

    ##############################################################################################
    def get_normals(self) -> Optional[np.ndarray]:
        """
        Get vertex normals from the surface mesh if available.

        Returns
        -------
        np.ndarray or None
            Array of normal vectors with shape (n_vertices, 3) if normals
            have been computed, None otherwise.

        Notes
        -----
        Returns None if normals haven't been computed yet. Use compute_normals()
        to calculate normals before calling this method.

        Examples
        --------
        >>> surface = Surface('lh.pial')
        >>> normals = surface.get_normals()
        >>> if normals is not None:
        ...     print(f"Found {len(normals)} normal vectors")
        ... else:
        ...     print("No normals computed yet")
        ...     surface.compute_normals()
        ...     normals = surface.get_normals()
        """

        if not self.is_loaded():
            return None

        return self.mesh.point_data.get("Normals", None)

    ##############################################################################################
    def load_annotation(
        self,
        annotation: Union[str, Path, cltfree.AnnotParcellation],
        parc_name: str = None,
    ) -> None:
        """
        Load parcellation annotation onto surface for visualization.

        Loads FreeSurfer annotation files or AnnotParcellation objects,
        storing labels and color information for region-based visualization.

        Parameters
        ----------
        annotation : str , Path or AnnotParcellation
            Path to annotation file (.annot) or AnnotParcellation object.

        parc_name : str
            Name for parcellation reference in visualizations.

        Raises
        ------
        FileNotFoundError
            If annotation file cannot be found.

        ValueError
            If invalid input type or vertex count mismatch.

        Examples
        --------
        >>> # Load Desikan-Killiany parcellation
        >>> surface.load_annotation('lh.aparc.annot', 'aparc')
        >>>
        >>> # Load from object
        >>> annot = AnnotParcellation('lh.aparc.a2009s.annot')
        >>> surface.load_annotation(annot, 'destrieux')
        """

        if isinstance(annotation, Path):
            annotation = str(annotation)

        # Handle different input types
        if isinstance(annotation, str):
            # Input is a file path
            if not os.path.isfile(annotation):
                raise FileNotFoundError(f"Annotation file not found: {annotation}")

            # Create AnnotParcellation object to benefit from its processing and cleaning
            annot_parc = cltfree.AnnotParcellation()
            annot_parc.load_from_file(parc_file=annotation, annot_id=parc_name)

        elif (
            hasattr(annotation, "codes")
            and hasattr(annotation, "regtable")
            and hasattr(annotation, "regnames")
        ):
            # Input is an AnnotParcellation object
            annot_parc = copy.deepcopy(annotation)
            if parc_name is not None:
                annot_parc.id = parc_name

        else:
            raise ValueError(
                "annot_input must be either a file path (str) or an AnnotParcellation object"
            )

        # Extract the processed and cleaned data from AnnotParcellation
        labels = annot_parc.codes
        reg_ctable = annot_parc.regtable.astype(np.float32)  # Ensure colors are float32
        reg_names = annot_parc.regnames  # Already processed as strings

        # Validate that the number of vertices matches
        if len(labels) != self.mesh.n_points:
            raise ValueError(
                f"Number of vertices in annotation ({len(labels)}) does not match surface ({self.mesh.n_points})"
            )
        # If parc_name is not provided, use the annot_id from AnnotParcellation

        if parc_name is None:
            parc_name = annot_parc.id

        # Store the parcellation data
        tmp_colors = reg_ctable[:, :3]

        reg_ctable[:, :3] = reg_ctable[:, :3] / 255  # Ensure colors are between 0 and 1

        # If all the opacity values are 0 set them to 1
        if np.all(reg_ctable[:, 3] == 0):
            reg_ctable[:, 3] = 1.0

        self._store_parcellation_data(labels, reg_ctable, reg_names, parc_name)

        # Store reference to AnnotParcellation object for advanced operations
        self.colortables[parc_name]["annot_object"] = annot_parc

    ##############################################################################################
    def _store_parcellation_data(
        self,
        labels: np.ndarray,
        reg_ctable: np.ndarray,
        reg_names: List[str],
        parc_name: str,
    ) -> None:
        """
        Store parcellation data and create color mappings.

        Internal method for organizing parcellation labels, colors, and names
        in surface object for visualization and analysis.

        Parameters
        ----------
        labels : np.ndarray
            Label values for each vertex.

        reg_ctable : np.ndarray
            Color table with RGBA values for each region.

        reg_names : list
            Region names corresponding to color table.

        parc_name : str
            Name of the parcellation.

        Notes
        -----
        Stores labels in mesh point data and creates organized color table
        structure. Also calls color table creation for visualization.
        """

        # Store labels in mesh
        self.mesh.point_data[parc_name] = labels

        # Store parcellation information in organized structure
        self.colortables[parc_name] = {
            "struct_names": reg_names,
            "color_table": reg_ctable,
            "lookup_table": None,  # Will be populated by _create_parcellation_colortable if needed
        }

    ##############################################################################################
    def _get_parcellation_data(
        self, annotation: Union[str, Path, cltfree.AnnotParcellation]
    ) -> cltfree.AnnotParcellation:
        """
        Load or retrieve parcellation data from annotation file or object.
        Handles both file paths and AnnotParcellation objects, ensuring
        consistent data retrieval for visualization and analysis.

        Parameters
        ----------
        annotation : str, Path or AnnotParcellation
            Path to annotation file (.annot) or AnnotParcellation object.

        Returns
        -------
        cltfree.AnnotParcellation
            AnnotParcellation object containing parcellation data.

        Raises
        ------
        FileNotFoundError
            If annotation file cannot be found and no colortable is available.

        ValueError
            If annotation input type is invalid or does not match expected formats.

        Notes
        -----
        - If annotation is a file path, it checks for existence and loads the data.

        - If annotation is an AnnotParcellation object, it returns a deep copy to avoid
        modifying the original object.

        - If the annotation file is not found, it attempts to load from colortables
        if available.
        - If no colortable is available, it raises a FileNotFoundError.

        Examples
        --------
        >>> # Load from annotation file
        >>> annot_parc = surface._get_parcellation_data('lh.aparc.annot')
        >>>
        """

        if isinstance(annotation, Path):
            annotation = str(annotation)

        # If map_name is not provided, use the first column name from the DataFrame
        if isinstance(annotation, str):
            # Check if the annotation file exists
            if not os.path.isfile(annotation) and annotation in self.mesh.point_data:
                # If the annotation file is not found, try to load it from the colortables

                # Extract the annotation data
                maps_array = self.mesh.point_data[annotation]

                # If there is a colortable for this map, use it
                if annotation in self.colortables:
                    ctable = self.colortables[annotation]["color_table"]
                    struct_names = self.colortables[annotation]["struct_names"]

                    # Create AnnotParcellation object from the data
                    parc = cltfree.AnnotParcellation()
                    parc.create_from_data(maps_array, ctable, struct_names)
                else:
                    raise FileNotFoundError(
                        f"Annotation file not found: {annotation} and no colortable available"
                    )

            elif os.path.isfile(annotation):
                # Create AnnotParcellation object to benefit from its processing and cleaning
                parc = cltfree.AnnotParcellation()
                parc.load_from_file(parc_file=annotation)
            else:
                raise FileNotFoundError(f"Annotation file not found: {annotation}")

        elif isinstance(annotation, cltfree.AnnotParcellation):
            parc = copy.deepcopy(
                annotation
            )  # Use a copy to avoid modifying the original object

        return parc

    ##############################################################################################
    def load_scalar_maps(
        self,
        scalar_map: Union[str, Path, np.ndarray, pd.DataFrame],
        annotation: Union[str, Path, cltfree.AnnotParcellation] = None,
        maps_names: Union[str, List[str]] = None,
    ) -> None:
        """
        Load data from a FreeSurfer vertex-wise map, a numpy array, a CSV file or
        pandas Dataframe onto surface for visualization.

        Handles both vertex-wise data (one value per vertex) and region-wise data
        (requires annotation for mapping to vertices). It is important that the CSV file
        has a header row with column names because the first row is used to name the maps.

        If it contains region-wise data then the Annotation file is mandatory.

        Parameters
        ----------
        scalar_map : str, Path, pd.DataFrame
            Path to a FreeSurfer vertex-wise map file, a CSV file, a numpy array or a
            pandas DataFrame.

        annotation : str, Path or AnnotParcellation, optional
            Annotation file/object for mapping region data to vertices.
            Required if the Dataframe has region-wise data. Default is None.

        maps_names : str or list, optional
            Names for scalar data. If None, uses column names from CSV.
            Default is None.

        Raises
        ------
        FileNotFoundError
            If map file or annotation file cannot be found.

        ValueError
            If annot_file required but not provided or invalid type.

        ValueError
            If maps_names length does not match number of columns in CSV.

        Notes
        -----
        Automatically detects if the array, the CSV or the array contains vertex-wise
        or region-wise data based on the number of rows. If the number of rows matches
        the number of vertices in the mesh, it is treated as vertex-wise data.
        Otherwise, it is treated as region-wise data and requires an annotation file to
        map region values to vertices.

        The annotation can be provided as a file path, a string or as an AnnotParcellation object.
        If the annotation file is not found, it will try to load it from the colortables
        associated with the surface.

        If maps_names is not provided, it uses the column names from the CSV file.
        If the annotation is provided as a string, it will try to load it as an AnnotParcellation
        object. If it is provided as an AnnotParcellation object, it will use it directly.


        Examples
        --------
        >>> surf_lh = cltsurf.Surface("/opt/freesurfer/subjects/fsaverage/surf/lh.pial")

        >>> # Example 1: Reading a region-wise map from a CSV file and selecting a specific column name
        >>> print("Example 1: Reading a region-wise map from a CSV file with a specific column name")
        >>> surf_lh.load_maps_scalar_maps("/tmp/values.csv",
                                        annotation="/opt/freesurfer/subjects/fsaverage/label/lh.aparc.annot",
                                        maps_names="region_index")

        >>> print("Loaded maps from CSV file for an specific column name:")
        >>> print(surf_lh.list_overlays())
        >>> print("")

        >>> # Example 2: Reading a region-wise map from a dataframe and selecting a column with specified name
        >>> import pandas as pd
        >>> values_df = pd.read_csv("/tmp/values.csv")
        >>> print("Example 2: Reading a region-wise map from a DataFrame with specified names")
        >>> surf_lh.load_maps_scalar_maps(values_df,
                                        annotation="/opt/freesurfer/subjects/fsaverage/label/lh.aparc.annot",
                                        maps_names=["value"])

        >>> print(" Loaded maps from DataFrame with specified names:")
        >>> print(surf_lh.list_overlays())
        >>> print("")

        >>> # Example 3: Reading a region-wise map from a numpy array without specifiying names
        >>> import pandas as pd
        >>> print("Example 3: Reading a region-wise map from a numpy array without specifying names")
        >>> values_df = pd.read_csv("/tmp/values.csv")
        >>> surf_lh.load_maps_scalar_maps(values_df.to_numpy(),
                                        annotation="/opt/freesurfer/subjects/fsaverage/label/lh.aparc.annot")

        >>> print("Loaded maps from numpy array without specifying names:")
        >>> print(surf_lh.list_overlays())
        >>> print("")

        >>> ######### Creating a csv with values as the number of vertices
        >>> import pandas as pd
        >>> n_points = surf_lh.mesh.n_points
        >>> values_df = pd.DataFrame({'vertex_index': np.arange(n_points), 'vertex_value': np.random.rand(n_points)})

        >>> values_df.to_csv("/tmp/values-vertexwise.csv", index=False)

        >>> # Example 4: Loading vertex-wise maps from a CSV file
        >>> print("Example 4: Loading vertex-wise maps from a CSV file")
        >>> surf_lh.load_maps_scalar_maps("/tmp/values-vertexwise.csv"
                                        )
        >>> print("Loaded vertex-wise maps from CSV file:")
        >>> print(surf_lh.list_overlays())
        >>> print("")

        >>> # Example 5: Reading a region-wise map from a numpy array and an specified name
        >>> print("Example 5: Reading a region-wise map from a numpy array with specified names")
        >>> import numpy as np
        >>> values_array = np.random.rand(n_points)
        >>> surf_lh.load_maps_scalar_maps(values_array,
                                        maps_names=["ex5_vertex_value_array"])
        >>> print("Loaded vertex-wise maps from numpy array:")
        >>> print(surf_lh.list_overlays())
        >>> print("")

        >>> # Example 6: Creating a numpy array without specifying names
        >>> values_array = np.random.rand(n_points)
        >>> print("Example 6: Creating a numpy array with values as the number of vertices without specifying names")
        >>> surf_lh.load_maps_scalar_maps(values_array)
        >>> print("Loaded vertex-wise maps from numpy array without specifying names:")
        >>> print(surf_lh.list_overlays())
        >>> print("")

        >>> # Example 7: Reading a FreeSurfer map file
        >>> print("Example 7: Reading a FreeSurfer map file")
        >>> surf_lh.load_maps_scalar_maps("/opt/freesurfer/subjects/fsaverage/surf/lh.thickness",
                                        maps_names=["cthickness"])
        >>> print("Loaded vertex-wise maps from FreeSurfer map file:")
        >>> print(surf_lh.list_overlays())
        >>> print("")

        >>> # Example 8: Reading multiple FreeSurfer map files
        >>> print("Example 8: Reading multiple FreeSurfer map files specifying names")
        >>> list_of_maps = [
            "/opt/freesurfer/subjects/fsaverage/surf/lh.thickness",
            "/opt/freesurfer/subjects/fsaverage/surf/lh.curv",
            "/opt/freesurfer/subjects/fsaverage/surf/lh.sulc"]

        >>> maps_names = ["thickness", "curvature", "sulc"]

        >>> for i, map_file in enumerate(list_of_maps):
            surf_lh.load_maps_scalar_maps(map_file, maps_names=maps_names[i])
        >>> print("Loaded vertex-wise maps from FreeSurfer map files:")
        >>> print(surf_lh.list_overlays())
        >>> print("")
        """

        if maps_names is not None:
            if isinstance(maps_names, str):
                maps_names = [maps_names]

        # Load the scalar map based on its type
        try:
            if isinstance(scalar_map, pd.DataFrame):
                # If map_file is a DataFrame, use it directly
                maps_df = copy.deepcopy(scalar_map)

                # Filter the columns that that are equal to the maps_names if maps_names is provided
                if maps_names is not None:
                    maps_df = maps_df[maps_names]

            elif isinstance(scalar_map, np.ndarray):

                if scalar_map.ndim == 1:
                    # If it is a row vector convert it to a column vector
                    # If scalar_map is a 1D numpy array, convert it to a 2D array
                    scalar_map = scalar_map[:, np.newaxis]

                # If the maps names are not provided, create default names
                if maps_names is None:
                    # Create default names for the maps
                    tmp_names = [f"map_{i}" for i in range(scalar_map.shape[1])]
                    maps_df = pd.DataFrame(scalar_map, columns=tmp_names)
                else:
                    if len(maps_names) != scalar_map.shape[1]:
                        raise ValueError(
                            "Length of maps_names must match the number of columns in the numpy array"
                        )
                    # If scalar_map is a numpy array, convert it to a DataFrame
                    maps_df = pd.DataFrame(scalar_map, columns=maps_names)

            else:
                if isinstance(scalar_map, Path):
                    scalar_map = str(scalar_map)

                if not os.path.isfile(scalar_map):
                    raise FileNotFoundError(f"Map file not found: {scalar_map}")

                # Read the map file into a DataFrame
                maps_df = cltmisc.smart_read_table(scalar_map)
                if maps_names is not None:
                    maps_df = maps_df[maps_names]

            if annotation is not None:
                if isinstance(annotation, Path):
                    annotation = str(annotation)

                if not isinstance(annotation, (str, cltfree.AnnotParcellation)):
                    raise ValueError(
                        "annotation must be a string or an AnnotParcellation object"
                    )

            if maps_names is not None:
                if isinstance(maps_names, str):
                    maps_names = [maps_names]

                elif not isinstance(maps_names, list):
                    raise ValueError("maps_names must be a string or a list of strings")

                if len(maps_names) != maps_df.shape[1]:
                    raise ValueError(
                        "Length of maps_names must match the number of columns in the DataFrame"
                    )

            else:
                # If maps_names is not provided, use the column names from the DataFrame
                maps_names = maps_df.columns.tolist()

            # If the number of rows of the dataframe is equal to the number of vertices, we can use it directly
            if maps_df.shape[0] == self.mesh.n_points:
                vertex_maps = maps_df.to_numpy()
            else:
                if annotation is None:
                    raise ValueError(
                        "annotation must be provided if map_file does not match the number of vertices"
                    )
                # Extracting the parcellation data
                parc = self._get_parcellation_data(annotation)

                vertex_maps = parc.map_values(
                    regional_values=maps_df, is_dataframe=True
                ).to_numpy()

            for i, map_name in enumerate(maps_names):
                # Ensure the map data is a 1D array
                map_data = vertex_maps[:, i]

                # Store the map data in the mesh point data
                self.mesh.point_data[map_name] = map_data

        except:
            if isinstance(scalar_map, (str, Path)):
                if not os.path.isfile(scalar_map):
                    raise FileNotFoundError(f"Map file not found: {scalar_map}")

                # Read the map file
                tmp_map = nib.freesurfer.read_morph_data(str(scalar_map))

                if tmp_map.shape[0] == self.mesh.n_points:
                    if maps_names is None:
                        # If maps_names is not provided, use the file name as the map name
                        map_name = os.path.splitext(os.path.basename(scalar_map))[0]
                    else:
                        if len(maps_names) != 1:
                            raise ValueError(
                                "maps_names must be a single string or a list with one name"
                            )

                        self.mesh.point_data[maps_names[0]] = tmp_map

                else:
                    raise ValueError(
                        f"Map file {scalar_map} does not match the number of vertices"
                    )

    ###############################################################################################
    def list_overlays(self) -> Dict[str, str]:
        """
        List all available surface overlays and their data types.

        Categorizes loaded data based on array dimensions and properties to
        identify scalar maps, color data, normals, and other overlay types.

        Returns
        -------
        dict
            Dictionary mapping overlay names to their types:
            - 'scalar': 1D arrays of scalar values per vertex
            - 'color': 2D arrays with RGB color values (shape: n_vertices, 3)
            - 'normals': 2D arrays with unit normal vectors (shape: n_vertices, 3)
            - 'unknown': Arrays with other dimensions or unrecognized format

        Notes
        -----
        Automatically detects data type based on:
        - 1D arrays: Classified as scalar data
        - 2D arrays with 3 columns: Checked for unit vectors (normals) vs colors
        - Other dimensions: Classified as unknown

        Normal vectors are identified by having unit length (norm ≈ 1) and
        containing negative values.

        Examples
        --------
        >>> # Load various data types
        >>> surface.load_scalar_map('thickness.mgh', 'thickness')
        >>> surface.load_annotation('aparc.annot', 'aparc')
        >>> surface.compute_normals()
        >>>
        >>> # List all overlays
        >>> overlays = surface.list_overlays()
        >>> print(overlays)
        {'surface': 'color', 'thickness': 'scalar', 'aparc': 'scalar', 'Normals': 'normals'}
        >>>
        >>> # Filter for scalar maps only
        >>> scalar_maps = {k: v for k, v in overlays.items() if v == 'scalar'}
        >>> print(f"Available scalar maps: {list(scalar_maps.keys())}")
        """

        overlays = {}

        for key in self.mesh.point_data.keys():
            tmp = self.mesh.point_data[key]
            if isinstance(tmp, np.ndarray) and tmp.ndim == 1:

                # If it's a 1D array, but has a color table, treat it as scalar with colortable
                if key in self.colortables:
                    overlays[key] = "scalar_with_colortable"
                else:
                    overlays[key] = "scalar"

            elif isinstance(tmp, np.ndarray) and tmp.ndim == 2:
                if tmp.shape[1] == 3:
                    # If there are negative values and the norm is equal to 1, it's likely normals
                    if np.all(np.round(np.linalg.norm(tmp, axis=1)) == 1) and np.any(
                        tmp < 0
                    ):
                        overlays[key] = "normals"
                    else:
                        overlays[key] = "color"
                else:
                    overlays[key] = "unknown"

        return overlays

    ##############################################################################################
    def set_active_overlay(self, overlay_name: str) -> None:
        """
        Set the active overlay for visualization.

        Parameters
        ----------
        overlay_name : str
            Name of the overlay to set as active

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the specified overlay is not found in mesh point data

        Examples
        --------
        >>> surface.set_active_overlay("thickness")
        >>> surface.set_active_overlay("aparc")
        """
        if overlay_name not in self.mesh.point_data:
            raise ValueError(f"Overlay '{overlay_name}' not found in mesh point data")

        self.mesh.set_active_scalars(overlay_name)

    ##############################################################################################
    def remove_overlay(self, overlay_name: str) -> None:
        """
        Set the active overlay for visualization.

        Designates which data array should be used as the primary scalar field
        for coloring and visualization in PyVista plots. This affects how the
        surface is colored when rendered.

        Parameters
        ----------
        overlay_name : str
            Name of the overlay to set as active. Must exist in mesh point data.

        Raises
        ------
        ValueError
            If the specified overlay is not found in mesh point data.

        Notes
        -----
        The active overlay determines which data is used for:
        - Surface coloring in visualizations
        - Colormap application
        - Scalar value display in interactive plots

        PyVista uses the active scalars for automatic coloring unless
        explicitly overridden in visualization methods.

        Examples
        --------
        >>> # Set thickness as active for visualization
        >>> surface.set_active_overlay('thickness')
        >>>
        >>> # Switch to parcellation display
        >>> surface.set_active_overlay('aparc')
        >>>
        >>> # Check available overlays first
        >>> overlays = surface.list_overlays()
        >>> if 'curvature' in overlays:
        ...     surface.set_active_overlay('curvature')
        """

        # Check if overlay exists
        if (
            overlay_name not in self.mesh.point_data
            and overlay_name not in self.colortables
        ):
            raise ValueError(f"Overlay '{overlay_name}' not found")

        # Remove from mesh point data
        if overlay_name in self.mesh.point_data:
            del self.mesh.point_data[overlay_name]

        # Remove from colortables storage
        if overlay_name in self.colortables:
            del self.colortables[overlay_name]

        # If this was the active scalar, reset to surface default
        try:
            active_scalars = self.mesh.active_scalars_name
            if active_scalars == overlay_name:
                if "surface" in self.mesh.point_data:
                    self.mesh.set_active_scalars("surface")
                else:
                    # Find the first available overlay
                    remaining_overlays = list(self.mesh.point_data.keys())
                    if remaining_overlays:
                        self.mesh.set_active_scalars(remaining_overlays[0])
        except:
            # If there's any issue with active scalars, just continue
            pass

    ##############################################################################################
    def get_overlay_info(self, overlay_name: str) -> Dict:
        """
        Get information about a specific surface overlay.

        Parameters
        ----------
        overlay_name : str
            Name of the overlay to query.

        Returns
        -------
        Dict
            Dictionary containing overlay metadata with keys:

            - 'name' : str
                Name of the overlay.
            - 'data_shape' : tuple
                Shape of the overlay data array.
            - 'data_type' : str
                NumPy data type of the overlay values.
            - 'has_colortable' : bool
                Whether the overlay has an associated color table.
            - 'num_regions' : int, optional
                Number of regions (if parcellation overlay).
            - 'region_names' : list of str, optional
                Names of regions (if parcellation overlay).
            - 'has_annot_object' : bool, optional
                Whether annotation object is available (if parcellation overlay).

        Raises
        ------
        ValueError
            If the overlay is not found.

        Examples
        --------
        >>> surface = Surface()
        >>> info = surface.get_overlay_info("aparc")
        >>> print(f"Overlay has {info['num_regions']} regions")
        >>> print(f"Data type: {info['data_type']}")
        """

        if overlay_name not in self.mesh.point_data:
            raise ValueError(f"Overlay '{overlay_name}' not found")

        info = {
            "name": overlay_name,
            "data_shape": self.mesh.point_data[overlay_name].shape,
            "data_type": str(self.mesh.point_data[overlay_name].dtype),
            "has_colortable": overlay_name in self.colortables,
        }

        # Add colortable info if available
        if overlay_name in self.colortables:
            ctable_info = self.colortables[overlay_name]
            info["num_regions"] = len(ctable_info["struct_names"])
            info["region_names"] = ctable_info["struct_names"]
            info["has_annot_object"] = "annot_object" in ctable_info

        return info

    ##############################################################################################
    def get_region_vertices(self, parc_name: str, region_name: str) -> np.ndarray:
        """
        Get vertices indices for a specific region in a parcellation.

        Parameters
        ----------
        parc_name : str
            Name of the parcellation

        region_name : str
            Name of the region

        Returns
        -------
        np.ndarray
            Array of vertices indices belonging to the region

        Raises
        ------
        ValueError
            If the parcellation is not found

        ValueError
            If the region is not found in the parcellation

        Examples
        --------
        >>> # Get vertices in the precentral gyrus
        >>> vertices = surface.get_region_vertices("aparc", "precentral")
        >>> print(f"Precentral region has {len(vertices)} vertices")
        >>>
        >>> # Get all vertices in superior frontal region
        >>> vertices = surface.get_region_vertices("aparc", "superiorfrontal")
        """
        if parc_name not in self.colortables:
            raise ValueError(f"Parcellation '{parc_name}' not found")

        # Use AnnotParcellation object if available for more robust lookup
        if "annot_object" in self.colortables[parc_name]:
            annot_obj = self.colortables[parc_name]["annot_object"]
            return annot_obj.get_region_vertices(region_name)
        else:
            # Fallback to manual lookup
            if region_name not in self.colortables[parc_name]["struct_names"]:
                raise ValueError(
                    f"Region '{region_name}' not found in parcellation '{parc_name}'"
                )

            # Find the label value for this region
            region_idx = self.colortables[parc_name]["struct_names"].index(region_name)
            label_value = self.colortables[parc_name]["color_table"][region_idx, 4]

            # Get vertices with this label
            labels = self.mesh.point_data[parc_name]
            return np.where(labels == label_value)[0]

    ##############################################################################################
    def get_region_info(self, parc_name: str, region_name: str) -> Dict:
        """
        Get comprehensive information about a region in a parcellation.

        Parameters
        ----------
        parc_name : str
            Name of the parcellation

        region_name : str
            Name of the region

        Returns
        -------
        Dict
            Dictionary with region information containing:
            - 'name': str, region name
            - 'index': int, region index in parcellation
            - 'label_value': int, label value used in annotation
            - 'color_rgb': np.ndarray, RGB color values (0-255)
            - 'color_rgba': np.ndarray, RGBA color values (0-255)
            - 'vertex_count': int, number of vertices in region
            - 'vertex_indices': np.ndarray, indices of vertices in region

        Raises
        ------
        ValueError
            If the parcellation or region is not found

        Examples
        --------
        >>> info = surface.get_region_info("aparc", "precentral")
        >>> print(f"Region: {info['name']}")
        >>> print(f"Vertices: {info['vertex_count']}")
        >>> print(f"Color: {info['color_rgb']}")
        """
        if parc_name not in self.colortables:
            raise ValueError(f"Parcellation '{parc_name}' not found")

        # Use AnnotParcellation object if available
        if "annot_object" in self.colortables[parc_name]:
            annot_obj = self.colortables[parc_name]["annot_object"]
            return annot_obj.get_region_info(region_name)
        else:
            # Fallback to manual calculation
            vertices = self.get_region_vertices(parc_name, region_name)
            region_idx = self.colortables[parc_name]["struct_names"].index(region_name)
            color_table = self.colortables[parc_name]["color_table"]

            return {
                "name": region_name,
                "index": region_idx,
                "label_value": color_table[region_idx, 4],
                "color_rgb": color_table[region_idx, :3],
                "color_rgba": color_table[region_idx, :4],
                "vertex_count": len(vertices),
                "vertex_indices": vertices,
            }

    ##############################################################################################
    def list_regions(self, parc_name: str) -> Union[pd.DataFrame, Dict]:
        """
        Get a summary of all regions in a parcellation.

        Parameters
        ----------
        parc_name : str
            Name of the parcellation

        Returns
        -------
        pd.DataFrame or Dict
            DataFrame with region information if AnnotParcellation object is available,
            otherwise a dictionary with basic information. Contains region names,
            label values, vertices counts, and colors.

        Raises
        ------
        ValueError
            If the parcellation is not found

        Examples
        --------
        >>> regions = surface.list_regions("aparc")
        >>> if isinstance(regions, pd.DataFrame):
        ...     print(regions.head())
        ... else:
        ...     for name, info in regions.items():
        ...         print(f"{name}: {info['vertex_count']} vertices")
        """

        if parc_name not in self.colortables:
            raise ValueError(f"Parcellation '{parc_name}' not found")

        # Use AnnotParcellation object if available
        if "annot_object" in self.colortables[parc_name]:
            annot_obj = self.colortables[parc_name]["annot_object"]
            return annot_obj.list_regions()
        else:
            # Fallback to basic information
            ctable_info = self.colortables[parc_name]
            regions = {}
            for i, name in enumerate(ctable_info["struct_names"]):
                color_table = ctable_info["color_table"]
                vertices = self.get_region_vertices(parc_name, name)
                regions[name] = {
                    "label_value": color_table[i, 4],
                    "vertex_count": len(vertices),
                    "color_rgb": color_table[i, :3].tolist(),
                }
            return regions

    ##############################################################################################
    def get_vertexwise_colors(
        self,
        overlay_name: str = "surface",
        colormap: str = "viridis",
        vmin: np.float64 = None,
        vmax: np.float64 = None,
    ) -> None:
        """
        Compute vertices colors for visualization based on the specified overlay.

        This method processes the overlay data and creates appropiate vertices colors
        for visualization, handling both scalar data (with colormaps) and
        categorical data (with discrete color tables).

        Parameters
        ----------
        overlay_name : str, optional
            Name of the overlay to visualize. If None, the first available overlay is used.

        colormap : str, optional
            Colormap to use for scalar overlays. If None, uses parcellation color table
            for categorical data or 'viridis' for scalar data.

        vmin : np.float64, optional
            Minimum value for scaling the colormap. If None, uses the minimum value of the overlay

        vmax : np.float64, optional
            Maximum value for scaling the colormap. If None, uses the maximum value of the overlay
        If both vmin and vmax are None, the colormap will be applied to the full range of the overlay values.
        If both are provided, they will be used to scale the colormap.

        Returns
        -------
        vertices_colors : np.ndarray
            Array of RGBA colors for each vertex in the mesh.

        Raises
        ------
        ValueError
            If the specified overlay is not found in the mesh point data

        ValueError
            If no overlays are available

        Notes
        -----
        This method sets the vertices colors based on the specified overlay.


        Examples
        --------
        >>> # Prepare colors for a parcellation (uses discrete colors)
        >>> surface.get_vertexwise_colors(overlay_name="aparc")
        >>>
        >>> # Prepare colors for scalar data with custom colormap
        >>> surface.get_vertexwise_colors(overlay_name="thickness", colormap="hot")
        >>>
        >>> # Prepare colors for the surface overlay
        >>> surface.get_vertexwise_colors()
        """

        # Get the list of overlays
        overlay_dict = self.list_overlays()

        # If the dictionary is empty
        overlays = list(overlay_dict.keys())
        if overlay_name is None:
            overlay_name = overlays[0] if overlay_dict else None

        if overlay_name not in overlays:
            raise ValueError(
                f"Overlay '{overlay_name}' not found. Available overlays: {', '.join(overlays)}"
            )

        # Getting the values of the overlay
        vertex_values = self.mesh.point_data[overlay_name]

        # if colortables is an attribute of the class, use it
        if hasattr(self, "colortables"):
            dict_ctables = self.colortables

            # Check if the overlay is on the colortables
            if overlay_name in dict_ctables.keys():
                # Use the colortable associated with the parcellation

                vertices_colors = cltmisc.get_colors_from_colortable(
                    vertex_values, self.colortables[overlay_name]["color_table"]
                )
            else:
                # Use the colormap for scalar data
                vertices_colors = cltmisc.values2colors(
                    vertex_values,
                    cmap=colormap,
                    output_format="rgb",
                    vmin=vmin,
                    vmax=vmax,
                )
        else:
            vertices_colors = cltmisc.values2colors(
                vertex_values,
                cmap=colormap,
                output_format="rgb",
                vmin=vmin,
                vmax=vmax,
            )

        return vertices_colors

    ##############################################################################################
    def prepare_colors(
        self,
        overlay_name: str = None,
        cmap: str = "viridis",
        vmin: np.float64 = None,
        vmax: np.float64 = None,
    ) -> None:
        """
        Prepare vertices colors for visualization based on the specified overlay.

        This method processes the overlay data and creates appropiate vertices colors
        for visualization, handling both scalar data (with colormaps) and
        categorical data (with discrete color tables).

        Parameters
        ----------
        overlay_name : str, optional
            Name of the overlay to visualize. If None, the first available overlay is used.

        cmap : str, optional
            Colormap to use for scalar overlays. If None, uses parcellation color table
            for categorical data or 'viridis' for scalar data.

        vmin : np.float64, optional
            Minimum value for scaling the colormap. If None, uses the minimum value of the overlay

        vmax : np.float64, optional
            Maximum value for scaling the colormap. If None, uses the maximum value of the overlay
        If both vmin and vmax are None, the colormap will be applied to the full range of the overlay values.
        If both are provided, they will be used to scale the colormap.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the specified overlay is not found in the mesh point data
        ValueError
            If no overlays are available

        Notes
        -----
        This method sets the vertices colors in the mesh based on the specified overlay.
        The colors are stored in the mesh's point_data under the key "RGB"
        and set as the active scalars for visualization.

        Examples
        --------
        >>> # Prepare colors for a parcellation (uses discrete colors)
        >>> surface.prepare_colors(overlay_name="aparc")
        >>>
        >>> # Prepare colors for scalar data with custom colormap
        >>> surface.prepare_colors(overlay_name="thickness", cmap="hot")
        >>>
        >>> # Prepare colors for first available overlay
        >>> surface.prepare_colors()
        """
        # Get the list of overlays
        overlay_dict = self.list_overlays()

        # If the dictionary is empty
        overlays = list(overlay_dict.keys())
        if overlay_name is None:
            overlay_name = overlays[0] if overlay_dict else None

        if overlay_name not in overlays:
            raise ValueError(
                f"Overlay '{overlay_name}' not found. Available overlays: {', '.join(overlays)}"
            )

        else:
            # Set the active overlay
            self.set_active_overlay(overlay_name)

        if overlay_dict[overlay_name] == "color":
            self.mesh.point_data["RGB"] = self.mesh.point_data[overlay_name]
            self.mesh.set_active_scalars("RGB")
            return

        else:
            # If no colormap is provided, use the default colormap for the overlay
            vertex_values = self.mesh.point_data[overlay_name]
            dict_ctables = self.colortables
            # Check if the overlay is a color or scalar type

            if overlay_name in dict_ctables.keys():
                # Use the colortable associated with the parcellation
                vertex_colors = cltfree.create_vertex_colors(
                    vertex_values, self.colortables[overlay_name]["color_table"]
                )

            else:
                vertex_colors = cltmisc.values2colors(
                    vertex_values,
                    cmap=cmap,
                    output_format="rgb",
                    vmin=vmin,
                    vmax=vmax,
                )

            self.mesh.point_data["RGB"] = vertex_colors
            self.mesh.set_active_scalars("RGB")

    ##############################################################################################
    def merge_surfaces(self, surfaces: Union["Surface", List["Surface"]]) -> "Surface":
        """
        Merge this surface with others into a single surface.

        This method merges multiple Surface objects by combining their geometries
        and point data. Only point_data fields that are present in ALL surfaces
        are retained in the merged result.

        Parameters
        ----------
        surfaces : List[Surface]
            List of Surface objects to merge with this surface

        Returns
        -------
        Surface
            New merged Surface object with hemisphere set to "unknown"

        Raises
        ------
        TypeError
            If surfaces is not a list or contains non-Surface objects
        ValueError
            If the surfaces list is empty

        Examples
        --------
        >>> # Merge left and right hemisphere surfaces
        >>> lh_surf = Surface("lh.pial")
        >>> rh_surf = Surface("rh.pial")
        >>> merged = lh_surf.merge_surfaces([rh_surf])
        >>> print(f"Merged surface has {merged.mesh.n_points} vertices")
        >>>
        >>> # Merge multiple surfaces
        >>> surf1 = Surface("surface1.pial")
        >>> surf2 = Surface("surface2.pial")
        >>> surf3 = Surface("surface3.pial")
        >>> merged = surf1.merge_surfaces([surf2, surf3])
        """

        if not isinstance(surfaces, list):
            surfaces = [surfaces]

        if len(surfaces) == 0:
            raise ValueError("surfaces list cannot be empty")

        # Check that all items in the list are Surface objects
        for i, surf in enumerate(surfaces):
            if not isinstance(surf, Surface):
                raise TypeError(f"Item at index {i} is not a Surface object")

        # Include this surface in the list
        all_surfaces = [self] + surfaces

        # Find common point_data fields across all surfaces
        common_fields = None
        for surf in all_surfaces:
            current_fields = set(surf.mesh.point_data.keys())
            if common_fields is None:
                common_fields = current_fields
            else:
                common_fields = common_fields.intersection(current_fields)

        # Convert to list for consistent ordering
        common_fields = list(common_fields)

        # Prepare meshes with only common fields
        meshes_to_merge = []
        for surf in all_surfaces:
            # Create a copy of the mesh
            mesh_copy = copy.deepcopy(surf.mesh)

            # Remove point_data fields that are not common
            fields_to_remove = set(mesh_copy.point_data.keys()) - set(common_fields)
            for field in fields_to_remove:
                del mesh_copy.point_data[field]

            meshes_to_merge.append(mesh_copy)

        # Merge all meshes using PyVista
        if len(meshes_to_merge) == 1:
            merged_mesh = meshes_to_merge[0]
        else:
            merged_mesh = pv.merge(meshes_to_merge)

        # Create new Surface object without calling __init__
        merged_surface = Surface.__new__(Surface)
        merged_surface.mesh = merged_mesh
        merged_surface.hemi = "unknown"
        merged_surface.surf = "merged_surface"

        # Merge colortables - only keep those for common fields
        merged_colortables = {}
        for surf in all_surfaces:
            for key, value in surf.colortables.items():
                if key in common_fields:
                    # If key already exists, keep the first one encountered
                    if key not in merged_colortables:
                        # Deep copy the colortable data to avoid reference issues
                        if isinstance(value, dict):
                            merged_colortables[key] = {}
                            for k, v in value.items():
                                if isinstance(v, (list, np.ndarray)):
                                    merged_colortables[key][k] = v.copy()
                                else:
                                    merged_colortables[key][k] = v
                        else:
                            merged_colortables[key] = value

        merged_surface.colortables = merged_colortables

        return merged_surface

    ##############################################################################################
    def save_surface(
        self,
        filename: str,
        format: str = "freesurfer",
        save_annotation: str = None,
        map_name: str = None,
        overwrite: bool = False,
    ) -> None:
        """
        Save the surface mesh to a file in the specified format.

        Exports the surface geometry (vertices and faces) and optionally associated
        data to various file formats including FreeSurfer, VTK, PLY, STL, and OBJ.

        Parameters
        ----------
        filename : str
            Output filename with or without extension. Extension will be added
            automatically if missing for some formats.

        format : str, default "freesurfer"
            Output format: 'freesurfer', 'vtk', 'ply', 'stl', or 'obj'.

        save_annotation : str, optional
            Path to save annotation file (for parcellation data). Only applicable
            for FreeSurfer.

        map_name : str, optional
            Name of overlay/parcellation to include with the surface data.

        overwrite : bool, default False
            Whether to overwrite existing files.

        Raises
        ------
        ValueError
            If filename is invalid, format is unsupported, or file exists and
            overwrite is False.
        FileNotFoundError
            If the output directory does not exist.

        Examples
        --------
        >>> surface.save_surface("lh.pial.vtk", format="vtk")
        >>> surface.save_surface("cortex.ply", format="ply", overwrite=True)
        """

        if not isinstance(filename, str):
            raise ValueError("filename must be a string")
        if not filename:
            raise ValueError("filename cannot be empty")

        # Check if the filename exists as a valid path
        if os.path.exists(filename) and not overwrite:
            raise ValueError(
                f"File '{filename}' already exists. Please set overwrite to True or choose a different name."
            )

        # Ensure the directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            raise FileNotFoundError(f"Directory '{directory}' does not exist")

        # Save the mesh using PyVista's built-in methods
        if format.lower() == "freesurfer":
            self.export_to_freesurfer(filename, save_annotation, map_name, overwrite)

        elif format.lower() == "obj":
            self.export_to_obj(filename, save_annotation, map_name, overwrite)

        elif format.lower() in ["vtk", "ply", "stl"]:

            # Substitute the file extension to match the format
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in [".vtk", ".ply", ".stl"]:
                if format.lower() == "vtk":
                    filename += ".vtk"
                elif format.lower() == "ply":
                    filename += ".ply"
                elif format.lower() == "stl":
                    filename += ".stl"
                else:
                    raise ValueError(
                        f"Unsupported file format: {format}. Supported formats are 'vtk', 'ply', 'stl'."
                    )

            self.export_to_pyvista(filename, save_annotation, map_name, overwrite)
            # Print a message indicating the file was saved
            print(f"Surface saved to {filename}")

        else:
            raise ValueError(
                f"Unsupported file format: {format}. Supported formats are 'vtk', 'ply', 'stl', 'obj', and 'freesurfer'."
            )

    ##############################################################################################
    def export_to_obj(
        self,
        filename: str,
        save_annotation: str = None,
        map_name: str = None,
        overwrite: bool = False,
    ) -> None:
        """
        Export the surface mesh to an OBJ file format.

        Writes surface geometry as a Wavefront OBJ file, which stores vertices
        and triangular faces in a simple text format widely supported by 3D
        software and visualization tools.

        Parameters
        ----------
        filename : str
            Output filename, should end with .obj extension.

        save_annotation : str, optional
            Path to save associated annotation file in FreeSurfer format.

        map_name : str, optional
            Name of parcellation/overlay to export alongside the geometry.

        overwrite : bool, default False
            Whether to overwrite existing files.

        Raises
        ------
        ValueError
            If filename is invalid or file exists and overwrite is False.
        FileNotFoundError
            If the output directory does not exist.

        Notes
        -----
        OBJ format uses 1-based indexing for face connectivity. The exported
        file includes vertex coordinates and triangular face definitions.

        Examples
        --------
        >>> surface.export_to_obj("brain_surface.obj")
        >>> surface.export_to_obj("lh.pial.obj", save_annotation="lh.aparc.annot", map_name="aparc")
        """

        # Validate filename
        if not isinstance(filename, str):
            raise ValueError("filename must be a string")
        if not filename:
            raise ValueError("filename cannot be empty")

        # Check if the filename exists as a valid path
        if os.path.exists(filename) and not overwrite:
            raise ValueError(
                f"File '{filename}' already exists. Please set overwrite to True or choose a different name."
            )

        # Ensure the directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            raise FileNotFoundError(f"Directory '{directory}' does not exist")

        # If save_annotation is provided, save the annotation data
        if save_annotation is not None:
            self.export_annotation(
                filename=save_annotation, parc_name=map_name, overwrite=overwrite
            )

        vertices = self.mesh.points
        faces = self.mesh.regular_faces

        with open(filename, "w") as f:
            f.write(f"# OBJ file exported from Surface class\n")
            f.write(f"# Vertices: {len(vertices)}\n")
            f.write(f"# Faces: {len(faces)}\n\n")

            # Write vertices
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

            # Write faces (OBJ uses 1-based indexing)
            f.write("\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        print(f"Surface exported to {filename}")

    def export_to_pyvista(
        self,
        filename: str,
        save_annotation: str = None,
        map_name: str = None,
        overwrite: bool = False,
    ) -> None:
        """
        Export surface to VTK, STL, or PLY format using PyVista.

        Saves the surface mesh in formats supported by PyVista, preserving
        geometry and optionally scalar data or colors. The file format is
        determined by the filename extension.

        Parameters
        ----------
        filename : str
            Output filename with extension (.vtk, .ply, or .stl).

        save_annotation : str, optional
            Path to save annotation file in FreeSurfer format.

        map_name : str, optional
            Name of overlay to include as scalar data or vertex colors.

        overwrite : bool, default False
            Whether to overwrite existing files.

        Raises
        ------
        ValueError
            If filename is invalid, map_name not found, or file exists and
            overwrite is False.
        FileNotFoundError
            If the output directory does not exist.

        Notes
        -----
        VTK format can store additional scalar data and colors. PLY and STL
        formats primarily store geometry. When map_name is specified, the
        overlay data is prepared as vertex colors using associated colortables.

        Examples
        --------
        >>> surface.export_to_pyvista("brain.vtk")
        >>> surface.export_to_pyvista("surface.ply", map_name="thickness")
        """

        # Validate filename
        if not isinstance(filename, str):
            raise ValueError("filename must be a string")
        if not filename:
            raise ValueError("filename cannot be empty")

        # Check if the filename exists as a valid path
        if os.path.exists(filename) and not overwrite:
            raise ValueError(
                f"File '{filename}' already exists. Please set overwrite to True or choose a different name."
            )

        # Ensure the directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            raise FileNotFoundError(f"Directory '{directory}' does not exist")

        # If save_annotation is provided, save the annotation data
        if save_annotation is not None:
            self.export_annotation(
                filename=save_annotation, parc_name=map_name, overwrite=overwrite
            )

        if map_name is not None:
            # Ensure the map_name is valid
            if not isinstance(map_name, str):
                raise ValueError("map_name must be a string")
            if not map_name:
                raise ValueError("map_name cannot be empty")

            # Ensure the map_name exists in the mesh point data
            if map_name not in self.mesh.point_data:
                raise ValueError(f"Map '{map_name}' not found in mesh point data")

            # Set the active scalars to the specified map_name
            self.mesh.set_active_scalars(map_name)

            # Prepare colors
            self.prepare_colors(
                overlay_name=map_name,
                cmap=None,  # Use the colortable for this map
                vmin=None,
                vmax=None,
            )

            # Save the mesh (no texture parameter needed for vertices colors)
            self.mesh.save(filename)

        else:
            # Use PyVista's built-in save method for VTK format
            self.mesh.save(filename)

    ##############################################################################################
    def export_to_freesurfer(
        self,
        filename: str,
        save_annotation: str = None,
        map_name: str = None,
        overwrite: bool = False,
    ) -> None:
        """
        Export surface to FreeSurfer binary format.

        Saves the surface mesh in FreeSurfer's native binary geometry format,
        which efficiently stores vertex coordinates and triangular face
        connectivity for neuroimaging applications.

        Parameters
        ----------
        filename : str
            Output filename, typically without extension (e.g., 'lh.pial').
        save_annotation : str, optional
            Path to save annotation file containing parcellation data.
        map_name : str, optional
            Name of parcellation to export with the annotation file.
        overwrite : bool, default False
            Whether to overwrite existing files.

        Raises
        ------
        ValueError
            If filename is invalid or file exists and overwrite is False.
        FileNotFoundError
            If the output directory does not exist.

        Notes
        -----
        FreeSurfer format is a compact binary representation optimized for
        neuroimaging workflows. The format stores only geometry data;
        additional data like parcellations are saved separately as .annot files.

        Examples
        --------
        >>> surface.export_to_freesurfer("lh.pial")
        >>> surface.export_to_freesurfer("rh.white", save_annotation="rh.aparc.annot", map_name="aparc")
        """

        if not isinstance(filename, str):
            raise ValueError("filename must be a string")
        if not filename:
            raise ValueError("filename cannot be empty")

        # Check if the filename exist as a valid path
        if os.path.exists(filename) and not overwrite:
            raise ValueError(
                f"File '{filename}' already exists. Please set overwrite to True or choose a different name."
            )

        # Ensure the directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            raise FileNotFoundError(f"Directory '{directory}' does not exist")

        # If save_annotation is provided, save the annotation data
        if save_annotation is not None:
            self.export_annotation(
                filename=save_annotation, parc_name=map_name, overwrite=overwrite
            )

        # Separating vertices and faces
        vertices = self.mesh.points
        faces = self.mesh.regular_faces

        # Use nibabel to write FreeSurfer geometry
        nib.freesurfer.write_geometry(filename, vertices, faces)

    ##############################################################################################
    def export_annotation(
        self,
        filename: str,
        parc_name: str,
        overwrite: bool = False,
    ) -> None:
        """
        Export parcellation data to a FreeSurfer annotation file.

        Saves vertex-wise parcellation labels, associated color lookup table,
        and region names in FreeSurfer's .annot format for use with FreeSurfer
        tools and visualization software.

        Parameters
        ----------
        filename : str
            Output filename for annotation file (should end with .annot).

        parc_name : str
            Name of parcellation overlay to export from the surface data.

        overwrite : bool, default False
            Whether to overwrite existing files.

        Raises
        ------
        ValueError
            If filename or parc_name is invalid, parcellation not found, or
            file exists and overwrite is False.
        FileNotFoundError
            If the output directory does not exist.

        Notes
        -----
        Requires the parcellation to have an associated colortable with region
        names and colors. The annotation format preserves the mapping between
        vertex labels, region names, and visualization colors.

        Examples
        --------
        >>> surface.export_annotation("lh.aparc.annot", "aparc")
        >>> surface.export_annotation("rh.destrieux.annot", "destrieux", overwrite=True)
        """

        # If save_annotation is provided, save the annotation data
        if not isinstance(filename, str):
            raise ValueError("The annotation filename must be a string")

        if not filename:
            raise ValueError("The annotation filename cannot be empty")

        # Check if the annotation file exists
        if os.path.isfile(filename) and not overwrite:
            raise ValueError(
                f"Annotation file '{filename}' already exists. Please set overwrite to True or choose a different name."
            )

        # Check if the directory exists
        annot_directory = os.path.dirname(filename)
        if annot_directory and not os.path.exists(annot_directory):
            raise FileNotFoundError(f"Directory '{annot_directory}' does not exist")

        if not isinstance(parc_name, str):
            raise ValueError("parc_name must be a string")
        if not parc_name:
            raise ValueError("parc_name cannot be empty")

        # Ensure the parc_name exists in the mesh point data
        if parc_name not in self.mesh.point_data:
            raise ValueError(f"Map '{parc_name}' not found in mesh point data")

        # Extract the annotation data
        maps_array = self.mesh.point_data[parc_name]

        # If there is a colortable for this map, use it
        if parc_name in self.colortables:
            ctable = self.colortables[parc_name]["color_table"]
            struct_names = self.colortables[parc_name]["struct_names"]

            # Saving the annotation data in FreeSurfer format
            annot_obj = cltfree.AnnotParcellation()
            annot_obj.create_from_data(
                maps_array, ctable, struct_names, annot_id=parc_name
            )
            annot_obj.save_annotation(filename, force=overwrite)
        else:
            print(
                f"Warning: No colortable found for map '{parc_name}'. Annotation file will not be saved."
            )

    ###############################################################################################
    def plot(
        self,
        overlay_name: str = "surface",
        cmap: str = "viridis",
        vmin: np.float64 = None,
        vmax: np.float64 = None,
        views: Union[str, List[str]] = ["lateral"],
        hemi: str = "lh",
        notebook: bool = False,
        show_colorbar: bool = False,
        colorbar_title: str = None,
        colorbar_position: str = "bottom",
        save_path: str = None,
    ):
        """
        Plot the surface with specified overlay and visualization parameters.

        Renders the surface mesh with optional overlays using PyVista, supporting
        multiple camera views, custom colormaps, and interactive or static output.
        Handles both categorical parcellation data and continuous scalar overlays.

        Parameters
        ----------
        overlay_name : str, default "surface"
            Name of the overlay to visualize from the surface's point data.

        cmap : str, optional
            Colormap for scalar data. If None, uses parcellation colors for
            categorical data or 'viridis' for scalar data.

        vmin : float, optional
            Minimum value for colormap scaling. If None, uses data minimum.

        vmax : float, optional
            Maximum value for colormap scaling. If None, uses data maximum.

        views : str or List[str], default ["lateral"]
            Camera view(s): 'lateral', 'medial', 'dorsal', 'ventral', 'anterior',
            'posterior', or multiple views like ['lateral', 'medial']. Also supports
            preset layouts: '4_views', '6_views', '8_views' with optional orientation.

        hemi : str, default "lh"
            Hemisphere to visualize: 'lh' (left) or 'rh' (right).

        notebook : bool, default False
            Whether to display in Jupyter notebook. If False, opens interactive window.

        show_colorbar : bool, default False
            Whether to display colorbar. Automatically determined if None.

        colorbar_title : str, optional
            Title for the colorbar. Uses overlay name if None.

        colorbar_position : str, default "bottom"
            Colorbar position: 'bottom', 'top', 'left', or 'right'.

        save_path : str, optional
            Path to save plot as image. If None, displays interactively.

        Returns
        -------
        Plotter
            PyVista plotter object for further customization.

        Raises
        ------
        ValueError
            If overlay not found or invalid view parameter.

        Examples
        --------
        >>> surface.plot(overlay_name="aparc")
        >>> surface.plot(overlay_name="thickness", cmap="hot", views="medial", show_colorbar=True)
        """

        # self.prepare_colors(overlay_name=overlay_name, cmap=cmap, vmin=vmin, vmax=vmax)

        dict_ctables = self.colortables
        if cmap is None:
            if overlay_name in dict_ctables.keys():
                show_colorbar = False

            else:
                show_colorbar = True

        else:
            show_colorbar = True

        from . import visualizationtools as cltvis

        plotter = cltvis.SurfacePlotter()

        plotter.plot_surfaces(
            self,
            hemi_id=hemi,
            views=views,
            map_names=overlay_name,
            colormaps=cmap,
            v_limits=(vmin, vmax),
            notebook=notebook,
            colorbar=show_colorbar,
            colorbar_titles=colorbar_title,
            colorbar_position=colorbar_position,
            save_path=save_path,
        )


def merge_surfaces_list(surface_list):
    """
    Merge a list of Surface objects into a single Surface object.

    Combines multiple Surface objects by merging their geometries and point
    data into a unified surface representation. Preserves all overlays and
    associated data from the input surfaces.

    Parameters
    ----------
    surface_list : List[Surface]
        List of Surface objects to merge. Must contain at least one surface.

    Returns
    -------
    Surface or None
        Merged Surface object containing combined geometries and data.
        Returns None if merging fails or list is empty.

    Raises
    ------
    TypeError
        If surface_list is not a list or contains non-Surface objects
    ValueError
        If the surface_list is empty

    Notes
    -----
    The merging process combines vertex coordinates, face connectivity, and
    point data from all input surfaces. The first surface serves as the base,
    with subsequent surfaces appended to create a unified mesh.

    Examples
    --------
    >>> surfaces = [surf1, surf2, surf3]
    >>> merged = merge_surfaces_list(surfaces)
    >>> print(f"Merged surface has {merged.mesh.n_points} vertices")
    """

    if not isinstance(surface_list, list):
        raise TypeError("surface_list must be a list")

    if any(not isinstance(surf, Surface) for surf in surface_list):
        raise TypeError("All items in surface_list must be Surface objects")

    # If the list is empty, return None
    if not surface_list:
        return None

    # If there's only one surface, return it as is
    if len(surface_list) == 1:
        return copy.deepcopy(surface_list[0])

    # Start with the first surface as the base for merging
    merged = copy.deepcopy(surface_list[0])

    # Iterate through the rest of the surfaces and merge them
    for surf in surface_list[1:]:
        try:
            # Use the merge_surfaces method of the Surface class
            # This will handle the merging logic and return a new Surface object
            # If the merge_surfaces method modifies the merged object in place,
            # we can just continue using the merged object

            # Most common: merge_surfaces returns a new object
            result = merged.merge_surfaces([surf])

            # If result is not None, update merged
            if result is not None:
                merged = result

            # If result is None, assume it modified merged in place
        except Exception as e:
            print(f"Merge failed: {e}")
            return None

    return merged
