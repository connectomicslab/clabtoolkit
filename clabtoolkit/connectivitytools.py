import warnings
from pathlib import Path
from typing import Literal

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.sparse import csr_matrix, issparse

from . import colorstools as cltcol
from . import misctools as cltmisc
from . import plottools as cltplot


class Connectome:
    """
    A class to represent and visualize brain connectivity data.

    Attributes:
    -----------
    name : str
        Name identifier for the connectome
    matrix : np.ndarray
        Connectivity matrix (n_regions x n_regions)
    region_coords : np.ndarray
        3D region_coords for each region (n_regions x 3)
    region_colors : np.ndarray
        RGB color values for each region (n_regions x 3)
    region_names : List[str]
        Names/labels for each brain region
    region_index : list[int]
        Index codes for each region. Always stored as a plain list of ints,
        regardless of whether a list, tuple, or numpy array was supplied.
    connectivity_type : str
        Type of connectivity ('unknown', 'structural', 'functional', 'effective', etc.)
    affine : np.ndarray
        4x4 affine transformation matrix
    n_regions : int
        Number of brain regions
    """

    #################################################################################
    def __init__(
        self,
        matrix: np.ndarray | csr_matrix | str | Path | None = None,
        name: str | None = None,
        region_coords: np.ndarray | None = None,
        region_names: list[str] | None = None,
        region_index: np.ndarray | list | tuple | None = None,
        region_colors: np.ndarray | list | None = None,
        connectivity_type: str = "unknown",
        affine: np.ndarray | None = None,
    ):
        """
        Initialize a Connectome object.

        Parameters:
        -----------
        matrix : np.ndarray, str, Path, or None
            Can be:
            - np.ndarray: Connectivity matrix (n_regions x n_regions)
            - str or Path: Path to HDF5 file to load
            - None: Create empty Connectome
        name : str, optional
            Name for the connectome. If loading from file and None, uses filename stem.
        region_coords : np.ndarray, optional
            3D region_coords for each region (n_regions x 3)
        region_names : List[str], optional
            Names/labels for each brain region
        region_index : list[int] or np.ndarray, optional
            Index codes for each region. Coerced to list[int] internally.
        region_colors : np.ndarray or List, optional
            RGB color values or hex strings for each region
        connectivity_type : str, optional
            Type of connectivity (default: 'unknown')
        affine : np.ndarray, optional
            4x4 affine transformation matrix

        Examples:
        ---------
        >>> # From matrix
        >>> matrix = np.random.rand(10, 10)
        >>> conn = Connectome(matrix)

        >>> # From file
        >>> conn = Connectome('/path/to/connectome.h5')

        >>> # Empty connectome
        >>> conn = Connectome(name='my_network')
        """
        self.type = connectivity_type

        # Handle different input types for data
        load_from_file = False
        filepath = None

        if matrix is not None:
            if isinstance(matrix, (str, Path)):
                # Load from file
                load_from_file = True
                filepath = Path(matrix)

                # Set default name from filename if not provided
                if name is None:
                    name = filepath.stem

                matrix = None

            elif isinstance(matrix, np.ndarray):
                # Use as connectivity matrix - nothing to transform here
                pass

            elif issparse(matrix):
                # Sparse input (e.g. CSR from networktools) - densify
                matrix = matrix.toarray()

            else:
                raise TypeError(
                    "The input matrix must be np.ndarray, scipy sparse matrix, "
                    f"str, Path, or None. Got {type(matrix)}"
                )

        self.name = name

        # Initialize matrix and derived attributes
        if matrix is not None:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Matrix must be square (NxN)")
            self.matrix = matrix.astype(np.float64)
            self.n_regions = matrix.shape[0]
        else:
            self.matrix = None
            self.n_regions = 0

        # Set region_coords
        if region_coords is not None:
            self.set_region_coordinates(region_coords)
        else:
            self.region_coords = None

        # Set colors
        if region_colors is not None:
            region_colors = cltcol.harmonize_colors(region_colors, "hex")
            self.set_region_colors(region_colors)
        elif self.n_regions > 0:
            # Only generate default colors if we have regions
            self.region_colors = cltcol.create_distinguishable_colors(self.n_regions)
        else:
            self.region_colors = None

        # Set region names
        if region_names is not None:
            self.set_region_names(region_names)
        elif self.n_regions > 0:
            # Only generate default names if we have regions
            self.region_names = cltmisc.create_names_from_indices(
                np.arange(self.n_regions) + 1
            )
        else:
            self.region_names = None

        # Set region index (always normalized to list[int])
        if region_index is not None:
            self.region_index = self._normalize_region_index(
                region_index, self.n_regions if self.matrix is not None else None
            )
        else:
            self.region_index = (
                list(range(self.n_regions)) if self.matrix is not None else None
            )

        # Set affine
        if affine is not None:
            if affine.shape != (4, 4):
                raise ValueError(f"Affine must be 4x4 array, got {affine.shape}")
            self.affine = affine.astype(np.float64)
        else:
            self.affine = np.eye(4)

        # Load from file if specified
        if load_from_file:
            self.load_h5(filepath)

    #################################################################################
    @staticmethod
    def _normalize_region_index(
        indices: list | np.ndarray | tuple | None,
        n_regions: int | None = None,
    ) -> list[int] | None:
        """
        Coerce any array-like of region indices into a plain list[int].

        Parameters:
        -----------
        indices : list, np.ndarray, tuple, or None
            Region indices in any array-like form.
        n_regions : int, optional
            If given, validates that len(indices) matches n_regions.

        Returns:
        --------
        list[int] or None
        """
        if indices is None:
            return None

        arr = np.asarray(indices)

        if n_regions is not None and len(arr) != n_regions:
            raise ValueError(
                f"Region index length ({len(arr)}) must match matrix size ({n_regions})"
            )

        return [int(i) for i in arr.tolist()]

    #################################################################################
    @classmethod
    def from_h5(cls, filename: str | Path, name: str | None = None) -> "Connectome":
        """
        Create a Connectome object from an HDF5 file.

        Parameters:
        -----------
        filename : str or Path
            Path to the HDF5 file containing connectivity data
        name : str, optional
            Name for the connectome. If None, uses filename stem.

        Returns:
        --------
        Connectome : New Connectome object with loaded data
        """
        filename = Path(filename)

        # Set default name from filename if not provided
        if name is None:
            name = filename.stem

        connectome = cls(name=name)
        connectome.load_h5(filename)
        return connectome

    #################################################################################
    @classmethod
    def from_csr(
        cls,
        csr_graph: csr_matrix,
        name: str | None = None,
        **kwargs,
    ) -> "Connectome":
        """
        Create a Connectome object from a scipy sparse matrix.

        The matrix is densified, since Connectome stores region-level
        connectivity as a dense array.

        Parameters:
        -----------
        csr_graph : scipy.sparse matrix
            Square sparse connectivity matrix (CSR preferred; any sparse
            format is accepted and converted).
        name : str, optional
            Name for the connectome.
        **kwargs
            Additional keyword arguments passed to Connectome.__init__
            (region_coords, region_names, region_index, region_colors,
            connectivity_type, affine).

        Returns:
        --------
        Connectome : New Connectome object

        Examples:
        ---------
        >>> from scipy.sparse import random as sprandom
        >>> A = sprandom(50, 50, density=0.05, format="csr")
        >>> conn = Connectome.from_csr(A, name="sparse_example")
        """
        if not issparse(csr_graph):
            raise TypeError(
                f"Input must be a scipy sparse matrix. Got {type(csr_graph)}"
            )
        if csr_graph.shape[0] != csr_graph.shape[1]:
            raise ValueError("Sparse matrix must be square")

        return cls(matrix=csr_graph.toarray(), name=name, **kwargs)

    #################################################################################
    @classmethod
    def from_csv(
        cls,
        filename: str | Path,
        name: str | None = None,
        region_coords: np.ndarray | None = None,
        region_names: list[str] | None = None,
        region_index: np.ndarray | list | tuple | None = None,
        region_colors: np.ndarray | list | None = None,
        connectivity_type: str = "unknown",
        affine: np.ndarray | None = None,
    ) -> "Connectome":
        """
        Create a Connectome object from a CSV file.

        Parameters:
        -----------
        filename : str or Path
            Path to the CSV file containing connectivity data
        name : str, optional
            Name for the connectome. If None, uses filename stem.
        region_coords : np.ndarray, optional
            3D region_coords for each region (n_regions x 3)
        region_names : List[str], optional
            Names/labels for each brain region
        region_index : list[int] or np.ndarray, optional
            Index codes for each region. Coerced to list[int] internally.
        region_colors : np.ndarray or List, optional
            RGB color values or hex strings for each region
        connectivity_type : str, optional
            Type of connectivity (default: 'unknown')
        affine : np.ndarray, optional
            4x4 affine transformation matrix

        Returns:
        --------
        Connectome : New Connectome object with loaded data
        """
        filename = Path(filename)

        # Set default name from filename if not provided
        if name is None:
            name = filename.stem

        connectome = cls()
        connectome.load_csv(
            filename,
            name,
            region_coords,
            region_names,
            region_index,
            region_colors,
            connectivity_type,
            affine,
        )
        return connectome

    #################################################################################
    def _calculate_node_sizes(
        self, property_type: str, threshold: float, scale: float, base_size: float
    ) -> np.ndarray:
        """
        Calculate node sizes based on different properties.

        Parameters:
        -----------
        property_type : str
            Type of property to use for sizing
        threshold : float
            Threshold for degree calculation
        scale : float
            Scale factor
        base_size : float
            Base size for nodes

        Returns:
        --------
        np.ndarray : Array of node sizes
        """
        if property_type == "uniform":
            return np.full(self.n_regions, base_size)

        elif property_type == "strength":
            # Total connectivity strength (sum of absolute connections)
            strengths = np.sum(np.abs(self.matrix), axis=1)
            normalized = (
                strengths / np.max(strengths) if np.max(strengths) > 0 else strengths
            )
            return normalized * 10 * scale + base_size

        elif property_type == "degree":
            # Number of connections above threshold
            degrees = np.sum(np.abs(self.matrix) > threshold, axis=1)
            normalized = degrees / np.max(degrees) if np.max(degrees) > 0 else degrees
            return normalized * 10 * scale + base_size

        elif property_type == "betweenness":
            try:
                import networkx as nx

                # Create graph from adjacency matrix
                G = nx.from_numpy_array(np.abs(self.matrix))
                centrality = nx.betweenness_centrality(G)
                values = np.array([centrality[i] for i in range(self.n_regions)])
                normalized = values / np.max(values) if np.max(values) > 0 else values
                return normalized * 10 * scale + base_size
            except ImportError:
                warnings.warn(
                    "NetworkX not available. Using strength instead of betweenness centrality.",
                    stacklevel=2,
                )
                return self._calculate_node_sizes(
                    "strength", threshold, scale, base_size
                )

        elif property_type == "eigenvector":
            try:
                import networkx as nx

                # Create graph from adjacency matrix
                G = nx.from_numpy_array(np.abs(self.matrix))
                try:
                    centrality = nx.eigenvector_centrality(G, max_iter=1000)
                    values = np.array([centrality[i] for i in range(self.n_regions)])
                    normalized = (
                        values / np.max(values) if np.max(values) > 0 else values
                    )
                    return normalized * 10 * scale + base_size
                except nx.PowerIterationFailedConvergence:
                    warnings.warn(
                        "Eigenvector centrality failed to converge. Using strength instead.",
                        stacklevel=2,
                    )
                    return self._calculate_node_sizes(
                        "strength", threshold, scale, base_size
                    )
            except ImportError:
                warnings.warn(
                    "NetworkX not available. Using strength instead of eigenvector centrality.",
                    stacklevel=2,
                )
                return self._calculate_node_sizes(
                    "strength", threshold, scale, base_size
                )

        else:
            raise ValueError(
                f"Unknown node size property: {property_type}. "
                f"Available options: 'uniform', 'strength', 'degree', 'betweenness', 'eigenvector'"
            )

    #################################################################################
    def load_csv(
        self,
        filename: str | Path,
        name: str | None = None,
        region_coords: np.ndarray | None = None,
        region_names: list[str] | None = None,
        region_index: np.ndarray | list | tuple | None = None,
        region_colors: np.ndarray | list | None = None,
        connectivity_type: str = "unknown",
        affine: np.ndarray | None = None,
    ) -> None:
        """
        Load connectivity data from a CSV file.

        Parameters:
        -----------
        filename : str or Path
            Path to the CSV file containing connectivity data
        name : str, optional
            Name for the connectome.
        region_coords : np.ndarray, optional
            3D region_coords for each region (n_regions x 3)
        region_names : List[str], optional
            Names/labels for each brain region
        region_index : list[int] or np.ndarray, optional
            Index codes for each region. Coerced to list[int] internally.
        region_colors : np.ndarray or List, optional
            RGB color values or hex strings for each region
        connectivity_type : str, optional
            Type of connectivity (default: 'unknown')
        affine : np.ndarray, optional
            4x4 affine transformation matrix
        """
        filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        matrix = np.loadtxt(filename, delimiter=",")

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"CSV must contain a square connectivity matrix, got shape {matrix.shape}"
            )

        self.matrix = matrix
        self.n_regions = self.matrix.shape[0]

        # Set region_coords
        if region_coords is not None:
            self.set_region_coordinates(region_coords)
        else:
            self.region_coords = None

        # Set colors
        if region_colors is not None:
            region_colors = cltcol.harmonize_colors(region_colors, "hex")
            self.set_region_colors(region_colors)
        elif self.n_regions > 0:
            # Only generate default colors if we have regions
            self.region_colors = cltcol.create_distinguishable_colors(self.n_regions)
        else:
            self.region_colors = None

        # Set region names
        if region_names is not None:
            self.set_region_names(region_names)
        elif self.n_regions > 0:
            # Only generate default names if we have regions
            self.region_names = cltmisc.create_names_from_indices(
                np.arange(self.n_regions) + 1
            )
        else:
            self.region_names = None

        # Set region index (always normalized to list[int])
        if region_index is not None:
            self.region_index = self._normalize_region_index(
                region_index, self.n_regions
            )
        else:
            self.region_index = list(range(self.n_regions))

        # Set affine
        if affine is not None:
            if affine.shape != (4, 4):
                raise ValueError(f"Affine must be 4x4 array, got {affine.shape}")
            self.affine = affine.astype(np.float64)
        else:
            self.affine = np.eye(4)

        if name is not None:
            self.name = name
        else:
            self.name = filename.stem

        self.type = connectivity_type if connectivity_type is not None else "unknown"

    #################################################################################
    def load_h5(self, filename: str | Path) -> None:
        """
        Load connectivity data from HDF5 file.

        Parameters:
        -----------
        filename : str or Path
            Path to the HDF5 file
        """
        filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        try:
            with h5py.File(filename, "r") as f:
                # Try to find data in 'connmat' group first, then root
                if "connmat" in f:
                    data_group = f["connmat"]
                else:
                    data_group = f

                # Load connectivity matrix (required).
                # Supports a dense dataset ("matrix") or a CSR group
                # ("matrix_csr" with data / indices / indptr and a shape attr).
                if "matrix" in data_group:
                    self.matrix = data_group["matrix"][:].astype(np.float64)
                elif "matrix_csr" in data_group:
                    self.matrix = self._read_csr_group(
                        data_group["matrix_csr"]
                    ).toarray()
                else:
                    raise KeyError(
                        "No 'matrix' dataset or 'matrix_csr' group found in HDF5 file"
                    )

                self.n_regions = self.matrix.shape[0]

                # Load coordinates (required for visualization)
                for key in ("coords", "gmcoords", "region_coords"):
                    if key in data_group:
                        self.region_coords = data_group[key][:]
                        if self.region_coords.shape[0] != self.n_regions:
                            raise ValueError(
                                "Number of coordinates doesn't match matrix size"
                            )
                        break
                else:
                    warnings.warn(
                        "No coordinates found. 3D visualization will not be available.",
                        stacklevel=2,
                    )

                for key in ("gmcolors", "region_colors", "colors"):
                    if key in data_group:
                        colors_data = data_group[key][:]
                        if key in ("gmcolors", "region_colors"):
                            # Hex / string formats written by save_h5 or legacy gmcolors
                            if colors_data.dtype.kind in ["S", "O"]:
                                colors_list = [
                                    (
                                        c.decode("utf-8")
                                        if isinstance(c, bytes)
                                        else str(c)
                                    )
                                    for c in colors_data
                                ]
                            else:
                                colors_list = colors_data.tolist()
                            self.region_colors = cltcol.harmonize_colors(colors_list)
                        else:  # "colors" — legacy numeric RGB array
                            self.region_colors = colors_data
                            if self.region_colors.shape[0] != self.n_regions:
                                warnings.warn(
                                    "Number of colors doesn't match matrix size",
                                    stacklevel=2,
                                )
                            elif np.max(self.region_colors) > 1:
                                self.region_colors = self.region_colors / 255.0
                        break

                # Load region names (optional)
                for key in ("gmregions", "name", "region_names"):
                    if key in data_group:
                        names_data = data_group[key][:]
                        if names_data.dtype.kind in ["S", "O"]:
                            self.region_names = [
                                n.decode("utf-8") if isinstance(n, bytes) else str(n)
                                for n in names_data
                            ]
                        else:
                            self.region_names = names_data.tolist()

                        if len(self.region_names) != self.n_regions:
                            warnings.warn(
                                "Number of region names doesn't match matrix size",
                                stacklevel=2,
                            )
                        break

                # Load region index (optional) — normalized to list[int]
                for key in ("gmindex", "index", "region_index"):
                    if key in data_group:
                        self.region_index = self._normalize_region_index(
                            data_group[key][:]
                        )
                        break
                else:
                    self.region_index = list(range(self.n_regions))

                # Load affine (optional)
                if "affine" in data_group:
                    self.affine = data_group["affine"][:]
                else:
                    self.affine = np.eye(4)

                # Load connectivity type (optional)
                if "type" in data_group.attrs:
                    self.type = data_group.attrs["type"]
                    if isinstance(self.type, bytes):
                        self.type = self.type.decode("utf-8")

        except Exception as e:
            raise RuntimeError(f"Error loading HDF5 file: {e}") from e

    #################################################################################
    def save_h5(
        self,
        filename: str | Path,
        compression: bool = True,
        sparse: bool | Literal["auto"] = "auto",
        sparse_density_threshold: float = 0.1,
    ) -> None:
        """
        Save Connectome to HDF5 file.

        Parameters:
        -----------
        filename : str or Path
            Output HDF5 filename
        compression : bool, optional
            Whether to use gzip compression (default: True)
        sparse : bool or "auto", optional
            How to store the connectivity matrix:
            - False: dense dataset "connmat/matrix"
            - True: CSR group "connmat/matrix_csr" (data, indices, indptr, shape)
            - "auto" (default): CSR if the fraction of nonzero entries in the
              full matrix is below ``sparse_density_threshold``, dense otherwise.
            Both layouts are read transparently by ``load_h5``.
        sparse_density_threshold : float, optional
            Nonzero fraction below which "auto" selects CSR (default: 0.1).
            CSR only saves space at low density, since each nonzero costs a
            value plus a column index.
        """
        if self.matrix is None:
            raise ValueError("No connectivity matrix to save")

        filename = Path(filename)

        with h5py.File(filename, "w") as f:
            # Create main group
            grp = f.create_group("connmat")

            # Save matrix (required), dense or CSR
            if sparse == "auto":
                nnz_fraction = (
                    np.count_nonzero(self.matrix) / self.matrix.size
                    if self.matrix.size > 0
                    else 0.0
                )
                use_sparse = nnz_fraction < sparse_density_threshold
            elif isinstance(sparse, bool):
                use_sparse = sparse
            else:
                raise ValueError(
                    f"sparse must be True, False or 'auto'. Got {sparse!r}"
                )

            comp = "gzip" if compression else None
            if use_sparse:
                self._write_csr_group(grp, "matrix_csr", self.to_csr(), comp)
            else:
                grp.create_dataset("matrix", data=self.matrix, compression=comp)

            # Save coordinates (if available)
            if self.region_coords is not None:
                grp.create_dataset("region_coords", data=self.region_coords)

            # Save colors (if available)
            if self.region_colors is not None:
                # Convert to hex strings
                colors_hex = cltcol.harmonize_colors(self.region_colors, "hex")
                # Ensure it's a list of strings
                if isinstance(colors_hex, np.ndarray):
                    colors_hex = colors_hex.tolist()
                # Save as UTF-8 encoded strings
                dt = h5py.string_dtype(encoding="utf-8")
                grp.create_dataset("region_colors", data=colors_hex, dtype=dt)

            # Save region names (if available)
            if self.region_names is not None:
                dt = h5py.string_dtype(encoding="utf-8")
                grp.create_dataset("region_names", data=self.region_names, dtype=dt)

            # Save region index
            if self.region_index is not None:
                grp.create_dataset("region_index", data=self.region_index)

            # Save affine
            grp.create_dataset("affine", data=self.affine)

            # Save metadata
            grp.attrs["type"] = self.type
            grp.attrs["n_regions"] = self.n_regions
            grp.attrs["density"] = self.get_density()
            grp.attrs["matrix_format"] = "csr" if use_sparse else "dense"

        print(f"Connectome saved to: {filename}")

    #################################################################################
    def get_region_names(self) -> list[str]:
        """
        Get region of interest (ROI) names. If not available, generate default names.

        Returns:
        --------
        List[str] : List of ROI names
        """
        if self.region_names is not None:
            return self.region_names
        else:
            return self.get_default_region_names()

    #################################################################################
    def get_region_colors(self) -> np.ndarray:
        """
        Get region of interest (ROI) colors. If not available, generate default colors.

        Returns:
        --------
        np.ndarray : Array of ROI colors
        """
        if self.region_colors is not None:
            return self.region_colors
        else:
            return self.get_default_region_colors()

    #################################################################################
    def get_region_coordinates(self) -> np.ndarray | None:
        """
        Get region of interest (ROI) coordinates.

        Returns:
        --------
        Optional[np.ndarray] : Array of ROI coordinates or None
        """
        return self.region_coords

    #################################################################################
    def get_region_indices(self) -> list[int] | None:
        """
        Get indices for brain regions.

        Returns:
        --------
        Optional[list[int]] : List of region indices or None
        """

        return self.region_index

    #################################################################################
    def set_region_coordinates(self, coordinates: np.ndarray) -> None:
        """
        Set 3D coordinates for brain regions.

        Parameters:
        -----------
        coordinates : np.ndarray
            Array of shape (n_regions, 3) with x, y, z coordinates
        """
        if self.matrix is not None and coordinates.shape != (self.n_regions, 3):
            raise ValueError(
                f"Coordinates shape {coordinates.shape} doesn't match expected ({self.n_regions}, 3)"
            )
        self.region_coords = coordinates.copy()

    #################################################################################
    def set_region_colors(self, colors: list | np.ndarray) -> None:
        """
        Set colors for brain regions.

        Parameters:
        -----------
        colors : np.ndarray or List
            Array of shape (n_regions, 3) with RGB values [0-1] or [0-255], or list of hex colors
        """
        if self.matrix is not None and len(colors) != self.n_regions:
            raise ValueError(
                f"Colors length {len(colors)} doesn't match expected ({self.n_regions})"
            )
        self.region_colors = cltcol.harmonize_colors(colors)

    #################################################################################
    def set_region_indices(self, indices: list[int] | np.ndarray) -> None:
        """
        Set indices for brain regions. Always stored internally as list[int],
        regardless of whether a list, tuple, or numpy array is supplied.

        Parameters:
        -----------
        indices : list[int] | np.ndarray
            List or array of region indices
        """

        n = self.n_regions if self.matrix is not None else None
        self.region_index = self._normalize_region_index(indices, n)

    #################################################################################
    def set_region_names(self, names: list[str]) -> None:
        """
        Set names for brain regions.

        Parameters:
        -----------
        names : List[str]
            List of region names
        """
        if self.matrix is not None and len(names) != self.n_regions:
            raise ValueError(
                f"Number of names {len(names)} doesn't match number of regions {self.n_regions}"
            )
        self.region_names = names.copy()

    #################################################################################
    def get_default_region_colors(self) -> np.ndarray:
        """
        Generate default colors for regions if not available.

        Returns:
        --------
        np.ndarray : RGB colors for each region
        """
        colors = cltcol.create_distinguishable_colors(self.n_regions)
        return colors

    def get_default_region_names(self) -> list[str]:
        """
        Generate default region names if not available.

        Returns:
        --------
        List[str] : Default region names
        """
        names = cltmisc.create_names_from_indices(np.arange(self.n_regions) + 1)
        return names

    #################################################################################
    def load_colortable(
        self, filename: str | Path | dict | cltcol.ColorTableLoader
    ) -> None:
        """
        Load color table for brain regions.

        Parameters:
        -----------
        filename : str | Path | dict | cltcol.ColorTableLoader
            Path to the LUT or TSV file containing region colors, a dictionary
            of colors/names/index, or an already-loaded ColorTableLoader.
        """
        if isinstance(filename, dict):

            colors = filename["color"]
            names = filename["name"]
            index = filename.get("index")
            self.set_region_colors(colors)
            self.set_region_names(names)
            if index is not None:
                self.set_region_indices(index)
            return

        if isinstance(filename, cltcol.ColorTableLoader):
            colors = filename.color
            names = filename.name
            index = getattr(filename, "index", None)
            self.set_region_colors(colors)
            self.set_region_names(names)
            if index is not None:
                self.set_region_indices(index)
            return

        filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        col_dict = cltcol.ColorTableLoader(filename)
        colors = col_dict.color
        names = col_dict.name
        index = getattr(col_dict, "index", None)

        self.set_region_colors(colors)
        self.set_region_names(names)
        if index is not None:
            self.set_region_indices(index)

    #################################################################################
    def to_csr(self, drop_diagonal: bool = False) -> csr_matrix:
        """
        Return the connectivity matrix as a scipy CSR sparse matrix.

        Useful for passing a Connectome to graph routines in networktools
        (e.g. ``connected_components``). The Connectome itself is not modified.

        Parameters:
        -----------
        drop_diagonal : bool, optional
            If True, self-connections are excluded (default: False).

        Returns:
        --------
        csr_matrix : Sparse (n_regions x n_regions) matrix with explicit zeros removed

        Examples:
        ---------
        >>> from clabtoolkit import networktools as cltnet
        >>> conn = Connectome.from_h5("sub-01_connectome.h5")
        >>> n_comp, labels, sizes = cltnet.connected_components(conn.to_csr())
        """
        if self.matrix is None:
            raise ValueError("No connectivity matrix available")

        csr = csr_matrix(self.matrix)
        if drop_diagonal:
            csr.setdiag(0)
        csr.eliminate_zeros()
        return csr

    #################################################################################
    @staticmethod
    def _write_csr_group(
        parent: h5py.Group,
        name: str,
        csr: csr_matrix,
        compression: str | None = "gzip",
    ) -> None:
        """
        Write a CSR matrix to an HDF5 group as data / indices / indptr datasets.

        The layout follows the common convention also used by AnnData (.h5ad):
        a group holding the three CSR arrays plus ``shape`` and
        ``encoding-type`` attributes.
        """
        g = parent.create_group(name)
        g.create_dataset("data", data=csr.data, compression=compression)
        g.create_dataset("indices", data=csr.indices, compression=compression)
        g.create_dataset("indptr", data=csr.indptr, compression=compression)
        g.attrs["shape"] = np.asarray(csr.shape, dtype=np.int64)
        g.attrs["encoding-type"] = "csr_matrix"

    #################################################################################
    @staticmethod
    def _read_csr_group(group: h5py.Group) -> csr_matrix:
        """
        Read a CSR matrix written by ``_write_csr_group``.
        """
        for key in ("data", "indices", "indptr"):
            if key not in group:
                raise KeyError(f"CSR group is missing the '{key}' dataset")
        if "shape" not in group.attrs:
            raise KeyError("CSR group is missing the 'shape' attribute")

        shape = tuple(int(x) for x in group.attrs["shape"])
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(f"Stored CSR matrix must be square. Got shape {shape}")

        return csr_matrix(
            (
                group["data"][:].astype(np.float64),
                group["indices"][:],
                group["indptr"][:],
            ),
            shape=shape,
        )

    #################################################################################
    def get_density(self) -> float:
        """
        Calculate the density of the connectivity matrix.

        Returns:
        -------
        float
            Proportion of non-zero connections (excluding diagonal)
        """
        if self.matrix is None:
            return 0.0

        n = self.matrix.shape[0]
        n_possible = n * (n - 1)  # Exclude diagonal

        # Count non-zero off-diagonal elements
        mask = ~np.eye(n, dtype=bool)
        n_connections = np.count_nonzero(self.matrix[mask])

        return n_connections / n_possible if n_possible > 0 else 0.0

    def get_connectivity_stats(self) -> dict:
        """
        Calculate basic connectivity statistics.

        Returns:
        --------
        dict : Dictionary with connectivity statistics
        """
        if self.matrix is None:
            return {}

        stats = {
            "n_regions": self.n_regions,
            "matrix_shape": self.matrix.shape,
            "min_strength": np.min(self.matrix),
            "max_strength": np.max(self.matrix),
            "mean_strength": np.mean(self.matrix),
            "std_strength": np.std(self.matrix),
            "density": self.get_density(),
            "node_strengths": np.sum(np.abs(self.matrix), axis=1),
        }

        if self.region_coords is not None:
            stats["coord_ranges"] = {
                "x": (
                    np.min(self.region_coords[:, 0]),
                    np.max(self.region_coords[:, 0]),
                ),
                "y": (
                    np.min(self.region_coords[:, 1]),
                    np.max(self.region_coords[:, 1]),
                ),
                "z": (
                    np.min(self.region_coords[:, 2]),
                    np.max(self.region_coords[:, 2]),
                ),
            }

        return stats

    #################################################################################
    def set_diagonal_to_zero(self):
        """
        Set the diagonal elements of the connectivity matrix to zero.
        """
        if self.matrix is None:
            raise ValueError("No connectivity matrix available")

        np.fill_diagonal(self.matrix, 0)

    def set_symmetric(self):
        """
        Make the connectivity matrix symmetric by averaging with its transpose.
        """
        if self.matrix is None:
            raise ValueError("No connectivity matrix available")

        self.matrix = (self.matrix + self.matrix.T) / 2

    #################################################################################
    def threshold(
        self,
        method: Literal["value", "sparsity"] = "value",
        threshold: float = 0.0,
        absolute: bool = True,
        binarize: bool = False,
        copy: bool = True,
    ) -> "Connectome":
        """
        Threshold the connectivity matrix.

        Parameters:
        ----------
        method : {'value', 'sparsity'}
            Thresholding method:
            - 'value': Keep connections above threshold value
            - 'sparsity': Keep top connections to achieve target sparsity
        threshold : float
            - For 'value': minimum connection strength to keep
            - For 'sparsity': target sparsity level (0-1), proportion of connections to keep
        absolute : bool, optional
            Use absolute values for thresholding (default: True)
        binarize : bool, optional
            Convert to binary matrix after thresholding (default: False)
        copy : bool, optional
            If True, return new Connectome object; if False, modify in place (default: True)

        Returns:
        -------
        Connectome
            Thresholded Connectome object (new if copy=True, self if copy=False)
        """
        if self.matrix is None:
            raise ValueError("No connectivity matrix available")

        matrix_thresh = self.matrix.copy()

        if method == "value":
            # Threshold by value
            if absolute:
                mask = np.abs(matrix_thresh) < threshold
            else:
                mask = matrix_thresh < threshold
            matrix_thresh[mask] = 0

        elif method == "sparsity":
            # Threshold by sparsity (keep top connections)
            if not 0 <= threshold <= 1:
                raise ValueError("Sparsity threshold must be between 0 and 1")

            # Get off-diagonal elements
            n = matrix_thresh.shape[0]
            mask_diag = ~np.eye(n, dtype=bool)
            values = matrix_thresh[mask_diag]

            # Use absolute values to determine which connections to keep
            if absolute:
                values_for_ranking = np.abs(values)
            else:
                values_for_ranking = values

            # Calculate how many connections to keep
            n_total = len(values)
            n_keep = int(n_total * threshold)

            if n_keep > 0:
                # Find threshold value (keep connections >= this value)
                sorted_values = np.sort(values_for_ranking)[::-1]
                value_threshold = sorted_values[min(n_keep - 1, len(sorted_values) - 1)]

                # Apply threshold
                if absolute:
                    mask = np.abs(matrix_thresh) < value_threshold
                else:
                    mask = matrix_thresh < value_threshold
                matrix_thresh[mask] = 0
            else:
                matrix_thresh[:] = 0

        else:
            raise ValueError(f"Unknown method: {method}. Use 'value' or 'sparsity'")

        # Binarize if requested
        if binarize:
            matrix_thresh = (matrix_thresh != 0).astype(np.float64)

        # Keep diagonal at zero
        np.fill_diagonal(matrix_thresh, 0)

        if copy:
            return Connectome(
                matrix=matrix_thresh,
                name=self.name,
                region_coords=(
                    self.region_coords.copy()
                    if self.region_coords is not None
                    else None
                ),
                region_colors=(
                    self.region_colors.copy()
                    if self.region_colors is not None
                    else None
                ),
                region_names=(
                    self.region_names.copy() if self.region_names is not None else None
                ),
                region_index=(
                    list(self.region_index) if self.region_index is not None else None
                ),
                connectivity_type=self.type,
                affine=self.affine.copy(),
            )
        else:
            # Modify in place
            self.matrix = matrix_thresh
            return self

    #################################################################################
    def get_subnetwork(
        self, region_indices: np.ndarray | list[int], copy: bool = True
    ) -> "Connectome":
        """
        Extract a subnetwork with selected regions.

        Parameters:
        ----------
        region_indices : np.ndarray or List[int]
            Positions (row/column indices) of regions to include
        copy : bool, optional
            If True, return new Connectome object; if False, modify in place (default: True)

        Returns:
        -------
        Connectome
            Subnetwork Connectome object
        """
        if self.matrix is None:
            raise ValueError("No connectivity matrix available")

        idx = np.array(region_indices)

        # Extract subnetwork data
        sub_matrix = self.matrix[np.ix_(idx, idx)]
        sub_coords = self.region_coords[idx] if self.region_coords is not None else None
        sub_colors = (
            None
            if self.region_colors is None
            else (
                [self.region_colors[i] for i in idx]
                if isinstance(self.region_colors, list)
                else self.region_colors[idx]
            )
        )
        sub_names = (
            [self.region_names[i] for i in idx]
            if self.region_names is not None
            else None
        )

        sub_index = (
            [self.region_index[i] for i in idx]
            if self.region_index is not None
            else None
        )

        if copy:
            return Connectome(
                matrix=sub_matrix,
                name=f"{self.name}_subnetwork" if self.name else "subnetwork",
                region_coords=sub_coords,
                region_colors=sub_colors,
                region_names=sub_names,
                region_index=sub_index,
                connectivity_type=self.type,
                affine=self.affine.copy(),
            )
        else:
            # Modify in place
            self.matrix = sub_matrix
            self.region_coords = sub_coords
            self.region_colors = sub_colors
            self.region_names = sub_names
            self.region_index = sub_index
            self.n_regions = len(idx)
            return self

    #################################################################################
    def copy(self) -> "Connectome":
        """
        Create a deep copy of the Connectome.

        Returns:
        -------
        Connectome
            Deep copy of the Connectome
        """
        return Connectome(
            matrix=self.matrix.copy() if self.matrix is not None else None,
            name=self.name,
            region_coords=(
                self.region_coords.copy() if self.region_coords is not None else None
            ),
            region_colors=(
                self.region_colors.copy() if self.region_colors is not None else None
            ),
            region_names=(
                self.region_names.copy() if self.region_names is not None else None
            ),
            region_index=(
                list(self.region_index) if self.region_index is not None else None
            ),
            connectivity_type=self.type,
            affine=self.affine.copy(),
        )

    #################################################################################
    def plot_matrix(
        self,
        figsize: tuple[int, int] = (12, 10),
        title: str | None = None,
        log_scale: bool = False,
        show_labels: bool = True,
        cmap: str = "RdBu_r",
        threshold: float | None = None,
        threshold_mode: str = "absolute",
    ) -> None:
        """
        Plot the connectivity matrix as a heatmap.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        show_labels : bool
            Whether to show region names on axes
        cmap : str
            Colormap for the heatmap
        threshold : float, optional
            Threshold value for displaying connections. Values below threshold will be set to 0.
        threshold_mode : str
            How to apply threshold: 'absolute' (abs(value) > threshold) or 'raw' (value > threshold)
        """
        if self.matrix is None:
            raise ValueError("No connectivity matrix available")

        plt.figure(figsize=figsize)

        # Apply threshold if specified
        matrix_to_plot = self.matrix.copy()
        if threshold is not None:
            if threshold_mode == "absolute":
                mask = np.abs(matrix_to_plot) < threshold
            else:  # raw mode
                mask = matrix_to_plot < threshold
            matrix_to_plot[mask] = 0
        # Apply log scale if specified
        if log_scale:
            # Use symmetric log scale to handle negative values
            matrix_to_plot = np.sign(matrix_to_plot) * np.log1p(np.abs(matrix_to_plot))

        # Create heatmap
        im = plt.imshow(matrix_to_plot, cmap=cmap, aspect="equal")
        plt.colorbar(im, label="Connection Strength")

        # Add threshold info to title
        if title is None:
            title = f"Connectivity Matrix - {self.name}"
            if threshold is not None:
                title += f" (threshold: {threshold}, mode: {threshold_mode})"

        # Add labels if available and requested
        if (
            show_labels
            and self.region_names is not None
            and len(self.region_names) < 50
        ):
            plt.xticks(
                range(len(self.region_names)),
                self.region_names,
                rotation=45,
                ha="right",
                fontsize=8,
            )
            plt.yticks(range(len(self.region_names)), self.region_names, fontsize=8)

        plt.title(title)
        plt.xlabel("Brain Regions")
        plt.ylabel("Brain Regions")
        plt.tight_layout()
        plt.show()

    #################################################################################
    def plot_circular_graph(
        self,
        figsize: tuple[int, int] = (12, 12),
        threshold: float | None = None,
        node_size_property: str = "strength",
        node_size_scale: float = 1000,
        edge_width_scale: float = 5,
        show_labels: bool = True,
        label_distance: float = 1.1,
        edge_alpha: float = 0.6,
        node_alpha: float = 0.8,
        edge_cmap: str = "plasma",
        layout_seed: int | None = 42,
    ) -> None:
        """
        Plot the connectivity matrix as a circular graph.

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        threshold : float, optional
            Minimum connection strength to display edges
        node_size_property : str
            Property to scale node sizes by: 'strength', 'degree', 'uniform'
        node_size_scale : float
            Scale factor for node sizes
        edge_width_scale : float
            Scale factor for edge widths
        show_labels : bool
            Whether to show region labels
        label_distance : float
            Distance of labels from nodes (1.0 = at node border)
        edge_alpha : float
            Transparency of edges (0-1)
        node_alpha : float
            Transparency of nodes (0-1)
        edge_cmap : str
            Colormap for edges based on connection strength
        layout_seed : int, optional
            Random seed for consistent layout
        """
        try:
            import networkx as nx
        except ImportError as err:
            raise ImportError(
                "NetworkX is required for circular graph visualization. "
                "Install with: pip install networkx"
            ) from err

        if self.matrix is None:
            raise ValueError("No connectivity matrix available")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create adjacency matrix for graph
        adj_matrix = self.matrix.copy()

        # Apply threshold if specified
        if threshold is not None:
            adj_matrix[np.abs(adj_matrix) < threshold] = 0

        # Create NetworkX graph
        G = nx.from_numpy_array(adj_matrix)

        # Get circular layout
        if layout_seed is not None:
            np.random.seed(layout_seed)
        pos = nx.circular_layout(G)

        # Calculate node sizes
        if node_size_property == "uniform":
            node_sizes = [
                node_size_scale * 0.1
            ] * self.n_regions  # Convert to reasonable size for circular plot
        elif node_size_property == "strength":
            strengths = np.sum(np.abs(adj_matrix), axis=1)
            if np.max(strengths) > 0:
                normalized_strengths = strengths / np.max(strengths)
            else:
                normalized_strengths = np.ones_like(strengths)
            node_sizes = normalized_strengths * node_size_scale + node_size_scale * 0.1
        elif node_size_property == "degree":
            degrees = np.array([G.degree(node) for node in G.nodes()])
            if np.max(degrees) > 0:
                normalized_degrees = degrees / np.max(degrees)
            else:
                normalized_degrees = np.ones_like(degrees)
            node_sizes = normalized_degrees * node_size_scale + node_size_scale * 0.1
        else:
            raise ValueError(f"Unknown node size property: {node_size_property}")

        # Get node colors
        node_colors = self.get_region_colors()

        # Get edge weights and colors
        edges = G.edges()
        edge_weights = []
        edge_colors = []

        for edge in edges:
            weight = abs(adj_matrix[edge[0], edge[1]])
            edge_weights.append(weight * edge_width_scale)
            edge_colors.append(weight)

        # Normalize edge colors
        if edge_colors:
            edge_colors = np.array(edge_colors)
            if np.max(edge_colors) > 0:
                edge_colors = edge_colors / np.max(edge_colors)

        # Draw edges
        if edges:
            nx.draw_networkx_edges(
                G,
                pos,
                width=edge_weights,
                edge_color=edge_colors,
                edge_cmap=plt.get_cmap(edge_cmap),
                alpha=edge_alpha,
                ax=ax,
            )

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=node_alpha,
            ax=ax,
        )

        # Add labels if requested
        if show_labels:
            region_names = self.get_region_names()

            # Create labels dictionary
            labels = {i: region_names[i] for i in range(self.n_regions)}

            # Calculate label positions
            label_pos = {}
            for node, (x, y) in pos.items():
                # Move labels slightly outward from nodes
                label_x = x * label_distance
                label_y = y * label_distance
                label_pos[node] = (label_x, label_y)

            # Draw labels
            nx.draw_networkx_labels(
                G, label_pos, labels=labels, font_size=8, font_weight="bold", ax=ax
            )

        # Set title
        title = f"Circular Graph - {self.name}"
        if threshold is not None:
            title += f" (threshold: {threshold})"
        if node_size_property != "uniform":
            title += f" (node size: {node_size_property})"

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        # Remove axes
        ax.set_axis_off()

        # Make layout tight
        plt.tight_layout()

        # Add colorbar for edges if there are edges
        if edges and len(edge_colors) > 0:
            # Create a dummy plot for colorbar
            sm = plt.cm.ScalarMappable(
                cmap=plt.get_cmap(edge_cmap),
                norm=plt.Normalize(
                    vmin=(
                        np.min(np.abs(adj_matrix)[adj_matrix != 0])
                        if threshold is None
                        else threshold
                    ),
                    vmax=np.max(np.abs(adj_matrix)),
                ),
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.1)
            cbar.set_label("Connection Strength", rotation=270, labelpad=20)

        plt.show()

    #################################################################################
    @classmethod
    def generate_connectome(
        cls,
        n_regions: int,
        method: str = "random",
        value_range: tuple[float, float] = (0.0, 1.0),
        sparsity: float | None = None,
        region_names: list[str] | None = None,
        region_colors: np.ndarray | list | None = None,
        region_coords: np.ndarray | None = None,
        connectivity_type: str = "unknown",
        name: str | None = None,
        symmetric: bool = True,
        n_modules: int = 4,
        distance_decay: float = 25.0,
        seed: int | None = None,
    ) -> "Connectome":
        """
        Generate a synthetic Connectome.

        Builds an (n_regions x n_regions) connectivity matrix using one of several
        generation strategies and wraps it in a Connectome object. Region names and
        colors are created automatically when not supplied (using the same helpers
        __init__ relies on), and 3D coordinates are generated on a sphere so the
        result is immediately usable with the 3D / circular visualizations.

        Parameters
        ----------
        n_regions : int
            Number of brain regions (matrix will be n_regions x n_regions).
        method : {'random', 'modular', 'distance'}
            Strategy used to fill the matrix:
              - 'random'   : i.i.d. uniform weights drawn from `value_range`.
              - 'modular'  : block/community structure — strong within-module,
                             weak between-module weights (alias: 'method1').
              - 'distance' : weights decay with the Euclidean distance between
                             region coordinates, i.e. nearby regions are more
                             strongly connected (alias: 'method2').
        value_range : (float, float)
            (low, high) bounds for the generated weights. Default (0.0, 1.0).
        sparsity : float, optional
            Fraction of off-diagonal edges to set to zero, in [0, 1].
            0.0 -> fully dense, 0.9 -> 90% of edges removed (very sparse).
            Edges are removed at random (symmetrically when `symmetric=True`).
            None (default) leaves the matrix dense.
        region_names : list of str, optional
            Region labels. Auto-generated when None.
        region_colors : np.ndarray or list, optional
            RGB values or hex strings. Auto-generated when None.
        region_coords : np.ndarray, optional
            (n_regions, 3) coordinates. Auto-generated on a sphere when None.
            Required by (and drives) the 'distance' method.
        connectivity_type : str
            Stored connectivity type. Default 'unknown'.
        name : str, optional
            Name for the connectome. Defaults to 'synthetic_<method>_<n_regions>'.
        symmetric : bool
            If True (default) the matrix is symmetric (undirected). Set False for
            directed weights.
        n_modules : int
            Number of communities for the 'modular' method. Default 4.
        distance_decay : float
            Characteristic length (same units as coordinates) controlling how fast
            weights fall off with distance in the 'distance' method. Default 25.0.
        seed : int, optional
            Seed for reproducible generation.

        Returns
        -------
        Connectome
            A new Connectome populated with the synthetic matrix, coordinates,
            names and colors.

        Examples
        --------
        >>> conn = Connectome.generate_connectome(84, seed=0)
        >>> conn = Connectome.generate_connectome(
        ...     100, method="modular", value_range=(-1, 1),
        ...     n_modules=6, sparsity=0.7, seed=1,
        ... )
        >>> conn = Connectome.generate_connectome(
        ...     coords.shape[0], method="distance", region_coords=coords,
        ... )
        """
        if n_regions <= 0:
            raise ValueError("n_regions must be a positive integer")

        if region_names is not None and len(region_names) != n_regions:
            raise ValueError(
                f"Length of region_names must match n_regions ({n_regions}), "
                f"got {len(region_names)}"
            )
        elif region_names is None:
            region_names = cltmisc.create_names_from_indices(np.arange(n_regions) + 1)

        if region_colors is not None:
            region_colors = cltcol.harmonize_colors(region_colors)
            if len(region_colors) != n_regions:
                raise ValueError(
                    f"Length of region_colors must match n_regions ({n_regions}), "
                    f"got {len(region_colors)}"
                )
        else:
            region_colors = cltcol.create_distinguishable_colors(n_regions)

        low, high = value_range
        if low > high:
            raise ValueError(
                f"value_range must be (low, high) with low <= high, got {value_range}"
            )
        span = high - low

        # Normalize method name and resolve aliases
        method = str(method).lower()
        method = {"method1": "modular", "method2": "distance"}.get(method, method)

        rng = np.random.default_rng(seed)

        # --- Coordinates (needed for 'distance', handy for visualization) -----
        if region_coords is None:
            region_coords = cltplot.generate_spherical_coords(n_regions, rng)
        else:
            region_coords = np.asarray(region_coords, dtype=float)
            if region_coords.shape != (n_regions, 3):
                raise ValueError(
                    f"region_coords must have shape ({n_regions}, 3), "
                    f"got {region_coords.shape}"
                )

        # --- Build the raw matrix ---------------------------------------------
        if method == "random":
            matrix = rng.uniform(low, high, size=(n_regions, n_regions))

        elif method == "modular":
            n_mod = max(1, min(int(n_modules), n_regions))
            # Contiguous, roughly balanced module assignment
            module_ids = (np.arange(n_regions) * n_mod) // n_regions
            same_module = module_ids[:, None] == module_ids[None, :]
            within = rng.uniform(low + 0.6 * span, high, size=(n_regions, n_regions))
            between = rng.uniform(low, low + 0.3 * span, size=(n_regions, n_regions))
            matrix = np.where(same_module, within, between)

        elif method == "distance":
            diff = region_coords[:, None, :] - region_coords[None, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            weights = np.exp(-dist / float(distance_decay))
            off = ~np.eye(n_regions, dtype=bool)
            if np.any(off):
                wmin, wmax = weights[off].min(), weights[off].max()
                norm = (
                    (weights - wmin) / (wmax - wmin)
                    if wmax > wmin
                    else np.full_like(weights, 0.5)
                )
            else:
                norm = np.zeros_like(weights)
            matrix = low + norm * span

        else:
            raise ValueError(
                f"Unknown method: {method!r}. "
                f"Use 'random', 'modular' ('method1') or 'distance' ('method2')."
            )

        # --- Enforce symmetry and zero diagonal -------------------------------
        if symmetric:
            upper = np.triu(matrix, k=1)
            matrix = upper + upper.T
        np.fill_diagonal(matrix, 0.0)

        # --- Apply sparsity (random edge removal) -----------------------------
        if sparsity is not None:
            if not 0.0 <= sparsity <= 1.0:
                raise ValueError("sparsity must be between 0 and 1")
            if symmetric:
                rows, cols = np.triu_indices(n_regions, k=1)
            else:
                rows, cols = np.where(~np.eye(n_regions, dtype=bool))
            n_zero = int(round(sparsity * rows.size))
            if n_zero > 0:
                sel = rng.choice(rows.size, size=n_zero, replace=False)
                matrix[rows[sel], cols[sel]] = 0.0
                if symmetric:
                    matrix[cols[sel], rows[sel]] = 0.0

        if name is None:
            name = f"synthetic_{method}_{n_regions}"

        return cls(
            matrix=matrix,
            name=name,
            region_coords=region_coords,
            region_names=region_names,
            region_colors=region_colors,
            connectivity_type=connectivity_type,
        )

    #################################################################################
    def visualize_3d(
        self,
        connectivity_threshold: float = 0.1,
        node_size_scale: float = 1.0,
        edge_width_scale: float = 1.0,
        show_edges: bool = True,
        show_labels: bool = False,
        background_color: str = "black",
        window_size: tuple[int, int] = (1200, 800),
        node_size_property: str = "strength",
        base_node_size: float = 0.5,
        notebook: bool | None = None,
        off_screen: bool | None = None,
    ) -> pv.Plotter:
        """
        Create a 3D visualization of the connectome using PyVista.

        Parameters:
        -----------
        connectivity_threshold : float
            Minimum connection strength to display edges
        node_size_scale : float
            Scale factor for node sizes
        edge_width_scale : float
            Scale factor for edge widths
        show_edges : bool
            Whether to show connectivity edges
        show_labels : bool
            Whether to show region labels
        background_color : str
            Background color for the plot
        window_size : tuple
            Window size (width, height)
        node_size_property : str
            Property to scale node sizes by:
            - 'strength': Total connectivity strength (sum of absolute connections)
            - 'degree': Number of connections above threshold
            - 'uniform': All nodes same size (base_node_size)
            - 'betweenness': Betweenness centrality (requires networkx)
            - 'eigenvector': Eigenvector centrality (requires networkx)
        base_node_size : float
            Base size for nodes when using 'uniform' or as minimum size for other properties
        notebook : bool, optional
            Whether to render inline for a Jupyter notebook (True) or in a
            separate interactive window (False). If None (default), PyVista
            auto-detects based on the current environment — this is what was
            happening implicitly before and is often wrong (e.g. it can try
            to pop a window from inside a headless notebook kernel, or fail
            to embed inline when it should). Pass explicitly to override.
        off_screen : bool, optional
            Render without opening any window/display at all — useful for
            headless environments (CI, remote servers) or when you only want
            to call ``plotter.screenshot(...)`` / use ``save_visualization``
            without ever displaying anything. If None (default), PyVista's
            own default is used (generally tied to a ``DISPLAY`` being
            available). Note that ``save_visualization`` does not need
            ``plotter.show()`` to produce a screenshot, so setting
            ``off_screen=True`` there is safe even outside a notebook.

        Returns:
        --------
        pv.Plotter : PyVista plotter object
        """
        if self.matrix is None:
            raise ValueError("No connectivity matrix available")
        if self.region_coords is None:
            raise ValueError("No coordinates available for 3D visualization")

        # Create plotter. `notebook`/`off_screen` are only passed through when
        # explicitly set, so leaving both at None preserves PyVista's own
        # default auto-detection behavior exactly as before.
        plotter_kwargs = {"window_size": window_size}
        if notebook is not None:
            plotter_kwargs["notebook"] = notebook
        if off_screen is not None:
            plotter_kwargs["off_screen"] = off_screen

        plotter = pv.Plotter(**plotter_kwargs)
        plotter.set_background(background_color)

        # Center coordinates around origin
        coords_centered = self.region_coords - np.mean(self.region_coords, axis=0)

        # Calculate node sizes based on selected property
        node_sizes = self._calculate_node_sizes(
            node_size_property, connectivity_threshold, node_size_scale, base_node_size
        )

        # Get colors (use provided or generate defaults)
        colors = self.get_region_colors()

        region_names = self.get_region_names()

        # Add nodes (brain regions)
        for i in range(self.n_regions):
            # Create sphere for each region
            sphere = pv.Sphere(radius=node_sizes[i], center=coords_centered[i])

            # Add sphere to plotter with color
            plotter.add_mesh(
                sphere,
                color=colors[i],
                opacity=0.8,
                smooth_shading=True,
                name=f"region_{i}",
            )

            # Add labels if requested
            if show_labels:
                plotter.add_point_labels(
                    coords_centered[i : i + 1],
                    [region_names[i]],
                    font_size=8,
                    text_color="white",
                )

        # Add connectivity edges
        if show_edges:
            # Get upper triangle indices (avoid duplicate edges)
            i_indices, j_indices = np.triu_indices(self.n_regions, k=1)

            for idx in range(len(i_indices)):
                i, j = i_indices[idx], j_indices[idx]
                connection_strength = abs(self.matrix[i, j])

                if connection_strength > connectivity_threshold:
                    # Create line between regions
                    points = np.array([coords_centered[i], coords_centered[j]])
                    line = pv.Line(points[0], points[1])

                    # Scale line width based on connection strength
                    line_width = connection_strength * edge_width_scale * 5 + 1

                    # Color edges based on connection strength
                    edge_color = plt.cm.plasma(
                        connection_strength / np.max(np.abs(self.matrix))
                    )[:3]

                    plotter.add_mesh(
                        line,
                        color=edge_color,
                        line_width=line_width,
                        opacity=0.6,
                        name=f"edge_{i}_{j}",
                    )

        # Set up camera and lighting
        plotter.camera_position = "xy"
        plotter.add_axes()

        # Add title
        title = f"Brain Connectivity Network - {self.name}"
        if node_size_property != "uniform":
            title += f" (node size: {node_size_property})"
        plotter.add_title(title, font_size=16, color="white")

        return plotter

    #################################################################################
    def save_visualization(self, filename: str, **kwargs) -> None:
        """
        Save a 3D visualization to file.

        Parameters:
        -----------
        filename : str
            Output filename for the visualization
        **kwargs : dict
            Additional arguments passed to visualize_3d(). Since this method
            only takes a screenshot and never calls plotter.show(), it
            defaults to off_screen=True (no window is ever displayed) unless
            you explicitly pass off_screen=False in kwargs.
        """
        kwargs.setdefault("off_screen", True)
        plotter = self.visualize_3d(**kwargs)
        plotter.screenshot(filename)
        plotter.close()

    #################################################################################
    def get_info(self) -> None:
        """
        Display comprehensive information about the connectome.

        Provides a formatted overview including region count, connectivity type,
        matrix statistics, coordinate ranges, and availability of optional data
        (colors, names, index). Useful for quick inspection and validation.

        The method displays:
            - Basic identification (name, type, number of regions)
            - Matrix statistics (min, max, mean ± SD, density)
            - Coordinate ranges per axis (if available)
            - Availability and shape of optional attributes

        Returns
        -------
        None
            Prints formatted information to stdout.

        Examples
        --------
        >>> conn = Connectome('/path/to/connectome.h5')
        >>> conn.get_info()
        ╔════════════════════════════════════════════════════════════════╗
        ║                     CONNECTOME OVERVIEW                        ║
        ╠════════════════════════════════════════════════════════════════╣
        ║ Name: my_connectome                                            ║
        ║ Type: structural                                               ║
        ║ Regions: 84                                                    ║
        ╠════════════════════════════════════════════════════════════════╣
        ║ MATRIX STATISTICS                                              ║
        ║   Shape:      84 × 84                                          ║
        ║   Range:      [0.000, 1.000]                                   ║
        ║   Mean ± SD:  0.123 ± 0.045                                    ║
        ║   Density:    0.312                                            ║
        ╠════════════════════════════════════════════════════════════════╣
        ║ COORDINATE RANGES                                              ║
        ║   X:  [-72.10,  68.40]                                         ║
        ║   Y:  [-98.50,  76.20]                                         ║
        ║   Z:  [-32.10,  78.90]                                         ║
        ╠════════════════════════════════════════════════════════════════╣
        ║ OPTIONAL DATA                                                  ║
        ║   Region names:   ✔  (84 entries)                              ║
        ║   Region colors:  ✔  (84 × 3)                                  ║
        ║   Region index:   ✔  (84 entries)                              ║
        ╚════════════════════════════════════════════════════════════════╝
        """

        def print_line(content, width=64):
            print(f"║{content.ljust(width)}║")

        width = 64
        print("╔" + "═" * width + "╗")
        print_line("CONNECTOME OVERVIEW".center(width))
        print("╠" + "═" * width + "╣")

        # Basic identification
        print_line(f" Name:    {self.name if self.name else 'N/A'}")
        print_line(f" Type:    {self.type}")

        if self.matrix is None:
            print_line(" No connectivity matrix loaded.")
            print("╚" + "═" * width + "╝")
            return

        print_line(f" Regions: {self.n_regions:,}")

        # Matrix statistics
        print("╠" + "═" * width + "╣")
        print_line(" MATRIX STATISTICS")
        print_line(f"   Shape:      {self.matrix.shape[0]} × {self.matrix.shape[1]}")
        print_line(
            f"   Range:      [{np.min(self.matrix):.4f},  {np.max(self.matrix):.4f}]"
        )
        print_line(
            f"   Mean ± SD:  {np.mean(self.matrix):.4f} ± {np.std(self.matrix):.4f}"
        )
        print_line(f"   Density:    {self.get_density():.4f}")

        # Coordinate ranges
        print("╠" + "═" * width + "╣")
        if self.region_coords is not None:
            print_line(" COORDINATE RANGES")

            # Nodes without coordinates are stored as NaN, so the ranges are
            # computed over the nodes that do have them
            n_missing = int(np.sum(np.all(np.isnan(self.region_coords), axis=1)))

            for axis, label in enumerate(("X", "Y", "Z")):
                column = self.region_coords[:, axis]
                if np.all(np.isnan(column)):
                    print_line(f"   {label}:  [     n/a,      n/a]")
                else:
                    lo = np.nanmin(column)
                    hi = np.nanmax(column)
                    print_line(f"   {label}:  [{lo:8.2f}, {hi:8.2f}]")

            if n_missing:
                print_line(f"   Nodes without coordinates: {n_missing}")
        else:
            print_line(" COORDINATE RANGES")
            print_line("   Not available — 3D visualization disabled")

        # Optional data availability
        print("╠" + "═" * width + "╣")
        print_line(" OPTIONAL DATA")
        tick = "✔"
        cross = "✘"

        if self.region_names is not None:
            print_line(f"   Region names:   {tick}  ({len(self.region_names)} entries)")
        else:
            print_line(f"   Region names:   {cross}")

        if self.region_colors is not None:
            n = len(self.region_colors)
            if hasattr(self.region_colors, "shape"):
                shape_str = " × ".join(str(s) for s in self.region_colors.shape)
            else:
                shape_str = str(n)
            print_line(f"   Region colors:  {tick}  ({shape_str})")
        else:
            print_line(f"   Region colors:  {cross}")

        if self.region_index is not None:
            print_line(f"   Region index:   {tick}  ({len(self.region_index)} entries)")
        else:
            print_line(f"   Region index:   {cross}")

        print("╚" + "═" * width + "╝")

    def __repr__(self) -> str:
        """String representation of the Connectome object."""
        if self.matrix is None:
            return f"Connectome(name='{self.name}', no data loaded)"
        return f"Connectome(name='{self.name}', type='{self.type}', n_regions={self.n_regions}, density={self.get_density():.3f})"
