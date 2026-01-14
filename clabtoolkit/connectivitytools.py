import numpy as np
import pyvista as pv
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, List, Tuple, Literal
import warnings

from . import colorstools as cltcol
from . import misctools as cltmisc


class Connectome:
    """
    A class to represent and visualize brain connectivity data.

    Attributes:
    -----------
    name : str
        Name identifier for the connectome
    matrix : np.ndarray
        Connectivity matrix (n_regions x n_regions)
    coordinates : np.ndarray
        3D coordinates for each region (n_regions x 3)
    colors : np.ndarray
        RGB color values for each region (n_regions x 3)
    region_names : List[str]
        Names/labels for each brain region
    region_index : np.ndarray
        Index codes for each region
    connectivity_type : str
        Type of connectivity ('structural', 'functional', 'effective', etc.)
    affine : np.ndarray
        4x4 affine transformation matrix
    n_regions : int
        Number of brain regions
    """

    def __init__(
        self,
        data: Optional[Union[np.ndarray, str, Path]] = None,
        name: Optional[str] = None,
        coordinates: Optional[np.ndarray] = None,
        colors: Union[np.ndarray, List] = None,
        region_names: Optional[List[str]] = None,
        region_index: Optional[np.ndarray] = None,
        connectivity_type: str = "structural",
        affine: Optional[np.ndarray] = None,
    ):
        """
        Initialize a Connectome object.

        Parameters:
        -----------
        data : np.ndarray, str, Path, or None
            Can be:
            - np.ndarray: Connectivity matrix (n_regions x n_regions)
            - str or Path: Path to HDF5 file to load
            - None: Create empty Connectome
        name : str, optional
            Name for the connectome. If loading from file and None, uses filename stem.
        coordinates : np.ndarray, optional
            3D coordinates for each region (n_regions x 3)
        colors : np.ndarray or List, optional
            RGB color values or hex strings for each region
        region_names : List[str], optional
            Names/labels for each brain region
        region_index : np.ndarray, optional
            Index codes for each region
        connectivity_type : str, optional
            Type of connectivity (default: 'structural')
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
        self.connectivity_type = connectivity_type

        # Handle different input types for data
        matrix = None
        load_from_file = False

        if data is not None:
            if isinstance(data, (str, Path)):
                # Load from file
                load_from_file = True
                filepath = Path(data)

                # Set default name from filename if not provided
                if name is None:
                    name = filepath.stem

            elif isinstance(data, np.ndarray):
                # Use as connectivity matrix
                matrix = data
            else:
                raise TypeError(
                    f"data must be np.ndarray, str, Path, or None. Got {type(data)}"
                )

        self.name = name

        # Initialize matrix and derived attributes
        if matrix is not None:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Matrix must be square (NxN)")
            self.matrix = matrix.astype(np.float64)
            self._n_regions = matrix.shape[0]
        else:
            self.matrix = None
            self._n_regions = 0

        # Set coordinates
        if coordinates is not None:
            self.set_coordinates(coordinates)
        else:
            self.coordinates = None

        # Set colors
        if colors is not None:
            colors = cltcol.harmonize_colors(colors, "hex")
            self.set_colors(colors)
        elif self._n_regions > 0:
            # Only generate default colors if we have regions
            self.colors = cltcol.create_distinguishable_colors(self._n_regions)
        else:
            self.colors = None

        # Set region names
        if region_names is not None:
            self.set_region_names(region_names)
        elif self._n_regions > 0:
            # Only generate default names if we have regions
            self.region_names = cltmisc.create_names_from_indices(
                np.arange(self._n_regions) + 1
            )
        else:
            self.region_names = None

        # Set region index
        if region_index is not None:
            if self.matrix is not None and len(region_index) != self._n_regions:
                raise ValueError(
                    f"Region index length ({len(region_index)}) must match matrix size ({self._n_regions})"
                )
            self.region_index = np.array(region_index)
        else:
            self.region_index = (
                np.arange(self._n_regions) if self.matrix is not None else None
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

    @property
    def n_regions(self) -> int:
        """Get the number of brain regions."""
        return self._n_regions

    @classmethod
    def from_h5(
        cls, filename: Union[str, Path], name: Optional[str] = None
    ) -> "Connectome":
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
                    "NetworkX not available. Using strength instead of betweenness centrality."
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
                        "Eigenvector centrality failed to converge. Using strength instead."
                    )
                    return self._calculate_node_sizes(
                        "strength", threshold, scale, base_size
                    )
            except ImportError:
                warnings.warn(
                    "NetworkX not available. Using strength instead of eigenvector centrality."
                )
                return self._calculate_node_sizes(
                    "strength", threshold, scale, base_size
                )

        else:
            raise ValueError(
                f"Unknown node size property: {property_type}. "
                f"Available options: 'uniform', 'strength', 'degree', 'betweenness', 'eigenvector'"
            )

    def load_h5(self, filename: Union[str, Path]) -> None:
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

                # Load connectivity matrix (required)
                if "matrix" in data_group:
                    self.matrix = data_group["matrix"][:]
                else:
                    raise KeyError("No 'matrix' dataset found in HDF5 file")

                self._n_regions = self.matrix.shape[0]

                # Load coordinates (required for visualization)
                if "coords" in data_group:
                    self.coordinates = data_group["coords"][:]
                    if self.coordinates.shape[0] != self._n_regions:
                        raise ValueError(
                            "Number of coordinates doesn't match matrix size"
                        )
                elif "gmcoords" in data_group:  # Alternative name
                    self.coordinates = data_group["gmcoords"][:]
                    if self.coordinates.shape[0] != self._n_regions:
                        raise ValueError(
                            "Number of coordinates doesn't match matrix size"
                        )
                elif "coordinates" in data_group:  # Alternative name
                    self.coordinates = data_group["coordinates"][:]
                    if self.coordinates.shape[0] != self._n_regions:
                        raise ValueError(
                            "Number of coordinates doesn't match matrix size"
                        )
                else:
                    warnings.warn(
                        "No coordinates found. 3D visualization will not be available."
                    )

                # Load colors (optional)
                if "gmcolors" in data_group:
                    colors_data = data_group["gmcolors"][:]
                    # Decode bytes to strings if necessary
                    if colors_data.dtype.kind in ["S", "O"]:
                        colors_list = [
                            c.decode("utf-8") if isinstance(c, bytes) else str(c)
                            for c in colors_data
                        ]
                    else:
                        colors_list = colors_data.tolist()
                    # Use harmonize_colors to convert to RGB array
                    self.colors = cltcol.harmonize_colors(colors_list)

                elif "colors" in data_group:  # Alternative name (legacy format)
                    self.colors = data_group["colors"][:]
                    if self.colors.shape[0] != self._n_regions:
                        warnings.warn("Number of colors doesn't match matrix size")
                    else:
                        # Normalize colors to [0,1] range if needed
                        if np.max(self.colors) > 1:
                            self.colors = self.colors / 255.0

                # Load region names (optional)
                if "gmregions" in data_group:
                    names_data = data_group["gmregions"][:]
                    # Decode bytes to strings if necessary
                    if names_data.dtype.kind in ["S", "O"]:
                        self.region_names = [
                            n.decode("utf-8") if isinstance(n, bytes) else str(n)
                            for n in names_data
                        ]
                    else:
                        self.region_names = names_data.tolist()

                    if len(self.region_names) != self._n_regions:
                        warnings.warn(
                            "Number of region names doesn't match matrix size"
                        )
                elif "name" in data_group:  # Alternative name
                    names_data = data_group["name"][:]
                    if names_data.dtype.kind in ["S", "O"]:
                        self.region_names = [
                            n.decode("utf-8") if isinstance(n, bytes) else str(n)
                            for n in names_data
                        ]
                    else:
                        self.region_names = names_data.tolist()

                    if len(self.region_names) != self._n_regions:
                        warnings.warn(
                            "Number of region names doesn't match matrix size"
                        )

                # Load region index (optional)
                if "gmindex" in data_group:
                    self.region_index = data_group["gmindex"][:]
                elif "index" in data_group:
                    self.region_index = data_group["index"][:]
                else:
                    self.region_index = np.arange(self._n_regions)

                # Load affine (optional)
                if "affine" in data_group:
                    self.affine = data_group["affine"][:]
                else:
                    self.affine = np.eye(4)

                # Load connectivity type (optional)
                if "type" in data_group.attrs:
                    self.connectivity_type = data_group.attrs["type"]
                    if isinstance(self.connectivity_type, bytes):
                        self.connectivity_type = self.connectivity_type.decode("utf-8")

        except Exception as e:
            raise RuntimeError(f"Error loading HDF5 file: {e}")

    def save_h5(self, filename: Union[str, Path], compression: bool = True) -> None:
        """
        Save Connectome to HDF5 file.

        Parameters:
        -----------
        filename : str or Path
            Output HDF5 filename
        compression : bool, optional
            Whether to use gzip compression (default: True)
        """
        if self.matrix is None:
            raise ValueError("No connectivity matrix to save")

        filename = Path(filename)

        with h5py.File(filename, "w") as f:
            # Create main group
            grp = f.create_group("connmat")

            # Save matrix (required)
            if compression:
                grp.create_dataset("matrix", data=self.matrix, compression="gzip")
            else:
                grp.create_dataset("matrix", data=self.matrix)

            # Save coordinates (if available)
            if self.coordinates is not None:
                grp.create_dataset("coords", data=self.coordinates)

            # Save colors (if available)
            if self.colors is not None:
                # Convert to hex strings
                colors_hex = cltcol.harmonize_colors(self.colors, "hex")
                # Ensure it's a list of strings
                if isinstance(colors_hex, np.ndarray):
                    colors_hex = colors_hex.tolist()
                # Save as UTF-8 encoded strings
                dt = h5py.string_dtype(encoding="utf-8")
                grp.create_dataset("gmcolors", data=colors_hex, dtype=dt)

            # Save region names (if available)
            if self.region_names is not None:
                dt = h5py.string_dtype(encoding="utf-8")
                grp.create_dataset("gmregions", data=self.region_names, dtype=dt)

            # Save region index
            if self.region_index is not None:
                grp.create_dataset("gmindex", data=self.region_index)

            # Save affine
            grp.create_dataset("affine", data=self.affine)

            # Save metadata
            grp.attrs["type"] = self.connectivity_type
            grp.attrs["n_regions"] = self.n_regions
            grp.attrs["density"] = self.get_density()

        print(f"Connectome saved to: {filename}")

    def get_roi_names(self) -> List[str]:
        """
        Get region of interest (ROI) names. If not available, generate default names.

        Returns:
        --------
        List[str] : List of ROI names
        """
        if self.region_names is not None:
            return self.region_names
        else:
            return self.get_default_roi_names()

    def get_roi_colors(self) -> np.ndarray:
        """
        Get region of interest (ROI) colors. If not available, generate default colors.

        Returns:
        --------
        np.ndarray : Array of ROI colors
        """
        if self.colors is not None:
            return self.colors
        else:
            return self.get_default_roi_colors()

    def get_roi_coordinates(self) -> Optional[np.ndarray]:
        """
        Get region of interest (ROI) coordinates.

        Returns:
        --------
        Optional[np.ndarray] : Array of ROI coordinates or None
        """
        return self.coordinates

    def set_coordinates(self, coordinates: np.ndarray) -> None:
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
        self.coordinates = coordinates.copy()

    def set_colors(self, colors: Union[List, np.ndarray]) -> None:
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
        self.colors = cltcol.harmonize_colors(colors)

    def set_region_names(self, names: List[str]) -> None:
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

    def get_default_roi_colors(self) -> np.ndarray:
        """
        Generate default colors for regions if not available.

        Returns:
        --------
        np.ndarray : RGB colors for each region
        """
        colors = cltcol.create_distinguishable_colors(self.n_regions)
        return colors

    def get_default_roi_names(self) -> List[str]:
        """
        Generate default region names if not available.

        Returns:
        --------
        List[str] : Default region names
        """
        names = cltmisc.create_names_from_indices(np.arange(self.n_regions) + 1)
        return names

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

        if self.coordinates is not None:
            stats["coord_ranges"] = {
                "x": (np.min(self.coordinates[:, 0]), np.max(self.coordinates[:, 0])),
                "y": (np.min(self.coordinates[:, 1]), np.max(self.coordinates[:, 1])),
                "z": (np.min(self.coordinates[:, 2]), np.max(self.coordinates[:, 2])),
            }

        return stats

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
            # Create new Connectome object
            return Connectome(
                data=matrix_thresh,
                name=self.name,
                coordinates=(
                    self.coordinates.copy() if self.coordinates is not None else None
                ),
                colors=self.colors.copy() if self.colors is not None else None,
                region_names=(
                    self.region_names.copy() if self.region_names is not None else None
                ),
                region_index=(
                    self.region_index.copy() if self.region_index is not None else None
                ),
                connectivity_type=self.connectivity_type,
                affine=self.affine.copy(),
            )
        else:
            # Modify in place
            self.matrix = matrix_thresh
            return self

    def get_subnetwork(
        self, region_indices: Union[np.ndarray, List[int]], copy: bool = True
    ) -> "Connectome":
        """
        Extract a subnetwork with selected regions.

        Parameters:
        ----------
        region_indices : np.ndarray or List[int]
            Indices of regions to include
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
        sub_coords = self.coordinates[idx] if self.coordinates is not None else None
        sub_colors = self.colors[idx] if self.colors is not None else None
        sub_names = (
            [self.region_names[i] for i in idx]
            if self.region_names is not None
            else None
        )
        sub_index = self.region_index[idx] if self.region_index is not None else None

        if copy:
            return Connectome(
                data=sub_matrix,
                name=f"{self.name}_subnetwork" if self.name else "subnetwork",
                coordinates=sub_coords,
                colors=sub_colors,
                region_names=sub_names,
                region_index=sub_index,
                connectivity_type=self.connectivity_type,
                affine=self.affine.copy(),
            )
        else:
            # Modify in place
            self.matrix = sub_matrix
            self.coordinates = sub_coords
            self.colors = sub_colors
            self.region_names = sub_names
            self.region_index = sub_index
            self._n_regions = len(idx)
            return self

    def copy(self) -> "Connectome":
        """
        Create a deep copy of the Connectome.

        Returns:
        -------
        Connectome
            Deep copy of the Connectome
        """
        return Connectome(
            data=self.matrix.copy() if self.matrix is not None else None,
            name=self.name,
            coordinates=(
                self.coordinates.copy() if self.coordinates is not None else None
            ),
            colors=self.colors.copy() if self.colors is not None else None,
            region_names=(
                self.region_names.copy() if self.region_names is not None else None
            ),
            region_index=(
                self.region_index.copy() if self.region_index is not None else None
            ),
            connectivity_type=self.connectivity_type,
            affine=self.affine.copy(),
        )

    def print_info(self) -> None:
        """Print comprehensive information about the connectome."""
        print(f"=== Connectome: {self.name} ===")

        if self.matrix is None:
            print("No connectivity data loaded.")
            return

        stats = self.get_connectivity_stats()
        print(f"Number of regions: {stats['n_regions']}")
        print(f"Connectivity type: {self.connectivity_type}")
        print(f"Matrix shape: {stats['matrix_shape']}")
        print(
            f"Connection strength range: [{stats['min_strength']:.3f}, {stats['max_strength']:.3f}]"
        )
        print(f"Mean ± SD: {stats['mean_strength']:.3f} ± {stats['std_strength']:.3f}")
        print(f"Network density: {stats['density']:.3f}")

        if "coord_ranges" in stats:
            print("Coordinate ranges:")
            for axis, (min_val, max_val) in stats["coord_ranges"].items():
                print(f"  {axis.upper()}: [{min_val:.1f}, {max_val:.1f}]")
        else:
            print("No coordinate data available")

        print(f"Colors available: {'Yes' if self.colors is not None else 'No'}")
        print(
            f"Region names available: {'Yes' if self.region_names is not None else 'No'}"
        )

        if self.region_names is not None:
            print(f"Sample regions: {self.region_names[:3]}...")

    def plot_matrix(
        self,
        figsize: Tuple[int, int] = (12, 10),
        show_labels: bool = True,
        cmap: str = "RdBu_r",
        threshold: Optional[float] = None,
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

        # Create heatmap
        im = plt.imshow(matrix_to_plot, cmap=cmap, aspect="equal")
        cbar = plt.colorbar(im, label="Connection Strength")

        # Add threshold info to title
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

    def plot_circular_graph(
        self,
        figsize: Tuple[int, int] = (12, 12),
        threshold: Optional[float] = None,
        node_size_property: str = "strength",
        node_size_scale: float = 1000,
        edge_width_scale: float = 5,
        show_labels: bool = True,
        label_distance: float = 1.1,
        edge_alpha: float = 0.6,
        node_alpha: float = 0.8,
        edge_cmap: str = "plasma",
        layout_seed: Optional[int] = 42,
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
        except ImportError:
            raise ImportError(
                "NetworkX is required for circular graph visualization. "
                "Install with: pip install networkx"
            )

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
        node_colors = self.get_roi_colors()

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
                edge_cmap=plt.cm.get_cmap(edge_cmap),
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
            # Get region names
            region_names = self.get_roi_names()

            # Create labels dictionary
            labels = {i: region_names[i] for i in range(self.n_regions)}

            # Calculate label positions
            label_pos = {}
            for node, (x, y) in pos.items():
                # Move labels slightly outward from nodes
                angle = np.arctan2(y, x)
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
                cmap=plt.cm.get_cmap(edge_cmap),
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

    def visualize_3d(
        self,
        connectivity_threshold: float = 0.1,
        node_size_scale: float = 1.0,
        edge_width_scale: float = 1.0,
        show_edges: bool = True,
        show_labels: bool = False,
        background_color: str = "black",
        window_size: Tuple[int, int] = (1200, 800),
        node_size_property: str = "strength",
        base_node_size: float = 0.5,
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

        Returns:
        --------
        pv.Plotter : PyVista plotter object
        """
        if self.matrix is None:
            raise ValueError("No connectivity matrix available")
        if self.coordinates is None:
            raise ValueError("No coordinates available for 3D visualization")

        # Create plotter
        plotter = pv.Plotter(window_size=window_size)
        plotter.set_background(background_color)

        # Center coordinates around origin
        coords_centered = self.coordinates - np.mean(self.coordinates, axis=0)

        # Calculate node sizes based on selected property
        node_sizes = self._calculate_node_sizes(
            node_size_property, connectivity_threshold, node_size_scale, base_node_size
        )

        # Get colors (use provided or generate defaults)
        colors = self.get_roi_colors()

        # Get region names (use provided or generate defaults)
        region_names = self.get_roi_names()

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

    def save_visualization(self, filename: str, **kwargs) -> None:
        """
        Save a 3D visualization to file.

        Parameters:
        -----------
        filename : str
            Output filename for the visualization
        **kwargs : dict
            Additional arguments passed to visualize_3d()
        """
        plotter = self.visualize_3d(**kwargs)
        plotter.screenshot(filename)
        plotter.close()

    def __repr__(self) -> str:
        """String representation of the Connectome object."""
        if self.matrix is None:
            return f"Connectome(name='{self.name}', no data loaded)"
        return f"Connectome(name='{self.name}', type='{self.connectivity_type}', n_regions={self.n_regions}, density={self.get_density():.3f})"
