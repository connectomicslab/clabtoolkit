import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from pathlib import Path
from typing import Optional, Union, List, Tuple, Literal
import warnings


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
        name: Optional[str] = None,
        matrix: Optional[np.ndarray] = None,
        coordinates: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        region_names: Optional[List[str]] = None,
        region_index: Optional[np.ndarray] = None,
        connectivity_type: str = "structural",
        affine: Optional[np.ndarray] = None,
    ):
        """
        Initialize a Connectome object.

        Parameters:
        -----------
        name : str, optional
            Name for the connectome. If None, will be set when loading data.
        matrix : np.ndarray, optional
            Connectivity matrix (n_regions x n_regions)
        coordinates : np.ndarray, optional
            3D coordinates for each region (n_regions x 3)
        colors : np.ndarray, optional
            RGB color values for each region (n_regions x 3)
        region_names : List[str], optional
            Names/labels for each brain region
        region_index : np.ndarray, optional
            Index codes for each region
        connectivity_type : str, optional
            Type of connectivity (default: 'structural')
        affine : np.ndarray, optional
            4x4 affine transformation matrix
        """
        self.name = name
        self.connectivity_type = connectivity_type

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
            self.set_colors(colors)
        else:
            self.colors = self.get_default_colors()

        # Set region names
        if region_names is not None:
            self.set_region_names(region_names)
        else:
            self.region_names = self.get_default_region_names()

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
        connectome.load_h5_data(filename)
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

    def load_h5_data(self, filename: Union[str, Path]) -> None:
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
                if "gmcoords" in data_group:
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
                    tmp_colors = data_group["gmcolors"][:]

                    tmp_colors = tmp_colors[:].tolist()
                    tmp_colors = [i.decode("utf-8") for i in tmp_colors]

                    self.colors = tmp_colors
                    if len(self.colors) != self._n_regions:
                        warnings.warn("Number of colors doesn't match matrix size")
                    # else:
                    #     # Normalize colors to [0,1] range if needed
                    #     if np.max(self.colors) > 1:
                    #         self.colors = self.colors / 255.0

                elif "colors" in data_group:  # Alternative name
                    self.colors = data_group["colors"][:]
                    if self.colors.shape[0] != self._n_regions:
                        warnings.warn("Number of colors doesn't match matrix size")
                    else:
                        # Normalize colors to [0,1] range if needed
                        if np.max(self.colors) > 1:
                            self.colors = self.colors / 255.0

                # Load region names (optional)
                if "gmregions" in data_group:
                    regions_data = data_group["gmregions"][:]
                    # Handle string datasets
                    if hasattr(regions_data, "dtype") and regions_data.dtype.kind in [
                        "S",
                        "U",
                    ]:
                        self.region_names = [
                            r.decode("utf-8") if isinstance(r, bytes) else str(r)
                            for r in regions_data
                        ]
                    else:
                        self.region_names = [str(r) for r in regions_data]

                    if len(self.region_names) != self._n_regions:
                        warnings.warn(
                            "Number of region names doesn't match matrix size"
                        )
                elif "name" in data_group:  # Alternative name
                    regions_data = data_group["name"][:]
                    if hasattr(regions_data, "dtype") and regions_data.dtype.kind in [
                        "S",
                        "U",
                    ]:
                        self.region_names = [
                            r.decode("utf-8") if isinstance(r, bytes) else str(r)
                            for r in regions_data
                        ]
                    else:
                        self.region_names = [str(r) for r in regions_data]

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
            return self.get_default_region_names()

    def get_roi_colors(self) -> List:
        """
        Get region of interest (ROI) colors. If not available, generate default colors.

        Returns:
        --------
        List : List of ROI colors
        """
        if self.colors is not None:
            return self.colors
        else:
            return self.get_default_colors()

    def get_roi_coordinates(self) -> Optional[np.ndarray]:
        """
        Get region of interest (ROI) coordinates.
        Returns:
        --------
        Optional[np.ndarray] : Array of ROI coordinates or None
        """
        return self.coordinates

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
                grp.create_dataset("gmcoords", data=self.coordinates)

            # Save colors (if available)
            if self.colors is not None:
                grp.create_dataset("gmcolors", data=self.colors)

            # Save region names (if available)
            if self.region_names is not None:
                # Convert to bytes for HDF5 storage
                regions_bytes = np.array(self.region_names, dtype="S")
                grp.create_dataset("gmregions", data=regions_bytes)

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

    def set_colors(self, colors: np.ndarray) -> None:
        """
        Set colors for brain regions.

        Parameters:
        -----------
        colors : np.ndarray
            Array of shape (n_regions, 3) with RGB values [0-1] or [0-255]
        """
        if self.matrix is not None and colors.shape[0] != self.n_regions:
            raise ValueError(
                f"Colors shape {colors.shape} doesn't match expected ({self.n_regions}, 3)"
            )

        colors = colors.copy()
        # Normalize to [0,1] if needed
        if np.max(colors) > 1:
            colors = colors / 255.0
        self.colors = colors

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

    def get_default_colors(self) -> np.ndarray:
        """
        Generate default colors for regions if not available.

        Returns
        -------
        np.ndarray
            Array of hex RGB colors for each region.
        """
        cmap = plt.cm.get_cmap("Set3", self.n_regions)

        # Extract the N colors by sampling the colormap
        colors = np.array([to_hex(cmap(i)) for i in range(self.n_regions)])

        return colors

    def get_default_region_names(self) -> List[str]:
        """
        Generate default region names if not available.

        Returns:
        --------
        List[str] : Default region names
        """
        return [f"Region_{i:03d}" for i in range(self.n_regions)]

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

    def copy(self) -> "Connectome":
        """
        Create a deep copy of the Connectome.

        Returns:
        -------
        Connectome
            Deep copy of the Connectome
        """
        return Connectome(
            name=self.name,
            matrix=self.matrix.copy() if self.matrix is not None else None,
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

    def __repr__(self) -> str:
        """String representation of the Connectome object."""
        if self.matrix is None:
            return f"Connectome(name='{self.name}', no data loaded)"
        return f"Connectome(name='{self.name}', type='{self.connectivity_type}', n_regions={self.n_regions}, density={self.get_density():.3f})"
