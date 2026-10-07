import copy
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import colorstools as cltcol

# Importing local modules
from . import misctools as cltmisc


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############            Section 1: Class and methods work with point clouds             ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
class PointCloud:
    """
    A class to represent and manipulate point clouds.

    Attributes:
        coords (np.ndarray): An array of shape (N, 3) representing the 3D coordinates of points.
        name (str): Name of the point cloud.
        affine (np.ndarray): Affine transformation matrix for the points.
        colortables (Dict): A dictionary to store colortable information for visualization.
            The region names and region colors supplied at construction time are stored
            in its "default" entry.
        point_data (Dict): A dictionary to store scalar data associated with each point.
    """

    def __init__(
        self,
        points: np.ndarray | pd.DataFrame = None,
        affine: np.ndarray = None,
        region_colors: str | list | np.ndarray = "#BFBDBD",
        region_names: str | list[str] = None,
        alpha: float = 1.0,
        name: str = "default",
    ) -> None:
        """
        Initializes the PointCloud object.

        Parameters:
        -----------
            points (np.ndarray or pd.DataFrame, optional):
                An array of shape (N, 3) representing the 3D coordinates of points,
                or a DataFrame with columns ['X', 'Y', 'Z'].

            affine (np.ndarray, optional):
                Affine transformation matrix for the points. Default is identity matrix.

            region_colors (str, list or np.ndarray, optional):
                Color(s) for the regions of the point cloud. It can be a single color
                (hex string or RGB array), or a list/array of colors. If more than one
                color is supplied, the number of colors must match the number of points
                (one region per point). Default is "#BFBDBD".

            region_names (str or list of str, optional):
                Name(s) of the regions. Its length must match the number of colors
                supplied in region_colors. If None, the names are created automatically
                as "region_1", "region_2", ... Default is None.

            alpha (float, optional):
                Opacity value for the point cloud (0-1). Default is 1.0.

            name (str, optional):
                Name of the point cloud. Default is "default".

        Raises:
        -------
            ValueError:
                If alpha is outside the range [0, 1], if the number of region names
                does not match the number of region colors, or if the number of region
                colors is neither 1 nor equal to the number of points.

        Examples:
        --------
        >>> # Single color for the whole point cloud
        >>> pc = PointCloud(points=np.random.rand(100, 3), region_colors="#FF0000")
        >>>
        >>> # One color and one name per point (e.g. region centroids)
        >>> pc = PointCloud(
        ...     points=centroids,
        ...     region_colors=["#FF0000", "#00FF00"],
        ...     region_names=["thalamus", "putamen"],
        ... )
        """

        # Initialize attributes
        self.name = name
        self.coords = None
        self.colortables: dict[str, dict] = {}
        self.point_data: dict[str, np.ndarray] = {}

        # Set affine (always initialize, even if points is None)
        self.affine = affine if affine is not None else np.eye(4)

        # Validate alpha value
        if isinstance(alpha, int):
            alpha = float(alpha)

        # If the alpha is not in the range [0, 1], raise an error
        if not (0 <= alpha <= 1):
            raise ValueError(f"Alpha value must be in the range [0, 1], got {alpha}")

        # Handle color input. Multiple colors mean multiple regions.
        region_colors = (
            cltcol.harmonize_colors(region_colors, output_format="rgb") / 255
        )
        n_regions = region_colors.shape[0]

        # Handle the region names. They are created automatically if not supplied.
        if region_names is None:
            region_names = [f"region_{i + 1}" for i in range(n_regions)]

        elif isinstance(region_names, str):
            region_names = [region_names]

        else:
            region_names = [str(reg_name) for reg_name in region_names]

        if len(region_names) != n_regions:
            raise ValueError(
                f"The number of region names ({len(region_names)}) must match the "
                f"number of region colors ({n_regions})"
            )

        tmp_ctable = cltcol.colors_to_table(
            colors=region_colors,
            alpha_values=alpha,
            values=np.arange(1, n_regions + 1),
        )
        tmp_ctable[:, :3] = tmp_ctable[:, :3] / 255  # Ensure colors are between 0 and 1

        # Store parcellation information in organized structure
        self.colortables["default"] = {
            "names": region_names,
            "color_table": tmp_ctable,
            "lookup_table": None,  # Will be populated by _create_parcellation_colortable if needed
        }

        # Validate and process input points
        if points is not None:
            if isinstance(points, pd.DataFrame):
                if not all(col in points.columns for col in ["X", "Y", "Z"]):
                    raise ValueError("DataFrame must contain 'X', 'Y', and 'Z' columns")
                self.coords = points[["X", "Y", "Z"]].to_numpy()

            elif isinstance(points, np.ndarray):
                if points.ndim != 2 or points.shape[1] != 3:
                    # If only X and Y are provided (2D points), add Z=0
                    if points.ndim == 2 and points.shape[1] == 2:
                        points = np.concatenate(
                            [points, np.zeros((points.shape[0], 1))], axis=1
                        )
                    else:
                        raise ValueError(
                            f"Points array must have shape (N, 3) or (N, 2), got shape {points.shape}"
                        )

                self.coords = points
            else:
                raise ValueError("points must be a numpy array or pandas DataFrame")

            # Initialize default point data. A single region is assigned to all the
            # points, otherwise each point belongs to its own region.
            n_points = len(self.coords)
            if n_regions == 1:
                default = np.full(n_points, int(tmp_ctable[0, 4]), dtype=int)

            elif n_regions == n_points:
                default = np.array(tmp_ctable[:, 4], dtype=int)

            else:
                raise ValueError(
                    f"The number of region colors ({n_regions}) must be 1 or equal to "
                    f"the number of points ({n_points})"
                )

            self.point_data["default"] = default

    ###############################################################################################
    def __len__(self) -> int:
        """
        Returns the number of points in the point cloud.

        Returns:
        -------
            int: Number of points.
        """
        return 0 if self.coords is None else len(self.coords)

    ###############################################################################################
    def __repr__(self) -> str:
        """
        String representation of the PointCloud object.

        Returns:
        -------
            str: Description of the point cloud.
        """
        n_points = len(self)
        n_attributes = len(self.point_data)
        return (
            f"PointCloud(name='{self.name}', "
            f"n_points={n_points}, "
            f"n_attributes={n_attributes})"
        )

    ###############################################################################################
    def __add__(self, other: "PointCloud") -> "PointCloud":
        """
        Concatenate two point clouds using the + operator.

        Parameters:
        -----------
            other (PointCloud):
                The PointCloud object to concatenate.

        Returns:
        -------
            PointCloud:
                A new PointCloud containing the concatenated data.

        Examples:
        --------
        >>> pc1 = PointCloud(points=np.random.rand(100, 3))
        >>> pc2 = PointCloud(points=np.random.rand(50, 3))
        >>> pc3 = pc1 + pc2  # Creates new point cloud with 150 points
        """
        return self.append(other, inplace=False)

    ###############################################################################################
    def copy(self) -> "PointCloud":
        """
        Creates a deep copy of the PointCloud object.

        Returns:
        -------
            PointCloud: A new PointCloud instance with copied data.
        """
        return copy.deepcopy(self)

    ###############################################################################################
    def add_point_data(
        self,
        data: np.ndarray,
        name: str,
        dtype: type = None,
    ) -> None:
        """
        Adds scalar data associated with each point.

        Parameters:
        -----------
            data (np.ndarray):
                Array of data values, one per point.

            name (str):
                Name for this data attribute.

            dtype (type, optional):
                Data type to cast the array to. If None, keeps original dtype.

        Raises:
        -------
            ValueError:
                If data length doesn't match number of points.
        """
        if self.coords is None:
            raise ValueError("Cannot add point data to an empty point cloud")

        if len(data) != len(self.coords):
            raise ValueError(
                f"Data length ({len(data)}) must match number of points ({len(self.coords)})"
            )

        if dtype is not None:
            data = np.array(data, dtype=dtype)
        else:
            data = np.array(data)

        self.point_data[name] = data

    ###############################################################################################
    def apply_affine(
        self, affine: np.ndarray, inverse: bool = False, inplace: bool = True
    ) -> Optional["PointCloud"]:
        """
        Applies an affine transformation to the point coordinates.

        Parameters:
        -----------
            affine (np.ndarray):
                A 4x4 affine transformation matrix.

            inverse (bool, default False):
                If True, applies the inverse of the affine transformation.

            inplace (bool):
                If True, modifies the current object. If False, returns a new object.
                Default is True.

        Returns:
        -------
            PointCloud or None:
                If inplace=False, returns a new transformed PointCloud.
                If inplace=True, returns None.

        Raises:
        -------
            ValueError: If the provided affine is not a 4x4 matrix.

        Examples:
        ---------
        >>> pc = PointCloud(coords)
        >>> pc.apply_affine(affine_matrix)

        >>> # Keeping the original point cloud untouched
        >>> moved = pc.apply_affine(affine_matrix, inplace=False)
        >>> print(moved.coords[:5])  # Transformed coordinates
        """
        if affine.shape != (4, 4):
            raise ValueError(
                f"Invalid affine transformation provided. It must be a 4x4 matrix, got shape {affine.shape}."
            )

        if inverse:
            affine = np.linalg.inv(affine)

        if self.coords is None:
            if inplace:
                return None
            else:
                return self.copy()

        # Create homogeneous coordinates
        ones = np.ones((len(self.coords), 1))
        homogeneous = np.hstack([self.coords, ones])

        # Apply transformation
        transformed = (affine @ homogeneous.T).T

        # Convert back to 3D coordinates
        new_coords = transformed[:, :3]

        if inplace:
            self.coords = new_coords
            self.affine = affine @ self.affine
            return None
        else:
            new_pc = self.copy()
            new_pc.coords = new_coords
            new_pc.affine = affine @ new_pc.affine
            return new_pc

    ###############################################################################################
    def filter_by_bounds(
        self,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        z_range: tuple[float, float] | None = None,
        inplace: bool = True,
    ) -> Optional["PointCloud"]:
        """
        Filters points based on spatial bounds.

        Parameters:
        -----------
            x_range (tuple, optional):
                (min, max) range for X coordinates.

            y_range (tuple, optional):
                (min, max) range for Y coordinates.

            z_range (tuple, optional):
                (min, max) range for Z coordinates.

            inplace (bool):
                If True, modifies the current object. If False, returns a new object.
                Default is True.

        Returns:
        -------
            PointCloud or None:
                If inplace=False, returns a new filtered PointCloud.
                If inplace=True, returns None.
        """
        if self.coords is None:
            if inplace:
                return None
            else:
                return self.copy()

        # Create mask for points within bounds
        mask = np.ones(len(self.coords), dtype=bool)

        if x_range is not None:
            mask &= (self.coords[:, 0] >= x_range[0]) & (
                self.coords[:, 0] <= x_range[1]
            )

        if y_range is not None:
            mask &= (self.coords[:, 1] >= y_range[0]) & (
                self.coords[:, 1] <= y_range[1]
            )

        if z_range is not None:
            mask &= (self.coords[:, 2] >= z_range[0]) & (
                self.coords[:, 2] <= z_range[1]
            )

        if inplace:
            self.coords = self.coords[mask]
            for key in self.point_data:
                self.point_data[key] = self.point_data[key][mask]
            return None
        else:
            new_pc = self.copy()
            new_pc.coords = new_pc.coords[mask]
            for key in new_pc.point_data:
                new_pc.point_data[key] = new_pc.point_data[key][mask]
            return new_pc

    ###############################################################################################
    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """
        Computes the bounding box of the point cloud.

        Returns:
        -------
            dict:
                Dictionary with 'x', 'y', 'z' keys, each containing (min, max) tuples.
        """
        if self.coords is None or len(self.coords) == 0:
            return {"x": (0, 0), "y": (0, 0), "z": (0, 0)}

        return {
            "x": (self.coords[:, 0].min(), self.coords[:, 0].max()),
            "y": (self.coords[:, 1].min(), self.coords[:, 1].max()),
            "z": (self.coords[:, 2].min(), self.coords[:, 2].max()),
        }

    ###############################################################################################
    def get_centroid(self) -> np.ndarray:
        """
        Computes the centroid (geometric center) of the point cloud.

        Returns:
        -------
            np.ndarray:
                Array of shape (3,) with the centroid coordinates [x, y, z].
        """
        if self.coords is None or len(self.coords) == 0:
            return np.array([0, 0, 0])

        return self.coords.mean(axis=0)

    ###############################################################################################
    def filter(
        self,
        condition: str,
        inplace: bool = True,
        **kwargs,
    ) -> Optional["PointCloud"]:
        """
        Filters points based on coordinate values or point_data attributes.

        Parameters:
        -----------
            condition (str):
                Condition string to filter points. Supported formats:
                - 'X > value': Keep points where X coordinate is greater than value.
                - 'Y < value': Keep points where Y coordinate is less than value.
                - 'Z >= value': Keep points where Z coordinate is greater than or equal to value.
                - 'intensity > value': Keep points where intensity attribute is greater than value.
                - 'min_value <= attribute <= max_value': Keep points within range.

                Available attributes: X, Y, Z (coordinates) or any key in point_data.

            inplace (bool):
                If True, modifies the current object. If False, returns a new filtered object.
                Default is True.

            **kwargs:
                Additional keyword arguments passed to the condition parser.

        Returns:
        -------
            PointCloud or None:
                If inplace=False, returns a new filtered PointCloud.
                If inplace=True, returns None.

        Raises:
        -------
            ValueError:
                If the specified attribute doesn't exist in coordinates or point_data,
                or if no points match the condition.

        Examples:
        --------
        >>> pc = PointCloud(points=np.random.rand(1000, 3) * 100)
        >>> pc.add_point_data(np.random.rand(1000), name="intensity")

        >>> # Filter by coordinate
        >>> pc.filter('X > 50')  # Keep points with X > 50

        >>> # Filter by point_data attribute
        >>> pc.filter('intensity > 0.5')  # Keep high intensity points

        >>> # Filter by range
        >>> pc.filter('20 <= Z <= 80')  # Keep points in Z range

        >>> # Create new filtered point cloud
        >>> pc_filtered = pc.filter('Y < 30', inplace=False)
        """
        if self.coords is None:
            raise ValueError("Cannot filter an empty point cloud")

        # Parse the condition to extract the attribute name
        cond_parts = cltmisc.parse_condition(condition)
        attr_name = cond_parts[0]

        # Check if attribute is a coordinate (X, Y, Z) or in point_data
        if attr_name in ["X", "Y", "Z"]:
            # Use coordinate values
            coord_idx = {"X": 0, "Y": 1, "Z": 2}[attr_name]
            attr_values = self.coords[:, coord_idx].reshape(-1, 1)
            kwargs.update({attr_name: attr_values})

        elif attr_name in self.point_data:
            # Use point_data values
            attr_values = self.point_data[attr_name].reshape(-1, 1)
            kwargs.update({attr_name: attr_values})

        else:
            # Attribute not found
            available_attrs = ["X", "Y", "Z"] + list(self.point_data.keys())
            raise ValueError(
                f"Attribute '{attr_name}' not found. "
                f"Available attributes: {available_attrs}"
            )

        # Get indices of points matching the condition
        indices = cltmisc.get_indices_by_condition(condition, **kwargs)

        if len(indices) == 0:
            raise ValueError(f"No points match the condition: {condition}")

        print(
            f"Filtered {len(indices)} points matching condition: {condition} "
            f"({len(indices)}/{len(self.coords)} points = {100*len(indices)/len(self.coords):.1f}%)"
        )

        # Create filtered data
        filtered_coords = self.coords[indices]
        filtered_point_data = {}

        for key in self.point_data.keys():
            filtered_point_data[key] = self.point_data[key][indices]

        # Handle colortables - update to reflect filtered data
        filtered_colortables = {}
        for map_name, ctable_info in self.colortables.items():
            if map_name in self.point_data:
                # Get unique values in the filtered data
                unique_values = np.unique(filtered_point_data[map_name])

                # Filter the colortable to only include used colors
                color_table = ctable_info["color_table"]
                names = ctable_info["names"]

                # Create mapping from index to position in color_table
                # color_table has shape (n_colors, 5) where last column is the index
                filtered_color_table = []
                filtered_names = []

                for i, row in enumerate(color_table):
                    idx = int(row[4])
                    if idx in unique_values:
                        filtered_color_table.append(row)
                        if i < len(names):
                            filtered_names.append(names[i])
                        else:
                            filtered_names.append(f"region_{idx}")

                if len(filtered_color_table) > 0:
                    filtered_colortables[map_name] = {
                        "names": filtered_names,
                        "color_table": np.array(filtered_color_table),
                        "lookup_table": ctable_info.get("lookup_table", None),
                    }
            else:
                # If colortable doesn't correspond to filtered point_data, keep as is
                filtered_colortables[map_name] = copy.deepcopy(ctable_info)

        # Apply changes
        if inplace:
            self.coords = filtered_coords
            self.point_data = filtered_point_data
            self.colortables = filtered_colortables
            return None
        else:
            # Create new PointCloud with filtered data
            new_pc = PointCloud(
                points=filtered_coords,
                affine=self.affine.copy(),
                name=f"{self.name}_filtered",
            )
            new_pc.point_data = filtered_point_data
            new_pc.colortables = filtered_colortables
            return new_pc

    ###############################################################################################
    def to_dataframe(
        self,
        include_data: bool = True,
        include_colortable: bool = False,
        colortable_name: str = "default",
    ) -> pd.DataFrame:
        """
        Converts the point cloud to a pandas DataFrame.

        Parameters:
        -----------
            include_data (bool):
                If True, includes all point_data attributes as columns.
                Default is True.

            include_colortable (bool):
                If True, includes index, name, and color columns from the colortable.
                Default is False.

            colortable_name (str):
                Name of the colortable to use for color information.
                Default is "default".

        Returns:
        -------
            pd.DataFrame:
                DataFrame with columns in order: index, name, color (if include_colortable=True),
                X, Y, Z, and optionally additional data columns.

        Raises:
        -------
            ValueError:
                If include_colortable is True but the specified colortable doesn't exist
                or the corresponding point_data key is missing.

        Examples:
        --------
        >>> pc = PointCloud(points=np.random.rand(100, 3))
        >>> df = pc.to_dataframe()  # Basic: X, Y, Z columns

        >>> df = pc.to_dataframe(include_colortable=True)
        >>> # Returns: index, name, color, X, Y, Z columns
        """
        if self.coords is None:
            base_cols = ["X", "Y", "Z"]
            if include_colortable:
                base_cols = ["index", "name", "color"] + base_cols
            return pd.DataFrame(columns=base_cols)

        # Helper function to convert RGB to hex

        # Initialize DataFrame with coordinates
        df = pd.DataFrame(self.coords, columns=["X", "Y", "Z"])

        # Add colortable information if requested
        if include_colortable:
            if colortable_name not in self.colortables:
                raise ValueError(
                    f"Colortable '{colortable_name}' not found. "
                    f"Available colortables: {list(self.colortables.keys())}"
                )

            if colortable_name not in self.point_data:
                raise ValueError(
                    f"Point data for '{colortable_name}' not found. "
                    f"Cannot map points to colortable. "
                    f"Available point_data keys: {list(self.point_data.keys())}"
                )

            # Get colortable information
            ctable = self.colortables[colortable_name]
            color_table = ctable["color_table"]
            names = ctable["names"]

            # Converting RGB to hex
            colors = cltcol.harmonize_colors(color_table[:, :3], output_format="hex")

            # Get point indices/values
            point_indices = self.point_data[colortable_name]

            # Create lookup dictionaries
            # color_table has shape (n_colors, 5) where columns are [r, g, b, alpha, index]
            index_to_color = {}
            index_to_name = {}

            for i, row in enumerate(color_table):
                idx = int(row[4])  # The index value
                index_to_color[idx] = colors[i]  # hexadecimal color
                if i < len(names):
                    index_to_name[idx] = names[i]
                else:
                    index_to_name[idx] = f"auto-roi-{idx:06d}"

            # Map each point to its color and name
            colors = []
            point_names = []

            for point_idx in point_indices:
                point_idx = int(point_idx)
                if point_idx in index_to_color:
                    colors.append(index_to_color[point_idx])
                    point_names.append(index_to_name[point_idx])
                else:
                    # Handle missing indices
                    colors.append("#000000")  # Black for undefined
                    point_names.append(f"auto-roi-{point_idx:06d}")

            # Insert colortable columns at the beginning
            df.insert(0, "color", colors)
            df.insert(0, "name", point_names)
            df.insert(0, "index", point_indices)

        # Add additional point data if requested
        if include_data:
            for key, data in self.point_data.items():
                # Skip the key we already used for colortable mapping
                if include_colortable and key == colortable_name:
                    continue
                df[key] = data

        return df

    ###############################################################################################
    def save(
        self,
        filename: str | Path,
        format: str = "npy",
        include_colortable: bool = False,
        colortable_name: str = "default",
    ) -> None:
        """
        Saves the point cloud to a file.

        Parameters:
        -----------
            filename (str or Path):
                Output filename.

            format (str):
                File format. Options: 'npy', 'csv', 'txt'.
                Default is 'npy'.

            include_colortable (bool):
                If True and format is 'csv' or 'txt', includes colortable information
                (index, name, color columns). Only applies to text formats.
                Default is False.

            colortable_name (str):
                Name of the colortable to use when include_colortable=True.
                Default is "default".

        Raises:
        -------
            ValueError:
                If format is not supported or if the point cloud is empty.
        """
        if self.coords is None:
            raise ValueError("Cannot save an empty point cloud")

        filename = Path(filename)

        if format == "npy":
            # Save as numpy archive with all data
            save_dict = {
                "coords": self.coords,
                "affine": self.affine,
                "name": self.name,
            }
            save_dict.update(self.point_data)
            # Writing through a file handle keeps the filename as given
            # (np.savez would append a .npz extension to it)
            with open(filename, "wb") as f:
                np.savez(f, **save_dict)

        elif format in ["csv", "txt"]:
            df = self.to_dataframe(
                include_data=True,
                include_colortable=include_colortable,
                colortable_name=colortable_name,
            )
            sep = "," if format == "csv" else "\t"
            df.to_csv(filename, sep=sep, index=False)

        else:
            raise ValueError(f"Unsupported format: {format}")

    ###############################################################################################
    @classmethod
    def load(cls, filename: str | Path, format: str = "npy") -> "PointCloud":
        """
        Loads a point cloud from a file.

        Parameters:
        -----------
            filename (str or Path):
                Input filename.

            format (str):
                File format. Options: 'npy', 'csv', 'txt'.
                Default is 'npy'.

        Returns:
        -------
            PointCloud:
                Loaded PointCloud object.

        Raises:
        -------
            ValueError:
                If format is not supported.

            FileNotFoundError:
                If file does not exist.
        """
        filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        if format == "npy":
            data = np.load(filename, allow_pickle=True)

            coords = data["coords"]
            affine = data["affine"] if "affine" in data else None
            name = str(data["name"]) if "name" in data else "default"

            pc = cls(points=coords, affine=affine, name=name)

            # Load additional point data
            for key in data.keys():
                if key not in ["coords", "affine", "name"]:
                    pc.point_data[key] = data[key]

            return pc

        elif format in ["csv", "txt"]:
            sep = "," if format == "csv" else "\t"
            df = pd.read_csv(filename, sep=sep)

            if not all(col in df.columns for col in ["X", "Y", "Z"]):
                raise ValueError("File must contain X, Y, Z columns")

            pc = cls(points=df)

            # Add any additional columns as point data
            for col in df.columns:
                if col not in ["X", "Y", "Z"]:
                    pc.add_point_data(df[col].values, name=col)

            return pc

        else:
            raise ValueError(f"Unsupported format: {format}")

    ###############################################################################################
    def load_colortable(
        self,
        lut_file: str | Path,
        map_name: str = "default",
        opacity: float | int | np.ndarray = 1.0,
        lut_type: str = "lut",
    ) -> None:
        """
        Loads a colortable from a file and associates it with a specified map name.

        Parameters:
        -----------
            lut_file (str or Path):
                Path to the colortable file.

            map_name (str):
                Name of the map to associate with the loaded colortable.
                Default is "default".

            opacity (float, int, or np.ndarray):
                Opacity value(s) for the colortable. Can be a single value
                (applied to all entries) or an array of values. Default is 1.0.

            lut_type (str):
                Type of lookup table to load. Currently only "lut" is supported.
                Default is "lut".

        Returns:
        -------
            None

        Raises:
        -------
            FileNotFoundError:
                If the specified colortable file does not exist.

            ValueError:
                If the colortable does not cover all IDs in the data or if
                lut_type is not supported.
        """
        if isinstance(lut_file, Path):
            lut_file = str(lut_file)

        if not os.path.isfile(lut_file):
            raise FileNotFoundError(
                f"The specified colortable file does not exist: {lut_file}"
            )

        # Load the colortable using the utility function
        if lut_type == "lut":
            lut_dict = cltcol.ColorTableLoader.read_luttable(lut_file)
        else:
            raise ValueError(
                f"Unsupported lut_type: '{lut_type}'. Currently only 'lut' is supported."
            )

        colors = lut_dict["color"]

        if map_name in self.point_data:
            values = np.unique(self.point_data[map_name])
            if len(values) != len(colors):
                raise ValueError(
                    f"Colortable in {lut_file} does not cover all IDs in point_data for map '{map_name}'."
                )
            color_table = cltcol.colors_to_table(colors=colors, values=values)
        else:
            color_table = cltcol.colors_to_table(colors=colors)

        if isinstance(opacity, (int, float)):
            # opacity is a scalar, apply to all entries
            opacity_array = np.full(color_table.shape[0], opacity)
        elif isinstance(opacity, np.ndarray):
            if len(opacity) != color_table.shape[0]:
                opacity_array = np.full(color_table.shape[0], opacity[0])
            else:
                opacity_array = np.array(opacity)
        else:
            opacity_array = np.full(color_table.shape[0], opacity)

        color_table[:, :3] = (
            color_table[:, :3] / 255
        )  # Ensure colors are between 0 and 1

        color_table[:, 3] = opacity_array  # Set opacity

        # Store parcellation information in organized structure
        self.colortables[map_name] = {
            "names": lut_dict["name"],
            "color_table": color_table,
            "lookup_table": None,
        }

    ###############################################################################################
    def append(
        self,
        other: "PointCloud",
        inplace: bool = True,
        fill_value: float = np.nan,
        handle_colortable_conflicts: str = "warn",
    ) -> Optional["PointCloud"]:
        """
        Appends another PointCloud to this one by concatenating coordinates and data.

        Parameters:
        -----------
            other (PointCloud):
                The PointCloud object to append.

            inplace (bool):
                If True, modifies the current object. If False, returns a new object.
                Default is True.

            fill_value (float):
                Value to use when filling missing point_data keys. Default is np.nan.

            handle_colortable_conflicts (str):
                How to handle colortable name conflicts. Options:
                - 'warn': Keep first colortable and warn (default)
                - 'overwrite': Use the new colortable
                - 'rename': Rename the new colortable as 'name_2'
                - 'skip': Skip the new colortable silently

        Returns:
        -------
            PointCloud or None:
                If inplace=False, returns a new concatenated PointCloud.
                If inplace=True, returns None.

        Raises:
        -------
            ValueError:
                If other is not a PointCloud object or if both point clouds are empty.

        Notes:
        -----
            - If point_data keys don't match, missing values are filled with fill_value
            - Affine matrices are compared; a warning is issued if they differ
            - The name of the resulting point cloud is kept from self

        Examples:
        --------
        >>> pc1 = PointCloud(points=np.random.rand(100, 3), name="cloud1")
        >>> pc2 = PointCloud(points=np.random.rand(50, 3), name="cloud2")
        >>> pc1.append(pc2)  # pc1 now has 150 points

        >>> # Or create a new combined cloud
        >>> pc3 = pc1.append(pc2, inplace=False)
        """
        import warnings

        # Validate input
        if not isinstance(other, PointCloud):
            raise ValueError("Can only append another PointCloud object")

        # Handle empty point clouds
        if self.coords is None and other.coords is None:
            raise ValueError("Cannot append two empty point clouds")

        if self.coords is None:
            if inplace:
                # Copy all data from other to self
                self.coords = other.coords.copy()
                self.affine = other.affine.copy()
                self.point_data = copy.deepcopy(other.point_data)
                self.colortables = copy.deepcopy(other.colortables)
                return None
            else:
                return other.copy()

        if other.coords is None:
            if inplace:
                return None
            else:
                return self.copy()

        # Check affine compatibility
        if not np.allclose(self.affine, other.affine):
            warnings.warn(
                "Affine matrices differ between point clouds. "
                "Using affine from the first point cloud.",
                UserWarning,
                stacklevel=2,
            )

        # Start with a copy if not inplace
        if inplace:
            target = self
        else:
            target = self.copy()

        # Concatenate coordinates
        target.coords = np.vstack([target.coords, other.coords])

        # Handle point_data - union of all keys
        all_keys = set(target.point_data.keys()) | set(other.point_data.keys())

        for key in all_keys:
            # Get data from both point clouds, or create fill arrays
            if key in target.point_data:
                data_self = target.point_data[key]
            else:
                data_self = np.full(len(self.coords), fill_value)

            if key in other.point_data:
                data_other = other.point_data[key]
            else:
                data_other = np.full(len(other.coords), fill_value)

            # Concatenate
            target.point_data[key] = np.concatenate([data_self, data_other])

        # Handle colortables
        for key, ctable_data in other.colortables.items():
            if key in target.colortables:
                if handle_colortable_conflicts == "warn":
                    warnings.warn(
                        f"Colortable '{key}' exists in both point clouds. "
                        f"Keeping colortable from first point cloud.",
                        UserWarning,
                        stacklevel=2,
                    )
                elif handle_colortable_conflicts == "overwrite":
                    target.colortables[key] = copy.deepcopy(ctable_data)
                elif handle_colortable_conflicts == "rename":
                    # Find a unique name
                    new_key = f"{key}_2"
                    counter = 2
                    while new_key in target.colortables:
                        counter += 1
                        new_key = f"{key}_{counter}"
                    target.colortables[new_key] = copy.deepcopy(ctable_data)
                    warnings.warn(
                        f"Colortable '{key}' renamed to '{new_key}' to avoid conflict.",
                        UserWarning,
                        stacklevel=2,
                    )
                elif handle_colortable_conflicts == "skip":
                    pass  # Do nothing, skip silently
                else:
                    raise ValueError(
                        f"Unknown handle_colortable_conflicts option: '{handle_colortable_conflicts}'"
                    )
            else:
                # No conflict, just add it
                target.colortables[key] = copy.deepcopy(ctable_data)

        if inplace:
            return None
        else:
            return target

    ###############################################################################################
    def get_info(self) -> None:
        """
        Display comprehensive information about the point cloud.

        Provides a formatted overview of the point cloud including point count,
        spatial transformations, bounding box, scalar data properties, and available
        colortables. Useful for quick inspection and validation of point cloud data.

        The method displays:
            - Basic point cloud identification and point count
            - Affine transformation matrix for spatial mapping
            - Bounding box information (spatial extent)
            - Centroid coordinates
            - Scalar data per point (with min/max statistics)
            - Available colortables for visualization

        Returns
        -------
        None
            Prints formatted information to stdout.

        Notes
        -----
        This method performs no modifications to the point cloud data. It only
        displays information for inspection purposes.

        Examples
        --------
        >>> pc = PointCloud(points=np.random.rand(10000, 3))
        >>> pc.add_point_data(np.random.rand(10000), name="intensity")
        >>> pc.get_info()
        ╔════════════════════════════════════════════════════════════════╗
        ║                    POINT CLOUD EXPLORATION                     ║
        ╠════════════════════════════════════════════════════════════════╣
        ║ Name: default                                                  ║
        ║ Points: 10,000                                                 ║
        ╠════════════════════════════════════════════════════════════════╣
        ║ AFFINE TRANSFORMATION MATRIX                                   ║
        ║   [[ 1.00   0.00   0.00   0.00]                                ║
        ║    [ 0.00   1.00   0.00   0.00]                                ║
        ║    [ 0.00   0.00   1.00   0.00]                                ║
        ║    [ 0.00   0.00   0.00   1.00]]                               ║
        ╠════════════════════════════════════════════════════════════════╣
        ║ SPATIAL INFORMATION                                            ║
        ║   Bounding Box:                                                ║
        ║     X: 0.0012 to 0.9998  (range: 0.9986)                       ║
        ║     Y: 0.0034 to 0.9987  (range: 0.9953)                       ║
        ║     Z: 0.0009 to 0.9995  (range: 0.9986)                       ║
        ║   Centroid: [0.5023, 0.4989, 0.5012]                           ║
        ╠════════════════════════════════════════════════════════════════╣
        ║ SCALAR DATA PER POINT (2 maps)                                 ║
        ║   default     Min: 1.0000    Max: 1.0000                       ║
        ║   intensity   Min: 0.0001    Max: 0.9999                       ║
        ╠════════════════════════════════════════════════════════════════╣
        ║ COLORTABLES (1 available)                                      ║
        ║   • default                                                    ║
        ╚════════════════════════════════════════════════════════════════╝
        """
        import numpy as np

        # Helper function for formatting numbers with thousands separator
        def format_number(num):
            if isinstance(num, (int, np.integer)):
                return f"{num:,}"
            return str(num)

        # Helper function to print a properly padded line
        def print_line(content, width=64):
            # Ensure content is exactly width characters, then add borders
            padded = content.ljust(width)
            print(f"║{padded}║")

        # Print header
        width = 64  # Content width (excluding borders)
        print("╔" + "═" * width + "╗")
        print_line("POINT CLOUD EXPLORATION".center(width), width)
        print("╠" + "═" * width + "╣")

        # Basic information
        print_line(f" Name: {self.name}", width)
        point_count = len(self) if self.coords is not None else 0
        print_line(f" Points: {format_number(point_count)}", width)

        # Affine transformation matrix
        print("╠" + "═" * width + "╣")
        if self.affine is not None:
            print_line(" AFFINE TRANSFORMATION MATRIX", width)
            affine_lines = str(self.affine).split("\n")
            for line in affine_lines:
                print_line(f"   {line}", width)
        else:
            print_line(" Affine transformation matrix: Not available", width)

        # Spatial information (bounding box and centroid)
        print("╠" + "═" * width + "╣")
        print_line(" SPATIAL INFORMATION", width)

        if self.coords is not None and len(self.coords) > 0:
            bounds = self.get_bounds()
            print_line("   Bounding Box:", width)

            for axis in ["x", "y", "z"]:
                min_val, max_val = bounds[axis]
                range_val = max_val - min_val
                axis_upper = axis.upper()
                print_line(
                    f"     {axis_upper}: {min_val:.4f} to {max_val:.4f}  (range: {range_val:.4f})",
                    width,
                )

            centroid = self.get_centroid()
            centroid_str = f"[{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]"
            print_line(f"   Centroid: {centroid_str}", width)
        else:
            print_line("   Not available (empty point cloud)", width)

        # Scalar data per point
        print("╠" + "═" * width + "╣")
        if hasattr(self, "point_data") and self.point_data:
            count = len(self.point_data)
            print_line(
                f" SCALAR DATA PER POINT ({count} {'map' if count == 1 else 'maps'})",
                width,
            )

            for map_name, values in self.point_data.items():
                if len(values) > 0:
                    min_val = np.nanmin(values)
                    max_val = np.nanmax(values)
                    print_line(
                        f"   {map_name:<12}  Min: {min_val:>8.4f}    Max: {max_val:>8.4f}",
                        width,
                    )
                else:
                    print_line(
                        f"   {map_name:<12}  (empty)",
                        width,
                    )
        else:
            print_line(" SCALAR DATA PER POINT (0 maps)", width)
            print_line("   No scalar data available", width)

        # Colortables
        print("╠" + "═" * width + "╣")
        if self.colortables:
            count = len(self.colortables)
            print_line(f" COLORTABLES ({count} available)", width)
            for name in self.colortables.keys():
                print_line(f"   • {name}", width)
        else:
            print_line(" COLORTABLES (0 available)", width)
            print_line("   No colortables available", width)

        # Footer
        print("╚" + "═" * width + "╝")

    ###################################################################################################
    def get_pointwise_colors(
        self,
        map_name: str = "default",
        colormap: str = "viridis",
        vmin: np.float64 = None,
        vmax: np.float64 = None,
        range_min: np.float64 = None,
        range_max: np.float64 = None,
        range_color: tuple = (128, 128, 128, 255),
    ) -> np.ndarray:
        """
        Compute streamlines colors for visualization based on the specified overlay.

        This method processes the overlay data and creates appropiate point colors
        for visualization, handling both scalar data (with colormaps) and
        categorical data (with discrete color tables).

        Parameters
        ----------
        map_name : str, optional
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

        range_min : np.float64, optional
            Minimum threshold for the overlay values. Values below this will be colored with range_color.
            If None, no minimum threshold is applied.

        range_max : np.float64, optional
            Maximum threshold for the overlay values. Values above this will be colored with range_color.
            If None, no maximum threshold is applied.

        range_color : List[int, int, int, int], optional
            RGBA color to use for values outside the specified range (range_min, range_max).
            Default is gray [128, 128, 128].

        Returns
        -------
        point_colors : ArraySequence
            Array of RGBA colors for each point in the tractogram.

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
        >>> tractogram.get_vertexwise_colors(map_name="aparc")
        >>>
        >>> # Prepare colors for scalar data with custom colormap
        >>> tractogram.get_vertexwise_colors(map_name="thickness", colormap="hot")
        >>>
        >>> # Prepare colors for the tractogram overlay
        >>> tractogram.get_vertexwise_colors()
        """

        # Get the list of overlays
        maps_list = self.list_maps()

        if map_name not in maps_list:
            raise ValueError(
                f"Overlay '{map_name}' not found. Available overlays: {', '.join(maps_list)}"
            )

        # Getting the values of the overlay
        data = self.point_data[map_name]

        # if colortables is an attribute of the class, use it
        if hasattr(self, "colortables"):
            dict_ctables = self.colortables

            # Check if the overlay is on the colortables
            if map_name in dict_ctables.keys():
                # Use the colortable associated with the parcellation

                point_colors = cltcol.get_colors_from_colortable(
                    data, self.colortables[map_name]["color_table"]
                )
            else:
                # Use the colormap for scalar data
                point_colors = cltcol.values2colors(
                    data,
                    cmap=colormap,
                    output_format="rgb",
                    vmin=vmin,
                    vmax=vmax,
                    range_min=range_min,
                    range_max=range_max,
                    range_color=range_color,
                )
        else:
            point_colors = cltcol.values2colors(
                data,
                cmap=colormap,
                output_format="rgb",
                vmin=vmin,
                vmax=vmax,
                range_min=range_min,
                range_max=range_max,
                range_color=range_color,
            )

        return point_colors

    ###############################################################################################
    def list_maps(self) -> list[str]:
        """
        Lists all available scalar maps in the point cloud.

        Returns:
        --------
            maps_per_point (set or None):
                Set of scalar map names stored per point. None if no maps are available.

        Examples:
        ---------
        >>> points = PointCloud(points)
        >>> maps = points.list_maps()
        >>> print("Available maps per point:", maps)

        """
        maps_per_point = []

        if hasattr(self, "point_data"):
            if self.point_data:
                maps_per_point = maps_per_point + list(self.point_data.keys())
            else:
                maps_per_point = None

        return maps_per_point

    ###############################################################################################
    def plot(
        self,
        maps: str | list[str] = "default",
        cmap: str = "viridis",
        vmin: np.float64 = None,
        vmax: np.float64 = None,
        range_min: np.float64 = None,
        range_max: np.float64 = None,
        range_color: tuple = (128, 128, 128, 255),
        views: str | list[str] = None,
        hemi: str = "lh",
        radius: float = 10.0,
        as_spheres: bool = True,
        use_opacity: bool = True,
        notebook: bool = False,
        show_colorbar: bool = False,
        colorbar_title: str = None,
        colorbar_position: str = "bottom",
        save_path: str = None,
        config: str | Path | dict = None,
    ):
        """
        Plot the point cloud with specified overlay and visualization parameters.

        Renders the tractogram with optional overlays using PyVista, supporting
        multiple camera views, custom colormaps, and interactive or static output.
        Handles both categorical parcellation data and continuous scalar overlays.

        Parameters
        ----------
        maps : str | list[str], default "default"
            Name of the overlay to visualize from the tractogram's point data.

        cmap : str, default "viridis"
            Colormap for scalar data. If None, uses parcellation colors for
            categorical data or 'viridis' for scalar data.
            If a list of maps is provided, the same colormap will be applied to all.

        vmin : float, optional
            Minimum value for colormap scaling. If None, uses data minimum.

        vmax : float, optional
            Maximum value for colormap scaling. If None, uses data maximum.

        range_min : float, optional
            Minimum value for the display range. If None, uses data minimum.

        range_max : float, optional
            Maximum value for the display range. If None, uses data maximum.

        range_color : tuple, default (128, 128, 128, 255)
            RGBA color for values outside the specified range.

        views : str | list[str], optional
            Camera views for the visualization. Can be a single view or a list of views.
            If None, defaults to ["lateral"].

        hemi : str, default "lh"
            Hemisphere to visualize ("lh" for left hemisphere, "rh" for right hemisphere).

        use_opacity : bool, default True
            Whether to use opacity in the visualization.

        notebook : bool, default False
            Whether to render the plot in a Jupyter notebook.

        show_colorbar : bool, default False
            Whether to display the colorbar.

        colorbar_title : str, optional
            Title for the colorbar.

        colorbar_position : str, default "bottom"
            Position of the colorbar ("bottom", "top", "left", "right").

        save_path : str, optional
            Path to save the rendered plot. If None, the plot is not saved.

        config : str | Path | dict, optional
            Configuration for the plotter. Can be a file path or a dictionary.

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
        >>> pointcloud.plot(maps="point_id")
        >>> pointcloud.plot(maps="point_id", cmap="hot", views="medial", show_colorbar=True)
        """

        # If the radius is not floating point, convert it to float
        if not isinstance(radius, float):
            # Convert the radius to float if it is not already with 2 decimal places
            radius = float(radius)
            radius = round(radius, 2)

        if views is None:
            views = ["lateral"]
        dict_ctables = self.colortables
        if cmap is None:
            if maps in dict_ctables.keys():
                show_colorbar = False

            else:
                show_colorbar = True

        else:
            show_colorbar = True

        from . import visualizationtools as cltvis
        from . import visualization_utils as visutils

        # loading the configuration if None
        if config is None:
            # Loading the default configuration file
            cwd = os.path.dirname(os.path.abspath(__file__))

            # Default to the standard configuration file
            def_config_file = os.path.join(cwd, "config", "viz_views.json")
            config = visutils.load_configs(def_config_file)

        # Detect if the radius is different from the configuration and update if necessary
        def_as_spheres = config["objs_conf"]["points"]["spheres"]
        def_radius = config["objs_conf"]["points"]["spheres_radius"]
        if as_spheres != def_as_spheres:
            config["objs_conf"]["points"]["spheres"] = as_spheres

        if radius != def_radius:
            config["objs_conf"]["points"]["spheres_radius"] = radius

        # Initialize the BrainPlotter
        plotter = cltvis.BrainPlotter()

        plotter.plot(
            self,
            hemi_id=hemi,
            views=views,
            map_names=maps,
            colormaps=cmap,
            v_limits=(vmin, vmax),
            range_color=range_color,
            v_range=(range_min, range_max),
            use_opacity=use_opacity,
            notebook=notebook,
            colorbar=show_colorbar,
            colorbar_titles=colorbar_title,
            colorbar_position=colorbar_position,
            save_path=save_path,
            config_file=config,
        )


###############################################################################################
def merge_pointclouds(
    pointclouds: list[str | Path | PointCloud],
    color_table: dict = None,
    map_name: str = "point_id",
) -> PointCloud | None:
    """
    Merges multiple point clouds into a single point cloud.

    It combines all points and associated data from the input point clouds
    into a new PointCloud object. The points of each input point cloud are
    labelled with a unique ID stored in the map `map_name`, and a color table
    is created to differentiate them.

    Parameters
    ----------
    pointclouds : list of PointCloud, str or Path
        Point clouds to merge. File paths are loaded with PointCloud.load
        (npy format).

    color_table : dict, optional
        A dictionary defining the color table of the merged point clouds. If None,
        a color table with distinguishable colors is created, using the names of
        the point clouds (or "pointcloud_1", "pointcloud_2", ... if the names are
        not unique). The dictionary should contain:
            - 'names': List of names, one per point cloud.
            - 'color_table': numpy.ndarray of shape (n_pointclouds, 5) with the RGBA
              colors (0-1 range) and the ID value of each point cloud.
            - 'lookup_table': Optional, can be None.

    map_name : str, optional
        Name of the map storing the ID of the point cloud each point comes from.
        Default is 'point_id'.

    Returns
    -------
    PointCloud or None
        A new PointCloud containing all the points and associated data of the
        input point clouds. None if the input list is empty.

    Raises
    ------
    TypeError
        If pointclouds is not a list, contains items that are not PointCloud
        objects or file paths, or if color_table is not a dictionary.

    ValueError
        If color_table does not contain the required keys or its size does not
        match the number of point clouds.

    Examples
    --------
    >>> pc1 = PointCloud(points=np.random.rand(100, 3), name="left")
    >>> pc2 = PointCloud(points=np.random.rand(50, 3), name="right")
    >>> merged = merge_pointclouds([pc1, pc2])
    >>> len(merged)
    150
    >>> merged.colortables["point_id"]["names"]
    ['left', 'right']
    """

    if not isinstance(pointclouds, list):
        raise TypeError("pointclouds must be a list")

    if any(not isinstance(pc, (str, Path, PointCloud)) for pc in pointclouds):
        raise TypeError(
            "All items in pointclouds must be PointCloud objects, file paths, or Path objects"
        )

    # If the list is empty, return None
    if not pointclouds:
        return None

    # Load the point clouds supplied as files
    pointclouds = [
        PointCloud.load(pc) if isinstance(pc, (str, Path)) else pc for pc in pointclouds
    ]
    n_clouds = len(pointclouds)

    if color_table is not None:
        if not isinstance(color_table, dict):
            raise TypeError("color_table must be a dictionary")

        required_keys = ["names", "color_table"]
        if not all(key in color_table for key in required_keys):
            raise ValueError(f"color_table must contain the keys: {required_keys}")
        if len(color_table["names"]) != n_clouds:
            raise ValueError(
                "Length of 'names' in color_table must match number of point clouds"
            )
        if color_table["color_table"].shape[0] != n_clouds:
            raise ValueError(
                "Number of rows in 'color_table' must match number of point clouds"
            )

        color_table_dict = {
            "names": list(color_table["names"]),
            "color_table": np.array(color_table["color_table"]),
            "lookup_table": color_table.get("lookup_table", None),
        }

    # Creating a colortable in case it is not provided
    else:
        colors = cltcol.create_distinguishable_colors(n_clouds)
        ctable = cltcol.colors_to_table(
            colors=colors, alpha_values=1, values=np.arange(1, n_clouds + 1)
        )
        ctable[:, :3] = ctable[:, :3] / 255  # Ensure colors are between 0 and 1

        cloud_names = [pc.name for pc in pointclouds]
        if len(set(cloud_names)) != n_clouds:
            cloud_names = [f"pointcloud_{i + 1}" for i in range(n_clouds)]

        color_table_dict = {
            "names": cloud_names,
            "color_table": ctable,
            "lookup_table": None,
        }

    cloud_values = color_table_dict["color_table"][:, 4].astype(int)

    # Concatenate the point clouds and label the points of each one with its ID
    merged = pointclouds[0].copy()
    point_ids = [np.full(len(pointclouds[0]), cloud_values[0], dtype=int)]

    for i, pc in enumerate(pointclouds[1:], start=1):
        merged.append(pc, inplace=True, handle_colortable_conflicts="skip")
        point_ids.append(np.full(len(pc), cloud_values[i], dtype=int))

    merged.point_data[map_name] = np.concatenate(point_ids)
    merged.colortables[map_name] = color_table_dict

    return merged


######################################################################################################
def smooth_curve_coordinates(points, sigma=1.0, iterations=1, window_size=5):
    """
    Smooth a 3D curve using Gaussian-weighted neighborhood averaging.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Array of 3D coordinates forming an ordered curve.
    sigma : float, optional
        Standard deviation for Gaussian weighting. Default is 1.0.
    iterations : int, optional
        Number of smoothing iterations. Default is 1.
    window_size : int, optional
        Size of the neighborhood window (must be odd). Default is 5.

    Returns
    -------
    smoothed : ndarray, shape (N, 3)
        Smoothed 3D coordinates.
    """
    smoothed = points.copy()
    half_window = window_size // 2

    # Create Gaussian weights
    x = np.arange(-half_window, half_window + 1)
    weights = np.exp(-0.5 * (x / sigma) ** 2)
    weights /= weights.sum()

    for _ in range(iterations):
        new_points = np.zeros_like(smoothed)

        for i in range(len(smoothed)):
            start = max(0, i - half_window)
            end = min(len(smoothed), i + half_window + 1)

            w_start = half_window - (i - start)
            w_end = w_start + (end - start)

            # Renormalize a copy, so the edge points do not modify the shared weights
            local_weights = weights[w_start:w_end] / weights[w_start:w_end].sum()

            new_points[i] = np.sum(smoothed[start:end] * local_weights[:, None], axis=0)

        smoothed = new_points

    return smoothed
