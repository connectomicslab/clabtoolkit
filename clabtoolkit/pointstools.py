import os
import numpy as np
import copy
import nibabel as nb

from typing import Union, List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import copy

# Importing local modules
from . import misctools as cltmisc
from . import parcellationtools as cltparc
from . import colorstools as cltcolors


# Utility function for interpolating streamline values
from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    TextColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)


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
        point_data (Dict): A dictionary to store scalar data associated with each point.
    """

    def __init__(
        self,
        points: Union[np.ndarray, pd.DataFrame] = None,
        affine: np.ndarray = None,
        color: Union[str, np.ndarray] = "#BFBDBD",
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

            color (str or np.ndarray, optional):
                Color for the point cloud. Can be a hex string or RGB array.
                Default is "#BFBDBD".

            alpha (float, optional):
                Opacity value for the point cloud (0-1). Default is 1.0.

            name (str, optional):
                Name of the point cloud. Default is "default".
        """

        # Initialize attributes
        self.name = name
        self.coords = None
        self.colortables: Dict[str, Dict] = {}
        self.point_data: Dict[str, np.ndarray] = {}

        # Set affine (always initialize, even if points is None)
        self.affine = affine if affine is not None else np.eye(4)

        # Validate alpha value
        if isinstance(alpha, int):
            alpha = float(alpha)

        # If the alpha is not in the range [0, 1], raise an error
        if not (0 <= alpha <= 1):
            raise ValueError(f"Alpha value must be in the range [0, 1], got {alpha}")

        # Handle color input
        color = cltcolors.harmonize_colors(color, output_format="rgb") / 255

        tmp_ctable = cltcolors.colors_to_table(colors=color, alpha_values=alpha)
        tmp_ctable[:, :3] = tmp_ctable[:, :3] / 255  # Ensure colors are between 0 and 1

        # Store parcellation information in organized structure
        self.colortables["default"] = {
            "names": ["default"],
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

            # Initialize default point data
            default = np.array(np.ones(len(self.coords)) * tmp_ctable[0, 4], dtype=int)
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
    def transform(
        self, affine: np.ndarray, inplace: bool = True
    ) -> Optional["PointCloud"]:
        """
        Applies an affine transformation to the point coordinates.

        Parameters:
        -----------
            affine (np.ndarray):
                A 4x4 affine transformation matrix.

            inplace (bool):
                If True, modifies the current object. If False, returns a new object.
                Default is True.

        Returns:
        -------
            PointCloud or None:
                If inplace=False, returns a new transformed PointCloud.
                If inplace=True, returns None.
        """
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
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
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
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
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
        >>> pc.filter_points('X > 50')  # Keep points with X > 50

        >>> # Filter by point_data attribute
        >>> pc.filter_points('intensity > 0.5')  # Keep high intensity points

        >>> # Filter by range
        >>> pc.filter_points('20 <= Z <= 80')  # Keep points in Z range

        >>> # Create new filtered point cloud
        >>> pc_filtered = pc.filter_points('Y < 30', inplace=False)
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
            colors = cltcolors.harmonize_colors(color_table[:, :3], output_format="hex")

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
        filename: Union[str, Path],
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
            np.savez(filename, **save_dict)

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
    def load(cls, filename: Union[str, Path], format: str = "npy") -> "PointCloud":
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
        lut_file: Union[str, Path],
        map_name: str = "default",
        opacity: Union[float, int, np.ndarray] = 1.0,
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
            lut_dict = cltparc.Parcellation.read_luttable(lut_file)
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
            color_table = cltcolors.colors_to_table(colors=colors, values=values)
        else:
            color_table = cltcolors.colors_to_table(colors=colors)

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
    def explore_pointcloud(self) -> None:
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
        >>> pc.explore_pointcloud()
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
        overlay_name: str = "default",
        colormap: str = "viridis",
        vmin: np.float64 = None,
        vmax: np.float64 = None,
        range_min: np.float64 = None,
        range_max: np.float64 = None,
        range_color: Tuple = (128, 128, 128, 255),
    ) -> np.ndarray:
        """
        Compute streamlines colors for visualization based on the specified overlay.

        This method processes the overlay data and creates appropiate point colors
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
        >>> tractogram.get_vertexwise_colors(overlay_name="aparc")
        >>>
        >>> # Prepare colors for scalar data with custom colormap
        >>> tractogram.get_vertexwise_colors(overlay_name="thickness", colormap="hot")
        >>>
        >>> # Prepare colors for the tractogram overlay
        >>> tractogram.get_vertexwise_colors()
        """

        # Get the list of overlays
        maps_list = self.list_maps()

        if overlay_name not in maps_list:
            raise ValueError(
                f"Overlay '{overlay_name}' not found. Available overlays: {', '.join(maps_list)}"
            )

        # Getting the values of the overlay
        data = self.point_data[overlay_name]

        # if colortables is an attribute of the class, use it
        if hasattr(self, "colortables"):
            dict_ctables = self.colortables

            # Check if the overlay is on the colortables
            if overlay_name in dict_ctables.keys():
                # Use the colortable associated with the parcellation

                point_colors = cltcol.get_colors_from_colortable(
                    data, self.colortables[overlay_name]["color_table"]
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
    def list_maps(self) -> List[str]:
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
        overlay_name: str = "default",
        cmap: str = "viridis",
        vmin: np.float64 = None,
        vmax: np.float64 = None,
        range_min: np.float64 = None,
        range_max: np.float64 = None,
        range_color: Tuple = (128, 128, 128, 255),
        views: Union[str, List[str]] = ["lateral"],
        hemi: str = "lh",
        use_opacity: bool = True,
        notebook: bool = False,
        show_colorbar: bool = False,
        colorbar_title: str = None,
        colorbar_position: str = "bottom",
        save_path: str = None,
        config: Union[str, Path, Dict] = None,
    ):
        """
        Plot the point cloud with specified overlay and visualization parameters.

        Renders the tractogram with optional overlays using PyVista, supporting
        multiple camera views, custom colormaps, and interactive or static output.
        Handles both categorical parcellation data and continuous scalar overlays.

        Parameters
        ----------
        overlay_name : str, default "default"
            Name of the overlay to visualize from the tractogram's point data.

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

        use_opacity : bool, default True
            Whether to use opacity settings from the tractogram overlays.

        plot_style : str, default "tube"
            Style for rendering streamlines: 'tube' or 'line'.

        vis_percentage : float, default 100
            Percentage of streamlines to visualize (0-100). Reduces number for
            faster rendering if less than 100.

        force_reduction : bool, default True
            Whether to force reduction of streamlines when vis_percentage < 100.

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

        config : str, Path or Dict, optional
            Configuration for visualization settings. Can be a file path or dictionary.

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
        >>> tractogram.plot(overlay_name="fa")
        >>> tractogram.plot(overlay_name="fa", cmap="hot", views="medial", show_colorbar=True)
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

        # Initialize the BrainPlotter
        plotter = cltvis.BrainPlotter()

        plotter.plot(
            self,
            hemi_id=hemi,
            views=views,
            map_names=overlay_name,
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
    tractograms: List[Union[str, Path, PointCloud]],
    color_table: dict = None,
    map_name: str = "point_id",
) -> Union[PointCloud, None]:
    """
    Merges multiple point clouds into a single tractogram.

    It combines all points and associated data from the input point clouds
    into a new Point object. Each point cloud's points are assigned unique IDs
    in the merged object, and a color table is created to differentiate them.

        color_table : dict, optional
            A dictionary defining the color table for the merged tractogram. If None, a default color table will be created.

            The dictionary should contain:
                - 'tractograms_names': List of names for each tractogram.
                - 'color_table': numpy.ndarray of shape (n_tractograms, 5) with RGBA colors and values.
                - 'lookup_table': Optional, can be None.

        map_name : str, optional
            Name of the map to store tract IDs in the merged tractogram. Default is 'tract_id'.

    Returns:
    --------
        merged_tractogram : Tractogram
            A new Tractogram object containing all streamlines and associated data
            from the input tractograms.

    Raises:
    -------
        ValueError: If the input list is empty or contains non-Tractogram objects.

    Examples:
    ---------
        >>> tract1 = Tractogram('tract1.trk')
        >>> tract2 = Tractogram('tract2.trk')
        >>> merged_tract = merge_tractograms([tract1, tract2])
        >>> print(f"Merged tractogram has {len(merged_tract.tracts)} streamlines")
    """

    if not isinstance(tractograms, list):
        raise TypeError("tractograms must be a list")

    if any(not isinstance(surf, (str, Path, PointCloud)) for surf in tractograms):
        raise TypeError(
            "All items in tractograms must be Tractogran objects, file paths, or Path objects"
        )

    # If the list is empty, return None
    if not tractograms:
        return None

    # If there's only one surface, return it as is
    if len(tractograms) == 1:
        return PointCloud(tractograms[0])

    n_tracts = len(tractograms)
    if color_table is not None:
        if not isinstance(color_table, dict):
            raise TypeError("color_table must be a dictionary")

        required_keys = ["names", "color_table"]
        if not all(key in color_table for key in required_keys):
            raise ValueError(f"color_table must contain the keys: {required_keys}")
        if len(color_table["names"]) != n_tracts:
            raise ValueError(
                "Length of 'names' in color_table must match number of tractograms"
            )
        if color_table["color_table"].shape[0] != n_tracts:
            raise ValueError(
                "Number of rows in 'color_table' must match number of tractograms"
            )

    # Creating a colortable in case it is not provided
    if color_table is None:
        colors = cltcolors.create_distinguishable_colors(n_tracts)
        color_table = cltcolors.colors_to_table(
            colors=colors, alpha_values=1, values=range(n_tracts)
        )
        color_table[:, :3] = (
            color_table[:, :3] / 255
        )  # Ensure colors are between 0 and 1
        color_table[:, 4] = np.arange(n_tracts) + 1  # Set the value column
        bundle_names = [f"tract_{i}" for i in range(n_tracts)]
        color_table_dict = {
            "names": bundle_names,
            "color_table": color_table,
            "lookup_table": None,
        }

    # Initialize lists to hold merged data

    # Add Rich progress bar around the main loop
    n_bundles = len(tractograms)
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:

        task = progress.add_task("Merging tractograms", total=n_bundles)

        for i, t in enumerate(tractograms):
            progress.update(
                task,
                description=f"Merging tractograms: {i + 1}/{n_bundles}",
                completed=i + 1,
            )

            if isinstance(t, (str, Path)):
                t = Tractogram(t)

            if i == 0:
                merged_tractogram = copy.deepcopy(t)
                bundle_ids = np.full(
                    (len(merged_tractogram.tracts), 1), color_table[i, 4]
                )

            else:
                merged_tractogram = merged_tractogram.add_tractogram(t)
                bundle_ids = np.vstack(
                    (
                        bundle_ids,
                        np.full(
                            (len(merged_tractogram.tracts) - len(bundle_ids), 1),
                            color_table[i, 4],
                        ),
                    )
                )

    merged_tractogram.data_per_streamline[map_name] = bundle_ids.reshape(
        -1, 1
    )  # Reshape to be a column vector
    merged_tractogram.colortables[map_name] = color_table_dict

    return merged_tractogram
