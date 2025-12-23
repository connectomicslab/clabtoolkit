import os
import numpy as np
import copy
import nibabel as nb
from nibabel.streamlines import ArraySequence
from scipy.interpolate import RegularGridInterpolator

from typing import Union, List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import copy

# Importing local modules
from . import misctools as cltmisc
from . import parcellationtools as cltparc
from . import colorstools as cltcol

from dipy.segment.clustering import QuickBundlesX, QuickBundles
from dipy.tracking.streamline import set_number_of_points
from dipy.io.stateful_tractogram import Space

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
############            Section 1: Class and methods work with tractograms              ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
class Tractogram:
    """
    A class to represent and manipulate tractograms.

    Attributes:
        tractogram_input (str): Path to the tractogram file.
        tracts (list): List of streamlines in the tractogram.
        affine (np.ndarray): Affine transformation matrix.
        header (dict): Header information from the tractogram file.
    """

    def __init__(
        self,
        tractogram_input: Union[nb.streamlines.Tractogram, str, Path] = None,
        tracts: ArraySequence = None,
        affine: np.ndarray = None,
        header: Dict = None,
        color: Union[str, np.ndarray] = "#BFBDBD",
        alpha: float = 1.0,
        name: str = "default",
    ) -> None:
        """
        Initializes the Tractogram class by loading a tractogram file.

        Parameters:
        -----------
            tractogram_input (str or nb.streamlines.Tractogram):
                Path to the tractogram file or nibabel Tractogram object.

        Raises:
        -------
            FileNotFoundError: If the specified tractogram file does not exist.
        """

        # Initialize attributes to None (empty instance)
        self.name = name
        self.tracts = None
        self.colortables: Dict[str, Dict] = {}

        # Validate alpha value
        if isinstance(alpha, int):
            alpha = float(alpha)

        # If the alpha is not in the range [0, 1], raise an error
        if not (0 <= alpha <= 1):
            raise ValueError(f"Alpha value must be in the range [0, 1], got {alpha}")

        # Handle color input
        color = cltcol.harmonize_colors(color, output_format="rgb") / 255

        tmp_ctable = cltcol.colors_to_table(colors=color, alpha_values=alpha)
        tmp_ctable[:, :3] = tmp_ctable[:, :3] / 255  # Ensure colors are between 0 and 1

        # Store parcellation information in organized structure
        self.colortables["default"] = {
            "names": ["default"],
            "color_table": tmp_ctable,
            "lookup_table": None,  # Will be populated by _create_parcellation_colortable if needed
        }

        # Validate input parameters
        if tractogram_input is not None and (
            tracts is not None or affine is not None or header is not None
        ):
            raise ValueError(
                "Cannot specify both tractogram_input and tracts/affine/header"
            )

        if tracts is not None and (affine is None or header is None):
            raise ValueError(
                "If tracts are provided, affine and header must also be provided"
            )

        if affine is not None and (tracts is None or header is None):
            raise ValueError(
                "If affine is provided, tracts and header must also be provided"
            )

        if header is not None and (tracts is None or affine is None):
            raise ValueError(
                "If header is provided, tracts and affine must also be provided"
            )

        # Load data based on input type
        if isinstance(tractogram_input, nb.streamlines.Tractogram):
            # Load from nibabel Tractogram object
            self.tract_file = "from_object"
            self.load_tractogram_from_object(tractogram_input)

            # Create the data_per_streamline for default map
            default = np.array(np.ones(len(self.tracts)) * tmp_ctable[:, 4], dtype=int)
            self.data_per_streamline["default"] = default.reshape(-1, 1)

        elif tractogram_input is not None:
            # Load from file path
            if isinstance(tractogram_input, Path):
                tractogram_input = str(tractogram_input)

            if isinstance(tractogram_input, str):
                if not os.path.isfile(tractogram_input):
                    raise FileNotFoundError(
                        f"Tractogram file not found: {tractogram_input}"
                    )

            self.tract_file = tractogram_input  # Store the actual path
            (
                self.tracts,
                self.affine,
                self.header,
                self.data_per_point,
                self.data_per_streamline,
            ) = self.load_tractogram_from_file(tractogram_input)

            # Create the data_per_streamline for default map
            default = np.array(np.ones(len(self.tracts)) * tmp_ctable[:, 4], dtype=int)
            self.data_per_streamline["default"] = default.reshape(-1, 1)

        elif tracts is not None and affine is not None and header is not None:
            # Load from direct components
            self.tract_file = "from_components"
            self.tracts = tracts
            self.affine = affine
            self.header = header

            # Initialize data dictionaries
            self.data_per_point = {}
            self.data_per_streamline = {}

            # Create the data_per_streamline for default map
            default = np.array(np.ones(len(self.tracts)) * tmp_ctable[:, 4], dtype=int)
            self.data_per_streamline["default"] = default.reshape(-1, 1)

            # Calculate streamline lengths (the method now stores them internally)
            self.compute_streamline_lengths()

        else:
            # Empty tractogram - initialize with None/empty values
            self.tract_file = None
            self.tracts = None
            self.affine = None
            self.header = None
            self.data_per_point = {}
            self.data_per_streamline = {}
            self.colortables = {}

    ###############################################################################################
    def load_tractogram_from_file(
        self, tractogram_input: str
    ) -> Tuple[List[np.ndarray], np.ndarray, Dict]:
        """
        Loads a tractogram file and extracts streamlines, affine transformation, and header information.

        Parameters:
        -----------
            tractogram_input (str):
                Path to the tractogram file.

        Returns:
        --------
            Tuple[List[np.ndarray], np.ndarray, Dict]:
                A tuple containing the list of streamlines, affine transformation matrix, and header information.

        """

        # Validate input file format
        if nb.streamlines.detect_format(tractogram_input) not in [
            nb.streamlines.TrkFile,
            nb.streamlines.TckFile,
        ]:
            raise ValueError(
                f"Invalid input file format: {tractogram_input}. Must be TRK or TCK."
            )

        if not os.path.isfile(tractogram_input):
            raise FileNotFoundError(
                f"The specified tractogram file does not exist: {tractogram_input}"
            )

        # Load the tractogram using nibabel
        tractogram = nb.streamlines.load(tractogram_input)
        streamlines = copy.deepcopy(tractogram.tractogram.streamlines)
        affine = copy.deepcopy(tractogram.tractogram.affine_to_rasmm)
        header = copy.deepcopy(tractogram.header)

        self.tracts = copy.deepcopy(streamlines)
        self.affine = copy.deepcopy(affine)
        self.header = copy.deepcopy(header)

        data_per_point = {}
        if hasattr(tractogram.tractogram, "data_per_point"):
            maps = list(tractogram.tractogram.data_per_point.keys())
            for m in maps:
                data_per_point[m] = copy.deepcopy(
                    tractogram.tractogram.data_per_point[m]
                )
        self.data_per_point = copy.deepcopy(data_per_point)

        data_per_streamline = {}
        if hasattr(tractogram.tractogram, "data_per_streamline"):
            maps = list(tractogram.tractogram.data_per_streamline.keys())
            for m in maps:
                data_per_streamline[m] = copy.deepcopy(
                    tractogram.tractogram.data_per_streamline[m]
                )

        # Compute and store lengths BEFORE final assignment
        lengths = self.compute_streamline_lengths()
        data_per_streamline["length"] = lengths
        self.data_per_streamline = copy.deepcopy(data_per_streamline)

        return streamlines, affine, header, data_per_point, data_per_streamline

    ###############################################################################################
    def load_tractogram_from_object(
        self, tractogram: nb.streamlines.Tractogram
    ) -> None:
        """
        Loads a tractogram from a nibabel Tractogram object.

        Parameters:
        -----------
            tractogram (nb.streamlines.Tractogram):
                A nibabel Tractogram object containing streamlines and metadata.
        Returns:
        -------
            None
        """
        if not isinstance(tractogram, nb.streamlines.Tractogram):
            raise TypeError("Input must be a nibabel Tractogram object")

        self.tracts = copy.deepcopy(tractogram.streamlines)
        self.affine = copy.deepcopy(tractogram.affine_to_rasmm)
        self.header = copy.deepcopy(tractogram.header)

        # Handle data_per_point consistently
        data_per_point = {}
        if hasattr(tractogram, "data_per_point"):
            maps = list(tractogram.data_per_point.keys())
            for m in maps:
                data_per_point[m] = copy.deepcopy(tractogram.data_per_point[m])
        self.data_per_point = copy.deepcopy(data_per_point)

        # Handle data_per_streamline consistently
        data_per_streamline = {}
        if hasattr(tractogram, "data_per_streamline"):
            maps = list(tractogram.data_per_streamline.keys())
            for m in maps:
                data_per_streamline[m] = copy.deepcopy(
                    tractogram.data_per_streamline[m]
                )

        # Compute and store lengths BEFORE final assignment
        lengths = self.compute_streamline_lengths()
        data_per_streamline["length"] = lengths
        self.data_per_streamline = copy.deepcopy(data_per_streamline)

    ###############################################################################################
    def load_colortable(
        self,
        lut_file: Union[str, Path],
        map_name: str = "default",
        opacity: np.ndarray = 1.0,
    ) -> None:
        """
        Loads a colortable from a file and associates it with a specified map name.

        Parameters:
        -----------
            colortable_path (str or Path):
                Path to the colortable file.

            map_name (str):
                Name of the map to associate with the loaded colortable.

        Returns:
        -------
            None

        """
        if isinstance(lut_file, Path):
            lut_file = str(lut_file)

        if not os.path.isfile(lut_file):
            raise FileNotFoundError(
                f"The specified colortable file does not exist: {lut_file}"
            )

        # Load the colortable using the utility function
        lut_dict = cltcol.ColorTableLoader.load_colortable(lut_file)

        colors = lut_dict["color"]
        if map_name in self.data_per_streamline or map_name not in self.data_per_point:
            if (
                map_name in self.data_per_streamline
                and map_name not in self.data_per_point
            ):
                values = np.unique(np.concatenate(self.data_per_streamline[map_name]))
                if len(values) != len(colors):
                    raise ValueError(
                        f"Colortable in {lut_file} does not cover all IDs in data_per_point for map '{map_name}'."
                    )
            elif (
                map_name not in self.data_per_streamline
                and map_name in self.data_per_point
            ):
                values = np.unique(np.concatenate(self.data_per_point[map_name]))
                if len(values) != len(colors):
                    raise ValueError(
                        f"Colortable in {lut_file} does not cover all IDs in data_per_point for map '{map_name}'."
                    )

            elif (
                map_name in self.data_per_streamline and map_name in self.data_per_point
            ):
                values = np.unique(np.concatenate(self.data_per_point[map_name]))
                if len(values) != len(colors):
                    raise ValueError(
                        f"Colortable in {lut_file} does not cover all IDs in data_per_point for map '{map_name}'."
                    )

            color_table = cltcol.colors_to_table(colors=colors, values=values)
        else:
            color_table = cltcol.colors_to_table(colors=colors)

        if isinstance(opacity, (int, float)):
            # opacity is a scalar, no need to check length
            opacity_array = np.full(color_table.shape[0], opacity)

        elif len(opacity) != color_table.shape[0]:
            opacity_array = np.full(color_table.shape[0], opacity[0])

        else:
            opacity_array = np.array(opacity)

        color_table[:, :3] = (
            color_table[:, :3] / 255
        )  # Ensure colors are between 0 and 1

        color_table[:, 3] = opacity_array  # Set uniform opacity

        # Store parcellation information in organized structure
        self.colortables[map_name] = {
            "names": lut_dict["name"],
            "color_table": color_table,
            "lookup_table": None,
        }

    ###############################################################################################
    def smooth_streamlines(self, iterations: int = 1, sigma: float = 1.0) -> None:
        """
        Smooth streamlines using a Gaussian filter.

        Parameters
        ----------
        iterations : int, optional
            Number of smoothing iterations to perform. Default is 1.
        sigma : float, optional
            Standard deviation for Gaussian kernel. Default is 1.0.

        Returns
        -------
        None
            The method modifies the streamlines in place.
        """

        # Check if the tractogram has streamlines loaded
        if not hasattr(self, "tracts") or self.tracts is None:
            raise ValueError(
                "No streamlines loaded. Please ensure the tractogram file was loaded correctly."
            )

        # Get streamlines (reference to tracts for consistency with class structure)
        streamlines = self.tracts

        # Check if the streamlines are empty
        if len(streamlines) == 0:
            raise ValueError(
                "Tractogram contains no streamlines. Please provide a valid tractogram file."
            )

        # Handle both ArraySequence and list inputs
        is_list_input = False
        if not isinstance(streamlines, nb.streamlines.array_sequence.ArraySequence):
            if isinstance(streamlines, list):
                streamlines = ArraySequence(streamlines)
                is_list_input = True

        # Smooth individual streamlines
        for i, streamline in enumerate(streamlines):

            smooth_st = cltmisc.smooth_curve_coordinates(
                streamline, iterations=iterations, sigma=sigma
            )
            streamlines[i] = smooth_st

        # Update the tractogram object with smoothed streamlines
        if is_list_input:
            self.tracts = list(streamlines)
        else:
            self.tracts = streamlines

    ###############################################################################################
    def resample_streamlines(
        self,
        num_points: int = 51,
        interp_method: str = "linear",
    ) -> Union[List[np.ndarray], nb.streamlines.array_sequence.ArraySequence]:
        """
        Resample streamlines to a specified number of points.

        Parameters
        ----------
        num_points : int, optional
            Number of points to resample each streamline to. Default is 51.

        Returns
        -------
        resampled_streamlines : List[np.ndarray] or nb.streamlines.array_sequence.ArraySequence
            Resampled streamlines in the same format as the input streamlines.
            Each streamline is a numpy array with shape (nb_points, 3).

        Raises
        ------
        ValueError
            If nb_points is not a positive integer.
        ValueError
            If the tractogram has no streamlines loaded.
        ValueError
            If streamlines are empty.
        ValueError
            If individual streamlines are not valid numpy arrays.

        Examples
        --------
        >>> tractogram = Tractogram('input.trk')
        >>> resampled_streamlines = tractogram.resample_streamlines(nb_points=100)
        >>> print(f"Resampled {len(resampled_streamlines)} streamlines to 100 points each")
        """
        # Check if nb_points is a positive integer
        if not isinstance(num_points, int) or num_points <= 0:
            raise ValueError(
                "Number of points (num_points) must be a positive integer."
            )

        # Check if the tractogram has streamlines loaded
        if not hasattr(self, "tracts") or self.tracts is None:
            raise ValueError(
                "No streamlines loaded. Please ensure the tractogram file was loaded correctly."
            )

        # Get streamlines (reference to tracts for consistency with class structure)
        streamlines = self.tracts

        # Check if the streamlines are empty
        if len(streamlines) == 0:
            raise ValueError(
                "Tractogram contains no streamlines. Please provide a valid tractogram file."
            )

        # Handle both ArraySequence and list inputs
        is_list_input = False
        if not isinstance(streamlines, nb.streamlines.array_sequence.ArraySequence):
            if isinstance(streamlines, list):
                streamlines = ArraySequence(streamlines)
                is_list_input = True

        # Validate individual streamlines
        for i, streamline in enumerate(streamlines):
            if not isinstance(streamline, np.ndarray):
                raise ValueError(f"Streamline {i} is not a valid numpy array.")

            if streamline.ndim != 2 or streamline.shape[1] != 3:
                raise ValueError(
                    f"Streamline {i} must be a 2D array with shape (n_points, 3), "
                    f"but has shape {streamline.shape}."
                )

            if streamline.shape[0] < 2:
                raise ValueError(
                    f"Streamline {i} must have at least 2 points for resampling, "
                    f"but has {streamline.shape[0]} points."
                )

        # Resample each streamline to the specified number of points
        # Use dipy's set_number_of_points for ArraySequence
        resampled_streamlines = set_number_of_points(streamlines, num_points)

        # Resampling the data_per_point if it exists
        if hasattr(self, "data_per_point") and self.data_per_point:
            per_point_map_names = list(self.data_per_point.keys())
            n_maps = len(per_point_map_names)

            # Create new data_per_point structure for resampled data
            new_data_per_point = {map_name: [] for map_name in per_point_map_names}

            # Process each streamline
            for streamline_idx, st in enumerate(streamlines):
                nst_points = len(st)
                orig_values = np.zeros((nst_points, n_maps))
                orig_coords = st[:, 0:3]
                new_coords = resampled_streamlines[streamline_idx][:, 0:3]

                # Extract original values for all maps for this streamline
                for map_idx, map_name in enumerate(per_point_map_names):
                    # Handle both 1D and 2D data_per_point arrays
                    map_data = self.data_per_point[map_name][streamline_idx]
                    if isinstance(map_data, np.ndarray):
                        if map_data.ndim == 2:
                            # If 2D (n_points, 1), flatten to 1D
                            orig_values[:, map_idx] = map_data.flatten()
                        else:
                            # If already 1D
                            orig_values[:, map_idx] = map_data
                    else:
                        # Convert to array if it's a list or other format
                        orig_values[:, map_idx] = np.array(map_data).flatten()

                # Interpolate values to new coordinates
                new_values = interpolate_streamline_values(
                    orig_coords, orig_values, new_coords, method=interp_method
                )

                # Store the interpolated values in the new structure
                for map_idx, map_name in enumerate(per_point_map_names):
                    # Determine original format and maintain it
                    original_data = self.data_per_point[map_name][streamline_idx]
                    if (
                        isinstance(original_data, np.ndarray)
                        and original_data.ndim == 2
                    ):
                        # Keep as 2D (n_points, 1)
                        new_data_per_point[map_name].append(
                            new_values[:, map_idx].reshape(-1, 1)
                        )
                    else:
                        # Keep as 1D
                        new_data_per_point[map_name].append(new_values[:, map_idx])

            # Replace the old data_per_point with the new resampled data
            # Convert lists to ArraySequence for nibabel compatibility
            for map_name in per_point_map_names:
                self.data_per_point[map_name] = ArraySequence(
                    new_data_per_point[map_name]
                )

        # Update the tractogram object with resampled streamlines
        if is_list_input:
            self.tracts = list(resampled_streamlines)
        else:
            self.tracts = resampled_streamlines

        # Update header if it exists
        if hasattr(self, "header") and self.header and "nb_streamlines" in self.header:
            self.header["nb_streamlines"] = len(self.tracts)

        # Convert back to list if the input was a list for return value
        if is_list_input:
            return list(resampled_streamlines)
        else:
            return resampled_streamlines

    ####################################################################################################
    def explore_tractogram(self):
        """
        Display comprehensive information about the tractogram.

        Provides a formatted overview of the tractogram including streamline count,
        spatial transformations, header metadata, scalar data properties, and available
        colortables. Useful for quick inspection and validation of tractogram data.

        The method displays:
            - Basic tractogram identification and streamline count
            - Affine transformation matrix for spatial mapping
            - Header information (dimensions, voxel sizes, origin)
            - Scalar data per point (with min/max statistics)
            - Scalar data per streamline (with min/max statistics)
            - Available colortables for visualization

        Returns
        -------
        None
            Prints formatted information to stdout.

        Notes
        -----
        This method performs no modifications to the tractogram data. It only
        displays information for inspection purposes.

        Examples
        --------
        >>> tractogram = Tractogram('input.trk')
        >>> tractogram.explore_tractogram()
        ╔════════════════════════════════════════════════════════════════╗
        ║                    TRACTOGRAM EXPLORATION                      ║
        ╠════════════════════════════════════════════════════════════════╣
        ║ Name: default                                                  ║
        ║ Streamlines: 15,000                                            ║
        ╠════════════════════════════════════════════════════════════════╣
        ║ AFFINE TRANSFORMATION MATRIX                                   ║
        ║   [[ 1.00   0.00   0.00  -90.00]                               ║
        ║    [ 0.00   1.00   0.00 -126.00]                               ║
        ║    [ 0.00   0.00   1.00  -72.00]                               ║
        ║    [ 0.00   0.00   0.00    1.00]]                              ║
        ╠════════════════════════════════════════════════════════════════╣
        ║ HEADER INFORMATION                                             ║
        ║   Dimensions:    181 × 217 × 181                               ║
        ║   Voxel Size:    1.00 × 1.00 × 1.00 mm                         ║
        ╠════════════════════════════════════════════════════════════════╣
        ║ SCALAR DATA PER POINT (2 maps)                                 ║
        ║   fa          Min: 0.0000    Max: 0.9200                       ║
        ║   md          Min: 0.0005    Max: 0.0020                       ║
        ╠════════════════════════════════════════════════════════════════╣
        ║ SCALAR DATA PER STREAMLINE (1 map)                             ║
        ║   length      Min: 20.50     Max: 150.30                       ║
        ╠════════════════════════════════════════════════════════════════╣
        ║ COLORTABLES (2 available)                                      ║
        ║   • default                                                    ║
        ║   • cluster_id                                                 ║
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
        print_line("TRACTOGRAM EXPLORATION".center(width), width)
        print("╠" + "═" * width + "╣")

        # Basic information
        print_line(f" Name: {self.name}", width)
        streamline_count = len(self.tracts) if self.tracts is not None else 0
        print_line(f" Streamlines: {format_number(streamline_count)}", width)

        # Affine transformation matrix
        print("╠" + "═" * width + "╣")
        if self.affine is not None:
            print_line(" AFFINE TRANSFORMATION MATRIX", width)
            affine_lines = str(self.affine).split("\n")
            for line in affine_lines:
                print_line(f"   {line}", width)
        else:
            print_line(" Affine transformation matrix: Not available", width)

        # Header information
        print("╠" + "═" * width + "╣")
        print_line(" HEADER INFORMATION", width)
        if self.header is not None:
            if "dim" in self.header:
                dim = self.header["dim"]
                dim_str = " × ".join(map(str, dim))
                print_line(f"   Dimensions:    {dim_str}", width)

            if "voxel_size" in self.header:
                voxel = self.header["voxel_size"]
                voxel_str = " × ".join(f"{v:.2f}" for v in voxel) + " mm"
                print_line(f"   Voxel Size:    {voxel_str}", width)

            if "origin" in self.header:
                origin = self.header["origin"]
                origin_str = " × ".join(f"{o:.2f}" for o in origin)
                print_line(f"   Origin:        {origin_str}", width)
        else:
            print_line("   Not available", width)

        # Scalar data per point
        print("╠" + "═" * width + "╣")
        if hasattr(self, "data_per_point") and self.data_per_point:
            count = len(self.data_per_point)
            print_line(
                f" SCALAR DATA PER POINT ({count} {'map' if count == 1 else 'maps'})",
                width,
            )

            for map_name, values in self.data_per_point.items():
                all_values = np.concatenate(values)
                min_val = np.nanmin(all_values)
                max_val = np.nanmax(all_values)
                print_line(
                    f"   {map_name:<12}  Min: {min_val:>8.4f}    Max: {max_val:>8.4f}",
                    width,
                )
        else:
            print_line(" SCALAR DATA PER POINT (0 maps)", width)
            print_line("   No scalar data available", width)

        # Scalar data per streamline
        print("╠" + "═" * width + "╣")
        if hasattr(self, "data_per_streamline") and self.data_per_streamline:
            count = len(self.data_per_streamline)
            print_line(
                f" SCALAR DATA PER STREAMLINE ({count} {'map' if count == 1 else 'maps'})",
                width,
            )

            for map_name, values in self.data_per_streamline.items():
                all_values = np.concatenate(values)
                min_val = np.nanmin(all_values)
                max_val = np.nanmax(all_values)
                print_line(
                    f"   {map_name:<12}  Min: {min_val:>8.2f}     Max: {max_val:>8.2f}",
                    width,
                )
        else:
            print_line(" SCALAR DATA PER STREAMLINE (0 maps)", width)
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

    ####################################################################################################
    def streamline_to_points(
        self, map_name: str, point_map_name: str = None
    ) -> ArraySequence:
        """
        Converts maps_per_streamline to maps_per_point by assigning the streamline value to all its points.
        This method creates a new set of data_per_point where each point in a streamline
        inherits the value of its corresponding streamline from data_per_streamline.

        Parameters
        ----------
        map_name : str, optional
            The name of the map in data_per_streamline to convert.

        point_map_name : str, optional
            The name of the new map to create in data_per_point. If None, uses map_name.

        Returns
        -------
            ArraySequence:
                An ArraySequence containing all points from all streamlines.
        Raises
        ------
            ValueError: If the specified map_name does not exist in data_per_streamline.
            ValueError: If the streamline value for the specified map is not a single value per streamline.

        Examples
        --------
        >>> tractogram = Tractogram('input.trk')
        >>> point_data = tractogram.streamline_to_points(map_name='cluster_id')
        >>> print(f"Converted streamline map 'cluster_id' to point data with {len(point_data)} points")

        """
        if (
            not hasattr(self, "data_per_streamline")
            or map_name not in self.data_per_streamline
        ):
            raise ValueError(f"Map '{map_name}' not found in data_per_streamline.")

        if point_map_name is None:
            point_map_name = map_name

        all_points = []
        for i, streamline in enumerate(self.tracts):
            n_points = len(streamline)
            streamline_value = self.data_per_streamline[map_name][i]
            if isinstance(streamline_value, (list, np.ndarray)):
                if len(streamline_value) != 1:
                    raise ValueError(
                        f"Streamline value for map '{map_name}' must be a single value per streamline."
                    )
                streamline_value = streamline_value[0]
            point_values = np.full((n_points, 1), streamline_value)
            all_points.append(point_values)

        self.data_per_point[point_map_name] = ArraySequence(all_points)

        # Check if the map has colortable information and propagate it if so
        if map_name in self.colortables:
            self.colortables[point_map_name] = copy.deepcopy(self.colortables[map_name])

        return ArraySequence(all_points)

    ####################################################################################################
    def points_to_streamline(
        self, map_name: str, streamline_map_name: str = None, metric: str = "mean"
    ) -> np.ndarray:
        """
        Converts maps_per_point to maps_per_streamline by averaging point values for each streamline.
        This method creates a new set of data_per_streamline where each streamline's value
        is the average of its corresponding points from data_per_point.

        Parameters
        ----------
        map_name : str
            The name of the map in data_per_point to convert.

        streamline_map_name : str, optional
            The name of the new map to create in data_per_streamline. If None, uses map_name.

        metric : str, optional
            The aggregation metric to use: "mean", "median", "max", or "min". Default is "mean".

        Returns
        -------
        np.ndarray
            A 2D array of shape (n_streamlines, n_features) containing aggregated values.
        """
        if not hasattr(self, "data_per_point") or map_name not in self.data_per_point:
            raise ValueError(f"Map '{map_name}' not found in data_per_point.")

        if streamline_map_name is None:
            streamline_map_name = map_name

        streamline_data = np.zeros(
            len(self.tracts), dtype=type(self.data_per_point[map_name][0][0])
        )
        for i, streamline in enumerate(self.tracts):
            point_values = self.data_per_point[map_name][i]

            if metric == "median":
                avg_value = np.median(point_values, axis=0)
            elif metric == "max":
                avg_value = np.max(point_values, axis=0)
            elif metric == "min":
                avg_value = np.min(point_values, axis=0)
            else:
                avg_value = np.mean(point_values, axis=0)

            streamline_data[i] = avg_value

        # Ensure it's 2D even for single scalar values
        if streamline_data.ndim == 1:
            streamline_data = streamline_data[:, np.newaxis]

        self.data_per_streamline[streamline_map_name] = streamline_data

        # Check if the map has colortable information and propagate it if so
        if map_name in self.colortables:
            self.colortables[streamline_map_name] = copy.deepcopy(
                self.colortables[map_name]
            )

        return streamline_data

    ####################################################################################################
    def add_tractogram(
        self, tract2add: Union["Tractogram", List["Tractogram"]]
    ) -> "Tractogram":
        """
        This method merges the current Tractogram with one or more other Tractogram objects.
        The resulting Tractogram contains all streamlines and associated metadata from the
        input Tractogram objects.

        Parameters
        ----------
        tractograms : Tractogram or list of Tractogram
            A single Tractogram object or a list of Tractogram objects to merge with the current one.

        Returns
        -------
            Tractogram:
                A new Tractogram object containing the merged streamlines and metadata.

        Raises
        ------
            TypeError: If any item in the input list is not a Tractogram object.

        Examples
        --------
        >>> tract1 = Tractogram('tract1.trk')
        >>> tract2 = Tractogram('tract2.trk')
        >>> merged_tract = tract1.add_tractogram(tract2)
        >>> print(f"Merged tractogram has {len(merged_tract.tracts)} streamlines")
        >>> merged_tract = tract1.add_tractogram('tract3.trk')
        >>> print(f"Merged tractogram has {len(merged_tract.tracts)} streamlines")

        """

        if isinstance(tract2add, (str, Path)):

            if isinstance(tract2add, str):
                if not os.path.isfile(tract2add):
                    raise FileNotFoundError(f"File '{tract2add}' not found")

            elif isinstance(tract2add, Path):
                # Check if Path is valid
                if not tract2add.exists():
                    raise FileNotFoundError(f"Path '{str(tract2add)}' does not exist")
                if not tract2add.is_file():
                    raise ValueError(f"Path '{str(tract2add)}' is not a file")

            # Load the tractogram from file
            tract2add = [copy.deepcopy(Tractogram(tract2add))]

        elif isinstance(tract2add, Tractogram):
            tract2add = [tract2add]

        if len(tract2add) == 0:
            raise ValueError("Tractograms list cannot be empty")

        # Check that all items in the list are tractogram objects
        for i, tract in enumerate(tract2add):
            if not isinstance(tract, Tractogram) and not isinstance(tract, str):
                raise TypeError(f"Item at index {i} is not a tractogram object")

        # Include this tractogram in the list
        all_tractograms = [self] + tract2add

        # Find common point_data fields across all tractograms
        common_data_per_points = None
        common_data_per_streamlines = None
        for tractogram in all_tractograms:
            current_data_per_points = set(tractogram.data_per_point.keys())

            if common_data_per_points is None:
                common_data_per_points = current_data_per_points
            else:
                common_data_per_points = common_data_per_points.intersection(
                    current_data_per_points
                )

            current_data_per_streamline = set(tractogram.data_per_streamline.keys())
            if common_data_per_streamlines is None:
                common_data_per_streamlines = current_data_per_streamline
            else:
                common_data_per_streamlines = common_data_per_streamlines.intersection(
                    current_data_per_streamline
                )

        # Convert to list for consistent ordering
        common_data_per_points = list(common_data_per_points)
        common_data_per_streamlines = list(common_data_per_streamlines)

        # Merge streamlines and metadata
        merged_streamlines = []
        tract_ids = []  # Changed variable name for clarity
        merged_data_per_point = {key: [] for key in common_data_per_points}
        merged_data_per_streamline = {key: [] for key in common_data_per_streamlines}
        merged_colortables = {}

        # Create a new colortable entry for tract IDs
        n_tractograms = len(all_tractograms)
        tract_id_colors = cltcol.create_distinguishable_colors(n_tractograms)

        merged_colortables["tract_id"] = {
            "names": [f"Tractogram_{i}" for i in range(n_tractograms)],
            "color_table": tract_id_colors,
            "lookup_table": None,
        }

        for id, tractogram in enumerate(all_tractograms):
            merged_streamlines.extend(tractogram.tracts)

            # Create tract_id for each streamline in this tractogram
            # Store as individual arrays (one per streamline) for consistency with data_per_streamline format
            for _ in range(len(tractogram.tracts)):
                tract_ids.append(np.array([id]))

            for key in common_data_per_points:
                merged_data_per_point[key].extend(tractogram.data_per_point[key])

            for key in common_data_per_streamlines:
                merged_data_per_streamline[key].extend(
                    tractogram.data_per_streamline[key]
                )

            for key, value in tractogram.colortables.items():
                if key in merged_colortables:
                    # If key already exists, keep the first one encountered
                    continue
                else:
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

        # Create new Tractogram object for the merged result
        merged_tractogram = Tractogram(
            tracts=ArraySequence(merged_streamlines),
            affine=self.affine,
            header=self.header,
            name="merged_tractogram",
        )
        merged_tractogram.data_per_point = merged_data_per_point
        merged_tractogram.data_per_streamline = merged_data_per_streamline

        # Add tract_id as ArraySequence (consistent with other data_per_streamline fields)
        merged_tractogram.data_per_streamline["tract_id"] = ArraySequence(tract_ids)
        merged_tractogram.colortables = merged_colortables

        return merged_tractogram

    ####################################################################################################
    def compute_centroids(
        self,
        method="qb",
        thresholds=[10],
    ):
        """
        Extract bundle centroids from tractogram and save them as .trk files.

        Parameters
        ----------
        method : str, optional
            Clustering method to use. Can be 'qbx' or 'qb'. Default is
            'qb'.

        nb_points : int, optional
            Number of points to resample the streamlines to. Default is 51.

        method : str, optional
            Clustering method to use. Can be 'qbx' or 'qb'. Default is 'qb'.

        thresholds : list of int, optional
            List of thresholds to use for clustering (only for qbx). Default is [10].
            If using 'qb', only the first threshold will be used.


        Returns
        -------
            centroids (ArraySequence):
                ArraySequence of centroids for each cluster.
            centroids_indexes (list of list of int):
                List of lists containing the indices of streamlines in each cluster.

        Raises
        ------
            ValueError: If the specified clustering method is not recognized.
            ValueError: If thresholds are not provided as a non-empty list for QuickBundlesX.

        Notes
        ------
            The centroids and their corresponding streamline indices are stored as attributes
            of the Tractogram object for later use.

        Examples
        --------
        >>> tractogram = Tractogram('input.trk')
        >>> centroids, indices = tractogram.compute_centroids(method='qb', thresholds=[10])
        >>> print(f"Computed {len(centroids)} centroids using QuickBundles with threshold 10")
        >>> ""
        >>> centroids_qbx, indices_qbx = tractogram.compute_centroids(method='qbx', thresholds=[20, 15, 10])
        >>> print(f"Computed {len(centroids_qbx)} centroids using QuickBundlesX with thresholds [20,
        15, 10]")

        """

        # === CLUSTERING ===
        if method == "qbx":
            if not isinstance(thresholds, list) or len(thresholds) == 0:
                raise ValueError(
                    "Thresholds must be a non-empty list for QuickBundlesX"
                )
            qbx = QuickBundlesX(thresholds)
            clusters = qbx.cluster(self.tracts)

        elif method == "qb":
            # Use the first threshold if provided, otherwise default to 10
            threshold = thresholds[0] if thresholds else 10
            qb = QuickBundles(threshold=threshold)
            clusters = qb.cluster(self.tracts)

        else:
            raise ValueError(f"Unknown clustering method: {method}. Use 'qb' or 'qbx'.")

        # Collect centroids and clustered streamlines with metadata
        centroids_list = []

        # Metadata for centroids
        centroids_indexes = []

        # Process each cluster
        for cluster in clusters:

            # Get the centroid of the cluster
            centroids_list.append(cluster.centroid)

            # Get the indices of streamlines in this cluster
            centroids_indexes.append(cluster.indices)

        # Convert to ArraySequence for nibabel
        self.centroids = ArraySequence(centroids_list)  # Store centroids in the object
        self.centroids_indexes = centroids_indexes  # Store indices in the object to use later for saving tracts for each cluster

        return self.centroids, self.centroids_indexes

    ####################################################################################################
    def label_streamlines_by_clusters(self):
        """
        Label each streamline in the tractogram with its corresponding cluster ID.

        This method assigns a cluster ID to each streamline based on the clustering
        results obtained from the `compute_centroids` method. The cluster IDs are stored
        in the `data_per_streamline` attribute of the Tractogram object under the key
        'cluster_id'.

        Returns:
        --------
            np.ndarray:
                Array of cluster IDs for each streamline in the tractogram.

        Raises:
        -------
            ValueError: If centroids or centroid indexes are not computed.

        Examples:
        ---------
        >>> tractogram = Tractogram('input.trk')
        >>> tractogram.compute_centroids(method='qb', thresholds=[10])
        >>> cluster_ids = tractogram.label_streamlines_by_clusters()
        >>> print(f"Assigned cluster IDs to {len(cluster_ids)} streamlines")


        """
        if not hasattr(self, "centroids") or not hasattr(self, "centroids_indexes"):
            raise ValueError(
                "Centroids and centroid indexes not computed. Please run compute_centroids() first."
            )

        n_streamlines = len(self.tracts)
        cluster_ids = np.full(n_streamlines, -1)  # Initialize with -1 (unlabeled)

        for cluster_id, indices in enumerate(self.centroids_indexes):
            for idx in indices:
                cluster_ids[idx] = cluster_id

        # Store the cluster IDs in data_per_streamline
        colors = cltcol.create_distinguishable_colors(len(self.centroids))
        colortable = cltcol.colors_to_table(
            colors=colors, alpha_values=1, values=range(len(self.centroids))
        )
        colortable[:, :3] = colortable[:, :3] / 255  # Ensure colors are between 0 and 1
        self.colortables["cluster_id"] = {
            "names": [f"cluster_{i}" for i in range(len(self.centroids))],
            "color_table": colortable,
            "lookup_table": None,
        }

        self.data_per_streamline["cluster_id"] = cluster_ids.reshape(
            -1, 1
        )  # Reshape to be a column vector

        return cluster_ids

    ###############################################################################################
    def get_cluster_streamlines(self, cluster_id: int) -> List[np.ndarray]:
        """
        Retrieve the streamlines belonging to a specific cluster.

        Parameters:
        -----------
            cluster_id (int):
                Index of the cluster to retrieve streamlines from.

        Returns:
        --------
            List[np.ndarray]:
                List of streamlines belonging to the specified cluster.

        Raises:
        -------
            ValueError: If centroids or centroid indexes are not computed.
            IndexError: If the specified cluster_id is out of range.

        Examples:
        ---------
        >>> tractogram = Tractogram('input.trk')
        >>> tractogram.compute_centroids(method='qb', thresholds=[10])
        >>> cluster_streamlines = tractogram.get_cluster_streamlines(cluster_id=0)
        >>> print(f"Cluster 0 has {len(cluster_streamlines)} streamlines")
        """
        if not hasattr(self, "centroids") or not hasattr(self, "centroids_indexes"):
            raise ValueError(
                "Centroids and centroid indexes not computed. Please run compute_centroids() first."
            )

        if cluster_id < 0 or cluster_id >= len(self.centroids):
            raise IndexError(
                f"Cluster ID {cluster_id} is out of range. Must be between 0 and {len(self.centroids)-1}."
            )

        # Get indices of streamlines in the specified cluster
        streamline_indices = self.centroids_indexes[cluster_id]

        # Retrieve the corresponding streamlines
        cluster_streamlines = [self.tracts[i] for i in streamline_indices]

        return cluster_streamlines

    ###############################################################################################
    def compute_streamline_lengths(self) -> np.ndarray:
        """
        Compute the lengths of all streamlines in the tractogram.

        This method both returns the lengths and stores them in data_per_streamline["length"].
        """
        if not hasattr(self, "tracts"):
            raise ValueError(
                "No streamlines loaded. Please ensure the tractogram file was loaded correctly."
            )
        else:
            if self.tracts is None or len(self.tracts) == 0:
                raise ValueError(
                    "Tractogram contains no streamlines. Please provide a valid tractogram file."
                )

        lengths = np.array(
            [np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1)) for s in self.tracts]
        )
        lengths = lengths.reshape(-1, 1)

        # Store in the object
        if not hasattr(self, "data_per_streamline"):
            self.data_per_streamline = {}
        self.data_per_streamline["length"] = lengths

        return lengths

    ##########################################################################################################
    def interpolate_on_tractogram(
        self,
        scal_map: str,
        interp_method: str = "linear",
        storage_mode: str = "data_per_point",
        map_name: str = "fa",
        reduction: str = "mean",
    ):
        """
        Interpolate scalar values (e.g., FA) from a NIfTI image onto a tractogram.

        This function loads a tractogram and a scalar map, then interpolates the scalar
        values at each streamline point or aggregates them per streamline. The resulting
        tractogram is saved with the interpolated values attached as metadata.

        Parameters
        ----------
        in_tract : str or Path
            Path to input .trk tractogram file. Must exist and be readable.

        scal_map : str or Path
            Path to scalar image (e.g., FA map in NIfTI format). Must exist and be readable.

        out_tract : str or Path
            Path to save the new tractogram with interpolated values. The parent directory
            must exist and be writable.

        interp_method : {'linear', 'nearest'}, default='linear'
            Interpolation method used for RegularGridInterpolator.
            - 'linear': Trilinear interpolation
            - 'nearest': Nearest neighbor interpolation

        storage_mode : {'data_per_point', 'data_per_streamline'}, default='data_per_point'
            Storage format for the interpolated values:
            - 'data_per_point': Store scalar value for each streamline point
            - 'data_per_streamline': Store aggregated scalar value per streamline

        map_name : str, default='fa'
            Name used for the scalar map in the output tractogram metadata.
            This will be the key in data_per_point or data_per_streamline.

        reduction : {'mean', 'median', 'min', 'max'}, default='mean'
            Aggregation method used when storage_mode='data_per_streamline'.
            Applied to all scalar values along each streamline to produce a single value.

        preserve_both_storage_modes : bool, default=False
            If True, preserve existing data in both data_per_point and data_per_streamline.
            **Warning**: This may cause visualization conflicts in some tools (FSLeyes, etc.)
            that have trouble rendering tractograms with both storage modes present.
            Use only when you specifically need both storage modes for different applications.

        Returns
        -------
        new_tractogram : nibabel.streamlines.Tractogram
            The tractogram object with interpolated scalar values attached.

        scalar_values_per_streamline : list of numpy.ndarray
            List containing the interpolated scalar values for each streamline.
            Each array has shape (n_points,) where n_points is the number of points
            in the corresponding streamline. Contains NaN for points outside the scalar map.

        Raises
        ------
        FileNotFoundError
            If input tractogram file or scalar map file does not exist.

        NotADirectoryError
            If the parent directory of the output path does not exist.

        PermissionError
            If the output directory is not writable.

        ValueError
            If interp_method is not 'linear' or 'nearest', if storage_mode is not
            'data_per_point' or 'data_per_streamline', or if reduction method is
            not one of 'mean', 'median', 'min', 'max'.

        IOError
            If there are issues reading the input files or writing the output file.

        Notes
        -----
        - Points outside the scalar map boundaries will have NaN values
        - Empty streamlines are handled gracefully with empty arrays
        - The function preserves the original tractogram's affine transformation
        - When using 'data_per_streamline' mode, the map name will be suffixed
        with the reduction method (e.g., 'fa_mean')
        - **Important**: By default, the function only populates the requested
        storage_mode and clears the other to prevent visualization conflicts.
        Many tools (FSLeyes, etc.) have trouble rendering tractograms with both
        data_per_point and data_per_streamline present simultaneously.
        - Set preserve_both_storage_modes=True only if you specifically need both
        storage modes for different applications, but be aware of potential
        visualization issues.

        Examples
        --------
        Basic usage with FA map:

        >>> new_tract, values = interpolate_on_tractogram(
        ...     'input.trk', 'fa_map.nii.gz', 'output_with_fa.trk',
        ...     map_name='fractional_anisotropy'
        ... )

        Using median aggregation per streamline:

        >>> new_tract, values = interpolate_on_tractogram(
        ...     'input.trk', 'md_map.nii.gz', 'output_with_md.trk',
        ...     storage_mode='data_per_streamline',
        ...     reduction='median',
        ...     map_name='mean_diffusivity'
        ... )

        Preserving both storage modes (use with caution):

        >>> new_tract, values = interpolate_on_tractogram(
        ...     'input.trk', 'fa_map.nii.gz', 'output_with_fa.trk',
        ...     preserve_both_storage_modes=True
        ... )
        # Warning: May cause visualization issues in FSLeyes and other tools
        """

        # --- Input validation ---
        scal_map = Path(scal_map)

        # Check if input files exist
        if not scal_map.exists():
            raise FileNotFoundError(f"Scalar map file not found: {scal_map}")

        # Validate parameters
        valid_interp_methods = ["linear", "nearest"]
        if interp_method not in valid_interp_methods:
            raise ValueError(
                f"Invalid interpolation method '{interp_method}'. "
                f"Choose from {valid_interp_methods}"
            )

        valid_storage_modes = ["data_per_point", "data_per_streamline"]
        if storage_mode not in valid_storage_modes:
            raise ValueError(
                f"Invalid storage mode '{storage_mode}'. "
                f"Choose from {valid_storage_modes}"
            )

        valid_reductions = ["mean", "median", "min", "max"]
        if reduction not in valid_reductions:
            raise ValueError(
                f"Invalid reduction method '{reduction}'. "
                f"Choose from {valid_reductions}"
            )

        streamlines = self.tracts

        # --- Load scalar image ---
        try:
            scalar_img = nb.load(str(scal_map))
        except Exception as e:
            raise IOError(f"Failed to load scalar map '{scal_map}': {e}")

        scalar_data = scalar_img.get_fdata()
        inv_affine = np.linalg.inv(scalar_img.affine)

        # Creating interpolation function
        x = np.arange(scalar_data.shape[0])
        y = np.arange(scalar_data.shape[1])
        z = np.arange(scalar_data.shape[2])
        my_interpolating_scalmap = RegularGridInterpolator(
            (x, y, z), scalar_data, method=interp_method
        )

        # --- Interpolate scalar values per streamline point ---
        scalar_values_per_point = []
        for sl in streamlines:
            if len(sl) == 0:
                scalar_values_per_point.append(np.array([]))
                continue

            ones = np.ones((len(sl), 1))
            coords_hom = np.hstack([sl, ones])
            voxel_coords = (inv_affine @ coords_hom.T).T[:, :3].T

            mask = (
                (voxel_coords[0] >= 0)
                & (voxel_coords[0] < scalar_data.shape[0])
                & (voxel_coords[1] >= 0)
                & (voxel_coords[1] < scalar_data.shape[1])
                & (voxel_coords[2] >= 0)
                & (voxel_coords[2] < scalar_data.shape[2])
            )

            values = np.full(voxel_coords.shape[1], np.nan)
            if np.any(mask):
                values[mask] = my_interpolating_scalmap(voxel_coords[:, mask].T)

            scalar_values_per_point.append(values)

        # --- Store scalar values ---
        # Handle storage mode conflicts - clear the other mode unless user explicitly wants both
        if storage_mode == "data_per_point":
            self.data_per_point[map_name] = scalar_values_per_point

        elif storage_mode == "data_per_streamline":
            reducer = {
                "mean": np.nanmean,
                "median": np.nanmedian,
                "min": np.nanmin,
                "max": np.nanmax,
            }[reduction]

            values = [
                reducer(v) if len(v) > 0 and not np.all(np.isnan(v)) else np.nan
                for v in scalar_values_per_point
            ]
            self.data_per_streamline[f"{map_name}_{reduction}"] = np.array(
                values
            ).reshape(-1, 1)

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
    ) -> None:
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
        map_list_dict = self.list_maps()

        st_maps = map_list_dict["maps_per_streamline"]
        pt_maps = map_list_dict["maps_per_point"]
        overlays = []
        if st_maps is not None:
            overlays = overlays + st_maps

        if pt_maps is not None:
            overlays = overlays + pt_maps

        if overlay_name not in overlays:
            raise ValueError(
                f"Overlay '{overlay_name}' not found. Available overlays: {', '.join(overlays)}"
            )

        # Getting the values of the overlay

        if overlay_name in st_maps:
            # Map the streamline values to points
            data = self.data_per_streamline[overlay_name]

        if overlay_name in pt_maps:
            data = self.data_per_point[overlay_name]

        # Concatenate all arrays into a single array for color mapping
        all_data = np.concatenate(data)
        lengths = [len(arr) for arr in data]

        # Calculate split indices (cumulative sum of lengths, excluding the last one)
        split_indices = np.cumsum(lengths)[:-1]

        # if colortables is an attribute of the class, use it
        if hasattr(self, "colortables"):
            dict_ctables = self.colortables

            # Check if the overlay is on the colortables
            if overlay_name in dict_ctables.keys():
                # Use the colortable associated with the parcellation

                point_colors = cltcol.get_colors_from_colortable(
                    all_data, self.colortables[overlay_name]["color_table"]
                )
            else:
                # Use the colormap for scalar data
                point_colors = cltcol.values2colors(
                    all_data,
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
                all_data,
                cmap=colormap,
                output_format="rgb",
                vmin=vmin,
                vmax=vmax,
                range_min=range_min,
                range_max=range_max,
                range_color=range_color,
            )

        # Split array_all back into a list of arrays
        point_colors = np.split(point_colors, split_indices)
        point_colors = ArraySequence(point_colors)

        return point_colors

    ###############################################################################################
    def reduce_streamlines(self, percentage: float = 50) -> None:
        """
        Reduces the number of streamline by a specified factor.

        Parameters:
        -----------
            percentage (np.float):
                Percentage of streamlines to keep (between 0 and 100).

        Returns:
        --------
            None

        Raises:
        -------
            ValueError: If percentage is not between 0 and 100.

        Notes:
        ------
            This method modifies the tractogram in place, reducing the number of streamlines
            by keeping every N-th streamline, where N is determined by the specified percentage.

        Examples:
        ---------
        >>> tractogram = Tractogram('input.trk')
        >>> tractogram.reduce_streamlines(percentage=25)
        >>> print(f"Reduced tractogram now has {len(tractogram.tracts)} streamlines")

        """

        if percentage <= 0 or percentage > 100:
            raise ValueError("Percentage must be between 0 and 100")

        # Calculate the reduction factor
        factor = np.ceil(100 / percentage).astype(int)

        # Reduce the streamlines
        reduced_streamlines = self.tracts[::factor]

        # Update the tractogram's streamlines and associated data, the maps and the header
        self.tracts = reduced_streamlines
        if hasattr(self, "data_per_point") and self.data_per_point:
            for key in self.data_per_point.keys():
                self.data_per_point[key] = self.data_per_point[key][::factor]
        if hasattr(self, "data_per_streamline") and self.data_per_streamline:
            for key in self.data_per_streamline.keys():
                self.data_per_streamline[key] = self.data_per_streamline[key][::factor]
        if hasattr(self, "header") and self.header:
            self.header["nb_streamlines"] = len(self.tracts)

    ###############################################################################################
    def filter(
        self,
        condition: str,
        **kwargs,
    ) -> Optional[List[np.ndarray]]:
        """
        Filters streamlines based on their lengths.

        Parameters:
        -----------
            condition (str):
                Condition string to filter streamlines. Supported formats:
                - 'length > X': Keep streamlines longer than X mm.
                - 'length < X': Keep streamlines shorter than X mm.
                - 'length >= X': Keep streamlines longer than or equal to X mm.
                - 'length <= X': Keep streamlines shorter than or equal to X mm.
                - 'length == X': Keep streamlines exactly X mm long.
                - 'length != X': Exclude streamlines exactly X mm long.
                - 'Xmin <= length <= Ymax': Keep streamlines between X and Y mm.

        Returns:
        --------
            filtered_streamlines (List[np.ndarray] or None):
                List of streamlines that match the condition. Returns None if no streamlines match.

        Raises:
        -------
            ValueError: If the condition format is invalid or if the specified map is not found.

        Notes:
        ------
            This method modifies the tractogram in place, updating the streamlines and associated
            data based on the filtering condition.

        """

        cond_parts = cltmisc.parse_condition(condition)
        map_name = cond_parts[0]

        # Check if the map exists in data_per_streamline
        # List the maps available in data_per_streamline
        map_list_dict = self.list_maps()
        if (map_name not in map_list_dict["maps_per_streamline"]) and (
            map_name not in map_list_dict["maps_per_point"]
        ):
            raise ValueError(
                f"Map '{map_name}' not found in data_per_point or data_per_streamline. "
                f"Available maps per point: {list(map_list_dict["maps_per_point"]) if map_list_dict["maps_per_point"] else 'None'}, "
                f"Available maps per streamline: {list(map_list_dict["maps_per_streamline"]) if map_list_dict["maps_per_streamline"] else 'None'}"
            )
        elif (map_name not in self.data_per_streamline) and (
            map_name in self.data_per_point
        ):
            # Compute the mean values per streamline and add to data_per_streamline
            mean_values = []
            for vals in self.data_per_point[map_name]:
                if len(vals) > 0:
                    mean_values.append(np.nanmean(vals))
                else:
                    mean_values.append(np.zeros((1,)))
            map_values = np.array(mean_values).reshape(-1, 1)
            kwargs.update({map_name: map_values})

        elif map_name in self.data_per_streamline:
            map_values = self.data_per_streamline[map_name]
            kwargs.update({map_name: map_values})

        indexes = cltmisc.get_indices_by_condition(condition, **kwargs)
        if len(indexes) == 0:
            print(f"No streamlines match the condition: {condition}")
            return None
        else:
            filtered_streamlines = [self.tracts[i] for i in indexes]
            print(
                f"Filtered {len(filtered_streamlines)} streamlines matching condition: {condition}"
            )
            # Update the tractogram's streamlines and associated data, the maps and the header
            self.tracts = filtered_streamlines
            if hasattr(self, "data_per_point") and self.data_per_point:
                for key in self.data_per_point.keys():
                    self.data_per_point[key] = [
                        self.data_per_point[key][i] for i in indexes
                    ]
            if hasattr(self, "data_per_streamline") and self.data_per_streamline:
                for key in self.data_per_streamline.keys():
                    self.data_per_streamline[key] = self.data_per_streamline[key][
                        indexes
                    ]
            if hasattr(self, "header") and self.header:
                self.header["nb_streamlines"] = len(self.tracts)

            # Remove it from the colortables if exists
            if hasattr(self, "colortables") and map_name in self.colortables:

                tmp_colortable = self.colortables[map_name]["color_table"]
                tmp_names = self.colortables[map_name]["names"]
                tmp_dict = {}
                tmp_dict[map_name] = tmp_colortable[
                    :, -1
                ]  # Assuming last column has the streamline IDs
                indexes = cltmisc.get_indices_by_condition(
                    condition, **{map_name: tmp_dict[map_name]}
                )
                # Filter the colortable and names
                filtered_colortable = np.array(
                    [
                        tmp_colortable[i]
                        for i in range(len(tmp_colortable))
                        if i not in indexes
                    ]
                )
                filtered_names = [
                    tmp_names[i] for i in range(len(tmp_names)) if i not in indexes
                ]
                self.colortables[map_name]["color_table"] = filtered_colortable
                self.colortables[map_name]["names"] = filtered_names

            return filtered_streamlines

    ###############################################################################################
    def list_maps(self) -> List[str]:
        """
        Lists all available scalar maps in the tractogram.

        Returns:
        --------
            maps_per_point (set or None):
                Set of scalar map names stored per point. None if no maps are available.

            maps_per_streamline (set or None):
                Set of scalar map names stored per streamline. None if no maps are available.

        Examples:
        ---------
        >>> tractogram = Tractogram('input.trk')
        >>> maps = tractogram.list_streamlines_maps()
        >>> print("Available scalar maps:", maps)
        """
        maps_per_point = []
        maps_per_streamline = []

        if hasattr(self, "data_per_point"):
            if self.data_per_point:
                maps_per_point = maps_per_point + list(self.data_per_point.keys())
            else:
                maps_per_point = None

        if hasattr(self, "data_per_streamline"):
            # if self.data_per_streamline is not an empty dict
            if self.data_per_streamline:
                maps_per_streamline = maps_per_streamline + list(
                    self.data_per_streamline.keys()
                )
            else:
                maps_per_streamline = None

        map_list = {
            "maps_per_point": maps_per_point,
            "maps_per_streamline": maps_per_streamline,
        }

        return map_list

    ##############################################################################################
    def get_maps_info(self) -> pd.DataFrame:
        """
        Retrieves information about scalar maps stored in the tractogram.

        Returns:
        --------
            pd.DataFrame:
                DataFrame containing map names, storage modes, and value statistics.

        Examples:
        ---------
        >>> tractogram = Tractogram('input.trk')
        >>> maps_info = tractogram.get_maps_info()
        >>> print(maps_info)
        """
        maps_info = []

        if hasattr(self, "data_per_point") and self.data_per_point:
            for key, values in self.data_per_point.items():
                all_values = np.concatenate(values) if values else np.array([])
                maps_info.append(
                    {
                        "map_name": key,
                        "storage_mode": "data_per_point",
                        "number_of_streamlines": len(values),
                        "num_values": len(all_values),
                        "min_value": (
                            np.nanmin(all_values) if len(all_values) > 0 else np.nan
                        ),
                        "max_value": (
                            np.nanmax(all_values) if len(all_values) > 0 else np.nan
                        ),
                        "mean_value": (
                            np.nanmean(all_values) if len(all_values) > 0 else np.nan
                        ),
                        "std_value": (
                            np.nanstd(all_values) if len(all_values) > 0 else np.nan
                        ),
                    }
                )

        if hasattr(self, "data_per_streamline") and self.data_per_streamline:
            for key, values in self.data_per_streamline.items():
                all_values = values.flatten() if values is not None else np.array([])
                maps_info.append(
                    {
                        "map_name": key,
                        "storage_mode": "data_per_streamline",
                        "number_of_streamlines": len(all_values),
                        "num_values": len(all_values),
                        "min_value": (
                            np.nanmin(all_values) if len(all_values) > 0 else np.nan
                        ),
                        "max_value": (
                            np.nanmax(all_values) if len(all_values) > 0 else np.nan
                        ),
                        "mean_value": (
                            np.nanmean(all_values) if len(all_values) > 0 else np.nan
                        ),
                        "std_value": (
                            np.nanstd(all_values) if len(all_values) > 0 else np.nan
                        ),
                    }
                )

        return pd.DataFrame(maps_info)

    ##############################################################################################
    def get_tractogram_info(self) -> Dict:
        """
        Retrieves basic information about the tractogram object.

        Returns:
        --------
            Dict:
                Dictionary containing number of streamlines, affine matrix, and header info.
        """
        n_streamlines = len(self.tracts) if self.tracts is not None else 0
        affine = self.affine if hasattr(self, "affine") else None
        mean_length = (
            np.mean(self.compute_streamline_lengths())
            if hasattr(self, "tracts")
            and self.tracts is not None
            and len(self.tracts) > 0
            else None
        )

        info = {
            "number_of_streamlines": n_streamlines,
            "mean_streamline_length": mean_length,
            "affine": affine,
            "header": self.header if hasattr(self, "header") else None,
        }
        return info

    ##############################################################################################
    def centroids_to_tractogram(self):
        """
        Converts the computed centroids into a new tractogram object.

        Returns:
        --------
            nb.streamlines.Tractogram:
                A new tractogram object containing the centroids as streamlines.

        Raises:
        -------
            ValueError: If centroids have not been computed yet.

        Examples:
        ---------
        >>> tractogram = Tractogram('input.trk')
        >>> tractogram.compute_centroids(method='qb', thresholds=[10])
        >>> centroids_tractogram = tractogram.centroids_to_tractogram()
        >>> print(f"Centroids tractogram has {len(centroids_tractogram.tractogram.streamlines)} streamlines")
        """

        if not hasattr(self, "centroids") or self.centroids is None:
            raise ValueError(
                "Centroids have not been computed yet. Please run compute_centroids() first."
            )

        centroids_tractogram = copy.deepcopy(self)
        centroids_tractogram.tracts = ArraySequence(copy.deepcopy(self.centroids))
        centroids_tractogram.affine = copy.deepcopy(self.affine)
        centroids_tractogram.header = copy.deepcopy(self.header)
        centroids_tractogram.data_per_point = {}
        centroids_tractogram.data_per_streamline = {}
        # Update header information
        if (
            centroids_tractogram.header
            and "nb_streamlines" in centroids_tractogram.header
        ):
            centroids_tractogram.header["nb_streamlines"] = len(self.centroids)

        centroids_tractogram.compute_streamline_lengths()

        return centroids_tractogram

    ##############################################################################################
    def save_tractogram(
        self,
        out_file: str,
        tracts: Optional[List[np.ndarray]] = None,
        affine: Optional[np.ndarray] = None,
        header: Optional[Dict] = None,
        file_type: str = "trk",
        overwrite: bool = False,
    ):
        """
        Saves the tractogram to a specified file.

        Parameters:
        -----------
            out_file (str):
                Path to the output tractogram file.

            tracts (Optional[List[np.ndarray]]):
                List of streamlines to save. If None, uses the current tracts.

            affine (Optional[np.ndarray]):
                Affine transformation matrix. If None, uses the current affine.

            header (Optional[Dict]):
                Header information. If None, uses the current header.

            file_type (str):
                Type of the output file ('tck' or 'trk'). Default is 'trk'.

            overwrite (bool):
                Whether to overwrite the existing file. Default is False.

        Returns:
        --------
            None

        Raises:
        -------
            ValueError: If the specified file type is not supported.

            FileExistsError: If the output file already exists and overwrite is False.

        Examples:
        ---------
        >>> tractogram = Tractogram('input.trk')
        >>> tractogram.save_tractogram('output.trk', file_type='trk', overwrite=True)
        >>> print("Tractogram saved to output.trk")
        """

        if tracts is None:
            tracts = self.tracts

        if affine is None:
            affine = self.affine

        if header is None:
            header = self.header

        if file_type not in ["tck", "trk"]:
            raise ValueError("Unsupported file type. Please specify 'tck' or 'trk'.")

        if os.path.isfile(out_file) and not overwrite:
            raise FileExistsError(
                f"The output file already exists: {out_file}. Use overwrite=True to overwrite it."
            )

        # Create a new tractogram object
        new_tractogram = nb.streamlines.Tractogram(tracts, affine_to_rasmm=affine)

        # Add data_per_point if available - ALWAYS rebuild ArraySequence
        if hasattr(self, "data_per_point") and self.data_per_point:
            for key, value in self.data_per_point.items():
                formatted_data = []

                # Handle different input types consistently
                if isinstance(value, ArraySequence):
                    # Convert ArraySequence to list first
                    value_list = list(value)
                elif isinstance(value, list):
                    value_list = value
                else:
                    # Handle other formats (numpy arrays, etc.)
                    raise TypeError(
                        f"Unsupported data_per_point format for key '{key}': {type(value)}"
                    )

                # Ensure each array in the list is properly formatted
                for arr in value_list:
                    arr = np.asarray(arr)
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    formatted_data.append(arr)

                # Always create a new ArraySequence to match current streamlines
                new_tractogram.data_per_point[key] = ArraySequence(formatted_data)

        # Add data_per_streamline if available
        if hasattr(self, "data_per_streamline") and self.data_per_streamline:
            for key, value in self.data_per_streamline.items():
                # Ensure proper numpy array format
                value = np.asarray(value)
                if value.ndim == 1:
                    value = value.reshape(-1, 1)
                new_tractogram.data_per_streamline[key] = value

        # Save the tractogram using nibabel
        if file_type == "tck":
            # Check if it finishes with .tck, if not, add it
            if not out_file.endswith(".tck"):
                out_file = os.path.splitext(out_file)[0] + ".tck"

            nb.streamlines.save(new_tractogram, out_file, header=header)
        elif file_type == "trk":
            # Check if it finishes with .trk, if not, add it
            if not out_file.endswith(".trk"):
                # Replace the extension with .trk
                out_file = os.path.splitext(out_file)[0] + ".trk"

            nb.streamlines.save(new_tractogram, out_file, header=header)

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
        plot_style: str = "tube",
        vis_percentage: float = 100,
        force_reduction: bool = True,
        notebook: bool = False,
        show_colorbar: bool = False,
        colorbar_title: str = None,
        colorbar_position: str = "bottom",
        save_path: str = None,
    ):
        """
        Plot the tractrogram with specified overlay and visualization parameters.

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

        # Reduce streamlines if needed for visualization
        n_streamlines = len(self.tracts)
        if vis_percentage < 100:
            self.reduce_streamlines(percentage=vis_percentage)

        if n_streamlines > 100000 and force_reduction:
            # Reduce to 100k streamlines
            reduction_percentage = (100000 / n_streamlines) * 100
            print(
                f"Number of streamlines is {n_streamlines}, reducing to {int(reduction_percentage)}% for faster visualization."
            )
            self.reduce_streamlines(percentage=reduction_percentage)

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
        )


################################# Helper Functions ################################
def resample_streamlines(
    in_streamlines: Union[
        List[np.ndarray], nb.streamlines.array_sequence.ArraySequence
    ],
    nb_points: int = 51,
) -> Union[List[np.ndarray], nb.streamlines.array_sequence.ArraySequence]:
    """
    Resample streamlines to a specified number of points.

    Parameters
    ----------
    in_streamlines : List[np.ndarray] or nb.streamlines.array_sequence.ArraySequence
        Input streamlines to be resampled.

    nb_points : int, optional
        Number of points to resample each streamline to. Default is 51.

    Returns
    -------
    resampled_streamlines : List[np.ndarray] or nb.streamlines.array_sequence.ArraySequence
        Resampled streamlines in the same format as the input.

    Raises
    ------
    ValueError
        If the input streamlines are not in the expected format.
    ValueError
        If the input streamlines are empty.
    ValueError
        If nb_points is not a positive integer.
    ValueError
        If individual streamlines are not valid numpy arrays.

    Examples
    --------
    >>> # With ArraySequence
    >>> in_streamlines = nb.streamlines.load('input.trk').streamlines
    >>> resampled = resample_streamlines(in_streamlines, nb_points=100)

    >>> # With list
    >>> streamlines_list = [np.random.rand(50, 3), np.random.rand(30, 3)]
    >>> resampled = resample_streamlines(streamlines_list, nb_points=100)
    """
    # Check if input is valid format
    is_array_sequence = isinstance(
        in_streamlines, nb.streamlines.array_sequence.ArraySequence
    )
    is_list = isinstance(in_streamlines, list)

    if not (is_array_sequence or is_list):
        raise ValueError(
            "Input streamlines must be either a list of numpy arrays or nibabel ArraySequence."
        )

    # Check if the input streamlines are empty
    if len(in_streamlines) == 0:
        raise ValueError(
            "Input streamlines are empty. Please provide valid streamlines."
        )

    # Check if nb_points is a positive integer
    if not isinstance(nb_points, int) or nb_points <= 0:
        raise ValueError("Number of points (nb_points) must be a positive integer.")

    # Validate individual streamlines
    for i, streamline in enumerate(in_streamlines):
        if not isinstance(streamline, np.ndarray):
            raise ValueError(f"Streamline {i} is not a valid numpy array.")

        if streamline.ndim != 2 or streamline.shape[1] != 3:
            raise ValueError(
                f"Streamline {i} must be a 2D array with shape (n_points, 3), "
                f"but has shape {streamline.shape}."
            )

        if streamline.shape[0] < 2:
            raise ValueError(
                f"Streamline {i} must have at least 2 points for resampling, "
                f"but has {streamline.shape[0]} points."
            )

    # Resample streamlines based on input type
    if is_array_sequence:
        # Use dipy's set_number_of_points directly for ArraySequence
        resampled_streamlines = set_number_of_points(in_streamlines, nb_points)
    else:
        # Handle list input
        resampled_streamlines = [
            set_number_of_points([streamline], nb_points)[0]
            for streamline in in_streamlines
        ]

    return resampled_streamlines


###############################################################################################
def interpolate_streamline_values(
    original_coords, original_values, new_coords, method="linear"
):
    """
    Interpolate values from original streamline coordinates to new coordinates.

    Parameters:
    -----------
        original_coords : np.ndarray
            Original coordinates with shape (n_points, 3) [X, Y, Z]

        original_values : np.ndarray
            Values at original coordinates with shape (n_points,) or (n_points, n_features)

        new_coords : np.ndarray
            New coordinates with shape (m_points, 3) [Xi, Yi, Zi]

        method : str
            Interpolation method: 'linear', 'nearest', 'cubic'. Default is 'linear'

    Returns:
    --------
        new_values : np.ndarray
            Interpolated values at new coordinates
    """
    from scipy.interpolate import interp1d

    # Calculate cumulative distance along original streamline
    orig_diffs = np.diff(original_coords, axis=0)
    orig_distances = np.cumsum(np.linalg.norm(orig_diffs, axis=1))
    orig_distances = np.concatenate([[0], orig_distances])  # Start at 0

    # Calculate cumulative distance along new streamline
    new_diffs = np.diff(new_coords, axis=0)
    new_distances = np.cumsum(np.linalg.norm(new_diffs, axis=1))
    new_distances = np.concatenate([[0], new_distances])  # Start at 0

    # Scale new distances to match original range
    new_distances = new_distances * (orig_distances[-1] / new_distances[-1])

    # Handle 1D vs 2D values
    original_values = np.asarray(original_values)
    if original_values.ndim == 1:
        # 1D case
        interpolator = interp1d(
            orig_distances,
            original_values,
            kind=method,
            bounds_error=False,
            fill_value="extrapolate",
        )
        new_values = interpolator(new_distances)
    else:
        # 2D case - interpolate each feature separately
        new_values = np.zeros((len(new_distances), original_values.shape[1]))
        for i in range(original_values.shape[1]):
            interpolator = interp1d(
                orig_distances,
                original_values[:, i],
                kind=method,
                bounds_error=False,
                fill_value="extrapolate",
            )
            new_values[:, i] = interpolator(new_distances)

    return new_values


###############################################################################################
def merge_tractograms(
    tractograms: List[Union[str, Path, Tractogram]],
    color_table: dict = None,
    map_name: str = "tract_id",
) -> Union[Tractogram, None]:
    """
    Merges multiple tractograms into a single tractogram.

    It combines streamlines and associated data from all input tractograms. If the input
    list is empty, it returns None. If there's only one tractogram, it returns it as is.

    Parameters:
    -----------
        tractograms : List[Union[str, Path, Tractogram]]
            List of Tractogram objects to be merged. Can also include file paths to tractogram files.

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

    if any(not isinstance(surf, (str, Path, Tractogram)) for surf in tractograms):
        raise TypeError(
            "All items in tractograms must be Tractogran objects, file paths, or Path objects"
        )

    # If the list is empty, return None
    if not tractograms:
        return None

    # If there's only one surface, return it as is
    if len(tractograms) == 1:
        return Tractogram(tractograms[0])

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
        colors = cltcol.create_distinguishable_colors(n_tracts)
        color_table = cltcol.colors_to_table(
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
