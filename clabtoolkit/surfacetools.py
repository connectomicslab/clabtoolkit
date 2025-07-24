import os
import numpy as np
import nibabel as nib
from typing import Union, List, Dict, Optional, Tuple
import pyvista as pv
import pandas as pd
import copy

# Importing local modules
from . import freesurfertools as cltfree
from . import visualizationtools as cltvis
from . import misctools as cltmisc


class Surface:
    """
    Surface class for loading and visualizing brain surface data with improved
    color table management and integration with AnnotParcellation.

    This class provides a comprehensive interface for working with brain surface
    data, including loading geometries, applying scalar maps, managing parcellations,
    and creating visualizations using PyVista.

    Attributes
    ----------
    surf : str
        Path to the surface file
    mesh : pv.PolyData
        PyVista mesh object containing the surface geometry and data
    hemi : str
        Hemisphere designation ('lh', 'rh', or 'unknown')
    colortables : Dict[str, Dict]
        Dictionary storing color table information for parcellations

    Examples
    --------
    >>> # Load a surface file
    >>> surface = Surface("path/to/lh.pial")
    >>>
    >>> # Load an annotation
    >>> surface.load_annotation("path/to/lh.aparc.annot", "aparc")
    >>>
    >>> # Load scalar data
    >>> surface.load_scalar_map("path/to/thickness.mgh", "thickness")
    >>>
    >>> # Visualize
    >>> surface.show(overlay_name="thickness", cmap="hot")
    """

    def __init__(self, surface_file: str, hemi: str = None) -> None:
        """
        Initialize a Surface object from a surface file.

        Parameters
        ----------
        surface_file : str
            Path to the surface file (e.g., FreeSurfer .pial, .white, .inflated files)
        hemi : str, optional
            Hemisphere designation ('lh' or 'rh'). If None, attempts to auto-detect
            from the filename or defaults to 'lh'

        Raises
        ------
        FileNotFoundError
            If the surface file does not exist
        ValueError
            If the surface file cannot be loaded or parsed

        Examples
        --------
        >>> # Load with auto-detection of hemisphere
        >>> surface = Surface("path/to/lh.pial")
        >>>
        >>> # Load with explicit hemisphere specification
        >>> surface = Surface("path/to/surface.pial", hemi="rh")
        """
        self.surf = surface_file
        self.mesh = self.load_surf()

        # Initialize color table storage BEFORE calling _store_parcellation_data
        self.colortables: Dict[str, Dict] = {}

        # Now create the temporary color table and store parcellation data
        tmp_ctable = cltfree.colors2colortable(
            np.array([[240, 240, 240]], dtype=np.uint8)
        )
        self._store_parcellation_data(
            np.ones((self.mesh.n_points,), dtype=np.uint32),
            tmp_ctable,
            ["surface"],
            "surface",
        )

        # Hemisphere detection
        if hemi is not None:
            self.hemi = hemi
        else:
            self.hemi = cltfree.detect_hemi(self.surf)

        # Fallback hemisphere detection from BIDS organization
        surf_name = os.path.basename(self.surf)
        detected_hemi = cltfree.detect_hemi(surf_name)

        if detected_hemi is None:
            self.hemi = "lh"  # Default to left hemisphere

    def load_surf(self) -> pv.PolyData:
        """
        Load surface geometry and create PyVista mesh.

        Returns
        -------
        pv.PolyData
            PyVista mesh object containing vertices, faces, and default surface colors

        Raises
        ------
        FileNotFoundError
            If the surface file cannot be found
        ValueError
            If the surface file format is not supported or corrupted

        Examples
        --------
        >>> surface = Surface("path/to/lh.pial")
        >>> mesh = surface.load_surf()
        >>> print(f"Number of vertices: {mesh.n_points}")
        >>> print(f"Number of faces: {mesh.n_cells}")
        """
        vertices, faces = nib.freesurfer.read_geometry(self.surf)

        # Add column with 3's to faces array for PyVista
        faces = np.c_[np.full(len(faces), 3), faces]

        mesh = pv.PolyData(vertices, faces)
        mesh.point_data["surface"] = (
            np.ones((len(vertices), 3), dtype=np.float32) * 240
        )  # Default colors

        return mesh

    def load_maps_from_csv(
        self,
        map_file: str,
        annot_file: Union[str, cltfree.AnnotParcellation] = None,
        map_name: str = None,
        cmap: str = "viridis",
    ) -> None:
        """
        Load scalar data (maps) from a CSV file and optionally an annotation file.

        This method can handle two scenarios:
        1. CSV with vertex-wise data (number of rows = number of vertices)
        2. CSV with region-wise data that needs to be mapped using an annotation

        Parameters
        ----------
        map_file : str
            Path to the CSV file containing scalar data
        annot_file : str or AnnotParcellation, optional
            Path to the annotation file or an AnnotParcellation object.
            Required if CSV contains region-wise data
        map_name : str, optional
            Name of the scalar data for reference. If None, uses column names from CSV
        cmap : str, default "viridis"
            Colormap name for visualizing the scalar data

        Raises
        ------
        FileNotFoundError
            If the map file or annotation file cannot be found
        ValueError
            If annot_file is required but not provided, or if it's not a valid type

        Examples
        --------
        >>> # Load vertex-wise data
        >>> surface.load_maps_from_csv("vertex_data.csv")
        >>>
        >>> # Load region-wise data with annotation
        >>> surface.load_maps_from_csv("region_data.csv", "lh.aparc.annot")
        >>>
        >>> # Load with custom map name
        >>> surface.load_maps_from_csv("data.csv", map_name="my_metric")
        """
        if not os.path.isfile(map_file):
            raise FileNotFoundError(f"Map file not found: {map_file}")

        maps_df = cltmisc.smart_read_table(map_file)

        # If the number oof rows of the dataframe is equal to the number of vertices, we can use it directly
        if maps_df.shape[0] == self.mesh.n_points:
            self.load_arrays_of_maps(maps_df)
        else:
            if annot_file is None:
                raise ValueError(
                    "annot_file must be provided if map_file does not match the number of vertices"
                )
            if not isinstance(annot_file, (str, cltfree.AnnotParcellation)):
                raise ValueError(
                    "annot_file must be a string or an AnnotParcellation object"
                )
            # If map_name is not provided, use the first column name from the DataFrame
            if isinstance(annot_file, str):
                # Check if the annotation file exists
                if not os.path.isfile(annot_file):
                    raise FileNotFoundError(f"Annotation file not found: {annot_file}")
                else:
                    parc = cltfree.AnnotParcellation(annot_file)
            elif isinstance(annot_file, cltfree.AnnotParcellation):
                parc = copy.deepcopy(
                    annot_file
                )  # Use a copy to avoid modifying the original object

            vertex_maps = parc.map_values(regional_values=maps_df, is_dataframe=True)
            self.load_arrays_of_maps(
                vertex_maps,
                map_names=maps_df.columns.tolist() if map_name is None else map_name,
            )

    def load_arrays_of_maps(
        self,
        maps_array: Union[np.ndarray, pd.DataFrame],
        map_names: Union[str, List[str]] = None,
        annot_file: Union[str, cltfree.AnnotParcellation] = None,
    ) -> None:
        """
        Load scalar data (maps) onto the surface for visualization.

        This method handles both vertex-wise and region-wise data arrays,
        automatically detecting the format and applying appropriate processing.

        Parameters
        ----------
        maps_array : np.ndarray or pd.DataFrame
            Array or DataFrame containing scalar data. Can be:
            - 1D array: Single map with one value per vertex
            - 2D array: Multiple maps with shape (n_vertices, n_maps)
            - DataFrame: Columns represent different maps
        map_names : str or List[str], optional
            Names for the scalar data. If not provided, default names will be generated.
            For DataFrame input, column names are used by default
        annot_file : str or AnnotParcellation, optional
            Annotation file or object for mapping region-wise data to vertices.
            Required if maps_array length doesn't match vertex count

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If maps_array is not a numpy array or pandas DataFrame
        ValueError
            If map_names is not a string or list of strings
        ValueError
            If the length of map_names does not match the number of columns in maps_array
        ValueError
            If annot_file is required but not provided or invalid
        FileNotFoundError
            If the annotation file cannot be found

        Examples
        --------
        >>> # Load single map as 1D array
        >>> data = np.random.rand(surface.mesh.n_points)
        >>> surface.load_arrays_of_maps(data, map_names="random_map")
        >>>
        >>> # Load multiple maps as 2D array
        >>> data = np.random.rand(surface.mesh.n_points, 3)
        >>> surface.load_arrays_of_maps(data, map_names=["map1", "map2", "map3"])
        >>>
        >>> # Load from DataFrame
        >>> df = pd.DataFrame({"thickness": thickness_data, "curvature": curv_data})
        >>> surface.load_arrays_of_maps(df)
        """

        if not isinstance(maps_array, np.ndarray) and not isinstance(
            maps_array, pd.DataFrame
        ):
            raise ValueError("maps_array must be a numpy array or a pandas DataFrame")

        if isinstance(maps_array, pd.DataFrame):
            # Convert DataFrame to numpy array
            map_names = maps_array.columns.tolist() if map_names is None else map_names
            maps_array = maps_array.to_numpy()

        # Check if maps_array has only one dimension if this is the case
        if maps_array.ndim == 1:
            maps_array = maps_array[:, np.newaxis]

        if map_names is not None:
            if isinstance(map_names, str):
                map_names = [map_names]
            elif not isinstance(map_names, list):
                raise ValueError("map_names must be a string or a list of strings")

            if len(map_names) != maps_array.shape[1]:
                raise ValueError(
                    "Length of map_names must match the number of columns in maps_array"
                )
        else:
            # Generate default names if not provided
            map_names = [f"map_{i}" for i in range(maps_array.shape[1])]

        if maps_array.shape[0] == self.mesh.n_points:
            # Store scalar data
            for i, map_name in enumerate(map_names):

                # Ensure the map data is a 1D array
                map_data = maps_array[:, i]

                # Store the map data in the mesh point data
                self.mesh.point_data[map_name] = map_data

        else:
            if annot_file is None:
                raise ValueError(
                    "annot_file must be provided if maps_array does not match the number of vertices"
                )
            if not isinstance(annot_file, (str, cltfree.AnnotParcellation)):
                raise ValueError(
                    "annot_file must be a string or an AnnotParcellation object"
                )

            # If map_names is not provided, use the first column name from the DataFrame
            if isinstance(annot_file, str):
                # Check if the annotation file exists
                if not os.path.isfile(annot_file):
                    raise FileNotFoundError(f"Annotation file not found: {annot_file}")
                else:
                    parc = cltfree.AnnotParcellation(annot_file)
            elif isinstance(annot_file, cltfree.AnnotParcellation):
                parc = copy.deepcopy(annot_file)

            vertex_maps = parc.map_values(
                regional_values=maps_array, is_dataframe=False
            )
            for i, map_name in enumerate(map_names):
                # Ensure the map data is a 1D array
                map_data = vertex_maps[:, i]

                # Store the map data in the mesh point data
                self.mesh.point_data[map_name] = map_data

    def load_scalar_map(self, map_file: str, map_name: str) -> None:
        """
        Load scalar data (map) onto the surface for visualization.

        This method loads FreeSurfer-format scalar files (e.g., .mgh, .mgz, .curv files)
        and stores the data as vertex-wise values on the surface.

        Parameters
        ----------
        map_file : str
            Path to the scalar data file (FreeSurfer format)
        map_name : str
            Name of the scalar data for reference and visualization

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the map file cannot be found
        ValueError
            If map_name is not a string
        ValueError
            If the scalar data cannot be loaded or doesn't match surface dimensions

        Examples
        --------
        >>> # Load cortical thickness data
        >>> surface.load_scalar_map("lh.thickness.mgh", "thickness")
        >>>
        >>> # Load curvature data
        >>> surface.load_scalar_map("lh.curv", "curvature")
        >>>
        >>> # Load custom scalar data
        >>> surface.load_scalar_map("custom_metric.mgh", "my_metric")
        """
        if not os.path.isfile(map_file):
            raise FileNotFoundError(f"Map file not found: {map_file}")

        if not isinstance(map_name, str):
            raise ValueError("map_name must be a string")

        # Read the map file
        map_data = nib.freesurfer.read_morph_data(map_file)

        # Store scalar data
        self.mesh.point_data[map_name] = map_data

    def load_annotation(
        self, annot_input: Union[str, "AnnotParcellation"], parc_name: str
    ) -> None:
        """
        Load annotation file or AnnotParcellation object onto the surface.

        This method loads parcellation data from FreeSurfer annotation files
        or AnnotParcellation objects, storing the labels and associated color
        information for visualization.

        Parameters
        ----------
        annot_input : str or AnnotParcellation
            Path to the annotation file (.annot) or an AnnotParcellation object
        parc_name : str
            Name for the parcellation (used for referencing in visualizations)

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the annotation file cannot be found
        ValueError
            If annot_input is not a valid file path or AnnotParcellation object
        ValueError
            If the number of vertices in annotation doesn't match the surface

        Examples
        --------
        >>> # Load FreeSurfer parcellation
        >>> surface.load_annotation("lh.aparc.annot", "aparc")
        >>>
        >>> # Load Destrieux parcellation
        >>> surface.load_annotation("lh.aparc.a2009s.annot", "destrieux")
        >>>
        >>> # Load from AnnotParcellation object
        >>> annot_obj = cltfree.AnnotParcellation("lh.aparc.annot")
        >>> surface.load_annotation(annot_obj, "aparc")
        """
        # Handle different input types
        if isinstance(annot_input, str):
            # Input is a file path
            if not os.path.isfile(annot_input):
                raise FileNotFoundError(f"Annotation file not found: {annot_input}")

            # Create AnnotParcellation object to benefit from its processing and cleaning
            annot_parc = cltfree.AnnotParcellation(annot_input)

        elif (
            hasattr(annot_input, "codes")
            and hasattr(annot_input, "regtable")
            and hasattr(annot_input, "regnames")
        ):
            # Input is an AnnotParcellation object
            annot_parc = annot_input
        else:
            raise ValueError(
                "annot_input must be either a file path (str) or an AnnotParcellation object"
            )

        # Extract the processed and cleaned data from AnnotParcellation
        labels = annot_parc.codes
        reg_ctable = annot_parc.regtable
        reg_names = annot_parc.regnames  # Already processed as strings

        # Validate that the number of vertices matches
        if len(labels) != self.mesh.n_points:
            raise ValueError(
                f"Number of vertices in annotation ({len(labels)}) does not match surface ({self.mesh.n_points})"
            )

        # Store the parcellation data
        self._store_parcellation_data(labels, reg_ctable, reg_names, parc_name)

        # Store reference to AnnotParcellation object for advanced operations
        self.colortables[parc_name]["annot_object"] = annot_parc

    def _store_parcellation_data(
        self,
        labels: np.ndarray,
        reg_ctable: np.ndarray,
        reg_names: List[str],
        parc_name: str,
    ) -> None:
        """
        Internal method to store parcellation data and create color mappings.

        Parameters
        ----------
        labels : np.ndarray
            Array of label values for each vertex
        reg_ctable : np.ndarray
            Color table array with RGBA values for each region
        reg_names : List[str]
            List of region names corresponding to the color table
        parc_name : str
            Name of the parcellation

        Returns
        -------
        None
        """
        # Store labels in mesh
        self.mesh.point_data[parc_name] = labels

        # Store parcellation information in organized structure
        self.colortables[parc_name] = {
            "struct_names": reg_names,
            "color_table": reg_ctable,
            "lookup_table": None,  # Will be populated by _create_parcellation_colortable if needed
        }

        # Create discrete color table for regions if needed
        self._create_parcellation_colortable(reg_ctable, reg_names, parc_name)

    def _create_parcellation_colortable(
        self, reg_ctable: np.ndarray, reg_names: List[str], parc_name: str
    ) -> None:
        """
        Create a PyVista color table for the parcellation.

        This method creates visualization-ready color tables for parcellation data.

        Parameters
        ----------
        reg_ctable : np.ndarray
            Color table array with RGBA values for each region
        reg_names : List[str]
            List of region names corresponding to the color table
        parc_name : str
            Name of the parcellation

        Returns
        -------
        None

        Notes
        -----
        This is a placeholder implementation that needs to be completed
        based on specific visualization requirements.
        """
        # Placeholder implementation - you'll need to implement this
        # based on how you want to create the PyVista LookupTable
        pass

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

    def list_overlays(self) -> Dict[str, str]:
        """
        List all available overlays and their types.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping overlay names to their types:
            - "scalar": 1D array of scalar values
            - "color": 3D array of RGB color values
            - "unknown": Arrays with other dimensions

        Examples
        --------
        >>> overlays = surface.list_overlays()
        >>> print(overlays)
        {'surface': 'color', 'thickness': 'scalar', 'aparc': 'scalar'}
        """
        overlays = {}

        for key in self.mesh.point_data.keys():
            tmp = self.mesh.point_data[key]
            if isinstance(tmp, np.ndarray) and tmp.ndim == 1:
                overlays[key] = "scalar"
            elif isinstance(tmp, np.ndarray) and tmp.ndim == 2:
                if tmp.shape[1] == 3:
                    overlays[key] = "color"
                else:
                    overlays[key] = "unknown"

        return overlays

    def remove_overlay(self, overlay_name: str) -> None:
        """
        Remove an overlay and its associated data.

        This method removes an overlay from both the mesh point data and
        the color tables storage, and handles active scalar management.

        Parameters
        ----------
        overlay_name : str
            Name of the overlay to remove

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the overlay is not found in either mesh point data or color tables

        Examples
        --------
        >>> surface.remove_overlay("thickness")
        >>> surface.remove_overlay("aparc")
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

    def get_overlay_info(self, overlay_name: str) -> Dict:
        """
        Get information about a specific overlay.

        Parameters
        ----------
        overlay_name : str
            Name of the overlay to get information about

        Returns
        -------
        Dict
            Dictionary containing overlay information with keys:
            - 'name': str, name of the overlay
            - 'data_shape': tuple, shape of the data array
            - 'data_type': str, numpy dtype of the data
            - 'has_colortable': bool, whether overlay has associated color table
            - 'num_regions': int, number of regions (if parcellation)
            - 'region_names': list, names of regions (if parcellation)
            - 'has_annot_object': bool, whether AnnotParcellation object is available

        Raises
        ------
        ValueError
            If the overlay is not found

        Examples
        --------
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

    def get_region_vertices(self, parc_name: str, region_name: str) -> np.ndarray:
        """
        Get vertex indices for a specific region in a parcellation.

        Parameters
        ----------
        parc_name : str
            Name of the parcellation
        region_name : str
            Name of the region

        Returns
        -------
        np.ndarray
            Array of vertex indices belonging to the region

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
            label values, vertex counts, and colors.

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

    def prepare_colors(
        self,
        overlay_name: str = None,
        cmap: str = None,
        vmin: np.float64 = None,
        vmax: np.float64 = None,
    ) -> None:
        """
        Prepare vertex colors for visualization based on the specified overlay.

        This method processes the overlay data and creates appropriate vertex colors
        for visualization, handling both scalar data (with colormaps) and
        categorical data (with discrete color tables).

        Parameters
        ----------
        overlay_name : str, optional
            Name of the overlay to visualize. If None, the first available overlay is used.
        cmap : str, optional
            Colormap to use for scalar overlays. If None, uses parcellation color table
            for categorical data or 'viridis' for scalar data.

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
        This method sets the vertex colors in the mesh based on the specified overlay.
        The colors are stored in the mesh's point_data under the key "vertex_colors"
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
        # If no colormap is provided, use the default colormap for the overlay

        vertex_values = self.mesh.point_data[overlay_name]
        dict_ctables = self.colortables
        # Check if the overlay is a color or scalar type

        if cmap is None:
            if overlay_name in dict_ctables.keys():
                # Use the colortable associated with the parcellation
                vertex_colors = cltfree.create_vertex_colors(
                    vertex_values, self.colortables[overlay_name]["color_table"]
                )

            else:
                vertex_colors = cltmisc.values2colors(
                    vertex_values,
                    cmap="viridis",
                    output_format="rgb",
                    vmin=vmin,
                    vmax=vmax,
                )

        else:
            vertex_colors = cltmisc.values2colors(
                vertex_values, cmap=cmap, output_format="rgb", vmin=vmin, vmax=vmax
            )

        self.mesh.point_data["vertex_colors"] = vertex_colors
        self.mesh.set_active_scalars("vertex_colors")

    def merge_surfaces(self, surfaces: List["Surface"]) -> "Surface":
        """
        Merge this surface with other surfaces into a single surface.

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
            raise TypeError("surfaces must be a list")

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

    def show(
        self,
        overlay_name: str = None,
        cmap: str = None,
        vmin: np.float64 = None,
        vmax: np.float64 = None,
        view: Union[str, List[str]] = "lateral",
        link_views: bool = False,
        colorbar_title: str = None,
        title: str = None,
        window_size: Tuple[int, int] = (1400, 900),
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
        Display the surface with the specified overlay and visualization parameters.

        This method creates an interactive 3D visualization of the brain surface
        with the specified overlay data, lighting, and camera settings.

        Parameters
        ----------
        overlay_name : str, optional
            Name of the overlay to visualize. If None, uses the first available overlay.
        cmap : str, optional
            Colormap for scalar data visualization. If None, uses parcellation colors
            for categorical data or 'viridis' for scalar data.
        view : str or List[str], default "lateral"
            Camera view(s) for visualization. Can be:
            - Single view: "lateral", "medial", "dorsal", "ventral", "anterior", "posterior"
            - Multiple views: ["lateral", "medial"] for side-by-side comparison
        link_views : bool, default False
            Whether to link camera movements across multiple views
        colorbar_title : str, optional
            Title for the colorbar. If None, uses the overlay name for scalar data.
        title : str, optional
            Main title for the visualization window
        window_size : Tuple[int, int], default (1400, 900)
            Size of the visualization window in pixels (width, height)
        background_color : str, default "white"
            Background color of the visualization
        ambient : float, default 0.2
            Ambient lighting coefficient (0.0 to 1.0)
        diffuse : float, default 0.5
            Diffuse lighting coefficient (0.0 to 1.0)
        specular : float, default 0.5
            Specular lighting coefficient (0.0 to 1.0)
        specular_power : int, default 50
            Specular power for surface shininess
        opacity : float, default 1.0
            Surface opacity (0.0 to 1.0)
        style : str, default "surface"
            Rendering style: "surface", "wireframe", or "points"
        smooth_shading : bool, default True
            Whether to use smooth shading for surface rendering
        intensity : float, default 0.2
            Light intensity (note: this parameter is not used by PyVista add_mesh)

        Returns
        -------
        Plotter object
            PyVista plotter object for further customization or interaction

        Raises
        ------
        ValueError
            If the specified overlay is not found
        ValueError
            If view parameter is not a string or list of strings

        Examples
        --------
        >>> # Basic visualization with parcellation
        >>> plotter = surface.show(overlay_name="aparc")
        >>>
        >>> # Scalar data with custom colormap and view
        >>> plotter = surface.show(overlay_name="thickness", cmap="hot", view="medial")
        >>>
        >>> # Multiple views with custom title
        >>> plotter = surface.show(overlay_name="curvature",
        ...                       view=["lateral", "medial"],
        ...                       title="Cortical Curvature",
        ...                       link_views=True)
        >>>
        >>> # Custom lighting and appearance
        >>> plotter = surface.show(overlay_name="thickness",
        ...                       ambient=0.3, diffuse=0.7,
        ...                       opacity=0.8, smooth_shading=False)
        """
        self.prepare_colors(overlay_name=overlay_name, cmap=cmap, vmin=vmin, vmax=vmax)

        dict_ctables = self.colortables
        if cmap is None:
            if overlay_name in dict_ctables.keys():
                show_colorbar = False

            else:
                show_colorbar = True

        else:
            show_colorbar = True

        # Import here to avoid circular import
        from . import visualizationtools as cltvis

        # Validate view parameter
        if isinstance(view, str):
            view = [view]
        elif not isinstance(view, list):
            raise ValueError("view must be a string or a list of strings")

        # Auto-detect colorbar title if not provided
        if show_colorbar and colorbar_title is None:
            colorbar_title = overlay_name  # Default title

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
