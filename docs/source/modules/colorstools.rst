colorstools module
==================

.. automodule:: clabtoolkit.colorstools
   :members:
   :undoc-members:
   :show-inheritance:

The colorstools module centralizes colour handling across the toolkit: format conversion and validation, palette generation, colour lookup table (LUT) loading and export, and colour visualization.

Key Features
------------
- Conversion between hexadecimal, 0-255 RGB and 0-1 RGB formats
- Colour validation and automatic range detection
- Random and perceptually distinguishable palette generation
- Mapping of numerical values to colours through matplotlib colormaps
- Loading of FreeSurfer LUT and TSV colour tables
- Export to FreeSurfer, FSL, nilearn and TSV formats
- Colour table and palette visualization
- Terminal colour codes via ``bcolors``

Main Classes
------------

ColorTableLoader
~~~~~~~~~~~~~~~~
Loads and manages colour lookup tables, and exports them to the formats used across neuroimaging suites.

Key Methods:
- ``load_colortable()``: Detect and load a LUT or TSV colour table automatically
- ``read_luttable()``: Read a FreeSurfer Color Lookup Table
- ``read_tsvtable()``: Read a TSV lookup table
- ``write_luttable()``: Write a FreeSurfer format lookup table
- ``write_tsvtable()``: Write a TSV format lookup table
- ``export()``: Export the loaded table to a specified format
- ``export_to_lutctab()``: Export to FreeSurfer LUT format
- ``export_to_tsvctab()``: Export to TSV format
- ``export_to_fslctab()``: Export to FSL LUT format
- ``export_to_nilearnctab()``: Export to nilearn-compatible format

bcolors
~~~~~~~
Terminal colour escape codes for coloured console output.

Main Functions
--------------

Format Conversion
~~~~~~~~~~~~~~~~~
- ``rgb2hex()``: Convert RGB values to a hexadecimal colour code
- ``hex2rgb()``: Convert a hexadecimal colour code to RGB
- ``multi_rgb2hex()``: Convert an array of RGB colours to hexadecimal
- ``multi_hex2rgb()``: Convert a list of hexadecimal colours to RGB
- ``normalize_rgb()``: Convert an RGB array to the 0-1 range regardless of input format
- ``harmonize_colors()``: Convert a list of colours to a consistent format
- ``readjust_colors()``: Wrapper around ``harmonize_colors()``
- ``invert_colors()``: Invert colours while preserving the input format

Validation
~~~~~~~~~~
- ``is_color_like()``: Validate a colour, including numpy arrays and lists
- ``is_valid_hex_color()``: Strict hexadecimal colour validation
- ``is_valid_rgb_255()``: Check for a valid 0-255 RGB array
- ``is_valid_rgb_01()``: Check for a valid 0-1 RGB array
- ``detect_rgb_range()``: Detect whether an RGB array uses the 0-255 or 0-1 range

Palette Generation
~~~~~~~~~~~~~~~~~~
- ``create_random_colors()``: Generate colours randomly or from a matplotlib colormap
- ``create_distinguishable_colors()``: Generate maximally distinguishable colours by perceptual spacing
- ``get_predefined_distinguishable_colors()``: Retrieve a predefined distinguishable palette
- ``get_colormaps_names()``: List matplotlib colormap names
- ``values2colors()``: Map numerical values to colours through a colormap

Colour Tables
~~~~~~~~~~~~~
- ``create_lut_dictionary()``: Build a LUT dictionary mapping parcel values to colours
- ``get_colors_from_colortable()``: Build per-vertex RGBA colours from parcellation labels
- ``colors_to_table()``: Convert a colour list into a colour table

Visualization
~~~~~~~~~~~~~
- ``visualize_colors()``: Display a list of colour codes in a labelled layout
- ``colortable_visualization()``: Render a FreeSurfer-style colour table to PNG

Common Usage Examples
---------------------

Format conversion and validation::

    import clabtoolkit.colorstools as cltcolors

    # Convert between formats
    hex_code = cltcolors.rgb2hex(255, 128, 0)
    rgb = cltcolors.hex2rgb("#ff8000")

    # Batch conversion
    hex_list = cltcolors.multi_rgb2hex([[255, 0, 0], [0, 255, 0]])
    rgb_array = cltcolors.multi_hex2rgb(["#ff0000", "#00ff00"])

    # Validate and normalize
    if cltcolors.is_valid_hex_color("#ff8000"):
        normalized = cltcolors.normalize_rgb([255, 128, 0])

    # Detect which range an array uses
    rng = cltcolors.detect_rgb_range([0.5, 0.2, 0.1])

Generating palettes::

    # Maximally distinguishable colours for plotting many regions
    colors = cltcolors.create_distinguishable_colors(n=20, output_format="hex")

    # Random colours, optionally drawn from a colormap
    colors = cltcolors.create_random_colors(n=10, output_format="hex", cmap="viridis")

    # Map values onto a colormap
    colors = cltcolors.values2colors(
        values=[0.1, 0.5, 0.9],
        cmap="jet",
        output_format="hex"
    )

    # Preview a palette
    cltcolors.visualize_colors(colors)

Loading and exporting colour tables::

    from clabtoolkit.colorstools import ColorTableLoader

    # Format is detected automatically for LUT and TSV inputs
    loader = ColorTableLoader("/path/to/lookup_table.lut")

    # Export to the format required downstream
    loader.export_to_tsvctab("/path/to/output.tsv", overwrite=True)
    loader.export_to_fslctab("/path/to/output_fsl.lut", overwrite=True)
    loader.export_to_nilearnctab("/path/to/output_nilearn.txt", overwrite=True)

    # Render the colour table as a PNG figure
    cltcolors.colortable_visualization(
        colortable="/path/to/lookup_table.lut",
        export_path="/path/to/colortable.png"
    )
