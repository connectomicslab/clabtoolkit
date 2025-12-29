import numpy as np
import os
import copy
from datetime import datetime
from typing import Union, List, Any, Optional
import re
import pandas as pd
from IPython.display import HTML
from pathlib import Path
from colorama import init, Fore, Style, Back

init(autoreset=True)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.colors import is_color_like as mpl_is_color_like
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import textwrap


import clabtoolkit.misctools as cltmisc


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############              Section 1: Methods dedicated to work with colors              ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
class bcolors:
    """
    This class is used to define the colors for the terminal output.
    It can be used to print the output in different colors.
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    OKYELLOW = "\033[93m"
    OKRED = "\033[91m"
    OKMAGENTA = "\033[95m"
    PURPLE = "\033[35m"
    OKCYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    ORANGE = "\033[48:5:208m%s\033[m"
    OKWHITE = "\033[97m"
    DARKWHITE = "\033[37m"
    OKBLACK = "\033[30m"
    OKGRAY = "\033[90m"
    OKPURPLE = "\033[35m"

    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"


####################################################################################################
def is_color_like(color) -> bool:
    """
    Extended color validation that handles numpy arrays and Python lists.

    Parameters
    ----------
    color : Any
        The color to validate. Can be:
        - Hex string (e.g., "#FF5733")
        - Numpy array ([R,G,B] as integers 0-255 or floats 0-1)
        - Python list ([R,G,B] as integers 0-255 or floats 0-1)

    Returns
    -------
    bool
        True if the color is valid, False otherwise.

    Examples
    --------------
        >>> is_color_like("#FF5733")  # Hex string
        True
        >>> is_color_like(np.array([255, 87, 51]))  # Numpy array
        True
        >>> is_color_like([255, 87, 51])  # Python list (integer)
        True
        >>> is_color_like([1.0, 0.34, 0.5])  # Python list (float)
        True
        >>> is_color_like("invalid_color")
        False
        >>> is_color_like([256, 0, 0])  # Out of range
        False
    """
    # Handle numpy arrays (existing functionality)
    if isinstance(color, np.ndarray):
        if color.shape == (3,) and np.issubdtype(color.dtype, np.integer):
            return (color >= 0).all() and (color <= 255).all()
        if color.shape == (3,) and np.issubdtype(color.dtype, np.floating):
            return (color >= 0).all() and (color <= 1).all()
        return False

    # Handle Python lists
    if isinstance(color, list):
        if len(color) == 3:
            # Check if all elements are integers (0-255)
            if all(isinstance(x, int) for x in color):
                return all(0 <= x <= 255 for x in color)
            # Check if all elements are floats (0-1)
            if all(isinstance(x, (float, np.floating)) for x in color):
                return all(0.0 <= x <= 1.0 for x in color)
        return False

    # Default to matplotlib's validator for strings and other types
    return mpl_is_color_like(color)


#####################################################################################################
def detect_rgb_range(rgb: Any) -> str:
    """
    Detect if an RGB array uses 0-255 or 0-1 range.

    This function analyzes RGB color values to determine whether they follow
    the 0-255 integer format (8-bit) or the 0-1 float format (normalized).

    Parameters
    ----------
    rgb : Any
        RGB color array/list containing 3 numeric values [R, G, B].
        Expected formats: [255, 128, 0] or [1.0, 0.5, 0.0]

    Returns
    -------
    str
        - "0-255" if any value is greater than 1
        - "0-1" if all values are between 0 and 1 (inclusive)
        - "invalid" if input is malformed or values are outside valid ranges

    Raises
    ------
    None
        This function does not raise any exceptions. Invalid inputs
        return "invalid" instead of raising errors.

    Examples
    --------
    >>> detect_rgb_range([255, 128, 0])
    '0-255'
    >>> detect_rgb_range([1.0, 0.5, 0.0])
    '0-1'
    >>> detect_rgb_range([0, 1, 0])
    '0-1'
    >>> detect_rgb_range([255, 0.5, 128])
    'invalid'
    >>> detect_rgb_range([300, 200, 100])
    'invalid'
    >>> detect_rgb_range([0.0, 0.0, 0.0])
    '0-1'
    >>> detect_rgb_range([255, 255, 255])
    '0-255'
    >>> detect_rgb_range([2, 1, 0])
    '0-255'
    >>> detect_rgb_range("not_a_list")
    'invalid'
    >>> detect_rgb_range([255, 128])
    'invalid'

    Notes
    -----
    - Expects exactly 3 numeric values (R, G, B)
    - Any value greater than 1 classifies the array as "0-255" range
    - All values between 0-1 (inclusive) classify the array as "0-1" range
    - Combinations like [0, 1, 0] are treated as "0-1" range
    - The 0-255 validator only accepts whole numbers (integers or floats like 128.0)
    - The 0-1 validator accepts any numeric values in the 0-1 range
    - Mixed ranges (e.g., [255, 0.5, 128]) are considered invalid
    - Out-of-range values (negative or > 255) result in "invalid" classification
    """
    # Validate input format
    if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
        return "invalid"

    # Check if all values are numeric
    try:
        values = [float(val) for val in rgb]
    except (ValueError, TypeError):
        return "invalid"

    # Check if all values are in 0-1 range
    in_zero_one = all(0.0 <= val <= 1.0 for val in values)

    # Check if all values are in 0-255 range
    in_zero_255 = all(0 <= val <= 255 for val in values)

    # Determine range based on values
    if not in_zero_one and not in_zero_255:
        return "invalid"

    # If any value > 1, it's definitely 0-255 range
    if any(val > 1 for val in values):
        return "0-255"

    # If all values <= 1, treat as 0-1 range
    # (This includes combinations of 0 and 1)
    return "0-1"


#####################################################################################################
def is_valid_rgb_255(rgb: Any) -> bool:
    """
    Check if RGB array contains valid 0-255 range values.

    Parameters
    ----------
    rgb : Any
        RGB color array/list to validate

    Returns
    -------
    bool
        True if all values are in 0-255 range, False otherwise

    Examples
    --------
    >>> is_valid_rgb_255([255, 128, 0])
    True
    >>> is_valid_rgb_255([0, 0, 0])
    True
    >>> is_valid_rgb_255([1, 1, 1])
    True
    >>> is_valid_rgb_255([128.0, 200.0, 50.0])
    True
    >>> is_valid_rgb_255([0.5, 0.3, 0.8])
    False
    >>> is_valid_rgb_255([128.5, 200.7, 50.2])
    False
    >>> is_valid_rgb_255([300, 200, 100])
    False
    """
    # Validate input format
    if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
        return False

    # Check if all values are numeric and in 0-255 range
    try:
        values = [float(val) for val in rgb]

        # All values must be in 0-255 range
        if not all(0 <= val <= 255 for val in values):
            return False

        # Only accept whole numbers (no decimal places)
        # This rejects both 0-1 format decimals and invalid decimals > 1
        for val in values:
            if val != int(val):
                return False

        return True
    except (ValueError, TypeError):
        return False


#####################################################################################################
def is_valid_rgb_01(rgb: Any) -> bool:
    """
    Check if RGB array contains valid 0-1 range values.

    Parameters
    ----------
    rgb : Any
        RGB color array/list to validate

    Returns
    -------
    bool
        True if all values are in 0-1 range, False otherwise

    Examples
    --------
    >>> is_valid_rgb_01([1.0, 0.5, 0.0])
    True
    >>> is_valid_rgb_01([0, 0, 0])
    True
    >>> is_valid_rgb_01([1, 1, 1])
    True
    >>> is_valid_rgb_01([255, 128, 0])
    False
    >>> is_valid_rgb_01([1.5, 0.5, 0.2])
    False
    """
    # Validate input format
    if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
        return False

    # Check if all values are numeric and in 0-1 range
    try:
        values = [float(val) for val in rgb]
        return all(0.0 <= val <= 1.0 for val in values)
    except (ValueError, TypeError):
        return False


#####################################################################################################
def normalize_rgb(rgb: Any) -> Union[List[float], None]:
    """
    Convert RGB array to 0-1 range regardless of input format.

    Parameters
    ----------
    rgb : Any
        RGB color array in either 0-255 or 0-1 format

    Returns
    -------
    List[float] or None
        RGB values normalized to 0-1 range, or None if invalid input
    """
    range_type = detect_rgb_range(rgb)

    if range_type == "invalid":
        return None
    elif range_type == "0-255":
        return [val / 255.0 for val in rgb]
    elif range_type == "0-1":
        return [float(val) for val in rgb]
    else:
        return None


####################################################################################################
def rgb2hex(r: Union[int, float], g: Union[int, float], b: Union[int, float]) -> str:
    """
    Convert RGB values to hexadecimal color code.
    Handles both integer (0-255) and normalized float (0-1) inputs.

    Parameters
    ----------
    r : int or float
        Red value (0-255 for integers, 0-1 for floats)
    g : int or float
        Green value (0-255 for integers, 0-1 for floats)
    b : int or float
        Blue value (0-255 for integers, 0-1 for floats)

    Returns
    -------
    str
        Hexadecimal color code in lowercase (e.g., "#ff0000")

    Raises
    ------
    ValueError
        If values are outside valid ranges (either 0-255 or 0-1)
    TypeError
        If input types are mixed (some ints and some floats)

    Examples
    --------
    >>> rgb2hex(255, 0, 0)      # Integer inputs
    '#ff0000'

    >>> rgb2hex(1.0, 0.0, 0.0)  # Normalized float inputs
    '#ff0000'

    >>> rgb2hex(0.5, 0.0, 1.0)  # Mixed range
    '#7f00ff'
    """
    # Check for mixed input types
    input_types = {type(r), type(g), type(b)}
    if len(input_types) > 1:
        raise TypeError(
            "All RGB components must be the same type (all int or all float)"
        )

    # Process based on input type
    if isinstance(r, float):
        # Validate normalized range
        if not (0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1):
            raise ValueError("Float values must be between 0 and 1")
        # Convert to 0-255 range
        r, g, b = (int(round(x * 255)) for x in (r, g, b))
    else:
        # Validate 0-255 range
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            raise ValueError("Integer values must be between 0 and 255")

    # Ensure values are within byte range after conversion
    r, g, b = (max(0, min(255, x)) for x in (r, g, b))

    return "#{:02x}{:02x}{:02x}".format(r, g, b)


####################################################################################################
def multi_rgb2hex(
    colors: Union[List[Union[str, list, np.ndarray]], np.ndarray],
) -> List[str]:
    """
    Function to convert rgb to hex for an array of colors.
    Note: If there are already elements in hexadecimal format the will not be transformed.

    Parameters
    ----------
    colors : list or numpy array
        List of rgb colors

    Returns
    -------
    hexcodes: list
        List of hexadecimal codes for the colors

    Examples
    --------------
        >>> colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        >>> hexcodes = multi_rgb2hex(colors)
        >>> print(hexcodes)  # Output: ['#ff0000', '#00ff00', '#0000ff']

    """

    # Harmonizing the colors
    hexcodes = harmonize_colors(colors, output_format="hex")

    return hexcodes


#######################################################################################################
def is_valid_hex_color(hex_color):
    """
    Strict validation that requires # prefix and only allows 6-digit format.

    This function validates hexadecimal color codes using a strict format that
    requires exactly 6 hexadecimal digits preceded by a hash (#) symbol.

    Parameters
    ----------
    hex_color : str
        The hex color string to validate. Must be in the format #RRGGBB
        where R, G, B are hexadecimal digits (0-9, A-F, a-f).

    Returns
    -------
    bool
        True if the input is a valid 6-digit hex color with # prefix,
        False otherwise.

    Raises
    ------
    None
        This function does not raise any exceptions. Invalid inputs
        return False instead of raising errors.

    Examples
    --------
    >>> is_valid_hex_color("#FF0000")
    True
    >>> is_valid_hex_color("#00FF00")
    True
    >>> is_valid_hex_color("#0000FF")
    True
    >>> is_valid_hex_color("#ffffff")
    True
    >>> is_valid_hex_color("#ABC123")
    True
    >>> is_valid_hex_color_strict("#FFF")
    False
    >>> is_valid_hex_color("FF0000")
    False
    >>> is_valid_hex_color("#GG0000")
    False
    >>> is_valid_hex_color("#FF0000FF")
    False
    >>> is_valid_hex_color("")
    False
    >>> is_valid_hex_color(None)
    False
    >>> is_valid_hex_color(123)
    False

    Notes
    -----
    - Only accepts 6-digit hexadecimal format (e.g., #RRGGBB)
    - Requires the # prefix
    - Case-insensitive for hex digits (A-F or a-f)
    - Does not accept 3-digit shorthand (e.g., #FFF)
    - Does not accept 8-digit format with alpha channel
    - Non-string inputs return False
    """
    if not isinstance(hex_color, str):
        return False

    pattern = r"^#[0-9A-Fa-f]{6}$"
    return bool(re.match(pattern, hex_color))


####################################################################################################
def hex2rgb(hexcode: str) -> tuple:
    """
    Function to convert hex to rgb

    Parameters
    ----------
    hexcode : str
        Hexadecimal code for the color

    Returns
    -------
    tuple
        Tuple with the rgb values

    Examples
    --------------
        >>> hexcode = "#FF5733"
        >>> rgb = hex2rgb(hexcode)
        >>> print(rgb)  # Output: (255, 87, 51)

    """
    # Convert hexadecimal color code to RGB values
    hexcode = hexcode.lstrip("#")
    return tuple(int(hexcode[i : i + 2], 16) for i in (0, 2, 4))


####################################################################################################
def multi_hex2rgb(hexcodes: Union[str, List[str]]) -> np.ndarray:
    """
    Function to convert a list of colores in hexadecimal format to rgb format.

    Parameters
    ----------
    hexcodes : list
        List of hexadecimal codes for the colors

    Returns
    -------
    rgb_list: np.array
        Array of rgb values

    Examples
    --------------
        >>> hexcodes = ["#FF5733", "#33FF57", "#3357FF"]
        >>> rgb_list = multi_hex2rgb(hexcodes)
        >>> print(rgb_list)  # Output: [[255, 87, 51], [51, 255, 87], [51, 87, 255]]

    """
    if isinstance(hexcodes, str):
        hexcodes = [hexcodes]

    rgb_list = [hex2rgb(hex_color) for hex_color in hexcodes]
    return np.array(rgb_list)


####################################################################################################
def invert_colors(
    colors: Union[List[Union[str, list, np.ndarray]], np.ndarray],
) -> Union[List[Union[str, list, np.ndarray]], np.ndarray]:
    """
    Invert colors while maintaining the original input format and value ranges.

    Parameters
    ----------
    colors : list or numpy array
        Input colors in any of these formats:
        - Hex strings (e.g., "#FF5733")
        - Python lists ([R,G,B] as integers 0-255 or floats 0-1)
        - Numpy arrays (integers 0-255 or floats 0-1)

    Returns
    -------
    Union[List[Union[str, list, np.ndarray]], np.ndarray]
        Inverted colors in the same format and range as input

    Examples
    --------
    >>> invert_colors([np.array([0.0, 0.0, 1.0]), np.array([0, 255, 243])])
    [array([1., 1., 0.]), array([255,   0,  12])]
    """
    if not isinstance(colors, (list, np.ndarray)):
        raise TypeError("Input must be a list or numpy array")

    # Store original formats and ranges
    input_types = []
    input_ranges = []  # '0-1' or '0-255'

    for color in colors:
        input_types.append(type(color))
        if isinstance(color, np.ndarray):
            if np.issubdtype(color.dtype, np.integer):
                input_ranges.append("0-255")
            else:
                input_ranges.append("0-1")
        elif isinstance(color, list):
            if all(isinstance(x, int) for x in color):
                input_ranges.append("0-255")
            else:
                input_ranges.append("0-1")
        else:  # hex string
            input_ranges.append("0-255")  # hex implies 0-255

    # Convert all to normalized (0-1) for inversion
    normalized_colors = []
    for color, orig_range in zip(colors, input_ranges):
        if orig_range == "0-255":
            if isinstance(color, str):
                hex_color = color.lstrip("#")
                rgb = (
                    np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0
                )
            elif isinstance(color, (list, np.ndarray)):
                rgb = np.array(color) / 255.0
            normalized_colors.append(rgb)
        else:
            normalized_colors.append(np.array(color))

    # Perform inversion in HSV space
    inverted = []
    for color in normalized_colors:
        hsv = rgb_to_hsv(color.reshape(1, 1, 3))
        hsv[..., 0] = (hsv[..., 0] + 0.5) % 1.0  # Hue rotation
        inverted_rgb = hsv_to_rgb(hsv).flatten()
        inverted.append(inverted_rgb)

    # Convert back to original formats and ranges
    result = []
    for inv_color, orig_type, orig_range in zip(inverted, input_types, input_ranges):
        if orig_range == "0-255":
            inv_color = (inv_color * 255).round().astype(np.uint8)

        if orig_type == str:
            result.append(
                to_hex(inv_color / 255 if orig_range == "0-255" else inv_color).lower()
            )
        elif orig_type == list:
            if orig_range == "0-255":
                result.append([int(x) for x in inv_color])
            else:
                result.append([float(x) for x in inv_color])
        else:  # numpy.ndarray
            if orig_range == "0-255":
                result.append(inv_color.astype(np.uint8))
            else:
                result.append(inv_color.astype(np.float64))

    # Return same container type as input
    return np.array(result) if isinstance(colors, np.ndarray) else result


####################################################################################################
def harmonize_colors(
    colors: Union[str, List[Union[str, list, np.ndarray]], np.ndarray],
    output_format: str = "hex",
) -> Union[List[str], List[np.ndarray]]:
    """
    Convert all colors in a list to a consistent format.
    Handles hex strings, RGB lists, and numpy arrays (both 0-255 and 0-1 ranges).

    Parameters
    ----------
    colors : list or numpy array
        List containing:
        - Hex strings (e.g., "#FF5733")
        - Python lists ([R,G,B] as integers 0-255 or floats 0-1)
        - Numpy arrays (integers 0-255 or floats 0-1)
    output_format : str, optional
        Output format ('hex', 'rgb', or 'rgbnorm'), defaults to 'hex'
        - 'hex': returns hexadecimal strings (e.g., '#ff5733')
        - 'rgb': returns RGB arrays with values 0-255 (uint8)
        - 'rgbnorm': returns normalized RGB arrays with values 0.0-1.0 (float64)

    Returns
    -------
    Union[List[str], List[np.ndarray]]
        List of colors in the specified format

    Examples
    --------
    >>> colors = ["#FF5733", [255, 87, 51], np.array([51, 87, 255])]
    >>> harmonize_colors(colors)
    ['#ff5733', '#ff5733', '#3357ff']

    >>> harmonize_colors(colors, output_format='rgb')
    [array([255,  87,  51], dtype=uint8),
    array([255,  87,  51], dtype=uint8),
    array([ 51,  87, 255], dtype=uint8)]

    >>> harmonize_colors(colors, output_format='rgbnorm')
    [array([1.        , 0.34117647, 0.2       ]),
    array([1.        , 0.34117647, 0.2       ]),
    array([0.2       , 0.34117647, 1.        ])]
    """

    if isinstance(colors, str):
        # Single color string input, convert to list for processing
        colors = [colors]

    if not isinstance(colors, (list, np.ndarray)):
        raise TypeError("Input must be a list or numpy array")

    output_format = output_format.lower()
    if output_format not in ["hex", "rgb", "rgbnorm"]:
        raise ValueError("output_format must be 'hex', 'rgb', or 'rgbnorm'")

    result = []

    for color in colors:
        if not is_color_like(color):
            raise ValueError(f"Invalid color: {color}")

        # Convert all inputs to numpy array first for consistent processing
        if isinstance(color, str):
            # Hex string -> convert to RGB array
            hex_color = color.lstrip("#")
            rgb_array = np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)])
        elif isinstance(color, list):
            # Python list -> convert to numpy array
            rgb_array = np.array(color)
        else:
            # Already numpy array
            rgb_array = color

        # Process based on output format
        if output_format == "hex":
            if np.issubdtype(rgb_array.dtype, np.integer):
                rgb_array = rgb_array / 255.0
            result.append(to_hex(rgb_array).lower())

        elif output_format == "rgbnorm":
            if np.issubdtype(rgb_array.dtype, np.integer):
                rgb_array = rgb_array / 255.0
            result.append(rgb_array.astype(np.float64))

        else:  # rgb format (0-255)
            if np.issubdtype(rgb_array.dtype, np.floating):
                rgb_array = rgb_array * 255
            result.append(rgb_array.astype(np.uint8))

    # Stacking the results
    if output_format != "hex":
        result = np.vstack(result)

    return result


####################################################################################################
def readjust_colors(
    colors: Union[List[Union[str, list, np.ndarray]], np.ndarray],
    output_format: str = "rgb",
) -> Union[list[str], np.ndarray]:
    """
    Function to readjust the colors to a certain format. It is just a wrapper from harmonize_colors function.

    Parameters
    ----------
    colors : list or numpy array
        List of colors

    Returns
    -------
    out_colors: list or numpy array
        List of colors in the desired format

    Examples
    --------------
        >>> colors = ["#FF5733", [255, 87, 51], np.array([51, 87, 255])]
        >>> out_colors = readjust_colors(colors, output_format='hex')
        >>> print(out_colors)  # Output: ['#ff5733', '#ff5733', '#3357ff']

        >>> out_colors = readjust_colors(colors, output_format='rgb')
        >>> print(out_colors)  # Output: [[255, 87, 51], [255, 87, 51], [51, 87, 255]]
    """

    output_format = output_format.lower()
    if output_format not in ["hex", "rgb", "rgbnorm"]:
        raise ValueError("output_format must be 'hex', 'rgb', or 'rgbnorm'")

    # harmonizing the colors
    out_colors = harmonize_colors(colors, output_format=output_format)

    return out_colors


####################################################################################################
def create_random_colors(
    n: int,
    output_format: str = "rgb",
    cmap: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> Union[list[str], np.ndarray]:
    """
    Generate n colors either randomly or from a specified matplotlib colormap.

    This function creates a collection of colors that can be used for data visualization,
    plotting, or other applications requiring distinct color schemes. Colors can be
    generated randomly or sampled from matplotlib colormaps for better visual harmony.

    Parameters
    ----------
    n : int
        Number of colors to generate. Must be a positive integer.
    output_format : str, default "rgb"
        Format of the output colors. Supported formats:
        - "rgb": RGB values as integers in range [0, 255]
        - "rgbnorm": RGB values as floats in range [0.0, 1.0]
        - "hex": Hexadecimal color strings (e.g., "#FF5733")
    cmap : str or None, default None
        Name of matplotlib colormap to use for color generation. If None,
        colors are generated randomly. Popular options include:
        - "viridis", "plasma", "inferno", "magma" (perceptually uniform)
        - "PiYG", "RdYlBu", "Spectral" (diverging)
        - "Set1", "Set2", "tab10" (qualitative)
        - "Blues", "Reds", "Greens" (sequential)
        See matplotlib.pyplot.colormaps() for full list.
    random_seed : int or None, default None
        Seed for random number generator to ensure reproducible results.
        Only used when cmap is None.

    Returns
    -------
    colors : list of str or numpy.ndarray
        Generated colors in the specified format:
        - If output_format is "hex": list of hex color strings
        - If output_format is "rgb" or "rgbnorm": numpy array of shape (n, 3)

    Raises
    ------
    ValueError
        If output_format is not one of the supported formats.
        If n is not a positive integer.
        If cmap is not a valid matplotlib colormap name.
    TypeError
        If n is not an integer.

    Examples
    --------
    Generate random colors:

    >>> colors = create_random_colors(3, output_format="hex")
    >>> print(colors)  # ['#A1B2C3', '#D4E5F6', '#789ABC']

    >>> colors = create_random_colors(3, output_format="rgb")
    >>> print(colors)  # [[161, 178, 195], [212, 229, 246], [120, 154, 188]]

    Generate colors from a colormap:

    >>> colors = create_random_colors(5, output_format="hex", cmap="PiYG")
    >>> print(colors)  # ['#8E0152', '#C994C7', '#F7F7F7', '#A1DAB4', '#276419']

    >>> colors = create_random_colors(4, output_format="rgbnorm", cmap="viridis")
    >>> print(colors)  # [[0.267, 0.005, 0.329], [0.229, 0.322, 0.545], ...]

    Notes
    -----
    - When using a colormap, colors are evenly spaced across the colormap range
    - Random colors are generated uniformly across RGB space and may not be
    visually harmonious
    - For better visual results with random colors, consider using the
    harmonize_colors() function (if available)
    - Colormaps provide better perceptual uniformity and accessibility
    """

    # Input validation
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n <= 0:
        raise ValueError("n must be a positive integer")

    output_format = output_format.lower()
    if output_format not in ["hex", "rgb", "rgbnorm"]:
        raise ValueError("output_format must be 'hex', 'rgb', or 'rgbnorm'")

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    if cmap is not None:
        # Generate colors from colormap
        try:
            colormap = plt.get_cmap(cmap)
        except ValueError:
            raise ValueError(
                f"'{cmap}' is not a valid matplotlib colormap name. "
                f"Use plt.colormaps() to see available options."
            )

        # Generate evenly spaced points across the colormap
        if n == 1:
            indices = [0.5]  # Use middle of colormap for single color
        else:
            indices = np.linspace(0, 1, n)

        # Get colors from colormap (returns RGBA, we take only RGB)
        colors_norm = np.array([colormap(idx)[:3] for idx in indices])

        if output_format == "rgbnorm":
            return colors_norm
        elif output_format == "rgb":
            return (colors_norm * 255).astype(int)
        else:  # hex
            return [rgb2hex(color[0], color[1], color[2]) for color in colors_norm]

    else:
        # Generate random colors
        colors = np.random.randint(0, 255, size=(n, 3))

        # Apply harmonization if the function is available
        try:
            colors = harmonize_colors(colors, output_format=output_format)
            return colors
        except NameError:
            # harmonize_colors function not available, proceed without harmonization
            pass

        if output_format == "rgb":
            return colors
        elif output_format == "rgbnorm":
            return colors / 255.0
        else:  # hex
            return ["#{:02x}{:02x}{:02x}".format(r, g, b) for r, g, b in colors]


#####################################################################################################
def get_colormaps_names(n, cmap_type="sequential"):
    """
    Get a list of colormap names from matplotlib. If n exceeds available colormaps, the list repeats.
    It can return either sequential or diverging colormaps.

    Parameters
    ----------
    n : int
        Number of colormaps to return

    cmap_type : str, optional
        Type of colormaps: "sequential" or "diverging" (default: "sequential")

    Returns
    -------
    list
        List of colormap names (repeats if n exceeds available colormaps)

    Examples
    --------
    >>> get_colormaps_names(5, cmap_type="sequential")
    ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

    >>> get_colormaps_names(3, cmap_type="diverging")
    ['PiYG', 'PRGn', 'BrBG']

    Notes
    -----
    - Uses matplotlib's built-in colormaps
    - If n exceeds available colormaps, the list repeats to fulfill the request
    """
    # Predefined lists of colormaps by type
    sequential_cmaps = [
        "viridis",
        "jet",
        "copper",
        "hot",
        "winter",
        "autumn",
        "spring",
        "summer",
        "bone",
        "cool",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        "Greys",
        "Purples",
        "Blues",
        "Greens",
        "Oranges",
        "Reds",
        "YlOrBr",
        "YlOrRd",
        "OrRd",
        "PuRd",
        "RdPu",
        "BuPu",
        "GnBu",
        "PuBu",
        "YlGnBu",
        "PuBuGn",
        "BuGn",
        "YlGn",
    ]

    diverging_cmaps = [
        "PiYG",
        "PRGn",
        "BrBG",
        "PuOr",
        "RdGy",
        "RdBu",
        "RdYlBu",
        "RdYlGn",
        "Spectral",
        "coolwarm",
        "bwr",
        "seismic",
    ]

    # Select the appropriate list
    if cmap_type == "sequential":
        cmap_list = sequential_cmaps
    elif cmap_type == "diverging":
        cmap_list = diverging_cmaps
    else:
        raise ValueError(
            f"cmap_type must be 'sequential' or 'diverging', got '{cmap_type}'"
        )

    # If n exceeds available colormaps, repeat them
    if n > len(cmap_list):
        repeats = (n // len(cmap_list)) + 1
        cmap_list = cmap_list * repeats

    return cmap_list[:n]


#########################################################################################################
def create_lut_dictionary(parc_values: Union[List[int], np.ndarray]) -> dict:
    """
    Create a lookup table (LUT) dictionary mapping parcel values to colors.

    Parameters
    ----------
    parc_values : List[int]
        List of integer parcel values.

    Returns
    -------
    dict
        Dictionary with keys:
        - "index": List of parcel IDs (excluding background id 0)
        - "name": List of region names (e.g., "Region_1", "Region_2", ...)
        - "color": List of hex color codes corresponding to each parcel ID

    Examples
    --------
    >>> parc_values = [0, 1, 2, 3, 4]
    >>> lut_dict = create_lut_dictionary(parc_values)
    >>> print(lut_dict)
    {'index': [1, 2, 3, 4],
     'name': ['Region_1', 'Region_2', 'Region_3', 'Region_4'],
     'color': ['#e6194b', '#3cb44b', '#ffe119', '#0082c8']}

    """
    # Remove the background id (0)
    sts_ids = np.array(parc_values)
    sts_ids = sts_ids[sts_ids != 0]
    sts_ids = sts_ids.astype(int).tolist()

    sts_names = [f"Region_{id}" for id in sts_ids]
    sts_colors = create_distinguishable_colors(len(sts_ids), output_format="hex")

    lut_dict = {}
    lut_dict["index"] = sts_ids
    lut_dict["name"] = sts_names
    lut_dict["color"] = sts_colors

    return lut_dict


###################################################################################################
def create_distinguishable_colors(
    n: int,
    output_format: str = "rgb",
    exclude_colors: Optional[list] = None,
    lightness_range: tuple[float, float] = (0.4, 0.85),
    saturation_range: tuple[float, float] = (0.5, 1.0),
    random_seed: Optional[int] = None,
) -> Union[list[str], np.ndarray]:
    """
    Generate n maximally distinguishable colors using perceptual color spacing.

    This function creates colors that are as visually distinct as possible from each
    other, making them ideal for categorical data visualization, plots with many
    categories, or any application requiring easily distinguishable colors.

    The algorithm uses HSV color space to distribute colors evenly across the hue
    spectrum while maintaining good saturation and lightness values for visibility.
    For very small sets (n ≤ 10), it can optionally use predefined maximally distinct
    color sets based on color theory research.

    Parameters
    ----------
    n : int
        Number of colors to generate. Must be a positive integer.

    output_format : str, default "rgb"
        Format of the output colors. Supported formats:
        - "rgb": RGB values as integers in range [0, 255]
        - "rgbnorm": RGB values as floats in range [0.0, 1.0]
        - "hex": Hexadecimal color strings (e.g., "#FF5733")

    exclude_colors : list of str or list of tuples, optional
        Colors to avoid when generating the palette. Can be hex strings or
        RGB tuples. The algorithm will try to maximize distance from these colors.

    lightness_range : tuple of float, default (0.4, 0.85)
        Range of lightness (V in HSV) to use, as (min, max) in [0, 1].
        Values closer to 0 are darker, closer to 1 are lighter.
        Default range avoids very dark and very light colors for better visibility.

    saturation_range : tuple of float, default (0.5, 1.0)
        Range of saturation (S in HSV) to use, as (min, max) in [0, 1].
        Values closer to 0 are more gray, closer to 1 are more vivid.
        Default range ensures colors are vibrant and easily distinguishable.

    random_seed : int or None, default None
        Seed for random number generator for reproducible saturation/lightness
        variations. If None, variations will be non-deterministic.

    Returns
    -------
    colors : list of str or numpy.ndarray
        Generated colors in the specified format:
        - If output_format is "hex": list of hex color strings
        - If output_format is "rgb" or "rgbnorm": numpy array of shape (n, 3)

    Raises
    ------
    ValueError
        If output_format is not one of the supported formats.
        If n is not a positive integer.
        If lightness_range or saturation_range values are not in [0, 1].

    TypeError
        If n is not an integer.

    Examples
    --------
    Generate distinguishable colors for a categorical plot:

    >>> colors = create_distinguishable_colors(5, output_format="hex")
    >>> print(colors)
    ['#E63946', '#06FFA5', '#3A86FF', '#FFBE0B', '#8338EC']

    >>> colors = create_distinguishable_colors(8, output_format="rgb")
    >>> print(colors.shape)
    (8, 3)

    Generate colors with custom lightness for dark backgrounds:

    >>> colors = create_distinguishable_colors(
    ...     6,
    ...     output_format="hex",
    ...     lightness_range=(0.6, 0.95),
    ...     saturation_range=(0.7, 1.0)
    ... )

    Exclude specific colors (e.g., avoid red):

    >>> colors = create_distinguishable_colors(
    ...     4,
    ...     output_format="hex",
    ...     exclude_colors=["#FF0000", "#CC0000"]
    ... )

    Notes
    -----
    - Colors are distributed evenly across the hue spectrum (360 degrees)
    - Saturation and lightness are varied slightly to increase distinctiveness
    - The algorithm prioritizes perceptual difference over aesthetic harmony
    - For small sets (n ≤ 6), consider also trying matplotlib's "tab10" colormap
    with create_random_colors(n, cmap="tab10") for comparison
    - Maximum recommended n is around 20-30 for truly distinguishable colors;
    beyond that, some colors will inevitably appear similar

    See Also
    --------
    create_random_colors : Generate random or colormap-based colors
    matplotlib.colors.rgb_to_hsv : Convert RGB to HSV color space
    """

    # Input validation
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n <= 0:
        raise ValueError("n must be a positive integer")

    output_format = output_format.lower()
    if output_format not in ["hex", "rgb", "rgbnorm"]:
        raise ValueError("output_format must be 'hex', 'rgb', or 'rgbnorm'")

    if not (0 <= lightness_range[0] <= lightness_range[1] <= 1):
        raise ValueError("lightness_range values must be in [0, 1] with min <= max")
    if not (0 <= saturation_range[0] <= saturation_range[1] <= 1):
        raise ValueError("saturation_range values must be in [0, 1] with min <= max")

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate evenly spaced hues
    hues = np.linspace(0, 1, n, endpoint=False)

    # Add a small random offset to starting hue for variety (deterministic if seed is set)
    hue_offset = np.random.uniform(0, 1 / n) if n > 1 else 0
    hues = (hues + hue_offset) % 1.0

    # Generate varied saturation and lightness values
    # Alternate between high and low values for adjacent colors to maximize difference
    saturations = np.zeros(n)
    lightnesses = np.zeros(n)

    for i in range(n):
        # Alternate patterns for better distinction between adjacent colors
        if i % 2 == 0:
            saturations[i] = np.random.uniform(
                saturation_range[0] + 0.4 * (saturation_range[1] - saturation_range[0]),
                saturation_range[1],
            )
            lightnesses[i] = np.random.uniform(
                lightness_range[0] + 0.3 * (lightness_range[1] - lightness_range[0]),
                lightness_range[1],
            )
        else:
            saturations[i] = np.random.uniform(
                saturation_range[0],
                saturation_range[0] + 0.6 * (saturation_range[1] - saturation_range[0]),
            )
            lightnesses[i] = np.random.uniform(
                lightness_range[0],
                lightness_range[0] + 0.7 * (lightness_range[1] - lightness_range[0]),
            )

    # Create HSV colors
    hsv_colors = np.column_stack([hues, saturations, lightnesses])

    # Convert to RGB (normalized [0, 1])
    rgb_colors_norm = np.array([hsv_to_rgb(hsv) for hsv in hsv_colors])

    # Handle excluded colors if provided
    if exclude_colors is not None:
        # This is a placeholder for more sophisticated exclusion logic
        # In a full implementation, you might adjust colors that are too close
        # to excluded colors
        pass

    # Convert to requested output format
    if output_format == "rgbnorm":
        return rgb_colors_norm
    elif output_format == "rgb":
        return (rgb_colors_norm * 255).astype(int)
    else:  # hex
        hex_colors = []
        for color in rgb_colors_norm:
            r, g, b = (color * 255).astype(int)
            hex_colors.append("#{:02X}{:02X}{:02X}".format(r, g, b))
        return hex_colors


####################################################################################################
def get_predefined_distinguishable_colors(
    n: int, output_format: str = "rgb"
) -> Union[list[str], np.ndarray]:
    """
    Get a predefined set of maximally distinguishable colors.

    This function returns carefully selected colors that are known to be highly
    distinguishable based on color theory research. Available for up to 20 colors.

    Parameters
    ----------
    n : int
        Number of colors (must be between 1 and 20).

    output_format : str, default "rgb"
        Format of the output: "rgb", "rgbnorm", or "hex".

    Returns
    -------
    colors : list or numpy.ndarray
        The predefined distinguishable colors.

    Notes
    -----
    Based on Kenneth Kelly's 22 colors of maximum contrast, optimized for
    both color-normal and color-blind viewers.
    """
    # Kelly's 22 colors of maximum contrast (excluding white and black)
    kelly_colors_hex = [
        "#F3C300",  # Vivid Yellow
        "#875692",  # Strong Purple
        "#F38400",  # Vivid Orange
        "#A1CAF1",  # Very Light Blue
        "#BE0032",  # Vivid Red
        "#C2B280",  # Grayish Yellow
        "#848482",  # Medium Gray
        "#008856",  # Vivid Green
        "#E68FAC",  # Strong Purplish Pink
        "#0067A5",  # Strong Blue
        "#F99379",  # Strong Yellowish Pink
        "#604E97",  # Strong Violet
        "#F6A600",  # Vivid Orange Yellow
        "#B3446C",  # Strong Purplish Red
        "#DCD300",  # Vivid Greenish Yellow
        "#882D17",  # Strong Reddish Brown
        "#8DB600",  # Vivid Yellowish Green
        "#654522",  # Deep Yellowish Brown
        "#E25822",  # Vivid Reddish Orange
        "#2B3D26",  # Dark Olive Green
    ]

    if n > len(kelly_colors_hex):
        raise ValueError(f"Only {len(kelly_colors_hex)} predefined colors available")

    selected_colors = kelly_colors_hex[:n]

    if output_format == "hex":
        return selected_colors

    # Convert hex to RGB
    rgb_colors = multi_hex2rgb(selected_colors)
    rgb_array = np.array(rgb_colors)

    if output_format == "rgb":
        return rgb_array
    else:  # rgbnorm
        return rgb_array / 255.0


###################################################################################################
def colortable_visualization(
    colortable: np.ndarray,
    region_names: Union[str, List[str]],
    columns: int = 2,
    export_path: str = None,
    title: str = "Color Table",
    alternating_bg: bool = False,
):
    """
    Color table visualization. Generates a PNG image displaying a FreeSurfer-style color table.

    Parameters
    ----------
    colortable : array-like, shape (N, 3), (N, 4), or (N, 5)
        FreeSurfer color table: [R, G, B] or [R, G, B, Alpha] or [R, G, B, Alpha, Value]

    region_names : list of str
        Region names corresponding to each row.

    columns : int, default=2
        Number of columns in layout.

    export_path : str, optional
        Path to save PNG file.

    title : str, default="FreeSurfer Color Table"
        Title displayed at the top.

    alternating_bg : bool, default=True
        Whether to shade alternating rows for readability.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object.

    Raises
    ------
    ValueError
        If colortable shape is invalid or region_names length mismatch.
    TypeError
        If region_names is not a string or a list of strings.

    Examples
    --------
    >>> # Example usage
    >>> colortable = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    >>> region_names = ["Region 1", "Region 2", "Region 3"]
    >>> fig = colortable_visualization(colortable
    ...     , region_names, columns=1, title="My Color Table")
    >>> plt.show()


    """

    colortable = np.array(colortable, dtype=float)
    n_regions = len(region_names)

    # Validate colortable shape
    if colortable.ndim != 2 or colortable.shape[1] not in [3, 4, 5]:
        raise ValueError("colortable must be a 2D array with 3, 4, or 5 columns")

    if colortable.shape[0] != n_regions:
        raise ValueError(
            "Length of region_names must match number of rows in colortable"
        )

    if not isinstance(region_names, (str, list)):
        raise TypeError("region_names must be a string or a list of strings")

    if isinstance(region_names, str):
        region_names = [region_names]

    elif isinstance(region_names, list):  # Validate all elements are strings
        if not all(isinstance(name, str) for name in region_names):
            raise TypeError("All elements in region_names list must be strings")

    colors = colortable[:, 0:3]
    colors = harmonize_colors(colors, output_format="rgb")
    colortable[:, 0:3] = colors

    # Layout
    rows_per_col = int(np.ceil(n_regions / columns))
    rect_width = 0.5
    rect_height = 0.35
    row_spacing = 0.5
    col_spacing = 5.5

    margin_left, margin_right, margin_top, margin_bottom = 0.5, 1.0, 1.2, 0.6
    fig_width = margin_left + columns * col_spacing + margin_right
    fig_height = margin_bottom + rows_per_col * (rect_height + row_spacing) + margin_top

    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="white")
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis("off")

    # Title
    ax.text(
        fig_width / 2,
        fig_height - 0.5,
        title,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )

    # Draw rows
    for i in range(n_regions):
        col = i // rows_per_col
        row = i % rows_per_col
        x = margin_left + col * col_spacing
        y = (
            fig_height
            - margin_top
            - (row + 1) * (rect_height + row_spacing)
            + row_spacing
        )

        # Background shading for readability
        if alternating_bg and row % 2 == 1:
            ax.add_patch(
                patches.Rectangle(
                    (x - 0.2, y - 0.1),
                    col_spacing - 0.3,
                    rect_height + 0.2,
                    facecolor="#efecec",
                    edgecolor="none",
                    zorder=0,
                )
            )
        elif alternating_bg and row % 2 == 0:
            ax.add_patch(
                patches.Rectangle(
                    (x - 0.2, y - 0.1),
                    col_spacing - 0.3,
                    rect_height + 0.2,
                    facecolor="#c8c8c8",
                    edgecolor="none",
                    zorder=0,
                )
            )

        # Get RGBA
        r, g, b = colortable[i, 0:3] / 255.0
        a = colortable[i, 3] / 255.0 if colortable.shape[1] >= 4 else 1.0

        # Color rectangle
        ax.add_patch(
            patches.Rectangle(
                (x, y),
                rect_width,
                rect_height,
                facecolor=(r, g, b, a),
                edgecolor="#444444",
                linewidth=0.8,
            )
        )

        # Text
        if colortable.shape[1] == 5:
            value = int(colortable[i, 4])
            rgb_label = f"#{value:02d} ({int(colortable[i,0])}, {int(colortable[i,1])}, {int(colortable[i,2])})"
        else:
            rgb_label = f"({int(colortable[i,0])}, {int(colortable[i,1])}, {int(colortable[i,2])})"

        # Wrap long names if needed
        name = textwrap.fill(region_names[i], width=30)
        label = f"{rgb_label} {name}"

        ax.text(
            x + rect_width + 0.2,
            y + rect_height / 2,
            label,
            ha="left",
            va="center",
            fontsize=10,
            fontfamily="monospace",
        )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if export_path:
        plt.savefig(
            export_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.15,
        )
        print(f"Saved: {export_path}")

    return fig


#####################################################################################################
def get_colors_from_colortable(
    labels: np.ndarray, reg_ctable: np.ndarray
) -> np.ndarray:
    """
    Create per-vertex RGBA colors based on parcellation labels.

    Assigns colors to vertices based on their parcellation region using
    the color table information.

    Parameters
    ----------
    labels : np.ndarray
        Array of parcellation labels for each vertex.

    reg_ctable : np.ndarray
        Color table with shape (N, 5) where first 3 columns are RGB values
        and column 4 contains region labels.

    Returns
    -------
    colors : np.ndarray
        Array of RGB colors for each vertex with shape (num_vertices, 3).
        Default color is gray (240, 240, 240) for unlabeled vertices.

    Examples
    --------
    >>> # Create vertex colors for visualization over a surface mesh
    >>> colors = get_colors_from_colortable(vertex_labels, color_table)
    >>> print(f"Colors shape: {colors.shape}")  # (num_vertices, 3)
    """

    # Automatically detect the range of the colors in reg_ctable
    if reg_ctable.shape[1] != 5:
        raise ValueError(
            "The color table must have 5 columns: R, G, B, A, and packed RGB value"
        )
    # Get the colors from the first 3 columns
    # This assumes the first 3 columns are RGB values
    colors_ctable = reg_ctable[:, :3].astype(np.uint8)

    # Check if all the colors are in the range 0-255
    if not ((colors_ctable.min() >= 0.0) and (colors_ctable.max() <= 1.0)):
        colors = np.ones((len(labels), 3), dtype=np.uint8) * 240  # Default gray
        colors = np.append(colors, np.zeros((len(labels), 1), dtype=np.uint8), axis=1)

    else:
        colors = np.ones((len(labels), 3), dtype=np.uint8) * 240 / 255  # Default gray
        colors = np.append(colors, np.ones((len(labels), 1), dtype=np.uint8), axis=1)

    for i, region_info in enumerate(reg_ctable):
        # Find vertices with this label
        indices = np.where(labels == region_info[4])[0]

        # Assign the region color (RGB from first 3 columns)
        if len(indices) > 0:
            colors[indices, :4] = region_info[:4]

    return colors


#####################################################################################################
def values2colors(
    values: Union[List[Union[int, float]], np.ndarray],
    cmap: str = "viridis",
    output_format: str = "hex",
    invert_cl: bool = False,
    invert_clmap: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
    range_color: tuple = (200, 200, 200),
) -> Union[List[str], np.ndarray]:
    """
    Map numerical values to colors using a specified colormap with optional inversions.

    This function takes a list or array of numerical values and maps them to colors
    using matplotlib colormaps. It provides options to invert the colormap gradient
    and/or invert the resulting colors to their complements.

    Parameters
    ----------
    values : list or numpy.ndarray
        Numerical values to map to colors. Can be integers or floats.

    cmap : str, default "viridis"
        Name of matplotlib colormap to use for color generation.

    output_format : str, default "hex"
        Format of the output colors. Supported formats:
        - "hex": Hexadecimal color strings (e.g., "#FF5733")
        - "rgb": RGB values as integers in range [0, 255]
        - "rgbnorm": RGB values as floats in range [0.0, 1.0]

    invert_cl : bool, default False
        If True, return the complementary colors instead of the original ones.

    invert_clmap : bool, default False
        If True, invert the gradient of the colormap before mapping values.

    vmin : float or None, default None
        Minimum value for colormap normalization. If None, uses range_min if provided,
        otherwise uses min of in-range values.

    vmax : float or None, default None
        Maximum value for colormap normalization. If None, uses range_max if provided,
        otherwise uses max of in-range values.

    range_min : float or None, default None
        Minimum threshold for values. Values below this will be set to range_color.

    range_max : float or None, default None
        Maximum threshold for values. Values above this will be set to range_color.

    range_color : tuple, default (200, 200, 200)
        Color to assign to out-of-range values, in RGB or RGBA format.

    Returns
    -------
    all_colors : list of str or numpy.ndarray
        Mapped colors in the specified format.

    Raises
    ------
    ValueError
        If output_format is not supported or cmap is invalid.

    TypeError
        If values is not a list or numpy array.
    """

    # Input validation
    if not isinstance(values, (list, np.ndarray)):
        raise TypeError("values must be a list or numpy array")

    values = np.array(values, dtype=float)
    if values.size == 0:
        raise ValueError("values array cannot be empty")

    # Check the range_color format
    if not (isinstance(range_color, tuple) and len(range_color) in [3, 4]):
        raise ValueError("range_color must be a tuple of length 3 (RGB) or 4 (RGBA)")

    output_format = output_format.lower()
    if output_format not in ["hex", "rgb", "rgbnorm"]:
        raise ValueError("output_format must be 'hex', 'rgb', or 'rgbnorm'")

    # Create mask for out-of-range values
    mask = np.zeros(len(values), dtype=bool)
    if range_min is not None:
        mask |= values < range_min

    if range_max is not None:
        mask |= values > range_max

    # Get values within range
    values_4_colors = values[~mask]

    # Get the matplotlib colormap
    try:
        colormap = plt.get_cmap(cmap)
    except ValueError:
        raise ValueError(
            f"'{cmap}' is not a valid matplotlib colormap name. "
            f"Use plt.colormaps() to see available options."
        )

    # Invert colormap if requested
    if invert_clmap:
        colormap = colormap.reversed()

    # Set vmin and vmax for normalization
    # Priority: explicit vmin/vmax > range_min/range_max > min/max of in-range values
    # Set vmin and vmax for normalization
    # Priority: explicit vmin/vmax > min/max of in-range values
    if vmin is None:
        if len(values_4_colors) > 0:
            vmin = np.nanmin(values_4_colors)
        else:
            vmin = 0.0

    if vmax is None:
        if len(values_4_colors) > 0:
            vmax = np.nanmax(values_4_colors)
        else:
            vmax = 1.0

    # Handle edge cases
    if not np.isfinite(vmin):
        vmin = 0.0

    if not np.isfinite(vmax):
        vmax = 1.0

    if vmax == vmin:
        # All values are the same, map to middle of colormap
        normalized_values = np.full_like(values_4_colors, 0.5, dtype=float)
    else:
        # Normalize values to [0, 1] range using vmin/vmax
        normalized_values = (values_4_colors - vmin) / (vmax - vmin)

    # Clip values to [0, 1] range
    normalized_values = np.clip(normalized_values, 0, 1)

    # Handle NaN values - map to a neutral color (middle of colormap)
    nan_mask = ~np.isfinite(values_4_colors)
    normalized_values[nan_mask] = 0.5

    # Map normalized values to colors using the continuous colormap
    mapped_colors = colormap(normalized_values)  # Returns RGBA values in [0,1]
    mapped_colors = np.squeeze(mapped_colors)

    # Convert to 0-255 range
    mapped_colors = (mapped_colors * 255).astype(np.uint8)

    # Remove alpha channel if present (take only RGB)
    if mapped_colors.ndim > 1 and mapped_colors.shape[-1] == 4:
        mapped_colors = mapped_colors[..., :3]
    elif mapped_colors.ndim == 1 and len(mapped_colors) == 4:
        mapped_colors = mapped_colors[:3]

    # Prepare range_color in RGB format
    range_color_rgb = np.array(range_color[:3], dtype=np.uint8)

    # Initialize output array with range_color and assign mapped colors to in-range values
    if len(values_4_colors) == 0:
        # All values are out of range
        if output_format == "rgb":
            all_colors = np.tile(range_color_rgb, (len(values), 1))
        elif output_format == "rgbnorm":
            all_colors = np.tile(range_color_rgb / 255.0, (len(values), 1))
        else:  # hex
            range_hex = f"#{range_color_rgb[0]:02x}{range_color_rgb[1]:02x}{range_color_rgb[2]:02x}"
            all_colors = [range_hex] * len(values)
    else:
        if output_format == "rgb":
            all_colors = np.tile(range_color_rgb, (len(values), 1))
            all_colors[~mask] = mapped_colors
        elif output_format == "rgbnorm":
            all_colors = np.tile(range_color_rgb / 255.0, (len(values), 1))
            all_colors[~mask] = mapped_colors / 255.0
        else:  # hex
            range_hex = f"#{range_color_rgb[0]:02x}{range_color_rgb[1]:02x}{range_color_rgb[2]:02x}"
            all_colors = [range_hex] * len(values)
            # Assign hex colors for in-range values
            in_range_indices = np.where(~mask)[0]
            if mapped_colors.ndim == 1:
                # Single color for single value
                r, g, b = mapped_colors[:3]
                all_colors[in_range_indices[0]] = (
                    f"#{int(r):02x}{int(g):02x}{int(b):02x}"
                )
            else:
                # Multiple colors
                for i, idx in enumerate(in_range_indices):
                    r, g, b = mapped_colors[i]
                    all_colors[idx] = f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    result_colors = all_colors

    # Apply color inversion if requested
    if invert_cl:
        if output_format == "hex":
            # For hex format, convert to RGB, invert, then back to hex
            rgb_colors = np.array(
                [
                    [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
                    for color in result_colors
                ]
            )
            inverted_rgb = 255 - rgb_colors
            result_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in inverted_rgb]
        else:
            # For rgb and rgbnorm formats
            if output_format == "rgb":
                result_colors = 255 - result_colors
            else:  # rgbnorm
                result_colors = 1.0 - result_colors

    return result_colors


#####################################################################################################
def colors_to_table(
    colors: Union[list, np.ndarray],
    alpha_values: np.ndarray = 0,
    values: np.ndarray = None,
) -> np.ndarray:
    """
    Convert color list to a color table.
    The color table will contain RGB values, alpha channel, and values or packed RGB values.

    This function harmonizes the input colors to RGB format, applies alpha values,
    and generates a color table with the specified values. It supports both
    hexadecimal color strings and RGB arrays. If values are not provided, it will
    generate a default packed RGB value for each color.

    If only the colors are provided, the function will create a color table
    with the RGB values, an alpha channel set to 0, and default packed RGB values.
    This structure is useful for creating a color table that can be used in FreeSurfer.

    Parameters
    ----------
    colors : list or np.ndarray
        List of hexadecimal color strings (e.g., ['#FF0000', '#00FF00'])
        or numpy array of RGB values. It can be also a list of mixture of
        hexadecimal strings and RGB arrays.

    alpha_values : np.ndarray
        Array of alpha values for each color. If a single value is provided,
        it will be applied to all colors.

    values : np.ndarray, optional
        Array of values corresponding to each color. If a single value is provided,

    Returns
    -------
    color_table : np.ndarray
        Color table with shape (N, 5) containing RGB values,
        alpha channel, and values or packed RGB values.

    Raises
    ------
    ValueError
        If colors is not a list or numpy array.

    Examples
    --------
    >>> # Convert hex colors to color table
    >>> hex_colors = ["#FF0000", "#00FF00", "#0000FF"]
    >>> ctab = colors2colortable(hex_colors)
    >>> print(f"Color table shape: {ctab.shape}")
    """

    if not isinstance(colors, (list, np.ndarray)):
        raise ValueError("The colors must be a list or a numpy array")

    colors = harmonize_colors(colors, output_format="rgb")

    # If values is None
    if values is None:
        values = np.zeros(np.shape(colors)[0], dtype=int)
        for i, color in enumerate(colors):
            values[i] = int(color[0]) + int(color[1]) * 2**8 + int(color[2]) * 2**16

    if hasattr(values, "__len__"):
        values_len = len(values)
    else:
        values_len = 1

    if values_len != np.shape(colors)[0]:
        raise ValueError(
            "The number of values must match the number of colors provided or a single value."
        )
    if hasattr(alpha_values, "__len__"):
        alpha_len = len(alpha_values)
    else:
        alpha_len = 1

    if alpha_len != np.shape(colors)[0]:
        if alpha_len != 1:

            raise ValueError(
                "The number of alpha values must match the number of colors provided or a single value."
            )
        else:
            if alpha_len == 1:
                alpha_values = np.ones(np.shape(colors)[0]) * alpha_values
            else:
                alpha_values = np.ones(np.shape(colors)[0]) * alpha_values[0]

    # Concatenate RGB values and alpha channel and values
    color_table = np.column_stack(
        (
            colors,
            alpha_values,
            values,
        )
    )

    return color_table


###################################################################################################
def visualize_colors(
    colors: Union[List[Union[str, list, np.ndarray]], np.ndarray],
    figsize: tuple = (10, 1),
    label_position: str = "below",  # or "above"
    label_rotation: int = 45,
    label_size: Optional[float] = None,
    spacing: float = 0.1,
    aspect_ratio: float = 0.1,
    background_color: str = "white",
    edge_color: Optional[str] = None,
) -> None:
    """
    Visualize a list of color codes in a clean, professional layout with configurable display options.

        Parameters
        ----------
        colors : List[str]
            List of hexadecimal color codes to visualize (e.g., ['#FF5733', '#33FF57'])
        figsize : tuple, optional
            Size of the figure in inches (width, height), by default (10, 2)
        label_position : str, optional
            Position of color labels relative to color bars ('above' or 'below'),
            by default "below"
        label_rotation : int, optional
            Rotation angle for labels in degrees (0-90), by default 45
        label_size : Optional[float], optional
            Font size for labels. If None, size is automatically determined based on
            number of colors, by default None
        spacing : float, optional
            Additional vertical space for labels (relative to bar height), by default 0.1
        aspect_ratio : float, optional
            Height/width ratio of color rectangles (0.1-1.0 recommended), by default 0.2
        background_color : str, optional
            Background color of the figure, by default "white"
        edge_color : Optional[str], optional
            Color for rectangle borders. None means no borders, by default None

        Returns
        -------
        None
            Displays a matplotlib figure with the color visualization

        Raises
        ------
        ValueError
            If any color code is invalid
            If label_position is not 'above' or 'below'

        Examples
        --------
        Basic usage:
        >>> colors = ['#FF5733', '#33FF57', '#3357FF']
        >>> visualize_colors(colors)

        Customized visualization:
        >>> visualize_colors(
        ...     colors,
        ...     figsize=(12, 3),
        ...     label_position='above',
        ...     label_rotation=30,
        ...     background_color='#f0f0f0',
        ...     edge_color='black'
        ... )

        Notes
        -----
        - All hex colors will be converted to lowercase for consistency
        - For large numbers of colors, consider increasing figsize or decreasing label_size
        - Edge colors can be used to improve visibility against similar backgrounds
    """

    # Convert RGB colors to hex if needed
    hex_colors = harmonize_colors(colors)

    # Validate colors
    for color in hex_colors:
        if not is_color_like(color):
            raise ValueError(f"Invalid color code: {color}")

    num_colors = len(hex_colors)
    if num_colors == 0:
        return

    # Create figure with specified background
    fig, ax = plt.subplots(figsize=figsize, facecolor=background_color)
    fig.tight_layout(pad=2)

    # Calculate dimensions
    rect_width = 1.0
    total_width = num_colors * rect_width
    rect_height = total_width * aspect_ratio

    # Automatic label size calculation if not specified
    if label_size is None:
        label_size = max(6, min(12, 100 / num_colors))

    # Set axis limits (with extra space for labels)
    y_offset = rect_height + spacing if label_position == "above" else -spacing
    ax.set_xlim(0, total_width)
    ax.set_ylim(
        -spacing if label_position == "below" else 0,
        rect_height + (spacing if label_position == "above" else 0),
    )

    # Remove axes for clean look
    ax.axis("off")

    # Determine edge color if not specified
    if edge_color is None:
        edge_color = "black" if background_color != "black" else "white"

    # Draw each color rectangle and label
    for i, color in enumerate(hex_colors):
        x_pos = i * rect_width

        # Draw the color rectangle (fixed property setting)
        rect = plt.Rectangle(
            (x_pos, 0),
            width=rect_width,
            height=rect_height,
            facecolor=color,
            linewidth=0.5 if edge_color else 0,
            edgecolor=edge_color,
        )
        ax.add_patch(rect)

        # Add the label
        label_y = (
            -0.02 * rect_height
            if label_position == "below"
            else rect_height + 0.02 * rect_height
        )
        va = "top" if label_position == "below" else "bottom"

        ax.text(
            x_pos + rect_width / 2,
            label_y,
            color.upper(),
            ha="center",
            va=va,
            rotation=label_rotation,
            fontsize=label_size,
            color="black" if background_color != "black" else "white",
            fontfamily="monospace",
        )

    # Adjust aspect ratio
    ax.set_aspect("auto")
    plt.show()


######################################################################################################
class ColorTableLoader:
    """Class for loading and managing color lookup tables."""

    def __init__(self, ctab_file: Union[str, Path, dict]):
        """
        Initialize ColorTableLoader by loading a color lookup table from a file.

        Parameters
        ----------
        ctab_file : str, Path, or dict
            Path to the color lookup table file (.txt, .lut, or .tsv)
            or a dictionary containing color table data.
            Attributes
            ----------
            index : list of int
                List of integer region codes (standard Python integers)

            name : list of str
                List of region name strings

            color : list
                List of color codes (format depends on source file)

            opacity : list of float
                List of opacity values (0-1)

            headerlines : list of str
                List of header lines from the color table file

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist

        ValueError
            If the file format cannot be determined or is invalid

        Examples
        --------
        >>> # Load a FreeSurfer LUT file
        >>> lut_loader = ColorTableLoader('FreeSurferColorLUT.txt')
        >>> print(lut_loader.index[:3])
        [0, 1, 2]
        >>> print(lut_loader.name[:3])
        """

        if isinstance(ctab_file, (str, Path)):
            col_dict = ColorTableLoader.load_colortable(
                in_file=ctab_file, filter_by_name=None
            )
        elif isinstance(ctab_file, dict):
            col_dict = copy.deepcopy(ctab_file)

            # Validate required keys (index amd name)
            required_keys = ["index"]
            for key in required_keys:
                if key not in col_dict:
                    raise ValueError(
                        f"Missing required key '{key}' in color table dictionary"
                    )

            if "name" not in col_dict.keys():
                col_dict["name"] = [f"Region_{idx}" for idx in col_dict["index"]]

            if "color" not in col_dict.keys():
                col_dict["color"] = create_distinguishable_colors(
                    n=len(col_dict["index"]), output_format="hex"
                )

            if "opacity" not in col_dict.keys():
                col_dict["opacity"] = [1.0] * len(col_dict["index"])

            if "headerlines" not in col_dict.keys():
                col_dict["headerlines"] = []

        else:
            raise ValueError("ctab_file must be a string or a dictionary")

        # Verify lengths of lists
        n_entries = len(col_dict["index"])
        for key in ["name", "color", "opacity"]:
            if len(col_dict[key]) != n_entries:
                raise ValueError(
                    f"Length of '{key}' does not match length of 'index' in color table dictionary"
                )

        self.index = col_dict["index"]
        self.name = col_dict["name"]
        self.color = col_dict["color"]
        self.opacity = col_dict["opacity"]
        self.headerlines = col_dict["headerlines"]

    @staticmethod
    def load_colortable(
        in_file: str, filter_by_name: Union[str, List[str]] = None
    ) -> dict:
        """
        Automatically detect and load a color lookup table from either LUT or TSV format.

        This method intelligently determines the file format (FreeSurfer LUT or TSV) and uses
        the appropriate parser to load the color table data. Detection is based on file extension
        and/or file content analysis.

        Parameters
        ----------
        in_file : str
            Path to the color lookup table file (.txt, .lut, or .tsv)

        filter_by_name : str or list of str, optional
            Filter regions by name substring(s). Default is None.

        Returns
        -------
        dict
            Dictionary with the following keys:
            - 'index': List of integer region codes (standard Python integers)
            - 'name': List of region name strings
            - 'color': List of color codes (format depends on source file)
            - Additional keys may be present depending on the file format

        Examples
        --------
        >>> # Load a FreeSurfer LUT file
        >>> lut_dict = ColorTableLoader.load_colortable('FreeSurferColorLUT.txt')
        >>> lut_dict['index'][:3]
        [0, 1, 2]
        >>> lut_dict['name'][:3]
        ['Unknown', 'Left-Cerebral-Exterior', 'Left-Cerebral-White-Matter']

        >>> # Load a TSV file
        >>> tsv_dict = ColorTableLoader.load_colortable('regions.tsv')
        >>> tsv_dict['index'][:3]
        [0, 1, 2]

        >>> # Load with filtering
        >>> hippo_dict = ColorTableLoader.load_colortable(
        ...     'FreeSurferColorLUT.txt',
        ...     filter_by_name='hippocampus'
        ... )

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        ValueError
            If the file format cannot be determined or is invalid

        Notes
        -----
        - File format detection uses both extension and content analysis
        - .tsv files are assumed to be TSV format
        - .txt and .lut files are analyzed to determine if they are LUT or TSV format
        - LUT format is identified by comment lines starting with '#'
        - TSV format is identified by tab-separated columns with headers
        """
        # Check if file exists
        if not os.path.exists(in_file):
            raise FileNotFoundError(f"Color table file not found: {in_file}")

        # Get file extension
        file_ext = os.path.splitext(in_file)[1].lower()

        # Detect file format
        file_format = ColorTableLoader._detect_format(in_file, file_ext)

        # Load the file using the appropriate method
        if file_format == "lut":
            colors_dict = ColorTableLoader.read_luttable(
                in_file, filter_by_name=filter_by_name
            )
        elif file_format == "tsv":
            colors_dict = ColorTableLoader.read_tsvtable(
                in_file, filter_by_name=filter_by_name
            )
        else:
            raise ValueError(f"Could not determine file format for: {in_file}")

        #  Force opacity values equal to 0 to be 255. This is because most
        # of the neuroimaging software interpret 0 opacity as fully opaque.
        #
        colors_dict["opacity"] = [
            255 if op == 0 else op for op in colors_dict["opacity"]
        ]

        # Force opacity values to be between 0 and 1
        colors_dict["opacity"] = [min(max(op, 0), 1) for op in colors_dict["opacity"]]

        return colors_dict

    @staticmethod
    def _detect_format(in_file: str, file_ext: str) -> str:
        """
        Detect the format of a color table file.

        Parameters
        ----------
        in_file : str
            Path to the file
        file_ext : str
            File extension (lowercase)

        Returns
        -------
        str
            Either 'lut' or 'tsv'

        Raises
        ------
        ValueError
            If file extension is unsupported or file is empty/invalid
        """
        if file_ext == ".tsv":
            return "tsv"

        if file_ext not in [".txt", ".lut", ""]:
            raise ValueError(
                f"Unsupported file extension: {file_ext}. Expected .txt, .lut, or .tsv"
            )

        # Analyze content for .txt, .lut, or extensionless files
        try:
            with open(in_file, "r", encoding="utf-8") as f:
                lines_to_check = [line.strip() for line in f if line.strip()]
        except UnicodeDecodeError:
            with open(in_file, "r") as f:
                lines_to_check = [line.strip() for line in f if line.strip()]

        if not lines_to_check:
            raise ValueError(f"File is empty: {in_file}")

        # Check for format indicators
        has_hash_comments = any(line.startswith("#") for line in lines_to_check)

        # Find first non-comment line
        first_non_comment = None
        for line in lines_to_check:
            if not line.startswith("#") and not line.startswith("\\\\"):
                first_non_comment = line
                break

        if not first_non_comment:
            raise ValueError(f"File contains only comments: {in_file}")

        has_tabs = "\t" in first_non_comment
        parts = first_non_comment.split("\t") if has_tabs else first_non_comment.split()

        if not parts:
            raise ValueError(f"First data line is empty or malformed: {in_file}")

        # Check if first column is numeric
        try:
            int(parts[0])
            is_numeric_first = True
        except (ValueError, IndexError):
            is_numeric_first = False

        # Determine format based on heuristics:
        # LUT format characteristics:
        #   - Has # comments
        #   - First non-comment line starts with a number (region code)
        #   - Space-separated or tab-separated
        # TSV format characteristics:
        #   - First line is a header (starts with column names like "index", "name")
        #   - Tab-separated
        #   - First column is typically text (header name)

        if has_tabs and not is_numeric_first:
            # Tab-separated with text header -> TSV
            return "tsv"
        elif has_hash_comments and is_numeric_first:
            # Has comments and numeric first column -> LUT
            return "lut"
        elif not has_tabs and is_numeric_first:
            # Space-separated with numeric first column -> LUT
            return "lut"
        elif has_tabs and is_numeric_first:
            # Tab-separated with numeric first column
            # Could be TSV without header, check for column names in parts
            if len(parts) >= 2 and parts[1].lower() in [
                "index",
                "name",
                "label",
                "region",
            ]:
                return "tsv"
            else:
                # Numeric data without clear header -> assume LUT
                return "lut"
        else:
            # Default fallback based on comments
            return "lut" if has_hash_comments else "tsv"

    @staticmethod
    def read_luttable(
        in_file: str, filter_by_name: Union[str, List[str]] = None
    ) -> dict:
        """
        Read and parse a FreeSurfer Color Lookup Table (LUT) file.

        This method reads a FreeSurfer color lookup table file and parses its contents into
        a structured dictionary containing region codes, names, and colors. The LUT file format
        follows FreeSurfer's standard format where each non-comment line contains a region code,
        name, and RGB color values.

        Parameters
        ----------
        in_file : str
            Path to the FreeSurfer color lookup table file (.txt or .lut)

        filter_by_name : str or list of str, optional
            Filter regions by name substring(s). If provided, only regions whose names
            contain any of the specified substrings will be returned. Default is None.

        Returns
        -------
        dict
            Dictionary with the following keys:
            - 'index': List of integer region codes (standard Python integers)
            - 'name': List of region name strings
            - 'color': List of hex color codes (format: '#RRGGBB')

        Examples
        --------
        >>> lut_dict = ColorTableLoader.read_luttable('FreeSurferColorLUT.txt')
        >>> print(f"Found {len(lut_dict['index'])} regions")
        Found 1234 regions

        >>> lut_dict['index'][:3]
        [0, 1, 2]
        >>> lut_dict['name'][:3]
        ['Unknown', 'Left-Cerebral-Exterior', 'Left-Cerebral-White-Matter']
        >>> lut_dict['color'][:3]
        ['#000000', '#4682b4', '#f5f5f5']

        >>> # Filter for hippocampus regions
        >>> hippo_dict = ColorTableLoader.read_luttable(
        ...     'FreeSurferColorLUT.txt',
        ...     filter_by_name='hippocampus'
        ... )
        >>> print(hippo_dict['name'])
        ['Left-Hippocampus', 'Right-Hippocampus', ...]

        >>> # Filter for multiple patterns
        >>> regions = ColorTableLoader.read_luttable(
        ...     'FreeSurferColorLUT.txt',
        ...     filter_by_name=['hippocampus', 'amygdala']
        ... )

        Notes
        -----
        - Comment lines (starting with '#') in the LUT file are ignored
        - Each non-comment line should have at least 5 elements: code, name, R, G, B
        - The returned region codes are standard Python integers, not numpy objects
        - Color values are converted from RGB to hexadecimal format
        - Filtering is case-insensitive and matches substrings
        """
        # Read the LUT file content
        try:
            with open(in_file, "r", encoding="utf-8") as f:
                lut_content = f.readlines()
        except UnicodeDecodeError:
            with open(in_file, "r") as f:
                lut_content = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"LUT file not found: {in_file}")
        except PermissionError:
            raise PermissionError(
                f"Permission denied when accessing LUT file: {in_file}"
            )

        # Initialize lists to store parsed data
        region_codes = []
        region_names = []
        region_colors_rgb = []
        region_opacities = []

        # Parse each non-comment line in the file
        headerlines = []
        for line in lut_content:
            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("\\\\"):
                parts = line.split()
                if (
                    line.startswith("#")
                    and parts[-1].lower() != "a"
                    and parts[-2].lower() != "b"
                ):
                    headerlines.append(line)

            # Split line into components
            parts = line.split()
            if len(parts) < 5:  # Need at least code, name, R, G, B
                continue

            # Extract data
            try:
                code = int(parts[0])  # Using Python's built-in int
                name = parts[1]

                if len(parts) == 6:
                    r, g, b, o = (
                        int(parts[2]),
                        int(parts[3]),
                        int(parts[4]),
                        int(parts[5]),
                    )
                else:
                    r, g, b = int(parts[2]), int(parts[3]), int(parts[4])
                    o = 1  # Default opacity if not provided

                region_codes.append(code)
                region_names.append(name)
                region_colors_rgb.append([r, g, b])
                region_opacities.append(o)

            except (ValueError, IndexError):
                # Skip malformed lines
                continue

        # Convert RGB colors to hex format
        try:
            # Use the existing multi_rgb2hex function if available
            region_colors_hex = multi_rgb2hex(np.array(region_colors_rgb))
        except (NameError, AttributeError):
            # Fallback to direct conversion if the function isn't available
            region_colors_hex = [
                f"#{r:02x}{g:02x}{b:02x}" for r, g, b in region_colors_rgb
            ]

        # Apply name filtering if requested
        if filter_by_name is not None:
            if isinstance(filter_by_name, str):
                filter_by_name = [filter_by_name]

            filtered_indices = cltmisc.get_indexes_by_substring(
                region_names, filter_by_name
            )

            # Filter the LUT based on the provided names
            region_codes = [region_codes[i] for i in filtered_indices]
            region_names = [region_names[i] for i in filtered_indices]
            region_colors_hex = [region_colors_hex[i] for i in filtered_indices]
            region_opacities = [region_opacities[i] for i in filtered_indices]

        # Create and return the result dictionary
        return {
            "index": region_codes,
            "name": region_names,
            "color": region_colors_hex,
            "opacity": region_opacities,
            "headerlines": headerlines,
        }

    @staticmethod
    def read_tsvtable(
        in_file: str, filter_by_name: Union[str, List[str]] = None
    ) -> dict:
        """
        Read and parse a TSV (Tab-Separated Values) lookup table file.

        This method reads a TSV file containing parcellation information and returns a dictionary
        with the data. The TSV file must contain at least 'index' and 'name' columns. If a 'color'
        column is present, it will be included in the returned dictionary.

        Parameters
        ----------
        in_file : str
            Path to the TSV lookup table file

        filter_by_name : str or list of str, optional
            Filter regions by name substring(s). If provided, only regions whose names
            contain any of the specified substrings will be returned. Default is None.

        Returns
        -------
        dict
            Dictionary with keys corresponding to column names in the TSV file.
            Must include at least:
            - 'index': List of integer region codes (standard Python integers)
            - 'name': List of region name strings
            May also include:
            - 'color': List of color codes if present in the TSV file
            - Any other columns present in the TSV file

        Raises
        ------
        FileNotFoundError
            If the specified TSV file does not exist
        ValueError
            If the TSV file does not contain required 'index' and 'name' columns,
            is empty, or cannot be parsed

        Examples
        --------
        >>> tsv_dict = ColorTableLoader.read_tsvtable('regions.tsv')
        >>> print(f"Columns: {list(tsv_dict.keys())}")
        Columns: ['index', 'name', 'color', 'abbreviation']

        >>> tsv_dict['index'][:3]
        [0, 1, 2]
        >>> tsv_dict['name'][:3]
        ['Unknown', 'Left-Cerebral-Exterior', 'Left-Cerebral-White-Matter']

        >>> # Filter for specific regions
        >>> filtered = ColorTableLoader.read_tsvtable(
        ...     'regions.tsv',
        ...     filter_by_name=['cortex', 'hippocampus']
        ... )

        Notes
        -----
        - The 'index' column values are converted to standard Python integers
        - All other columns are preserved in their original format
        - Filtering is case-insensitive and matches substrings
        """
        # Check if file exists
        if not os.path.exists(in_file):
            raise FileNotFoundError(f"TSV file not found: {in_file}")

        try:
            # Read the TSV file into a pandas DataFrame
            tsv_df = pd.read_csv(in_file, sep="\t")

            # Check for required columns
            required_columns = ["index", "name"]
            missing_columns = [
                col for col in required_columns if col not in tsv_df.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"TSV file missing required columns: {', '.join(missing_columns)}"
                )

            # Convert DataFrame to dictionary
            tsv_dict = tsv_df.to_dict(orient="list")

            # Convert index values to integers
            if "index" in tsv_dict:
                tsv_dict["index"] = [int(x) for x in tsv_dict["index"]]

            if "opacity" in tsv_dict:
                tsv_dict["opacity"] = [tsv_dict["opacity"][i] for i in filtered_indices]
            else:
                tsv_dict["opacity"] = [1] * len(tsv_dict["index"])

            # Apply name filtering if requested
            if filter_by_name is not None:
                if isinstance(filter_by_name, str):
                    filter_by_name = [filter_by_name]

                filtered_indices = cltmisc.get_indexes_by_substring(
                    tsv_dict["name"], filter_by_name
                )

                # Filter all columns based on the provided names
                tsv_dict = {
                    key: [tsv_dict[key][i] for i in filtered_indices]
                    for key in tsv_dict.keys()
                }

                #

            tsv_dict["headerlines"] = []

            return tsv_dict

        except pd.errors.EmptyDataError:
            raise ValueError(
                f"The TSV file is empty or improperly formatted: {in_file}"
            )
        except pd.errors.ParserError:
            raise ValueError(f"The TSV file could not be parsed correctly: {in_file}")
        except Exception as e:
            raise ValueError(f"Error reading TSV file {in_file}: {str(e)}")

    @staticmethod
    def write_luttable(
        lut_df: Union[pd.DataFrame, dict],
        out_file: str = None,
        boolappend: bool = False,
        force: bool = True,
    ):
        """
        Write a FreeSurfer format lookup table file.

        This method creates a FreeSurfer-compatible color lookup table file from region
        codes, names, and colors. The output follows the standard FreeSurfer LUT format
        with optional header lines.

        Parameters
        ----------
        lut_df : pd.DataFrame or dict
            DataFrame or dictionary containing the following keys/columns:
            - 'index': List of integer region codes
            - 'name': List of region name strings
            - 'color': List of colors in RGB format (as list/array of [R, G, B] values) or
            hexadecimal format (as list of '#RRGGBB' strings)
            - 'opacity': List of opacity values (0-1 range)
            - 'headerlines': Optional list of header lines to include at the top of the file. If the list is empty, a default
            header with timestamp will be generated. Default is None.
            - If a dictionary is provided, it must contain the same keys.

        out_file : str, optional
            Output file path. If None, returns formatted lines without writing to file.
            Default is None.

        boolappend : bool, optional
            If True, append to existing file. If False, create new file or overwrite.
            Default is False.

        force : bool, optional
            If True, overwrite existing files without warning. If False, warn before
            overwriting. Default is True.

        Returns
        -------
        list
            List of formatted LUT lines as strings

        Examples
        --------
        >>> # Create a simple LUT file
        >>> ColorTableLoader.write_luttable(
        ...     codes=[1, 2, 3],
        ...     names=['region1', 'region2', 'region3'],
        ...     colors=['#FF0000', '#00FF00', '#0000FF'],
        ...     out_file='output.lut'
        ... )

        >>> # Use RGB colors instead
        >>> ColorTableLoader.write_luttable(
        ...     codes=[1, 2],
        ...     names=['cortex', 'white_matter'],
        ...     colors=[[255, 0, 0], [255, 255, 255]],
        ...     out_file='parcellation.lut'
        ... )

        >>> # Append to existing file
        >>> ColorTableLoader.write_luttable(
        ...     codes=[4],
        ...     names=['new_region'],
        ...     colors=['#FFFF00'],
        ...     out_file='output.lut',
        ...     boolappend=True
        ... )

        Notes
        -----
        - Output format follows FreeSurfer LUT specification
        - RGB values should be in range [0, 255]
        - Hex colors should be in format '#RRGGBB'
        - Alpha channel is always set to 0 in output
        """

        codes = lut_df["index"]
        names = lut_df["name"]
        colors = lut_df["color"]
        opacities = lut_df["opacity"]
        headerlines = []

        # Move the opacity to the range of 0-255
        opacities = [int(op * 255) for op in opacities]

        # Check if the file already exists and if the force parameter is False
        if out_file is not None:
            if os.path.exists(out_file) and not force:
                print("Warning: The file already exists. It will be overwritten.")

            out_dir = os.path.dirname(out_file)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)

        happend_bool = True  # Boolean to append the headerlines
        if headerlines is None:
            happend_bool = (
                False  # Only add this if it is the first time the file is created
            )
            now = datetime.now()
            date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
            headerlines = [
                "# $Id: {} {} \n".format(out_file, date_time),
            ]
        elif isinstance(headerlines, str):
            headerlines = [headerlines]
        elif isinstance(headerlines, list):
            pass
        else:
            raise ValueError("The headerlines parameter must be a list or a string")

        if boolappend:
            if not os.path.exists(out_file):
                raise ValueError(f"Cannot append: file does not exist: {out_file}")
            else:
                with open(out_file, "r") as file:
                    luttable = file.readlines()

                luttable = [l.strip("\n\r") for l in luttable]
                luttable = ["\n" if element == "" else element for element in luttable]

                if happend_bool:
                    luttable = luttable + headerlines
        else:
            luttable = headerlines

        luttable.append("#\n")
        luttable.append(
            "{:<4} {:<50} {:>3} {:>3} {:>3} {:>3}".format(
                "#No.", "Label Name:", "R", "G", "B", "A"
            )
        )

        # Handle different color input formats
        if isinstance(colors, list):
            if isinstance(colors[0], str):
                colors = harmonize_colors(colors)
                colors = multi_hex2rgb(colors)
            elif isinstance(colors[0], list):
                colors = np.array(colors)
            elif isinstance(colors[0], np.ndarray):
                colors = np.vstack(colors)
        elif isinstance(colors, np.ndarray):
            pass  # Already in correct format
        else:
            raise ValueError("Colors must be a list or numpy array")

        # Add regions to table
        for roi_pos, roi_name in enumerate(names):
            if roi_pos == 0:
                luttable.append("\n")

            luttable.append(
                "{:<4} {:<50} {:>3} {:>3} {:>3} {:>3}".format(
                    codes[roi_pos],
                    names[roi_pos],
                    colors[roi_pos, 0],
                    colors[roi_pos, 1],
                    colors[roi_pos, 2],
                    opacities[roi_pos],
                )
            )
        luttable.append("\n")

        # Write to file if path provided
        if out_file is not None:
            if os.path.isfile(out_file) and force:
                with open(out_file, "w") as colorLUT_f:
                    colorLUT_f.write("\n".join(luttable))
            elif not os.path.isfile(out_file):
                with open(out_file, "w") as colorLUT_f:
                    colorLUT_f.write("\n".join(luttable))

        return luttable

    @staticmethod
    def write_tsvtable(
        tsv_df: Union[pd.DataFrame, dict],
        out_file: str,
        boolappend: bool = False,
        force: bool = False,
    ):
        """
        Write a TSV format lookup table file.

        This method creates a tab-separated values (TSV) file from a pandas DataFrame or
        dictionary containing parcellation information. The data must include at least
        'index' and 'name' columns/keys.

        Parameters
        ----------
        tsv_df : pd.DataFrame or dict
            Data to write with index/name/color information. Must contain at least
            'index' and 'name' keys/columns.

        out_file : str
            Output file path for the TSV file

        boolappend : bool, optional
            If True, append to existing TSV file. If False, create new file or overwrite.
            Default is False.

        force : bool, optional
            If True, overwrite existing files without warning. If False, warn before
            overwriting. Default is False.

        Returns
        -------
        str
            Path to the output TSV file

        Raises
        ------
        ValueError
            If the input data does not contain required 'index' and 'name' keys/columns,
            if colors are not in hexadecimal format, or if append is requested but
            file doesn't exist

        Examples
        --------
        >>> # Write from dictionary
        >>> data = {
        ...     'index': [1, 2, 3],
        ...     'name': ['region1', 'region2', 'region3'],
        ...     'color': ['#FF0000', '#00FF00', '#0000FF']
        ... }
        >>> ColorTableLoader.write_tsvtable(data, 'regions.tsv', force=True)
        'regions.tsv'

        >>> # Write from DataFrame
        >>> import pandas as pd
        >>> df = pd.DataFrame(data)
        >>> ColorTableLoader.write_tsvtable(df, 'regions.tsv', force=True)

        >>> # Append to existing file
        >>> new_data = {
        ...     'index': [4],
        ...     'name': ['region4'],
        ...     'color': ['#FFFF00']
        ... }
        >>> ColorTableLoader.write_tsvtable(new_data, 'regions.tsv', boolappend=True)

        Notes
        -----
        - Output is tab-separated with column headers
        - RGB colors are automatically converted to hexadecimal format
        - When appending, columns are matched by name; missing values are filled with empty strings
        - The output file includes a header row with column names
        """
        # Check if the file already exists and if the force parameter is False
        if os.path.exists(out_file) and not force and not boolappend:
            print("Warning: The TSV file already exists. It will be overwritten.")

        out_dir = os.path.dirname(out_file)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Convert DataFrame to dictionary if needed
        if isinstance(tsv_df, pd.DataFrame):
            tsv_dict = tsv_df.to_dict(orient="list")
        else:
            tsv_dict = tsv_df.copy()  # Create a copy to avoid modifying original

        # Validate required columns
        if "name" not in tsv_dict.keys() or "index" not in tsv_dict.keys():
            raise ValueError("The dictionary must contain the keys 'index' and 'name'")

        # Process colors if present
        if "color" in tsv_dict.keys():
            temp_colors = tsv_dict["color"]

            if isinstance(temp_colors, list):
                if isinstance(temp_colors[0], str):
                    if temp_colors[0][0] != "#":
                        raise ValueError(
                            "The colors must be in hexadecimal format (starting with #)"
                        )
                elif isinstance(temp_colors[0], list):
                    colors = np.array(temp_colors)
                    seg_hexcol = multi_rgb2hex(colors)
                    tsv_dict["color"] = seg_hexcol
            elif isinstance(temp_colors, np.ndarray):
                seg_hexcol = multi_rgb2hex(temp_colors)
                tsv_dict["color"] = seg_hexcol

        # Handle append mode
        if boolappend:
            if not os.path.exists(out_file):
                raise ValueError(f"Cannot append: file does not exist: {out_file}")
            else:
                tsv_orig = ColorTableLoader.read_tsvtable(in_file=out_file)

                # Create a list with the common keys between tsv_orig and tsv_dict
                common_keys = list(set(tsv_orig.keys()) & set(tsv_dict.keys()))

                # List all the keys for both dictionaries
                all_keys = list(set(tsv_orig.keys()) | set(tsv_dict.keys()))

                # Concatenate values for common keys
                for key in common_keys:
                    tsv_orig[key] = tsv_orig[key] + tsv_dict[key]

                # Fill missing values for non-common keys
                for key in all_keys:
                    if key not in common_keys:
                        if key in tsv_orig.keys():
                            tsv_orig[key] = tsv_orig[key] + [""] * len(tsv_dict["name"])
                        elif key in tsv_dict.keys():
                            tsv_orig[key] = [""] * len(tsv_orig["name"]) + tsv_dict[key]

                tsv_dict = tsv_orig

        # Convert dictionary to DataFrame
        tsv_df = pd.DataFrame(tsv_dict)

        # Write to file
        if os.path.isfile(out_file) and force:
            with open(out_file, "w") as tsv_file:
                tsv_file.write(tsv_df.to_csv(sep="\t", index=False))
        elif not os.path.isfile(out_file):
            with open(out_file, "w") as tsv_file:
                tsv_file.write(tsv_df.to_csv(sep="\t", index=False))

        return out_file

    #################################################
    def export(
        self,
        out_ctab: Union[str, Path],
        out_format: str = "fsl",
        headerlines: Union[list, str] = None,
        append: bool = False,
        overwrite: bool = True,
    ):
        """
        Export the loaded color table to specified format.

        Parameters
        ----------
        out_ctab : str
            Path for output color table file.

        out_format : str, optional
            Output format. Options are 'lut', 'tsv', 'fsl' or 'nilearn'.
            Default is 'fsl'.

        overwrite : bool, optional
            Whether to overwrite the output file if it already exists.
            Default: True

        Examples
        --------
        >>> parcellation.export_to_file('fsl_colors.lut', out_format='fslctab')
        >>> parcellation.export_to_file('nilearn_colors.tsv', out_format='nilearnctab')
        """

        if out_format.lower() == "fsl":
            self.export_to_fslctab(out_ctab, overwrite=overwrite)

        elif out_format.lower() == "nilearn":
            self.export_to_nilearnctab(out_ctab, overwrite=overwrite)

        elif out_format.lower() == "lut":
            self.export_to_lutctab(
                out_ctab, overwrite=overwrite, headerlines=headerlines, append=append
            )

        elif out_format.lower() == "tsv":
            self.export_to_tsvctab(out_ctab, overwrite=overwrite)

        else:
            raise ValueError(
                f"Unsupported output format: {out_format}. "
                "Supported formats are 'lut', 'tsv', 'fsl', or 'nilearn'."
            )

    ######################################################################################################
    def export_to_fslctab(self, out_ctab: Union[str, Path], overwrite: bool = True):
        """
        Export the loaded color table to FSL LUT format.

        Parameters
        ----------
        out_ctab : str
            Path for output FSL LUT file.

        Examples
        --------

        """

        # Convert to Path objects
        if isinstance(out_ctab, str):
            out_ctab = Path(out_ctab)

        # Check if output directory exists
        if not out_ctab.parent.exists():
            raise FileNotFoundError(
                f"Output directory does not exist: {out_ctab.parent}"
            )

        # Check if output file exists and handle overwrite
        if out_ctab.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {out_ctab}. Use overwrite=True to overwrite."
            )

        st_codes_lut = self.index
        st_names_lut = self.name
        st_colors_lut = harmonize_colors(self.color, output_format="rgb")

        lut_lines = []
        for roi_pos, st_code in enumerate(st_codes_lut):
            st_name = st_names_lut[roi_pos]
            lut_lines.append(
                "{:<4} {:>3.5f} {:>3.5f} {:>3.5f} {:<40} ".format(
                    st_code,
                    st_colors_lut[roi_pos, 0] / 255,
                    st_colors_lut[roi_pos, 1] / 255,
                    st_colors_lut[roi_pos, 2] / 255,
                    st_name,
                )
            )

        if os.path.isfile(out_ctab) or overwrite:
            with open(out_ctab, "w") as colorLUT_f:
                colorLUT_f.write("\n".join(lut_lines))

    ######################################################################################################
    def export_to_nilearnctab(
        self, out_ctab: Union[str, Path], overwrite: bool = False
    ) -> str:
        """
        Export the color table to nilearn-compatible format.

        This function reads a color lookup table and converts it to a pandas-readable format
        that nilearn's NiftiLabelsMasker can use.

        Parameters
        ----------

        out_ctab : str or Path
            Path for the output file. Directory must exist.

        overwrite : bool, optional
            Whether to overwrite the output file if it already exists.
            Default: False


        Raises
        ------
        FileNotFoundError
            If input_lut_path doesn't exist or output directory doesn't exist
        FileExistsError
            If output file exists and overwrite=False
        ValueError
            If no valid data lines are found in the input file

        Examples
        --------

        """
        # Convert to Path objects
        if isinstance(out_ctab, str):
            out_ctab = Path(out_ctab)

        # Check if output directory exists
        if not out_ctab.parent.exists():
            raise FileNotFoundError(
                f"Output directory does not exist: {out_ctab.parent}"
            )

        # Check if output file exists and handle overwrite
        if out_ctab.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {out_ctab}. Use overwrite=True to overwrite."
            )

        rgb_colors = harmonize_colors(self.color, output_format="rgb")
        r = rgb_colors[:, 0]
        g = rgb_colors[:, 1]
        b = rgb_colors[:, 2]

        if self.opacity is not None:
            # Ensure opacity values are in the correct range [0, 255]
            a = [min(max(int(opacity * 255), 0), 255) for opacity in self.opacity]

        else:
            a = [255] * len(self.color)

        df = pd.DataFrame(
            {
                "index": self.index,
                "name": self.name,
                "R": r,
                "G": g,
                "B": b,
                "A": a,
            }
        )

        df = df.sort_values("index").reset_index(drop=True)

        if not os.path.isfile(out_ctab) or overwrite:
            df.to_csv(out_ctab, sep=" ", index=False)

    ######################################################################################################
    def export_to_lutctab(
        self,
        out_ctab: Union[str, Path] = None,
        headerlines: Union[list, str] = None,
        append: bool = False,
        overwrite: bool = False,
    ) -> str:
        """
        Export the color table to LUT format.
        This function writes the color table to a FreeSurfer-compatible LUT file.

        Parameters
        ----------
        out_ctab : str or Path
            Path for the output LUT file. If None, returns the LUT lines as a list of strings.

        headerlines : list or str, optional
            Custom header lines to include at the top of the file. If None, a default
            header with timestamp will be generated. Default is None.

        append : bool, optional
            If True, append to existing file. If False, create new file or overwrite.
            Default is False.

        overwrite : bool, optional
            Whether to overwrite the output file if it already exists.
            Default: False

        Raises
        ------
        FileNotFoundError
            If output directory doesn't exist

        FileExistsError
            If output file exists and overwrite=False

        """

        if out_ctab is not None:
            # Convert to Path objects
            if isinstance(out_ctab, str):
                out_ctab = Path(out_ctab)

            # Check if output file exists and handle overwrite
            if out_ctab.exists() and not overwrite and not append:
                raise FileExistsError(
                    f"Output file already exists: {out_ctab}. Use overwrite=True to overwrite."
                )

        # Write LUT file
        codes = self.index
        names = self.name
        colors = self.color

        if self.opacity is not None:
            opacities = self.opacity

            # Ensure opacity values are in the correct range [0, 255]
            opacities = [min(max(int(opacity * 255), 0), 255) for opacity in opacities]

        else:
            opacities = [255] * len(self.color)

        if headerlines is None:
            happend_bool = (
                False  # Only add this if it is the first time the file is created
            )
            now = datetime.now()
            date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
            headerlines = ["# $Id: {} {} \n".format(str(out_ctab), date_time)]
        elif isinstance(headerlines, str):
            headerlines = [headerlines]

        elif isinstance(headerlines, list):
            pass
        else:
            raise ValueError("The headerlines parameter must be a list or a string")

        if append:
            with open(str(out_ctab), "r") as file:
                luttable = file.readlines()

            luttable = [l.strip("\n\r") for l in luttable]
            luttable = ["\n" if element == "" else element for element in luttable]

            if happend_bool:
                luttable = luttable + headerlines

        else:
            luttable = headerlines

        headerlines.append(
            "{:<4} {:<50} {:>3} {:>3} {:>3} {:>3}".format(
                "#No.", "Label Name:", "R", "G", "B", "A"
            )
        )

        # Handle different color input formats
        if isinstance(colors, list):
            if isinstance(colors[0], str):
                colors = harmonize_colors(colors)
                colors = multi_hex2rgb(colors)

            elif isinstance(colors[0], list):
                colors = np.array(colors)

            elif isinstance(colors[0], np.ndarray):
                colors = np.vstack(colors)

        elif isinstance(colors, np.ndarray):
            pass  # Already in correct format

        else:
            raise ValueError("Colors must be a list or numpy array")

        # Add regions to table
        for roi_pos, roi_name in enumerate(names):
            if roi_pos == 0:
                luttable.append("\n")

            luttable.append(
                "{:<4} {:<50} {:>3} {:>3} {:>3} {:>3}".format(
                    codes[roi_pos],
                    names[roi_pos],
                    colors[roi_pos, 0],
                    colors[roi_pos, 1],
                    colors[roi_pos, 2],
                    opacities[roi_pos],
                )
            )
        luttable.append("\n")

        # Write to file if path provided
        if out_ctab is not None:
            if not os.path.isfile(out_ctab) or overwrite:
                with open(out_ctab, "w") as colorLUT_f:
                    colorLUT_f.write("\n".join(luttable))

            return str(out_ctab)

        else:
            return luttable

    ###########################################################################################
    def export_to_tsvctab(
        self, out_ctab: Union[str, Path] = None, overwrite: bool = False
    ) -> str:
        """
        Export the color table to TSV format.

        This function writes the color table to a tab-separated values (TSV) file.

        Parameters
        ----------
        out_ctab : str or Path
            Path for the output TSV file.

        overwrite : bool, optional
            Whether to overwrite the output file if it already exists.
            Default: False

        Raises
        ------
        FileNotFoundError
            If output directory doesn't exist

        FileExistsError
            If output file exists and overwrite=False

        """

        if out_ctab is not None:
            # Convert to Path objects
            if isinstance(out_ctab, str):
                out_ctab = Path(out_ctab)

            # Check if output directory exists
            if not out_ctab.parent.exists():
                raise FileNotFoundError(
                    f"Output directory does not exist: {out_ctab.parent}"
                )

            # Check if output file exists and handle overwrite
            if out_ctab.exists() and not overwrite:
                raise FileExistsError(
                    f"Output file already exists: {out_ctab}. Use overwrite=True to overwrite."
                )

        colors = harmonize_colors(self.color, output_format="hex")
        opaciity = self.opacity if self.opacity is not None else [1] * len(self.color)

        df = pd.DataFrame(
            {
                "index": self.index,
                "name": self.name,
                "color": colors,
                "opacity": opaciity,
            }
        )

        df = df.sort_values("index").reset_index(drop=True)

        if out_ctab is not None:
            if not os.path.isfile(out_ctab) or overwrite:
                df.to_csv(out_ctab, sep="\t", index=False)

            return str(out_ctab)

        else:
            return df
