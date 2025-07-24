import numpy as np
import h5py

from typing import Union, Dict, List, Tuple, Set, Any, Optional

import shlex
import os
import argparse
from datetime import datetime
import pandas as pd
import inspect
import sys
import types
import re
import json
import pandas as pd
import inspect
import types

from pathlib import Path
from colorama import init, Fore, Style, Back

init(autoreset=True)

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.colors import is_color_like as mpl_is_color_like
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from typing import Union, List, Optional

####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############         Section 1: Methods dedicated to improve the documentation          ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################


class SmartFormatter(argparse.HelpFormatter):
    """
    Class to format the help message

    This class is used to format the help message in the argparse module. It allows to use the "R|" prefix to print the help message as raw text.

    For example:
    parser = argparse.ArgumentParser(description='''R|This is a raw text help message.
    It can contain multiple lines.
    It will be printed as raw text.''', formatter_class=SmartFormatter)

    parser.print_help()

    Parameters
    ----------
    argparse : argparse.HelpFormatter
        HelpFormatter class from the argparse module

    Returns
    -------
    argparse.HelpFormatter
        HelpFormatter class from the argparse module

    """

    ###################################################################################################
    def split_lines(self, text, width):
        """
        This function is used to split the lines of the help message.
        It allows to use the "R|" prefix to print the help message as raw text.
        For example:
        parser = argparse.ArgumentParser(description='''R|This is a raw text help message.
        It can contain multiple lines.
        It will be printed as raw text.''', formatter_class=SmartFormatter)
        parser.print_help()

        Parameters
        ----------
        text : str
            Text to be split
        width : int
            Width of the text

        Returns
        -------
        text : str
            Text split in lines

        """
        if text.startswith("R|"):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter.split_lines
        return argparse.HelpFormatter.split_lines(self, text, width)


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############          Section 2: Methods dedicated to work with progress bar            ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################


# Print iterations progress
def printprogressbar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printend="\r",
):
    """
    Call in a loop to create terminal progress bar

    Parameters
    ----------

        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printend    - Optional  : end character (e.g. "\r", "\r\n") (Str)

    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledlength = int(length * iteration // total)
    bar = fill * filledlength + "-" * (length - filledlength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printend)
    # Print New Line on Complete
    if iteration == total:
        print()


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############              Section 3: Methods dedicated to work with colors              ############
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

    How to Use:
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

    How to Use:
    --------------
        >>> colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        >>> hexcodes = multi_rgb2hex(colors)
        >>> print(hexcodes)  # Output: ['#ff0000', '#00ff00', '#0000ff']

    """

    # Harmonizing the colors
    hexcodes = harmonize_colors(colors, output_format="hex")

    return hexcodes


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

    How to Use:
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

    How to Use:
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
    colors: Union[List[Union[str, list, np.ndarray]], np.ndarray],
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

    How to Use:
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


###################################################################################################
def values2colors(
    values: Union[List[Union[int, float]], np.ndarray],
    cmap: str = "viridis",
    output_format: str = "hex",
    invert_cl: bool = False,
    invert_clmap: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
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
        Minimum value for colormap normalization. If None, uses min(values).
    vmax : float or None, default None
        Maximum value for colormap normalization. If None, uses max(values).

    Returns
    -------
    colors : list of str or numpy.ndarray
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

    output_format = output_format.lower()
    if output_format not in ["hex", "rgb", "rgbnorm"]:
        raise ValueError("output_format must be 'hex', 'rgb', or 'rgbnorm'")

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

    # Set vmin and vmax
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)

    # Handle edge cases
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0
    if vmax == vmin:
        # All values are the same, map to middle of colormap
        normalized_values = np.full_like(values, 0.5, dtype=float)
    else:
        # Normalize values to [0, 1] range using vmin/vmax
        normalized_values = (values - vmin) / (vmax - vmin)

    # Clip values to [0, 1] range
    normalized_values = np.clip(normalized_values, 0, 1)

    # Handle NaN values - map to a neutral color (middle of colormap)
    nan_mask = ~np.isfinite(values)
    normalized_values[nan_mask] = 0.5

    # Map normalized values to colors using the continuous colormap
    mapped_colors = colormap(normalized_values)  # This returns RGBA values in [0,1]

    # Remove alpha channel if present (take only RGB)
    if mapped_colors.shape[-1] == 4:
        mapped_colors = mapped_colors[..., :3]

    # Convert to the desired output format
    if output_format == "rgbnorm":
        result_colors = mapped_colors
    elif output_format == "rgb":
        result_colors = (mapped_colors * 255).astype(np.uint8)
    else:  # hex
        result_colors = [
            f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            for r, g, b in mapped_colors
        ]

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


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############              Section 4: Methods dedicated to work with dates               ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################


def find_closest_date(dates_list: list, target_date: str, date_fmt: str = "%Y%m%d"):
    """
    Function to find the closest date in a list of dates with respect to a target date.
    It also returns the index of the closest date in the list.

    Parameters
    ----------
    dates_list : list
        List of dates in string format.

    target_date : str
        Target date in string format.

    date_fmt : str
        Date format. Default is '%Y%m%d'

    Returns
    -------
    closest_date: str
        Closest date in the list to the target date

    closest_index: int
        Index of the closest date in the list

    time_diff: int
        Time difference in days between the target date and the closest date in the list.
        If the target date is not in the list, it will return the time difference in days.

    How to Use:
    --------------
        >>> dates_list = ["20230101", "20230201", "20230301"]
        >>> target_date = "20230215"
        >>> closest_date, closest_index, time_diff = find_closest_date(dates_list, target_date)
        >>> print(closest_date)  # Output: "20230201"
        >>> print(closest_index)  # Output: 1
        >>> print(time_diff)      # Output: 14

    Raises
    ------
    ValueError
        If the target_date is not in the correct format or if the dates_list is empty.

    TypeError
        If the target_date is not a string or if the dates_list is not a list of strings.

    """

    # Convert target_date to a datetime object
    target_date = datetime.strptime(str(target_date), date_fmt)

    # Convert all dates in the list to datetime objects
    dates_list_dt = [datetime.strptime(str(date), date_fmt) for date in dates_list]

    # Find the index of the date with the minimum difference from the target date
    closest_index = min(
        range(len(dates_list_dt)), key=lambda i: abs(dates_list_dt[i] - target_date)
    )

    # Get the closest date from the list using the index
    closest_date = dates_list_dt[closest_index]

    # Get the time difference between the target date and the closest date in days
    time_diff = abs(closest_date - target_date).days

    # Convert the closest date back to the 'YYYYMMDD' format
    return closest_date.strftime(date_fmt), closest_index, time_diff


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############      Section 5: Methods dedicated to create and work with indices,         ############
############           to search for elements in a list, etc                            ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################


def build_indices(
    range_vector: List[Union[int, tuple, list, str, np.ndarray]], nonzeros: bool = True
) -> List[int]:
    """
    Build a list of unique, sorted indices from a vector containing integers, tuples, lists,
    NumPy arrays, or strings representing values, ranges, or comma-separated expressions.

    Supports:
        - Integers: added as-is.
        - Tuples of 2 integers: expanded into range(start, end+1).
        - Lists or np.ndarray: flattened and added as integers.
        - Strings:
            - "8-10"       → [8, 9, 10]
            - "11:13"      → [11, 12, 13]
            - "14:2:22"    → [14, 16, 18, 20, 22]
            - "5"          → [5]
            - "1, 2, 3"    → [1, 2, 3]
            - "1, 2, 4:10, 16-20, 25, 0" → parsed into all segments

    Parameters
    ----------
    range_vector : list of int, tuple, list, np.ndarray, or str
        The input elements to parse into a list of integers.

    nonzeros : bool, optional
        If True, zero values will be removed. Default is True.

    Returns
    -------
    List[int]
        A sorted list of unique indices.

    Raises
    ------
    ValueError
        If any item cannot be interpreted correctly.

    Example
    -------
    >>> range_vector = [1, (2, 5), [6, 7], np.array([0, 0, 0]), "8-10", "11:13", "14:2:22", "1, 2, 4:10, 16-20, 25, 0"]
    >>> build_indices(range_vector)
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 25]

    >>> build_indices(range_vector, nonzeros=False)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 25]

    """

    indexes = []

    def parse_string(expr: str) -> List[int]:
        result = []
        parts = [p.strip() for p in expr.split(",") if p.strip()]
        for part in parts:
            if "-" in part:
                start, end = map(int, part.split("-"))
                result.extend(range(start, end + 1))
            elif ":" in part:
                nums = list(map(int, part.split(":")))
                if len(nums) == 2:
                    result.extend(range(nums[0], nums[1] + 1))
                elif len(nums) == 3:
                    result.extend(range(nums[0], nums[2] + 1, nums[1]))
                else:
                    raise ValueError(f"Invalid colon-range format: '{part}'")
            else:
                result.append(int(part))
        return result

    for item in range_vector:
        try:
            if isinstance(item, (int, np.integer)):
                indexes.append([int(item)])

            elif isinstance(item, tuple) and len(item) == 2:
                start, end = item
                indexes.append(list(range(int(start), int(end) + 1)))

            elif isinstance(item, list):
                indexes.append([int(x) for x in item])

            elif isinstance(item, np.ndarray):
                if item.ndim == 0:
                    indexes.append([int(item)])
                else:
                    indexes.append([int(x) for x in item.tolist()])

            elif isinstance(item, str):
                indexes.append(parse_string(item))

            else:
                raise ValueError(f"Unsupported input type: {item}")

        except Exception as e:
            raise ValueError(f"Error processing item '{item}': {e}")

    flat = [x for sublist in indexes for x in sublist]

    if nonzeros:
        flat = [x for x in flat if x != 0]

    return sorted(set(flat))


####################################################################################################
def get_indices_by_condition(condition: str, **kwargs):
    """
    Evaluate a logical condition involving an array and optional scalar variables,
    and return the indices where the condition holds true.

    Parameters
    ----------
    condition : str
        A condition string to evaluate, e.g.:
            - "bvals > 1000"
            - "bmin <= bvals <= bmax"
            - "bvals != bval"
        Supports chained comparisons and scalar literals directly in the expression.

    **kwargs : dict
        Variable bindings for any names used in the condition string. Must include exactly
        one array (list or np.ndarray) that represents the main vector to filter.

    Returns
    -------
    np.ndarray
        Indices where the condition evaluates to True.

    Raises
    ------
    ValueError
        If:
            - The condition references variables not in kwargs (excluding literals)
            - No array variable is found
            - More than one array-like variable is provided
            - The condition does not yield a boolean array

    Examples
    --------
    >>> bvals = np.array([0, 500, 1000, 2000, 3000])
    >>> get_indices_by_condition("bvals > 1000", bvals=bvals)
    array([3, 4])

    >>> get_indices_by_condition("bmin <= bvals <= bmax", bvals=bvals, bmin=800, bmax=2500)
    array([2, 3])
    """
    condition = condition.replace(" ", "")

    # Extract all words used in the condition
    var_names = set(re.findall(r"\b[a-zA-Z_]\w*\b", condition))

    # Identify array-like variables
    array_vars = [k for k, v in kwargs.items() if isinstance(v, (list, np.ndarray))]

    if len(array_vars) != 1:
        raise ValueError("Exactly one variable must be a list or numpy array.")

    array_var = array_vars[0]

    # Check if any required variables (excluding literals) are missing
    missing_vars = var_names - set(kwargs.keys())
    if missing_vars:
        raise ValueError(f"Missing variable(s): {', '.join(missing_vars)}")

    # Convert all inputs to appropriate types for evaluation
    local_vars = {
        k: np.array(v) if isinstance(v, (list, np.ndarray)) else v
        for k, v in kwargs.items()
    }

    def rewrite_chained_comparisons(expr: str) -> str:
        # Replace "a <= b <= c" with "(a <= b) & (b <= c)"
        pattern = r"(\b\w+\b)(<=|<|>=|>)(\b\w+\b)(<=|<|>=|>)(\b\w+\b)"
        while True:
            match = re.search(pattern, expr)
            if not match:
                break
            a, op1, b, op2, c = match.groups()
            expr = expr.replace(f"{a}{op1}{b}{op2}{c}", f"({a}{op1}{b})&({b}{op2}{c})")
        return expr

    safe_expr = rewrite_chained_comparisons(condition)

    try:
        result = eval(safe_expr, {}, local_vars)
    except Exception as e:
        raise ValueError(f"Error evaluating condition: {e}")

    if not isinstance(result, np.ndarray) or result.dtype != bool:
        raise ValueError("The condition did not produce a valid boolean mask.")

    return np.where(result)[0]


####################################################################################################
def get_values_by_condition(condition: str, **kwargs):
    """
    Evaluate a logical condition involving an array and optional scalar variables,
    and return the values where the condition holds true.

    Parameters
    ----------
    condition : str
        A condition string to evaluate, e.g.:
            - "bvals > 1000"
            - "bmin <= bvals <= bmax"
            - "bvals != bval"
        Supports chained comparisons and scalar literals directly in the expression.

    **kwargs : dict
        Variable bindings for any names used in the condition string. Must include exactly
        one array (list or np.ndarray) that represents the main vector to filter.

    Returns
    -------
    np.ndarray
        Values where the condition evaluates to True.

    Raises
    ------
    ValueError
        If:
            - The condition references variables not in kwargs (excluding literals)
            - No array variable is found
            - More than one array-like variable is provided
            - The condition does not yield a boolean array

    Examples
    --------
    >>> bvals = np.array([0, 500, 1000, 2000, 3000])
    >>> get_values_by_condition("bvals > 1000", bvals=bvals)
    array([2000, 3000])

    >>> get_values_by_condition("bmin <= bvals <= bmax", bvals=bvals, bmin=800, bmax=2500)
    array([1000, 2000])
    """

    condition = condition.replace(" ", "")
    # Reuse the logic from get_indices_by_condition but return values instead of indices
    indices = get_indices_by_condition(condition, **kwargs)

    # Extract the array variable from kwargs
    array_var = next(k for k, v in kwargs.items() if isinstance(v, (list, np.ndarray)))

    tmp = np.array(remove_duplicates(kwargs[array_var][indices]))

    return tmp.tolist()


####################################################################################################
def build_indices_with_conditions(
    inputs: List[Union[int, tuple, list, str, np.ndarray]],
    nonzeros: bool = True,
    **kwargs,
) -> List[int]:
    """
    Combine numeric, range, and condition-based inputs into a unified list of indices.
    Parameters
    ----------
    inputs : list
        Mixed list containing integers, lists, arrays, or strings with comma-separated numeric ranges or conditions.

    nonzeros : bool
        If True, removes zeros from the output.

    **kwargs : dict
        Variables used for evaluating conditions (must include exactly one array-like for conditions).

    Returns
    -------
    List[int]
        Sorted, unique list of resulting indices.

    Raises
    ------
    ValueError
        If any item cannot be interpreted correctly.
        If the condition references variables not in kwargs (excluding literals).
        If no array variable is found.
        If more than one array-like variable is provided.
        If the condition does not yield a boolean array.
        If the condition is invalid.

        Usage:
        -------
        # Test 2: Pure range strings
        >>> input2 = ["1:4", "5-7", "8:2:10"]
        >>> print(f"Input: {input2}")
        >>> result = build_indices_with_conditions(input2, nonzeros=False)
        >>> print(f"Result: {result}")
        >>> print("Expected: [1,2,3,4,5,6,7,8,10]")

        # Test 3: Mixed numeric and range strings
        >>> input3 = [0, 9, "2:4", "6-8"]
        >>> print(f"Input: {input3}")
        >>> result = build_indices_with_conditions(input3, nonzeros=False)
        >>> print(f"Result: {result}")
        >>> print("Expected: [0,2,3,4,6,7,8,9]")

        # Test 4: Value-based conditions (returns INDICES where condition is true)
        >>> input4 = ["5<=data<=20"]
        >>> print(f"Input: {input4}")
        >>> result = build_indices_with_conditions(input4, data=data)
        >>> print(f"Result: {result}")
        >>> print("Expected: [1,2,3,4] (indices where data is between 5 and 20)")

        # Test 5: Mixed indices and conditions
        >>> input5 = [0, "2:4", "data == 0", 9]
        >>> print(f"Input: {input5}")
        >>> result = build_indices_with_conditions(input5, data=data)
        >>> print(f"Result: {result}")
        >>> print("Expected: [2,3,4,9] (indices including where data==0)")

        # Test 6: Non-zero filtering
        >>> input6 = [0, "0:3", "data != 0", 9]
        >>> print(f"Input: {input6}")
        >>> result = build_indices_with_conditions(input6, data=data, nonzeros=True)
        >>> print(f"Result: {result}")

        # Test 7: Complex mixed case
        >>> input7 = [0, "data > threshold", "1:3, 5-7", np.array([8,9])]
        >>> print(f"Input: {input7}")
        >>> result = build_indices_with_conditions(input7, data=data, threshold=threshold)
        >>> print(f"Result: {result}")
        >>> print("Expected: [0,1,2,3,5,6,7,8,9] (all valid indices)")

    """

    all_values = []

    for item in inputs:
        if isinstance(item, str):
            parts = [p.strip() for p in item.split(",") if p.strip()]
            for part in parts:
                if any(op in part for op in ["<", ">", "=", "!"]):
                    try:
                        condition_indices = get_indices_by_condition(part, **kwargs)
                        all_values += condition_indices.tolist()
                    except Exception as e:
                        raise ValueError(f"Invalid condition '{part}': {e}")
                else:
                    try:
                        range_values = build_indices([part], nonzeros=nonzeros)
                        all_values += range_values
                    except Exception as e:
                        raise ValueError(f"Invalid range expression '{part}': {e}")
        else:
            try:
                range_values = build_indices([item], nonzeros=nonzeros)
                all_values += range_values
            except Exception as e:
                raise ValueError(f"Invalid input item '{item}': {e}")

    final_result = sorted(set(all_values))
    if nonzeros:
        final_result = [v for v in final_result if v != 0]

    return final_result


####################################################################################################
def build_values_with_conditions(
    inputs: List[Union[int, tuple, list, str, np.ndarray]],
    nonzeros: bool = True,
    **kwargs,
) -> List[int]:
    """
    Combine numeric, range, and condition-based inputs into a unified list of values.

    Parameters
    ----------
    inputs : list
        Mixed list containing integers, lists, arrays, or strings with comma-separated numeric ranges or conditions.

    nonzeros : bool
        If True, removes zeros from the output.

    **kwargs : dict
        Variables used for evaluating conditions (must include exactly one array-like for conditions).

    Returns
    -------
    List[int]
        Sorted, unique list of resulting values.
    """
    all_values = []

    for item in inputs:
        if isinstance(item, str):
            # Split comma-separated sections in the string
            parts = [p.strip() for p in item.split(",")]
            for part in parts:
                if any(op in part for op in ["<", ">", "=", "!"]):
                    try:
                        condition_values = get_values_by_condition(part, **kwargs)
                        all_values += condition_values
                    except Exception as e:
                        raise ValueError(f"Invalid condition '{part}': {e}")
                else:
                    try:
                        range_values = build_indices([part], nonzeros=nonzeros)
                        all_values += range_values
                    except Exception as e:
                        raise ValueError(f"Invalid range expression '{part}': {e}")
        else:
            # Delegate everything else to build_indices
            try:
                range_values = build_indices([item], nonzeros=nonzeros)
                all_values += range_values
            except Exception as e:
                raise ValueError(f"Invalid input item '{item}': {e}")

    final_result = sorted(set(all_values))
    if nonzeros:
        final_result = [v for v in final_result if v != 0]

    return final_result


####################################################################################################
def remove_duplicates(input_list: list):
    """
    Function to remove duplicates from a list while preserving the order

    Parameters
    ----------
    input_list : list
        List of elements

    Returns
    -------
    unique_list: list
        List of unique elements

    How to Use:
    --------------
        >>> input_list = [1, 2, 2, 3, 4, 4, 5]
        >>> unique_list = remove_duplicates(input_list)
        >>> print(unique_list)  # Output: [1, 2, 3, 4, 5]

    """

    unique_list = []
    seen_elements = set()

    for element in input_list:
        if element not in seen_elements:
            unique_list.append(element)
            seen_elements.add(element)

    return unique_list


####################################################################################################
def select_ids_from_file(subj_ids: list, ids_file: Union[list, str]) -> list:
    """
    Function to select the ids from a list of ids that are in a file.
    It can be used to select the ids from a list of subjects that are in a file.

    Parameters
    ----------
    subj_ids : list
        List of subject ids.
    ids_file : str or list
        File with the ids to select.

    Returns
    -------
    out_ids: list
        List of ids that are in the file.

    How to Use:
    --------------
        >>> subj_ids = ["sub-01", "sub-02", "sub-03"]
        >>> ids_file = "ids.txt" # Column-wise text file with the ids to select (i.e. "sub-01", "sub-03")
        >>> out_ids = select_ids_from_file(subj_ids, ids_file)
        >>> print(out_ids)  # Output: ["sub-01", "sub-03"]
    """

    # Read the ids from the file
    out_ids = []  # Initialize out_ids to avoid potential use before assignment

    if isinstance(ids_file, str):
        if os.path.exists(ids_file):
            with open(ids_file) as file:
                t1s2run = [line.rstrip() for line in file]

            out_ids = [s for s in subj_ids if any(xs in s for xs in t1s2run)]

    elif isinstance(ids_file, list):
        out_ids = list_intercept(subj_ids, ids_file)

    return out_ids


####################################################################################################
def filter_by_substring(
    input_list: list,
    or_filter: Union[str, list],
    and_filter: Union[str, list] = None,
    bool_case: bool = False,
) -> list:
    """
    Function to filter a list of elements by a substrings.

    Parameters
    ----------
    input_list : list
        List of elements

    or_filter : str or list
        Substring to filter. It can be a string or a list of strings.
        It functions as an OR filter, meaning that if any of the substrings are found in the element,
        it will be included in the filtered list.

    and_filter : str or list, optional
        Substring to filter. It can be a string or a list of strings.
        It functions as an AND filter, meaning that all of the substrings must be found in the element

    bool_case : bool
        Boolean to indicate if the search is case sensitive. Default is False

    Returns
    -------
    filtered_list: list
        List of elements that contain the substring

    How to Use:
    --------------
        >>> input_list = ["apple", "banana", "cherry", "date"]
        >>> or_filter = ["app", "ch"]
        >>> filtered_list = filter_by_substring(input_list, or_filter)
        >>> print(filtered_list)  # Output: ['apple', 'cherry']

    """

    if isinstance(input_list, str):
        input_list = [input_list]

    # Rise an error if input_list is not a list
    if not isinstance(input_list, list):
        raise ValueError("The input input_list must be a list.")

    # Convert the or_filter to a list
    if isinstance(or_filter, str):
        or_filter = [or_filter]

    # Convert the or_filter and input_list to lower case
    if not bool_case:
        tmp_substr = [e.lower() for e in or_filter]
        tmp_input_list = [e.lower() for e in input_list]

    else:
        tmp_substr = or_filter
        tmp_input_list = input_list

    # Get the indexes of the list elements that contain any of the strings in the list aa
    indexes = [
        i for i, x in enumerate(tmp_input_list) if any(a in x for a in tmp_substr)
    ]

    # Convert indexes to a numpy array
    indexes = np.array(indexes)

    # Select the atlas_files with the indexes
    filtered_list = [input_list[i] for i in indexes]

    # Remove the duplicates from the filtered list
    filtered_list = remove_duplicates(filtered_list)

    if and_filter is not None:
        # Convert the and_filter to a list
        if isinstance(and_filter, str):
            and_filter = [and_filter]

        # Convert the and_filter to lower case
        if not bool_case:
            tmp_and_filter = [e.lower() for e in and_filter]
            tmp_filtered_list = [e.lower() for e in filtered_list]
        else:
            tmp_and_filter = and_filter
            tmp_filtered_list = filtered_list

        # Get the indexes of the list elements that contain all of the strings in the list tmp_and_filter
        indexes = [
            i
            for i, x in enumerate(tmp_filtered_list)
            if all(a in x for a in tmp_and_filter)
        ]

        # Convert indexes to a numpy array
        indexes = np.array(indexes)

        # Select the filtered_list with the indexes
        filtered_list = [filtered_list[i] for i in indexes]

    return filtered_list


####################################################################################################
def get_indexes_by_substring(
    input_list: list,
    substr: Union[str, list],
    invert: bool = False,
    bool_case: bool = False,
    match_entire_word: bool = False,
):
    """
    Function extracts the indexes of the elements of a list of elements that contain
    any of the substrings of another list.

    Parameters
    ----------
    input_list : list
        List of elements

    substr : str or list
        Substring to filter. It can be a string or a list of strings

    invert : bool
        Boolean to indicate if the indexes are inverted. Default is False
        If True, the indexes of the elements that do not contain any of the substrings are returned.

    bool_case : bool
        Boolean to indicate if the search is case sensitive. Default is False

    match_entire_word : bool
        Boolean to indicate if the search is a whole word match. Default is False

    Returns
    -------
    indexes: list
        List of indexes that contain any of the substring

    How to Use:
    --------------
        >>> input_list = ["apple", "banana", "cherry", "date"]
        >>> substr = ["ap", "ch"]
        >>> indexes = get_indexes_by_substring(input_list, substr)
        >>> print(indexes)  # Output: [0, 2]

        >>> input_list = ["apple", "banana", "cherry", "date"]
        >>> substr = ["apple", "banana"]
        >>> indexes = get_indexes_by_substring(input_list, substr, invert=True)
        >>> print(indexes)  # Output: [2, 3]

        >>> input_list = ["apple", "banana", "cherry", "date"]
        >>> substr = ["apple", "cherry"]
        >>> indexes = get_indexes_by_substring(input_list, substr, match_entire_word=True)
        >>> print(indexes) # Output: [0, 2]
    """

    # Rise an error if input_list is not a list
    if not isinstance(input_list, list):
        raise ValueError("The input input_list must be a list.")

    # Convert the substr to a list
    if isinstance(substr, str):
        substr = [substr]

    # Convert the substr and input_list to lower case
    if not bool_case:
        tmp_substr = [e.lower() for e in substr]
        tmp_input_list = [e.lower() for e in input_list]

    else:
        tmp_substr = substr
        tmp_input_list = input_list

    # Get the indexes of the list elements that contain any of the strings in the list aa
    if match_entire_word:
        indexes = [
            i for i, x in enumerate(tmp_input_list) if any(a == x for a in tmp_substr)
        ]
    else:
        indexes = [
            i for i, x in enumerate(tmp_input_list) if any(a in x for a in tmp_substr)
        ]

    # Convert indexes to a numpy array
    indexes = np.array(indexes)

    if invert:
        indexes = np.setdiff1d(np.arange(0, len(input_list)), indexes)

    return indexes


####################################################################################################
def remove_substrings(
    list1: Union[str, List[str]], list2: Union[str, List[str]]
) -> List[str]:
    """
    Remove substrings from each element of list1 that match any string in list2.

    Parameters
    ----------
    list1 : Union[str, List[str]]
        A string or a list of strings to process.

    list2 : Union[str, List[str]]
        A string or a list of strings to be removed from each element in list1.
        If a single string is provided, it will be converted to a list internally.

    Returns
    -------
    List[str]
        A new list with the substrings removed from each element of list1.

    Raises
    ------
    TypeError
        If list1 is not a list of strings or list2 is not a string or list of strings.

    Examples
    --------
    >>> remove_substrings(["hello_world", "test_world", "worldwide"], "world")
    ['hello_', 'test_', 'wide']

    >>> remove_substrings(["apple_pie", "banana_pie", "cherry_pie"], ["pie", "_"])
    ['apple', 'banana', 'cherry']
    """

    if isinstance(list1, str):
        list1 = [list1]

    elif not isinstance(list1, list) or not all(isinstance(s, str) for s in list1):
        raise TypeError("list1 must be a list of strings.")

    if isinstance(list2, str):
        list2 = [list2]

    elif not isinstance(list2, list) or not all(isinstance(s, str) for s in list2):
        raise TypeError("list2 must be a string or a list of strings.")

    result = []
    for item in list1:
        for sub in list2:
            item = item.replace(sub, "")
        result.append(item)

    return result


####################################################################################################
def replace_substrings(
    strings: Union[str, List[str]],
    substrings: Union[str, List[str]],
    replaced_by: Union[str, List[str]],
    bool_case: bool = True,
) -> List[str]:
    """
    Replace substrings or regex patterns in each element of a list of strings.

    Parameters
    ----------
    strings : Union[str, List[str]]
        A string or a list of strings to modify.
    substrings : Union[str, List[str]]
        A string or list of substrings or regular expression patterns to search for.
    replaced_by : Union[str, List[str]]
        A string or list of replacement strings corresponding to each substring pattern.
    bool_case : bool, optional
        If False, the matching will be case-insensitive. Default is True (case-sensitive).

    Returns
    -------
    List[str]
        A new list of strings with the specified patterns replaced.

    Raises
    ------
    TypeError
        If inputs are not strings or lists of strings.
    ValueError
        If `substrings` and `replaced_by` have different lengths.

    Examples
    --------
    >>> replace_substrings_regex("Hello_World", "World", "Earth", bool_case=False)
    ['Hello_Earth']

    >>> replace_substrings_regex(["abc123", "ABC123"], ["abc", "123"], ["xyz", "789"], bool_case=False)
    ['xyz789', 'xyz789']
    """
    # Normalize inputs to lists
    if isinstance(strings, str):
        strings = [strings]
    if isinstance(substrings, str):
        substrings = [substrings]
    if isinstance(replaced_by, str):
        replaced_by = [replaced_by]

    # Validate inputs
    if not (
        isinstance(strings, list)
        and all(isinstance(s, str) for s in strings)
        and isinstance(substrings, list)
        and all(isinstance(s, str) for s in substrings)
        and isinstance(replaced_by, list)
        and all(isinstance(s, str) for s in replaced_by)
    ):
        raise TypeError("All inputs must be strings or lists of strings.")

    if len(substrings) != len(replaced_by):
        raise ValueError("`substrings` and `replaced_by` must have the same length.")

    flags = 0 if bool_case else re.IGNORECASE
    compiled_patterns = [re.compile(pat, flags) for pat in substrings]

    result = []
    for s in strings:
        for pattern, replacement in zip(compiled_patterns, replaced_by):
            s = pattern.sub(replacement, s)
        result.append(s)

    return result


####################################################################################################
def list_intercept(list1: list, list2: list):
    """
    Function to intercept the elements from 2 different lists.

    Parameters
    ----------
    list1 : list
        List of elements
    list2 : list
        List of elements

    Returns
    -------
    int_list: list
        List of elements that are in both lists

    How to Use:
    --------------
        >>> list1 = [1, 2, 3, 4, 5]
        >>> list2 = [3, 4, 5, 6, 7]
        >>> int_list = list_intercept(list1, list2)
        >>> print(int_list)  # Output: [3, 4, 5]

    """

    # Rise an error if list1 or list2 are not lists
    if not isinstance(list1, list):
        raise ValueError("The input list1 must be a list.")

    if not isinstance(list2, list):
        raise ValueError("The input list2 must be a list.")

    # Create a list of elements that are in both lists
    int_list = [value for value in list1 if value in list2]

    return int_list


####################################################################################################
def ismember_from_list(a, b):
    """
    Function to check if elements of a are in b

    Parameters
    ----------
    a : list
        List of elements to check
    b : list
        List of elements to check against

    Returns
    -------
    values: list
        List of unique elements in a
    idx: list
        List of indices of elements in a that are in b

    How to Use:
    --------------
        >>> a = [1, 2, 3, 4, 5]
        >>> b = [3, 4, 5, 6, 7]
        >>> values, idx = ismember_from_list(a, b)
        >>> print(values)  # Output: [3, 4, 5]
        >>> print(idx)     # Output: [0, 1, 2]
    """

    values, indices = np.unique(a, return_inverse=True)
    is_in_list = np.isin(a, b)
    idx = indices[is_in_list].astype(int)

    return values, idx


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############   Section 6: Methods dedicated to find directories, remove empty folders   ############
############     find all the files inside a certain directory, etc                     ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def get_leaf_directories(root_dir: str) -> list:
    """
    Finds all folders inside the given directory that do not contain any subfolders.

    Parameters:.
    ----------
    root_dir :str
        The path to the root directory where the search will be performed.

    Returns:
    -------
    leaf_folders: list
        A list of absolute paths to folders that do not contain any subfolders.

    How to Use:
    --------------
        >>> root_directory = "/path/to/your/folder"
        >>> leaf_folders = get_leaf_directories(root_directory)
        >>> print("Leaf folders:", leaf_folders)
    """

    if not os.path.isdir(root_dir):
        raise ValueError(f"Invalid directory: {root_dir}")

    leaf_folders = []
    for foldername, subfolders, _ in os.walk(root_dir):
        if not subfolders:  # If the folder has no subfolders, it's a leaf folder
            leaf_folders.append(foldername)

    return leaf_folders


####################################################################################################
def remove_trailing_separators(path: str) -> str:
    """
    Remove all trailing path separators (unless at root).

    Parameters
    ----------
    path : str
        The path from which to remove trailing separators.

    Returns
    -------
    str
        The path with trailing separators removed.

    Usage example:
    >>> path = "/path/to/directory///"
    >>> print(remove_trailing_separators(path))
    "/path/to/directory/"

    """
    stripped = path.rstrip(os.sep)
    return stripped if stripped else os.sep


####################################################################################################
def get_all_files(
    in_dir: Union[str, Path],
    recursive: bool = True,
    or_filter: Union[str, List[str]] = None,
    and_filter: Union[str, List[str]] = None,
    bool_case: bool = False,
) -> list:
    """
    Function to detect all the files in a directory and its subdirectories

    Parameters
    ----------
    in_dir : str
        Input directory

    recursive : bool
        If True, the function will search recursively in all subdirectories.
        If False, it will only search in the specified directory.

    Raises
    ------
    ValueError
        If the input directory does not exist or is not a directory.
    ValueError
        If the input directory is empty.
    ValueError
        If the input directory is not a string.
    ValueError
        If the input directory is not a valid path.

    Returns
    -------
    files: list
        List of files in the directory and its subdirectories

    Examples:
    ----------------
        >>> in_dir = "/path/to/directory"
        >>> files = get_all_files(in_dir)
        >>> print(files)  # Output: List of files in the directory and its subdirectories
    """

    if isinstance(in_dir, str):
        try:
            in_dir = Path(in_dir)
        except Exception as e:
            raise ValueError(f"Invalid input directory path: {in_dir}. Error: {e}")
    elif not isinstance(in_dir, Path):
        raise ValueError("The input in_dir must be a string or a Path object.")

    # If the input directory is a file, raise an error
    if in_dir.is_file():
        raise ValueError(f"The input path is a file, not a directory: {in_dir}")

    # If the input directory is a symlink, raise an error
    if in_dir.is_symlink():
        raise ValueError(f"The input path is a symlink, not a directory: {in_dir}")

    if not isinstance(in_dir, Path):
        raise ValueError("The input in_dir must be a string or a Path object.")

    if not in_dir.exists():
        raise ValueError(f"The input directory does not exist: {in_dir}")
    if not in_dir.is_dir():
        raise ValueError(f"The input path is not a directory: {in_dir}")

    if not in_dir.is_absolute():
        raise ValueError(f"The input path is not an absolute path: {in_dir}")

    if not os.listdir(in_dir):
        raise ValueError(f"The input directory is empty: {in_dir}")

    if or_filter is not None:
        if isinstance(or_filter, str):
            or_filter = [or_filter]

        if not isinstance(or_filter, list):
            raise ValueError("The or_filter must be a string or a list of strings.")

    # Initialize an empty list to store the file paths
    if not recursive:
        all_files = [
            os.path.join(in_dir, f)
            for f in os.listdir(in_dir)
            if os.path.isfile(os.path.join(in_dir, f))
        ]
    else:

        all_files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(in_dir):
            for file in f:
                all_files.append(os.path.join(r, file))

    if or_filter is not None:
        all_files = filter_by_substring(
            all_files, or_filter=or_filter, and_filter=None, bool_case=bool_case
        )

    if and_filter is not None:
        all_files = filter_by_substring(
            all_files,
            or_filter=and_filter[0],
            and_filter=and_filter,
            bool_case=bool_case,
        )

    # Check if the list of files is empty
    if not all_files:
        raise ValueError(f"No files found in the directory: {in_dir}")

    return all_files


####################################################################################################
def rename_folders(
    folder_paths: List[str], replacements: Dict[str, str], bool_case: bool = True
) -> List[Tuple[str, str]]:
    """
    Rename folders by replacing specified strings in their paths.

    This function identifies all unique directory paths that need renaming (including parent
    directories) and renames them from deepest to shallowest to avoid conflicts.

    Parameters
    ----------
    folder_paths : List[str]
        List of folder paths to be renamed.
    replacements : Dict[str, str]
        Dictionary where keys are old strings to replace and values are new strings.
        Replacements are applied in order of string length (longest first) to avoid
        partial matches.
    bool_case : bool, optional
        If True (default), performs case-sensitive replacement.
        If False, performs case-insensitive replacement.

    Returns
    -------
    List[Tuple[str, str]]
        List of tuples containing (old_path, new_path) for successfully renamed folders.

    Raises
    ------
    OSError
        If there are permission issues or filesystem errors during renaming.
    FileNotFoundError
        If a folder path doesn't exist when attempting to rename.
    FileExistsError
        If the target path already exists.

    Example
    -------
    >>> folder_paths = [
    ...     '/data/sub-CHUVL488/ses-V1/dwi',
    ...     '/data/sub-CHUVL488/ses-V1/func'
    ... ]
    >>> replacements = {"CHUVL488": "L488", "V1": "S1"}
    >>> renamed = rename_folders(folder_paths, replacements, bool_case=True)
    >>> print(f"Renamed {len(renamed)} folders")
    """

    def apply_replacements(path: str) -> str:
        """Apply all replacements to a path string."""
        new_path = path
        # Sort replacements by length (longer strings first) to avoid partial matches
        sorted_replacements = sorted(
            replacements.items(), key=lambda x: len(x[0]), reverse=True
        )

        for old_str, new_str in sorted_replacements:
            if bool_case:
                new_path = new_path.replace(old_str, new_str)
            else:
                pattern = re.escape(old_str)
                new_path = re.sub(pattern, new_str, new_path, flags=re.IGNORECASE)
        return new_path

    # Collect all unique directory paths that need to be considered for renaming
    all_directories: Set[str] = set()

    for folder_path in folder_paths:
        # Add the folder itself and all its parent directories
        current_path = folder_path
        while current_path and current_path != "/":
            all_directories.add(current_path)
            current_path = os.path.dirname(current_path)

    # Build rename operations for directories that actually need renaming
    rename_operations = []

    for old_path in all_directories:
        new_path = apply_replacements(old_path)

        # Only add to operations if path actually changed
        if old_path != new_path:
            rename_operations.append((old_path, new_path))

    # Sort by path depth (deepest first) to avoid parent-child conflicts
    rename_operations.sort(key=lambda x: x[0].count("/"), reverse=True)

    # Execute rename operations
    successfully_renamed = []

    for old_path, new_path in rename_operations:
        try:
            if os.path.exists(old_path):
                # Create parent directories of new path if they don't exist
                parent_dir = os.path.dirname(new_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)

                # Check if target already exists
                if os.path.exists(new_path):
                    print(f"Warning: Target already exists, skipping: {new_path}")
                    continue

                # Perform the rename
                os.rename(old_path, new_path)
                successfully_renamed.append((old_path, new_path))
                print(f"Renamed: {old_path} -> {new_path}")
            else:
                print(f"Warning: Path does not exist: {old_path}")

        except OSError as e:
            print(f"Error renaming {old_path}: {e}")
            continue

    return successfully_renamed


####################################################################################################
def remove_empty_folders(start_path, deleted_folders=None):
    """
    Recursively removes empty directories starting from start_path.
    Returns a list of all directories that were deleted.

    Parameters:
    ----------
        start_path : str
            The directory path to start searching from

        deleted_folders : list
            A list to store the paths of deleted directories. If None, a new list will be created.

    Returns:
    -------
        deleted_folders : list
            A list of all directories that were deleted.

    How to Use:
    --------------
        >>> deleted_folders = remove_empty_folders("/path/to/start")
        >>> print("Deleted folders:", deleted_folders)
    --------------
    """
    if deleted_folders is None:
        deleted_folders = []

    # Walk through the directory tree bottom-up (deepest first)
    for root, dirs, files in os.walk(start_path, topdown=False):
        # Check each directory in current level
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                # Try to remove the directory (will only succeed if empty)
                os.rmdir(dir_path)
                deleted_folders.append(dir_path)
                # print(f"Removed empty directory: {dir_path}")  # Optional logging
            except OSError:
                # Directory not empty or other error - we'll ignore it
                pass

    # Finally, try to remove the starting directory itself if it's now empty
    try:
        os.rmdir(start_path)
        deleted_folders.append(start_path)
        # print(f"Removed empty directory: {start_path}")  # Optional logging
    except OSError:
        pass

    return deleted_folders


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############        Section 7: Methods dedicated to strings and characters              ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def rem_duplicate_char(strcad: str, dchar: str):
    """
    This function removes duplicate characters from strings.

    Parameters
    ----------
    strcad : str
        Input string
    dchar : str

    Returns
    ---------
    str or list
        String with the duplicate characters removed.

    """

    chars = []
    prev = None

    for c in strcad:
        if c != dchar:
            chars.append(c)
            prev = c
        else:
            if prev != c:
                chars.append(c)
                prev = c

    return "".join(chars)


####################################################################################################
def create_names_from_indices(
    indices: Union[int, List[int], np.ndarray],
    prefix: str = "auto-roi",
    sufix: str = None,
) -> list[str]:
    """
    Generates a list of region names with the format "auto-roi-000001"
    based on a list of indices, using list comprehension.

    Parameters
    ----------
    indices : int or list
        A single integer or a list of integers representing the indices.

    prefix : str
        A prefix to add to the region names. Default is "auto-roi"

    sufix : str
        A sufix to add to the region names. Default is None

    Returns
    -------
    list[str]
        A list of formatted region names.

    Examples
    ---------
    >>> indices = [1, 2, 3]
    >>> names = create_names_from_indices(indices)
    >>> print(names)  # Output: ['auto-roi-000001', 'auto-roi-000002', 'auto-roi-000003']

    >>> indices = 5
    >>> names = create_names_from_indices(indices)
    >>> print(names)  # Output: ['auto-roi-000005']

    >>> indices = np.array([10, 20, 30])
    >>> names = create_names_from_indices(indices, sufix="lh")
    >>> print(names)  # Output: ['auto-roi-000010-lh', 'auto-roi-000020-lh', 'auto-roi-000030-lh']

    >>> indices = [1, 2, 3]
    >>> names = create_names_from_indices(indices, prefix="ctx")
    >>> print(names)  # Output: ['ctx-000001', 'ctx-000002', 'ctx-000003']

    """

    # Check if indices is a single integer or a list of integers
    if isinstance(indices, int):
        indices = [indices]
    elif isinstance(indices, np.ndarray):
        indices = indices.tolist()
    elif not isinstance(indices, list):
        raise ValueError("Indices must be an integer, list, or numpy array.")
    elif not all(isinstance(i, int) for i in indices):
        raise ValueError("All elements in indices must be integers.")

    if sufix is not None:
        names = [f"{prefix}-{index:06d}-{sufix}" for index in indices]
    else:
        # Generate names with the specified prefix and formatted index
        names = [f"{prefix}-{index:06d}" for index in indices]

    return names


####################################################################################################
def correct_names(
    regnames: list,
    prefix: str = None,
    sufix: str = None,
    lower: bool = False,
    remove: list = None,
    replace: list = None,
):
    """
    Correcting region names. It can be used to add a prefix or sufix to the region names, lower the region names, remove or replace substrings in the region names.

    Parameters
    ----------
    regnames : list
        List of region names
    prefix : str
        Prefix to add to the region names. Default is None
    sufix : str
        Sufix to add to the region names. Default is None
    lower : bool
        Boolean to indicate if the region names should be lower case. Default is False
    remove : list
        List of substrings to remove from the region names. Default is None
    replace : list
        List of substrings to replace in the region names. Default is None.
        It can be a list of tuples or a list of lists. The first element is the substring to replace and the second element is the substring to replace with.
        For example: replace = [["old", "new"], ["old2", "new2"]]

    Returns
    -------
    regnames: list
        List of corrected region names

    How to Use:
    --------------
        >>> regnames = ["ctx-lh-1", "ctx-rh-2", "ctx-lh-3"]
        >>> prefix = "ctx-"
        >>> sufix = "-lh"
        >>> lower = True
        >>> remove = ["ctx-"]
        >>> replace = [["lh", "left"], ["rh", "right"]]
        >>> corrected_names = correct_names(regnames, prefix, sufix, lower, remove, replace)
        >>> print(corrected_names)  # Output: ['left-1-lh', 'right-2-lh', 'left-3-lh']

    """

    # Add prefix to the region names
    if prefix is not None:
        # If temp_name do not starts with ctx- then add it
        regnames = [
            name if name.startswith(prefix) else prefix + "{}".format(name)
            for name in regnames
        ]

    # Add sufix to the region names
    if sufix is not None:
        # If temp_name do not ends with - then add it
        regnames = [
            name if name.endswith(sufix) else "{}".format(name) + sufix
            for name in regnames
        ]

    # Lower the region names
    if lower:
        regnames = [name.lower() for name in regnames]

    # Remove the substring item from the region names
    if remove is not None:

        for item in remove:

            # Remove the substring item from the region names
            regnames = [name.replace(item, "") for name in regnames]

    # Replace the substring item from the region names
    if replace is not None:

        if isinstance(replace, list):
            if all(isinstance(item, list) for item in replace):
                for item in replace:
                    # Replace the substring item from the region names
                    regnames = [name.replace(item[0], item[1]) for name in regnames]
            else:
                regnames = [name.replace(replace[0], replace[1]) for name in regnames]

    return regnames


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############    Section 8: Methods dedicated to work with dictionaries and dataframes   ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def remove_empty_keys_or_values(d: dict) -> dict:
    """
    Remove dictionary entries with empty keys, keys with only spaces, or empty values.

    Parameters:
    ----------

    d : dict
        The dictionary to remove entries from.

    Returns:
    --------

    d : dict
        The dictionary with the empty entries removed.

    How to Use:
    --------------
        >>> my_dict = {'key1': 'value1', 'key2': '', '': 'value3', 'key4': None}
        >>> cleaned_dict = remove_empty_keys_or_values(my_dict)
        >>> print(cleaned_dict)  # Output: {'key1': 'value1', 'key4': None}
    """
    keys_to_remove = [
        key
        for key in d
        if not key
        or (isinstance(key, str) and key.strip() == "")
        or not d[key]
        or (isinstance(d[key], str) and d[key].strip() == "")
    ]

    for key in keys_to_remove:
        del d[key]

    return d


####################################################################################################
def save_dictionary_to_json(data_dictionary: dict, json_file_path: str):
    """
    Saves a Python dictionary to a JSON file.

    Parameters
    ----------
    data_dictionary : dict
        The dictionary to be saved.

    file_path : str
        The path to the JSON file where the dictionary will be saved.

    Returns
    -------
    None
        This function does not return anything. It only saves the dictionary to a JSON file.

    Example
    -------
    >>> data = {'key': 'value'}
    >>> save_dictionary_to_json(data, 'data.json')
    ----------
        data_dictionary (dict): The dictionary to be saved.
        file_path (str): The path to the JSON file where the dictionary will be saved.
    """

    # Check if the file path is valid
    if not isinstance(json_file_path, str):
        raise ValueError("The file path must be a string.")
    if not json_file_path.endswith(".json"):
        raise ValueError("The file path must end with '.json'.")
    # Check if the dictionary is valid
    if not isinstance(data_dictionary, dict):
        raise ValueError("The data must be a dictionary.")

    try:
        with open(json_file_path, "w") as json_file:
            json.dump(data_dictionary, json_file, indent=4)
        print(f"Dictionary successfully saved to: {json_file_path}")
    except Exception as e:
        print(f"An error occurred while saving the dictionary: {e}")


####################################################################################################
def read_file_with_separator_detection(
    file_path, sample_size=10, possible_seps=None, **kwargs
):
    """
    Reads a delimited text file with automatic separator detection.

    Parameters
    ----------
    file_path : str or Path
        Path to the delimited input file (e.g., CSV or TSV).
    sample_size : int, optional
        Number of lines to sample for separator detection (default is 10).
    possible_seps : list of str, optional
        List of possible separators to try (default is [',', '\\t']).
    **kwargs :
        Additional keyword arguments passed to `pandas.read_csv`.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the parsed file data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty or no valid separator could be detected.

    Examples
    --------
    >>> df = read_file_with_separator_detection("data.txt")
    >>> df.head()
    """
    if possible_seps is None:
        possible_seps = [",", "\t"]

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist")

    # Read a sample of lines to detect the delimiter
    sample_lines = []
    with open(file_path, "r", encoding="utf-8") as file:
        for _ in range(sample_size):
            try:
                sample_lines.append(next(file))
            except StopIteration:
                break

    if not sample_lines:
        raise ValueError("File is empty")

    # Count the number of times each possible separator appears in the sample
    sep_counts = {
        sep: sum(line.count(sep) for line in sample_lines) for sep in possible_seps
    }
    valid_seps = {sep: count for sep, count in sep_counts.items() if count > 0}

    # Fallback to space separator if nothing valid was found
    if not valid_seps:
        space_count = sum(line.count(" ") for line in sample_lines)
        if space_count == 0:
            raise ValueError("Could not detect a valid separator in the file")
        detected_sep = " "
    else:
        detected_sep = max(valid_seps.items(), key=lambda x: x[1])[0]

    # Attempt to read the file using the detected separator
    try:
        df = pd.read_csv(file_path, sep=detected_sep, engine="python", **kwargs)
    except Exception:
        try:
            df = pd.read_csv(file_path, sep=detected_sep, **kwargs)
        except Exception as e:
            raise ValueError(
                f"Failed to read file with detected separator '{detected_sep}': {str(e)}"
            )

    return df


####################################################################################################
def smart_read_table(
    file_path: Union[str, Path], sample_size: int = 10, possible_seps=None, **kwargs
):
    """
    Reads a delimited file using pandas auto-detection or fallback separator detection.

    Parameters
    ----------
    file_path : str or Path
        Path to the delimited input file.
    sample_size : int, optional
        Number of lines to sample for fallback detection (default is 10).
    possible_seps : list of str, optional
        List of separators to try if auto-detection fails (default is [',', '\\t', ';', '|']).
    **kwargs :
        Additional keyword arguments passed to `pandas.read_csv`.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the parsed file data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty or no valid separator could be detected.

    Examples
    --------
    >>> df = read_file_with_fallback_detection("data.txt")
    >>> df.shape
    """
    if possible_seps is None:
        possible_seps = [",", "\t", ";", "|"]

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist")

    # Try pandas built-in separator auto-detection
    try:
        df = pd.read_csv(file_path, sep=None, engine="python", **kwargs)
        return df
    except Exception:
        # If auto-detection fails, fall back to manual separator detection
        return read_file_with_separator_detection(
            file_path, sample_size, possible_seps, **kwargs
        )


####################################################################################################
def extract_string_values(data_dict: Union[str, dict], only_last_key=True) -> dict:
    """
    Recursively extracts all keys with string values from a nested dictionary. It will avoid keys
    that are lists or other types. The keys can be either the leaf key name or the full path.

    Parameters:
    -----------
        data_dict: A nested dictionary to search through
        only_last_key: If True, uses only the leaf key name; if False, uses the full path

    Returns:
    --------
        A dictionary where keys are either leaf keys or paths to string values,
        and values are the corresponding strings

    Examples:
        >>> data = {
        ...     "a": {
        ...         "b": "value1",
        ...         "c": {
        ...             "d": "value2"
        ...         }
        ...     },
        ...     "e": ["list", "of", "values"],
        ...     "f": "value3"
        ... }
        >>>
        >>> # With only_last_key=True (default)
        >>> extract_string_values(data)
        {'b': 'value1', 'd': 'value2', 'f': 'value3'}
        >>>
        >>> # With only_last_key=False
        >>> extract_string_values(data, only_last_key=False)
        {'a.b': 'value1', 'a.c.d': 'value2', 'f': 'value3'}
    """

    if isinstance(data_dict, str):
        # Check if the string is a valid JSON file path
        if os.path.isfile(data_dict):
            # Load the custom JSON file
            with open(data_dict, "r") as file:
                data_dict = json.load(file)
        else:
            # If the file does not exist, raise an error
            raise ValueError(f"Invalid file path: {data_dict}")

    result = {}

    def explore_dict(d, path=""):
        if not isinstance(d, dict):
            return

        for key, value in d.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, str):
                # Use either just the key or the full path based on the parameter
                result_key = key if only_last_key else current_path
                result[result_key] = value
            elif isinstance(value, dict):
                explore_dict(value, current_path)
            # Skip lists and other types

    explore_dict(data_dict)
    return result


####################################################################################################
def expand_and_concatenate(df_add: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands df_add to match the number of rows in df and concatenates them along columns.

    Parameters:
    -----------
        df_add : pd.DataFrame
            DataFrame with a single row to be replicated.

        df : pd.DataFrame
            DataFrame to which df_add will be concatenated.

    Returns:
    --------
        pd.DataFrame: Concatenated DataFrame with df_add repeated and merged with df.


    """

    df_expanded = pd.concat([df_add] * len(df), ignore_index=True)

    # Detect if there is a column in df that exists in df_add. If so, assign the values from df to df_add and remove the column from df
    for col in df.columns:
        if col in df_add.columns:
            df_expanded[col] = df[col].values
            df = df.drop(columns=[col])

    df = df.reset_index(drop=True)  # Ensure clean index
    return pd.concat([df_expanded, df], axis=1)


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############        Section 9: Methods dedicated to containerization assistance         ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################


def generate_container_command(
    bash_args,
    technology: str = "local",
    image_path: str = None,
    license_path: str = None,
) -> list:
    """
    This function generates the command to run a bash command inside a container

    Parameters
    ----------
    bash_args : list
        List of arguments for the bash command

    technology : str
        Container technology ("docker" or "singularity"). Default is "local"

    image_path : str
        Path to the container image. Default is None

    Returns
    -------
    container_cmd: list
        List with the command to run the bash command locally or inside the container

    How to Use:
    --------------
        >>> bash_args = ["bash", "-c", "echo Hello World"]
        >>> container_cmd = generate_container_command(bash_args, technology="docker", image_path="/path/to/image")
        >>> print(container_cmd)

    """

    # Checks if the variable "a_list" is a list
    if isinstance(bash_args, str):
        bash_args = shlex.split(bash_args)

    path2mount = []
    if technology in ["docker", "singularity"]:

        # Adding the container image path and the bash command arguments
        if image_path is not None:
            if not os.path.exists(image_path):
                raise ValueError(f"The container image {image_path} does not exist.")
        else:
            raise ValueError(
                "The image path is required for Singularity containerization."
            )

        # Checking if the arguments are files or directories
        container_cmd = []
        bind_mounts = []

        for arg in bash_args:  # Checking if the arguments are files or directories
            abs_arg_path = os.path.dirname(arg)
            if os.path.exists(abs_arg_path):
                bind_mounts.append(
                    abs_arg_path
                )  # Adding the argument to the bind mounts

        if bind_mounts:  # Adding the bind mounts to the container command
            # Detect only the unique elements in the list bind_mounts
            bind_mounts = list(set(bind_mounts))
            for mount_path in bind_mounts:
                if technology == "singularity":  # Using Singularity technology
                    path2mount.extend(["--bind", f"{mount_path}:{mount_path}"])

                elif technology == "docker":  # Using Docker technology
                    path2mount.extend(["-v", f"{mount_path}:{mount_path}"])

        # Creating the container command
        if technology == "singularity":  # Using Singularity technology
            container_cmd.append("singularity")  # singularity command
            container_cmd.append("run")

        # Using Docker technology
        elif technology == "docker":
            container_cmd.append("docker")  # docker command
            container_cmd.append("run")

        container_cmd = container_cmd + path2mount

        container_cmd.append(image_path)
        container_cmd.extend(bash_args)

    else:  # No containerization
        container_cmd = bash_args

    return container_cmd


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############       Section 10: Methods to print information and signatures              ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################


def format_signature(sig: inspect.Signature):
    """
    Formats a function signature with ANSI colors.

    """
    parts = [f"{bcolors.OKWHITE}({bcolors.ENDC}"]
    params = list(sig.parameters.values())
    for i, p in enumerate(params):
        param_str = f"{bcolors.DARKCYAN}{p.name}{bcolors.ENDC}"

        if p.annotation != inspect.Parameter.empty:
            annotation = (
                p.annotation.__name__
                if hasattr(p.annotation, "__name__")
                else str(p.annotation)
            )
            param_str += f": {bcolors.OKPURPLE}{annotation}{bcolors.ENDC}"

        if p.default != inspect.Parameter.empty:
            param_str += f"{bcolors.OKGRAY} = {repr(p.default)}{bcolors.ENDC}"

        parts.append(param_str)
        if i < len(params) - 1:
            parts.append(f"{bcolors.OKWHITE}, {bcolors.ENDC}")
    parts.append(f"{bcolors.OKWHITE}){bcolors.ENDC}")
    return "".join(parts)


####################################################################################################
def show_module_contents(module):
    """
    Displays all classes and functions in a given module with colored formatting.
    Accepts a module object or module name (str).
    """
    if isinstance(module, str):
        try:
            module = sys.modules.get(module) or __import__(module)
        except ImportError:
            print(
                f"{bcolors.FAIL}Module '{module}' could not be imported.{bcolors.ENDC}"
            )
            return
    elif not isinstance(module, types.ModuleType):
        print(
            f"{bcolors.FAIL}Invalid input: must be a module object or module name string.{bcolors.ENDC}"
        )
        return

    print(
        f"{bcolors.HEADER}{bcolors.BOLD}📦 Contents of module '{module.__name__}':{bcolors.ENDC}\n"
    )

    # Classes
    print(f"{bcolors.OKBLUE}{bcolors.BOLD}📘 Classes:{bcolors.ENDC}")
    for name in sorted(dir(module)):
        try:
            obj = getattr(module, name)
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                print(f"  {bcolors.OKBLUE}- {name}{bcolors.ENDC}")

                doc = inspect.getdoc(obj)
                if doc:
                    first_line = doc.split("\n")[0]
                    print(f"    {bcolors.OKGRAY}# {first_line}{bcolors.ENDC}")

                for method_name, method in inspect.getmembers(
                    obj, predicate=inspect.isfunction
                ):
                    if (
                        method.__module__ == module.__name__
                        and method.__qualname__.startswith(obj.__name__ + ".")
                    ):
                        sig = inspect.signature(method)
                        formatted_sig = format_signature(sig)
                        print(
                            f"    {bcolors.OKYELLOW}• {method_name}{bcolors.ENDC}{formatted_sig}"
                        )
                        method_doc = inspect.getdoc(method)

                        if method_doc:
                            first_line = method_doc.split("\n")[0]
                            print(f"      {bcolors.OKGRAY}# {first_line}{bcolors.ENDC}")

                print(f"    {bcolors.OKWHITE}{'─'*60}{bcolors.ENDC}\n")
        except Exception:
            continue

    # Functions
    print(f"\n{bcolors.OKGREEN}{bcolors.BOLD}🔧 Functions:{bcolors.ENDC}")
    for name in sorted(dir(module)):
        try:
            obj = getattr(module, name)
            if inspect.isfunction(obj) and obj.__module__ == module.__name__:
                sig = inspect.signature(obj)
                formatted_sig = format_signature(sig)
                print(f"  {bcolors.OKYELLOW}- {name}{bcolors.ENDC}{formatted_sig}")
                doc = inspect.getdoc(obj)
                if doc:
                    print(f"    {bcolors.OKGRAY}# {doc.splitlines()[0]}{bcolors.ENDC}")
        except Exception:
            continue


####################################################################################################
def h5explorer(
    file_path: str,
    max_datasets_per_group: int = 20,
    max_attrs: int = 5,
    show_values: bool = True,
) -> Dict[str, Any]:
    """
    Print the hierarchical structure of an HDF5 file with colors and tree visualization.

    This function displays HDF5 file contents in a tree-like structure with color-coded
    elements, detailed information about datasets and groups, and limits the number of
    datasets shown per group to avoid overwhelming output.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file to analyze
    max_datasets_per_group : int, default=20
        Maximum number of datasets to display per group before truncating.
        Groups will show all child groups but limit datasets to this number.
    max_attrs : int, default=5
        Maximum number of attributes to display per item before truncating
    show_values : bool, default=True
        Whether to show attribute values in the output. If False, only
        attribute names are displayed.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing file statistics with keys:
        - 'groups': total number of groups in the file
        - 'datasets': total number of datasets in the file
        - 'total_size_mb': total size of all datasets in megabytes
        - 'file_path': path to the analyzed file

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    OSError
        If the file cannot be opened or is not a valid HDF5 file
    Exception
        For other HDF5 reading errors or invalid file formats

    Example
    -------
    >>> stats = print_h5_structure("/path/to/data.h5", max_datasets_per_group=10)
    📁 data/ (group)
    ├── 📊 measurements [1000 × 256] float64 (2.0 MB)
    │   └── 🏷️ @units = 'volts'
    ├── 📁 metadata/ (group)
    │   └── 📊 info scalar string (0.0 MB)
    └── 📊 results [100 × 50] complex128 (0.8 MB)

    >>> print(f"File contains {stats['datasets']} datasets")
    File contains 15 datasets

    Notes
    -----
    - Groups are displayed with 📁 (red color)
    - Datasets are displayed with 📊 (green color)
    - Attributes are displayed with 🏷️ (yellow color)
    - Tree structure uses Unicode box-drawing characters
    - Large groups show first N datasets + truncation message
    - Requires colorama package for colored output
    """

    def _get_tree_chars(is_last: bool, depth: int) -> str:
        """Generate tree characters for visual hierarchy."""
        if depth == 0:
            return ""

        chars = ""
        for i in range(depth - 1):
            chars += "│   "

        if is_last:
            chars += "└── "
        else:
            chars += "├── "

        return chars

    def _format_dtype(dtype: np.dtype) -> str:
        """Format numpy dtype for display."""
        if dtype.names:  # Compound dtype
            return f"compound({len(dtype.names)} fields)"
        return str(dtype)

    def _format_shape(shape: Tuple[int, ...]) -> str:
        """Format array shape for display."""
        if shape == ():
            return "scalar"
        return f"[{' × '.join(map(str, shape))}]"

    def _format_attribute_value(value: Any) -> str:
        """Format an attribute value for display."""
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return f" = {value.item()}"
            elif value.size <= 5:
                return f" = {value.tolist()}"
            else:
                return f" = [{_format_shape(value.shape)} array]"
        elif isinstance(value, (bytes, np.bytes_)):
            return f" = '{value.decode('utf-8', errors='ignore')}'"
        else:
            return f" = {value}"

    def _print_attributes(obj: h5py.HLObject, depth: int, prefix: str = "") -> None:
        """Print attributes of an HDF5 object."""
        if not obj.attrs:
            return

        attrs = list(obj.attrs.items())
        for i, (name, value) in enumerate(attrs[:max_attrs]):
            is_last_attr = i == len(attrs[:max_attrs]) - 1
            attr_prefix = _get_tree_chars(is_last_attr, depth + 1)

            # Format attribute value
            if show_values:
                val_str = _format_attribute_value(value)
            else:
                val_str = ""

            print(
                f"{prefix}{attr_prefix}"
                f"{Fore.YELLOW}@{name}{Style.RESET_ALL}"
                f"{Fore.CYAN}{val_str}{Style.RESET_ALL}"
            )

        if len(attrs) > max_attrs:
            more_attrs = len(attrs) - max_attrs
            attr_prefix = _get_tree_chars(True, depth + 1)
            print(
                f"{prefix}{attr_prefix}"
                f"{Style.DIM}... {more_attrs} more attributes{Style.RESET_ALL}"
            )

    def _print_item(
        name: str,
        obj: h5py.HLObject,
        depth: int = 0,
        prefix: str = "",
        is_last: bool = True,
    ) -> None:
        """Recursively print HDF5 items with proper handling of dataset limits."""
        tree_chars = _get_tree_chars(is_last, depth)

        if isinstance(obj, h5py.Group):
            # Print group
            print(
                f"{prefix}{tree_chars}"
                f"{Fore.RED}📁 {name}{Style.RESET_ALL} "
                f"{Style.DIM}(group){Style.RESET_ALL}"
            )

            stats["groups"] += 1

            # Print group attributes
            _print_attributes(obj, depth, prefix)

            # Print group contents with dataset limiting
            items = list(obj.items())
            datasets = [(n, o) for n, o in items if isinstance(o, h5py.Dataset)]
            groups = [(n, o) for n, o in items if isinstance(o, h5py.Group)]

            # Combine groups first, then limited datasets
            display_items = groups + datasets[:max_datasets_per_group]

            for i, (child_name, child_obj) in enumerate(display_items):
                is_last_child = (i == len(display_items) - 1) and len(
                    datasets
                ) <= max_datasets_per_group
                _print_item(child_name, child_obj, depth + 1, prefix, is_last_child)

            # Show truncation message if needed
            if len(datasets) > max_datasets_per_group:
                truncated_count = len(datasets) - max_datasets_per_group
                truncation_prefix = _get_tree_chars(True, depth + 1)
                print(
                    f"{prefix}{truncation_prefix}"
                    f"{Style.DIM}... {truncated_count} more datasets (showing first {max_datasets_per_group}){Style.RESET_ALL}"
                )

        elif isinstance(obj, h5py.Dataset):
            # Print dataset
            shape_str = _format_shape(obj.shape)
            dtype_str = _format_dtype(obj.dtype)
            size_mb = obj.nbytes / (1024 * 1024)

            print(
                f"{prefix}{tree_chars}"
                f"{Fore.GREEN}📊 {name}{Style.RESET_ALL} "
                f"{Fore.BLUE}{shape_str}{Style.RESET_ALL} "
                f"{Fore.MAGENTA}{dtype_str}{Style.RESET_ALL} "
                f"{Style.DIM}({size_mb:.1f} MB){Style.RESET_ALL}"
            )

            stats["datasets"] += 1
            stats["total_size"] += obj.nbytes

            # Print dataset attributes
            _print_attributes(obj, depth, prefix)

    def _count_all_items(obj: h5py.HLObject, counts: Dict[str, int]) -> None:
        """Recursively count all items in the HDF5 file."""
        for item in obj.values():
            if isinstance(item, h5py.Group):
                counts["groups"] += 1
                _count_all_items(item, counts)
            elif isinstance(item, h5py.Dataset):
                counts["datasets"] += 1
                counts["total_size"] += item.nbytes

    # Initialize statistics
    stats = {"groups": 0, "datasets": 0, "total_size": 0}

    try:
        print(f"\n{Back.BLUE}{Fore.WHITE} HDF5 File Structure {Style.RESET_ALL}")
        print(f"{Style.BRIGHT}File: {file_path}{Style.RESET_ALL}\n")

        with h5py.File(file_path, "r") as f:
            # Print root attributes if any
            if f.attrs:
                print(f"{Fore.YELLOW}Root Attributes:{Style.RESET_ALL}")
                _print_attributes(f, -1, "")
                print()

            # Print file contents
            items = list(f.items())
            if not items:
                print(f"{Style.DIM}(empty file){Style.RESET_ALL}")
            else:
                for i, (name, obj) in enumerate(items):
                    is_last = i == len(items) - 1
                    _print_item(name, obj, 0, "", is_last)

            # Count all items for accurate statistics
            total_counts = {"groups": 0, "datasets": 0, "total_size": 0}
            _count_all_items(f, total_counts)

        # Print legend
        print(f"\n{Style.BRIGHT}Legend:{Style.RESET_ALL}")
        print(f"📁 {Fore.RED}Groups{Style.RESET_ALL} - containers/folders")
        print(f"📊 {Fore.GREEN}Datasets{Style.RESET_ALL} - data arrays")
        print(f"🏷️ {Fore.YELLOW}Attributes{Style.RESET_ALL} - metadata")

        # Print file statistics (use total counts, not display counts)
        print(f"\n{Style.BRIGHT}File Statistics:{Style.RESET_ALL}")
        print(f"📁 Total Groups: {total_counts['groups']}")
        print(f"📊 Total Datasets: {total_counts['datasets']}")
        print(f"💾 Total Size: {total_counts['total_size'] / (1024*1024):.1f} MB")

        # Return statistics
        return {
            "groups": total_counts["groups"],
            "datasets": total_counts["datasets"],
            "total_size_mb": total_counts["total_size"] / (1024 * 1024),
            "file_path": file_path,
        }

    except FileNotFoundError:
        print(f"{Fore.RED}Error: File '{file_path}' not found{Style.RESET_ALL}")
        raise
    except OSError as e:
        print(f"{Fore.RED}Error: Cannot open file '{file_path}' - {e}{Style.RESET_ALL}")
        raise
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        raise


def h5explorer_simple(file_path: str, max_datasets_per_group: int = 20) -> None:
    """
    Print a simplified version of the HDF5 structure without colors (for basic terminals).

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file to analyze
    max_datasets_per_group : int, default=20
        Maximum number of datasets to display per group before truncating

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    OSError
        If the file cannot be opened or is not a valid HDF5 file

    Example
    -------
    >>> print_h5_structure_simple("data.h5", max_datasets_per_group=10)
    HDF5 Structure: data.h5
    --------------------------------------------------
    📁 data/ (group)
    📊 measurements [1000, 256] float64
    📁 metadata/ (group)
        📊 info () <U10
    ... 5 more datasets
    """

    def _print_item_simple(name: str, obj: h5py.HLObject, depth: int = 0) -> None:
        """Print items in simple format without colors."""
        indent = "  " * depth

        if isinstance(obj, h5py.Group):
            print(f"{indent}📁 {name}/ (group)")

            # Apply same dataset limiting logic
            items = list(obj.items())
            datasets = [(n, o) for n, o in items if isinstance(o, h5py.Dataset)]
            groups = [(n, o) for n, o in items if isinstance(o, h5py.Group)]

            # Show all groups, limited datasets
            for child_name, child_obj in groups + datasets[:max_datasets_per_group]:
                _print_item_simple(child_name, child_obj, depth + 1)

            if len(datasets) > max_datasets_per_group:
                truncated = len(datasets) - max_datasets_per_group
                print(f"{'  ' * (depth + 1)}... {truncated} more datasets")

        elif isinstance(obj, h5py.Dataset):
            shape = obj.shape if obj.shape != () else "scalar"
            print(f"{indent}📊 {name} {shape} {obj.dtype}")

    try:
        print(f"HDF5 Structure: {file_path}")
        print("-" * 50)

        with h5py.File(file_path, "r") as f:
            for name, obj in f.items():
                _print_item_simple(name, obj)

    except Exception as e:
        print(f"Error: {e}")
        raise


def show_object_content(obj, show_private=False, show_dunder=False):
    """
    Print object properties and methods with ANSI colors, similar to inspect.help().

    Parameters:
        obj: The object to inspect
        show_private (bool): Whether to show private methods/attributes (starting with _)
        show_dunder (bool): Whether to show dunder methods (starting with __)

    Returns:
        None: Prints the colorized help information to stdout

    Examples:
        >>> show_object_content(str)
        >>> show_object_content(my_object, show_private=True)
        >>> show_object_content(MyClass, show_dunder=True)
    """
    # Get object info
    obj_type = type(obj)
    obj_name = getattr(obj, "__name__", str(obj))
    module_name = getattr(obj_type, "__module__", "unknown")

    # Print header
    print(f"{bcolors.BOLD}{bcolors.HEADER}{'='*60}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.HEADER}✅ INSPECTION COMPLETE{bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.HEADER}{'='*60}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.HEADER}🔍 OBJECT INSPECTOR{bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.HEADER}{'='*60}{bcolors.ENDC}")

    # Object information
    print(
        f"{bcolors.BOLD}{bcolors.OKCYAN}📦 Object:{bcolors.ENDC} {bcolors.OKWHITE}{obj_name}{bcolors.ENDC}"
    )
    print(
        f"{bcolors.BOLD}{bcolors.OKCYAN}🏷️  Type:{bcolors.ENDC} {bcolors.OKWHITE}{obj_type.__name__}{bcolors.ENDC}"
    )
    print(
        f"{bcolors.BOLD}{bcolors.OKCYAN}📁 Module:{bcolors.ENDC} {bcolors.OKWHITE}{module_name}{bcolors.ENDC}"
    )

    # Class docstring
    doc = inspect.getdoc(obj)
    if doc:
        print(f"\n{bcolors.BOLD}{bcolors.OKGREEN}📝 Description:{bcolors.ENDC}")
        print(f"{bcolors.ITALIC}{bcolors.OKGRAY}{doc}{bcolors.ENDC}")

    # Get all members
    members = inspect.getmembers(obj)

    # Categorize members
    methods = []
    properties = []
    attributes = []

    for name, value in members:
        # Filter based on visibility preferences
        if not show_dunder and name.startswith("__") and name.endswith("__"):
            continue
        if not show_private and name.startswith("_") and not name.startswith("__"):
            continue

        if inspect.ismethod(value) or inspect.isfunction(value):
            methods.append((name, value))
        elif inspect.isdatadescriptor(value) or isinstance(value, property):
            properties.append((name, value))
        else:
            attributes.append((name, value))

    # Print Methods
    if methods:
        print(
            f"\n{bcolors.BOLD}{bcolors.OKBLUE}{'─'*15} ⚙️  METHODS {'─'*15}{bcolors.ENDC}"
        )
        for name, method in sorted(methods):
            try:
                sig = inspect.signature(method)
                print(
                    f"{bcolors.BOLD}{bcolors.OKYELLOW}🔧 {name}{bcolors.ENDC}{bcolors.OKWHITE}{sig}{bcolors.ENDC}"
                )

                # Method docstring
                method_doc = inspect.getdoc(method)
                if method_doc:
                    # Show first line of docstring
                    first_line = method_doc.split("\n")[0]
                    print(
                        f"    {bcolors.ITALIC}{bcolors.OKGRAY}{first_line}{bcolors.ENDC}"
                    )
            except (ValueError, TypeError):
                print(
                    f"{bcolors.BOLD}{bcolors.OKYELLOW}🔧 {name}{bcolors.ENDC}{bcolors.OKGRAY}(signature unavailable){bcolors.ENDC}"
                )
            print()

    # Print Properties
    if properties:
        print(
            f"{bcolors.BOLD}{bcolors.OKMAGENTA}{'─'*15} 🏠 PROPERTIES {'─'*12}{bcolors.ENDC}"
        )
        for name, prop in sorted(properties):
            print(f"{bcolors.BOLD}{bcolors.PURPLE}🔑 {name}{bcolors.ENDC}")

            # Property docstring
            prop_doc = inspect.getdoc(prop)
            if prop_doc:
                first_line = prop_doc.split("\n")[0]
                print(f"    {bcolors.ITALIC}{bcolors.OKGRAY}{first_line}{bcolors.ENDC}")
            print()

    # Print Attributes
    if attributes:
        print(
            f"{bcolors.BOLD}{bcolors.DARKCYAN}{'─'*15} 📊 ATTRIBUTES {'─'*12}{bcolors.ENDC}"
        )
        for name, attr in sorted(attributes):
            attr_type = type(attr).__name__
            attr_repr = repr(attr)

            # Truncate long representations
            if len(attr_repr) > 50:
                attr_repr = attr_repr[:47] + "..."

            print(
                f"{bcolors.BOLD}{bcolors.OKCYAN}📌 {name}{bcolors.ENDC} "
                f"{bcolors.OKGRAY}({attr_type}){bcolors.ENDC}: "
                f"{bcolors.DARKWHITE}{attr_repr}{bcolors.ENDC}"
            )

    # MRO (Method Resolution Order) for classes
    if inspect.isclass(obj):
        mro = inspect.getmro(obj)
        if len(mro) > 1:
            print(
                f"\n{bcolors.BOLD}{bcolors.WARNING}{'─'*10} 🏗️  METHOD RESOLUTION ORDER {'─'*10}{bcolors.ENDC}"
            )
            for i, cls in enumerate(mro):
                print(
                    f"{bcolors.OKYELLOW}🔗 {i+1}.{bcolors.ENDC} {bcolors.OKWHITE}{cls.__name__}{bcolors.ENDC} "
                    f"{bcolors.OKGRAY}({cls.__module__}){bcolors.ENDC}"
                )

    print(f"{bcolors.BOLD}{bcolors.HEADER}{'='*60}{bcolors.ENDC}")


def search_methods(obj, keyword, case_sensitive=False):
    """
    Search for methods/attributes containing a keyword in name or docstring.

    Parameters:
        obj: The object to search in
        keyword (str): The keyword to search for
        case_sensitive (bool): Whether the search should be case sensitive

    Returns:
        None: Prints the search results to stdout

    Examples:
        >>> search_methods(str, "find")
        >>> search_methods(my_toolkit, "config", case_sensitive=True)
        >>> search_methods(pandas.DataFrame, "drop")

    Note:
        Searches both method names and their docstrings for the keyword.
        Results are displayed with colorized output showing matches.
    """
    if not case_sensitive:
        keyword = keyword.lower()

    print(
        f"{bcolors.BOLD}{bcolors.HEADER}🔍 SEARCH RESULTS for '{keyword}'{bcolors.ENDC}"
    )
    print(f"{bcolors.BOLD}{bcolors.HEADER}{'='*40}{bcolors.ENDC}")

    members = inspect.getmembers(obj)
    found = False

    for name, value in members:
        name_match = keyword in (name.lower() if not case_sensitive else name)
        doc_match = False

        # Check docstring
        doc = inspect.getdoc(value)
        if doc:
            doc_match = keyword in (doc.lower() if not case_sensitive else doc)

        if name_match or doc_match:
            found = True
            print(f"{bcolors.BOLD}{bcolors.OKYELLOW}✨ {name}{bcolors.ENDC}")

            if inspect.ismethod(value) or inspect.isfunction(value):
                try:
                    sig = inspect.signature(value)
                    print(
                        f"  {bcolors.OKGRAY}📋 Signature:{bcolors.ENDC} {bcolors.OKWHITE}{sig}{bcolors.ENDC}"
                    )
                except:
                    pass

            if doc:
                first_line = doc.split("\n")[0]
                if len(first_line) > 80:
                    first_line = first_line[:77] + "..."
                print(f"  {bcolors.ITALIC}{bcolors.OKGRAY}{first_line}{bcolors.ENDC}")

            print()

    if not found:
        print(f"{bcolors.WARNING}❌ No matches found for '{keyword}'{bcolors.ENDC}")
