

import numpy as np
from typing import Union
import shlex
import os
import argparse
from datetime import datetime
import pandas as pd
import inspect
import sys
import types

####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############           Methods dedicated to improve the documentation                   ############
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
############                    Methods related to progress bar                         ############
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
############                    Methods dedicated to work with colors                   ############
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
def rgb2hex(r: int, g: int, b: int) -> str:
    """
    Function to convert rgb to hex

    Parameters
    ----------
    r : int
        Red value
    g : int
        Green value
    b : int
        Blue value

    Returns
    -------
    hexcode: str
        Hexadecimal code for the color

    Example Usage:
    --------------
        >>> r = 255
        >>> g = 0
        >>> b = 0
        >>> hexcode = rgb2hex(r, g, b)
        >>> print(hexcode)  # Output: "#ff0000" 
    
    """

    return "#{:02x}{:02x}{:02x}".format(r, g, b)

####################################################################################################
def multi_rgb2hex(colors: Union[list, np.ndarray]) -> list:
    """
    Function to convert rgb to hex for an array of colors

    Parameters
    ----------
    colors : list or numpy array
        List of rgb colors

    Returns
    -------
    hexcodes: list
        List of hexadecimal codes for the colors

    Example Usage:
    --------------
        >>> colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        >>> hexcodes = multi_rgb2hex(colors)
        >>> print(hexcodes)  # Output: ['#ff0000', '#00ff00', '#0000ff']
        
    """
    if len(colors) > 0:
        # If all the values in the list are between 0 and 1, then the values are multiplied by 255
        colors = readjust_colors(colors)

        hexcodes = []
        if isinstance(colors, list):
            for indcol, color in enumerate(colors):
                if isinstance(colors[indcol], str):
                    hexcodes.append(colors[indcol])

                elif isinstance(colors[indcol], np.ndarray):
                    hexcodes.append(rgb2hex(color[0], color[1], color[2]))

        elif isinstance(colors, np.ndarray):
            nrows, ncols = colors.shape
            for i in np.arange(0, nrows):
                hexcodes.append(rgb2hex(colors[i, 0], colors[i, 1], colors[i, 2]))
    else:
        hexcodes = []

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

    Example Usage:
    --------------
        >>> hexcode = "#FF5733"
        >>> rgb = hex2rgb(hexcode)
        >>> print(rgb)  # Output: (255, 87, 51)
        
    """
    # Convert hexadecimal color code to RGB values
    hexcode = hexcode.lstrip("#")
    return tuple(int(hexcode[i : i + 2], 16) for i in (0, 2, 4))

####################################################################################################
def multi_hex2rgb(hexcodes: list) -> np.ndarray:
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

    Example Usage:
    --------------
        >>> hexcodes = ["#FF5733", "#33FF57", "#3357FF"]
        >>> rgb_list = multi_hex2rgb(hexcodes)
        >>> print(rgb_list)  # Output: [[255, 87, 51], [51, 255, 87], [51, 87, 255]]
        
    """

    rgb_list = [hex2rgb(hex_color) for hex_color in hexcodes]
    return np.array(rgb_list)

####################################################################################################
def invert_colors(colors: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
    """
    Function to invert the colors by finding its complementary color.

    Parameters
    ----------
    colors : list or numpy array
        List of colors

    Returns
    -------
    colors: list or numpy array
        List of inverted colors 
        
    Example Usage:
    ----------------
        >>> colors = ["#FF5733", "#33FF57", np.array([51, 87, 255])]
        >>> inverted_colors = invert_colors(colors)
        >>> print(inverted_colors)  # Output: ['#00aacc', '#cc00a8', [204, 168, 0]]
    """

    bool_norm = False
    if isinstance(colors, list):

        if isinstance(colors[0], str):
            # Convert the hexadecimal colors to rgb
            colors = multi_hex2rgb(colors)
            color_type = "hex"

        elif isinstance(colors[0], np.ndarray):
            colors = np.array(colors)
            color_type = "arraylist"

            if all(map(lambda x: max(x) < 1, colors)):
                colors = [color * 255 for color in colors]
                bool_norm = True

        else:
            raise ValueError(
                "If colors is a list, it must be a list of hexadecimal colors or a list of rgb colors"
            )

    elif isinstance(colors, np.ndarray):
        color_type = "array"
        if np.max(colors) <= 1:
            colors = colors * 255
            bool_norm = True
    else:
        raise ValueError("The colors must be a list of colors or a numpy array")

    ## Inverting the colors
    colors = 255 - colors

    if color_type == "hex":
        colors = multi_rgb2hex(colors)

    elif color_type == "arraylist":
        if bool_norm:
            colors = colors / 255

        # Create a list of colors where each row is an element in the list
        colors = [list(color) for color in colors]

    elif color_type == "array":
        if bool_norm:
            colors = colors / 255

    return colors

####################################################################################################
def harmonize_colors(colors: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
    """
    Function to harmonize the colors in a list. The colors can be in hexadecimal or rgb format.
    If the list contains colors in multiple formats, the function will convert all the colors to hexadecimal format.

    Parameters
    ----------
    colors : list or numpy array
        List of colors

    Returns
    -------
    colors: list
        List of colors in hexadecimal format

    Example Usage:
    ----------------
        >>> colors = ["#FF5733", "#33FF57", np.array([51, 87, 255])]
        >>> harmonized_colors = harmonize_colors(colors)
        >>> print(harmonized_colors)  # Output: ['#ff5733', '#33ff57', '#3393ff']
        
    """

    bool_tmp = all(isinstance(x, np.ndarray) for x in colors)
    if bool_tmp:
        hexcodes = []
        for indcol, color in enumerate(colors):
            if isinstance(colors[indcol], str):
                hexcodes.append(colors[indcol])

            elif isinstance(colors[indcol], np.ndarray):
                hexcodes.append(rgb2hex(color[0], color[1], color[2]))
        colors = hexcodes

    return colors

####################################################################################################
def readjust_colors(colors: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
    """
    Function to readjust the colors to the range 0-255

    Parameters
    ----------
    colors : list or numpy array
        List of colors

    Returns
    -------
    colors: Numpy array
        List of colors normalized

    """

    if isinstance(colors, list):

        # If all the values in the list are between 0 and 1, then the values are multiplied by 255
        if not isinstance(colors[0], str):
            if all(map(lambda x: max(x) < 1, colors)):
                colors = [color * 255 for color in colors]

        bool_tmp = all(isinstance(x, np.ndarray) for x in colors)
        if bool_tmp:
            hexcodes = []
            for indcol, color in enumerate(colors):
                if isinstance(colors[indcol], str):
                    hexcodes.append(colors[indcol])

                elif isinstance(colors[indcol], np.ndarray):
                    hexcodes.append(rgb2hex(color[0], color[1], color[2]))
            colors = hexcodes

    elif isinstance(colors, np.ndarray):
        nrows, ncols = colors.shape

        # If all the values in the array are between 0 and 1, then the values are multiplied by 255
        if np.max(colors) <= 1:
            colors = colors * 255

    return colors

####################################################################################################
def create_random_colors(n: int, fmt: str = "rgb") -> np.ndarray:
    """
    Function to create a list of n random colors

    Parameters
    ----------
    n : int
        Number of colors
        
    fmt : str
        Format of the colors. It can be 'rgb', 'rgbnorm' or 'hex'. Default is 'rgb'.

    Returns
    -------
    colors: list
        List of random colors
        
    Example Usage:
    ----------------
        >>> colors = create_random_colors(5)
        >>> print(colors)  # Output: [[123, 45, 67], [89, 12, 34], ...]
        

    """

    # Create a numpy array with n random colors in the range 0-255
    colors = np.random.randint(0, 255, size=(n, 3))
    
    if fmt == "hex":
        # Convert the colors to hexadecimal format
        colors = multi_rgb2hex(colors)
    elif fmt == "rgbnorm":
        # Normalize the colors to the range 0-1
        colors = colors / 255

    return colors

####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############                    Methods dedicated to work with dates                    ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################

def find_closest_date(dates_list: list, target_date: str, date_fmt: str = "%Y%m%d"):
    """
    Function to find the closest date in a list of dates to a target date.
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
        
    Example Usage:
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
############           Methods dedicated to create and work with indexes,               ############
############           to search for elements in a list, etc                            ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def build_indexes(range_vector: list, nonzeros: bool = True):
    """
    Function to build the indexes from a range vector that can contain integers, tuples, lists or strings.

    For example:
    range_vector = [1, (2, 5), [6, 7], "8-10", "11:13", "14:2:22"]

    In this example the tuple (2, 5) will be converted to [2, 3, 4, 5]
    The list [6, 7] will be kept as it is
    The string "8-10" will be converted to [8, 9, 10]
    The string "11:13" will be converted to [11, 12, 13]
    The string "14:2:22" will be converted to [14, 16, 18, 20, 22]

    All this values will be flattened and unique values will be returned.
    In this case the output will be [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22]

    Parameters
    ----------
    range_vector : list
        List of ranges

    nonzeros : bool
        Boolean to indicate if the zeros are removed. Default is True

    Returns
    -------
    indexes: list
        List of indexes

    Example Usage:
    --------------
        >>> range_vector = [1, (2, 5), [6, 7], "8-10", "11:13", "14:2:22"]
        >>> indexes = build_indexes(range_vector)
        >>> print(indexes)  # Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22]
        
    """

    indexes = []
    for i in range_vector:
        if isinstance(i, tuple):

            # Apend list from the minimum to the maximum value
            indexes.append(list(range(i[0], i[1] + 1)))

        elif isinstance(i, (int, np.integer)):
            # Append the value as an integer
            indexes.append([i])

        elif isinstance(i, list):
            # Append the values in the values in the list
            indexes.append(i)

        elif isinstance(i, str):

            # Find if the strin contains "-" or ":"
            if "-" in i:
                # Split the string by the "-"
                i = i.split("-")
                indexes.append(list(range(int(i[0]), int(i[1]) + 1)))
            elif ":" in i:
                # Split the string by the ":"
                i = i.split(":")
                if len(i) == 2:
                    indexes.append(list(range(int(i[0]), int(i[1]) + 1)))
                elif len(i) == 3:

                    # Append the values in the range between the minimum to the maximum value of the elements of the list with a step
                    indexes.append(list(range(int(i[0]), int(i[2]) + 1, int(i[1]))))

            else:

                try:
                    # Append the value as an integer
                    indexes.append([int(i)])
                except:
                    pass

    indexes = [item for sublist in indexes for item in sublist]

    if nonzeros:
        # Remove the elements with 0
        indexes = [x for x in indexes if x != 0]

    # Flatten the list and unique the values
    indexes = remove_duplicates(indexes)

    return indexes

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

    Example Usage:
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
def select_ids_from_file(subj_ids: list, ids_file: Union[list, str]):
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

    Example Usage:
    --------------
        >>> subj_ids = ["sub-01", "sub-02", "sub-03"]
        >>> ids_file = "ids.txt" # Column-wise text file with the ids to select (i.e. "sub-01", "sub-03")
        >>> out_ids = select_ids_from_file(subj_ids, ids_file)
        >>> print(out_ids)  # Output: ["sub-01", "sub-03"]
    """

    # Read the ids from the file
    if isinstance(ids_file, str):
        if os.path.exists(ids_file):
            with open(ids_file) as file:
                t1s2run = [line.rstrip() for line in file]

            out_ids = [s for s in subj_ids if any(xs in s for xs in t1s2run)]

    if isinstance(ids_file, list):
        out_ids = list_intercept(subj_ids, ids_file)

    return out_ids

####################################################################################################
def filter_by_substring(
    input_list: list, 
    or_filter: Union[str, list], 
    and_filter: Union[str, list] = None, 
    bool_case: bool = False) -> list:
    
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
        
    Example Usage:
    --------------
        >>> input_list = ["apple", "banana", "cherry", "date"]
        >>> or_filter = ["app", "ch"]
        >>> filtered_list = filter_by_substring(input_list, or_filter)
        >>> print(filtered_list)  # Output: ['apple', 'cherry']

    """

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
    indexes = [i for i, x in enumerate(tmp_input_list) if any(a in x for a in tmp_substr)]

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
        indexes = [i for i, x in enumerate(tmp_filtered_list) if all(a in x for a in tmp_and_filter)]
        
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
    match_entire_world: bool = False,
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

    match_entire_world : bool
        Boolean to indicate if the search is a whole word match. Default is False

    Returns
    -------
    indexes: list
        List of indexes that contain any of the substring

    Example Usage:
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
        >>> indexes = get_indexes_by_substring(input_list, substr, match_entire_world=True)
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
    if match_entire_world:
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
    
    Example Usage:
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

    Example Usage:
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
############     Methods dedicated to find directories, remove empty folders            ############
############     find all the files inside a certain directory, etc                     ############                                                 ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def detect_leaf_directories(root_dir: str) -> list:
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

    Example Usage:
    --------------
        >>> root_directory = "/path/to/your/folder"
        >>> leaf_folders = detect_leaf_directories(root_directory)
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
def detect_recursive_files(in_dir):
    """
    Function to detect all the files in a directory and its subdirectories

    Parameters
    ----------
    in_dir : str
        Input directory

    Returns
    -------
    files: list
        List of files in the directory and its subdirectories

    Example Usage:
    ----------------
        >>> in_dir = "/path/to/directory"
        >>> files = detect_recursive_files(in_dir)
        >>> print(files)  # Output: List of files in the directory and its subdirectories
    """

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(in_dir):
        for file in f:
            files.append(os.path.join(r, file))

    return files

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
    
    Example Usage:
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
                #print(f"Removed empty directory: {dir_path}")  # Optional logging
            except OSError:
                # Directory not empty or other error - we'll ignore it
                pass
    
    # Finally, try to remove the starting directory itself if it's now empty
    try:
        os.rmdir(start_path)
        deleted_folders.append(start_path)
        #print(f"Removed empty directory: {start_path}")  # Optional logging
    except OSError:
        pass
    
    return deleted_folders

####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############              Methods dedicated to strings and characters                   ############
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
    Example Usage:
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
############        Methods dedicated to work with dictionaries and dataframes          ############
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
        
    Example Usage:
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
def expand_and_concatenate(
    df_add: pd.DataFrame, 
    df: pd.DataFrame
    ) -> pd.DataFrame:
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
############            Methods dedicated to help with containarization                 ############
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
        
    Example Usage:
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
############            Methods to print modules information and signatures             ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################

def format_signature(sig: inspect.Signature):
    """Formats a function signature with ANSI colors."""
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
            print(f"{bcolors.FAIL}Module '{module}' could not be imported.{bcolors.ENDC}")
            return
    elif not isinstance(module, types.ModuleType):
        print(f"{bcolors.FAIL}Invalid input: must be a module object or module name string.{bcolors.ENDC}")
        return

    print(f"{bcolors.HEADER}{bcolors.BOLD}📦 Contents of module '{module.__name__}':{bcolors.ENDC}\n")

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

                for method_name, method in inspect.getmembers(obj, predicate=inspect.isfunction):
                    if method.__module__ == module.__name__ and method.__qualname__.startswith(obj.__name__ + "."):
                        sig = inspect.signature(method)
                        formatted_sig = format_signature(sig)
                        print(f"    {bcolors.OKYELLOW}• {method_name}{bcolors.ENDC}{formatted_sig}")
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