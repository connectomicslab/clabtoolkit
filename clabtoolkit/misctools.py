import numpy as np
from typing import Union
import shlex
import os
import argparse
from datetime import datetime

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
    def split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter.split_lines
        return argparse.HelpFormatter.split_lines(self, text, width)
    
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
    @params:
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


def rgb2hex(r:int, 
            g:int, 
            b:int):
    
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

    """

    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def multi_rgb2hex(colors: Union[list, np.ndarray]):
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

    """

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

    return hexcodes

def hex2rgb(hexcode: str):
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

    """
    # Convert hexadecimal color code to RGB values
    hexcode = hexcode.lstrip('#')
    return tuple(int(hexcode[i:i+2], 16) for i in (0, 2, 4))

def search_in_list(ref_list, list2look):
    """
    Search for the index of the elements in list2look in ref_list
    
    Parameters
    ----------
    ref_list : list
        Reference list
    list2look : list
        List to look for
        
    Returns
    -------
    list
        List of indices of the elements in list2look in ref_list
    
    """
    ret = []
    for v in list2look:
        index = ref_list.index(v)
        ret.append(index)
    return ret

def find_closest_date(dates_list: list,
                        target_date: str,
                        date_fmt:str='%Y%m%d'):
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
        
    
    """
    
    # Convert target_date to a datetime object
    target_date = datetime.strptime(str(target_date), date_fmt)
    
    # Convert all dates in the list to datetime objects
    dates_list_dt = [datetime.strptime(str(date), date_fmt) for date in dates_list]
    
    # Find the index of the date with the minimum difference from the target date
    closest_index = min(range(len(dates_list_dt)), key=lambda i: abs(dates_list_dt[i] - target_date))
    
    # Get the closest date from the list using the index
    closest_date = dates_list_dt[closest_index]
    
    # Get the time difference between the target date and the closest date in days
    time_diff = abs(closest_date - target_date).days
    
    
    # Convert the closest date back to the 'YYYYMMDD' format
    return closest_date.strftime(date_fmt), closest_index, time_diff



def multi_hex2rgb(hexcodes: list):
    """
    Function to convert hex to rgb for an array of colors

    Parameters
    ----------
    hexcodes : list
        List of hexadecimal codes for the colors

    Returns
    -------
    rgb_list: np.array
        Array of rgb values

    """

    rgb_list = [hex2rgb(hex_color) for hex_color in hexcodes]
    return np.array(rgb_list)

def build_indexes(range_vector: list, 
                    nonzeros: bool = True
                    ):
    """
    Function to build the indexes from a range vector. The range vector can contain integers, tuples, lists or strings.

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

    """

    indexes = []
    for i in range_vector:
        if isinstance(i, tuple):

            # Apend list from the minimum to the maximum value
            indexes.append(list(range(i[0], i[1]+1)))

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
                indexes.append(list(range(int(i[0]), int(i[1])+1)))
            elif ":" in i:
                # Split the string by the ":"
                i = i.split(":")
                if len(i) == 2:
                    indexes.append(list(range(int(i[0]), int(i[1])+1)))
                elif len(i) == 3:
                    
                    # Append the values in the range between the minimum to the maximum value of the elements of the list with a step
                    indexes.append(list(range(int(i[0]), int(i[2])+1, int(i[1]))))

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

    """

    
    unique_list = []
    seen_elements = set()

    for element in input_list:
        if element not in seen_elements:
            unique_list.append(element)
            seen_elements.add(element)

    return unique_list

def select_ids_from_file(subids: list, 
                            idfile: Union[list, str]):
    """
    Function to select the ids from a list of ids that are in a file.
    It can be used to select the ids from a list of subjects that are in a file.

    Parameters
    ----------
    subids : list
        List of subject ids.
    idfile : str or list
        File with the ids to select.
        
    Returns
    -------
    out_ids: list
        List of ids that are in the file.
    
    """
    
    # Read the ids from the file
    if isinstance(idfile, str):
        if os.path.exists(idfile):
            with open(idfile) as file:
                t1s2run = [line.rstrip() for line in file]

            out_ids = [s for s in subids if any(xs in s for xs in t1s2run)]
    
    if isinstance(idfile, list):
        out_ids = list_intercept(subids, idfile)

    return out_ids


def filter_by_substring(list1: list,
                        substr: Union[str, list], 
                        boolcase: bool = False):
    """
    Function to filter a list of elements by a substrings. 
    
    Parameters
    ----------
    list1 : list
        List of elements
        
    substr : str or list
        Substring to filter. It can be a string or a list of strings
    
    boolcase : bool
        Boolean to indicate if the search is case sensitive. Default is False

    Returns
    -------
    out_ids: list
        List of ids that contain any of the substring

    """
    
    # Rise an error if list1 is not a list  
    if not isinstance(list1, list):
        raise ValueError("The input list1 must be a list.")
    
    # Convert the substr to a list
    if isinstance(substr, str):
        substr = [substr]
    
    # Convert the substr and list1 to lower case
    if not boolcase:
        tmp_substr = [e.lower() for e in substr]
        tmp_list1 = [e.lower() for e in list1]
    
    else:
        tmp_substr = substr
        tmp_list1 = list1
        
        
    # Get the indexes of the list elements that contain any of the strings in the list aa
    indexes = [i for i, x in enumerate(tmp_list1) if any(a in x for a in tmp_substr)]

    # Convert indexes to a numpy array
    indexes = np.array(indexes)

    # Select the atlas_files with the indexes
    filt_list = [list1[i] for i in indexes]
    
    filt_list = remove_duplicates(filt_list)
    
    return filt_list

def get_indexes_by_substring(list1: list,
                        substr: Union[str, list], 
                        invert: bool = False,
                        boolcase: bool = False):
    """
    Function extracts the indexes of the elements of a list of elements that contain
    any of the substrings of anothre list. 
    
    Parameters
    ----------
    list1 : list
        List of elements
        
    substr : str or list
        Substring to filter. It can be a string or a list of strings
    
    boolcase : bool
        Boolean to indicate if the search is case sensitive. Default is False

    Returns
    -------
    indexes: list
        List of indexes that contain any of the substring

    """
    
    # Rise an error if list1 is not a list  
    if not isinstance(list1, list):
        raise ValueError("The input list1 must be a list.")
    
    # Convert the substr to a list
    if isinstance(substr, str):
        substr = [substr]
    
    # Convert the substr and list1 to lower case
    if not boolcase:
        tmp_substr = [e.lower() for e in substr]
        tmp_list1 = [e.lower() for e in list1]
    
    else:
        tmp_substr = substr
        tmp_list1 = list1
        
        
    # Get the indexes of the list elements that contain any of the strings in the list aa
    indexes = [i for i, x in enumerate(tmp_list1) if any(a in x for a in tmp_substr)]

    # Convert indexes to a numpy array
    indexes = np.array(indexes)

    if invert:
        indexes = np.setdiff1d(np.arange(0, len(list1)), indexes)
        
    return indexes


def list_intercept(list1: list,
                    list2: list):
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

    """
    
    # Rise an error if list1 or list2 are not lists  
    if not isinstance(list1, list):
        raise ValueError("The input list1 must be a list.")
    
    if not isinstance(list2, list):
        raise ValueError("The input list2 must be a list.")
    
    # Create a list of elements that are in both lists
    int_list = [value for value in list1 if value in list2]
    
    return int_list


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
    
    """
    
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(in_dir):
        for file in f:
            files.append(os.path.join(r, file))

    return files

def rem_dupplicate_char(strcad: str, 
                        dchar: str):
    """
    This function removes duplicate characters from strings.
    
    Parameters
    ----------
    s : list or str
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

    return ''.join(chars)

def invert_colors(colors: Union[list, np.ndarray]):
    """
    Function to invert the colors by finding its complementary color. 

    Parameters
    ----------
    colors : list or numpy array
        List of colors

    Returns
    -------
    colors: Numpy array
        List of colors

    """
    
    bool_norm = False
    if isinstance(colors, list):

        if isinstance(colors[0], str):
            # Convert the hexadecimal colors to rgb
            colors = multi_hex2rgb(colors)
            color_type = 'hex'
            
        elif isinstance(colors[0], np.ndarray):
            colors = np.array(colors)
            color_type = 'arraylist'
            
            if all(map(lambda x: max(x) < 1, colors)):
                colors = [color * 255 for color in colors]
                bool_norm = True
                
        else:
            raise ValueError("If colors is a list, it must be a list of hexadecimal colors or a list of rgb colors")
        
    elif isinstance(colors, np.ndarray):
        color_type = 'array'
        if np.max(colors) <= 1:
            colors = colors * 255
            bool_norm = True
    else:
        raise ValueError("The colors must be a list of colors or a numpy array")

    ## Inverting the colors
    colors = 255 - colors
    
    if color_type == 'hex':
        colors = multi_rgb2hex(colors)
    
    elif color_type == 'arraylist':
        if bool_norm:
            colors = colors / 255
        
        # Create a list of colors where each row is an element in the list
        colors = [list(color) for color in colors]
    
    elif color_type == 'array':
        if bool_norm:
            colors = colors / 255
    
    return colors

def harmonize_colors(colors: Union[list, np.ndarray]):
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

def readjust_colors(colors: Union[list, np.ndarray]):
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

class bcolors:
    """
    Class to define the colors for the terminal output.
    
    
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    OKYELLOW = '\033[93m'
    OKRED = '\033[91m'
    OKMAGENTA = '\033[95m'
    PURPLE = '\033[35m'
    OKCYAN = '\033[96m'
    DARKCYAN = "\033[36m"
    ORANGE = "\033[48:5:208m%s\033[m"
    OKWHITE = '\033[97m'
    DARKWHITE = '\033[37m'
    OKBLACK = '\033[30m'
    OKGRAY = '\033[90m'
    OKPURPLE = '\033[35m'
    
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'

def create_random_colors(n: int):
    """
    Function to create a list of n random colors

    Parameters
    ----------
    n : int
        Number of colors

    Returns
    -------
    colors: list
        List of random colors

    """

    # Create a numpy array with n random colors in the range 0-255
    colors = np.random.randint(0, 255, size=(n, 3))

    return colors

def correct_names(regnames: list, 
                    prefix: str = None, 
                    sufix: str = None, 
                    lower: bool = False,
                    remove: list = None,
                    replace: list = None):
        
    """
    Correcting region names
    @params:
        regnames   - Required  : List of region names:
        prefix     - Optional  : Add prefix to the region names:
        sufix      - Optional  : Add sufix to the region names:
        lower      - Optional  : Lower the region names. Default is False:
        remove     - Optional  : Remove the substring item from the region names:
        replace    - Optional  : Replace the substring item from the region names:
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

    """

    values, indices = np.unique(a, return_inverse=True)
    is_in_list = np.isin(a, b)
    idx = indices[is_in_list].astype(int)

    return values, idx

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
    
    """
    keys_to_remove = [
        key for key in d 
        if not key or (isinstance(key, str) and key.strip() == "") or 
        not d[key] or (isinstance(d[key], str) and d[key].strip() == "")
    ]

    for key in keys_to_remove:
        del d[key]

    return d

def generate_container_command(bash_args, 
                                technology:str = "local", 
                                image_path:str = None, 
                                license_path:str = None):
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
            raise ValueError("The image path is required for Singularity containerization.")
    
        # Checking if the arguments are files or directories
        container_cmd = []
        bind_mounts = []
            
        for arg in bash_args: # Checking if the arguments are files or directories
            abs_arg_path = os.path.dirname(arg)
            if os.path.exists(abs_arg_path):
                bind_mounts.append(abs_arg_path) # Adding the argument to the bind mounts

        if bind_mounts: # Adding the bind mounts to the container command
            # Detect only the unique elements in the list bind_mounts
            bind_mounts = list(set(bind_mounts))
            for mount_path in bind_mounts:
                if technology == "singularity": # Using Singularity technology
                    path2mount.extend(['--bind', f'{mount_path}:{mount_path}'])
                    
                elif technology == "docker": # Using Docker technology
                    path2mount.extend(['-v', f'{mount_path}:{mount_path}'])

        # Creating the container command
        if technology == "singularity": # Using Singularity technology
            container_cmd.append('singularity') # singularity command
            container_cmd.append('run')

        # Using Docker technology
        elif technology == "docker":
            container_cmd.append('docker') # docker command
            container_cmd.append('run')
        
        container_cmd = container_cmd + path2mount
        
        container_cmd.append(image_path)
        container_cmd.extend(bash_args)

    else: # No containerization
        container_cmd = bash_args
    

    return container_cmd