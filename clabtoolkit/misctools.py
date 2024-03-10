import numpy as np
from typing import Union
import shlex
import os

# Print iterations progress
def _printprogressbar(
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


def _rgb2hex(r:int, 
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

def _multi_rgb2hex(colors: Union[list, np.ndarray]):
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
    colors = _readjust_colors(colors)

    hexcodes = []
    if isinstance(colors, list):
        for color in colors:
            hexcodes.append(_rgb2hex(color[0], color[1], color[2]))
    
    elif isinstance(colors, np.ndarray):
        nrows, ncols = colors.shape
        for i in np.arange(0, nrows):
            hexcodes.append(_rgb2hex(colors[i, 0], colors[i, 1], colors[i, 2]))

    return hexcodes

def _hex2rgb(hexcode: str):
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

def _multi_hex2rgb(hexcodes: list):
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

    rgb_list = [_hex2rgb(hex_color) for hex_color in hexcodes]
    return np.array(rgb_list)

def _build_indexes(range_vector: list):
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

        elif isinstance(i, int):
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

    # Remove the elements with 0
    indexes = [x for x in indexes if x != 0]

    # Flatten the list and unique the values
    indexes = _remove_duplicates(indexes)

    return indexes

def _remove_duplicates(input_list: list):
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

def _readjust_colors(colors: Union[list, np.ndarray]):
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
        if max(max(colors)) <= 1:
            colors = [color * 255 for color in colors]

    elif isinstance(colors, np.ndarray):
        nrows, ncols = colors.shape

        # If all the values in the array are between 0 and 1, then the values are multiplied by 255
        if np.max(colors) <= 1:
            colors = colors * 255
    
    return colors

def _create_random_colors(n: int):
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

def _correct_names(regnames: list, 
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
            
            for item in replace:

                # Replace the substring item from the region names
                regnames = [name.replace(item[0], item[1]) for name in regnames]

    return regnames


def _my_ismember(a, b):
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

def _generate_container_command(bash_args, technology:str = "local", image_path:str = None):
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

        
    container_cmd = []
    # Creating the container command
    if technology == "singularity": # Using Singularity technology
        container_cmd.append('singularity') # singularity command
        container_cmd.append('run')

        # Checking if the arguments are files or directories
        bind_mounts = []

        for arg in bash_args: # Checking if the arguments are files or directories
            abs_arg_path = os.path.dirname(arg)
            if os.path.exists(abs_arg_path):
                bind_mounts.append(abs_arg_path) # Adding the argument to the bind mounts

        if bind_mounts: # Adding the bind mounts to the container command
            for mount_path in bind_mounts:
                container_cmd.extend(['--bind', f'{mount_path}:{mount_path}'])

        # Adding the container image path and the bash command arguments
        if image_path is not None:
            if not os.path.exists(image_path):
                raise ValueError(f"The container image {image_path} does not exist.")
        else:
            raise ValueError("The image path is required for Singularity containerization.")
        
        container_cmd.append(image_path)
        container_cmd.extend(bash_args)

    # Using Docker technology
    elif technology == "docker":
        container_cmd.append('docker') # docker command
        container_cmd.append('run')
        
        for arg in bash_args: # Checking if the arguments are files or directories
            abs_arg_path = os.path.dirname(arg)
            if os.path.exists(abs_arg_path):
                bind_mounts.append(abs_arg_path) # Adding the argument to the bind mounts

        if bind_mounts: # Adding the bind mounts to the container command
            for mount_path in bind_mounts:
                container_cmd.extend(['-v', f'{mount_path}:{mount_path}'])

        # Adding the container image path and the bash command arguments
        if image_path is not None:
            if not os.path.exists(image_path):
                raise ValueError(f"The container image {image_path} does not exist.")
        else:
            raise ValueError("The image path is required for Docker containerization.")
        
        container_cmd.append(image_path)
        container_cmd.extend(bash_args)

    else: # No containerization
        container_cmd = bash_args
    

    return container_cmd