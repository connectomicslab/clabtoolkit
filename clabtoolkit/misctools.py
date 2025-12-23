import numpy as np
import h5py
import uuid

from typing import Union, Dict, List, Tuple, Set, Any, Optional, Literal


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
import importlib
from IPython.display import HTML, display
from IPython import get_ipython

from pathlib import Path
from colorama import init, Fore, Style, Back

init(autoreset=True)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_hex
from matplotlib.colors import is_color_like as mpl_is_color_like
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import textwrap

from typing import Union, List, Optional

from . import colorstools as cltcolors

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

    Examples
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
def parse_condition(condition: str) -> Tuple[Optional[str], List[str]]:
    """
    Parse a condition string to extract the main variable and limit variables.

    Parameters:
    ----------
    condition : str
        Condition string to parse (e.g., "bmin <= bvals <= bmax").

    Returns:
    -------
    main_variable : str or None
        The main variable in the condition (e.g., "bvals").
        Returns None if no valid variable is found.
    limit_variables : list of str
        List of limit variables (e.g., ["bmin", "bmax"]).
        Returns an empty list if no limits are found.
    Examples:
    --------
    >>> condition = "bmin <= bvals <= bmax"
    >>> var, limits = parse_condition(condition)
    >>> print(var)     # Output: "bvals"
    >>> print(limits)  # Output: ["bmin", "bmax"]

    >>> condition = "bvals > 1000"
    >>> var, limits = parse_condition(condition)
    >>> print(var)     # Output: "bvals"
    >>> print(limits)  # Output: ["1000"]

    >>> condition = "1000 < bvals <= 2000"
    >>> var, limits = parse_condition(condition)
    >>> print(var)     # Output: "bvals"
    >>> print(limits)  # Output: ["1000", "2000"]

    >>> condition = "bvals != bval"
    >>> var, limits = parse_condition(condition)
    >>> print(var)     # Output: "bvals"
    >>> print(limits)  # Output: ["bval"]

    >>> condition = "invalid condition"
    >>> var, limits = parse_condition(condition)
    >>> print(var)     # Output: None
    >>> print(limits)  # Output: []

    """
    # Remove spaces for easier parsing
    condition = condition.strip()

    # Define comparison operators (order matters - longer operators first)
    operators = ["<=", ">=", "==", "!=", "<", ">"]

    # Pattern to match chained comparison: limit1 op1 var op2 limit2
    chained_pattern = r"(\w+)\s*(<=|>=|<|>|==|!=)\s*(\w+)\s*(<=|>=|<|>|==|!=)\s*(\w+)"

    # Check for chained comparison first
    chained_match = re.match(chained_pattern, condition)
    if chained_match:
        limit1, op1, var, op2, limit2 = chained_match.groups()
        return var, [limit1, limit2]

    # Pattern for simple comparison: var op limit or limit op var
    simple_pattern = r"(\w+)\s*(<=|>=|<|>|==|!=)\s*(\w+)"
    simple_match = re.match(simple_pattern, condition)

    if simple_match:
        left, op, right = simple_match.groups()

        # Determine which is the main variable and which is the limit
        # Heuristic: assume the variable that appears in multiple conditions
        # or follows common naming patterns is the main variable

        # For now, we'll use a simple heuristic:
        # - If one side is a number, the other is the variable
        # - If both are identifiers, assume the first one is the variable
        # - You can customize this logic based on your specific needs

        if is_numeric(right):
            return left, [right]
        elif is_numeric(left):
            return right, [left]
        else:
            # Both are identifiers - use heuristic or return first as variable
            # You might want to customize this based on your naming conventions
            return left, [right]

    return None, []


def is_numeric(s: str) -> bool:
    """Check if a string represents a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def analyze_condition(condition: str) -> dict:
    """
    Analyze a condition string and return detailed information.

    Parameters:
    ----------
    condition : str
        Condition string to analyze (e.g., "bmin <= bvals <= bmax").

    Returns:
    -------
    info : dict
        Dictionary containing:
            - 'original_condition': Original condition string.
            - 'main_variable': The main variable in the condition.
            - 'limit_variables': List of limit variables.
            - 'operators': List of operators found in the condition.
            - 'is_chained': Boolean indicating if the condition is chained.
            - 'is_valid': Boolean indicating if the condition was parsed successfully.

    Examples:
    --------
    >>> condition = "bmin <= bvals <= bmax"
    >>> info = analyze_condition(condition)
    >>> print(info)
    {'original_condition': 'bmin <= bvals <= bmax',
        'main_variable': bvals'
        'limit_variables': ['bmin', 'bmax'],
        'operators': ['<=', '<='],
        'is_chained': True,
        'is_valid': True}

    """
    variable, limits = parse_condition(condition)

    # Extract operators for additional info
    operators_found = []
    for op in ["<=", ">=", "==", "!=", "<", ">"]:
        if op in condition:
            operators_found.append(op)

    return {
        "original_condition": condition,
        "main_variable": variable,
        "limit_variables": limits,
        "operators": operators_found,
        "is_chained": len(limits) > 1,
        "is_valid": variable is not None,
    }


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

    Examples
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

    Examples
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

    Examples
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

    Examples
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


#################################################################################################
def to_list(item):
    """
    Convert single items to lists for consistent handling.

    Parameters
    ----------
    item : any
        A single item or a list of items.

    Returns
    -------
    list
        A list containing the item(s).

    Examples
    --------
    >>> to_list(5)
    [5]
    >>> to_list([1, 2, 3])
    [1, 2, 3]
    >>> to_list("hello")
    ['hello']
    >>> to_list(["a", "b"])
    ['a', 'b']
    """
    if not isinstance(item, list):
        return [item]
    return item


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

    Examples
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

    Examples
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

    Parameters.
    ----------
    root_dir :str
        The path to the root directory where the search will be performed.

    Returns
    -------
    leaf_folders: list
        A list of absolute paths to folders that do not contain any subfolders.

    Examples
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

    Examples
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
    Rename folders based on specified string replacements. Handles nested directories and avoids conflicts.

    Parameters
    ----------
    folder_paths : List[str]
        List of folder paths to be renamed.

    replacements : Dict[str, str]
        Dictionary mapping substrings to be replaced (keys) with their replacements (values).

    bool_case : bool
        If True, replacements are case-sensitive; if False, they are case-insensitive.

    Returns
    -------
    List[Tuple[str, str]]
        List of tuples containing (old_path, new_path) for each successfully renamed folder.

    Examples
    --------------
        >>> folder_paths = ["/data/sub-01/session1", "/data/sub-02/session2"]
        >>> replacements = {"sub-": "subject-", "session": "sess"}
        >>> renamed = rename_folders(folder_paths, replacements, bool_case=True)
        >>> print(renamed)
        [('/data/sub-01/session1', '/data/subject-01/sess1'),
        ('/data/sub-02/session2', '/data/subject-02/sess2')]
    """

    def apply_replacements(path: str) -> str:
        """Apply all replacements to a path string."""
        new_path = path
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

    # Find unique directories that need renaming
    dirs_to_process = set()

    for folder_path in folder_paths:
        # Add this path and all its parents
        current = folder_path
        while current and current != "/":
            dirs_to_process.add(current)
            current = os.path.dirname(current)

    # Find which directories actually need renaming
    rename_operations = []
    for old_path in dirs_to_process:
        new_path = apply_replacements(old_path)
        if old_path != new_path and os.path.exists(old_path):
            rename_operations.append((old_path, new_path))

    # Sort by depth (shallowest first)
    rename_operations.sort(key=lambda x: x[0].count("/"))

    # Remove any child directories of directories we're already renaming
    filtered_operations = []
    for old_path, new_path in rename_operations:
        # Check if this is a child of something we're already renaming
        is_child = False
        for parent_old, parent_new in filtered_operations:
            if old_path.startswith(parent_old + "/"):
                is_child = True
                break

        if not is_child:
            filtered_operations.append((old_path, new_path))

    # Execute the renames
    successfully_renamed = []
    for old_path, new_path in filtered_operations:
        try:
            # Create parent directory if needed
            parent_dir = os.path.dirname(new_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            # Check if target exists
            if os.path.exists(new_path):
                print(f"Warning: Target already exists, skipping: {new_path}")
                continue

            # Rename
            os.rename(old_path, new_path)
            successfully_renamed.append((old_path, new_path))
            print(f"Renamed: {old_path} -> {new_path}")

        except OSError as e:
            print(f"Error renaming {old_path}: {e}")

    return successfully_renamed


####################################################################################################
def remove_empty_folders(start_path, deleted_folders=None):
    """
    Recursively removes empty directories starting from start_path.
    Returns a list of all directories that were deleted.

    Parameters
    ----------
        start_path : str
            The directory path to start searching from

        deleted_folders : list
            A list to store the paths of deleted directories. If None, a new list will be created.

    Returns
    -------
        deleted_folders : list
            A list of all directories that were deleted.

    Examples
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


#########################################################################################################
def create_temporary_filename(
    tmp_dir: str = "/tmp", prefix: str = "tmp", extension: str = ".nii.gz"
) -> str:
    """
    Create a temporary filename with a unique identifier.

    Parameters
    ----------
    tmp_dir : str
        The directory where the temporary file will be created. Default is "/tmp".

    prefix : str
        The prefix for the temporary filename. Default is "tmp".

    extension : str
        The file extension for the temporary file. Default is ".nii.gz".

    Returns
    -------
    str
        A unique temporary filename with the specified prefix and extension.

    Examples
    --------
    >>> tmp_filename = create_temporary_filename()
    >>> print(tmp_filename)  # Output: /tmp/tmp_<unique_id>.nii.gz
    """

    # Validate the temporary directory
    if not os.path.isdir(tmp_dir):
        raise ValueError(f"The specified temporary directory does not exist: {tmp_dir}")

    if not os.access(tmp_dir, os.W_OK):
        raise ValueError(
            f"The specified temporary directory is not writable: {tmp_dir}"
        )

    # Generate a unique identifier
    unique_id = str(uuid.uuid4())

    # Create the temporary filename
    tmp_filename = os.path.join(tmp_dir, f"{prefix}_{unique_id}{extension}")

    # Ensure the filename is unique
    while os.path.exists(tmp_filename):
        unique_id = str(uuid.uuid4())
        tmp_filename = os.path.join(tmp_dir, f"{prefix}{unique_id}{extension}")

    return tmp_filename


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

    Examples
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


#####################################################################################################
def get_real_basename(file_name: str) -> str:
    """
    Extracts the base name of a file without its extension.

    Parameters
    ----------
    file_name : str
        The full path to the file.

    Returns
    -------
    str
        The base name of the file without its extension.

    Examples
    --------
    >>> get_real_basename("/path/to/image.nii.gz")
    'image'
    >>> get_real_basename("image.jpg")
    'image'
    """

    file_basename = os.path.basename(file_name)

    # Remove the file extension
    if file_basename.endswith(".nii.gz"):
        file_basename = file_basename[:-7]
    else:
        # Get the string until the last dot
        file_basename = file_basename.rsplit(".", 1)[0]

    return file_basename


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############    Section 8: Methods dedicated to work with dictionaries and dataframes   ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def load_json(json_file_path: Union[str, Path]) -> dict:
    """
    Loads a JSON file and returns its contents as a Python dictionary.

    Parameters
    ----------
    json_file_path : Union[str, Path]
        The path to the JSON file to be loaded.

    Returns
    -------
    dict
        A dictionary containing the data from the JSON file.

    Examples
    --------
    >>> data_dict = load_json_to_dictionary("data.json")
    >>> print(data_dict)  # Output: Contents of the JSON file as a dictionary
    """

    # Check if the file path is valid
    if not isinstance(json_file_path, (str, Path)):
        raise ValueError("The file path must be a string or a Path object.")

    if isinstance(json_file_path, str):
        json_file_path = Path(json_file_path)

    if not json_file_path.exists():
        raise FileNotFoundError(f"The file {json_file_path} does not exist.")

    ##
    try:
        with open(json_file_path, "r") as json_file:
            data_dictionary = json.load(json_file)

        return data_dictionary

    except Exception as e:
        raise ValueError(f"An error occurred while loading the JSON file: {e}")


#####################################################################################################
def remove_empty_keys_or_values(d: dict) -> dict:
    """
    Remove dictionary entries with empty keys, keys with only spaces, or empty values.

    Parameters
    ----------

    d : dict
        The dictionary to remove entries from.

    Returns
    --------

    d : dict
        The dictionary with the empty entries removed.

    Examples
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

    import clabtoolkit.bidstools as cltbids

    if possible_seps is None:
        possible_seps = [",", "\t", ";", "|"]

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist")

    # Try pandas built-in separator auto-detection
    try:

        # Read the table, take the column names, look on the BIDs columns names and then, read those columns as str)
        tmp_bids = cltbids.entities4table()
        bids_entities = list(tmp_bids.values())
        df_tmp = pd.read_csv(file_path, sep=None, engine="python", **kwargs)
        cols = df_tmp.columns.tolist()

        if any(col in bids_entities for col in cols):
            kwargs["dtype"] = {col: str for col in cols if col in bids_entities}

        df = pd.read_csv(
            file_path, sep=None, dtype={"Run": str}, engine="python", **kwargs
        )

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

    Parameters
    -----------
        data_dict: A nested dictionary to search through
        only_last_key: If True, uses only the leaf key name; if False, uses the full path

    Returns
    --------
        A dictionary where keys are either leaf keys or paths to string values,
        and values are the corresponding strings

    Examples
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


#####################################################################################################
def print_dict_tree(data, indent="", is_last=True, max_str_len=50, max_items=None):
    """
    Print a dictionary as a tree structure.

    Parameters:
    -----------
    data : dict, list, or any
        The data structure to visualize

    indent : str
        Current indentation (used internally for recursion)

    is_last : bool
        Whether this is the last item at current level (used internally)

    max_str_len : int
        Maximum length for string values before truncation

    max_items : int or None
        Maximum number of items to show in lists/dicts (None for all)

    Example:
    --------
    >>> data = {
    ...     'project': 'neuroimaging',
    ...     'metadata': {
    ...         'subjects': 150,
    ...         'scans': ['T1', 'T2', 'fMRI']
    ...     }
    ... }
    >>> print_dict_tree(data)
    """

    if isinstance(data, dict):
        items = list(data.items())
        if max_items and len(items) > max_items:
            items = items[:max_items]
            truncated = True
        else:
            truncated = False

        for i, (key, value) in enumerate(items):
            is_last_item = (i == len(items) - 1) and not truncated

            # Print the key
            connector = "└── " if is_last_item else "├── "
            print(f"{indent}{connector}{key}: ", end="")

            # Handle the value
            if isinstance(value, (dict, list)):
                print(f"({type(value).__name__}, {len(value)} items)")
                extension = "    " if is_last_item else "│   "
                print_dict_tree(
                    value, indent + extension, is_last_item, max_str_len, max_items
                )
            else:
                value_str = str(value)
                if len(value_str) > max_str_len:
                    value_str = value_str[:max_str_len] + "..."
                print(f"{value_str}")

        if truncated:
            connector = "└── " if is_last else "├── "
            print(f"{indent}{connector}... ({len(data) - max_items} more items)")

    elif isinstance(data, list):
        items = data
        if max_items and len(items) > max_items:
            items = items[:max_items]
            truncated = True
        else:
            truncated = False

        for i, item in enumerate(items):
            is_last_item = (i == len(items) - 1) and not truncated
            connector = "└── " if is_last_item else "├── "

            if isinstance(item, (dict, list)):
                print(
                    f"{indent}{connector}[{i}]: ({type(item).__name__}, {len(item)} items)"
                )
                extension = "    " if is_last_item else "│   "
                print_dict_tree(
                    item, indent + extension, is_last_item, max_str_len, max_items
                )
            else:
                item_str = str(item)
                if len(item_str) > max_str_len:
                    item_str = item_str[:max_str_len] + "..."
                print(f"{indent}{connector}[{i}]: {item_str}")

        if truncated:
            connector = "└── " if is_last else "├── "
            print(f"{indent}{connector}... ({len(data) - max_items} more items)")

    else:
        # For non-dict, non-list items at the root level
        value_str = str(data)
        if len(value_str) > max_str_len:
            value_str = value_str[:max_str_len] + "..."
        print(value_str)


#####################################################################################################
def explore_dictionary(data, max_depth=None, current_depth=0):
    """
    Get a simplified structure of a nested dictionary showing only keys and types.

    Parameters:
    -----------
    data : dict, list, or any
        The data structure to analyze

    max_depth : int or None
        Maximum depth to explore (None for unlimited)

    current_depth : int
        Current depth level (used internally)

    Returns:
    --------
    dict or str
        Simplified structure representation

    Example:
    --------
    >>> structure = get_dict_structure(data, max_depth=2)
    >>> print_dict_tree(structure)
    """
    if max_depth is not None and current_depth >= max_depth:
        return f"<{type(data).__name__}>"

    if isinstance(data, dict):
        return {
            key: explore_dictionary(value, max_depth, current_depth + 1)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        if len(data) == 0:
            return "[]"
        # Show structure of first item as representative
        return [explore_dictionary(data[0], max_depth, current_depth + 1)]
    else:
        return f"<{type(data).__name__}>"


#####################################################################################################
def get_dict_info(data):
    """
    Print summary information about a dictionary structure.

    Parameters:
    -----------
    data : dict, list, or any
        The data structure to summarize
    """

    def count_items(obj, counts=None):
        if counts is None:
            counts = {"dicts": 0, "lists": 0, "total_keys": 0, "max_depth": 0}

        def recurse(o, depth=0):
            counts["max_depth"] = max(counts["max_depth"], depth)

            if isinstance(o, dict):
                counts["dicts"] += 1
                counts["total_keys"] += len(o)
                for v in o.values():
                    recurse(v, depth + 1)
            elif isinstance(o, list):
                counts["lists"] += 1
                for item in o:
                    recurse(item, depth + 1)

        recurse(obj)
        return counts

    info = count_items(data)
    print(f"Dictionary Summary:")
    print(f"  Type: {type(data).__name__}")
    if isinstance(data, dict):
        print(f"  Top-level keys: {len(data)}")
        print(f"  Total nested dicts: {info['dicts']}")
    print(f"  Total nested lists: {info['lists']}")
    print(f"  Total keys (all levels): {info['total_keys']}")
    print(f"  Maximum depth: {info['max_depth']}")
    print()


####################################################################################################
def update_dict(orig_dict, new_dict, merge_lists=False, allow_new_keys=False):
    """
    Deep update dictionary with type safety and optional new key support.

    Provides type-safe updates for existing keys while optionally allowing
    new keys to be added. Type checking ensures that existing values can
    only be updated with values of the same type.

    Parameters
    ----------
    orig_dict : dict
        The original dictionary to be updated. This dictionary is modified
        in-place.

    new_dict : dict
        Dictionary containing the updates to apply.

    merge_lists : bool, optional
        If True, lists are extended rather than replaced. Default is False.

    allow_new_keys : bool, optional
        If True, new keys from new_dict are added to orig_dict.
        If False, only existing keys can be updated. Default is False.

    Returns
    -------
    dict
        The updated original dictionary.

    Raises
    ------
    TypeError
        If an update value has a different type than the original value
        for existing keys.

    KeyError
        If update_dict contains new keys and allow_new_keys is False.

    Examples
    --------
    >>> # With allow_new_keys=True
    >>> original = {'name': 'John', 'items': [1, 2]}
    >>> updates = {'name': 'Jane', 'age': 30, 'items': [3, 4]}
    >>> deep_update_flexible(original, updates, merge_lists=True, allow_new_keys=True)
    {'name': 'Jane', 'items': [1, 2, 3, 4], 'age': 30}
    """
    for key, update_value in new_dict.items():
        if key in orig_dict:
            orig_value = orig_dict[key]

            # Check type compatibility for existing keys
            if type(orig_value) != type(update_value):
                raise TypeError(
                    f"Type mismatch for key '{key}': "
                    f"expected {type(orig_value).__name__}, "
                    f"got {type(update_value).__name__}"
                )

            # Handle different types
            if isinstance(orig_value, dict):
                update_dict(orig_value, update_value, merge_lists, allow_new_keys)

            elif isinstance(orig_value, list):
                if merge_lists:
                    orig_dict[key].extend(update_value)
                else:
                    orig_dict[key] = update_value
            else:
                orig_dict[key] = update_value
        else:
            # Handle new keys
            if allow_new_keys:
                orig_dict[key] = update_value
            else:
                raise KeyError(f"Key '{key}' not found in original dictionary")

    return orig_dict


####################################################################################################
def expand_and_concatenate(df_add: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands df_add to match the number of rows in df and concatenates them along columns.

    Parameters
    -----------
        df_add : pd.DataFrame
            DataFrame with a single row to be replicated.

        df : pd.DataFrame
            DataFrame to which df_add will be concatenated.

    Returns
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

    Examples
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
def is_notebook():
    """
    Check if code is running in a Jupyter notebook environment.

    Returns
    -------
    bool
        True if running in Jupyter notebook, False if in terminal or other environment

    Notes
    -----
    Uses IPython's get_ipython() to detect the shell type:
    - 'ZMQInteractiveShell' indicates Jupyter notebook or qtconsole
    - 'TerminalInteractiveShell' indicates IPython terminal
    - Other types or exceptions indicate standard Python interpreter
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except (NameError, AttributeError):
        return False  # Probably standard Python interpreter


#####################################################################################################
def format_signature(sig: inspect.Signature, notebook_mode=False):
    """
    Format a function signature with colors appropriate for the environment.

    Parameters
    ----------
    sig : inspect.Signature
        The function signature object to format
    notebook_mode : bool, optional
        If True, format with HTML styling for notebooks.
        If False, format with ANSI color codes for terminal, by default False

    Returns
    -------
    str
        Formatted signature string with appropriate styling

    Examples
    --------
    >>> import inspect
    >>> def example_func(name: str, age: int = 25): pass
    >>> sig = inspect.signature(example_func)
    >>> format_signature(sig, notebook_mode=False)
    '(name: str, age = 25)'  # With ANSI colors in terminal

    Notes
    -----
    - Parameter names are colored in cyan/blue
    - Type annotations are colored in purple
    - Default values are colored in gray
    - Automatically handles different annotation types and missing defaults
    """
    if notebook_mode:
        return _format_signature_html(sig)
    else:
        return _format_signature_ansi(sig)


#####################################################################################################
def _format_signature_ansi(sig: inspect.Signature):
    """
    Format a function signature with ANSI color codes for terminal display.

    Parameters
    ----------
    sig : inspect.Signature
        The function signature object to format

    Returns
    -------
    str
        Signature string with ANSI color codes using bcolors class

    Notes
    -----
    Uses the bcolors class for consistent terminal coloring:
    - DARKCYAN for parameter names
    - OKPURPLE for type annotations
    - OKGRAY for default values
    - OKWHITE for punctuation (parentheses, commas)
    """
    parts = [f"{cltcolors.bcolors.OKWHITE}({cltcolors.bcolors.ENDC}"]
    params = list(sig.parameters.values())
    for i, p in enumerate(params):
        param_str = f"{cltcolors.bcolors.DARKCYAN}{p.name}{cltcolors.bcolors.ENDC}"
        if p.annotation != inspect.Parameter.empty:
            annotation = (
                p.annotation.__name__
                if hasattr(p.annotation, "__name__")
                else str(p.annotation)
            )
            param_str += (
                f": {cltcolors.bcolors.OKPURPLE}{annotation}{cltcolors.bcolors.ENDC}"
            )
        if p.default != inspect.Parameter.empty:
            param_str += f"{cltcolors.bcolors.OKGRAY} = {repr(p.default)}{cltcolors.bcolors.ENDC}"
        parts.append(param_str)
        if i < len(params) - 1:
            parts.append(f"{cltcolors.bcolors.OKWHITE}, {cltcolors.bcolors.ENDC}")
    parts.append(f"{cltcolors.bcolors.OKWHITE}){cltcolors.bcolors.ENDC}")
    return "".join(parts)


######################################################################################################
def _format_signature_html(sig: inspect.Signature):
    """
    Format a function signature with HTML styling for Jupyter notebook display.

    Parameters
    ----------
    sig : inspect.Signature
        The function signature object to format

    Returns
    -------
    str
        Signature string with inline HTML styling

    Notes
    -----
    Uses inline CSS styles for notebook compatibility:
    - Cyan blue (#36a3d9) for parameter names
    - Purple (#9d4edd) for type annotations
    - Gray (#95a5a6) for default values
    - Light gray (#97a3b3) for punctuation
    """
    parts = ['<span style="color: #97a3b3;">(</span>']
    params = list(sig.parameters.values())
    for i, p in enumerate(params):
        param_str = f'<span style="color: #36a3d9;">{p.name}</span>'
        if p.annotation != inspect.Parameter.empty:
            annotation = (
                p.annotation.__name__
                if hasattr(p.annotation, "__name__")
                else str(p.annotation)
            )
            param_str += f': <span style="color: #9d4edd;">{annotation}</span>'
        if p.default != inspect.Parameter.empty:
            param_str += f'<span style="color: #95a5a6;"> = {repr(p.default)}</span>'
        parts.append(param_str)
        if i < len(params) - 1:
            parts.append('<span style="color: #97a3b3;">, </span>')
    parts.append('<span style="color: #97a3b3;">)</span>')
    return "".join(parts)


######################################################################################################
def show_module_contents(module, show_private=False, show_inherited=True):
    """
    Display all classes and functions in a given module with colored formatting.

    Works in both Jupyter notebooks and terminal environments, automatically
    detecting the environment and using appropriate styling (ANSI colors for
    terminal, HTML for notebooks).

    Parameters
    ----------
    module : module object or str
        A module object or module name (str) to inspect
    show_private : bool, optional
        Whether to show private members (names starting with _), by default False
    show_inherited : bool, optional
        Whether to show inherited methods in classes, by default True

    Returns
    -------
    None
        Displays the module contents directly (prints to terminal or renders HTML)

    Examples
    --------
    >>> show_module_contents('json')
    📦 Contents of module 'json':
    ...

    >>> import math
    >>> show_module_contents(math, show_private=True)
    📦 Contents of module 'math':
    ...

    >>> show_module_contents('pathlib', show_inherited=False)
    📦 Contents of module 'pathlib':
    ...

    Notes
    -----
    - Automatically detects Jupyter notebook vs terminal environment
    - Shows function signatures with type annotations and default values
    - Displays class methods and docstrings
    - Only shows members defined in the target module (not imported)
    """
    notebook_mode = is_notebook()

    # Handle module input
    if isinstance(module, str):
        module_name = module
        try:
            module = sys.modules.get(module)
            if module is None:
                module = importlib.import_module(module_name)
        except ImportError as e:
            error_msg = f"Module '{module_name}' could not be imported: {e}"
            if notebook_mode:
                display(
                    HTML(
                        f'<span style="color: #e74c3c; font-weight: bold;">{error_msg}</span>'
                    )
                )
            else:
                print(f"{cltcolors.bcolors.FAIL}{error_msg}{cltcolors.bcolors.ENDC}")
            return
        except Exception as e:
            error_msg = f"Error importing module '{module_name}': {e}"
            if notebook_mode:
                display(
                    HTML(
                        f'<span style="color: #e74c3c; font-weight: bold;">{error_msg}</span>'
                    )
                )
            else:
                print(f"{cltcolors.bcolors.FAIL}{error_msg}{cltcolors.bcolors.ENDC}")
            return
    elif not isinstance(module, types.ModuleType):
        error_msg = "Invalid input: must be a module object or module name string."
        if notebook_mode:
            display(
                HTML(
                    f'<span style="color: #e74c3c; font-weight: bold;">{error_msg}</span>'
                )
            )
        else:
            print(f"{cltcolors.bcolors.FAIL}{error_msg}{cltcolors.bcolors.ENDC}")
        return

    # Helper function to filter private members
    def should_show(name):
        return show_private or not name.startswith("_")

    # Get all members
    all_classes = []
    all_functions = []

    for name in sorted(dir(module)):
        if not should_show(name):
            continue

        try:
            obj = getattr(module, name)
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                all_classes.append((name, obj))
            elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
                all_functions.append((name, obj))
        except Exception:
            continue

    # Build output based on environment
    if notebook_mode:
        _display_notebook_output(module, all_classes, all_functions, show_inherited)
    else:
        _display_terminal_output(module, all_classes, all_functions, show_inherited)


##
def _display_notebook_output(module, all_classes, all_functions, show_inherited=True):
    """
    Display formatted module contents for Jupyter notebooks using HTML.

    Parameters
    ----------
    module : module object
        The module whose contents to display
    all_classes : list of tuple
        List of (name, class_object) tuples for classes in the module
    all_functions : list of tuple
        List of (name, function_object) tuples for functions in the module
    show_inherited : bool, optional
        Whether to show inherited methods in classes, by default True

    Notes
    -----
    Uses IPython.display.HTML to render formatted content with:
    - Styled headers and sections
    - Color-coded class and function names
    - Formatted signatures with type annotations
    - Docstring previews
    - Clean visual separators
    """
    html = f"""
    <div style="font-family: 'Courier New', monospace; line-height: 1.6;">
        <h3 style="color: #9d4edd; margin-bottom: 5px;">📦 Contents of module '{module.__name__}'</h3>
    """

    if hasattr(module, "__file__") and module.__file__:
        html += f'<p style="color: #95a5a6; font-size: 0.9em; margin: 0;">Path: {module.__file__}</p>'

    html += "<br>"

    # Classes section
    if all_classes:
        html += f'<h4 style="color: #3498db; margin-bottom: 10px;">📘 Classes ({len(all_classes)}):</h4>'

        for name, cls in all_classes:
            html += f'<div style="margin-left: 20px; margin-bottom: 15px;">'
            html += f'<strong style="color: #3498db;">{name}</strong><br>'

            # Class docstring
            doc = inspect.getdoc(cls)
            if doc:
                first_line = doc.split("\n")[0]
                html += f'<span style="color: #95a5a6; font-style: italic;">    # {first_line}</span><br>'

            # Methods
            methods = []
            for method_name, method in inspect.getmembers(
                cls, predicate=inspect.ismethod
            ):
                if show_inherited or method.__qualname__.startswith(cls.__name__ + "."):
                    if not method_name.startswith("_"):
                        methods.append((method_name, method))

            for method_name, method in inspect.getmembers(
                cls, predicate=inspect.isfunction
            ):
                if show_inherited or method.__qualname__.startswith(cls.__name__ + "."):
                    if not method_name.startswith("_"):
                        methods.append((method_name, method))

            if methods:
                for method_name, method in sorted(methods):
                    try:
                        sig = inspect.signature(method)
                        formatted_sig = format_signature(sig, notebook_mode=True)
                        html += f'<span style="color: #f39c12; margin-left: 20px;">• {method_name}</span>{formatted_sig}<br>'

                        method_doc = inspect.getdoc(method)
                        if method_doc:
                            first_line = method_doc.split("\n")[0]
                            html += f'<span style="color: #95a5a6; font-style: italic; margin-left: 40px;">      # {first_line}</span><br>'
                    except Exception:
                        html += f'<span style="color: #f39c12; margin-left: 20px;">• {method_name}</span> (signature unavailable)<br>'

            html += '<div style="border-bottom: 1px solid #ecf0f1; margin: 10px 0; width: 60%;"></div>'
            html += "</div>"

    # Functions section
    if all_functions:
        html += f'<h4 style="color: #27ae60; margin-bottom: 10px;">🔧 Functions ({len(all_functions)}):</h4>'

        for name, func in all_functions:
            try:
                sig = inspect.signature(func)
                formatted_sig = format_signature(sig, notebook_mode=True)
                html += f'<div style="margin-left: 20px; margin-bottom: 8px;">'
                html += f'<span style="color: #27ae60; font-weight: bold;">{name}</span>{formatted_sig}<br>'

                doc = inspect.getdoc(func)
                if doc:
                    first_line = doc.split("\n")[0]
                    html += f'<span style="color: #95a5a6; font-style: italic; margin-left: 20px;">    # {first_line}</span>'
                html += "</div>"
            except Exception:
                html += f'<div style="margin-left: 20px;"><span style="color: #27ae60;">{name}</span> (signature unavailable)</div>'

    # Summary
    total_items = len(all_classes) + len(all_functions)
    if total_items == 0:
        html += '<p style="color: #f39c12;">No public classes or functions found in this module.</p>'
    else:
        html += f'<p style="color: #2c3e50; font-weight: bold; margin-top: 20px;">Total: {len(all_classes)} classes, {len(all_functions)} functions</p>'

    html += "</div>"
    display(HTML(html))


#######################################################################################################
def _display_terminal_output(module, all_classes, all_functions, show_inherited=True):
    """
    Display formatted module contents for terminal using ANSI color codes.

    Parameters
    ----------
    module : module object
        The module whose contents to display
    all_classes : list of tuple
        List of (name, class_object) tuples for classes in the module
    all_functions : list of tuple
        List of (name, function_object) tuples for functions in the module
    show_inherited : bool, optional
        Whether to show inherited methods in classes, by default True

    Notes
    -----
    Uses bcolors class for consistent terminal coloring:
    - Headers in bold with color highlighting
    - Class names in blue, function names in yellow
    - Method signatures with color-coded parameters
    - Gray italic text for docstring previews
    - Visual separators using Unicode characters
    """
    print(
        f"{cltcolors.bcolors.HEADER}{cltcolors.bcolors.BOLD}📦 Contents of module '{module.__name__}':{cltcolors.bcolors.ENDC}\n"
    )

    if hasattr(module, "__file__") and module.__file__:
        print(
            f"{cltcolors.bcolors.OKGRAY}   Path: {module.__file__}{cltcolors.bcolors.ENDC}\n"
        )

    # Classes section
    if all_classes:
        print(
            f"{cltcolors.bcolors.OKBLUE}{cltcolors.bcolors.BOLD}📘 Classes ({len(all_classes)}):{cltcolors.bcolors.ENDC}"
        )

        for name, cls in all_classes:
            print(
                f"  {cltcolors.bcolors.OKBLUE}{cltcolors.bcolors.BOLD}{name}{cltcolors.bcolors.ENDC}"
            )

            # Class docstring
            doc = inspect.getdoc(cls)
            if doc:
                first_line = doc.split("\n")[0]
                print(
                    f"    {cltcolors.bcolors.OKGRAY}# {first_line}{cltcolors.bcolors.ENDC}"
                )

            # Methods
            methods = []
            for method_name, method in inspect.getmembers(
                cls, predicate=inspect.ismethod
            ):
                if show_inherited or method.__qualname__.startswith(cls.__name__ + "."):
                    if not method_name.startswith("_"):
                        methods.append((method_name, method))

            for method_name, method in inspect.getmembers(
                cls, predicate=inspect.isfunction
            ):
                if show_inherited or method.__qualname__.startswith(cls.__name__ + "."):
                    if not method_name.startswith("_"):
                        methods.append((method_name, method))

            if methods:
                for method_name, method in sorted(methods):
                    try:
                        sig = inspect.signature(method)
                        formatted_sig = format_signature(sig, notebook_mode=False)
                        print(
                            f"    {cltcolors.bcolors.OKYELLOW}• {method_name}{cltcolors.bcolors.ENDC}{formatted_sig}"
                        )

                        method_doc = inspect.getdoc(method)
                        if method_doc:
                            first_line = method_doc.split("\n")[0]
                            print(
                                f"      {cltcolors.bcolors.OKGRAY}# {first_line}{cltcolors.bcolors.ENDC}"
                            )
                    except Exception:
                        print(
                            f"    {cltcolors.bcolors.OKYELLOW}• {method_name}{cltcolors.bcolors.ENDC} (signature unavailable)"
                        )

            print(
                f"    {cltcolors.bcolors.OKWHITE}{'─' * 60}{cltcolors.bcolors.ENDC}\n"
            )

    # Functions section
    if all_functions:
        print(
            f"\n{cltcolors.bcolors.OKGREEN}{cltcolors.bcolors.BOLD}🔧 Functions ({len(all_functions)}):{cltcolors.bcolors.ENDC}"
        )

        for name, func in all_functions:
            try:
                sig = inspect.signature(func)
                formatted_sig = format_signature(sig, notebook_mode=False)
                print(
                    f"  {cltcolors.bcolors.OKYELLOW}{name}{cltcolors.bcolors.ENDC}{formatted_sig}"
                )

                doc = inspect.getdoc(func)
                if doc:
                    first_line = doc.split("\n")[0]
                    print(
                        f"    {cltcolors.bcolors.OKGRAY}# {first_line}{cltcolors.bcolors.ENDC}"
                    )
            except Exception:
                print(
                    f"  {cltcolors.bcolors.OKYELLOW}{name}{cltcolors.bcolors.ENDC} (signature unavailable)"
                )
        print()

    # Summary
    total_items = len(all_classes) + len(all_functions)
    if total_items == 0:
        print(
            f"{cltcolors.bcolors.WARNING}No public classes or functions found in this module.{cltcolors.bcolors.ENDC}"
        )
    else:
        print(
            f"{cltcolors.bcolors.OKWHITE}Total: {len(all_classes)} classes, {len(all_functions)} functions{cltcolors.bcolors.ENDC}"
        )


# ####################################################################################################
# def show_module_contents(module):
#     """
#     Displays all classes and functions in a given module with colored formatting.
#     Accepts a module object or module name (str).
#     """
#     if isinstance(module, str):
#         try:
#             module = sys.modules.get(module) or __import__(module)
#         except ImportError:
#             print(
#                 f"{cltcolors.bcolors.FAIL}Module '{module}' could not be imported.{cltcolors.bcolors.ENDC}"
#             )
#             return
#     elif not isinstance(module, types.ModuleType):
#         print(
#             f"{cltcolors.bcolors.FAIL}Invalid input: must be a module object or module name string.{cltcolors.bcolors.ENDC}"
#         )
#         return

#     print(
#         f"{cltcolors.bcolors.HEADER}{cltcolors.bcolors.BOLD}📦 Contents of module '{module.__name__}':{cltcolors.bcolors.ENDC}\n"
#     )

#     # Classes
#     print(f"{cltcolors.bcolors.OKBLUE}{cltcolors.bcolors.BOLD}📘 Classes:{cltcolors.bcolors.ENDC}")
#     for name in sorted(dir(module)):
#         try:
#             obj = getattr(module, name)
#             if inspect.isclass(obj) and obj.__module__ == module.__name__:
#                 print(f"  {cltcolors.bcolors.OKBLUE}- {name}{cltcolors.bcolors.ENDC}")

#                 doc = inspect.getdoc(obj)
#                 if doc:
#                     first_line = doc.split("\n")[0]
#                     print(f"    {cltcolors.bcolors.OKGRAY}# {first_line}{cltcolors.bcolors.ENDC}")

#                 for method_name, method in inspect.getmembers(
#                     obj, predicate=inspect.isfunction
#                 ):
#                     if (
#                         method.__module__ == module.__name__
#                         and method.__qualname__.startswith(obj.__name__ + ".")
#                     ):
#                         sig = inspect.signature(method)
#                         formatted_sig = format_signature(sig)
#                         print(
#                             f"    {cltcolors.bcolors.OKYELLOW}• {method_name}{cltcolors.bcolors.ENDC}{formatted_sig}"
#                         )
#                         method_doc = inspect.getdoc(method)

#                         if method_doc:
#                             first_line = method_doc.split("\n")[0]
#                             print(f"      {cltcolors.bcolors.OKGRAY}# {first_line}{cltcolors.bcolors.ENDC}")

#                 print(f"    {cltcolors.bcolors.OKWHITE}{'─'*60}{cltcolors.bcolors.ENDC}\n")
#         except Exception:
#             continue

#     # Functions
#     print(f"\n{cltcolors.bcolors.OKGREEN}{cltcolors.bcolors.BOLD}🔧 Functions:{cltcolors.bcolors.ENDC}")
#     for name in sorted(dir(module)):
#         try:
#             obj = getattr(module, name)
#             if inspect.isfunction(obj) and obj.__module__ == module.__name__:
#                 sig = inspect.signature(obj)
#                 formatted_sig = format_signature(sig)
#                 print(f"  {cltcolors.bcolors.OKYELLOW}- {name}{cltcolors.bcolors.ENDC}{formatted_sig}")
#                 doc = inspect.getdoc(obj)
#                 if doc:
#                     print(f"    {cltcolors.bcolors.OKGRAY}# {doc.splitlines()[0]}{cltcolors.bcolors.ENDC}")
#         except Exception:
#             continue


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


#####################################################################################################
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


######################################################################################################
def show_object_content(obj, show_private=False, show_dunder=False):
    """
    Inspect and display object properties and methods with colored formatting.

    Provides a comprehensive view of any Python object similar to inspect.help(),
    but with colored formatting for better readability. Works in both Jupyter
    notebooks and terminal environments, automatically detecting the environment
    and using appropriate styling.

    Parameters
    ----------
    obj : object
        The object to inspect (class, instance, function, module, etc.)

    show_private : bool, optional
        Whether to show private methods/attributes (starting with single _),
        by default False

    show_dunder : bool, optional
        Whether to show dunder/magic methods (starting and ending with __),
        by default False

    Returns
    -------
    None
        Displays the inspection information (prints to terminal or renders HTML)

    Examples
    --------
    >>> show_object_content(str)
    ═══════════════════════════════════════════════════════════
    🔍 OBJECT INSPECTOR
    ═══════════════════════════════════════════════════════════
    📦 Object: str
    🏷️  Type: type
    📁 Module: builtins
    ...

    >>> class MyClass:
    ...     def method(self): pass
    >>> show_object_content(MyClass(), show_private=True)
    # Shows private methods and attributes

    >>> import json
    >>> show_object_content(json.loads, show_dunder=True)
    # Shows function details with dunder methods

    Notes
    -----
    - Automatically detects Jupyter notebook vs terminal environment
    - Uses ANSI color codes for terminal, HTML styling for notebooks
    - Categorizes members into methods, properties, and attributes
    - Shows method signatures with color-coded parameters
    - Displays first line of docstrings for quick reference
    - Includes Method Resolution Order (MRO) for class objects
    - Truncates long attribute representations to 50 characters
    - Works with any Python object: classes, instances, functions, modules

    See Also
    --------
    show_module_contents : For inspecting entire modules
    inspect.help : Built-in Python inspection function
    """
    notebook_mode = is_notebook()

    # Get object info
    obj_type = type(obj)
    obj_name = getattr(obj, "__name__", str(obj))
    module_name = getattr(obj_type, "__module__", "unknown")

    # Get all members and categorize them
    members = inspect.getmembers(obj)
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

    # Build output based on environment
    if notebook_mode:
        _display_object_notebook_output(
            obj, obj_type, obj_name, module_name, methods, properties, attributes
        )
    else:
        _display_object_terminal_output(
            obj, obj_type, obj_name, module_name, methods, properties, attributes
        )


########################################################################################################
def _display_object_notebook_output(
    obj, obj_type, obj_name, module_name, methods, properties, attributes
):
    """
    Display formatted object inspection for Jupyter notebooks using HTML.

    Parameters
    ----------
    obj : object
        The object being inspected

    obj_type : type
        The type of the object

    obj_name : str
        Name of the object

    module_name : str
        Module where the object is defined

    methods : list of tuple
        List of (name, method_object) tuples for methods

    properties : list of tuple
        List of (name, property_object) tuples for properties

    attributes : list of tuple
        List of (name, attribute_value) tuples for attributes

    Notes
    -----
    Uses IPython.display.HTML to render formatted content with styled sections
    and color-coded information display.
    """
    html = f"""
    <div style="font-family: 'Courier New', monospace; line-height: 1.6; border: 2px solid #9d4edd; padding: 20px; border-radius: 8px;">
        <h2 style="color: #9d4edd; text-align: center; margin: 0; padding: 10px 0; border-bottom: 2px solid #9d4edd;">🔍 OBJECT INSPECTOR</h2>
        
        <div style="margin: 15px 0;">
            <strong style="color: #36a3d9;">📦 Object:</strong> <span style="color: #2c3e50; font-weight: bold;">{obj_name}</span><br>
            <strong style="color: #36a3d9;">🏷️ Type:</strong> <span style="color: #2c3e50; font-weight: bold;">{obj_type.__name__}</span><br>
            <strong style="color: #36a3d9;">📁 Module:</strong> <span style="color: #2c3e50; font-weight: bold;">{module_name}</span>
        </div>
    """

    # Object docstring
    doc = inspect.getdoc(obj)
    if doc:
        html += f"""
        <div style="margin: 15px 0;">
            <h4 style="color: #27ae60; margin-bottom: 8px;">📝 Description:</h4>
            <p style="color: #95a5a6; font-style: italic; margin-left: 20px; background: #f8f9fa; padding: 10px; border-radius: 4px;">{doc}</p>
        </div>
        """

    # Methods section
    if methods:
        html += f"""
        <div style="margin: 20px 0;">
            <h4 style="color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 5px;">⚙️ METHODS ({len(methods)})</h4>
        """

        for name, method in sorted(methods):
            html += f'<div style="margin: 10px 0; margin-left: 20px;">'

            try:
                sig = inspect.signature(method)
                formatted_sig = format_signature(sig, notebook_mode=True)
                html += f'<strong style="color: #f39c12;">🔧 {name}</strong>{formatted_sig}<br>'

                # Method docstring
                method_doc = inspect.getdoc(method)
                if method_doc:
                    first_line = method_doc.split("\n")[0]
                    html += f'<span style="color: #95a5a6; font-style: italic; margin-left: 20px;">    {first_line}</span>'
            except (ValueError, TypeError):
                html += f'<strong style="color: #f39c12;">🔧 {name}</strong> <span style="color: #95a5a6;">(signature unavailable)</span>'

            html += "</div>"

        html += "</div>"

    # Properties section
    if properties:
        html += f"""
        <div style="margin: 20px 0;">
            <h4 style="color: #9d4edd; border-bottom: 2px solid #9d4edd; padding-bottom: 5px;">🏠 PROPERTIES ({len(properties)})</h4>
        """

        for name, prop in sorted(properties):
            html += f'<div style="margin: 10px 0; margin-left: 20px;">'
            html += f'<strong style="color: #8e44ad;">🔑 {name}</strong><br>'

            # Property docstring
            prop_doc = inspect.getdoc(prop)
            if prop_doc:
                first_line = prop_doc.split("\n")[0]
                html += f'<span style="color: #95a5a6; font-style: italic; margin-left: 20px;">    {first_line}</span>'

            html += "</div>"

        html += "</div>"

    # Attributes section
    if attributes:
        html += f"""
        <div style="margin: 20px 0;">
            <h4 style="color: #1abc9c; border-bottom: 2px solid #1abc9c; padding-bottom: 5px;">📊 ATTRIBUTES ({len(attributes)})</h4>
        """

        for name, attr in sorted(attributes):
            attr_type = type(attr).__name__
            attr_repr = repr(attr)

            # Truncate long representations
            if len(attr_repr) > 50:
                attr_repr = attr_repr[:47] + "..."

            html += f"""
            <div style="margin: 8px 0; margin-left: 20px;">
                <strong style="color: #16a085;">📌 {name}</strong> 
                <span style="color: #95a5a6;">({attr_type})</span>: 
                <span style="color: #2c3e50; background: #f8f9fa; padding: 2px 6px; border-radius: 3px;">{attr_repr}</span>
            </div>
            """

        html += "</div>"

    # MRO (Method Resolution Order) for classes
    if inspect.isclass(obj):
        mro = inspect.getmro(obj)
        if len(mro) > 1:
            html += f"""
            <div style="margin: 20px 0;">
                <h4 style="color: #e67e22; border-bottom: 2px solid #e67e22; padding-bottom: 5px;">🏗️ METHOD RESOLUTION ORDER</h4>
            """

            for i, cls in enumerate(mro):
                html += f"""
                <div style="margin: 5px 0; margin-left: 20px;">
                    <span style="color: #f39c12;">🔗 {i+1}.</span> 
                    <strong style="color: #2c3e50;">{cls.__name__}</strong> 
                    <span style="color: #95a5a6;">({cls.__module__})</span>
                </div>
                """

            html += "</div>"

    html += "</div>"
    display(HTML(html))


#########################################################################################################
def _display_object_terminal_output(
    obj, obj_type, obj_name, module_name, methods, properties, attributes
):
    """
    Display formatted object inspection for terminal using ANSI color codes.

    Parameters
    ----------
    obj : object
        The object being inspected

    obj_type : type
        The type of the object

    obj_name : str
        Name of the object

    module_name : str
        Module where the object is defined

    methods : list of tuple
        List of (name, method_object) tuples for methods

    properties : list of tuple
        List of (name, property_object) tuples for properties

    attributes : list of tuple
        List of (name, attribute_value) tuples for attributes

    Notes
    -----
    Uses bcolors class for consistent ANSI terminal coloring with styled
    headers, colored sections, and formatted information display.
    """
    # Print header
    print(
        f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.HEADER}{'='*60}{cltcolors.bcolors.ENDC}"
    )
    print(
        f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.HEADER}✅ INSPECTION COMPLETE{cltcolors.bcolors.ENDC}"
    )
    print(
        f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.HEADER}{'='*60}{cltcolors.bcolors.ENDC}"
    )
    print(
        f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.HEADER}🔍 OBJECT INSPECTOR{cltcolors.bcolors.ENDC}"
    )
    print(
        f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.HEADER}{'='*60}{cltcolors.bcolors.ENDC}"
    )

    # Object information
    print(
        f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.OKCYAN}📦 Object:{cltcolors.bcolors.ENDC} {cltcolors.bcolors.OKWHITE}{obj_name}{cltcolors.bcolors.ENDC}"
    )
    print(
        f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.OKCYAN}🏷️  Type:{cltcolors.bcolors.ENDC} {cltcolors.bcolors.OKWHITE}{obj_type.__name__}{cltcolors.bcolors.ENDC}"
    )
    print(
        f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.OKCYAN}📁 Module:{cltcolors.bcolors.ENDC} {cltcolors.bcolors.OKWHITE}{module_name}{cltcolors.bcolors.ENDC}"
    )

    # Class docstring
    doc = inspect.getdoc(obj)
    if doc:
        print(
            f"\n{cltcolors.bcolors.BOLD}{cltcolors.bcolors.OKGREEN}📝 Description:{cltcolors.bcolors.ENDC}"
        )
        print(
            f"{cltcolors.bcolors.ITALIC}{cltcolors.bcolors.OKGRAY}{doc}{cltcolors.bcolors.ENDC}"
        )

    # Print Methods
    if methods:
        print(
            f"\n{cltcolors.bcolors.BOLD}{cltcolors.bcolors.OKBLUE}{'─'*15} ⚙️  METHODS {'─'*15}{cltcolors.bcolors.ENDC}"
        )
        for name, method in sorted(methods):
            try:
                sig = inspect.signature(method)
                formatted_sig = format_signature(sig, notebook_mode=False)
                print(
                    f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.OKYELLOW}🔧 {name}{cltcolors.bcolors.ENDC}{formatted_sig}"
                )

                # Method docstring
                method_doc = inspect.getdoc(method)
                if method_doc:
                    # Show first line of docstring
                    first_line = method_doc.split("\n")[0]
                    print(
                        f"    {cltcolors.bcolors.ITALIC}{cltcolors.bcolors.OKGRAY}{first_line}{cltcolors.bcolors.ENDC}"
                    )
            except (ValueError, TypeError):
                print(
                    f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.OKYELLOW}🔧 {name}{cltcolors.bcolors.ENDC}{cltcolors.bcolors.OKGRAY}(signature unavailable){cltcolors.bcolors.ENDC}"
                )
            print()

    # Print Properties
    if properties:
        print(
            f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.OKMAGENTA}{'─'*15} 🏠 PROPERTIES {'─'*12}{cltcolors.bcolors.ENDC}"
        )
        for name, prop in sorted(properties):
            print(
                f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.PURPLE}🔑 {name}{cltcolors.bcolors.ENDC}"
            )

            # Property docstring
            prop_doc = inspect.getdoc(prop)
            if prop_doc:
                first_line = prop_doc.split("\n")[0]
                print(
                    f"    {cltcolors.bcolors.ITALIC}{cltcolors.bcolors.OKGRAY}{first_line}{cltcolors.bcolors.ENDC}"
                )
            print()

    # Print Attributes
    if attributes:
        print(
            f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.DARKCYAN}{'─'*15} 📊 ATTRIBUTES {'─'*12}{cltcolors.bcolors.ENDC}"
        )
        for name, attr in sorted(attributes):
            attr_type = type(attr).__name__
            attr_repr = repr(attr)

            # Truncate long representations
            if len(attr_repr) > 50:
                attr_repr = attr_repr[:47] + "..."

            print(
                f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.OKCYAN}📌 {name}{cltcolors.bcolors.ENDC} "
                f"{cltcolors.bcolors.OKGRAY}({attr_type}){cltcolors.bcolors.ENDC}: "
                f"{cltcolors.bcolors.DARKWHITE}{attr_repr}{cltcolors.bcolors.ENDC}"
            )

    # MRO (Method Resolution Order) for classes
    if inspect.isclass(obj):
        mro = inspect.getmro(obj)
        if len(mro) > 1:
            print(
                f"\n{cltcolors.bcolors.BOLD}{cltcolors.bcolors.WARNING}{'─'*10} 🏗️  METHOD RESOLUTION ORDER {'─'*10}{cltcolors.bcolors.ENDC}"
            )
            for i, cls in enumerate(mro):
                print(
                    f"{cltcolors.bcolors.OKYELLOW}🔗 {i+1}.{cltcolors.bcolors.ENDC} {cltcolors.bcolors.OKWHITE}{cls.__name__}{cltcolors.bcolors.ENDC} "
                    f"{cltcolors.bcolors.OKGRAY}({cls.__module__}){cltcolors.bcolors.ENDC}"
                )

    print(
        f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.HEADER}{'='*60}{cltcolors.bcolors.ENDC}"
    )


########################################################################################################
def print_dict_tree(
    data: Dict[Any, Any],
    prefix: str = "",
    is_last: bool = True,
    max_value_length: int = 50,
) -> None:
    """
    Print dictionary in a tree-like structure with ANSI colors.

    Parameters
    ----------
    data (dict):
        The dictionary to print

    prefix (str):
        Prefix for the current level (used for recursion)

    is_last (bool):
        Whether the current item is the last in its level

    max_value_length (int):
        Maximum length of value strings before truncation

    Returns
    -------
    None:
        Prints the dictionary structure to stdout

    Examples
    --------
    >>> my_dict = {"key1": "value1", "key2": {"subkey1": "subvalue1", "subkey2": "subvalue2"}}
    >>> print_dict_tree(my_dict)
    ├── key1: value1
    └── key2/
        ├── subkey1: subvalue1
        └── subkey2: subvalue2

    Notes
    -----
    - Uses Unicode box-drawing characters for tree structure
    - Color codes for keys and values for better readability
    - Truncates long values to avoid clutter

    """
    if not isinstance(data, dict):
        return

    items = list(data.items())

    for i, (key, value) in enumerate(items):
        is_last_item = i == len(items) - 1
        current_prefix = "└── " if is_last_item else "├── "

        if isinstance(value, dict):
            print(
                f"{prefix}{cltcolors.bcolors.OKWHITE}{current_prefix}{cltcolors.bcolors.OKBLUE}{cltcolors.bcolors.BOLD}{key}/{cltcolors.bcolors.ENDC}"
            )
            extension = "    " if is_last_item else "│   "
            print_dict_tree(value, prefix + extension, is_last_item, max_value_length)
        else:
            # Handle long values by truncating
            value_str = str(value)
            if len(value_str) > max_value_length:
                value_str = value_str[: max_value_length - 3] + "..."

            print(
                f"{prefix}{cltcolors.bcolors.OKWHITE}{current_prefix}{cltcolors.bcolors.OKYELLOW}{key}{cltcolors.bcolors.ENDC}: {cltcolors.bcolors.OKGRAY}{value_str}{cltcolors.bcolors.ENDC}"
            )


#####################################################################################################
def search_methods(obj, keyword, case_sensitive=False):
    """
    Search for methods/attributes containing a keyword in name or docstring.

    Parameters
        obj: The object to search in
        keyword (str): The keyword to search for
        case_sensitive (bool): Whether the search should be case sensitive

    Returns
        None: Prints the search results to stdout

    Examples
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
        f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.HEADER}🔍 SEARCH RESULTS for '{keyword}'{cltcolors.bcolors.ENDC}"
    )
    print(
        f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.HEADER}{'='*40}{cltcolors.bcolors.ENDC}"
    )

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
            print(
                f"{cltcolors.bcolors.BOLD}{cltcolors.bcolors.OKYELLOW}✨ {name}{cltcolors.bcolors.ENDC}"
            )

            if inspect.ismethod(value) or inspect.isfunction(value):
                try:
                    sig = inspect.signature(value)
                    print(
                        f"  {cltcolors.bcolors.OKGRAY}📋 Signature:{cltcolors.bcolors.ENDC} {cltcolors.bcolors.OKWHITE}{sig}{cltcolors.bcolors.ENDC}"
                    )
                except:
                    pass

            if doc:
                first_line = doc.split("\n")[0]
                if len(first_line) > 80:
                    first_line = first_line[:77] + "..."
                print(
                    f"  {cltcolors.bcolors.ITALIC}{cltcolors.bcolors.OKGRAY}{first_line}{cltcolors.bcolors.ENDC}"
                )

            print()

    if not found:
        print(
            f"{cltcolors.bcolors.WARNING}❌ No matches found for '{keyword}'{cltcolors.bcolors.ENDC}"
        )


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

            local_weights = weights[w_start:w_end]
            local_weights /= local_weights.sum()

            new_points[i] = np.sum(smoothed[start:end] * local_weights[:, None], axis=0)

        smoothed = new_points

    return smoothed
