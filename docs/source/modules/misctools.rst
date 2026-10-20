misctools module
================

.. automodule:: clabtoolkit.misctools
   :members:
   :undoc-members:
   :show-inheritance:

The misctools module collects general-purpose helpers used across the clabtoolkit package: index and condition parsing, list and string filtering, file system and JSON utilities, table reading, container commands and tools to explore Python objects and HDF5 files.

Key Features
------------
- Build index lists from integers, ranges (``"3:5"``), tuples and logical conditions
- Filter lists and rename strings by substrings or regular expressions
- Clean and standardize region names
- File system helpers: list files, find leaf folders, rename and remove folders, create temporary filenames
- Read and write JSON, merge and compare dictionaries, read delimited tables with separator detection
- Build the command that runs a command line inside a container
- Inspect modules, objects and HDF5 files from the terminal or a notebook

Main Functions
--------------

Indices and conditions
~~~~~~~~~~~~~~~~~~~~~~
- ``build_indices``, ``build_indices_with_conditions``, ``build_values_with_conditions``
- ``get_indices_by_condition``, ``get_values_by_condition``, ``parse_condition``, ``analyze_condition``

Lists and strings
~~~~~~~~~~~~~~~~~
- ``filter_by_substring``, ``get_indexes_by_substring``, ``remove_substrings``, ``replace_substrings``
- ``to_list``, ``list_intercept``, ``ismember``, ``remove_duplicates``, ``select_ids_from_file``
- ``correct_names``, ``create_names_from_indices``, ``remove_consecutive_duplicates``

Files, folders and tables
~~~~~~~~~~~~~~~~~~~~~~~~~
- ``get_all_files``, ``get_leaf_directories``, ``rename_folders``, ``remove_empty_folders``, ``create_temporary_filename``
- ``load_json``, ``save_dictionary_to_json``, ``update_dict``, ``compare_dicts``
- ``smart_read_table``, ``drop_empty_columns``, ``expand_and_concatenate``
- ``generate_container_command``

Exploration and display
~~~~~~~~~~~~~~~~~~~~~~~
- ``show_module_contents``, ``show_object_content``, ``search_methods``
- ``h5explorer``, ``h5explorer_simple``, ``printprogressbar``, ``is_notebook``
- ``SmartFormatter``: ``argparse`` formatter that keeps the line breaks of help messages starting with ``R|``

Common Usage Examples
---------------------

Indices and conditions::

    import numpy as np
    from clabtoolkit import misctools as cltmisc

    # Integers, tuples and "start:stop" strings are expanded and sorted
    cltmisc.build_indices([1, (3, 5), "7:9"])  # [1, 3, 4, 5, 7, 8, 9]

    # Select the values of an array that fulfil a condition
    x = np.array([1, 2, 3, 4])
    cltmisc.get_indices_by_condition("x > 2", x=x)  # array([2, 3])

Lists and names::

    names = ["ctx-lh-a", "ctx-rh-b", "subc-x"]
    cltmisc.filter_by_substring(names, "ctx-lh")  # ['ctx-lh-a']
    cltmisc.replace_substrings(["a_b"], {"_": "-"})  # ['a-b']

    cltmisc.correct_names(["Left Thalamus"], prefix="subc-", case="lower",
                          replacements={" ": "-"})  # ['subc-left-thalamus']
    cltmisc.create_names_from_indices([1, 2], prefix="roi", padding=3)  # ['roi-001', 'roi-002']

Files and tables::

    files = cltmisc.get_all_files("/path/to/dataset", or_filter=".nii.gz")
    tmp = cltmisc.create_temporary_filename(prefix="simulated_", extension=".nii.gz")
    df = cltmisc.smart_read_table("/path/to/table.tsv")

Enhanced argument parsing::

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=cltmisc.SmartFormatter,
        description="Enhanced command-line tool",
    )
    parser.add_argument(
        "--input",
        help="R|Input file path\n"
             "Supports multiple formats:\n"
             "  - NIfTI (.nii.gz)\n"
             "  - FreeSurfer (.mgh)",
    )

Exploring objects::

    import clabtoolkit.parcellationtools as cltparc

    cltmisc.show_module_contents(cltparc)
    cltmisc.search_methods(cltparc.Parcellation, "keep")
