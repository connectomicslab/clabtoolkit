misctools module
================

.. automodule:: clabtoolkit.misctools
   :members:
   :undoc-members:
   :show-inheritance:

The misctools module provides essential utility functions and helper classes that support all other modules in the clabtoolkit package.

Key Features
------------
- Enhanced command-line argument parsing
- File system operations and path management
- Color processing and conversion utilities
- Documentation generation helpers
- Progress tracking and console output
- String manipulation and validation

Main Classes
------------

SmartFormatter
~~~~~~~~~~~~~~
Enhanced argparse formatter for better help text display.

Key Methods:
- Custom help text formatting with preserved line breaks
- Improved readability for complex command-line interfaces
- Support for structured help documentation

Common Usage Examples
---------------------

Enhanced argument parsing::

    from clabtoolkit.misctools import SmartFormatter
    import argparse
    
    # Create parser with enhanced formatting
    parser = argparse.ArgumentParser(
        formatter_class=SmartFormatter,
        description="Enhanced command-line tool"
    )
    
    parser.add_argument(
        '--input', 
        help='R|Input file path\n'
             'Supports multiple formats:\n'
             '  - NIfTI (.nii.gz)\n'  
             '  - FreeSurfer (.mgh)\n'
             '  - GIFTI (.gii)'
    )

Utility functions::

    # File system operations
    from clabtoolkit import misctools as utils
    
    # Validate file existence and format
    if utils.validate_file_format(filepath, ['.nii.gz', '.mgh']):
        print("Valid neuroimaging file")
    
    # Color processing
    rgb_color = utils.hex_to_rgb('#FF5733')
    hex_color = utils.rgb_to_hex((255, 87, 51))
    
    # Progress tracking
    utils.display_progress("Processing subjects", current=5, total=20)

String and path utilities::

    # Path manipulation
    clean_path = utils.normalize_path("/path/with/../redundant/./parts")
    
    # String validation
    if utils.validate_subject_id("sub-001"):
        print("Valid BIDS subject ID")
    
    # Generate unique identifiers
    unique_id = utils.generate_unique_id(prefix="session")