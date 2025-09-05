# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Mock imports for ReadTheDocs build - comprehensive list
autodoc_mock_imports = [
    # 3D Visualization
    'pyvista',
    'vtk', 
    'fury',
    'trame',
    'trame_client',
    'trame_server',
    'trame_vtk',
    'trame_vuetify',
    'trame_common',
    'trame.app',
    'trame.widgets',
    'trame.html',
    
    # Neuroimaging
    'dipy',
    'dipy.io',
    'dipy.io.streamline',
    'dipy.io.stateful_tractogram',
    'dipy.segment',
    'dipy.segment.clustering',
    'dipy.tracking',
    'dipy.tracking.streamline',
    'nilearn',
    'nilearn.plotting',
    'nilearn.maskers',
    'pydicom',
    'nibabel',
    'nibabel.streamlines',
    'nibabel.orientations',
    'nibabel.processing',
    
    # Image Processing
    'skimage',
    'skimage.measure',
    'skimage.morphology',
    'skimage.filters',
    'cv2',
    
    # Scientific Computing - additional submodules
    'scipy.sparse',
    'scipy.ndimage',
    'scipy.spatial',
    'scipy.interpolate',
    'scipy.optimize',
    
    # Plotting and UI
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.patches',
    'matplotlib.colors',
    'matplotlib.figure',
    'matplotlib.axes',
    'seaborn',
    'plotly',
    'rich',
    'rich.progress',
    'rich.console',
    'rich.panel',
    'colorama',
    'screeninfo',
    
    # Interactive Computing
    'ipython',
    'IPython',
    'IPython.display',
    'jupyter',
    'ipywidgets',
    
    # GUI and System
    'tkinter',
    'pygame',
    
    # Data Processing
    'h5py',
    'openpyxl',
    'lxml',
    'beautifulsoup4',
    'bs4',
    
    # Parallel and Performance
    'joblib',
    'numba',
    'cupy',
    'dask',
    
    # Machine Learning
    'sklearn',
    'scikit-learn',
    'torch',
    'tensorflow',
    
    # Network Analysis
    'networkx',
    'igraph',
    
    # Statistical Analysis
    'statsmodels',
    'pingouin',
    'patsy',
    
    # Misc utilities
    'tqdm',
    'click',
    'typer',
    'pydantic',
    'marshmallow',
    'requests',
    'urllib3',
    'certifi',
    'fsspec',
    'pooch',
    'duecredit'
]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'clabtoolkit'
copyright = '2024, Yasser Alemán-Gómez'
author = 'Yasser Alemán-Gómez'
release = '0.3.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'show-inheritance': True,
    'exclude-members': '__weakref__'
}

# Show more detailed autodoc information
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Additional autodoc settings for better documentation generation
autodoc_preserve_defaults = True
autodoc_class_signature = 'mixed'

# Ensure modules can be imported even with missing dependencies
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add clabtoolkit to the Python path
sys.path.insert(0, os.path.abspath('../../clabtoolkit'))

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'nibabel': ('https://nipy.org/nibabel/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_theme = 'alabaster'  # Built-in theme, no extra install needed

html_static_path = ['_static']

html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
