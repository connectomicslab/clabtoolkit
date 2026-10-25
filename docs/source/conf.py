# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# All the dependencies are installed when the documentation is built (pip install -e .),
# so nothing is mocked: mocked types cannot be used in annotations such as "np.ndarray | csr_matrix".
autodoc_mock_imports = []

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "clabtoolkit"
copyright = "2024-2026, Yasser Alemán-Gómez"
author = "Yasser Alemán-Gómez"
release = "0.6.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = []

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
    "exclude-members": "__weakref__",
}

# Show more detailed autodoc information
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Additional autodoc settings for better documentation generation
autodoc_preserve_defaults = True
autodoc_class_signature = "mixed"

# Ensure modules can be imported even with missing dependencies
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add clabtoolkit to the Python path
sys.path.insert(0, os.path.abspath("../../clabtoolkit"))

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme = 'alabaster'  # Built-in theme, no extra install needed

html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_logo = "_static/clabtoolkit_logo.png"
html_favicon = "_static/clabtoolkit_favicon.ico"

html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}


def _skip_module_docstring(app, what, name, obj, options, lines):
    """The overview of each module is written at the top of its page in modules/*.rst."""
    if what == "module":
        del lines[:]


def setup(app):
    app.connect("autodoc-process-docstring", _skip_module_docstring)
