"""Sphinx configuration for Peach documentation.

Follows scVerse documentation standards.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add source to path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))
sys.path.insert(0, str(HERE.parent / "src"))

# Import package for version
import peach

# Project information
project = "🍑 PEACH"
author = "Alexander Honkala"
copyright = f"{datetime.now():%Y}, {author}"
version = peach.__version__
release = version

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxext.opengraph",
    "myst_parser",
]

# Templates path
templates_path = ["_templates"]

# Source suffix
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Master document
master_doc = "index"

# Language
language = "en"

# Excluded patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "workflows/**"]

# Pygments style
pygments_style = "sphinx"

# HTML theme configuration
html_theme = "sphinx_book_theme"
html_title = "🍑 PEACH"
html_theme_options = {
    "repository_url": "https://github.com/xhonkala/PEACH",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "navigation_with_keys": True,
}

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
# html_logo = "_static/logo.png"  # Uncomment when logo is added

# Napoleon settings for docstring parsing
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "signature"
autodoc_typehints_description_target = "documented_params"
autodoc_mock_imports = ["torch", "plotly", "matplotlib"]

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
}

# MyST parser settings
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]

# nbsphinx settings
nbsphinx_execute = "never"
nbsphinx_allow_errors = False
nbsphinx_kernel_name = "python3"

# Copybutton settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# OpenGraph settings
ogp_site_url = "https://scpeach.readthedocs.io/"
ogp_image = "https://scpeach.readthedocs.io/en/latest/_static/logo.png"

# Suppress warnings
suppress_warnings = ["autosummary.import_cycle"]

# Custom setup
def setup(app):
    """Setup custom Sphinx application."""
    app.add_css_file("css/custom.css")