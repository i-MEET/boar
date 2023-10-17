# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


# add boar to sys.path
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
import boar

import datetime
from importlib.metadata import metadata

# isort: off
from sphinx_gallery.sorting import (  # pylint: disable=no-name-in-module
    ExplicitOrder,
    ExampleTitleSortKey,
)


project = 'boar'
copyright = '2023, Vincent M. Le Corre, Larry Lüer'
author = 'Vincent M. Le Corre, Larry Lüer'
release = '1.0.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx_gallery.load_style",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.mathjax",  # Maths visualization
    "sphinx.ext.graphviz",  # Dependency diagrams
    "sphinx_copybutton",
    "notfound.extension",
    'sphinx_search.extension',
    'sphinx_gallery.gen_gallery',
    "sphinx_gallery.load_style",
    ]

templates_path = ['_templates']
exclude_patterns = []
source_suffix = ['.rst', '.md']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for sphinx_gallery ----------------------------------------------
from plotly.io._sg_scraper import plotly_sg_scraper
image_scrapers = ('matplotlib', plotly_sg_scraper,)

sphinx_gallery_conf = {
     'doc_module': ('plotly',),
     'examples_dirs': 'examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     'filename_pattern': r"\.ipynb", # Notebooks to include
     'image_scrapers': image_scrapers,
    #  'ignore_pattern': r"__init__\.py", # Ignore this file
    #  'subsection_order': ExplicitOrder([
    #                 '../../examples/DD_Fit_fake_OPV.ipynb',
    #                 '../../examples/DD_Fit_real_OPV.ipynb',]),


}

nbsphinx_execute = 'never' # so we don't run the notebooks

html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "i-MEET", # Username
    "github_repo": "boar", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/docs/source/", # Path in the checkout to the docs root
}