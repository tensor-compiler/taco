# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../../build/lib/'))

# -- Project information -----------------------------------------------------
project = 'TACO Python API Reference'
copyright = '2022 TACO Development Team'
author = 'TACO Development Team'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'numpydoc'
]

templates_path = ['_templates']

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_title = 'TACO Python API Reference'

# -- Extension configuration -------------------------------------------------
numpydoc_attributes_as_param_list = True
numpydoc_class_members_toctree = False
numpydoc_show_class_members = False

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {'https://docs.python.org/': None}
