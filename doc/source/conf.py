# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path
from datetime import date
sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[2] / 'cobrawap' / 'pipeline'))
# -- Project information -----------------------------------------------------


# The master toctree document.
master_doc = 'index'


# General information about the project.
project = 'Collaborative Brain Wave Analysis Pipeline (Cobrawap)'
authors = u'Cobrawap authors and contributors'
copyright = u"2017-{this_year}, {authors}".format(this_year=date.today().year,
                                                  authors=authors)

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
with open(os.path.join(root_dir, 'VERSION')) as version_file:
    # The full version, including alpha/beta/rc tags.
    release = version_file.read().strip()

# The short X.Y version.
version = '.'.join(release.split('.')[:-1])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinxarg.ext',
    # 'myst_parser',
]

source_suffix = ['.rst', '.md']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates', '_templates/autosummary']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Required to automatically create a summary page for each function listed in
# the autosummary fields of each module.
autosummary_generate = True

# Set to False to not overwrite the custom _toctree/*.rst
autosummary_generate_overwrite = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'font_family': 'Arial',
    'page_width': '1200px',  # default is 940
    'sidebar_width': '280px',  # default is 220
    'logo': 'cobrawap_logo.png',  # add logo to sidebar
    'fixed_sidebar': 'true'
}

html_favicon = '../images/cobrawap_icon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static', '../images']

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True
