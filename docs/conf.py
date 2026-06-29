"""Sphinx configuration for the libdamp documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import libdamp  # noqa: E402

project = "libdamp"
copyright = "2026, Simon Schwär, Manuel Peters, Meinard Müller"
author = "Simon Schwär, Manuel Peters, Meinard Müller"
release = libdamp.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # parses NumPy-style docstrings, used throughout libdamp
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]
