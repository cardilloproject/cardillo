# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "cardillo3"
html_title = (
    "cardillo: Solvers, algorithms and benchmark systems for nonsmooth dynamics"
)
author = "Jonas Harsch, Giuseppe Capobianco, Simon Eugster"
copyright = "2022, " + author

# The full version, including alpha/beta/rc tags
release = "3.0.0"

# -- General configuration ---------------------------------------------------

# Enabled extensions
extensions = [
    "sphinxcontrib.bibtex",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    "sphinx.ext.todo",
    "myst_nb",
    "sphinx_copybutton",
]
# sphinxcontrib.bibtex -> For citations
# sphinx_math_dollar   -> For wrting inline math using dollar sign ( $ ).
#                         Use $\mathrm{R}$ for inline math.
#                         Use $$\mathrm{R}$$ for new full line equation.

# Config for myst_nb
jupyter_execute_notebooks = "cache"

# Config for sphinx.ext.todo
todo_include_todos = True

# Config for sphinxcontrib.bibtex
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "plain"

# Config for sphinx_math_dollar
mathjax3_config = {
    "tex2jax": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    },
    # set LaTeX macros, see https://stackoverflow.com/a/60497853/7280763
    # note: bm and diff are not working correct and are defined here
    "tex": {
        "macros": {
            "bm": ["{\\boldsymbol #1}", 1],  # hack for \bm in mathjax
            "diff": [
                "{\\mathrm{d} #1}",
                1,
                "",
            ],
        }
    },
}

import re

# load external LaTeX macros, see https://stackoverflow.com/a/60497853/7280763
with open("mathsymbols.tex", "r") as f:
    for line in f:
        macros = re.findall(
            r"\\(DeclareRobustCommand|newcommand){\\(.*?)}(\[(\d)\])?{(.+)}", line
        )
        for macro in macros:
            if len(macro[2]) == 0:
                mathjax3_config["tex"]["macros"][macro[1]] = "{" + macro[4] + "}"
            else:
                mathjax3_config["tex"]["macros"][macro[1]] = [
                    "{" + macro[4] + "}",
                    int(macro[3]),
                ]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"
