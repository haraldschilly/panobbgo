#!/usr/bin/env python

from setuptools import setup, find_packages

from psnobfit import __version__

setup(
    name = "psnobfit",
    version = __version__, 
    packages = find_packages(),

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires = ['docutils>=0.3', "numpy >= 1.5.0", "IPython >= 0.12" ],

    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst']#,
    #    # And include any *.msg files found in the 'hello' package, too:
    #    'hello': ['*.msg'],
    },

    test_suite = "psnobfit.test",

    # metadata for upload to PyPI
    author           = "Harald Schilly",
    author_email     = "harald.schilly@univie.ac.at",
    description      = "Parallel Snobfit",
    url              = "",
    license          = 'BSD',
    keywords = "optimization blackbox stochastic noisy parallel",

    # could also include long_description, download_url, classifiers, etc.
)
