#!/usr/bin/env python

from setuptools import setup#, find_packages

from panobbgo_problems import __version__

setup(
    name = "panobbgo_problems",
    version = __version__, 
    packages = ['panobbgo_problems'] ,#find_packages(),

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires = [
        'docutils >= 0.3', 
        #"numpy    >= 1.5.0", 
        #"scipy    >= 0.9.0",
        "IPython  >= 0.12" 
    ],

    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst']#,
    #    # And include any *.msg files found in the 'hello' package, too:
    #    'hello': ['*.msg'],
    },

    #test_suite = "psnobfit.test",

    # metadata for upload to PyPI
    author           = "Harald Schilly",
    author_email     = "harald.schilly@univie.ac.at",
    description      = "Problems for panobbgo",
    url              = "http://github.com/haraldschilly/panobbgo",
    license          = 'Apache 2.0',
    keywords = "optimization blackbox stochastic noisy parallel black-box ipython distributed cluster",

    # could also include long_description, download_url, classifiers, etc.
)
