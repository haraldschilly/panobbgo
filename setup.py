#!/usr/bin/env python
# -*- coding: utf8 -*-

from setuptools import setup  # , find_packages

from panobbgo import __version__

setup(
    name="panobbgo",
    version=__version__,
    packages=['panobbgo'],  # find_packages(),

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        'docutils >= 0.3',
        "numpy    >= 1.5.0",
        "scipy    >= 0.9.0",
        "IPython  >= 0.12",
        "nose     >= 1.1.2",
        "mock     >= 1.0.1"
    ],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst']  # ,
        # And include any *.msg files found in the 'hello' package, too:
        #    'hello': ['*.msg'],
    },

    test_suite="nose.collector",

    # metadata for upload to PyPI
    author="Harald Schilly",
    author_email="harald.schilly@univie.ac.at",
    description="Parallel Noisy Black-Box Global Optimization",
    url="http://github.com/haraldschilly/panobbgo",
    license='Apache 2.0',
    keywords="optimization blackbox stochastic noisy parallel black-box ipython distributed cluster",

    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Operating System :: POSIX :: Linux"
    ]
)
