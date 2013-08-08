#!/usr/bin/env python
# -*- coding: utf8 -*-

from setuptools import setup  # , find_packages
from pip.req import parse_requirements
from panobbgo import __version__

# read requirements.txt
install_reqs = parse_requirements("requirements.txt")
required = [str(ir.req) for ir in install_reqs]

setup(
    name="panobbgo",
    version=__version__,
    packages=['panobbgo'],  # find_packages(),

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=required,

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
