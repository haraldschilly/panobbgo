#!/usr/bin/env python

import os
import sys
import numpy
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core      import setup

# -----------------------------------------------------
if len(sys.argv) == 1:
    sys.argv.append('install')

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


# ------------------------------------------------------
def configuration(parent_package='',
                  top_path=None
                  ):
    if  numpy.__version__ < '1.0.0':
        raise RuntimeError, 'numpy version %s or higher required, but got %s'\
              % ('1.0.0', numpy.__version__)
    
    config = Configuration(None, parent_package, top_path)

    config.set_options( ignore_setup_xxx_py=True,
                        assume_default_configuration=True,
                        delegate_options_to_subpackages=True,
                        quiet=True
                       )
     
    config.add_subpackage('psnobfit')
    
    config.add_data_files( ('psnobfit', '*.txt') )
    config.add_data_files( ('psnobfit', 'psnobfit.epydoc') )
    #config.get_version(os.path.join('snobfit','version.py'))   # sets config.version

    return config



def setup_package():
    old_path   = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0,local_path)
    sys.path.insert(0,os.path.join(local_path, 'snobfit')) # to retrive version

    try:
        from version import version as version
        setup( name = 'snobfit',
               version = version, #will be overwritten by configuration version
               maintainer       = "Harald Schilly",
               maintainer_email = "harald@schil.ly",
               description      = "Parallel Version of Snobfit",
               url              = "",
               license          = 'BSD',
               configuration=configuration
               )
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return


# -----------------------------------------------------
if __name__ == '__main__':
    setup_package()

