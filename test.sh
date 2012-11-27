#!/usr/bin/env bash

# assume modern unittest and python 2.7

python -m unittest discover -s `dirname "$0"` -p '*_test.py'

exit $?
