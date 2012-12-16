#!/usr/bin/env bash

FILES="panobbgo*/*.py *.py doc/source/*.py"

pep8 --max-line-length=110 $FILES

# autopep8 -i --max-line-length=110 $FILES
